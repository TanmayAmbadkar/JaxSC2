"""End-to-end regressions for resets, action masks, updates, and checkpoints."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.env.env import JaxSC2Env, CentralAction
from algorithms.common.rollout import step_and_reset, policy_masks, evaluate_episodes
from algorithms.ppo import PPO
from algorithms.mask_ppo import MaskPPO
from algorithms.a2c.trainer import A2CTrainer
from algorithms.ppo.eval import evaluate
from algorithms.mask_ppo.eval import evaluate as evaluate_masked


def assert_tree_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def test_terminal_reward_is_retained_and_next_state_resets():
    env = JaxSC2Env("V1_Base")
    env.max_steps = 1
    _, state = env.reset(jax.random.PRNGKey(0))
    action = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    _, next_state, reward, done = jax.jit(
        lambda key, state: step_and_reset(env, key, state, action)
    )(jax.random.PRNGKey(1), state)
    assert bool(done)
    assert float(reward) < -14.
    assert int(next_state.timestep) == 0
    assert not bool(next_state.smax_state.terminal)


def test_masks_forbid_invisible_targets_and_handle_no_targets():
    env = JaxSC2Env("V1_Base")
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(enemy_visible=jnp.array([False, True, False]))
    mask = policy_masks(env, state)
    np.testing.assert_array_equal(mask["target"], [False, True, False])
    state = state.replace(enemy_visible=jnp.zeros(3, dtype=bool))
    mask = policy_masks(env, state)
    assert not bool(mask["verb"][2])
    np.testing.assert_array_equal(mask["target"], [True, False, False])


@pytest.mark.parametrize("algorithm", [PPO, MaskPPO, A2CTrainer])
def test_training_checkpoints_restore_and_continue(algorithm, tmp_path):
    config = {
        "NUM_ENVS": 2, "ROLLOUT_LEN": 4, "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 1, "CKPT_INTERVAL": 1,
        "LOG_DIR": str(tmp_path / "logs"), "CKPT_DIR": str(tmp_path / "checkpoints"),
    }
    env = JaxSC2Env("V1_Base")
    env.max_steps = 2
    model = algorithm(config)

    def train(agent, steps):
        if algorithm is A2CTrainer:
            agent.env = env
            agent.train(steps)
        else:
            agent.train(env, steps)

    train(model, 0)
    initial_params = model.state.params
    train(model, 16)
    assert int(model.state.step) == (2 if algorithm is A2CTrainer else 4)
    # All policy heads, as well as the critic, should receive gradient updates.
    for name in initial_params:
        if name.startswith("Dense_"):
            assert not np.array_equal(initial_params[name]["kernel"], model.state.params[name]["kernel"])
    path = tmp_path / "checkpoints" / "ckpt_16.pkl"
    with path.open("rb") as file:
        checkpoint = pickle.load(file)
    assert checkpoint["step"] == 16
    assert int(checkpoint["optimizer_step"]) == int(model.state.step)
    assert_tree_equal(checkpoint["params"], model.state.params)
    assert_tree_equal(checkpoint["opt_state"], model.state.opt_state)

    restored = algorithm(config)
    restored.load(path)  # Supports loading before model initialization.
    train(restored, 0)
    assert restored.total_steps == 16
    assert not isinstance(restored.state, tuple)
    assert_tree_equal(restored.state.params, model.state.params)
    assert_tree_equal(restored.state.opt_state, model.state.opt_state)
    assert_tree_equal(restored.rms, model.rms)
    train(restored, 8)
    assert restored.total_steps == 24
    assert int(restored.state.step) > int(model.state.step)

    evaluator = evaluate_masked if algorithm is MaskPPO else evaluate
    metrics = evaluator(env, restored.state.params, restored.model, jax.random.PRNGKey(9),
                        num_episodes=2, max_steps=4)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert 0 <= float(metrics["eval/total_win_rate"]) <= 1


def test_maskppo_samples_only_allowed_actions_and_resets(tmp_path):
    actions = []

    class RestrictedEnv(JaxSC2Env):
        def get_action_mask(self, state):
            mask = super().get_action_mask(state)
            mask.update(
                verb=jnp.array([False, True, False]),
                direction=jnp.arange(8) == 2,
                target=jnp.arange(8) == 1,
            )
            return mask

        def step(self, key, state, action):
            jax.debug.callback(
                lambda t, v, d, target: actions.append((int(t), int(v), int(d), int(target))),
                state.timestep, action.verb[0], action.direction[0], action.target[0],
            )
            return super().step(key, state, action)

    env = RestrictedEnv("V1_Base")
    env.max_steps = 2
    model = MaskPPO({
        "NUM_ENVS": 2, "ROLLOUT_LEN": 4, "NUM_MINIBATCHES": 2, "UPDATE_EPOCHS": 2,
        "LOG_DIR": str(tmp_path / "logs"),
    })
    model.train(env, 16)
    jax.effects_barrier()
    assert len(actions) == 16
    assert {item[0] for item in actions} == {0, 1}
    assert all(item[1:] == (1, 2, 1) for item in actions)


def test_evaluation_counts_victory_once_and_freezes_episode():
    class BeaconEnv(JaxSC2Env):
        def reset(self, key):
            _, state = super().reset(key)
            state = state.replace(beacon_pos=state.smax_state.unit_positions[0])
            return self.get_obs(state), state

    class NoopModel:
        def apply(self, params, obs):
            return {
                "verb_logits": jnp.array([1., 0., 0.]),
                "direction_logits": jnp.zeros(8), "target_logits": jnp.zeros(3),
            }

    env = BeaconEnv("V1_Base")
    key = jax.random.PRNGKey(0)
    short = evaluate_episodes(env, None, NoopModel(), key, 2, 1)
    long = evaluate_episodes(env, None, NoopModel(), key, 2, 5)
    assert_tree_equal(short, long)
    assert jnp.all(long[1])
    assert not jnp.any(long[2])
