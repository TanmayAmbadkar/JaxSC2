"""Balanced objectives and backwards-compatible reward profiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.rewards import REWARD_PROFILES


@pytest.fixture(scope="module", params=["legacy", "balanced"])
def reward_env(request):
    env = JaxSC2Env("V2_Base", reward_profile=request.param)
    _, state = env.reset(jax.random.PRNGKey(0))
    action = CentralAction(jnp.ones(env.num_allies, dtype=bool), 0, 0, -1)
    return env, state, action, jax.jit(env.step)


def test_balanced_profile_is_default_and_keeps_nonobjective_weights():
    env = JaxSC2Env()
    assert env.reward_profile == "balanced"
    assert env.reward_weights["navigation_progress"] == env.reward_weights["combat_progress"] == 2.
    assert env.reward_weights["beacon_win"] == env.reward_weights["combat_win"] == 25.
    for key in ("enemy_damage", "ally_damage", "enemy_kill", "defeat", "timeout"):
        assert REWARD_PROFILES["legacy"][key] == env.reward_weights[key]
    # Each instance gets an independent copy, not a mutable global reference.
    env.reward_weights["combat_win"] = 999.
    assert JaxSC2Env().reward_weights["combat_win"] == 25.


def test_unknown_reward_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown reward profile"):
        JaxSC2Env(reward_profile="typo")


def test_progress_rewards_use_profile_coefficients(reward_env):
    env, state, action, step = reward_env
    key = jax.random.PRNGKey(1)
    _, _, _, _, before = step(key, state, action)
    shifted = state.replace(prev_mean_dist=state.prev_mean_dist + .1,
                            prev_enemy_dist=state.prev_enemy_dist + .1)
    _, _, _, _, after = step(key, shifted, action)
    np.testing.assert_allclose(after["nav_reward"] - before["nav_reward"],
                               .1 * env.reward_weights["navigation_progress"], atol=1e-6)
    np.testing.assert_allclose(after["combat_shaping"] - before["combat_shaping"],
                               .1 * env.reward_weights["combat_progress"], atol=1e-6)


@pytest.mark.parametrize("outcome", ["beacon", "combat", "both", "defeat", "timeout"])
def test_terminal_rewards_and_episode_endings(reward_env, outcome):
    env, state, action, step = reward_env
    if outcome in ("beacon", "both"):
        state = state.replace(beacon_pos=state.smax_state.unit_positions[0])
    if outcome in ("combat", "both"):
        smax = state.smax_state
        state = state.replace(smax_state=smax.replace(
            unit_alive=smax.unit_alive.at[env.num_allies:].set(False),
            unit_health=smax.unit_health.at[env.num_allies:].set(0.)))
    if outcome == "defeat":
        smax = state.smax_state
        state = state.replace(smax_state=smax.replace(
            unit_alive=smax.unit_alive.at[:env.num_allies].set(False),
            unit_health=smax.unit_health.at[:env.num_allies].set(0.)))
    if outcome == "timeout":
        state = state.replace(timestep=env.max_steps - 1)
    _, _, reward, done, info = step(jax.random.PRNGKey(1), state, action)
    assert bool(done) and np.isfinite(float(reward))
    name = {"beacon": "beacon_win", "combat": "combat_win", "both": "beacon_win",
            "defeat": "defeat", "timeout": "timeout"}[outcome]
    assert float(info["terminal_reward"]) == env.reward_weights[name]
    assert bool(info["nav_success"]) == (outcome in ("beacon", "both"))
    assert bool(info["combat_success"]) == (outcome in ("combat", "both"))
