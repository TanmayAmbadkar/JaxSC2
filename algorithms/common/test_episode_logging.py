"""Episode logging must include terminal rewards and span rollout boundaries."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.common.episode_logging import EpisodeAccumulator, EpisodeTotals, EpisodeLogWindow, accumulate_episodes
from JaxSC2.env.env import JaxSC2Env


def test_episode_returns_span_rollouts_and_reset_independently():
    initial = EpisodeAccumulator(jnp.zeros(2), jnp.zeros(2, dtype=jnp.int32))
    running, first = jax.jit(accumulate_episodes)(
        initial, jnp.array([[1., 10.], [2., 20.]]), jnp.array([[False, False], [False, True]]))
    np.testing.assert_array_equal(running.returns, [3., 0.])
    assert float(first.return_sum) == 30.
    assert int(first.length_sum) == 2 and int(first.count) == 1
    running, second = jax.jit(accumulate_episodes)(
        running, jnp.array([[25., 4.], [7., 5.]]), jnp.array([[True, False], [False, True]]))
    assert float(second.return_sum) == 28. + 9.
    assert int(second.length_sum) == 3 + 2 and int(second.count) == 2
    np.testing.assert_array_equal(running.returns, [7., 0.])
    np.testing.assert_array_equal(running.lengths, [1, 0])


def test_log_window_weights_episodes_and_does_not_emit_false_zero_returns():
    window = EpisodeLogWindow()
    metrics, display = window.emit()
    assert metrics == {"train/completed_episodes": 0}
    assert "n/a" in display and "train/episode_reward" not in metrics
    window.add(EpisodeTotals(10., 3, 1))
    window.add(EpisodeTotals(80., 15, 4))
    metrics, _ = window.emit()
    assert metrics["train/episode_reward"] == 18.
    assert metrics["train/episode_length"] == 3.6
    assert metrics["train/completed_episodes"] == 5
    assert window.count == 0


@pytest.mark.parametrize("module,class_name", [
    ("algorithms.ppo.ppo", "PPO"),
    ("algorithms.mask_ppo.mask_ppo", "MaskPPO"),
    ("algorithms.a2c.trainer", "A2CTrainer"),
])
def test_training_logs_completed_raw_returns_not_step_rewards(module, class_name, monkeypatch, capsys):
    records = []

    class FakeLogger:
        def __init__(self, path):
            pass

        def log(self, step, metrics):
            records.append((step, metrics))

        def close(self):
            pass

    class ThreeStepEnv(JaxSC2Env):
        def step(self, key, state, action):
            state = state.replace(timestep=state.timestep + 1)
            return self.get_obs(state), state, jnp.float32(1.), state.timestep == 3, {}

    trainer_module = importlib.import_module(module)
    logger_module = importlib.import_module("algorithms.ppo.ppo" if class_name == "A2CTrainer" else module)
    monkeypatch.setattr(logger_module, "Logger", FakeLogger)
    model = getattr(trainer_module, class_name)({
        "NUM_ENVS": 1, "ROLLOUT_LEN": 2, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1,
        "LOG_INTERVAL": 1, "EVAL_INTERVAL": 0, "CKPT_INTERVAL": 100,
    })
    env = ThreeStepEnv("V1_Base")
    if class_name == "A2CTrainer":
        model.env = env
        model.train(6)
    else:
        model.train(env, 6)
    assert len(records) == 3
    assert "train/episode_reward" not in records[0][1]
    for _, metrics in records[1:]:
        assert metrics["train/episode_reward"] == 3.
        assert metrics["train/episode_length"] == 3.
        assert metrics["train/completed_episodes"] == 1
        assert metrics["train/step_reward"] == 1.
        assert "train/mean_reward" not in metrics
    assert "Episode Reward    3.00" in capsys.readouterr().out
