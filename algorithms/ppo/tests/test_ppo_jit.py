"""The rollout/update graph should compile once across consecutive updates."""

from JaxSC2.env.env import JaxSC2Env
from algorithms.ppo.trainer import PPOTrainer


def test_ppo_jit_stability(tmp_path):
    class CountTracesEnv(JaxSC2Env):
        trace_count = 0

        def step(self, key, state, action):
            self.trace_count += 1
            return super().step(key, state, action)

    trainer = PPOTrainer({
        "NUM_ENVS": 2, "ROLLOUT_LEN": 4, "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1, "LOG_DIR": str(tmp_path / "logs"),
    })
    trainer.env = CountTracesEnv("V1_Base")
    trainer.train(16)
    assert int(trainer.state.step) == 2
    assert trainer.env.trace_count == 1


if __name__ == "__main__":
    trainer = PPOTrainer({
        "NUM_ENVS": 4, "ROLLOUT_LEN": 16, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1,
    })
    trainer.train(4 * 16)
    print("PPO Smoke Test Passed")
