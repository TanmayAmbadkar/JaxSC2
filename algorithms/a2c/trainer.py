"""A2C reuses the batched rollout engine with one unclipped gradient update."""

from JaxSC2.env.env import JaxSC2Env
from algorithms.ppo.ppo import PPO
from algorithms.a2c.a2c_logic import a2c_loss_multi_head


class A2CTrainer(PPO):
    loss_fn = staticmethod(a2c_loss_multi_head)

    def __init__(self, config=None):
        settings = {
            "NUM_ENVS": 64, "ROLLOUT_LEN": 128, "LR": 7e-4,
            "GAMMA": 0.99, "GAE_LAMBDA": 1.0, "VARIANT_NAME": "V1_Base",
            "LOG_DIR": "runs/a2c/logs", "CKPT_DIR": "runs/a2c/checkpoints",
        }
        settings.update(config or {})
        # A2C takes a single update over the complete on-policy rollout.
        settings.update(UPDATE_EPOCHS=1, NUM_MINIBATCHES=1)
        super().__init__(settings)
        self.env = JaxSC2Env(variant_name=self.config["VARIANT_NAME"])

    def train(self, total_steps):
        return super().train(self.env, total_steps)


if __name__ == "__main__":
    A2CTrainer().train(100000)
