"""Compatibility entry point for the original PPOTrainer API."""

from JaxSC2.env.env import JaxSC2Env
from algorithms.ppo.ppo import PPO


class PPOTrainer(PPO):
    def __init__(self, config=None):
        super().__init__(config)
        self.env = JaxSC2Env(variant_name=self.config.get("VARIANT_NAME", "V1_Base"))

    def train(self, total_steps):
        return super().train(self.env, total_steps)
