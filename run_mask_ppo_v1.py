#!/usr/bin/env python
"""Launch MaskPPO experiment on V1_Base variant."""

from algorithms.mask_ppo import MaskPPO
from JaxSC2.env.env import JaxSC2Env


def main():
    config = {
        "NUM_ENVS": 32,
        "ROLLOUT_LEN": 512,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 16,
        "LR": 3e-4,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.995,
        "GAE_LAMBDA": 0.95,
        "ENTROPY_COEFF": 0.01,
        "VF_COEFF": 0.5,
        "LOG_INTERVAL": 10,
        "CKPT_INTERVAL": 200,
        "SEED": 42,
    }

    variant_name = "V1_Base"
    env = JaxSC2Env(variant_name=variant_name)
    model = MaskPPO(config=config)

    print(f"MaskPPO {variant_name} | gamma=0.995 | total_steps=30,000,000")
    model.train(env, total_steps=30_000_000)


if __name__ == "__main__":
    main()
