#!/usr/bin/env python
"""Benchmark MaskPPO throughput for ~1.64M steps (100 iterations)."""

from algorithms.mask_ppo import MaskPPO
from JaxSC2.env.env import JaxSC2Env


def main():
    config = {
        "NUM_ENVS": 32,
        "ROLLOUT_LEN": 512,
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 16,
        "LR": 2e-4,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.995,
        "GAE_LAMBDA": 0.95,
        "ENTROPY_COEFF": 0.01,
        "VF_COEFF": 0.5,
        "LOG_INTERVAL": 100,   # single log at end of 100 iters
        "CKPT_INTERVAL": 9999, # no checkpoints during benchmark
        "SEED": 42,
    }

    variant_name = "V3_Navigate"
    env = JaxSC2Env(variant_name=variant_name)
    model = MaskPPO(config=config)

    batch_size = config["NUM_ENVS"] * config["ROLLOUT_LEN"]
    total_steps = 100 * batch_size

    print(f"=== Benchmark: {total_steps:,} steps ({100} iterations) ===")
    print(f"NUM_ENVS={config['NUM_ENVS']} | ROLLOUT_LEN={config['ROLLOUT_LEN']}")
    print(f"UPDATE_EPOCHS={config['UPDATE_EPOCHS']} | NUM_MINIBATCHES={config['NUM_MINIBATCHES']}")
    print(f"Variant: {variant_name}")
    print()

    model.train(env, total_steps=total_steps)


if __name__ == "__main__":
    main()
