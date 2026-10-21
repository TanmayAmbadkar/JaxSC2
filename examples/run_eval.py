#!/usr/bin/env python
"""Evaluation script for trained agents.

Usage:
    python run_eval.py --algo mask_ppo --ckpt runs/mask_ppo/checkpoints/ckpt_29491200.pkl --variant V2_Combat
    python run_eval.py --algo ppo --ckpt runs/ppo/checkpoints/ckpt_10M.pkl --variant V3_Navigate --num_episodes 64
    python run_eval.py --algo mask_ppo --ckpt checkpoint.pkl --variant V2_Combat --output results.json
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.getcwd())

def _evaluate_policy(env, params, rng, num_episodes, max_steps, masked):
    from algorithms.common.rollout import evaluate_episodes

    rewards, nav, combat = evaluate_episodes(
        env, params, env.model, rng, num_episodes, max_steps, masked=masked
    )
    results = {
        "total_rewards": rewards.tolist(),
        "nav_wins": nav.tolist(),
        "combat_wins": combat.tolist(),
    }
    return results, compute_stats(results)


def evaluate_masked_policy(env, params, rng, num_episodes=64, max_steps=500):
    return _evaluate_policy(env, params, rng, num_episodes, max_steps, masked=True)


def evaluate_standard_policy(env, params, rng, num_episodes=64, max_steps=500):
    return _evaluate_policy(env, params, rng, num_episodes, max_steps, masked=False)


def compute_stats(results):
    rewards = jnp.array(results["total_rewards"])
    nav_wins = jnp.array(results["nav_wins"], dtype=jnp.float32)
    combat_wins = jnp.array(results["combat_wins"], dtype=jnp.float32)
    total_wins = jnp.maximum(nav_wins, combat_wins)

    return {
        "num_episodes": int(len(rewards)),
        "metrics": {
            "reward_total": float(jnp.sum(rewards)),
            "reward_mean": float(jnp.mean(rewards)),
            "reward_std": float(jnp.std(rewards)) if len(rewards) > 1 else 0.0,
        },
        "win_rates": {
            "nav_win_rate": float(jnp.mean(nav_wins)),
            "combat_win_rate": float(jnp.mean(combat_wins)),
            "total_win_rate": float(jnp.mean(total_wins)),
        },
    }


def save_json(stats, results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    clean_episodes = []
    for i, (rew, nav, combat) in enumerate(
        zip(results["total_rewards"], results["nav_wins"], results["combat_wins"])
    ):
        clean_episodes.append({
            "episode_id": i,
            "total_reward": float(rew),
            "nav_win": bool(nav),
            "combat_win": bool(combat),
        })

    output = {
        "stats": stats,
        "episodes": clean_episodes,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


def print_summary(stats):
    m = stats["metrics"]
    w = stats["win_rates"]

    print("\n" + "=" * 50)
    print(f"Evaluation Results ({stats['num_episodes']} episodes)")
    print("=" * 50)

    print(f"\nReward:")
    print(f"   Total: {m['reward_total']:.2f}")
    print(f"   Mean ± Std: {m['reward_mean']:.2f} ± {m['reward_std']:.2f}")

    print(f"\nWin Rates:")
    print(f"   Navigation (beacon):  {w['nav_win_rate']:.3f}")
    print(f"   Combat (kills):       {w['combat_win_rate']:.3f}")
    print(f"   Total (nav or combat): {w['total_win_rate']:.3f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent on JaxSC2")
    parser.add_argument(
        "--algo", type=str, choices=["mask_ppo", "ppo"], default="mask_ppo"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to trained checkpoint (.pkl)"
    )
    parser.add_argument(
        "--variant", type=str, default="V2_Combat", help="Environment variant (e.g. V1_Base, V2_Combat)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=64, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    args = parser.parse_args()

    from JaxSC2.env.env import JaxSC2Env
    env = JaxSC2Env(variant_name=args.variant)

    if args.algo == "mask_ppo":
        from algorithms.mask_ppo import MaskPPO
        main_model = MaskPPO(config={"NUM_ENVS": 1})
    else:
        from algorithms.ppo import PPO
        main_model = PPO(config={"NUM_ENVS": 1})

    main_model.load(args.ckpt)
    main_model.train(env, total_steps=0)
    loaded_step = main_model.total_steps
    params = main_model.state.params

    env.model = main_model.model

    print(f"Loaded checkpoint (step {loaded_step}) from {args.ckpt}")
    print(f"Evaluating on variant: {args.variant} ({env.num_allies}v{env.num_enemies})")

    rng = jax.random.PRNGKey(args.seed)

    if args.algo == "mask_ppo":
        results, stats = evaluate_masked_policy(
            env, params, rng, args.num_episodes, args.max_steps
        )
    else:
        results, stats = evaluate_standard_policy(
            env, params, rng, args.num_episodes, args.max_steps
        )

    print_summary(stats)
    save_json(stats, results, args.output)


if __name__ == "__main__":
    main()
