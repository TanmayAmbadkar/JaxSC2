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

from algorithms.common.checkpoint import load_checkpoint
from algorithms.common.utils import flatten_obs
from JaxSC2.env.env import PerUnitAction, build_action_mask


def evaluate_masked_policy(env, params, rng, num_episodes=64, max_steps=500):
    """Evaluate MaskPPO model.

    MaskPPO model returns (v_logits, d_logits, t_logits) tuple when called
    with dummy masks. We pass all-ones dummy masks to trigger this path,
    then apply real action masks externally (matching mask_ppo/eval.py).

    MaskPPO model returns combined logits without masks, making separate
    head access impossible - dummy masks are required.
    """
    num_enemies = env.num_enemies

    dummy_verb_mask = jnp.ones((1, 3), dtype=jnp.bool_)
    dummy_dir_mask = jnp.ones((1, 8), dtype=jnp.bool_)
    dummy_tgt_mask = jnp.ones((1, num_enemies), dtype=jnp.bool_)

    def run_episode(rng, dummy_v, dummy_d, dummy_t):
        obs, state = env.reset(rng)

        def _step(carry, _):
            state, obs, total_reward, done = carry

            v_logits, d_logits, t_logits = env.model.apply(
                params, flatten_obs(obs), dummy_v, dummy_d, dummy_t
            )

            mask = build_action_mask(state, env.num_allies)
            v_logits = jnp.where(mask["verb"][:, :, None], v_logits, -1e9)

            verb_idx = jnp.argmax(v_logits, axis=-1)[:, 0]
            dir_idx = jnp.argmax(d_logits, axis=-1)[:, 0]
            tgt_idx = jnp.argmax(t_logits, axis=-1)[:, 0]

            action = PerUnitAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
                verb=verb_idx[None, :],
                direction=dir_idx[None, :],
                target=tgt_idx[None, :],
            )

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, new_done, info = env.step(
                step_rng, state, action
            )

            reward = jnp.where(done, 0.0, reward)

            next_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done, a, b), state, next_state
            )
            next_obs = jnp.where(done.astype(jnp.bool_), obs, next_obs)

            return (next_state, next_obs, total_reward + reward, new_done), None

        init_carry = (state, obs, 0.0, False)
        final_carry, _ = jax.lax.scan(_step, init_carry, None, length=max_steps)
        _, _, total_reward, _ = final_carry

        return total_reward


    rngs = jax.random.split(rng, num_episodes)
    rewards = jax.vmap(
        lambda r: run_episode(r, dummy_verb_mask, dummy_dir_mask, dummy_tgt_mask)
    )(rngs)

    def get_flags(ep_rng):
        obs, state = env.reset(ep_rng)

        def _step(carry, _):
            state, obs, done = carry

            v_logits, d_logits, t_logits = env.model.apply(
                params, flatten_obs(obs), dummy_verb_mask, dummy_dir_mask, dummy_tgt_mask
            )

            mask = build_action_mask(state, env.num_allies)
            v_logits = jnp.where(mask["verb"][:, :, None], v_logits, -1e9)

            verb_idx = jnp.argmax(v_logits, axis=-1)[:, 0]
            dir_idx = jnp.argmax(d_logits, axis=-1)[:, 0]
            tgt_idx = jnp.argmax(t_logits, axis=-1)[:, 0]

            action = PerUnitAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
                verb=verb_idx[None, :],
                direction=dir_idx[None, :],
                target=tgt_idx[None, :],
            )

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, new_done, info = env.step(
                step_rng, state, action
            )

            nav_flag = info["beacon_reached"].astype(jnp.bool_) & ~done
            combat_flag = (info["enemies_killed"].astype(jnp.bool_) > 0) & ~done
            new_done = done | new_done

            next_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.astype(jnp.bool_), a, b), state, next_state
            )
            next_obs = jnp.where(done.astype(jnp.bool_), obs, next_obs)

            return (next_state, next_obs, new_done), (nav_flag, combat_flag)

        init_carry = (state, obs, False)
        final_carry, flags = jax.lax.scan(_step, init_carry, None, length=max_steps)
        _, _, _ = final_carry
        nav_flags, combat_flags = flags

        return jnp.any(nav_flags), jnp.any(combat_flags)

    rngs = jax.random.split(rng, num_episodes)
    nav_wins_arr, combat_wins_arr = jax.vmap(get_flags)(rngs)

    total_rewards = float(jnp.sum(rewards))
    reward_mean = float(jnp.mean(rewards))
    reward_std = float(jnp.std(rewards)) if num_episodes > 1 else 0.0

    nav_win_rate = float(jnp.mean(nav_wins_arr))
    combat_win_rate = float(jnp.mean(combat_wins_arr))
    total_win_rate = float(jnp.maximum(nav_win_rate, combat_win_rate))

    return {
        "total_rewards": rewards.tolist(),
        "nav_wins": nav_wins_arr.tolist(),
        "combat_wins": combat_wins_arr.tolist(),
    }, {
        "num_episodes": int(num_episodes),
        "metrics": {
            "reward_total": total_rewards,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
        },
        "win_rates": {
            "nav_win_rate": nav_win_rate,
            "combat_win_rate": combat_win_rate,
            "total_win_rate": total_win_rate,
        },
    }


def evaluate_standard_policy(env, params, rng, num_episodes=64, max_steps=500):
    """Evaluate Standard PPO model (MultiHeadActorCritic).

    MultiHeadActorCritic returns dict {verb_logits, direction_logits, target_logits}
    which is indexed directly without needing dummy masks.
    """

    def run_episode(rng):
        obs, state = env.reset(rng)

        def _step(carry, _):
            state, obs, total_reward, done = carry

            output = env.model.apply(params, flatten_obs(obs))

            mask = build_action_mask(state, env.num_allies)

            verb_logits = jnp.where(
                mask["verb"][:, :, None], output["verb_logits"], -1e9
            )
            dir_logits = jnp.where(
                mask["direction"][:, :, None], output["direction_logits"], -1e9
            )
            tgt_logits = jnp.where(
                mask["target"][:, :, None], output["target_logits"], -1e9
            )

            verb_idx = jnp.argmax(verb_logits, axis=-1)[:, 0]
            dir_idx = jnp.argmax(dir_logits, axis=-1)[:, 0]
            tgt_idx = jnp.argmax(tgt_logits, axis=-1)[:, 0]

            action = PerUnitAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
                verb=verb_idx[None, :],
                direction=dir_idx[None, :],
                target=tgt_idx[None, :],
            )

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, new_done, info = env.step(
                step_rng, state, action
            )

            reward = jnp.where(done, 0.0, reward)

            next_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.astype(jnp.bool_), a, b), state, next_state
            )
            next_obs = jnp.where(done.astype(jnp.bool_), obs, next_obs)

            return (next_state, next_obs, total_reward + reward, new_done), None

        init_carry = (state, obs, 0.0, False)
        final_carry, _ = jax.lax.scan(_step, init_carry, None, length=max_steps)
        _, _, total_reward, _ = final_carry

        return total_reward


    rngs = jax.random.split(rng, num_episodes)
    rewards = jax.vmap(run_episode)(rngs)

    def get_flags(ep_rng):
        obs, state = env.reset(ep_rng)

        def _step(carry, _):
            state, obs, done = carry

            output = env.model.apply(params, flatten_obs(obs))
            mask = build_action_mask(state, env.num_allies)

            verb_logits = jnp.where(
                mask["verb"][:, :, None], output["verb_logits"], -1e9
            )
            dir_logits = jnp.where(
                mask["direction"][:, :, None], output["direction_logits"], -1e9
            )
            tgt_logits = jnp.where(
                mask["target"][:, :, None], output["target_logits"], -1e9
            )

            verb_idx = jnp.argmax(verb_logits, axis=-1)[:, 0]
            dir_idx = jnp.argmax(dir_logits, axis=-1)[:, 0]
            tgt_idx = jnp.argmax(tgt_logits, axis=-1)[:, 0]

            action = PerUnitAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
                verb=verb_idx[None, :],
                direction=dir_idx[None, :],
                target=tgt_idx[None, :],
            )

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, new_done, info = env.step(
                step_rng, state, action
            )

            nav_flag = info["beacon_reached"].astype(jnp.bool_) & ~done
            combat_flag = (info["enemies_killed"].astype(jnp.bool_) > 0) & ~done
            new_done = done | new_done

            next_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.astype(jnp.bool_), a, b), state, next_state
            )
            next_obs = jnp.where(done.astype(jnp.bool_), obs, next_obs)

            return (next_state, next_obs, new_done), (nav_flag, combat_flag)

        init_carry = (state, obs, False)
        final_carry, flags = jax.lax.scan(_step, init_carry, None, length=max_steps)
        _, _, _ = final_carry
        nav_flags, combat_flags = flags

        return jnp.any(nav_flags), jnp.any(combat_flags)

    rngs = jax.random.split(rng, num_episodes)
    nav_wins_arr, combat_wins_arr = jax.vmap(get_flags)(rngs)

    reward_mean = float(jnp.mean(rewards))
    reward_std = float(jnp.std(rewards)) if num_episodes > 1 else 0.0

    nav_win_rate = float(jnp.mean(nav_wins_arr))
    combat_win_rate = float(jnp.mean(combat_wins_arr))
    total_win_rate = float(jnp.maximum(nav_win_rate, combat_win_rate))

    return {
        "total_rewards": rewards.tolist(),
        "nav_wins": nav_wins_arr.tolist(),
        "combat_wins": combat_wins_arr.tolist(),
    }, {
        "num_episodes": int(num_episodes),
        "metrics": {
            "reward_total": float(jnp.sum(rewards)),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
        },
        "win_rates": {
            "nav_win_rate": nav_win_rate,
            "combat_win_rate": combat_win_rate,
            "total_win_rate": total_win_rate,
        },
    }


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

    trained_state, loaded_step = load_checkpoint(args.ckpt, main_model.state)
    params = trained_state.params

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
