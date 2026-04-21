#!/usr/bin/env python
"""Launch MaskPPO experiment on V2_Combat variant."""

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

    variant_name = "V2_Combat"
    env = JaxSC2Env(variant_name=variant_name)
    model = MaskPPO(config=config)

    print(f"MaskPPO {variant_name} | gamma=0.995 | total_steps=30,000,000")
    model.train(env, total_steps=30_000_000)

    # Auto-evaluate on latest checkpoint
    import glob as _glob
    import os as _os

    import jax
    import jax.numpy as jnp

    from algorithms.common.checkpoint import load_checkpoint
    from algorithms.common.utils import flatten_obs
    from JaxSC2.env.env import PerUnitAction, build_action_mask

    ckpt_dir = "runs/mask_ppo/checkpoints"
    ckpts = sorted(_glob.glob(f"{ckpt_dir}/ckpt_*.pkl"))
    if not ckpts:
        print("No checkpoints found, skipping eval")
        return

    latest = ckpts[-1]
    trained_state, step = load_checkpoint(latest, model.state)
    params = trained_state.params

    num_episodes = 64
    print(f"\nEvaluating checkpoint {step} ({latest})...")

    eval_env = JaxSC2Env(variant_name=variant_name)
    eval_env.model = model.model

    def _run_episode(ep_rng, rng):
        obs, state = eval_env.reset(rng)

        def _step(carry, _):
            s, o, tr, done = carry

            v_logits, d_logits, t_logits = eval_env.model.apply(
                params, flatten_obs(o)
            )
            mask = build_action_mask(s, eval_env.num_allies)
            v_logits = jnp.where(mask["verb"][:, :, None], v_logits, -1e9)

            verb_idx = jnp.argmax(v_logits, axis=-1)[:, 0]
            dir_idx = jnp.argmax(d_logits, axis=-1)[:, 0]
            tgt_idx = jnp.argmax(t_logits, axis=-1)[:, 0]

            action = PerUnitAction(
                who_mask=jnp.ones((eval_env.num_allies,), dtype=jnp.bool_),
                verb=verb_idx[None, :],
                direction=dir_idx[None, :],
                target=tgt_idx[None, :],
            )

            rng_, s_rng = jax.random.split(rng)
            _, next_s, reward, new_done, _ = eval_env.step(s_rng, s, action)
            reward = jnp.where(done, 0.0, reward)

            next_s = jax.tree_util.tree_map(
                lambda a, b: jnp.where(new_done.astype(jnp.bool_), a, b), s, next_s
            )

            return (next_s, o, tr + reward, new_done), None

        init = (state, obs, 0.0, False)
        final, _ = jax.lax.scan(_step, init, None, length=500)
        return final[0][2]

    rngs = jax.random.split(jax.random.PRNGKey(42), num_episodes)
    rewards = jax.vmap(_run_episode)(rngs, rngs).tolist()

    print(f"\nReward: {jnp.mean(rewards):.2f} ± {jnp.std(rewards):.2f}")
    print(f"Total:  {jnp.sum(rewards):.2f}\n")


if __name__ == "__main__":
    main()
