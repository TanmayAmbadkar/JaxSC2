"""Episode handling and masks shared by training and evaluation."""

import jax
import jax.numpy as jnp

from JaxSC2.env.env import PerUnitAction
from algorithms.common.utils import flatten_obs


def step_and_reset(env, rng, state, action):
    """Return the terminal reward/done, but start the next episode when done."""
    step_rng, reset_rng = jax.random.split(rng)
    obs, next_state, reward, done, _ = env.step(step_rng, state, action)
    obs, next_state = jax.lax.cond(
        done, lambda _: env.reset(reset_rng), lambda _: (obs, next_state), None
    )
    return flatten_obs(obs), next_state, reward, done


def policy_masks(env, state):
    """Masks for an army-level command, with a harmless unused-target fallback."""
    mask = env.get_action_mask(state)
    targets = mask["target"][:env.num_enemies].astype(jnp.bool_)
    # Target/direction heads are sampled even for NOOP. Keep the unused target
    # distribution defined when no attack is available; ATTACK stays masked.
    targets = targets.at[0].set(targets[0] | ~jnp.any(targets))
    return {
        "verb": mask["verb"].astype(jnp.bool_),
        "direction": mask["direction"].astype(jnp.bool_),
        "target": targets,
    }


def evaluate_episodes(env, params, model, rng, num_episodes, max_steps, masked=False):
    """Run deterministic episodes, freezing state and accumulating flags at done."""
    def run_episode(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        def step(carry, _):
            state, obs, reward_sum, done, nav, combat, key = carry
            vector = flatten_obs(obs)
            if masked:
                mask = policy_masks(env, state)
                verb, direction, target = model.apply(
                    params, vector, mask["verb"], mask["direction"], mask["target"]
                )
            else:
                output = model.apply(params, vector)
                verb, direction, target = (
                    output["verb_logits"], output["direction_logits"], output["target_logits"]
                )
            action = PerUnitAction(
                state.smax_state.unit_alive[:env.num_allies],
                jnp.full(env.num_allies, jnp.argmax(verb), dtype=jnp.int32),
                jnp.full(env.num_allies, jnp.argmax(direction), dtype=jnp.int32),
                jnp.full(env.num_allies, jnp.argmax(target), dtype=jnp.int32),
            )
            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, terminal, info = env.step(step_key, state, action)
            next_state = jax.tree_util.tree_map(
                lambda old, new: jnp.where(done, old, new), state, next_state
            )
            next_obs = jax.tree_util.tree_map(
                lambda old, new: jnp.where(done, old, new), obs, next_obs
            )
            return (
                next_state, next_obs, reward_sum + jnp.where(done, 0., reward),
                done | terminal, nav | (~done & info["nav_success"]),
                combat | (~done & info["combat_success"]), key,
            ), None

        final, _ = jax.lax.scan(
            step, (state, obs, 0., False, False, False, key), None, length=max_steps
        )
        return final[2], final[4], final[5]

    return jax.vmap(run_episode)(jax.random.split(rng, num_episodes))


def evaluation_metrics(rewards, nav, combat):
    return {
        "eval/mean_reward": jnp.mean(rewards),
        "eval/nav_win_rate": jnp.mean(nav),
        "eval/combat_win_rate": jnp.mean(combat),
        "eval/total_win_rate": jnp.mean(nav | combat),
    }
