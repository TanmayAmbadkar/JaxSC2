import jax
import jax.numpy as jnp
from JaxSC2.env.env import JaxSC2Env, CentralAction
from algorithms.common.utils import flatten_obs, decode_action

def evaluate(env, params, model, rng, num_episodes=32, max_steps=400):
    """
    Evaluates the current policy on a set of episodes deterministically.
    Disaggregates success into Navigation (Beacon) and Combat (Kills).
    """
    
    def run_episode(rng):
        obs, state = env.reset(rng)
        
        def _step(carry, _):
            state, obs, total_reward, done, nav_flag, combat_flag, rng = carry
            
            # Deterministic Action (Standard PPO)
            obs_flat = flatten_obs(obs)
            logits, _ = model.apply(params, obs_flat)
            action_idx = jnp.argmax(logits, axis=-1)
            
            verb, direction, target = decode_action(action_idx)
            action = CentralAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
                verb=verb, direction=direction, target=target
            )
            
            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, next_done, info = env.step(step_rng, state, action)
            
            # Masking
            reward = jnp.where(done, 0.0, reward)
            
            # Logic: Outcome only counts if we weren't already finished.
            current_nav = jnp.where(done, 0.0, info["beacon_reached"].astype(jnp.float32))
            current_combat = jnp.where(done, 0.0, info["enemies_killed"].astype(jnp.float32))
            
            new_done = done | next_done
            new_nav_flag = jnp.maximum(nav_flag, current_nav)
            new_combat_flag = jnp.maximum(combat_flag, current_combat)
            
            # Freeze state/obs
            next_state = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), state, next_state)
            next_obs = jnp.where(done, obs, next_obs)
            
            return (next_state, next_obs, total_reward + reward, new_done, new_nav_flag, new_combat_flag, rng), None

        init_carry = (state, obs, 0.0, False, 0.0, 0.0, rng)
        final_carry, _ = jax.lax.scan(_step, init_carry, None, length=max_steps)
        
        # Reward, NavWin, CombatWin
        return final_carry[2], final_carry[4], final_carry[5]

    # Parallelize episodes using vmap
    rngs = jax.random.split(rng, num_episodes)
    rewards, nav_wins, combat_wins = jax.vmap(run_episode)(rngs)
    
    return {
        "eval/mean_reward": jnp.mean(rewards),
        "eval/nav_win_rate": jnp.mean(nav_wins),
        "eval/combat_win_rate": jnp.mean(combat_wins),
        "eval/total_win_rate": jnp.mean(jnp.maximum(nav_wins, combat_wins))
    }
