import jax
import jax.numpy as jnp
from algorithms.common.utils import flatten_obs, decode_action
from JaxSC2.env.twobridge import CentralAction, get_action_masks

def evaluate(env, params, model, rng, num_episodes=16, max_steps=500):
    def run_episode(rng):
        # Step logic
        def _step(carry, _):
            state, obs, total_reward, done, nav_flag, combat_flag, rng = carry
            
            # Deterministic Action with Masking
            obs_flat = flatten_obs(obs)
            mask = get_action_masks(state, env.num_allies)
            logits, _ = model.apply(params, obs_flat, mask)
            action_idx = jnp.argmax(logits, axis=-1)
            
            verb, direction, target = decode_action(action_idx)
            action = CentralAction(
                who_mask=jnp.ones((env.num_allies,), dtype=jnp.int32),
                verb=verb, direction=direction, target=target
            )
            
            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, new_done, info = env.step(step_rng, state, action)
            
            # Masking
            reward = jnp.where(done, 0.0, reward)
            
            # Nav/Combat Win Tracking (Guarded by ~done)
            new_nav_flag = nav_flag | (info["nav_success"] & ~done)
            new_combat_flag = combat_flag | (info["combat_success"] & ~done)
            
            # Freeze state/obs
            next_state = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), state, next_state)
            next_obs = jnp.where(done, obs, next_obs)
            
            return (next_state, next_obs, total_reward + reward, new_done, new_nav_flag, new_combat_flag, rng), None

        rng, reset_rng = jax.random.split(rng)
        init_obs, init_state = env.reset(reset_rng)
        init_carry = (init_state, init_obs, 0.0, False, False, False, rng)

        final_carry, _ = jax.lax.scan(_step, init_carry, None, length=max_steps)
        _, _, episode_reward, _, nav_win, combat_win, _ = final_carry
        
        return episode_reward, jnp.float32(nav_win), jnp.float32(combat_win)

    rngs = jax.random.split(rng, num_episodes)
    rewards, nav_wins, combat_wins = jax.vmap(run_episode)(rngs)
    
    total_wins = jnp.maximum(nav_wins, combat_wins)
    
    return {
        "eval/mean_reward": jnp.mean(rewards),
        "eval/nav_win_rate": jnp.mean(nav_wins),
        "eval/combat_win_rate": jnp.mean(combat_wins),
        "eval/total_win_rate": jnp.mean(total_wins)
    }
