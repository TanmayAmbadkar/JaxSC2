import jax
import jax.random
import jax.numpy as jnp
import numpy as np
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

def run_interactive_demo():
    print("Launching Interactive TwoBridge UI (Hunter Mode)...")
    env = JaxSC2Env(variant_name="V2_Base", enemy_ai=True, enemy_mode="aggressive")
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)
    
    trajectory = [state_to_frame(state)]
    
    print("Simulating 500 steps for trajectory...")
    from JaxSC2.visualizations.render_demo import get_smart_direction

    for t in range(500):
        if t % 50 == 0: print(f"  Simulating step {t}/500...")
        ally_alive = state.smax_state.unit_alive[:5]
        if not jnp.any(ally_alive): break
        
        # Smart Heuristic (Hunter Mode)
        centroid = jnp.mean(state.smax_state.unit_positions[:5], axis=0, where=ally_alive[:, None])
        enemy_alive = state.smax_state.unit_alive[5:]
        enemy_pos = state.smax_state.unit_positions[5:]
        dist = jnp.linalg.norm(centroid - enemy_pos, axis=-1)
        valid_dist = jnp.where(enemy_alive, dist, 999.0)
        closest = jnp.argmin(valid_dist)
        
        if valid_dist[closest] < 7.0: # In weapon range
            verb, target, direction = 2, int(closest), 0
        else:
            # Seek enemies first, then the beacon
            if enemy_alive.any():
                verb, target, direction = 1, 0, get_smart_direction(centroid, enemy_pos[closest])
            else:
                verb, target, direction = 1, 0, get_smart_direction(centroid, state.beacon_pos)
        
        action = CentralAction(who_mask=ally_alive.astype(np.int32), 
                               verb=verb, 
                               direction=direction, 
                               target=target)
        
        rng, step_rng = jax.random.split(rng)
        obs, state, _, done, _ = env.step(step_rng, state, action)
        trajectory.append(state_to_frame(state))
        if done: break
        
    print(f"Simulation complete ({len(trajectory)} frames).")
    print("Initializing Pygame Window...")
    # Headless=False to open window
    renderer = ProductionRenderer(headless=False, trails_enabled=True)
    renderer.run_interactive(trajectory, interp_steps=4, fps=60)

if __name__ == "__main__":
    run_interactive_demo()
