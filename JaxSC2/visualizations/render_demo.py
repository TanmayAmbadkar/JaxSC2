import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

def fig_to_array(fig):
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba())
    return rgba[:, :, :3]

def get_smart_direction(pos, target):
    unit_x, target_x = pos[0], target[0]
    on_left = unit_x < 14.4
    target_on_right = target_x > 17.6
    on_right = unit_x > 17.6
    target_on_left = target_x < 14.4
    
    if (on_left and target_on_right) or (on_right and target_on_left):
        bridge_y = 8.0 if pos[1] < 16.0 else 24.0
        bridge_x = 14.0 if on_left else 18.0
        sub_target = jnp.array([bridge_x, bridge_y])
    else:
        sub_target = target
        
    diff = sub_target - pos
    angle = (jnp.arctan2(diff[0], diff[1]) + 2*jnp.pi) % (2*jnp.pi)
    return int(jnp.round(angle / (jnp.pi / 4)) % 8)

def run_demonstration(variant="V2_Base", out_path="combat_nav.gif", out_spatial="spatial_obs.gif", mode="smart"):
    env = JaxSC2Env(variant_name=variant, use_spatial_obs=True, resolution=64, enemy_ai=True, enemy_mode="aggressive")
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)
    
    trajectory = [state_to_frame(state)]
    frames_spatial = []
    
    fig_spat, axs_spat = plt.subplots(2, 4, figsize=(16, 8))
    titles = ["Terrain (S)", "Ally Pos (S)", "Enemy Pos (S)", "Ally HP (S)",
              "Terrain (M)", "Ally (M)", "Enemy (M)", "Beacon (M)"]
    
    print(f"Starting {mode} simulation...")
    # 1. Collect Trajectory
    for t in range(200):
        ally_alive = state.smax_state.unit_alive[:5]
        if not jnp.any(ally_alive): break
            
        if mode == "chaos":
            rng, a_rng = jax.random.split(rng)
            verb = jax.random.randint(a_rng, (), 1, 3)
            direction = jax.random.randint(a_rng, (), 0, 8)
            target = jax.random.randint(a_rng, (), 0, 5)
            action = CentralAction(who_mask=ally_alive.astype(jnp.int32), verb=int(verb), direction=int(direction), target=int(target))
        else:
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
            action = CentralAction(who_mask=ally_alive.astype(jnp.int32), verb=verb, direction=direction, target=target)
        
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, _ = env.step(step_rng, state, action)
        trajectory.append(state_to_frame(state))
        
        if mode != "chaos":
            screen, minimap = obs['screen'], obs['minimap'] 
            imgs = [screen[0], screen[5], screen[6], screen[8], minimap[0], minimap[1], minimap[3], minimap[5]]
            for i, ax in enumerate(axs_spat.flatten()):
                ax.clear(); ax.imshow(imgs[i], cmap='viridis' if i in [3, 7] else 'gray'); ax.set_title(titles[i]); ax.axis('off')
            frames_spatial.append(fig_to_array(fig_spat))
        if done: break
            
    # 2. Render Trajectory
    print(f"Rendering {out_path} using ProductionRenderer...")
    try:
        renderer = ProductionRenderer(headless=True, trails_enabled=True)
        renderer.render_episode(trajectory, save_path=out_path, interp_steps=4)
        if mode != "chaos":
            imageio.mimsave(out_spatial, frames_spatial, fps=10)
    except Exception as e: print(f"Rendering failed: {e}")

if __name__ == "__main__":
    run_demonstration(mode="smart")
    run_demonstration(out_path="chaos_nav.gif", mode="chaos")
