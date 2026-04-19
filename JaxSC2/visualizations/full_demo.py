"""
JaxSC2 Full Demo Suite — All Scenarios

Generates GIFs for all combat and navigation scenarios with different enemy AI modes.
Runs on the twobridge conda environment.

Scenarios:
  Combat: V2_Combat (5v5) × Guard/Aggressive
  Navigate: V1_Navigate (5v3), V2_Navigate (5v5), V3_Navigate (5v8) × Aggressive

Usage:
  PYTHONPATH=/path/to/Sc2Jax conda run -n twobridge python JaxSC2/visualizations/full_demo.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['SDL_VIDEODRIVER'] = 'dummy'

import jax
import jax.numpy as jnp
import numpy as np
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Smart Combat Agent ──────────────────────────────────────────────────
def combat_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos,
                 num_allies, combat_cfg):
    centroid = jnp.mean(ally_pos[ally_alive], axis=0) if jnp.any(ally_alive) else ally_pos[0]
    
    dists = jnp.linalg.norm(enemy_pos - centroid[None, :], axis=1)
    valid_dists = jnp.where(enemy_alive, dists, 999.0)
    closest = jnp.argmin(valid_dists)
    
    if valid_dists[closest] < combat_cfg["attack_range"]:
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=2, direction=0, target=int(closest)
        )
    else:
        angle = (jnp.arctan2(
            enemy_pos[closest][0] - centroid[0],
            enemy_pos[closest][1] - centroid[1]
        ) + 2 * jnp.pi) % (2 * jnp.pi)
        direction = int(jnp.round(angle / (jnp.pi / 4)) % 8)
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=1, direction=int(direction), target=0
        )


# ─── Smart Navigation Agent ──────────────────────────────────────────────
def navigation_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos,
                     num_allies, combat_cfg):
    centroid = jnp.mean(ally_pos[ally_alive], axis=0) if jnp.any(ally_alive) else ally_pos[0]
    
    dists = jnp.linalg.norm(enemy_pos - centroid[None, :], axis=1)
    valid_dists = jnp.where(enemy_alive, dists, 999.0)
    closest_dist = jnp.min(valid_dists)
    
    # Navigate toward beacon, only engage if very close (self-defense)
    if closest_dist < 4.0:
        closest = jnp.argmin(valid_dists)
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=2, direction=0, target=int(closest)
        )
    else:
        # Smart direction with bridge awareness
        unit_x = centroid[0]
        target_x = beacon_pos[0]
        needs_crossing = (unit_x < 14.4 and target_x > 17.6) or \
                         (unit_x > 17.6 and target_x < 14.4)
        
        if needs_crossing:
            bridge_y = 8.0 if centroid[1] < 16.0 else 24.0
            bridge_x = 14.0 if unit_x < 16.0 else 18.0
            target_pos = jnp.array([bridge_x, bridge_y])
        else:
            target_pos = beacon_pos
        
        angle = (jnp.arctan2(target_pos[0] - centroid[0], target_pos[1] - centroid[1]) + 2 * jnp.pi) % (2 * jnp.pi)
        direction = int(jnp.round(angle / (jnp.pi / 4)) % 8)
        
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=1, direction=int(direction), target=0
        )


# ─── Run Scenario ────────────────────────────────────────────────────────
def run_scenario(env, agent_fn, max_steps=300, seed=42):
    rng = jax.random.PRNGKey(seed)
    obs, state = env.reset(rng)
    
    trajectory = [state_to_frame(state)]
    
    for step in range(max_steps):
        ally_alive = state.smax_state.unit_alive[:env.num_allies]
        
        if not jnp.any(ally_alive):
            break
        
        ally_pos = state.smax_state.unit_positions[:env.num_allies]
        enemy_pos = state.smax_state.unit_positions[env.num_allies:]
        enemy_alive = state.smax_state.unit_alive[env.num_allies:]
        
        action = agent_fn(
            ally_pos=ally_pos, ally_alive=ally_alive,
            enemy_pos=enemy_pos, enemy_alive=enemy_alive,
            beacon_pos=state.beacon_pos,
            num_allies=env.num_allies,
            combat_cfg=env.combat_cfg
        )
        
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)
        trajectory.append(state_to_frame(state))
        
        if done:
            break
    
    ally_alive_final = state.smax_state.unit_alive[:env.num_allies]
    enemy_alive_final = state.smax_state.unit_alive[env.num_allies:]
    
    if jnp.any(ally_alive_final) and not jnp.any(enemy_alive_final):
        outcome = "Allies Win (Combat)"
    elif jnp.any(ally_alive_final) and info.get("beacon_reached", False):
        outcome = "Allies Win (Nav)"
    elif not jnp.any(ally_alive_final):
        outcome = "Enemies Win"
    else:
        outcome = "Timeout"
    
    return trajectory, {
        "steps": len(trajectory) - 1,
        "outcome": outcome,
    }


# ─── Save GIF ────────────────────────────────────────────────────────────
def save_gif(trajectory, metadata, filename):
    renderer = ProductionRenderer(headless=True, trails_enabled=True)
    save_path = os.path.join(OUTPUT_DIR, filename)
    renderer.render_episode(trajectory, save_path=save_path, interp_steps=4)
    
    print(f"  ✓ {filename} ({metadata['steps']} steps, {metadata['outcome']})")


# ─── Main Demo Suite ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("JaxSC2 Full Demo Suite — All Scenarios")
    print("=" * 70)
    print()
    
    scenarios = [
        # Combat scenarios
        {
            "variant": "V2_Combat",
            "enemy_mode": "guard",
            "agent_fn": combat_agent,
            "title": "5v5 Combat — Guard AI",
            "filename": "combat_5v5_guard.gif",
            "seed": 42,
        },
        {
            "variant": "V2_Combat",
            "enemy_mode": "aggressive",
            "agent_fn": combat_agent,
            "title": "5v5 Combat — Aggressive AI",
            "filename": "combat_5v5_aggressive.gif",
            "seed": 123,
        },
        # Navigation scenarios
        {
            "variant": "V1_Navigate",
            "enemy_mode": "aggressive",
            "agent_fn": navigation_agent,
            "title": "5v3 Navigate — Aggressive AI",
            "filename": "navigate_5v3_aggressive.gif",
            "seed": 42,
        },
        {
            "variant": "V2_Navigate",
            "enemy_mode": "aggressive",
            "agent_fn": navigation_agent,
            "title": "5v5 Navigate — Aggressive AI",
            "filename": "navigate_5v5_aggressive.gif",
            "seed": 123,
        },
        {
            "variant": "V3_Navigate",
            "enemy_mode": "aggressive",
            "agent_fn": navigation_agent,
            "title": "5v8 Navigate — Aggressive AI",
            "filename": "navigate_5v8_aggressive.gif",
            "seed": 456,
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {scenario['title']}")
        
        env = JaxSC2Env(
            variant_name=scenario["variant"],
            enemy_ai=True,
            enemy_mode=scenario["enemy_mode"]
        )
        
        trajectory, metadata = run_scenario(
            env=env,
            agent_fn=scenario["agent_fn"],
            max_steps=300,
            seed=scenario["seed"]
        )
        
        save_gif(trajectory, metadata, scenario["filename"])
        print()
    
    print("=" * 70)
    print(f"All GIFs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
