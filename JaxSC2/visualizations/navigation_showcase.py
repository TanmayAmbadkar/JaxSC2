"""
JaxSC2 Navigation Showcase — Full Episode Visualizations

Generates GIFs showing navigation scenarios where allies must reach the beacon:
  - 5v3 Navigate — Light resistance, focus on navigation
  - 5v5 Navigate — Medium resistance
  - 5v8 Navigate — Heavy resistance, must fight through

Usage:
  python JaxSC2/visualizations/navigation_showcase.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

OUTPUT_DIR = "JaxSC2/visualizations/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Navigation Agent ────────────────────────────────────────────────────
def navigation_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos,
                     num_allies, combat_cfg):
    """
    Smart navigation agent that prioritizes reaching the beacon:
    - Engages enemies only when in attack range
    - Uses smart pathfinding to cross bridges
    - Focuses on beacon when enemies are dead
    """
    centroid = jnp.mean(ally_pos[ally_alive], axis=0) if jnp.any(ally_alive) else ally_pos[0]
    
    # Direction helper: 0-7 cardinal/intercardinal
    def direction_to(pos):
        diff = pos - centroid
        angle = (jnp.arctan2(diff[0], diff[1]) + 2 * jnp.pi) % (2 * jnp.pi)
        return int(jnp.round(angle / (jnp.pi / 4)) % 8)
    
    # Smart direction with bridge awareness
    def smart_direction_to(pos):
        unit_x = centroid[0]
        target_x = pos[0]
        
        # Check if we need to cross the cliff (x=16 in 32-unit map)
        needs_crossing = (unit_x < 14.4 and target_x > 17.6) or \
                         (unit_x > 17.6 and target_x < 14.4)
        
        if needs_crossing:
            # Aim for nearest bridge
            bridge_y = 8.0 if centroid[1] < 16.0 else 24.0
            bridge_x = 14.0 if unit_x < 16.0 else 18.0
            sub_target = jnp.array([bridge_x, bridge_y])
        else:
            sub_target = pos
        
        return direction_to(sub_target)
    
    # Check if any enemies are in attack range
    dists = jnp.linalg.norm(enemy_pos - centroid[None, :], axis=1)
    valid_dists = jnp.where(enemy_alive, dists, 999.0)
    closest_dist = jnp.min(valid_dists)
    
    in_attack_range = closest_dist < combat_cfg["attack_range"]
    
    # In navigation mode, engage enemies only if they're very close (self-defense)
    self_defense_range = 4.0
    
    if in_attack_range and closest_dist < self_defense_range:
        # Self-defense: attack the closest enemy
        closest = jnp.argmin(valid_dists)
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=2,  # Attack
            direction=0,
            target=int(closest)
        )
    else:
        # Navigate toward beacon (primary objective)
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=1,  # Move
            direction=smart_direction_to(beacon_pos),
            target=0
        )


# ─── Run Scenario ────────────────────────────────────────────────────────
def run_scenario(env, agent_fn, max_steps=300, seed=42):
    """Run a full episode until termination."""
    rng = jax.random.PRNGKey(seed)
    obs, state = env.reset(rng)
    
    trajectory = [state_to_frame(state)]
    ally_hp_history = []
    enemy_hp_history = []
    
    for step in range(max_steps):
        ally_alive = state.smax_state.unit_alive[:env.num_allies]
        
        if not jnp.any(ally_alive):
            break
        
        ally_pos = state.smax_state.unit_positions[:env.num_allies]
        enemy_pos = state.smax_state.unit_positions[env.num_allies:]
        enemy_alive = state.smax_state.unit_alive[env.num_allies:]
        
        action = agent_fn(
            ally_pos=ally_pos,
            ally_alive=ally_alive,
            enemy_pos=enemy_pos,
            enemy_alive=enemy_alive,
            beacon_pos=state.beacon_pos,
            num_allies=env.num_allies,
            combat_cfg=env.combat_cfg
        )
        
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)
        trajectory.append(state_to_frame(state))
        
        ally_hp = jnp.sum(state.smax_state.unit_health[:env.num_allies] * 
                         state.smax_state.unit_alive[:env.num_allies])
        enemy_hp = jnp.sum(state.smax_state.unit_health[env.num_allies:] * 
                          state.smax_state.unit_alive[env.num_allies:])
        ally_hp_history.append(float(ally_hp))
        enemy_hp_history.append(float(enemy_hp))
        
        if done:
            break
    
    # Determine outcome
    ally_alive_final = state.smax_state.unit_alive[:env.num_allies]
    enemy_alive_final = state.smax_state.unit_alive[env.num_allies:]
    
    if jnp.any(ally_alive_final) and info.get("beacon_reached", False):
        outcome = "Allies Win (Navigation)"
    elif jnp.any(ally_alive_final) and not jnp.any(enemy_alive_final):
        outcome = "Allies Win (Combat)"
    elif not jnp.any(ally_alive_final):
        outcome = "Enemies Win"
    else:
        outcome = "Timeout"
    
    return trajectory, {
        "steps": len(trajectory) - 1,
        "outcome": outcome,
        "ally_hp_final": ally_hp_history[-1] if ally_hp_history else 0,
        "enemy_hp_final": enemy_hp_history[-1] if enemy_hp_history else 0,
    }


# ─── Save GIF ────────────────────────────────────────────────────────────
def save_gif(trajectory, metadata, filename):
    """Save trajectory as GIF."""
    renderer = ProductionRenderer(headless=True, trails_enabled=True)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    renderer.render_episode(trajectory, save_path=save_path, interp_steps=4)
    
    print(f"  ✓ {filename}")
    print(f"    Steps: {metadata['steps']} | Outcome: {metadata['outcome']}")
    print(f"    Ally HP: {metadata['ally_hp_final']:.1f} | Enemy HP: {metadata['enemy_hp_final']:.1f}")


# ─── Main Showcase ───────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("JaxSC2 Navigation Showcase — Full Episode Visualizations")
    print("=" * 70)
    print()
    
    scenarios = [
        {
            "variant": "V1_Navigate",
            "enemy_mode": "aggressive",
            "title": "5v3 Navigate — Light Resistance",
            "desc": "Few enemies, focus on beacon navigation and bridge crossing",
            "filename": "navigate_5v3_aggressive.gif",
            "seed": 42,
        },
        {
            "variant": "V2_Navigate",
            "enemy_mode": "aggressive",
            "title": "5v5 Navigate — Medium Resistance",
            "desc": "Balanced fight while navigating toward beacon",
            "filename": "navigate_5v5_aggressive.gif",
            "seed": 123,
        },
        {
            "variant": "V3_Navigate",
            "enemy_mode": "aggressive",
            "title": "5v8 Navigate — Heavy Resistance",
            "desc": "Must fight through 8 enemies to reach beacon",
            "filename": "navigate_5v8_aggressive.gif",
            "seed": 456,
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {scenario['title']}")
        print(f"    {scenario['desc']}")
        
        env = JaxSC2Env(
            variant_name=scenario["variant"],
            enemy_ai=True,
            enemy_mode=scenario["enemy_mode"]
        )
        
        trajectory, metadata = run_scenario(
            env=env,
            agent_fn=navigation_agent,
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
