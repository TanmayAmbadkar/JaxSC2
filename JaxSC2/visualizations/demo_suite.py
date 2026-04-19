"""
Comprehensive JaxSC2 Demo Suite — Full Episode Visualizations

Generates GIFs for all combat and navigation scenarios with different enemy AI modes.
Each demo runs until episode termination (beacon reached, all enemies dead, or all allies dead).

Scenarios:
  Combat: V1 (5v3), V2 (5v5), V3 (5v8) × Static/Guard/Patrol/Aggressive
  Navigate: V1 (5v3), V2 (5v5), V3 (5v8) × Static/Guard/Patrol/Aggressive

Usage:
  python JaxSC2/visualizations/demo_suite.py          # Run all demos
  python JaxSC2/visualizations/demo_suite.py --mode combat   # Combat only
  python JaxSC2/visualizations/demo_suite.py --mode navigate # Navigate only
  python JaxSC2/visualizations/demo_suite.py --variant V1_Base # Single variant
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

OUTPUT_DIR = "JaxSC2/visualizations/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Smart Heuristic Agent ───────────────────────────────────────────────
def get_smart_action(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos,
                     num_allies, combat_cfg):
    """
    Smart heuristic that makes the demo visually interesting:
    - Engages enemies in range with appropriate targeting
    - Flanks and maneuvers when out of range
    - Prioritizes beacon in navigation mode
    """
    centroid = jnp.mean(ally_pos[ally_alive], axis=0) if jnp.any(ally_alive) else ally_pos[0]
    
    # Find closest alive enemy
    valid_dists = jnp.where(enemy_alive, 
                            jnp.linalg.norm(enemy_pos - centroid[None, :], axis=1), 
                            999.0)
    closest_enemy = jnp.argmin(valid_dists)
    dist_to_closest = valid_dists[closest_enemy]
    
    in_attack_range = dist_to_closest < combat_cfg["attack_range"]
    
    # Direction helper: 0-7 cardinal/intercardinal
    def direction_to(pos):
        diff = pos - centroid
        angle = (jnp.arctan2(diff[0], diff[1]) + 2 * jnp.pi) % (2 * jnp.pi)
        return int(jnp.round(angle / (jnp.pi / 4)) % 8)
    
    # Bridge crossing helper for navigation
    def smart_direction_to(pos):
        unit_x = centroid[0]
        target_x = pos[0]
        
        # Check if we need to cross the cliff
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
    
    if in_attack_range and dist_to_closest < 999.0:
        # Attack the closest enemy
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=2,  # Attack
            direction=0,
            target=int(closest_enemy)
        )
    else:
        # Move toward closest enemy or beacon
        if dist_to_closest < 999.0:
            # Pursue enemy
            target_pos = enemy_pos[closest_enemy]
        else:
            # No enemies alive, move to beacon
            target_pos = beacon_pos
        
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=1,  # Move
            direction=smart_direction_to(target_pos),
            target=0
        )


# ─── Demo Runner ──────────────────────────────────────────────────────────
def run_demo(variant_name, enemy_mode, mode_type, rng_seed=42, max_steps=300):
    """
    Run a single scenario and return trajectory + metadata.
    
    Args:
        variant_name: e.g., "V1_Base", "V2_Combat", "V3_Navigate"
        enemy_mode: "static", "guard", "patrol", "aggressive"
        mode_type: "combat" or "navigate" (for labeling)
        rng_seed: random seed for reproducibility
        max_steps: maximum steps per episode
    
    Returns:
        trajectory, metadata dict
    """
    env = JaxSC2Env(
        variant_name=variant_name,
        enemy_ai=True,
        enemy_mode=enemy_mode
    )
    
    rng = jax.random.PRNGKey(rng_seed)
    obs, state = env.reset(rng)
    
    trajectory = [state_to_frame(state)]
    ally_hp_history = []
    enemy_hp_history = []
    
    for step in range(max_steps):
        ally_alive = state.smax_state.unit_alive[:env.num_allies]
        
        if not jnp.any(ally_alive):
            break
        
        # Get smart action for allies
        ally_pos = state.smax_state.unit_positions[:env.num_allies]
        enemy_pos = state.smax_state.unit_positions[env.num_allies:]
        enemy_alive = state.smax_state.unit_alive[env.num_allies:]
        
        action = get_smart_action(
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
        
        # Record HP for metadata
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
    
    if jnp.any(ally_alive_final) and not jnp.any(enemy_alive_final):
        outcome = "Allies Win (Combat)"
    elif jnp.any(ally_alive_final) and info.get("beacon_reached", False):
        outcome = "Allies Win (Navigation)"
    elif not jnp.any(ally_alive_final):
        outcome = "Enemies Win"
    else:
        outcome = "Timeout"
    
    metadata = {
        "variant": variant_name,
        "enemy_mode": enemy_mode,
        "mode_type": mode_type,
        "steps": len(trajectory) - 1,
        "outcome": outcome,
        "ally_hp_final": ally_hp_history[-1] if ally_hp_history else 0,
        "enemy_hp_final": enemy_hp_history[-1] if enemy_hp_history else 0,
        "ally_count": env.num_allies,
        "enemy_count": env.num_enemies,
    }
    
    return trajectory, metadata


def save_gif(trajectory, metadata, filename):
    """Save trajectory as GIF with metadata overlay."""
    renderer = ProductionRenderer(headless=True, trails_enabled=True)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    renderer.render_episode(trajectory, save_path=save_path, interp_steps=4)
    
    print(f"  ✓ Saved: {save_path} ({metadata['steps']} steps, {metadata['outcome']})")


# ─── Main Demo Suite ──────────────────────────────────────────────────────
def run_all_demos(mode_filter=None, variant_filter=None):
    """
    Run all demo scenarios and generate GIFs.
    
    Args:
        mode_filter: "combat", "navigate", or None for all
        variant_filter: e.g., "V1_Base" for single variant, or None for all
    """
    variants = ["V1_Base", "V2_Combat", "V3_Navigate"]
    enemy_modes = ["static", "guard", "patrol", "aggressive"]
    
    print("=" * 70)
    print("JaxSC2 Demo Suite — Full Episode Visualizations")
    print("=" * 70)
    
    if mode_filter:
        print(f"Mode filter: {mode_filter}")
    if variant_filter:
        print(f"Variant filter: {variant_filter}")
    
    print()
    
    demo_configs = []
    
    for variant in variants:
        # Determine mode type from variant name
        if "Combat" in variant:
            mode_type = "combat"
        elif "Navigate" in variant:
            mode_type = "navigate"
        else:  # Base mode — show both combat and navigate
            demo_configs.append((variant, "guard", "combat"))
            demo_configs.append((variant, "aggressive", "navigate"))
            continue
        
        # Apply mode filter
        if mode_filter and mode_type != mode_filter:
            continue
        
        for enemy_mode in enemy_modes:
            demo_configs.append((variant, enemy_mode, mode_type))
    
    # Apply variant filter
    if variant_filter:
        demo_configs = [c for c in demo_configs if c[0] == variant_filter]
    
    print(f"Running {len(demo_configs)} demos...")
    print()
    
    for i, (variant, enemy_mode, mode_type) in enumerate(demo_configs):
        print(f"[{i+1}/{len(demo_configs)}] {variant} vs {enemy_mode} ({mode_type})")
        
        # Map variant to seed for reproducibility
        seed_map = {
            "V1_Base": 42,
            "V2_Combat": 123,
            "V3_Navigate": 456,
        }
        
        try:
            trajectory, metadata = run_demo(
                variant_name=variant,
                enemy_mode=enemy_mode,
                mode_type=mode_type,
                rng_seed=seed_map.get(variant, 42),
                max_steps=300
            )
            
            # Generate filename
            mode_short = {"combat": "C", "navigate": "N"}.get(mode_type, "?")
            filename = f"{variant}_{enemy_mode[:4]}{mode_short}_full.gif"
            
            save_gif(trajectory, metadata, filename)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print()
    
    print("=" * 70)
    print(f"Done! All GIFs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    mode_filter = None
    variant_filter = None
    
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode_filter = sys.argv[idx + 1]
    
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        variant_filter = sys.argv[idx + 1]
    
    run_all_demos(mode_filter=mode_filter, variant_filter=variant_filter)
