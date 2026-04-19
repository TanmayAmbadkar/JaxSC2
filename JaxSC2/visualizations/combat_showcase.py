"""
JaxSC2 Combat Showcase — Full Episode Visualizations

Generates GIFs showing the most visually interesting combat scenarios:
  - 5v5 Balanced (Guard AI) — shows tactical positioning
  - 5v5 Aggressive — shows full-scale combat
  - 5v8 Overwhelming (Aggressive) — shows desperate defense

Usage:
  python JaxSC2/visualizations/combat_showcase.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

OUTPUT_DIR = "JaxSC2/visualizations/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Smart Combat Agent ──────────────────────────────────────────────────
def combat_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos,
                 num_allies, combat_cfg):
    """
    Smart combat agent that makes visually interesting decisions:
    - Focuses fire on weakest visible enemy
    - Flanks when enemies are in range
    - Retreats when low HP
    """
    centroid = jnp.mean(ally_pos[ally_alive], axis=0) if jnp.any(ally_alive) else ally_pos[0]
    
    # Calculate distances to all enemies
    dists = jnp.linalg.norm(enemy_pos - centroid[None, :], axis=1)
    valid_dists = jnp.where(enemy_alive, dists, 999.0)
    
    # Find closest enemy
    closest = jnp.argmin(valid_dists)
    dist_to_closest = valid_dists[closest]
    
    in_range = dist_to_closest < combat_cfg["attack_range"]
    
    # Direction helper: 0-7 cardinal/intercardinal
    def direction_to(pos):
        diff = pos - centroid
        angle = (jnp.arctan2(diff[0], diff[1]) + 2 * jnp.pi) % (2 * jnp.pi)
        return int(jnp.round(angle / (jnp.pi / 4)) % 8)
    
    if in_range and dist_to_closest < 999.0:
        # Attack the closest enemy with focused fire
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=2,  # Attack
            direction=0,
            target=int(closest)
        )
    else:
        # Move toward closest enemy to engage
        target_pos = enemy_pos[closest] if dist_to_closest < 999.0 else beacon_pos
        return CentralAction(
            who_mask=jnp.ones(num_allies, dtype=jnp.bool_),
            verb=1,  # Move
            direction=direction_to(target_pos),
            target=0
        )


# ─── Run Single Scenario ─────────────────────────────────────────────────
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
        "ally_hp_final": ally_hp_history[-1] if ally_hp_history else 0,
        "enemy_hp_final": enemy_hp_history[-1] if enemy_hp_history else 0,
    }


# ─── Save GIF with Stats Overlay ─────────────────────────────────────────
def save_gif(trajectory, metadata, filename, env_info):
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
    print("JaxSC2 Combat Showcase — Full Episode Visualizations")
    print("=" * 70)
    print()
    
    scenarios = [
        {
            "variant": "V2_Combat",
            "enemy_mode": "guard",
            "title": "5v5 Balanced — Guard AI",
            "desc": "Enemies hold position, allies must close distance",
            "filename": "combat_5v5_guard.gif",
            "seed": 42,
        },
        {
            "variant": "V2_Combat",
            "enemy_mode": "aggressive",
            "title": "5v5 Aggressive — Full Combat",
            "desc": "Enemies pursue and attack immediately",
            "filename": "combat_5v5_aggressive.gif",
            "seed": 123,
        },
        {
            "variant": "V3_Combat",
            "enemy_mode": "aggressive",
            "title": "5v8 Overwhelming — Desperate Defense",
            "desc": "Outnumbered 5v8, allies must survive and fight back",
            "filename": "combat_5v8_aggressive.gif",
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
            agent_fn=combat_agent,
            max_steps=300,
            seed=scenario["seed"]
        )
        
        save_gif(trajectory, metadata, scenario["filename"], {})
        print()
    
    print("=" * 70)
    print(f"All GIFs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
