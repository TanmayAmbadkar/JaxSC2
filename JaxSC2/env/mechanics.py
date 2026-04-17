import jax
import jax.numpy as jnp
from typing import Dict

def apply_hard_collisions(unit_positions, unit_alive, unit_radius: float, stiffness: float = 1.2):
    """
    Hard body blocking. Stronger repulsion than soft collision.
    """
    pos = unit_positions
    alive = unit_alive
    N = pos.shape[0]
    eps = 1e-6

    pos_i = pos[:, None, :]
    pos_j = pos[None, :, :]
    diff = pos_i - pos_j
    dist = jnp.linalg.norm(diff, axis=-1)

    mask_not_self = 1.0 - jnp.eye(N, dtype=pos.dtype)
    alive_mask = (alive[:, None] * alive[None, :]) * mask_not_self

    # Enforce separation: radius * 2
    min_dist = 2 * unit_radius
    overlap = (dist < min_dist) * (dist > eps) * alive_mask
    penetration = (min_dist - dist) * overlap

    direction = jnp.where(dist[..., None] > eps, diff / (dist[..., None] + eps), 0.0)
    
    # Stiffness factor makes collisions "harder"
    displacement = 0.5 * stiffness * penetration[..., None] * direction
    total_disp = jnp.sum(displacement, axis=1)

    # Clamp to prevent instability while ensuring separation
    max_disp = unit_radius * 0.8
    disp_norm = jnp.linalg.norm(total_disp, axis=-1, keepdims=True)
    total_disp = total_disp * jnp.minimum(1.0, max_disp / (disp_norm + eps))

    return pos + total_disp

def integrate_velocity(pos, vel, accel_vec, max_speed, friction=0.9):
    """
    Continuous double-integrator physics.
    """
    # 1. Update Velocity
    new_vel = (vel + accel_vec) * friction
    
    v_norm = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
    new_vel = new_vel * jnp.minimum(1.0, max_speed / (v_norm + 1e-6))
    
    # 3. Update Position
    new_pos = pos + new_vel
    return new_pos, new_vel

def compute_visibility(ally_pos, ally_alive, enemy_pos, vision_radius: float):
    """
    Computes visibility mask for enemies based on distance to alive allies.
    Returns: (num_enemies,) boolean mask.
    """
    # Pairwise distances: (N_allies, N_enemies)
    diff = ally_pos[:, None, :] - enemy_pos[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # Visibility condition per (ally, enemy) pair
    # Mask out dead allies
    visible_matrix = (dists < vision_radius) & ally_alive[:, None]
    
    # Visible if ANY ally sees the enemy
    return jnp.any(visible_matrix, axis=0)  # (num_enemies,)

def update_projectiles(proj_pos, proj_vel, proj_target, proj_active, proj_team, unit_pos, unit_alive, unit_teams):
    """
    Ballistic projectiles: they hit the FIRST enemy they collide with.
    """
    new_p_pos = proj_pos + proj_vel # dt=1.0
    
    # 1. Pairwise Distance Check (32 projectiles x N units)
    # dists shape: (32, N)
    diff = new_p_pos[:, None, :] - unit_pos[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    
    # 2. Collision Conditions
    # Hit if: projectile active, unit alive, unit is enemy, and distance < 0.6
    is_enemy = proj_team[:, None] != unit_teams[None, :]
    potential_hits = proj_active[:, None] & unit_alive[None, :] & is_enemy & (dists < 0.6)
    
    # 3. Resolve first hit per projectile
    hit_triggered = jnp.any(potential_hits, axis=1)
    hit_targets = jnp.argmax(potential_hits, axis=1).astype(jnp.int32) # Returns first enemy index hit
    
    # 4. Cleanup: lost projectiles (traveled too far or hit)
    # Note: we still use proj_target for distance-to-intent checks to prevent infinite flight
    safe_targets = jnp.where(proj_target >= 0, proj_target, 0).astype(jnp.int32)
    target_pos = unit_pos[safe_targets]
    dist_to_intended = jnp.linalg.norm(new_p_pos - target_pos, axis=-1)
    lost = proj_active & (dist_to_intended > 40.0)
    
    new_p_active = proj_active & (~hit_triggered) & (~lost)
    
    return new_p_pos, new_p_active, hit_triggered, hit_targets

def apply_high_fidelity_combat(
    unit_health,
    unit_alive,
    unit_positions,
    unit_types,
    unit_teams,
    attack_timers,
    targets,
    proj_state, # (p_pos, p_vel, p_target, p_damage, p_active, p_team)
    combat_cfg: Dict
):
    """
    Dual-phase combat: Windup -> Fire (Spawn/Instant) -> Cooldown.
    """
    windup = combat_cfg["windup"]
    cooldown = combat_cfg.get("cooldown", 8)
    total_cycle = windup + cooldown
    N_units = unit_health.shape[0]
    
    can_attack = unit_alive & (targets >= 0)
    new_timers = jnp.where(can_attack, attack_timers + 1, 0).astype(jnp.int32)
    firing = (new_timers == windup) & can_attack
    
    target_idx = jnp.where(targets >= 0, targets, 0).astype(jnp.int32)
    t_pos = unit_positions[target_idx]
    dist = jnp.linalg.norm(unit_positions - t_pos, axis=-1)
    in_range = dist < combat_cfg["type_ranges"][unit_types]
    
    melee_fire = firing & (unit_types == 0) & in_range
    ranged_fire = firing & (unit_types == 1) & in_range
    
    p_pos, p_vel, p_target, p_damage, p_active, p_team = proj_state
    
    # a. Update existing projectiles (Ballistic hit resolution)
    p_pos, p_active, hit_triggered, hit_targets = update_projectiles(
        p_pos, p_vel, p_target, p_active, p_team, unit_positions, unit_alive, unit_teams
    )
    
    # b. Spawn new projectiles
    vec_all = t_pos - unit_positions
    vel_all = (vec_all / jnp.maximum(1e-6, jnp.linalg.norm(vec_all, axis=-1, keepdims=True))) * 1.5
    
    p_pos = p_pos.at[:N_units].set(jnp.where(ranged_fire[:, None], unit_positions, p_pos[:N_units]))
    p_vel = p_vel.at[:N_units].set(jnp.where(ranged_fire[:, None], vel_all, p_vel[:N_units]))
    p_target = p_target.at[:N_units].set(jnp.where(ranged_fire, target_idx, p_target[:N_units]).astype(jnp.int32))
    p_active = p_active.at[:N_units].set(jnp.where(ranged_fire, True, p_active[:N_units]))
    p_damage = p_damage.at[:N_units].set(jnp.where(ranged_fire, combat_cfg["type_damages"][unit_types], p_damage[:N_units]))
    p_team = p_team.at[:N_units].set(jnp.where(ranged_fire, unit_teams[:N_units], p_team[:N_units]).astype(jnp.int32))
    
    new_timers = jnp.where(new_timers >= total_cycle, 0, new_timers)
    
    # c. Damage Resolution
    damage_matrix = combat_cfg["damage_matrix"]
    melee_dmg_val = combat_cfg["type_damages"][unit_types] * damage_matrix[unit_types, unit_types[target_idx]]
    
    damage_to_apply = jnp.zeros_like(unit_health)
    damage_to_apply = damage_to_apply.at[target_idx].add(jnp.where(melee_fire, melee_dmg_val, 0.0))
    damage_to_apply = damage_to_apply.at[hit_targets].add(jnp.where(hit_triggered, p_damage, 0.0))
    
    new_health = jnp.clip(unit_health - damage_to_apply, 0.0, 200.0)
    new_alive = new_health > 0
    final_proj_state = (p_pos, p_vel, p_target, p_damage, p_active, p_team)
    
    return new_health, new_alive, new_timers, final_proj_state

def update_persistent_targets(current_targets, unit_positions, unit_alive, target_pool_positions, target_pool_alive, unit_ranges, leash_extra: float = 2.0):
    """
    SC2-style persistent targeting. Units stick to a target unless it dies or leaves leash range.
    """
    N = current_targets.shape[0]
    
    # 1. Check current target validity
    safe_targets = jnp.where(current_targets >= 0, current_targets, 0)
    target_alive = target_pool_alive[safe_targets]
    
    diff = unit_positions[:, None, :] - target_pool_positions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    
    # Row i corresponds to unit i's distance to its current target
    # This is a bit inefficient to compute all dists, but cleaner for JAX vmap
    # Actually let's just compute the distance to the safe target
    dist_to_current = jnp.linalg.norm(unit_positions - target_pool_positions[safe_targets], axis=-1)
    
    # Leash: range + extra margin to prevent jitter
    is_valid = (current_targets >= 0) & target_alive & (dist_to_current < (unit_ranges + leash_extra))
    
    # 2. Pick new closest if invalid
    # Mask out dead enemies and find min dist
    valid_pool_dists = jnp.where(target_pool_alive[None, :], dists, 999.0)
    new_closest = jnp.argmin(valid_pool_dists, axis=1)
    
    # Only update if NOT valid
    final_targets = jnp.where(is_valid, current_targets, new_closest)
    return final_targets
