import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Dict

@dataclass
class FogState:
    last_seen_pos: jnp.ndarray   # (num_enemies, 2)
    last_seen_alive: jnp.ndarray # (num_enemies,)
    last_seen_hp: jnp.ndarray    # (num_enemies,)
    last_seen_time: jnp.ndarray  # (num_enemies,)

def apply_mass_collisions(
    unit_pos: jnp.ndarray,      # (N, 2)
    unit_alive: jnp.ndarray,    # (N,)
    unit_types: jnp.ndarray,    # (N,)
    type_radius: jnp.ndarray,   # (T,)
    type_mass: jnp.ndarray,     # (T,)
    eps: float = 1e-6,
    max_step_scale: float = 0.5
):
    N = unit_pos.shape[0]

    pos_i = unit_pos[:, None, :]        # (N,1,2)
    pos_j = unit_pos[None, :, :]        # (1,N,2)
    diff = pos_i - pos_j                # (N,N,2)

    dist = jnp.linalg.norm(diff, axis=-1)  # (N,N)

    # Radii + masses
    r_i = type_radius[unit_types][:, None]
    r_j = type_radius[unit_types][None, :]
    m_i = type_mass[unit_types][:, None]
    m_j = type_mass[unit_types][None, :]

    min_dist = r_i + r_j

    # Mask for alive and not self (vectorized, no jnp.eye allocation)
    _not_self = (jnp.arange(N)[:, None] != jnp.arange(N)[None, :])
    valid_mask = (unit_alive[:, None] & unit_alive[None, :]) & _not_self

    overlap = (dist < min_dist) & (dist > eps) & valid_mask
    penetration = (min_dist - dist) * overlap

    # Stable direction
    direction = jnp.where(dist[..., None] > eps, diff / (dist[..., None] + eps), 0.0)

    # Mass-weighted split
    total_mass = m_i + m_j + eps
    w_i = (m_j / total_mass)  # how much i moves due to j
    # (N,N,1)
    w_i = w_i[..., None]

    disp_ij = penetration[..., None] * direction * w_i  # (N,N,2)

    total_disp = jnp.sum(disp_ij, axis=1)  # (N,2)

    # Stability clamp
    max_step = jnp.take(type_radius, unit_types) * max_step_scale
    max_step = max_step[:, None]
    disp_norm = jnp.linalg.norm(total_disp, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_step / (disp_norm + eps))

    total_disp = total_disp * scale

    return unit_pos + total_disp

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

    mask_not_self = (jnp.arange(N)[:, None] != jnp.arange(N)[None, :]).astype(pos.dtype)
    alive_mask = (alive[:, None] * alive[None, :]) * mask_not_self

    # Enforce separation: radius * 2
    min_dist = 2 * unit_radius
    # Fallback clamp: apply only when penetration exceeds 20% of radius to prevent double-counting soft forces
    threshold = unit_radius * 0.2
    overlap = (dist < (min_dist - threshold)) * (dist > eps) * alive_mask
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
    new_vel = new_vel * jnp.minimum(1.0, jnp.asarray(max_speed)[..., None] / (v_norm + 1e-6))
    
    # 3. Update Position
    new_pos = pos + new_vel
    return new_pos, new_vel

def compute_visibility(ally_pos, ally_alive, enemy_pos, vision_radius):
    diff = ally_pos[:, None, :] - enemy_pos[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    return jnp.any((dists < vision_radius) & ally_alive[:, None], axis=0)

def update_fog_memory(state: FogState, ally_pos, ally_alive, enemy_pos, enemy_alive, enemy_hp, vision_radius, current_time):
    visible = compute_visibility(ally_pos, ally_alive, enemy_pos, vision_radius)
    
    new_pos = jnp.where(visible[:, None], enemy_pos, state.last_seen_pos)
    new_alive = jnp.where(visible, enemy_alive, state.last_seen_alive)
    new_hp = jnp.where(visible, enemy_hp, state.last_seen_hp)
    new_time = jnp.where(visible, current_time, state.last_seen_time)
    
    return state.replace(
        last_seen_pos=new_pos,
        last_seen_alive=new_alive,
        last_seen_hp=new_hp,
        last_seen_time=new_time
    ), visible

def update_projectiles(proj_pos, proj_vel, proj_target, proj_active, proj_team, unit_pos, unit_alive, unit_teams):
    """Intersect each traveled segment with enemy circles; choose first contact."""
    new_p_pos = proj_pos + proj_vel
    offset = proj_pos[:, None, :] - unit_pos[None, :, :]
    speed_sq = jnp.sum(proj_vel ** 2, axis=-1)[:, None]
    b = jnp.sum(offset * proj_vel[:, None, :], axis=-1)
    c = jnp.sum(offset ** 2, axis=-1) - 0.6 ** 2
    discriminant = b ** 2 - speed_sq * c
    entry = (-b - jnp.sqrt(jnp.maximum(discriminant, 0.))) / jnp.maximum(speed_sq, 1e-12)
    entry = jnp.where(c <= 0., 0., entry)
    intersects = (discriminant >= 0.) & (entry >= 0.) & (entry <= 1.)
    intersects &= (speed_sq > 1e-12) | (c <= 0.)
    is_enemy = proj_team[:, None] != unit_teams[None, :]
    potential_hits = proj_active[:, None] & unit_alive[None, :] & is_enemy & intersects
    hit_triggered = jnp.any(potential_hits, axis=1)
    hit_targets = jnp.argmin(jnp.where(potential_hits, entry, jnp.inf), axis=1).astype(jnp.int32)
    
    safe_targets = jnp.where(proj_target >= 0, proj_target, 0).astype(jnp.int32)
    target_pos = unit_pos[safe_targets]
    dist_to_intended = jnp.linalg.norm(new_p_pos - target_pos, axis=-1)
    lost = proj_active & (dist_to_intended > 40.0)
    
    new_p_active = proj_active & (~hit_triggered) & (~lost)
    return new_p_pos, new_p_active, hit_triggered, hit_targets

def apply_high_fidelity_combat(
    unit_health, unit_alive, unit_positions, unit_types, unit_teams, unit_armor,
    attack_timers, targets, proj_state, combat_cfg: Dict
):
    windups = combat_cfg["type_windups"][unit_types]
    cooldowns = combat_cfg["type_cooldowns"][unit_types]
    total_cycle = windups + cooldowns
    bonus_matrix = combat_cfg.get("bonus_matrix", jnp.zeros_like(combat_cfg["damage_matrix"]))
    N_units = unit_health.shape[0]
    
    valid_target = (targets >= 0) & (targets < N_units)
    target_idx = jnp.clip(targets, 0, N_units - 1).astype(jnp.int32)
    t_pos = unit_positions[target_idx]
    dist = jnp.linalg.norm(unit_positions - t_pos, axis=-1)
    in_range = (unit_alive) & (targets >= 0) & (dist < combat_cfg["type_ranges"][unit_types])
    
    can_attack = (unit_alive & valid_target & in_range & unit_alive[target_idx]
                  & (unit_teams != unit_teams[target_idx]))
    # Once a shot fires, cooldown continues even if the target dies or leaves.
    recovering = attack_timers >= windups
    new_timers = jnp.where(recovering | can_attack, attack_timers + 1, 0).astype(jnp.int32)
    firing = (new_timers == windups) & can_attack & ~recovering
    
    melee_fire = firing & (unit_types == 0)
    ranged_fire = firing & (unit_types != 0)
    
    p_pos, p_vel, p_target, p_damage, p_active, p_team = proj_state[:6]
    # Older callers may pass six arrays containing already-mitigated damage.
    p_types = proj_state[6] if len(proj_state) == 7 else jnp.full_like(p_target, -1)
    
    # 1. Update Projectiles
    p_pos, p_active, hit_triggered, hit_targets = update_projectiles(
        p_pos, p_vel, p_target, p_active, p_team, unit_positions, unit_alive, unit_teams
    )
    # Preserve impact damage before free slots can be reused by new shots.
    shot_types = jnp.maximum(p_types, 0)
    hit_types = unit_types[hit_targets]
    effective_damage = jnp.maximum(
        0.5, p_damage * combat_cfg["damage_matrix"][shot_types, hit_types]
        + bonus_matrix[shot_types, hit_types] - unit_armor[hit_targets],
    )
    hit_damage = jnp.where(hit_triggered, jnp.where(p_types < 0, p_damage, effective_damage), 0.)
    
    # 2. Spawn logic with Slot Allocation
    vec_all = t_pos - unit_positions
    vel_all = (vec_all / jnp.maximum(1e-6, jnp.linalg.norm(vec_all, axis=-1, keepdims=True))) * 1.5
    
    slot_mask = ~p_active # (32,)
    
    def spawn_one(state, i):
        p_pos, p_vel, p_target, p_damage, p_active, p_team, p_types, free_slots, d_count = state
        should_fire = ranged_fire[i]
        
        # Find first free slot
        idx = jnp.argmax(free_slots)
        can_fire_idx = should_fire & free_slots[idx]
        
        # Update if firing
        p_pos = p_pos.at[idx].set(jnp.where(can_fire_idx, unit_positions[i], p_pos[idx]))
        p_vel = p_vel.at[idx].set(jnp.where(can_fire_idx, vel_all[i], p_vel[idx]))
        p_target = p_target.at[idx].set(jnp.where(can_fire_idx, target_idx[i], p_target[idx]))
        p_active = p_active.at[idx].set(jnp.where(can_fire_idx, True, p_active[idx]))
        p_team = p_team.at[idx].set(jnp.where(can_fire_idx, unit_teams[i], p_team[idx]))
        
        # Store the weapon's base damage/type, and mitigate against the unit
        # actually hit later (which can differ from the intended target).
        p_damage = p_damage.at[idx].set(jnp.where(
            can_fire_idx, combat_cfg["type_damages"][unit_types[i]], p_damage[idx]))
        p_types = p_types.at[idx].set(jnp.where(can_fire_idx, unit_types[i], p_types[idx]))
        
        # Consume slot
        new_free_slots = free_slots.at[idx].set(jnp.where(can_fire_idx, False, free_slots[idx]))
        
        # Track saturation: if should fire but NO slot was free
        dropped = should_fire & (~free_slots[idx])
        new_d_count = d_count + dropped.astype(jnp.int32)
        
        return (p_pos, p_vel, p_target, p_damage, p_active, p_team, p_types, new_free_slots, new_d_count), None

    # We use scan to allocate slots one by one for firing units
    init_dropped = jnp.int32(0)
    final_p_st, _ = jax.lax.scan(spawn_one, (p_pos, p_vel, p_target, p_damage, p_active, p_team, p_types, slot_mask, init_dropped), jnp.arange(N_units))
    p_pos, p_vel, p_target, p_damage, p_active, p_team, p_types, _, dropped_shots = final_p_st

    new_timers = jnp.where(new_timers >= total_cycle, 0, new_timers)
    
    # c. Damage Resolution
    def compute_effective_damage(atk_types, def_types, base_dmg, def_armor):
        mult = combat_cfg["damage_matrix"][atk_types, def_types]
        bonus = bonus_matrix[atk_types, def_types]
        raw = base_dmg * mult + bonus
        return jnp.maximum(0.5, raw - def_armor) # Minimum 0.5 damage

    melee_dmg_val = compute_effective_damage(
        unit_types, unit_types[target_idx],
        combat_cfg["type_damages"][unit_types],
        unit_armor[target_idx]
    )
    
    damage_to_apply = jnp.zeros_like(unit_health)
    damage_to_apply = damage_to_apply.at[target_idx].add(jnp.where(melee_fire, melee_dmg_val, 0.0))
    damage_to_apply = damage_to_apply.at[hit_targets].add(hit_damage)
    
    new_health = jnp.clip(unit_health - damage_to_apply, 0.0, 200.0)
    new_alive = new_health > 0
    if "suicide_type" in combat_cfg:
        # Optional archetype effect shared by either team. Dead units cannot
        # detonate; multiple blasts add, with no friendly-fire damage.
        distances = jnp.linalg.norm(unit_positions[:, None, :] - unit_positions[None, :, :], axis=-1)
        hostile = unit_teams[:, None] != unit_teams[None, :]
        triggered = (new_alive & (unit_types == combat_cfg["suicide_type"]) &
                     jnp.any(hostile & new_alive[None, :] &
                             (distances < combat_cfg["suicide_trigger"]), axis=1))
        blasts = (triggered[:, None] & hostile &
                  (distances < combat_cfg["suicide_radius"]))
        new_health = jnp.maximum(0., new_health -
                                 jnp.sum(blasts, axis=0) * combat_cfg["suicide_damage"])
        new_health = jnp.where(triggered, 0., new_health)
        new_alive = new_health > 0
    final_proj_state = (p_pos, p_vel, p_target, p_damage, p_active, p_team, p_types)
    
    return new_health, new_alive, new_timers, final_proj_state, dropped_shots

def update_persistent_targets(current_targets, unit_positions, unit_alive, target_pool_positions, target_pool_alive, unit_teams, pool_teams, unit_ranges, leash_extra: float = 2.0, pool_visible=None):
    """
    SC2-style persistent targeting. Units stick to a target unless it dies or leaves leash range.
    """
    N = unit_positions.shape[0]
    
    # 1. Check current target validity
    safe_targets = jnp.where(current_targets >= 0, current_targets, 0)
    target_alive = target_pool_alive[safe_targets]
    
    diff = unit_positions[:, None, :] - target_pool_positions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)
    
    dist_to_current = jnp.linalg.norm(unit_positions - target_pool_positions[safe_targets], axis=-1)
    
    # Leash check
    is_valid = (current_targets >= 0) & target_alive & (dist_to_current < (unit_ranges + leash_extra))
    if pool_visible is not None:
        is_valid &= pool_visible[jnp.arange(N), safe_targets]
    
    # 2. Pick new closest if invalid
    # Mask out dead units AND teammates
    enemy_mask = (unit_teams[:, None] != pool_teams[None, :])
    if pool_visible is not None:
        enemy_mask &= pool_visible
    valid_pool_dists = jnp.where(target_pool_alive[None, :] & enemy_mask, dists, jnp.inf)
    new_closest = jnp.argmin(valid_pool_dists, axis=1)
    
    # Ensure if NO enemy alive, we set to -1
    no_enemy = jnp.all(valid_pool_dists == jnp.inf, axis=1)
    new_closest = jnp.where(no_enemy, -1, new_closest)
    
    # Only update if NOT valid
    final_targets = jnp.where(is_valid, current_targets, new_closest)
    return final_targets
