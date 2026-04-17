import jax.numpy as jnp

def compute_rewards_and_dones(state, next_smax_state, num_allies, num_enemies, map_width, map_height):
    """
    Computes the multi-objective reward shaping and exclusive terminal conditions.
    """
    map_dims = jnp.array([map_width, map_height])
    
    ally_pos = next_smax_state.unit_positions[:num_allies]
    ally_alive = next_smax_state.unit_alive[:num_allies]
    ally_health = next_smax_state.unit_health[:num_allies]
    
    enemy_pos = next_smax_state.unit_positions[num_allies:]
    enemy_alive = next_smax_state.unit_alive[num_allies:]
    enemy_health = next_smax_state.unit_health[num_allies:]
    
    # ---------------------------------------------------------
    # 1. Navigation Shaping (Beacon)
    # ---------------------------------------------------------
    dists_to_beacon = jnp.linalg.norm((ally_pos - state.beacon_pos) / map_dims, axis=-1)
    mean_dist_beacon = jnp.where(jnp.sum(ally_alive) > 0, 
                                 jnp.sum(dists_to_beacon * ally_alive) / jnp.sum(ally_alive), 
                                 state.prev_mean_dist)
    r_nav = state.prev_mean_dist - mean_dist_beacon
    
    # ---------------------------------------------------------
    # 2. Combat Shaping (3-Part Design)
    # ---------------------------------------------------------
    # A. Health Based
    current_enemy_hp = jnp.sum(enemy_health)
    current_ally_hp = jnp.sum(ally_health)
    delta_enemy_hp = state.prev_enemy_health - current_enemy_hp
    delta_ally_hp = state.prev_ally_health - current_ally_hp
    r_combat_health = (delta_enemy_hp - delta_ally_hp) / 100.0 # Scaled down
    
    # B. Spatial Based (Distance to enemy centroid)
    enemy_centroid = jnp.where(jnp.sum(enemy_alive) > 0,
                               jnp.sum(enemy_pos * enemy_alive[:, None], axis=0) / jnp.sum(enemy_alive),
                               jnp.zeros(2))
    
    dists_to_enemy = jnp.linalg.norm((ally_pos - enemy_centroid) / map_dims, axis=-1)
    mean_dist_enemy = jnp.where((jnp.sum(ally_alive) > 0) & (jnp.sum(enemy_alive) > 0),
                                jnp.sum(dists_to_enemy * ally_alive) / jnp.sum(ally_alive),
                                state.prev_enemy_centroid_dist)
    r_combat_spatial = state.prev_enemy_centroid_dist - mean_dist_enemy
    
    # C. Event Based (Kills/Losses)
    prev_ally_alive = state.smax_state.unit_alive[:num_allies]
    prev_enemy_alive = state.smax_state.unit_alive[num_allies:]
    
    kills = jnp.sum(prev_enemy_alive.astype(jnp.int32)) - jnp.sum(enemy_alive.astype(jnp.int32))
    losses = jnp.sum(prev_ally_alive.astype(jnp.int32)) - jnp.sum(ally_alive.astype(jnp.int32))
    r_combat_events = (kills * 1.0) - (losses * 1.0)
    
    r_combat = r_combat_health + r_combat_spatial + r_combat_events
    
    # ---------------------------------------------------------
    # 3. Terminal Conditions
    # ---------------------------------------------------------
    beacon_reached = jnp.any((dists_to_beacon < 0.05) & ally_alive)
    enemies_all_dead = jnp.sum(enemy_alive) == 0
    allies_all_dead = jnp.sum(ally_alive) == 0
    timeout = state.timestep >= 300
    
    is_tie = enemies_all_dead & allies_all_dead
    done = beacon_reached | enemies_all_dead | allies_all_dead | timeout
    
    r_term = jnp.where(
        beacon_reached, 25.0,
        jnp.where(is_tie, 0.0,
        jnp.where(enemies_all_dead, 10.0,
        jnp.where(allies_all_dead, -10.0,
        jnp.where(timeout, -15.0, 0.0))))
    )
    
    # ---------------------------------------------------------
    # 4. Total Reward (With Scaling)
    # ---------------------------------------------------------
    # Research-grade scaling to balance Nav vs Combat
    r_nav_scaled = 2.0 * r_nav
    r_combat_scaled = 0.5 * r_combat
    
    total_reward = r_nav_scaled + r_combat_scaled + r_term
    
    return total_reward, done, mean_dist_beacon, current_enemy_hp, current_ally_hp, mean_dist_enemy
