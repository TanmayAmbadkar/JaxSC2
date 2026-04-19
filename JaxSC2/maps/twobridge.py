import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict
import chex

class VariantConfig(NamedTuple):
    id: int
    name: str
    n_ally: int
    n_enemy: int
    layout_type: int # 0: Base, 1: Combat, 2: Navigate

class TwoBridgeMap:
    """
    Encapsulates the 'Two-Bridge' scenario logic:
    - Regional layouts (R1-R6)
    - Cliff/Bridge terrain constraints
    - Navigation and Combat reward shaping
    """
    def __init__(self):
        # 32x32 Normalized Coordinates
        self.REGION_COORDS = jnp.array([
            [0.1, 0.3, 0.7, 0.9], # 0: R1 (L_TOP)
            [0.1, 0.3, 0.4, 0.6], # 1: R2 (L_MID)
            [0.1, 0.3, 0.1, 0.3], # 2: R3 (L_BOT)
            [0.7, 0.9, 0.7, 0.9], # 3: R4 (R_TOP)
            [0.7, 0.9, 0.4, 0.6], # 4: R5 (R_MID)
            [0.7, 0.9, 0.1, 0.3], # 5: R6 (R_BOT)
        ])
        
        self.LEFT_REGIONS = jnp.array([0, 1, 2])
        self.RIGHT_REGIONS = jnp.array([3, 4, 5])
        
        self.VARIANTS = {
            "V1_Base":     VariantConfig(0, "V1_Base",     5, 3, 0),
            "V1_Combat":   VariantConfig(1, "V1_Combat",   5, 3, 1),
            "V1_Navigate": VariantConfig(2, "V1_Navigate", 5, 3, 2),
            "V2_Base":     VariantConfig(3, "V2_Base",     5, 5, 0),
            "V2_Combat":   VariantConfig(4, "V2_Combat",   5, 5, 1),
            "V2_Navigate": VariantConfig(5, "V2_Navigate", 5, 5, 2),
            "V3_Base":     VariantConfig(6, "V3_Base",     5, 8, 0),
            "V3_Combat":   VariantConfig(7, "V3_Combat",   5, 8, 1),
            "V3_Navigate": VariantConfig(8, "V3_Navigate", 5, 8, 2),
        }

    def get_spawn_regions(self, rng, layout_type):
        """Unified region selection for Two-Bridge."""
        rng_ally, rng_enemy, rng_beacon = jax.random.split(rng, 3)
        
        def select_base():
            a_idx = jax.random.choice(rng_ally, self.LEFT_REGIONS)
            b_idx = jax.random.choice(rng_beacon, self.RIGHT_REGIONS)
            e_mask = (self.RIGHT_REGIONS != b_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(e_mask), 1.0)
            e_probs = e_mask / denom
            e_idx = jax.random.choice(rng_enemy, self.RIGHT_REGIONS, p=e_probs)
            return a_idx, e_idx, b_idx
            
        def select_combat():
            b_idx = jax.random.choice(rng_beacon, self.LEFT_REGIONS)
            a_idx = jax.random.choice(rng_ally, self.RIGHT_REGIONS)
            e_mask = (self.RIGHT_REGIONS != a_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(e_mask), 1.0)
            e_probs = e_mask / denom
            e_idx = jax.random.choice(rng_enemy, self.RIGHT_REGIONS, p=e_probs)
            return a_idx, e_idx, b_idx
            
        def select_navigate():
            e_idx = jax.random.choice(rng_enemy, self.LEFT_REGIONS)
            a_idx = jax.random.choice(rng_ally, self.RIGHT_REGIONS)
            b_mask = (self.RIGHT_REGIONS != a_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(b_mask), 1.0)
            b_probs = b_mask / denom
            b_idx = jax.random.choice(rng_beacon, self.RIGHT_REGIONS, p=b_probs)
            return a_idx, e_idx, b_idx
            
        return jax.lax.switch(layout_type, [select_base, select_combat, select_navigate])

    def enforce_constraints(self, next_pos, prev_pos, map_w, map_h, unit_teams):
        """Bridge-and-Cliff terrain logic. Only constrains ally units (team=0)."""
        # Normalized x=0.5 is the cliff
        px = prev_pos[..., 0] / map_w
        py = prev_pos[..., 1] / map_h
        nx = next_pos[..., 0] / map_w
        ny = next_pos[..., 1] / map_h
        
        # Crossing the central cliff (x=0.5)
        cross_cliff = ((px < 0.5) & (nx >= 0.5)) | ((px >= 0.5) & (nx < 0.5))
        
        # Bridges (y ranges)
        on_bridge_1 = (ny > 0.2) & (ny < 0.3)
        on_bridge_2 = (ny > 0.7) & (ny < 0.8)
        blocked = cross_cliff & ~(on_bridge_1 | on_bridge_2)
        
        # Only constrain ally units (team=0), enemies move freely
        is_ally = unit_teams == 0
        
        # Apply constraint only to ally units that are blocked
        constrained = blocked & is_ally
        
        # Reset X to previous valid side if constrained (keep shape (N,))
        safe_nx = jnp.where(constrained, px, nx)
        
        # Reconstruct position: keep Y unchanged, use safe X
        safe_nx_scaled = safe_nx * map_w
        return next_pos.at[..., 0].set(safe_nx_scaled)

    def compute_reward(self, state, next_smax_state, num_allies, num_enemies, map_w, map_h):
        """Unified reward shaping for Two-Bridge."""
        dims = jnp.array([map_w, map_h])
        ally_pos = next_smax_state.unit_positions[:num_allies]
        enemy_pos = next_smax_state.unit_positions[num_allies:]
        ally_alive = next_smax_state.unit_alive[:num_allies]
        enemy_alive = next_smax_state.unit_alive[num_allies:]
        
        # 1. Navigation Shaping (Distance to Beacon)
        dist_to_beacon = jnp.linalg.norm((ally_pos - state.beacon_pos) / dims, axis=-1)
        mean_dist = jnp.sum(dist_to_beacon * ally_alive) / jnp.maximum(jnp.sum(ally_alive), 1.0)
        nav_reward = (state.prev_mean_dist - mean_dist) * 2.0
        
        # 2. Combat Shaping (HP Damage + Kills)
        current_enemy_hp = jnp.sum(next_smax_state.unit_health[num_allies:] * enemy_alive)
        enemy_dmg_reward = (state.prev_enemy_health - current_enemy_hp) * 0.01 * 0.5
        
        current_ally_hp = jnp.sum(next_smax_state.unit_health[:num_allies] * ally_alive)
        ally_dmg_penalty = (state.prev_ally_health - current_ally_hp) * 0.01 * 0.5
        
        # Kill Bonuses
        enemy_killed = (jnp.sum(state.smax_state.unit_alive[num_allies:]) - jnp.sum(enemy_alive)) * 0.2
        
        # Total Reward
        total_reward = nav_reward + enemy_dmg_reward - ally_dmg_penalty + enemy_killed
        
        # Termination
        beacon_reached = jnp.any((dist_to_beacon < 0.05) & ally_alive)
        enemies_dead = jnp.sum(enemy_alive) == 0
        allies_dead = jnp.sum(ally_alive) == 0
        done = beacon_reached | enemies_dead | allies_dead
        
        return total_reward, done, mean_dist, current_enemy_hp, current_ally_hp, 0.0
