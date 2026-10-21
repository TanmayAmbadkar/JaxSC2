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

    # Normalized geometry shared by collision checks, feature maps and rendering.
    DITCH_X = (.45, .55)
    BRIDGE_Y = ((.2, .3), (.7, .8))

    def obstacle_rectangles(self, map_w, map_h, unit_radius=0.):
        """Three ditch rectangles, conservatively expanded for a unit's radius."""
        x0, x1 = self.DITCH_X
        (b0, b1), (b2, b3) = self.BRIDGE_Y
        rectangles = jnp.array([
            [x0 * map_w, 0., x1 * map_w, b0 * map_h],
            [x0 * map_w, b1 * map_h, x1 * map_w, b2 * map_h],
            [x0 * map_w, b3 * map_h, x1 * map_w, map_h],
        ])
        return rectangles + jnp.asarray(unit_radius)[..., None, None] * jnp.array([-1., -1., 1., 1.])

    def terrain_mask(self, positions, map_w, map_h):
        """True where a point lies in the visible ditch footprint."""
        rectangles = self.obstacle_rectangles(map_w, map_h)
        inside = ((positions[..., None, :] > rectangles[:, :2])
                  & (positions[..., None, :] < rectangles[:, 2:])).all(axis=-1)
        return jnp.any(inside, axis=-1)

    def is_walkable(self, positions, map_w, map_h, unit_radius=.6, tolerance=1e-5):
        """Check world bounds and radius clearance for arbitrary leading axes."""
        radius = jnp.asarray(unit_radius)
        rectangles = self.obstacle_rectangles(map_w, map_h, radius)
        inside = ((positions[..., None, :] > rectangles[..., :2] + tolerance)
                  & (positions[..., None, :] < rectangles[..., 2:] - tolerance)).all(axis=-1)
        bounded = ((positions >= radius[..., None] - tolerance)
                   & (positions <= jnp.array([map_w, map_h]) - radius[..., None] + tolerance)).all(axis=-1)
        return bounded & ~jnp.any(inside, axis=-1)

    def route_waypoints(self, positions, goals, map_w, map_h, unit_radius=.6):
        """Stay in a bridge corridor until the entire unit reaches the far bank."""
        middle = map_w * .5
        bank_left = self.DITCH_X[0] * map_w - unit_radius - .5
        bank_right = self.DITCH_X[1] * map_w + unit_radius + .5
        on_bridge = (positions[:, 0] > bank_left) & (positions[:, 0] < bank_right)
        left = positions[:, 0] < middle
        goal_left = goals[:, 0] < middle
        crossing = (left != goal_left) | on_bridge
        bridges = jnp.array([(a + b) / 2 for a, b in self.BRIDGE_Y]) * map_h
        entry_cost = jnp.abs(positions[:, 1, None] - bridges)
        cost = entry_cost + jnp.abs(goals[:, 1, None] - bridges) + .05 * entry_cost
        bridge_index = jnp.argmin(jnp.where(on_bridge[:, None], entry_cost, cost), axis=-1)
        bridge_y = bridges[bridge_index]
        aligned = jnp.abs(positions[:, 1] - bridge_y) < .4
        near_x = jnp.where(left, bank_left, bank_right)
        far_x = jnp.where(goal_left, bank_left, bank_right)
        waypoint = jnp.stack([jnp.where(aligned | on_bridge, far_x, near_x), bridge_y], axis=-1)
        return jnp.where(crossing[:, None], waypoint, goals)

    def enforce_constraints(self, next_pos, prev_pos, map_w, map_h, unit_teams, unit_radius=.6):
        """Sweep unit centers against radius-expanded ditch rectangles, for both teams.

        Resolve the first contact, remove its inward displacement, then slide
        along the edge. Three contacts suffice for this axis-aligned geometry.
        Radius expansion is conservative at corners and also prevents tunneling.
        """
        radius = jnp.broadcast_to(jnp.asarray(unit_radius), prev_pos.shape[:-1])
        limits = jnp.array([map_w, map_h])
        rectangles = self.obstacle_rectangles(map_w, map_h, radius)
        start = jnp.clip(prev_pos, radius[:, None], limits - radius[:, None])

        # Recover an invalid input position (e.g. an older saved simulation).
        def repair(i, position):
            box = rectangles[:, i]
            inside = ((position > box[:, :2]) & (position < box[:, 2:])).all(axis=-1)
            x, y = position[:, 0], position[:, 1]
            candidates = jnp.stack([
                jnp.stack([box[:, 0], y], -1), jnp.stack([box[:, 2], y], -1),
                jnp.stack([x, box[:, 1]], -1), jnp.stack([x, box[:, 3]], -1),
            ], axis=1)
            valid = ((candidates >= radius[:, None, None])
                     & (candidates <= limits - radius[:, None, None])).all(axis=-1)
            distance = jnp.sum((candidates - position[:, None, :]) ** 2, axis=-1)
            closest = jnp.argmin(jnp.where(valid, distance, jnp.inf), axis=-1)
            recovered = candidates[jnp.arange(len(position)), closest]
            return jnp.where(inside[:, None], recovered, position)

        start = jax.lax.fori_loop(0, 3, repair, start)
        desired = jnp.clip(next_pos, radius[:, None], limits - radius[:, None])

        def slide(_, carry):
            position, remaining = carry
            parallel = jnp.abs(remaining[:, None, :]) < 1e-8
            delta = jnp.where(parallel, 1., remaining[:, None, :])
            near = (rectangles[..., :2] - position[:, None, :]) / delta
            far = (rectangles[..., 2:] - position[:, None, :]) / delta
            entry_axis, exit_axis = jnp.minimum(near, far), jnp.maximum(near, far)
            parallel_inside = ((position[:, None, :] > rectangles[..., :2])
                               & (position[:, None, :] < rectangles[..., 2:]))
            entry_axis = jnp.where(parallel, jnp.where(parallel_inside, -jnp.inf, jnp.inf), entry_axis)
            exit_axis = jnp.where(parallel, jnp.where(parallel_inside, jnp.inf, -jnp.inf), exit_axis)
            entry, exit = jnp.max(entry_axis, axis=-1), jnp.min(exit_axis, axis=-1)
            hits = (entry >= -1e-6) & (entry < 1.) & (exit > jnp.maximum(entry, 0.))
            times = jnp.where(hits, jnp.maximum(entry, 0.), jnp.inf)
            obstacle = jnp.argmin(times, axis=-1)
            first = times[jnp.arange(len(position)), obstacle]
            contact = jnp.isfinite(first)
            fraction = jnp.where(contact, jnp.maximum(first - 1e-5, 0.), 1.)
            position = position + remaining * fraction[:, None]
            normal_axis = jnp.argmax(entry_axis[jnp.arange(len(position)), obstacle], axis=-1)
            remaining = remaining * (1. - fraction[:, None])
            remaining = jnp.where(
                contact[:, None] & (jnp.arange(2) == normal_axis[:, None]), 0., remaining)
            return position, remaining

        position, _ = jax.lax.fori_loop(0, 3, slide, (start, desired - start))
        return jnp.clip(position, radius[:, None], limits - radius[:, None])

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
