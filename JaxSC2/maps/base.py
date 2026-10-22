"""Terrain contracts independent of task objectives and opponent policies."""

from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from JaxSC2.maps.twobridge import TwoBridgeMap


class Terrain(Protocol):
    width: float
    height: float
    def obstacle_rectangles(self, map_w, map_h, unit_radius=0.): ...
    def is_walkable(self, positions, map_w, map_h, unit_radius=.6, tolerance=1e-5): ...
    def enforce_constraints(self, new_pos, old_pos, map_w, map_h, teams, unit_radius=.6): ...
    def route_waypoints(self, positions, goals, map_w, map_h, unit_radius=.6): ...
    def sample_positions(self, key, count, margin=2.): ...


def sample_positions(terrain, key, count, margin=2.):
    candidates = jax.random.uniform(key, (count, 64, 2), minval=margin,
                                    maxval=jnp.array([terrain.width, terrain.height]) - margin)
    valid = terrain.is_walkable(candidates, terrain.width, terrain.height, .6)
    chosen = candidates[jnp.arange(count), jnp.argmax(valid, axis=1)]
    fallback = getattr(terrain, "spawn_fallback", jnp.array([4., 4.]))
    return jnp.where(jnp.any(valid, axis=1)[:, None], chosen, fallback)


class ArenaMap:
    """A rectangular arena, optionally with axis-aligned obstacles.

    Obstacle collision uses the same swept/sliding solver as TwoBridge. Routing
    is direct: obstacle layouts need a task-specific planner, not hidden physics
    pathfinding. The initial mini-game layouts are open arenas.
    """
    DITCH_X = (0., 0.)
    BRIDGE_Y = ()

    def __init__(self, width=32., height=32., obstacles=(), name="arena"):
        if width < 8 or height < 8:
            raise ValueError("Arena dimensions must be at least 8")
        self.width, self.height, self.name = float(width), float(height), name
        self.rectangles = jnp.asarray(obstacles, dtype=jnp.float32).reshape((-1, 4))
        boxes = np.asarray(self.rectangles)
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, :2] >= boxes[:, 2:]):
            raise ValueError("Obstacle rectangles must have finite increasing bounds")
        grid = np.stack(np.meshgrid(np.linspace(.61, width - .61, 64),
                                    np.linspace(.61, height - .61, 64)), axis=-1).reshape((-1, 2))
        blocked = np.any(np.all((grid[:, None, :] > boxes[None, :, :2] - .6) &
                                (grid[:, None, :] < boxes[None, :, 2:] + .6), axis=-1), axis=-1)
        if np.all(blocked):
            raise ValueError("Arena has no radius-clear spawn point")
        self.spawn_fallback = jnp.asarray(grid[np.argmax(~blocked)], dtype=jnp.float32)
        # Only used by the compatibility world's reset; tasks own their spawns.
        self.REGION_COORDS = jnp.array([[.05, .35, .1, .9], [.65, .95, .1, .9]])

    def get_spawn_regions(self, key, layout_type):
        return jnp.int32(0), jnp.int32(1), jnp.int32(1)

    def obstacle_rectangles(self, map_w, map_h, unit_radius=0.):
        return self.rectangles + jnp.asarray(unit_radius)[..., None, None] * jnp.array([-1., -1., 1., 1.])

    def terrain_mask(self, positions, map_w, map_h):
        boxes = self.rectangles
        return jnp.any(((positions[..., None, :] > boxes[:, :2]) &
                        (positions[..., None, :] < boxes[:, 2:])).all(axis=-1), axis=-1)

    def is_walkable(self, positions, map_w, map_h, unit_radius=.6, tolerance=1e-5):
        return TwoBridgeMap.is_walkable(self, positions, map_w, map_h, unit_radius, tolerance)

    def enforce_constraints(self, new_pos, old_pos, map_w, map_h, teams, unit_radius=.6):
        if self.rectangles.shape[0]:
            return TwoBridgeMap.enforce_constraints(self, new_pos, old_pos, map_w, map_h, teams, unit_radius)
        radius = jnp.asarray(unit_radius)
        return jnp.clip(new_pos, radius[..., None], jnp.array([map_w, map_h]) - radius[..., None])

    def route_waypoints(self, positions, goals, map_w, map_h, unit_radius=.6):
        return goals

    def sample_positions(self, key, count, margin=2.):
        return sample_positions(self, key, count, margin)
