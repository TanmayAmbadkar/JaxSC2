"""JIT-compatible scripted agents using per-unit movement and attack commands."""

import jax.numpy as jnp

from JaxSC2.env.env import PerUnitAction


def bridge_waypoints(env, positions, goals):
    """Use the map's radius-aware route until units have fully exited the bridge."""
    return env.map_instance.route_waypoints(
        positions, goals, env.map_width, env.map_height, env.unit_radius,
    )


def movement_directions(env, state, goals):
    """Quantize a velocity-compensated heading to eight movement directions."""
    positions = state.smax_state.unit_positions[:env.num_allies]
    waypoints = bridge_waypoints(env, positions, goals)
    desired = waypoints - positions - 3. * state.unit_velocities[:env.num_allies]
    angles = jnp.arctan2(desired[:, 0], desired[:, 1])
    return (jnp.rint(angles / (jnp.pi / 4)).astype(jnp.int32) % 8)


def beacon_agent(env, state):
    """Move every living ally toward the beacon, routing across bridges as needed."""
    alive = state.smax_state.unit_alive[:env.num_allies]
    goals = jnp.broadcast_to(state.beacon_pos, (env.num_allies, 2))
    return PerUnitAction(
        alive, jnp.where(alive, 1, 0).astype(jnp.int32),
        movement_directions(env, state, goals), jnp.full(env.num_allies, -1, dtype=jnp.int32),
    )


def combat_agent(env, state):
    """Approach known enemies, stop in each unit's weapon range, and attack.

    With fog enabled, use only sightings/memory and sweep map regions until an
    enemy is found. The agent never reads hidden enemy positions or health.
    """
    s = state.smax_state
    ally_pos = s.unit_positions[:env.num_allies]
    alive = s.unit_alive[:env.num_allies]
    enemy_pos = state.fog_state.last_seen_pos
    known = state.fog_state.last_seen_alive & (state.fog_state.last_seen_time >= 0)
    distances = jnp.linalg.norm(ally_pos[:, None, :] - enemy_pos[None, :, :], axis=-1)
    score = distances + .01 * state.fog_state.last_seen_hp[None, :]
    target = jnp.argmin(jnp.where(known[None, :], score, jnp.inf), axis=-1)
    goals = enemy_pos[target]
    # Search one region per unit, alternating banks/bridges over time.
    regions = env.map_instance.REGION_COORDS
    centers = jnp.stack([(regions[:, 0] + regions[:, 1]) * env.map_width / 2,
                         (regions[:, 2] + regions[:, 3]) * env.map_height / 2], axis=-1)
    search = centers[(jnp.arange(env.num_allies) + state.timestep // 100) % len(regions)]
    goals = jnp.where(jnp.any(known), goals, search)
    ranges = env.unit_type_range[s.unit_types[:env.num_allies]]
    target_distance = jnp.linalg.norm(goals - ally_pos, axis=-1)
    in_range = (jnp.any(known) & state.enemy_visible[target]
                & (target_distance < ranges * .85))
    return PerUnitAction(
        alive, jnp.where(alive, jnp.where(in_range, 2, 1), 0).astype(jnp.int32),
        movement_directions(env, state, goals),
        jnp.where(jnp.any(known), target, -1).astype(jnp.int32),
    )
