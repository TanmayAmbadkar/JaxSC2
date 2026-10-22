"""Deterministic, JAX-compatible enemy movement and target selection.

Controllers use simulation state, as the original enemy AI does. Player fog does
not limit enemy perception. No controller teleports units or bypasses physics.
"""

from typing import NamedTuple

import jax.numpy as jnp


LEGACY_ENEMY_MODES = ("static", "guard", "patrol", "aggressive")
COMBAT_ENEMY_MODES = (
    "beacon_defender", "bridge_blocker", "pursuer", "kiter",
    "focus_fire", "ambusher", "adaptive_defender",
)
ENEMY_MODES = LEGACY_ENEMY_MODES + COMBAT_ENEMY_MODES


class EnemyOrders(NamedTuple):
    verb: jnp.ndarray
    direction: jnp.ndarray
    target: jnp.ndarray  # global unit index, or -1
    can_attack: jnp.ndarray
    alerted: jnp.ndarray


def idle_orders(env, state):
    zeros = jnp.zeros(env.num_enemies, dtype=jnp.int32)
    return EnemyOrders(zeros, zeros, zeros - 1,
                       jnp.ones(env.num_enemies, dtype=bool), state.enemy_alerted)


def _bridge_posts(env, home, bridges):
    """Hold radius-clear exits on each enemy's original bank, with row spacing."""
    left = home[:, 0] < env.map_width * .5
    x = jnp.where(left, env.map_instance.DITCH_X[0] * env.map_width - env.unit_radius - .8,
                  env.map_instance.DITCH_X[1] * env.map_width + env.unit_radius + .8)
    # Spread larger squads away from the exit instead of stacking every unit.
    x += jnp.where(left, -1., 1.) * (jnp.arange(env.num_enemies) // 2) * 1.3
    y = jnp.mean(jnp.asarray(env.map_instance.BRIDGE_Y), axis=1)[bridges] * env.map_height
    return jnp.stack([x, y], axis=-1)


def _legacy_orders(env, state, pos, ally_pos, ally_alive, alive):
    centroid = env._compute_centroid(ally_pos, ally_alive)
    delta = centroid - pos
    waypoint = env.map_instance.route_waypoints(
        pos, jnp.broadcast_to(centroid, pos.shape), env.map_width, env.map_height, env.unit_radius)
    delta_move = waypoint - pos
    horizontal = jnp.abs(delta_move[:, 0]) > jnp.abs(delta_move[:, 1])
    direction = jnp.where(horizontal, jnp.where(delta_move[:, 0] > 0, 1, 3),
                          jnp.where(delta_move[:, 1] > 0, 0, 2))
    nearby = jnp.linalg.norm(delta, axis=-1) < env.combat_cfg["aggression_radius"]
    patrol = jnp.full(env.num_enemies, (state.timestep // 30) % 4, dtype=jnp.int32)
    encoded = jnp.where(
        env.enemy_mode == "aggressive", direction,
        jnp.where(env.enemy_mode == "patrol", patrol,
                  jnp.where((env.enemy_mode == "guard") & nearby, direction, 4)))
    encoded = jnp.where(alive, encoded, 4)
    idle = idle_orders(env, state)
    return idle._replace(verb=jnp.where(encoded < 4, 1, 0).astype(jnp.int32),
                         direction=jnp.where(encoded < 4, encoded * 2, 0).astype(jnp.int32))


def get_enemy_orders(env, state):
    s = state.smax_state
    n = env.num_allies
    pos, alive = s.unit_positions[n:], s.unit_alive[n:]
    ally_pos, ally_alive = s.unit_positions[:n], s.unit_alive[:n]
    if env.enemy_mode in LEGACY_ENEMY_MODES:
        return _legacy_orders(env, state, pos, ally_pos, ally_alive, alive)

    distances = jnp.linalg.norm(pos[:, None, :] - ally_pos[None, :, :], axis=-1)
    distances = jnp.where(ally_alive[None, :], distances, jnp.inf)
    nearest = jnp.argmin(distances, axis=-1)
    target = nearest
    has_allies = jnp.any(ally_alive)
    ranges = env.unit_type_range[s.unit_types[n:]]
    alerted = state.enemy_alerted
    can_attack = jnp.ones(env.num_enemies, dtype=bool)
    goals = ally_pos[target]
    move = distances[jnp.arange(env.num_enemies), target] > ranges * .9

    if env.enemy_mode == "beacon_defender":
        angles = 2. * jnp.pi * jnp.arange(env.num_enemies) / env.num_enemies
        posts = state.beacon_pos + 2.5 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        threat = ally_alive & (jnp.linalg.norm(ally_pos - state.beacon_pos, axis=-1) < 8.)
        target = jnp.argmin(jnp.where(threat[None, :], distances, jnp.inf), axis=-1)
        intercept = jnp.any(threat) & (jnp.linalg.norm(pos - state.beacon_pos, axis=-1) < 10.)
        goals = jnp.where(intercept[:, None], ally_pos[target], posts)
        move = jnp.where(intercept, distances[jnp.arange(env.num_enemies), target] > ranges * .9,
                         jnp.linalg.norm(posts - pos, axis=-1) > .6)
        # Defenders may shoot while relocating, but do not chase beyond their leash.
        target = jnp.where(intercept, target, nearest)
    elif env.enemy_mode in ("bridge_blocker", "adaptive_defender", "ambusher"):
        if env.enemy_mode == "adaptive_defender":
            centroid = env._compute_centroid(ally_pos, ally_alive)
            centers = jnp.mean(jnp.asarray(env.map_instance.BRIDGE_Y), axis=1) * env.map_height
            bridge = jnp.argmin(jnp.abs(centers - centroid[1]))
            bridges = jnp.full(env.num_enemies, bridge)
        elif env.enemy_mode == "ambusher":
            centers = jnp.mean(jnp.asarray(env.map_instance.BRIDGE_Y), axis=1) * env.map_height
            bridges = jnp.argmin(jnp.abs(state.enemy_home[:, 1, None] - centers), axis=-1)
        else:
            bridges = jnp.arange(env.num_enemies) % 2
        posts = _bridge_posts(env, state.enemy_home, bridges)
        if env.enemy_mode == "ambusher":
            # One living sentry triggers the squad. The alert stays latched.
            trigger = jnp.any((distances < env.vision_radius) & alive[:, None])
            alerted = alerted | trigger
            can_attack = alerted
            intercept = alerted
        else:
            intercept = (jnp.min(distances, axis=-1) < ranges + 2.) & (
                jnp.linalg.norm(pos - posts, axis=-1) < 4.)
        goals = jnp.where(intercept[:, None], ally_pos[target], posts)
        move = jnp.where(intercept, distances[jnp.arange(env.num_enemies), target] > ranges * .9,
                         jnp.linalg.norm(posts - pos, axis=-1) > .6)
    elif env.enemy_mode == "pursuer":
        # Prefer nearby low-health, isolated allies rather than an army centroid.
        pair_dist = jnp.linalg.norm(ally_pos[:, None, :] - ally_pos[None, :, :], axis=-1)
        others = ally_alive[None, :] & ~jnp.eye(n, dtype=bool)
        isolation = jnp.minimum(jnp.min(jnp.where(others, pair_dist, jnp.inf), axis=-1), 8.)
        score = distances + .02 * s.unit_health[:n] - .25 * isolation
        target = jnp.argmin(score, axis=-1)
        goals = ally_pos[target]
        move = distances[jnp.arange(env.num_enemies), target] > ranges * .9
    elif env.enemy_mode == "focus_fire":
        centroid = env._compute_centroid(pos, alive)
        hp_fraction = s.unit_health[:n] / env.unit_type_hp[s.unit_types[:n]]
        score = hp_fraction + .05 * jnp.linalg.norm(ally_pos - centroid, axis=-1)
        shared = jnp.argmin(jnp.where(ally_alive, score, jnp.inf))
        target = jnp.full(env.num_enemies, shared, dtype=jnp.int32)
        goals = ally_pos[target]
        move = distances[jnp.arange(env.num_enemies), target] > ranges * .9
    elif env.enemy_mode == "kiter":
        distance = jnp.min(distances, axis=-1)
        ranged = s.unit_types[n:] != 0
        timers = state.attack_timers[n:]
        windup = env.combat_cfg["type_windups"][s.unit_types[n:]]
        cooldown = timers > windup + env.combat_cfg.get("backswing", 1)
        retreat = ranged & ((distance < ranges * .65) | (cooldown & (distance < ranges + .5)))
        away = pos - ally_pos[target]
        norm = jnp.linalg.norm(away, axis=-1, keepdims=True)
        away = jnp.where(norm > 1e-5, away / jnp.maximum(norm, 1e-5), jnp.array([1., 0.]))
        retreat_goal = pos + away * 3.
        # A wall is not an escape route; try a lateral sidestep instead.
        valid = env.map_instance.is_walkable(retreat_goal, env.map_width, env.map_height, env.unit_radius)
        lateral = pos + jnp.stack([-away[:, 1], away[:, 0]], axis=-1) * 2.
        retreat_goal = jnp.where(valid[:, None], retreat_goal, lateral)
        goals = jnp.where(retreat[:, None], retreat_goal, goals)
        move = move | retreat

    goals = jnp.clip(goals, env.unit_radius,
                     jnp.array([env.map_width, env.map_height]) - env.unit_radius)
    waypoint = env.map_instance.route_waypoints(
        pos, goals, env.map_width, env.map_height, env.unit_radius)
    delta = waypoint - pos
    direction = (jnp.rint(jnp.arctan2(delta[:, 0], delta[:, 1]) / (jnp.pi / 4)).astype(jnp.int32) % 8)
    moving = alive & has_allies & move
    target = jnp.where(alive & has_allies & can_attack, target, -1).astype(jnp.int32)
    return EnemyOrders(moving.astype(jnp.int32), direction, target,
                       can_attack & alive, alerted & alive)
