"""Observable-state per-unit baselines and an explicit economy macro controller."""

import jax.numpy as jnp

from JaxSC2.env.env import PerUnitAction
from JaxSC2.scripted.agents import beacon_agent, combat_agent, movement_directions
from JaxSC2.tasks.economy import macro_mask, supply


def scripted_action(env, state):
    action, n = env.no_op(), env.num_allies
    world, s = state.world, state.world.smax_state
    alive = s.unit_alive[:n]
    if env.spec.kind == "beacon":
        return action.replace(units=beacon_agent(env.core, world))
    if env.spec.kind == "shards":
        chosen, targets = jnp.zeros(env.entity_capacity, dtype=bool), []
        for i in range(n):
            distance = jnp.linalg.norm(state.entity_positions - s.unit_positions[i], axis=-1)
            available = (state.entity_amount > 0) & ~chosen
            available = jnp.where(jnp.any(available), available, state.entity_amount > 0)
            target = jnp.argmin(jnp.where(available, distance, jnp.inf))
            targets.append(target)
            chosen = chosen.at[target].set(True)
        goals = state.entity_positions[jnp.array(targets)]
        units = PerUnitAction(alive, jnp.where(alive, 1, 0), movement_directions(env.core, world, goals),
                               jnp.full(n, -1, dtype=jnp.int32))
        return action.replace(units=units)
    if not env.economic:
        units = combat_agent(env.core, world)
        known = world.fog_state.last_seen_alive & (world.fog_state.last_seen_time >= 0)
        distance = jnp.linalg.norm(s.unit_positions[:n, None, :] - world.fog_state.last_seen_pos[None, :, :], axis=-1)
        nearest = jnp.argmin(jnp.where(known[None, :], distance, jnp.inf), axis=1)
        closest = jnp.min(jnp.where(known[None, :], distance, jnp.inf), axis=1)
        retreat = jnp.any(known) & (closest < 4.5) & (s.unit_types[:n] == 1)
        away = s.unit_positions[:n] - world.fog_state.last_seen_pos[nearest]
        away /= jnp.maximum(jnp.linalg.norm(away, axis=-1, keepdims=True), 1e-5)
        goals = jnp.clip(s.unit_positions[:n] + away * 3., 1., env.dims - 1.)
        dirs = movement_directions(env.core, world, goals)
        # Search a 4x4 grid rather than reading hidden enemy locations.
        cell = (jnp.arange(n) + world.timestep // 160) % 16
        search = jnp.stack([(cell % 4 + .5) / 4 * env.map_width,
                             (cell // 4 + .5) / 4 * env.map_height], axis=-1)
        dirs = jnp.where(jnp.any(known), dirs, movement_directions(env.core, world, search))
        units = units.replace(direction=jnp.where(retreat | ~jnp.any(known), dirs, units.direction),
                               verb=jnp.where(alive & (retreat | ~jnp.any(known)), 1, units.verb))
        return action.replace(units=units)

    # Stable per-worker assignments. Jobs continue while a global macro command
    # is issued; carrying workers do not switch resources until depositing.
    workers = alive & (s.unit_types[:n] == 4)
    resource_order = jnp.argsort(jnp.where((state.entity_kind == 2) & (state.entity_amount > 0),
                                         jnp.arange(env.entity_capacity), env.entity_capacity))
    count = jnp.maximum(1, jnp.sum((state.entity_kind == 2) & (state.entity_amount > 0)))
    targets = resource_order[jnp.arange(n) % count]
    refined = jnp.any((state.building_kind == 4)[:, None] &
                      (state.building_remaining == 0)[:, None] &
                      (jnp.linalg.norm(state.building_positions[:, None, :] - state.entity_positions[None, :, :], axis=-1) < 1.), axis=0)
    gas_nodes = refined & (state.entity_kind == 3) & (state.entity_amount > 0)
    gas_target = jnp.argmax(gas_nodes)
    targets = jnp.where(jnp.any(gas_nodes) & (jnp.arange(n) % 4 == 0), gas_target, targets)
    units = PerUnitAction(alive, jnp.where(workers, 3, 0).astype(jnp.int32),
                           jnp.zeros(n, dtype=jnp.int32), targets.astype(jnp.int32))
    mask = macro_mask(env, state)
    used, cap = supply(state)
    pending = jnp.sum(state.queue_kind >= 0)
    depot_exists = jnp.any(state.building_kind == 2)
    depot_pending = jnp.any((state.building_kind == 2) & (state.building_remaining > 0))
    need_depot = ((used + pending >= cap - 1) | ((env.spec.kind == "production") & ~depot_exists)) & ~depot_pending
    need_barracks = (env.spec.kind == "production") & ~jnp.any(state.building_kind == 3)
    need_refinery = (env.spec.kind == "economy") & ~jnp.any(state.building_kind == 4)
    need_cc = (env.spec.kind == "economy") & (jnp.sum(workers) >= 16) & (jnp.sum(state.building_kind == 1) < 2)
    build = jnp.where(need_depot & mask[2], 2,
                      jnp.where(need_barracks & mask[3], 3,
                                jnp.where(need_refinery & mask[4], 4, jnp.where(need_cc & mask[1], 1, 0))))
    busy = jnp.any((state.building_worker[:, None] == jnp.arange(n)[None, :]) &
                   (state.building_remaining[:, None] > 0), axis=0)
    builder = jnp.argmax(jnp.where(workers & ~busy, jnp.arange(n) + 1, 0))
    available_builder = jnp.any(workers & ~busy)
    base = state.building_positions[0]
    offsets = jnp.array([[0., 4.], [0., -4.], [4., 0.], [-4., 0.], [4., 4.],
                         [4., -4.], [-4., 4.], [-4., -4.], [0., 8.], [0., -8.]])
    sites = base + offsets
    free = env.map.is_walkable(sites, env.map_width, env.map_height, 1.)
    free &= jnp.all((jnp.linalg.norm(sites[:, None, :] - state.building_positions[None, :, :], axis=-1) > 2.6)
                     | (state.building_kind[None, :] == 0), axis=1)
    free &= jnp.all((jnp.linalg.norm(sites[:, None, :] - state.entity_positions[None, :, :], axis=-1) > 1.2)
                     | (state.entity_amount[None, :] <= 0), axis=1)
    site = sites[jnp.argmax(free)]
    geysers = state.entity_kind == 3
    site = jnp.where(build == 4, state.entity_positions[jnp.argmax(geysers)], site)
    build = jnp.where(available_builder & (jnp.any(free) | (build == 4)), build, 0)
    goals = jnp.broadcast_to(site, (n, 2))
    dirs = movement_directions(env.core, world, goals)
    near = jnp.linalg.norm(s.unit_positions[builder] - site) < 1.7
    units = units.replace(verb=units.verb.at[builder].set(jnp.where(build > 0, jnp.where(near, 0, 1), units.verb[builder])),
                           direction=units.direction.at[builder].set(jnp.where(build > 0, dirs[builder], units.direction[builder])))
    ready = (state.building_remaining == 0) & (state.queue_kind < 0)
    train_marine = (env.spec.kind == "production") & mask[6]
    train_worker = (env.spec.kind == "economy") & (jnp.sum(workers) < 16) & mask[5]
    train = jnp.where(train_marine, 6, jnp.where(train_worker, 5, 0))
    producer = jnp.argmax(ready & (state.building_kind == jnp.where(train == 5, 1, 3)))
    macro = jnp.where(build > 0, jnp.where(near, build, 0), train)
    actor = jnp.where(build > 0, builder, producer)
    return action.replace(units=units, macro=macro.astype(jnp.int32),
                           actor=actor.astype(jnp.int32), placement=site)
