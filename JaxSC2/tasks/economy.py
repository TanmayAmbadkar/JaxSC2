"""Bounded economy/production mechanics for the first mini-game adaptations.

Workers physically travel, harvest finite deposits, carry cargo, and return to a
completed command center. Buildings have costs, placement, construction time,
tech prerequisites, and queues. Structures are not attackable in this version.
"""

import jax
import jax.numpy as jnp

from JaxSC2.env.env import PerUnitAction


COST = jnp.array([0., 400., 100., 150., 75., 50., 50.])
DURATION = jnp.array([0, 60, 20, 35, 20, 12, 18], dtype=jnp.int32)


def supply(state):
    complete = (state.building_kind > 0) & (state.building_remaining == 0)
    capacity = jnp.sum(jnp.where(complete & (state.building_kind == 1), 15,
                                jnp.where(complete & (state.building_kind == 2), 8, 0)))
    used = jnp.sum(state.world.smax_state.unit_alive[:state.worker_target.shape[0]])
    return used, capacity


def initialize(env, state):
    n, s = env.num_allies, state.world.smax_state
    base = jnp.array([env.map_width * .20, env.map_height * .5])
    i = jnp.arange(n)
    positions = base + jnp.stack([(i % 4) * 1.4 - 2.1, (i // 4) * 1.4 - 2.1], axis=-1)
    positions = jnp.clip(positions, 1., env.dims - 1.)
    s = s.replace(unit_positions=s.unit_positions.at[:n].set(positions),
                  unit_types=s.unit_types.at[:n].set(4),
                  unit_health=s.unit_health.at[:n].set(jnp.where(s.unit_alive[:n], 45., 0.)))
    mineral_count = 16 if env.spec.kind == "economy" else 8
    e = jnp.arange(env.entity_capacity)
    nodes = jnp.stack([env.map_width * .4 + (e // 8) * 1.5,
                       env.map_height * .5 + (e % 8 - 3.5) * 1.6], axis=-1)
    nodes = nodes.at[16:20].set(jnp.array([[9., 6.], [9., 26.], [23., 6.], [23., 26.]]) * env.dims / 32.)
    kinds = jnp.where(e < mineral_count, 2, 0)
    if env.spec.kind == "economy":
        kinds = kinds.at[16:20].set(3)
    return state.replace(
        world=state.world.replace(smax_state=s), minerals=jnp.float32(50.),
        entity_positions=nodes, entity_kind=kinds,
        entity_amount=jnp.where(kinds == 2, 1500., jnp.where(kinds == 3, 2000., 0.)),
        building_positions=state.building_positions.at[0].set(base),
        building_kind=state.building_kind.at[0].set(1))


def macro_mask(env, state):
    mask = jnp.array([True, False, False, False, False, False, False])
    if not env.economic:
        return mask
    n, s = env.num_allies, state.world.smax_state
    complete = (state.building_kind > 0) & (state.building_remaining == 0)
    builders = s.unit_alive[:n] & (s.unit_types[:n] == 4)
    free_building = jnp.any(state.building_kind == 0)
    can_build = jnp.any(builders) & free_building
    depot = jnp.any(complete & (state.building_kind == 2))
    gas = jnp.any((state.entity_kind == 3) & (state.entity_amount > 0))
    used, cap = supply(state)
    reserved = jnp.sum(state.queue_kind >= 0)
    room = (used + reserved < cap) & (used + reserved < n)
    cc = jnp.any(complete & (state.building_kind == 1) & (state.queue_kind < 0))
    barracks = jnp.any(complete & (state.building_kind == 3) & (state.queue_kind < 0))
    allowed = jnp.array([True, can_build, can_build, can_build & depot,
                         can_build & gas, room & cc, room & barracks])
    return mask | (allowed & (state.minerals >= COST) & ~state.done)


def prepare(env, state, action):
    """Validate/charge one macro and translate persistent worker jobs to motion."""
    n, s = env.num_allies, state.world.smax_state
    op = jnp.clip(action.macro, 0, 6)
    worker = jnp.clip(action.actor, 0, n - 1)
    producer = jnp.clip(action.actor, 0, env.building_capacity - 1)
    slot = jnp.argmax(state.building_kind == 0)
    complete = (state.building_kind > 0) & (state.building_remaining == 0)
    busy = jnp.any((state.building_worker[:, None] == jnp.arange(n)[None, :]) &
                   (state.building_remaining[:, None] > 0), axis=0)
    room = jnp.all(jnp.linalg.norm(state.building_positions - action.placement, axis=-1)
                   + jnp.where(state.building_kind > 0, 0., 1e5) > 2.6)
    near = jnp.linalg.norm(s.unit_positions[worker] - action.placement) <= 2.
    terrain_ok = env.map.is_walkable(action.placement, env.map_width, env.map_height, 1.)
    resource_dist = jnp.linalg.norm(state.entity_positions - action.placement, axis=-1)
    no_resource = jnp.all((resource_dist > 1.2) | (state.entity_amount <= 0))
    geyser = jnp.any((state.entity_kind == 3) & (resource_dist < .8))
    construct = ((op >= 1) & (op <= 4) & macro_mask(env, state)[op] &
                 (action.actor >= 0) & (action.actor < n) & s.unit_alive[worker] &
                 (s.unit_types[worker] == 4) & ~busy[worker] & room & near & terrain_ok &
                 jnp.all(jnp.isfinite(action.placement)) & jnp.where(op == 4, geyser, no_resource))
    required = jnp.where(op == 5, 1, 3)
    train = ((op >= 5) & (op <= 6) & macro_mask(env, state)[op] &
             (action.actor >= 0) & (action.actor < env.building_capacity) &
             complete[producer] & (state.building_kind[producer] == required) &
             (state.queue_kind[producer] < 0))
    valid_op = (action.macro >= 0) & (action.macro <= 6)
    construct, train = construct & valid_op, train & valid_op
    accepted = ((action.macro == 0) | construct | train) & valid_op
    state = state.replace(
        minerals=state.minerals - jnp.where(construct | train, COST[op], 0.),
        building_positions=state.building_positions.at[slot].set(jnp.where(construct, action.placement, state.building_positions[slot])),
        building_kind=state.building_kind.at[slot].set(jnp.where(construct, op, state.building_kind[slot])),
        building_remaining=state.building_remaining.at[slot].set(jnp.where(construct, DURATION[op], state.building_remaining[slot])),
        building_worker=state.building_worker.at[slot].set(jnp.where(construct, worker, state.building_worker[slot])),
        queue_kind=state.queue_kind.at[producer].set(jnp.where(train, op - 5, state.queue_kind[producer])),
        queue_remaining=state.queue_remaining.at[producer].set(jnp.where(train, DURATION[op], state.queue_remaining[producer])))
    units = action.units
    safe_target = jnp.clip(units.target, 0, env.entity_capacity - 1)
    requests = (units.who_mask & (units.verb == 3) & (units.target >= 0) &
                (units.target < env.entity_capacity) & (state.entity_amount[safe_target] > 0) &
                (s.unit_types[:n] == 4) & s.unit_alive[:n] & (state.cargo == 0))
    jobs = jnp.where(requests, safe_target, state.worker_target)
    jobs = jnp.where(units.who_mask & ((units.verb == 1) | (units.verb == 2)), -1, jobs)
    jobs = jobs.at[worker].set(jnp.where(construct, -1, jobs[worker]))
    state = state.replace(worker_target=jobs)
    # Existing jobs persist through NOOP and macro commands.
    safe_job = jnp.maximum(jobs, 0)
    cc = (state.building_kind == 1) & (state.building_remaining == 0)
    home_dist = jnp.linalg.norm(s.unit_positions[:n, None, :] - state.building_positions[None, :, :], axis=-1)
    home_idx = jnp.argmin(jnp.where(cc[None, :], home_dist, jnp.inf), axis=1)
    returning = (state.cargo >= 5.) | ((state.cargo > 0) & ((state.entity_amount[safe_job] <= 0) | (jobs < 0)))
    goals = jnp.where(returning[:, None], state.building_positions[home_idx], state.entity_positions[safe_job])
    goals = jnp.where((returning & ~jnp.any(cc))[:, None], s.unit_positions[:n], goals)
    assigned = (state.building_worker[:, None] == jnp.arange(n)[None, :]) & (state.building_remaining[:, None] > 0)
    busy = jnp.any(assigned, axis=0)
    site = jnp.argmax(assigned, axis=0)
    goals = jnp.where(busy[:, None], state.building_positions[site], goals)
    working = s.unit_alive[:n] & (s.unit_types[:n] == 4) & ((jobs >= 0) | (state.cargo > 0) | busy)
    waypoint = env.map.route_waypoints(s.unit_positions[:n], goals, env.map_width, env.map_height)
    delta = waypoint - s.unit_positions[:n]
    direction = jnp.rint(jnp.arctan2(delta[:, 0], delta[:, 1]) / (jnp.pi / 4)).astype(jnp.int32) % 8
    threshold = jnp.where(busy, .6, jnp.where(returning, 2., .8))
    verb = jnp.where(jnp.linalg.norm(delta, axis=-1) > threshold, 1, 0)
    # Unrecognized resource verbs cannot accidentally become an attack command.
    original = jnp.where(units.verb <= 2, units.verb, 0)
    return state, PerUnitAction(units.who_mask | working,
                                jnp.where(working, verb, original),
                                jnp.where(working, direction, units.direction), units.target), accepted


def advance_economy(env, state):
    n, s = env.num_allies, state.world.smax_state
    workers = s.unit_alive[:n] & (s.unit_types[:n] == 4)
    safe_builder = jnp.clip(state.building_worker, 0, n - 1)
    near_site = jnp.linalg.norm(s.unit_positions[safe_builder] - state.building_positions, axis=-1) <= 2.2
    progressing = (state.building_remaining > 0) & (state.building_worker >= 0) & workers[safe_builder] & near_site
    remaining = jnp.maximum(0, state.building_remaining - progressing.astype(jnp.int32))
    state = state.replace(building_remaining=remaining,
                          building_worker=jnp.where(remaining == 0, -1, state.building_worker))
    busy = jnp.any((state.building_worker[:, None] == jnp.arange(n)[None, :]) &
                   (remaining[:, None] > 0), axis=0)
    cc = (state.building_kind == 1) & (remaining == 0)
    distance_home = jnp.linalg.norm(s.unit_positions[:n, None, :] - state.building_positions[None, :, :], axis=-1)
    at_home = jnp.any((distance_home <= 2.2) & cc[None, :], axis=1)
    job = jnp.maximum(state.worker_target, 0)
    returning = (state.cargo >= 5.) | ((state.cargo > 0) & ((state.entity_amount[job] <= 0) | (state.worker_target < 0)))
    deposit = workers & ~busy & at_home & returning
    mineral_gain = jnp.sum(jnp.where(deposit & (state.cargo_kind == 2), state.cargo, 0.))
    gas_gain = jnp.sum(jnp.where(deposit & (state.cargo_kind == 3), state.cargo, 0.))
    state = state.replace(cargo=jnp.where(deposit, 0., state.cargo),
                          minerals=state.minerals + mineral_gain, gas=state.gas + gas_gain,
                          gathered_minerals=state.gathered_minerals + mineral_gain,
                          gathered_gas=state.gathered_gas + gas_gain)

    def harvest(i, st):
        node = jnp.maximum(st.worker_target[i], 0)
        nearby = jnp.linalg.norm(s.unit_positions[i] - st.entity_positions[node]) < 1.2
        refinery = jnp.any((st.building_kind == 4) & (st.building_remaining == 0) &
                            (jnp.linalg.norm(st.building_positions - st.entity_positions[node], axis=-1) < 1.))
        allowed = ((st.entity_kind[node] == 2) | ((st.entity_kind[node] == 3) & refinery))
        can = (workers[i] & ~busy[i] & (st.worker_target[i] >= 0) & nearby &
               allowed & ~returning[i] & (st.cargo[i] < 5.) &
               ((st.cargo[i] == 0) | (st.cargo_kind[i] == st.entity_kind[node])))
        amount = jnp.where(can, jnp.minimum(1., jnp.minimum(st.entity_amount[node], 5. - st.cargo[i])), 0.)
        return st.replace(entity_amount=st.entity_amount.at[node].add(-amount),
                           cargo=st.cargo.at[i].add(amount),
                           cargo_kind=st.cargo_kind.at[i].set(jnp.where(amount > 0, st.entity_kind[node], st.cargo_kind[i])))
    state = jax.lax.fori_loop(0, n, harvest, state)
    ticking = (state.queue_kind >= 0) & (state.building_remaining == 0)
    state = state.replace(queue_remaining=jnp.maximum(0, state.queue_remaining - ticking.astype(jnp.int32)))

    def produce(i, st):
        world, army = st.world, st.world.smax_state
        free = ~army.unit_alive[:n]
        slot = jnp.argmax(free)
        ready = (st.queue_kind[i] >= 0) & (st.queue_remaining[i] == 0) & jnp.any(free)
        marine = st.queue_kind[i] == 1
        unit_type = jnp.where(marine, 1, 4)
        angle = slot * 2.4
        pos = st.building_positions[i] + 2. * jnp.array([jnp.sin(angle), jnp.cos(angle)])
        pos = env.map.enforce_constraints(pos[None, :], pos[None, :], env.map_width, env.map_height,
                                           jnp.array([0]))[0]
        army = army.replace(
            unit_alive=army.unit_alive.at[slot].set(army.unit_alive[slot] | ready),
            unit_health=army.unit_health.at[slot].set(jnp.where(ready, 45., army.unit_health[slot])),
            unit_types=army.unit_types.at[slot].set(jnp.where(ready, unit_type, army.unit_types[slot])),
            unit_positions=army.unit_positions.at[slot].set(jnp.where(ready, pos, army.unit_positions[slot])))
        world = world.replace(smax_state=army,
                              unit_velocities=world.unit_velocities.at[slot].set(jnp.where(ready, jnp.zeros(2), world.unit_velocities[slot])),
                              attack_timers=world.attack_timers.at[slot].set(jnp.where(ready, 0, world.attack_timers[slot])),
                              persistent_targets=world.persistent_targets.at[slot].set(jnp.where(ready, -1, world.persistent_targets[slot])))
        return st.replace(world=world,
                           queue_kind=st.queue_kind.at[i].set(jnp.where(ready, -1, st.queue_kind[i])),
                           cargo=st.cargo.at[slot].set(jnp.where(ready, 0., st.cargo[slot])),
                           cargo_kind=st.cargo_kind.at[slot].set(jnp.where(ready, 0, st.cargo_kind[slot])),
                           worker_target=st.worker_target.at[slot].set(jnp.where(ready, -1, st.worker_target[slot])),
                           marines_built=st.marines_built + (ready & marine).astype(jnp.int32))
    state = jax.lax.fori_loop(0, env.building_capacity, produce, state)
    score = (state.gathered_minerals + state.gathered_gas) if env.spec.kind == "economy" else state.marines_built.astype(jnp.float32)
    return state.replace(score=score)
