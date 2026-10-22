"""Task facade composing a terrain, shared simulator, objective, and opponent."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from JaxSC2.env.env import JaxSC2Env, PerUnitAction, DictWrapper
from JaxSC2.env.mechanics import FogState, update_fog_memory
from JaxSC2.maps.base import ArenaMap
from JaxSC2.maps.twobridge import VariantConfig
from JaxSC2.tasks.types import TASKS, UNITS, TaskState, TaskAction
from JaxSC2.tasks import economy
from JaxSC2.tasks.objectives import transition


class MiniGameEnv:
    """SC2-inspired adaptations with fixed capacities and masked inactive slots.

    Terrain and unit dynamics are independent of objective rewards. These tasks
    are not binary SC2-map imports, PySC2 API replicas, or exact unit-stat ports.
    """
    entity_capacity = 24
    building_capacity = 16

    def __init__(self, name, terrain=None, max_steps=None, ally_capacity=None,
                 enemy_mode=None, enable_fow=None, **world_options):
        if name not in TASKS:
            raise ValueError(f"Unknown task {name!r}; choose from {tuple(TASKS)}")
        spec = TASKS[name]
        if ally_capacity is not None:
            if ally_capacity < spec.initial_allies:
                raise ValueError("Ally capacity must fit the initial army")
            spec = replace(spec, ally_capacity=ally_capacity)
        self.spec, self.name = spec, name
        self.max_steps = spec.max_steps if max_steps is None else max_steps
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.map = terrain if terrain is not None else ArenaMap(spec.width, spec.height, name=name)
        if enemy_mode in ("bridge_blocker", "adaptive_defender", "ambusher") and not self.map.BRIDGE_Y:
            raise ValueError(f"Enemy mode {enemy_mode} requires a terrain with bridge anchors")
        self.num_allies, self.num_enemies = spec.ally_capacity, spec.enemy_capacity
        self.num_units = self.num_allies + self.num_enemies
        self.economic = spec.kind in ("economy", "production")
        self.core = JaxSC2Env(
            name, map_instance=self.map,
            variant_config=VariantConfig(0, name, self.num_allies, self.num_enemies, 0),
            units=UNITS, max_enemy_slots=self.num_enemies,
            projectile_capacity=max(32, 2 * self.num_units),
            enemy_ai=spec.initial_enemies > 0,
            enemy_mode=spec.opponent if enemy_mode is None else enemy_mode,
            enable_fow=spec.fog if enable_fow is None else enable_fow,
            **world_options,
        )
        self.core.max_steps = self.max_steps
        if spec.kind == "banelings":
            self.core.combat_cfg.update(suicide_type=3, suicide_trigger=1.6,
                                        suicide_radius=2.2, suicide_damage=35.)
        self.map_width, self.map_height = self.core.map_width, self.core.map_height
        self.dims = jnp.array([self.map_width, self.map_height])
        self.observation_space = DictWrapper({
            "shape": (7 * self.num_units + 5 * self.entity_capacity + 7 * self.building_capacity + 12,)})
        self.action_space = DictWrapper({"nvec": jnp.array([4, 8, max(self.num_enemies, self.entity_capacity)])})

    def no_op(self):
        zeros = jnp.zeros(self.num_allies, dtype=jnp.int32)
        units = PerUnitAction(jnp.ones(self.num_allies, dtype=bool), zeros, zeros, zeros - 1)
        return TaskAction(units, jnp.int32(0), jnp.int32(0), jnp.zeros(2))

    def _refresh(self, world, clear_memory=False):
        s = world.smax_state
        n = self.num_allies
        fog = world.fog_state
        if clear_memory:
            fog = FogState(jnp.zeros((self.num_enemies, 2)), jnp.zeros(self.num_enemies, dtype=bool),
                           jnp.zeros(self.num_enemies), jnp.full(self.num_enemies, -1, dtype=jnp.int32))
        fog, visible = update_fog_memory(
            fog, s.unit_positions[:n], s.unit_alive[:n], s.unit_positions[n:],
            s.unit_alive[n:], s.unit_health[n:], self.core.vision_radius, world.timestep)
        if not self.core.enable_fow:
            visible = jnp.ones(self.num_enemies, dtype=bool)
            fog = FogState(s.unit_positions[n:], s.unit_alive[n:], s.unit_health[n:],
                           jnp.full(self.num_enemies, world.timestep, dtype=jnp.int32))
        return world.replace(fog_state=fog, enemy_visible=visible & s.unit_alive[n:])

    def spawn_armies(self, key, world, initial=False):
        """Wave reset: preserve veteran health, add bounded reinforcements."""
        spec, n = self.spec, self.num_allies
        s = world.smax_state
        flip = jax.random.bernoulli(key)
        ai, ei = jnp.arange(n), jnp.arange(self.num_enemies)
        allies = jnp.stack([5. + (ai // 12) * 1.4,
                            self.map_height / 2 + (ai % 12 - 5.5) * 1.7], axis=-1)
        enemies = jnp.stack([self.map_width - 5. - (ei // 12) * 1.4,
                             self.map_height / 2 + (ei % 12 - 5.5) * 1.7], axis=-1)
        positions = jnp.concatenate([allies, enemies])
        positions = positions.at[:, 0].set(jnp.where(flip, self.map_width - positions[:, 0], positions[:, 0]))
        positions = self.map.enforce_constraints(positions, positions, self.map_width,
                                                  self.map_height, self.core.unit_teams)
        types = jnp.concatenate([jnp.ones(n, dtype=jnp.int32),
                                jnp.full(self.num_enemies, 2 if spec.kind == "roaches" else 0, dtype=jnp.int32)])
        if spec.kind == "banelings":
            types = types.at[n + 6:].set(3)
        if initial:
            alive = jnp.concatenate([ai < spec.initial_allies, ei < spec.initial_enemies])
            hp = jnp.where(alive, self.core.unit_type_hp[types], 0.)
        else:
            available = ~s.unit_alive[:n]
            added = available & (jnp.cumsum(available) <= spec.reinforcements)
            ally_alive = s.unit_alive[:n] | added
            alive = jnp.concatenate([ally_alive, jnp.ones(self.num_enemies, dtype=bool)])
            hp = jnp.concatenate([jnp.where(added, self.core.unit_type_hp[1], s.unit_health[:n]),
                                   self.core.unit_type_hp[types[n:]]])
        if spec.kind == "search":
            center = self.dims / 2
            positions = positions.at[:n].set(center + jnp.stack([ai * 1.4 - 1.4, jnp.zeros(n)], axis=-1))
            candidates = self.map.sample_positions(key, self.num_enemies * 64).reshape((self.num_enemies, 64, 2))
            far = jnp.linalg.norm(candidates - center, axis=-1) > 10.
            enemies = candidates[jnp.arange(self.num_enemies), jnp.argmax(far, axis=1)]
            enemies = enemies.at[:2].set(center + jnp.array([[5., 0.], [0., 5.]]))
            positions = positions.at[n:].set(enemies)
        world = world.replace(
            smax_state=s.replace(unit_positions=positions, unit_types=types, unit_alive=alive,
                                 unit_health=hp, terminal=False),
            unit_velocities=jnp.zeros_like(world.unit_velocities),
            persistent_targets=jnp.full_like(world.persistent_targets, -1),
            attack_timers=jnp.zeros_like(world.attack_timers), proj_active=jnp.zeros_like(world.proj_active),
            enemy_home=positions[n:], enemy_alerted=jnp.zeros(self.num_enemies, dtype=bool))
        return self._refresh(world, clear_memory=True)

    def distant_beacon(self, key, positions):
        candidates = self.map.sample_positions(key, 64)
        valid = jnp.all(jnp.linalg.norm(candidates[:, None, :] - positions[None, :, :], axis=-1) > 5., axis=1)
        return candidates[jnp.argmax(valid)]

    def reset(self, key):
        key, world_key, spawn_key, entity_key = jax.random.split(key, 4)
        _, world = self.core.reset(world_key)
        world = self.spawn_armies(spawn_key, world, initial=True)
        spec, n, k, b = self.spec, self.num_allies, self.entity_capacity, self.building_capacity
        if spec.kind in ("beacon", "shards"):
            positions = self.map.sample_positions(spawn_key, self.num_units)
            world = world.replace(smax_state=world.smax_state.replace(unit_positions=positions))
            world = world.replace(beacon_pos=self.distant_beacon(key, positions[:n]))
        state = TaskState(
            world=world, entity_positions=self.map.sample_positions(entity_key, k),
            entity_amount=jnp.zeros(k), entity_kind=jnp.zeros(k, dtype=jnp.int32),
            building_positions=jnp.zeros((b, 2)), building_kind=jnp.zeros(b, dtype=jnp.int32),
            building_remaining=jnp.zeros(b, dtype=jnp.int32), building_worker=jnp.full(b, -1, dtype=jnp.int32),
            queue_kind=jnp.full(b, -1, dtype=jnp.int32), queue_remaining=jnp.zeros(b, dtype=jnp.int32),
            worker_target=jnp.full(n, -1, dtype=jnp.int32), cargo=jnp.zeros(n),
            cargo_kind=jnp.zeros(n, dtype=jnp.int32), minerals=jnp.float32(0.), gas=jnp.float32(0.),
            gathered_minerals=jnp.float32(0.), gathered_gas=jnp.float32(0.),
            marines_built=jnp.int32(0), wave=jnp.int32(0), score=jnp.float32(0.), done=jnp.bool_(False))
        if spec.kind == "shards":
            state = state.replace(entity_kind=state.entity_kind.at[:20].set(1),
                                  entity_amount=state.entity_amount.at[:20].set(1.))
        if self.economic:
            from JaxSC2.tasks.economy import initialize
            state = initialize(self, state)
        state = state.replace(world=self._refresh(state.world, clear_memory=True))
        return self.get_obs(state), state

    def step(self, key, state, action):
        from JaxSC2.tasks.objectives import transition
        from JaxSC2.tasks.economy import prepare, advance_economy
        if isinstance(action, PerUnitAction):
            action = self.no_op().replace(units=action)
        key, physics_key = jax.random.split(key)
        prepared, units, accepted = prepare(self, state, action) if self.economic else (state, action.units, action.macro == 0)
        world, physics = self.core.advance(physics_key, prepared.world, units)
        if self.economic:
            next_state = advance_economy(self, prepared.replace(world=world))
        else:
            next_state = transition(self, key, state, world)
        # Objectives, not the simulator, decide episode termination.
        failed = ~jnp.any(next_state.world.smax_state.unit_alive[:self.num_allies])
        done = failed | (next_state.world.timestep >= self.max_steps)
        next_state = next_state.replace(done=done, world=next_state.world.replace(
            smax_state=next_state.world.smax_state.replace(terminal=done)))
        next_state = jax.tree_util.tree_map(lambda old, new: jnp.where(state.done, old, new), state, next_state)
        reward = jnp.where(state.done, 0., next_state.score - state.score)
        info = self.info(state, next_state)
        info["invalid_macro"] = ~state.done & ~accepted
        info["dropped_shots"] = jnp.where(state.done, 0, physics["dropped_shots"])
        return self.get_obs(next_state), next_state, reward, next_state.done, info

    def info(self, previous, state):
        combat = self.spec.kind in ("search", "roaches", "banelings")
        navigation = self.spec.kind in ("beacon", "shards")
        success = (state.wave > 0) if combat else (state.score > 0)
        return {
            "score": state.score, "task_success": success, "win": success,
            "nav_success": success & navigation, "combat_success": success & combat,
            "wave": state.wave, "minerals": state.minerals, "gas": state.gas,
            "marines_built": state.marines_built,
            "capacity_exhausted": jnp.all(state.world.smax_state.unit_alive[:self.num_allies]),
        }

    def get_obs(self, state):
        world, s, n = state.world, state.world.smax_state, self.num_allies
        known = world.fog_state.last_seen_alive & (world.fog_state.last_seen_time >= 0)
        mask = jnp.concatenate([s.unit_alive[:n], known])
        pos = jnp.concatenate([s.unit_positions[:n], world.fog_state.last_seen_pos])
        hp = jnp.concatenate([s.unit_health[:n], world.fog_state.last_seen_hp])
        units = jnp.concatenate([
            (pos - self.dims / 2) / self.dims,
            (hp / self.core.unit_type_hp[s.unit_types])[:, None],
            (s.unit_types / (len(UNITS) - 1))[:, None], mask[:, None],
            jnp.concatenate([s.unit_alive[:n], world.enemy_visible])[:, None],
            (self.core.unit_teams == 0)[:, None]], axis=-1) * mask[:, None]
        active_entities = state.entity_amount > 0
        entities = jnp.concatenate([state.entity_positions / self.dims, (state.entity_kind / 3)[:, None],
                                    (state.entity_amount / 2000)[:, None], active_entities[:, None]], axis=-1)
        entities *= active_entities[:, None]
        active_buildings = state.building_kind > 0
        buildings = jnp.concatenate([
            state.building_positions / self.dims, (state.building_kind / 4)[:, None],
            (state.building_remaining / 80)[:, None], ((state.queue_kind + 1) / 2)[:, None],
            (state.queue_remaining / 20)[:, None], (state.building_remaining == 0)[:, None]], axis=-1)
        buildings *= active_buildings[:, None]
        from JaxSC2.tasks.economy import supply
        used, capacity = supply(state)
        goal = world.beacon_pos / self.dims if self.spec.kind == "beacon" else jnp.zeros(2)
        globals_ = jnp.concatenate([jnp.array([
            world.timestep / self.max_steps, state.score / 100, state.minerals / 1000,
            state.gas / 1000, state.gathered_minerals / 1000, state.gathered_gas / 1000,
            used / self.num_allies, capacity / 200, state.wave / 10, state.marines_built / 100]), goal])
        return {"vector": jnp.concatenate([units.ravel(), entities.ravel(), buildings.ravel(), globals_]),
                "units": units, "unit_mask": mask, "entities": entities, "entity_mask": active_entities,
                "buildings": buildings, "building_mask": active_buildings, "global": globals_}

    def get_action_mask(self, state):
        from JaxSC2.tasks.economy import macro_mask
        alive = state.world.smax_state.unit_alive[:self.num_allies] & ~state.done
        can_attack = jnp.any(state.world.enemy_visible) & jnp.any(alive)
        verbs = jnp.broadcast_to(jnp.array([True, True, can_attack, self.economic]),
                                 (self.num_allies, 4)) & alive[:, None]
        verbs = verbs.at[:, 3].set(alive & (state.world.smax_state.unit_types[:self.num_allies] == 4) & self.economic)
        return {"who": alive, "unit_verb": verbs, "verb": jnp.array([True, jnp.any(alive), can_attack]),
                "direction": jnp.ones(8, dtype=bool),
                "target": state.world.smax_state.unit_alive[self.num_allies:] & state.world.enemy_visible,
                "entities": state.entity_amount > 0, "macro": macro_mask(self, state)}
