"""Static scenario specifications and fixed-shape episode/action state."""

from dataclasses import dataclass as configclass
from typing import Protocol

from flax.struct import dataclass
import jax.numpy as jnp

from JaxSC2.env.env import JaxSC2State, PerUnitAction


@configclass(frozen=True)
class Weapon:
    damage: float
    range: float
    windup: int = 2
    cooldown: int = 8


@configclass(frozen=True)
class Unit:
    max_hp: float
    speed: float
    accel: float
    mass: float
    armor: float
    weapon: Weapon


# Adapted archetypes, not SC2 unit-stat replicas. Type zero is melee in the
# shared simulator; banelings have zero weapon range and use a suicide effect.
UNITS = (
    Unit(35., .32, .10, 1., 0., Weapon(5., 1.5, 1, 6)),       # zergling
    Unit(45., .25, .08, 1., 0., Weapon(6., 6.)),               # marine
    Unit(145., .22, .06, 3., 1., Weapon(12., 4., 3, 10)),      # roach
    Unit(30., .30, .09, 1., 0., Weapon(0., 0.)),               # baneling
    Unit(45., .25, .08, 1., 0., Weapon(0., 0.)),               # worker
)


@configclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    ally_capacity: int
    enemy_capacity: int
    initial_allies: int
    initial_enemies: int
    width: float = 32.
    height: float = 32.
    max_steps: int = 600
    fog: bool = False
    opponent: str = "aggressive"
    reinforcements: int = 0


TASKS = {
    s.name: s for s in (
        TaskSpec("MoveToBeacon", "beacon", 1, 1, 1, 0, max_steps=300, opponent="static"),
        TaskSpec("CollectMineralShards", "shards", 2, 1, 2, 0, opponent="static"),
        TaskSpec("FindAndDefeatZerglings", "search", 3, 25, 3, 25,
                 width=48., height=48., max_steps=1200, fog=True, opponent="static"),
        TaskSpec("DefeatRoaches", "roaches", 32, 4, 9, 4, max_steps=1200, reinforcements=5),
        TaskSpec("DefeatZerglingsAndBanelings", "banelings", 32, 10, 9, 10,
                 max_steps=1200, reinforcements=4),
        TaskSpec("CollectMineralsAndGas", "economy", 32, 1, 12, 0,
                 max_steps=1800, opponent="static"),
        TaskSpec("BuildMarines", "production", 48, 1, 12, 0,
                 max_steps=2400, opponent="static"),
    )
}


@dataclass
class TaskAction:
    units: PerUnitAction
    # Macro: 0 none, 1 command center, 2 depot, 3 barracks, 4 refinery,
    # 5 train worker, 6 train marine. Actor is a worker slot for construction,
    # or a building slot for training. Placement is a world coordinate.
    macro: jnp.ndarray
    actor: jnp.ndarray
    placement: jnp.ndarray


@dataclass
class TaskState:
    world: JaxSC2State
    entity_positions: jnp.ndarray
    entity_amount: jnp.ndarray
    entity_kind: jnp.ndarray  # 0 absent, 1 shard, 2 mineral field, 3 gas geyser
    building_positions: jnp.ndarray
    building_kind: jnp.ndarray  # 0 absent, 1 command center, 2 depot, 3 barracks, 4 refinery
    building_remaining: jnp.ndarray
    building_worker: jnp.ndarray
    queue_kind: jnp.ndarray  # -1 empty, 0 worker, 1 marine
    queue_remaining: jnp.ndarray
    worker_target: jnp.ndarray
    cargo: jnp.ndarray
    cargo_kind: jnp.ndarray
    minerals: jnp.ndarray
    gas: jnp.ndarray
    gathered_minerals: jnp.ndarray
    gathered_gas: jnp.ndarray
    marines_built: jnp.ndarray
    wave: jnp.ndarray
    score: jnp.ndarray
    done: jnp.ndarray

    @property
    def smax_state(self):
        """Compatibility view for army-policy rollouts; not duplicated state."""
        return self.world.smax_state


class Task(Protocol):
    """Objective lifecycle independent of terrain and simulation internals."""
    spec: TaskSpec
    def reset(self, key): ...
    def step(self, key, state: TaskState, action: TaskAction): ...
    def get_obs(self, state: TaskState): ...
