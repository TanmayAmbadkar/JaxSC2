# Composable SC2-inspired task suite

This is the first functional adaptation of the seven DeepMind/PySC2 mini-game
ideas. These are **our own task implementations**, not imported `.SC2Map` files,
PySC2 action/API compatibility, or exact StarCraft II rules and statistics.
The source task descriptions are available in the
[official PySC2 documentation](https://github.com/google-deepmind/pysc2/blob/master/docs/mini_games.md).

## Architecture

```text
JaxSC2.make(task_name)
  ├─ terrain: ArenaMap / TwoBridgeMap
  ├─ simulator: JaxSC2Env.advance
  │    └─ inertia, collisions, swept terrain, weapons, projectiles, fog memory
  ├─ task: objective transitions or economy/production rules
  ├─ opponent: independent enemy controller
  └─ observations/actions: masked entities + task-specific macro interface
```

`advance` performs simulation only. The legacy `JaxSC2Env.step` still applies
TwoBridge rewards and beacon-or-combat termination. `MiniGameEnv.step` instead
applies its own rewards, wave/production events, and termination. Consequently,
combat tasks cannot end successfully by reaching a beacon.

Implementation locations:

- [Terrain contract and arena](../JaxSC2/maps/base.py)
- [Shared simulator](../JaxSC2/env/env.py)
- [Task specifications and state](../JaxSC2/tasks/types.py)
- [Task composition and entity observations](../JaxSC2/tasks/env.py)
- [Navigation and combat objectives](../JaxSC2/tasks/objectives.py)
- [Economy and production](../JaxSC2/tasks/economy.py)
- [Scripted task controllers](../JaxSC2/tasks/scripted.py)

## Tasks

| Name | Initial state | Objective / reward | Default step limit |
| --- | --- | --- | --- |
| `MoveToBeacon` | 1 marine, 1 beacon | +1 per reach; beacon relocates, episode continues | 300 |
| `CollectMineralShards` | 2 marines, 20 shards | +1 per shard; all shards respawn after clearance | 600 |
| `FindAndDefeatZerglings` | 3 marines, 25 stationary zerglings, fog, 48×48 arena | +1 per enemy defeated, -1 per allied death; enemy waves respawn | 1,200 |
| `DefeatRoaches` | 9 marines, 4 roaches | +10 per enemy defeated, -1 per allied death; new waves add up to 5 marines | 1,200 |
| `DefeatZerglingsAndBanelings` | 9 marines, 6 zerglings, 4 banelings | +5 per enemy defeated, -1 per allied death; new waves add up to 4 marines | 1,200 |
| `CollectMineralsAndGas` | 12 workers, command center, 16 mineral fields, 4 geysers | Deposited minerals + gas; spending does not reduce score | 1,800 |
| `BuildMarines` | 12 workers, command center, 8 mineral fields | +1 per newly produced marine | 2,400 |

These limits are simulation ticks, **not SC2 seconds**. Timed tasks end at the
limit; loss of all allied units also ends the episode. Combat wave clearance is
recorded but does not end the continuous wave task. Veterans preserve health;
wave repositioning, new enemies, and reinforcements are explicit task events.

The initial suite uses open rectangular arenas (32×32 except the search task).
These seven names represent different task mechanics, not seven reconstructed
SC2 terrain files. Geometry can be supplied independently:

```python
from JaxSC2 import make, available_tasks
from JaxSC2.maps.twobridge import TwoBridgeMap

env = make("DefeatRoaches", max_steps=600)
navigation_on_bridges = make("MoveToBeacon", terrain=TwoBridgeMap())
print(available_tasks())  # Nine original TwoBridge variants + seven new tasks
```

Arena obstacles use the same radius-aware swept/sliding collision solver.
Direct arena routing is not a general obstacle planner. Bridge-specific enemy
modes require bridge terrain; their use on an open arena fails explicitly.

## Entity capacities and observations

Episode arrays have static shapes for `jax.jit`, `jax.vmap`, and `jax.lax.scan`.
Inactive slots use alive/presence masks; producing or reinforcing units activates
slots without resizing arrays. Default allied capacities are 1, 2, 3, 32, 32, 32,
and 48 respectively; enemy capacities range from a single inactive placeholder
to 25 enemies. Increase allied capacity at construction, for example
`make("BuildMarines", ally_capacity=64)`.

At capacity, production is blocked; reinforcements are capped to free slots.
Queues reserve supply and unit capacity. Newly created units reset their timers,
velocity, cargo, jobs, and targeting state. Capacities are part of the scenario
and must be held fixed for comparable experiments.

Observations are dictionaries containing:

| Field | Shape | Meaning |
| --- | --- | --- |
| `units` | `(ally_capacity + enemy_capacity, 7)` | Normalized position, HP, type, presence, visibility, team |
| `unit_mask` | `(unit_capacity,)` | Living allies / known last-seen enemies |
| `entities` | `(24, 5)` | Resource/shard position, type, amount, active flag |
| `entity_mask` | `(24,)` | Nonempty resource/shard slots |
| `buildings` | `(16, 7)` | Position, type, construction, queue, completion |
| `building_mask` | `(16,)` | Occupied structure slots |
| `global` | `(12,)` | Time, score, resources, supply, waves, production, beacon goal |
| `vector` | `(7U + 5×24 + 7×16 + 12,)` | Flattened policy input; `U` is total unit capacity |

Hidden enemies expose last-seen memory, not current hidden coordinates or health.
The search task enables fog by default. Resources and owned structures are known.
There is no persistent explored-terrain grid, high-ground occlusion, or cloaking.

The original TwoBridge vector remains **63-dimensional** for its standard roster.
Old checkpoints must be evaluated on their original environment/reward profile;
they are not interchangeable with new entity schemas.

## Actions and economy

`TaskAction` contains per-unit `PerUnitAction` plus a single macro command:

```python
import jax
import jax.numpy as jnp
from JaxSC2 import make

env = make("BuildMarines")
obs, state = env.reset(jax.random.PRNGKey(0))
action = env.no_op().replace(
    macro=jnp.int32(5),   # Queue a worker for 50 minerals
    actor=jnp.int32(0),   # Command-center building slot
)
obs, state, reward, done, info = jax.jit(env.step)(jax.random.PRNGKey(1), state, action)
```

Unit verbs: 0 NOOP, 1 MOVE, 2 ATTACK, 3 GATHER. GATHER's target is a resource
slot; ATTACK's target is an enemy slot. Gather jobs persist until overridden.
Full cargo, or cargo from an exhausted node, returns to a completed command center.
Workers cannot mix resource types in one cargo load. No deposit is scored twice.

| Macro | Actor | Mineral cost | Duration (ticks) |
| --- | --- | --- | --- |
| 0 none | Ignored | 0 | 0 |
| 1 command center | Worker unit slot | 400 | 60 |
| 2 supply depot | Worker unit slot | 100 | 20 |
| 3 barracks | Worker unit slot | 150 | 35 |
| 4 refinery | Worker unit slot | 75 | 20 |
| 5 worker | Completed command-center slot | 50 | 12 |
| 6 marine | Completed barracks slot | 50 | 18 |

Construction requires a living, available worker near a valid placement, a free
structure slot, and sufficient money. Barracks require a completed depot;
refineries must be placed at geysers. Only completed refineries permit gas
harvesting. Command centers provide 15 supply; completed depots add 8.
Invalid commands are no-ops and reported as `invalid_macro` without charging.
Buildings construct over time; units appear only when paid queues finish.

`get_action_mask` exposes army masks, per-unit `unit_verb`, resource masks, and
macro availability. Macro masks are necessary conditions; actor and placement
validity is checked in `step`. Structure footprints are not movement blockers,
and structures cannot be attacked or destroyed in this first version. There are
no upgrades, repairs, refunds, queue cancellation, or full technology trees.

## Baselines and evaluation

```bash
conda run --no-capture-output -n twobridge env PYTHONPATH=. \
  python -m JaxSC2.tasks.evaluate --seeds 0 42 --render none

conda run --no-capture-output -n twobridge env PYTHONPATH=. \
  python -m JaxSC2.tasks.evaluate --seeds 42 --render all --output runs/task_suite_gallery
```

Open `runs/task_suite_gallery/index.html`. GIFs are an **observer view**: hidden
enemies may be drawn, while scripted policies use only sightings and memory.
JSON records scores, wave counts, resource totals, production, capacities, fog,
opponent, horizons, and terrain violations. `task_success` means positive score
for navigation/economy, or at least one cleared wave for combat. It does not mean
the task is solved optimally. Raw scores are not comparable across task families.

Seed-42 scripted validation in the initial implementation:

| Task | Recorded progress |
| --- | --- |
| MoveToBeacon | 4 beacon reaches |
| CollectMineralShards | 32 shards, including one complete respawn cycle |
| FindAndDefeatZerglings | 24 enemies defeated; no full wave cleared |
| DefeatRoaches | 2 waves cleared; eventually lost all allied units |
| DefeatZerglingsAndBanelings | 7 waves cleared |
| CollectMineralsAndGas | 1,090 minerals and 230 gas deposited |
| BuildMarines | 20 marines produced |

All seven runs recorded zero terrain violations. These are functionality checks,
not multi-seed algorithm benchmark results or claims of SC2-equivalent behavior.

## Learning support and remaining work

Existing PPO/MaskPPO can consume the entity vector and army action interface for
the five navigation/combat tasks. They still broadcast a shared command; these
are not decentralized per-unit learners. Use `PPO(...).train(make("MoveToBeacon"), steps)`.

For a short, isolated training experiment (not launched automatically):

```bash
python examples/run_task_ppo.py --task MoveToBeacon --total-steps 100000
python examples/run_task_ppo.py --task DefeatRoaches --algorithm mask_ppo --max-steps 600
```

This new launcher rounds the target to complete rollout batches, records its
configuration, and saves a final checkpoint/evaluation in a fresh run directory.
It is foreground-only; existing long-running TwoBridge launchers are unchanged.

**The economy tasks currently have scripted macro controllers and a complete
environment action interface, but no learned macro-policy trainer.** Existing
army-only PPO/MaskPPO reject them explicitly rather than silently omitting macro
actions. A learner with unit/resource selection, build placement, and macro heads
is a separate next step. No long training runs are automatically launched.

Other follow-ups include richer terrain routing, additional layouts, persistent
exploration, attackable structures, policy memory, per-unit learned control,
multi-seed baselines, and held-out opponent/map evaluation.
