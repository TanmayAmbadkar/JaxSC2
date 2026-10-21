# Scripted mechanics evaluation

Two deterministic baselines control individual allies:

- `beacon_agent(env, state)` heads for the beacon, using a bridge waypoint when it must cross the cliff.
- `combat_agent(env, state)` approaches enemies and attacks within each unit type's range. With fog enabled it uses sightings and last-seen memory, and searches map regions when it knows no enemy location.

These are mechanics baselines, not learned policies. Stationary enemies still auto-attack, and a simple scripted army is not guaranteed to beat a larger force before the 300-step limit. The report distinguishes reaching the objective, dealing damage, defeat, and timeout.

Both teams obey the same ditch footprint, including unit-radius clearance. Routes
remain aligned with a bridge until the complete unit reaches the opposite bank.
Every episode also records `terrain_valid` and `terrain_violations`, so visual and
objective success can be checked against internal terrain safety.

Run from the repository root in the `twobridge` environment:

```bash
conda run --no-capture-output -n twobridge env PYTHONPATH=. \
  python -m JaxSC2.scripted.evaluate --render all
```

This evaluates both agents on all nine variants with five settings: baseline, collisions disabled, armor disabled, fog enabled, and latency buffer length three. Each pair uses the same spawn seed. It writes 90 episode records to `runs/scripted_evaluation/results.json` and a GIF gallery to `runs/scripted_evaluation/index.html`.

For a focused demo:

```bash
conda run --no-capture-output -n twobridge env PYTHONPATH=. \
  python -m JaxSC2.scripted.evaluate --variants V1_Base V2_Combat \
  --settings baseline fog --enemy-modes static aggressive --render all \
  --output runs/scripted_demo
```

Use `--enemy-modes all` to cover static, guard, patrol, and aggressive opponents; `--seeds 0 42 123` for multiple layouts; `--render none` for metrics only. `--max-steps` explicitly changes the episode budget and is recorded in every result. Blue units are allies, red units are enemies, cyan is the beacon, and yellow marks projectiles. The rendered bridge openings match the movement constraints.

```python
import jax
from JaxSC2.env.env import JaxSC2Env
from JaxSC2.scripted import beacon_agent, combat_agent

env = JaxSC2Env("V1_Base")
obs, state = env.reset(jax.random.PRNGKey(0))
action = beacon_agent(env, state)
obs, state, reward, done, info = jax.jit(env.step)(jax.random.PRNGKey(1), state, action)
```
