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

Use `--enemy-modes all` to cover all eleven opponents listed below;
`--seeds 0 42 123` for multiple layouts; `--render none` for metrics only.
The full nine-variant, five-setting, eleven-mode matrix has 990 episodes per seed.
`--max-steps` explicitly changes the episode budget and is recorded in every result.
Blue units are allies, red units are enemies, cyan is the beacon, and yellow marks
projectiles. The rendered bridge openings match the movement constraints.

## Enemy modes

Pass `enemy_ai=True, enemy_mode="..."` to `JaxSC2Env`, or use `--enemy-mode`
in the PPO launcher and `--enemy-modes` in this evaluation tool.

| Mode | Behavior |
| --- | --- |
| `static` | Stay in place; auto-attack enemies in range. |
| `guard` | Approach the ally centroid when within the aggression radius. |
| `patrol` | Cycle through four cardinal movement directions every 30 steps. |
| `aggressive` | Chase the living allies' centroid, routing through bridges. |
| `beacon_defender` | Assemble around the beacon; intercept nearby allies within a defensive leash. |
| `bridge_blocker` | Split across both bridge exits on the original bank; hold positions and intercept nearby allies. |
| `pursuer` | Chase individual exposed allies, favoring nearby, isolated, low-health targets. |
| `kiter` | Melee units close in; ranged/tank units retreat when too close or during cooldown, respecting windup locks. |
| `focus_fire` | Coordinate on one vulnerable ally; retarget when it dies. |
| `ambusher` | Stage near the closest bridge exit; suppress attacks until a living sentry sees an ally, then latch into pursuit. |
| `adaptive_defender` | Shift to the bridge nearest the current ally centroid; intercept local threats. |

Controllers share the normal movement, inertia, terrain, collisions, weapon range,
and attack-cycle mechanics. They do not teleport to defensive positions. Enemy
controllers use full simulation state, like the original AI; `enable_fow` limits
the player observation, not enemy perception. Ambush activation additionally
requires proximity within the configured vision radius. There is no stealth or
cover bonus. Adaptive defense is a reactive heuristic, not a learned strategy.

Enemy home positions and ambush alerts are episode state, reset by `env.reset`.
Advanced target selection is available through `env.get_enemy_orders(state)`;
`get_enemy_actions` remains a four-direction movement-only compatibility view.
Selecting one mode does not randomly mix modes between episodes.

Render just the seven new modes (14 episodes):

```bash
conda run --no-capture-output -n twobridge env PYTHONPATH=. \
  python -m JaxSC2.scripted.evaluate --variants V2_Base --settings baseline \
  --enemy-modes beacon_defender bridge_blocker pursuer kiter focus_fire ambusher adaptive_defender \
  --render all --output runs/scripted_combat_modes
```

Train a separate run against a selected opponent:

```bash
python examples/run_ppo.py --enemy-mode beacon_defender --reward-profile balanced \
  --total-steps 30000000 --background
```

The mode and exact reward weights are saved in each run's configuration; changing
the opponent is rejected when resuming an existing run.

For a 600-step training/evaluation horizon, add `--max-episode-steps 600` to the
PPO launcher. This is independent of `--rollout-len`: partial episode returns
carry across rollouts. The horizon is saved in `config.json`; resume must use
the original horizon as well as the same opponent and reward profile.

```python
import jax
from JaxSC2.env.env import JaxSC2Env
from JaxSC2.scripted import beacon_agent, combat_agent

env = JaxSC2Env("V1_Base")
obs, state = env.reset(jax.random.PRNGKey(0))
action = beacon_agent(env, state)
obs, state, reward, done, info = jax.jit(env.step)(jax.random.PRNGKey(1), state, action)
```
