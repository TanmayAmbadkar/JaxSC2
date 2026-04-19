# Visualization Reference Guide

The `JaxSC2/visualizations/` directory contains scripts for generating GIFs, interactive demos, and spatial observation plots. All scripts use the `ProductionRenderer` for frame-by-frame rendering and operate on 32×32 (or configurable resolution) Twobridge maps.

---

## Quick Start — Generate Your First GIF

```bash
# The simplest entry point: smart agent demo with spatial observation plots
python JaxSC2/visualizations/render_demo.py

# Outputs (in current directory):
#   combat_nav.gif     — Smart agent navigating toward beacon + fighting enemies
#   chaos_nav.gif      — Random action demo for comparison
#   spatial_obs.gif    — 2×4 subplot showing terrain/ally/enemy/health at each timestep
```

---

## Script Reference

| Script | Purpose | Run Command | Output Files |
|---|---|---|---|
| `demo_suite.py` | **Full configurable demo runner** — batch across all variants/enemy modes | `python JaxSC2/visualizations/demo_suite.py` | Multiple GIFs in `output/` |
| `full_demo.py` | **Comprehensive scenario showcase** — combat + navigation agents | `python JaxSC2/visualizations/full_demo.py` | Multiple GIFs in `output/` |
| `combat_showcase.py` | **Combat-focused** — different AI engagement styles | `python JaxSC2/visualizations/combat_showcase.py` | GIFs in `output/` |
| `navigation_showcase.py` | **Navigation-focused** — beacon pathfinding + bridge crossing | `python JaxSC2/visualizations/navigation_showcase.py` | GIFs in `output/` |
| `run_ui.py` | **Interactive Pygame UI** — live stepping with window | `python JaxSC2/visualizations/run_ui.py` | Pygame window (no file output) |
| `render_demo.py` | **Smart/chaos agent demo** — with spatial observation plots | `python JaxSC2/visualizations/render_demo.py` | 3 GIFs in current dir |

---

## Detailed Script Descriptions

### `demo_suite.py` — Batch Demo Runner

Runs a comprehensive set of demos across all variant/enemy-mode combinations. Supports CLI filtering to limit output.

**CLI Arguments:**
```bash
# Run all combat demos only
python JaxSC2/visualizations/demo_suite.py --mode combat

# Run navigate-only demos
python JaxSC2/visualizations/demo_suite.py --mode navigate

# Run specific variant only
python JaxSC2/visualizations/demo_suite.py --variant V3_Combat

# Combined filter
python JaxSC2/visualizations/demo_suite.py --mode combat --variant V1_Combat
```

**Internal Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `mode_filter` | `None` | If set, only run demos matching mode ("combat" or "navigate") |
| `variant_filter` | `None` | If set, only run demos matching variant (e.g., "V2_Base") |
| `rng_seed` | 42 | PRNG seed for reproducibility |
| `max_steps` | 300 | Maximum steps per episode before truncation |

**Smart Agent Logic:**
The demo suite uses a heuristic "smart" agent (not a trained policy) for visualization:

```python
# 1. Compute centroid of all alive allies
centroid = mean(ally_positions[ally_alive])

# 2. Find closest living enemy
enemy_distances = norm(centroid - enemy_positions, axis=-1)
closest_enemy = argmin(where(enemy_alive, distances, 999.0))

# 3. Decide action based on distance to closest enemy
if distance_to_closest < 7.0:        # In weapon range
    action = ATTACK(target=closest_enemy, direction=0)
else:
    if enemies_alive.any():           # Engage enemies first
        action = MOVE(direction=direction_toward(closest_enemy))
    else:                             # All enemies dead → go to beacon
        action = MOVE(direction=direction_toward(beacon))

# Bridge routing: if agent and target are on opposite sides of cliff,
# route through nearest bridge midpoint first then toward final target
```

**Outputs:**
- GIFs saved to `JaxSC2/visualizations/output/` directory (created automatically)
- Console progress: `Running demo: V2_Base / aggressive ...`

---

### `full_demo.py` — Comprehensive Scenario Showcase

Runs a full suite of combat and navigation scenarios with dedicated agent logic for each mode.

**Agent Functions:**

```python
combat_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos, num_allies, combat_cfg)
```

- **Behavior:** Prioritizes attacking nearest enemies; reaches beacon only after all enemies are eliminated.
- **Decision:** If any enemy is in weapon range → ATTACK; else if enemies alive → MOVE toward closest; else → MOVE toward beacon.

```python
navigation_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos, num_allies, combat_cfg)
```

- **Behavior:** Prioritizes reaching the beacon; ignores enemies unless they block the path.
- **Decision:** If enemy is at beacon position AND in range → slight avoidance maneuver; else → MOVE directly toward beacon.

```python
run_scenario(env, agent_fn, max_steps=300, seed=42)
```

- Runs a single episode using the provided agent function. Collects frames and metadata, then calls `save_gif()`.

```python
save_gif(trajectory, metadata, filename)
```

- Renders the trajectory using `ProductionRenderer` and saves to disk.

**What It Demonstrates:**
- Combat agent successfully engaging and eliminating enemies before navigating to beacon
- Navigation agent reaching the beacon even with passive enemy presence
- Bridge crossing strategies (splitting force, timing)
- Type advantage exploitation (if the agent has learned composition play via training)

---

### `combat_showcase.py` — Combat-Focused Demos

Generates GIFs showing different enemy AI behaviors and combat outcomes. Ideal for:
- Comparing trained agent performance against random/enemy AI
- Demonstrating type advantage mechanics (Melee→Ranged, Ranged→Tank, Tank→Melee)
- Visualizing projectile trajectories and damage application

**Enemy AI Modes Demonstrated:**

| Mode | Behavior | Visual Characteristic |
|---|---|---|
| `"random"` | Random MOVE/ATTACK with random direction/target | Chaotic, unpredictable movement; agents cluster randomly |
| `"aggressive"` | Always moves toward nearest enemy and attacks when in range | Forward pressure; agents rush together |
| *(future modes)* | — | Extendable for defensive, flanking, etc. |

**What It Demonstrates:**
- Projectile ballistics (Gauss Rifle and Plasma Cannon trajectories)
- Windup/cooldown visual effects — agents pause between attacks
- Collision resolution — units repelling each other during close combat
- Damage application and health bar changes (via `state_to_frame`)

---

### `navigation_showcase.py` — Navigation-Focused Demos

Generates GIFs showing beacon navigation, with emphasis on:
- Bridge crossing strategies (which bridge to use when)
- Pathfinding around cliff walls
- Multi-unit coordination toward a common target

**What It Demonstrates:**
- Agents learning to route through the correct bridge (based on spawn position)
- Units maintaining formation while crossing (or failing to — for untrained agents)
- Beacon proximity detection and final approach behavior

---

### `run_ui.py` — Interactive Pygame UI (Hunter Mode)

Opens a live Pygame window that renders the simulation in real-time. You can step through episodes manually and observe the environment state at each timestep.

**Behavior:**
```python
run_interactive_demo()
# Creates JaxSC2Env → resets → steps through trajectory using render_demo's
# get_smart_direction function for direction logic. Opens a Pygame window
# showing the current frame at each step. Press any key to advance.
```

**Usage:**
```bash
# Requires Pygame installed (already in requirements.txt)
python JaxSC2/visualizations/run_ui.py

# Opens a window showing the TwoBridge map.
# Each key press advances one simulation step.
# Press Q or close the window to exit.
```

**What Makes It Useful:**
- Debugging: Observe exactly what the environment state looks like at each timestep (positions, projectiles, health)
- Agent inspection: Load a trained model and replace the smart agent with your policy to see how it plays
- Education: Step through slowly to understand the 12-phase step loop (collision → combat → fog → reward)

**Output:** Interactive window only — no file output. Use `ProductionRenderer.render_episode()` with a collected trajectory if you want to save the result as a GIF.

---

### `render_demo.py` — Smart/Chaos Agent with Spatial Observations

The most feature-rich demo script. Generates three types of output:
1. **Main GIF** (`combat_nav.gif`): Standard frame-by-frame rendering of the simulation
2. **Chaos GIF** (`chaos_nav.gif`): Same as above but with random actions
3. **Spatial Observation GIF** (`spatial_obs.gif`): 2×4 subplot showing terrain, ally positions, enemy positions, and health maps

**Function:**
```python
run_demonstration(
    variant="V2_Base",          # Scenario variant (see §6 of Instructions.md)
    out_path="combat_nav.gif",  # Output path for main GIF
    out_spatial="spatial_obs.gif",  # Output path for spatial observation subplot
    mode="smart"                # "smart" (heuristic agent) or "chaos" (random actions)
)
```

**Spatial Observation Subplots:**

The 2×4 grid shows:
| Row | Column | Content |
|---|---|---|
| **Top (2D)** | 0: Terrain | Grayscale map showing cliff and bridges |
| | 1: Ally Positions | Heatmap of ally unit locations |
| | 2: Enemy Positions | Heatmap of visible enemy locations (zeros for hidden) |
| | 3: Ally Health | Colored heatmap of ally HP |
| **Bottom (Minimaps)** | 4-7: Minimap variants | Various aggregated views of the map state |

**Smart Direction Logic:**
```python
def get_smart_direction(pos, target):
    # Determine if crossing the cliff is needed
    on_left = pos[0] < 14.4          # Left of cliff gap
    target_on_right = target[0] > 17.6  # Right of cliff gap
    
    if on_left and target_on_right or on_right and target_on_left:
        # Must cross cliff — route through nearest bridge midpoint first
        bridge_y = 8.0 if pos[1] < 16 else 24.0
        bridge_x = 14.0 if on_left else 18.0
        sub_target = [bridge_x, bridge_y]
    else:
        sub_target = target
    
    # Convert to 8-octant direction angle
    diff = sub_target - pos
    angle = (arctan2(diff[0], diff[1]) + 2π) % 2π
    return round(angle / (π/4)) % 8   # Map to direction index 0-7
```

**Dependencies:** Requires `matplotlib` (Agg backend, set via `matplotlib.use('Agg')`) and `imageio`. The spatial subplot uses matplotlib's canvas buffer for frame extraction.

---

## Customizing Demos

### Adding a New Agent Policy

To create your own agent visualization script, follow this pattern:

```python
import jax
from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

def my_agent(ally_pos, ally_alive, enemy_pos, enemy_alive, beacon_pos):
    """Return a CentralAction for each alive ally."""
    # Your logic here — return verb, direction, target as integers
    
    centroid = jnp.mean(ally_pos[ally_alive], axis=0)
    nearest_enemy_idx = argmin(where(enemy_alive, norm(centroid - enemy_pos), 999.0))
    
    if distance_to(centroid, enemy_pos[nearest_enemy_idx]) < 7.0:
        return CentralAction(who_mask=ally_alive, verb=2, direction=0, target=int(nearest_enemy_idx))
    else:
        return CentralAction(who_mask=ally_alive, verb=1, direction=get_direction(centroid, beacon_pos), target=0)

# Run
env = JaxSC2Env(variant_name="V1_Base", enemy_ai=True, enemy_mode="aggressive")
rng = jax.random.PRNGKey(42)
obs, state = env.reset(rng)

trajectory = [state_to_frame(state)]
for _ in range(300):
    rng, step_rng = jax.random.split(rng)
    action = my_agent(state.smax_state.unit_positions[:5],
                      state.smax_state.unit_alive[:5],
                      state.smax_state.unit_positions[5:],
                      state.smax_state.unit_alive[5:],
                      state.beacon_pos)
    obs, state, _, done, _ = env.step(step_rng, state, action)
    trajectory.append(state_to_frame(state))
    if done: break

renderer = ProductionRenderer(headless=True, trails_enabled=True)
renderer.render_episode(trajectory, save_path="my_agent.gif", interp_steps=4)
```

### Customizing the Renderer

| Parameter | Default | Description |
|---|---|---|
| `headless` | `False` | Run without display (needed for server/GIF-only output) |
| `trails_enabled` | `True` | Show displacement trails behind moving units |

```python
renderer = ProductionRenderer(headless=True, trails_enabled=False)  # No display, no trails
```

### Adjusting Observation Resolution

The `use_spatial_obs` and `resolution` parameters in `JaxSC2Env.__init__`:

```python
# Low-res spatial (fast, for debugging)
env = JaxSC2Env(use_spatial_obs=True, resolution=16)

# High-res spatial (good for GIFs)
env = JaxSC2Env(use_spatial_obs=True, resolution=64)

# Default vector observation (no spatial maps)
env = JaxSC2Env(use_spatial_obs=False, resolution=32)  # resolution only matters for spatial
```

---

## Output Directory

All scripts create or write to `JaxSC2/visualizations/output/` unless otherwise specified. This directory is:
- Created automatically on first run (via `os.makedirs`)
- Not tracked in git (add to `.gitignore` if it isn't already)
- Contains one GIF per scenario run

```bash
# Clean output before regenerating all demos
rm -rf JaxSC2/visualizations/output/*

# Regenerate everything
python JaxSC2/visualizations/demo_suite.py
```

---

## Troubleshooting Visualization Scripts

### "ModuleNotFoundError: No module named 'JaxSC2'"
```bash
export PYTHONPATH=$PYTHONPATH:.  # Or: cd to project root and add . to sys.path
```

### "pygame.error: No available video device" (UI scripts)
Run with `headless=True` or on a server:
```bash
# For rendering without display
python JaxSC2/visualizations/render_demo.py  # Already headless (uses ProductionRenderer(headless=True))

# For UI scripts on a server, use Xvfb:
xvfb-run python JaxSC2/visualizations/run_ui.py
```

### "Matplotlib backend error" (render_demo spatial plots)
Ensure matplotlib's Agg backend is set before importing pyplot:
```python
import matplotlib
matplotlib.use('Agg')  # Must come before plt import
import matplotlib.pyplot as plt
```

### "GIF output empty or frozen"
The trajectory list must include the initial frame:
```python
trajectory = [state_to_frame(state)]  # Initial frame at t=0
for step in range(300):
    # ... step env ...
    trajectory.append(state_to_frame(state))  # Frame after each step
# If done=True before appending any frames, the GIF will be empty
```
