# Maps, Rewards, and Constraints

This directory contains the logic governing environmental layouts, reward shaping, and physical constraints. All terrain is defined on a normalized 0–1 grid that maps to the actual simulation resolution (default: 32×32).

---

## 🗺 Twobridge Map (`twobridge.py`)

### Layout Overview

The default map is a **32×32 grid** with a vertical cliff wall at x=0.5 (normalized) that splits the map into left (ally) and right (enemy) halves. Units must cross through one of two bridges to reach the other side.

### Terrain Features

| Feature | Position (normalized) | Position (32×32 grid) | Effect |
|---|---|---|---|
| **Cliff wall** | x = 0.5 | x = 16 (full height) | Blocks all movement except at bridge gaps |
| **Bridge 1** | y ∈ (0.2, 0.3) | y = 6–10 | Allows passage through cliff wall |
| **Bridge 2** | y ∈ (0.7, 0.8) | y = 22–26 | Allows passage through cliff wall |
| **Ally spawn regions** | 6 boxes on left half (x < 0.5) | x ∈ [0, 16) | Procedural spawn within these boxes |
| **Enemy spawn regions** | 6 boxes on right half (x > 0.5) | x ∈ [16, 32] | Procedural spawn within these boxes |
| **Beacon** | Right half, near center | ~ (x>16, y≈16) | Navigation target for ally units |

### Regional Coordinates (`REGION_COORDS`)

Six spawn regions define where allies and enemies can appear. Each is a normalized box: `[xmin, xmax, ymin, ymax]`.

```
Region 0: [0.05, 0.20, 0.10, 0.30]    # Top-left
Region 1: [0.25, 0.40, 0.10, 0.30]    # Upper-left
Region 2: [0.05, 0.20, 0.70, 0.90]    # Bottom-left
Region 3: [0.25, 0.40, 0.70, 0.90]    # Lower-left
Region 4: [0.60, 0.75, 0.10, 0.30]    # Upper-right (enemy)
Region 5: [0.60, 0.75, 0.70, 0.90]    # Lower-right (enemy)
```

The environment randomly selects spawn positions within these boxes at reset time. This procedural generation ensures agents don't memorize exact starting positions.

### Variant Naming Convention

Variants follow the pattern `{Strength}_{Mode}`:

| Component | Options | Description |
|---|---|---|
| **Strength** | `V1`, `V2`, `V3` | Enemy count: 3, 5, or 8 enemies |
| **Mode** | `Base`, `Combat`, `Navigate` | Scenario objectives (see below) |

### Variant Details

| Variant Name | Allies | Enemies | Objective |
|---|---|---|---|
| `V1_Base` | 5 | 3 | Navigate to beacon + fight enemies en route (easiest) |
| `V1_Combat` | 5 | 3 | Kill all enemies, beacon irrelevant |
| `V1_Navigate` | 5 | 3 | Reach beacon, enemies are passive/ignored |
| `V2_Base` | 5 | 5 | Balanced scenario with navigation + combat |
| `V2_Combat` | 5 | 5 | Pure combat, no navigation goal |
| `V2_Navigate` | 5 | 5 | Navigate through passive enemies |
| `V3_Base` | 5 | 8 | Hard scenario, overwhelming numbers require strategy |
| `V3_Combat` | 5 | 8 | Hard combat, composition strategy essential |
| `V3_Navigate` | 5 | 8 | Navigate through heavy enemy presence |

### Bridge Strategy Implications

The two-bridge design creates natural choke points that agents must learn to exploit:

1. **Split force**: Sending some units through each bridge prevents the enemy from blocking all paths simultaneously
2. **Bridge timing**: Crossing when the enemy is engaged on the other side gives a numerical advantage
3. **Bridge denial**: Holding one bridge while attacking through the other forces the enemy to split their response
4. **Leash on bridges**: Units cluster at bridge exits after crossing, making them vulnerable to focused fire

The `enforce_bridge_terrain` function only constrains **ally** units (not enemies). Enemies can move through the cliff freely, simulating map asymmetry where allies must follow terrain while enemies have open movement (or this is a design choice for training difficulty).

---

## ⚖️ Rewards (`rewards.py`)

The `compute_rewards_and_dones` function (in `env.py`, not a separate file in the current structure) implements the multi-objective reward signal:

### Reward Components

```python
# 1. Navigation shaping (dense, high weight)
nav_reward = (prev_dist_to_beacon - new_dist_to_beacon) * 2.0
# Encourages moving toward the beacon every step

# 2. Combat shaping (sparse, moderate weight)
enemy_dmg_reward = (prev_hp - curr_hp) * 0.01 * 0.5
# Reward for dealing damage to enemies (scaled by health differential)

# 3. Friendly fire / being hit penalty
ally_dmg_penalty = -damage_taken_by_allies
# Penalizes losing HP (not just damage dealt, but also damage received)

# 4. Kill bonus (sparse, flat)
enemy_killed_reward = 0.2 per unit killed
# One-time bonus when an enemy's HP drops to zero

# Total reward = nav_reward + enemy_dmg_reward + ally_dmg_penalty + enemy_killed_reward
```

### Weight Rationale

| Component | Multiplier | Why this weight? |
|---|---|---|
| `nav_reward` | ×2.0 | Primary signal for Navigate/Base modes; needs to be strong enough to overcome exploration noise |
| `enemy_dmg` | ×0.5 × 0.01 | Scaled down because raw HP numbers (45-200) are large; 0.5 prevents combat from dominating navigation |
| `ally_dmg_penalty` | -1.0× | Direct penalty — no scaling needed since it's already a loss signal; keeps agents from suicidal behavior |
| `enemy_killed` | +0.2 flat | Sparse but substantial enough to signal "win" when combined with nav_reward |

### Done Conditions

An episode ends (and returns a done flag) when **any** of these conditions is met:

| Condition | Trigger | Rationale |
|---|---|---|
| `beacon_reached` | Any ally reaches beacon position | Success for Navigate/Base modes |
| `enemies_dead` | All enemies have HP ≤ 0 | Success for Combat/Base modes (zero-sum) |
| `allies_dead` | All allies have HP ≤ 0 | Failure — agent lost the engagement |
| `timestep >= 300` | Timeout | Prevents infinite episodes; ~15 seconds of simulation at 20 steps/sec equivalent |

---

## 🚧 Constraints (`enforce_bridge_terrain`)

### How Bridge Enforcement Works

The function `enforce_bridge_terrain` (implemented in the environment step loop, not a separate module) uses **segment-crossing detection** to prevent units from moving through the cliff wall:

```python
for each unit:
    prev_x = old_position.x
    new_x = new_position.x

    if (prev_x < 16 and new_x > 16) or (prev_x > 16 and new_x < 16):
        # Unit crossed the cliff boundary (x=16 in 32×32 grid)
        bridge_y = get_bridge_y_for_unit(unit.current_y)  # Which bridge is this unit near?
        if not in_bridge_range(unit.current_y, bridge_y):
            # Not within a bridge gap — reject the movement
            unit.position = clamp_to_bridge_side(prev_x, unit.current_y)
```

**Key implementation details:**
- The cliff is at grid column x=16 (normalized 0.5 × resolution).
- Bridge detection checks if the unit's y-coordinate is within ±2 grid units of either bridge center.
- If movement crosses the cliff AND isn't near a bridge, the position is "clamped" to prevent crossing.
- **Only ally units are constrained.** Enemy movement is not checked against bridges (this creates an asymmetry that increases training difficulty — enemies can reposition freely while allies must plan bridge crossings).

### Cliff Collision Penalty

In addition to blocking movement, units that are *on* cliff tiles (x=16 and not in bridge range) receive a small negative reward each step they remain there. This prevents agents from "backing up" against the cliff wall as a defensive strategy.

---

## 🛠 Authoring New Maps

### Step-by-Step Guide

#### 1. Create the Map Module

Create `JaxSC2/maps/newmap.py`:

```python
from JaxSC2.maps.twobridge import TwoBridgeMap

class NewMap(TwoBridgeMap):
    VARIANT_NAME = "NewMap"
    
    def __init__(self, resolution=32):
        super().__init__(resolution)
```

#### 2. Define Region Coordinates

Add a `REGION_COORDS` list with your spawn boxes:
```python
REGION_COORDS = [
    [0.1, 0.3, 0.2, 0.5],   # [xmin, xmax, ymin, ymax] in normalized coords
    ...
]
```

Each box should contain at least one valid spawn position. The environment will randomly pick positions within these boxes at reset time.

#### 3. Define the Cliff (if any)

```python
CLIFF_X = 0.5              # Normalized x position of cliff wall (None for no cliff)
BRIDGES = [                 # List of (y_min, y_max) for each bridge gap
    (0.3, 0.4),             # Bridge 1 at y=30%-40%
    (0.6, 0.7),             # Bridge 2 at y=60%-70%
]
```

Set `CLIFF_X = None` if you want an open map with no bridge constraints.

#### 4. Define the Beacon Position

```python
BEACON_POS = jnp.array([0.75, 0.5])   # [x, y] in normalized coords
```

This is the navigation target for Base and Navigate modes.

#### 5. Register Variants

Add your variant names to the environment's variant configuration in `JaxSC2/env/env.py`:

```python
ALL_VARIANTS = {
    ...existing variants...,
    "NewMap_Base": {"num_enemies": 5, "mode": "base"},
    "NewMap_Combat": {"num_enemies": 5, "mode": "combat"},
    # etc.
}
```

#### 6. Update Enforcer Logic (if needed)

If your map has different terrain features (e.g., multiple cliffs, circular bridges), modify the `enforce_bridge_terrain` function in `env.py`. The key contract is:

- Input: unit position, previous position
- Output: adjusted position (if movement is blocked) or original position

---

## 📁 File Index

| File | Purpose |
|---|---|
| `twobridge.py` | Default map: cliff + two bridges, 6 spawn regions, beacon position, variant definitions |
| `README.md` (this file) | You are here — map authoring and reward reference |
| `__init__.py` | Re-exports variant configuration to the environment |

