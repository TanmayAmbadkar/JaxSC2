# Maps, Rewards, and Constraints

This directory contains the logic governing environmental layouts, reward shaping, and physical constraints. All terrain is defined on a normalized 0–1 grid that maps to the actual simulation resolution (default: 32×32).

---

## 🗺 Twobridge Map (`twobridge.py`)

### Layout Overview

The default map is a **32×32 grid** with a ditch spanning x=(0.45–0.55), or x=14.4–17.6 in world coordinates. Both armies must cross through one of two bridges. Collision geometry, spatial features, rendering, and scripted routing share the constants in `TwoBridgeMap`.

### Terrain Features

| Feature | Position (normalized) | Position (32×32 grid) | Effect |
|---|---|---|---|
| **Ditch** | x ∈ (0.45, 0.55) | x = 14.4–17.6 | Blocks both teams except at bridge gaps |
| **Bridge 1** | y ∈ (0.2, 0.3) | y = 6.4–9.6 | Unit centers need radius clearance from both edges |
| **Bridge 2** | y ∈ (0.7, 0.8) | y = 22.4–25.6 | Unit centers need radius clearance from both edges |
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

`TwoBridgeMap.enforce_constraints` constrains both teams, including position changes caused by unit collisions. Scripted agents stay inside a bridge corridor until their full footprint has cleared the opposite bank.

---

## ⚖️ Rewards

`JaxSC2Env.step` computes rewards using versioned weights from
[`env/rewards.py`](../env/rewards.py). The default profile is `balanced`;
`JaxSC2Env(..., reward_profile="legacy")` retains the previous training setup.

| Component | Balanced (default) | Legacy |
|---|---|---|
| Decrease in mean beacon distance | ×2.0 | ×2.0 |
| Decrease in mean distance to enemy centroid | ×2.0 | ×0.5 |
| Reach beacon (terminal) | +25 | +25 |
| Defeat all enemies (terminal) | +25 | +10 |
| Enemy HP removed | +0.005/HP | +0.005/HP |
| Allied HP lost | -0.005/HP | -0.005/HP |
| Enemy killed | +0.2 | +0.2 |
| All allies defeated (terminal) | -10 | -10 |
| Timeout (terminal) | -15 | -15 |

Distances are normalized by map dimensions and averaged over living allies.
Enemy-centroid approach is dense shaping, separate from damage and kill rewards.
Moving away produces negative shaping. Total reward is the sum of both approach
terms, damage/kill terms, and the applicable terminal bonus. Simultaneous beacon
and combat success receives one +25 bonus, not +50.

All variants use these same weights; Base/Combat/Navigate affect spawn layouts,
not task-specific victory conditions. Equal weights remove the explicit bonus
imbalance but do not remove differences in task difficulty, risk, or duration.

### Done Conditions

An episode ends (and returns a done flag) when **any** of these conditions is met:

| Condition | Trigger | Rationale |
|---|---|---|
| `beacon_reached` | Any living ally within normalized distance 0.05 | Navigation success in every variant |
| `enemies_dead` | All enemies have HP ≤ 0 | Combat success in every variant |
| `allies_dead` | All allies have HP ≤ 0 | Failure — agent lost the engagement |
| `timestep >= 300` | Timeout | Prevents infinite episodes; ~15 seconds of simulation at 20 steps/sec equivalent |

---

## 🚧 Constraints (`TwoBridgeMap.enforce_constraints`)

The ditch is the union of three rectangles between x=14.4 and 17.6, separated
by the two bridge openings. `DITCH_X` and `BRIDGE_Y` define this footprint for
physics, rendering and spatial feature layers.

For movement, each rectangle is expanded by the unit radius. This conservative
clearance also covers corners. A radius-0.6 unit can cross the lower bridge with
its center at y=7.0–9.0, and the upper bridge at y=23.0–25.0.

The solver sweeps the whole requested displacement against these expanded
rectangles, stops at first contact, removes inward displacement, and slides along
the edge. World bounds also include radius clearance. Both armies obey these
constraints even when unit-to-unit collision resolution is disabled.

The environment applies terrain constraints after resolving unit collisions,
then removes velocity components rejected by terrain. Scripted routing keeps its
bridge waypoint until the unit has fully exited the ditch, rather than turning
as soon as it crosses x=16.

`is_walkable` checks complete unit clearance. Scripted evaluations record
`terrain_valid` and `terrain_violations` separately from objective success.
There is no additional cliff-contact penalty in the reward function.

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
DITCH_X = (0.45, 0.55)     # Normalized near and far banks
BRIDGE_Y = (              # Exactly two (y_min, y_max) bridge openings
    (0.3, 0.4),             # Bridge 1 at y=30%-40%
    (0.6, 0.7),             # Bridge 2 at y=60%-70%
)
```

These class attributes are shared by terrain physics, features, rendering, and
routing. For an open map, provide a map implementation with no obstacle
rectangles and a corresponding constraint method.

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

If your map has different terrain features (e.g., multiple ditches or curved bridges), implement the corresponding geometry and `enforce_constraints` method on the map. The key contract is:

- Input: proposed positions, previous positions, map dimensions, teams, and radii
- Output: adjusted positions with valid swept paths and complete unit clearance

---

## 📁 File Index

| File | Purpose |
|---|---|
| `twobridge.py` | Default map: cliff + two bridges, 6 spawn regions, beacon position, variant definitions |
| `README.md` (this file) | You are here — map authoring and reward reference |
| `__init__.py` | Re-exports variant configuration to the environment |

