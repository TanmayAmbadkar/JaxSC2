# Maps, Rewards, and Constraints

This directory contains the logic governing environmental layouts, reward shaping, and physical constraints.

## 🗺 Layouts (`layouts.py`)

Layouts define the regional boundaries and unit spawn locations for different scenario "Variants".
- **`VARIANTS`**: A dictionary mapping variant names (e.g., `V1_Base`, `V2_Navigate`) to their configuration (number of units, layout type).
- **`REGION_COORDS`**: Defines normalized coordinate boxes `[xmin, xmax, ymin, ymax]` for procedural map generation.

## ⚖️ Rewards (`rewards.py`)

The `compute_rewards_and_dones` function implements the multi-objective reward signal:
- **Navigation Shaping**: Reward based on progress toward the beacon (2.0x weight).
- **Combat Shaping**: Sparse rewards for health differentials and unit kills (0.5x weight).
- **Collision Penalties**: Negative rewards for time spent in contact with cliffs or obstacles.

## 🚧 Constraints (`constraints.py`)

This module enforces physical boundaries that cannot be bypassed by units.
- **`enforce_bridge_terrain`**: Uses segment-crossing detection to prevent units from moving through the central cliff wall, except via the designated bridges.
- **Cliff Logic**: The cliff is positioned at `x = 0.5` (normalized) with gaps at specific `y` ranges.

## 🛠 Customizing Logic

To add a new reward component (e.g., ammunition conservation):
1. Add the necessary logic to `JaxSC2/maps/rewards.py`.
2. Update the `compute_rewards_and_dones` return signature.
3. Update `JaxSC2Env.step` to pass the new metrics to the `info` dictionary.
