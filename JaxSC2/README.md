# JaxSC2 Environment Engine

`JaxSC2` is the core simulation engine for the Sc2Jax framework. It provides a highly optimized, physically verified StarCraft II simulation environment built entirely in JAX.

## 🏛 Architecture

- **`JaxSC2Env`**: The primary environment class. It handles high-level transitions, centralized action decoding, and observation assembly.
- **`mechanics.py`**: A low-level physics engine implementing continuous movement integration, hard-body collisions, and high-fidelity combat (projectiles/armor).
- **`base.py`**: Contains the `SmaxState` schema, `MultiAgentEnv` base classes, and JIT-friendly `Space` definitions.

## 🛠 Extending the Environment

### 1. Adding New Mechanics
To add new physics (e.g., elevation, secondary weapon systems), modify `JaxSC2/env/mechanics.py`. Ensure all functions are pure JAX (`@jax.jit` compatible) and use `lax.cond` or `jnp.where` for branching.

### 2. Modifying Unit Types
Unit characteristics (HP, Speed, Range) are defined in `JaxSC2Env.__init__`. You can modify `self.unit_type_hp`, `self.unit_type_speed`, etc., to create new agent roles.

### 3. Creating New Varianats
Add new layout configurations in `JaxSC2/maps/layouts.py` by defining a new dictionary in `VARIANTS`.

## 📡 Observation Specs

The default observation is a **63-dimensional vector**:
- **Ally Features (20-dim)**: Relative X/Y, Health, Cooldown for 5 units.
- **Enemy Features (32-dim)**: Relative X/Y, Health, Cooldown for up to 8 units.
- **Enemy Mask (8-dim)**: Binary mask for unit existence.
- **Global Data (3-dim)**: Beacon relative position and normalized timestep.

## 🎮 Action Specs

Actions are passed via the `CentralAction` dataclass:
- `who_mask`: (num_allies,) binary mask for selection.
- `verb`: 0 (No-op), 1 (Move), 2 (Attack).
- `direction`: 0-7 (Octal directions).
- `target`: 0-7 (Index of target enemy).
