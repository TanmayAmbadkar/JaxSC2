# Contributing to Sc2Jax

Thank you for your interest in contributing! This guide covers everything needed to make a clean, reviewable contribution.

---

## Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Sc2Jax.git
cd Sc2Jax

# 2. Set up the conda environment (or use existing one)
conda create -n twobridge python=3.10 -y
conda activate twobridge

# 3. Install dependencies
pip install -r requirements.txt pytest

# 4. Configure paths
export PYTHONPATH=$PYTHONPATH:.

# 5. Run the smoke test suite
python algorithms/ppo/tests/test_ppo_jit.py
python algorithms/a2c/tests/test_a2c_jit.py

# 6. Run environment integrity tests
python JaxSC2/tests/test_env.py
python JaxSC2/tests/test_mechanics.py

# 7. Verify TensorBoard logging works
tensorboard --logdir runs/ --port 6007
```

---

## Coding Standards

### General Principles

- **Pure functions for physics.** Every function in `JaxSC2/env/mechanics.py` is pure (no side effects, input → output). This is non-negotiable — it enables JIT compilation.
- **No Python-side loops inside `@jax.jit` decorated functions.** Use `lax.scan`, `lax.map`, or `lax.fori_loop` for iteration. The JIT smoke tests will catch this.
- **No `as any`, `@ts-ignore`, or type suppression.** JAX is statically typed via shape inference. If you need flexibility, use `chex.assert_shape` or explicit casting.
- **No mutating state outside of dataclasses.** Use `flax.struct.dataclass` with `.replace()` for updates. Never mutate arrays in-place (`x[0] = 5` inside a JIT function will fail).

### Naming Conventions

| Context | Convention | Example |
|---|---|---|
| Variables & functions | `snake_case` | `apply_mass_collisions`, `num_enemies` |
| Classes & dataclasses | `PascalCase` | `SmaxState`, `MaskedActorCritic` |
| Constants & config keys | `UPPER_SNAKE_CASE` | `GAMMA`, `ROLLOUT_LEN`, `NUM_ENVS` |
| Private helpers (internal) | `_leading_underscore` | `_get_advantages` |

### File Layout

```
JaxSC2/env/
  __init__.py      # Re-export public API
  base.py          # SMAX, SmaxState, Space (no env logic)
  env.py           # JaxSC2Env class (only file importing from maps/)
  mechanics.py     # Pure physics functions (no env dependency)
  renderer.py      # Pygame rendering (optional, not needed for training)
  units.py         # Unit blueprints

algorithms/
  ppo/             # One algorithm per subdirectory
    __init__.py
    model.py       # Neural network architecture
    ppo_logic.py   # Loss function only
    ppo.py         # Trainer class
  common/          # Shared by all algorithms
```

Each algorithm subdirectory must contain at minimum: `model.py`, `_logic.py`, and `{algorithm}.py` (trainer).

### Type Hints & Docstrings

- **All public functions need type hints.** Internal helpers can skip if obvious.
- **Docstrings follow Google style** (not NumPy or Sphinx). At minimum: one-line summary, Args, Returns.
- **Physics functions MUST document their equations.** If a function modifies positions or health, the reader should be able to reproduce the math from the docstring alone.

Example:
```python
def apply_mass_collisions(prev_state: SmaxState) -> SmaxState:
    """Resolve pairwise collisions using mass-weighted repulsion.

    For each pair of collided units i, j:
        overlap = (r_i + r_j) - |pos_i - pos_j|
        if overlap > 0:
            r_i = mass_j / (mass_i + mass_j)
            r_j = mass_i / (mass_i + mass_j)
            pos_i += overlap * r_i * normal
            pos_j -= overlap * r_j * normal

    Args:
        prev_state: Current simulation state with positions and masses.

    Returns:
        Updated SmaxState with resolved positions (velocities unchanged).
    """
```

---

## Git Workflow & PR Process

### Branch Naming

Use descriptive, hyphenated names:
```
fix/health-clamp-minimum    # Bug fix
feat/new-unit-type          # New feature (unit)
docs/mechanics-docstrings   # Documentation update
refactor/gae-function       # Code restructuring
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
fix: clamp damage to minimum 0.5 instead of 0

The previous implementation allowed zero-damage when armor ≥ damage,
causing infinite combat loops. Adding a floor of 0.5 ensures units
always lose at least some health per hit.

Fixes #42
```

### Pull Request Template

When opening a PR, include:

**Description of changes** — What and why, not just what.
```
This adds a Tank unit type with higher HP/armor but slower speed.
The damage_matrix is extended to [3×3] with Tank advantage over Ranged.
```

**Testing performed** — What you ran and the output.
```bash
python JaxSC2/tests/test_env.py           # All pass
python algorithms/mask_ppo/tests/test_mask_ppo_jit.py  # JIT stable
python run_mask_ppo_v1.py --dry-run       # Trains 100 steps, reward > 0
```

**Screenshots (if applicable)** — For renderer/map changes, include before/after GIFs.

**Breaking changes** — If the PR changes any public API (function signatures, config keys, observation shape), call it out explicitly.

---

## Testing Expectations

### Minimum Tests for Any PR

| Type | Requirement |
|---|---|
| **Smoke test** | `env.reset()` and `env.step(action)` run without error in JIT mode |
| **Shape test** | Output shapes match expected dimensions (e.g., observation is still 63-dim) |
| **JIT test** | At least one `@jax.jit` wrapper around the core logic compiles without retracing errors |

### Where to Put Tests

- **Environment changes** → `JaxSC2/tests/`
- **Algorithm changes** → `algorithms/{algo}/tests/`
- **Shared utilities** → `algorithms/common/tests/` (create this if it doesn't exist)

### Test Structure

```python
"""Tests for mechanics.py collision resolution."""

import jax
import jax.numpy as jnp
import pytest
from JaxSC2.env.mechanics import apply_mass_collisions, apply_hard_collisions
from JaxSC2.env.base import SmaxState

def _make_dummy_state(num_units=4):
    """Helper: create a minimal valid SmaxState for testing."""
    pos = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32)
    vel = jnp.zeros_like(pos)
    return SmaxState(
        unit_positions=pos,
        unit_velocities=vel,
        unit_health=jnp.ones((num_units,), dtype=jnp.float32) * 100,
        unit_type=jnp.arange(num_units),
        # ... fill remaining fields with defaults ...
    )

def test_apply_mass_collisions_no_overlap():
    """When units don't overlap, positions should be unchanged."""
    state = _make_dummy_state()  # Well-separated points
    result = apply_mass_collisions(state)
    jnp.testing.assert_array_almost_equal(result.unit_positions, state.unit_positions)

def test_apply_mass_collisions_jit_stable():
    """Should compile under @jax.jit without retracing."""
    state = _make_dummy_state()

    @jax.jit
    def collide(s):
        return apply_mass_collisions(s)

    result = collide(state)  # First call compiles, second validates
    _ = collide(result)

def test_apply_hard_collisions_reduces_overlap():
    """When units are within 20% of radius sum, hard repulsion should push them apart."""
    # Create two units at distance 0.1 (well within collision threshold)
    # ... test that distance after > distance before ...
```

---

## Performance Bar

Since Sc2Jax is a high-throughput framework, PRs are expected to maintain or improve performance:

- **Benchmark before and after** if changing core loops. Use the SPS numbers from `README.md` as a baseline:
  - Random policy (32 envs): ~110k SPS
  - MaskPPO (32 envs): ~19k SPS
- **No regressions >5%** without explicit justification. If a change improves correctness but costs 20% speed, document the tradeoff and propose incremental optimization steps.
- **Profile with `jax.profiler`** if a PR touches the JIT compilation path and you suspect overhead:
  ```python
  import jax.profiler
  jax.profiler.start_trace("/tmp/jax_profile")
  # ... run training loop ...
  jax.profiler.stop_trace()
  ```

---

## Code Review Checklist (for reviewers)

- [ ] All new code is pure functions (no mutation, no side effects outside dataclasses)
- [ ] JIT smoke test passes (`@jax.jit` compiles without retracing errors)
- [ ] Type hints present on all public functions
- [ ] Docstrings follow Google style; physics equations documented in mechanics changes
- [ ] No `as any` / type suppression
- [ ] Config keys use `UPPER_SNAKE_CASE`; no magic numbers without constants or comments
- [ ] Changes to `damage_matrix`, `bonus_matrix`, or unit stats documented with design rationale
- [ ] New test(s) added; existing tests still pass

---

## Getting Help

If you're stuck:
1. Run the full test suite — if it passes, your environment is set up correctly
2. Check `ARCHITECTURE.md` for the system-level overview
3. Open an issue with a minimal reproducible example (or failing test)
