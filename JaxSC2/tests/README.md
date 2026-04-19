# Test Suite — Complete Guide

The `tests/` directory contains validation scripts to ensure environment integrity and physics correctness. Algorithm test suites live in their respective directories (`algorithms/*/tests/`).

---

## 🧪 Running All Tests

```bash
# Set up path first (required for all tests)
export PYTHONPATH=$PYTHONPATH:.

# 1. Environment integrity test (reset/step across all variants)
python JaxSC2/tests/test_env.py

# 2. Physics mechanics test (collisions, projectiles, combat)
python JaxSC2/tests/test_mechanics.py

# 3. Algorithm JIT stability tests
python algorithms/ppo/tests/test_ppo_jit.py     # PPO training loop compiles and runs 1 step
python algorithms/a2c/tests/test_a2c_jit.py     # A2C training loop compiles and runs 1 step
python algorithms/mask_ppo/tests/test_mask_ppo_jit.py  # MaskPPO compiles with action masks
```

### Expected Output

All tests should print "All tests passed ✅" (or similar success message) with no errors. Each test includes:
- A JIT warm-up call (first compilation)
- A validation call (second execution, checks for retracing errors)
- Shape assertions on outputs

---

## 📋 Test Inventory & What Each Covers

### Environment Tests (`test_env.py`)

| Test | What It Validates |
|---|---|
| `test_reset_shapes` | Observation is 63-dim, state has correct unit count for each variant |
| `test_step_success` | Single step returns (obs, state, reward, done, info) without error — JIT compiled |
| `test_all_variants` | All 9 variants (V1/V2/V3 × Base/Combat/Navigate) reset and step correctly |
| `test_per_unit_actions` | PerUnitAction with different who_mask/verb/direction/target values step correctly |
| `test_action_masks` | Masked actions produce zero probability (invalid verbs/directions/targets can't be sampled) |
| `test_done_conditions` | Episodes end when beacon reached, all enemies dead, or all allies dead |

### Mechanics Tests (`test_mechanics.py`)

| Test | What It Validates |
|---|---|
| `test_mass_collisions_no_overlap` | Non-overlapping units have unchanged positions after collision check |
| `test_mass_collisions_overlap` | Overlapping units are pushed apart proportional to inverse mass ratio |
| `test_hard_collisions_threshold` | Hard repulsion activates only when distance < 20% of radius sum |
| `test_projectile_hit_detection` | Projectile within hit_radius (0.6) of target unit registers a hit |
| `test_projectile_lost_range` | Projectile beyond 40 units from spawn is removed (no crash) |
| `test_combat_windup_cooldown` | Unit cannot fire during windup; cooldown prevents immediate re-fire |
| `test_damage_formula` | Damage = clamp(dmg*mult+bonus-armor, min=0.5) produces correct values for all type pairs |
| `test_fog_updates` | Invisible enemies have zero observation features; visible enemies appear correctly |

### Algorithm JIT Tests (`algorithms/*/tests/test_*_jit.py`)

Each algorithm's JIT test follows the same pattern:
```python
# 1. Initialize model with dummy inputs
model = MyModel(...)
params = model.init(rng, dummy_obs)

# 2. Wrap the training iteration in @jax.jit
@jax.jit
def train_step(params, ...):
    return my_algorithm.train_iteration(...)

# 3. Warm-up compilation (first call) - may take several seconds
train_step(params, ...)

# 4. Validation run (second call) - must complete without retracing errors
result = train_step(params, ...)

# 5. Assert output shapes match expected dimensions
assert result[0].params["Dense_0"]["kernel"].shape == (63, 256)
```

These tests ensure:
- No Python-side loops inside JIT-compiled code (would cause retracing)
- All shapes are stable across iterations (no dynamic shapes causing OOM or errors)
- The training loop can compile and execute end-to-end

---

## 🏗 Adding New Tests

### Creating a New Test File

```bash
# For environment changes:
touch JaxSC2/tests/test_new_feature.py

# For algorithm changes:
touch algorithms/{algo}/tests/test_{algo}_new_feature.py
```

### Test Template

Every test should follow this pattern:

```python
"""Tests for <feature>."""

import jax
import jax.numpy as jnp
import pytest
from <package>.<module> import <function_to_test>


def _make_dummy_state(num_units=4):
    """Helper: create a minimal valid SmaxState for testing.

    Override this with your own fixtures if needed, but keep it minimal —
    only include fields that the test actually uses.
    """
    pos = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32)
    return SmaxState(
        unit_positions=pos,
        # ... fill required fields with safe defaults ...
    )


class TestApplyNewPhysics:
    """Tests for the apply_new_physics function."""

    def test_no_change_when_not_needed(self):
        """When conditions aren't met, output should equal input."""
        state = _make_dummy_state()
        result = apply_new_physics(state)
        jnp.testing.assert_array_almost_equal(
            result.unit_positions, state.unit_positions
        )

    def test_jit_stable(self):
        """Should compile under @jax.jit without retracing errors."""
        state = _make_dummy_state()

        @jax.jit
        def collide(s):
            return apply_new_physics(s)

        # Warm-up compilation
        _ = collide(state)
        # Validation — second call must not retrace
        result2 = collide(collide(state))

    def test_correct_values(self):
        """Specific numerical assertion for known inputs."""
        state = _make_dummy_state()
        result = apply_new_physics(state)
        # Assert specific values, not just shapes:
        assert result.unit_health[0] < state.unit_health[0]  # Some damage was dealt
```

### Test Conventions

1. **JIT stability is mandatory.** Every test involving a JAX function must include at least one JIT-compiled call. This catches retracing errors that wouldn't appear in eager mode.

2. **No random seeds in tests.** Use deterministic inputs (`jnp.array([[0.1, 0.2], ...])`). If randomness is unavoidable (e.g., testing the action sampling), fix the PRNG key:
   ```python
   rng = jax.random.PRNGKey(0)
   action = sample_action(rng, logits)
   ```

3. **Prefer `jnp.testing.assert_array_almost_equal`** over raw `assert x == y` for array comparisons. Floating-point drift from JIT compilation can cause exact equality failures even when the physics is correct.

4. **One test per function.** If a module has 5 functions, create at least 5 test methods (or separate test classes). This makes failures immediately traceable.

5. **Include a "smoke" test for the full loop.** At minimum, one end-to-end test that calls `env.reset()` → `env.step()` 10 times in JIT mode. This catches integration errors that unit tests might miss.

### Adding a Test for New Mechanics

When you add a new function to `mechanics.py`, create a corresponding test:

```python
# JaxSC2/tests/test_my_mechanic.py

import jax
from JaxSC2.env.mechanics import apply_my_mechanic

def test_apply_my_mechanic_basic():
    """Basic correctness check with known inputs."""
    state = _make_dummy_state()  # helper from conftest.py or inline
    result = apply_my_mechanic(state)
    # Assert: positions changed as expected, or unchanged if conditions not met

def test_apply_my_mechanic_jit():
    """Ensure JIT compilation succeeds."""
    @jax.jit
    def fn(state):
        return apply_my_mechanic(state)
    
    state = _make_dummy_state()
    _ = fn(state)   # warm-up
    _ = fn(fn(state))  # validation
```

---

## 🐛 Debugging Test Failures

### "XLA Error: retracing detected"

This means Python control flow leaked into a JIT-compiled function. Check:
1. No `if/else` blocks that depend on Python values (use `jnp.where`)
2. No `for` loops over data (use `lax.scan` or `lax.map`)
3. No `.item()` calls (converts array to Python scalar, breaks JIT)

### "Shape mismatch in dynamic_slice"

A tensor shape changed between warm-up and validation compilation. Check:
1. No variable batch sizes (all tests should use the same `NUM_ENVS`)
2. No dynamic number of enemies (use fixed unit counts in tests)

### "KeyError: 'vector'"

The observation format changed. This usually means `flatten_obs` or the env's step function signature was modified. Check that all tests use the same observation extraction method as the production code.

### "AssertionError: arrays not almost equal"

Floating-point drift from JIT compilation. Increase the tolerance or check if the physics parameters changed:
```python
# Default: rtol=1e-6, atol=1e-8
jnp.testing.assert_array_almost_equal(result, expected, rtol=1e-5)
```

---

## 📁 Test File Layout

```
JaxSC2/tests/
  test_env.py           # Environment reset/step across all variants
  test_mechanics.py     # Physics functions: collisions, projectiles, combat
  simulate_combat.py    # Non-test simulation script (demonstration, not pytest)

algorithms/ppo/tests/
  test_ppo_jit.py       # PPO JIT stability + shape validation

algorithms/mask_ppo/tests/
  test_mask_ppo_jit.py  # MaskPPO JIT stability with action masks

algorithms/a2c/tests/
  test_a2c_jit.py       # A2C JIT stability (no clip, MC GAE)

algorithms/common/tests/
  # [NOT YET CREATED — add here for new shared utility tests]
```

Add a `tests/` subdirectory to any new algorithm or common module you create.
