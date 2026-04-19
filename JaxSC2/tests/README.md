# Test Suite Certification

The `tests/` directory contains critical validation scripts to ensure environment integrity and JIT stability.

## 🧪 Running Tests

Ensure your `PYTHONPATH` is set correctly:
```bash
export PYTHONPATH=$PYTHONPATH:.
```

### 1. Environment Integrity
Validates that the environment resets and steps without errors across different variants.
```bash
python JaxSC2/tests/test_env.py
```

### 2. Physical Mechanics
Certifies collision detection, weapon ballistics, and combat logic.
```bash
python JaxSC2/tests/test_mechanics.py
```

### 3. Algorithm JIT Stability
These tests ensure that the training loops do not trigger Python side-effects or retraces during execution.
```bash
python algorithms/ppo/tests/test_ppo_jit.py
python algorithms/a2c/tests/test_a2c_jit.py
python algorithms/mask_ppo/tests/test_mask_ppo_jit.py
```

## 🏗 Adding New Tests
When adding new features (e.g., new obs types):
1. Create a smoke test that calls `env.reset` and `env.step` with the new feature enabled.
2. Wrap the loop in `jax.jit` to confirm it is compiled successfully.
3. Assert that the output shapes match the expected feature dimensions.
