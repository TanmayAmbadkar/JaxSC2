# Algorithms Suite

The `algorithms/` directory contains high-performance RL implementations optimized for the JAX-native Sc2Jax environment.

## 📦 Available Algorithms

- **`ppo/`**: Standard Proximal Policy Optimization. Reliable and well-tested for both navigation and combat.
- **`mask_ppo/`**: Specialized PPO with binary logit masking. Used to enforce action constraints and prevent illegal actions during exploration.
- **`a2c/`**: Advantage Actor-Critic. Optimized for high parallelism and single-pass updates, achieving the highest SPS.

## 🛠 Shared Utilities (`common/`)

All algorithms share a set of core JAX utilities to ensure consistency:
- **`compute_gae`**: Generalized Advantage Estimation.
- **`RunningMeanStd`**: Online normalization for observations and rewards.
- **`checkpoint.py`**: Resilient state saving/loading with `opt_state` fallback.
- **`logging.py`**: Clean TensorBoard-compatible logging.

## 📈 JIT Optimization Pattern

All trainers follow the `jax.lax.scan` pattern for both rollouts and optimization epochs. This ensures that the entire training iteration is compiled into a single XLA computation, eliminating CPU-GPU/MPS synchronization bottlenecks.

## 🧩 Extending Algorithms

To implement a new algorithm (e.g., SAC, Q-learning):
1. Create a new subfolder in `algorithms/`.
2. Define a `logic.py` for the loss function.
3. Inherit from a `Trainer` pattern that uses `algorithms.common.utils` for rollout collection.
4. Verify JIT stability by ensuring no Python control flow is used inside the `train_iteration`.
