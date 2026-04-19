# Advantage Actor-Critic (A2C)

A high-throughput, synchronous A2C implementation.

## ✨ Highlights
- **Speed**: Achieves the highest SPS (~24.5k) by using a single-pass update per rollout.
- **Parallelism**: Efficiently scales to hundreds of parallel environments via `jax.vmap`.
- **Logic**: Implements a standard A2C loss with entropy regularization.

## ⚙️ Key Hyperparameters
- `GAE_LAMBDA`: Default `1.0` (monte-carlo style returns). Reduce if variance is too high.
- `NUM_ENVS`: Can be much larger than PPO to saturate device throughput.

## 🛠 Extension Guide
A2C uses the same `a2c_loss` function in `logic.py`. To implement n-step returns or other variance reduction techniques, modify the return calculation in `trainer.py`.
