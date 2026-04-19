# Proximal Policy Optimization (PPO)

A standard, high-throughput PPO implementation for Sc2Jax.

## 🗝 Highlights
- **Architecture**: Feed-forward Categorical Actor-Critic.
- **Optimization**: Multi-epoch mini-batch updates with CLIP loss.
- **Normalization**: Online RunningMeanStd for both observations and value returns.

## ⚙️ Key Hyperparameters
- `LEARN_RATE`: Default `3e-4`.
- `ENTROPY_COEFF`: Default `0.01` (crucial for exploration in combat).
- `CLIP_EPS`: Default `0.2`.
- `NUM_MINIBATCHES`: Controls the gradient update granularity.

## 🛠 Extension Guide
To modify the network architecture, edit `algorithms/ppo/model.py`. The `ActorCritic` class uses Flax Linen; ensure any new layers are JAX-compatible.
