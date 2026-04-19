# Sc2Jax: Standalone High-Throughput MARL Framework

Sc2Jax is a lightweight, high-performance StarCraft II research framework built entirely in JAX. It provides a standalone ecosystem for training and benchmarking Reinforcement Learning agents with 100% JAX-native control flow, multi-agent vectorization, and physical verification.

## 🚀 Key Features

- **JAX-Native Physics**: All environment logic (collisions, combat, rewards) is implemented in JAX, enabling seamless `vmap` and `jit` transformations.
- **High Throughput**: Achieves >110k SPS for raw environment steps and >17k SPS during multi-epoch PPO training on Apple Silicon.
- **Tri-Algorithm Stack**: Production-ready implementations of **A2C**, **PPO**, and **MaskPPO** (with action-masking).
- **Standalone Architecture**: Zero dependency on heavy external MARL libraries; local `base.py` provides all necessary abstractions.
- **Physical Fidelity**: Includes continuous kinematics, projectile ballistics, and heterogeneous unit types (Melee, Ranged, Tank).

## 🛠 Installation

```bash
# Ensure JAX and Flax are installed
pip install jax jaxlib flax optax chex gymnasium
# Clone and set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
```

## 📊 Benchmarks (Apple Silicon)

| Scenario | Envs | Steps Per Second (SPS) |
|---|---|---|
| **Random Policy** | 32 | **110,320** |
| **A2C** | 64 | **24,541** |
| **MaskPPO** | 32 | **19,046** |
| **Standard PPO** | 32 | **17,057** |

## 📖 Library Structure

- `JaxSC2/`: Core environment engine.
  - `env/`: Physics, base classes, and `JaxSC2Env`.
  - `maps/`: Terrain constraints, rewards, and variants.
  - `tests/`: Mechanics and integrity test suites.
- `algorithms/`: Standardized RL implementations.
  - `ppo/`: Proximal Policy Optimization.
  - `mask_ppo/`: PPO with invalid-action masking.
  - `a2c/`: Advantage Actor-Critic.
  - `common/`: Shared utilities (GAE, RMS, Logging, Checkpointing).

## 🧪 Quick Start

Run the PPO smoke test to verify JIT stability:
```bash
python algorithms/ppo/tests/test_ppo_jit.py
```

Launch training:
```python
from algorithms.ppo.trainer import PPOTrainer
trainer = PPOTrainer({"VARIANT_NAME": "V1_Base"})
trainer.train(total_steps=1000000)
```

## 🤝 Contributing

See the individual `README.md` files in `JaxSC2/` and `algorithms/` for details on how to extend the environment or customize the RL logic.
