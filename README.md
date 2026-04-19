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

## 📖 Documentation

| Resource | Description |
|---|---|
| [Instructions.md](Instructions.md) | Setup, loading environments/agents, training, rendering, configuration |
| [Architecture](ARCHITECTURE.md) | Deep-dive into the environment engine and algorithm stack |
| [Contributing](CONTRIBUTING.md) | Coding standards, JIT rules, PR process, test expectations |
| [Mechanics](JaxSC2/env/mechanics.md) | Physics equations: collisions, velocity integration, combat, fog of war |
| [Unit Reference](JaxSC2/env/units.md) | Unit stats, weapons, type advantage matrix, design rationale |
| [Map Authoring](JaxSC2/maps/README.md) | Map structure, variants, bridge strategy, creating custom maps |
| [Test Suite](JaxSC2/tests/README.md) | Running tests, adding new tests, debugging JIT errors |
| [Common Utilities](algorithms/common/README.md) | GAE, Welford RMS, action encoding, checkpointing |
| [Visualizations](VISUALIZATIONS.md) | All demo scripts with CLI args, agent logic, customization guide |
| [Algorithm READMEs](algorithms/README.md) | PPO, MaskPPO, A2C overview |

## 📖 Library Structure

- `JaxSC2/`: Core environment engine.
  - `env/`: Physics, base classes, and `JaxSC2Env`. **See:** [mechanics.md](JaxSC2/env/mechanics.md), [units.md](JaxSC2/env/units.md)
  - `maps/`: Terrain constraints, rewards, and variants. **See:** [map authoring guide](JaxSC2/maps/README.md)
  - `tests/`: Mechanics and integrity test suites. **See:** [test suite guide](JaxSC2/tests/README.md)
- `algorithms/`: Standardized RL implementations. **See:** [common utilities](algorithms/common/README.md)
  - `ppo/`: Proximal Policy Optimization.
  - `mask_ppo/`: PPO with invalid-action masking.
  - `a2c/`: Advantage Actor-Critic.

## 🧪 Quick Start

Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
# or manually: pip install jax flax optax chex gymnasium pygame imageio matplotlib tensorboardX
```

Set up paths:
```bash
export PYTHONPATH=$PYTHONPATH:.
```

Run the PPO smoke test to verify JIT stability:
```bash
python algorithms/ppo/tests/test_ppo_jit.py
```

Launch training:
```python
from algorithms.mask_ppo import MaskPPO
from JaxSC2.env.env import JaxSC2Env

config = {"NUM_ENVS": 32, "ROLLOUT_LEN": 512, "UPDATE_EPOCHS": 10}
env = JaxSC2Env(variant_name="V1_Base")
model = MaskPPO(config=config)
model.train(env, total_steps=30_000_000)
```

## 🎬 Visualizations

Generate demo GIFs:
```bash
python JaxSC2/visualizations/render_demo.py        # Smart + chaos agent demos
python JaxSC2/visualizations/demo_suite.py          # Full variant suite with CLI filters
python JaxSC2/visualizations/run_ui.py              # Interactive Pygame window (press key to step)
```

See [VISUALIZATIONS.md](VISUALIZATIONS.md) for the full script reference.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, the PR process, and test expectations.
See [ARCHITECTURE.md](ARCHITECTURE.md) for a system-level overview before contributing.
