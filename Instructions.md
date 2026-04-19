# JaxSC2 — Instructions

## 1. Project Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or venv

### Install Dependencies

```bash
# Using the conda environment you already have
conda activate twobridge
pip install -r requirements.txt

# Or manually:
pip install jax flax optax chex gymnasium numpy pygame imageio matplotlib tensorboardX
```

### Path Configuration

The project uses two import paths. Set this in your shell or at the top of scripts:

```bash
export PYTHONPATH=$PYTHONPATH:.
```

Or add to your script:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

---

## 2. Loading Environments

### Basic Environment

```python
from JaxSC2.env.env import JaxSC2Env

env = JaxSC2Env(variant_name="V1_Base")
```

### Key Constructor Parameters (`JaxSC2Env.__init__`)

| Parameter | Default | Description |
|---|---|---|
| `variant_name` | `"V1_Base"` | Scenario variant (see §6) |
| `use_spatial_obs` | `False` | Enable 2D spatial observation maps (screen/minimats) instead of 63-dim vector |
| `resolution` | `32` | Map resolution for spatial observations |
| `enemy_ai` | `False` | Enable AI-controlled enemy units |
| `enemy_mode` | `"random"` | Enemy behavior mode: `"random"`, `"aggressive"`, etc. |

### Reset and Step

```python
import jax

# Reset returns (observation, state) — requires PRNG key for determinism
rng = jax.random.PRNGKey(42)
obs, env_state = env.reset(rng)

# Step returns (next_obs, next_state, reward, done, info_dict)
rng, step_rng = jax.random.split(rng)
next_obs, next_state, reward, done, info = env.step(step_rng, env_state, action)
```

### Observation Formats

**Vector observation (default):** 63-dim flat array:
- 20 ally features → 4 units × (rel_x, rel_y, health, cooldown, pad)
- 32 enemy features → 4 units × (rel_x, rel_y, health, cooldown, type, alive, pad×2)
- 8 action masks → verb_mask(3) + direction_mask(8), flattened
- 3 global features

**Spatial observation (`use_spatial_obs=True`):** Dict with `"screen"` and `"minimaps"` arrays for visual debugging.

### Action Formats

**PerUnitAction (native):**
```python
from JaxSC2.env.env import PerUnitAction

action = PerUnitAction(
    who_mask=jnp.ones((5,), dtype=jnp.bool_),  # Which allies act (N=allies)
    verb=jnp.array([1], dtype=jnp.int32),       # 0=NOOP, 1=MOVE, 2=ATTACK
    direction=jnp.array([3], dtype=jnp.int32),   # 0–7 (8 octal directions)
    target=jnp.array([1], dtype=jnp.int32),      # Enemy index (0–num_enemies-1)
)
```

**CentralAction:** Single action applied to all allies simultaneously:
```python
from JaxSC2.env.env import CentralAction

action = CentralAction(
    who_mask=ally_alive, verb=1, direction=3, target=0
)
```

---

## 3. Loading Agents / Algorithms

### MaskPPO (Recommended — handles action masking natively)

```python
from algorithms.mask_ppo import MaskPPO

config = {
    "NUM_ENVS": 32,        # Parallel environments via vmap
    "ROLLOUT_LEN": 512,     # Steps per rollout before update
    "UPDATE_EPOCHS": 10,    # Epochs per update
    "NUM_MINIBATCHES": 16,  # Minibatches per epoch
    "LR": 3e-4,             # Learning rate (Adam)
    "CLIP_EPS": 0.2,        # PPO clip epsilon
    "GAMMA": 0.995,         # Discount factor
    "GAE_LAMBDA": 0.95,     # GAE lambda (0=TD, 1=MC)
    "ENTROPY_COEFF": 0.01,  # Entropy bonus for exploration
    "VF_COEFF": 0.5,        # Value function loss weight
}

model = MaskPPO(config=config)
```

### Standard PPO (no action masking — all actions always valid)

```python
from algorithms.ppo.trainer import PPOTrainer

trainer = PPOTrainer(config=config)
```

### A2C (no clipping, Monte Carlo returns)

```python
from algorithms.a2c.trainer import A2CTrainer

trainer = A2CTrainer(config={
    "VARIANT_NAME": "V1_Base",  # A2C uses config dict instead of env constructor arg
    "NUM_ENVS": 64,
    "ROLLOUT_LEN": 128,         # Shorter rollouts for A2C
    "LR": 7e-4,                 # Higher LR (no clipping)
    "GAMMA": 0.99,
    "GAE_LAMBDA": 1.0,          # Monte Carlo returns (no bootstrapping)
})
```

### Model Architecture

All algorithms use the same multi-head structure:

```
Input (63-dim) → LayerNorm → Dense(256) → ReLU → Dense(256) → ReLU
  ├─ Value head: Dense(1)          → value scalar (for GAE)
  ├─ Verb head:   Dense(3)         → NOOP/MOVE/ATTACK logits
  ├─ Dir head:    Dense(8)         → 8-direction logits
  └─ Target head: Dense(num_enemies) → enemy selection logits

MaskPPO adds: -1e9 fill for invalid actions before softmax
```

### Loading Saved Checkpoints

```python
from algorithms.common.checkpoint import load_checkpoint

# Reconstruct model first (same architecture as during training)
rng = jax.random.PRNGKey(42)
dummy_obs = jnp.zeros((1, 63))
model = MaskedActorCritic(action_dim=num_enemies)
params = model.init(rng, dummy_obs)

tx = optax.adam(config["LR"])
train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Load
train_state, step = load_checkpoint("runs/mask_ppo/checkpoints/ckpt_100000.pkl", train_state)
print(f"Loaded checkpoint from step {step}")
```

---

## 4. Training & Evaluation

### Full Training Loop (MaskPPO)

```python
from algorithms.mask_ppo import MaskPPO
from JaxSC2.env.env import JaxSC2Env

config = {
    "NUM_ENVS": 32, "ROLLOUT_LEN": 512, "UPDATE_EPOCHS": 10,
    "NUM_MINIBATCHES": 16, "LR": 3e-4, "CLIP_EPS": 0.2,
    "GAMMA": 0.995, "ENTROPY_COEFF": 0.01, "VF_COEFF": 0.5,
    "LOG_INTERVAL": 10, "CKPT_INTERVAL": 200,
}

env = JaxSC2Env(variant_name="V1_Base")
model = MaskPPO(config=config)

# Train for 30M steps (~2 hours on Apple Silicon M-series)
model.train(env, total_steps=30_000_000)
```

### Quick Smoke Test (100k steps)

```bash
python algorithms/ppo/tests/test_ppo_jit.py
```

### Evaluation (Post-Training)

```python
from algorithms.ppo.eval import evaluate

# Returns metrics dict: mean_reward, nav_win_rate, combat_win_rate, total_win_rate
eval_metrics = evaluate(
    env=env,                    # Environment instance
    params=train_state.params,  # Loaded model parameters
    model=model,                # Model instance (for apply_fn)
    rng=jax.random.PRNGKey(42), # Evaluation PRNG key
    num_episodes=32,            # Parallel episodes via vmap
    max_steps=400,              # Max steps per episode during eval
)

print(eval_metrics)
# {
#   "eval/mean_reward": 0.42,
#   "eval/nav_win_rate": 0.65,      # % episodes reaching beacon
#   "eval/combat_win_rate": 0.35,   # % episodes killing all enemies
#   "eval/total_win_rate": 0.75,    # Either outcome wins
# }
```

### Viewing Training Logs

TensorBoard logs are written to `runs/mask_ppo/logs/` (MaskPPO) or `runs/{variant}_a2c/logs/` (A2C):

```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```

---

## 5. Rendering & Visualization

### Pygame Renderer (Live)

The `ProductionRenderer` uses Pygame for real-time rendering:

```python
from JaxSC2.env.renderer import ProductionRenderer, state_to_frame
from JaxSC2.env.env import JaxSC2Env
import jax

env = JaxSC2Env(variant_name="V1_Base")
rng = jax.random.PRNGKey(42)
obs, state = env.reset(rng)

renderer = ProductionRenderer(headless=False, trails_enabled=True)

# Render a single episode to GIF
trajectory = [state_to_frame(state)]
for _ in range(300):
    rng, step_rng = jax.random.split(rng)
    # ... apply actions ...
    obs, state, _, done, _ = env.step(step_rng, state, action)
    trajectory.append(state_to_frame(state))
    if done: break

renderer.render_episode(trajectory, save_path="episode.gif", interp_steps=4)
```

### Pre-built Demo Scripts

| Script | What It Does |
|---|---|
| `run_mask_ppo_v1.py` | MaskPPO training on V1_Base (5v3) |
| `run_mask_ppo_v3.py` | MaskPPO training on V3_Base (5v8) |
| `run_mask_ppo_gamma099.py` | MaskPPO with γ=0.99 |
| `run_mask_ppo_gamma0995.py` | MaskPPO with γ=0.995 (default) |
| `JaxSC2/visualizations/render_demo.py` | Smart + chaos agent demo with spatial observation plots |
| `JaxSC2/visualizations/combat_showcase.py` | Combat scenario visualization |
| `JaxSC2/visualizations/navigation_showcase.py` | Navigation scenario visualization |
| `JaxSC2/visualizations/full_demo.py` | Combined combat + navigation demo suite |
| `JaxSC2/visualizations/demo_suite.py` | Full interactive demo runner |
| `JaxSC2/visualizations/run_ui.py` | Interactive Pygame UI (no training) |

Run a demo:
```bash
python JaxSC2/visualizations/render_demo.py
# Outputs: combat_nav.gif, chaos_nav.gif, spatial_obs.gif
```

---

## 6. Configuration & Variants

### Variant Naming Convention `{N}v{M}_{Mode}`

| Component | Options |
|---|---|
| **Strength** | `V1` = 5v3 · `V2` = 5v5 · `V3` = 5v8 |
| **Mode** | `Base` (navigate + fight) · `Combat` (pure combat) · `Navigate` (pathfinding only) |

### Available Variants (9 total)

```
V1_Base, V1_Combat, V1_Navigate    # 5 allies vs 3 enemies
V2_Base, V2_Combat, V2_Navigate    # 5 allies vs 5 enemies
V3_Base, V3_Combat, V3_Navigate    # 5 allies vs 8 enemies
```

### Usage

```python
# Via variant_name (MaskPPO / PPO)
env = JaxSC2Env(variant_name="V3_Combat")

# Via config (A2C trainer)
trainer = A2CTrainer(config={"VARIANT_NAME": "V3_Navigate"})

# Via launch script argument (scripts accept variant_name internally)
python run_mask_ppo_v3.py  # Hardcoded to V1_Base; modify variant_name in script
```

### Map Structure (Twobridge)

The default map is a 32×32 grid with:
- **Cliff wall** at x=0.5 splits left and right halves (only allies are terrain-constrained)
- **Two bridges**: one at y≈6.4 (20%), one at y≈20.5 (64%)
- **6 spawn regions** for varied initial positions
- **Beacon position** on the right side (navigation target)

---

## 7. Custom Environments & Maps

### Adding a New Unit Type

Edit `JaxSC2/env/units.py`:

```python
from JaxSC2.env.units import WeaponDef, UnitDefinition

# 1. Define weapon
class SniperWeapon(WeaponDef):
    def __init__(self):
        super().__init__(damage=12, range_=10.0, windup=5, cooldown=15)

# 2. Define unit
def create_sniper():
    return UnitDefinition(
        name="Sniper", index=3,  # Next available index
        hp=30, speed=0.2, accel=0.06, mass=0.8, armor=0.5,
        weapon=SniperWeapon(),
    )

# 3. Register in unit registry (end of units.py)
UNIT_REGISTRY[3] = create_sniper
```

### Modifying Type Advantages

Edit `JaxSC2/env/env.py` — look for the `damage_matrix` and `bonus_matrix`:

```python
# Matrix shape: [row_type] × [col_type], row attacks col
damage_matrix = jnp.array([
    [1.0, 1.5, 0.7],   # Melee → [Melee, Ranged, Tank]
    [0.8, 1.0, 1.5],   # Ranged → ...
    [1.5, 0.7, 1.0],   # Tank → ...
    [?, ?, ?],         # Sniper (new row)
])  # and add a column for [?, ?, ?, ?] at end of each existing row
```

### Creating a New Map

Create `maps/newmap.py`:

```python
from JaxSC2.maps.twobridge import TwoBridgeMap

class NewMap(TwoBridgeMap):
    VARIANT_NAME = "NewMap"
    
    def __init__(self, resolution=32):
        super().__init__(resolution)
        # Override: terrain mask, bridges, spawn coords, beacon position
```

Then register in the variant system (`JaxSC2/env/env.py`) and reference via `variant_name="NewMap_Combat"`.

### Changing Reward Formulation

In `JaxSC2/env/env.py`, modify the reward computation step in the main loop:

```python
# Current formula (env.py):
nav_reward = (prev_dist - new_dist) * 2.0
enemy_dmg_reward = (prev_hp - curr_hp) * 0.01 * 0.5
ally_dmg_penalty = -damage_to_allies
enemy_killed_reward = 0.2

# Modify any component, e.g.:
reward = nav_reward * 1.5 + enemy_dmg_reward - ally_dmg_penalty * 0.3
```

---

## 8. Debugging, Testing & Multi-Environment Training

### Test Suite

Run the JIT compilation smoke test:
```bash
python algorithms/ppo/tests/test_ppo_jit.py    # PPO training stability
python algorithms/a2c/tests/test_a2c_jit.py    # A2C training stability
```

### Debugging JIT Errors

JAX JIT compilation errors can be cryptic. Use these techniques:

```python
# 1. Disable JIT to get Python tracebacks (slow but debuggable)
import jax
jax.config.update("jax_disable_jit", True)

# 2. Use jnp.printing to inspect intermediate values (works with JIT in newer JAX)
import jax.numpy as jnp
jnp.printing.describe_array(intermediates)

# 3. Compile in advance with random inputs to catch shape errors
env = JaxSC2Env(variant_name="V1_Base")
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng)

# Warm-up compilation (this will reveal JIT errors immediately)
dummy_model = MaskedActorCritic(action_dim=3)
dummy_params = dummy_model.init(rng, jnp.zeros((1, 63)), 
    jnp.ones((1,3), dtype=bool), jnp.ones((1,8), dtype=bool), 
    jnp.ones((1,3), dtype=bool))
```

### Multi-Environment Training (vmap)

Both MaskPPO and A2C are designed for massive parallelism via `jax.vmap`:

```python
# The NUM_ENVS config parameter controls this automatically:
config = {"NUM_ENVS": 64, "ROLLOUT_LEN": 512}
# → 64 parallel environments vmap'd, producing 32,768 transitions per rollout

# Benchmarks on Apple Silicon:
# - Random policy (32 envs):    ~110,000 steps/sec
# - A2C (64 envs):              ~24,500 steps/sec
# - MaskPPO (32 envs):          ~19,000 steps/sec
# - Standard PPO (32 envs):     ~17,000 steps/sec

# For GPU/TPU: ensure JAX is installed with CUDA/Metal backend
pip install "jax[cuda12]"  # or jax[tpu] / pip install mlx
```

### Monitoring During Training

Training prints progress every `LOG_INTERVAL` iterations:

```
Iter   10 | Step    163840 | Reward     0.42 | SPS  19,046
Iter   20 | Step    327680  | Reward     1.18 | SPS  19,234
...
>>> EVAL: Reward 2.35 | Win (Nav: 70.3%, Combat: 41.2%, Total: 81.3%)
```

Watch live via TensorBoard:
```bash
tensorboard --logdir runs/ --port 6006
```

### Checkpoint Management

Checkpoints are saved as pickle files: `runs/{algo}/checkpoints/ckpt_{step}.pkl`

```python
# To resume training from a checkpoint, load it and continue:
from algorithms.common.checkpoint import load_checkpoint

# Reconstruct the same model architecture used during training
model = MaskedActorCritic(action_dim=3)  # action_dim must match
params = model.init(rng, dummy_obs)
tx = optax.adam(3e-4)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Load
state, loaded_step = load_checkpoint("runs/mask_ppo/checkpoints/ckpt_500000.pkl", state)

# Continue training from step 500,000
model.state = state
model.train(env, total_steps=1_000_000)  # trains from step 500k to 1.5M
```
