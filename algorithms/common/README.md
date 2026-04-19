# Common Utilities — Shared Algorithms Code

The `algorithms/common/` directory contains utilities shared across all three algorithms (PPO, MaskPPO, A2C). Every new algorithm should import from here rather than reimplementing these functions.

---

## Running Mean Squared Deviation (`RunningMeanStd`)

**File:** `utils.py` (lines 5-29)  
**Algorithm:** Used by all algorithms for reward normalization.

### What It Does

Maintains a running estimate of mean and variance using **Welford's online algorithm**. This is numerically stable for arbitrary batch sizes and doesn't require storing all past observations.

### Interface

```python
from algorithms.common.utils import RunningMeanStd, update_rms

# Initial state (small epsilon on var to avoid division by zero)
rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

# Update with a batch of rewards (shape: (batch_size,))
rms = update_rms(rms, rewards)

# Access running statistics
mean = rms.mean   # Shape: (1,) — scalar wrapped in array
var  = rms.var    # Running variance estimate
```

### Welford's Algorithm (Batch Mode)

The implementation computes `update_rms` in a single pass using the parallel Welford formula:

```python
# Given running stats (mean_r, var_r, count_r) and batch statistics:
delta = batch_mean - running_mean
new_mean = running_mean + delta * batch_count / (running_count + batch_count)

M2_running = var_r * running_count
M2_batch   = sum((x - batch_mean)^2) for x in batch

new_var = (M2_running + M2_batch + delta^2 * running_count * batch_count / total_count) / total_count
```

**Why Welford over naive?** The naive formula `mean(x^2) - mean(x)^2` suffers from catastrophic cancellation when values are large. Welford's incremental approach is numerically stable.

**Why batch mode over single-element?** Processing entire reward batches at once is faster under JIT (one `lax.scan` call vs many). Single-element updates would require unrolling the scan.

### Usage Pattern in Training Loops

```python
# Per-iteration update (in train_iteration):
rms = update_rms(rms, rew_b.flatten())          # Update with current batch rewards
norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)   # Normalize rewards by running stddev
```

The normalized rewards are then used as inputs to `compute_gae`. This prevents reward scale from affecting the PPO clip ratio or GAE advantage magnitude.

---

## Generalized Advantage Estimation (`compute_gae`)

**File:** `utils.py` (lines 31-48)  
**Algorithm:** Used by PPO and MaskPPO (GAE_LAMBDA < 1.0) and A2C (GAE_LAMBDA = 1.0).

### What It Does

Computes Generalized Advantage Estimation (Schulman et al., 2016) — a low-variance, low-bias estimator of action advantages that interpolates between TD(0) and Monte Carlo returns.

### Interface

```python
from algorithms.common.utils import compute_gae

advantages, returns = compute_gae(
    rewards=norm_rew_b,       # (rollout_len, num_envs) — already normalized
    values=val_b,             # (rollout_len, num_envs) — value predictions from policy
    dones=done_b,             # (rollout_len, num_envs) — episode termination flags
    gamma=GAMMA,              # Discount factor (typically 0.995)
    lam=GAE_LAMBDA,           # Mix factor: 0=TD(0), 1=Monte Carlo (A2C)
    last_val=last_value,      # (num_envs,) — value of final state for bootstrapping
)

# Returns:
# advantages: (rollout_len, num_envs) — estimated advantage for each step
# returns:    (rollout_len, num_envs) — A_t + V(s_t), used as target for value loss
```

### GAE Equation

```python
delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
A^GAE_t = delta_{t-1} + gamma * lambda * (1 - d_{t-2}) * A^GAE_{t-2} + ...
```

Where:
- `delta_t` is the TD error at step t
- `d_t` is the done flag (1.0 if episode ended, 0.0 otherwise)
- `gamma * lambda` controls the bias-variance tradeoff:
  - **GAE_LAMBDA=0 (PPO default for some configs):** Pure TD(0) — low variance, high bias
  - **GAE_LAMBDA=0.95 (default PPO):** Balanced — used by both PPO and MaskPPO
  - **GAE_LAMBDA=1.0 (A2C):** Pure Monte Carlo — zero bias, higher variance

### JIT Implementation

Implemented via `jax.lax.scan` with `reverse=True` to iterate backward through the rollout:

```python
def _get_advantages(gae_and_next_value, transition):
    gae, next_value = gae_and_next_value
    r, v, d = transition  # reward, value prediction, done flag
    
    delta = r + gamma * next_value * (1.0 - d) - v
    gae = delta + gamma * lam * (1.0 - d) * gae
    
    return (gae, v), gae  # carry forward next_value for previous step

_, advantages = jax.lax.scan(
    _get_advantages,
    (jnp.zeros_like(last_val), last_val),  # initial carry: gae=0, next_value=last_val
    (rewards, values, dones),               # sequence to scan over
    reverse=True                            # backward iteration
)
```

**Key insight:** The carry state `(gae, next_value)` is updated in each iteration. When scanning backward, `next_value` becomes the value of the *previous* step in the rollout (which is the "next" state from that step's perspective). This elegantly computes the recursive GAE formula in a single JIT-compiled pass.

---

## Observation Flattening (`flatten_obs`)

**File:** `utils.py` (lines 50-61)  
**Algorithm:** Used by all algorithms to convert environment observations to model input.

### What It Does

Converts the observation from any format (dict or flat array) into a consistent `(batch, features)` shape. Handles both single observations `(D,)` and batched observations `(B, D)`.

### Interface

```python
from algorithms.common.utils import flatten_obs

# From vector observation (default env mode)
obs = {"vector": jnp.zeros((32, 63))}      # batch of 32
flat = flatten_obs(obs)                      # shape: (32, 63)

# From raw array
obs = jnp.zeros((1, 63))                    # single observation
flat = flatten_obs(obs)                      # shape: (1, 63) — no change for unbatched

# From spatial observation
obs = {"screen": ..., "minimats": ...}       # dict with 2D arrays
flat = flatten_obs(obs)                      # shape: (32, -1) — flattened all features
```

### Implementation Notes

```python
def flatten_obs(obs):
    if not isinstance(obs, jnp.ndarray):       # Dict → extract vector key
        obs = obs["vector"]
    
    if obs.ndim == 1:                          # Single observation: (D,) → return as-is
        return obs
    
    return obs.reshape(obs.shape[0], -1)       # Batched: (B, D) → (B, D)
```

The function is deliberately simple — no complex reshaping. The 63-dim vector observation already has the right shape; `flatten_obs` is mainly a safety wrapper for spatial obs or dict cases.

---

## Action Encoding / Decoding

**File:** `utils.py` (lines 63-102)  
**Algorithm:** Used for logging, evaluation, and Gym compatibility.

### Legacy Flat Action Encoding (`decode_action` / `encode_per_unit_action`)

```python
# OLD: Single flat index into 3×8×num_targets space
flat_idx = encode_per_unit_action(verb, direction, target)
# Returns: verb + 3 * direction + 3 * 8 * target_packed

verb, direction, target = decode_per_unit_action(flat_idx, num_targets)
# Returns: (verb % 3), ((flat // 3) % 8), flat // 24 - (target==0)
```

### Legacy Central Action Decoding (`decode_action`) — DEPRECATED

```python
# Maps flat 17-dim action index to (Verb, Direction, Target)
# 0 → Stop, 1-8 → Move, 9-16 → Attack
verb, direction, target = decode_action(action_idx)
```

**Status:** Deprecated in favor of `PerUnitAction`. Kept for backward compatibility with evaluation code and Gym wrapper.

### Per-Unit Action Encoding (Current)

```python
# Pack per-unit action arrays into flat index for storage/logging
flat_idx = encode_per_unit_action(verb, direction, target)

# Unpack back into components
verb, direction, target = decode_per_unit_action(flat_idx, num_targets)
```

---

## Logging (`Logger`)

**File:** `logging.py` (lines 1-23)  
**Algorithm:** Used by all algorithms for TensorBoard metric logging.

### Interface

```python
from algorithms.common.logging import Logger

logger = Logger("runs/mask_ppo/logs")   # Creates directory + TensorBoard writer

# Log scalar metrics at a given step
logger.log(step=1000, metrics={
    "train/mean_reward": 0.42,
    "train/sps": 19046.0,
    "train/value_loss": 0.15,
})

logger.close()   # Flush and close writer (called automatically at end of train())
```

### Implementation Details

Uses `tensorboardX.SummaryWriter` (not the native TensorBoard Python API). This works without requiring a separate `tensorboard` binary installation — it writes Protocol Buffer events directly to disk, which TensorBoard can then read.

```python
class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log(self, step, metrics: dict):
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(k, float(v), step)
            except (TypeError, ValueError):
                continue  # Skip non-numeric values silently

    def close(self):
        self.writer.close()
```

**Note:** Metrics with `train/`, `eval/`, or other prefixes are organized into tabs in TensorBoard. The slash is a TensorBoard convention for grouping related metrics.

---

## Checkpointing (`save_checkpoint` / `load_checkpoint`)

**File:** `checkpoint.py` (lines 1-39)  
**Algorithm:** Used by all algorithms for saving/loading trained models.

### Interface

```python
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint
from flax.training.train_state import TrainState

# Save
save_checkpoint("runs/mask_ppo/checkpoints/ckpt_500000.pkl", state, step=500000)

# Load
loaded_state, loaded_step = load_checkpoint(
    "runs/mask_ppo/checkpoints/ckpt_500000.pkl",
    train_state  # The TrainState to load into (must have same architecture)
)
```

### Resilience Design

The `load_checkpoint` function handles two failure modes:

1. **Missing file:** Returns the input state unchanged with step=0 (no crash).
2. **Opt_state restore failure:** If the optimizer state is corrupted or from a different architecture, it loads only `params` and returns the unmodified opt_state. This allows resuming training even if the optimizer state can't be restored (you'll get a brief warmup period as Adam's momentum resets).

```python
def load_checkpoint(path, state: TrainState):
    if not os.path.exists(path):
        return state, 0
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    try:
        new_state = state.replace(params=data["params"], opt_state=data["opt_state"])
    except Exception as e:
        print(f"Warning: opt_state restore failed ({e}), loading params only.")
        new_state = state.replace(params=data["params"])
    
    return new_state, data["step"]
```

### Checkpoint Format

```python
{
    "params": PyTree[Array],         # JAX parameter tree (flax-compatible)
    "opt_state": PyTree[Array],      # Optimizer state (Adam's mu/v for optax)
    "step": int,                     # Global training step
}
```

File size: Typically 10-50 MB depending on model architecture. For MaskPPO with two Dense(256) layers + 4 heads, expect ~3 MB.

---

## Base Algorithm (`BaseAlgorithm`)

**File:** `base.py` (lines 1-31)  
**Algorithm:** Abstract base class for all algorithm classes.

### Interface

```python
from algorithms.common.base import BaseAlgorithm

class MaskPPO(BaseAlgorithm):
    def __init__(self, config=None):
        default_config = { "NUM_ENVS": 32, ... }
        if config:
            default_config.update(config)
        super().__init__(default_config)  # Stores self.config

    def train(self, env, total_steps: int):
        ...  # concrete implementation
```

### Purpose

Provides a uniform interface across algorithms:
- `self.config` — the initialized configuration dictionary (from `__init__`)
- Extensible hook methods for new algorithms to override

**Note:** The base class is intentionally minimal. It doesn't enforce any specific training loop structure — each algorithm defines its own `train()` method signature.
