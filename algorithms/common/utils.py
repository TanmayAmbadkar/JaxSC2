import jax
import jax.numpy as jnp
from typing import NamedTuple

class RunningMeanStd(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float

def update_rms(rms: RunningMeanStd, x: jnp.ndarray) -> RunningMeanStd:
    """
    Update RunningMeanStd state using Welford's online algorithm.
    Computes batch statistics in a single pass for efficiency.
    """
    batch_count = jnp.array(x.shape[0], dtype=jnp.float32)
    
    batch_mean = jnp.mean(x, axis=0)
    batch_m2 = jnp.sum(jnp.square(x - batch_mean), axis=0)
    
    total_count = rms.count + batch_count
    
    delta = batch_mean - rms.mean
    new_mean = rms.mean + delta * batch_count / jnp.maximum(total_count, 1.0)
    
    m_a = rms.var * jnp.maximum(rms.count, 1.0)
    M2 = m_a + batch_m2 + jnp.square(delta) * rms.count * batch_count / jnp.maximum(total_count, 1.0)
    new_var = M2 / jnp.maximum(total_count, 1.0)
    
    return RunningMeanStd(mean=new_mean, var=jnp.where(total_count > 0, new_var, rms.var), count=total_count)

def compute_gae(rewards, values, dones, gamma, lam, last_val):
    """
    Generalized Advantage Estimation using lax.scan for JIT safety.
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        r, v, d = transition
        delta = r + gamma * next_value * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return (gae, v), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        (rewards, values, dones),
        reverse=True
    )
    return advantages, advantages + values

def flatten_obs(obs):
    """
    Maps dict observations or raw arrays into a flattened feature vector.
    Handles both single (D,) and batched (B, D) observations.
    """
    if not isinstance(obs, jnp.ndarray):
        obs = obs["vector"]
    
    if obs.ndim == 1:
        return obs
    
    return obs.reshape(obs.shape[0], -1)

def decode_action(action_idx):
    """
    Maps a flat 17-dim action index to (Verb, Direction, Target).
    DEPRECATED: Use per-unit actions instead.
    
    - 0: Stop
    - 1-8: Move
    - 9-16: Attack
    """
    verb = jnp.where(action_idx == 0, 0, jnp.where(action_idx <= 8, 1, 2))
    direction = jnp.where(verb == 1, action_idx - 1, 0).astype(jnp.int32)
    target = jnp.where(verb == 2, action_idx - 9, 0).astype(jnp.int32)
    return verb.astype(jnp.int32), direction.astype(jnp.int32), target.astype(jnp.int32)

def encode_per_unit_action(verb, direction, target):
    """Pack per-unit action arrays into a single flat index for storage.
    
    Usage: Index = verb + 3 * direction + 3 * 8 * target
    Range: [0, 3*8*num_targets)
    
    verb: (N,) int32 in [0, 3]
    direction: (N,) int32 in [0, 8]  
    target: (N,) int32 in [-1, num_targets-1], -1 maps to 0
    """
    target_packed = jnp.clip(target, 0, None)  # -1 -> 0 (not needed for loss, just storage)
    return verb + 3 * direction + 3 * 8 * target_packed

def decode_per_unit_action(flat_idx, num_targets):
    """Unpack flat index back into (verb, direction, target) arrays.
    
    flat_idx: (N,) int32
    Returns: (verb, direction, target) each (N,), target in [-1, num_targets-1]
    """
    rem = flat_idx
    verb = rem % 3
    rem = rem // 3
    direction = rem % 8
    target_packed = rem // 8
    target = jnp.where(target_packed == 0, -1, target_packed)
    return verb.astype(jnp.int32), direction.astype(jnp.int32), target.astype(jnp.int32)
