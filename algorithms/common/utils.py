import jax
import jax.numpy as jnp
from typing import NamedTuple

class RunningMeanStd(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float

def update_rms(rms: RunningMeanStd, x: jnp.ndarray) -> RunningMeanStd:
    """
    Update RunningMeanStd state using Welford's algorithm (JAX-friendly).
    """
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = jnp.array(x.shape[0], dtype=jnp.float32)
    
    total_count = rms.count + batch_count
    
    delta = batch_mean - rms.mean
    new_mean = rms.mean + delta * batch_count / total_count
    
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * rms.count * batch_count / total_count
    new_var = M2 / total_count
    
    return RunningMeanStd(mean=new_mean, var=new_var, count=total_count)

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
        # Assuming dict with "vector" key
        obs = obs["vector"]
    
    if obs.ndim == 1:
        return obs # (D,) -> (D,) for single samples
    
    # Batch case: (B, ...) -> (B, D)
    return obs.reshape(obs.shape[0], -1)

def decode_action(action_idx):
    """
    Maps a flat 17-dim action index to (Verb, Direction, Target).
    - 0: Stop
    - 1-8: Move
    - 9-16: Attack
    """
    verb = jnp.where(action_idx == 0, 0, jnp.where(action_idx <= 8, 1, 2))
    direction = jnp.where(verb == 1, action_idx - 1, 0).astype(jnp.int32)
    target = jnp.where(verb == 2, action_idx - 9, 0).astype(jnp.int32)
    return verb.astype(jnp.int32), direction.astype(jnp.int32), target.astype(jnp.int32)
