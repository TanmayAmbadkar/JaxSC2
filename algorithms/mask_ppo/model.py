import flax.linen as nn
import jax.numpy as jnp

class MaskedActorCritic(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x, action_mask):
        x = nn.LayerNorm()(x)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        
        # Always mask logits: invalid actions get -1e9
        logits = jnp.where(action_mask, logits, jnp.full_like(logits, -1e9))
        
        return logits, value
