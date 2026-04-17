import jax
import jax.numpy as jnp
import flax.linen as nn

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        
        return logits, value
