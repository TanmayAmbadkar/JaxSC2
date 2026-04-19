import flax.linen as nn
import jax.numpy as jnp

class MultiHeadActorCritic(nn.Module):
    """
    Multi-head Actor-Critic for per-unit action spaces.
    
    Outputs:
        verb_logits:      (N, 3)   - NO_OP, MOVE, ATTACK
        direction_logits: (N, 8)   - 8 directions
        target_logits:    (N, num_enemies) - enemy targets
        value:            ()       - state value
    """
    action_dim: int  # num_enemies for target head
    hidden_size: int = 256
    
    @nn.compact
    def __call__(self, x):
        # Shared backbone
        x = nn.LayerNorm()(x)
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        
        # Verb head: 3 actions (NO_OP, MOVE, ATTACK)
        verb_logits = nn.Dense(3)(x)
        
        # Direction head: 8 directions
        direction_logits = nn.Dense(8)(x)
        
        # Target head: num_enemies targets (action_dim)
        target_logits = nn.Dense(self.action_dim)(x)
        
        # Value head
        value = nn.Dense(1)(x).squeeze(-1)
        
        return {
            "verb_logits": verb_logits,
            "direction_logits": direction_logits,
            "target_logits": target_logits,
            "value": value,
        }

class ActorCritic(nn.Module):
    """Legacy single-head model for backward compatibility."""
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        
        return logits, value
