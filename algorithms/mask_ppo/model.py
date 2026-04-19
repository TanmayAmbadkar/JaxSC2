import flax.linen as nn
import jax.numpy as jnp


class MaskedActorCritic(nn.Module):
    action_dim: int = 17

    @nn.compact
    def __call__(self, x, verb_mask=None, dir_mask=None, tgt_mask=None, get_value_only=False):
        x = nn.LayerNorm()(x)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))

        if get_value_only:
            return nn.Dense(1)(x).squeeze(-1)

        # Value head always created
        value = nn.Dense(1)(x).squeeze(-1)

        if verb_mask is not None:
            v_logits = nn.Dense(3)(x)
            d_logits = nn.Dense(8)(x)
            t_dim = tgt_mask.shape[-1] if tgt_mask is not None else 3
            t_logits = nn.Dense(t_dim)(x)

            v_mask2d = jnp.squeeze(verb_mask, axis=(1,) if verb_mask.ndim == 3 else ())
            d_mask2d = jnp.squeeze(dir_mask, axis=(1,) if dir_mask.ndim == 3 else ())
            t_mask2d = jnp.squeeze(tgt_mask, axis=(1,) if tgt_mask.ndim == 3 else ())

            v_logits = jnp.where(v_mask2d, v_logits, -1e9)
            d_logits = jnp.where(d_mask2d, d_logits, -1e9)
            t_logits = jnp.where(t_mask2d, t_logits, -1e9)

            return v_logits, d_logits, t_logits
        else:
            logits = nn.Dense(self.action_dim)(x)
            return logits, value
