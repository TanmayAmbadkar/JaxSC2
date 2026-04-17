import jax
import jax.numpy as jnp
import optax

def ppo_loss(params, apply_fn, obs, actions, old_logp, advantages, returns, clip_eps, ent_coeff=0.01, vf_coeff=0.5):
    logits, values = apply_fn(params, obs)
    
    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)

    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - values) ** 2)
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp, axis=-1))

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    # --- Aux metrics ---
    clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
    approx_kl = jnp.mean(old_logp - logp_act)
    explained_var = jnp.where(
        jnp.var(returns) > 1e-8,
        1.0 - jnp.var(returns - values) / jnp.var(returns),
        0.0
    )

    aux = {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/critic": value_loss,
        "loss/entropy": entropy,
        "train/clip_frac": clip_frac,
        "train/approx_kl": approx_kl,
        "train/explained_var": explained_var,
    }

    return total_loss, aux

def masked_ppo_loss(params, apply_fn, obs, actions, old_logp, 
                    advantages, returns, action_masks, clip_eps, ent_coeff, vf_coeff=0.5):
    # Pass masks into forward pass
    logits, values = apply_fn(params, obs, action_masks)
    
    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)

    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - values) ** 2)
    
    # Entropy over valid actions
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * logp, axis=-1))

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    # Aux metrics
    clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
    approx_kl = jnp.mean(old_logp - logp_act)
    explained_var = jnp.where(
        jnp.var(returns) > 1e-8,
        1.0 - jnp.var(returns - values) / jnp.var(returns),
        0.0
    )

    aux = {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/critic": value_loss,
        "loss/entropy": entropy,
        "train/clip_frac": clip_frac,
        "train/approx_kl": approx_kl,
        "train/explained_var": explained_var,
    }

    return total_loss, aux
