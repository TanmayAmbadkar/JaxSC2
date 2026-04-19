import jax
import jax.numpy as jnp

def ppo_loss(params, apply_fn, obs, actions, old_logp, advantages, returns, clip_eps, ent_coeff=0.01, vf_coeff=0.5):
    """PPO loss - assumes advantages are already normalized outside."""
    logits, values = apply_fn(params, obs)
    
    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)

    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    
    # NOTE: advantages are normalized BEFORE calling this function (in the training loop)
    # to ensure consistent normalization across all minibatches
    
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


def ppo_loss_multi_head(params, apply_fn, obs, verb_act, dir_act, tgt_act, 
                        old_logp, advantages, returns, clip_eps, ent_coeff=0.01, vf_coeff=0.5):
    """Multi-head PPO loss for per-unit action spaces (verb, direction, target)."""
    output = apply_fn(params, obs)
    verb_logits = output["verb_logits"]
    direction_logits = output["direction_logits"]
    target_logits = output["target_logits"]
    values = output["value"]
    
    # Compute log probabilities for each head
    verb_logp = jax.nn.log_softmax(verb_logits)
    dir_logp = jax.nn.log_softmax(direction_logits)
    tgt_logp = jax.nn.log_softmax(target_logits)
    
    verb_logp_act = jnp.take_along_axis(verb_logp, verb_act[:, None], axis=-1).squeeze(-1)
    dir_logp_act = jnp.take_along_axis(dir_logp, dir_act[:, None], axis=-1).squeeze(-1)
    tgt_logp_act = jnp.take_along_axis(tgt_logp, tgt_act[:, None], axis=-1).squeeze(-1)
    
    # Combined log probability
    logp_act = verb_logp_act + dir_logp_act + tgt_logp_act
    
    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - values) ** 2)
    
    # Entropy from all heads
    verb_probs = jax.nn.softmax(verb_logits)
    dir_probs = jax.nn.softmax(direction_logits)
    tgt_probs = jax.nn.softmax(target_logits)
    
    verb_entropy = -jnp.sum(verb_probs * verb_logp, axis=-1)
    dir_entropy = -jnp.sum(dir_probs * dir_logp, axis=-1)
    tgt_entropy = -jnp.sum(tgt_probs * tgt_logp, axis=-1)
    entropy = jnp.mean(verb_entropy + dir_entropy + tgt_entropy)

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
    """Masked PPO loss - assumes advantages are already normalized outside."""
    logits, values = apply_fn(params, obs, action_masks)
    
    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)

    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    
    # NOTE: advantages are normalized BEFORE calling this function (in the training loop)
    
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - values) ** 2)
    
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * logp, axis=-1))

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
