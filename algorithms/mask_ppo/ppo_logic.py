import jax
import jax.numpy as jnp


def masked_ppo_loss(params, apply_fn, obs, action_indices, old_logp,
                    advantages, returns, verb_mask, dir_mask, tgt_mask,
                    clip_eps=0.2, ent_coeff=0.01, vf_coeff=0.5):
    """Legacy flat-mode PPO loss for backward compat."""
    logits, value = apply_fn(params, obs)

    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, action_indices[:, :, None], axis=-1).squeeze(-1)

    ratio = jnp.exp(logp_act - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - value) ** 2)

    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * logp, axis=-1))

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
    approx_kl = jnp.mean(old_logp - logp_act)
    explained_var = jnp.where(
        jnp.var(returns) > 1e-8,
        1.0 - jnp.var(returns - value) / jnp.var(returns),
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


def masked_ppo_loss_multih(params, apply_fn, obs,
                           verb_idx, dir_idx, tgt_idx,
                           old_logp_v, old_logp_d, old_logp_t,
                           advantages, returns,
                           verb_mask, dir_mask, tgt_mask,
                           clip_eps=0.2, ent_coeff=0.01, vf_coeff=0.5):
    """Multi-head PPO loss with separate heads for verb, direction, target.

    Actions are scalar per batch item: verb_idx (B,), dir_idx (B,), tgt_idx (B,).
    Shapes returned from model: v_logits (B, 3), d_logits (B, 8), t_logits (B, num_enemies).
    """
    v_logits, d_logits, t_logits = apply_fn(params, obs, verb_mask, dir_mask, tgt_mask)

    value = apply_fn(params, obs, get_value_only=True)

    logp_v = jax.nn.log_softmax(v_logits)
    logp_d = jax.nn.log_softmax(d_logits)
    logp_t = jax.nn.log_softmax(t_logits)

    # Gather log-probs of taken actions: (B,)
    logp_act_v = jnp.take_along_axis(logp_v, verb_idx[:, None], axis=-1).squeeze(-1)
    logp_act_d = jnp.take_along_axis(logp_d, dir_idx[:, None], axis=-1).squeeze(-1)
    logp_act_t = jnp.take_along_axis(logp_t, tgt_idx[:, None], axis=-1).squeeze(-1)

    old_logp = old_logp_v + old_logp_d + old_logp_t
    new_logp = logp_act_v + logp_act_d + logp_act_t

    ratio = jnp.exp(new_logp - old_logp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    value_loss = jnp.mean((returns - value) ** 2)

    probs_v = jax.nn.softmax(v_logits)
    probs_d = jax.nn.softmax(d_logits)
    probs_t = jax.nn.softmax(t_logits)
    entropy_v = -jnp.mean(jnp.sum(probs_v * logp_v, axis=-1))
    entropy_d = -jnp.mean(jnp.sum(probs_d * logp_d, axis=-1))
    entropy_t = -jnp.mean(jnp.sum(probs_t * logp_t, axis=-1))
    entropy = entropy_v + entropy_d + entropy_t

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
    approx_kl = jnp.mean(old_logp - new_logp)
    explained_var = jnp.where(
        jnp.var(returns) > 1e-8,
        1.0 - jnp.var(returns - value) / jnp.var(returns),
        0.0
    )

    aux = {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/critic": value_loss,
        "loss/entropy_v": entropy_v,
        "loss/entropy_d": entropy_d,
        "loss/entropy_t": entropy_t,
        "train/clip_frac": clip_frac,
        "train/approx_kl": approx_kl,
        "train/explained_var": explained_var,
    }

    return total_loss, aux
