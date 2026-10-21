import jax
import jax.numpy as jnp


def a2c_loss_multi_head(params, apply_fn, obs, verb_act, dir_act, tgt_act,
                        old_logp, advantages, returns, clip_eps=0.2,
                        ent_coeff=0.01, vf_coeff=0.5):
    """Unclipped policy gradient over all sampled command heads."""
    output = apply_fn(params, obs)
    logp_act = jnp.zeros_like(advantages)
    entropy = jnp.zeros_like(advantages)
    for name, action in (("verb_logits", verb_act), ("direction_logits", dir_act),
                         ("target_logits", tgt_act)):
        logp = jax.nn.log_softmax(output[name])
        logp_act = logp_act + jnp.take_along_axis(logp, action[:, None], axis=-1).squeeze(-1)
        entropy = entropy - jnp.sum(jax.nn.softmax(output[name]) * logp, axis=-1)
    policy_loss = -jnp.mean(logp_act * advantages)
    value_loss = jnp.mean((returns - output["value"]) ** 2)
    mean_entropy = jnp.mean(entropy)
    loss = policy_loss + vf_coeff * value_loss - ent_coeff * mean_entropy
    return loss, {
        "loss/total": loss, "loss/policy": policy_loss,
        "loss/critic": value_loss, "loss/entropy": mean_entropy,
    }

def a2c_loss(params, apply_fn, obs, actions, advantages, returns, ent_coeff=0.01, vf_coeff=0.5):
    """A2C loss - assumes advantages are already normalized outside."""
    logits, values = apply_fn(params, obs)
    
    logp = jax.nn.log_softmax(logits)
    logp_act = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)

    # NOTE: advantages are normalized BEFORE calling this function (in the training loop)
    
    policy_loss = -jnp.mean(logp_act * advantages)
    
    value_loss = jnp.mean((returns - values) ** 2)
    
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * logp, axis=-1))

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    aux = {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/critic": value_loss,
        "loss/entropy": entropy,
    }

    return total_loss, aux
