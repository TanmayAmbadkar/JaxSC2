import jax
import jax.numpy as jnp

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
