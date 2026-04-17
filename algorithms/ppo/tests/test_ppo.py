import jax
import jax.numpy as jnp
from algorithms.ppo.trainer import PPOTrainer
from algorithms.ppo.ppo_logic import ppo_loss

def test_trainer_init():
    trainer = PPOTrainer(config={"NUM_ENVS": 4, "ROLLOUT_LEN": 16})
    assert trainer.config["NUM_ENVS"] == 4
    assert trainer.config["ROLLOUT_LEN"] == 16

def test_ppo_loss_shape():
    batch_size = 16
    obs = jnp.zeros((batch_size, 63))
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    old_logp = jnp.zeros(batch_size)
    adv = jnp.zeros(batch_size)
    ret = jnp.zeros(batch_size)
    
    params = {} # Dummy
    def dummy_apply(p, x):
        return jnp.zeros((batch_size, 17)), jnp.zeros((batch_size,))
    
    loss = ppo_loss(params, dummy_apply, obs, actions, old_logp, adv, ret, clip_eps=0.2)
    assert loss.shape == ()
    assert not jnp.isnan(loss)

def test_trainer_smoke():
    # Very small run to ensure no crashes
    trainer = PPOTrainer(config={
        "NUM_ENVS": 2, 
        "ROLLOUT_LEN": 8,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "LOG_INTERVAL": 1,
        "EVAL_INTERVAL": 1
    })
    # Run only 16 steps (1 iteration)
    trainer.train(16)
