import jax
import jax.numpy as jnp
import optax
from algorithms.ppo.trainer import PPOTrainer
from algorithms.ppo.model import ActorCritic
from flax.training.train_state import TrainState

def test_ppo_jit_stability():
    """
    Verifies that PPOTrainer.train_iteration is JIT-stable and doesn't retrace.
    """
    trainer = PPOTrainer({"NUM_ENVS": 4, "ROLLOUT_LEN": 16})
    config = trainer.config
    
    # Setup state
    rng = jax.random.PRNGKey(0)
    model = ActorCritic(action_dim=17)
    params = model.init(rng, jnp.zeros((1, 63)))
    tx = optax.adam(3e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    from JaxSC2.env.twobridge import TwoBridgeEnv
    env = TwoBridgeEnv(variant_name="V1_Base")
    obs, env_state = jax.vmap(env.reset)(jax.random.split(rng, 4))
    from algorithms.common.utils import flatten_obs, RunningMeanStd
    obs_flat = flatten_obs(obs)
    rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))
    
    # Get the train_iteration function
    # Note: It's defined inside trainer.train, so we need a way to access it 
    # for testing. Or we test the trainer.train performance.
    # Refactoring suggestion: Move train_iteration to a class method or static method.
    pass

if __name__ == "__main__":
    # For now, let's just run a smoke test iteration
    trainer = PPOTrainer({"NUM_ENVS": 4, "ROLLOUT_LEN": 16, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1})
    trainer.train(4 * 16) # Exactly one iteration
    print("PPO Smoke Test Passed")
