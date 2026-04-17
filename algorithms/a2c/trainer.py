import jax
import jax.numpy as jnp
import optax
import time
from flax.training.train_state import TrainState

from JaxSC2.env.twobridge import TwoBridgeEnv, CentralAction
from algorithms.ppo.model import ActorCritic 
from algorithms.a2c.a2c_logic import a2c_loss
from algorithms.common.utils import compute_gae, flatten_obs, decode_action, RunningMeanStd, update_rms
from algorithms.common.logging import Logger
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint
from algorithms.ppo.eval import evaluate # A2C can use the same eval logic

class A2CTrainer:
    """Advantage Actor-Critic (A2C) trainer optimized for massive parallelism.

    This implementation focuses on high sample throughput by utilizing a 
    single-pass update strategy (no multiple epochs). It is designed to scale 
    with `num_envs` and leverages JAX transformation for maximum efficiency.

    Attributes:
        config (dict): Base hyperparameters for A2C training.
    """
    def __init__(self, config=None):
        self.default_config = {
            "NUM_ENVS": 64, # A2C often benefits from more parallel envs
            "ROLLOUT_LEN": 128, # Shorter rollouts for A2C (standard)
            "LR": 7e-4, 
            "GAMMA": 0.99,
            "GAE_LAMBDA": 1.0, # Standard A2C usually doesn't use GAE but we can for stability
            "ENTROPY_COEFF": 0.01,
            "VF_COEFF": 0.5,
            "LOG_INTERVAL": 10,
            "EVAL_INTERVAL": 50,
            "CKPT_INTERVAL": 200,
            "VARIANT_NAME": "V1_Base"
        }
        if config:
            self.default_config.update(config)
        self.config = self.default_config

    def train(self, total_steps: int):
        """Executes the high-throughput A2C training cycle.

        Performs a single vectorized rollout followed by a single gradient 
        ascent step. This high-parallelism strategy is optimized for JAX's 
        vectorization capabilities, providing maximum sample efficiency 
        per second of wall-clock time.

        Args:
            total_steps (int): Total steps across all environments.
        """
        config = self.config
        num_envs = config["NUM_ENVS"]
        rollout_len = config["ROLLOUT_LEN"]
        batch_size = num_envs * rollout_len

        print("="*50)
        print(f"JaxSC2 A2C Trainer | Variant: {config['VARIANT_NAME']}")
        print("="*50)
        
        env = TwoBridgeEnv(variant_name=config["VARIANT_NAME"], use_spatial_obs=False)
        logger = Logger(f"runs/{config['VARIANT_NAME']}_a2c/logs")
        ckpt_dir = f"runs/{config['VARIANT_NAME']}_a2c/checkpoints"
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        dummy_obs = jnp.zeros((1, 63))
        model = ActorCritic(action_dim=17)
        params = model.init(init_rng, dummy_obs)
        
        tx = optax.adam(config["LR"])
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

        def rollout_step(carry, _):
            params, env_state, obs, rng = carry
            logits, value = model.apply(params, obs)
            rng, sub_rng = jax.random.split(rng)
            action_idx = jax.random.categorical(sub_rng, logits)
            
            def single_env_step(s, r, a_idx):
                v, d, t = decode_action(a_idx)
                act = CentralAction(who_mask=jnp.ones((env.num_allies,), dtype=jnp.int32), verb=v, direction=d, target=t)
                o, s2, rew, done, i = env.step(r, s, act)
                return o, s2, rew, done

            rng, step_rng = jax.random.split(rng)
            next_obs, next_env_state, reward, done = jax.vmap(single_env_step)(
                env_state, jax.random.split(step_rng, num_envs), action_idx
            )
            next_obs_flat = flatten_obs(next_obs)
            
            transition = (obs, action_idx, value, reward, done)
            return (params, next_env_state, next_obs_flat, rng), transition

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms):
            carry = (state.params, env_state, obs, rng)
            (params, next_env_state, next_obs, next_rng), transitions = jax.lax.scan(
                rollout_step, carry, None, length=rollout_len
            )
            obs_b, act_b, val_b, rew_b, done_b = transitions
            
            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)
            mean_reward = jnp.mean(rew_b)
            
            _, last_val = model.apply(params, next_obs) 
            adv_b, ret_b = compute_gae(norm_rew_b, val_b, done_b, config["GAMMA"], config["GAE_LAMBDA"], last_val)
            
            # A2C Update (Single pass)
            obs_f = obs_b.reshape((-1, obs_b.shape[-1]))
            act_f = act_b.reshape((-1,))
            adv_f = adv_b.reshape((-1,))
            ret_f = ret_b.reshape((-1,))
            
            # Explicitly pass state.params for consistency
            def compute_loss(params):
                return a2c_loss(
                    params, state.apply_fn, obs_f, act_f, adv_f, ret_f, 
                    ent_coeff=config["ENTROPY_COEFF"], 
                    vf_coeff=config["VF_COEFF"]
                )
            
            (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            
            return state, next_env_state, next_obs, next_rng, rms, aux, mean_reward

        # Rollout Start
        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size
        
        for i in range(1, num_iters + 1):
            state, env_state, obs_flat, rng, rms, aux, mean_reward = train_iteration(state, env_state, obs_flat, rng, rms)
            global_step = i * batch_size
            
            if i % config["LOG_INTERVAL"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                log_dict = {
                    "train/sps": sps,
                    "train/mean_reward": float(mean_reward),
                    "train/rew_std": float(jnp.sqrt(rms.var)[0]),
                    **{k: float(v) for k, v in aux.items()}
                }
                logger.log(global_step, log_dict)
                print(
                    f"Iter {i:4d} | Step {global_step:8d} | "
                    f"Loss {float(aux['loss/total']):7.4f} | "
                    f"PL {float(aux['loss/policy']):6.4f} | "
                    f"VL {float(aux['loss/critic']):6.4f} | "
                    f"Ent {float(aux['loss/entropy']):5.3f} | "
                    f"SPS {sps:,.0f}",
                    flush=True
                )
                
            if i % config["EVAL_INTERVAL"] == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = evaluate(env, state.params, model, eval_rng)
                logger.log(global_step, eval_metrics)
                print(f">>> EVAL: Reward {eval_metrics['eval/mean_reward']:.2f} | Win (Nav: {eval_metrics['eval/nav_win_rate']:.1%}, Combat: {eval_metrics['eval/combat_win_rate']:.1%}, Total: {eval_metrics['eval/total_win_rate']:.1%})", flush=True)
                
            if i % config["CKPT_INTERVAL"] == 0:
                save_checkpoint(f"{ckpt_dir}/ckpt_{global_step}.pkl", state, global_step)

        logger.close()

if __name__ == "__main__":
    trainer = A2CTrainer()
    trainer.train(100_000_000)
