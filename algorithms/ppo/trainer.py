import jax
import jax.numpy as jnp
import optax
import time
from flax.training.train_state import TrainState

from JaxSC2.env.twobridge import TwoBridgeEnv, CentralAction
from algorithms.ppo.model import ActorCritic
from algorithms.ppo.ppo_logic import ppo_loss
from algorithms.common.utils import compute_gae, flatten_obs, decode_action, RunningMeanStd, update_rms
from algorithms.common.logging import Logger
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint
from algorithms.ppo.eval import evaluate

class PPOTrainer:
    """Standard Proximal Policy Optimization (PPO) trainer for Sc2Jax.

    This trainer implements a high-throughput, JAX-native PPO pipeline specifically 
    optimized for vectorized environments. It utilizes `jax.lax.scan` for both 
    rollout collection and optimization epochs to eliminate Python-loop overhead.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters like 
            CLIP_EPS, ENTROPY_COEFF, and LEARN_RATE.
    """
    def __init__(self, config=None):
        self.default_config = {
            "NUM_ENVS": 32,
            "ROLLOUT_LEN": 512,
            "UPDATE_EPOCHS": 10,
            "NUM_MINIBATCHES": 16,
            "LR": 3e-4,
            "CLIP_EPS": 0.2,
            "GAMMA": 0.995,
            "GAE_LAMBDA": 0.95,
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
        """Executes the standard PPO training loop.

        This method orchestrates the JIT-compiled training iteration. It manages 
        vectorized rollout collection, advantage calculation (GAE), and 
        mini-batch optimization via `lax.scan`. Includes periodic evaluation 
        and checkpointing.

        Args:
            total_steps (int): Total number of environment steps to train for.
        """
        config = self.config
        num_envs = config["NUM_ENVS"]
        rollout_len = config["ROLLOUT_LEN"]
        batch_size = num_envs * rollout_len
        minibatch_size = batch_size // config["NUM_MINIBATCHES"]

        print("="*50)
        print(f"JaxSC2 Standard PPO Trainer | Variant: {config['VARIANT_NAME']}")
        print("="*50)
        
        env = TwoBridgeEnv(variant_name=config["VARIANT_NAME"], use_spatial_obs=False)
        logger = Logger(f"runs/{config['VARIANT_NAME']}_ppo/logs")
        ckpt_dir = f"runs/{config['VARIANT_NAME']}_ppo/checkpoints"
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        # Init Network
        dummy_obs = jnp.zeros((1, 63))
        model = ActorCritic(action_dim=17)
        params = model.init(init_rng, dummy_obs)
        
        tx = optax.adam(config["LR"])
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

        # JIT Closure Components
        def rollout_step(carry, _):
            params, env_state, obs, rng = carry
            logits, value = model.apply(params, obs)
            rng, sub_rng = jax.random.split(rng)
            action_idx = jax.random.categorical(sub_rng, logits)
            logp = jnp.take_along_axis(
                jax.nn.log_softmax(logits), action_idx[:, None], axis=1
            ).squeeze(-1)
            
            def single_env_step(s, r, a_idx):
                v, d, t = decode_action(a_idx)
                act = CentralAction(who_mask=jnp.ones((env.num_allies,), dtype=jnp.int32), verb=v, direction=d, target=t)
                o, s2, rew, done, i = env.step(r, s, act)
                return o, s2, rew, done

            rng, step_rng = jax.random.split(rng)
            next_obs, next_env_state, reward, done = jax.vmap(single_env_step)(env_state, jax.random.split(step_rng, num_envs), action_idx)
            next_obs_flat = flatten_obs(next_obs)
            
            transition = (obs, action_idx, logp, value, reward, done)
            return (params, next_env_state, next_obs_flat, rng), transition

        def update_minibatch(train_state, batch):
            obs, act, logp, adv, ret = batch
            (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                train_state.params, train_state.apply_fn, obs, act, logp, adv, ret, 
                clip_eps=config["CLIP_EPS"], 
                ent_coeff=config["ENTROPY_COEFF"], 
                vf_coeff=config["VF_COEFF"]
            )
            return train_state.apply_gradients(grads=grads), aux

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms):
            carry = (state.params, env_state, obs, rng)
            (params, next_env_state, next_obs, next_rng), transitions = jax.lax.scan(rollout_step, carry, None, length=rollout_len)
            obs_b, act_b, logp_b, val_b, rew_b, done_b = transitions
            
            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)
            mean_reward = jnp.mean(rew_b)
            
            _, last_val = model.apply(params, next_obs) 
            adv_b, ret_b = compute_gae(norm_rew_b, val_b, done_b, config["GAMMA"], config["GAE_LAMBDA"], last_val)
            
            # Optimization: Flatten outside epoch loop
            batch = (obs_b, act_b, logp_b, adv_b, ret_b)
            f_batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

            update_state = (state, next_rng)
            def epoch_loop(update_carry, _):
                ts, r = update_carry
                r, perm_rng = jax.random.split(r)
                permutation = jax.random.permutation(perm_rng, batch_size)
                s_batch = jax.tree_util.tree_map(lambda x: x[permutation], f_batch)
                
                def minibatch_loop(ts_inner, i):
                    start = i * minibatch_size
                    mb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size), s_batch)
                    return update_minibatch(ts_inner, mb)
                
                ts, iter_aux = jax.lax.scan(minibatch_loop, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, r), iter_aux

            (state, final_rng), aux_b = jax.lax.scan(epoch_loop, update_state, None, length=config["UPDATE_EPOCHS"])
            mean_aux = jax.tree_util.tree_map(jnp.mean, aux_b)
            return state, next_env_state, next_obs, final_rng, rms, mean_aux, mean_reward

        # Rollout Start
        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size
        
        for i in range(1, num_iters + 1):
            state, env_state, obs_flat, rng, rms, mean_aux, mean_reward = train_iteration(state, env_state, obs_flat, rng, rms)
            global_step = i * batch_size
            
            if i % config["LOG_INTERVAL"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                log_dict = {
                    "train/sps": sps,
                    "train/mean_reward": float(mean_reward),
                    "train/rew_std": float(jnp.sqrt(rms.var)[0]),
                    **{k: float(v) for k, v in mean_aux.items()}
                }
                logger.log(global_step, log_dict)
                print(
                    f"Iter {i:4d} | Step {global_step:8d} | "
                    f"Loss {float(mean_aux['loss/total']):7.4f} | "
                    f"PL {float(mean_aux['loss/policy']):6.4f} | "
                    f"VL {float(mean_aux['loss/critic']):6.4f} | "
                    f"Ent {float(mean_aux['loss/entropy']):5.3f} | "
                    f"KL {float(mean_aux['train/approx_kl']):.4f} | "
                    f"ClipF {float(mean_aux['train/clip_frac']):.3f} | "
                    f"EV {float(mean_aux['train/explained_var']):.3f} | "
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
    trainer = PPOTrainer()
    trainer.train(100_000_000)
