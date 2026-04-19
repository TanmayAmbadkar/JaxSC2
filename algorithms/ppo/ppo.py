import jax
import jax.numpy as jnp
import optax
import time
from typing import Any, Dict, Optional
from flax.training.train_state import TrainState

from algorithms.common.base import BaseAlgorithm
from algorithms.ppo.model import MultiHeadActorCritic
from algorithms.ppo.ppo_logic import ppo_loss_multi_head
from algorithms.common.utils import compute_gae, flatten_obs, RunningMeanStd, update_rms
from algorithms.common.logging import Logger
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint

class PPO(BaseAlgorithm):
    """
    Multi-head PPO implementation for per-unit action spaces.
    
    Action space: (verb, direction, target) with separate heads for each.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
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
            "SEED": 42
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        
        self.state = None
        self.model = None
        self.rms = None

    def train(self, env: Any, total_steps: int):
        config = self.config
        num_envs = config["NUM_ENVS"]
        rollout_len = config["ROLLOUT_LEN"]
        batch_size = num_envs * rollout_len
        minibatch_size = batch_size // config["NUM_MINIBATCHES"]

        # 1. Environment-Aware Initialization
        obs_dim = 63 # Default for TwoBridge
        if hasattr(env, "observation_space"):
            if hasattr(env.observation_space, "shape"):
                obs_dim = env.observation_space.shape[0]

        num_enemies = getattr(env, "num_enemies", 5)
        
        print(f"PPO Training started | Obs Dim: {obs_dim} | Num Enemies: {num_enemies}")
        
        logger = Logger(f"runs/ppo/logs")
        ckpt_dir = f"runs/ppo/checkpoints"
        
        rng = jax.random.PRNGKey(config["SEED"])
        rng, init_rng = jax.random.split(rng)
        
        # Init Network - Multi-head ActorCritic
        dummy_obs = jnp.zeros((1, obs_dim))
        self.model = MultiHeadActorCritic(action_dim=num_enemies)
        params = self.model.init(init_rng, dummy_obs)
        
        tx = optax.adam(config["LR"])
        self.state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        self.rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

        # --- JAX-Native Rollout Step (Closure) ---
        def rollout_step(carry, _):
            params, env_state, obs, rng = carry
            output = self.model.apply(params, obs)
            verb_logits = output["verb_logits"]
            direction_logits = output["direction_logits"]
            target_logits = output["target_logits"]
            value = output["value"]
            
            rng, sub_rng = jax.random.split(rng)
            
            # Sample from each head independently
            verb_idx = jax.random.categorical(sub_rng, verb_logits)
            direction_idx = jax.random.categorical(jax.random.split(sub_rng, num_envs), direction_logits)
            target_idx = jax.random.categorical(jax.random.split(sub_rng, num_envs), target_logits)
            
            # Compute log probabilities for each head
            verb_logp = jnp.take_along_axis(
                jax.nn.log_softmax(verb_logits), verb_idx[:, None], axis=-1
            ).squeeze(-1)
            
            direction_logp = jnp.take_along_axis(
                jax.nn.log_softmax(direction_logits), direction_idx[:, None], axis=-1
            ).squeeze(-1)
            
            target_logp = jnp.take_along_axis(
                jax.nn.log_softmax(target_logits), target_idx[:, None], axis=-1
            ).squeeze(-1)
            
            # Combined log probability (sum of all heads)
            logp = verb_logp + direction_logp + target_logp
            
            # This logic assumes JaxSC2Env step signature
            def single_env_step(s, r, v_idx, d_idx, t_idx):
                num_allies = getattr(env, "num_allies", 5)
                from JaxSC2.env.env import PerUnitAction
                act = PerUnitAction(
                    who_mask=jnp.ones((num_allies,), dtype=jnp.bool_),
                    verb=jnp.array([v_idx], dtype=jnp.int32),
                    direction=jnp.array([d_idx], dtype=jnp.int32),
                    target=jnp.array([t_idx], dtype=jnp.int32),
                )
                o, s2, rew, done, i = env.step(r, s, act)
                return o, s2, rew, done

            rng, step_rng = jax.random.split(rng)
            next_obs, next_env_state, reward, done = jax.vmap(single_env_step)(
                env_state, 
                jax.random.split(step_rng, num_envs), 
                verb_idx,
                direction_idx,
                target_idx
            )
            next_obs_flat = flatten_obs(next_obs)
            
            transition = (obs, verb_idx, direction_idx, target_idx, logp, value, reward, done)
            return (params, next_env_state, next_obs_flat, rng), transition

        def update_minibatch(train_state, batch):
            obs, verb_act, dir_act, tgt_act, logp, adv, ret = batch
            (loss, aux), grads = jax.value_and_grad(ppo_loss_multi_head, has_aux=True)(
                train_state.params, train_state.apply_fn, obs, verb_act, dir_act, tgt_act, 
                logp, adv, ret, 
                clip_eps=config["CLIP_EPS"], 
                ent_coeff=config["ENTROPY_COEFF"], 
                vf_coeff=config["VF_COEFF"]
            )
            return train_state.apply_gradients(grads=grads), aux

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms):
            carry = (state.params, env_state, obs, rng)
            (params, next_env_state, next_obs, next_rng), transitions = jax.lax.scan(
                rollout_step, carry, None, length=rollout_len
            )
            obs_b, verb_b, dir_b, tgt_b, logp_b, val_b, rew_b, done_b = transitions
            
            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)
            mean_reward = jnp.mean(rew_b)
            
            _, last_val = self.model.apply(params, next_obs) 
            adv_b, ret_b = compute_gae(norm_rew_b, val_b, done_b, config["GAMMA"], config["GAE_LAMBDA"], last_val)
            
            # Normalize advantages ONCE per rollout (before splitting into minibatches)
            adv_b = (adv_b - jnp.mean(adv_b)) / (jnp.std(adv_b) + 1e-8)
            
            batch = (obs_b, verb_b, dir_b, tgt_b, logp_b, adv_b, ret_b)
            f_batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

            update_state = (state, next_rng)
            def epoch_loop(update_carry, _):
                ts, r = update_carry
                r, perm_rng = jax.random.split(r)
                permutation = jax.random.permutation(perm_rng, batch_size)
                s_batch = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size), s_batch)
                
                def minibatch_loop(ts_inner, i):
                    start = i * minibatch_size
                    mb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size), s_batch)
                    return update_minibatch(ts_inner, mb)
                
                ts, iter_aux = jax.lax.scan(minibatch_loop, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, r), iter_aux

            (state, final_rng), aux_b = jax.lax.scan(epoch_loop, update_state, None, length=config["UPDATE_EPOCHS"])
            mean_aux = jax.tree_util.tree_map(jnp.mean, aux_b)
            return state, next_env_state, next_obs, final_rng, rms, mean_aux, mean_reward

        # --- Training Loop ---
        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size
        
        curr_state = self.state
        for i in range(1, num_iters + 1):
            curr_state, env_state, obs_flat, rng, self.rms, mean_aux, mean_reward = train_iteration(
                curr_state, env_state, obs_flat, rng, self.rms
            )
            global_step = i * batch_size
            
            if i % config["LOG_INTERVAL"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                print(f"Iter {i:4d} | Step {global_step:8d} | Reward {float(mean_reward):7.2f} | SPS {sps:,.0f}")
                logger.log(global_step, {"train/mean_reward": float(mean_reward), "train/sps": sps, **{k: float(v) for k, v in mean_aux.items()}})
            
            if i % config["CKPT_INTERVAL"] == 0:
                self.save(f"{ckpt_dir}/ckpt_{global_step}.pkl")

        self.state = curr_state
        logger.close()

    def save(self, path: str):
        save_checkpoint(path, self.state, 0)

    def load(self, path: str):
        self.state = load_checkpoint(path, self.state)
