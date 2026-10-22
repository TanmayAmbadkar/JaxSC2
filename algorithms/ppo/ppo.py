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
from algorithms.common.rollout import step_and_reset
from algorithms.common.episode_logging import EpisodeAccumulator, EpisodeLogWindow, accumulate_episodes

class PPO(BaseAlgorithm):
    """
    Multi-head PPO implementation for per-unit action spaces.
    
    Action space: (verb, direction, target) with separate heads for each.
    """
    loss_fn = staticmethod(ppo_loss_multi_head)

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
        self.total_steps = 0
        self._pending_checkpoint = None

    def train(self, env: Any, total_steps: int):
        if getattr(env, "economic", False):
            raise ValueError("Economy tasks require a TaskAction policy with macro heads; standard PPO controls army actions only")
        config = self.config
        num_envs = config["NUM_ENVS"]
        rollout_len = config["ROLLOUT_LEN"]
        batch_size = num_envs * rollout_len
        if config["NUM_MINIBATCHES"] < 1 or batch_size % config["NUM_MINIBATCHES"]:
            raise ValueError("NUM_MINIBATCHES must be positive and divide the rollout batch")
        minibatch_size = batch_size // config["NUM_MINIBATCHES"]

        # 1. Environment-Aware Initialization
        obs_dim = 63 # Default for TwoBridge
        if hasattr(env, "observation_space"):
            if hasattr(env.observation_space, "shape"):
                obs_dim = env.observation_space.shape[0]

        num_enemies = getattr(env, "num_enemies", 5)
        
        print(f"{type(self).__name__} Training started | Obs Dim: {obs_dim} | Num Enemies: {num_enemies}")
        
        logger = Logger(config.get("LOG_DIR", "runs/ppo/logs"))
        ckpt_dir = config.get("CKPT_DIR", "runs/ppo/checkpoints")
        
        rng = jax.random.PRNGKey(config["SEED"])
        rng, init_rng = jax.random.split(rng)
        
        # Init Network - Multi-head ActorCritic
        if self.state is None:
            self.model = MultiHeadActorCritic(action_dim=num_enemies)
            params = self.model.init(init_rng, jnp.zeros((1, obs_dim)))
            self.state = TrainState.create(
                apply_fn=self.model.apply, params=params, tx=optax.adam(config["LR"])
            )
            self.rms = RunningMeanStd(jnp.zeros(1), jnp.ones(1), jnp.array(1e-4))
        if self._pending_checkpoint is not None:
            self.load(self._pending_checkpoint)

        # --- JAX-Native Rollout Step (Closure) ---
        def rollout_step(carry, _):
            params, env_state, obs, rng = carry
            output = self.model.apply(params, obs)
            verb_logits = output["verb_logits"]
            direction_logits = output["direction_logits"]
            target_logits = output["target_logits"]
            value = output["value"]
            
            rng, verb_rng, dir_rng, target_rng = jax.random.split(rng, 4)
            
            # Sample from each head independently
            verb_idx = jax.random.categorical(verb_rng, verb_logits)
            direction_idx = jax.random.categorical(dir_rng, direction_logits)
            target_idx = jax.random.categorical(target_rng, target_logits)
            
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
                    who_mask=s.smax_state.unit_alive[:num_allies],
                    verb=jnp.full(num_allies, v_idx, dtype=jnp.int32),
                    direction=jnp.full(num_allies, d_idx, dtype=jnp.int32),
                    target=jnp.full(num_allies, t_idx, dtype=jnp.int32),
                )
                return step_and_reset(env, r, s, act)

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
            (loss, aux), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
                train_state.params, train_state.apply_fn, obs, verb_act, dir_act, tgt_act, 
                logp, adv, ret, 
                clip_eps=config["CLIP_EPS"], 
                ent_coeff=config["ENTROPY_COEFF"], 
                vf_coeff=config["VF_COEFF"]
            )
            return train_state.apply_gradients(grads=grads), aux

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms, episode_accumulator):
            carry = (state.params, env_state, obs, rng)
            (params, next_env_state, next_obs, next_rng), transitions = jax.lax.scan(
                rollout_step, carry, None, length=rollout_len
            )
            obs_b, verb_b, dir_b, tgt_b, logp_b, val_b, rew_b, done_b = transitions
            episode_accumulator, completed = accumulate_episodes(episode_accumulator, rew_b, done_b)
            
            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)
            mean_reward = jnp.mean(rew_b)
            
            last_val = self.model.apply(params, next_obs)["value"]
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
                s_batch = jax.tree_util.tree_map(lambda x: x[permutation], f_batch)
                
                def minibatch_loop(ts_inner, i):
                    start = i * minibatch_size
                    mb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size), s_batch)
                    return update_minibatch(ts_inner, mb)
                
                ts, iter_aux = jax.lax.scan(minibatch_loop, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, r), iter_aux

            (state, final_rng), aux_b = jax.lax.scan(epoch_loop, update_state, None, length=config["UPDATE_EPOCHS"])
            mean_aux = jax.tree_util.tree_map(jnp.mean, aux_b)
            return state, next_env_state, next_obs, final_rng, rms, mean_aux, mean_reward, episode_accumulator, completed

        # --- Training Loop ---
        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size
        
        curr_state = self.state
        start_step = self.total_steps
        episode_accumulator = EpisodeAccumulator(jnp.zeros(num_envs), jnp.zeros(num_envs, dtype=jnp.int32))
        episode_window = EpisodeLogWindow()
        for i in range(1, num_iters + 1):
            curr_state, env_state, obs_flat, rng, self.rms, mean_aux, mean_reward, episode_accumulator, completed = train_iteration(
                curr_state, env_state, obs_flat, rng, self.rms, episode_accumulator,
            )
            episode_window.add(completed)
            global_step = start_step + i * batch_size
            global_iteration = global_step // batch_size
            self.state = curr_state
            self.total_steps = global_step
            
            if global_iteration % config["LOG_INTERVAL"] == 0 or i == num_iters:
                elapsed = time.time() - start_time
                sps = (global_step - start_step) / elapsed
                episode_metrics, display = episode_window.emit()
                print(f"Iter {global_iteration:4d} | Step {global_step:8d} | {display} | SPS {sps:,.0f}")
                logger.log(global_step, {**episode_metrics, "train/step_reward": float(mean_reward),
                           "train/sps": sps, **{k: float(v) for k, v in mean_aux.items()}})

            if config.get("EVAL_INTERVAL", 0) and global_iteration % config["EVAL_INTERVAL"] == 0:
                from algorithms.ppo.eval import evaluate
                rng, eval_rng = jax.random.split(rng)
                metrics = evaluate(
                    env, curr_state.params, self.model, eval_rng,
                    num_episodes=config.get("EVAL_EPISODES", 16),
                    max_steps=config.get("EVAL_MAX_STEPS", env.max_steps),
                )
                logger.log(global_step, metrics)
            
            if global_iteration % config["CKPT_INTERVAL"] == 0:
                self.save(f"{ckpt_dir}/ckpt_{global_step}.pkl")

        self.state = curr_state
        logger.close()

    def save(self, path: str):
        if self.state is None:
            raise ValueError("Initialize the model with train(env, 0) before saving")
        save_checkpoint(path, self.state, self.total_steps, rms=self.rms)

    def load(self, path: str):
        if self.state is None:
            self._pending_checkpoint = path
            return
        self.state, self.total_steps, metadata = load_checkpoint(path, self.state, return_metadata=True)
        self.rms = metadata.get("rms", self.rms)
        self._pending_checkpoint = None
