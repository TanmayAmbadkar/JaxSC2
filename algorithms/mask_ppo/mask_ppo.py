import jax
import jax.numpy as jnp
import optax
import time
from typing import Any, Dict, Optional
from flax.training.train_state import TrainState

from algorithms.common.base import BaseAlgorithm
from algorithms.mask_ppo.model import MaskedActorCritic
from algorithms.mask_ppo.ppo_logic import masked_ppo_loss_multih, masked_ppo_loss
from algorithms.common.utils import compute_gae, flatten_obs, RunningMeanStd, update_rms
from algorithms.common.logging import Logger
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint

from JaxSC2.env.env import PerUnitAction


class MaskPPO(BaseAlgorithm):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "NUM_ENVS": 32,
            "ROLLOUT_LEN": 512,
            "UPDATE_EPOCHS": 5,
            "NUM_MINIBATCHES": 16,
            "LR": 2e-4,
            "CLIP_EPS": 0.2,
            "GAMMA": 0.995,
            "GAE_LAMBDA": 0.95,
            "ENTROPY_COEFF": 0.01,
            "VF_COEFF": 0.5,
            "LOG_INTERVAL": 10,
            "CKPT_INTERVAL": 200,
            "SEED": 42,
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
        num_allies = getattr(env, "num_allies", 5)
        num_enemies = getattr(env, "num_enemies", 3)

        batch_size = num_envs * rollout_len
        minibatch_size = batch_size // config["NUM_MINIBATCHES"]

        print(f"MaskPPO Training started | Allies: {num_allies} | Enemies: {num_enemies} | Batch: {batch_size}")

        logger = Logger(f"runs/mask_ppo/logs")
        ckpt_dir = f"runs/mask_ppo/checkpoints"

        rng = jax.random.PRNGKey(config["SEED"])
        rng, init_rng = jax.random.split(rng)

        dummy_mask = jnp.ones((1, 3), dtype=jnp.bool_)
        dir_mask = jnp.ones((1, 8), dtype=jnp.bool_)
        tgt_mask = jnp.ones((1, num_enemies), dtype=jnp.bool_)
        self.model = MaskedActorCritic(action_dim=17)
        params = self.model.init(init_rng, jnp.zeros((1, 63)), dummy_mask, dir_mask, tgt_mask)
        tx = optax.adam(config["LR"])
        self.state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        self.rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

        loss_mask_v = jnp.ones((1, 3), dtype=jnp.bool_)
        loss_mask_d = jnp.ones((1, 8), dtype=jnp.bool_)
        loss_mask_t = jnp.ones((1, num_enemies), dtype=jnp.bool_)

        def rollout_step(carry, _):
            params_c, env_state_c, obs_c, rng_c = carry

            v_logits, d_logits, t_logits = self.model.apply(
                params_c, obs_c, loss_mask_v, loss_mask_d, loss_mask_t
            )

            # Collect value for GAE at each step (needed with lax.scan)
            value_c = self.model.apply(params_c, obs_c, get_value_only=True)

            rng_c, sub_rng = jax.random.split(rng_c)

            def single_env_step(s, r, v_l, d_l, t_l):
                verb_idx = jax.random.categorical(jax.random.split(r)[0], v_l)
                dir_idx = jax.random.categorical(jax.random.split(r)[1], d_l)
                tgt_idx = jax.random.categorical(jax.random.split(r)[2], t_l)

                logp_v = jnp.take_along_axis(
                    jax.nn.log_softmax(v_l), jnp.array([verb_idx]), axis=-1
                ).squeeze(-1)
                logp_d = jnp.take_along_axis(
                    jax.nn.log_softmax(d_l), jnp.array([dir_idx]), axis=-1
                ).squeeze(-1)
                logp_t = jnp.take_along_axis(
                    jax.nn.log_softmax(t_l), jnp.array([tgt_idx]), axis=-1
                ).squeeze(-1)

                rng2, step_rng = jax.random.split(r)
                act = PerUnitAction(
                    who_mask=jnp.ones((num_allies,), dtype=jnp.bool_),
                    verb=jnp.array([verb_idx], dtype=jnp.int32),
                    direction=jnp.array([dir_idx], dtype=jnp.int32),
                    target=jnp.array([tgt_idx], dtype=jnp.int32),
                )
                o, s2, rew, done, _ = env.step(step_rng, s, act)
                return o, s2, rew, done, verb_idx, dir_idx, tgt_idx, logp_v, logp_d, logp_t

            rng_c, split_rng = jax.random.split(rng_c)
            rngs_split = jax.random.split(split_rng, num_envs)

            results = jax.vmap(single_env_step)(
                env_state_c, rngs_split, v_logits, d_logits, t_logits
            )

            next_obs_v = results[0]
            next_env_state_v = results[1]

            return (params_c, next_env_state_v, next_obs_v, rng_c), (
                obs_c,  # observations at each step
                results[4], results[5], results[6],  # verb, dir, tgt
                results[7], results[8], results[9],  # logp_v, logp_d, logp_t
                results[2], results[3], value_c      # rew, done, value (at each step)
            )

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms):
            carry = (state.params, env_state, obs, rng)
            (_, next_env_state, next_obs, _), transitions = jax.lax.scan(
                rollout_step, carry, None, length=rollout_len
            )

            verb_b = transitions[1]   # (rollout_len, num_envs)
            dir_b = transitions[2]
            tgt_b = transitions[3]
            logp_v_b = transitions[4]
            logp_d_b = transitions[5]
            logp_t_b = transitions[6]
            rew_b = transitions[7]     # (rollout_len, num_envs)
            done_b = transitions[8]    # (rollout_len, num_envs)
            val_b = transitions[9]     # (rollout_len, num_envs) - values collected per step

            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)

            last_val = self.model.apply(state.params, next_obs, get_value_only=True)

            # GAE uses (rollout_len, num_envs) shapes with last_val as (num_envs,)
            adv_b, ret_b = compute_gae(norm_rew_b, val_b, done_b, config["GAMMA"], config["GAE_LAMBDA"], last_val)

            perm_rng, r = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)

            # Flatten everything for minibatch updates
            obs_b = transitions[0]  # (rollout_len, num_envs, feature_dim)
            obs_b_reshaped = obs_b.reshape((batch_size,) + obs_b.shape[2:])
            verb_b_reshaped = verb_b.reshape((batch_size,))
            dir_b_reshaped = dir_b.reshape((batch_size,))
            tgt_b_reshaped = tgt_b.reshape((batch_size,))
            logp_v_reshaped = logp_v_b.reshape((batch_size,))
            logp_d_reshaped = logp_d_b.reshape((batch_size,))
            logp_t_reshaped = logp_t_b.reshape((batch_size,))

            adv_flat = (adv_b - jnp.mean(adv_b)) / (jnp.std(adv_b) + 1e-8)
            adv_b_reshaped = adv_flat.reshape((batch_size,))
            ret_b_reshaped = ret_b.flatten().reshape((batch_size,))
            adv_b_reshaped = adv_b.reshape((batch_size,))
            ret_b_reshaped = ret_b.reshape((batch_size,))

            update_state = (state, r)

            def epoch_loop(update_carry, _):
                ts, r = update_carry
                r, epoch_rng = jax.random.split(r)
                perm = jax.random.permutation(epoch_rng, batch_size)

                def minibatch_loop(ts_inner, i):
                    start = i * minibatch_size
                    s_batch = jax.tree_util.tree_map(
                        lambda x: jax.lax.dynamic_slice_in_dim(x[perm], start, minibatch_size),
                        (obs_b_reshaped, verb_b_reshaped, dir_b_reshaped, tgt_b_reshaped,
                         logp_v_reshaped, logp_d_reshaped, logp_t_reshaped,
                         adv_b_reshaped, ret_b_reshaped)
                    )

                    (loss, aux), grads = jax.value_and_grad(masked_ppo_loss_multih, has_aux=True)(
                        ts_inner.params, ts_inner.apply_fn, s_batch[0],
                        s_batch[1], s_batch[2], s_batch[3],
                        s_batch[4], s_batch[5], s_batch[6],
                        s_batch[7], s_batch[8],
                        loss_mask_v,
                        loss_mask_d,
                        loss_mask_t,
                        clip_eps=config["CLIP_EPS"],
                        ent_coeff=config["ENTROPY_COEFF"],
                        vf_coeff=config["VF_COEFF"]
                    )

                    return ts_inner.apply_gradients(grads=grads), aux

                ts, iter_aux = jax.lax.scan(minibatch_loop, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, r), iter_aux

            (state, final_rng), aux_b = jax.lax.scan(epoch_loop, update_state, None, length=config["UPDATE_EPOCHS"])
            mean_aux = jax.tree_util.tree_map(jnp.mean, aux_b)

            return state, next_env_state, next_obs, final_rng, rms, mean_aux, jnp.mean(rew_b)

        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size

        curr_state = self.state
        for i in range(1, num_iters + 1):
            curr_state, env_state, obs_flat, rng, self.rms, mean_aux, rew_scalar = train_iteration(
                curr_state, env_state, obs_flat, rng, self.rms
            )
            global_step = i * batch_size

            if i % config["LOG_INTERVAL"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-6)
                mean_reward_val = float(rew_scalar)
                print(f"Iter {i:4d} | Step {global_step:8d} | Reward {mean_reward_val:7.2f} | SPS {sps:,.0f}")
                logger.log(global_step, {"train/mean_reward": mean_reward_val, "train/sps": sps,
                    **{k: float(v) for k, v in mean_aux.items()}})

            if i % config["CKPT_INTERVAL"] == 0:
                self.save(f"{ckpt_dir}/ckpt_{global_step}.pkl")

        self.state = curr_state
        logger.close()

    def save(self, path: str):
        save_checkpoint(path, self.state, 0)

    def load(self, path: str):
        self.state = load_checkpoint(path, self.state)
