import jax
import jax.numpy as jnp
import optax
import time
from flax.training.train_state import TrainState

from JaxSC2.env.env import JaxSC2Env, PerUnitAction
from algorithms.ppo.model import MultiHeadActorCritic 
from algorithms.a2c.a2c_logic import a2c_loss
from algorithms.common.utils import compute_gae, flatten_obs, RunningMeanStd, update_rms
from algorithms.common.logging import Logger
from algorithms.common.checkpoint import save_checkpoint, load_checkpoint
from algorithms.ppo.eval import evaluate

class A2CTrainer:
    """Advantage Actor-Critic (A2C) trainer with multi-head actions.

    This implementation uses per-unit action spaces (verb, direction, target)
    with separate policy heads for each component.
    """
    def __init__(self, config=None):
        self.default_config = {
            "NUM_ENVS": 64,
            "ROLLOUT_LEN": 128,
            "LR": 7e-4, 
            "GAMMA": 0.99,
            "GAE_LAMBDA": 1.0,
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
        config = self.config
        num_envs = config["NUM_ENVS"]
        rollout_len = config["ROLLOUT_LEN"]
        batch_size = num_envs * rollout_len

        print("="*50)
        print(f"JaxSC2 A2C Trainer | Variant: {config['VARIANT_NAME']}")
        print("="*50)
        
        env = JaxSC2Env(variant_name=config["VARIANT_NAME"], use_spatial_obs=False)
        logger = Logger(f"runs/{config['VARIANT_NAME']}_a2c/logs")
        ckpt_dir = f"runs/{config['VARIANT_NAME']}_a2c/checkpoints"
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        num_enemies = env.num_enemies
        dummy_obs = jnp.zeros((1, 63))
        model = MultiHeadActorCritic(action_dim=num_enemies)
        params = model.init(init_rng, dummy_obs)
        
        tx = optax.adam(config["LR"])
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        rms = RunningMeanStd(mean=jnp.zeros(1), var=jnp.ones(1), count=jnp.array(1e-4))

        def rollout_step(carry, _):
            params, env_state, obs, rng = carry
            output = model.apply(params, obs)
            verb_logits = output["verb_logits"]
            direction_logits = output["direction_logits"]
            target_logits = output["target_logits"]
            value = output["value"]
            
            rng, sub_rng = jax.random.split(rng)
            
            verb_idx = jax.random.categorical(sub_rng, verb_logits)
            direction_idx = jax.random.categorical(jax.random.split(sub_rng, num_envs), direction_logits)
            target_idx = jax.random.categorical(jax.random.split(sub_rng, num_envs), target_logits)
            
            def single_env_step(s, r, v_idx, d_idx, t_idx):
                act = PerUnitAction(
                    who_mask=jnp.ones((env.num_allies,), dtype=jnp.bool_),
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
            
            transition = (obs, verb_idx, direction_idx, target_idx, value, reward, done)
            return (params, next_env_state, next_obs_flat, rng), transition

        @jax.jit
        def train_iteration(state, env_state, obs, rng, rms):
            carry = (state.params, env_state, obs, rng)
            (params, next_env_state, next_obs, next_rng), transitions = jax.lax.scan(
                rollout_step, carry, None, length=rollout_len
            )
            obs_b, verb_b, dir_b, tgt_b, val_b, rew_b, done_b = transitions
            
            rms = update_rms(rms, rew_b.flatten())
            norm_rew_b = rew_b / jnp.sqrt(rms.var + 1e-8)
            mean_reward = jnp.mean(rew_b)
            
            _, last_val = model.apply(params, next_obs) 
            adv_b, ret_b = compute_gae(norm_rew_b, val_b, done_b, config["GAMMA"], config["GAE_LAMBDA"], last_val)
            
            batch = (obs_b, adv_b, ret_b)
            f_batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

            update_state = (state, next_rng)
            def epoch_loop(update_carry, _):
                ts, r = update_carry
                r, perm_rng = jax.random.split(r)
                permutation = jax.random.permutation(perm_rng, batch_size)
                s_batch = jax.tree_util.tree_map(lambda x: x[permutation], f_batch)
                
                def minibatch_loop(ts_inner, i):
                    start = i * rollout_len
                    mb = jax.tree_util.tree_map(lambda x: x[start:start + rollout_len], s_batch)
                    return update_a2c_step(ts_inner, mb), None
                
                ts, _ = jax.lax.scan(minibatch_loop, ts, jnp.arange(1))
                return (ts, r), None

            (state, final_rng), _ = jax.lax.scan(epoch_loop, update_state, None, length=1)
            return state, next_env_state, next_obs, final_rng, rms, mean_reward

        def update_a2c_step(train_state, batch):
            obs, adv, ret = batch
            (loss, aux), grads = jax.value_and_grad(a2c_loss, has_aux=True)(
                train_state.params, train_state.apply_fn, obs, adv, ret, 
                ent_coeff=config["ENTROPY_COEFF"], 
                vf_coeff=config["VF_COEFF"]
            )
            return train_state.apply_gradients(grads=grads), aux

        # Training Loop
        rng, start_rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(start_rng, num_envs))
        obs_flat = flatten_obs(obs)
        start_time = time.time()
        num_iters = total_steps // batch_size
        
        curr_state = state
        for i in range(1, num_iters + 1):
            curr_state, env_state, obs_flat, rng, rms, mean_reward = train_iteration(
                curr_state, env_state, obs_flat, rng, rms
            )
            global_step = i * batch_size
            
            if i % config["LOG_INTERVAL"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                print(
                    f"Iter {i:4d} | Step {global_step:8d} | "
                    f"Reward {float(mean_reward):7.2f} | "
                    f"SPS {sps:,.0f}",
                    flush=True
                )
                
            if i % config["EVAL_INTERVAL"] == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = evaluate(env, curr_state.params, model, eval_rng)
                print(f">>> EVAL: Reward {eval_metrics['eval/mean_reward']:.2f} | Win (Nav: {eval_metrics['eval/nav_win_rate']:.1%}, Combat: {eval_metrics['eval/combat_win_rate']:.1%}, Total: {eval_metrics['eval/total_win_rate']:.1%})", flush=True)
                
            if i % config["CKPT_INTERVAL"] == 0:
                save_checkpoint(f"{ckpt_dir}/ckpt_{global_step}.pkl", curr_state, global_step)

        logger.close()

if __name__ == "__main__":
    trainer = A2CTrainer()
    trainer.train(100000)
