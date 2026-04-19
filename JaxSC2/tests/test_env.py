import jax
import jax.numpy as jnp
import pytest
from JaxSC2.env.env import JaxSC2Env, CentralAction

def test_env_reset():
    env = JaxSC2Env(variant_name="V1_Base")
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    
    assert state.smax_state.unit_positions.shape == (env.num_allies + env.num_enemies, 2)
    assert jnp.all(state.smax_state.unit_alive)
    assert state.timestep == 0

def test_env_step():
    env = JaxSC2Env(variant_name="V1_Base")
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    
    # Simple no-op action
    action = CentralAction(
        who_mask=jnp.ones(env.num_allies, dtype=jnp.bool_),
        verb=0,
        direction=0,
        target=0
    )
    
    rng, step_rng = jax.random.split(rng)
    obs, next_state, reward, done, info = env.step(step_rng, state, action)
    
    assert next_state.timestep == 1
    assert not done
    assert "win" in info

@pytest.mark.parametrize("variant", ["V1_Base", "V2_Base", "V3_Base"])
def test_all_variants(variant):
    env = JaxSC2Env(variant_name=variant)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert state.smax_state.unit_positions.shape[0] == env.num_allies + env.num_enemies
