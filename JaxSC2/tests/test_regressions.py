"""Behavioral regressions for targeting, projectile damage, AI, and spatial JIT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.mechanics import apply_high_fidelity_combat


def combat(types, teams, targets, timers, projectiles=None):
    env = JaxSC2Env("V1_Base")
    if projectiles is None:
        projectiles = (
            jnp.zeros((32, 2)), jnp.zeros((32, 2)), jnp.full(32, -1),
            jnp.zeros(32), jnp.zeros(32, dtype=bool), jnp.zeros(32, dtype=jnp.int32),
        )
    return apply_high_fidelity_combat(
        jnp.full(2, 200.), jnp.ones(2, dtype=bool), jnp.array([[0., 0.], [1., 0.]]),
        jnp.array(types), jnp.array(teams), jnp.zeros(2), jnp.array(timers),
        jnp.array(targets), projectiles, env.combat_cfg,
    )


def test_manual_target_is_enemy_index_and_cannot_hurt_allies():
    env = JaxSC2Env("V1_Base")
    _, state = env.reset(jax.random.PRNGKey(0))
    # Ally zero can hit enemy zero; the other units are well separated.
    positions = jnp.array([
        [2., 2.], [2., 6.], [2., 10.], [2., 14.], [2., 18.],
        [3.3, 2.], [28., 8.], [28., 14.],
    ])
    state = state.replace(
        smax_state=state.smax_state.replace(unit_positions=positions),
        attack_timers=state.attack_timers.at[5:].set(3),
    )
    action = CentralAction(jnp.array([True, False, False, False, False]), 2, 0, 0)
    _, next_state, _, _, _ = jax.jit(env.step)(jax.random.PRNGKey(1), state, action)
    assert int(next_state.persistent_targets[0]) == env.num_allies
    assert float(next_state.smax_state.unit_health[0]) == 100.
    assert next_state.smax_state.unit_health[env.num_allies] < 100.
    # Defense in depth: even an erroneous friendly target must not cause damage.
    hp, *_ = combat([0, 0], [0, 0], [1, -1], [0, 0])
    np.testing.assert_array_equal(hp, [200., 200.])


def test_tank_fires_projectile_and_deals_damage():
    hp, _, _, projectiles, _ = combat([2, 0], [0, 1], [1, -1], [2, 0])
    assert int(projectiles[4].sum()) == 1
    np.testing.assert_array_equal(hp, [200., 200.])
    hp, *_ = combat([2, 0], [0, 1], [1, -1], [3, 0], projectiles)
    assert float(hp[1]) < 200.


def test_reusing_projectile_slot_preserves_incoming_damage():
    projectiles = (
        jnp.zeros((32, 2)).at[0].set(jnp.array([-.5, 0.])),
        jnp.zeros((32, 2)).at[0].set(jnp.array([1.5, 0.])),
        jnp.full(32, -1).at[0].set(1), jnp.zeros(32).at[0].set(99.),
        jnp.zeros(32, dtype=bool).at[0].set(True), jnp.zeros(32, dtype=jnp.int32),
    )
    hp, _, _, projectiles, _ = combat([1, 0], [0, 1], [1, -1], [1, 0], projectiles)
    assert float(hp[1]) == 101.
    assert bool(projectiles[4][0])  # A new shot now occupies the same slot.
    assert float(projectiles[3][0]) == 5.  # Base damage; armor is applied on impact.


def test_attack_selection_obeys_latency():
    env = JaxSC2Env("V1_Base", latency_delay=3)
    _, state = env.reset(jax.random.PRNGKey(0))
    positions = jnp.array([
        [2., 2.], [2., 6.], [2., 10.], [2., 14.], [2., 18.],
        [3.3, 2.], [28., 8.], [28., 14.],
    ])
    state = state.replace(smax_state=state.smax_state.replace(unit_positions=positions))
    noop = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    attack = CentralAction(jnp.array([True, False, False, False, False]), 2, 0, 2)
    step = jax.jit(env.step)
    _, state, *_ = step(jax.random.PRNGKey(1), state, attack)
    assert state.persistent_targets[0] != 7
    # Keep the manually selected enemy visible to allow the delayed command.
    state = state.replace(enemy_visible=jnp.ones(3, dtype=bool))
    _, state, *_ = step(jax.random.PRNGKey(2), state, noop)
    state = state.replace(enemy_visible=jnp.ones(3, dtype=bool))
    _, state, *_ = step(jax.random.PRNGKey(3), state, noop)
    assert int(state.persistent_targets[0]) == 7
    assert not bool(state.action_buffer.who_masks[0, 1])


@pytest.mark.parametrize("mode", ["aggressive", "patrol"])
def test_enemy_movement_modes_move_units(mode):
    env = JaxSC2Env("V1_Base", enemy_ai=True, enemy_mode=mode)
    _, state = env.reset(jax.random.PRNGKey(0))
    positions = jnp.array([
        [2., 2.], [2., 6.], [2., 10.], [2., 14.], [2., 18.],
        [28., 2.], [28., 8.], [28., 14.],
    ])
    state = state.replace(smax_state=state.smax_state.replace(unit_positions=positions))
    noop = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    _, next_state, *_ = jax.jit(env.step)(jax.random.PRNGKey(1), state, noop)
    assert jnp.all(jnp.linalg.norm(next_state.unit_velocities[5:], axis=-1) > 0)
    if mode == "aggressive":
        assert jnp.all(next_state.smax_state.unit_positions[5:, 0] < positions[5:, 0])


def test_spatial_reset_and_step_support_jit_and_vmap():
    env = JaxSC2Env("V1_Base", use_spatial_obs=True, resolution=16)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    assert obs["screen"].shape == (2, 17, 16, 16)
    assert obs["minimap"].shape == (2, 7, 16, 16)
    assert jnp.any(obs["minimap"][:, 3] > 0)  # Enemy player-relative category.
    action = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    obs, *_ = jax.jit(jax.vmap(lambda key, state: env.step(key, state, action)))(keys, states)
    assert obs["vector"].shape == (2, 63)
    assert jnp.all(jnp.isfinite(obs["screen"]))
