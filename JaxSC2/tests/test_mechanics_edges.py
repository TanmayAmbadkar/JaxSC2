"""Regressions for cooldowns, projectile interception, toggles, fog, and bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.env.env import JaxSC2Env, JaxSC2GymEnv, CentralAction
from JaxSC2.env.mechanics import apply_high_fidelity_combat, update_projectiles


def projectile_state():
    return (jnp.zeros((32, 2)), jnp.zeros((32, 2)), jnp.full(32, -1),
            jnp.zeros(32), jnp.zeros(32, dtype=bool), jnp.zeros(32, dtype=jnp.int32),
            jnp.full(32, -1))


def separated_state(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    positions = jnp.array([
        [2., 2.], [2., 6.], [2., 10.], [2., 14.], [2., 18.],
        [28., 2.], [28., 8.], [28., 14.],
    ])
    return state.replace(smax_state=state.smax_state.replace(unit_positions=positions))


def test_swept_projectile_hits_first_enemy_along_segment():
    result = jax.jit(update_projectiles)(
        jnp.array([[0., 0.]]), jnp.array([[2., 0.]]), jnp.array([0]),
        jnp.array([True]), jnp.array([0]),
        jnp.array([[1.5, 0.], [.75, 0.]]), jnp.ones(2, dtype=bool), jnp.ones(2, dtype=jnp.int32),
    )
    assert bool(result[2][0])
    assert int(result[3][0]) == 1
    assert not bool(result[1][0])


def test_cooldown_continues_when_target_leaves_range():
    env = JaxSC2Env("V1_Base")
    def step(health, timers, projectiles, x):
        return apply_high_fidelity_combat(
            health, jnp.ones(2, dtype=bool), jnp.array([[0., 0.], [x, 0.]]),
            jnp.zeros(2, dtype=jnp.int32), jnp.array([0, 1]), jnp.zeros(2),
            timers, jnp.array([1, -1]), projectiles, env.combat_cfg,
        )
    step = jax.jit(step)
    health, _, timers, projectiles, _ = step(jnp.full(2, 200.), jnp.zeros(2, dtype=jnp.int32), projectile_state(), 1.)
    assert float(health[1]) == 192.
    for x in (4., 1.):
        health, _, timers, projectiles, _ = step(health, timers, projectiles, x)
    assert float(health[1]) == 192.
    assert int(timers[0]) == 3


def test_intercepted_shot_uses_actual_targets_type_and_armor():
    env = JaxSC2Env("V1_Base")
    types = jnp.array([1, 2, 0])
    health, timers, projectiles = jnp.array([45., 200., 100.]), jnp.array([1, 0, 0]), projectile_state()
    for _ in range(2):
        health, _, timers, projectiles, _ = apply_high_fidelity_combat(
            health, jnp.ones(3, dtype=bool), jnp.array([[0., 0.], [3., 0.], [1.5, 0.]]),
            types, jnp.array([0, 1, 1]), env.unit_armor[types], timers,
            jnp.array([1, -1, -1]), projectiles, env.combat_cfg,
        )
    assert float(health[1]) == 200.
    assert float(health[2]) == 97.


def test_collision_flag_changes_overlap_resolution():
    outputs = []
    for enabled in (False, True):
        env = JaxSC2Env("V1_Base", enable_collision=enabled)
        state = separated_state(env)
        positions = state.smax_state.unit_positions.at[1].set(jnp.array([2.5, 2.]))
        state = state.replace(smax_state=state.smax_state.replace(unit_positions=positions))
        _, next_state, *_ = jax.jit(env.step)(
            jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1))
        outputs.append(next_state.smax_state.unit_positions)
    np.testing.assert_array_equal(outputs[0][0], [2., 2.])
    assert float(outputs[1][0, 0]) < 2.


def test_armor_flag_changes_received_damage():
    health = []
    for enabled in (False, True):
        env = JaxSC2Env("V1_Base", enable_collision=False, enable_armor=enabled)
        state = separated_state(env)
        positions = state.smax_state.unit_positions.at[5].set(jnp.array([3.3, 2.]))
        state = state.replace(smax_state=state.smax_state.replace(unit_positions=positions))
        _, state, *_ = jax.jit(env.step)(
            jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1))
        health.append(float(state.smax_state.unit_health[0]))
    assert health == [92., 93.]


def test_fog_reset_does_not_reveal_unseen_enemies():
    env = JaxSC2Env("V1_Base", enable_fow=True)
    obs, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert not jnp.any(state.enemy_visible)
    np.testing.assert_array_equal(state.fog_state.last_seen_hp, 0.)
    np.testing.assert_array_equal(state.fog_state.last_seen_pos, 0.)
    np.testing.assert_array_equal(obs[20:52], 0.)


def test_disabled_fog_reports_current_enemy_state():
    env = JaxSC2Env("V1_Base", enable_fow=False, enemy_ai=True, enemy_mode="patrol")
    state = separated_state(env)
    _, state, *_ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1))
    assert jnp.all(state.enemy_visible)
    np.testing.assert_array_equal(state.fog_state.last_seen_pos, state.smax_state.unit_positions[5:])


def test_dead_ally_does_not_provide_vision():
    env = JaxSC2Env("V1_Base", enable_fow=True)
    state = separated_state(env)
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[0].set(jnp.array([27., 2.])),
        unit_alive=state.smax_state.unit_alive.at[0].set(False),
        unit_health=state.smax_state.unit_health.at[0].set(0.),
    ))
    _, state, *_ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1))
    assert not jnp.any(state.enemy_visible)
    assert not jnp.any(env.get_action_mask(state)["target"])


@pytest.mark.parametrize("position,direction", [([31.99, 2.], 2), ([.01, 2.], 6),
                                                ([2., 31.99], 0), ([2., .01], 4)])
def test_units_cannot_move_outside_map(position, direction):
    env = JaxSC2Env("V1_Base", enable_collision=False)
    state = separated_state(env)
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[0].set(jnp.array(position))))
    _, state, *_ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 1, direction, -1))
    assert jnp.all(state.smax_state.unit_positions >= 0.)
    assert jnp.all(state.smax_state.unit_positions <= 32.)


def test_gym_can_select_all_allies_and_reports_truncation():
    wrapper = JaxSC2GymEnv("V1_Base", enable_collision=False)
    obs, _ = wrapper.reset(seed=0)
    assert isinstance(obs, np.ndarray) and wrapper.observation_space.contains(obs)
    assert wrapper.action_space.contains(np.array([31, 1, 2, 2]))
    wrapper._state = separated_state(wrapper.env)
    wrapper.env.max_steps = 1
    obs, _, terminated, truncated, _ = wrapper.step(np.array([31, 1, 2, 2]))
    assert jnp.all(wrapper._state.unit_velocities[:5, 0] > 0)
    assert not terminated and truncated
    assert wrapper.observation_space.contains(obs)
    from gymnasium.utils.env_checker import check_env
    check_env(JaxSC2GymEnv("V1_Base"), skip_render_check=True)
