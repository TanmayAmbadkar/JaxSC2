"""Terrain must contain full unit footprints, swept paths, and both armies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.maps.twobridge import TwoBridgeMap
from JaxSC2.env.env import JaxSC2Env, CentralAction, PerUnitAction
from JaxSC2.scripted.agents import bridge_waypoints


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("start,desired", [
    ([16.8, 8.9], [16.8, 9.8]),   # Leaving a bridge sideways.
    ([16.8, 24.9], [16.8, 25.8]), # The reported upper-bridge failure.
    ([13.7, 16.], [14.6, 16.]),   # Entering the ditch's near edge.
    ([18.3, 16.], [17.4, 16.]),   # Entering from the opposite bank.
    ([13., 16.], [19., 16.]),     # Large movement cannot tunnel across.
    ([13.7, 9.2], [14.4, 8.9]),  # Diagonal corner contact.
])
def test_ditch_blocks_radius_and_swept_motion_for_both_teams(team, start, desired):
    terrain = TwoBridgeMap()
    start, desired = jnp.array([start]), jnp.array([desired])
    result = jax.jit(terrain.enforce_constraints)(desired, start, 32, 32, jnp.array([team]))
    assert jnp.all(terrain.is_walkable(result, 32, 32))
    assert not jnp.allclose(result, desired)
    # The renderer may interpolate frames; the resulting segment must also be safe.
    samples = start + jnp.linspace(0., 1., 32)[:, None, None] * (result - start)
    assert jnp.all(terrain.is_walkable(samples, 32, 32))


@pytest.mark.parametrize("bridge_y", [8., 24.])
@pytest.mark.parametrize("team", [0, 1])
def test_bridge_permits_crossing_in_both_directions(bridge_y, team):
    terrain = TwoBridgeMap()
    start = jnp.array([[13., bridge_y], [19., bridge_y]])
    desired = start[::-1]
    result = terrain.enforce_constraints(desired, start, 32, 32, jnp.full(2, team))
    np.testing.assert_allclose(result, desired)


def test_wall_contact_slides_without_sticking():
    terrain = TwoBridgeMap()
    result = terrain.enforce_constraints(
        jnp.array([[14.2, 18.]]), jnp.array([[13.7, 16.]]), 32, 32, jnp.array([0]))
    assert float(result[0, 0]) <= 13.8 + 1e-5
    np.testing.assert_allclose(result[0, 1], 18.)
    assert jnp.all(terrain.is_walkable(result, 32, 32))


def test_invalid_old_state_recovers_to_walkable_ground():
    terrain = TwoBridgeMap()
    illegal = jnp.array([[16., 16.], [16.8, 25.8]])
    result = terrain.enforce_constraints(illegal, illegal, 32, 32, jnp.array([0, 1]))
    assert jnp.all(terrain.is_walkable(result, 32, 32))


def test_route_waits_until_whole_unit_exits_bridge():
    env = JaxSC2Env("V1_Base")
    positions = jnp.array([[16.8, 8.], [15.2, 24.], [19., 8.]])
    goals = jnp.array([[28., 20.], [4., 10.], [28., 20.]])
    waypoints = bridge_waypoints(env, positions, goals)
    assert float(waypoints[0, 0]) > 17.6 + env.unit_radius
    assert float(waypoints[0, 1]) == 8.
    assert float(waypoints[1, 0]) < 14.4 - env.unit_radius
    assert float(waypoints[1, 1]) == 24.
    np.testing.assert_array_equal(waypoints[2], goals[2])


def test_collision_push_cannot_put_unit_in_ditch():
    env = JaxSC2Env("V1_Base")
    _, state = env.reset(jax.random.PRNGKey(0))
    position = state.smax_state.unit_positions.at[0].set(jnp.array([16., 24.99]))
    position = position.at[1].set(jnp.array([16., 24.1]))
    state = state.replace(smax_state=state.smax_state.replace(unit_positions=position))
    _, state, *_ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1))
    assert jnp.all(env.map_instance.is_walkable(state.smax_state.unit_positions, 32, 32))
    assert float(state.smax_state.unit_positions[0, 1]) <= 25. + 1e-5
    assert float(state.unit_velocities[0, 1]) == 0.


def test_random_actions_keep_both_teams_on_terrain_under_vmap():
    env = JaxSC2Env("V3_Base", enemy_ai=True, enemy_mode="aggressive", latency_delay=3)

    def episode(key):
        _, state = env.reset(key)
        # Stress contact resolution with a crowded bridge and opposing banks.
        x = jnp.where(jnp.arange(env.num_units) % 2 == 0, 13.7, 18.3)
        y = jnp.linspace(7.1, 8.9, env.num_units)
        state = state.replace(smax_state=state.smax_state.replace(unit_positions=jnp.stack([x, y], -1)))

        def step(carry, _):
            state, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            direction = jax.random.randint(action_key, (env.num_allies,), 0, 8)
            action = PerUnitAction(jnp.ones(5, dtype=bool), jnp.ones(5, dtype=jnp.int32),
                                   direction, jnp.full(5, -1, dtype=jnp.int32))
            _, state, *_ = env.step(step_key, state, action)
            valid = env.map_instance.is_walkable(state.smax_state.unit_positions, 32, 32)
            return (state, key), jnp.all(valid)

        return jax.lax.scan(step, (state, key), None, length=100)[1]

    assert jnp.all(jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(7), 8)))


def test_spatial_features_use_shared_terrain_footprint():
    env = JaxSC2Env("V1_Base", use_spatial_obs=True, resolution=32)
    obs, _ = jax.jit(env.reset)(jax.random.PRNGKey(0))
    x, y = jnp.meshgrid(jnp.linspace(0, 32, 32), jnp.linspace(0, 32, 32), indexing="xy")
    expected = env.map_instance.terrain_mask(jnp.stack([x, y], -1), 32, 32)
    np.testing.assert_array_equal(obs["minimap"][0], expected)
