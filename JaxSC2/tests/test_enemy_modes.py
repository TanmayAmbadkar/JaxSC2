"""Distinct enemy tactics, safe targets, persistent state, and compiled rollouts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2.env.env import JaxSC2Env, CentralAction
from JaxSC2.env.enemy_ai import ENEMY_MODES, COMBAT_ENEMY_MODES
from JaxSC2.scripted.evaluate import evaluate_case


def scenario(mode):
    env = JaxSC2Env("V2_Base", enemy_ai=True, enemy_mode=mode, enable_collision=False)
    _, state = env.reset(jax.random.PRNGKey(42))
    allies = jnp.array([[4., 6.], [4., 8.], [4., 10.], [4., 12.], [4., 14.]])
    enemies = jnp.array([[24., 8.], [24., 10.], [25., 12.], [26., 14.], [26., 16.]])
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=jnp.concatenate([allies, enemies])), enemy_home=enemies)
    return env, state


def direction_vectors(orders):
    angle = orders.direction * jnp.pi / 4
    return jnp.stack([jnp.sin(angle), jnp.cos(angle)], axis=-1)


def test_unknown_enemy_mode_does_not_silently_become_guard():
    with pytest.raises(ValueError, match="Unknown enemy mode"):
        JaxSC2Env(enemy_mode="typo")


@pytest.mark.parametrize("mode", ENEMY_MODES)
def test_all_modes_support_compiled_batched_rollouts_with_fog_and_safe_terrain(mode):
    env = JaxSC2Env("V2_Base", enemy_ai=True, enemy_mode=mode, enable_fow=True)
    env.max_steps = 45
    rows, _ = evaluate_case(env, seed=42)
    assert len(rows) == 2
    assert all(row["units_in_bounds"] and row["terrain_valid"] for row in rows)
    assert all(np.isfinite(row["reward"]) for row in rows)


@pytest.mark.parametrize("mode", COMBAT_ENEMY_MODES)
def test_new_orders_are_deterministic_and_never_target_dead_allies(mode):
    env, state = scenario(mode)
    s = state.smax_state
    state = state.replace(smax_state=s.replace(unit_alive=s.unit_alive.at[0].set(False)))
    get_orders = jax.jit(env.get_enemy_orders)
    first, second = get_orders(state), get_orders(state)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert jnp.all((first.target == -1) | ((first.target > 0) & (first.target < env.num_allies)))
    assert jnp.all((first.direction >= 0) & (first.direction < 8))
    state = state.replace(smax_state=state.smax_state.replace(
        unit_alive=state.smax_state.unit_alive.at[:env.num_allies].set(False)))
    empty = get_orders(state)
    assert jnp.all(empty.target == -1) and jnp.all(empty.verb == 0)
    state = state.replace(smax_state=state.smax_state.replace(
        unit_alive=jnp.zeros(env.num_units, dtype=bool)))
    dead = get_orders(state)
    assert jnp.all(dead.verb == 0) and not jnp.any(dead.alerted)


def test_beacon_defender_moves_toward_beacon_not_distant_allies():
    env, state = scenario("beacon_defender")
    state = state.replace(beacon_pos=jnp.array([26., 25.]))
    orders = env.get_enemy_orders(state)
    assert jnp.all(orders.verb == 1)
    vectors = direction_vectors(orders)
    assert jnp.all(vectors[:, 1] > 0.)


def test_bridge_blocker_holds_both_exits_instead_of_chasing():
    env, state = scenario("bridge_blocker")
    posts = jnp.array([[19., 8.], [19., 24.], [20.3, 8.], [20.3, 24.], [21.6, 8.]])
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[5:].set(posts)))
    orders = env.get_enemy_orders(state)
    assert jnp.all(orders.verb == 0)
    assert jnp.all(env.map_instance.is_walkable(posts, 32, 32, env.unit_radius))


def test_pursuer_selects_exposed_individual_not_army_centroid():
    env, state = scenario("pursuer")
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[0].set(jnp.array([21., 8.]))))
    orders = env.get_enemy_orders(state)
    assert jnp.all(orders.target == 0)


def test_kiter_retreats_during_cooldown_but_melee_closes_distance():
    env, state = scenario("kiter")
    positions = state.smax_state.unit_positions
    positions = positions.at[:5].set(positions[5:] - jnp.array([5., 0.]))
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=positions, unit_types=state.smax_state.unit_types.at[5:].set(1)))
    ready = env.get_enemy_orders(state)
    assert int(ready.verb[0]) == 0  # Shoot, don't always run away.
    cooldown = state.replace(attack_timers=state.attack_timers.at[5:].set(4))
    retreat = env.get_enemy_orders(cooldown)
    assert int(retreat.verb[0]) == 1 and float(direction_vectors(retreat)[0, 0]) > 0.
    melee = cooldown.replace(smax_state=cooldown.smax_state.replace(
        unit_types=cooldown.smax_state.unit_types.at[5:].set(0)))
    approach = env.get_enemy_orders(melee)
    assert int(approach.verb[0]) == 1 and float(direction_vectors(approach)[0, 0]) < 0.


def test_focus_fire_coordinates_and_retargets_after_death():
    env, state = scenario("focus_fire")
    positions = state.smax_state.unit_positions.at[:5, 0].set(22.)
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=positions, unit_health=state.smax_state.unit_health.at[2].set(1.)))
    orders = env.get_enemy_orders(state)
    assert jnp.all(orders.target == 2)
    state = state.replace(smax_state=state.smax_state.replace(
        unit_alive=state.smax_state.unit_alive.at[2].set(False)))
    orders = env.get_enemy_orders(state)
    assert len(set(orders.target.tolist())) == 1 and int(orders.target[0]) != 2
    # The simulator, not just the controller, must apply the coordinated targets.
    noop = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    _, next_state, *_ = jax.jit(env.step)(jax.random.PRNGKey(0), state, noop)
    np.testing.assert_array_equal(next_state.persistent_targets[5:], orders.target)


def test_ambush_trigger_latches_and_reset_clears_it():
    env, state = scenario("ambusher")
    orders = env.get_enemy_orders(state)
    assert not jnp.any(orders.alerted) and jnp.all(orders.target == -1)
    nearby = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[0].set(jnp.array([22., 8.]))))
    triggered = env.get_enemy_orders(nearby)
    assert jnp.all(triggered.alerted) and jnp.all(triggered.target >= 0)
    later = env.get_enemy_orders(state.replace(enemy_alerted=triggered.alerted))
    assert jnp.all(later.alerted)
    _, reset = env.reset(jax.random.PRNGKey(0))
    assert not jnp.any(reset.enemy_alerted)
    np.testing.assert_array_equal(reset.enemy_home, reset.smax_state.unit_positions[5:])


def test_untriggered_ambusher_suppresses_automatic_target_acquisition():
    env, state = scenario("ambusher")
    env.vision_radius = .1  # Allow an in-weapon-range ally without triggering.
    state = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[0].set(jnp.array([23., 8.]))))
    noop = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    _, next_state, *_ = jax.jit(env.step)(jax.random.PRNGKey(0), state, noop)
    assert jnp.all(next_state.persistent_targets[5:] == -1)
    assert not jnp.any(next_state.enemy_alerted)


def test_adaptive_defender_switches_to_threatened_bridge():
    env, state = scenario("adaptive_defender")
    positions = state.smax_state.unit_positions.at[5:, 0].set(24.).at[5:, 1].set(16.)
    lower = state.replace(smax_state=state.smax_state.replace(
        unit_positions=positions.at[:5, 1].set(8.)))
    upper = state.replace(smax_state=state.smax_state.replace(
        unit_positions=positions.at[:5, 1].set(24.)))
    assert jnp.all(direction_vectors(env.get_enemy_orders(lower))[:, 1] < 0.)
    assert jnp.all(direction_vectors(env.get_enemy_orders(upper))[:, 1] > 0.)


def test_disabled_ai_keeps_movement_off_for_new_modes():
    env, state = scenario("pursuer")
    env.enemy_ai = False
    noop = CentralAction(jnp.ones(5, dtype=bool), 0, 0, -1)
    _, next_state, *_ = jax.jit(env.step)(jax.random.PRNGKey(0), state, noop)
    np.testing.assert_array_equal(next_state.smax_state.unit_positions[5:], state.smax_state.unit_positions[5:])
