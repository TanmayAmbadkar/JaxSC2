"""Task semantics, general entity capacities, and economy conservation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxSC2 import make, available_tasks
from JaxSC2.tasks.types import TASKS
from JaxSC2.tasks.scripted import scripted_action
from JaxSC2.tasks.economy import prepare, advance_economy, macro_mask
from JaxSC2.maps.base import ArenaMap


@pytest.mark.parametrize("name", TASKS)
def test_every_task_supports_jit_vmap_and_fixed_shape_observations(name):
    env = make(name, max_steps=2)
    keys = jax.random.split(jax.random.PRNGKey(42), 2)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    assert obs["vector"].shape == (2,) + env.observation_space.shape
    action = env.no_op()
    step = jax.jit(jax.vmap(lambda k, s: env.step(k, s, action)))
    _, states, rewards, done, _ = step(keys, states)
    assert not jnp.any(done) and jnp.all(jnp.isfinite(rewards))
    _, states, _, done, _ = step(keys, states)
    assert jnp.all(done)
    old = states
    _, states, rewards, done, _ = step(keys, states)
    assert jnp.all(rewards == 0) and jnp.all(done)
    for before, after in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(states)):
        np.testing.assert_array_equal(before, after)


def test_factory_validates_tasks_capacities_and_horizons():
    assert len(available_tasks()) == 16
    with pytest.raises(ValueError, match="Unknown task"):
        make("typo")
    with pytest.raises(ValueError):
        make("BuildMarines", ally_capacity=2)
    with pytest.raises(ValueError):
        make("MoveToBeacon", max_steps=0)
    legacy = make("V2_Base")
    assert legacy.observation_space.shape == (63,)
    larger = make("BuildMarines", ally_capacity=64)
    obs, state = larger.reset(jax.random.PRNGKey(0))
    assert obs["units"].shape == (65, 7)
    assert int(jnp.sum(state.world.smax_state.unit_alive)) == 12


def test_arena_uses_generic_obstacle_count_and_radius_clearance():
    terrain = ArenaMap(obstacles=[(14., 0., 18., 20.)])
    start = jnp.array([[12., 8.], [20., 8.]])
    result = jax.jit(terrain.enforce_constraints)(start[::-1], start, 32., 32., jnp.array([0, 1]))
    assert jnp.all(terrain.is_walkable(result, 32., 32.))
    assert float(result[0, 0]) < 14. and float(result[1, 0]) > 18.


def test_beacon_relocates_and_does_not_end_episode():
    env = make("MoveToBeacon")
    _, state = env.reset(jax.random.PRNGKey(0))
    goal = state.world.smax_state.unit_positions[0]
    state = state.replace(world=state.world.replace(beacon_pos=goal))
    _, next_state, reward, done, info = jax.jit(env.step)(jax.random.PRNGKey(1), state, env.no_op())
    assert float(reward) == 1. and not bool(done) and bool(info["nav_success"])
    assert float(jnp.linalg.norm(next_state.world.beacon_pos - goal)) > 5.


def test_shard_claims_are_unique_and_respawn_without_ending():
    env = make("CollectMineralShards")
    _, state = env.reset(jax.random.PRNGKey(0))
    point = jnp.array([10., 10.])
    s = state.world.smax_state
    state = state.replace(world=state.world.replace(smax_state=s.replace(
        unit_positions=s.unit_positions.at[:2].set(point))),
        entity_positions=state.entity_positions.at[0].set(point),
        entity_amount=jnp.zeros(env.entity_capacity).at[0].set(1.))
    _, next_state, reward, done, _ = env.step(jax.random.PRNGKey(1), state, env.no_op())
    assert float(reward) == 1. and int(next_state.wave) == 1 and not bool(done)
    assert int(jnp.sum(next_state.entity_amount)) == 20


@pytest.mark.parametrize("name", ["FindAndDefeatZerglings", "DefeatRoaches", "DefeatZerglingsAndBanelings"])
def test_combat_cannot_win_by_touching_a_beacon(name):
    env = make(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    world = state.world.replace(beacon_pos=state.world.smax_state.unit_positions[0])
    _, _, reward, done, info = env.step(jax.random.PRNGKey(1), state.replace(world=world), env.no_op())
    assert not bool(done) and not bool(info["nav_success"])
    assert not bool(info["combat_success"]) and float(reward) == 0.


def test_wave_reinforcements_preserve_veteran_health_and_capacity():
    env = make("DefeatRoaches", ally_capacity=12)
    _, state = env.reset(jax.random.PRNGKey(0))
    s = state.world.smax_state
    world = state.world.replace(smax_state=s.replace(
        unit_health=s.unit_health.at[0].set(12.).at[12:].set(0.),
        unit_alive=s.unit_alive.at[12:].set(False)))
    from JaxSC2.tasks.objectives import transition
    next_state = jax.jit(lambda st, w: transition(env, jax.random.PRNGKey(1), st, w))(state, world)
    assert int(next_state.wave) == 1 and float(next_state.score) == 40.
    assert float(next_state.world.smax_state.unit_health[0]) == 12.
    assert int(jnp.sum(next_state.world.smax_state.unit_alive[:12])) == 12
    assert jnp.all(next_state.world.smax_state.unit_health[9:12] == 45.)
    assert jnp.all(next_state.world.smax_state.unit_alive[12:])


def test_baneling_explosion_damages_multiple_allies_and_consumes_attacker():
    env = make("DefeatZerglingsAndBanelings")
    _, state = env.reset(jax.random.PRNGKey(0))
    n, s = env.num_allies, state.world.smax_state
    pos = s.unit_positions.at[:2].set(jnp.array([[10., 10.], [11.5, 10.]]))
    pos = pos.at[n + 6].set(jnp.array([10.5, 10.]))
    state = state.replace(world=state.world.replace(smax_state=s.replace(unit_positions=pos)))
    _, result, _, _, _ = jax.jit(env.step)(jax.random.PRNGKey(1), state, env.no_op())
    assert not bool(result.world.smax_state.unit_alive[n + 6])
    assert jnp.all(result.world.smax_state.unit_health[:2] < 45.)


def test_search_observation_does_not_reveal_hidden_enemy_positions():
    env = make("FindAndDefeatZerglings")
    obs, state = env.reset(jax.random.PRNGKey(0))
    hidden = ~state.world.enemy_visible
    assert jnp.any(hidden)
    np.testing.assert_array_equal(obs["units"][3:][hidden], 0.)
    pos = state.world.smax_state.unit_positions.at[3:].add(jnp.where(hidden[:, None], .1, 0.))
    altered = state.replace(world=state.world.replace(smax_state=state.world.smax_state.replace(unit_positions=pos)))
    np.testing.assert_array_equal(obs["vector"], env.get_obs(altered)["vector"])


def test_invalid_macros_do_not_charge_or_instantly_create_units():
    env = make("BuildMarines")
    _, state = env.reset(jax.random.PRNGKey(0))
    # No barracks: requesting a marine must be rejected even with money.
    state = state.replace(minerals=jnp.float32(1000.))
    action = env.no_op().replace(macro=jnp.int32(6), actor=jnp.int32(0))
    prepared, _, accepted = prepare(env, state, action)
    assert not bool(accepted) and float(prepared.minerals) == 1000.
    assert jnp.all(prepared.queue_kind == -1)
    assert not bool(macro_mask(env, state)[6])
    for op in (-1, 7):
        prepared, _, accepted = prepare(env, state, action.replace(macro=jnp.int32(op)))
        assert not bool(accepted) and float(prepared.minerals) == 1000.


def test_paid_production_takes_time_and_obeys_supply():
    env = make("BuildMarines")
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(minerals=jnp.float32(100.), building_kind=state.building_kind.at[1].set(3),
                          building_positions=state.building_positions.at[1].set(jnp.array([6., 20.])))
    action = env.no_op().replace(macro=jnp.int32(6), actor=jnp.int32(1))
    prepared, _, accepted = prepare(env, state, action)
    assert bool(accepted) and float(prepared.minerals) == 50.
    assert int(prepared.marines_built) == 0 and int(prepared.queue_remaining[1]) == 18
    # Idle/deactivated workers prevent incidental collection during this unit test.
    for _ in range(17):
        prepared = advance_economy(env, prepared)
    assert int(prepared.marines_built) == 0
    prepared = advance_economy(env, prepared)
    assert int(prepared.marines_built) == 1 and float(prepared.score) == 1.
    assert int(jnp.sum(prepared.world.smax_state.unit_alive)) == 13
    # Supply blocks additional queues when the cap is reached.
    s = prepared.world.smax_state
    full = prepared.replace(minerals=jnp.float32(1000.), world=prepared.world.replace(
        smax_state=s.replace(unit_alive=s.unit_alive.at[:15].set(True))))
    assert not bool(macro_mask(env, full)[6])


def test_resource_conservation_and_refinery_requirement():
    env = make("CollectMineralsAndGas")
    _, state = env.reset(jax.random.PRNGKey(0))
    s = state.world.smax_state
    state = state.replace(entity_amount=state.entity_amount.at[0].set(.5),
                          worker_target=state.worker_target.at[:2].set(0),
                          world=state.world.replace(smax_state=s.replace(
                              unit_positions=s.unit_positions.at[:2].set(state.entity_positions[0]))))
    next_state = advance_economy(env, state)
    assert float(next_state.entity_amount[0]) == 0.
    assert float(jnp.sum(next_state.cargo[:2])) == .5
    assert float(next_state.score) == 0.  # Cargo is not banked yet.
    gas = state.replace(worker_target=state.worker_target.at[0].set(16),
                         world=state.world.replace(smax_state=s.replace(
                             unit_positions=s.unit_positions.at[0].set(state.entity_positions[16]))))
    assert float(advance_economy(env, gas).cargo[0]) == 0.
    gas = gas.replace(building_kind=gas.building_kind.at[1].set(4),
                       building_positions=gas.building_positions.at[1].set(gas.entity_positions[16]))
    assert float(advance_economy(env, gas).cargo[0]) == 1.


def test_spending_resources_does_not_reduce_gathering_score():
    env = make("CollectMineralsAndGas")
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(minerals=jnp.float32(100.), gathered_minerals=jnp.float32(50.), score=jnp.float32(50.))
    action = env.no_op().replace(macro=jnp.int32(5), actor=jnp.int32(0))
    prepared, _, accepted = prepare(env, state, action)
    assert bool(accepted) and float(prepared.minerals) == 50.
    assert float(advance_economy(env, prepared).score) == 50.


@pytest.mark.parametrize("name", TASKS)
def test_scripted_policy_compiles_for_each_task(name):
    env = make(name)
    _, state = env.reset(jax.random.PRNGKey(42))
    action = jax.jit(lambda s: scripted_action(env, s))(state)
    _, state, reward, _, _ = jax.jit(env.step)(jax.random.PRNGKey(0), state, action)
    assert np.isfinite(float(reward))
    assert jnp.all(env.map.is_walkable(state.world.smax_state.unit_positions,
                                       env.map_width, env.map_height))


@pytest.mark.parametrize("algorithm", ["PPO", "MaskPPO"])
@pytest.mark.parametrize("name", ["MoveToBeacon", "FindAndDefeatZerglings"])
def test_existing_army_trainers_support_new_navigation_and_combat_tasks(algorithm, name, tmp_path):
    from algorithms.ppo import PPO
    from algorithms.mask_ppo import MaskPPO
    model = {"PPO": PPO, "MaskPPO": MaskPPO}[algorithm]({
        "NUM_ENVS": 1, "ROLLOUT_LEN": 2, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1,
        "LOG_INTERVAL": 1, "EVAL_INTERVAL": 0, "CKPT_INTERVAL": 100,
        "LOG_DIR": str(tmp_path / "logs"), "CKPT_DIR": str(tmp_path / "checkpoints"),
    })
    model.train(make(name, max_steps=2), 4)
    assert model.total_steps == 4
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(model.state.params))


def test_army_trainers_explicitly_reject_economy_without_macro_heads():
    from algorithms.ppo import PPO
    from algorithms.mask_ppo import MaskPPO
    for cls in (PPO, MaskPPO):
        with pytest.raises(ValueError, match="macro heads"):
            cls().train(make("BuildMarines"), 0)


def test_construction_cost_time_and_supply_completion():
    from JaxSC2.tasks.economy import supply
    env = make("BuildMarines")
    _, state = env.reset(jax.random.PRNGKey(0))
    site = state.building_positions[0] + jnp.array([0., 4.])
    s = state.world.smax_state
    state = state.replace(minerals=jnp.float32(100.), world=state.world.replace(
        smax_state=s.replace(unit_positions=s.unit_positions.at[0].set(site))))
    action = env.no_op().replace(macro=jnp.int32(2), actor=jnp.int32(0), placement=site)
    state, _, accepted = prepare(env, state, action)
    assert bool(accepted) and float(state.minerals) == 0.
    assert int(supply(state)[1]) == 15  # An unfinished depot grants no supply.
    for _ in range(20):
        state = advance_economy(env, state)
    assert int(supply(state)[1]) == 23


def test_same_objective_can_run_on_twobridge_terrain():
    from JaxSC2.maps.twobridge import TwoBridgeMap
    env = make("MoveToBeacon", terrain=TwoBridgeMap())
    _, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert jnp.all(env.map.is_walkable(state.world.smax_state.unit_positions, 32., 32.))
    _, result, reward, _, _ = jax.jit(env.step)(jax.random.PRNGKey(1), state, scripted_action(env, state))
    assert np.isfinite(float(reward))
    assert jnp.all(env.map.is_walkable(result.world.smax_state.unit_positions, 32., 32.))


def test_invalid_terrain_and_bridge_specific_modes_fail_clearly():
    with pytest.raises(ValueError, match="no radius-clear spawn"):
        ArenaMap(obstacles=[(0., 0., 32., 32.)])
    with pytest.raises(ValueError, match="bridge anchors"):
        make("DefeatRoaches", enemy_mode="bridge_blocker")


@pytest.mark.parametrize("name", ["CollectMineralsAndGas", "BuildMarines"])
def test_economy_scripted_baseline_makes_end_to_end_progress(name):
    from JaxSC2.tasks.evaluate import make_rollout
    env = make(name)
    final, _ = make_rollout(env)(jnp.stack([jax.random.PRNGKey(42)]))
    state = jax.tree_util.tree_map(lambda x: x[0], final[0])
    assert float(state.score) > 0 and int(final[2][0]) == 0
    assert jnp.all(state.entity_amount >= 0) and float(state.minerals) >= 0
    if name == "CollectMineralsAndGas":
        assert float(state.gathered_gas) > 0 and float(state.gathered_minerals) > 0
        assert jnp.sum(state.building_kind == 1) == 2
    else:
        assert int(state.marines_built) > 0 and jnp.any(state.building_kind == 3)
