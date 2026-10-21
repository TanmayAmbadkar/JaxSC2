"""Scripted baselines should demonstrate objective progress on every layout."""

import jax
import jax.numpy as jnp
import pytest

from JaxSC2.env.env import JaxSC2Env
from JaxSC2.maps.twobridge import TwoBridgeMap
from JaxSC2.scripted.agents import beacon_agent, combat_agent
from JaxSC2.scripted.evaluate import evaluate_case


@pytest.mark.parametrize("variant", list(TwoBridgeMap().VARIANTS))
def test_scripted_agents_make_goal_progress(variant):
    env = JaxSC2Env(variant, enemy_ai=True, enemy_mode="static")
    rows, _ = evaluate_case(env, seed=42)
    assert rows[0]["nav_success"]
    assert rows[1]["enemy_damage"] > 0.
    assert all(row["units_in_bounds"] for row in rows)
    assert all(row["terrain_valid"] for row in rows)


def test_fog_combat_agent_does_not_use_hidden_enemy_positions():
    env = JaxSC2Env("V1_Base", enable_fow=True)
    _, state = env.reset(jax.random.PRNGKey(0))
    altered = state.replace(smax_state=state.smax_state.replace(
        unit_positions=state.smax_state.unit_positions.at[5:].set(jnp.array([[20., 0.], [30., 10.], [20., 30.]]))))
    first, second = combat_agent(env, state), combat_agent(env, altered)
    for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        assert jnp.array_equal(a, b)
    assert jnp.all(beacon_agent(env, state).verb == 1)
