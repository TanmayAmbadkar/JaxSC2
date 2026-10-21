"""Numerical checks for normalization and lossless per-unit action packing."""

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.common.utils import RunningMeanStd, update_rms, encode_per_unit_action, decode_per_unit_action


def test_running_variance_uses_exact_prior_count():
    prior = RunningMeanStd(jnp.zeros(1), jnp.ones(1), jnp.array(1e-4))
    result = jax.jit(update_rms)(prior, jnp.zeros(2))
    np.testing.assert_allclose(result.var, 1e-4 / 2.0001, rtol=1e-6)


def test_running_statistics_match_direct_batch_statistics():
    first, second = jnp.array([1., 2., 3.]), jnp.array([-2., 5., 9.])
    prior = RunningMeanStd(jnp.zeros(1), jnp.ones(1), jnp.array(0.))
    result = update_rms(update_rms(prior, first), second)
    data = jnp.concatenate([first, second])
    np.testing.assert_allclose(result.mean, jnp.mean(data), rtol=1e-6)
    np.testing.assert_allclose(result.var, jnp.var(data), rtol=1e-6)


def test_all_actions_round_trip_including_no_target_and_target_zero():
    verb, direction, target = jnp.meshgrid(jnp.arange(3), jnp.arange(8), jnp.arange(-1, 3), indexing="ij")
    verb, direction, target = [x.flatten() for x in (verb, direction, target)]
    packed = jax.jit(encode_per_unit_action)(verb, direction, target)
    assert len(np.unique(packed)) == len(packed)
    for actual, expected in zip(decode_per_unit_action(packed, 3), (verb, direction, target)):
        np.testing.assert_array_equal(actual, expected)
