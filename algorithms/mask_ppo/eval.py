"""Deterministic evaluation using the same action masks as MaskPPO training."""

from algorithms.common.rollout import evaluate_episodes, evaluation_metrics


def evaluate(env, params, model, rng, num_episodes=16, max_steps=500):
    return evaluation_metrics(*evaluate_episodes(
        env, params, model, rng, num_episodes, max_steps, masked=True
    ))
