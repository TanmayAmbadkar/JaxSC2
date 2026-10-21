"""Deterministic evaluation for the multi-head PPO/A2C policy."""

from algorithms.common.rollout import evaluate_episodes, evaluation_metrics


def evaluate(env, params, model, rng, num_episodes=32, max_steps=400):
    return evaluation_metrics(*evaluate_episodes(
        env, params, model, rng, num_episodes, max_steps
    ))
