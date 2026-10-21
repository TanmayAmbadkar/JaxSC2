"""Raw episode-return accounting, independent of reward normalization and GAE."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EpisodeAccumulator(NamedTuple):
    returns: jnp.ndarray
    lengths: jnp.ndarray


class EpisodeTotals(NamedTuple):
    return_sum: jnp.ndarray
    length_sum: jnp.ndarray
    count: jnp.ndarray


def accumulate_episodes(accumulator, rewards, dones):
    """Sum raw rewards through termination, carrying partial episodes across rollouts.

    Inputs have shape (time, environments). Completed episodes are emitted only
    on their done step, including terminal rewards. Incomplete returns/lengths
    remain in the accumulator and do not contribute to the reported mean.
    """
    def step(carry, transition):
        reward, done = transition
        returns, lengths = carry.returns + reward, carry.lengths + 1
        totals = EpisodeTotals(
            jnp.sum(jnp.where(done, returns, 0.)),
            jnp.sum(jnp.where(done, lengths, 0)), jnp.sum(done),
        )
        return EpisodeAccumulator(jnp.where(done, 0., returns), jnp.where(done, 0, lengths)), totals

    accumulator, totals = jax.lax.scan(step, accumulator, (rewards, dones))
    return accumulator, EpisodeTotals(*(jnp.sum(value) for value in totals))


class EpisodeLogWindow:
    """Weight completed episodes equally across an entire logging interval."""

    def __init__(self):
        self.return_sum = 0.
        self.length_sum = 0
        self.count = 0

    def add(self, totals):
        self.return_sum += float(totals.return_sum)
        self.length_sum += int(totals.length_sum)
        self.count += int(totals.count)

    def emit(self):
        metrics = {"train/completed_episodes": self.count}
        if self.count:
            metrics.update({
                "train/episode_reward": self.return_sum / self.count,
                "train/episode_length": self.length_sum / self.count,
            })
            display = (f'Episode Reward {metrics["train/episode_reward"]:7.2f} | '
                       f'EpLen {metrics["train/episode_length"]:6.1f} | Episodes {self.count}')
        else:
            display = "Episode Reward     n/a | EpLen    n/a | Episodes 0"
        self.return_sum, self.length_sum, self.count = 0., 0, 0
        return metrics, display
