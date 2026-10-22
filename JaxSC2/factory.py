"""Stable entry point for legacy TwoBridge and composable mini-game tasks."""

from JaxSC2.maps.twobridge import TwoBridgeMap
from JaxSC2.tasks.types import TASKS


def available_tasks():
    return tuple(TwoBridgeMap().VARIANTS) + tuple(TASKS)


def make(name="V2_Base", **kwargs):
    if name in TASKS:
        from JaxSC2.tasks.env import MiniGameEnv
        return MiniGameEnv(name, **kwargs)
    if name in TwoBridgeMap().VARIANTS:
        from JaxSC2.env.env import JaxSC2Env
        max_steps = kwargs.pop("max_steps", None)
        env = JaxSC2Env(name, **kwargs)
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be positive")
            env.max_steps = max_steps
        return env
    raise ValueError(f"Unknown task {name!r}; choose from {available_tasks()}")
