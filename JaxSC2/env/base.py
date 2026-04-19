import jax
import jax.numpy as jnp
import chex
from flax.struct import dataclass
from typing import Tuple, Dict, Optional, Union, OrderedDict, Sequence

# --- Jittable Spaces (Minimal Gymnax-style) ---

class Space:
    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError
    def contains(self, x: jnp.int_) -> bool:
        raise NotImplementedError

class Discrete(Space):
    def __init__(self, num_categories: int, dtype=jnp.int32):
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype
    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)
    def contains(self, x: jnp.int_) -> bool:
        return jnp.logical_and(x >= 0, x < self.n)

class Box(Space):
    def __init__(self, low: float, high: float, shape: Tuple[int], dtype: jnp.dtype = jnp.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)
    def contains(self, x: jnp.int_) -> bool:
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))

# --- Abstract Multi-Agent Env ---

@dataclass
class State:
    done: chex.Array
    step: int

class MultiAgentEnv:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.observation_spaces = {}
        self.action_spaces = {}

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        raise NotImplementedError

    def step(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        return self.step_env(key, state, actions)

    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        raise NotImplementedError

# --- SMAX Lite (Schema & Constants) ---

@dataclass
class SmaxState:
    unit_positions: chex.Array
    unit_alive: chex.Array
    unit_teams: chex.Array
    unit_health: chex.Array
    unit_types: chex.Array
    unit_weapon_cooldowns: chex.Array
    prev_movement_actions: chex.Array
    prev_attack_actions: chex.Array
    time: int
    terminal: bool

class SMAX(MultiAgentEnv):
    """Lite version of SMAX constants for JaxSC2Env dependency injection."""
    def __init__(self, num_allies=5, num_enemies=5, **kwargs):
        super().__init__(num_allies + num_enemies)
        self.num_allies = num_allies
        self.num_enemies = num_enemies
        self.map_width = 32
        self.map_height = 32
        
        # Default SC2 Stats (Marine, Marauder, Stalker, Zealot, Zergling, Hydralisk)
        self.unit_type_health = jnp.array([45.0, 125.0, 160, 150, 35, 80])
        self.unit_type_radiuses = jnp.array([0.375, 0.5625, 0.625, 0.5, 0.375, 0.625])
        
        # Override with custom combat cfg from trainer if provided
        if "unit_type_attacks" in kwargs:
            self.unit_type_attacks = kwargs["unit_type_attacks"]
