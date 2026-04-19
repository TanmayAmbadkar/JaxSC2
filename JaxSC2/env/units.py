from abc import ABC, abstractmethod
from flax.struct import dataclass
import jax.numpy as jnp

# --- Blueprints (Abstract Classes) ---

class AbstractWeapon(ABC):
    @property
    @abstractmethod
    def damage(self) -> float: pass

    @property
    @abstractmethod
    def range(self) -> float: pass

    @property
    @abstractmethod
    def windup(self) -> int: pass

    @property
    @abstractmethod
    def cooldown(self) -> int: pass

class AbstractUnit(ABC):
    @property
    @abstractmethod
    def max_hp(self) -> float: pass

    @property
    @abstractmethod
    def speed(self) -> float: pass

    @property
    @abstractmethod
    def accel(self) -> float: pass

    @property
    @abstractmethod
    def mass(self) -> float: pass

    @property
    @abstractmethod
    def armor(self) -> float: pass

    @property
    @abstractmethod
    def weapon(self) -> AbstractWeapon: pass

# --- Concrete Weapon Implementations ---

@dataclass
class BladeWeapon(AbstractWeapon):
    damage: float = 8.0
    range: float = 1.5
    windup: int = 1
    cooldown: int = 6

@dataclass
class GaussRifle(AbstractWeapon):
    damage: float = 5.0
    range: float = 6.0
    windup: int = 2
    cooldown: int = 8

@dataclass
class PlasmaCannon(AbstractWeapon):
    damage: float = 4.0
    range: float = 4.0
    windup: int = 3
    cooldown: int = 12

# --- Concrete Unit Implementations ---

@dataclass
class MeleeUnit(AbstractUnit):
    weapon: AbstractWeapon = BladeWeapon()
    max_hp: float = 100.0
    speed: float = 0.3
    accel: float = 0.10
    mass: float = 2.0
    armor: float = 1.0

@dataclass
class RangedUnit(AbstractUnit):
    weapon: AbstractWeapon = GaussRifle()
    max_hp: float = 45.0
    speed: float = 0.25
    accel: float = 0.08
    mass: float = 1.0
    armor: float = 0.0

@dataclass
class TankUnit(AbstractUnit):
    weapon: AbstractWeapon = PlasmaCannon()
    max_hp: float = 200.0
    speed: float = 0.18
    accel: float = 0.05
    mass: float = 5.0
    armor: float = 2.0
