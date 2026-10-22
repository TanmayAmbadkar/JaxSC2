import jax
import jax.numpy as jnp
import chex
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Union, Any, Optional
from flax.struct import dataclass

from JaxSC2.env.base import SMAX, SmaxState
from JaxSC2.maps.twobridge import TwoBridgeMap
from JaxSC2.env.mechanics import (
    apply_mass_collisions, 
    apply_hard_collisions,
    integrate_velocity,
    update_persistent_targets,
    update_fog_memory, 
    compute_visibility,
    apply_high_fidelity_combat,
    FogState
)
from JaxSC2.env.units import MeleeUnit, RangedUnit, TankUnit
from JaxSC2.env.rewards import REWARD_PROFILES
from JaxSC2.env.enemy_ai import ENEMY_MODES, COMBAT_ENEMY_MODES, get_enemy_orders, idle_orders

@dataclass
class ActionBuffer:
    verbs: jnp.ndarray      # (delay, N) - per-unit actions per delay slot
    directions: jnp.ndarray # (delay,)
    targets: jnp.ndarray    # (delay,)
    who_masks: jnp.ndarray  # (delay, N)

@dataclass
class JaxSC2State:
    smax_state: SmaxState
    beacon_pos: jnp.ndarray
    prev_mean_dist: float
    prev_enemy_health: float
    prev_ally_health: float
    prev_enemy_dist: float
    unit_velocities: jnp.ndarray
    attack_timers: jnp.ndarray
    persistent_targets: jnp.ndarray
    proj_pos: jnp.ndarray
    proj_vel: jnp.ndarray
    proj_target: jnp.ndarray
    proj_damage: jnp.ndarray
    proj_active: jnp.ndarray
    proj_team: jnp.ndarray
    proj_types: jnp.ndarray
    fog_state: FogState
    action_buffer: ActionBuffer
    enemy_visible: jnp.ndarray # (num_enemies,)
    enemy_home: jnp.ndarray  # initial positions: stable defensive bank/anchors
    enemy_alerted: jnp.ndarray  # latched ambush activation, reset each episode
    timestep: int = 0

def init_action_buffer(delay, num_units):
    return ActionBuffer(
        verbs=jnp.zeros((delay, num_units), dtype=jnp.int32),
        directions=jnp.zeros((delay, num_units), dtype=jnp.int32),
        targets=jnp.zeros((delay, num_units), dtype=jnp.int32),
        who_masks=jnp.zeros((delay, num_units), dtype=jnp.bool_),
    )

def push_and_pop_actions(buffer: ActionBuffer, new_action):
    verbs = jnp.roll(buffer.verbs, shift=-1, axis=0)
    dirs  = jnp.roll(buffer.directions, shift=-1, axis=0)
    targs = jnp.roll(buffer.targets, shift=-1, axis=0)
    who = jnp.roll(buffer.who_masks, shift=-1, axis=0)

    verbs = verbs.at[-1].set(new_action.verb)
    dirs  = dirs.at[-1].set(new_action.direction)
    targs = targs.at[-1].set(new_action.target)
    who = who.at[-1].set(new_action.who_mask)

    exec_action = PerUnitAction(
        who_mask=who[0],
        verb=verbs[0],
        direction=dirs[0],
        target=targs[0],
    )
    return ActionBuffer(verbs, dirs, targs, who), exec_action

@dataclass(frozen=True)
class PerUnitAction:
    who_mask: jnp.ndarray   # Shape: (N,) - Boolean, which allies are active
    verb: jnp.ndarray       # Shape: (N,) - 0: NO_OP, 1: MOVE, 2: ATTACK
    direction: jnp.ndarray  # Shape: (N,) - 0-7 cardinal/intercardinal directions
    target: jnp.ndarray     # Shape: (N,) - 0 to (num_enemies-1), -1 = no target


def CentralAction(who_mask, verb, direction, target):
    """Compatibility constructor broadcasting an army command to selected units."""
    who_mask = jnp.asarray(who_mask, dtype=jnp.bool_)
    return PerUnitAction(
        who_mask,
        jnp.broadcast_to(jnp.asarray(verb, dtype=jnp.int32), who_mask.shape),
        jnp.broadcast_to(jnp.asarray(direction, dtype=jnp.int32), who_mask.shape),
        jnp.broadcast_to(jnp.asarray(target, dtype=jnp.int32), who_mask.shape),
    )

def build_action_mask(state: JaxSC2State, num_allies: int) -> Dict[str, jnp.ndarray]:
    """
    Returns per-unit action masks for SC2 semantics.
    
    verb_mask:      (N, 3) - [NO_OP valid, MOVE valid, ATTACK valid]
    direction_mask: (N, 8) - all directions valid if MOVE
    target_mask:    (N, num_enemies) - only visible & alive enemies
    
    All masks are False for dead units (who_mask = ally_alive).
    """
    s = state.smax_state
    
    # Alive mask (N,)
    ally_alive = s.unit_alive[:num_allies]
    
    # Verb masks (N, 3)
    verb_mask = jnp.stack([
        ally_alive,           # NO_OP always valid if alive
        ally_alive,           # MOVE always valid if alive
        jnp.zeros(num_allies, dtype=jnp.bool_)  # ATTACK - refined below
    ], axis=-1)
    
    # Attack requires a visible, living enemy and a living attacker.
    enemy_alive = s.unit_alive[num_allies:]
    any_enemy_alive = jnp.any(enemy_alive & state.enemy_visible)
    attack_valid = jnp.where(any_enemy_alive, 
                             jnp.ones(num_allies, dtype=jnp.bool_),
                             jnp.zeros(num_allies, dtype=jnp.bool_))
    verb_mask = verb_mask.at[:, 2].set(attack_valid & ally_alive)
    
    # Direction masks (N, 8) - always valid if alive
    direction_mask = jnp.tile(ally_alive[:, None], (1, 8))
    
    # Target masks (N, num_enemies) - only visible & alive enemies
    visible = state.enemy_visible  # (num_enemies,)
    target_mask = jnp.outer(ally_alive, visible & enemy_alive)  # (N, num_enemies)
    
    return {
        "verb": verb_mask,           # (N, 3)
        "direction": direction_mask, # (N, 8)
        "target": target_mask,       # (N, num_enemies)
    }

def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Apply softmax with masking - invalid actions get -inf logits."""
    masked_logits = jnp.where(mask, logits, -1e9)
    return jax.nn.softmax(masked_logits, axis=-1)

class DictWrapper:
    """Helper to provide .shape or .n matching gym.spaces interface."""
    def __init__(self, data):
        self.__setattr__("shape", data.get("shape"))
        self.__setattr__("nvec", data.get("nvec"))
        self.__setattr__("n", data.get("n"))

class JaxSC2Env:
    """
    JAX-Native Two-Bridge Environment Suite for Per-Unit RL.
    
    Action Space (Per-Unit):
        - who_mask:   (N,) boolean - which allies are active this step
        - verb:       (N,) int32   - 0: NO_OP, 1: MOVE, 2: ATTACK
        - direction:  (N,) int32   - 0-7 cardinal/intercardinal directions
        - target:     (N,) int32   - 0 to (num_enemies-1), -1 = no target
    
    Observation Space (Vector):
        - 5 Allies (4 features: [rel_x, rel_y, health, cooldown]): 20-dim.
        - 8 Enemies (4 features: [rel_x, rel_y, health, cooldown]): 32-dim.
        - Enemy Mask (1.0 if enemy exists, 0.0 if padded): 8-dim.
        - Global (Beacon [x,y], Timestep): 3-dim.
    
    Action Masks:
        - verb_mask:      (N, 3) - [NO_OP valid, MOVE valid, ATTACK valid]
        - direction_mask: (N, 8) - all directions valid if alive
        - target_mask:    (N, num_enemies) - only visible & alive enemies
    """
    def __init__(self, variant_name="V2_Base", use_spatial_obs=False, resolution=64, latency_delay=0,
                 reward_profile="balanced", map_instance=None, variant_config=None,
                 units=None, max_enemy_slots=None, projectile_capacity=32, **kwargs):
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown reward profile: {reward_profile}")
        self.reward_profile = reward_profile
        self.reward_weights = dict(REWARD_PROFILES[reward_profile])
        self.map_instance = map_instance if map_instance is not None else TwoBridgeMap()
        self.variant_config = (variant_config if variant_config is not None
                               else self.map_instance.VARIANTS[variant_name])
        self.num_allies = self.variant_config.n_ally
        self.num_enemies = self.variant_config.n_enemy
        self.num_units = self.num_allies + self.num_enemies
        if self.num_allies < 1 or self.num_enemies < 1 or projectile_capacity < 1:
            raise ValueError("World capacities must be positive; represent absent units with alive masks")
        self.max_enemy_slots = max(self.num_enemies, 8) if max_enemy_slots is None else max_enemy_slots
        if self.max_enemy_slots < self.num_enemies:
            raise ValueError("Observation enemy capacity cannot be smaller than world capacity")
        self.projectile_capacity = projectile_capacity
        
        self.use_spatial_obs = use_spatial_obs
        self.resolution = resolution
        self.latency_delay = latency_delay
        
        self.enemy_ai = kwargs.pop("enemy_ai", False)
        self.enemy_mode = kwargs.pop("enemy_mode", "guard")
        if self.enemy_mode not in ENEMY_MODES:
            raise ValueError(f"Unknown enemy mode {self.enemy_mode!r}; choose from {ENEMY_MODES}")
        self.mode_id = ENEMY_MODES.index(self.enemy_mode)
        
        # Initialize SMAX
        smax_combat_cfg = {}
        if kwargs.get("enable_armor", True):
            smax_combat_cfg["unit_type_attacks"] = jnp.array([0.0] * 6)
            
        self.smax_env = SMAX(
            num_allies=self.num_allies,
            num_enemies=self.num_enemies,
            **smax_combat_cfg
        )
        
        self.max_steps = 300
        self.map_width = getattr(self.map_instance, "width", self.smax_env.map_width)
        self.map_height = getattr(self.map_instance, "height", self.smax_env.map_height)
        self.map_dimensions = jnp.array([self.map_width, self.map_height])
        
        self.unit_teams = jnp.concatenate([
            jnp.zeros(self.num_allies, dtype=jnp.int32),
            jnp.ones(self.num_enemies, dtype=jnp.int32)
        ])
        
        # Advanced Mechanics Config (Toggleable)
        self.enable_collision = kwargs.get("enable_collision", True)
        self.enable_fow = kwargs.get("enable_fow", False)
        self.enable_armor = kwargs.get("enable_armor", True)
        
        self.unit_radius = 0.6
        self.vision_radius = 6.0
        
        # Unit Heterogeneity Schema (Melee, Ranged, Tank)
        custom_units = units is not None
        self.units = list(units) if custom_units else [MeleeUnit(), RangedUnit(), TankUnit()]
        
        # Aggregate stats into JAX arrays for simulation
        self.unit_type_hp = jnp.array([u.max_hp for u in self.units])
        self.unit_type_range = jnp.array([u.weapon.range for u in self.units])
        self.unit_type_speed = jnp.array([u.speed for u in self.units])
        self.unit_type_accel = jnp.array([u.accel for u in self.units])
        self.unit_type_damage = jnp.array([u.weapon.damage for u in self.units])
        self.type_radius = jnp.array([self.unit_radius] * len(self.units))
        self.type_mass = jnp.array([u.mass for u in self.units])
        self.unit_armor = jnp.array([u.armor for u in self.units])

        self.combat_cfg = {
            "base_damage": 5.0,
            "attack_range": 6.0,
            "windup": self.units[1].weapon.windup,
            "backswing": 1,
            "cooldown": self.units[1].weapon.cooldown, 
            "turn_rate": 0.4,
            "damage_matrix": jnp.array([
                [1.0, 1.5, 0.7],
                [0.8, 1.0, 1.5],
                [1.5, 0.7, 1.0]
            ]),
            "bonus_matrix": jnp.array([
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 0.0]
            ]),
            "aggression_radius": 8.0,
            "type_ranges": self.unit_type_range,
            "type_damages": self.unit_type_damage,
            "type_windups": jnp.array([u.weapon.windup for u in self.units]),
            "type_cooldowns": jnp.array([u.weapon.cooldown for u in self.units]),
        }
        if custom_units:
            self.combat_cfg["damage_matrix"] = jnp.ones((len(self.units), len(self.units)))
            self.combat_cfg["bonus_matrix"] = jnp.zeros((len(self.units), len(self.units)))
        # Hoist constants for JIT efficiency
        self._angles = jnp.linspace(0, 2*jnp.pi, 8, endpoint=False)
        
        # Precompute NOT_SELF_MASK for collision functions (N x N)
        max_units = self.num_units
        i_idx = jnp.arange(max_units)[:, None]
        j_idx = jnp.arange(max_units)[None, :]
        self._not_self_mask = (i_idx != j_idx).astype(jnp.float32)
        
        # SB3-style Space definitions (Native)
        obs_dim = 4 * self.num_allies + 5 * self.max_enemy_slots + 3
        self.observation_space = DictWrapper({"shape": (obs_dim,)})
        self.action_space = DictWrapper({
            "nvec": jnp.array([2, 3, 8, 8]), 
            "n": 17
        })

    def _compute_centroid(self, ally_pos, ally_alive):
        """Compute centroid of alive allies. Returns (2,) array."""
        alive_mask = ally_alive[:, None]
        sum_pos = jnp.sum(ally_pos * alive_mask, axis=0)
        count = jnp.sum(ally_alive)
        return jnp.where(count > 0, sum_pos / jnp.maximum(count, 1.0), jnp.zeros(2))

    def reset(self, rng: chex.PRNGKey) -> Tuple[Union[jnp.ndarray, Dict], JaxSC2State]:
        rng_region, rng_ally_pos, rng_enemy_pos, rng_beacon_pos = jax.random.split(rng, 4)
        
        # 1. JIT-Compilable Region Selection (No Overlap)
        a_global_idx, e_global_idx, b_global_idx = self.map_instance.get_spawn_regions(
            rng_region,
            self.variant_config.layout_type
        )
        
        # 2. Sample dense clusters (Centroid + small offset)
        def sample_dense(rng, g_idx, num):
            rng_x, rng_y, rng_o = jax.random.split(rng, 3)
            bounds = self.map_instance.REGION_COORDS[g_idx]
            
            cx = jax.random.uniform(rng_x, (), minval=(bounds[0]+0.05)*self.map_width, maxval=(bounds[1]-0.05)*self.map_width)
            cy = jax.random.uniform(rng_y, (), minval=(bounds[2]+0.05)*self.map_height, maxval=(bounds[3]-0.05)*self.map_height)
            
            offsets = jax.random.uniform(rng_o, (num, 2), minval=-1.5, maxval=1.5)
            pos = jnp.stack([cx, cy], axis=-1) + offsets
            return jnp.clip(pos, .05, jnp.array([self.map_width, self.map_height]) - .05)

        ally_pos = sample_dense(rng_ally_pos, a_global_idx, self.num_allies)
        enemy_pos = sample_dense(rng_enemy_pos, e_global_idx, self.num_enemies)
        
        bounds_b = self.map_instance.REGION_COORDS[b_global_idx]
        rng_bx, rng_by = jax.random.split(rng_beacon_pos)
        bx = jax.random.uniform(rng_bx, (), minval=bounds_b[0]*self.map_width, maxval=bounds_b[1]*self.map_width)
        by = jax.random.uniform(rng_by, (), minval=bounds_b[2]*self.map_height, maxval=bounds_b[3]*self.map_height)
        beacon_pos = jnp.array([bx, by])
        
        # 3. Construct SMAX State
        unit_positions = jnp.concatenate([ally_pos, enemy_pos])
        unit_teams = jnp.zeros((self.num_allies + self.num_enemies,), dtype=jnp.int32)
        unit_teams = unit_teams.at[self.num_allies:].set(1)
        
        base_types = jnp.array([0, 1, 2], dtype=jnp.int32)
        a_types = jnp.tile(base_types, (self.num_allies // 3 + 1))[:self.num_allies]
        e_types = jnp.tile(base_types, (self.num_enemies // 3 + 1))[:self.num_enemies]
        unit_types = jnp.concatenate([a_types, e_types])
        
        unit_health = self.unit_type_hp[unit_types]
        
        smax_state = SmaxState(
            unit_positions=unit_positions,
            unit_alive=jnp.ones((self.num_allies + self.num_enemies,), dtype=jnp.bool_),
            unit_teams=unit_teams,
            unit_health=unit_health,
            unit_types=unit_types,
            unit_weapon_cooldowns=jnp.zeros((self.num_allies + self.num_enemies,)),
            prev_movement_actions=jnp.zeros((self.num_allies + self.num_enemies, 2)),
            prev_attack_actions=jnp.zeros((self.num_allies + self.num_enemies,), dtype=jnp.int32),
            time=0,
            terminal=False,
        )
        
        initial_mean_dist = jnp.mean(jnp.linalg.norm((ally_pos - beacon_pos) / self.map_dimensions, axis=-1))
        initial_enemy_dist = jnp.mean(jnp.linalg.norm((ally_pos - jnp.mean(enemy_pos, axis=0)) / self.map_dimensions, axis=-1))
        
        visible = (compute_visibility(ally_pos, smax_state.unit_alive[:self.num_allies],
                                      enemy_pos, self.vision_radius)
                   if self.enable_fow else jnp.ones(self.num_enemies, dtype=jnp.bool_))
        fog_state = FogState(
            last_seen_pos=jnp.where(visible[:, None], enemy_pos, 0.),
            last_seen_alive=visible,
            last_seen_hp=jnp.where(visible, smax_state.unit_health[self.num_allies:], 0.),
            last_seen_time=jnp.where(visible, 0, -1).astype(jnp.int32),
        )
        action_buffer = init_action_buffer(max(1, self.latency_delay), self.num_allies)

        state = JaxSC2State(
            smax_state=smax_state,
            beacon_pos=beacon_pos,
            prev_mean_dist=initial_mean_dist,
            prev_enemy_health=jnp.sum(smax_state.unit_health[self.num_allies:]),
            prev_ally_health=jnp.sum(smax_state.unit_health[:self.num_allies]),
            prev_enemy_dist=initial_enemy_dist,
            unit_velocities=jnp.zeros((self.num_units, 2)),
            attack_timers=jnp.zeros(self.num_units, dtype=jnp.int32),
            persistent_targets=jnp.ones(self.num_units, dtype=jnp.int32) * -1,
            proj_pos=jnp.zeros((self.projectile_capacity, 2)),
            proj_vel=jnp.zeros((self.projectile_capacity, 2)),
            proj_target=jnp.ones(self.projectile_capacity, dtype=jnp.int32) * -1,
            proj_damage=jnp.zeros(self.projectile_capacity),
            proj_active=jnp.zeros(self.projectile_capacity, dtype=jnp.bool_),
            proj_team=jnp.zeros(self.projectile_capacity, dtype=jnp.int32),
            proj_types=jnp.full(self.projectile_capacity, -1, dtype=jnp.int32),
            fog_state=fog_state,
            action_buffer=action_buffer,
            enemy_visible=visible,
            enemy_home=enemy_pos,
            enemy_alerted=jnp.zeros(self.num_enemies, dtype=jnp.bool_),
            timestep=0
        )
        
        return self.get_obs(state), state

    def advance(self, rng, state, action):
        """Advance only simulation dynamics; tasks own reward and termination."""
        # 1. Action Latency Buffer
        new_buffer, exec_action = push_and_pop_actions(state.action_buffer, action)
        
        # Get Actions - Enemy AI or default
        new_key, rng_enemy = jax.random.split(rng)
        
        enemy_orders = jax.lax.cond(
            self.enemy_ai,
            lambda _: self.get_enemy_orders(state, rng_enemy),
            lambda _: idle_orders(self, state),
            operand=None
        )
        
        e_verb, e_dir = enemy_orders.verb, enemy_orders.direction

        # Concatenate ally + enemy — always (N_total,) inside vmap; vmap adds B automatically
        ally_verb = exec_action.verb       # (N_allies,)
        ally_dir  = exec_action.direction  # (N_allies,)
        verbs = jnp.concatenate([ally_verb, e_verb], axis=0)  # (N_total,)
        dirs  = jnp.concatenate([ally_dir,  e_dir],  axis=0)  # (N_total,)

        # Unit types — (N_total,), no [None, :] expansion
        u_types = state.smax_state.unit_types  # (N_total,)

        # Decode angles: (N_total,)
        dirs_mod = jnp.where(dirs >= 0, dirs % 8, 0)
        angles = jnp.take(self._angles, dirs_mod)  # (N_total,)

        # Acceleration magnitudes: (N_total,)
        accel_mags = self.unit_type_accel[u_types]  # (N_total,)

        # Direction vectors: (N_total, 2)
        direction_vecs = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (N_total, 2)

        # Who mask: pad ally mask with True for enemies — (N_total,)
        who_masks = jnp.concatenate([
            exec_action.who_mask.astype(bool),
            jnp.ones(self.num_enemies, dtype=bool)
        ], axis=0)  # (N_total,)

        # Alive mask: (N_total,) — no expansion needed
        alive_masks = state.smax_state.unit_alive  # (N_total,)

        # Combine: active = who & alive, move = active & verb==1
        active_masks = who_masks & alive_masks          # (N_total,)
        move_masks   = active_masks & (verbs == 1)      # (N_total,)

        # Windup lock: (N_total,)
        windups  = self.combat_cfg["type_windups"][u_types]  # (N_total,)
        backswing = self.combat_cfg.get("backswing", 1)
        is_locked = (state.attack_timers > 0) & (state.attack_timers <= (windups + backswing))

        # Final acceleration: (N_total, 2) — use [:, None] to broadcast (N,) with (N, 2)
        accel_vecs = (direction_vecs
                      * accel_mags[:, None]
                      * move_masks[:, None]
                      * (~is_locked)[:, None])

        # 3. Apply Turn-Rate Inertia — no shape branching, vmap handles batching
        def apply_inertia(accel, vel):
            v_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)   # (N, 1)
            has_vel = v_norm > 1e-3
            a_mag   = jnp.linalg.norm(accel, axis=-1, keepdims=True) # (N, 1)
            turn_rate = self.combat_cfg.get("turn_rate", 0.4)
            heading = jnp.arctan2(vel[:, 1], vel[:, 0])
            desired = jnp.arctan2(accel[:, 1], accel[:, 0])
            delta = jnp.arctan2(jnp.sin(desired - heading), jnp.cos(desired - heading))
            heading = heading + jnp.clip(delta, -turn_rate, turn_rate)
            rotated = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1) * a_mag
            return jnp.where(has_vel, rotated, accel)

        accel_vecs = apply_inertia(accel_vecs, state.unit_velocities)  # (N_total, 2)
        
        # 4. Physics Integration
        next_pos, next_vel = integrate_velocity(
            state.smax_state.unit_positions, 
            state.unit_velocities, 
            accel_vecs, 
            self.unit_type_speed[state.smax_state.unit_types]
        )
        
        # 5. Dual-Layer Collisions
        if self.enable_collision:
            next_pos = apply_mass_collisions(
                next_pos, state.smax_state.unit_alive, state.smax_state.unit_types,
                self.type_radius, self.type_mass,
            )
            next_pos = apply_hard_collisions(
                next_pos, state.smax_state.unit_alive, unit_radius=0.3, stiffness=1.2,
            )

        # Radius-aware terrain constraints apply identically to both teams.
        terrain_input = next_pos
        next_pos = self.map_instance.enforce_constraints(
            next_pos, 
            state.smax_state.unit_positions,
            self.map_width, 
            self.map_height,
            self.unit_teams,
            unit_radius=self.type_radius[state.smax_state.unit_types],
        )
        # Collisions/terrain can reject movement; do not retain velocity through
        # an outer boundary or the cliff. Dead units must not drift either.
        corrected = jnp.abs(next_pos - terrain_input) > 1e-6
        next_vel = jnp.where(corrected, 0., next_vel)
        next_vel = jnp.where(state.smax_state.unit_alive[:, None], next_vel, 0.)
        
        # 6. Target Persistence
        current_targets = update_persistent_targets(
            state.persistent_targets,
            next_pos,
            state.smax_state.unit_alive,
            next_pos,
            state.smax_state.unit_alive,
            self.unit_teams,
            self.unit_teams,
            self.unit_type_range[state.smax_state.unit_types],
            leash_extra=2.5,
            pool_visible=jnp.where(
                self.unit_teams[:, None] == 0,
                jnp.concatenate([jnp.ones(self.num_allies, dtype=jnp.bool_),
                                 state.enemy_visible])[None, :], True,
            ),
        )
        
        # Per-unit target override from action
        target = jnp.clip(exec_action.target, 0, self.num_enemies - 1)
        manual_valid = (
            (exec_action.verb == 2) & exec_action.who_mask
            & (exec_action.target >= 0) & (exec_action.target < self.num_enemies)
            & state.smax_state.unit_alive[self.num_allies:][target]
            & state.enemy_visible[target]
        )
        manual_targets = jnp.where(
            manual_valid,
            target + self.num_allies,
            -1
        )
        next_targets = current_targets.at[:self.num_allies].set(
            jnp.where(manual_targets >= 0, manual_targets, current_targets[:self.num_allies])
        )
        if self.enemy_mode in COMBAT_ENEMY_MODES:
            # Explicit targets enable coordinated focus fire and suppression
            # before an ambush triggers. Disabled AI retains legacy auto-attack.
            next_targets = next_targets.at[self.num_allies:].set(jnp.where(
                self.enemy_ai, enemy_orders.target, next_targets[self.num_allies:]))
        
        # 7. High-Fidelity Combat
        unit_armor_per_unit = (self.unit_armor[state.smax_state.unit_types]
                              if self.enable_armor else jnp.zeros(self.num_units))
        proj_st = (state.proj_pos, state.proj_vel, state.proj_target, state.proj_damage,
                   state.proj_active, state.proj_team, state.proj_types)
        next_hp, next_alive, next_timers, next_proj_st, dropped_shots = apply_high_fidelity_combat(
            state.smax_state.unit_health,
            state.smax_state.unit_alive,
            next_pos,
            state.smax_state.unit_types,
            self.unit_teams,
            unit_armor_per_unit,
            state.attack_timers,
            next_targets,
            proj_st,
            self.combat_cfg,
        )
        
        # 8. Update visibility based on vision radius
        ally_pos = next_pos[:self.num_allies]
        enemy_pos = next_pos[self.num_allies:]
        ally_alive = next_alive[:self.num_allies]
        enemy_alive = next_alive[self.num_allies:]
        
        # 9. Fog of War Memory Update
        next_fog_state, fog_visibility = update_fog_memory(
            state.fog_state,
            ally_pos,
            ally_alive,
            enemy_pos,
            enemy_alive,
            next_hp[self.num_allies:],
            self.vision_radius,
            state.timestep + 1,
        )
        
        if not self.enable_fow:
            fog_visibility = jnp.ones(self.num_enemies, dtype=jnp.bool_)
            next_fog_state = FogState(
                enemy_pos, enemy_alive, next_hp[self.num_allies:],
                jnp.full(self.num_enemies, state.timestep + 1, dtype=jnp.int32),
            )
        new_visibility = fog_visibility & enemy_alive
        
        world = state.replace(
            smax_state=state.smax_state.replace(
                unit_positions=next_pos, unit_alive=next_alive, unit_health=next_hp,
                unit_weapon_cooldowns=jnp.zeros_like(state.smax_state.unit_weapon_cooldowns),
                prev_movement_actions=jnp.zeros_like(state.smax_state.prev_movement_actions),
                prev_attack_actions=jnp.zeros_like(state.smax_state.prev_attack_actions),
                time=state.timestep + 1, terminal=False),
            unit_velocities=next_vel, attack_timers=next_timers, persistent_targets=next_targets,
            proj_pos=next_proj_st[0], proj_vel=next_proj_st[1], proj_target=next_proj_st[2],
            proj_damage=next_proj_st[3], proj_active=next_proj_st[4],
            proj_team=next_proj_st[5], proj_types=next_proj_st[6],
            fog_state=next_fog_state, action_buffer=new_buffer, enemy_visible=new_visibility,
            enemy_alerted=enemy_orders.alerted, timestep=state.timestep + 1)
        return world, {"dropped_shots": dropped_shots}

    def step(self, rng, state, action):
        """Compatibility TwoBridge task; other tasks call advance directly."""
        next_state, physics_info = self.advance(rng, state, action)
        next_pos = next_state.smax_state.unit_positions
        next_hp = next_state.smax_state.unit_health
        next_alive = next_state.smax_state.unit_alive
        # 10. Compute Rewards
        dims = jnp.array([self.map_width, self.map_height])
        ally_pos_obs = next_pos[:self.num_allies]
        enemy_pos_obs = next_pos[self.num_allies:]
        ally_alive_obs = next_alive[:self.num_allies]
        enemy_alive_obs = next_alive[self.num_allies:]
        
        # Navigation Reward (Distance to Beacon)
        dist_to_beacon = jnp.linalg.norm((ally_pos_obs - state.beacon_pos) / dims, axis=-1)
        mean_dist = jnp.sum(dist_to_beacon * ally_alive_obs) / jnp.maximum(jnp.sum(ally_alive_obs), 1.0)
        weights = self.reward_weights
        nav_reward = (state.prev_mean_dist - mean_dist) * weights["navigation_progress"]
        
        # Combat shaping: bidirectional distance to enemy centroid (Panda et al., ICML 2026)
        alive_enemy_float = enemy_alive_obs.astype(jnp.float32)
        combat_centroid = jnp.sum(enemy_pos_obs * alive_enemy_float[:, None], axis=0) / jnp.maximum(jnp.sum(alive_enemy_float), 1.0)
        dist_to_combat_centroid = (
            jnp.sum(jnp.linalg.norm((ally_pos_obs - combat_centroid) / dims, axis=-1) * ally_alive_obs.astype(jnp.float32), axis=0)
            / jnp.maximum(jnp.sum(ally_alive_obs.astype(jnp.float32)), 1.0)
        )
        combat_shaping = (state.prev_enemy_dist - dist_to_combat_centroid) * weights["combat_progress"]
        
        # Combat Reward (HP Damage + Kills)
        current_enemy_hp = jnp.sum(next_hp[self.num_allies:] * enemy_alive_obs)
        enemy_dmg_reward = (state.prev_enemy_health - current_enemy_hp) * weights["enemy_damage"]
        
        current_ally_hp = jnp.sum(next_hp[:self.num_allies] * ally_alive_obs)
        ally_dmg_penalty = (state.prev_ally_health - current_ally_hp) * weights["ally_damage"]
        
        # Kill Bonuses
        enemy_killed = (jnp.sum(state.smax_state.unit_alive[self.num_allies:]) - jnp.sum(enemy_alive_obs)) * weights["enemy_kill"]
        
        # Total Reward (step-level)
        total_reward = nav_reward + combat_shaping + enemy_dmg_reward - ally_dmg_penalty + enemy_killed
        
        # Termination
        beacon_reached = jnp.any((dist_to_beacon < 0.05) & ally_alive_obs)
        enemies_dead = jnp.sum(enemy_alive_obs) == 0
        allies_dead = jnp.sum(ally_alive_obs) == 0
        done = beacon_reached | enemies_dead | allies_dead | (state.timestep + 1 >= self.max_steps)
        
        # Either objective earns the same terminal bonus in the balanced profile.
        # Simultaneous victories receive one bonus, not a double payout.
        terminal_reward = jnp.where(beacon_reached, weights["beacon_win"],
            jnp.where(enemies_dead, weights["combat_win"],
                jnp.where(allies_dead, weights["defeat"],
                    jnp.where(done, weights["timeout"], 0.0))))
        total_reward = total_reward + terminal_reward
        
        # 11. Build Info
        info = {
            "beacon_reached": beacon_reached,
            "nav_success": beacon_reached,
            "combat_success": enemies_dead,
            "win": beacon_reached | enemies_dead,
            "enemies_killed": enemy_killed,
            "nav_reward": nav_reward,
            "combat_shaping": combat_shaping,
            "combat_reward": enemy_dmg_reward - ally_dmg_penalty + enemy_killed,
            "terminal_reward": terminal_reward,
        }
        
        next_state = next_state.replace(
            smax_state=next_state.smax_state.replace(terminal=done),
            prev_mean_dist=mean_dist, prev_enemy_health=current_enemy_hp,
            prev_ally_health=current_ally_hp, prev_enemy_dist=dist_to_combat_centroid)
        info.update(physics_info)
        return self.get_obs(next_state), next_state, total_reward, done, info

    def get_enemy_actions(self, state: JaxSC2State, rng=None) -> jnp.ndarray:
        """Compatibility four-direction encoding; advanced orders also have targets."""
        orders = self.get_enemy_orders(state, rng)
        return jnp.where(orders.verb == 1, ((orders.direction + 1) // 2) % 4, 4)

    def get_enemy_orders(self, state: JaxSC2State, rng=None):
        """Per-enemy movement, target selection, and persistent ambush activation."""
        return get_enemy_orders(self, state)

    def get_obs(self, state: JaxSC2State) -> Union[jnp.ndarray, Dict]:
        vector_obs = self.build_vector_obs(state)
        if not self.use_spatial_obs:
            return vector_obs
        
        return {
            "vector": vector_obs,
            **self.build_spatial_features(state)
        }

    def build_vector_obs(self, state: JaxSC2State, max_enemies=None) -> jnp.ndarray:
        """
        SC2-semantics-aligned observation (63-dim, invariant across variants)
        """
        max_enemies = self.max_enemy_slots if max_enemies is None else max_enemies
        s = state.smax_state
        dims = self.map_dimensions

        ally_pos = s.unit_positions[:self.num_allies]
        enemy_pos = s.unit_positions[self.num_allies:]

        ally_alive = s.unit_alive[:self.num_allies]
        enemy_alive = s.unit_alive[self.num_allies:]

        # 1. Reference frame (SC2-style: relative to ally centroid)
        centroid = self._compute_centroid(ally_pos, ally_alive)

        # 2. Allies
        max_hp = jnp.max(self.unit_type_hp)
        f_rel_pos = (ally_pos - centroid) / dims
        f_hp = s.unit_health[:self.num_allies] / max_hp
        f_cd = jnp.clip(state.attack_timers[:self.num_allies].astype(jnp.float32) / jnp.maximum(1.0, self.combat_cfg["windup"]), 0.0, 1.0)
        
        # SC2 Reality: Dead units vanish from the observation
        f_feats = jnp.concatenate([f_rel_pos, f_hp[:, None], f_cd[:, None]], axis=-1)
        f_feats = (f_feats * ally_alive[:, None]).flatten()

        # 3. Enemies (Use Fog memory for non-visible units)
        max_hp = jnp.max(self.unit_type_hp)
        e_pos_obs = state.fog_state.last_seen_pos
        e_alive_obs = state.fog_state.last_seen_alive
        e_hp_obs = state.fog_state.last_seen_hp
        
        e_rel_pos = (e_pos_obs - centroid) / dims
        e_hp_norm = e_hp_obs / max_hp
        
        e_windups = self.combat_cfg["type_windups"][state.smax_state.unit_types[self.num_allies:]]
        e_cd = state.attack_timers[self.num_allies:].astype(jnp.float32) / jnp.maximum(1.0, e_windups)
        e_cd = jnp.where(state.enemy_visible, e_cd, 0.)

        e_feats = jnp.concatenate([e_rel_pos, e_hp_norm[:, None], e_cd[:, None]], axis=-1)
        e_feats = e_feats * e_alive_obs[:, None]
        
        e_feats = jnp.pad(e_feats, ((0, max_enemies - self.num_enemies), (0, 0))).flatten()

        enemy_exist_mask = jnp.concatenate([
            jnp.ones(self.num_enemies),
            jnp.zeros(max_enemies - self.num_enemies)
        ])

        # 5. Global features (relative/normalized)
        beacon_rel = (state.beacon_pos - centroid) / dims
        global_feats = jnp.array([
            beacon_rel[0],
            beacon_rel[1],
            state.timestep / self.max_steps
        ])

        return jnp.concatenate([f_feats, e_feats, enemy_exist_mask, global_feats])

    def get_action_mask(self, state: JaxSC2State):
        """
        SC2-style action masks: who, verb, target.
        """
        s = state.smax_state
        ally_alive = s.unit_alive[:self.num_allies]
        enemy_alive = s.unit_alive[self.num_allies:]

        who_mask = ally_alive.astype(jnp.float32)

        # VERB mask: [noop, move, attack]
        can_attack = jnp.float32(jnp.any(enemy_alive & state.enemy_visible) & jnp.any(ally_alive))
        move_valid = jnp.float32(jnp.any(ally_alive))
        verb_mask = jnp.array([
            1.0,                # noop always valid
            move_valid,         # move valid if any alive
            can_attack          # attack only if enemies exist
        ])

        # TARGET mask: which enemies can be targeted (fixed size 8)
        e_visible_alive = enemy_alive & state.enemy_visible
        
        target_mask = jnp.concatenate([
            e_visible_alive.astype(jnp.float32),
            jnp.zeros(self.max_enemy_slots - self.num_enemies)
        ])

        direction_mask = jnp.ones(8, dtype=jnp.float32)

        return {
            "who": who_mask,
            "verb": verb_mask,
            "direction": direction_mask,
            "target": target_mask,
        }

    def build_spatial_features(self, state: JaxSC2State) -> Dict[str, jnp.ndarray]:
        if not self.use_spatial_obs:
            return {}

        res = self.resolution
        smax_state = state.smax_state

        TYPE_MAP = jnp.arange(1, len(self.units) + 1, dtype=jnp.float32)
        ally_types = TYPE_MAP[smax_state.unit_types[:self.num_allies]]
        enemy_types = TYPE_MAP[smax_state.unit_types[self.num_allies:]]

        def get_grid_obs(positions, healths, alives, visibility, map_bounds, grid_res):
            clamped_pos = jnp.where(alives[:, None] & visibility[:, None], positions, -100.0)
            grid_pos = ((clamped_pos - map_bounds[:2]) / (map_bounds[2:] - map_bounds[:2]) * grid_res).astype(jnp.int32)
            in_view = (grid_pos >= 0).all(axis=-1) & (grid_pos < grid_res).all(axis=-1) & alives & visibility
            indices = grid_pos[:, 1] * grid_res + grid_pos[:, 0]
            chan_pos_flat = jnp.zeros(grid_res * grid_res)
            chan_hp_flat = jnp.zeros(grid_res * grid_res)
            chan_pos_flat = chan_pos_flat.at[indices].add(in_view.astype(jnp.float32))
            chan_hp_flat = chan_hp_flat.at[indices].add(healths * in_view)
            return chan_pos_flat.reshape(grid_res, grid_res), chan_hp_flat.reshape(grid_res, grid_res)

        def get_terrain_mask(map_bounds, grid_res):
            x_grid, y_grid = jnp.meshgrid(jnp.linspace(map_bounds[0], map_bounds[2], grid_res),
                                          jnp.linspace(map_bounds[1], map_bounds[3], grid_res),
                                          indexing='xy')
            return self.map_instance.terrain_mask(
                jnp.stack([x_grid, y_grid], axis=-1), self.map_width, self.map_height,
            ).astype(jnp.float32)

        def get_player_relative_grid(positions, alives, visibility_for_ally, map_bounds, grid_res, team_id=0):
            """Returns (res, res, 5) for categories bg=0, self=1, ally=2, neutral=3, enemy=4."""
            n = len(positions)
            grid = jnp.zeros((5, grid_res * grid_res))

            clamped_pos = jnp.where(alives[:, None] & visibility_for_ally[:, None], positions, -100.0)
            grid_pos = ((clamped_pos - map_bounds[:2]) / (map_bounds[2:] - map_bounds[:2]) * grid_res).astype(jnp.int32)
            in_view = (grid_pos >= 0).all(axis=-1) & (grid_pos < grid_res).all(axis=-1) & alives & visibility_for_ally
            indices = grid_pos[:, 1] * grid_res + grid_pos[:, 0]

            teams = smax_state.unit_teams
            cat = jnp.where(teams == team_id, 1, 4)
            for c in range(5):
                flat = jnp.zeros(grid_res * grid_res)
                flat = flat.at[indices].add(((cat == c) & in_view).astype(jnp.float32))
                grid = grid.at[c].set(flat)
            grid = grid.at[0].set((jnp.sum(grid[1:], axis=0) == 0).astype(jnp.float32))
            return jnp.moveaxis(grid.reshape(5, grid_res, grid_res), 0, -1)

        def get_visibility_grid(positions, alives, visibility_for_ally, map_bounds, grid_res):
            grid = jnp.zeros(grid_res * grid_res)
            clamped_pos = jnp.where(alives[:, None] & visibility_for_ally[:, None], positions, -100.0)
            grid_pos = ((clamped_pos - map_bounds[:2]) / (map_bounds[2:] - map_bounds[:2]) * grid_res).astype(jnp.int32)
            in_visible = (grid_pos >= 0).all(axis=-1) & (grid_pos < grid_res).all(axis=-1) & alives & visibility_for_ally
            indices = grid_pos[:, 1] * grid_res + grid_pos[:, 0]
            flat = jnp.zeros(grid_res * grid_res)
            flat = flat.at[indices].add(in_visible.astype(jnp.float32))
            return flat.reshape(grid_res, grid_res)

        m_bounds = jnp.array([0, 0, self.map_width, self.map_height])
        m_ally_p, m_ally_h = get_grid_obs(smax_state.unit_positions[:self.num_allies],
                                          smax_state.unit_health[:self.num_allies]/45.0,
                                          smax_state.unit_alive[:self.num_allies],
                                          jnp.ones(self.num_allies, dtype=jnp.bool_),
                                          m_bounds, res)
        m_enemy_p, m_enemy_h = get_grid_obs(smax_state.unit_positions[self.num_allies:],
                                            smax_state.unit_health[self.num_allies:]/45.0,
                                            smax_state.unit_alive[self.num_allies:],
                                            state.enemy_visible,
                                            m_bounds, res)
        m_beacon_p, _ = get_grid_obs(state.beacon_pos[None, :], jnp.array([1.0]), jnp.array([True]),
                                     jnp.array([True]), m_bounds, res)
        m_terrain = get_terrain_mask(m_bounds, res)

        m_vis = get_visibility_grid(smax_state.unit_positions[:self.num_allies],
                                    smax_state.unit_alive[:self.num_allies],
                                    jnp.ones(self.num_allies, dtype=jnp.bool_),
                                    m_bounds, res)
        m_player_rel = get_player_relative_grid(smax_state.unit_positions,
                                                 jnp.concatenate([smax_state.unit_alive[:self.num_allies],
                                                                  smax_state.unit_alive[self.num_allies:]]),
                                                 jnp.concatenate([jnp.ones(self.num_allies, dtype=jnp.bool_),
                                                                  state.enemy_visible]),
                                                 m_bounds, res, team_id=0)

        centroid = self._compute_centroid(smax_state.unit_positions[:self.num_allies], smax_state.unit_alive[:self.num_allies])
        cam_y = ((centroid[1] / self.map_height) * res).astype(jnp.int32)
        cam_x = ((centroid[0] / self.map_width) * res).astype(jnp.int32)
        half_cam = 2
        y_lo, y_hi = jnp.clip(cam_y-half_cam, 0, res-1), jnp.clip(cam_y+half_cam+1, 0, res)
        x_lo, x_hi = jnp.clip(cam_x-half_cam, 0, res-1), jnp.clip(cam_x+half_cam+1, 0, res)
        grid_y, grid_x = jnp.meshgrid(jnp.arange(res), jnp.arange(res), indexing="ij")
        m_camera = ((grid_y >= y_lo) & (grid_y < y_hi)
                    & (grid_x >= x_lo) & (grid_x < x_hi)).astype(jnp.float32)

        minimap = jnp.stack([
            m_terrain,
            m_vis,
            m_player_rel[..., 0],
            m_player_rel[..., 4],
            m_ally_p,
            m_enemy_p,
            m_camera,
        ], axis=0)


        view_size = 8.0
        s_bounds = jnp.array([centroid[0]-view_size, centroid[1]-view_size,
                              centroid[0]+view_size, centroid[1]+view_size])

        s_ally_p, s_ally_h = get_grid_obs(smax_state.unit_positions[:self.num_allies],
                                          smax_state.unit_health[:self.num_allies]/45.0,
                                          smax_state.unit_alive[:self.num_allies],
                                          jnp.ones(self.num_allies, dtype=jnp.bool_),
                                          s_bounds, res)
        s_enemy_p, s_enemy_h = get_grid_obs(smax_state.unit_positions[self.num_allies:],
                                            smax_state.unit_health[self.num_allies:]/45.0,
                                            smax_state.unit_alive[self.num_allies:],
                                            state.enemy_visible,
                                            s_bounds, res)
        s_beacon_p, _ = get_grid_obs(state.beacon_pos[None, :], jnp.array([1.0]), jnp.array([True]),
                                     jnp.array([True]), s_bounds, res)
        s_terrain = get_terrain_mask(s_bounds, res)

        s_vis = get_visibility_grid(smax_state.unit_positions[:self.num_allies],
                                    smax_state.unit_alive[:self.num_allies],
                                    jnp.ones(self.num_allies, dtype=jnp.bool_),
                                    s_bounds, res)

        s_player_rel = get_player_relative_grid(
            jnp.concatenate([smax_state.unit_positions[:self.num_allies],
                             smax_state.unit_positions[self.num_allies:]]),
            jnp.concatenate([smax_state.unit_alive[:self.num_allies],
                             smax_state.unit_alive[self.num_allies:]]),
            jnp.concatenate([jnp.ones(self.num_allies, dtype=jnp.bool_),
                             state.enemy_visible]),
            s_bounds, res, team_id=0)

        screen = jnp.zeros((17, res, res))
        screen = screen.at[0].set(s_terrain)
        screen = screen.at[1].set(s_vis)
        for c in range(5):
            screen = screen.at[2+c].set(s_player_rel[..., c])
        screen = screen.at[7].set(s_ally_p)
        screen = screen.at[8].set(s_enemy_p)
        screen = screen.at[9].set(s_beacon_p)
        screen = screen.at[10].set(s_ally_h)
        screen = screen.at[11].set(s_enemy_h)

        return {"screen": screen, "minimap": minimap}

class JaxSC2GymEnv(gym.Env):
    """
    A stateful Gymnasium wrapper for the JAX-native JaxSC2Env.
    Enables usage with standard RL libraries via gym.make().
    """
    def __init__(self, variant_name="V2_Base", **kwargs):
        super().__init__()
        self.env = JaxSC2Env(variant_name=variant_name, **kwargs)
        self.num_allies = self.env.num_allies
        self.num_enemies = self.env.num_enemies
        
        max_cooldown = float(jnp.max(
            (self.env.combat_cfg["type_windups"] + self.env.combat_cfg["type_cooldowns"] - 1)
            / self.env.combat_cfg["type_windups"]))
        enemy_slots = self.env.max_enemy_slots
        low = np.array([-1., -1., 0., 0.] * (self.num_allies + enemy_slots)
                       + [0.] * enemy_slots + [-1., -1., 0.], dtype=np.float32)
        high = np.array([1.] * (4 * self.num_allies) + [1., 1., 1., max_cooldown] * enemy_slots
                        + [1.] * (enemy_slots + 3), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([1 << self.num_allies, 3, 8, self.num_enemies])
        
        self._state = None
        self._rng = jax.random.PRNGKey(0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        
        self._rng, reset_rng = jax.random.split(self._rng)
        obs, self._state = self.env.reset(reset_rng)
        
        mask = self.env.get_action_mask(self._state)
        info = {"action_mask": jax.tree_util.tree_map(np.asarray, mask)}
        
        if isinstance(obs, dict):
            obs = obs["vector"]
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        who_val, verb, direction, target = action
        
        # JAX-safe bitmask decoding (no Python list comprehension)
        bit_masks = (who_val >> jnp.arange(self.num_allies)) & 1
        who_mask = bit_masks.astype(jnp.bool_)
        
        # Per-unit action with all units active (batched over envs)
        per_unit_action = PerUnitAction(
            who_mask=who_mask,
            verb=jnp.full(self.num_allies, verb, dtype=jnp.int32),
            direction=jnp.full(self.num_allies, direction, dtype=jnp.int32),
            target=jnp.full(self.num_allies, target, dtype=jnp.int32),
        )
        
        self._rng, step_rng = jax.random.split(self._rng)
        obs, self._state, reward, done, info = self.env.step(step_rng, self._state, per_unit_action)
        
        mask = self.env.get_action_mask(self._state)
        info["action_mask"] = mask
        info = jax.tree_util.tree_map(np.asarray, info)
        
        if isinstance(obs, dict):
            obs = obs["vector"]
            
        time_limit = bool(self._state.timestep >= self.env.max_steps)
        terminated = bool(info["nav_success"] | info["combat_success"]
                          | ~jnp.any(self._state.smax_state.unit_alive[:self.num_allies]))
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, bool(done) and time_limit and not terminated, info

    def render(self):
        pass
