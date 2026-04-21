import jax
import jax.numpy as jnp
import chex
import gymnasium as gym
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
    apply_high_fidelity_combat,
    FogState
)
from JaxSC2.env.units import MeleeUnit, RangedUnit, TankUnit

@dataclass
class ActionBuffer:
    verbs: jnp.ndarray      # (delay, N) - per-unit actions per delay slot
    directions: jnp.ndarray # (delay,)
    targets: jnp.ndarray    # (delay,)

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
    fog_state: FogState
    action_buffer: ActionBuffer
    enemy_visible: jnp.ndarray # (num_enemies,)
    timestep: int = 0

def init_action_buffer(delay, num_units):
    return ActionBuffer(
        verbs=jnp.zeros((delay, num_units), dtype=jnp.int32),
        directions=jnp.zeros((delay, num_units), dtype=jnp.int32),
        targets=jnp.zeros((delay, num_units), dtype=jnp.int32),
    )

def push_and_pop_actions(buffer: ActionBuffer, new_action):
    verbs = jnp.roll(buffer.verbs, shift=-1, axis=0)
    dirs  = jnp.roll(buffer.directions, shift=-1, axis=0)
    targs = jnp.roll(buffer.targets, shift=-1, axis=0)

    verbs = verbs.at[-1].set(new_action.verb)
    dirs  = dirs.at[-1].set(new_action.direction)
    targs = targs.at[-1].set(new_action.target)

    exec_action = PerUnitAction(
        who_mask=new_action.who_mask,
        verb=verbs[0],
        direction=dirs[0],
        target=targs[0],
    )
    return ActionBuffer(verbs, dirs, targs), exec_action

@dataclass(frozen=True)
class PerUnitAction:
    who_mask: jnp.ndarray   # Shape: (N,) - Boolean, which allies are active
    verb: jnp.ndarray       # Shape: (N,) - 0: NO_OP, 1: MOVE, 2: ATTACK
    direction: jnp.ndarray  # Shape: (N,) - 0-7 cardinal/intercardinal directions
    target: jnp.ndarray     # Shape: (N,) - 0 to (num_enemies-1), -1 = no target

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
    
    # Attack validity: only valid if at least one enemy exists
    enemy_alive = s.unit_alive[num_allies:]
    any_enemy_alive = jnp.any(enemy_alive)
    attack_valid = jnp.where(any_enemy_alive, 
                             jnp.ones(num_allies, dtype=jnp.bool_),
                             jnp.zeros(num_allies, dtype=jnp.bool_))
    verb_mask = verb_mask.at[:, 2].set(attack_valid)
    
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
    def __init__(self, variant_name="V2_Base", use_spatial_obs=False, resolution=64, latency_delay=0, **kwargs):
        self.map_instance = TwoBridgeMap()
        self.variant_config = self.map_instance.VARIANTS[variant_name]
        self.num_allies = self.variant_config.n_ally
        self.num_enemies = self.variant_config.n_enemy
        self.num_units = self.num_allies + self.num_enemies
        
        self.use_spatial_obs = use_spatial_obs
        self.resolution = resolution
        self.latency_delay = latency_delay
        
        self.enemy_ai = kwargs.pop("enemy_ai", False)
        self.enemy_mode = kwargs.pop("enemy_mode", "guard")
        self.mode_id = {
            "static": 0,
            "guard": 1,
            "patrol": 2,
            "aggressive": 3
        }.get(self.enemy_mode, 1)
        
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
        self.map_width = self.smax_env.map_width
        self.map_height = self.smax_env.map_height
        self.map_dimensions = jnp.array([self.map_width, self.map_height])
        
        # Team indices for combat engine (0: Ally, 1: Enemy)
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
        self.units = [MeleeUnit(), RangedUnit(), TankUnit()]
        
        # Aggregate stats into JAX arrays for simulation
        self.unit_type_hp = jnp.array([u.max_hp for u in self.units])
        self.unit_type_range = jnp.array([u.weapon.range for u in self.units])
        self.unit_type_speed = jnp.array([u.speed for u in self.units])
        self.unit_type_accel = jnp.array([u.accel for u in self.units])
        self.unit_type_damage = jnp.array([u.weapon.damage for u in self.units])
        self.type_radius = jnp.array([self.unit_radius] * 3)
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
        # Hoist constants for JIT efficiency
        self._angles = jnp.linspace(0, 2*jnp.pi, 8, endpoint=False)
        
        # Precompute NOT_SELF_MASK for collision functions (N x N)
        max_units = self.num_units
        i_idx = jnp.arange(max_units)[:, None]
        j_idx = jnp.arange(max_units)[None, :]
        self._not_self_mask = (i_idx != j_idx).astype(jnp.float32)
        
        # SB3-style Space definitions (Native)
        self.observation_space = DictWrapper({"shape": (63,)})
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
        rng_ally_region, rng_enemy_region, rng_beacon_region = jax.random.split(rng, 3)
        rng_ally_pos, rng_enemy_pos, rng_beacon_pos = jax.random.split(rng, 3)
        
        # 1. JIT-Compilable Region Selection (No Overlap)
        a_global_idx, e_global_idx, b_global_idx = self.map_instance.get_spawn_regions(
            rng_ally_region, 
            self.variant_config.layout_type
        )
        
        # 2. Sample dense clusters (Centroid + small offset)
        def sample_dense(rng, g_idx, num):
            rng_c, rng_o = jax.random.split(rng)
            bounds = self.map_instance.REGION_COORDS[g_idx]
            
            cx = jax.random.uniform(rng_c, (), minval=(bounds[0]+0.05)*self.map_width, maxval=(bounds[1]-0.05)*self.map_width)
            cy = jax.random.uniform(rng_c, (), minval=(bounds[2]+0.05)*self.map_height, maxval=(bounds[3]-0.05)*self.map_height)
            
            offsets = jax.random.uniform(rng_o, (num, 2), minval=-1.5, maxval=1.5)
            pos = jnp.stack([cx, cy], axis=-1) + offsets
            return jnp.clip(pos, 0.05, 31.95)

        ally_pos = sample_dense(rng_ally_pos, a_global_idx, self.num_allies)
        enemy_pos = sample_dense(rng_enemy_pos, e_global_idx, self.num_enemies)
        
        bounds_b = self.map_instance.REGION_COORDS[b_global_idx]
        bx = jax.random.uniform(rng_beacon_pos, (), minval=bounds_b[0]*self.map_width, maxval=bounds_b[1]*self.map_width)
        by = jax.random.uniform(rng_beacon_pos, (), minval=bounds_b[2]*self.map_height, maxval=bounds_b[3]*self.map_height)
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
        
        fog_state = FogState(
            last_seen_pos=enemy_pos,
            last_seen_alive=jnp.ones(self.num_enemies, dtype=jnp.bool_),
            last_seen_hp=smax_state.unit_health[self.num_allies:],
            last_seen_time=jnp.zeros(self.num_enemies, dtype=jnp.int32)
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
            proj_pos=jnp.zeros((32, 2)),
            proj_vel=jnp.zeros((32, 2)),
            proj_target=jnp.ones(32, dtype=jnp.int32) * -1,
            proj_damage=jnp.zeros(32),
            proj_active=jnp.zeros(32, dtype=jnp.bool_),
            proj_team=jnp.zeros(32, dtype=jnp.int32),
            fog_state=fog_state,
            action_buffer=action_buffer,
            enemy_visible=jnp.ones(self.num_enemies, dtype=jnp.bool_),
            timestep=0
        )
        
        return self.get_obs(state), state

    def step(self, rng: chex.PRNGKey, state: JaxSC2State, action: PerUnitAction) -> Tuple[Union[jnp.ndarray, Dict], JaxSC2State, float, bool, dict]:
        # 1. Action Latency Buffer
        new_buffer, exec_action = push_and_pop_actions(state.action_buffer, action)
        
        # Get Actions - Enemy AI or default
        new_key, rng_enemy = jax.random.split(rng)
        
        enemy_actions = jax.lax.cond(
            self.enemy_ai,
            lambda _: self.get_enemy_actions(state, rng_enemy),
            lambda _: jnp.ones(self.num_enemies, dtype=jnp.int32) * 4,
            operand=None
        )
        
        # Translate enemy actions to verb/dir — always (N_enemies,) inside vmap
        e_verb = jnp.where(enemy_actions < 4, 1, 0)
        e_verb = jnp.where(enemy_actions >= 5, 2, e_verb)
        e_dir = jnp.where(enemy_actions < 4, enemy_actions * 2, 0)

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
            vel_dir = vel / jnp.maximum(1e-6, v_norm)               # (N, 2)
            a_mag   = jnp.linalg.norm(accel, axis=-1, keepdims=True) # (N, 1)
            turn_rate = self.combat_cfg.get("turn_rate", 0.4)
            blended = (1.0 - turn_rate) * (vel_dir * a_mag) + turn_rate * accel
            return jnp.where(has_vel, blended, accel)

        accel_vecs = apply_inertia(accel_vecs, state.unit_velocities)  # (N_total, 2)
        
        # 4. Physics Integration
        next_pos, next_vel = integrate_velocity(
            state.smax_state.unit_positions, 
            state.unit_velocities, 
            accel_vecs, 
            self.unit_type_speed[state.smax_state.unit_types]
        )
        
        # 5. Dual-Layer Collisions
        next_pos = apply_mass_collisions(
            next_pos,
            state.smax_state.unit_alive,
            state.smax_state.unit_types,
            self.type_radius,
            self.type_mass
        )
        next_pos = apply_hard_collisions(
            next_pos,
            state.smax_state.unit_alive,
            unit_radius=0.3,
            stiffness=1.2
        )
        
        # Terrain Constraints (ally units only)
        next_pos = self.map_instance.enforce_constraints(
            next_pos, 
            state.smax_state.unit_positions,
            self.map_width, 
            self.map_height,
            self.unit_teams
        )
        
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
            leash_extra=2.5
        )
        
        # Per-unit target override from action
        manual_targets = jnp.where(
            (action.verb == 2) & action.who_mask, 
            action.target, 
            -1
        )
        next_targets = current_targets.at[:self.num_allies].set(
            jnp.where(manual_targets >= 0, manual_targets, current_targets[:self.num_allies])
        )
        
        # 7. High-Fidelity Combat
        unit_armor_per_unit = self.unit_armor[state.smax_state.unit_types]  # (N_total,)
        proj_st = (state.proj_pos, state.proj_vel, state.proj_target, state.proj_damage, state.proj_active, state.proj_team)
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
        
        dists_to_enemies = jnp.linalg.norm(
            ally_pos[:, None, :] - enemy_pos[None, :, :],
            axis=-1
        )
        visibility = dists_to_enemies < self.vision_radius
        new_visibility = jnp.any(visibility, axis=0) & enemy_alive
        
        # 10. Compute Rewards
        dims = jnp.array([self.map_width, self.map_height])
        ally_pos_obs = next_pos[:self.num_allies]
        enemy_pos_obs = next_pos[self.num_allies:]
        ally_alive_obs = next_alive[:self.num_allies]
        enemy_alive_obs = next_alive[self.num_allies:]
        
        # Navigation Reward (Distance to Beacon)
        dist_to_beacon = jnp.linalg.norm((ally_pos_obs - state.beacon_pos) / dims, axis=-1)
        mean_dist = jnp.sum(dist_to_beacon * ally_alive_obs) / jnp.maximum(jnp.sum(ally_alive_obs), 1.0)
        nav_reward = (state.prev_mean_dist - mean_dist) * 2.0
        
        # Combat shaping: bidirectional distance to enemy centroid (Panda et al., ICML 2026)
        alive_enemy_float = enemy_alive_obs.astype(jnp.float32)
        combat_centroid = jnp.sum(enemy_pos_obs * alive_enemy_float[:, None], axis=0) / jnp.maximum(jnp.sum(alive_enemy_float), 1.0)
        dist_to_combat_centroid = (
            jnp.sum(jnp.linalg.norm((ally_pos_obs - combat_centroid) / dims, axis=-1) * ally_alive_obs.astype(jnp.float32), axis=0)
            / jnp.maximum(jnp.sum(ally_alive_obs.astype(jnp.float32)), 1.0)
        )
        combat_shaping = (state.prev_enemy_dist - dist_to_combat_centroid) * 0.5
        
        # Combat Reward (HP Damage + Kills)
        current_enemy_hp = jnp.sum(next_hp[self.num_allies:] * enemy_alive_obs)
        enemy_dmg_reward = (state.prev_enemy_health - current_enemy_hp) * 0.01 * 0.5
        
        current_ally_hp = jnp.sum(next_hp[:self.num_allies] * ally_alive_obs)
        ally_dmg_penalty = (state.prev_ally_health - current_ally_hp) * 0.01 * 0.5
        
        # Kill Bonuses
        enemy_killed = (jnp.sum(state.smax_state.unit_alive[self.num_allies:]) - jnp.sum(enemy_alive_obs)) * 0.2
        
        # Total Reward (step-level)
        total_reward = nav_reward + combat_shaping + enemy_dmg_reward - ally_dmg_penalty + enemy_killed
        
        # Termination
        beacon_reached = jnp.any((dist_to_beacon < 0.05) & ally_alive_obs)
        enemies_dead = jnp.sum(enemy_alive_obs) == 0
        allies_dead = jnp.sum(ally_alive_obs) == 0
        done = beacon_reached | enemies_dead | allies_dead | (state.timestep >= self.max_steps)
        
        # Terminal Rewards (Panda et al., ICML 2026)
        terminal_reward = jnp.where(beacon_reached, 25.0,
            jnp.where(enemies_dead, 10.0,
                jnp.where(allies_dead, -10.0,
                    jnp.where(done, -15.0, 0.0))))
        total_reward = total_reward + terminal_reward
        
        # 11. Build Info
        info = {
            "beacon_reached": beacon_reached,
            "nav_success": beacon_reached,
            "combat_success": enemies_dead,
            "enemies_killed": enemy_killed,
            "nav_reward": nav_reward,
            "combat_shaping": combat_shaping,
            "combat_reward": enemy_dmg_reward - ally_dmg_penalty + enemy_killed,
            "terminal_reward": terminal_reward,
        }
        
        # 12. Build Next State
        next_state = JaxSC2State(
            smax_state=SmaxState(
                unit_positions=next_pos,
                unit_alive=next_alive,
                unit_teams=self.unit_teams,
                unit_health=next_hp,
                unit_types=state.smax_state.unit_types,
                unit_weapon_cooldowns=jnp.zeros_like(state.smax_state.unit_weapon_cooldowns),
                prev_movement_actions=jnp.zeros_like(state.smax_state.prev_movement_actions),
                prev_attack_actions=jnp.zeros_like(state.smax_state.prev_attack_actions),
                time=state.timestep + 1,
                terminal=False,
            ),
            beacon_pos=state.beacon_pos,
            prev_mean_dist=mean_dist,
            prev_enemy_health=current_enemy_hp,
            prev_ally_health=current_ally_hp,
            prev_enemy_dist=dist_to_combat_centroid,
            unit_velocities=next_vel,
            attack_timers=next_timers,
            persistent_targets=next_targets,
            proj_pos=next_proj_st[0],
            proj_vel=next_proj_st[1],
            proj_target=next_proj_st[2],
            proj_damage=next_proj_st[3],
            proj_active=next_proj_st[4],
            proj_team=next_proj_st[5],
            fog_state=next_fog_state,
            action_buffer=new_buffer,
            enemy_visible=new_visibility,
            timestep=state.timestep + 1,
        )
        
        return self.get_obs(next_state), next_state, total_reward, done, info

    def get_enemy_actions(self, state: JaxSC2State, rng=None) -> jnp.ndarray:
        """Enemy AI: simple directional behaviors. Depends on state so outer vmap batches correctly."""
        enemy_alive = state.smax_state.unit_alive[self.num_allies:]  # (N_enemies,)
        # mode_id: 0=static(4), 1=guard(6/left), 2=patrol(6), 3=aggressive(6)
        base_action = jnp.where(
            self.mode_id == 0,
            jnp.ones(self.num_enemies, dtype=jnp.int32) * 4,   # static: stay
            jnp.ones(self.num_enemies, dtype=jnp.int32) * 6    # all others: move left
        )
        # Dead enemies stay
        return jnp.where(enemy_alive, base_action, jnp.ones_like(base_action) * 4)

    def get_obs(self, state: JaxSC2State) -> Union[jnp.ndarray, Dict]:
        vector_obs = self.build_vector_obs(state)
        if not self.use_spatial_obs:
            return vector_obs
        
        return {
            "vector": vector_obs,
            **self.build_spatial_features(state)
        }

    def build_vector_obs(self, state: JaxSC2State, max_enemies=8) -> jnp.ndarray:
        """
        SC2-semantics-aligned observation (63-dim, invariant across variants)
        """
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
        can_attack = jnp.float32(jnp.any(enemy_alive))
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
            jnp.zeros(8 - self.num_enemies)
        ])

        direction_mask = jnp.ones(8, dtype=jnp.float32)

        return {
            "who": who_mask,
            "verb": verb_mask,
            "direction": direction_mask,
            "target": target_mask,
        }

    def build_spatial_features(self, state: JaxSC2State) -> Dict[str, jnp.ndarray]:
        """
        Mimics Experiment 3's 17-channel feature_screen and 7-channel feature_minimap.
        """
        if not self.use_spatial_obs:
            return {}
            
        res = self.resolution
        smax_state = state.smax_state
        
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
            x_grid, y_grid = jnp.meshgrid(jnp.linspace(map_bounds[0]/self.map_width, map_bounds[2]/self.map_width, grid_res),
                                          jnp.linspace(map_bounds[1]/self.map_height, map_bounds[3]/self.map_height, grid_res),
                                          indexing='xy')
            in_cliff_x = (x_grid > 0.45) & (x_grid < 0.55)
            on_bridge_1 = (y_grid > 0.2) & (y_grid < 0.3)
            on_bridge_2 = (y_grid > 0.7) & (y_grid < 0.8)
            return (in_cliff_x & ~(on_bridge_1 | on_bridge_2)).astype(jnp.float32)

        # 1. Global Minimap (7 Channels)
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
        minimap = jnp.stack([m_terrain, m_ally_p, m_ally_h, m_enemy_p, m_enemy_h, m_beacon_p, jnp.zeros((res, res))])

        # 2. Camera-Locked Screen (17 Channels)
        centroid = self._compute_centroid(smax_state.unit_positions[:self.num_allies], smax_state.unit_alive[:self.num_allies])
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
        
        screen = jnp.zeros((17, res, res))
        screen = screen.at[0].set(s_terrain)
        screen = screen.at[5].set(s_ally_p)
        screen = screen.at[6].set(s_enemy_p)
        screen = screen.at[7].set(s_beacon_p)
        screen = screen.at[8].set(s_ally_h)
        screen = screen.at[9].set(s_enemy_h)

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
        
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(63,), dtype=jnp.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 3, 8, 8])
        
        self._state = None
        self._rng = jax.random.PRNGKey(0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        
        self._rng, reset_rng = jax.random.split(self._rng)
        obs, self._state = self.env.reset(reset_rng)
        
        mask = self.env.get_action_mask(self._state)
        info = {"action_mask": mask}
        
        if isinstance(obs, dict):
            return obs["vector"], info
        return obs, info

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
        
        if isinstance(obs, dict):
            obs = obs["vector"]
            
        return obs, float(reward), bool(done), False, info

    def render(self):
        pass