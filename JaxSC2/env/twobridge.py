import jax
import jax.numpy as jnp
import chex
import gymnasium as gym
from typing import Tuple, Dict, Union, Any, Optional
from flax.struct import dataclass

from jaxmarl.environments.smax.smax_env import SMAX, State as SmaxState
from JaxSC2.maps.layouts import VARIANTS, REGION_COORDS, LEFT_REGIONS, RIGHT_REGIONS
from JaxSC2.maps.constraints import enforce_bridge_terrain
from JaxSC2.maps.rewards import compute_rewards_and_dones
from JaxSC2.env.mechanics import (
    apply_hard_collisions, 
    integrate_velocity,
    update_persistent_targets,
    compute_visibility, 
    apply_high_fidelity_combat
)

@dataclass
class TwoBridgeState:
    smax_state: SmaxState            
    beacon_pos: jnp.ndarray          
    prev_mean_dist: jnp.float32      
    prev_enemy_health: jnp.float32   
    prev_ally_health: jnp.float32    
    prev_enemy_centroid_dist: jnp.float32 
    timestep: jnp.int32              
    variant_id: jnp.int32            
    attack_timers: jnp.ndarray       # (num_allies + num_enemies,)
    enemy_visible: jnp.ndarray       # (num_enemies,)
    key: chex.PRNGKey                # PRNG key for stochastic behaviors
    unit_velocities: jnp.ndarray     # (num_allies + num_enemies, 2)
    persistent_targets: jnp.ndarray  # (num_allies + num_enemies,)
    # --- Projectile Buffer (fixed size 32) ---
    proj_pos: jnp.ndarray            # (32, 2)
    proj_vel: jnp.ndarray            # (32, 2)
    proj_target: jnp.ndarray         # (32,) index of target unit
    proj_damage: jnp.ndarray         # (32,)
    proj_active: jnp.ndarray         # (32,) bool mask
    proj_team: jnp.ndarray           # (32,) team index (0:Ally, 1:Enemy)

@dataclass
class CentralAction:
    who_mask: jnp.ndarray   # Shape: (5,) - Boolean [0, 1]
    verb: jnp.int32         # Shape: () - 0: No-op, 1: Move, 2: Attack
    direction: jnp.int32    # Shape: () - 0-3 cardinal directions
    target: jnp.int32       # Shape: () - 0 to (N_E - 1) enemy index

def get_action_masks(state: TwoBridgeState, num_allies: int) -> jnp.ndarray:
    """
    Returns shape (17,) boolean mask for valid actions.
    - 0: Stop (Always valid)
    - 1-8: Move (Always valid for now)
    - 9-16: Attack (Valid if enemy exists, is alive, and is visible)
    """
    # Visibility and Aliveness
    enemy_alive = state.smax_state.unit_alive[num_allies:] # (num_enemies,)
    enemy_visible = state.enemy_visible # (num_enemies,)
    
    attack_valid = enemy_alive & enemy_visible # (num_enemies,)
    
    # Pad to max 8 enemies for consistent action space (17-dim)
    pad_len = 8 - attack_valid.shape[0]
    attack_valid_padded = jnp.concatenate([
        attack_valid, 
        jnp.zeros(pad_len, dtype=jnp.bool_)
    ])
    
    stop_valid = jnp.ones(1, dtype=jnp.bool_)
    move_valid = jnp.ones(8, dtype=jnp.bool_)
    
    return jnp.concatenate([stop_valid, move_valid, attack_valid_padded])

class TwoBridgeEnv:
    """
    JAX-Native Two-Bridge Environment Suite for Centralized RL.
    
    This environment wraps SMAX (StarCraft Multi-Agent Challenge in JAX) to provide
    a high-throughput, physically verified benchmark for arbitration between 
    navigation and combat objectives.
    
    TECHNICAL HANDOFF SPECS:
    -----------------------
    1. ACTION SPACE (Discrete):
       - Verb: 0: No-op, 1: Move, 2: Attack.
       - Direction (if Move): 0: North (+y), 1: East (+x), 2: South (-y), 3: West (-x).
       - Target (if Attack): Index 0 to (Max Enemies - 1).
    
    2. OBSERVATION SPACE (Vector - 63-dim):
       - 5 Allies (4 features: [rel_x, rel_y, health, cooldown]): 20-dim.
       - 8 Enemies (4 features: [rel_x, rel_y, health, cooldown]): 32-dim.
       - Enemy Mask (1.0 if enemy exists, 0.0 if padded): 8-dim.
       - Global (Beacon [x,y], Timestep): 3-dim.
    
    3. SPATIAL FEATURES (Experiment 3):
       - Minimap (7 channels): [0:Terrain, 1:AllyPos, 2:AllyHP, 3:EnemyPos, 4:EnemyHP, 5:BeaconPos, 6:Empty]
       - Screen (17 channels): [0:Terrain, 5:AllyPos, 6:EnemyPos, 7:BeaconPos, 8:AllyHP, 9:EnemyHP, others:0]
         (Aligned with PySC2 feature layer offsets)
    
    4. PHYSICS & REWARDS:
       - SMAX Bypass: Internal SMAX rewards and terminal signals are COMPLETELY IGNORED.
         Logic transitions are governed by jax_twobridge.envs.rewards and constraints.
       - Collision: Uses segment-crossing detection for the central cliff (x=0.5).
       - Reward Scaling: 2.0x weight for Navigation Shaping, 0.5x for Combat Shaping.
    """
    def __init__(self, variant_name="V2_Base", use_spatial_obs=False, resolution=64, **kwargs):
        self.variant_config = VARIANTS[variant_name]
        self.num_allies = self.variant_config.n_ally
        self.num_enemies = self.variant_config.n_enemy
        
        self.enemy_ai = kwargs.pop("enemy_ai", False)
        self.enemy_mode = kwargs.pop("enemy_mode", "guard")
        self.mode_id = {
            "static": 0,
            "guard": 1,
            "patrol": 2,
            "aggressive": 3
        }.get(self.enemy_mode, 1)
        
        # Initialize SMAX
        # If custom armor/combat mechanics are enabled, we disable SMAX's internal damage logic
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
        
        # Spatial Feature Config
        self.use_spatial_obs = use_spatial_obs
        self.resolution = resolution # Default 64x64 to match paper
        
        # Advanced Mechanics Config (Toggleable)
        self.enable_collision = kwargs.get("enable_collision", True)
        self.enable_fow = kwargs.get("enable_fow", False)
        self.enable_armor = kwargs.get("enable_armor", True)
        
        self.unit_radius = 0.6
        self.vision_radius = 6.0
        
        # Unit Heterogeneity Constants (Melee, Ranged, Tank)
        self.unit_type_hp = jnp.array([100.0, 45.0, 200.0])
        self.unit_type_range = jnp.array([1.5, 6.0, 4.0])
        self.unit_type_speed = jnp.array([0.3, 0.25, 0.18])
        self.unit_type_accel = jnp.array([0.1, 0.08, 0.05])
        self.unit_type_damage = jnp.array([8.0, 5.0, 4.0])

        self.combat_cfg = {
            "base_damage": 5.0,
            "attack_range": 6.0,
            "windup": 2, # steps
            "backswing": 1, # post-fire lock
            "cooldown": 8, # steps recovery
            "turn_rate": 0.4, # inertia lerp factor
            "damage_matrix": jnp.array([
                [1.0, 1.5, 0.7],  # Melee (0) vs (M, R, T) 
                [0.8, 1.0, 1.5],  # Ranged (1) vs (M, R, T)
                [1.5, 0.7, 1.0]   # Tank (2)   vs (M, R, T)
            ]),
            "aggression_radius": 8.0,
            "type_ranges": self.unit_type_range,
            "type_damages": self.unit_type_damage
        }

    def reset(self, rng: chex.PRNGKey) -> Tuple[Union[jnp.ndarray, Dict], TwoBridgeState]:
        rng_ally_region, rng_enemy_region, rng_beacon_region = jax.random.split(rng, 3)
        rng_ally_pos, rng_enemy_pos, rng_beacon_pos = jax.random.split(rng, 3)
        
        # 1. JIT-Compilable Region Selection (No Overlap)
        def select_base():
            a_idx = jax.random.choice(rng_ally_region, LEFT_REGIONS)
            b_idx = jax.random.choice(rng_beacon_region, RIGHT_REGIONS)
            e_mask = (RIGHT_REGIONS != b_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(e_mask), 1.0)
            e_probs = e_mask / denom
            e_idx = jax.random.choice(rng_enemy_region, RIGHT_REGIONS, p=e_probs)
            return a_idx, e_idx, b_idx
            
        def select_combat():
            b_idx = jax.random.choice(rng_beacon_region, LEFT_REGIONS)
            a_idx = jax.random.choice(rng_ally_region, RIGHT_REGIONS)
            e_mask = (RIGHT_REGIONS != a_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(e_mask), 1.0)
            e_probs = e_mask / denom
            e_idx = jax.random.choice(rng_enemy_region, RIGHT_REGIONS, p=e_probs)
            return a_idx, e_idx, b_idx
            
        def select_navigate():
            e_idx = jax.random.choice(rng_enemy_region, LEFT_REGIONS)
            a_idx = jax.random.choice(rng_ally_region, RIGHT_REGIONS)
            b_mask = (RIGHT_REGIONS != a_idx).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(b_mask), 1.0)
            b_probs = b_mask / denom
            b_idx = jax.random.choice(rng_beacon_region, RIGHT_REGIONS, p=b_probs)
            return a_idx, e_idx, b_idx
            
        a_global_idx, e_global_idx, b_global_idx = jax.lax.switch(
            self.variant_config.layout_type,
            [select_base, select_combat, select_navigate]
        )
        
        # 2. Sample dense clusters (Centroid + small offset)
        def sample_dense(rng, g_idx, num):
            rng_c, rng_o = jax.random.split(rng)
            bounds = REGION_COORDS[g_idx]
            
            # Centroid (within region, leaving margin for offsets)
            cx = jax.random.uniform(rng_c, (), minval=(bounds[0]+0.05)*self.map_width, maxval=(bounds[1]-0.05)*self.map_width)
            cy = jax.random.uniform(rng_c, (), minval=(bounds[2]+0.05)*self.map_height, maxval=(bounds[3]-0.05)*self.map_height)
            
            # Offsets (tight cluster of ~1.5m radius)
            offsets = jax.random.uniform(rng_o, (num, 2), minval=-1.5, maxval=1.5)
            # Ensure within map boundaries [0.05, 31.95]
            pos = jnp.stack([cx, cy], axis=-1) + offsets
            return jnp.clip(pos, 0.05, 31.95)

        ally_pos = sample_dense(rng_ally_pos, a_global_idx, self.num_allies)
        enemy_pos = sample_dense(rng_enemy_pos, e_global_idx, self.num_enemies)
        
        # Beacon doesn't need a cluster
        bounds_b = REGION_COORDS[b_global_idx]
        bx = jax.random.uniform(rng_beacon_pos, (), minval=bounds_b[0]*self.map_width, maxval=bounds_b[1]*self.map_width)
        by = jax.random.uniform(rng_beacon_pos, (), minval=bounds_b[2]*self.map_height, maxval=bounds_b[3]*self.map_height)
        beacon_pos = jnp.array([bx, by])
        
        # 3. Construct SMAX State
        unit_positions = jnp.concatenate([ally_pos, enemy_pos])
        unit_teams = jnp.zeros((self.num_allies + self.num_enemies,), dtype=jnp.int32)
        unit_teams = unit_teams.at[self.num_allies:].set(1)
        
        # Unit Heterogeneity: [0: Melee, 1: Ranged, 2: Tank]
        # Distribution: robust cycling for any count
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
        
        state = TwoBridgeState(
            smax_state=smax_state,
            beacon_pos=beacon_pos,
            prev_mean_dist=initial_mean_dist,
            prev_enemy_health=jnp.sum(unit_health[self.num_allies:]),
            prev_ally_health=jnp.sum(unit_health[:self.num_allies]),
            prev_enemy_centroid_dist=initial_enemy_dist,
            timestep=0,
            variant_id=self.variant_config.id,
            attack_timers=jnp.zeros((self.num_allies + self.num_enemies,), dtype=jnp.int32),
            enemy_visible=jnp.ones((self.num_enemies,), dtype=jnp.bool_), # All visible initially
            key=rng,
            unit_velocities=jnp.zeros((self.num_allies + self.num_enemies, 2)),
            persistent_targets=jnp.ones((self.num_allies + self.num_enemies,), dtype=jnp.int32) * -1,
            proj_pos=jnp.zeros((32, 2)),
            proj_vel=jnp.zeros((32, 2)),
            proj_target=jnp.ones((32,), dtype=jnp.int32) * -1,
            proj_damage=jnp.zeros((32,)),
            proj_active=jnp.zeros((32,), dtype=jnp.bool_),
            proj_team=jnp.zeros((32,), dtype=jnp.int32)
        )
        
        return self.get_obs(state), state

    def step(self, rng: chex.PRNGKey, state: TwoBridgeState, action: CentralAction) -> Tuple[Union[jnp.ndarray, Dict], TwoBridgeState, jnp.ndarray, jnp.ndarray, Dict]:
        # 1. Update RNG for stochastic behaviors
        new_key, action_rng = jax.random.split(state.key)
        
        # 2. Get Ally and Enemy Actions
        ally_actions = self.translate_action(action)[:self.num_allies]
        enemy_actions = jax.lax.cond(
            self.enemy_ai,
            lambda _: self.get_enemy_actions(state, action_rng),
            lambda _: jnp.ones(self.num_enemies, dtype=jnp.int32) * 4, # Stop
            operand=None
        )
        actions = jnp.concatenate([ally_actions, enemy_actions])

        # 3. Continuous Physics Integration (Double-Integrator)
        def decode_accel(i, a):
            is_move = a < 8 # 8 directions (0-7)
            accel_mag = self.unit_type_accel[state.smax_state.unit_types[i]]
            
            # 8-Directional Decoding (0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW)
            angle = (a % 8) * (jnp.pi / 4.0)
            vec = jnp.array([jnp.sin(angle), jnp.cos(angle)])
            
            # Movement Lock: units cannot move during wind-up OR backswing (Commitment)
            # Wind-up + Backswing phase: Timer > 0 AND Timer <= (Windup + Backswing)
            windup = self.combat_cfg["windup"]
            backswing = self.combat_cfg.get("backswing", 1)
            is_locked = (state.attack_timers[i] > 0) & (state.attack_timers[i] <= (windup + backswing))
            return vec * accel_mag * is_move * (~is_locked)

        accel_vecs = jax.vmap(decode_accel)(jnp.arange(self.num_allies + self.num_enemies), actions)
        
        # 3.1 Turn-Rate Inertia (Lerp acceleration with current velocity direction)
        def apply_inertia(i, accel, vel):
            v_norm = jnp.linalg.norm(vel)
            # If moving, blend acceleration with velocity direction
            # Otherwise allow full acceleration from rest
            has_vel = v_norm > 1e-3
            vel_dir = vel / jnp.maximum(1e-6, v_norm)
            
            # Target direction
            a_mag = jnp.linalg.norm(accel)
            
            turn_rate = self.combat_cfg.get("turn_rate", 0.4)
            # Blended accel
            blended_accel = (1.0 - turn_rate) * (vel_dir * a_mag) + turn_rate * accel
            return jnp.where(has_vel, blended_accel, accel)
            
        accel_vecs = jax.vmap(apply_inertia)(jnp.arange(self.num_allies + self.num_enemies), accel_vecs, state.unit_velocities)
        max_speeds = self.unit_type_speed[state.smax_state.unit_types]
        
        next_pos, next_vel = jax.vmap(integrate_velocity)(
            state.smax_state.unit_positions,
            state.unit_velocities,
            accel_vecs,
            max_speeds
        )

        # 4. Physical Constrains (Collisions & Terrain)
        # Pass 1: Hard Collision
        next_pos = jax.lax.cond(
            self.enable_collision,
            lambda p: apply_hard_collisions(p, state.smax_state.unit_alive, self.unit_radius),
            lambda p: p,
            next_pos
        )
        
        # Pass 2: Terrain
        next_pos = enforce_bridge_terrain(
            next_pos, 
            state.smax_state.unit_positions,
            self.map_width, 
            self.map_height
        )
        
        # Pass 3: Stabilizing Collision
        next_pos = jax.lax.cond(
            self.enable_collision,
            lambda p: apply_hard_collisions(p, state.smax_state.unit_alive, self.unit_radius),
            lambda p: p,
            next_pos
        )

        # Pass 4: Final Terrain Check (Security)
        next_pos = enforce_bridge_terrain(
            next_pos, 
            state.smax_state.unit_positions,
            self.map_width, 
            self.map_height
        )
        
        next_smax_state = state.smax_state.replace(unit_positions=next_pos)

        # 5. Target Persistence & Focus Fire Logic
        # a. Integrate manual actions (Attack overrides persistence)
        is_attack = (actions >= 5) & (actions <= 12)
        raw_target = actions - 5
        # Map to global indices: Allies target [5...12], Enemies target [0...4]
        mapped_target = jnp.where(
            jnp.arange(self.num_allies + self.num_enemies) < self.num_allies,
            raw_target + self.num_allies,
            raw_target
        )
        current_targets = jnp.where(is_attack, mapped_target, state.persistent_targets)

        all_ranges = self.unit_type_range[state.smax_state.unit_types]
        
        # b. Update Persistence (Normalize to POOL indices first)
        ally_targets = update_persistent_targets(
            current_targets[:self.num_allies] - self.num_allies,
            next_pos[:self.num_allies],
            state.smax_state.unit_alive[:self.num_allies],
            next_pos[self.num_allies:],
            state.smax_state.unit_alive[self.num_allies:],
            all_ranges[:self.num_allies]
        )
        
        enemy_targets = update_persistent_targets(
            current_targets[self.num_allies:],
            next_pos[self.num_allies:],
            state.smax_state.unit_alive[self.num_allies:],
            next_pos[:self.num_allies],
            state.smax_state.unit_alive[:self.num_allies],
            all_ranges[self.num_allies:]
        )
        
        new_targets = jnp.concatenate([
            ally_targets + self.num_allies, 
            enemy_targets
        ])

        # 6. High-Fidelity Combat (Ballistics / Cooldowns / Backswing)
        proj_state = (state.proj_pos, state.proj_vel, state.proj_target, state.proj_damage, state.proj_active, state.proj_team)
        new_health, new_alive, new_timers, next_proj_state = jax.lax.cond(
            self.enable_armor,
            lambda _: apply_high_fidelity_combat(
                next_smax_state.unit_health,
                next_smax_state.unit_alive,
                next_smax_state.unit_positions,
                next_smax_state.unit_types,
                next_smax_state.unit_teams,
                state.attack_timers,
                new_targets,
                proj_state,
                self.combat_cfg
            ),
            lambda _: (next_smax_state.unit_health, next_smax_state.unit_alive, state.attack_timers, proj_state),
            operand=None
        )
        
        next_smax_state = next_smax_state.replace(
            unit_health=new_health,
            unit_alive=new_alive
        )

        # 7. Fog of War
        enemy_visible = jax.lax.cond(
            self.enable_fow,
            lambda _: compute_visibility(
                next_smax_state.unit_positions[:self.num_allies],
                next_smax_state.unit_alive[:self.num_allies],
                next_smax_state.unit_positions[self.num_allies:],
                self.vision_radius
            ),
            lambda _: jnp.ones((self.num_enemies,), dtype=jnp.bool_),
            operand=None
        )
        
        # 8. Rewards & Termination
        reward, done, next_mean_dist, next_enemy_hp, next_ally_hp, next_enemy_dist = compute_rewards_and_dones(
            state, next_smax_state, self.num_allies, self.num_enemies, self.map_width, self.map_height
        )
        
        # Outcome flags for telemetry
        beacon_reached = jnp.any((jnp.linalg.norm((next_smax_state.unit_positions[:self.num_allies] - state.beacon_pos) / jnp.array([self.map_width, self.map_height]), axis=-1) < 0.05) & next_smax_state.unit_alive[:self.num_allies])
        enemies_all_dead = jnp.sum(next_smax_state.unit_alive[self.num_allies:]) == 0
        allies_all_dead = jnp.sum(next_smax_state.unit_alive[:self.num_allies]) == 0
        timeout = state.timestep >= self.max_steps
        
        win = (beacon_reached | enemies_all_dead) & ~allies_all_dead
        loss = allies_all_dead | timeout
        
        info = {
            "win": win,
            "loss": loss,
            "timeout": timeout,
            "beacon_reached": beacon_reached,
            "enemies_killed": enemies_all_dead
        }

        next_state = state.replace(
            smax_state=next_smax_state,
            prev_mean_dist=next_mean_dist,
            prev_enemy_health=next_enemy_hp,
            prev_ally_health=next_ally_hp,
            prev_enemy_centroid_dist=next_enemy_dist,
            timestep=state.timestep + 1,
            attack_timers=new_timers,
            enemy_visible=enemy_visible,
            key=new_key,
            unit_velocities=next_vel,
            persistent_targets=new_targets,
            proj_pos=next_proj_state[0],
            proj_vel=next_proj_state[1],
            proj_target=next_proj_state[2],
            proj_damage=next_proj_state[3],
            proj_active=next_proj_state[4],
            proj_team=next_proj_state[5]
        )
        
        return self.get_obs(next_state), next_state, reward, done, info

    def get_obs(self, state: TwoBridgeState) -> Union[jnp.ndarray, Dict]:
        vector_obs = self.build_vector_obs(state)
        if not self.use_spatial_obs:
            return vector_obs
        
        return {
            "vector": vector_obs,
            **self.build_spatial_features(state)
        }

    def build_vector_obs(self, state: TwoBridgeState, max_enemies=8) -> jnp.ndarray:
        """
        SC2-semantics-aligned observation (63-dim, invariant across variants)
        """
        s = state.smax_state
        dims = self.map_dimensions

        ally_pos = s.unit_positions[:self.num_allies]
        enemy_pos = s.unit_positions[self.num_allies:]

        ally_alive = s.unit_alive[:self.num_allies]
        enemy_alive = s.unit_alive[self.num_allies:]

        # --- 1. Reference frame (SC2-style: relative to ally centroid) ---
        alive_mask = ally_alive[:, None]
        centroid = jnp.sum(ally_pos * alive_mask, axis=0) / jnp.maximum(jnp.sum(ally_alive), 1.0)

        # --- 2. Allies ---
        f_rel_pos = (ally_pos - centroid) / dims
        f_hp = s.unit_health[:self.num_allies] / self.smax_env.unit_type_health[0]
        # Use OUR attack timers for cooldown (normalized by windup and clipped)
        f_cd = jnp.clip(state.attack_timers[:self.num_allies].astype(jnp.float32) / jnp.maximum(1.0, self.combat_cfg["windup"]), 0.0, 1.0)
        
        # SC2 Reality: Dead units vanish from the observation
        f_feats = jnp.concatenate([f_rel_pos, f_hp[:, None], f_cd[:, None]], axis=-1)
        f_feats = (f_feats * ally_alive[:, None]).flatten()

        # --- 3. Enemies ---
        e_rel_pos = (enemy_pos - centroid) / dims
        e_hp = s.unit_health[self.num_allies:] / self.smax_env.unit_type_health[0]
        # Normalized and clipped cooldown
        e_cd = jnp.clip(state.attack_timers[self.num_allies:].astype(jnp.float32) / jnp.maximum(1.0, self.combat_cfg["windup"]), 0.0, 1.0)

        e_feats = jnp.concatenate([e_rel_pos, e_hp[:, None], e_cd[:, None]], axis=-1)

        # Apply FOW masking ONLY to features (consistent with visibility)
        e_feats = e_feats * state.enemy_visible[:, None]

        # Pad to max_enemies
        e_feats = jnp.pad(e_feats, ((0, max_enemies - self.num_enemies), (0, 0))).flatten()

        # --- 4. Enemy existence mask (Existence ONLY, not visibility) ---
        enemy_exist_mask = jnp.concatenate([
            jnp.ones(self.num_enemies),
            jnp.zeros(max_enemies - self.num_enemies)
        ])

        # --- 5. Global features (relative/normalized) ---
        beacon_rel = (state.beacon_pos - centroid) / dims
        global_feats = jnp.array([
            beacon_rel[0],
            beacon_rel[1],
            state.timestep / self.max_steps
        ])

        return jnp.concatenate([f_feats, e_feats, enemy_exist_mask, global_feats])

    def get_action_mask(self, state: TwoBridgeState):
        """
        SC2-style action masks: who, verb, target.
        """
        s = state.smax_state
        ally_alive = s.unit_alive[:self.num_allies]
        enemy_alive = s.unit_alive[self.num_allies:]

        # WHO mask: which allies are available to act
        who_mask = ally_alive.astype(jnp.float32)

        # VERB mask: [noop, move, attack]
        can_attack = jnp.any(enemy_alive)
        verb_mask = jnp.array([
            1.0,                # noop always valid
            jnp.any(ally_alive),# move valid if any alive
            can_attack.astype(jnp.float32) # attack only if enemies exist
        ])

        # TARGET mask: which enemies can be targeted (fixed size 8)
        # SC2 Logic: Unit can only be targeted if it is alive AND visible
        e_visible_alive = enemy_alive * state.enemy_visible
        
        # If focusing on a target, only that target (or closest) can be actioned?
        # Actually in SC2, player CAN switch. The heuristic is for the AI/Auto-attack.
        # We'll allow the policy full flexibility.
        target_mask = jnp.concatenate([
            e_visible_alive.astype(jnp.float32),
            jnp.zeros(8 - self.num_enemies)
        ])

        # 3. DIRECTION mask: 8 cardinal/diagonal directions
        direction_mask = jnp.ones(8, dtype=jnp.float32)

        return {
            "who": who_mask,
            "verb": verb_mask,
            "direction": direction_mask,
            "target": target_mask
        }

    def get_enemy_actions(self, state: TwoBridgeState, rng: chex.PRNGKey) -> jnp.ndarray:
        """
        Unified enemy policy supporting Static, Guard, Patrol, and Aggressive modes.
        """
        s = state.smax_state
        ally_pos = s.unit_positions[:self.num_allies]
        enemy_pos = s.unit_positions[self.num_allies:]
        ally_alive = s.unit_alive[:self.num_allies]
        enemy_alive = s.unit_alive[self.num_allies:]

        any_ally_alive = jnp.any(ally_alive)

        # Pairwise distances (num_enemies, num_allies)
        diff = enemy_pos[:, None, :] - ally_pos[None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)

        # Mask dead allies
        valid_dists = jnp.where(ally_alive[None, :], dists, 999.0)
        closest_ally = jnp.argmin(valid_dists, axis=1)
        min_enemy_dist = jnp.min(valid_dists, axis=1)

        in_range = min_enemy_dist < self.combat_cfg["attack_range"]

        def do_attack(targets):
            # SMAX Team-1 attack indices are reversed relative to ally ordering.
            smax_target_indices = (self.num_allies - 1) - targets
            return smax_target_indices + 5

        def do_pursue(targets, positions):
            # Simple cardinal movement toward target
            target_pos = ally_pos[targets]
            vec = target_pos - positions
            # Vectorized direction calculation
            return jnp.where(
                jnp.abs(vec[:, 0]) > jnp.abs(vec[:, 1]),
                jnp.where(vec[:, 0] > 0, 1, 3), # East or West
                jnp.where(vec[:, 1] > 0, 0, 2)  # North or South
            )

        # Mode Implementations 
        def mode_static(_):
            return jnp.ones(self.num_enemies, dtype=jnp.int32) * 4

        def mode_guard(_):
            return jnp.where(in_range, do_attack(closest_ally), 4)

        def mode_patrol(_):
            # Any enemy sees ally within aggression radius -> Coordinated Aggressive mode
            any_engaged = jnp.any(min_enemy_dist < self.combat_cfg["aggression_radius"])
            
            # Roaming: random cardinal move
            roam = jax.random.randint(rng, (self.num_enemies,), 0, 4)
            
            coordinated_response = jnp.where(in_range, do_attack(closest_ally), do_pursue(closest_ally, enemy_pos))
            return jnp.where(any_engaged, coordinated_response, roam)

        def mode_aggressive(_):
            return jnp.where(in_range, do_attack(closest_ally), do_pursue(closest_ally, enemy_pos))

        enemy_actions = jax.lax.switch(
            self.mode_id,
            [mode_static, mode_guard, mode_patrol, mode_aggressive],
            operand=None
        )
        
        # Finally, ensure enemies only act if any ally is alive, else STOP
        return jnp.where(any_ally_alive, enemy_actions, 4)

    def build_spatial_features(self, state: TwoBridgeState) -> Dict[str, jnp.ndarray]:
        """
        Mimics Experiment 3's 17-channel feature_screen and 7-channel feature_minimap.
        """
        res = self.resolution
        smax_state = state.smax_state
        
        def get_grid_obs(positions, healths, alives, visibility, map_bounds, grid_res):
            # Strictly zero out coordinates of dead or invisible units
            clamped_pos = jnp.where(alives[:, None] & visibility[:, None], positions, -100.0)
            
            grid_pos = ((clamped_pos - map_bounds[:2]) / (map_bounds[2:] - map_bounds[:2]) * grid_res).astype(jnp.int32)
            in_view = (grid_pos >= 0).all(axis=-1) & (grid_pos < grid_res).all(axis=-1) & alives & visibility
            
            # Map indices, ensuring dead/invisible units (with pos -100) are safely OOB
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
                                          jnp.ones(self.num_allies, dtype=jnp.bool_), # Allies always visible to themselves
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
        centroid = jnp.mean(smax_state.unit_positions[:self.num_allies], axis=0)
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
        screen = screen.at[0].set(s_terrain)  # height_map
        screen = screen.at[5].set(s_ally_p)   # player_relative (ally)
        screen = screen.at[6].set(s_enemy_p)  # player_relative (enemy)
        screen = screen.at[7].set(s_beacon_p) # player_relative (neutral/beacon)
        screen = screen.at[8].set(s_ally_h)   # hit_points (ally)
        screen = screen.at[9].set(s_enemy_h)  # hit_points (enemy)

        return {"screen": screen, "minimap": minimap}

    def translate_action(self, central_action: CentralAction) -> jnp.ndarray:
        """
        Maps centralized representation to SMAX discrete integers.
        In this high-fidelity version: 0-7 are 8 directions of movement.
        Attack actions start at 8? No, SMAX might have its own indexing.
        Actually, we override movement/combat entirely in step, 
        so translate_action just needs to preserve the integers.
        """
        def get_unit_action(i):
            is_selected = central_action.who_mask[i]
            
            def do_noop(_): return 4 # Stop acts as no-op
            # central_action.direction: 0:North (+y), 1:East (+x), 2:South (-y), 3:West (-x)
            def do_move(_): return central_action.direction
            # Map target to SMAX attack offset
            def do_attack(_): return central_action.target + 5
            
            mapped_action = jax.lax.switch(
                central_action.verb,
                [do_noop, do_move, do_attack],
                operand=None
            )
            return jax.lax.select(is_selected == 1, mapped_action, 4)

        ally_actions = jax.vmap(get_unit_action)(jnp.arange(self.num_allies))
        return jnp.pad(ally_actions, (0, self.num_enemies), constant_values=4)

class TwoBridgeGymEnv(gym.Env):
    """
    A stateful Gymnasium wrapper for the JAX-native TwoBridgeEnv.
    Enables usage with standard RL libraries via gym.make().
    """
    def __init__(self, variant_name="V2_Base", **kwargs):
        super().__init__()
        self.env = TwoBridgeEnv(variant_name=variant_name, **kwargs)
        self.num_allies = self.env.num_allies
        self.num_enemies = self.env.num_enemies
        
        # Define Spaces (Assuming Vector Obs for standard Gym)
        # 63-dim vector space. Relative positions are in range [-1, 1]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(63,), dtype=jnp.float32)
        
        # Action Space: MultiDiscrete [WhoMask(32), Verb(3), Direction(8), Target(8)]
        # We simplify selection mask to an integer for Gym (0-31 representing binary flags)
        self.action_space = gym.spaces.MultiDiscrete([32, 3, 8, 8])
        
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
        # Decode MultiDiscrete back to CentralAction
        # NOTE: who_val is now Ignored to match SC2 conventions (all alive units selected).
        # This reduces action entropy and matches standard SMAC policy assumptions.
        who_val, verb, direction, target = action
        
        who_mask = self._state.smax_state.unit_alive[:self.num_allies]
        
        central_action = CentralAction(
            who_mask=who_mask,
            verb=verb,
            direction=direction,
            target=target
        )
        
        self._rng, step_rng = jax.random.split(self._rng)
        obs, self._state, reward, done, info = self.env.step(step_rng, self._state, central_action)
        
        mask = self.env.get_action_mask(self._state)
        info["action_mask"] = mask
        
        if isinstance(obs, dict):
            obs = obs["vector"]
            
        return obs, float(reward), bool(done), False, info

    def render(self):
        # We can implement a visualizer here or use evaluate.py logic
        pass
