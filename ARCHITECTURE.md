# JaxSC2 Architecture — Deep Dive

## What It Is

JaxSC2 is a **fully differentiable, JIT-compiled StarCraft II combat simulator** built on JAX. It's a mini-environment for learning multi-agent decision-making with SC2-like mechanics: unit movement, targeting, combat windup/cooldown, fog of war, terrain bridges, and type advantages — all vectorized across batches with `jax.vmap` and compiled with `@jax.jit`.

---

## 1. Environment Engine (`JaxSC2/env/`)

### Core Architecture

The environment follows a **12-phase deterministic step loop** per timestep:

```
Action Buffer → Enemy AI → Decode Actions → Accel Calc
→ Turn-Rate Inertia → Velocity Integration → Mass Collision
→ Hard Collision → Terrain Constraints → Target Persistence + Override
→ Combat (Projectiles+Melee) → Fog Update → Visibility Check
→ Reward Computation → State Build
```

This is a **physics-respecting simulation**, not just discrete turn-based. Every phase transforms `SmaxState` (position, velocity, health, type, team) via pure JAX functions.

### Physics (`mechanics.py` — 287 lines)

| Function | What It Does | Key Params |
|---|---|---|
| `apply_mass_collisions` | Pairwise repulsion weighted by mass/radius | r_i, r_j radii; m_i, m_j masses |
| `apply_hard_collisions` | Strong repulsion when units overlap too much | stiffness=1.2, threshold at 20% radius |
| `integrate_velocity` | Continuous double-integrator dynamics | friction, max_speed clamp |
| `update_projectiles` | Ballistic projectiles with distance cutoff | hit radius=0.6, lost at >40 |
| `apply_high_fidelity_combat` | Windup/cooldown cycle, slot-based projectile spawning | lax.scan over weapon slots |
| `update_persistent_targets` | SC2-style target sticking (leash behavior) | leash_extra=2.0 |

**Combat damage formula:**
```python
damage = clamp(damage * multiplier + bonus - armor, min=0.5)
```

### Unit Roster (`units.py`)

| Type | Index | HP | Speed | Mass | Armor | Weapon |
|---|---|---|---|---|---|---|
| Melee (Marine-like) | 0 | 100 | 0.3 | 2.0 | 1.0 | Blade: dmg=8, rng=1.5, windup=1, cd=6 |
| Ranged (Zealot-like) | 1 | 45 | 0.25 | 1.0 | 0.0 | Gauss Rifle: dmg=5, rng=6.0, windup=2, cd=8 |
| Tank (Stalker-like) | 2 | 200 | 0.18 | 5.0 | 2.0 | Plasma Cannon: dmg=4, rng=4.0, windup=3, cd=12 |

**Type advantage matrix (row attacks col):**
```python
[[1.0, 1.5, 0.7],    # Melee → [Melee, Ranged(+50%), Tank(-30%)]
 [0.8, 1.0, 1.5],    # Ranged → [Melee(-20%), Tank(+50%)]
 [1.5, 0.7, 1.0]]    # Tank → [Melee(+50%), Ranged(-30%)]
```

Plus `bonus_matrix` shifting damage by +2 for certain type pairs. This creates a **rock-paper-scissors meta** that agents must learn to exploit.

### Map (`maps/twobridge.py`)
- **32×32 grid** with cliff at x=0.5 forcing bridge crossing
- **Two bridges** at y=(0.2–0.3) and (y=0.7–0.8) — only allies are terrain-constrained
- **6 regional spawn coordinates** for varied initial positioning

Variants: V1(5v3), V2(5v5), V3(5v8) × {Base, Combat, Navigate} = 9 scenarios

### Observation Space (63-dim vector)
- **20 ally features**: 4 units × 5 dims (relative X, relative Y, health, cooldown, __pad__)
- **32 enemy features**: 4 units × 8 dims (relative X, relative Y, health, cooldown, type, alive, __pad__×2)
- **8 action masks**: verb_mask(3) + direction_mask(8) flattened
- **3 global features**

### Action Space (PerUnitAction with bitmasks)
```python
who_mask: bool[N]        # Which ally to act on (sparse — one-hot style)
verb:     int[N]         # 0=NOOP, 1=MOVE, 2=ATTACK
direction: int[N]        # 0–7 (8-directional octal)
target:   int[N]         # Enemy index 0–(num_enemies-1)
```

**Action masks** dynamically mask invalid actions: verb_mask, direction_mask, target_mask built from ally_alive and enemy_visible+alive. This is **critical** — it's why MaskPPO exists.

### Reward Function
```python
nav_reward = (prev_dist - new_dist) * 2.0        # Encourage approaching beacon
enemy_dmg_reward = (prev_hp - curr_hp) * 0.01 * 0.5   # Reward dealing damage
ally_dmg_penalty = -damage_to_allies               # Penalize friendly fire / being hit
enemy_killed_reward = 0.2                          # Flat bonus per kill
```

### Done Conditions
`beacon_reached | enemies_dead | allies_dead | timestep >= 300`

### Fog of War
FogState tracks per-unit: `last_seen_pos`, `alive`, `hp`, `time` of last sighting. Invisible enemies have 0 visibility — agents must scout to maintain information.

### Gym Wrapper (`JaxSC2GymEnv`)
Converts the native action space to `gym.spaces.MultiDiscrete([2, 3, 8, 8])` with bitmask decoding for compatibility.

---

## 2. Algorithm Stack (`algorithms/`)

### MaskPPO (Main Algorithm) — `algorithms/mask_ppo/`

**Why it exists:** Standard PPO can't handle action masking. MaskPPO solves this with **multi-head independent softmax + -1e9 fill**.

**Model (`MaskedActorCritic` — 37 lines):**
```
Input (63-dim) → LayerNorm → Dense(256) → ReLU → Dense(256) → ReLU
  ├─ Value head: Dense(1)          → value scalar
  ├─ Verb head:   Dense(3)         → verb logits (NOOP/MOVE/ATTACK)
  ├─ Dir head:    Dense(8)         → direction logits (0-7 octal)
  └─ Target head: Dense(t_dim)     → target logits (num_enemies)
```

**Masking mechanism:** Before softmax, -1e9 is added to invalid logits (masked positions). This makes P(invalid) ≈ 0 without any special sampling logic.

**Loss (`masked_ppo_loss_multih`):**
- Separate `log_softmax` per head (verb, direction, target)
- Combined log-prob: `logp = verb_logp + dir_logp + target_logp`
- Multi-head entropy: `entropy = -sum_of_head_entropies` (encourages exploration across all heads)
- PPO clip + value loss + entropy bonus

**Training Loop:**
```python
vmap rollout over envs → lax.scan 512 steps → collect (obs, act, rew, logp, val, done)
→ compute GAE → flatten batch → shuffle → scan minibatches over 10 epochs
→ JIT-compiled train_iteration via @jax.jit
```

**Key insight:** The entire rollout + training is JIT-compiled. `lax.scan` unrolls the 512-step rollout and the epoch loop, making it extremely fast on GPU/TPU.

### PPO (Standard) — `algorithms/ppo/`

Same backbone as MaskPPO but **returns raw logits dict** `{verb_logits, direction_logits, target_logits, value}` without masking. The loss (`ppo_loss_multi_head`) is structurally identical but has no mask arguments. Useful for comparison or environments where all actions are always valid.

### A2C — `algorithms/a2c/`

Single-pass actor-critic with:
- **GAE_LAMBDA = 1.0** (pure Monte Carlo, no bootstrapping)
- **No PPO clip / epsilon** — just advantage-weighted policy gradient + value loss
- Same multi-head action structure

### Common Utilities (`algorithms/common/`)

| Utility | Algorithm |
|---|---|
| `RunningMeanStd` (Welford online) | Per-timestep normalization without batch statistics |
| `compute_gae` (lax.scan, reverse) | Generalized Advantage Estimation via backward scan |
| `flatten_obs` / `encode_action` / `decode_action` | Space conversion between env and algorithm |

---

## 3. Key Design Patterns & Why They Matter

### Action Masking is Central (Not an Afterthought)
The PerUnitAction has all three mask tensors. Most SC2 agents ignore this and pad — MaskPPO makes it native by filling masked logits with -1e9 before softmax. This preserves the multi-head independent sampling while guaranteeing invalid actions get zero probability.

### Continuous Physics in Discrete Action Space
Actions are discrete (MOVE/ATTACK + 8 directions), but the physics is continuous (double-integrator, velocity, collision). Agents learn **kinematic control** — how to move toward a target with inertia, not just point-and-shoot. The turn-rate inertia and mass collisions make movement feel weighty.

### Rock-Paper-Scissors Meta Creates Strategic Depth
The type advantage matrix isn't cosmetic — it fundamentally changes optimal behavior. A 5v3 Ranged vs Melee fight might be easy (ranged advantage), but a 5v8 Tank fight would be brutal. This creates emergent composition play without explicit unit selection.

### Bridge Map Forces Strategic Decision-Making
The cliff at x=0.5 with only two bridges means agents must learn: (1) which bridge to use, (2) whether to split force or push together, (3) timing relative to enemy bridge crossing. This is a simplified version of real StarCraft's "choke point" strategy.

### Full JIT Compilation for Speed
Every computational bottleneck uses JAX primitives: `@jax.jit` on train_iteration, `lax.scan` for rollout/epoch loops, `jax.vmap` over environment batches. Thousands of parallel episodes compile and run in the time a non-JAX implementation trains one.

### Fog of War Creates Information Asymmetry
Agents don't see invisible enemies. The FogState tracks last-seen position, creating a partial observability problem (POMDP) that makes the task significantly harder than fully observable MDPs. Memory-based agents and exploration strategies become necessary.

---

## 4. Architecture Diagram

```
┌─────────────────────────┐     ┌──────────────────────┐
│    JaxSC2 Environment   │     │  Training Algorithms  │
├─────────────────────────┤     ├──────────────────────┤
│  JaxSC2Env (859 lines)  │     │  MaskPPO ★ (main)    │
│  ─────────────────      │     │   ├─ MaskedActorCritic│
│  Step Loop (12 phases): │     │   ├─ masked_ppo_loss  │
│    1. Action Buffer     │     │   └─ JIT train_step   │
│    2. Enemy AI          │     │                       │
│    3. Decode Actions    │     │  PPO                   │
│    4-5. Physics         │     │   (same model, no mask)│
│    6. Combat            │     │                       │
│    7. Fog + Visibility  │     │  A2C                   │
│    8. Reward            │     │   (no clip, MC GAE)   │
│    9. State Build       │     └──────────────────────┘
├─────────────────────────┤            │
│ Mechanics:              │     ┌──────┴──────┐
│   - Mass/Hard Collision │     │  Common     │
│   - Double-Integrator   │     │  Utilities  │
│   - Projectile Physics  │     ├─ RunningMSD  │
│   - Windup/Cooldown     │     ├─ compute_GAE│
│   - Persistent Targets  │     └─ enc/dec    │
├─────────────────────────┤            │
│ Units: Melee/Ranged/Tank│     ┌──────┴──────┐
│  + Damage Matrix        │     │  Renderer   │
├─────────────────────────┤     ├─ Pygame GUI │
│ Map: Twobridge (32×32)  │     └─────────────┘
│  + Bridges, Variants    │
│                         │
│ Obs: 63-dim (20+32+8+3) │
│ Act: PerUnitAction      │
│  (who,verb,dir,target)  │
└─────────────────────────┘
```

★ = MaskPPO is the primary algorithm — PPO and A2C are comparison variants.

---

## 5. Key Files

| File | Lines | Purpose |
|---|---|---|
| `JaxSC2/env/base.py` | 88 | SMAX base class, SmaxState dataclass, Space classes |
| `JaxSC2/env/mechanics.py` | 287 | All physics: collisions, velocity, projectiles, combat, targets |
| `JaxSC2/env/renderer.py` | 244 | Pygame-based renderer, 60fps GIF output, world-to-screen camera |
| `JaxSC2/env/units.py` | 99 | Unit blueprints and weapon definitions (Melee/Ranged/Tank) |
| `JaxSC2/env/env.py` | 859 | JaxSC2Env main environment + JaxSC2GymEnv wrapper |
| `JaxSC2/maps/twobridge.py` | 140 | Terrain, bridge constraints, spawn coords, variant definitions |
| `algorithms/mask_ppo/model.py` | 37 | MaskedActorCritic multi-head model with -1e9 masking |
| `algorithms/mask_ppo/ppo_logic.py` | 113 | masked_ppo_loss_multih: per-head log-softmax, combined logp |
| `algorithms/mask_ppo/mask_ppo.py` | 243 | Full training loop: vmap rollout, GAE, JIT train_iteration |
| `algorithms/ppo/model.py` | 56 | MultiHeadActorCritic (without masking) |
| `algorithms/ppo/ppo.py` | 217 | Standard PPO training loop (no mask args) |
| `algorithms/a2c/a2c_logic.py` | 29 | A2C loss: no clip, GAE_LAMBDA=1.0 |
| `algorithms/common/utils.py` | 102 | RunningMeanStd, compute_gae, action encode/decode |
| `algorithms/common/base.py` | 31 | BaseAlgorithm ABC |

---

## 6. Notable Implementation Details

1. **`apply_high_fidelity_combat` uses `lax.scan` over weapon slots** — Each unit can have multiple weapons; the scan processes them in parallel. Windup/cooldown is tracked per-slot as integer counters decremented each step.

2. **Damage clamped to min=0.5** — Prevents zero-damage from fully armored targets, ensures combat always resolves (avoids deadlock).

3. **Reward shaping** — `enemy_dmg*(prev_hp-curr_hp)*0.01*0.5` gives credit based on target's HP loss (not raw damage dealt), which accounts for variable armor mitigation — aligns with real SC2 mechanics.

4. **`enemies_dead` and `allies_dead` both trigger done** — Zero-sum episode. No partial credit for surviving longer without winning. Harder but clearer signal.

5. **Gym wrapper converts to MultiDiscrete([2,3,8,8])** — The `who_mask` is encoded as a bitmask (0 or 1), while the other three dims are independent finite alphabets. Compatible with standard Gym-based RL libraries but loses sparse structure of PerUnitAction.

6. **FogState tracks `time`** — How long since an enemy was last seen. This isn't used directly in the observation but could be extended for temporal reasoning about hidden units.
