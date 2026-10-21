# Sc2Jax vs. True StarCraft II — Similarities and Differences

This document outlines how the Sc2Jax research environment relates to actual StarCraft II (Blizzard, 2010). Sc2Jax is a **conceptual proxy**, not a simulation — it retains the *essence* of SC2 micro-management while stripping away everything that doesn't serve RL research.

The reference for "true SC2" here is the **SC2LE** (StarCraft II Learning Environment) paper (Vinyals et al., 2017, DeepMind + Blizzard), which exposed StarCraft II as an RL environment via the PySC2 Python wrapper.

---

## A. What Sc2Jax Shares with True SC2 (Similarities)

These are the core design choices that make Sc2Jax a faithful *abstraction* of SC2 rather than an unrelated game.

### 1. Type-Advantage Combat Loop
Sc2Jax captures SC2's signature rock-paper-scissors combat triangle: **Melee > Ranged > Tank**, with damage multipliers and type-bonus matrices. This mirrors how real SC2 units like Zealot vs Marine vs Siege Tank interact — each unit type has strengths and weaknesses against others. The concept of asymmetric unit matchups is central to SC2 strategy and is preserved in Sc2Jax.

### 2. Bridge-Choked Combat Dynamics
The TwoBridge map forces two groups of units to fight across narrow choke points, with a cliff wall blocking lateral movement. This closely mirrors real SC2 maps like Abyssal Reef (the ladder map used in the SC2LE paper) where armies are funneled through choke points, creating natural melee-ranged separation and forcing strategic positioning. The "two-army vs two-army" combat pattern is a canonical SC2 micro-battle structure.

### 3. Persistent Targeting with Leash Range
Units lock onto a target and stay engaged until it dies or moves beyond a configurable leash distance. This is exactly how SC2's `attack` and `attack-move` commands work — units maintain target locks and won't chase enemies beyond their leash range, which is a critical micro-management mechanic in real SC2.

### 4. Fog of War (Simplified)
Sc2Jax implements a memory-based vision system: when an enemy moves out of sight, the agent retains knowledge of its last-seen position, HP, and alive status. This mirrors SC2's fog-of-war where units persist in memory after being spotted, even if they retreat behind walls. The SC2LE paper explicitly notes this: "unit state is remembered at position where last seen."

### 5. Projectile Ballistics
Ranged units fire projectiles with ballistic travel time — hits are not instant. This matches real SC2 where units like Marines fire bullets that take time to reach their target, and the projectile must travel in a specific direction. The distance-based hit detection in Sc2Jax is a simplified but faithful version of this.

### 6. Reward Shaping: Combat + Navigation
Sc2Jax combines navigation (reach the beacon) and combat (kill enemies) into a single reward signal. This mirrors the SC2LE design philosophy where mini-games like CollectMineralShards (+1 per shard) and FindAndDefeatZerglings (+1 enemy, -1 ally) use dense per-step rewards rather than sparse game-end outcomes.

### 7. Multi-Agent Unit Control
Sc2Jax has multiple units acting simultaneously as independent agents with shared policies — each unit receives a directed action (via `who_mask`) and executes it. This mirrors the core SC2 paradigm where a single player controls a large army of independent units, each needing to make decisions about movement and targeting. SMAC (Samvelyan et al., 2019) formalizes this as the multi-agent RL problem in SC2.

### 8. Step-Based Simulation (Approximate Tick Rate)
Sc2Jax advances in discrete steps, mirroring SC2LE's game loop where the engine runs at 16 Hz ("normal speed") or 22.4 Hz ("fast speed"). The SC2LE paper acts every **8 game frames** (≈180 APM equivalent); Sc2Jax's step size is configurable and represents a comparable level of temporal abstraction.

### 9. Deterministic Simulation
SC2LE is frame-locked deterministic (with weapon speed randomness of ±1–2 steps and random update order, both seedable). Sc2Jax achieves **full determinism** — PRNG keys are explicitly threaded through all function calls, making it reproducible across platforms. Both share the property that simulations can be reproduced given the same seed.

---

## B. Where Sc2Jax Diverges from True SC2 (Differences)

### 1. Action Space

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Action structure** | ~300 function-based actions with typed arguments (screen coords, minimap coords, unit tags, control group IDs, queued bool) | **3 verbs per unit**: NO_OP / MOVE / ATTACK — flat discrete output |
| **Action format** | `FunctionCall` protos: e.g., `[Attack_screen, [[queued], [screen_point]]]` — hierarchical composition of function ID + arguments | Per-unit `PerUnitAction` dataclass: `(who_mask, verb, direction, target)` — 4 arrays |
| **Direction** | Continuous screen coordinates (N×N grid) or minimap coordinates (64×64) | Octal directions: 8 discrete angles (N, NE, E, SE, S, SW, W, NW) |
| **Targeting** | Click-to-select from thousands of objects; unit tag lists for multi-target spells (e.g., Storm, Blinding Burst) | Discrete index into alive, visible enemies only (0–7 max target slots) |
| **Action queuing** | Shift-queue: multiple ordered commands per unit (move → attack-move → build) | Single action per step, **no queuing** at all |
| **Unit selection** | Control groups (P0–P9), filter, lasso select, single/multi-select with selection actions | No selection layer — all alive units act each step via `who_mask` (implicit full selection) |
| **Abilities** | 500+ abilities including spells (Halo Jump, Stasis Field, Blink, Metabolic Boost), building construction | **None** — only basic move and attack. No spells, no abilities beyond the 3 verbs |
| **Total action space** | ~10⁸ combinations at 64×64 resolution (function ID × argument values) | ~17 flat actions for all units combined — heavily reduced discrete space |

### 2. Unit Types & Stats

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Unit roster** | 125+ unique units across Terran / Zerg / Protoss (Marine, Zealot, Hydralisk, Siege Tank, Medivac, etc.) | **3 abstract types**: MeleeUnit, RangedUnit, TankUnit — no race distinction |
| **Armor system** | Armored / Organic / Mechanical + structure type modifiers (40+ distinct weapons with specific bonuses) | Single armor value per unit; damage matrix × bonus matrix — simplified type advantage |
| **Weapon variety** | 40+ distinct weapons with splash damage, pierce, bonuses vs types, different windup/cooldown per weapon | Simplified damage matrix; **no splash, no pierce**, single windup/cooldown per type |
| **Unit creation** | Production from buildings over time (minerals/gas economy) with train queue | **Static composition** — all units present from step 0, no production or economy |
| **Upgrade system** | Ground armor / weapon levels via research labs, tech tree progression | No upgrades — all stats fixed at initialization |
| **Special abilities** | Units have unique abilities: Marine Stimpack, Zealot Blink, Siege Tank siege mode, Medivac heal | No special abilities — units behave purely as damage dealers with movement and attack |

### 3. Map & Terrain

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Map variety** | 10,000+ maps of varying sizes and designs; SC2LE provides Abyssal Reef LE plus 7 mini-game maps | Single **TwoBridge** map — 32×32 grid with cliff wall + two bridge gaps; procedurally configured variants |
| **Terrain complexity** | Elevation (affects LOS and attack angle), ramps, choke points, creep zones, power fields | Flat 2D plane — only cliff wall blocks movement; no elevation or pathfinding costs |
| **Map size** | Typically 64×64 to 512×512 (Blizzard resolution); SC2LE minimap is fixed at 64×64 | Fixed 32×32 **normalized** grid (values in [0,1]) — much smaller |
| **Terrain enforcement** | A* pathfinding with navigation mesh prevents impossible movement; unit stacking constraints | Simplified bridge crossing: detects x-axis cliff crossing, rejects if not near bridge gap; no pathfinding |
| **Variants** | Hardcoded per map (symmetric or asymmetric spawns) | 9 variants: 3 army strength configs (V1=3v3, V2=5v5, V3=8v8) × 3 modes (Base/Combat/Navigate) |
| **Spawn randomness** | Fixed symmetric spawn locations on ladder maps | Procedural spawns within 6 predefined regions per side |

### 4. Physics & Movement

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Movement model** | Discrete tick-based movement with acceleration curves and max speed caps per unit type; terrain effects on speed | Continuous double-integrator: `vel = (vel + accel) × friction`, clamped to max speed |
| **Collision** | Bounding-circle collision with priority-based pushaway (bigger units push smaller); unit stacking limits | Two models: **soft** (mass-weighted repulsion via `apply_mass_collisions`) + **hard** (stiffness-based blocking via `apply_hard_collisions`) |
| **Pathfinding** | A* on navigation mesh; units follow paths around obstacles, through bridges, with unit stacking constraints | **No pathfinding** — direction is a discrete octal command; physics handles detours via collision forces |
| **Turn rate** | Units have minimum turning radius (especially large units like Colossus); cannot turn instantaneously | **No turn rate** — velocity changes instantaneously within friction limits |
| **Projectile physics** | Mix of hitscan (instant) and projectile-based weapons; projectiles travel at defined speeds, affected by unit movement | All ranged weapons are **projectile-based** with ballistic travel; hit detection via distance check at projectile position |

### 5. Combat Mechanics

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Attack cycle** | Windup → fire → cooldown per weapon definition; some weapons have auto-cast enabled | Same cycle (windup + cooldown) but with **projectile slot allocation** if too many ranged units fire simultaneously |
| **Auto-attack / focus fire** | Units auto-engage enemies in attack range; AI and humans use focus-fire to prioritize low-HP targets | **No auto-engagement** — units wait for explicit attack command; nearest-enemy selection only during retargeting |
| **Damage calculation** | `damage = max(1, base × type_bonus + armor_bonus - armor)` — minimum 1 damage per hit | `damage = max(0.5, base × type_bonus + bonus - armor)` — minimum 0.5 (fractional) |
| **Damage types** | Normal / Crush / Ignore Armor + multiple bonus categories (Structure, Biological, Mechanical) | Single type-advantage matrix + scalar bonus — no damage type categories |
| **Splash / AoE** | Multiple weapons deal area damage (Baneling explosion, Roach bile, Siege Tank bombardment) | **No splash** — one projectile = one hit per unit |
| **Piercing shots** | Dark Templar, Mutalisks, and others hit multiple units in a line | **No piercing** — each projectile hits the closest enemy only |
| **Interrupt / dismiss** | Dismiss (Archon), Blink, Temporal Rift can cancel attacks or move units out of combat | No interrupt or dismiss mechanics |

### 6. Information & Perception

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Observation architecture** | Feature layers rendered at configurable resolution (minimap + screen, typically 64×64) | **Dual mode**: fixed-length 63-dim vector (default, `use_spatial_obs=False`) **or** spatial feature layers (`use_spatial_obs=True`) |
| **Feature layers (minimap)** | 7 typical layers: height_map, visibility, creep, camera, player_id, player_relative [0–4], selected | **7-channel minimap** (when `use_spatial_obs=True`): terrain mask, ally position, ally health, enemy position, enemy health, beacon position, empty — mimics SC2LE Experiment 3 |
| **Feature layers (screen)** | ~17 screen layers: height_map, visibility, creep, power, player_id, player_relative, unit_type, selected, hit_points (log), energy (log), shields (log), unit_density, unit_density_aa | **17-channel screen** (when `use_spatial_obs=True`): terrain, ally position, enemy position, beacon position, ally health, enemy health — camera-locked to army centroid with 8.0 view radius |
| **Non-spatial observations** | Player stats (11-tensor: minerals, vespene, food used/cap/army/workers), control groups (10×2), single/multi select (7-tensor/unit), build queue, available actions mask | **Fixed 63-dim vector** (default): 20-dim ally features (5 units × 4) + 32-dim enemy features (8 units × 4) + 8-dim enemy mask + 3-dim global |
| **Fog of war** | Complex vision system: persistent memory until game end for dead units; detection reveals stealth/burrow | Simplified `FogState`: last-seen position, HP, alive flag per enemy; **no persistent memory across episodes** |
| **Unit detection** | Detection units (e.g., Science Ship, Observer) reveal cloaked/digger enemies within detection radius | **No stealth or cloak** — all units always visible when in vision range |
| **Screen / minimap distinction** | Two view modes: screen (detailed local area) and minimap (coarse full-world view); camera must be actively moved via `move_camera` function | **Screen is camera-locked** to army centroid (auto-tracks); minimap covers full map. No manual camera control needed — camera follows army automatically |
| **Information completeness** | RL agents (e.g., AlphaStar) receive a flattened state dictionary with hundreds of features per unit | Minimalist — only relative position, health, cooldown for nearby allies/enemies within vision radius (vector mode) |

### 7. Game Structure & Economy

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Economy** | Minerals + vespene harvesting by worker units (SCV/Drone/Probe); resources accumulate and are spent on production | **No economy** — all units present from step 0, no resources, no worker management |
| **Buildings / structures** | Command Center, Barracks, Forge, etc. — must be built, protected, and upgraded | **No buildings** — purely unit-vs-unit combat on an open map |
| **Tech tree** | Gateway → Forge → Stargate progression; research unlocks units and upgrades | No tech tree — all 3 unit types available immediately |
| **Supply** | Supply cap limits max army size (Supply Depots increase it; over-supply prevents production) | No supply system — army sizes are fixed per variant config (3v3, 5v5, or 8v8) |
| **Win conditions** | Destroy enemy buildings OR achieve surrender (time-based); SC2LE full game: 30-minute time limit = tie | Kill all enemies OR reach beacon (Navigate mode) OR timeout at step 300 |
| **Match length** | Typically 5–60 minutes of real time (~1500–18,000 game steps at 30 ticks/s) | Fixed max **300 steps** (~15 seconds simulated) — designed for fast RL training |
| **Episode structure** | Continuous game with no natural episode boundary (except 30-min time limit or surrender) | Discrete episodes of fixed length, reset after win/loss/timeout |

### 8. Multi-Agent Coordination

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Player vs agents** | Single human player controls entire army (5v5 at lowest ladder; 30+ APM just for camera) | No human player — RL agents control all units; batched JAX simulation with 32–64 parallel environments |
| **Subgrouping** | Control groups, selection filters, attack-move commands create organic subgroups during gameplay | All units act simultaneously via `who_mask` — no dynamic subgrouping; single policy across all units |
| **Communication** | Pings, chat, visual coordination between team members (in co-op); humans coordinate subgroups verbally | No inter-agent communication — each unit's policy is independent, conditioned only on its observation vector |
| **APM** | Humans: 30–500 APM; SC2LE experiments act every 8 frames ≈ 180 APM | No APM concept — JAX computes all actions in a single vectorized step |
| **Learning paradigm** | Human players learn from thousands of hours of gameplay; AlphaStar trained via imitation + RL on 200K core games | Batched JAX simulation: full JIT compilation for throughput; >17,000 SPS on Apple Silicon; no human demonstration data needed |

### 9. Technical Architecture

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Engine** | Proprietary Blizzard C++ engine (unmodified), compiled binary; accessed via gRPC | Pure Python/JAX — everything is JIT-compileable via `jax.jit` and `jax.vmap` |
| **Determinism** | Frame-locked deterministic; weapon randomness ±1–2 steps, random update order; seedable | **Fully deterministic** — PRNG keys explicitly threaded through all function calls |
| **Observation delivery** | API calls over network to local SC2 process; frames rendered by game engine, transformed via `features.transform_obs()` | Direct JAX array computation — environment state lives in JAX arrays, no rendering pipeline during training |
| **Observation format** | Dict of numpy arrays: feature layers (resolution-dependent), structured data (`player_common`, `score_cumulative`) | Flat 63-dim JAX array — ready for neural network input, fixed shape |
| **Action delivery** | `FunctionCall` protos sent over gRPC: ability ID + typed arguments (screen point, minimap point, unit tags) | `PerUnitAction` dataclass (`who_mask`, `verb`, `direction`, `target`) passed directly into JAX step function |
| **Rendering** | Full 3D engine; feature layers rendered as layered maps at configurable resolution (64×64 typical) | Optional Pygame renderer for visualization only — never used during training, no 3D engine |
| **Batching** | Limited — each episode runs in a separate SC2 process; parallel episodes require multiple binary instances (~400 MB each) | **Native batch support** — all environments run in a single JIT graph via `jax.vmap`; 32–64 parallel envs at zero overhead |
| **Replays** | Full `.SC2Replay` files with exact input history; can be replayed frame-accurately | No replays — state serialized via pickle for checkpoints only (`save_checkpoint` / `load_checkpoint`) |
| **Launch overhead** | ~5–10 seconds per SC2 binary launch; memory usage ~400 MB per instance | Near-zero — pure Python import, JIT compiles on first call (~2–5 seconds) |
| **Testing** | Limited — test via end-to-end episodes; no ability to inspect intermediate state easily | Full introspectability — every array in `JaxSC2State` is accessible; independent mechanics tests (`tests/`) |

### 10. Reward Design Philosophy

| Aspect | True SC2 (SC2LE/PySC2) | Sc2Jax |
|---|---|---|
| **Full game rewards** | Two options: (1) Ternary — +1 win / 0 tie / -1 loss; (2) Blizzard score — running cumulative sum of resources + units alive | Not applicable — Sc2Jax focuses on micro-battles, not full economy games |
| **Mini-game rewards** | Dense per-step: +1 kill, -1 death, +1 resource collected (varies by mini-game) | Dense per-step: +5 for kill, +10 for beacon reach, -0.1 per step (time penalty) |
| **Reward sparsity** | SC2LE found that even with dense rewards, no agent learned to win the full 1v1 game; Blizzard score converged to trivial "preserve initial mining" strategy | Same lesson: Sc2Jax avoids the full-game problem entirely by focusing on tractable micro-battles |

---

## C. Summary: What Sc2Jax Retains vs. Discards

### Retained (the "essence" of SC2)
- **Type advantage**: Melee > Ranged > Tank rock-paper-scissors cycle
- **Bridge-choked combat dynamics**: Force splitting across choke points, natural melee-ranged separation
- **Persistent targeting with leash range** — units stay locked on targets until dead or beyond leash distance
- **Fog of war**: Simplified memory-based vision (last-seen position tracking)
- **Projectile ballistics**: Ranged weapons have travel time, not instant hitscan
- **Reward shaping**: Combined navigation + combat signals (dense per-step rewards)
- **Multi-agent unit control**: Multiple independent units acting simultaneously under a shared policy

### Discarded (irrelevant or prohibitive for RL research)
- **Economy & production**: Minerals, gas, workers, buildings, supply — completely absent in Sc2Jax
- **Tech trees & upgrades**: No research, no race progression, all unit types available immediately
- **Unit selection mechanics**: Control groups, lasso select, filter — replaced by implicit full-army control
- **Spell/ability system**: 500+ abilities including spells — reduced to 3 verbs (NO_OP, MOVE, ATTACK)
- **Pixel-based observations**: Feature layers at 64×64 resolution — Sc2Jax supports both: fixed 63-dim vector (default) **and** optional spatial feature layers (`use_spatial_obs=True`) with 7-channel minimap + 17-channel screen mimicking SC2LE Experiment 3
- **A* pathfinding**: Navigation mesh with unit stacking — replaced by octal direction + collision forces
- **Realistic terrain**: Elevation, ramps, creep zones — replaced by flat plane with cliff wall
- **Full game length**: 5–60 minute matches — compressed to 300 steps (~15 seconds)
- **Binary dependency**: SC2 game engine — replaced by pure JAX simulation

---

## D. Sc2Jax vs PySC2 — The Two Design Philosophies

PySC2 (Vinyals et al., 2017, DeepMind) and Sc2Jax are both StarCraft II environments for RL research, but they represent fundamentally different approaches.

### PySC2: The Wrapper Approach
```
Agent ↔ PySC2 env (Python) ↔ gRPC ↔ SC2 binary (Blizzard C++) ↔ Actual game simulation
```

- **Requires real SC2 game** (~10 GB) — talks to the actual binary via gRPC
- **500+ abilities** exposed through the SC2 Machine Learning API
- **Feature layer observations**: 7 minimap + ~13 screen layers at configurable resolution (typically 64×64)
- **Action grouping**: ~300 function-based actions with typed arguments (screen coords, minimap coords, unit tags)
- **Mini-game suite**: 7 curated mini-games (MoveToBeacon through BuildMarines) plus SMAC's 1,000+ scenarios
- **Throughput**: ~50–200 SPS (limited by binary launch + rendering overhead)
- **Fidelity**: Exact SC2 physics, units, abilities — the real game as an RL environment

### Sc2Jax: The From-Scratch Approach
```
Agent ↔ JaxSC2Env (Python/JAX) ↔ jax.jit(jnp arrays) ↔ Self-contained simulation
```

- **Zero game dependency** — runs on any platform with JAX (CPU / GPU / TPU / Apple Silicon)
- **3 unit types, 3 actions** — heavily simplified but fully transparent simulation
- **Fixed 63-dim vector observations** — no pixels, no feature layers, ready for neural network input
- **9 variants**: 3 army sizes × 3 modes — procedurally configured from a single map template
- **Throughput**: >17,000 SPS on Apple Silicon (pure JIT execution, no binary overhead)
- **Fidelity**: Simplified but captures the essential micro-combat dynamics of SC2

### Why Two Approaches Exist

| Design Goal | PySC2 (wrapper) wins | Sc2Jax (rewrite) wins |
|---|---|---|
| **Fidelity to SC2** | Exact game physics, 125+ units, 500+ abilities | Simplified but fully transparent and modifiable |
| **Accessibility** | Requires SC2 game purchase/installation | Zero dependencies beyond JAX ecosystem |
| **Throughput** | ~50–200 SPS | >17,000 SPS |
| **Research iteration** | Slow (binary launch/stop cycle) | Instant (change code, next run starts immediately) |
| **Platform** | Primarily Linux (officially supported) | Any platform with JAX support |
| **Extensibility** | Limited to what SC2 API exposes | Complete control — every equation is source code you can change |
| **Reproducibility** | Good but version-dependent (replays tied to game versions) | Perfect — fully deterministic, identical across platforms |
| **Best use case** | Benchmarking against human-level AI; studying real SC2 strategies | Rapid RL algorithm development; ablation studies on simplified mechanics |

**Bottom line**: PySC2 is the **gold-standard interface to real SC2** — if you want agents that learn from actual StarCraft II mechanics. Sc2Jax is the **high-throughput research playground** — if you want to iterate fast on RL algorithms with SC2-inspired mechanics, without any game binary dependency. They solve different problems in the same research space.
