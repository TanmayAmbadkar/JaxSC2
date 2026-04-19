# JaxSC2 Mechanics — Physics Reference

The `mechanics.py` module (287 lines) implements the entire physics simulation as pure, JIT-compilable JAX functions. All state is stored in `SmaxState` (defined in `base.py`) and every function returns a new `SmaxState` — no mutation.

---

## Overview: The 12-Phase Step Loop

Mechanics functions are called in this order during each `JaxSC2Env.step()` call:

```
Phase 4: accel_calc               → computes per-unit acceleration from actions
Phase 5: turn_rate_inertia         → velocity smoothing (prevents instant direction change)
Phase 6: integrate_velocity        → position + velocity update via double-integrator
Phase 7: apply_mass_collisions     → mass-weighted pairwise collision resolution
Phase 8: apply_hard_collisions     → strong repulsion for overlapping units
Phase 9: enforce_bridge_terrain    → map constraints (cliff + bridges)
Phase 10: update_persistent_targets → SC2-style target sticking with leash
Phase 6b: apply_high_fidelity_combat → projectiles, melee, damage application
```

---

## 1. Velocity Integration (`integrate_velocity`)

**What it does:** Updates unit position and velocity using continuous double-integrator dynamics. This is the primary motion primitive — all actions (MOVE/ATTACK) feed into acceleration, which drives velocity, which drives position.

**Equations:**
```python
new_vel = (vel + accel) * friction * clamp(max_mag, max_speed)
new_pos = pos + new_vel
```

**Parameters:**
- `friction` — Damping factor (typically ~0.95). Applied each step to simulate drag.
- `max_speed` — Per-type speed cap from unit blueprint (e.g., Melee=0.3, Tank=0.18)

**Effect:** Units accelerate smoothly toward their target direction but cannot stop or turn instantly. This creates the "weighty" feel of movement — a Tank takes several steps to reverse direction even if commanded immediately.

---

## 2. Mass Collision Resolution (`apply_mass_collisions`)

**What it does:** Resolves pairwise collisions between units using mass-weighted repulsion. Prevents units from occupying the same space while respecting their relative "heaviness."

**Equations:**
```python
for each pair (i, j):
    r_i = mass_j / (mass_i + mass_j)   # lighter unit gets pushed more
    r_j = 1 - r_i

    overlap = (radius_i + radius_j) - |pos_i - pos_j|
    if overlap > 0:
        normal = (pos_i - pos_j) / |pos_i - pos_j|
        push_i = overlap * r_i * normal   # direction toward i
        pos_i += push_i
        pos_j -= push_j                   # equal and opposite
```

**Effect:** A Tank (mass=5.0) barely moves when bumped by a Melee unit (mass=2.0), but two Ranged units (mass=1.0 each) push each other apart equally. This matters in combat formations — heavy units hold ground while light units get scattered.

**Radii calculation:** `radius = mass * base_radius_factor` (implicit, derived from unit mass).

---

## 3. Hard Collision Repulsion (`apply_hard_collisions`)

**What it does:** Applies strong repulsive force when units overlap beyond a tight threshold. This is the "prevent unit stacking" safety net that kicks in when mass collision alone isn't enough.

**Equations:**
```python
for each pair (i, j):
    dist = |pos_i - pos_j|
    threshold = 0.2 * (radius_i + radius_j)   # 20% of combined radii
    if dist < threshold:
        stiffness = 1.2
        push_force = stiffness * (threshold - dist) / threshold
        normal = (pos_i - pos_j) / |pos_i - pos_j|
        pos_i += push_force * normal
        pos_j -= push_force * normal
```

**Effect:** When units get extremely close (< 20% of radius sum), they are forcefully separated. The stiffness parameter (1.2) determines how aggressively. This prevents "clumping" and ensures combat spreads units out naturally.

**Why both mass + hard collisions?** Mass collision handles normal movement overlaps (gentle). Hard collision handles edge cases where units somehow end up too close due to velocity or projectile impact (forceful).

---

## 4. Projectile Physics (`update_projectiles`)

**What it does:** Updates all active projectiles each step, checking for hits and expiration.

**Equations & Rules:**
```python
for each projectile:
    proj_pos += velocity * direction     # ballistic movement

    if |proj_pos - target_unit.pos| <= 0.6:   # HIT radius = 0.6
        apply_damage(target_unit, projectile.damage)
        remove_projectile()

    if |proj_start - proj_pos| > 40:          # Lost range = 40
        remove_projectile()
```

**Effect:** Projectiles travel in straight lines (no homing). They expire at distance 40 from spawn — this is much farther than any weapon's range, so projectiles typically hit or miss within their own weapon's range. The 0.6 hit radius creates a small "kill zone" around the target unit, making long-range shooting slightly easier than point-blank precision.

---

## 5. High-Fidelity Combat (`apply_high_fidelity_combat`)

**What it does:** The most complex function (100+ lines). Implements SC2-style weapon management with windup/cooldown cycles and slot-based projectile spawning.

**Weapon Cycle:**
```python
for each unit:
    for each weapon slot (only Melee has 1, Ranged/Tank have multiple):

        if windup_counter > 0:
            windup_counter -= 1              # Still charging up, no attack

        elif cooldown_counter > 0:
            cooldown_counter -= 1            # Just fired, reloading

        elif can_attack(target_in_range):
            spawn_projectile()               # Create at unit position + offset
            windup_counter = weapon.windup   # Reset to windup phase

    if in_melee_range(target):
        apply_damage(target)                   # Instant melee damage, no projectile
```

**Key Mechanics:**
- **Windup phase:** After firing, the weapon enters windup. During this time the unit cannot fire again (even if a new target is available).
- **Cooldown phase:** After the weapon finishes windup, it fires. Then cooldown begins — the unit is "reloading" and cannot fire for `weapon.cooldown` steps.
- **Slot-based:** Units with multiple weapons have parallel windup/cooldown counters. Firing weapon 0 doesn't affect weapon 1's readiness.
- **Melee vs Ranged:** Melee weapons use direct damage application (no projectile). Ranged/Tank spawn projectiles that travel to the target.

**Projectile spawning via `lax.scan`:** The function uses `lax.scan` over all weapon slots simultaneously, making it JIT-compile efficiently even for units with many weapons.

---

## 6. Persistent Targets (`update_persistent_targets`)

**What it does:** Implements SC2's "leash" behavior — units stick to their previously selected target even if it moves, unless the distance exceeds a leash threshold.

**Equations:**
```python
leash_extra = 2.0

for each ally:
    if target is alive and within (weapon_range + leash_extra):
        continue_targeting(target)         # Stick to current target

    else:
        find_new_closest_enemy()           # Re-acquire target
```

**Effect:** Units won't constantly switch targets mid-combat. If their target runs away, they'll continue pursuing it for up to `leash_extra` (2.0) units beyond the weapon range before giving up and switching to a closer enemy. This creates natural "dueling" behavior — two units will chase each other rather than constantly switching to the nearest third party.

**Why this matters:** Without persistent targets, agents would waste actions micro-managing target switches every frame. With it, combat naturally resolves into focused engagements.

---

## 7. Combat Damage Formula

The final damage calculation applied after hit detection:

```python
damage = clamp(
    base_damage * type_multiplier + bonus,
    min=0.5   # Never deal less than 0.5 damage per hit
)
damage = max(damage - armor, min=0.5)       # Armor reduction (floor still 0.5)
unit.hp -= damage
```

| Parameter | Source | Example |
|---|---|---|
| `base_damage` | Weapon definition (e.g., Blade.dmg=8) | 8 |
| `type_multiplier` | `damage_matrix[unit_type][target_type]` | 1.5 (Melee→Ranged) |
| `bonus` | `bonus_matrix[unit_type][target_type]` | +2 |
| `armor` | Unit's armor stat (e.g., Tank.armor=2.0) | 2.0 |

**Example:** Melee unit attacks Ranged Tank:
```
damage = clamp(8 * 1.5 + 2 - 2, min=0.5) = clamp(12, min=0.5) = 12
```

Melee unit attacks Tank (no advantage):
```
damage = clamp(8 * 0.7 + 2 - 2, min=0.5) = clamp(5.6, min=0.5) = 5.6
```

---

## 8. Fog of War (`FogState`)

**What it does:** Tracks visibility state for enemy units. Invisible enemies are completely hidden from the observation — their features appear as zeros or padding values.

**FogState fields (per enemy unit):**
```python
@dataclass
class FogState:
    last_seen_pos: jnp.ndarray       # (x, y) position when last visible
    alive: bool                      # Was the unit alive at last sighting?
    hp: float                        # HP at last sighting (for health bar estimate)
    time: int                        # Steps since last sighting
```

**Visibility update (per step):**
1. After combat resolves, check if any ally can "see" each enemy (within sensor range).
2. If visible: update `FogState` with current position, HP, alive status, reset timer.
3. If not visible: increment `time` counter but keep existing state unchanged.

**Effect on observation:** In the 63-dim observation vector, invisible enemies have their health and type features set to 0 (or a special "unknown" value). The agent must actively position units to maintain information about hidden enemies. This creates a scout-vs-conceal dynamic that adds POMDP complexity beyond what the physics alone provides.

---

## Parameter Glossary

| Symbol | Value | Description |
|---|---|---|
| `friction` | ~0.95 per step | Velocity damping for smooth movement |
| `stiffness` (hard collider) | 1.2 | Repulsion force multiplier when units significantly overlap |
| `hard_collision_threshold` | 0.2 × (r_i + r_j) | Distance below which hard collision activates |
| `leash_extra` | 2.0 | Extra range beyond weapon for target persistence |
| `hit_radius` (projectile) | 0.6 | Distance from target center for projectile hit detection |
| `lost_range` (projectile) | 40 | Distance from spawn where projectile expires |
| `min_damage` | 0.5 | Floor on all damage calculations (prevents zero-damage deadlock) |

## JIT Compilation Notes

All mechanics functions must be:
1. **Pure** — no global state, no I/O, no random numbers (pass PRNG key as argument)
2. **No Python loops** — use `lax.scan` for iteration over units or weapon slots
3. **Shape-stable** — no dynamic shapes that would cause XLA retracing (use `lax.dynamic_slice` if slices vary)

If a new mechanic is added, test it with:
```python
import jax

@jax.jit
def test_mechanic(state):
    return apply_new_physics(state)

# Warm-up compilation
state = make_dummy_state()
test_mechanic(state)
# Second call validates no retracing errors
test_mechanic(state.copy())
```
