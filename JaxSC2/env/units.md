# JaxSC2 Unit Reference

This document defines all unit types, their stats, weapons, and the design rationale behind each stat.

---

## Unit Types

Units are indexed by integer: `0 = Melee`, `1 = Ranged`, `2 = Tank`. The index is used in the observation vector (5th feature of enemy unit data) and in the damage/bonus matrices.

### Melee Unit (Index 0) — "Marine-like Assault"

| Stat | Value | Rationale |
|---|---|---|
| HP | 100 | Medium survivability — can absorb ~2-3 ranged shots before dying |
| Speed | 0.3 | Fastest unit — designed to close distance quickly |
| Acceleration | 0.10 | Rapid acceleration for positioning and flanking |
| Mass | 2.0 | Medium mass — gets pushed around by Tanks but overpowers Ranged |
| Armor | 1.0 | Light protection — mitigates ~1 damage per hit from most weapons |
| Weapon | BladeWeapon (dmg=8, range=1.5, windup=1, cooldown=6) | Fast attack cycle: ~1 hit per 5 steps (6-1 cooldown) |

**Design intent:** Melee is the "glass cannon" that excels against Ranged (1.5× damage) but struggles against Tank (0.7× damage). Its high speed allows it to exploit the bridge choke points and flank enemies before Ranged units can react. The short windup (1 step) means it attacks almost immediately when in range — great for burst damage but limited DPS due to the 6-step cooldown.

**Optimal behavior:** Close distance fast, engage Ranged first (type advantage), avoid Tank engagements unless outnumbered.

---

### Ranged Unit (Index 1) — "Zealot-like Striker"

| Stat | Value | Rationale |
|---|---|---|
| HP | 45 | Fragile — can absorb ~1 melee hit before dying. Encourages kiting. |
| Speed | 0.25 | Moderate speed — slower than Melee, faster than Tank. Balanced mobility. |
| Acceleration | 0.08 | Slower acceleration — takes ~2-3 steps to reach full speed from rest. |
| Mass | 1.0 | Lightest unit — easily pushed by collisions, doesn't hold ground well. |
| Armor | 0.0 | No armor — every point of incoming damage is real. Vulnerable to everything. |
| Weapon | GaussRifle (dmg=5, range=6.0, windup=2, cooldown=8) | Long-range weapon: ~1 hit per 7 steps (8-1 cooldown), can attack before being engaged. |

**Design intent:** Ranged is the "high risk, high reward" unit — massive range advantage (6.0 vs 1.5 for Melee) but extremely fragile when engaged. The type advantage (Tank: +50%) makes it the primary anti-Tank unit despite having no armor. When it attacks Melee, it only deals 80% damage (0.8×), making melee engagements punitive.

**Optimal behavior:** Engage from maximum range, kite away when Melee approaches, save for Tank fights where its type advantage is decisive. Poor against melee-heavy compositions.

---

### Tank Unit (Index 2) — "Stalker-like Heavy"

| Stat | Value | Rationale |
|---|---|---|
| HP | 200 | Double Melee, ~4× Ranged — designed to absorb sustained fire. Can survive 2-3 ranged hits and still be functional. |
| Speed | 0.18 | Slowest unit — takes ~5+ steps to reach top speed from rest. Tanky but predictable. |
| Acceleration | 0.05 | Very slow acceleration — movement is lumbering, making it an easy projectile target when moving. |
| Mass | 5.0 | Heaviest unit — dominates collision resolution, pushes through light units. |
| Armor | 2.0 | Substantial armor — absorbs ~2 damage per hit, significantly reducing incoming DPS. |
| Weapon | PlasmaCannon (dmg=4, range=4.0, windup=3, cooldown=12) | Slow but reliable: ~1 hit per 9 steps (12-3 cooldown), moderate range. |

**Design intent:** Tank is the "anchor" unit — high HP and armor make it an excellent frontline blocker, but its slow speed and acceleration make positioning difficult. The 4.0 range is a compromise: not as long as Ranged (6.0) but better than Melee (1.5). Against Melee, it deals +50% damage (1.5×) and its armor makes melee attacks less effective against it. However, Ranged exploits Tank with +50% damage while Tank's armor often can't absorb the difference.

**Optimal behavior:** Hold choke points (bridges), absorb damage while Ranged/Melee flank, avoid open-field chases against faster units.

---

## Weapon Reference

```python
@dataclass
class WeaponDef:
    damage: float     # Base damage before type multiplier and armor reduction
    range_: float     # Max distance to spawn a projectile (or melee hit)
    windup: int       # Steps to enter attack phase after cooldown ends
    cooldown: int     # Total cycle length (windup + fire + reload)

# Melee weapon
@dataclass
class BladeWeapon(WeaponDef):
    def __init__(self):
        super().__init__(damage=8, range_=1.5, windup=1, cooldown=6)

# Ranged weapon
@dataclass
class GaussRifle(WeaponDef):
    def __init__(self):
        super().__init__(damage=5, range_=6.0, windup=2, cooldown=8)

# Tank weapon
@dataclass
class PlasmaCannon(WeaponDef):
    def __init__(self):
        super().__init__(damage=4, range_=4.0, windup=3, cooldown=12)
```

### Weapon Cycle Timing

The cycle is: `windup → fire → cooldown - windup` (reload phase)

| Weapon | Windup | Fire | Reload (cd-windup) | Total Cycle | Approx. DPS* |
|---|---|---|---|---|---|
| Blade | 1 step | Instant (melee) | 5 steps | ~6 steps/hit | 8/6 ≈ **1.33** |
| Gauss | 2 steps | Projectile (0.6 hit radius) | 6 steps | ~8 steps/hit | 5/8 ≈ **0.63** |
| Plasma | 3 steps | Projectile (0.6 hit radius) | 9 steps | ~12 steps/hit | 4/12 ≈ **0.33** |

\* Rough DPS — actual depends on hit rate (projectiles can miss, windup means not all steps produce hits).

---

## Type Advantage Matrix

```python
damage_matrix = [
    # vs Melee  vs Ranged  vs Tank
    [1.0,      1.5,       0.7],     # Melee attacks
    [0.8,      1.0,       1.5],     # Ranged attacks
    [1.5,      0.7,       1.0],     # Tank attacks
]

bonus_matrix = [
    # vs Melee  vs Ranged  vs Tank
    [+0,       +2,        -1],      # Melee bonus shifts
    [-1,       +0,        +2],      # Ranged bonus shifts
    [+2,       -1,        +0],      # Tank bonus shifts
]
```

### How the Matrix Works

The final damage formula from `mechanics.md` §7:
```python
damage = clamp(base_damage * damage_matrix[row][col] + bonus_matrix[row][col] - armor, min=0.5)
```

**Concrete examples:**

Melee (weapon: Blade, dmg=8) attacks Ranged Target:
```
damage = clamp(8 * 1.5 + 2 - 0, min=0.5) = clamp(14, min=0.5) = 14
Ranged (HP=45) dies in ~3 hits.
```

Tank (weapon: Plasma, dmg=4) attacks Melee Target:
```
damage = clamp(4 * 1.5 + 2 - 1, min=0.5) = clamp(7, min=0.5) = 7
Melee (HP=100, armor=1) dies in ~14 hits.
```

Ranged (weapon: Gauss, dmg=5) attacks Tank Target:
```
damage = clamp(5 * 1.0 + 0 - 2, min=0.5) = clamp(3, min=0.5) = 3
Tank (HP=200, armor=2) dies in ~67 hits — but Ranged can maintain range.
```

**Strategic implication:** The matrix creates a closed rock-paper-scissors cycle with asymmetric magnitudes:
- Melee→Ranged is **very** strong (1.5× +2 bonus) — Melee should always prioritize Ranged
- Tank→Melee is **moderately** strong (1.5× +2 bonus) — Tank wants to engage Melee close
- Ranged→Tank is **moderately** strong (1.5×) but offset by Tank's armor — Ranged wins on range, not raw damage
- The reverse directions are weak (0.7× or 0.8×) — engaging the wrong counter is costly

---

## Unit Composition (Scenarios)

Each scenario variant uses a fixed composition of all 3 unit types. The standard setup is:

**Every ally team:**
```
Allies (5 units, typically): 3× Melee + 2× Ranged
(or mixed based on variant configuration)

Enemies (N units, varies):    Proportional mix of Melee/Ranged/Tank
```

The exact composition per variant is defined in `maps/twobridge.py` under the `VARIANTS` dictionary. The number of enemies scales with variant:
- V1: 3 enemies (easier, good for learning)
- V2: 5 enemies (balanced)
- V3: 8 enemies (hard, requires composition strategy)

---

## Adding a New Unit Type

To add a new unit (e.g., "Sniper"):

1. **Define the weapon** in `units.py`:
```python
class SniperWeapon(WeaponDef):
    def __init__(self):
        super().__init__(damage=12, range_=10.0, windup=5, cooldown=15)
```

2. **Define the unit**:
```python
def create_sniper():
    return UnitDefinition(
        name="Sniper", index=3,  # Next available index
        hp=30, speed=0.2, accel=0.06, mass=0.8, armor=0.5,
        weapon=SniperWeapon(),
    )
```

3. **Extend the matrices** in `env.py` (both `damage_matrix` and `bonus_matrix` grow by 1 row + 1 column):
```python
damage_matrix = jnp.array([
    [1.0, 1.5, 0.7,   ?],     # existing + new row/col for Sniper
    [0.8, 1.0, 1.5,   ?],
    [1.5, 0.7, 1.0,   ?],
    [?,   ?,   ?,     1.0],   # Sniper's own row
])

bonus_matrix = jnp.array([
    [+0, +2, -1,  ?],   # extend each row and add new column
    [-1, +0, +2,  ?],
    [+2, -1, +0,  ?],
    [?,  ?,  ?,   +0],
])
```

4. **Update the observation space** — if num_units > 4, increase the batch dimensions in `env.py` from `(N, ..., 5)` to `(new_N, ..., 5)`.

5. **Add tests** — verify the new unit's damage calculations match expected values (see `JaxSC2/tests/test_mechanics.py`).
