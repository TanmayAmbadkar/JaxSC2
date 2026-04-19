import jax
import jax.numpy as jnp
from JaxSC2.env.mechanics import (
    apply_mass_collisions, 
    integrate_velocity, 
    update_fog_memory, 
    FogState, 
    apply_high_fidelity_combat
)

def test_mass_collisions():
    # Heavy unit (Mass 5) vs Light unit (Mass 1)
    # They overlap. Heavy unit should move less than light unit.
    # Dist 1.0, Radii 0.6 -> Overlap 0.2 (below max_step clamp 0.3)
    pos = jnp.array([[10.0, 10.0], [11.0, 10.0]])
    alive = jnp.array([True, True])
    unit_types = jnp.array([2, 1]) # Tank (Heavy), Ranged (Light)
    
    type_radius = jnp.array([0.6, 0.6, 0.6])
    type_mass = jnp.array([2.0, 1.0, 5.0]) # Melee=2, Ranged=1, Tank=5
    
    new_pos = apply_mass_collisions(pos, alive, unit_types, type_radius, type_mass)
    
    disp_heavy = jnp.linalg.norm(new_pos[0] - pos[0])
    disp_light = jnp.linalg.norm(new_pos[1] - pos[1])
    
    # Heavy (Tank, i=2 -> index 0) has mass 5
    # Light (Ranged, i=1 -> index 1) has mass 1
    # Disp light should be 5x disp heavy
    assert disp_light > disp_heavy
    assert jnp.abs(disp_light - 5.0 * disp_heavy) < 1e-4

def test_integrate_velocity():
    pos = jnp.array([0.0, 0.0])
    vel = jnp.array([0.0, 0.0])
    accel = jnp.array([1.0, 0.0]) # Move East
    max_speed = 0.3
    
    new_pos, new_vel = integrate_velocity(pos, vel, accel, max_speed)
    
    assert new_pos[0] > 0
    assert new_vel[0] > 0
    assert jnp.linalg.norm(new_vel) <= max_speed + 1e-6

def test_fog_memory():
    fog = FogState(
        last_seen_pos=jnp.array([[0.0, 0.0]]),
        last_seen_alive=jnp.array([True]),
        last_seen_time=jnp.array([0])
    )
    
    ally_pos = jnp.array([[0.0, 0.0]])
    enemy_pos = jnp.array([[20.0, 0.0]]) # OOB
    vision_radius = 10.0
    
    new_fog, visible = update_fog_memory(fog, ally_pos, jnp.array([True]), enemy_pos, jnp.array([True]), vision_radius, 1)
    
    assert visible[0] == False
    # Memory should persist (0,0) from initialization since it wasn't seen at (20,0)
    assert jnp.array_equal(new_fog.last_seen_pos[0], jnp.array([0.0, 0.0]))
    
    # Move enemy into vision
    enemy_pos_v = jnp.array([[5.0, 0.0]])
    new_fog_v, visible_v = update_fog_memory(new_fog, ally_pos, jnp.array([True]), enemy_pos_v, jnp.array([True]), vision_radius, 2)
    
    assert visible_v[0] == True
    assert jnp.array_equal(new_fog_v.last_seen_pos[0], jnp.array([5.0, 0.0]))
    assert new_fog_v.last_seen_time[0] == 2

def test_armor_and_bonus():
    # 3 Units: Melee (0), Ranged (1), Tank (2)
    unit_types = jnp.array([0, 2]) # Melee (Atk) vs Tank (Def)
    unit_health = jnp.array([100.0, 100.0])
    unit_alive = jnp.array([True, True])
    unit_positions = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    unit_teams = jnp.array([0, 1])
    unit_armor = jnp.array([0.0, 5.0]) # Tank has 5 armor
    
    targets = jnp.array([1, 0])
    attack_timers = jnp.array([0, 0])
    
    # Dummy proj state
    proj_state = (jnp.zeros((32, 2)), jnp.zeros((32, 2)), jnp.ones(32, dtype=jnp.int32)*-1, 
                  jnp.zeros(32), jnp.zeros(32, dtype=jnp.bool_), jnp.zeros(32, dtype=jnp.int32))
    
    combat_cfg = {
        "type_windups": jnp.array([1, 1, 1]),
        "type_cooldowns": jnp.array([6, 6, 6]),
        "type_ranges": jnp.array([1.5, 6.0, 4.0]),
        "type_damages": jnp.array([10.0, 5.0, 4.0]),
        "damage_matrix": jnp.ones((3, 3)),
        "bonus_matrix": jnp.zeros((3, 3))
    }
    
    # Melee (index 0) fires at index 1 (Tank).
    # Base dmg 10. Tank armor 5. Net dmg = 5.
    hp1, _, _, _ = apply_high_fidelity_combat(
        unit_health, unit_alive, unit_positions, unit_types, unit_teams, 
        unit_armor, attack_timers, targets, proj_state, combat_cfg
    )
    
    # Tank health (index 1) should be 95
    assert hp1[1] == 95.0
    
    # Test bonus damage
    combat_cfg["bonus_matrix"] = combat_cfg["bonus_matrix"].at[0, 2].set(10.0) # Melee bonus vs Tank
    hp2, _, _, _ = apply_high_fidelity_combat(
        unit_health, unit_alive, unit_positions, unit_types, unit_teams, 
        unit_armor, attack_timers, targets, proj_state, combat_cfg
    )
    # Base 10 + Bonus 10 - Armor 5 = 15 dmg
    assert hp2[1] == 85.0
