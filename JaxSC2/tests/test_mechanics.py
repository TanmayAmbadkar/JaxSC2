import jax
import jax.numpy as jnp
from JaxSC2.env.mechanics import apply_hard_collisions, integrate_velocity, compute_visibility

def test_hard_collisions():
    # Place two units nearly on top of each other
    pos = jnp.array([[10.0, 10.0], [10.001, 10.0]])
    alive = jnp.array([True, True])
    radius = 0.6
    
    new_pos = apply_hard_collisions(pos, alive, radius)
    dist = jnp.linalg.norm(new_pos[0] - new_pos[1])
    
    # Should be pushed apart
    assert dist > 0.01
    # Should not be further than required by stiffness
    assert dist <= 2 * radius * 1.2 # stiffness factor

def test_integrate_velocity():
    pos = jnp.array([0.0, 0.0])
    vel = jnp.array([0.0, 0.0])
    accel = jnp.array([1.0, 0.0]) # Move East
    max_speed = 0.3
    
    new_pos, new_vel = integrate_velocity(pos, vel, accel, max_speed)
    
    assert new_pos[0] > 0
    assert new_vel[0] > 0
    assert jnp.linalg.norm(new_vel) <= max_speed + 1e-6

def test_visibility():
    ally_pos = jnp.array([[0.0, 0.0]])
    enemy_pos = jnp.array([[5.0, 0.0], [20.0, 0.0]])
    vision_radius = 10.0
    
    visible = compute_visibility(ally_pos, jnp.array([True]), enemy_pos, vision_radius)
    
    assert visible[0] == True  # Within 10m
    assert visible[1] == False # Further than 10m
