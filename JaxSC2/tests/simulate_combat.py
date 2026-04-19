import jax
import jax.numpy as jnp
from JaxSC2.env.env import JaxSC2Env, CentralAction

def simulate_battle():
    print("==================================================")
    print("SC2JAX HETEROGENEOUS COMBAT SIMULATION")
    print("==================================================")
    
    # Initialize environment
    env = JaxSC2Env(variant_name="V2_Combat")
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)
    
    # FORCE POSITIONS CLOSER FOR IMMEDIATE COMBAT
    # Allies at (10, 15), Enemies at (11, 15) -> 1m distance
    unit_pos = state.smax_state.unit_positions
    ally_pos = jnp.tile(jnp.array([10.0, 15.0]), (5, 1)) + jax.random.uniform(rng, (5, 2), minval=-0.5, maxval=0.5)
    enemy_pos = jnp.tile(jnp.array([12.0, 15.0]), (5, 1)) + jax.random.uniform(rng, (5, 2), minval=-0.5, maxval=0.5)
    new_unit_pos = jnp.concatenate([ally_pos, enemy_pos])
    state = state.replace(smax_state=state.smax_state.replace(unit_positions=new_unit_pos))
    
    print(f"Engagement Distance: {jnp.linalg.norm(ally_pos[0] - enemy_pos[0]):.2f}m")
    print(f"Unit Types (Allies): {state.smax_state.unit_types[:5]}")
    print(f"Unit Types (Enemies): {state.smax_state.unit_types[5:]}")
    
    # Run for 50 steps with a simple aggressive policy
    for i in range(50):
        # Action: All allies select and attack first enemy
        action = CentralAction(
            who_mask=jnp.ones(5, dtype=jnp.bool_),
            verb=2, # Attack
            direction=0,
            target=0
        )
        
        obs, state, reward, done, info = env.step(rng, state, action)
        
        if i % 10 == 0:
            a_hp = jnp.sum(state.smax_state.unit_health[:5] * state.smax_state.unit_alive[:5])
            e_hp = jnp.sum(state.smax_state.unit_health[5:] * state.smax_state.unit_alive[5:])
            print(f"Step {i:02d} | Ally HP: {a_hp:6.1f} | Enemy HP: {e_hp:6.1f} | Alive: {jnp.sum(state.smax_state.unit_alive[:5])}v{jnp.sum(state.smax_state.unit_alive[5:])}")
            
    print("--------------------------------------------------")
    print("Simulation Complete.")
    print(f"Final Ally Health: {jnp.sum(state.smax_state.unit_health[:5] * state.smax_state.unit_alive[:5]):.1f}")
    print(f"Final Enemy Health: {jnp.sum(state.smax_state.unit_health[5:] * state.smax_state.unit_alive[5:]):.1f}")
    print("==================================================")

if __name__ == "__main__":
    simulate_battle()
