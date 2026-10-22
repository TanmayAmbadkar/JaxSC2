"""Pure objective transitions: no navigation victory leaks into combat tasks."""

import jax
import jax.numpy as jnp


def transition(env, key, state, world):
    kind, n = env.spec.kind, env.num_allies
    s, old = world.smax_state, state.world.smax_state
    next_state = state.replace(world=world)
    if kind == "beacon":
        reached = jnp.any(s.unit_alive[:n] &
                          (jnp.linalg.norm(s.unit_positions[:n] - world.beacon_pos, axis=-1) < 1.2))
        goal = env.distant_beacon(key, s.unit_positions[:n])
        return next_state.replace(score=state.score + reached.astype(jnp.float32),
                                  world=world.replace(beacon_pos=jnp.where(reached, goal, world.beacon_pos)))
    if kind == "shards":
        distance = jnp.linalg.norm(state.entity_positions[:, None, :] - s.unit_positions[None, :n, :], axis=-1)
        collected = (state.entity_amount > 0) & jnp.any((distance < 1.2) & s.unit_alive[None, :n], axis=-1)
        amount = jnp.where(collected, 0., state.entity_amount)
        exhausted = ~jnp.any(amount > 0)
        candidates = env.map.sample_positions(key, env.entity_capacity * 64).reshape((env.entity_capacity, 64, 2))
        valid = jnp.all(jnp.linalg.norm(candidates[:, :, None, :] - s.unit_positions[None, None, :n, :], axis=-1) > 2., axis=-1)
        positions = candidates[jnp.arange(env.entity_capacity), jnp.argmax(valid, axis=1)]
        return next_state.replace(
            entity_amount=jnp.where(exhausted, (state.entity_kind == 1).astype(jnp.float32), amount),
            entity_positions=jnp.where(exhausted, positions, state.entity_positions),
            wave=state.wave + exhausted.astype(jnp.int32), score=state.score + jnp.sum(collected))
    killed = jnp.sum(old.unit_alive[n:] & ~s.unit_alive[n:])
    deaths = jnp.sum(old.unit_alive[:n] & ~s.unit_alive[:n])
    weight = 10. if kind == "roaches" else 5. if kind == "banelings" else 1.
    cleared = ~jnp.any(s.unit_alive[n:]) & jnp.any(s.unit_alive[:n])
    # Waves are continuous tasks. Victory is recorded, not an episode-ending
    # terminal bonus; health is preserved for surviving allied veterans.
    world = jax.lax.cond(cleared, lambda w: env.spawn_armies(key, w), lambda w: w, world)
    return next_state.replace(world=world, wave=state.wave + cleared.astype(jnp.int32),
                              score=state.score + weight * killed - deaths)
