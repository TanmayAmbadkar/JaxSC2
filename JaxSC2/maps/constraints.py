import jax.numpy as jnp

def enforce_bridge_terrain(proposed_pos: jnp.ndarray, old_pos: jnp.ndarray, map_width: float, map_height: float) -> jnp.ndarray:
    """
    Prevents units from walking through the central cliff and staying within map bounds.
    Uses 'Push-to-Safety' instead of just reverting to old_pos.
    """
    # 1. Map Boundary Clamping
    clamped_x = jnp.clip(proposed_pos[:, 0], 0.2, map_width - 0.2)
    clamped_y = jnp.clip(proposed_pos[:, 1], 0.2, map_height - 0.2)
    new_pos = jnp.stack([clamped_x, clamped_y], axis=-1)
    
    # 2. Cliff Zone Check (x in [0.45, 0.55] relative)
    x_rel = new_pos[:, 0] / map_width
    y_rel = new_pos[:, 1] / map_height
    
    in_cliff_x = (x_rel > 0.45) & (x_rel < 0.55)
    
    # Bridge Coordinates (matches render_demo)
    # Bridge 1: 0.15 - 0.35, Bridge 2: 0.65 - 0.85
    on_bridge_y = ((y_rel >= 0.15) & (y_rel <= 0.35)) | ((y_rel >= 0.65) & (y_rel <= 0.85))
    
    in_cliff_illegal = in_cliff_x & (~on_bridge_y)
    
    # Push to nearest safe X edge if illegal
    # If old_x was < 0.45, push to 0.45. If was > 0.55, push to 0.55.
    old_x_rel = old_pos[:, 0] / map_width
    safe_x_rel = jnp.where(old_x_rel <= 0.45, 0.449, 0.551)
    safe_x = safe_x_rel * map_width
    
    final_x = jnp.where(in_cliff_illegal, safe_x, new_pos[:, 0])
    
    return jnp.stack([final_x, new_pos[:, 1]], axis=-1)
