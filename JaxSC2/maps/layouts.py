import jax.numpy as jnp
from typing import NamedTuple

class VariantConfig(NamedTuple):
    id: int
    name: str
    n_ally: int
    n_enemy: int
    layout_type: int # 0: Base, 1: Combat, 2: Navigate

# We define regions as a fixed JAX array so it is fully JIT-compilable.
# Shape: (6, 4) -> [x_min, x_max, y_min, y_max] for regions R1 through R6
REGION_COORDS = jnp.array([
    [0.1, 0.3, 0.7, 0.9], # 0: R1 (L_TOP)
    [0.1, 0.3, 0.4, 0.6], # 1: R2 (L_MID)
    [0.1, 0.3, 0.1, 0.3], # 2: R3 (L_BOT)
    [0.7, 0.9, 0.7, 0.9], # 3: R4 (R_TOP)
    [0.7, 0.9, 0.4, 0.6], # 4: R5 (R_MID)
    [0.7, 0.9, 0.1, 0.3], # 5: R6 (R_BOT)
])

# Pre-defined side indices for easy masking
LEFT_REGIONS = jnp.array([0, 1, 2])
RIGHT_REGIONS = jnp.array([3, 4, 5])

# Definition of the 9 configurations
VARIANTS = {
    # V1 (Easy): 5v3
    "V1_Base":     VariantConfig(0, "V1_Base",     5, 3, 0),
    "V1_Combat":   VariantConfig(1, "V1_Combat",   5, 3, 1),
    "V1_Navigate": VariantConfig(2, "V1_Navigate", 5, 3, 2),
    
    # V2 (Medium): 5v5
    "V2_Base":     VariantConfig(3, "V2_Base",     5, 5, 0),
    "V2_Combat":   VariantConfig(4, "V2_Combat",   5, 5, 1),
    "V2_Navigate": VariantConfig(5, "V2_Navigate", 5, 5, 2),
    
    # V3 (Hard): 5v8
    "V3_Base":     VariantConfig(6, "V3_Base",     5, 8, 0),
    "V3_Combat":   VariantConfig(7, "V3_Combat",   5, 8, 1),
    "V3_Navigate": VariantConfig(8, "V3_Navigate", 5, 8, 2),
}
