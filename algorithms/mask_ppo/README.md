# Masked Proximal Policy Optimization (MaskPPO)

A specialized PPO implementation that enforces valid-action constraints.

## 🛡 Valid-Action Masking
In StarCraft-style combat, many actions (like attacking a dead or invisible enemy) are illegal. MaskPPO handles this by:
1.  **Logit Masking**: Invalid action logits are set to large negative values (`-1e8`) before the softmax.
2.  **Zero-Loss Gradient**: The policy loss is implicitly masked because the probability of selecting an illegal action is zero.

## 🗝 Highlights
- **Model**: Uses `MaskedActorCritic` which expects an `action_mask` during the forward pass.
- **Trainer**: Automatically stores action masks in the rollout buffer.

## 🛠 Extension Guide
If you add new action types to the environment, update the `get_action_masks` function in `twobridge.py` and ensure the mask shape remains consistent with the algorithm's expectations.
