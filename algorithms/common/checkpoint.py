import os
import pickle
from flax.training import train_state

def save_checkpoint(path, state: train_state.TrainState, step, **metadata):
    """
    Saves model parameters and step count to a pickle file.
    """
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "params": state.params,
            "opt_state": state.opt_state,
            "step": step,
            "optimizer_step": state.step,
            "metadata": metadata,
        }, f)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, state: train_state.TrainState, return_metadata=False):
    """
    Loads model parameters and optimizer state from a pickle file.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return (state, 0, {}) if return_metadata else (state, 0)
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    try:
        new_state = state.replace(
            params=data["params"],
            opt_state=data["opt_state"],
            step=data.get("optimizer_step", state.step),
        )
    except Exception as e:
        print(f"Warning: opt_state restore failed ({e}), loading params only.")
        new_state = state.replace(params=data["params"])
        
    print(f"Loaded checkpoint from {path} (Step: {data['step']})")
    if return_metadata:
        return new_state, data["step"], data.get("metadata", {})
    return new_state, data["step"]
