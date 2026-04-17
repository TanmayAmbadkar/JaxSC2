import jax
from algorithms.mask_ppo.trainer import MaskedPPOTrainer

def test_mask_ppo_smoke():
    trainer = MaskedPPOTrainer({"NUM_ENVS": 4, "ROLLOUT_LEN": 16, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1})
    trainer.train(4 * 16)
    print("MaskPPO Smoke Test Passed")

if __name__ == "__main__":
    test_mask_ppo_smoke()
