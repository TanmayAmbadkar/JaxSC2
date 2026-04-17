import jax
from algorithms.a2c.trainer import A2CTrainer

def test_a2c_smoke():
    trainer = A2CTrainer({"NUM_ENVS": 4, "ROLLOUT_LEN": 16})
    trainer.train(4 * 16)
    print("A2C Smoke Test Passed")

if __name__ == "__main__":
    test_a2c_smoke()
