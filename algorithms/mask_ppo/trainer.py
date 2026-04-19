import argparse

from algorithms.mask_ppo import MaskPPO
from JaxSC2.env.env import JaxSC2Env


def main():
    parser = argparse.ArgumentParser(description="MaskPPO Training")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=30_000_000)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=200)
    args = parser.parse_args()

    config = {
        "NUM_ENVS": args.num_envs,
        "ROLLOUT_LEN": args.rollout_len,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "LR": args.lr,
        "CLIP_EPS": args.clip_eps,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "ENTROPY_COEFF": args.entropy_coeff,
        "VF_COEFF": args.vf_coeff,
        "LOG_INTERVAL": args.log_interval,
        "CKPT_INTERVAL": args.ckpt_interval,
        "SEED": args.seed,
    }

    env = JaxSC2Env(variant_name="V1_Base")
    model = MaskPPO(config=config)

    print(f"MaskPPO Training | gamma={args.gamma} | total_steps={args.total_steps}")
    model.train(env, total_steps=args.total_steps)


if __name__ == "__main__":
    main()
