"""Train the existing army policy on a new navigation/combat task adaptation."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from JaxSC2 import make
from JaxSC2.tasks.types import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--algorithm", choices=["ppo", "mask_ppo"], default="ppo")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.total_steps, args.num_envs, args.rollout_len) < 1:
        parser.error("Training sizes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("Episode limit must be positive")
    if TASKS[args.task].kind in ("economy", "production"):
        parser.error("Economy tasks need a macro-policy learner; use the scripted task evaluator for now")
    from algorithms.ppo import PPO
    from algorithms.mask_ppo import MaskPPO
    from algorithms.ppo.eval import evaluate as evaluate_ppo
    from algorithms.mask_ppo.eval import evaluate as evaluate_masked
    import jax

    env = make(args.task, max_steps=args.max_steps)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or Path("runs") / f'{args.algorithm}_{args.task}_{timestamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch = args.num_envs * args.rollout_len
    target = math.ceil(args.total_steps / batch) * batch
    config = {"NUM_ENVS": args.num_envs, "ROLLOUT_LEN": args.rollout_len,
              "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": math.gcd(batch, 8),
              "SEED": args.seed, "LOG_INTERVAL": 1, "EVAL_INTERVAL": 25,
              "EVAL_EPISODES": 16, "CKPT_INTERVAL": 50,
              "LOG_DIR": str(output / "logs"), "CKPT_DIR": str(output / "checkpoints")}
    model = {"ppo": PPO, "mask_ppo": MaskPPO}[args.algorithm](config)
    metadata = {"algorithm": args.algorithm, "task": args.task,
                "max_steps": env.max_steps, "requested_steps": args.total_steps, "target_steps": target,
                "observation_dim": env.observation_space.shape[0], "config": model.config,
                "controller": "shared army command; not decentralized per-unit control"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    model.train(env, target)
    model.save(output / "checkpoints" / f"final_{model.total_steps}.pkl")
    evaluate = evaluate_ppo if args.algorithm == "ppo" else evaluate_masked
    metrics = evaluate(env, model.state.params, model.model, jax.random.PRNGKey(args.seed + 1),
                       num_episodes=64, max_steps=env.max_steps)
    result = {name: float(value) for name, value in metrics.items()}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(output, result)


if __name__ == "__main__":
    main()
