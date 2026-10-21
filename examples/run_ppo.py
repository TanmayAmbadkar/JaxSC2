"""Launch standard PPO with isolated run artifacts and optional background mode.

Example: python examples/run_ppo.py --total-steps 30000000 --background
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="V2_Base")
    parser.add_argument("--total-steps", type=int, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=500)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enemy-mode", choices=["static", "guard", "patrol", "aggressive"], default="guard")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume weights/optimizer state toward the total target step count")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if min(args.total_steps, args.num_envs, args.rollout_len, args.update_epochs,
           args.num_minibatches, args.checkpoint_interval) < 1 or args.eval_interval < 0:
        parser.error("Training sizes and checkpoint interval must be positive; evaluation interval may be zero")
    batch = args.num_envs * args.rollout_len
    if batch % args.num_minibatches:
        parser.error("Minibatches must divide NUM_ENVS * ROLLOUT_LEN")
    if args.resume is not None:
        args.resume = args.resume.resolve()
        if not args.resume.is_file():
            parser.error(f"Checkpoint does not exist: {args.resume}")
    target = math.ceil(args.total_steps / batch) * batch
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (args.run_dir or REPO / "runs" / f"ppo_{args.variant}_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.background:
        if (run_dir / "pid").exists():
            previous_pid = int((run_dir / "pid").read_text())
            try:
                os.kill(previous_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Run process {previous_pid} is still alive; refusing a second writer")
        # A separate session and explicit file descriptors keep the training
        # process alive after this launcher or the invoking terminal exits.
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[arg for arg in sys.argv[1:] if arg != "--background"],
                   "--run-dir", str(run_dir)]
        with (run_dir / "training.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        (run_dir / "pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "run_dir": str(run_dir), "target_steps": target}, indent=2))
        return

    import jax
    import numpy as np
    from algorithms.ppo import PPO
    from algorithms.ppo.eval import evaluate
    from JaxSC2.env.env import JaxSC2Env

    # Refuse to overwrite an existing run's metadata or checkpoint namespace.
    if (run_dir / "config.json").exists() and args.resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}")
    env = JaxSC2Env(args.variant, enemy_ai=True, enemy_mode=args.enemy_mode,
                   enable_collision=True, enable_armor=True, enable_fow=False)
    model = PPO({
        "NUM_ENVS": args.num_envs, "ROLLOUT_LEN": args.rollout_len,
        "UPDATE_EPOCHS": args.update_epochs, "NUM_MINIBATCHES": args.num_minibatches,
        "LOG_INTERVAL": 1, "CKPT_INTERVAL": args.checkpoint_interval,
        "EVAL_INTERVAL": args.eval_interval, "EVAL_EPISODES": 16,
        "SEED": args.seed, "LOG_DIR": str(run_dir / "logs"),
        "CKPT_DIR": str(run_dir / "checkpoints"),
    })
    metadata = {
        "algorithm": "PPO", "variant": args.variant, "enemy_ai": True,
        "enemy_mode": args.enemy_mode, "enable_collision": True,
        "enable_armor": True, "enable_fow": False, "max_episode_steps": env.max_steps,
        "requested_steps": args.total_steps, "target_steps": target,
        "config": model.config, "devices": [str(device) for device in jax.devices()],
        "python": sys.executable, "jax_version": jax.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    else:
        original = json.loads((run_dir / "config.json").read_text())
        keys = ("variant", "enemy_mode", "enable_collision", "enable_armor", "enable_fow", "target_steps")
        if any(original[key] != metadata[key] for key in keys) or original["config"] != model.config:
            raise ValueError("Resume arguments must match the existing run configuration")
    (run_dir / "pid").write_text(str(os.getpid()) + "\n")

    def status(state, **extra):
        temporary = run_dir / "status.tmp"
        temporary.write_text(json.dumps({
            "status": state, "pid": os.getpid(), "target_steps": target,
            "completed_steps": model.total_steps,
            "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
        }, indent=2))
        temporary.replace(run_dir / "status.json")

    save = model.save

    def save_with_progress(path):
        save(path)
        status("running", checkpoint=str(path))

    model.save = save_with_progress

    status("running")
    print(json.dumps(metadata, indent=2), flush=True)
    try:
        if args.resume is not None:
            model.load(args.resume)
            model.train(env, 0)
            if model.total_steps > target or (target - model.total_steps) % batch:
                raise ValueError("Checkpoint step count must align with this run's rollout batches and target")
            history_path = run_dir / "resume_history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            history.append({"checkpoint": str(args.resume), "step": model.total_steps,
                            "at": datetime.now(timezone.utc).isoformat(),
                            "logging": "completed raw episode returns; incomplete episodes restart on resume"})
            history_path.write_text(json.dumps(history, indent=2))
            status("running", resumed_from=str(args.resume))
            print(f"Resumed at {model.total_steps:,} steps; target {target:,}. Episode counters start with fresh environments.", flush=True)
        model.train(env, target - model.total_steps)
        if not all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(model.state.params)):
            raise FloatingPointError("Training produced non-finite model parameters")
        final_path = run_dir / "checkpoints" / f"final_{model.total_steps}.pkl"
        model.save(final_path)
        metrics = evaluate(env, model.state.params, model.model,
                           jax.random.PRNGKey(args.seed + 1), num_episodes=64, max_steps=env.max_steps)
        results = {name: float(value) for name, value in metrics.items()}
        (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
        status("complete", checkpoint=str(final_path), evaluation=results)
        print("Training complete", model.total_steps, results, flush=True)
    except BaseException as error:
        if model.state is not None:
            model.save(run_dir / "checkpoints" / f"interrupted_{model.total_steps}.pkl")
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
