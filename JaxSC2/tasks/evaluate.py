"""Run all task adaptations with scripted agents; emit metrics and GIF gallery."""

import argparse
import html
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from JaxSC2.factory import make
from JaxSC2.tasks.types import TASKS
from JaxSC2.tasks.scripted import scripted_action


def make_rollout(env, save_history=False):
    def episode(key):
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)

        def step(carry, _):
            key, state, count, violations = carry
            key, step_key = jax.random.split(key)
            action = scripted_action(env, state)
            _, result, _, _, _ = env.step(step_key, state, action)
            walkable = env.map.is_walkable(result.world.smax_state.unit_positions,
                                           env.map_width, env.map_height)
            bad = jnp.sum(~walkable & result.world.smax_state.unit_alive)
            return (key, result, count + (~state.done).astype(jnp.int32),
                    violations + jnp.where(state.done, 0, bad)), result if save_history else None

        final, history = jax.lax.scan(step, (key, state, jnp.int32(0), jnp.int32(0)), None,
                                       length=env.max_steps)
        return final[1:], history
    return jax.jit(jax.vmap(episode))


def render(env, states, length, path, stride=8):
    import imageio.v2 as imageio
    import pygame
    from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

    renderer = ProductionRenderer(headless=True, width=480, height=480,
                                  map_dims=(env.map_width, env.map_height), terrain=env.map)
    renderer.zoom = .92
    font = pygame.font.Font(None, 19)
    frames = []
    for tick in sorted(set([*range(0, length, stride), length - 1])):
        state = jax.tree_util.tree_map(lambda x: x[tick], states)
        frame = state_to_frame(state.world, env.core.unit_type_hp, env.core.unit_type_range)
        if env.spec.kind != "beacon":
            frame.pop("beacon_pos")
        surface = renderer.render_frame(frame, frame, 1.)
        for pos, amount, kind in zip(state.entity_positions, state.entity_amount, state.entity_kind):
            if amount <= 0:
                continue
            color = (0, 220, 255) if kind == 1 else (100, 150, 255) if kind == 2 else (80, 230, 100)
            pygame.draw.circle(surface, color, renderer.world_to_screen(pos), 4)
        for i, kind in enumerate(state.building_kind):
            if kind == 0:
                continue
            x, y = renderer.world_to_screen(state.building_positions[i])
            color = (180, 180, 100) if state.building_remaining[i] else (80, 180, 240)
            pygame.draw.rect(surface, color, (x - 9, y - 7, 18, 14), 2)
            surface.blit(font.render(("", "CC", "D", "B", "G")[int(kind)], True, color), (x - 8, y - 22))
        labels = [f'{env.name} | scripted | step {tick + 1}',
                  f'Score {float(state.score):.1f} | Waves {int(state.wave)} | Blue {int(jnp.sum(state.world.smax_state.unit_alive[:env.num_allies]))}']
        if env.economic:
            labels.append(f'Minerals {float(state.minerals):.0f} | Gas {float(state.gas):.0f} | Marines built {int(state.marines_built)}')
        for line, label in enumerate(labels):
            surface.blit(font.render(label, True, (255, 255, 255)), (8, 8 + line * 20))
        frames.append(pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy())
    frames.extend([frames[-1]] * 10)
    imageio.mimsave(path, frames, duration=80, loop=0)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["all", *TASKS], default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render", choices=["none", "all"], default="none")
    parser.add_argument("--output", type=Path, default=Path("runs/task_suite"))
    args = parser.parse_args()
    names = TASKS if "all" in args.tasks else args.tasks
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        env = make(name, max_steps=args.max_steps)
        final, history = jax.device_get(make_rollout(env, args.render == "all")(
            jnp.stack([jax.random.PRNGKey(seed) for seed in args.seeds])))
        for i, seed in enumerate(args.seeds):
            state = jax.tree_util.tree_map(lambda x: x[i], final[0])
            row = {"task": name, "seed": seed, "steps": int(final[1][i]), "max_steps": env.max_steps,
                   "score": float(state.score), "waves": int(state.wave),
                   "gathered_minerals": float(state.gathered_minerals), "gathered_gas": float(state.gathered_gas),
                   "marines_built": int(state.marines_built),
                   "allies_alive": int(np.sum(state.world.smax_state.unit_alive[:env.num_allies])),
                   "terrain_violations": int(final[2][i]),
                   "observation_dim": env.observation_space.shape[0],
                   "enemy_mode": env.core.enemy_mode, "fog": env.core.enable_fow,
                   "ally_capacity": env.num_allies, "enemy_capacity": env.num_enemies,
                   "controller": "scripted", "adaptation": "simplified; not exact SC2 semantics"}
            if args.render == "all":
                row["gif"] = f'{name}_{seed}.gif'
                episode = jax.tree_util.tree_map(lambda x: x[i], history)
                render(env, episode, row["steps"], args.output / row["gif"])
            rows.append(row)
            print(json.dumps(row), flush=True)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    cards = "".join(f'<article><h2>{html.escape(r["task"])} / seed {r["seed"]}</h2>'
                    f'<p>Score {r["score"]:.1f}; waves {r["waves"]}; terrain violations {r["terrain_violations"]}</p>'
                    + (f'<img src="{r["gif"]}" width="480">' if "gif" in r else "") + '</article>' for r in rows)
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SC2-inspired task suite</title>'
        '<style>body{background:#101916;color:#eef5f1;font:16px system-ui;margin:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}img{max-width:100%}</style>'
        '<h1>SC2-inspired task adaptations</h1><p>Scripted baselines, not trained policies. '
        'Observer view: hidden enemies may be shown; policy observations obey fog. '
        'Simplified economy and unit stats, not exact SC2 replicas.</p><main>' + cards + '</main>')
    print(f'Gallery: {(args.output / "index.html").resolve()}')


if __name__ == "__main__":
    main()
