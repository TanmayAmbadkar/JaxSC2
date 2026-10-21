"""Scenario matrix evaluation and visual artifacts for deterministic baselines."""

import argparse
import html
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from JaxSC2.env.env import JaxSC2Env
from JaxSC2.maps.twobridge import TwoBridgeMap
from JaxSC2.scripted.agents import beacon_agent, combat_agent


SETTINGS = {
    "baseline": {},
    "no_collision": {"enable_collision": False},
    "no_armor": {"enable_armor": False},
    "fog": {"enable_fow": True},
    "latency": {"latency_delay": 3},
}


def make_rollout(env):
    """Compile both agents together; identical seeds give identical initial states."""
    def episode(key, agent_id):
        key, reset_key = jax.random.split(key)
        _, initial = env.reset(reset_key)

        def step(carry, _):
            state, done, reward_sum, length, nav, combat, key = carry
            action = jax.lax.cond(
                agent_id == 0, lambda _: beacon_agent(env, state),
                lambda _: combat_agent(env, state), None,
            )
            key, step_key = jax.random.split(key)
            _, next_state, reward, terminal, info = env.step(step_key, state, action)
            next_state = jax.tree_util.tree_map(
                lambda old, new: jnp.where(done, old, new), state, next_state
            )
            carry = (
                next_state, done | terminal, reward_sum + jnp.where(done, 0., reward),
                length + (~done).astype(jnp.int32),
                nav | (~done & info["nav_success"]),
                combat | (~done & info["combat_success"]), key,
            )
            return carry, next_state

        final, history = jax.lax.scan(
            step, (initial, False, 0., jnp.int32(0), False, False, key),
            None, length=env.max_steps,
        )
        return initial, final[:6], history

    return jax.jit(jax.vmap(episode))


def evaluate_case(env, seed=42, rollout=None):
    keys = jnp.broadcast_to(jax.random.PRNGKey(seed), (2, 2))
    initial, final, history = (rollout or make_rollout(env))(keys, jnp.arange(2))
    initial, final, history = jax.device_get((initial, final, history))
    rows, episodes = [], []
    for index, agent in enumerate(("beacon", "combat")):
        start = jax.tree_util.tree_map(lambda x: x[index], initial)
        end = jax.tree_util.tree_map(lambda x: x[index], final[0])
        length = int(final[3][index])
        trajectory = jax.tree_util.tree_map(lambda x: x[index, :length], history)
        alive = end.smax_state.unit_alive
        nav, combat = bool(final[4][index]), bool(final[5][index])
        outcome = ("beacon_reached" if nav else "enemies_defeated" if combat
                   else "allies_defeated" if not np.any(alive[:env.num_allies]) else "timeout")
        start_dist = np.min(np.linalg.norm(
            start.smax_state.unit_positions[:env.num_allies] - start.beacon_pos, axis=-1))
        end_dist = np.min(np.where(alive[:env.num_allies], np.linalg.norm(
            end.smax_state.unit_positions[:env.num_allies] - end.beacon_pos, axis=-1), np.inf))
        enemy_damage = float(np.sum(start.smax_state.unit_health[env.num_allies:])
                             - np.sum(end.smax_state.unit_health[env.num_allies:]))
        pos = trajectory.smax_state.unit_positions
        bounded = bool(np.all(pos >= -1e-5) and np.all(pos <= np.array([env.map_width, env.map_height]) + 1e-5))
        walkable = np.asarray(env.map_instance.is_walkable(
            jnp.asarray(pos), env.map_width, env.map_height,
            env.type_radius[start.smax_state.unit_types],
        ))
        violations = int(np.sum(~walkable & trajectory.smax_state.unit_alive))
        rows.append({
            "agent": agent, "seed": seed, "steps": length, "outcome": outcome,
            "reward": float(final[2][index]), "nav_success": nav, "combat_success": combat,
            "allies_alive": int(np.sum(alive[:env.num_allies])),
            "enemies_alive": int(np.sum(alive[env.num_allies:])),
            "enemy_damage": enemy_damage,
            "beacon_distance_initial": float(start_dist),
            "beacon_distance_final": float(end_dist) if np.isfinite(end_dist) else None,
            "units_in_bounds": bounded,
            "terrain_valid": violations == 0, "terrain_violations": violations,
            "goal_progress": bool((end_dist < start_dist - 1.) if agent == "beacon" else enemy_damage > 0.),
        })
        episodes.append((start, trajectory))
    return rows, episodes


def render_gif(env, episode, row, path, stride=3):
    """Show the full map, exact bridge apertures, objective, and live HP/outcome."""
    import imageio.v2 as imageio
    import pygame
    from JaxSC2.env.renderer import ProductionRenderer, state_to_frame

    start, states = episode
    renderer = ProductionRenderer(headless=True, width=480, height=480,
                                  map_dims=(env.map_width, env.map_height), terrain=env.map_instance)
    renderer.zoom = .9
    font = pygame.font.Font(None, 20)
    frames = []
    previous = state_to_frame(start)
    for tick in range(0, row["steps"], max(1, stride)):
        state = jax.tree_util.tree_map(lambda x: x[tick], states)
        frame = state_to_frame(state)
        surface = renderer.render_frame(previous, frame, 1.)
        label = f'{row["variant"]} | {row["agent"]} | {row["setting"]}'
        surface.blit(font.render(label, True, (255, 255, 255)), (10, 10))
        surface.blit(font.render(f'Step {tick + 1} | Blue: {np.sum(frame["unit_alive"][:env.num_allies])} '
                                 f'Red: {np.sum(frame["unit_alive"][env.num_allies:])}',
                                 True, (255, 255, 255)), (10, 30))
        frames.append(pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy())
        previous = frame
    # Always include the actual final state and make the result readable.
    state = jax.tree_util.tree_map(lambda x: x[-1], states)
    last = state_to_frame(state)
    surface = renderer.render_frame(last, last, 1.)
    surface.blit(font.render(f'{row["variant"]} | {row["agent"]} | {row["setting"]}',
                             True, (255, 255, 255)), (10, 10))
    surface.blit(font.render(row["outcome"].replace("_", " "), True, (255, 255, 255)), (10, 30))
    final_frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()
    frames.extend([final_frame] * 12)
    imageio.mimsave(path, frames, duration=80, loop=0)
    pygame.quit()


def write_gallery(rows, output):
    cards = []
    for row in rows:
        if "gif" not in row:
            continue
        title = html.escape(f'{row["variant"]} / {row["agent"]} / {row["setting"]} / {row["enemy_mode"]}')
        cards.append(f'<article><h2>{title}</h2><img loading="lazy" src="{html.escape(row["gif"])}">'
                     f'<p>{row["outcome"]} · {row["steps"]} steps · '
                     f'enemy damage {row["enemy_damage"]:.1f} · '
                     f'terrain violations {row["terrain_violations"]}</p></article>')
    content = ('<!doctype html><meta charset="utf-8"><title>Scripted agent evaluation</title>'
               '<style>body{background:#101916;color:#eef5f1;font:16px system-ui;margin:24px}'
               'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}'
               'article{background:#1b2b24;padding:16px;border-radius:12px}img{width:100%}'
               'h2{font-size:17px}a{color:#8ee1dc}</style>'
               '<h1>Scripted beacon and combat agents</h1>'
               '<p>Blue allies · red enemies · cyan beacon · yellow projectiles. '
               'Both agents use the same spawn seed. Fog runs use sightings and memory only. '
               'Stationary enemies still auto-attack. Wins are measured separately from goal progress. '
               'Both armies obey the same ditch boundaries with unit-radius clearance.</p>'
               '<p><a href="results.json">All settings and metrics (JSON)</a></p><main>'
               + ''.join(cards) + '</main>')
    (output / "index.html").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--settings", nargs="+", default=["all"])
    parser.add_argument("--enemy-modes", "--enemy-mode", nargs="+",
                        choices=["all", "static", "guard", "patrol", "aggressive"], default=["static"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("runs/scripted_evaluation"))
    parser.add_argument("--render", choices=["none", "baseline", "all"], default="baseline")
    args = parser.parse_args()
    variants = list(TwoBridgeMap().VARIANTS) if "all" in args.variants else args.variants
    settings = list(SETTINGS) if "all" in args.settings else args.settings
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if any(v not in TwoBridgeMap().VARIANTS for v in variants) or any(s not in SETTINGS for s in settings):
        parser.error("Unknown variant or setting")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ["static", "guard", "patrol", "aggressive"] if "all" in args.enemy_modes else args.enemy_modes
    for setting in settings:
        for variant in variants:
            for mode in modes:
                env = JaxSC2Env(variant, enemy_ai=True, enemy_mode=mode, **SETTINGS[setting])
                env.max_steps = args.max_steps
                rollout = make_rollout(env)
                for seed in args.seeds:
                    results, episodes = evaluate_case(env, seed, rollout)
                    for row, episode in zip(results, episodes):
                        row.update(variant=variant, setting=setting, enemy_mode=mode,
                                   max_steps=args.max_steps, enable_collision=env.enable_collision,
                                   enable_armor=env.enable_armor, enable_fow=env.enable_fow,
                                   latency_delay=env.latency_delay)
                        if args.render == "all" or (args.render == "baseline" and setting == "baseline"):
                            name = f'{variant}_{row["agent"]}_{setting}_{mode}_{seed}.gif'
                            render_gif(env, episode, row, args.output / name)
                            row["gif"] = name
                        rows.append(row)
                        print(f'{variant:12} {setting:12} {mode:10} {row["agent"]:6} '
                              f'{row["outcome"]:16} steps={row["steps"]:3} '
                              f'damage={row["enemy_damage"]:6.1f} progress={row["goal_progress"]}', flush=True)
                    # Persist partial results during long matrix runs as well.
                    (args.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
                    write_gallery(rows, args.output)
    print(f'Wrote {len(rows)} episodes to {args.output / "index.html"}')


if __name__ == "__main__":
    main()
