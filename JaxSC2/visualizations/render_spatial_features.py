#!/usr/bin/env python3

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import jax, jax.numpy as jnp

from JaxSC2.env.env import JaxSC2Env


def _normalize(data):
    if data.size == 0 or np.all(np.isnan(data)):
        return np.zeros_like(data)
    mn, mx = float(np.min(data)), float(np.max(data))
    return np.clip((data - mn) / (mx - mn + 1e-9), 0.0, 1.0)


def _build_frame(screen, minimap):
    H = W = 64

    screen_labels = [
        "terrain",    # gray
        "visibility", # gray
        "p_rel_bg",   # gray
        "p_rel_self", # gray
        "p_rel_ally", # gray
        "p_rel_neut", # gray
        "p_rel_enemy","# magma
        "ally_pos",   # magma
        "enemy_pos",  # magma
        "beacon_pos", # magma
        "ally_hp",    # magma
        "enemy_hp",   # magma
    ]

    minimap_labels = [
        "terrain",       # gray
        "visibility",    # gray
        "p_rel_self_0",  # gray
        "p_rel_enemy_0", # magma
        "ally_pos",      # magma
        "enemy_pos",     # magma
        "camera",        # gray
    ]

    all_labels = screen_labels + minimap_labels  # 19 total
    n_channels = len(all_labels)

    is_density = lambda name: 'pos' in name or 'hp' in name
    all_cmaps = ['magma' if is_density(l) else 'gray' for l in all_labels]

    n_cols = 6
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows + 2, n_cols + 1, figsize=(n_cols * 40, (n_rows + 2) * 35), squeeze=False)

    for r in range(n_rows + 2):
        for c in range(n_cols + 1):
            axes[r][c].axis('off')

    for idx in range(n_channels):
        label = all_labels[idx]
        cmap  = all_cmaps[idx]

        if idx < len(screen_labels):
            data = np.array(screen[idx]) if screen is not None else None
        elif idx < len(all_labels):
            rel = idx - len(screen_labels)
            data = np.array(minimap[rel]) if minimap is not None else None
        else:
            data = None

        r = idx // n_cols + 2
        c = idx % n_cols

        ax = axes[r][1 + c]
        ax.axis('off')

        if data is not None and np.any(data):
            im = ax.imshow(_normalize(data), cmap=cmap, vmin=0, vmax=1)
        else:
            im = ax.imshow(np.zeros((H, W)), cmap='gray', vmin=0, vmax=1)
        ax.set_title(label, fontsize=7)

    plt.tight_layout(pad=0.1, h_pad=3.0)
    fig.canvas.draw()
    arr = np.array(fig.canvas.renderer.buffer_rgba())
    rgb_arr = np.stack([arr[..., 0], arr[..., 1], arr[..., 2]], axis=-1)
    plt.close(fig)
    return rgb_arr


def _random_action(num_allies, num_enemies, key):
    probs = jax.random.uniform(key)
    who_mask = jnp.ones(num_allies, dtype=jnp.bool_) if probs > 0.3 else jnp.zeros(
        num_allies, dtype=jnp.bool_)
    verb = int(jax.random.uniform(key) * 3.0) % 3
    direction = int(jax.random.uniform(key) * 8.0) % 8
    target_idx = int(jax.random.randint(key, (), 0, num_enemies)) if num_enemies > 0 else 0
    target = jnp.full(num_allies, target_idx, dtype=jnp.int32)
    return type('PerUnitAction', (), {
        'who_mask': who_mask,
        'verb': verb,
        'direction': direction,
        'target': target
    })()


def main():
    parser = argparse.ArgumentParser(description="Visualize spatial feature channels as GIF.")
    parser.add_argument('--variant', default='V2_Combat')
    parser.add_argument('--save', dest='gif_path', default='spatial_features_rollout.gif')
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    env = JaxSC2Env(variant_name=args.variant, use_spatial_obs=True, resolution=64)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    frames = []
    for step in range(args.max_steps):
        feats = env.build_spatial_features(state)
        screen = np.array(feats['screen']) if 'screen' in feats else None
        minimap = np.array(feats['minimap']) if 'minimap' in feats else None

        frame = _build_frame(screen, minimap)
        frames.append(frame)

        rng, step_rng = jax.random.split(rng)
        action = _random_action(env.num_allies, env.num_enemies, step_rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)

        if done:
            break

    gif_path = args.gif_path
    os.makedirs(os.path.dirname(gif_path) or '.', exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Saved {gif_path} ({len(frames)} frames)")


if __name__ == '__main__':
    main()
