"""Roll out the trained PPO policy and save PNG frames (offscreen render).
Avoids the dm_control viewer (no GUI required).

Usage:
    py -3.10 render_frames.py
"""
import importlib.util
import os
import pickle
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))


def load_main():
    spec = importlib.util.spec_from_file_location(
        'p', os.path.join(HERE, '18330723_hw4_p2.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load_main()
    env = m.make_env(seed=42)

    actor = m.GaussianActor(m.XDIM, m.UDIM)
    actor.load_state_dict(th.load(os.path.join(HERE, 'actor.pt'), map_location='cpu'))
    actor.eval()

    with open(os.path.join(HERE, 'obs_norm.pkl'), 'rb') as f:
        d = pickle.load(f)
    mean = d['mean']
    std  = np.sqrt(d['var']) + 1e-8

    capture_at = [40, 120, 240, 400, 600, 800]
    frames = {}
    ep_ret = 0.0

    ts = env.reset()
    for t in range(max(capture_at) + 1):
        x_raw  = m.obs2vec(ts.observation)
        x_norm = (x_raw - mean) / std
        with th.no_grad():
            mu, _ = actor(th.from_numpy(x_norm.astype(np.float32)).unsqueeze(0))
        u = np.clip(mu.squeeze(0).numpy(), -1.0, 1.0)
        ts = env.step(u)
        ep_ret += float(ts.reward) if ts.reward is not None else 0.0
        if t in capture_at:
            frames[t] = env.physics.render(height=320, width=480, camera_id=0)
        if ts.last():
            break

    print(f'Captured {len(frames)} frames; episode return so far: {ep_ret:.2f}')

    for t, img in frames.items():
        path = os.path.join(HERE, f'walker_step{t:04d}.png')
        mpimg.imsave(path, img)
        print(f'Saved {path}')

    keys = sorted(frames.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 3))
    for ax, t in zip(axes, keys):
        ax.imshow(frames[t])
        ax.set_title(f't = {t}')
        ax.axis('off')
    plt.tight_layout()
    strip = os.path.join(HERE, 'walker_strip.png')
    plt.savefig(strip, dpi=150, bbox_inches='tight')
    print(f'Saved {strip}')


if __name__ == '__main__':
    main()
