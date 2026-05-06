"""
ESE 650 HW4 Problem 2: PPO for the DM Control Walker.

Implements PPO (clip variant) with the standard tricks needed to make a
24-D / 6-D bipedal walker actually walk:
    * Hidden-256 ReLU MLPs with orthogonal init
    * Running observation normalization (with mirror-symmetrized stats)
    * GAE-lambda advantage
    * Mini-batch SGD (256 / 10 epochs) per rollout
    * Linear LR annealing
    * Entropy bonus
    * Value-function clipping
    * Gradient clipping
    * Mirror-symmetry regularizer (left <-> right) for both pi and V
    * Best-checkpoint saving

Run:
    cd hw4/p2
    py -3.10 18330723_hw4_p2.py

Inspect a checkpoint:
    py -3.10 view.py
"""

import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Constants and helpers
# ==============================================================================
TASK = 'walk'                # 'walk' or 'stand'
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, 'runs', f'seed{SEED}')

XDIM = 14 + 1 + 9            # = 24
UDIM = 6

# Mirror permutations for the dm_control planar walker (left <-> right).
# State layout: orientations[0:14] + height[14] + velocity[15:24].
#   orientations: torso[0:2], R_thigh[2:4], R_leg[4:6], R_foot[6:8],
#                 L_thigh[8:10], L_leg[10:12], L_foot[12:14]
#   velocity:     root[15:18], R_hip/knee/ankle[18:21], L_hip/knee/ankle[21:24]
# Action: [R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle]
STATE_PERM = th.tensor(
    [0, 1, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7,
     14,
     15, 16, 17, 21, 22, 23, 18, 19, 20],
    dtype=th.long,
)
ACTION_PERM = th.tensor([3, 4, 5, 0, 1, 2], dtype=th.long)


def mirror_state(s):
    return s.index_select(-1, STATE_PERM)


def mirror_action(a):
    return a.index_select(-1, ACTION_PERM)


def obs2vec(obs):
    return np.array(
        obs['orientations'].tolist() + [obs['height']] + obs['velocity'].tolist(),
        dtype=np.float32,
    )


def make_env(seed=SEED):
    """Lazy import keeps this module importable without dm_control."""
    from dm_control import suite
    rng = np.random.RandomState(seed)
    return suite.load('walker', TASK, task_kwargs={'random': rng})


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


# ==============================================================================
# Networks
# ==============================================================================
class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy with state-independent log_std (clamped)."""
    def __init__(self, xdim, udim, hdim=256):
        super().__init__()
        self.fc1 = _ortho(nn.Linear(xdim, hdim), gain=np.sqrt(2))
        self.fc2 = _ortho(nn.Linear(hdim, hdim), gain=np.sqrt(2))
        self.mu  = _ortho(nn.Linear(hdim, udim), gain=0.01)
        self.log_std = nn.Parameter(-0.5 * th.ones(udim))

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.mu(h)
        std = th.exp(self.log_std.clamp(min=-1.0)).expand_as(mu)
        return mu, std

    def dist(self, x):
        mu, std = self.forward(x)
        return th.distributions.Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, xdim, hdim=256):
        super().__init__()
        self.fc1 = _ortho(nn.Linear(xdim, hdim), gain=np.sqrt(2))
        self.fc2 = _ortho(nn.Linear(hdim, hdim), gain=np.sqrt(2))
        self.v   = _ortho(nn.Linear(hdim, 1),    gain=1.0)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.v(h).squeeze(-1)


# ==============================================================================
# Running observation normalizer
# ==============================================================================
class ObsNormalizer:
    """Welford running mean/var; symmetrized so that
    mirror(normalize(x)) == normalize(mirror(x))."""
    def __init__(self, xdim):
        self.mean  = np.zeros(xdim, dtype=np.float64)
        self.var   = np.ones(xdim,  dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        bm, bv, bc = batch.mean(0), batch.var(0), batch.shape[0]
        delta = bm - self.mean
        total = self.count + bc
        self.mean = self.mean + delta * bc / total
        self.var = (self.var * self.count + bv * bc
                    + delta ** 2 * self.count * bc / total) / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def symmetrize(self, perm):
        p = perm.cpu().numpy()
        self.mean = 0.5 * (self.mean + self.mean[p])
        self.var  = 0.5 * (self.var  + self.var[p])


# ==============================================================================
# Rollout + GAE
# ==============================================================================
def collect_batch(env, actor, critic, obs_norm,
                  steps_per_batch=8000, gamma=0.99, lam=0.95, T=1000):
    states_norm, actions, log_probs = [], [], []
    advantages, returns_, old_values = [], [], []
    ep_returns, ep_lens = [], []

    steps = 0
    while steps < steps_per_batch:
        ep_states_raw, ep_actions, ep_logps, ep_rewards = [], [], [], []
        ts = env.reset()
        x_raw = obs2vec(ts.observation)

        for _ in range(T):
            x_norm = obs_norm.normalize(x_raw)
            x_th = th.from_numpy(x_norm).float().unsqueeze(0)
            with th.no_grad():
                d = actor.dist(x_th)
                u = d.sample()
                lp = d.log_prob(u).sum(-1)

            ep_states_raw.append(x_raw)
            ep_actions.append(u.squeeze(0).numpy())
            ep_logps.append(lp.item())

            r = env.step(np.clip(u.squeeze(0).numpy(), -1.0, 1.0))
            ep_rewards.append(r.reward if r.reward is not None else 0.0)
            x_raw = obs2vec(r.observation)
            if r.last():
                break

        ep_len = len(ep_rewards)
        steps += ep_len
        ep_returns.append(float(sum(ep_rewards)))
        ep_lens.append(ep_len)

        s_raw = np.array(ep_states_raw, dtype=np.float32)
        obs_norm.update(s_raw)
        s_normed = obs_norm.normalize(s_raw).astype(np.float32)

        with th.no_grad():
            vals = critic(th.from_numpy(s_normed)).numpy()
            if r.last():
                last_val = 0.0
            else:
                last_val = critic(
                    th.from_numpy(obs_norm.normalize(x_raw).astype(np.float32))
                    .unsqueeze(0)).item()

        ep_adv = np.zeros(ep_len, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(ep_len)):
            next_val = vals[t + 1] if t + 1 < ep_len else last_val
            delta = ep_rewards[t] + gamma * next_val - vals[t]
            gae = delta + gamma * lam * gae
            ep_adv[t] = gae
        ep_ret = ep_adv + vals

        states_norm.append(s_normed)
        actions.extend(ep_actions)
        log_probs.extend(ep_logps)
        advantages.append(ep_adv)
        returns_.append(ep_ret)
        old_values.append(vals)

    states_norm = np.concatenate(states_norm, 0)
    actions     = np.array(actions, dtype=np.float32)
    log_probs   = np.array(log_probs, dtype=np.float32)
    adv_all     = np.concatenate(advantages, 0)
    ret_all     = np.concatenate(returns_, 0)
    val_all     = np.concatenate(old_values, 0)
    adv_all     = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

    batch = dict(
        states=th.from_numpy(states_norm),
        actions=th.from_numpy(actions),
        log_probs=th.from_numpy(log_probs),
        advantages=th.from_numpy(adv_all),
        returns=th.from_numpy(ret_all),
        old_values=th.from_numpy(val_all),
    )
    return batch, ep_returns, ep_lens


# ==============================================================================
# PPO trainer
# ==============================================================================
def train(
    total_timesteps=3_000_000,
    steps_per_batch=8000,
    gamma=0.99, lam=0.95, clip_ratio=0.2,
    lr=3e-4, epochs=10, mini_batch_size=256,
    target_kl=0.02, max_grad_norm=0.5,
    ent_coef=5e-3, vf_clip=0.2,
    sym_coef_pi=0.5, sym_coef_vf=0.1,
    seed=SEED, run_dir=None,
):
    run_dir = run_dir or RUN_DIR
    os.makedirs(run_dir, exist_ok=True)
    th.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(seed)
    actor = GaussianActor(XDIM, UDIM)
    critic = Critic(XDIM)
    pi_opt = th.optim.Adam(actor.parameters(),  lr=lr, eps=1e-5)
    vf_opt = th.optim.Adam(critic.parameters(), lr=lr, eps=1e-5)
    obs_norm = ObsNormalizer(XDIM)

    timesteps_so_far = 0
    all_rets, all_ts = [], []
    best_avg = -float('inf')
    t0 = time.time()
    iter_i = 0

    while timesteps_so_far < total_timesteps:
        # Linear LR anneal
        frac = max(0.0, 1.0 - timesteps_so_far / total_timesteps)
        cur_lr = lr * frac
        for pg in pi_opt.param_groups: pg['lr'] = cur_lr
        for pg in vf_opt.param_groups: pg['lr'] = cur_lr

        batch, ep_rets, ep_lens = collect_batch(
            env, actor, critic, obs_norm, steps_per_batch, gamma, lam)
        obs_norm.symmetrize(STATE_PERM)
        bsz = batch['states'].shape[0]
        timesteps_so_far += bsz
        for r in ep_rets:
            all_rets.append(r); all_ts.append(timesteps_so_far)

        avg_ret = float(np.mean(ep_rets))
        avg_len = float(np.mean(ep_lens))
        elapsed = time.time() - t0
        iter_i += 1
        print(f'iter {iter_i:3d} | step {timesteps_so_far:>7d} | eps {len(ep_rets):3d} '
              f'| len {avg_len:5.0f} | ret {avg_ret:7.1f} | lr {cur_lr:.1e} | t {elapsed:.0f}s',
              flush=True)

        if avg_ret > best_avg:
            best_avg = avg_ret
            th.save(actor.state_dict(),  os.path.join(run_dir, 'actor_best.pt'))
            th.save(critic.state_dict(), os.path.join(run_dir, 'critic_best.pt'))
            with open(os.path.join(run_dir, 'obs_norm.pkl'), 'wb') as f:
                pickle.dump({'mean': obs_norm.mean, 'var': obs_norm.var,
                             'count': obs_norm.count}, f)
            print(f'   -> new best ({best_avg:.1f}) saved', flush=True)

        s_th, a_th = batch['states'], batch['actions']
        old_lp_th  = batch['log_probs']
        adv_th     = batch['advantages']
        ret_th     = batch['returns']
        oldv_th    = batch['old_values']
        N = s_th.shape[0]

        for ep in range(epochs):
            idx = np.random.permutation(N)
            for st in range(0, N, mini_batch_size):
                mb = idx[st:st + mini_batch_size]
                ms, ma   = s_th[mb], a_th[mb]
                molp     = old_lp_th[mb]
                madv     = adv_th[mb]
                mret     = ret_th[mb]
                moldv    = oldv_th[mb]

                # ----- Actor update -----
                d = actor.dist(ms)
                new_lp = d.log_prob(ma).sum(-1)
                ent    = d.entropy().sum(-1).mean()

                ratio  = th.exp(new_lp - molp)
                clip_a = th.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                pi_clip = -th.min(ratio * madv, clip_a * madv).mean()

                # Mirror symmetry: pi(s) = mirror(pi(mirror(s)))
                mu_curr, _ = actor(ms)
                mu_mirr, _ = actor(mirror_state(ms))
                sym_pi = ((mu_curr - mirror_action(mu_mirr)) ** 2).mean()

                pi_loss = pi_clip - ent_coef * ent + sym_coef_pi * sym_pi

                pi_opt.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                pi_opt.step()

                # ----- Critic update (with value clipping) -----
                vp = critic(ms)
                vc = moldv + th.clamp(vp - moldv, -vf_clip, vf_clip)
                v_loss = 0.5 * th.max((vp - mret) ** 2,
                                       (vc - mret) ** 2).mean()
                v_mirr = critic(mirror_state(ms))
                sym_v  = ((vp - v_mirr) ** 2).mean()
                vf_loss = v_loss + sym_coef_vf * sym_v

                vf_opt.zero_grad()
                vf_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                vf_opt.step()

            # KL early stop
            with th.no_grad():
                d = actor.dist(s_th)
                approx_kl = (old_lp_th - d.log_prob(a_th).sum(-1)).mean().item()
            if approx_kl > 1.5 * target_kl:
                break

    # Final save
    th.save(actor.state_dict(),  os.path.join(run_dir, 'actor.pt'))
    th.save(critic.state_dict(), os.path.join(run_dir, 'critic.pt'))
    with open(os.path.join(run_dir, 'obs_norm_final.pkl'), 'wb') as f:
        pickle.dump({'mean': obs_norm.mean, 'var': obs_norm.var,
                     'count': obs_norm.count}, f)
    with open(os.path.join(run_dir, 'logs.json'), 'w') as f:
        json.dump([{'timestep': int(t), 'return': float(r)}
                   for t, r in zip(all_ts, all_rets)], f)

    # Symlink-like copy of best to top-level for view.py + report
    import shutil
    shutil.copy(os.path.join(run_dir, 'actor_best.pt'),
                os.path.join(HERE, 'actor.pt'))
    shutil.copy(os.path.join(run_dir, 'critic_best.pt'),
                os.path.join(HERE, 'critic.pt'))
    shutil.copy(os.path.join(run_dir, 'obs_norm.pkl'),
                os.path.join(HERE, 'obs_norm.pkl'))

    # Logs as numpy for easy plotting
    np.savez(os.path.join(HERE, 'ppo_log.npz'),
             timesteps=np.array(all_ts), returns=np.array(all_rets))

    # Plot
    rets = np.array(all_rets)
    ts   = np.array(all_ts)
    win = max(20, len(rets) // 50)
    if len(rets) >= win:
        ma = np.convolve(rets, np.ones(win) / win, mode='valid')
        ma_ts = ts[win - 1:]
    else:
        ma, ma_ts = rets, ts
    plt.figure(figsize=(8, 5))
    plt.plot(ts, rets, alpha=0.3, color='#1f77b4', label='per-episode')
    plt.plot(ma_ts, ma, lw=2, color='#d62728', label=f'{win}-ep moving avg')
    plt.xlabel('Environment timesteps'); plt.ylabel('Episode return')
    plt.title(f'PPO on Walker-{TASK}'); plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'ppo_returns.png'), dpi=150)
    print(f'\nBest avg-batch return during training: {best_avg:.1f}')
    print(f'Saved {os.path.join(HERE, "ppo_returns.png")}')


# ==============================================================================
# Evaluation / viewer
# ==============================================================================
def evaluate(actor_path=None, norm_path=None):
    """Launch dm_control viewer with the trained policy (deterministic mean
    actions). Requires dm_control + obs_norm.pkl beside the checkpoint."""
    from dm_control import viewer
    actor_path = actor_path or os.path.join(HERE, 'actor.pt')
    norm_path  = norm_path  or os.path.join(HERE, 'obs_norm.pkl')

    actor = GaussianActor(XDIM, UDIM)
    actor.load_state_dict(th.load(actor_path, map_location='cpu'))
    actor.eval()

    with open(norm_path, 'rb') as f:
        d = pickle.load(f)
    mean = d['mean']
    std  = np.sqrt(d['var']) + 1e-8

    env = make_env(SEED)

    def policy(timestep):
        x = obs2vec(timestep.observation)
        x = (x - mean) / std
        with th.no_grad():
            mu, _ = actor(th.from_numpy(x.astype(np.float32)).unsqueeze(0))
        return np.clip(mu.squeeze(0).numpy(), -1.0, 1.0)

    viewer.launch(env, policy=policy)


if __name__ == '__main__':
    train()
