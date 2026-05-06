# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coursework repository for **ESE 650: Learning in Robotics** (UPenn, Spring 2025). Contains homework solutions and lecture materials covering probabilistic state estimation, MDPs, deep generative models, and reinforcement learning.

## Repository Layout

- `hw1/` — Bayesian filtering (histogram filter) and HMMs (forward-backward, Baum-Welch, Viterbi)
- `hw2/` — Kalman filtering: EKF for parameter estimation (`p1/`), quaternion UKF for 3D orientation tracking (`p2/`)
- `hw3/` — `p1/` policy iteration on a 10×10 grid MDP; `p2/` particle-filter SLAM on KITTI; `p3/` NeRF on a synthetic Lego scene
- `hw4/` — `p2/` PPO on the DeepMind Control `walker` task (`p1/` is written-only)
- `lec/` — Lecture notes by chapter (`ch2`–`ch13`), each with slides PDF and optional `.tex` cheatsheet/summary; see `lec/README.md`
- `proj/` — Final project materials
- `hw/` — Older duplicate of hw1/hw2, gitignored

## Running Code

All Python scripts are standalone — no build system, package manager, or virtual environment is configured. Scripts use **relative paths**, so always run them from their own directory:

```bash
cd hw2/p2 && python estimate_rot.py
cd hw3/p2 && python main.py
cd hw3/p3 && python NeRF.py
cd hw4/p2 && py -3.10 18330723_hw4_p2.py    # hw4/p2 needs Python 3.10 (dm_control)
```

### Dependencies by homework

- `hw1`, `hw2`: `numpy`, `scipy`, `matplotlib`, `jupyter`
- `hw3/p2`: adds `tqdm`, `click`
- `hw3/p3`: adds `torch`, `opencv-python` (`cv2`), `imageio`
- `hw4/p2`: adds `torch`, `dm_control` (MuJoCo bindings — needs Python 3.10)

### Data files

Sensor / dataset files committed in-repo: `hw2/p2/imu/`, `hw2/p2/vicon/`, `hw3/p3/data/transforms_colmap.json`. **Not** in repo (gitignored, must be obtained externally before running):

- `hw3/p2/KITTI/` — KITTI LiDAR `.bin` files and poses for SLAM
- `hw3/p3/data/images/` — Lego synthetic NeRF training images

## Key Architecture Details

### Quaternion UKF (`hw2/p2`)

The most complex piece of code. Key files:

- `quaternion.py` — provided `Quaternion` class (Hamilton multiplication, axis-angle, Euler). The `from_rotm` method has a known fragility: it fails silently when `acos` input is out of `[-1,1]` due to floating-point error (see commit `ebf3535`).
- `estimate_rot.py` — full UKF. State is `[quaternion, angular_velocity]` (7D state, 6D covariance in tangent space). Uses gradient-descent quaternion mean (Kraft EK §3.4) and a double-cover fix for axis-angle computation.

Calibration constants are hardcoded: 200-sample stationary window for bias, 1.20× scale on the Wz gyro channel.

### Particle-filter SLAM (`hw3/p2`)

- `slam.py` defines `slam_t` and `map_t`. **Hardcoded KITTI map bounds** (`xmin/xmax = -700/700`, `zmin/zmax = -500/900`) at the top of `map_t.__init__` — re-tune these if running on a different sequence.
- Code by Pratik Chaudhari uses `s` instead of `self` throughout (`def method(s, ...)`). Preserve this convention when editing.
- Driver is `main.py`; helpers in `load_data.py` and `utils.py`.

### NeRF (`hw3/p3`)

- `NeRF.py` is the entire pipeline (model, ray sampling, training, novel view rendering). Reads `data/transforms_colmap.json` and `data/images/`. Resize images from 800×800 to ~200×200 for tractable training (and adjust focal length accordingly — noted in `load_colmap_data` docstring).

### PPO walker (`hw4/p2`)

- `18330723_hw4_p2.py` is the trainer; `walker.py` contains the provided skeleton (note `s`/`self` convention here too); `view.py` replays a saved checkpoint via `dm_control` viewer; `render_frames.py` renders strip images.
- Implementation relies on the standard PPO stabilizers: orthogonal init, running obs normalization (mirror-symmetrized), GAE-λ, mini-batch SGD, LR annealing, entropy bonus, value clipping, gradient clipping, and a left/right mirror-symmetry regularizer on both π and V.

## File / Submission Conventions

- Submission files follow `18330723_hw{N}_p{X}{part}.py` (student ID prefix). Don't rename these — the autograder relies on the pattern.
- Reports live alongside problem code as `*_report.pdf` (or under `report/` for hw2/hw3).

## LaTeX

Lecture notes and reports use LaTeX. Build artifacts (`*.aux`, `*.synctex.gz`, `lec/**/out/`, `*.log`) are gitignored. The compiled `*.pdf` is generally committed.
