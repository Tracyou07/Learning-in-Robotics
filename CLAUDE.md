# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a coursework repository for **ESE 650: Learning in Robotics** (UPenn, Spring 2025). It contains homework solutions and lecture materials covering probabilistic state estimation for robotic systems.

## Repository Layout

- `hw1/` — Bayesian filtering (histogram filter) and Hidden Markov Models (forward-backward, Baum-Welch, Viterbi)
- `hw2/` — Kalman filtering: EKF for parameter estimation (`p1/`), quaternion UKF for 3D orientation tracking (`p2/`)
- `lec/` — Lecture notes and chapter materials (PDFs, LaTeX)
- `proj/` — Final project materials
- `hw/` — Older duplicate of hw1/hw2, excluded via `.gitignore`

## Running Code

All Python scripts are standalone. No build system, package manager, or virtual environment is configured. Run scripts directly:

```bash
python hw2/p2/estimate_rot.py
python hw1/p1/example_test.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `jupyter` (install via pip).

Sensor data files (`.npy`) are committed in `hw2/p2/imu/` and `hw2/p2/vicon/`. Scripts use relative paths, so run them from their containing directory (e.g., `cd hw2/p2 && python estimate_rot.py`).

## Key Architecture Details

### Quaternion UKF (hw2/p2)

The most complex piece of code. Key files:

- `quaternion.py` — Provided `Quaternion` class with Hamilton multiplication, axis-angle conversion, Euler angle extraction. The `from_rotm` method has a known fragility: it fails silently when `acos` input is out of `[-1,1]` due to floating-point error (see commit `ebf3535`).
- `estimate_rot.py` — Full UKF implementation. State is `[quaternion, angular_velocity]` (7D state, 6D covariance in tangent space). Uses gradient-descent quaternion mean (Kraft EK Sec 3.4) and a double-cover fix for axis-angle computation.

Calibration constants are hardcoded: 200-sample stationary window for bias, 1.20× scale on Wz gyro channel.

### File Naming Convention

Homework submission files follow the pattern `18330723_hw{N}_p{X}{part}.py` (student ID prefix).

## LaTeX

Lecture notes in `lec/` use LaTeX with build artifacts (`.aux`, `.synctex.gz`, `out/`) excluded via `.gitignore`.
