"""
ESE 650 HW3 Problem 1: Policy Iteration on a 10x10 Grid MDP
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# Grid definition
# ==============================================================================
# 10x10 grid, (x, y) with (0,0) at bottom-left
# x = column (0..9 left to right), y = row (0..9 bottom to top)
#
# Convention: we store things as grid[x, y]
#
# Obstacles: all border cells + interior obstacle at (3,2)
# Start: (1,1) blue
# Goal:  (8,1) green
# Other labeled free cells: (3,6), (4,4), (5,7), (7,5)

GRID_SIZE = 10
GAMMA = 0.9

# Build obstacle map
is_obstacle = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

# All border cells are obstacles
for i in range(GRID_SIZE):
    is_obstacle[i, 0] = True   # bottom row
    is_obstacle[i, 9] = True   # top row
    is_obstacle[0, i] = True   # left column
    is_obstacle[9, i] = True   # right column

# Interior obstacle
is_obstacle[3, 2] = True

# Make sure start and goal are free
START = (1, 1)
GOAL = (8, 1)
is_obstacle[START] = False
is_obstacle[GOAL] = False

# Other labeled free cells (from figure)
for cell in [(3, 6), (4, 4), (5, 7), (7, 5)]:
    is_obstacle[cell] = False

# Total number of states = 100 (10x10)
N_STATES = GRID_SIZE * GRID_SIZE

# Actions: 0=North, 1=South, 2=East, 3=West
ACTIONS = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}
N_ACTIONS = 4

# Action deltas: (dx, dy)
ACTION_DELTAS = {
    0: (0, 1),   # North: y+1
    1: (0, -1),  # South: y-1
    2: (1, 0),   # East:  x+1
    3: (-1, 0),  # West:  x-1
}

# Transition probabilities
# P(intended) = 0.7, P(left of intended) = 0.1, P(right of intended) = 0.1, P(stay) = 0.1
# Left/right relative to the intended direction
def get_left_right(action):
    """Return (left_action, right_action) relative to the intended action."""
    if action == 0:    # North -> left=West, right=East
        return 3, 2
    elif action == 1:  # South -> left=East, right=West
        return 2, 3
    elif action == 2:  # East -> left=North, right=South
        return 0, 1
    else:              # West -> left=South, right=North
        return 1, 0


def state_index(x, y):
    """Convert (x, y) to flat state index."""
    return x * GRID_SIZE + y


def index_to_xy(s):
    """Convert flat state index to (x, y)."""
    return s // GRID_SIZE, s % GRID_SIZE


def attempt_move(x, y, action):
    """Attempt to move from (x,y) in direction action.
    If move goes out of bounds or into obstacle, stay in place."""
    dx, dy = ACTION_DELTAS[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and not is_obstacle[nx, ny]:
        return nx, ny
    else:
        return x, y


def get_reward(x, y):
    """Reward for being in state (x, y).
    Goal: +10, Obstacle: -10, otherwise: -1 per step."""
    if (x, y) == GOAL:
        return 0  # At goal, no cost (can stay)
    if is_obstacle[x, y]:
        return -10
    return -1


# ==============================================================================
# Build transition model
# ==============================================================================
# T[s, a, s'] = probability of going from s to s' under action a
# R[s, a] = expected immediate reward for taking action a in state s

T = np.zeros((N_STATES, N_ACTIONS, N_STATES))
R = np.zeros((N_STATES, N_ACTIONS))

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        s = state_index(x, y)

        # Goal state: stay with 0 cost regardless of action
        if (x, y) == GOAL:
            for a in range(N_ACTIONS):
                T[s, a, s] = 1.0
                R[s, a] = 0.0
            continue

        for a in range(N_ACTIONS):
            left_a, right_a = get_left_right(a)

            # Outcomes: (probability, resulting_action_or_stay)
            outcomes = [
                (0.7, a),         # intended
                (0.1, left_a),    # left of intended
                (0.1, right_a),   # right of intended
                (0.1, None),      # stay in place
            ]

            expected_reward = 0.0
            for prob, act in outcomes:
                if act is None:
                    nx, ny = x, y  # stay
                else:
                    nx, ny = attempt_move(x, y, act)
                ns = state_index(nx, ny)
                T[s, a, ns] += prob
                expected_reward += prob * get_reward(x, y)

            R[s, a] = expected_reward


# ==============================================================================
# Policy Evaluation
# ==============================================================================
def policy_evaluation(policy, tol=1e-10, max_iter=10000):
    """Evaluate a policy by solving V = R_pi + gamma * T_pi * V.
    policy: array of shape (N_STATES,) with action indices.
    Returns V: array of shape (N_STATES,).
    """
    V = np.zeros(N_STATES)
    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            a = policy[s]
            V_new[s] = R[s, a] + GAMMA * np.dot(T[s, a, :], V)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V


def policy_evaluation_linear(policy):
    """Solve V = R_pi + gamma * T_pi * V exactly via linear system.
    (I - gamma * T_pi) V = R_pi
    """
    T_pi = np.zeros((N_STATES, N_STATES))
    R_pi = np.zeros(N_STATES)
    for s in range(N_STATES):
        a = policy[s]
        T_pi[s, :] = T[s, a, :]
        R_pi[s] = R[s, a]
    V = np.linalg.solve(np.eye(N_STATES) - GAMMA * T_pi, R_pi)
    return V


# ==============================================================================
# Policy Improvement
# ==============================================================================
def policy_improvement(V):
    """Given value function V, return greedy policy."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    for a in range(N_ACTIONS):
        Q[:, a] = R[:, a] + GAMMA * T[:, a, :] @ V
    return np.argmax(Q, axis=1)


# ==============================================================================
# Plotting utilities
# ==============================================================================
def value_to_grid(V):
    """Convert flat value array to 10x10 grid for plotting.
    Returns array where grid_val[y, x] = V[state_index(x, y)]
    so that imshow shows it with y increasing upward.
    """
    grid_val = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid_val[y, x] = V[state_index(x, y)]
    return grid_val


def plot_value_heatmap(V, title, filename, policy=None):
    """Plot value function as heatmap. Optionally overlay policy arrows."""
    fig, ax = plt.subplots(figsize=(8, 8))

    grid_val = value_to_grid(V)

    # Mask obstacles for distinct coloring
    masked_val = np.copy(grid_val)

    im = ax.imshow(masked_val, origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='Value J(x)')

    # Mark obstacles
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if is_obstacle[x, y]:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                           facecolor='gray', edgecolor='black', alpha=0.7))

    # Mark start and goal
    ax.plot(START[0], START[1], 's', color='blue', markersize=12, label='Start (1,1)')
    ax.plot(GOAL[0], GOAL[1], 's', color='lime', markersize=12, label='Goal (8,1)')

    # Add value text
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if not is_obstacle[x, y]:
                ax.text(x, y, f'{V[state_index(x, y)]:.1f}',
                        ha='center', va='center', fontsize=6, color='white',
                        fontweight='bold')

    # Overlay policy arrows
    if policy is not None:
        arrow_dx = {0: 0, 1: 0, 2: 0.3, 3: -0.3}
        arrow_dy = {0: 0.3, 1: -0.3, 2: 0, 3: 0}
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if not is_obstacle[x, y] and (x, y) != GOAL:
                    a = policy[state_index(x, y)]
                    ax.arrow(x, y, arrow_dx[a], arrow_dy[a],
                             head_width=0.15, head_length=0.08,
                             fc='red', ec='red', alpha=0.8)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


# ==============================================================================
# (b) Policy Evaluation: initial policy = always go East
# ==============================================================================
print("=" * 60)
print("Part (b): Policy Evaluation with 'always go East' policy")
print("=" * 60)

# Initial policy: always go East (action 2)
policy_east = np.full(N_STATES, 2, dtype=int)

# Evaluate
V_east = policy_evaluation_linear(policy_east)

print(f"Value at start (1,1): {V_east[state_index(1, 1)]:.4f}")
print(f"Value at goal  (8,1): {V_east[state_index(8, 1)]:.4f}")

plot_value_heatmap(V_east,
                   r"Policy Evaluation: $J^\pi(x)$ with $\pi$ = always East",
                   "part_b_value_heatmap.png")


# ==============================================================================
# (c) Policy Iteration: 4 iterations
# ==============================================================================
print("\n" + "=" * 60)
print("Part (c): Policy Iteration (4 iterations)")
print("=" * 60)

policy = np.copy(policy_east)  # start from "always East"

for iteration in range(4):
    # Policy evaluation
    V = policy_evaluation_linear(policy)

    # Plot current policy and value
    plot_value_heatmap(V,
                       f"Policy Iteration {iteration + 1}: Value + Policy",
                       f"part_c_iteration_{iteration + 1}.png",
                       policy=policy)

    print(f"Iteration {iteration + 1}:")
    print(f"  Value at start (1,1): {V[state_index(1, 1)]:.4f}")
    print(f"  Value at goal  (8,1): {V[state_index(8, 1)]:.4f}")

    # Policy improvement
    policy_new = policy_improvement(V)

    # Check convergence
    n_changed = np.sum(policy_new != policy)
    print(f"  Policies changed: {n_changed}")

    policy = policy_new

# Final evaluation after 4 improvements
V_final = policy_evaluation_linear(policy)
plot_value_heatmap(V_final,
                   "Policy Iteration: Final Policy (after 4 iterations)",
                   "part_c_final.png",
                   policy=policy)

print(f"\nFinal policy value at start (1,1): {V_final[state_index(1, 1)]:.4f}")
print(f"Final policy value at goal  (8,1): {V_final[state_index(8, 1)]:.4f}")

print("\nDone! All plots saved to p1/ directory.")
