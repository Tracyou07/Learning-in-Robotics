import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# Problem 1(a): Simulate the dynamical system
# ═══════════════════════════════════════════════════════
# System:
#   x_{k+1} = a * x_k + eps_k,   eps_k ~ N(0, 1)
#   y_k     = sqrt(x_k^2 + 1) + nu_k,  nu_k ~ N(0, 1/2)
# True parameter: a = -1
# Initial state:  x_0 ~ N(mean=1, variance=2)
# ═══════════════════════════════════════════════════════

np.random.seed(0)

a_true = -1
T = 100

# x_0 ~ N(1, 2): mean=1, variance=2, std=sqrt(2)
x0 = np.random.normal(loc=1, scale=np.sqrt(2))

x = np.zeros(T + 1)
y = np.zeros(T)
x[0] = x0

for k in range(T):
    eps_k = np.random.normal(0, 1)             # eps_k ~ N(0, 1)
    nu_k  = np.random.normal(0, np.sqrt(0.5))  # nu_k  ~ N(0, 1/2)
    y[k]     = np.sqrt(x[k]**2 + 1) + nu_k
    x[k + 1] = a_true * x[k] + eps_k

# Save for use in p1b and p1c
np.save('p1_data.npy', {'x': x, 'y': y, 'T': T, 'a_true': a_true})
print(f"x_0  = {x0:.4f}  (sampled from N(1, 2))")
print(f"T    = {T} observations")
print(f"Data saved to p1_data.npy")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

axes[0].plot(range(T + 1), x, lw=1.5, label=r'$x_k$ (true state)')
axes[0].axhline(0, color='k', lw=0.5, ls='--')
axes[0].set_ylabel(r'$x_k$')
axes[0].set_title('True State Trajectory')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(range(T), y, 'o', markersize=3, color='tab:orange',
             label=r'$y_k$ (observations)')
axes[1].set_xlabel('k')
axes[1].set_ylabel(r'$y_k$')
axes[1].set_title(r'Noisy Observations  $y_k = \sqrt{x_k^2+1} + \nu_k$')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig('p1a_simulation.png', dpi=150)
plt.show()
