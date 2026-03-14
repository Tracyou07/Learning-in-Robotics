import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# Problem 1(c): Plot EKF estimate of a vs true value
# ═══════════════════════════════════════════════════════

# Load EKF results from p1b
res    = np.load('p1_ekf.npy', allow_pickle=True).item()
mu_a   = res['mu_a']
sig_a  = res['sig_a']
T      = res['T']
a_true = res['a_true']

ks = np.arange(1, T + 1)

fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(a_true, color='k', lw=2, ls='--', label='True $a = -1$')
ax.plot(ks, mu_a, color='tab:blue', lw=1.5, label=r'$\mu_k$ (EKF mean)')
ax.fill_between(ks,
                mu_a - sig_a,
                mu_a + sig_a,
                alpha=0.3, color='tab:blue', label=r'$\mu_k \pm \sigma_k$')

ax.set_xlabel('$k$')
ax.set_ylabel('Estimate of $a$')
ax.set_title('EKF Estimation of Unknown Parameter $a$')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('p1c_ekf.png', dpi=150)
plt.show()
