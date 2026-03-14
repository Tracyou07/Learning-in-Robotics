import numpy as np

# ═══════════════════════════════════════════════════════
# Problem 1(b): EKF to estimate unknown parameter a
# ═══════════════════════════════════════════════════════
# Augmented state: z_k = [x_k, a]^T  ∈ R²
#
# Process model:  f(z) = [a*x, a]^T
#   Jacobian F = [[a, x],    evaluated at current estimate
#                 [0, 1 ]]
#
# Observation model: h(z) = sqrt(x^2 + 1)
#   Jacobian H = [x / sqrt(x^2+1),  0]   (shape 1×2)
#
# Process noise:      Q = diag(1, 0)  — only x has noise
# Measurement noise:  R = 0.5
# ═══════════════════════════════════════════════════════

# Load simulation data from p1a
data  = np.load('p1_data.npy', allow_pickle=True).item()
y     = data['y']
T     = data['T']
a_true = data['a_true']

# ── Noise matrices ───────────────────────────
Q = np.array([[1.0, 0.0],
              [0.0, 0.0]])   # process noise covariance (2×2)
R = 0.5                      # measurement noise variance

# ── Initialization ───────────────────────────
# x_0 ~ N(1, 2)  →  mu_x = 1,  var_x = 2
# a unknown       →  mu_a = -1.0, var_a = 10 (vague prior with negative sign)
#
# NOTE: the observation h(x) = sqrt(x^2+1) is an even function of x,
# so it cannot distinguish +x from -x. This creates two equally valid modes:
#   - a ≈ +1 (x stays positive)  — reached if initialized with mu_a > 0
#   - a ≈ -1 (x alternates sign) — reached if initialized with mu_a < 0
# The EKF can only track one mode, so initialization determines which
# solution is found. We use mu_a = -1 as a prior that a is negative.
mu    = np.array([1.0, -1.0])
Sigma = np.array([[2.0, 0.0],
                  [0.0, 10.0]])

I2 = np.eye(2)

mu_a_hist  = np.zeros(T)    # E[a | y_1,...,y_k]
sig_a_hist = np.zeros(T)    # std(a | y_1,...,y_k)

# ── EKF loop ─────────────────────────────────
for k in range(T):

    x_est, a_est = mu

    # ── Prediction ───────────────────────────
    mu_pred = np.array([a_est * x_est,   # f: x part
                        a_est         ]) # f: a part (constant)

    F = np.array([[a_est, x_est],  # Jacobian (1×2)
                  [0.0,   1.0  ]])

    Sigma_pred = F @ Sigma @ F.T + Q

    # ── Update ───────────────────────────────
    x_p   = mu_pred[0]
    y_hat = np.sqrt(x_p**2 + 1)                          # predicted observation
    H     = np.array([[x_p / np.sqrt(x_p**2 + 1), 0.0]]) # Jacobian (1×2)

    S = (H @ Sigma_pred @ H.T).item() + R  # innovation variance (scalar)
    K = (Sigma_pred @ H.T) / S             # Kalman gain (2×1)

    mu    = mu_pred + K.flatten() * (y[k] - y_hat)
    Sigma = (I2 - K @ H) @ Sigma_pred

    # Store a-component
    mu_a_hist[k]  = mu[1]
    sig_a_hist[k] = np.sqrt(Sigma[1, 1])

print(f"True a            = {a_true}")
print(f"Final estimate    = {mu_a_hist[-1]:.4f} ± {sig_a_hist[-1]:.4f}")

# Save results for p1c
np.save('p1_ekf.npy', {'mu_a': mu_a_hist, 'sig_a': sig_a_hist,
                        'T': T, 'a_true': a_true})
print("EKF results saved to p1_ekf.npy")
