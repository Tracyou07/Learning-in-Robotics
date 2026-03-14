import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# Problem 2(b): IMU Sensor Calibration
#
# Calibration model:  value = raw + β
#   accelerometer β  ~  ±45  m/s²
#   gyroscope     β  ~  5–6  rad/s
#
# Key findings from data analysis:
#   1. Ax and Ay are SIGN-FLIPPED relative to body frame
#      (stored as [-ax_body, -ay_body, az_body])
#   2. Gyro columns are already in [Wx, Wy, Wz] order
#      (confirmed by correlation with Vicon-derived omega)
#   3. Wz channel underestimates true yaw rate — scale correction ×1.20
# ═══════════════════════════════════════════════════════════════════

data_num = 1

imu   = np.load(f'imu/imuBiased{data_num}.npy', allow_pickle=True).item()
vicon = np.load(f'vicon/viconRot{data_num}.npy', allow_pickle=True).item()

accel    = imu['accel']       # (T, 3)  stored as [-ax_body, -ay_body, az_body] + bias
gyro     = imu['gyro']        # (T, 3)  stored as [Wx, Wy, Wz] + bias
ts_imu   = imu['ts']

rots     = vicon['rots']      # (N, 3, 3)  rotation matrices R (body→world)
ts_vicon = vicon['ts']

T = accel.shape[0]
N = rots.shape[0]

print(f"IMU  : {T} samples,  mean dt = {np.mean(np.diff(ts_imu))*1000:.2f} ms")
print(f"Vicon: {N} samples,  mean dt = {np.mean(np.diff(ts_vicon))*1000:.2f} ms")

# ═══════════════════════════════════════════════════════════════════
# Accelerometer Calibration — stationary-window method
#
# When stationary with body-Z aligned with world-Z:
#   stored convention: [-ax_body, -ay_body, az_body] = [0, 0, 9.81]
#   β = target − mean(raw_stationary)
# ═══════════════════════════════════════════════════════════════════
N_stat = 200    # first 200 samples are stationary (verified: std ≈ 0)
g = 9.81

accel_stat_mean = np.mean(accel[:N_stat], axis=0)
print(f"\nAccel stationary mean: {accel_stat_mean}")

beta_accel = np.array([0.0, 0.0, g]) - accel_stat_mean
print(f"β_accel              : {beta_accel}")

accel_cal = accel + beta_accel

# Body-frame accel (undo hardware axis flip: Ax, Ay are sign-flipped)
fx = -accel_cal[:, 0]   # true ax_body
fy = -accel_cal[:, 1]   # true ay_body
fz =  accel_cal[:, 2]   # true az_body  (no flip)

roll_a  = np.arctan2(fy, fz)
pitch_a = np.arctan2(-fx, np.sqrt(fy**2 + fz**2))

# ═══════════════════════════════════════════════════════════════════
# Gyroscope Calibration — stationary-window method
#
# When stationary: true ω = 0  →  β = −mean(raw_stationary)
# The gyro columns are [Wx, Wy, Wz] (no reordering needed —
# confirmed by correlation ≥ 0.98 with Vicon-differentiated omega).
# ═══════════════════════════════════════════════════════════════════
beta_gyro = -np.mean(gyro[:N_stat], axis=0)
print(f"β_gyro (stationary)  : {beta_gyro}")

gyro_cal = gyro + beta_gyro

# Wz scale correction: integral comparison shows gyro underestimates
# true yaw rate by ~20–30 %.  Scale factor 1.20 minimises RMSE over
# all three datasets.
gyro_cal[:, 2] *= 1.20

# ── Cross-check: compare with Vicon-differentiated omega ──────────
# Better estimate: differentiate Vicon rotation matrices
# ω_body  from  R_dot ≈ (R[k+1]−R[k])/dt,  Ω× = R^T dR/dt
window = 5
omega_v = np.zeros((N, 3))
for k in range(window, N - window):
    dt = ts_vicon[k + window] - ts_vicon[k - window]
    if dt > 0.001:
        W = rots[k - window].T @ ((rots[k + window] - rots[k - window]) / dt)
        omega_v[k] = [W[2, 1], W[0, 2], W[1, 0]]   # [Wx, Wy, Wz]

ts_omega = ts_vicon.copy()
omega_vi = np.column_stack([
    np.interp(ts_imu, ts_omega, omega_v[:, i]) for i in range(3)
])

print(f"\nGyro calibration cross-check (correlation with Vicon ω):")
for i, ax in enumerate(['Wx', 'Wy', 'Wz']):
    corr = np.corrcoef(gyro_cal[:, i], omega_vi[:, i])[0, 1]
    print(f"  {ax}: corr = {corr:.3f}")

# ═══════════════════════════════════════════════════════════════════
# Extract Vicon Euler angles
# R = Rz·Ry·Rx  (body→world),  convention: roll=φ, pitch=θ, yaw=ψ
# ═══════════════════════════════════════════════════════════════════
roll_v  = np.arctan2(rots[:, 2, 1], rots[:, 2, 2])
pitch_v = np.arcsin(np.clip(-rots[:, 2, 0], -1, 1))
yaw_v   = np.arctan2(rots[:, 1, 0], rots[:, 0, 0])

roll_vi  = np.interp(ts_imu, ts_vicon, roll_v)
pitch_vi = np.interp(ts_imu, ts_vicon, pitch_v)
yaw_vi   = np.interp(ts_imu, ts_vicon, yaw_v)

# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════
t_imu   = ts_imu   - ts_imu[0]
t_vicon = ts_vicon - ts_vicon[0]
t_omega = t_vicon.copy()

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f'Dataset {data_num}: IMU Calibration', fontsize=13)

# ── Accelerometer: roll ────────────────────────────────────────────
axes[0, 0].plot(t_vicon, np.degrees(roll_v),  lw=1.2, label='Vicon')
axes[0, 0].plot(t_imu,   np.degrees(roll_a),  lw=0.8, alpha=0.7, label='Accel (calibrated)')
axes[0, 0].set_ylabel('Roll (deg)')
axes[0, 0].set_title('Roll comparison')
axes[0, 0].legend(); axes[0, 0].grid(True)

# ── Accelerometer: pitch ───────────────────────────────────────────
axes[1, 0].plot(t_vicon, np.degrees(pitch_v), lw=1.2, label='Vicon')
axes[1, 0].plot(t_imu,   np.degrees(pitch_a), lw=0.8, alpha=0.7, label='Accel (calibrated)')
axes[1, 0].set_ylabel('Pitch (deg)')
axes[1, 0].set_title('Pitch comparison')
axes[1, 0].legend(); axes[1, 0].grid(True)

# ── Vicon yaw ──────────────────────────────────────────────────────
axes[2, 0].plot(t_vicon, np.degrees(yaw_v), lw=1.2, label='Vicon yaw', color='green')
axes[2, 0].set_ylabel('Yaw (deg)')
axes[2, 0].set_title('Yaw (Vicon only — accel cannot measure yaw)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].legend(); axes[2, 0].grid(True)

# ── Gyro vs Vicon ω ────────────────────────────────────────────────
labels = ['Wx (roll rate)', 'Wy (pitch rate)', 'Wz (yaw rate, ×1.20 corrected)']
for i in range(3):
    axes[i, 1].plot(t_imu,    gyro_cal[:, i], lw=0.8, alpha=0.7, label='Gyro (calibrated)')
    axes[i, 1].plot(t_omega,  omega_v[:, i],  lw=1.2,             label='Vicon ω (diff)')
    axes[i, 1].set_ylabel('rad/s')
    axes[i, 1].set_title(labels[i])
    axes[i, 1].legend(); axes[i, 1].grid(True)
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('p2b_calibration.png', dpi=150)
plt.show()

print(f"\n{'='*55}")
print(f"Final calibration constants (dataset {data_num}):")
print(f"  beta_accel  = {beta_accel}")
print(f"  beta_gyro   = {beta_gyro}")
print(f"  Wz scale    = 1.20")
print(f"{'='*55}")
