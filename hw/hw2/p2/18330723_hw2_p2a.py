import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Problem 2(a): Understanding the data
#
# Load IMU and Vicon data, print basic statistics.
# ═══════════════════════════════════════════════════════════════════

data_num = 1

imu   = np.load(f'imu/imuBiased{data_num}.npy', allow_pickle=True).item()
accel = imu['accel']   # (T, 3)  raw accelerometer readings
gyro  = imu['gyro']    # (T, 3)  raw gyroscope readings  [Wx, Wy, Wz]
ts    = imu['ts']      # (T,)    unix timestamps
T     = np.shape(imu['ts'])[0]

vicon     = np.load(f'vicon/viconRot{data_num}.npy', allow_pickle=True).item()
rots      = vicon['rots']   # (N, 3, 3)  rotation matrices (body→world)
ts_vicon  = vicon['ts']     # (N,)       vicon timestamps

print(f"=== Dataset {data_num} ===")
print(f"IMU  : T={T} samples,  dt_mean={np.mean(np.diff(ts))*1000:.2f} ms")
print(f"Vicon: N={rots.shape[0]} samples, dt_mean={np.mean(np.diff(ts_vicon))*1000:.2f} ms")
print(f"Duration: {ts[-1]-ts[0]:.2f} s")
print()
print(f"accel shape: {accel.shape},  raw mean: {np.mean(accel, axis=0)}")
print(f"gyro  shape: {gyro.shape},   raw mean: {np.mean(gyro,  axis=0)}")
print()

# ── Stationary-window calibration ──────────────────────────────────
N_stat     = 200
beta_accel = np.array([0., 0., 9.81]) - np.mean(accel[:N_stat], axis=0)
beta_gyro  = -np.mean(gyro[:N_stat], axis=0)
print(f"beta_accel (stationary method): {beta_accel}")
print(f"beta_gyro  (stationary method): {beta_gyro}")
