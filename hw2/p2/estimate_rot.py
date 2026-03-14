import math
import numpy as np
from scipy.linalg import sqrtm
from quaternion import Quaternion



def estimate_rot(data_num=1):
    # ── Load IMU data ──────────────────────────────────────────────
    imu   = np.load(f"imu/imuBiased{data_num}.npy", allow_pickle=True).item()
    accel = imu['accel']   # (T,3)  stored as [-ax_body, -ay_body, az_body] + β
    gyro  = imu['gyro']    # (T,3)  stored as [Wx, Wy, Wz] + β
    ts    = imu['ts']      # (T,)   unix timestamps
    T     = ts.shape[0]

    # ── Calibration from stationary window (first 200 samples) ────
    # value = raw + β  →  β = target − mean(raw)
    N_stat = 200
    # Stationary target for accel (stored convention): [0, 0, 9.81]
    beta_accel = np.array([0., 0., 9.81]) - np.mean(accel[:N_stat], axis=0)
    # Stationary target for gyro: zero angular velocity
    beta_gyro  = -np.mean(gyro[:N_stat], axis=0)

    # ── Apply calibration ─────────────────────────────────────────
    accel_cal = accel + beta_accel
    gyro_cal  = gyro  + beta_gyro
    # Wz scale correction: this IMU's Wz channel underestimates yaw rate by ~20%
    gyro_cal[:, 2] *= 1.20

    # ── Convert accel to body frame (undo Ax,Ay hardware flip) ────
    # stored convention: [-ax_body, -ay_body, az_body]
    a_body = np.column_stack([-accel_cal[:, 0],
                              -accel_cal[:, 1],
                               accel_cal[:, 2]])

    # ── UKF constants ──────────────────────────────────────────────
    n_cov   = 6                           # covariance dimension (6×6)
    g_world = np.array([0., 0., 9.81])    # gravity in world frame (z=up)

    # Process noise per unit time (scaled by dt each step)
    R_proc = np.diag([1e-4, 1e-4, 1e-4,   # orientation noise  (rad²/s)
                      1e-2, 1e-2, 1e-2])   # angular velocity noise (rad²/s³)

    # Measurement noise covariance
    # High accel noise avoids corruption during dynamic motion; gyro dominates
    Q_meas = np.diag([2e2,  2e2,  2e2,    # accelerometer (m²/s⁴) – trust gyro more
                      1e-3, 1e-3, 1e-3])  # gyroscope     (rad²/s²)

    # ── Initialize state from first accel reading ──────────────────
    # Get initial pitch/roll from first calibrated accel measurement
    a0 = a_body[0].copy()
    a0_n = a0 / max(np.linalg.norm(a0), 1e-9)
    phi0   = np.arctan2(a0_n[1], a0_n[2])                            # roll
    theta0 = np.arctan2(-a0_n[0], np.sqrt(a0_n[1]**2 + a0_n[2]**2)) # pitch
    # Build quaternion from ZYX Euler angles (yaw=0, pitch=theta0, roll=phi0)
    cr, sr = np.cos(phi0/2),   np.sin(phi0/2)
    cp, sp = np.cos(theta0/2), np.sin(theta0/2)
    q_init = np.array([cr*cp, sr*cp, cr*sp, -sr*sp])
    q_est  = Quaternion(q_init[0], q_init[1:])
    q_est.normalize()

    w_est = gyro_cal[0].copy()           # initial angular velocity
    Sigma = np.eye(n_cov) * 1e-2         # initial covariance

    # ══════════════════════════════════════════════════════════════
    # Helper: safe axis_angle with quaternion double-cover fix
    # Ensures the error quaternion has positive scalar → θ ∈ [0,π]
    # ══════════════════════════════════════════════════════════════
    def safe_axis_angle(q):
        if q.scalar() < 0:
            q = Quaternion(-q.scalar(), -q.vec())
        return q.axis_angle()

    # ══════════════════════════════════════════════════════════════
    # Helper: quaternion mean via gradient descent  (EK Sec 3.4)
    # ══════════════════════════════════════════════════════════════
    def quat_mean_gd(qs, q_init):
        q_bar = Quaternion(q_init.scalar(), q_init.vec().copy())
        E = np.zeros((3, len(qs)))
        for _ in range(100):
            for i, qi in enumerate(qs):
                E[:, i] = safe_axis_angle(qi * q_bar.inv())
            e_bar = np.mean(E, axis=1)
            if np.linalg.norm(e_bar) < 1e-8:
                break
            dq = Quaternion()
            dq.from_axis_angle(e_bar)
            q_bar = dq * q_bar
            q_bar.normalize()
        for i, qi in enumerate(qs):
            E[:, i] = safe_axis_angle(qi * q_bar.inv())
        return q_bar, E

    # ══════════════════════════════════════════════════════════════
    # Helper: generate 2n sigma points
    # W_i = ±√n · col_i(√Σ)          6D vectors, 2n total
    # sigma_q_i = q_mean * from_axis_angle(W_i[:3])
    # sigma_w_i = w_mean + W_i[3:]
    # ══════════════════════════════════════════════════════════════
    def gen_sigma_points(q_mean, w_mean, Cov):
        S    = np.real(sqrtm(Cov))
        cols = np.sqrt(n_cov) * np.hstack([S, -S])   # (6, 2n)
        sq, sw = [], np.zeros((2 * n_cov, 3))
        for i in range(2 * n_cov):
            wi = cols[:, i]
            dq = Quaternion()
            dq.from_axis_angle(wi[:3])
            qi = q_mean * dq
            qi.normalize()
            sq.append(qi)
            sw[i] = w_mean + wi[3:]
        return sq, sw

    # ══════════════════════════════════════════════════════════════
    # Helper: rotate world vector to body frame
    # g_body = q^{-1} · g · q   (homework Appendix)
    # ══════════════════════════════════════════════════════════════
    def world_to_body(q, v):
        return (q.inv() * Quaternion(0, v) * q).vec()

    # ── Output arrays ──────────────────────────────────────────────
    roll  = np.zeros(T)
    pitch = np.zeros(T)
    yaw   = np.zeros(T)

    # ══════════════════════════════════════════════════════════════
    # Main UKF loop
    # ══════════════════════════════════════════════════════════════
    for k in range(T):
        dt = ts[k] - ts[k - 1] if k > 0 else ts[1] - ts[0]

        # ──────────────────────────────────────────────────────────
        # PREDICTION STEP
        # ──────────────────────────────────────────────────────────

        # Step 1: generate sigma points (absorb process noise into Σ)
        sq, sw = gen_sigma_points(q_est, w_est, Sigma + R_proc * dt)

        # Step 2: propagate through process model
        #   q_{k+1} = q_k · q_Δ,   q_Δ = from_axis_angle(ω · dt)
        #   ω_{k+1} = ω_k
        pq, pw = [], np.zeros((2 * n_cov, 3))
        for i in range(2 * n_cov):
            q_delta = Quaternion()
            q_delta.from_axis_angle(sw[i] * dt)
            qi_new = sq[i] * q_delta
            qi_new.normalize()
            pq.append(qi_new)
            pw[i] = sw[i]

        # Step 3: predicted mean  μ_{k+1|k}
        w_pred         = np.mean(pw, axis=0)
        q_pred, E_pred = quat_mean_gd(pq, q_est)

        # Step 4: predicted covariance  Σ_{k+1|k}
        W_prime    = np.vstack([E_pred, (pw - w_pred).T])   # (6, 2n)
        Sigma_pred = (1 / (2 * n_cov)) * (W_prime @ W_prime.T)
        Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T) + np.eye(n_cov) * 1e-9

        # ──────────────────────────────────────────────────────────
        # MEASUREMENT UPDATE STEP
        # ──────────────────────────────────────────────────────────

        # Step 5: new sigma points from predicted distribution
        sq2, sw2 = gen_sigma_points(q_pred, w_pred, Sigma_pred)

        # Step 6: measurement model
        #   ẑ_accel = q^{-1} · g_world · q   (gravity in body frame)
        #   ẑ_gyro  = ω
        Z = np.zeros((6, 2 * n_cov))
        for i in range(2 * n_cov):
            Z[:3, i] = world_to_body(sq2[i], g_world)
            Z[3:, i] = sw2[i]

        # Step 7: measurement mean, Σ_yy, Σ_xy
        z_mean   = np.mean(Z, axis=1)
        Z_diff   = Z - z_mean[:, None]
        Sigma_yy = (1 / (2 * n_cov)) * (Z_diff @ Z_diff.T) + Q_meas

        W_prime2 = np.zeros((6, 2 * n_cov))
        for i, qi in enumerate(sq2):
            W_prime2[:3, i] = safe_axis_angle(qi * q_pred.inv())
            W_prime2[3:, i] = sw2[i] - w_pred
        Sigma_xy = (1 / (2 * n_cov)) * (W_prime2 @ Z_diff.T)

        # Step 8: Kalman gain
        K = Sigma_xy @ np.linalg.inv(Sigma_yy)

        # Step 9: innovation and update
        innov   = np.concatenate([a_body[k], gyro_cal[k]]) - z_mean
        K_innov = K @ innov   # (6,)

        dq_upd = Quaternion()
        dq_upd.from_axis_angle(K_innov[:3])
        q_est = dq_upd * q_pred
        q_est.normalize()

        w_est  = w_pred + K_innov[3:]
        Sigma  = Sigma_pred - K @ Sigma_yy @ K.T
        Sigma  = 0.5 * (Sigma + Sigma.T) + np.eye(n_cov) * 1e-9

        # Step 10: extract Euler angles
        euler    = q_est.euler_angles()
        roll[k]  = euler[0]
        pitch[k] = euler[1]
        yaw[k]   = euler[2]

    return roll, pitch, yaw
