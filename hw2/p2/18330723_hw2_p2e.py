import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from quaternion import Quaternion

# ═══════════════════════════════════════════════════════════════════
# Problem 2(e): UKF Analysis and Debugging
#
# Runs the full UKF and plots:
#   1. Quaternion q (UKF mean) vs Vicon quaternion
#   2. Covariance diagonal over time
#   3. Angular velocity ω (UKF estimate vs calibrated gyro)
#   4. Calibrated gyroscope readings
# ═══════════════════════════════════════════════════════════════════

def run_ukf_with_history(data_num=1):
    """Run UKF and collect full state history for plotting."""
    # ── Load data ──────────────────────────────────────────────────
    imu      = np.load(f"imu/imuBiased{data_num}.npy", allow_pickle=True).item()
    accel    = imu['accel']
    gyro     = imu['gyro']    # already [Wx, Wy, Wz] — no reordering needed
    ts       = imu['ts']
    T        = ts.shape[0]

    vicon    = np.load(f"vicon/viconRot{data_num}.npy", allow_pickle=True).item()
    rots     = vicon['rots']
    ts_vicon = vicon['ts']

    # ── Calibration (stationary-window, N=200) ─────────────────────
    N_stat     = 200
    beta_accel = np.array([0., 0., 9.81]) - np.mean(accel[:N_stat], axis=0)
    beta_gyro  = -np.mean(gyro[:N_stat], axis=0)

    accel_cal  = accel + beta_accel
    gyro_cal   = gyro  + beta_gyro
    gyro_cal[:, 2] *= 1.20           # Wz scale correction

    a_body = np.column_stack([-accel_cal[:, 0],
                              -accel_cal[:, 1],
                               accel_cal[:, 2]])

    # ── UKF parameters ─────────────────────────────────────────────
    n_cov   = 6
    g_world = np.array([0., 0., 9.81])

    R_proc = np.diag([1e-4, 1e-4, 1e-4,   # orientation process noise
                      1e-2, 1e-2, 1e-2])   # angular velocity process noise

    Q_meas = np.diag([2e2,  2e2,  2e2,    # accel measurement noise (high → trust gyro)
                      1e-3, 1e-3, 1e-3])  # gyro measurement noise

    # ── Initialize from first accel reading ────────────────────────
    a0   = a_body[0] / max(np.linalg.norm(a_body[0]), 1e-9)
    phi0 = np.arctan2(a0[1], a0[2])
    th0  = np.arctan2(-a0[0], np.sqrt(a0[1]**2 + a0[2]**2))
    cr, sr = np.cos(phi0 / 2), np.sin(phi0 / 2)
    cp, sp = np.cos(th0  / 2), np.sin(th0  / 2)
    q_est  = Quaternion(cr * cp, [sr * cp, cr * sp, -sr * sp])
    q_est.normalize()

    w_est  = gyro_cal[0].copy()
    Sigma  = np.eye(n_cov) * 1e-2

    # ── Helpers ────────────────────────────────────────────────────
    def sax(q):
        if q.scalar() < 0:
            q = Quaternion(-q.scalar(), -q.vec())
        return q.axis_angle()

    def qmean(qs, qi):
        qb = Quaternion(qi.scalar(), qi.vec().copy())
        E  = np.zeros((3, len(qs)))
        for _ in range(50):
            for i, q in enumerate(qs):
                E[:, i] = sax(q * qb.inv())
            eb = np.mean(E, axis=1)
            if np.linalg.norm(eb) < 1e-8:
                break
            dq = Quaternion(); dq.from_axis_angle(eb)
            qb = dq * qb; qb.normalize()
        for i, q in enumerate(qs):
            E[:, i] = sax(q * qb.inv())
        return qb, E

    def gsp(qm, wm, C):
        S    = np.real(sqrtm(C))
        cols = np.sqrt(n_cov) * np.hstack([S, -S])
        sq, sw = [], np.zeros((2 * n_cov, 3))
        for i in range(2 * n_cov):
            wi = cols[:, i]
            dq = Quaternion(); dq.from_axis_angle(wi[:3])
            qi = qm * dq; qi.normalize()
            sq.append(qi); sw[i] = wm + wi[3:]
        return sq, sw

    def w2b(q, v):
        return (q.inv() * Quaternion(0, v) * q).vec()

    # ── Output buffers ─────────────────────────────────────────────
    q_hist   = np.zeros((T, 4))
    w_hist   = np.zeros((T, 3))
    cov_diag = np.zeros((T, 6))

    # ── Main UKF loop ──────────────────────────────────────────────
    for k in range(T):
        dt = ts[k] - ts[k - 1] if k > 0 else ts[1] - ts[0]

        # PREDICTION
        sq, sw = gsp(q_est, w_est, Sigma + R_proc * dt)
        pq, pw = [], np.zeros((2 * n_cov, 3))
        for i in range(2 * n_cov):
            qd = Quaternion(); qd.from_axis_angle(sw[i] * dt)
            qi = sq[i] * qd; qi.normalize()
            pq.append(qi); pw[i] = sw[i]

        wp       = np.mean(pw, axis=0)
        qp, Ep   = qmean(pq, q_est)
        Wp       = np.vstack([Ep, (pw - wp).T])
        Sp       = (1 / (2 * n_cov)) * (Wp @ Wp.T)
        Sp       = 0.5 * (Sp + Sp.T) + np.eye(n_cov) * 1e-9

        # UPDATE
        sq2, sw2 = gsp(qp, wp, Sp)
        Z = np.zeros((6, 2 * n_cov))
        for i in range(2 * n_cov):
            Z[:3, i] = w2b(sq2[i], g_world)
            Z[3:, i] = sw2[i]

        zm   = np.mean(Z, axis=1)
        Zd   = Z - zm[:, None]
        Syy  = (1 / (2 * n_cov)) * (Zd @ Zd.T) + Q_meas

        Wp2  = np.zeros((6, 2 * n_cov))
        for i, qi in enumerate(sq2):
            Wp2[:3, i] = sax(qi * qp.inv())
            Wp2[3:, i] = sw2[i] - wp
        Sxy  = (1 / (2 * n_cov)) * (Wp2 @ Zd.T)

        K    = Sxy @ np.linalg.inv(Syy)
        innov = np.concatenate([a_body[k], gyro_cal[k]]) - zm
        Ki   = K @ innov

        dq   = Quaternion(); dq.from_axis_angle(Ki[:3])
        q_est = dq * qp; q_est.normalize()
        w_est = wp + Ki[3:]
        Sigma = Sp - K @ Syy @ K.T
        Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(n_cov) * 1e-9

        q_hist[k]   = q_est.q
        w_hist[k]   = w_est
        cov_diag[k] = np.diag(Sigma)

    # ── Vicon → quaternion ─────────────────────────────────────────
    vicon_q = np.zeros((len(ts_vicon), 4))
    for i, R in enumerate(rots):
        qv = Quaternion()
        try:
            qv.from_rotm(R)
        except Exception:
            qv = Quaternion()
        vicon_q[i] = qv.q

    return {
        'ts':      ts,
        'ts_v':    ts_vicon,
        'q':       q_hist,
        'w':       w_hist,
        'cov':     cov_diag,
        'gyro':    gyro_cal,
        'vicon_q': vicon_q,
    }


def plot_results(d, data_num):
    ts    = d['ts']   - d['ts'][0]
    ts_v  = d['ts_v'] - d['ts_v'][0]
    q     = d['q']
    w     = d['w']
    cov   = d['cov']
    gyro  = d['gyro']
    vq    = d['vicon_q']

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(f'Dataset {data_num}: UKF Orientation Estimation', fontsize=13)

    # ── Plot 1: Quaternion components ─────────────────────────────
    ax = axes[0]
    labels_q = ['w', 'x', 'y', 'z']
    colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax.plot(ts,   q[:, i],    lw=0.8, color=colors[i],
                label=f'UKF q{labels_q[i]}')
        ax.plot(ts_v, vq[:, i],   lw=1.2, color=colors[i],
                ls='--', alpha=0.6, label=f'Vicon q{labels_q[i]}')
    ax.set_ylabel('Quaternion component')
    ax.set_title('Quaternion q: UKF (solid) vs Vicon (dashed)')
    ax.legend(ncol=4, fontsize=8); ax.grid(True)

    # ── Plot 2: Covariance diagonal ───────────────────────────────
    ax = axes[1]
    cov_labels = [r'$\sigma^2(\delta\phi_x)$', r'$\sigma^2(\delta\phi_y)$',
                  r'$\sigma^2(\delta\phi_z)$',
                  r'$\sigma^2(\omega_x)$',     r'$\sigma^2(\omega_y)$',
                  r'$\sigma^2(\omega_z)$']
    for i in range(6):
        ax.plot(ts, cov[:, i], lw=0.8, label=cov_labels[i])
    ax.set_ylabel('Variance')
    ax.set_title('Covariance diagonal (UKF uncertainty)')
    ax.legend(ncol=3, fontsize=8)
    ax.set_yscale('log'); ax.grid(True)

    # ── Plot 3: Angular velocity ──────────────────────────────────
    ax = axes[2]
    w_labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    for i in range(3):
        ax.plot(ts, w[:, i],     lw=0.8, label=f'UKF {w_labels[i]}')
        ax.plot(ts, gyro[:, i],  lw=0.6, ls='--', alpha=0.5,
                label=f'Gyro {w_labels[i]}')
    ax.set_ylabel('rad/s')
    ax.set_title('Angular velocity: UKF (solid) vs calibrated gyro (dashed)')
    ax.legend(ncol=3, fontsize=8); ax.grid(True)

    # ── Plot 4: Calibrated gyroscope ──────────────────────────────
    ax = axes[3]
    for i, lbl in enumerate([r'$W_x$', r'$W_y$', r'$W_z$ (×1.20 corrected)']):
        ax.plot(ts, gyro[:, i], lw=0.8, label=lbl)
    ax.set_ylabel('rad/s')
    ax.set_xlabel('Time (s)')
    ax.set_title('Calibrated gyroscope readings')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    fname = f'p2e_dataset{data_num}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.show()


if __name__ == '__main__':
    for data_num in [1, 2, 3]:
        print(f"\n── Dataset {data_num} ──────────────────────────────────")
        d = run_ukf_with_history(data_num)
        plot_results(d, data_num)

        # Compute RMSE for reporting
        from estimate_rot import estimate_rot
        roll, pitch, yaw = estimate_rot(data_num)
        vicon  = np.load(f"vicon/viconRot{data_num}.npy", allow_pickle=True).item()
        imu_ts = np.load(f"imu/imuBiased{data_num}.npy", allow_pickle=True).item()['ts']
        rots   = vicon['rots']; tsv = vicon['ts']
        rv  = np.arctan2(rots[:,2,1], rots[:,2,2])
        pv  = np.arcsin(np.clip(-rots[:,2,0], -1, 1))
        yv  = np.arctan2(rots[:,1,0], rots[:,0,0])
        rvi = np.interp(imu_ts, tsv, rv)
        pvi = np.interp(imu_ts, tsv, pv)
        yvi = np.interp(imu_ts, tsv, yv)
        print(f"  Roll  RMSE: {np.sqrt(np.mean((roll-rvi)**2)):.4f}")
        print(f"  Pitch RMSE: {np.sqrt(np.mean((pitch-pvi)**2)):.4f}")
        print(f"  Yaw   RMSE: {np.sqrt(np.mean((yaw-yvi)**2)):.4f}")
