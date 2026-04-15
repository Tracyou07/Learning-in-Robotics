# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        # s.xmin, s.xmax = -400, 1100
        # s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        ix = np.clip(
            np.floor((x - s.xmin) / s.resolution).astype(int), 0, s.szx - 1
        )
        iz = np.clip(
            np.floor((z - s.zmin) / s.resolution).astype(int), 0, s.szz - 1
        )
        return np.vstack((ix, iz))

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        n = p.shape[1]
        cumsum = np.cumsum(w)
        # stratified resampling: u_i = (i + U(0,1)) / N
        u = (np.arange(n) + np.random.uniform(0, 1, n)) / n
        new_p = np.zeros_like(p)
        idx = np.searchsorted(cumsum, u)
        idx = np.clip(idx, 0, n - 1)
        new_p = p[:, idx]
        new_w = np.ones(n) / n
        return new_p, new_w

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """
        # 1. Convert LiDAR points to homogeneous coordinates
        # points is (N, 3), transpose to (3, N), then make homogeneous (4, N)
        pts_h = make_homogeneous_coords_3d(points[:, :3].T)  # (4, N)

        # 2. Transform Velodyne Frame -> Camera Frame using Tr (3x4)
        Tr = s.calib  # (3, 4)
        pts_cam = Tr @ pts_h  # (3, N)

        # 3. From camera frame to world frame
        # Build 4x4 transform from particle state [x, z, theta]
        x, z, theta = p[0], p[1], p[2]
        c, sn = np.cos(theta), np.sin(theta)
        # R_y consistent with euler_to_so3(0, theta, 0):
        #   [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        T_world = np.array([
            [c,  0, sn, x],
            [0,  1,  0, 0],
            [-sn, 0, c, z],
            [0,  0,  0, 1]
        ])
        pts_cam_h = make_homogeneous_coords_3d(pts_cam)  # (4, N)
        pts_world = T_world @ pts_cam_h  # (4, N)

        return pts_world[:3, :].T  # (N, 3)

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)

        # Extract [x, z, yaw] from pose at t and t-1
        pose_t = s.poses[t]    # (3, 4)
        pose_t1 = s.poses[t-1]  # (3, 4)

        # yaw = atan2(-R[2,0], R[0,0]) matching euler_to_so3 convention
        p_t = np.array([pose_t[0, 3], pose_t[2, 3],
                        np.arctan2(-pose_t[2, 0], pose_t[0, 0])])
        p_t1 = np.array([pose_t1[0, 3], pose_t1[2, 3],
                         np.arctan2(-pose_t1[2, 0], pose_t1[0, 0])])

        return smart_minus_2d(p_t, p_t1)

    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        u = s.get_control(t)
        for i in range(s.n):
            noise = np.random.multivariate_normal(np.zeros(3), s.Q)
            s.p[:, i] = smart_plus_2d(s.p[:, i], u + noise)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        log_w = np.log(w) + obs_logp
        log_w -= slam_t.log_sum_exp(log_w)
        return np.exp(log_w)

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """

        # Load and clean LiDAR data
        lidar_path = os.path.join(s.lidar_dir, s.lidar_files[t])
        pc = load_kitti_lidar_data(lidar_path)
        pc = clean_point_cloud(pc)
        points = pc[:, :3]  # (N, 3)

        # Compute observation log-probability for each particle
        obs_logp = np.zeros(s.n)
        for i in range(s.n):
            # Transform LiDAR points to world coordinates
            world_pts = s.lidar2world(s.p[:, i], points)
            # Get grid cells for occupied points (use x and z)
            cells = s.map.grid_cell_from_xz(world_pts[:, 0], world_pts[:, 2])
            # Sum binarized map values at occupied cells
            obs_logp[i] = np.sum(s.map.cells[cells[0], cells[1]])

        # Update weights
        s.w = s.update_weights(s.w, obs_logp)

        # Use best particle to update the map
        best_idx = np.argmax(s.w)
        best_p = s.p[:, best_idx]
        world_pts = s.lidar2world(best_p, points)
        occ_cells = s.map.grid_cell_from_xz(world_pts[:, 0], world_pts[:, 2])

        # Update log-odds: accumulate evidence at observed occupied cells.
        # We do NOT blanket-decrement every cell each step: in a moving-car
        # scenario most cells are only observed a handful of times, so a
        # per-step global "free" decrement pushes them far below the
        # occupancy threshold even if they were genuinely occupied.  Sticking
        # to an occupied-only accumulator gives a much more legible map.
        s.map.log_odds[occ_cells[0], occ_cells[1]] += s.lidar_log_odds_occ

        # Clip log-odds
        np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max, out=s.map.log_odds)

        # Update binarized map
        s.map.cells = (s.map.log_odds > s.map.log_odds_thresh).astype(np.int8)

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
