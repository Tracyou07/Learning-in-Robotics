# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import click, tqdm, random

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx)

    # Trajectory using odometry (xz and yaw) in the lidar data
    d = slam.poses
    pose = np.column_stack([d[:,0,3], d[:,1,3], d[:,2,3]]) # X Y Z
    plt.figure(1)
    plt.clf()
    plt.title('Trajectory using onboard odometry')
    plt.plot(pose[:,0], pose[:,2])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(pose[0])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)
    plt.figure(2)
    plt.clf()
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))

def run_observation_step(src_dir, log_dir, idx, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.5)
    slam.read_data(src_dir, idx)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    d = slam.poses
    pose = np.column_stack([d[t0,0,3], d[t0,1,3], np.arctan2(-d[t0,2,0], d[t0,0,0])])
    logging.debug('> Initializing 1 particle at: {}'.format(pose))
    slam.init_particles(n=1,p=pose.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=50 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.5, Q=np.diag([1e-4,1e-4,1e-5]))
    slam.read_data(src_dir, idx)

    os.makedirs(log_dir, exist_ok=True)

    T = len(slam.lidar_files)
    d = slam.poses
    T = min(T, len(d))

    # Extract initial pose [x, z, yaw]
    p0 = np.array([d[0, 0, 3], d[0, 2, 3],
                    np.arctan2(-d[0, 2, 0], d[0, 0, 0])])

    # Initialize map with first observation using 1 particle at the initial pose
    slam.init_particles(n=1, p=p0.reshape(3, 1), w=np.array([1.0]))
    slam.observation_step(t=0)

    # Initialize n=50 particles at the initial pose
    n = 50
    slam.init_particles(n=n, p=np.tile(p0.reshape(3, 1), (1, n)),
                        w=np.ones(n) / n)

    best_trajectory = [p0.copy()]

    for t in tqdm.tqdm(range(1, T)):
        slam.dynamics_step(t)
        slam.observation_step(t)
        slam.resample_particles()

        best_idx = np.argmax(slam.w)
        best_trajectory.append(slam.p[:, best_idx].copy())

    best_trajectory = np.array(best_trajectory)

    # Plot odometry trajectory from poses
    odom_x = d[:T, 0, 3]
    odom_z = d[:T, 2, 3]

    # Plot binarized map
    plt.figure(figsize=(12, 10))
    plt.imshow(slam.map.cells.T, cmap='gray_r', origin='lower',
               extent=[slam.map.xmin, slam.map.xmax,
                       slam.map.zmin, slam.map.zmax])
    plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'r-', linewidth=1,
             label='SLAM trajectory')
    plt.plot(odom_x, odom_z, 'b--', linewidth=1, alpha=0.5,
             label='Odometry trajectory')
    plt.legend()
    plt.title('Particle Filter SLAM')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.axis('equal')
    plt.savefig(os.path.join(log_dir, 'slam_%s.jpg' % idx))
    logging.info('> Saved slam plot to %s' % os.path.join(log_dir, 'slam_%s.jpg' % idx))
    plt.close()

    # Plot just the map
    plt.figure(figsize=(12, 10))
    plt.imshow(slam.map.cells.T, cmap='gray_r', origin='lower',
               extent=[slam.map.xmin, slam.map.xmax,
                       slam.map.zmin, slam.map.zmax])
    plt.title('Occupancy Grid Map')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.savefig(os.path.join(log_dir, 'map_%s.jpg' % idx))
    logging.info('> Saved map plot to %s' % os.path.join(log_dir, 'map_%s.jpg' % idx))
    plt.close()

    return best_trajectory


@click.command()
@click.option('--src_dir', default='./KITTI/', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='00', help='dataset number', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx)
        return p

if __name__=='__main__':
    main()
