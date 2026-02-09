import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        # 旋转 cmap 到与 actions/observations/belief_states 一致的坐标系
        cmap_rot = np.rot90(cmap, -1)
        N, M = cmap_rot.shape

        # Step 1: Prediction — b_bar = T · b
        bel_bar = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                # 从 (i,j) 出发，执行 action 后的目标格子
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < N and 0 <= nj < M:
                    bel_bar[ni, nj] += 0.9 * belief[i, j]  # 成功转移
                    bel_bar[i, j] += 0.1 * belief[i, j]    # 原地不动
                else:
                    bel_bar[i, j] += 1.0 * belief[i, j]    # 出界，留原地

        # Step 2: Update — b_new = η · M_z · b_bar
        # 用旋转后的 cmap 做观测更新
        obs_prob = np.where(cmap_rot == observation, 0.9, 0.1)
        bel_new = obs_prob * bel_bar

        # 归一化
        bel_new /= np.sum(bel_new)

        return bel_new
