import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        T = len(self.Observations)
        N = len(self.Initial_distribution)
        alpha = np.zeros((T, N))

        # alpha_1(x) = pi(x) * M(x, y_1)
        alpha[0] = self.Initial_distribution * self.Emission[:, self.Observations[0]]

        # alpha_{k+1}(x') = sum_x alpha_k(x) * T(x,x') * M(x', y_{k+1})
        for k in range(1, T):
            for j in range(N):
                alpha[k, j] = np.sum(alpha[k-1] * self.Transition[:, j]) * self.Emission[j, self.Observations[k]]

        return alpha

    def backward(self):
        T = len(self.Observations)
        N = len(self.Initial_distribution)
        beta = np.zeros((T, N))

        # beta_T(x) = 1
        beta[T-1] = 1.0

        # beta_k(x) = sum_{x'} T(x,x') * M(x', y_{k+1}) * beta_{k+1}(x')
        for k in range(T-2, -1, -1):
            for i in range(N):
                beta[k, i] = np.sum(self.Transition[i, :] * self.Emission[:, self.Observations[k+1]] * beta[k+1])

        return beta

    def gamma_comp(self, alpha, beta):
        # gamma_k(x) = alpha_k(x) * beta_k(x) / sum_x alpha_T(x)
        gamma = alpha * beta
        # 逐行归一化
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        return gamma

    def xi_comp(self, alpha, beta):
        # ξ_k(x,x') = η · α_k(x) · T_{x,x'} · M_{x',y_{k+1}} · β_{k+1}(x')
        T_time = len(self.Observations)
        N = len(self.Initial_distribution)

        # xi has shape (T-1, N, N) - no transition after last timestep
        xi = np.zeros((T_time - 1, N, N))

        for k in range(T_time - 1):
            # For each state pair (i, j)
            for i in range(N):
                for j in range(N):
                    xi[k, i, j] = (alpha[k, i]
                                   * self.Transition[i, j]
                                   * self.Emission[j, self.Observations[k+1]]
                                   * beta[k+1, j])

            # Normalize so sum over all (i,j) equals 1
            xi[k] /= np.sum(xi[k])

        return xi

    def update(self, gamma, xi):
        N = len(self.Initial_distribution)
        num_obs = self.Emission.shape[1]  # number of possible observations

        # 1. Update initial distribution: π' = γ_1
        new_init_state = gamma[0].copy()

        # 2. Update transition matrix T'
        T_prime = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                numerator = np.sum(xi[:, i, j])  # sum over k=0 to T-2 (T-1 transitions)
                denominator = np.sum(gamma[:-1, i])  # sum over k=0 to T-2 (exclude last)
                T_prime[i, j] = numerator / denominator if denominator > 0 else 0

        # 3. Update emission matrix M'
        M_prime = np.zeros((N, num_obs))
        for i in range(N):
            denominator = np.sum(gamma[:, i])  # sum over all timesteps
            for y in range(num_obs):
                # Sum gamma[k,i] only when observation at k equals y
                numerator = np.sum(gamma[self.Observations == y, i])
                M_prime[i, y] = numerator / denominator if denominator > 0 else 0

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, T_prime, M_prime, new_init_state):
        # P(Y_1,...,Y_t | λ) using original model
        # Already computed in forward(): P = sum of alpha_t(x) over all x
        alpha_original = self.forward()
        P_original = np.sum(alpha_original[-1])

        # P(Y_1,...,Y_t | λ') using updated model
        # Create new HMM with updated parameters and run forward
        hmm_prime = HMM(self.Observations, T_prime, M_prime, new_init_state)
        alpha_prime = hmm_prime.forward()
        P_prime = np.sum(alpha_prime[-1])

        return P_original, P_prime
