import numpy as np
import sys
sys.path.append('..')
from HMM import HMM

# 观测映射: LA=0, NY=1, null=2
obs_map = {'null': 2, 'LA': 0, 'NY': 1}
obs_seq_raw = ['null','LA','LA','null','NY','null','NY','NY','NY','null',
               'NY','NY','NY','NY','NY','null','null','LA','LA','NY']
observations = np.array([obs_map[o] for o in obs_seq_raw])

T = np.array([[0.5, 0.5],
              [0.5, 0.5]])

M = np.array([[0.4, 0.1, 0.5],
              [0.1, 0.5, 0.4]])

pi = np.array([0.5, 0.5])

hmm = HMM(observations, T, M, pi)

alpha = hmm.forward()
beta = hmm.backward()
gamma = hmm.gamma_comp(alpha, beta)

state_names = ['LA', 'NY']

print("=== Forward (alpha) ===")
print(f"{'k':>3} | {'alpha(LA)':>12} | {'alpha(NY)':>12}")
print("-" * 35)
for k in range(len(observations)):
    print(f"{k+1:3d} | {alpha[k,0]:12.6e} | {alpha[k,1]:12.6e}")

print("\n=== Backward (beta) ===")
print(f"{'k':>3} | {'beta(LA)':>12} | {'beta(NY)':>12}")
print("-" * 35)
for k in range(len(observations)):
    print(f"{k+1:3d} | {beta[k,0]:12.6e} | {beta[k,1]:12.6e}")

print("\n=== Smoothing (gamma) ===")
print(f"{'k':>3} | {'gamma(LA)':>10} | {'gamma(NY)':>10} | {'Most likely':>10}")
print("-" * 50)
for k in range(len(observations)):
    ml = state_names[np.argmax(gamma[k])]
    print(f"{k+1:3d} | {gamma[k,0]:10.4f} | {gamma[k,1]:10.4f} | {ml:>10}")

# 验证 pointwise most likely sequence
most_likely = [state_names[np.argmax(gamma[k])] for k in range(len(observations))]
expected = ['LA','LA','LA','LA','NY','LA','NY','NY','NY','LA',
            'NY','NY','NY','NY','NY','LA','LA','LA','LA','NY']

print(f"\nMost likely sequence:  {most_likely}")
print(f"Expected sequence:     {expected}")
print(f"Match: {most_likely == expected}")
