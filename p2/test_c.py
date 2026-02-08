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

# Compute α, β, γ
alpha = hmm.forward()
beta = hmm.backward()
gamma = hmm.gamma_comp(alpha, beta)

# Compute ξ
xi = hmm.xi_comp(alpha, beta)

# Update parameters
T_prime, M_prime, pi_prime = hmm.update(gamma, xi)

state_names = ['LA', 'NY']
obs_names = ['LA', 'NY', 'null']

print("=" * 70)
print("PART (c): UPDATE HMM PARAMETERS")
print("=" * 70)

print("\n1. INITIAL DISTRIBUTION")
print("-" * 40)
print("     State |  pi (original) | pi' (updated)  |     Change")
print("-" * 40)
for i, state in enumerate(state_names):
    change = pi_prime[i] - pi[i]
    print(f"{state:>10} | {pi[i]:15.4f} | {pi_prime[i]:15.4f} | {change:+10.4f}")

print("\n2. TRANSITION MATRIX T")
print("-" * 60)
print("     From -> To  |  T (original)  |  T' (updated)  |     Change")
print("-" * 60)
for i, state_from in enumerate(state_names):
    for j, state_to in enumerate(state_names):
        change = T_prime[i, j] - T[i, j]
        transition_name = f"{state_from} -> {state_to}"
        print(f"{transition_name:>15} | {T[i,j]:15.4f} | {T_prime[i,j]:15.4f} | {change:+10.4f}")

print("\n3. EMISSION MATRIX M")
print("-" * 60)
print("    State, Obs  |  M (original)  |  M' (updated)  |     Change")
print("-" * 60)
for i, state in enumerate(state_names):
    for j, obs in enumerate(obs_names):
        change = M_prime[i, j] - M[i, j]
        emission_name = f"{state}, {obs}"
        print(f"{emission_name:>15} | {M[i,j]:15.4f} | {M_prime[i,j]:15.4f} | {change:+10.4f}")

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Check row sums = 1
print("\nTransition matrix T' row sums:")
for i, state in enumerate(state_names):
    row_sum = np.sum(T_prime[i])
    check = '✓' if abs(row_sum - 1.0) < 1e-6 else '✗'
    print(f"  {state}: {row_sum:.6f} {check}")

print("\nEmission matrix M' row sums:")
for i, state in enumerate(state_names):
    row_sum = np.sum(M_prime[i])
    check = '✓' if abs(row_sum - 1.0) < 1e-6 else '✗'
    print(f"  {state}: {row_sum:.6f} {check}")

pi_sum = np.sum(pi_prime)
pi_check = '✓' if abs(pi_sum - 1.0) < 1e-6 else '✗'
print(f"\nInitial distribution pi' sum: {pi_sum:.6f} {pi_check}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("\nKey insights from updated model lambda':")
likely_start = 'LA' if pi_prime[0] > pi_prime[1] else 'NY'
print(f"1. Initial state bias: {likely_start} (pi'[LA]={pi_prime[0]:.3f})")

t_la_trend = 'higher' if T_prime[0,0] > T[0,0] else 'lower'
print(f"2. LA->LA transition: {T_prime[0,0]:.3f} ({t_la_trend} than original)")

t_ny_trend = 'higher' if T_prime[1,1] > T[1,1] else 'lower'
print(f"3. NY->NY transition: {T_prime[1,1]:.3f} ({t_ny_trend} than original)")

la_most_obs = obs_names[np.argmax(M_prime[0])]
la_prob = M_prime[0, np.argmax(M_prime[0])]
print(f"4. Most common observation in LA: {la_most_obs} (prob={la_prob:.3f})")

ny_most_obs = obs_names[np.argmax(M_prime[1])]
ny_prob = M_prime[1, np.argmax(M_prime[1])]
print(f"5. Most common observation in NY: {ny_most_obs} (prob={ny_prob:.3f})")
