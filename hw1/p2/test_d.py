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

# Compute probabilities
P_original, P_prime = hmm.trajectory_probability(T_prime, M_prime, pi_prime)

print("=" * 70)
print("PART (d): COMPARE MODEL LIKELIHOODS")
print("=" * 70)

print("\nObservation sequence (20 timesteps):")
print(obs_seq_raw)

print("\n" + "-" * 70)
print("LIKELIHOOD COMPARISON")
print("-" * 70)

print(f"\nP(Y_1,...,Y_t | λ)  = {P_original:.6e}")
print(f"P(Y_1,...,Y_t | λ') = {P_prime:.6e}")

ratio = P_prime / P_original
print(f"\nRatio P(λ') / P(λ)  = {ratio:.6f}")

improvement = (P_prime - P_original) / P_original * 100
print(f"Relative improvement = {improvement:+.2f}%")

print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)

if P_prime > P_original:
    print("\n✓ SUCCESS: P(Y_1,...,Y_t | λ') > P(Y_1,...,Y_t | λ)")
    print("\nThe updated model λ' is MORE LIKELY to generate the observed")
    print("sequence than the original model λ.")
    print("\nThis confirms the Baum-Welch algorithm successfully improved")
    print("the model parameters to better fit the observations.")
else:
    print("\n✗ UNEXPECTED: P(Y_1,...,Y_t | λ') <= P(Y_1,...,Y_t | λ)")
    print("\nThis should not happen with correct Baum-Welch implementation.")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("\nThe Baum-Welch algorithm is an EM (Expectation-Maximization) algorithm:")
print("\n1. E-step: Compute γ and ξ (expected sufficient statistics)")
print("   - γ_k(x): Expected time in state x at timestep k")
print("   - ξ_k(x,x'): Expected transitions from x to x' at timestep k")
print("\n2. M-step: Update parameters to maximize likelihood")
print("   - π' = γ_1 (initial state distribution)")
print("   - T'_{x,x'} = (expected x→x' transitions) / (expected time in x)")
print("   - M'_{x,y} = (expected time in x with obs y) / (expected time in x)")

print(f"\nAfter one iteration, the likelihood increased by {improvement:.2f}%.")
print("Multiple iterations would continue to improve the model until convergence.")

print("\n" + "=" * 70)
print("LOG-LIKELIHOOD (for numerical stability)")
print("=" * 70)

log_P_original = np.log(P_original)
log_P_prime = np.log(P_prime)

print(f"\nlog P(Y_1,...,Y_t | λ)  = {log_P_original:.4f}")
print(f"log P(Y_1,...,Y_t | λ') = {log_P_prime:.4f}")
print(f"\nDifference: {log_P_prime - log_P_original:.4f}")
