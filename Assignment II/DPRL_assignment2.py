"""
DPRL Assignment 2: System Maintenance Problem
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================================
# PARAMETERS
# ============================================================================
N_STATES = 10
P_DETERIORATE = 0.1
REWARD_OPERATING = 1.0
COST_PREVENTIVE = 5.0
COST_CORRECTIVE = 25.0

# ============================================================================
# PART B.1: SIMULATION (NO PREVENTIVE REPAIR)
# ============================================================================
def simulate_corrective_only(n_steps=1000000, seed=42):
    np.random.seed(seed)
    c1, c2 = 1, 1
    total_reward = 0.0

    for _ in range(n_steps):
        if c1 == N_STATES or c2 == N_STATES:
            cost = (COST_CORRECTIVE if c1 == N_STATES else 0) + \
                   (COST_CORRECTIVE if c2 == N_STATES else 0)
            c1 = 1 if c1 == N_STATES else c1
            c2 = 1 if c2 == N_STATES else c2
            total_reward -= cost
            continue

        total_reward += REWARD_OPERATING

        if np.random.rand() < P_DETERIORATE:
            c1 = min(c1 + 1, N_STATES)
        if np.random.rand() < P_DETERIORATE:
            c2 = min(c2 + 1, N_STATES)

    return total_reward / n_steps

# ============================================================================
# PART B.2: STATIONARY DISTRIBUTION
# ============================================================================
def compute_stationary_distribution():
    n_states = N_STATES * N_STATES
    P = np.zeros((n_states, n_states))
    r = np.zeros(n_states)

    for s in range(n_states):
        c1 = (s % N_STATES) + 1
        c2 = (s // N_STATES) + 1

        if c1 == N_STATES or c2 == N_STATES:
            new_c1 = 1 if c1 == N_STATES else c1
            new_c2 = 1 if c2 == N_STATES else c2
            next_s = (new_c1 - 1) + (new_c2 - 1) * N_STATES
            P[s, next_s] = 1.0
            cost = (COST_CORRECTIVE if c1 == N_STATES else 0) + \
                   (COST_CORRECTIVE if c2 == N_STATES else 0)
            r[s] = -cost
        else:
            r[s] = REWARD_OPERATING
            for d1 in [0, 1]:
                for d2 in [0, 1]:
                    prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                           (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                    new_c1 = min(c1 + d1, N_STATES)
                    new_c2 = min(c2 + d2, N_STATES)
                    next_s = (new_c1 - 1) + (new_c2 - 1) * N_STATES
                    P[s, next_s] += prob

    pi = np.ones(n_states) / n_states
    for _ in range(100000):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, 1) < 1e-12:
            break
        pi = pi_next

    return pi @ r

# ============================================================================
# PART B.3: POISSON EQUATION
# ============================================================================
def poisson_equation_no_preventive(max_iter=10000, tol=1e-8):
    """
    Poisson equation + RVI.
    """
    V = np.zeros((N_STATES + 1, N_STATES + 1))
    ref_state = (1, 1)
    gain = 0.0

    for iteration in range(max_iter):
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):
                if c1 == N_STATES or c2 == N_STATES:
                    new_c1 = 1 if c1 == N_STATES else c1
                    new_c2 = 1 if c2 == N_STATES else c2
                    cost = (COST_CORRECTIVE if c1 == N_STATES else 0) + \
                           (COST_CORRECTIVE if c2 == N_STATES else 0)
                    V_new[c1, c2] = -cost + V[new_c1, new_c2]
                else:
                    value = REWARD_OPERATING
                    for d1 in [0, 1]:
                        for d2 in [0, 1]:
                            prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                                   (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                            new_c1 = min(c1 + d1, N_STATES)
                            new_c2 = min(c2 + d2, N_STATES)
                            value += prob * V[new_c1, new_c2]
                    V_new[c1, c2] = value

        gain = V_new[ref_state] - V[ref_state]
        V_new = V_new - V_new[ref_state]

        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

    return gain, V

# ============================================================================
# PART C: LIMITED PREVENTIVE
# ============================================================================
def value_iteration_limited_preventive(max_iter=10000, tol=1e-8):
    V = np.zeros((N_STATES + 1, N_STATES + 1))
    policy = np.zeros((N_STATES + 1, N_STATES + 1), dtype=int)
    ref_state = (1, 1)

    for iteration in range(max_iter):
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):

                if c1 == N_STATES and c2 == N_STATES:
                    V_new[c1, c2] = -2 * COST_CORRECTIVE + V[1, 1]
                    policy[c1, c2] = 3

                elif c1 == N_STATES and c2 < N_STATES:
                    val1 = -COST_CORRECTIVE + V[1, c2]
                    val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                    if val2 > val1:
                        V_new[c1, c2] = val2
                        policy[c1, c2] = 3
                    else:
                        V_new[c1, c2] = val1
                        policy[c1, c2] = 1

                elif c1 < N_STATES and c2 == N_STATES:
                    val1 = -COST_CORRECTIVE + V[c1, 1]
                    val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                    if val2 > val1:
                        V_new[c1, c2] = val2
                        policy[c1, c2] = 3
                    else:
                        V_new[c1, c2] = val1
                        policy[c1, c2] = 2

                else:
                    value = REWARD_OPERATING
                    for d1 in [0, 1]:
                        for d2 in [0, 1]:
                            prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                                   (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                            new_c1 = min(c1 + d1, N_STATES)
                            new_c2 = min(c2 + d2, N_STATES)
                            value += prob * V[new_c1, new_c2]
                    V_new[c1, c2] = value
                    policy[c1, c2] = 0

        gain = V_new[ref_state] - V[ref_state]
        V_new = V_new - V_new[ref_state]

        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

    return gain, V, policy

# ============================================================================
# PART D: FULL PREVENTIVE
# ============================================================================
def value_iteration_full_preventive(max_iter=10000, tol=1e-8):
    V = np.zeros((N_STATES + 1, N_STATES + 1))
    policy = np.zeros((N_STATES + 1, N_STATES + 1), dtype=int)
    ref_state = (1, 1)

    for iteration in range(max_iter):
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):

                if c1 == N_STATES or c2 == N_STATES:

                    if c1 == N_STATES and c2 == N_STATES:
                        V_new[c1, c2] = -2 * COST_CORRECTIVE + V[1, 1]
                        policy[c1, c2] = 3

                    elif c1 == N_STATES and c2 < N_STATES:
                        val1 = -COST_CORRECTIVE + V[1, c2]
                        best_value = val1
                        best_action = 1
                        val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                        if val2 > best_value:
                            best_value = val2
                            best_action = 3
                        V_new[c1, c2] = best_value
                        policy[c1, c2] = best_action

                    elif c2 == N_STATES and c1 < N_STATES:
                        val1 = -COST_CORRECTIVE + V[c1, 1]
                        best_value = val1
                        best_action = 2
                        val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                        if val2 > best_value:
                            best_value = val2
                            best_action = 3
                        V_new[c1, c2] = best_value
                        policy[c1, c2] = best_action

                    continue

                best_value = -np.inf
                best_action = 0

                value = REWARD_OPERATING
                for d1 in [0, 1]:
                    for d2 in [0, 1]:
                        prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                               (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                        new_c1 = min(c1 + d1, N_STATES)
                        new_c2 = min(c2 + d2, N_STATES)
                        value += prob * V[new_c1, new_c2]
                best_value = value

                value = -COST_PREVENTIVE + V[1, c2]
                if value > best_value:
                    best_value = value
                    best_action = 1

                value = -COST_PREVENTIVE + V[c1, 1]
                if value > best_value:
                    best_value = value
                    best_action = 2

                value = -2 * COST_PREVENTIVE + V[1, 1]
                if value > best_value:
                    best_value = value
                    best_action = 3

                V_new[c1, c2] = best_value
                policy[c1, c2] = best_action

        gain = V_new[ref_state] - V[ref_state]
        V_new = V_new - V_new[ref_state]

        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

    return gain, V, policy

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_policy(policy, title, filename=None):
    grid = policy[1:, 1:]

    plt.figure(figsize=(10, 8))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = plt.imshow(grid.T, cmap=cmap, norm=norm, origin='lower', aspect='auto',
                    extent=[0.5, N_STATES + 0.5, 0.5, N_STATES + 0.5])

    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['0: Do nothing', '1: Repair C1',
                             '2: Repair C2', '3: Repair both'])

    plt.xlabel('Component 2 State')
    plt.ylabel('Component 1 State')
    plt.title(title)
    plt.xticks(range(1, N_STATES + 1))
    plt.yticks(range(1, N_STATES + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 80)
    print(" " * 20 + "DPRL ASSIGNMENT 2: SYSTEM MAINTENANCE")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("PART B.1: Simulation (corrective only)")
    print("=" * 80)
    avg_reward_sim = simulate_corrective_only()
    print(f"φ* = {avg_reward_sim:.6f}\n")

    print("=" * 80)
    print("PART B.2: Stationary distribution")
    print("=" * 80)
    avg_reward_stat = compute_stationary_distribution()
    print(f"φ* = {avg_reward_stat:.6f}\n")

    print("=" * 80)
    print("PART B.3: Poisson equation (Value iteration)")
    print("=" * 80)
    gain_b3, V_b3 = poisson_equation_no_preventive()
    print(f"φ* = {gain_b3:.6f}\n")

    print("=" * 80)
    print("PART C: Value iteration (limited preventive)")
    print("=" * 80)
    gain_c, V_c, policy_c = value_iteration_limited_preventive()
    print(f"φ* = {gain_c:.6f}\n")

    print("Policy analysis (sample states):")
    action_names = ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both']
    print("  (1,1)  ->", action_names[int(policy_c[1,1])])
    print("  (10,1) ->", action_names[int(policy_c[10,1])])
    print("  (1,10) ->", action_names[int(policy_c[1,10])])
    print("  (10,5) ->", action_names[int(policy_c[10,5])])
    print("  (10,10)->", action_names[int(policy_c[10,10])])

    plot_policy(policy_c, "Optimal Policy - Limited Preventive",
                "Optimal_Policy_Limited_Preventive.png")
    print()

    print("=" * 80)
    print("PART D: Value iteration (full preventive)")
    print("=" * 80)
    gain_d, V_d, policy_d = value_iteration_full_preventive()
    print(f"φ* = {gain_d:.6f}\n")

    plot_policy(policy_d, "Optimal Policy - Full Preventive",
                "Optimal_Policy_Full_Preventive.png")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"B.1: φ* = {avg_reward_sim:.6f}")
    print(f"B.2: φ* = {avg_reward_stat:.6f}")
    print(f"B.3: φ* = {gain_b3:.6f}")
    print(f"C:   φ* = {gain_c:.6f}")
    print(f"D:   φ* = {gain_d:.6f}")
    print("\nExpected: B.1 ≈ B.2 ≈ B.3 ≤ C ≤ D\n")

    print(f"Limited preventive improvement: +{100*(gain_c-gain_b3)/abs(gain_b3):.1f}%")
    print(f"Full preventive improvement:    +{100*(gain_d-gain_b3)/abs(gain_b3):.1f}%")
    print("=" * 80)

    print("\nPART D POLICY ANALYSIS:")
    print("-" * 80)
    test_states = [(1,1), (5,5), (9,1), (1,9), (9,9), (10,1), (1,10), (10,10)]
    for c1, c2 in test_states:
        action = int(policy_d[c1, c2])
        print(f"({c1:2d},{c2:2d}) -> {action}: {action_names[action]}")

    print("\nPolicy counts:")
    for action in range(4):
        count = np.sum(policy_d[1:, 1:] == action)
        print(f"Action {action:1d} ({action_names[action]:15s}): {count:3d} states")

    print("=" * 80 + "\n")
