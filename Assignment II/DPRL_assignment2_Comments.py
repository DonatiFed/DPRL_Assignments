"""
DPRL Assignment 2: System Maintenance Problem
==============================================

This assignment solves a Markov Decision Process (MDP) for a system with 2 components.
The objective is to maximize the long-run average reward φ*.

Problem Description:
- 2 components, both must function for system to work
- Each component has 10 states (1-10), where 10 = failed
- Deterioration: 10% probability per time unit to move to next state
- Reward: +1 when system operates (both components < 10, no repair)
- Costs: Preventive repair = 5, Corrective repair = 25
- Repair takes 1 time unit, components return to state 1
- No deterioration during repair

Parts:
  B) No preventive repair (corrective only) - solved 3 ways
  C) Limited preventive: can repair one component if other failed
  D) Full preventive: can repair any component(s) in any state
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# ============================================================================
# PARAMETERS
# ============================================================================
N_STATES = 10  # States 1-10 for each component
P_DETERIORATE = 0.1  # Probability of deterioration per time unit
REWARD_OPERATING = 1.0  # Reward when system is operating
COST_PREVENTIVE = 5.0  # Cost of preventive repair per component
COST_CORRECTIVE = 25.0  # Cost of corrective repair per component


# ============================================================================
# PART B.1: SIMULATION (NO PREVENTIVE REPAIR)
# ============================================================================
def simulate_corrective_only(n_steps=1000000, seed=42):
    """
    Simulate system with corrective repair only.
    """
    np.random.seed(seed)
    c1, c2 = 1, 1
    total_reward = 0.0

    for _ in range(n_steps):
        # Check if repair needed
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
    """
    Compute average reward using stationary distribution.
    """
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

    # Power iteration
    pi = np.ones(n_states) / n_states
    for _ in range(100000):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, 1) < 1e-12:
            break
        pi = pi_next

    return pi @ r


# ============================================================================
# PART B.3: POISSON EQUATION (NO PREVENTIVE REPAIR)
# ============================================================================
def poisson_equation_no_preventive(max_iter=10000, tol=1e-8):
    """
    Solve Poisson equation using Relative Value Iteration (RVI).
    Poisson equation:
        φ + V(s) = r(s) + Σ_s' P(s'|s) V(s')
    where:
        - φ is the average reward (gain)
        - V(s) is the relative value function (V in code)
        - r(s) is the immediate reward
        - P(s'|s) is the transition probability
    RVI algorithm:
        1. For each state, compute:
            V_new(s) = r(s) + Σ_s' P(s'|s) V(s')
        2. Compute gain:
            φ = V_new(s_ref) - V(s_ref)
        3. Anchor:
            V_new = V_new - φ
        4. Repeat until convergence.
"""

    V = np.zeros((N_STATES + 1, N_STATES + 1))  # Relative value function h(s)
    ref_state = (1, 1)  # Reference state for anchoring
    gain = 0.0  # Average reward φ

    for iteration in range(max_iter):
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):
                if c1 == N_STATES or c2 == N_STATES:
                    # Corrective repair
                    new_c1 = 1 if c1 == N_STATES else c1
                    new_c2 = 1 if c2 == N_STATES else c2
                    cost = (COST_CORRECTIVE if c1 == N_STATES else 0) + \
                           (COST_CORRECTIVE if c2 == N_STATES else 0)
                    V_new[c1, c2] = -cost + V[new_c1, new_c2]
                else:
                    # Normal operation
                    value = REWARD_OPERATING
                    for d1 in [0, 1]:
                        for d2 in [0, 1]:
                            prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                                   (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                            new_c1 = min(c1 + d1, N_STATES)
                            new_c2 = min(c2 + d2, N_STATES)
                            value += prob * V[new_c1, new_c2]
                    V_new[c1, c2] = value

        # RVI: compute gain and anchor
        gain = V_new[ref_state] - V[ref_state]
        V_new = V_new - V_new[ref_state]

        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

    return gain, V


# ============================================================================
# PART C: LIMITED PREVENTIVE REPAIR
# ============================================================================
def value_iteration_limited_preventive(max_iter=10000, tol=1e-8):
    """
    Value iteration with limited preventive repair.
    
    Policy constraint: Can repair component preventively ONLY if other has failed.
    - If c1=10, c2<10: Choose between (repair c1) or (repair both)
    - If c1<10, c2=10: Choose between (repair c2) or (repair both)
    - If c1=10, c2=10: Must repair both (corrective)
    - If c1<10, c2<10: Cannot do preventive (action 0 only)
    
    Uses RVI to find optimal policy and average reward φ*.
    """
    V = np.zeros((N_STATES + 1, N_STATES + 1))  # Relative value function
    policy = np.zeros((N_STATES + 1, N_STATES + 1), dtype=int)  # Optimal policy
    ref_state = (1, 1)  # Reference state
    gain = 0.0  # Average reward φ

    for iteration in range(max_iter): 
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):
                
                # Both failed
                if c1 == N_STATES and c2 == N_STATES:
                    V_new[c1, c2] = -2 * COST_CORRECTIVE + V[1, 1]
                    policy[c1, c2] = 3
                
                # Only c1 failed - can optionally repair c2 preventively
                elif c1 == N_STATES and c2 < N_STATES:
                    # Option 1: Repair c1 only
                    val1 = -COST_CORRECTIVE + V[1, c2]
                    # Option 2: Repair both
                    val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                    
                    if val2 > val1:
                        V_new[c1, c2] = val2
                        policy[c1, c2] = 3
                    else:
                        V_new[c1, c2] = val1
                        policy[c1, c2] = 1
                
                # Only c2 failed - can optionally repair c1 preventively
                elif c1 < N_STATES and c2 == N_STATES:
                    # Option 1: Repair c2 only
                    val1 = -COST_CORRECTIVE + V[c1, 1]
                    # Option 2: Repair both
                    val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                    
                    if val2 > val1:
                        V_new[c1, c2] = val2
                        policy[c1, c2] = 3
                    else:
                        V_new[c1, c2] = val1
                        policy[c1, c2] = 2
                
                # Both operational - no preventive allowed
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

        # RVI
        gain = V_new[ref_state] - V[ref_state]
        V_new = V_new - V_new[ref_state]

        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

    return gain, V, policy


# ============================================================================
# PART D: FULL PREVENTIVE REPAIR
# ============================================================================
def value_iteration_full_preventive(max_iter=10000, tol=1e-8):
    """
    Value iteration with full preventive repair.
    
    Available actions:
      - In non-failure states (c1<10, c2<10):
          0: Do nothing
          1: Preventive repair on component 1
          2: Preventive repair on component 2
          3: Preventive repair on both
      - In failure states (at least one == 10):
          All failed components are repaired correctively in this step.
          Healthy components may additionally be repaired preventively.
    """
    V = np.zeros((N_STATES + 1, N_STATES + 1))  # Relative value function
    policy = np.zeros((N_STATES + 1, N_STATES + 1), dtype=int)  # Optimal policy
    ref_state = (1, 1)  # Reference state
    gain = 0.0  # Average reward φ

    for iteration in range(max_iter):
        V_new = np.zeros((N_STATES + 1, N_STATES + 1))

        for c1 in range(1, N_STATES + 1):
            for c2 in range(1, N_STATES + 1):

                # ----------------------------------------------------------
                # FAILURE STATES: at least one component failed
                # ----------------------------------------------------------
                if c1 == N_STATES or c2 == N_STATES:
                    # Both failed: only corrective on both
                    if c1 == N_STATES and c2 == N_STATES:
                        V_new[c1, c2] = -2 * COST_CORRECTIVE + V[1, 1]
                        policy[c1, c2] = 3  # "repair both"
                    
                    # Only c1 failed: corrective on c1, optionally preventive on c2
                    elif c1 == N_STATES and c2 < N_STATES:
                        # Option 1: corrective on c1 only
                        val1 = -COST_CORRECTIVE + V[1, c2]
                        best_value = val1
                        best_action = 1  # "repair c1"

                        # Option 2: corrective on c1 + preventive on c2
                        val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                        if val2 > best_value:
                            best_value = val2
                            best_action = 3  # "repair both"

                        V_new[c1, c2] = best_value
                        policy[c1, c2] = best_action

                    # Only c2 failed: corrective on c2, optionally preventive on c1
                    elif c2 == N_STATES and c1 < N_STATES:
                        # Option 1: corrective on c2 only
                        val1 = -COST_CORRECTIVE + V[c1, 1]
                        best_value = val1
                        best_action = 2  # "repair c2"

                        # Option 2: corrective on c2 + preventive on c1
                        val2 = -COST_CORRECTIVE - COST_PREVENTIVE + V[1, 1]
                        if val2 > best_value:
                            best_value = val2
                            best_action = 3  # "repair both"

                        V_new[c1, c2] = best_value
                        policy[c1, c2] = best_action

                    continue  # done with failure states

                # ----------------------------------------------------------
                # NON-FAILURE STATES: full preventive freedom
                # ----------------------------------------------------------
                best_value = -np.inf
                best_action = 0

                # Action 0: Do nothing
                value = REWARD_OPERATING
                for d1 in [0, 1]:
                    for d2 in [0, 1]:
                        prob = (P_DETERIORATE if d1 else 1 - P_DETERIORATE) * \
                               (P_DETERIORATE if d2 else 1 - P_DETERIORATE)
                        new_c1 = min(c1 + d1, N_STATES)
                        new_c2 = min(c2 + d2, N_STATES)
                        value += prob * V[new_c1, new_c2]
                best_value = value
                best_action = 0

                # Action 1: Preventive repair on c1
                value = -COST_PREVENTIVE + V[1, c2]
                if value > best_value:
                    best_value = value
                    best_action = 1

                # Action 2: Preventive repair on c2
                value = -COST_PREVENTIVE + V[c1, 1]
                if value > best_value:
                    best_value = value
                    best_action = 2

                # Action 3: Preventive repair on both
                value = -2 * COST_PREVENTIVE + V[1, 1]
                if value > best_value:
                    best_value = value
                    best_action = 3

                V_new[c1, c2] = best_value
                policy[c1, c2] = best_action

        # RVI
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
    """
    Plot optimal policy heatmap.
    """
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
    cbar.ax.set_yticklabels(['0: Do nothing', '1: Repair C1', '2: Repair C2', '3: Repair both'])

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

    # Part B.1
    print("=" * 80)
    print("PART B.1: Simulation (corrective only)")
    print("=" * 80)
    avg_reward_sim = simulate_corrective_only()
    print(f"φ* = {avg_reward_sim:.6f}\n")

    # Part B.2
    print("=" * 80)
    print("PART B.2: Stationary distribution")
    print("=" * 80)
    avg_reward_stat = compute_stationary_distribution()
    print(f"φ* = {avg_reward_stat:.6f}\n")

    # Part B.3
    print("=" * 80)
    print("PART B.3: Poisson equation (Value iteration)")
    print("=" * 80)
    gain_b3, V_b3 = poisson_equation_no_preventive()
    print(f"φ* = {gain_b3:.6f}\n")

    # Part C
    print("=" * 80)
    print("PART C: Value iteration (limited preventive)")
    print("=" * 80)
    gain_c, V_c, policy_c = value_iteration_limited_preventive()
    print(f"φ* = {gain_c:.6f}")
    
    print("\nPolicy analysis (sample states):")
    print("  (1, 1) -> Action {}: {}".format(int(policy_c[1,1]), 
          ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both'][int(policy_c[1,1])]))
    print("  (10, 1) -> Action {}: {} (C1 failed, C2 good)".format(int(policy_c[10,1]),
          ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both'][int(policy_c[10,1])]))
    print("  (1, 10) -> Action {}: {} (C1 good, C2 failed)".format(int(policy_c[1,10]),
          ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both'][int(policy_c[1,10])]))
    print("  (10, 5) -> Action {}: {} (C1 failed, C2 moderate)".format(int(policy_c[10,5]),
          ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both'][int(policy_c[10,5])]))
    print("  (10, 10) -> Action {}: {} (both failed)".format(int(policy_c[10,10]),
          ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both'][int(policy_c[10,10])]))
    
    plot_policy(policy_c, "Optimal Policy - Limited Preventive", 
                "Optimal_Policy_Limited_Preventive.png")
    print()

    # Part D
    print("=" * 80)
    print("PART D: Value iteration (full preventive)")
    print("=" * 80)
    gain_d, V_d, policy_d = value_iteration_full_preventive()
    print(f"φ* = {gain_d:.6f}")
    plot_policy(policy_d, "Optimal Policy - Full Preventive",
                "Optimal_Policy_Full_Preventive.png")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"B.1 (Simulation):              φ* = {avg_reward_sim:.6f}")
    print(f"B.2 (Stationary distribution): φ* = {avg_reward_stat:.6f}")
    print(f"B.3 (Poisson equation):        φ* = {gain_b3:.6f}")
    print(f"C   (Limited preventive):      φ* = {gain_c:.6f}")
    print(f"D   (Full preventive):         φ* = {gain_d:.6f}")
    print("\nExpected: B.1 ≈ B.2 ≈ B.3 ≤ C ≤ D")
    print(f"\nImprovements:")
    print(f"  Limited preventive: +{100*(gain_c-gain_b3)/abs(gain_b3):.1f}%")
    print(f"  Full preventive:    +{100*(gain_d-gain_b3)/abs(gain_b3):.1f}%")
    print("=" * 80)
    
    # Policy analysis for Part D
    print("\nPART D POLICY ANALYSIS:")
    print("-" * 80)
    print("Sample states (C1, C2) -> Action:")
    test_states = [(1,1), (5,5), (9,1), (1,9), (9,9), (10,1), (1,10), (10,10)]
    action_names = ['Do nothing', 'Repair C1', 'Repair C2', 'Repair both']
    for c1, c2 in test_states:
        action = int(policy_d[c1, c2])
        print(f"  ({c1:2d}, {c2:2d}) -> {action}: {action_names[action]}")
    
    print("\nPolicy counts:")
    for action in range(4):
        count = np.sum(policy_d[1:, 1:] == action)
        print(f"  Action {action} ({action_names[action]:15s}): {count:3d} states")
    print("=" * 80 + "\n")
