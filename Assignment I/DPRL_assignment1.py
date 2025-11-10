import numpy as np
import matplotlib.pyplot as plt
import random
import os
from matplotlib.colors import ListedColormap

# Problem parameters
T = 150
INITIAL_INVENTORY = 5
PROFIT_PER_SALE = 1.0
HOLDING_COST = 0.1
DELIVERY_PROB = 0.5
MAX_INVENTORY = 15


def compute_expected_value(inventory, time, action, V_next):
    demand_prob = time / T
    expected_value = 0.0

    if action == 0:  # don't order
        for demand in [0, 1]:
            prob_demand = demand_prob if demand == 1 else (1 - demand_prob)
            pre_sale_inventory = inventory
            sales = min(pre_sale_inventory, demand)
            post_sale_inventory = pre_sale_inventory - sales
            reward = PROFIT_PER_SALE * sales - HOLDING_COST * post_sale_inventory
            expected_value += prob_demand * (reward + V_next[post_sale_inventory])

    else:  # order 1 item
        for delivery in [0, 1]:
            prob_delivery = DELIVERY_PROB if delivery == 1 else (1 - DELIVERY_PROB)
            pre_sale_inventory = min(inventory + delivery, MAX_INVENTORY)

            for demand in [0, 1]:
                prob_demand = demand_prob if demand == 1 else (1 - demand_prob)
                sales = min(pre_sale_inventory, demand)
                post_sale_inventory = pre_sale_inventory - sales
                reward = PROFIT_PER_SALE * sales - HOLDING_COST * post_sale_inventory
                expected_value += prob_delivery * prob_demand * (reward + V_next[post_sale_inventory])

    return expected_value


def solve_dp():
    V = np.zeros((T + 1, MAX_INVENTORY + 1))
    policy = np.zeros((T + 1, MAX_INVENTORY + 1), dtype=int)

    for t in range(T, 0, -1):
        V_next = V[t]

        for inventory in range(MAX_INVENTORY + 1):
            value_no_order = compute_expected_value(inventory, t, 0, V_next)
            value_order = compute_expected_value(inventory, t, 1, V_next)

            if value_order > value_no_order:
                V[t - 1][inventory] = value_order
                policy[t - 1][inventory] = 1
            else:
                V[t - 1][inventory] = value_no_order
                policy[t - 1][inventory] = 0

    return V, policy


V, policy = solve_dp()

expected_maximal_reward = V[0][INITIAL_INVENTORY]
print(f"\nExpected maximal reward: {expected_maximal_reward:.15f}")


def simulate_single_episode(policy):
    inventory = INITIAL_INVENTORY
    total_reward = 0.0

    for t in range(1, T + 1):
        action = policy[t - 1][inventory]

        if action == 1 and random.random() < DELIVERY_PROB:
            pre_sale_inventory = min(inventory + 1, MAX_INVENTORY)
        else:
            pre_sale_inventory = inventory

        demand_prob = t / T
        demand = 1 if random.random() < demand_prob else 0

        sales = min(pre_sale_inventory, demand)
        post_sale_inventory = pre_sale_inventory - sales
        reward = PROFIT_PER_SALE * sales - HOLDING_COST * post_sale_inventory
        total_reward += reward
        inventory = post_sale_inventory

    return total_reward


def run_simulations(policy, n_simulations=1000):
    rewards = [simulate_single_episode(policy) for _ in range(n_simulations)]
    return rewards


print("\nRunning 1000 simulations: ")
simulation_rewards = run_simulations(policy, n_simulations=1000)
avg_simulated_reward = np.mean(simulation_rewards)
std_simulated_reward = np.std(simulation_rewards)

print(f"Average simulated reward: {avg_simulated_reward:.15f}")
print(f"Standard deviation: {std_simulated_reward:.15f}")
print(f"Difference from expected: {abs(avg_simulated_reward - expected_maximal_reward):.15f}")

# Get script directory and create plots folder there
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Policy heatmap (binary colors + correct save order)
plt.figure(figsize=(10, 6))
binary_cmap = ListedColormap(["#d7191c", "#1a9641"])
plt.imshow(policy[1:, :].T, aspect='auto', cmap=binary_cmap, interpolation='nearest')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Inventory Level', fontsize=12)
plt.title('Optimal Ordering Policy Over Time', fontsize=14)
cbar = plt.colorbar(ticks=[0, 1])
cbar.ax.set_yticklabels(['No Order (0)', 'Order (1)'])
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, 'optimal_policy.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(simulation_rewards, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(expected_maximal_reward, color='r', linestyle='--', linewidth=2, label=f'Expected: {expected_maximal_reward:.6f}')
plt.axvline(avg_simulated_reward, color='g', linestyle='--', linewidth=2, label=f'Simulated Avg: {avg_simulated_reward:.6f}')
plt.xlabel('Total Reward', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Simulated Rewards (n=1000)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, 'simulation_rewards.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
