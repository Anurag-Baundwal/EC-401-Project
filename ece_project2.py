import numpy as np


# Number of users/agents
num_cells = 2
num_users_per_cell = 50

# Number of actions
# 3 - increase, decrease, keep same
num_actions = 3

# Number of states
# ie, number of power levels
num_states = 10

max_power = 19.95 #19.95 watts or 43 dBm
power_step = max_power/num_actions # max power divived by number of power levels

# Learning rate
learning_rate = 5 * (10**-3)

# Discount factor
discount_factor = 0.95

# epsilon_greedy for exploration phase
exploration_rate = 1

# # Q table:
# q_table = []
# for i in range(num_cells):
#     cell_q_table = []
#     for j in range(num_users_per_cell):
#         user_q_table = np.zeros((num_states, num_actions))
#         cell_q_table.append(user_q_table)
#     q_tables.append(cell_q_table)
q_table = [[np.zeros((num_states, num_actions)) for j in range(num_users_per_cell)] for i in range(num_cells)]


# Initialize total reward
total_reward = 0

# Define the reward matrix
reward_matrix = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0]])

# Define the transition matrix
transition_matrix = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1]
])

# Define the episode length
num_steps = 5000

# Loop over the episodes
for i in range(num_steps):
    # Initialize state
    state = 0
    
    # Initialize rewards for each agent
    rewards = []
    for i in range(num_cells):
        cell_rewards = []
        for j in range(num_users_per_cell):
            cell_rewards.append(0)
        rewards.append(cell_rewards)
    
    # Loop over each agent
    for i in range(num_cells):
        for j in range(num_users_per_cell):
            # Choose an action using epsilon-greedy policy
            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_tables[i][j][state, :])
            else:
                action = np.random.randint(num_actions)
            
            # Execute the action and observe the reward and new state
            new_state = np.argmax(transition_matrix[state, action, :])
            reward = reward_matrix[new_state, i]
            
            # Update Q table
            q_tables[i][j][state, action] = q_tables[i][j][state, action] * (1 - learning_rate) + \
                                            learning_rate * (reward + discount_factor * np.max(q_tables[i][j][new_state, :]))
            
            # Update total reward and agent reward
            total_reward += reward
            rewards[i][j] += reward
            
            # Update state
            state = new_state
    
    # Print progress
    if (i + 1) % 1000 == 0:
        print(f"Episode {i+1}/{num_steps}, Total Reward: {total_reward}")
