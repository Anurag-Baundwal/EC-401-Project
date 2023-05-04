import numpy as np
from helpers import *

num_steps = 5000
num_exploration_steps = 2500
num_power_levels = 21  # 5% change in power level between each state, so 21 states including 0 and 100%
power_step = 0.05  # 5% change in power level

learning_rate = 0.005 #
discount_factor = 0.95 # 

# Initialize Q-tables
# one for each user in each bs
q_tables = [np.zeros((num_power_levels, num_users)) for _ in range(num_cells)]

# Generate channel matrix and path loss before starting the training loop
channel_matrices = [generate_channel_matrix(num_users, num_antennas) for _ in range(num_cells)]
path_loss_values = [path_loss(distance_bs_user, wavelength) for _ in range(num_users)]

# Initialize power levels for each user
power_levels = np.zeros((num_cells, num_users))

for step in range(num_steps):
    # Summary of this loop:
    # Make each base station transmit
    # Compute transmitted and received signals
    # Compute reward
    # Update q-tables


    # # Choose a random base station for this training step
    # cell_idx = np.random.randint(num_cells)
    
    # OUTER LOOP
    # for each cell (or each base station), we transmit
    for cell_idx, h in enumerate(channel_matrices):
        # Generate transmitted symbols
        #transmitted_symbols = generate_bpsk_symbols(num_users)
        # Generate transmitted symbol vector for the selected base station
        s = generate_bpsk_symbols(num_users)
        # Generate zero-forcing precoding matrix for the selected base station
        F = zero_forcing_precoder(channel_matrices[cell_idx], power_levels[cell_idx])
        # Compute the received symbols
        y = compute_received_symbols(channel_matrices[cell_idx], F, s, N0)
        # Compute SINR
        sinr = compute_sinr(channel_matrices[cell_idx], F, path_loss_values, N0)
        # Compute the average rate
        R_avg = compute_average_rate(sinr, num_users, channel_bandwidth)

        # Exploration or exploitation
        if step < num_exploration_steps:
            exploration = True
        else:
            exploration = False

        for user_idx in range(num_users):
            # Choose an action using epsilon-greedy policy (only during exploration steps)
            if exploration:
                action = np.random.randint(num_power_levels)
            else:
                action = np.argmax(q_tables[cell_idx][:, user_idx])

            # Update power levels based on the action
            new_power_level = max(min(action * power_step, 1), 0)
            power_levels[cell_idx, user_idx] = new_power_level

            # Update the zero-forcing precoding matrix with the new power level
            F_new = zero_forcing_precoder(channel_matrices[cell_idx], power_levels[cell_idx])

            # Compute the new received symbols
            y_new = compute_received_symbols(channel_matrices[cell_idx], F_new, s, N0)

            # Compute the new SINR
            sinr_new = compute_sinr(channel_matrices[cell_idx], F_new, path_loss_values, N0)

            # Compute the new average rate
            R_avg_new = compute_average_rate(sinr_new, num_users, channel_bandwidth)

            # Calculate the reward (change in average rate)
            reward = R_avg_new - R_avg

            # Update the Q-table
            q_tables[cell_idx][action, user_idx] = (1 - learning_rate) * q_tables[cell_idx][action, user_idx] + learning_rate * (reward + discount_factor * np.max(q_tables[cell_idx][:, user_idx]))

    # Print progress
    if (step % 1000 == 0):
        print(f"Step {step + 1}/{num_steps}, Total Reward: {total_reward}")
