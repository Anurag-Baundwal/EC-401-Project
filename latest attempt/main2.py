import numpy as np
from helpers import *
from penalty_fun import *

num_cells = 1  # number of cells or base stations or agents
num_users = 10  # number of users per cell
num_antennas = 8  # number of antennas per base station
num_steps = 50
num_power_levels = 10
max_power_per_user = 1
power_step = max_power_per_user / (num_power_levels - 1)

learning_rate = 0.005
discount_factor = 0.95
exploration_rate = 1
exploration_rate_decay = 1 - (1 / (num_steps - 2500))

distance_bs_user = 50  # 50 meters

# Convert Pm (43 dBm) to linear scale (19.95 watts)
Pm = 10 ** (43 / 10) / 1000

# Initialize Q-tables
q_tables = [np.zeros((num_power_levels, num_users)) for _ in range(num_cells)]

# Initialize total reward
total_reward = 0

# Generate channel matrices for each cell
channel_matrices = [generate_channel_matrix(num_users, num_antennas) for _ in range(num_cells)]

power_levels = np.zeros((num_cells, num_users))
steps_taken = 0
for step in range(num_steps):
    for cell_idx, h in enumerate(channel_matrices):
        # Generate transmitted symbols
        transmitted_symbols = generate_bpsk_symbols(num_users)
        
        sinr_values = []  # Initialize an empty array to store SINR values for all users

        for user_idx in range(num_users):
            # Choose an action using epsilon-greedy policy
            # action = choose_action(q_tables[cell_idx][:, user_idx], exploration_rate, num_power_levels)
            action = choose_action(q_tables[cell_idx][:, user_idx], exploration_rate, num_power_levels)

            steps_taken += 1

            # update the power levels based on the action taken
            power_levels = apply_action(action, power_levels, cell_idx, user_idx, power_step, Pm)

            # precoder matrix needs to be recomputed
            w = zero_forcing_precoder(h, power_levels[cell_idx])

            # Compute the received symbols for the given user based on h, w, and transmitted symbols, and N0
            # remember that h here is for the corresponding cell 
            # and w here has been computed for the given user and the corresponding base station in his cell
            received_symbols = compute_received_symbols(h, w, transmitted_symbols, N0)

            # Calculate the SINR for the given user
            # doubt - path_loss is what? ------------------------------------------------------------------------------DOUBT
            # ok path_path is being computed based on distance and wavelength
            sinr = compute_sinr(h, w, path_loss(distance_bs_user, wavelength), N0)
            sinr_values.append(sinr)
            
            ############## IMPORTANT ################
            # Calculate the average rate
            # channel bandwidth is the bandwidth of the base station - 30 MHz
            average_rate = compute_average_rate(sinr_values, num_users, channel_bandwidth)
            
            ###### IMPORTANT ###########
            # Calculate the reward using the penalty function
            reward = 1

            # Update the Q-table
            q_tables[cell_idx][action, user_idx] = (1 - learning_rate) * q_tables[cell_idx][action, user_idx] + learning_rate * (reward + discount_factor * np.max(q_tables[cell_idx][:, user_idx]))

            # Update total reward
            total_reward += reward
    # Decay exploration rate if the number of steps taken is less than 2500
    if steps_taken < 2500:
        exploration_rate *= exploration_rate_decay

    # Print progress
    if (step + 1) % 1000 == 0:
        print(f"Step {step + 1}/{num_steps}, Total Reward: {total_reward}")
        print(f"Average rate: {average_rate}")

