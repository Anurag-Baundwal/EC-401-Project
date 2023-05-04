import numpy as np

def generate_channel_matrix(num_senders, num_receivers):
    h_real = np.random.normal(0, 1, (num_receivers, num_senders))
    h_imag = np.random.normal(0, 1, (num_receivers, num_senders))
    h = h_real + 1j * h_imag
    return h

def zero_forcing_precoder(h, p):
    h_hermitian = h.conj().T
    w = np.linalg.inv(h_hermitian @ h) @ h_hermitian
    w_normalized = w / np.sqrt(np.sum(np.abs(w)**2, axis=1)).reshape(-1, 1)
    w_power_allocated = np.sqrt(p) * w_normalized
    return w_power_allocated

def sender(num_senders, num_receivers, num_symbols):
    transmitted_symbols = np.random.randint(0, 2, (num_senders, num_symbols))
    return transmitted_symbols

def receiver(h, w, transmitted_symbols, snr):
    noise_power = np.power(10, -snr / 10)
    noise_shape = (h.shape[0], transmitted_symbols.shape[1])  # Update this line to fix the shape
    noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, noise_shape) + 1j * np.random.normal(0, 1, noise_shape))
    received_symbols = h @ transmitted_symbols + noise
    recovered_symbols = w @ received_symbols
    return recovered_symbols

# Parameters
num_cells = 2
num_users_per_cell = 50
num_antennas = 10
num_symbols = 100
num_episodes = 5000
num_power_levels = 10

learning_rate = 0.005
discount_factor = 0.95
exploration_rate = 1

max_power_per_user = 1
power_step = max_power_per_user / (num_power_levels - 1)
snr = 10

# Initialize Q-tables
q_tables = [np.zeros((num_power_levels, num_users_per_cell)) for _ in range(num_cells)]

# Initialize total reward
total_reward = 0

for episode in range(num_episodes):
    # Generate channel matrix for each cell
    channel_matrices = [generate_channel_matrix(num_antennas, num_users_per_cell) for _ in range(num_cells)]

    # Initialize power levels for each user
    power_levels = np.zeros((num_cells, num_users_per_cell))

    for cell_idx, h in enumerate(channel_matrices):
        # Generate transmitted symbols
        transmitted_symbols = sender(num_antennas, num_users_per_cell, num_symbols)

        for user_idx in range(num_users_per_cell):
            # Choose an action using epsilon-greedy policy
            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_tables[cell_idx][:, user_idx])
            else:
                action = np.random.randint(num_power_levels)

            # Update power levels based on the action
            new_power_level = max(min(power_levels[cell_idx, user_idx] + (action - 1) * power_step, max_power_per_user), 0)
            power_levels[cell_idx, user_idx] = new_power_level

            # Compute the precoder matrix
            w = zero_forcing_precoder(h, power_levels[cell_idx])

            # Compute the received symbols
            received_symbols = receiver(h, w, transmitted_symbols, snr)

            # Calculate the reward (SINR)
            signal_power = np.abs(w[cell_idx][user_idx] @ h[cell_idx][:, user_idx])**2
            interference_power = np.sum(np.abs(w[user_idx] @ h)**2) - signal_power
            noise_power = np.power(10, -snr / 10)
            sinr = signal_power / (interference_power + noise_power)
            reward = np.log2(1 + sinr)

            # Update the Q-table
            q_tables[cell_idx][action, user_idx] = (1 - learning_rate) * q_tables[cell_idx][action, user_idx] + learning_rate * (reward + discount_factor * np.max(q_tables[cell_idx][:, user_idx]))

            # Update total reward
            total_reward += reward

    # Print progress
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
