import numpy as np

num_cells = 2 # number of cells or base stations or agents
num_users = 10 # number of users per cell
num_antennas = 8 # number of antennas per base station

# helper 1
# Generates the transmitted symbol vector for a base station
def generate_bpsk_symbols(num_users):
    bits = np.random.randint(0, 2, num_users)
    symbols = 2 * bits - 1  # Map 0 to -1 and 1 to 1
    return symbols[:, np.newaxis]

# helper 2
# Generates the channel matrix for a base station and the users in that cell
def generate_channel_matrix(num_users, num_antennas):
    H = np.random.randn(num_users, num_antennas) + 1j * np.random.randn(num_users, num_antennas)
    return H

# helper 3
# Generates the zero-forcing precoding matrix for a base station
# power_levels is the vector of powers allocated for users by the l-th BS
# def zero_forcing_precoder(H, power_levels):
#     H_H = H.conj().T  # Conjugate transpose of H - hermitian
#     inv_term = np.linalg.inv(H @ H_H)
#     F = H_H @ inv_term
#     # F = F * np.sqrt(power_levels[:, np.newaxis])  # Apply power levels
#     F = F * np.sqrt(power_levels.reshape(-1, 1))  # Apply power levels

#     return F

def zero_forcing_precoder(h, power_levels):
    F = np.linalg.pinv(h)  # Compute pseudo-inverse of h
    F = F * np.sqrt(power_levels).T  # Apply power levels
    return F

# helper 4
# Generates noise
def generate_noise(num_users, noise_variance):
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(num_users) + 1j * np.random.randn(num_users))
    return noise[:, np.newaxis]

# finally use helpers 1, 2, 3, 4 to compute the received symbols
def compute_received_symbols(H, F, s, noise_variance):
    w = generate_noise(num_users, noise_variance)
    y = H @ (F @ s) + w
    return y


###########################
# Constants
incumbent_frequency = 2.8e9  # 2.8 GHz
light_speed = 3e8  # speed of light in m/s
wavelength = light_speed / incumbent_frequency  # wavelength in meters
distance_incumbent_licensee = 15000  # 15 km
distance_bs_user = 50  # 50 meters
channel_bandwidth = 30e6  # 30 MHz
N0_dbm = -174  # -174 dBm
N0 = 10 ** ((N0_dbm - 30) / 10)  # convert to linear scale

# Compute the path loss
def path_loss(distance_bs_user, wavelength):
    return (4 * np.pi * distance_bs_user / wavelength) ** 2

# SINR
# compute the sinr for a given user when given the channel matrix between that user and the bs, the precoder matrix, the path loss values and 
# def compute_sinr(H, F, path_loss, N0):
#     HF = np.abs(H @ F) ** 2
#     sinr = HF / path_loss / N0
#     return sinr

def compute_sinr(h, w, path_loss, N0):
    F = np.abs(np.matmul(h, w)) ** 2
    interference_plus_noise = np.diag(path_loss * (np.matmul(np.abs(w.T) ** 2, np.abs(h.T) ** 2) - F) + N0)
    sinr = F / interference_plus_noise

    # Add print statements to debug
    print(f"h: {h}, w: {w}, path_loss: {path_loss}, N0: {N0}")
    print(f"F: {F}, interference_plus_noise: {interference_plus_noise}, sinr: {sinr}")

    return sinr

# Average rate
# goal is to maximise this subject to some constraints
def compute_average_rate(sinr_values, num_users, channel_bandwidth):
    print(f"sinr_values: {sinr_values}, num_users: {num_users}, channel_bandwidth: {channel_bandwidth}")
    sinr_values = np.array(sinr_values)
    print(f"rates: {rates}")
    rates = np.log2(1 + sinr_values)
    R_avg = np.sum(rates) * channel_bandwidth / num_users
    print(f"average_rate: {average_rate}")
    return R_avg

# Action selection function
# def choose_action(state, q_table, exploration_rate):
#     if np.random.uniform(0, 1) < exploration_rate:
#         action = np.random.randint(0, 3, size=(num_cells, num_users))
#     else:
#         action = np.argmax(q_table[state], axis=2)
#     return action

# Action selection function
def choose_action(q_table, exploration_rate, num_power_levels):
    if np.random.uniform(0, 1) < exploration_rate:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(q_table)
    return action

# Perform the action and enforce the max power constraint
# def apply_action(power_levels, action, Pm):
#     new_power_levels = np.copy(power_levels)
#     new_power_levels += action - 1  # Increment, decrement, or no change
#     total_power = np.sum(new_power_levels)
    
#     if total_power > Pm:  # If the total power exceeds the max power constraint
#         return power_levels  # Return the original power levels (no change)
#     else:
#         return new_power_levels
    
def apply_action(action, power_levels, cell_idx, user_idx, power_step, Pm):
    new_power_levels = np.copy(power_levels)
    new_power_levels[cell_idx, user_idx] += (action - 1) * power_step  # Increment, decrement, or no change
    total_power = np.sum(new_power_levels)

    if total_power > Pm:  # If the total power exceeds the max power constraint
        return power_levels  # Return the original power levels (no change)
    else:
        return new_power_levels