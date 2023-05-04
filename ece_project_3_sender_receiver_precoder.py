import numpy as np

def generate_channel_matrix(num_senders, num_receivers):
    h_real = np.random.normal(0, 1, (num_receivers, num_senders))
    h_imag = np.random.normal(0, 1, (num_receivers, num_senders))
    h = h_real + 1j * h_imag
    return h

def zero_forcing_precoder(h, p):
    # h: channel matrix
    # p: allocated power
    h_hermitian = h.conj().T
    w = np.linalg.inv(h_hermitian @ h) @ h_hermitian
    w_normalized = w / np.sqrt(np.sum(np.abs(w)**2, axis=1)).reshape(-1, 1)
    w_power_allocated = np.sqrt(p) * w_normalized
    return w_power_allocated

def sender(num_senders, num_receivers, num_symbols):
    # 0s and 1s matrix of size num_senders x num_symbols
    transmitted_symbols = np.random.randint(0, 2, (num_senders, num_symbols))
    return transmitted_symbols

def receiver(h, w, transmitted_symbols, snr):
    noise_power = np.power(10, -snr / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, transmitted_symbols.shape) + 1j * np.random.normal(0, 1, transmitted_symbols.shape))
    received_symbols = h @ transmitted_symbols + noise
    recovered_symbols = w @ received_symbols
    return recovered_symbols