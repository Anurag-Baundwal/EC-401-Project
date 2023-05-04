def compute_received_signal(H, F, s, w):
    y = H @ F @ s + w
    return y

# @ is for matrix multiplication in python

##################
# zero forcing precoding matrix F
def calculate_F(H, p):

    # calculate H^H
    H_H = H.conj().T
    
    # calculate (H * H^H)^-1
    H_H_inv = np.linalg.inv(H @ H_H)
    
    # calculate sqrt(p) as a diagonal matrix
    sqrt_p = np.diag(np.sqrt(p))
    
    # calculate F using equation (2)
    F = H_H @ H_H_inv @ sqrt_p
    
    return F

#############

def calc_received_signal(H, F, s, w, n):
    desired_signal = (H @ F)[n, n] * s[n]
    interference = np.sum((H @ F)[:, n] * s) - desired_signal
    noise = np.random.normal(0, np.sqrt(w))
    return desired_signal + interference + noise

############

def calculate_SINR(l, n_l, H, F, p, N_l, B, N0):
    numerator = abs(H[l] @ F[l] @ np.diag(np.sqrt(p[l]))[:, n_l])**2
    denominator = 0
    for k in range(N_l):
        if k != n_l:
            denominator += abs(H[l] @ F[l] @ np.diag(np.sqrt(p[l]))[:, k])**2 / (B * N0 + calculate_interference(l, n_l, k, H, F, p, N_l))
    return numerator / denominator

############

def calc_SINR_eq5(HF, ul_noise, ul_var):
    numerator = np.square(np.abs(HF.diagonal()))
    denominator = ul_noise / ul_var
    return numerator / denominator

#########

