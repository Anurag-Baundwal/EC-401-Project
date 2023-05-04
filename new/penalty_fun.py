def penalty_function(power_hat, channel_quality_ratio, T, tau, kappa_1, kappa_2, rho):
    penalty = 0
    lower_limit = rho * channel_quality_ratio * (T - tau)
    upper_limit = rho * channel_quality_ratio * (T + tau)
    
    if power_hat < lower_limit:
        penalty = kappa_1 * (lower_limit - power_hat)
    elif power_hat > upper_limit:
        penalty = kappa_2 * (power_hat - upper_limit)

    return penalty