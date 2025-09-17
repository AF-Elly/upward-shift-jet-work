import numpy as np
from scipy import optimize
import xarray as xr

Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
Cp=1005    # specific heat capacity of air
def calculate_beta(T, delta_T, p):
    # T_fine = interpolate_to_fine(T, p, p_fine)
    # delta_T_fine = interpolate_to_fine(delta_T, p, p_fine)
    # Calculate es, dT_dp, and dT_des as before
    dT_dp = np.gradient(T, p, axis=0)

    # Calculate beta for each time, level, and latitude
    def func(beta, delta_T, dT_dp, T, level):
        # p_adjusted = np.tile(p_fine, (181, 1)).T
        return delta_T - (beta - 1) * (p[level] * dT_dp - Rv * T ** 2 / Lv)

    # Solve for beta
    beta_init = 1.15
    beta = np.empty_like(T)
    for lat in range(T.shape[1]):
        # for level in range(p.size):
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[5, lat], dT_dp[5, lat], T[5, lat], 5))
    # Return the calculated beta array
    return beta

def calculate_beta_2D(T, delta_T, p):
    # T_fine = interpolate_to_fine(T, p, p_fine)
    # delta_T_fine = interpolate_to_fine(delta_T, p, p_fine)
    # Calculate es, dT_dp, and dT_des as before
    dT_dp = np.gradient(T, p, axis=0)

    # Calculate beta for each time, level, and latitude
    def func(beta, delta_T, dT_dp, T, level):
        # p_adjusted = np.tile(p_fine, (181, 1)).T
        return delta_T - (beta - 1) * (p[level] * dT_dp - Rv * T ** 2 / Lv)

    # Solve for beta
    beta_init = 1.15
    beta = np.empty_like(T)
    for lon in range(T.shape[2]):
        for lat in range(T.shape[1]):
            # for level in range(p.size):
            beta[:, lat, lon] = optimize.newton(func, beta_init, args=(delta_T[5, lat, lon], dT_dp[5, lat, lon], T[5, lat, lon], 5))
    # Return the calculated beta array
    return beta

#T ′(p) = T (βp) − (β−1/β)* R v/ L *T(βp)**2.
def calculate_transformed_t(t, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    t_betap = np.empty_like(t)

    for lat in range(t.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        t_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], t[::-1, lat])
    t_prime = t_betap-(beta-1)*Rv*t_betap**2/(beta*Lv)
    # Return the calculated u_prime array
    return t_prime


def calculate_transformed_t_3D(t, beta, p,pp):
    # Initialize the output array
    t_betap = np.empty_like(t)
    # Loop through months and latitudes
    for month in range(t.shape[0]):
        for lat in range(t.shape[2]):
            transformed_p = beta[:, lat, :] * pp
            # Vectorized interpolation for each longitude
            for lon in range(t.shape[3]):
                t_betap[month, :, lat, lon] = np.interp(transformed_p[::-1, lon], p[::-1], t[month, ::-1, lat, lon])[
                                              ::-1]
    t_prime = t_betap - (beta - 1) * Rv * t_betap ** 2 / (beta * Lv)

    return t_prime

def calculate_transformed_u(u, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    u_betap = np.empty_like(u)

    for lat in range(u.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        u_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], u[::-1, lat])
    u_prime = u_betap
    # Return the calculated u_prime array
    return u_prime

def calculate_transformed_q(q, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    q_betap = np.empty_like(q)

    for lat in range(q.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        q_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], q[::-1, lat])
    q_prime = q_betap
    # Return the calculated u_prime array
    return q_prime

def calculate_transformed_zg(t, zg, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    t_betap = np.empty_like(zg)
    zg_betap = np.empty_like(zg)
    for lat in range(zg.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        t_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], t[::-1, lat])
        zg_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], zg[::-1, lat])
    zg_prime = zg_betap + Cp * ((beta-1)*Rv*t_betap**2/(beta*Lv))
    # Return the calculated u_prime array
    return zg_prime
