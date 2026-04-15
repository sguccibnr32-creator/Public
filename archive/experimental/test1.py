import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import stats

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
print('インポートOK')

SIGMA_CR_CONST = 1.6625e18

def sigma_cr_mpc(z_l, z_s):
    if z_s <= z_l + 0.05:
        return np.inf
    Dl  = cosmo.angular_diameter_distance(z_l).value
    Ds  = cosmo.angular_diameter_distance(z_s).value
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    if Dls <= 0:
        return np.inf
    return SIGMA_CR_CONST * Ds / (Dl * Dls)

print('sigma_cr test:', round(sigma_cr_mpc(0.2, 0.8), 3))

def nfw_delta_sigma(R_mpc, rho_s, r_s):
    x = np.atleast_1d(R_mpc / r_s).astype(float)
    kappa_s = rho_s * r_s
    h = np.zeros_like(x)
    f = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 1:
            sq = np.sqrt(1 - xi**2)
            h[i] = np.arctanh(sq) / sq + np.log(xi / 2)
            f[i] = 1/(xi**2-1) * (1 - 2/sq * np.arctanh(np.sqrt((1-xi)/(1+xi))))
        elif xi == 1:
            h[i] = 1 + np.log(0.5)
            f[i] = 1.0/3.0
        else:
            sq = np.sqrt(xi**2 - 1)
            h[i] = np.arctan(sq) / sq + np.log(xi / 2)
            f[i] = 1/(xi**2-1) * (1 - 2/sq * np.arctan(np.sqrt((xi-1)/(xi+1))))
    Sigma_bar = 4 * kappa_s / x**2 * h
    Sigma     = 2 * kappa_s * f
    return Sigma_bar - Sigma

print('nfw test:', round(nfw_delta_sigma(np.array([0.3]), 4.8e14, 0.169)[0], 2))
print('全関数OK')
