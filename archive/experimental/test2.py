import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import stats

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Sigma_cr の正しい定数 [M_sun/pc^2 -> M_sun/Mpc^2 への換算確認]
# c^2/(4piG) in M_sun/Mpc^2
# c = 3e8 m/s, G = 6.674e-11, M_sun=1.989e30 kg, Mpc=3.086e22 m
c   = 2.998e8
G   = 6.674e-11
Msun = 1.989e30
Mpc  = 3.086e22

CONST = c**2 / (4 * np.pi * G) * Mpc / Msun
print('Sigma_cr定数 [M_sun/Mpc^2 * Mpc]:', CONST)

def sigma_cr_mpc(z_l, z_s):
    if z_s <= z_l + 0.05:
        return np.inf
    Dl  = cosmo.angular_diameter_distance(z_l).value  # Mpc
    Ds  = cosmo.angular_diameter_distance(z_s).value
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    if Dls <= 0:
        return np.inf
    return CONST * Ds / (Dl * Dls)   # M_sun/Mpc^2

scr = sigma_cr_mpc(0.2, 0.8)
print('Sigma_cr(z_l=0.2, z_s=0.8) =', round(scr/1e15, 4), 'x10^15 M_sun/Mpc^2')
print('典型値は ~3-5 x 10^15 M_sun/Mpc^2 が正常')

def nfw_delta_sigma(R_mpc, rho_s, r_s):
    x = np.atleast_1d(R_mpc / r_s).astype(float)
    kappa_s = rho_s * r_s
    h = np.zeros_like(x)
    f = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 1:
            sq = np.sqrt(max(1 - xi**2, 1e-10))
            h[i] = np.arctanh(sq) / sq + np.log(xi / 2)
            f[i] = 1/(xi**2-1) * (1 - 2/sq * np.arctanh(np.sqrt(max((1-xi)/(1+xi),0))))
        elif xi == 1:
            h[i] = 1 + np.log(0.5)
            f[i] = 1.0/3.0
        else:
            sq = np.sqrt(xi**2 - 1)
            h[i] = np.arctan(sq) / sq + np.log(xi / 2)
            f[i] = 1/(xi**2-1) * (1 - 2/sq * np.arctan(np.sqrt(max((xi-1)/(xi+1),0))))
    Sigma_bar = 4 * kappa_s / x**2 * h
    Sigma     = 2 * kappa_s * f
    return Sigma_bar - Sigma

ds = nfw_delta_sigma(np.array([0.3, 0.5, 1.0]), 4.8e14, 0.169)
print('NFW DeltaSigma [M_sun/Mpc^2]:')
for r, d in zip([0.3,0.5,1.0], ds):
    print(f'  R={r}Mpc: {d:.4e}  ({d/scr:.4f} = gamma_t)')

print()
print('gamma_t @ R=0.3Mpc が 0.01~0.3 の範囲なら正常')
