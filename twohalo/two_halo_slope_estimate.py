#!/usr/bin/env python3
"""two_halo_slope_estimate.py — 2-halo contribution to gc-M* slope"""

import sys
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))

G_SI = 6.674e-11
Msun = 1.989e30
Mpc = 3.086e22
kpc = 3.086e19
c_light = 2.998e8
H0 = 70.0
h = H0 / 100.0
rho_crit = 3 * (H0*1e3/Mpc)**2 / (8*np.pi*G_SI)
Omega_m = 0.3
rho_mean = Omega_m * rho_crit
a0_SI = 1.2e-10

MSTAR_BINS = [
    (8.5, 10.3, 9.4),
    (10.3, 10.6, 10.45),
    (10.6, 10.8, 10.70),
    (10.8, 11.0, 10.90),
]


def log_Mhalo(logMstar):
    logMs_pivot = 10.5
    logMh_pivot = 12.0
    slope = 1.5 if logMstar < logMs_pivot else 2.5
    return logMh_pivot + slope * (logMstar - logMs_pivot)


def sigma_M(logMh):
    logM_arr = np.array([10, 11, 12, 13, 14, 15])
    sig_arr = np.array([2.0, 1.2, 0.80, 0.55, 0.35, 0.22])
    f = interp1d(logM_arr, sig_arr, fill_value='extrapolate')
    return float(f(logMh))


def halo_bias(logMh):
    delta_c = 1.686
    sig = sigma_M(logMh)
    nu = delta_c / sig
    b = 1.0 + (nu**2 - 1) / delta_c
    return max(b, 0.5)


def xi_mm(r_Mpc):
    r0 = 5.0 / h
    gamma = 1.8
    if r_Mpc < 0.01:
        return (0.01 / r0)**(-gamma)
    return (r_Mpc / r0)**(-gamma)


def Sigma_mm(R_Mpc, L_max=100.0):
    def integrand(l):
        r = np.sqrt(R_Mpc**2 + l**2)
        return xi_mm(r)
    result, _ = quad(integrand, -L_max, L_max, limit=100)
    return rho_mean * result * Mpc


def Delta_Sigma_2h(R_Mpc, logMstar):
    logMh = log_Mhalo(logMstar)
    b = halo_bias(logMh)
    Sig = Sigma_mm(R_Mpc)
    Sig_Msun_pc2 = Sig / Msun * (3.086e16)**2
    return b * Sig_Msun_pc2


def Delta_Sigma_1h(R_Mpc, logMstar):
    v_flat = 10**(0.25 * (logMstar - 10.0) + 2.1)
    sigma_v = 0.7 * v_flat
    sigma_v_ms = sigma_v * 1e3
    R_m = R_Mpc * Mpc
    DS = sigma_v_ms**2 / (2 * G_SI * R_m)
    return DS / Msun * (3.086e16)**2


def compute_gc_with_2halo(logMstar, R_range=(0.03, 2.0), N_R=15):
    R_arr = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), N_R)
    M_gal = 10**logMstar * Msun
    g_bar_arr = G_SI * M_gal / (R_arr * Mpc)**2

    ESD_to_g = 4 * G_SI * Msun / (3.086e16)**2

    g_1h_arr = np.array([Delta_Sigma_1h(R, logMstar) * ESD_to_g for R in R_arr])
    g_2h_arr = np.array([Delta_Sigma_2h(R, logMstar) * ESD_to_g for R in R_arr])
    g_obs_arr = g_1h_arr + g_2h_arr
    f_2h = g_2h_arr / g_obs_arr

    def chi2_gc(log_gc):
        gc = 10**log_gc * a0_SI
        x = np.sqrt(g_bar_arr / gc)
        x = np.clip(x, 0, 50)
        model = g_bar_arr / (1.0 - np.exp(-x))
        valid = (g_obs_arr > 0) & (g_bar_arr > 0)
        return np.sum((np.log10(g_obs_arr[valid]) - np.log10(model[valid]))**2)

    result = minimize_scalar(chi2_gc, bounds=(-2, 2), method='bounded')
    gc_a0 = 10**result.x
    return gc_a0, R_arr, f_2h, g_1h_arr, g_2h_arr


def main():
    print("="*70)
    print("2-halo contribution to gc-M* slope")
    print("="*70)

    print("\nPart 1: SHMR & Halo Bias")
    print("-"*50)
    print(f"  {'logM*':>8s} {'logM_h':>8s} {'b(M_h)':>8s} {'v_flat':>8s}")
    for lo, hi, logMs in MSTAR_BINS:
        logMh = log_Mhalo(logMs)
        b = halo_bias(logMh)
        vf = 10**(0.25*(logMs-10)+2.1)
        print(f"  {logMs:8.1f} {logMh:8.2f} {b:8.2f} {vf:8.0f}")

    b1 = halo_bias(log_Mhalo(MSTAR_BINS[0][2]))
    b4 = halo_bias(log_Mhalo(MSTAR_BINS[3][2]))
    print(f"\n  b(Bin4)/b(Bin1) = {b4/b1:.2f}")

    print("\nPart 2: 2-halo fraction vs R")
    print("-"*50)
    R_plot = np.logspace(-1.5, 0.5, 30)
    print(f"  {'R [Mpc]':>10s}", end='')
    for _, _, logMs in MSTAR_BINS:
        print(f"  f2h(M{logMs:.1f})", end='')
    print()

    f2h_all = {}
    for _, _, logMs in MSTAR_BINS:
        f2h_arr = []
        for R in R_plot:
            ds1h = Delta_Sigma_1h(R, logMs)
            ds2h = Delta_Sigma_2h(R, logMs)
            f2h = ds2h / (ds1h + ds2h) if (ds1h + ds2h) > 0 else 0
            f2h_arr.append(f2h)
        f2h_all[logMs] = np.array(f2h_arr)

    for i_r in range(0, len(R_plot), 5):
        R = R_plot[i_r]
        print(f"  {R:10.3f}", end='')
        for _, _, logMs in MSTAR_BINS:
            print(f"  {f2h_all[logMs][i_r]:10.3f}", end='')
        print()

    print("\nPart 3: gc with/without 2-halo per M* bin")
    print("-"*50)
    gc_no2h = []
    gc_with2h = []
    for lo, hi, logMs in MSTAR_BINS:
        gc_1h, _, _, _, _ = compute_gc_with_2halo(logMs, R_range=(0.03, 0.3))
        gc_full, R_arr, f_2h, g_1h, g_2h = compute_gc_with_2halo(logMs)
        gc_no2h.append(gc_1h)
        gc_with2h.append(gc_full)
        f2h_mean = np.mean(f_2h)
        print(f"  logM*={logMs:.1f}: gc(1h)={gc_1h:.3f}, "
              f"gc(1h+2h)={gc_full:.3f}, ratio={gc_full/gc_1h:.3f}, "
              f"<f_2h>={f2h_mean:.3f}")

    gc_no2h = np.array(gc_no2h)
    gc_with2h = np.array(gc_with2h)
    logMs_arr = np.array([b[2] for b in MSTAR_BINS])
    log_gc_no2h = np.log10(gc_no2h)
    log_gc_with2h = np.log10(gc_with2h)

    slope_no2h = np.polyfit(logMs_arr, log_gc_no2h, 1)[0]
    slope_with2h = np.polyfit(logMs_arr, log_gc_with2h, 1)[0]
    slope_2h_contrib = slope_with2h - slope_no2h

    print(f"\n  gc-M* slope:")
    print(f"    1-halo only:         {slope_no2h:+.4f}")
    print(f"    1-halo + 2-halo:     {slope_with2h:+.4f}")
    print(f"    2-halo contribution: {slope_2h_contrib:+.4f}")

    print("\n" + "="*70)
    print("Part 4: Slope budget")
    print("="*70)
    slope_C15 = 0.075
    slope_ModelB = 0.017
    slope_obs = 0.166
    slope_obs_err = 0.041
    total = slope_C15 + slope_ModelB + slope_2h_contrib
    residual = (slope_obs - total) / slope_obs_err

    print(f"  C15 (SPARC):              {slope_C15:+.4f}")
    print(f"  Model B (v2, max):        {slope_ModelB:+.4f}")
    print(f"  2-halo (this estimate):   {slope_2h_contrib:+.4f}")
    print(f"  Total predicted:          {total:+.4f}")
    print(f"  HSC observed:             {slope_obs:+.4f} +/- {slope_obs_err}")
    print(f"  Residual:                 {residual:+.2f} sigma")

    if abs(residual) < 1.0:
        print("  >>> Resolved within 1 sigma")
    elif abs(residual) < 2.0:
        print("  >>> Within 2 sigma")
    else:
        print("  >>> Over 2 sigma remains")

    print(f"\n  Contribution breakdown (vs observed):")
    print(f"    C15:     {slope_C15/slope_obs*100:.0f}%")
    print(f"    Model B: {slope_ModelB/slope_obs*100:.0f}%")
    print(f"    2-halo:  {slope_2h_contrib/slope_obs*100:.0f}%")
    print(f"    Total:   {total/slope_obs*100:.0f}%")

    print("\n" + "="*70)
    print("Part 5: R-dependence consistency")
    print("="*70)
    for R_test, R_name in [(0.066, 'Inner(66kpc)'), (0.275, 'Mid(275kpc)'),
                            (1.137, 'Outer(1137kpc)')]:
        f2h_test = []
        for _, _, logMs in MSTAR_BINS:
            ds1h = Delta_Sigma_1h(R_test, logMs)
            ds2h = Delta_Sigma_2h(R_test, logMs)
            f2h_test.append(ds2h / (ds1h + ds2h) if (ds1h + ds2h) > 0 else 0)
        f2h_slope = np.polyfit(logMs_arr, np.log10(np.array(f2h_test) + 1e-10), 1)[0]
        print(f"  {R_name:>18s}: f_2h={f2h_test[0]:.3f}-{f2h_test[3]:.3f}, "
              f"slope_contrib~{f2h_slope:+.3f}")

    # figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    ax = axes[0, 0]
    for i, (_, _, logMs) in enumerate(MSTAR_BINS):
        ax.plot(R_plot * 1e3, f2h_all[logMs], color=colors[i], lw=2,
                label=f'logM*={logMs:.1f}')
    for R_kpc, label in [(66, 'Inner'), (275, 'Mid'), (1137, 'Outer')]:
        ax.axvline(R_kpc, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('f_2h')
    ax.set_title('(a) 2-halo fraction vs R')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    logMs_plot = np.linspace(8, 12, 100)
    b_plot = [halo_bias(log_Mhalo(lm)) for lm in logMs_plot]
    ax.plot(logMs_plot, b_plot, 'k-', lw=2)
    for i, (_, _, logMs) in enumerate(MSTAR_BINS):
        b = halo_bias(log_Mhalo(logMs))
        ax.plot(logMs, b, 'o', color=colors[i], ms=10, zorder=5)
    ax.set_xlabel('log10(M*/Msun)')
    ax.set_ylabel('Halo bias b(M)')
    ax.set_title('(b) Halo bias')

    ax = axes[1, 0]
    ax.plot(logMs_arr, gc_no2h, 's--', color='blue', ms=8, label='1-halo only')
    ax.plot(logMs_arr, gc_with2h, 'o-', color='red', ms=8, label='1-halo + 2-halo')
    ax.set_xlabel('log10(M*/Msun)'); ax.set_ylabel('gc / a0')
    ax.set_title(f'(c) 1h slope={slope_no2h:+.3f}, 1h+2h slope={slope_with2h:+.3f}')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    labels = ['C15', 'Model B', '2-halo', 'Total', 'HSC']
    vals = [slope_C15, slope_ModelB, slope_2h_contrib, total, slope_obs]
    errs = [0, 0, 0, 0, slope_obs_err]
    cols = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50', '#F44336']
    ax.bar(range(5), vals, color=cols, alpha=0.8,
           yerr=errs, capsize=5, edgecolor='black', lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('gc-M* slope')
    ax.set_title('(d) Slope budget')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f'{v:+.3f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(slope_obs, color='red', ls=':', lw=1, alpha=0.3)

    plt.tight_layout()
    figpath = os.path.join(BASE, 'two_halo_slope_estimate.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nfig: {figpath}")

    print("\n" + "="*70)
    print("Caveats")
    print("="*70)
    print("  1. SHMR, halo bias, xi_mm are all approximate (10-30% uncertainty)")
    print("  2. SIS 1-halo is simplified (NFW would improve)")
    print("  3. Order-of-magnitude estimate, not precision")
    print("  4. Exact 2-halo measurement requires HSC Phase B decomposition")


if __name__ == '__main__':
    main()
