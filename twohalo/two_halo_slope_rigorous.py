#!/usr/bin/env python3
"""two_halo_slope_rigorous.py — 2-halo slope estimate with colossus"""

import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))

from colossus.cosmology import cosmology as col_cosmo
from colossus.lss import bias as col_bias
from colossus.halo import concentration as col_conc
from colossus.halo import profile_nfw
from colossus.lss import peaks

cosmo = col_cosmo.setCosmology('planck18')

z_lens = 0.2
a0_SI = 1.2e-10
G_SI = 6.674e-11
Msun = 1.989e30
kpc_m = 3.086e19
Mpc_m = 3.086e22

MSTAR_BINS = [
    (8.5, 10.3, 9.4,  'Bin1'),
    (10.3, 10.6, 10.45, 'Bin2'),
    (10.6, 10.8, 10.70, 'Bin3'),
    (10.8, 11.0, 10.90, 'Bin4'),
]


def Mhalo_from_Mstar(logMstar, z=0.2):
    M1 = 10**(11.590 + 1.195 * z / (1+z))
    N0 = 0.0351 - 0.0247 * z / (1+z)
    beta0 = 1.376 - 0.826 * z / (1+z)
    gamma0 = 0.608 + 0.329 * z / (1+z)

    Mstar_target = 10**logMstar

    def mstar_model(logMh):
        Mh = 10**logMh
        ratio = Mh / M1
        return 2.0 * Mh * N0 / (ratio**(-beta0) + ratio**gamma0)

    logMh_lo, logMh_hi = 9.0, 16.0
    for _ in range(100):
        logMh_mid = 0.5 * (logMh_lo + logMh_hi)
        if mstar_model(logMh_mid) < Mstar_target:
            logMh_lo = logMh_mid
        else:
            logMh_hi = logMh_mid
    return logMh_mid


def nfw_esd(R_kpc_arr, logMhalo, z=0.2):
    Mhalo = 10**logMhalo
    c = col_conc.concentration(Mhalo, '200c', z, model='duffy08')
    prof = profile_nfw.NFWProfile(M=Mhalo, c=c, z=z, mdef='200c')

    R_kpc = np.atleast_1d(R_kpc_arr).astype(float)
    R_kpc = np.maximum(R_kpc, 1.0)
    try:
        ds = prof.deltaSigma(R_kpc)
    except Exception:
        ds = np.zeros_like(R_kpc)
    esd_Msun_pc2 = ds * cosmo.h / 1e6
    return esd_Msun_pc2


def twohalo_esd(R_kpc_arr, logMhalo, z=0.2):
    Mhalo = 10**logMhalo
    b = col_bias.haloBias(Mhalo, model='tinker10', z=z, mdef='200c')

    R_kpc = np.atleast_1d(R_kpc_arr).astype(float)
    esd_arr = np.zeros_like(R_kpc)

    for i, Rk in enumerate(R_kpc):
        R_Mpc = Rk / 1e3

        def integrand(l_Mpc):
            r = np.sqrt(R_Mpc**2 + l_Mpc**2)
            if r < 0.01:
                r = 0.01
            r_com = r * (1 + z) * cosmo.h
            try:
                return cosmo.correlationFunction(r_com, z=z)
            except Exception:
                return 0.0

        L_max = 100.0
        result, _ = quad(integrand, -L_max, L_max, limit=200,
                         epsabs=1e-10, epsrel=1e-6)

        rho_m_phys = cosmo.rho_m(z) * (1 + z)**3 / cosmo.h**2
        Sigma_mm = rho_m_phys * result * 1e3
        Sigma_2h = b * Sigma_mm
        esd_arr[i] = Sigma_2h / 1e6
    return esd_arr


def fit_gc(g_bar, g_obs):
    valid = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_obs) & np.isfinite(g_bar)
    if valid.sum() < 3:
        return np.nan
    gb = g_bar[valid]
    go = g_obs[valid]

    def chi2(log_gc):
        gc = 10**log_gc * a0_SI
        x = np.clip(np.sqrt(gb / gc), 0, 50)
        model = gb / (1.0 - np.exp(-x))
        return np.sum((np.log10(go) - np.log10(model))**2)

    result = minimize_scalar(chi2, bounds=(-3, 3), method='bounded')
    return 10**result.x


def main():
    print("="*70)
    print("2-halo slope estimate (colossus rigorous)")
    print("="*70)
    print(f"Cosmology: {cosmo.name}, z_lens={z_lens}")
    print(f"rho_m(z) = {cosmo.rho_m(z_lens):.4e} Msun h^2/kpc^3\n")

    R_kpc = np.logspace(np.log10(30), np.log10(3000), 20)

    print("Part 1: SHMR + Halo Bias")
    print("-"*50)
    print(f"  {'logM*':>8s} {'logM_h':>8s} {'c_200':>8s} {'b(M)':>8s} {'nu':>8s}")
    bin_info = []
    for lo, hi, logMs, label in MSTAR_BINS:
        logMh = Mhalo_from_Mstar(logMs, z=z_lens)
        Mh = 10**logMh
        c = col_conc.concentration(Mh, '200c', z_lens, model='duffy08')
        b = col_bias.haloBias(Mh, model='tinker10', z=z_lens, mdef='200c')
        nu = peaks.peakHeight(Mh, z_lens)
        print(f"  {logMs:8.1f} {logMh:8.2f} {c:8.2f} {b:8.3f} {nu:8.3f}")
        bin_info.append({'logMs': logMs, 'logMh': logMh, 'c': c, 'b': b, 'label': label})

    print(f"\n  b(Bin4)/b(Bin1) = {bin_info[3]['b']/bin_info[0]['b']:.2f}")

    print("\nPart 2: ESD per M* bin")
    print("-"*50)
    esd_1h_all = {}
    esd_2h_all = {}
    f_2h_all = {}

    for info in bin_info:
        logMs = info['logMs']
        logMh = info['logMh']
        label = info['label']
        print(f"\n  {label} (logM*={logMs:.1f}, logM_h={logMh:.2f}):")

        esd_1h = nfw_esd(R_kpc, logMh, z=z_lens)
        esd_1h_all[label] = esd_1h

        print(f"    Computing 2-halo...", end='', flush=True)
        esd_2h = twohalo_esd(R_kpc, logMh, z=z_lens)
        esd_2h_all[label] = esd_2h
        print(" done")

        esd_total = esd_1h + esd_2h
        f_2h = np.where(esd_total > 0, esd_2h / esd_total, 0)
        f_2h_all[label] = f_2h

        for R_show in [50, 200, 500, 1000, 2000]:
            idx = np.argmin(np.abs(R_kpc - R_show))
            print(f"    R={R_kpc[idx]:6.0f}: ESD_1h={esd_1h[idx]:.3e}, "
                  f"ESD_2h={esd_2h[idx]:.3e}, f_2h={f_2h[idx]:.3f}")

    print("\n" + "="*70)
    print("Part 3: gc fits")
    print("-"*50)
    ESD_to_gobs = 5.580e-13

    gc_1h_list = []
    gc_total_list = []
    logMs_list = []

    for info in bin_info:
        logMs = info['logMs']
        label = info['label']
        esd_1h = esd_1h_all[label]
        esd_2h = esd_2h_all[label]
        esd_total = esd_1h + esd_2h

        g_obs_1h = esd_1h * ESD_to_gobs
        g_obs_total = esd_total * ESD_to_gobs

        M_gal = 10**logMs * Msun
        g_bar = G_SI * M_gal / (R_kpc * kpc_m)**2

        gc_1h = fit_gc(g_bar, g_obs_1h)
        gc_tot = fit_gc(g_bar, g_obs_total)

        gc_1h_list.append(gc_1h)
        gc_total_list.append(gc_tot)
        logMs_list.append(logMs)

        if np.isfinite(gc_1h) and gc_1h > 0:
            print(f"  {label}: gc(1h)={gc_1h:.3f}, gc(1h+2h)={gc_tot:.3f}, "
                  f"ratio={gc_tot/gc_1h:.3f}")
        else:
            print(f"  {label}: fit failed")

    gc_1h_arr = np.array(gc_1h_list)
    gc_total_arr = np.array(gc_total_list)
    logMs_arr = np.array(logMs_list)

    valid = (gc_1h_arr > 0) & (gc_total_arr > 0) & np.isfinite(gc_1h_arr) & np.isfinite(gc_total_arr)
    if valid.sum() >= 2:
        slope_1h = np.polyfit(logMs_arr[valid], np.log10(gc_1h_arr[valid]), 1)[0]
        slope_total = np.polyfit(logMs_arr[valid], np.log10(gc_total_arr[valid]), 1)[0]
        slope_2h = slope_total - slope_1h
    else:
        slope_1h = slope_total = slope_2h = float('nan')

    print(f"\n  Slopes:")
    print(f"    1-halo only:         {slope_1h:+.4f}")
    print(f"    1-halo + 2-halo:     {slope_total:+.4f}")
    print(f"    2-halo contribution: {slope_2h:+.4f}")

    print("\n" + "="*70)
    print("Part 4: Slope budget")
    print("="*70)
    slope_C15 = 0.075
    slope_modelB = 0.017
    total_pred = slope_C15 + slope_modelB + slope_2h
    residual = (0.166 - total_pred) / 0.041
    print(f"  C15 (SPARC):         {slope_C15:+.4f}")
    print(f"  Model B (v2):        {slope_modelB:+.4f}")
    print(f"  2-halo (rigorous):   {slope_2h:+.4f}")
    print(f"  Total:               {total_pred:+.4f}")
    print(f"  HSC observed:        +0.166 +/- 0.041")
    print(f"  Residual:            {residual:+.2f} sigma")

    print("\n" + "="*70)
    print("Part 5: R-dependent test")
    print("="*70)
    for R_test, rname in [(66, 'Inner'), (275, 'Mid'), (1137, 'Outer')]:
        idx = np.argmin(np.abs(R_kpc - R_test))
        f2h_vals = [f_2h_all[info['label']][idx] for info in bin_info]
        print(f"  {rname} (R~{R_test} kpc): f_2h = "
              f"[{', '.join(f'{v:.3f}' for v in f2h_vals)}]")
        if all(v > 0 for v in f2h_vals):
            f2h_slope = np.polyfit(logMs_arr, np.log10(f2h_vals), 1)[0]
            print(f"    d log(f_2h)/d logM* = {f2h_slope:+.3f}")

    # figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    ax = axes[0, 0]
    for i, info in enumerate(bin_info):
        label = info['label']
        ax.plot(R_kpc, esd_1h_all[label], '--', color=colors[i], lw=1.5, alpha=0.7)
        ax.plot(R_kpc, esd_1h_all[label] + esd_2h_all[label], '-', color=colors[i],
                lw=2, label=f'{label} (logM*={info["logMs"]:.1f})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('R [kpc]'); ax.set_ylabel('ESD [Msun/pc^2]')
    ax.set_title('(a) ESD: solid=1h+2h, dashed=1h')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, info in enumerate(bin_info):
        ax.plot(R_kpc, f_2h_all[info['label']], color=colors[i], lw=2,
                label=info['label'])
    ax.axvline(66, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axvline(1137, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xscale('log'); ax.set_xlabel('R [kpc]')
    ax.set_ylabel('f_2h'); ax.set_title('(b) 2-halo fraction')
    ax.legend(fontsize=9); ax.set_ylim(0, 1)

    ax = axes[1, 0]
    if valid.sum() >= 2:
        ax.plot(logMs_arr[valid], gc_1h_arr[valid], 's--', color='blue', ms=8,
                label=f'1-halo (slope={slope_1h:+.3f})')
        ax.plot(logMs_arr[valid], gc_total_arr[valid], 'o-', color='red', ms=8,
                label=f'1h+2h (slope={slope_total:+.3f})')
    ax.set_xlabel('log10(M*/Msun)'); ax.set_ylabel('gc / a0')
    ax.set_title(f'(c) 2-halo contrib = {slope_2h:+.4f}')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    labels = ['C15', 'Model B', '2-halo', 'Total', 'HSC']
    vals = [slope_C15, slope_modelB, slope_2h, total_pred, 0.166]
    errs = [0, 0, 0, 0, 0.041]
    cols = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50', '#F44336']
    ax.bar(range(5), vals, color=cols, alpha=0.8,
           yerr=errs, capsize=5, edgecolor='black', lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('gc-M* slope'); ax.set_title('(d) Slope budget (rigorous)')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f'{v:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    figpath = os.path.join(BASE, 'two_halo_slope_rigorous.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nfig: {figpath}")


if __name__ == '__main__':
    main()
