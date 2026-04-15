# -*- coding: utf-8 -*-
"""
phase_a_step5_rar.py

Step 5: HSC ESD -> RAR -> C15 vs MOND comparison.
"""

import os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

OUTPUT_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase_a_output")
BROUWER_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\brouwer2021")

a0 = 1.2e-10
pc_m = 3.0857e16
Mpc_pc = 1e6
G_SI = 6.674e-11
Msun_kg = 1.989e30

# g_obs [m/s^2] = 4 * G_SI * ESD[Msun/pc^2] * Msun_kg / pc_m^2
# Brouwer+2021 Eq.7 coefficient: ~5.58e-13 per unit (h70 Msun/pc^2)
GOBS_FACTOR = 4 * G_SI * Msun_kg / pc_m**2

MSTAR_BINS = [8.5, 10.3, 10.6, 10.8, 11.0, 11.5]
N_MBINS = len(MSTAR_BINS) - 1
MSTAR_BIN_CENTERS = [0.5 * (MSTAR_BINS[i] + MSTAR_BINS[i+1]) for i in range(N_MBINS)]

BIAS_HSC = 1.0


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def load_esd_hsc(filename):
    data = np.loadtxt(OUTPUT_DIR / filename, comments='#')
    return {
        'R_Mpc': data[:, 0],
        'ESD_t': data[:, 1],
        'ESD_x': data[:, 2],
        'N_pairs': data[:, 3].astype(int),
    }


def load_esd_brouwer(filename):
    filepath = BROUWER_DIR / filename
    if not filepath.exists():
        return None
    data = np.loadtxt(filepath, comments='#')
    return {
        'g_bar': data[:, 0],
        'ESD_t': data[:, 1],
        'ESD_x': data[:, 2],
        'error': data[:, 3],
        'bias':  data[:, 4],
    }


def estimate_gbar_sis(R_Mpc, logMstar):
    M_star = 10**logMstar
    M_bar = 1.4 * M_star
    R_pc = R_Mpc * Mpc_pc
    R_m = R_pc * pc_m
    return G_SI * M_bar * Msun_kg / R_m**2


def estimate_gbar_per_bin(R_Mpc_arr, logMstar_median):
    return np.array([estimate_gbar_sis(R, logMstar_median) for R in R_Mpc_arr])


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4 * gc * g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def c15_gc(logMstar, Yd=0.5):
    import math
    kpc_m = 3.0857e19
    Mbar = 10**logMstar * 1.4
    vflat = (G_SI * a0 * Mbar * Msun_kg)**0.25 / 1e3
    hR = 10**(0.35 * (logMstar - 10.0) + 0.5)
    return 0.584 * Yd**(-0.361) * math.sqrt(a0 * (vflat * 1e3)**2 / (hR * kpc_m))


def fit_gc(g_bar, g_obs, g_err=None):
    if g_err is None:
        g_err = np.ones_like(g_obs) * 0.1 * g_obs

    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(g_bar, gc)
        valid = (gp > 0) & (g_obs > 0)
        if valid.sum() < 3:
            return 1e30
        res = np.log10(g_obs[valid]) - np.log10(gp[valid])
        w = 1.0 / np.maximum((g_err[valid] / (g_obs[valid] * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)

    result = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**result.x


def main():
    section("STEP 5: HSC LENSING RAR + C15 vs MOND")

    section("Loading HSC ESD profiles")
    hsc_all = load_esd_hsc('esd_profile_all.txt')
    print(f"  All lenses: {len(hsc_all['R_Mpc'])} radial bins, "
          f"{hsc_all['N_pairs'].sum():,} total pairs")

    hsc_mbins = []
    for mb in range(N_MBINS):
        fname = f'esd_profile_mbin_{mb+1}.txt'
        if (OUTPUT_DIR / fname).exists():
            d = load_esd_hsc(fname)
            hsc_mbins.append(d)
            print(f"  M* bin {mb+1}: {d['N_pairs'].sum():,} pairs")
        else:
            hsc_mbins.append(None)

    section("Loading Brouwer+2021 KiDS ESD")
    brouwer_all = load_esd_brouwer('Fig-4-5-C1_KiDS-isolated_Nobins.txt')
    if brouwer_all:
        print(f"  Brouwer all: {len(brouwer_all['g_bar'])} points")

    brouwer_mbins = []
    for mb in range(4):
        fname = f'Fig-9_KiDS-isolated_Massbin-{mb+1}.txt'
        d = load_esd_brouwer(fname)
        brouwer_mbins.append(d)
        if d:
            print(f"  Brouwer bin {mb+1}: {len(d['g_bar'])} points")

    section("HSC ESD -> RAR conversion")
    rar_results = []

    for mb in range(N_MBINS):
        if hsc_mbins[mb] is None:
            continue

        esd = hsc_mbins[mb]
        logM_med = MSTAR_BIN_CENTERS[mb]

        g_obs = GOBS_FACTOR * esd['ESD_t'] / BIAS_HSC
        g_bar = estimate_gbar_per_bin(esd['R_Mpc'], logM_med)

        n_pairs = np.maximum(esd['N_pairs'], 1)
        esd_err = np.abs(esd['ESD_t']) / np.sqrt(n_pairs) * 2.0
        g_err = GOBS_FACTOR * esd_err / BIAS_HSC

        valid = (g_obs > 0) & (g_bar > 0) & np.isfinite(g_obs) & np.isfinite(g_bar)

        if valid.sum() < 3:
            print(f"\n  M* bin {mb+1}: too few valid points")
            continue

        gc_fit = fit_gc(g_bar[valid], g_obs[valid], g_err[valid])
        gc_c15 = c15_gc(logM_med)

        g_mond = mond_rar(g_bar[valid])
        g_memb = membrane_rar(g_bar[valid], gc_fit)
        g_c15 = membrane_rar(g_bar[valid], gc_c15)

        res_mond = np.log10(g_obs[valid]) - np.log10(g_mond)
        res_memb = np.log10(g_obs[valid]) - np.log10(g_memb)
        res_c15 = np.log10(g_obs[valid]) - np.log10(g_c15)

        w = 1.0 / np.maximum((g_err[valid] / (g_obs[valid] * np.log(10)))**2, 1e-10)
        chi2_mond = np.sum(w * res_mond**2)
        chi2_memb = np.sum(w * res_memb**2)
        chi2_c15 = np.sum(w * res_c15**2)
        dof = valid.sum()

        result = {
            'mbin': mb + 1, 'logM': logM_med,
            'gc_fit': gc_fit / a0, 'gc_c15': gc_c15 / a0,
            'chi2_mond': chi2_mond, 'chi2_c15': chi2_c15, 'chi2_fit': chi2_memb,
            'dof': dof,
            'scatter_mond': np.std(res_mond), 'scatter_c15': np.std(res_c15),
            'g_bar': g_bar, 'g_obs': g_obs, 'g_err': g_err,
            'valid': valid,
        }
        rar_results.append(result)

        print(f"\n  M* bin {mb+1} (logM*={logM_med:.2f}):")
        print(f"    gc_fit = {gc_fit/a0:.3f} a0")
        print(f"    gc_C15 = {gc_c15/a0:.3f} a0  (ratio fit/C15 = {gc_fit/gc_c15:.2f})")
        print(f"    chi2(MOND) = {chi2_mond:.1f}  (dof={dof})")
        print(f"    chi2(C15)  = {chi2_c15:.1f}")
        print(f"    chi2(fit)  = {chi2_memb:.1f}")
        print(f"    Delta-chi2 (MOND - C15) = {chi2_mond - chi2_c15:+.1f}")

    section("RAR SUMMARY TABLE")
    if rar_results:
        print(f"\n  {'Bin':>4s} {'logM*':>6s} {'gc_fit/a0':>10s} {'gc_C15/a0':>10s} "
              f"{'ratio':>8s} {'chi2_M':>8s} {'chi2_C':>8s} {'Dchi2':>8s} {'dof':>5s}")
        for r in rar_results:
            dchi = r['chi2_mond'] - r['chi2_c15']
            print(f"  {r['mbin']:>4d} {r['logM']:6.2f} {r['gc_fit']:10.3f} "
                  f"{r['gc_c15']:10.3f} {r['gc_fit']/r['gc_c15']:8.2f} "
                  f"{r['chi2_mond']:8.1f} {r['chi2_c15']:8.1f} "
                  f"{dchi:+8.1f} {r['dof']:>5d}")

        logM_arr = np.array([r['logM'] for r in rar_results])
        log_gc_fit = np.log10([r['gc_fit'] for r in rar_results])
        log_gc_c15 = np.log10([r['gc_c15'] for r in rar_results])

        if len(logM_arr) >= 3:
            slope_fit = np.polyfit(logM_arr, log_gc_fit, 1)[0]
            slope_c15 = np.polyfit(logM_arr, log_gc_c15, 1)[0]
            print(f"\n  gc-M* slopes:")
            print(f"    HSC observed:  {slope_fit:+.3f}")
            print(f"    C15 predicted: {slope_c15:+.3f}")
            print(f"    MOND expected: 0.000")

    if brouwer_mbins and any(b is not None for b in brouwer_mbins):
        section("HSC vs KiDS (Brouwer) COMPARISON")
        for mb in range(min(N_MBINS, 4)):
            if hsc_mbins[mb] is None or brouwer_mbins[mb] is None:
                continue
            hsc_esd = hsc_mbins[mb]['ESD_t']
            kids_esd = brouwer_mbins[mb]['ESD_t']
            kids_bias = brouwer_mbins[mb]['bias']
            hsc_mean = np.mean(hsc_esd[hsc_esd > 0])
            kids_mean = np.mean(kids_esd / kids_bias)
            print(f"\n  Bin {mb+1}: <ESD_t>_HSC = {hsc_mean:.2f}, "
                  f"<ESD_t>_KiDS = {kids_mean:.2f}, "
                  f"ratio = {hsc_mean/kids_mean:.2f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(18, 11))

        ax = axs[0, 0]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        for r in rar_results:
            mb = r['mbin'] - 1
            v = r['valid']
            ax.loglog(r['g_bar'][v], r['g_obs'][v], 'o', ms=4,
                      color=colors[mb % len(colors)],
                      label=f"Bin {r['mbin']} (logM*={r['logM']:.1f})")
        gb_line = np.logspace(-16, -10, 100)
        ax.loglog(gb_line, mond_rar(gb_line), 'k--', lw=1.5, label='MOND')
        ax.loglog(gb_line, gb_line, 'k:', lw=0.5, alpha=0.3, label='g_obs=g_bar')
        ax.set_xlabel('g_bar (m/s^2)')
        ax.set_ylabel('g_obs (m/s^2)')
        ax.set_title('HSC Lensing RAR (per M* bin)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        ax = axs[0, 1]
        if rar_results:
            lm = [r['logM'] for r in rar_results]
            gc_fit = [r['gc_fit'] for r in rar_results]
            gc_c15 = [r['gc_c15'] for r in rar_results]
            ax.semilogy(lm, gc_fit, 'rs-', ms=8, lw=2, label='HSC gc (fit)')
            ax.semilogy(lm, gc_c15, 'b^--', ms=8, lw=1.5, label='C15 prediction')
            ax.axhline(1.0, color='gray', ls=':', lw=1, label='MOND (gc=a0)')
            ax.set_xlabel('log10(M*/Msun)')
            ax.set_ylabel('gc / a0')
            ax.set_title('gc per M* bin: HSC vs C15')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axs[0, 2]
        if rar_results:
            bins_x = [r['mbin'] for r in rar_results]
            dchi = [r['chi2_mond'] - r['chi2_c15'] for r in rar_results]
            ax.bar(bins_x, dchi, color=['green' if d > 0 else 'red' for d in dchi])
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('M* bin')
            ax.set_ylabel('Delta-chi2 (MOND - C15)')
            ax.set_title('+ve = C15 preferred')
            ax.grid(True, alpha=0.3)

        for pi, r in enumerate(rar_results[:3]):
            ax = axs[1, pi]
            v = r['valid']
            gb = r['g_bar'][v]
            go = r['g_obs'][v]
            ge = r['g_err'][v]
            ax.errorbar(gb, go, yerr=ge, fmt='ko', ms=4, capsize=2, label='HSC data')
            gb_line = np.logspace(np.log10(gb.min()*0.5),
                                  np.log10(gb.max()*2), 50)
            ax.loglog(gb_line, mond_rar(gb_line), 'b--', lw=1.5, label='MOND')
            ax.loglog(gb_line, membrane_rar(gb_line, r['gc_fit'] * a0),
                      'r-', lw=1.5, label=f"Memb (gc={r['gc_fit']:.2f}a0)")
            ax.loglog(gb_line, membrane_rar(gb_line, r['gc_c15'] * a0),
                      'g:', lw=1.5, label=f"C15 (gc={r['gc_c15']:.2f}a0)")
            ax.loglog(gb_line, gb_line, 'k:', lw=0.5, alpha=0.3)
            ax.set_xlabel('g_bar')
            ax.set_ylabel('g_obs')
            ax.set_title(f"Bin {r['mbin']} (logM*={r['logM']:.1f})")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out_png = OUTPUT_DIR / 'hsc_lensing_rar.png'
        plt.savefig(out_png, dpi=120)
        print(f"\n  Plot saved: {out_png}")

    except Exception as e:
        print(f"\n  [Plot error: {e}]")

    section("Saving RAR data")
    for r in rar_results:
        v = r['valid']
        out = np.column_stack([r['g_bar'][v], r['g_obs'][v], r['g_err'][v]])
        fname = OUTPUT_DIR / f"hsc_rar_mbin_{r['mbin']}.txt"
        np.savetxt(fname, out,
                   header=f"g_bar(m/s2)  g_obs(m/s2)  g_err  "
                          f"[logM*={r['logM']:.2f}, gc_fit={r['gc_fit']:.3f}a0]",
                   fmt='%.6e')
        print(f"  Saved: {fname}")

    section("STEP 5 COMPLETE")


if __name__ == '__main__':
    main()
