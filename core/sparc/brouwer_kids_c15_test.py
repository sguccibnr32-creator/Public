# -*- coding: utf-8 -*-
"""
brouwer_kids_c15_test.py
Brouwer+2021 KiDS-1000 Lensing RAR: C15 Verification
"""

import os, sys, math
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

G_SI = 6.674e-11
a0 = 1.2e-10
Msun = 1.989e30
pc_m = 3.0857e16
kpc_m = 3.0857e19
Mpc_m = 3.0857e22
h70 = 1.0

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

MSTAR_BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
MSTAR_BIN_LOG_CENTERS = [0.5*(MSTAR_BIN_EDGES[i]+MSTAR_BIN_EDGES[i+1])
                          for i in range(4)]
MSTAR_BIN_LABELS = [
    f"Bin {i+1}: log M* = [{MSTAR_BIN_EDGES[i]:.1f}, {MSTAR_BIN_EDGES[i+1]:.1f}]"
    for i in range(4)
]


def estimate_vflat(log_mstar):
    Mbar = 10**log_mstar * 1.4
    return (G_SI * a0 * Mbar * Msun)**0.25 / 1e3


def estimate_hR(log_mstar):
    return 10**(0.35*(log_mstar - 10.0) + 0.5)


def c15_predict_gc(log_mstar, Yd=0.5):
    vflat = estimate_vflat(log_mstar)
    hR = estimate_hR(log_mstar)
    vflat_m = vflat * 1e3
    hR_m = hR * kpc_m
    Sigma_dyn = vflat_m**2 / hR_m
    return 0.584 * Yd**(-0.361) * math.sqrt(a0 * Sigma_dyn)


def load_esd_file(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'col1':  data[:, 0],
        'ESD_t': data[:, 1],
        'ESD_x': data[:, 2],
        'error': data[:, 3],
        'bias':  data[:, 4],
        'n_points': len(data),
    }


def esd_to_gobs(ESD_t, bias):
    return 5.580e-13 * ESD_t / bias


def esd_error_to_gobs_error(error, bias):
    return 5.580e-13 * error / bias


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4*gc*g_bar)) / 2


def fit_gc_to_rar(g_bar, g_obs, g_obs_err):
    def chi2(log_gc):
        gc = 10**log_gc
        g_pred = membrane_rar(g_bar, gc)
        res = np.log10(g_obs) - np.log10(g_pred)
        w = 1.0 / (g_obs_err / (g_obs * np.log(10)))**2
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        return np.sum(w * res**2)
    result = minimize_scalar(chi2, bounds=(-14, -9), method='bounded')
    gc_best = 10**result.x
    chi2_min = result.fun
    try:
        from scipy.optimize import brentq
        dlg_up = brentq(lambda dlg: chi2(result.x + dlg) - chi2_min - 1.0, 1e-4, 2)
        dlg_dn = brentq(lambda dlg: chi2(result.x - dlg) - chi2_min - 1.0, 1e-4, 2)
        gc_err = gc_best * (10**((dlg_up + dlg_dn)/2) - 1)
    except:
        gc_err = gc_best * 0.5
    return gc_best, gc_err, chi2_min


def main():
    print("=" * 70)
    print("Brouwer+2021 KiDS-1000 Lensing RAR: C15 Verification")
    print("=" * 70)

    main_file = BASE / "Fig-4-5-C1_KiDS-isolated_Nobins.txt"
    if not main_file.exists():
        print(f"[ERROR] Main RAR file not found: {main_file}")
        return

    print("\n--- T1: Main RAR (all M* combined) ---")
    d = load_esd_file(main_file)
    g_bar = d['col1']
    g_obs = esd_to_gobs(d['ESD_t'], d['bias'])
    g_err = esd_error_to_gobs_error(d['error'], d['bias'])

    valid = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_obs) & np.isfinite(g_err) & (g_err > 0)
    g_bar, g_obs, g_err = g_bar[valid], g_obs[valid], g_err[valid]

    print(f"Data points (valid): {len(g_bar)}/{d['n_points']}")
    print(f"g_bar range: [{g_bar.min():.2e}, {g_bar.max():.2e}] m/s^2")
    print(f"g_bar/a0 range: [{g_bar.min()/a0:.2e}, {g_bar.max()/a0:.2e}]")
    print(f"g_obs range: [{g_obs.min():.2e}, {g_obs.max():.2e}] m/s^2")

    gc_all, gc_all_err, chi2_all = fit_gc_to_rar(g_bar, g_obs, g_err)
    print(f"\nFitted gc (all M*): {gc_all:.3e} m/s^2 = {gc_all/a0:.4f} a0")
    print(f"SPARC median gc: 0.24 a0 = {0.24*a0:.3e} m/s^2")

    g_mond = mond_rar(g_bar)
    g_memb = membrane_rar(g_bar, gc_all)
    res_mond = np.log10(g_obs / g_mond)
    res_memb = np.log10(g_obs / g_memb)
    print(f"\nResiduals (log10 g_obs/g_pred):")
    print(f"  MOND:     median = {np.median(res_mond):+.4f}, std = {np.std(res_mond):.4f}")
    print(f"  Membrane: median = {np.median(res_memb):+.4f}, std = {np.std(res_memb):.4f}")

    print("\n" + "=" * 70)
    print("T2: M* BIN gc FITTING (C15 Yd^{-0.36} TEST)")
    print("=" * 70)

    bin_results = []
    for i in range(4):
        fname = BASE / f"Fig-9_KiDS-isolated_Massbin-{i+1}.txt"
        if not fname.exists():
            print(f"  [SKIP] {fname.name} not found")
            continue
        d = load_esd_file(fname)
        gb = d['col1']
        go = esd_to_gobs(d['ESD_t'], d['bias'])
        ge = esd_error_to_gobs_error(d['error'], d['bias'])
        valid = (gb > 0) & (go > 0) & np.isfinite(go) & np.isfinite(ge) & (ge > 0)
        gb, go, ge = gb[valid], go[valid], ge[valid]
        if len(gb) < 3:
            print(f"  Bin {i+1}: too few points ({len(gb)})")
            continue
        gc_fit, gc_err, chi2_fit = fit_gc_to_rar(gb, go, ge)
        log_mstar_center = MSTAR_BIN_LOG_CENTERS[i]
        gc_c15 = c15_predict_gc(log_mstar_center)
        vflat_est = estimate_vflat(log_mstar_center)
        hR_est = estimate_hR(log_mstar_center)
        ratio = gc_fit / gc_c15
        bin_results.append({
            'bin': i+1, 'log_mstar': log_mstar_center,
            'gc_fit': gc_fit, 'gc_fit_a0': gc_fit/a0,
            'gc_c15': gc_c15, 'gc_c15_a0': gc_c15/a0,
            'ratio': ratio, 'n_pts': len(gb),
            'vflat_est': vflat_est, 'hR_est': hR_est,
            'chi2': chi2_fit,
        })
        print(f"\n  Bin {i+1}: {MSTAR_BIN_LABELS[i]}")
        print(f"    N points: {len(gb)}")
        print(f"    g_bar range: [{gb.min():.2e}, {gb.max():.2e}] m/s^2")
        print(f"    gc_fit = {gc_fit:.3e} m/s^2 = {gc_fit/a0:.4f} a0")
        print(f"    gc_C15 = {gc_c15:.3e} m/s^2 = {gc_c15/a0:.4f} a0")
        print(f"    ratio gc_fit/gc_C15 = {ratio:.3f}")
        print(f"    (vflat_est = {vflat_est:.1f} km/s, hR_est = {hR_est:.2f} kpc)")

    if len(bin_results) >= 3:
        print("\n" + "=" * 70)
        print("T3: gc SCALING WITH M*")
        print("=" * 70)
        log_mstar = np.array([r['log_mstar'] for r in bin_results])
        log_gc_fit = np.log10([r['gc_fit'] for r in bin_results])
        log_gc_c15 = np.log10([r['gc_c15'] for r in bin_results])
        coeffs_fit = np.polyfit(log_mstar, log_gc_fit, 1)
        coeffs_c15 = np.polyfit(log_mstar, log_gc_c15, 1)
        print(f"\nlog gc vs log M* slope:")
        print(f"  Observed (lensing fit): {coeffs_fit[0]:+.3f}")
        print(f"  C15 prediction:         {coeffs_c15[0]:+.3f}")
        print(f"  MOND prediction:        0.000 (gc = const)")
        if abs(coeffs_fit[0]) > 0.05:
            match = "MATCHES" if coeffs_fit[0]*coeffs_c15[0] > 0 else "CONTRADICTS"
            print(f"\n  -> gc varies with M*: MOND violated")
            print(f"  -> C15 slope direction {match} observation")
        else:
            print(f"\n  -> gc approximately constant: consistent with MOND")
        gc_fits = np.array([r['gc_fit'] for r in bin_results])
        gc_c15s = np.array([r['gc_c15'] for r in bin_results])
        corr = np.corrcoef(np.log10(gc_fits), np.log10(gc_c15s))[0, 1]
        print(f"\nCorrelation log(gc_fit) vs log(gc_C15): r = {corr:.3f}")
        print(f"\n{'Bin':>4s} {'logM*':>6s} {'gc_fit/a0':>10s} {'gc_C15/a0':>10s} "
              f"{'ratio':>7s} {'N':>4s}")
        for r in bin_results:
            print(f"  {r['bin']:>2d}  {r['log_mstar']:6.2f}  {r['gc_fit_a0']:10.4f}  "
                  f"{r['gc_c15_a0']:10.4f}  {r['ratio']:7.3f}  {r['n_pts']:4d}")

    dwarf_file = BASE / "Fig-10_KiDS-isolated-dwarfs_Nobins.txt"
    if dwarf_file.exists():
        print("\n" + "=" * 70)
        print("T4: DWARF SUBSAMPLE")
        print("=" * 70)
        d = load_esd_file(dwarf_file)
        gb = d['col1']
        go = esd_to_gobs(d['ESD_t'], d['bias'])
        ge = esd_error_to_gobs_error(d['error'], d['bias'])
        valid = (gb > 0) & (go > 0) & np.isfinite(go) & np.isfinite(ge) & (ge > 0)
        gb, go, ge = gb[valid], go[valid], ge[valid]
        if len(gb) >= 3:
            gc_dwarf, _, _ = fit_gc_to_rar(gb, go, ge)
            print(f"Dwarf gc = {gc_dwarf:.3e} m/s^2 = {gc_dwarf/a0:.4f} a0")
            gc_c15_dwarf = c15_predict_gc(9.0)
            print(f"C15 (logM*=9): {gc_c15_dwarf:.3e} = {gc_c15_dwarf/a0:.4f} a0")
            print(f"Ratio: {gc_dwarf/gc_c15_dwarf:.3f}")

    gama_file = BASE / "Fig-4-C1_GAMA-isolated_Nobins.txt"
    if gama_file.exists():
        print("\n--- T5: GAMA cross-check ---")
        d = load_esd_file(gama_file)
        gb = d['col1']
        go = esd_to_gobs(d['ESD_t'], d['bias'])
        ge = esd_error_to_gobs_error(d['error'], d['bias'])
        valid = (gb > 0) & (go > 0) & np.isfinite(go) & np.isfinite(ge) & (ge > 0)
        gb, go, ge = gb[valid], go[valid], ge[valid]
        if len(gb) >= 3:
            gc_gama, _, _ = fit_gc_to_rar(gb, go, ge)
            print(f"GAMA gc = {gc_gama:.3e} m/s^2 = {gc_gama/a0:.4f} a0")
            print(f"KiDS gc = {gc_all:.3e} m/s^2 = {gc_all/a0:.4f} a0")
            print(f"Ratio GAMA/KiDS: {gc_gama/gc_all:.3f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        ax = axs[0]
        d = load_esd_file(main_file)
        gb_m = d['col1']
        go_m = esd_to_gobs(d['ESD_t'], d['bias'])
        ge_m = esd_error_to_gobs_error(d['error'], d['bias'])
        v = (gb_m > 0) & (go_m > 0) & np.isfinite(go_m) & np.isfinite(ge_m)
        ax.errorbar(gb_m[v]/a0, go_m[v]/a0, yerr=ge_m[v]/a0, fmt='ko', ms=5,
                    capsize=2, label=f'KiDS all (gc={gc_all/a0:.3f} a0)')
        colors = ['blue', 'green', 'orange', 'red']
        for i in range(4):
            fname = BASE / f"Fig-9_KiDS-isolated_Massbin-{i+1}.txt"
            if fname.exists():
                d = load_esd_file(fname)
                gb_i = d['col1']
                go_i = esd_to_gobs(d['ESD_t'], d['bias'])
                vi = (gb_i > 0) & (go_i > 0) & np.isfinite(go_i)
                ax.plot(gb_i[vi]/a0, go_i[vi]/a0, 's', color=colors[i], ms=4,
                        label=f'Bin {i+1} (logM*={MSTAR_BIN_LOG_CENTERS[i]:.1f})')
        gb_range = np.logspace(-6, 0, 200) * a0
        ax.loglog(gb_range/a0, mond_rar(gb_range)/a0, 'g-', lw=1.5, label='MOND (gc=a0)')
        ax.loglog(gb_range/a0, membrane_rar(gb_range, gc_all)/a0, 'r--', lw=1.5,
                  label=f'Membrane (gc={gc_all/a0:.3f} a0)')
        ax.loglog(gb_range/a0, gb_range/a0, 'k:', lw=0.7, alpha=0.5, label='1:1')
        ax.set_xlabel('g_bar / a0')
        ax.set_ylabel('g_obs / a0')
        ax.set_title('KiDS-1000 Lensing RAR + C15')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(1e-3, 10)
        ax.grid(True, which='both', alpha=0.2)

        ax = axs[1]
        if len(bin_results) >= 2:
            lm = [r['log_mstar'] for r in bin_results]
            gc_f = [r['gc_fit_a0'] for r in bin_results]
            gc_c = [r['gc_c15_a0'] for r in bin_results]
            ax.semilogy(lm, gc_f, 'ro-', ms=8, lw=2, label='gc from lensing fit')
            ax.semilogy(lm, gc_c, 'b^--', ms=8, lw=1.5, label='gc from C15')
            ax.axhline(1.0, color='g', ls=':', lw=1, label='MOND (gc=a0)')
            ax.axhline(0.24, color='gray', ls=':', lw=1, label='SPARC median')
            ax.set_xlabel('log10(M*/Msun)')
            ax.set_ylabel('gc / a0')
            ax.set_title('gc vs Stellar Mass: C15 Test')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = BASE / "brouwer_kids_c15_test.png"
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved: {out_png}")
    except Exception as e:
        print(f"\n[Plot error: {e}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Main RAR: gc_all = {gc_all/a0:.4f} a0")
    if bin_results:
        gc_range = [r['gc_fit_a0'] for r in bin_results]
        print(f"M* bins: gc range [{min(gc_range):.4f}, {max(gc_range):.4f}] a0")
        if max(gc_range)/min(gc_range) > 1.5:
            print(f"  -> gc varies by {max(gc_range)/min(gc_range):.1f}x")
            print(f"  -> MOND (gc=const) VIOLATED")
        else:
            print(f"  -> gc approximately constant")
    print(f"\nC15 prediction quality:")
    for r in bin_results:
        status = "OK" if 0.3 < r['ratio'] < 3.0 else "OUTLIER"
        print(f"  Bin {r['bin']}: ratio = {r['ratio']:.2f} [{status}]")


if __name__ == '__main__':
    main()
