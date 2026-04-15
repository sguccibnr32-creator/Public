# -*- coding: utf-8 -*-
"""
brouwer_kids_c15_hotgas_per_mbin.py

Estimate hot-gas-corrected gc per M* bin using Fig-4 hotgas/no-hg pair
as universal correction factor applied to Fig-9 Massbin data.

CAVEATS:
  - Assumes hot-gas correction is M*-independent (only g_bar-dependent)
  - In reality, more massive halos have proportionally more hot gas,
    so this approximation under-corrects high-M* bins and over-corrects
    low-M* bins
  - Provides first-order estimate, not Brouwer-pipeline equivalent
"""

import os, math
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

a0 = 1.2e-10
G_SI = 6.674e-11
Msun = 1.989e30
kpc_m = 3.0857e19

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

MSTAR_BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
MSTAR_BIN_LOG_CENTERS = [0.5*(MSTAR_BIN_EDGES[i]+MSTAR_BIN_EDGES[i+1]) for i in range(4)]


def load_esd(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'g_bar': data[:, 0], 'ESD_t': data[:, 1],
        'error': data[:, 3], 'bias':  data[:, 4],
    }


def esd_to_gobs(ESD_t, bias):
    return 5.580e-13 * ESD_t / bias


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4*gc*g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def fit_gc(g_bar, g_obs, g_err):
    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(g_bar, gc)
        res = np.log10(g_obs) - np.log10(gp)
        w = 1.0 / np.maximum((g_err / (g_obs * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)
    res = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**res.x


def estimate_vflat(log_mstar):
    Mbar = 10**log_mstar * 1.4
    return (G_SI * a0 * Mbar * Msun)**0.25 / 1e3


def estimate_hR(log_mstar):
    return 10**(0.35*(log_mstar - 10.0) + 0.5)


def c15_predict_gc(log_mstar, Yd=0.5):
    vflat = estimate_vflat(log_mstar)
    hR = estimate_hR(log_mstar)
    return 0.584 * Yd**(-0.361) * math.sqrt(a0 * (vflat*1e3)**2 / (hR * kpc_m))


def main():
    print("=" * 72)
    print("HOT GAS CORRECTION PER M* BIN (proxy method)")
    print("=" * 72)

    # ---------- Build correction factor from Fig-4 pair ----------
    f_nohg = BASE / "Fig-4-5-C1_KiDS-isolated_Nobins.txt"
    f_hg   = BASE / "Fig-4_KiDS-isolated_hotgas_Nobins.txt"
    d_nohg = load_esd(f_nohg)
    d_hg   = load_esd(f_hg)

    assert np.allclose(d_nohg['g_bar'], d_hg['g_bar']), "g_bar bins differ!"
    g_bar_grid = d_nohg['g_bar']
    correction_factor = d_hg['ESD_t'] / d_nohg['ESD_t']

    print("\nUniversal hot-gas correction factor C(g_bar) = ESD_hg / ESD_nohg:")
    print(f"{'g_bar/a0':>12s}  {'ESD_nohg':>10s}  {'ESD_hg':>10s}  {'C factor':>10s}")
    for i in range(len(g_bar_grid)):
        print(f"  {g_bar_grid[i]/a0:>10.2e}    {d_nohg['ESD_t'][i]:10.3f}  "
              f"{d_hg['ESD_t'][i]:10.3f}  {correction_factor[i]:10.3f}")

    # ---------- Apply to each M* bin ----------
    print("\n" + "=" * 72)
    print("PER-BIN gc FITTING (no_hg vs hot_gas corrected)")
    print("=" * 72)

    bin_results = []
    for i in range(4):
        fname = BASE / f"Fig-9_KiDS-isolated_Massbin-{i+1}.txt"
        if not fname.exists():
            print(f"  Bin {i+1}: file not found")
            continue

        d = load_esd(fname)
        gb_bin = d['g_bar']
        ESD_orig = d['ESD_t']
        bias = d['bias']
        err = d['error']

        # Match g_bar grid
        if not np.allclose(gb_bin, g_bar_grid):
            # Interpolate correction in log-g_bar space
            from scipy.interpolate import interp1d
            log_C_interp = interp1d(
                np.log10(g_bar_grid), np.log10(correction_factor),
                bounds_error=False, fill_value='extrapolate')
            C_bin = 10**log_C_interp(np.log10(gb_bin))
        else:
            C_bin = correction_factor.copy()

        # No-hg fit
        go_nohg = esd_to_gobs(ESD_orig, bias)
        ge_nohg = esd_to_gobs(err, bias)
        valid_n = (gb_bin > 0) & (go_nohg > 0) & np.isfinite(go_nohg) & (ge_nohg > 0)
        gc_nohg = fit_gc(gb_bin[valid_n], go_nohg[valid_n], ge_nohg[valid_n])

        # HG-corrected fit
        ESD_corr = ESD_orig * C_bin
        err_corr = err * C_bin   # propagate (multiplicative scaling)
        go_hg = esd_to_gobs(ESD_corr, bias)
        ge_hg = esd_to_gobs(err_corr, bias)
        valid_h = (gb_bin > 0) & (go_hg > 0) & np.isfinite(go_hg) & (ge_hg > 0)
        gc_hg = fit_gc(gb_bin[valid_h], go_hg[valid_h], ge_hg[valid_h])

        # MOND residuals (HG-corrected)
        res_mond_hg = np.log10(go_hg[valid_h] / mond_rar(gb_bin[valid_h]))
        res_memb_hg = np.log10(go_hg[valid_h] / membrane_rar(gb_bin[valid_h], gc_hg))

        # C15 prediction
        gc_c15 = c15_predict_gc(MSTAR_BIN_LOG_CENTERS[i])

        bin_results.append({
            'bin': i+1, 'logM': MSTAR_BIN_LOG_CENTERS[i],
            'gc_nohg': gc_nohg/a0, 'gc_hg': gc_hg/a0,
            'gc_c15': gc_c15/a0,
            'reduction': gc_nohg/gc_hg,
            'mond_med': float(np.median(res_mond_hg)),
            'mond_std': float(np.std(res_mond_hg)),
            'memb_med': float(np.median(res_memb_hg)),
            'memb_std': float(np.std(res_memb_hg)),
        })

        print(f"\n  Bin {i+1} (logM* = {MSTAR_BIN_LOG_CENTERS[i]:.1f}):")
        print(f"    gc_nohg = {gc_nohg/a0:.3f} a0")
        print(f"    gc_hg   = {gc_hg/a0:.3f} a0  (reduction factor {gc_nohg/gc_hg:.2f}x)")
        print(f"    gc_C15  = {gc_c15/a0:.3f} a0  (prediction)")
        print(f"    HG-corrected residuals:")
        print(f"      MOND:     median={np.median(res_mond_hg):+.4f}, std={np.std(res_mond_hg):.4f}")
        print(f"      Membrane: median={np.median(res_memb_hg):+.4f}, std={np.std(res_memb_hg):.4f}")

    # ---------- Summary ----------
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(f"{'Bin':>4s} {'logM*':>6s} {'gc_nohg/a0':>11s} {'gc_hg/a0':>10s} "
          f"{'reduction':>10s} {'gc_C15/a0':>10s} {'gc_hg/gc_C15':>13s}")
    for r in bin_results:
        ratio_c15 = r['gc_hg'] / r['gc_c15']
        print(f"  {r['bin']:>2d}  {r['logM']:6.2f}  {r['gc_nohg']:11.3f}  "
              f"{r['gc_hg']:10.3f}  {r['reduction']:10.2f}  "
              f"{r['gc_c15']:10.3f}  {ratio_c15:13.2f}")

    # ---------- gc-M* scaling ----------
    if len(bin_results) >= 3:
        print("\n--- gc vs M* slopes ---")
        logM = np.array([r['logM'] for r in bin_results])
        log_gc_nohg = np.log10([r['gc_nohg'] for r in bin_results])
        log_gc_hg   = np.log10([r['gc_hg']   for r in bin_results])
        log_gc_c15  = np.log10([r['gc_c15']  for r in bin_results])

        s_nohg = np.polyfit(logM, log_gc_nohg, 1)[0]
        s_hg   = np.polyfit(logM, log_gc_hg,   1)[0]
        s_c15  = np.polyfit(logM, log_gc_c15,  1)[0]

        print(f"  log gc vs log M* slope:")
        print(f"    No hot gas:  {s_nohg:+.3f}")
        print(f"    Hot gas:     {s_hg:+.3f}")
        print(f"    C15 pred:    {s_c15:+.3f}")
        print(f"    MOND pred:   0.000")

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: correction factor
        ax = axs[0]
        ax.semilogx(g_bar_grid/a0, correction_factor, 'ko-', ms=6)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('g_bar / a0')
        ax.set_ylabel('C(g_bar) = ESD_hg / ESD_nohg')
        ax.set_title('Universal Hot-Gas Correction Factor (from Fig-4 pair)')
        ax.grid(True, alpha=0.3)
        ax.text(1e-4, 0.5, 'Strong reduction\nat low g_bar\n(large R)',
                fontsize=9, color='red')

        # Panel 2: gc per bin (nohg vs hg)
        ax = axs[1]
        if bin_results:
            lm = [r['logM'] for r in bin_results]
            ax.semilogy(lm, [r['gc_nohg'] for r in bin_results], 'ko-', ms=8,
                        lw=2, label='no hot gas (raw)')
            ax.semilogy(lm, [r['gc_hg'] for r in bin_results], 'rs-', ms=8,
                        lw=2, label='hot gas corrected')
            ax.semilogy(lm, [r['gc_c15'] for r in bin_results], 'b^--', ms=8,
                        lw=1.5, label='C15 prediction')
            ax.axhline(1.0, color='g', ls=':', lw=1, label='MOND (gc=a0)')
            ax.axhline(0.24, color='gray', ls=':', lw=1, label='SPARC median')
            ax.set_xlabel('log10(M*/Msun)')
            ax.set_ylabel('gc / a0')
            ax.set_title('Per-bin gc: Hot Gas Effect vs C15 Prediction')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = BASE / "brouwer_kids_c15_hotgas_per_mbin.png"
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved: {out_png}")
    except Exception as e:
        print(f"\n[Plot error: {e}]")


if __name__ == '__main__':
    main()
