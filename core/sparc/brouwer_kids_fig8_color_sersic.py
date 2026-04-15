# -*- coding: utf-8 -*-
"""
brouwer_kids_fig8_color_sersic.py

Test C15 morphological-type bridge prediction using Brouwer+2021 Fig-8:
  - Color bins (u-r): blue vs red
  - Sersic bins (n): low vs high

C15 + bridge (rho=+0.89) predicts: gc(red/high-n) > gc(blue/low-n)

Usage: uv run --with scipy --with matplotlib python brouwer_kids_fig8_color_sersic.py
Place in: D:\\...\\brouwer2021\\

Author: Sakaguchi Shinobu / Sakaguchi Seimensho
Date: 2026-04-14
"""

import os, sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

# ---------- Constants ----------
a0 = 1.2e-10       # m/s^2
ESD_TO_GOBS = 5.580e-13

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

# f_gas scan (reduced set - focus on sign stability)
F_GAS_GRID = [0.0, 0.3, 0.5, 1.0]


# ====================================================================
#  DATA LOADING
# ====================================================================

def load_esd(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'g_bar': data[:, 0],
        'ESD_t': data[:, 1],
        'error': data[:, 3],
        'bias':  data[:, 4],
    }


def load_covmatrix_2bin(filepath, n_gbar=15):
    """Load 30x30 covariance matrix for 2-bin data.

    Format: bin_min[m], bin_min[n], g_bar[i], g_bar[j], cov, corr, bias
    """
    data = np.loadtxt(filepath, comments='#')

    pairs_m = list(zip(data[:, 0], data[:, 2]))
    all_pairs = sorted(set(pairs_m))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}

    n = len(all_pairs)
    cov_raw = np.zeros((n, n))
    bias_cov = np.zeros((n, n))

    for row in data:
        key_i = (row[0], row[2])
        key_j = (row[1], row[3])
        if key_i in pair_to_idx and key_j in pair_to_idx:
            i = pair_to_idx[key_i]
            j = pair_to_idx[key_j]
            cov_raw[i, j] = row[4]
            bias_cov[i, j] = row[6]

    # Avoid division by zero
    bias_cov[bias_cov == 0] = 1.0
    cov_corrected = cov_raw / bias_cov

    return cov_corrected, all_pairs


def find_fig8_files(obs_type):
    """Find Fig-8 files for a given observable type (Color or Sersic).

    Try multiple naming patterns.
    """
    patterns = [
        # Pattern 1: Fig-8_*_{type}bin-{N}.txt
        (f"Fig-8_KiDS-isolated_{obs_type}bin-1.txt",
         f"Fig-8_KiDS-isolated_{obs_type}bin-2.txt",
         f"Fig-8_RAR-KiDS-isolated_{obs_type}bins_covmatrix.txt"),
        # Pattern 2: alternative naming
        (f"Fig-8_KiDS-isolated_{obs_type}bin-1.txt",
         f"Fig-8_KiDS-isolated_{obs_type}bin-2.txt",
         f"Fig-8_KiDS-isolated_{obs_type}bins_covmatrix.txt"),
    ]

    # Try to find files by glob
    import glob
    bin1_candidates = glob.glob(str(BASE / f"Fig-8*{obs_type}*1*.txt"))
    bin2_candidates = glob.glob(str(BASE / f"Fig-8*{obs_type}*2*.txt"))
    cov_candidates = glob.glob(str(BASE / f"Fig-8*{obs_type}*cov*.txt"))

    if bin1_candidates and bin2_candidates:
        cov = cov_candidates[0] if cov_candidates else None
        return bin1_candidates[0], bin2_candidates[0], cov

    # Try explicit patterns
    for p1, p2, pc in patterns:
        f1, f2 = BASE / p1, BASE / p2
        fc = BASE / pc
        if f1.exists() and f2.exists():
            return str(f1), str(f2), str(fc) if fc.exists() else None

    return None, None, None


# ====================================================================
#  PHYSICS
# ====================================================================

def esd_to_gobs(ESD_t, bias):
    return ESD_TO_GOBS * ESD_t / bias


def gobs_to_esd(g_obs, bias):
    return g_obs * bias / ESD_TO_GOBS


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4 * gc * g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def fit_gc(g_bar, g_obs, g_err):
    """Fit gc minimizing weighted chi2 in log-space."""
    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(g_bar, gc)
        res = np.log10(np.maximum(g_obs, 1e-30)) - np.log10(np.maximum(gp, 1e-30))
        w = 1.0 / np.maximum((g_err / (g_obs * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)
    result = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**result.x


def compute_correction_factor_scaled(C_nominal, f_gas):
    C_f = 1.0 - f_gas * (1.0 - C_nominal)
    return np.maximum(C_f, 0.01)


# ====================================================================
#  SINGLE OBSERVABLE ANALYSIS
# ====================================================================

def analyze_observable(obs_type, label_low, label_high):
    """Full analysis for one observable (Color or Sersic).

    Args:
        obs_type: 'Color' or 'Sersic'
        label_low: label for bin 1 (e.g., 'Blue' or 'Low-n')
        label_high: label for bin 2 (e.g., 'Red' or 'High-n')
    """
    print(f"\n{'='*72}")
    print(f"  {obs_type} ANALYSIS: {label_low} (bin1) vs {label_high} (bin2)")
    print(f"{'='*72}")

    f1, f2, fc = find_fig8_files(obs_type)

    if f1 is None:
        print(f"  ERROR: Fig-8 {obs_type} files not found in {BASE}")
        print(f"  Searched for: Fig-8*{obs_type}*")
        import glob
        all_fig8 = glob.glob(str(BASE / "Fig-8*"))
        print(f"  Available Fig-8 files: {[os.path.basename(f) for f in all_fig8]}")
        return None

    print(f"  Bin 1 ({label_low}): {os.path.basename(f1)}")
    print(f"  Bin 2 ({label_high}): {os.path.basename(f2)}")
    print(f"  Covmatrix: {os.path.basename(fc) if fc else 'NOT FOUND'}")

    d1 = load_esd(f1)
    d2 = load_esd(f2)

    n_pts = len(d1['g_bar'])
    print(f"  g_bar points per bin: {n_pts}")

    # Load hot gas correction if available
    f_nohg = BASE / "Fig-4-5-C1_KiDS-isolated_Nobins.txt"
    f_hg = BASE / "Fig-4_KiDS-isolated_hotgas_Nobins.txt"

    has_hg = f_nohg.exists() and f_hg.exists()
    if has_hg:
        d_nohg = load_esd(f_nohg)
        d_hg = load_esd(f_hg)
        C_nominal_full = d_hg['ESD_t'] / d_nohg['ESD_t']
        g_bar_hg_grid = d_nohg['g_bar']

        # Interpolate to bin g_bar grids
        from scipy.interpolate import interp1d
        log_C_interp = interp1d(
            np.log10(g_bar_hg_grid), np.log10(C_nominal_full),
            bounds_error=False, fill_value='extrapolate')
        C_nom_1 = 10**log_C_interp(np.log10(d1['g_bar']))
        C_nom_2 = 10**log_C_interp(np.log10(d2['g_bar']))

    # Load covariance matrix
    has_cov = fc is not None and os.path.exists(fc)
    if has_cov:
        cov_raw, pair_map = load_covmatrix_2bin(fc)
        print(f"  Covariance: {cov_raw.shape}, cond={np.linalg.cond(cov_raw):.1e}")

    # ---------- f_gas scan ----------
    print(f"\n--- gc fitting across f_gas ---")
    print(f"  {'f_gas':>6s}  {'gc_'+label_low+'/a0':>14s}  {'gc_'+label_high+'/a0':>14s}"
          f"  {'ratio(H/L)':>12s}  {'Delta(dex)':>12s}  {'Prediction':>12s}")

    results = []

    for f_gas in F_GAS_GRID:
        # Apply hot gas correction
        if has_hg and f_gas > 0:
            C1 = compute_correction_factor_scaled(C_nom_1, f_gas)
            C2 = compute_correction_factor_scaled(C_nom_2, f_gas)
        else:
            C1 = np.ones_like(d1['ESD_t'])
            C2 = np.ones_like(d2['ESD_t'])

        esd1 = d1['ESD_t'] * C1
        esd2 = d2['ESD_t'] * C2
        err1 = d1['error'] * C1
        err2 = d2['error'] * C2

        go1 = esd_to_gobs(esd1, d1['bias'])
        ge1 = esd_to_gobs(err1, d1['bias'])
        go2 = esd_to_gobs(esd2, d2['bias'])
        ge2 = esd_to_gobs(err2, d2['bias'])

        v1 = (d1['g_bar'] > 0) & (go1 > 0) & np.isfinite(go1) & (ge1 > 0)
        v2 = (d2['g_bar'] > 0) & (go2 > 0) & np.isfinite(go2) & (ge2 > 0)

        gc1 = fit_gc(d1['g_bar'][v1], go1[v1], ge1[v1])
        gc2 = fit_gc(d2['g_bar'][v2], go2[v2], ge2[v2])

        ratio = gc2 / gc1
        delta_dex = np.log10(gc2 / gc1)
        pred_ok = "OK (gc_H > gc_L)" if gc2 > gc1 else "FAIL"

        # MOND and membrane residuals per bin
        res_mond_1 = np.log10(go1[v1] / mond_rar(d1['g_bar'][v1]))
        res_mond_2 = np.log10(go2[v2] / mond_rar(d2['g_bar'][v2]))
        res_memb_1 = np.log10(go1[v1] / membrane_rar(d1['g_bar'][v1], gc1))
        res_memb_2 = np.log10(go2[v2] / membrane_rar(d2['g_bar'][v2], gc2))

        results.append({
            'f_gas': f_gas,
            'gc1': gc1/a0, 'gc2': gc2/a0,
            'ratio': ratio, 'delta_dex': delta_dex,
            'pred_ok': gc2 > gc1,
            'mond_scatter_1': np.std(res_mond_1),
            'mond_scatter_2': np.std(res_mond_2),
            'memb_scatter_1': np.std(res_memb_1),
            'memb_scatter_2': np.std(res_memb_2),
        })

        print(f"  {f_gas:6.1f}  {gc1/a0:14.3f}  {gc2/a0:14.3f}"
              f"  {ratio:12.2f}  {delta_dex:+12.3f}  {pred_ok:>12s}")

    # ---------- Formal chi2 with covariance ----------
    if has_cov:
        print(f"\n--- Formal chi2 with {cov_raw.shape[0]}x{cov_raw.shape[0]} covariance ---")

        for f_gas in [0.0, 1.0]:
            if has_hg and f_gas > 0:
                C1 = compute_correction_factor_scaled(C_nom_1, f_gas)
                C2 = compute_correction_factor_scaled(C_nom_2, f_gas)
            else:
                C1 = np.ones_like(d1['ESD_t'])
                C2 = np.ones_like(d2['ESD_t'])

            esd1 = d1['ESD_t'] * C1
            esd2 = d2['ESD_t'] * C2

            go1 = esd_to_gobs(esd1, d1['bias'])
            go2 = esd_to_gobs(esd2, d2['bias'])
            ge1 = esd_to_gobs(d1['error'] * C1, d1['bias'])
            ge2 = esd_to_gobs(d2['error'] * C2, d2['bias'])

            v1 = (d1['g_bar'] > 0) & (go1 > 0) & np.isfinite(go1) & (ge1 > 0)
            v2 = (d2['g_bar'] > 0) & (go2 > 0) & np.isfinite(go2) & (ge2 > 0)

            gc1 = fit_gc(d1['g_bar'][v1], go1[v1], ge1[v1])
            gc2 = fit_gc(d2['g_bar'][v2], go2[v2], ge2[v2])
            gc_uniform = fit_gc(
                np.concatenate([d1['g_bar'][v1], d2['g_bar'][v2]]),
                np.concatenate([go1[v1], go2[v2]]),
                np.concatenate([ge1[v1], ge2[v2]]))

            # Build data and model vectors (2*n_pts)
            esd_data = np.concatenate([esd1, esd2])
            bias_all = np.concatenate([d1['bias'], d2['bias']])
            gbar_all = np.concatenate([d1['g_bar'], d2['g_bar']])

            # Model: MOND (gc=a0 uniform)
            esd_mond = gobs_to_esd(mond_rar(gbar_all), bias_all)

            # Model: membrane uniform gc
            esd_unif = gobs_to_esd(membrane_rar(gbar_all, gc_uniform), bias_all)

            # Model: membrane split gc
            esd_split = np.concatenate([
                gobs_to_esd(membrane_rar(d1['g_bar'], gc1), d1['bias']),
                gobs_to_esd(membrane_rar(d2['g_bar'], gc2), d2['bias']),
            ])

            # Scale covariance for hot gas
            C_vec = np.concatenate([C1, C2])
            cov_scaled = cov_raw * np.outer(C_vec, C_vec)

            cond = np.linalg.cond(cov_scaled)
            if cond > 1e12:
                cov_scaled += 0.01 * np.diag(np.diag(cov_scaled))

            try:
                cov_inv = np.linalg.inv(cov_scaled)

                def chi2_cov(d, m):
                    r = d - m
                    return float(r @ cov_inv @ r)

                c2_mond = chi2_cov(esd_data, esd_mond)
                c2_unif = chi2_cov(esd_data, esd_unif)
                c2_split = chi2_cov(esd_data, esd_split)

                n_data = len(esd_data)

                print(f"\n  f_gas = {f_gas:.1f}:")
                print(f"    chi2(MOND, gc=a0)      = {c2_mond:8.1f}  (dof={n_data})")
                print(f"    chi2(membrane, uniform) = {c2_unif:8.1f}  (dof={n_data-1})")
                print(f"    chi2(membrane, split)   = {c2_split:8.1f}  (dof={n_data-2})")
                print(f"    Delta-chi2 (uniform - split) = {c2_unif - c2_split:+.1f}"
                      f"  {'-> split preferred' if c2_split < c2_unif else '-> uniform sufficient'}")
                print(f"    Delta-chi2 (MOND - split)    = {c2_mond - c2_split:+.1f}")
                print(f"    gc_uniform = {gc_uniform/a0:.3f} a0")
                print(f"    gc_{label_low} = {gc1/a0:.3f} a0, "
                      f"gc_{label_high} = {gc2/a0:.3f} a0")
            except np.linalg.LinAlgError:
                print(f"\n  f_gas = {f_gas:.1f}: covariance inversion failed")

    # ---------- Sign stability summary ----------
    all_ok = all(r['pred_ok'] for r in results)
    n_ok = sum(r['pred_ok'] for r in results)

    print(f"\n--- Sign stability: gc({label_high}) > gc({label_low}) ---")
    print(f"  {n_ok}/{len(results)} f_gas values: "
          f"{'STABLE across all f_gas' if all_ok else 'UNSTABLE'}")

    return results


# ====================================================================
#  MAIN
# ====================================================================

def main():
    print("=" * 72)
    print("  BROUWER+2021 Fig-8: COLOR / SERSIC BIN ANALYSIS")
    print("  C15 morphological-type bridge test")
    print("=" * 72)

    # List available Fig-8 files
    import glob
    fig8_files = sorted(glob.glob(str(BASE / "Fig-8*")))
    print(f"\nAvailable Fig-8 files ({len(fig8_files)}):")
    for f in fig8_files:
        sz = os.path.getsize(f)
        print(f"  {os.path.basename(f):55s}  ({sz:,d} bytes)")

    # Color analysis
    results_color = analyze_observable('Color', 'Blue', 'Red')

    # Sersic analysis
    results_sersic = analyze_observable('Sersic', 'Low-n', 'High-n')

    # ====================================================================
    #  GRAND SUMMARY
    # ====================================================================
    print(f"\n{'='*72}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*72}")

    print(f"\nC15 + morphological-type bridge prediction:")
    print(f"  gc(Red) > gc(Blue)    [early-type -> higher gc]")
    print(f"  gc(High-n) > gc(Low-n) [bulge-dominated -> higher gc]")

    if results_color:
        print(f"\nColor results (f_gas=0 / 1.0):")
        for r in results_color:
            if r['f_gas'] in [0.0, 1.0]:
                print(f"  f={r['f_gas']:.0f}: gc_Blue={r['gc1']:.3f}, "
                      f"gc_Red={r['gc2']:.3f}, "
                      f"ratio={r['ratio']:.2f}, "
                      f"{'OK' if r['pred_ok'] else 'FAIL'}")

    if results_sersic:
        print(f"\nSersic results (f_gas=0 / 1.0):")
        for r in results_sersic:
            if r['f_gas'] in [0.0, 1.0]:
                print(f"  f={r['f_gas']:.0f}: gc_Low-n={r['gc1']:.3f}, "
                      f"gc_High-n={r['gc2']:.3f}, "
                      f"ratio={r['ratio']:.2f}, "
                      f"{'OK' if r['pred_ok'] else 'FAIL'}")

    # Assess overall
    color_stable = results_color and all(r['pred_ok'] for r in results_color)
    sersic_stable = results_sersic and all(r['pred_ok'] for r in results_sersic)

    print(f"\n  Color sign stability:  {'STABLE' if color_stable else 'UNSTABLE'}")
    print(f"  Sersic sign stability: {'STABLE' if sersic_stable else 'UNSTABLE'}")

    if color_stable and sersic_stable:
        print(f"\n  >> Both observables show gc ordering consistent with C15")
        print(f"     morphological-type bridge across all f_gas values.")
        print(f"     Grade: B (qualitative confirmation, not quantitatively decisive)")
    elif color_stable or sersic_stable:
        stable = "Color" if color_stable else "Sersic"
        print(f"\n  >> {stable} shows stable ordering; the other is unstable.")
        print(f"     Grade: B- (partial confirmation)")
    else:
        print(f"\n  >> Neither observable shows stable gc ordering.")
        print(f"     Grade: C (inconclusive)")

    # ====================================================================
    #  PLOT
    # ====================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        datasets = [
            (results_color, 'Color', 'Blue', 'Red', axs[0]),
            (results_sersic, 'Sersic', 'Low-n', 'High-n', axs[1]),
        ]

        for data, obs, lo, hi, ax in datasets:
            if data is None:
                ax.text(0.5, 0.5, f'{obs}: files not found',
                        transform=ax.transAxes, ha='center')
                continue

            fgs = [r['f_gas'] for r in data]
            gc_lo = [r['gc1'] for r in data]
            gc_hi = [r['gc2'] for r in data]

            ax.plot(fgs, gc_lo, 'bo-', ms=8, lw=2, label=f'{lo} (bin 1)')
            ax.plot(fgs, gc_hi, 'rs-', ms=8, lw=2, label=f'{hi} (bin 2)')
            ax.axhline(1.0, color='gray', ls=':', lw=1, label='MOND (gc=a0)')

            # Shade region where prediction holds
            ax.fill_between(fgs,
                            [min(gc_lo)*0.5]*len(fgs),
                            [max(gc_hi)*1.5]*len(fgs),
                            alpha=0.05, color='green',
                            where=[r['pred_ok'] for r in data])

            ax.set_xlabel('f_gas (M_hot/M*)')
            ax.set_ylabel('gc / a0')
            ax.set_title(f'{obs} bins: gc({hi}) > gc({lo}) ?')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = BASE / "brouwer_kids_fig8_color_sersic.png"
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved: {out_png}")
    except Exception as e:
        print(f"\n[Plot error: {e}]")

    print(f"\n{'='*72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
