#!/usr/bin/env python3
"""
model_b_R_dependent_slope.py

Model B R 依存スロープ検証 — Phase B Step 3 既存出力を使用。

予測: Model B では P_rewire(r) が r_s 周辺で最大 → 大 R で gc-M* slope が急
  - Null (MOND): slope = 0 at all R
  - C15 only: slope ~ +0.075, R 非依存
  - C15 + Model B: slope が R と共に増加

手法: g_bar ビンから R_eff = sqrt(G*M/g_bar) を計算、inner/mid/outer に分割。
既存 phase_b_combined_mbin_{1-4}.txt を読み直して高速化。
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE_B_DIR = os.path.join(BASE, "phase_b_output")

G_SI = 6.674e-11
Msun = 1.989e30
kpc = 3.086e19
a0_SI = 1.2e-10

N_GBAR = 15
MSTAR_BINS = [(8.5, 10.3), (10.3, 10.6), (10.6, 10.8), (10.8, 11.0)]
MSTAR_CENTERS = [9.4, 10.45, 10.70, 10.90]  # From Phase B Step 3 actual bin means
N_MBINS = len(MSTAR_BINS)

# Range splits (g_bar bins). Low g_bar = large R = outer.
RANGES = [
    ('Outer (low g_bar, large R)', list(range(0, 5))),
    ('Mid',                         list(range(5, 10))),
    ('Inner (high g_bar, small R)', list(range(10, 15))),
]


def load_phase_b_data():
    """Load existing Phase B Step 3 per-mbin output."""
    data = []
    for mb in range(1, N_MBINS + 1):
        fp = os.path.join(PHASE_B_DIR, f"phase_b_combined_mbin_{mb}.txt")
        arr = np.loadtxt(fp, comments='#')
        # Columns: g_bar, g_obs, g_obs_err, N_pairs
        data.append({
            'g_bar': arr[:, 0],
            'g_obs': arr[:, 1],
            'g_obs_err': arr[:, 2],
            'npair': arr[:, 3].astype(int)
        })
    return data


def mond_rar(g_bar, gc):
    x = np.sqrt(np.maximum(g_bar / gc, 0))
    x = np.clip(x, 1e-6, 50)
    return g_bar / (1.0 - np.exp(-x))


def fit_gc(g_bar, g_obs, g_err, mask=None):
    if mask is None:
        mask = np.ones(len(g_bar), dtype=bool)
    valid = mask & np.isfinite(g_bar) & np.isfinite(g_obs) & np.isfinite(g_err)
    valid &= (g_obs > 0) & (g_bar > 0) & (g_err > 0)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan

    gb = g_bar[valid]; go = g_obs[valid]; ge = g_err[valid]

    def chi2(log_gc):
        gc_val = 10**log_gc * a0_SI
        model = mond_rar(gb, gc_val)
        return np.sum(((go - model) / ge)**2)

    res = minimize_scalar(chi2, bounds=(-2, 2), method='bounded')
    gc_best = 10**res.x
    chi2_min = res.fun
    target = chi2_min + 1.0

    try:
        x_up = brentq(lambda x: chi2(x) - target, res.x, res.x + 2.0)
        gc_up = 10**x_up
    except (ValueError, RuntimeError):
        gc_up = 10**(res.x + 0.5)
    try:
        x_lo = brentq(lambda x: chi2(x) - target, res.x - 2.0, res.x)
        gc_lo = 10**x_lo
    except (ValueError, RuntimeError):
        gc_lo = 10**(res.x - 0.5)
    gc_err = (gc_up - gc_lo) / 2.0
    dof = max(valid.sum() - 1, 1)
    return gc_best, gc_err, chi2_min / dof


def compute_slope(gc_vals, gc_errs, logM):
    """Weighted log-log fit for gc-M* slope."""
    valid = np.isfinite(gc_vals) & (gc_vals > 0) & np.isfinite(gc_errs) & (gc_errs > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    x = logM[valid]
    y = np.log10(gc_vals[valid])
    ye = gc_errs[valid] / (gc_vals[valid] * np.log(10))
    w = 1.0 / ye**2
    S = np.sum(w); Sx = np.sum(w * x); Sy = np.sum(w * y)
    Sxx = np.sum(w * x**2); Sxy = np.sum(w * x * y)
    det = S * Sxx - Sx**2
    if det <= 0:
        return np.nan, np.nan
    slope = (S * Sxy - Sx * Sy) / det
    slope_err = np.sqrt(S / det)
    return slope, slope_err


def main():
    print("=" * 70)
    print("Model B R依存スロープテスト")
    print("=" * 70)
    print("予測: gc-M* slope が大 R (低 g_bar) ほど急")
    print("  帰無仮説 (MOND): slope = 0 at all R")
    print("  C15 only: slope ~ +0.075, R 非依存")
    print("  C15 + Model B: slope が R と共に増加")
    print()

    print("Phase B Step 3 既存出力を読み込み...")
    data = load_phase_b_data()
    g_bar = data[0]['g_bar']
    print(f"  {N_MBINS} M*bins, {N_GBAR} g_bar bins per bin")
    for mb in range(N_MBINS):
        npair = data[mb]['npair'].sum()
        print(f"  M* bin {mb+1} (logM*~{MSTAR_CENTERS[mb]:.2f}): "
              f"total pairs = {npair:,}")

    # R_eff per (M*bin, gbar_bin)
    print("\n  R_eff [kpc] per (M*bin, g_bar bin):")
    R_eff = np.zeros((N_MBINS, N_GBAR))
    for mb in range(N_MBINS):
        M_gal_kg = 10**MSTAR_CENTERS[mb] * Msun
        for gb in range(N_GBAR):
            R_m = np.sqrt(G_SI * M_gal_kg / g_bar[gb])
            R_eff[mb, gb] = R_m / kpc

    print(f"  {'g_bar_bin':>9s} {'g_bar':>10s}", end='')
    for mb in range(N_MBINS):
        print(f"  Bin{mb+1} R[kpc]", end='')
    print()
    for gb in range(N_GBAR):
        tag = 'OUTER' if gb < 5 else ('MID' if gb < 10 else 'INNER')
        print(f"  {gb:9d} {g_bar[gb]:10.2e}", end='')
        for mb in range(N_MBINS):
            print(f"  {R_eff[mb, gb]:11.0f}", end='')
        print(f"  <-- {tag}")

    # Analysis per range
    print("\n" + "=" * 70)
    print("R 範囲別 gc-M* スロープ")
    print("=" * 70)

    logM_arr = np.array(MSTAR_CENTERS)
    gc_results = {}

    print(f"\n  {'Range':>30s}  {'Bin1':>14s}  {'Bin2':>14s}  "
          f"{'Bin3':>14s}  {'Bin4':>14s}  {'slope':>12s}")
    print("  " + "-" * 110)

    for rname, rbins in RANGES:
        mask = np.zeros(N_GBAR, dtype=bool)
        mask[rbins] = True

        R_mean_range = np.mean(R_eff[:, rbins])

        gc_list = np.zeros(N_MBINS)
        gc_err_list = np.zeros(N_MBINS)
        for mb in range(N_MBINS):
            d = data[mb]
            gc, gce, chi2r = fit_gc(d['g_bar'], d['g_obs'], d['g_obs_err'],
                                     mask=mask)
            gc_list[mb] = gc
            gc_err_list[mb] = gce

        slope, slope_err = compute_slope(gc_list, gc_err_list, logM_arr)

        gc_strs = []
        for i in range(N_MBINS):
            if np.isfinite(gc_list[i]):
                gc_strs.append(f"{gc_list[i]:.2f}+/-{gc_err_list[i]:.2f}")
            else:
                gc_strs.append("---")

        slope_str = f"{slope:+.4f}+/-{slope_err:.4f}" if np.isfinite(slope) else "---"
        print(f"  {rname:>30s}  {gc_strs[0]:>14s}  {gc_strs[1]:>14s}  "
              f"{gc_strs[2]:>14s}  {gc_strs[3]:>14s}  {slope_str:>12s}")

        gc_results[rname] = {
            'gc': gc_list, 'gc_err': gc_err_list,
            'slope': slope, 'slope_err': slope_err,
            'R_mean': R_mean_range, 'bins': rbins
        }

    # Full
    gc_full = np.zeros(N_MBINS); gc_full_err = np.zeros(N_MBINS)
    for mb in range(N_MBINS):
        d = data[mb]
        gc, gce, _ = fit_gc(d['g_bar'], d['g_obs'], d['g_obs_err'])
        gc_full[mb] = gc; gc_full_err[mb] = gce
    slope_full, slope_full_err = compute_slope(gc_full, gc_full_err, logM_arr)
    gc_f_strs = [f"{gc:.2f}+/-{gce:.2f}" for gc, gce in zip(gc_full, gc_full_err)]
    slope_f_str = f"{slope_full:+.4f}+/-{slope_full_err:.4f}"
    print(f"  {'Full (all bins)':>30s}  {gc_f_strs[0]:>14s}  {gc_f_strs[1]:>14s}  "
          f"{gc_f_strs[2]:>14s}  {gc_f_strs[3]:>14s}  {slope_f_str:>12s}")

    # Model B prediction comparison
    print("\n" + "=" * 70)
    print("Model B 予測との比較")
    print("=" * 70)

    slopes = {rn: gc_results[rn]['slope'] for rn, _ in RANGES}
    slope_errs = {rn: gc_results[rn]['slope_err'] for rn, _ in RANGES}

    for rn, _ in RANGES:
        s = slopes[rn]; se = slope_errs[rn]
        R = gc_results[rn]['R_mean']
        if np.isfinite(s):
            print(f"  {rn}: slope = {s:+.4f} +/- {se:.4f} (R_mean ~ {R:.0f} kpc)")

    s_outer = slopes[RANGES[0][0]]
    s_mid   = slopes[RANGES[1][0]]
    s_inner = slopes[RANGES[2][0]]
    se_outer = slope_errs[RANGES[0][0]]
    se_inner = slope_errs[RANGES[2][0]]

    if np.isfinite(s_outer) and np.isfinite(s_inner):
        dslope = s_outer - s_inner
        dslope_err = np.sqrt(se_outer**2 + se_inner**2)
        sigma = dslope / dslope_err if dslope_err > 0 else np.nan

        print(f"\n  Delta slope (Outer - Inner) = {dslope:+.4f} +/- {dslope_err:.4f}")
        print(f"  Significance: {sigma:+.2f} sigma")
        print(f"\n  Model B 予測: 大R で slope が急になるため Delta > 0 を予想")
        if dslope > 2 * dslope_err:
            print(f"  >> 2sigma以上の positive gradient -> Model B 予測整合")
        elif dslope > 0:
            print(f"  >> Positive gradient but < 2sigma -> Model B 予測と同符号、統計不足")
        elif abs(dslope) < dslope_err:
            print(f"  >> Near-zero gradient -> Model B 寄与は均一または検出限界以下")
        else:
            print(f"  >> Negative gradient -> Model B 予測と不整合")

        # Quantitative: Model B predicts +0.091 total. Under R-dependent model:
        # slope(R) = 0.075 + 0.091 * weight(R)
        # If weight ~ 0 at inner, ~2 at outer, dslope ~ 0.18.
        # If weight evenly distributed, dslope ~ 0 (uniform contribution).
        print(f"\n  定量: Model B 全体寄与 = +0.091")
        if np.isfinite(s_inner):
            residual_inner = s_inner - 0.075
            print(f"    Inner slope - C15 予測 = {residual_inner:+.4f}")
        if np.isfinite(s_outer):
            residual_outer = s_outer - 0.075
            print(f"    Outer slope - C15 予測 = {residual_outer:+.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors_mbin = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    # (a) RAR per M* bin
    ax = axes[0, 0]
    for mb in range(N_MBINS):
        d = data[mb]
        valid = (d['g_obs'] > 0) & np.isfinite(d['g_obs'])
        if valid.sum() == 0:
            continue
        ax.errorbar(d['g_bar'][valid]/a0_SI, d['g_obs'][valid]/a0_SI,
                    yerr=d['g_obs_err'][valid]/a0_SI,
                    fmt='o', ms=5, color=colors_mbin[mb], alpha=0.8,
                    label=f"Bin{mb+1} (logM*={MSTAR_CENTERS[mb]:.2f})")
    gb_plot = np.logspace(-5, 1, 100) * a0_SI
    ax.plot(gb_plot/a0_SI, mond_rar(gb_plot, a0_SI)/a0_SI,
            'k--', lw=1.5, label='MOND (gc=a0)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar / a0'); ax.set_ylabel('g_obs / a0')
    ax.set_title('(a) RAR per M* bin')
    ax.legend(fontsize=8)

    # (b) gc per R range x M*
    ax = axes[0, 1]
    x_pos = np.arange(N_MBINS)
    width = 0.25
    colors_range = ['#9C27B0', '#FF9800', '#2196F3']
    for ri, (rname, _) in enumerate(RANGES):
        gc_arr = gc_results[rname]['gc']
        gc_err_arr = gc_results[rname]['gc_err']
        valid = np.isfinite(gc_arr) & (gc_arr > 0)
        gc_plot = np.where(valid, gc_arr, 0)
        err_plot = np.where(valid, gc_err_arr, 0)
        ax.bar(x_pos + (ri - 1) * width, gc_plot, width,
               yerr=err_plot, capsize=3,
               color=colors_range[ri], alpha=0.7,
               label=rname.split('(')[0].strip())
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Bin{i+1}\n{MSTAR_CENTERS[i]:.2f}' for i in range(N_MBINS)])
    ax.set_ylabel('gc / a0')
    ax.set_title('(b) gc per R range x M*')
    ax.legend(fontsize=7)

    # (c) slope vs R
    ax = axes[1, 0]
    R_means = [gc_results[rn]['R_mean'] for rn, _ in RANGES]
    slope_vals = [gc_results[rn]['slope'] for rn, _ in RANGES]
    slope_errs_arr = [gc_results[rn]['slope_err'] for rn, _ in RANGES]
    ax.errorbar(R_means, slope_vals, yerr=slope_errs_arr,
                fmt='ko-', ms=10, lw=2, capsize=5, label='Observed')
    ax.axhline(0.075, color='blue', ls='--', lw=1.5, label='C15 (+0.075)')
    ax.axhline(0.166, color='red', ls='--', lw=1.5, label='HSC Full (+0.166)')
    ax.axhline(0.0, color='gray', ls=':', lw=1, label='MOND (0)')
    ax.fill_between([min(R_means)*0.5, max(R_means)*2],
                    0.166 - 0.041, 0.166 + 0.041,
                    color='red', alpha=0.1)
    # Qualitative Model B prediction
    R_pred = np.logspace(1.5, 3.5, 50)
    slope_pred = 0.075 + 0.091 * (1.0 - np.exp(-R_pred / 300.0))
    ax.plot(R_pred, slope_pred, 'g-', lw=2, alpha=0.5,
            label='Model B qual. pred.')
    ax.set_xscale('log')
    ax.set_xlabel('Effective R [kpc]')
    ax.set_ylabel('gc-M* slope')
    ax.set_title('(c) Slope vs R range -- Model B diagnostic')
    ax.legend(fontsize=7, loc='best')

    # (d) summary bar
    ax = axes[1, 1]
    labels = ['Outer\n(large R)', 'Mid', 'Inner\n(small R)', 'Full']
    vals = slope_vals + [slope_full]
    errs = slope_errs_arr + [slope_full_err]
    colors_bar = ['#9C27B0', '#FF9800', '#2196F3', '#4CAF50']
    ax.bar(range(4), vals, color=colors_bar, alpha=0.8,
           yerr=errs, capsize=5, edgecolor='black')
    ax.axhline(0.075, color='blue', ls='--', lw=1, label='C15 (+0.075)')
    ax.axhline(0.166, color='red', ls='--', lw=1, label='HSC (+0.166)')
    ax.axhline(0.0, color='gray', ls=':', lw=0.5, label='MOND')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('gc-M* slope')
    ax.set_title('(d) Slope decomposition')
    ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.01, f'{v:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fp = os.path.join(BASE, 'model_b_R_dependent_slope.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {fp}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Inner slope:  {s_inner:+.4f} +/- {se_inner:.4f}")
    print(f"  Mid slope:    {s_mid:+.4f} +/- {slope_errs[RANGES[1][0]]:.4f}")
    print(f"  Outer slope:  {s_outer:+.4f} +/- {se_outer:.4f}")
    print(f"  Full slope:   {slope_full:+.4f} +/- {slope_full_err:.4f}")
    print(f"  HSC observed: +0.166 +/- 0.041")
    print(f"  C15 pred:     +0.075")
    print()
    if np.isfinite(s_outer) and np.isfinite(s_inner):
        if s_outer > s_inner + 0.03 and dslope > 2*dslope_err:
            print("  >> Outer > Inner (>2sigma): Model B 予測と整合")
        elif s_outer > s_inner:
            print("  >> Outer > Inner but <2sigma: Suggestive, needs more data")
        elif abs(s_outer - s_inner) < dslope_err:
            print("  >> Outer ~ Inner: R 依存性は検出されず")
        else:
            print("  >> Outer < Inner: Model B 予測と不整合")


if __name__ == '__main__':
    main()
