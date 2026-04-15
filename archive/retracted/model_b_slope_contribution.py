#!/usr/bin/env python3
"""
model_b_slope_contribution.py

Model B (Eq 10a-10d) の kappa_total/kappa_A が gc-M* スロープに
与える寄与を定量化。HSC観測+0.166 vs C15予測+0.075 の差+0.091 を説明できるか検証。
"""

import os
import csv
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
TA3_CSV = os.path.join(BASE, 'TA3_gc_independent.csv')
PH1_CSV = os.path.join(BASE, 'phase1', 'sparc_results.csv')
MRT_FILE = os.path.join(BASE, 'SPARC_Lelli2016c.mrt')


def load_sparc():
    """3-file merge: TA3 + phase1 + MRT."""
    ta3 = {}
    with open(TA3_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row['gc_over_a0'])
            except ValueError:
                continue
            if name and gc_a0 > 0:
                ta3[name] = {'gc_over_a0': gc_a0}

    ph1 = {}
    with open(PH1_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                ud = float(row['ud'])
                vflat = float(row['vflat'])
            except ValueError:
                continue
            if name:
                ph1[name] = {'ud': ud, 'vflat': vflat}

    # MRT using split() pattern (sep>=4 then in_data)
    mrt = {}
    in_data = False
    sep = 0
    with open(MRT_FILE, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4:
                    in_data = True
                continue
            if not in_data:
                continue
            p = line.split()
            if len(p) < 18:
                continue
            try:
                # Columns per memory: [0]Galaxy [1]T [7]L36 [9]Reff [10]SBeff
                # [11]Rdisk [12]SBdisk0 [13]MHI [14]RHI [15]Vflat [17]Q
                mrt[p[0]] = {
                    'T_type': int(p[1]),
                    'L36': float(p[7]),
                    'Reff': float(p[9]),        # kpc (effective radius)
                    'Rdisk': float(p[11]),      # kpc (disk scale length = hR)
                    'MHI': float(p[13]),        # 10^9 Msun
                    'RHI': float(p[14]),        # kpc (HI radius)
                    'Vflat': float(p[15]),      # km/s
                }
            except (ValueError, IndexError):
                continue

    galaxies = []
    for name in ta3:
        if name not in ph1 or name not in mrt:
            continue
        m = mrt[name]; p = ph1[name]; t = ta3[name]

        Mstar = p['ud'] * m['L36'] * 1e9  # Msun
        vflat = p['vflat'] if p['vflat'] > 0 else m['Vflat']
        if Mstar <= 0 or vflat <= 0:
            continue

        hR = m['Rdisk']
        # r_s proxy: use RHI (HI extent) or 2*Rdisk as fallback
        r_s_est = m['RHI'] if m['RHI'] > 0 else 2.0 * m['Rdisk']

        galaxies.append({
            'name': name,
            'gc_a0': t['gc_over_a0'],
            'Mstar': Mstar, 'logMstar': np.log10(Mstar),
            'v_flat': vflat, 'hR': hR,
            'r_s_est': r_s_est,
            'T_type': m['T_type'],
            'MHI': m['MHI'] * 1e9,
        })

    print(f"Merged: {len(galaxies)} galaxies")
    return galaxies


def T_transition(r, r_s, w=3.0):
    arg = np.clip(w * (r / r_s - 1.0), -20, 20)
    return 0.5 * (1.0 + np.tanh(arg))


def compute_kappa_ratio_LOS(R_kpc, r_s_kpc, xi_kpc, delta_kpc,
                             k_cross, alpha, beta_AB, L_max_kpc=3000.0):
    """kappa_total/kappa_A at projected R via LOS integration (SIS weighting)."""
    N_pts = 200
    l_arr = np.linspace(-L_max_kpc, L_max_kpc, N_pts)
    dl = l_arr[1] - l_arr[0]

    sum_rho = 0.0
    sum_rho_P = 0.0
    for l in l_arr:
        r = np.sqrt(R_kpc**2 + l**2)
        if r < 0.1:
            continue
        rho = 1.0 / r**2
        T_val = T_transition(r, r_s_kpc)
        n_fold = k_cross * (T_val / r)**2  # [1/kpc^2]
        P_rw = 1.0 - np.exp(-np.pi * (2.0 * delta_kpc)**2 * n_fold)
        sum_rho += rho * dl
        sum_rho_P += rho * P_rw * dl

    if sum_rho < 1e-30:
        return 1.0

    q = sum_rho_P / sum_rho
    attn = np.exp(-delta_kpc / xi_kpc) if xi_kpc > 0 else 1.0
    return (1.0 + alpha * beta_AB * q) * (q + (1.0 - q) * attn)


def compute_effective_kappa_ratio(r_s_kpc, k_cross, alpha, beta_AB,
                                   f_xi, f_delta,
                                   R_min_kpc=30.0, R_max_kpc=2000.0,
                                   N_R=10):
    xi_kpc = f_xi * r_s_kpc
    delta_kpc = f_delta * xi_kpc

    R_arr = np.logspace(np.log10(R_min_kpc), np.log10(R_max_kpc), N_R)
    ratios = np.array([compute_kappa_ratio_LOS(R, r_s_kpc, xi_kpc, delta_kpc,
                                                k_cross, alpha, beta_AB)
                       for R in R_arr])
    weights = 1.0 / R_arr  # ESD weighting
    eff_ratio = np.average(ratios, weights=weights)
    return eff_ratio, R_arr, ratios


def main():
    galaxies = load_sparc()

    alpha = 2.0
    beta_AB = 0.503
    k_cross_values = [10.0, 50.0, 200.0, 403.4]
    f_xi_values = [0.5, 1.0, 2.0, 5.0]
    f_delta = 0.70

    mstar_bins = [
        (8.5, 10.3, 'Bin1'),
        (10.3, 10.6, 'Bin2'),
        (10.6, 10.8, 'Bin3'),
        (10.8, 11.0, 'Bin4'),
    ]

    print("\n" + "="*70)
    print("Model B slope contribution to gc-M*")
    print("="*70)
    print(f"Target: Delta_slope = +0.091 (obs +0.166 - C15 +0.075)")
    print(f"alpha = {alpha}, beta_AB = {beta_AB:.3f}, f_delta = {f_delta}")
    print()

    print(f"{'k_cross':>8s} {'f_xi':>6s} | "
          f"{'Bin1 kr':>8s} {'Bin2 kr':>8s} {'Bin3 kr':>8s} {'Bin4 kr':>8s} | "
          f"{'slope':>8s} {'gap':>8s}")
    print("-"*80)

    results_all = []
    best_match = None
    best_delta = 999.0

    for k_cross in k_cross_values:
        for f_xi in f_xi_values:
            for g in galaxies:
                eff_ratio, _, _ = compute_effective_kappa_ratio(
                    g['r_s_est'], k_cross, alpha, beta_AB, f_xi, f_delta)
                g['kappa_ratio'] = eff_ratio

            bin_logM = []
            bin_log_kr = []
            for lo, hi, label in mstar_bins:
                subset = [g for g in galaxies if lo <= g['logMstar'] < hi]
                if len(subset) < 3:
                    bin_logM.append(0.5*(lo+hi)); bin_log_kr.append(0.0)
                    continue
                bin_logM.append(np.mean([g['logMstar'] for g in subset]))
                bin_log_kr.append(np.mean([np.log10(g['kappa_ratio']) for g in subset]))

            bin_logM = np.array(bin_logM); bin_log_kr = np.array(bin_log_kr)
            slope_kr = np.polyfit(bin_logM, bin_log_kr, 1)[0] if len(bin_logM) >= 2 else 0.0
            gap = slope_kr - 0.091
            match = abs(gap) < 0.03

            if abs(gap) < best_delta:
                best_delta = abs(gap)
                best_match = (k_cross, f_xi, slope_kr, bin_logM.copy(), bin_log_kr.copy())

            kr_strs = [f"{10**v:.4f}" for v in bin_log_kr]
            print(f"{k_cross:8.1f} {f_xi:6.1f} | "
                  f"{kr_strs[0]:>8s} {kr_strs[1]:>8s} {kr_strs[2]:>8s} {kr_strs[3]:>8s} | "
                  f"{slope_kr:+8.4f} {gap:+8.4f} "
                  f"{'<<*>>' if match else ''}")

            results_all.append({
                'k_cross': k_cross, 'f_xi': f_xi,
                'slope': slope_kr, 'gap': gap,
                'bin_logM': bin_logM.copy(), 'bin_log_kr': bin_log_kr.copy(),
            })

    print("\n" + "="*70)
    print("Best match")
    print("="*70)
    if best_match:
        k_cross_best, f_xi_best, slope_best, bm, blk = best_match
        slope_total = 0.075 + slope_best
        gap = 0.166 - slope_total
        sigma = abs(gap) / 0.041
        print(f"  k_cross = {k_cross_best:.1f}")
        print(f"  f_xi (xi/r_s) = {f_xi_best:.1f}")
        print(f"  Model B slope = {slope_best:+.4f}")
        print(f"  C15 + Model B = {slope_total:+.4f}")
        print(f"  Observed      = +0.166 +/- 0.041")
        print(f"  Residual      = {gap:+.4f} ({sigma:.1f} sigma)")
        print()
        print(f"  Bin kappa_total/kappa_A:")
        for i, (lm, lkr) in enumerate(zip(bm, blk)):
            print(f"    Bin {i+1}: logM*={lm:.2f}, kappa_ratio={10**lkr:.4f} "
                  f"(+{(10**lkr - 1)*100:.2f}%)")

        rs_median = np.median([g['r_s_est'] for g in galaxies])
        xi_med = f_xi_best * rs_median
        delta_med = f_delta * xi_med
        print(f"\n  Representative (r_s_median = {rs_median:.1f} kpc):")
        print(f"    xi = {xi_med:.1f} kpc, delta = {delta_med:.1f} kpc")

    # Full galaxy scatter (best)
    logM_all = []; logkr_all = []
    if best_match:
        k_cross_best, f_xi_best = best_match[0], best_match[1]
        for g in galaxies:
            eff_ratio, _, _ = compute_effective_kappa_ratio(
                g['r_s_est'], k_cross_best, alpha, beta_AB, f_xi_best, f_delta)
            g['kappa_ratio_best'] = eff_ratio
            logM_all.append(g['logMstar'])
            logkr_all.append(np.log10(eff_ratio))
        logM_all = np.array(logM_all); logkr_all = np.array(logkr_all)
        coeffs_all = np.polyfit(logM_all, logkr_all, 1)
        from scipy.stats import spearmanr
        rho, pv = spearmanr(logM_all, logkr_all)
        print(f"\n  All galaxies: slope={coeffs_all[0]:+.4f}, Spearman rho={rho:+.3f} (p={pv:.2e})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Scan heatmap
    ax = axes[0, 0]
    k_arr = sorted(set(r['k_cross'] for r in results_all))
    f_arr = sorted(set(r['f_xi'] for r in results_all))
    Z = np.zeros((len(f_arr), len(k_arr)))
    for r in results_all:
        i = f_arr.index(r['f_xi']); j = k_arr.index(r['k_cross'])
        Z[i, j] = r['slope']
    im = ax.imshow(Z, aspect='auto', origin='lower',
                   extent=[0, len(k_arr), 0, len(f_arr)],
                   cmap='RdYlBu_r', vmin=-0.05, vmax=0.20)
    ax.set_xticks(np.arange(len(k_arr)) + 0.5)
    ax.set_xticklabels([f'{k:g}' for k in k_arr])
    ax.set_yticks(np.arange(len(f_arr)) + 0.5)
    ax.set_yticklabels([str(f) for f in f_arr])
    ax.set_xlabel('k_cross'); ax.set_ylabel('f_xi')
    ax.set_title('(a) Model B slope contribution')
    plt.colorbar(im, ax=ax, label='Delta slope')
    for r in results_all:
        i = f_arr.index(r['f_xi']); j = k_arr.index(r['k_cross'])
        ax.text(j + 0.5, i + 0.5, f"{r['slope']:+.3f}",
                ha='center', va='center', fontsize=7,
                color='white' if abs(r['slope']) > 0.1 else 'black')

    # (b) Best match bin
    ax = axes[0, 1]
    if best_match:
        bm, blk = best_match[3], best_match[4]
        kr_vals = [10**v for v in blk]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        ax.bar(range(4), kr_vals, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'Bin{i+1}\nlogM*={bm[i]:.1f}' for i in range(4)], fontsize=8)
        ax.axhline(1.0, color='gray', ls='--', label='kappa_A (no Model B)')
        ax.set_ylabel('kappa_total / kappa_A')
        ax.set_title(f'(b) Best: k_cross={best_match[0]:g}, f_xi={best_match[1]}')
        ax.legend(fontsize=8)
        for i, kr in enumerate(kr_vals):
            ax.text(i, kr + 0.005, f'+{(kr-1)*100:.1f}%', ha='center', va='bottom', fontsize=8)

    # (c) Individual scatter
    ax = axes[1, 0]
    if best_match and len(logM_all) > 0:
        sc = ax.scatter(logM_all, logkr_all, c=[g['T_type'] for g in galaxies],
                       cmap='coolwarm', s=20, alpha=0.7, edgecolors='gray', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label='T-type')
        x_fit = np.linspace(7.5, 11.5, 100)
        ax.plot(x_fit, np.polyval(coeffs_all, x_fit), 'k-', lw=2,
                label=f'slope={coeffs_all[0]:+.4f}')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel('log10(M*/Msun)')
        ax.set_ylabel('log10(kappa_total/kappa_A)')
        ax.set_title('(c) Individual galaxies (best)')
        ax.legend(fontsize=9)

    # (d) Slope decomposition
    ax = axes[1, 1]
    slope_B = best_match[2] if best_match else 0.0
    labels = ['C15\n(SPARC)', 'Model B\ncontrib', 'C15+B\n(pred)', 'HSC\n(obs)']
    vals = [0.075, slope_B, 0.075 + slope_B, 0.166]
    errs = [0.0, 0.0, 0.0, 0.041]
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#F44336']
    ax.bar(range(4), vals, color=colors, alpha=0.8, yerr=errs, capsize=5, edgecolor='black')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('gc-M* slope')
    ax.set_title('(d) Slope decomposition')
    ax.axhline(0.166, color='red', ls=':', alpha=0.5)
    ax.axhline(0.075, color='blue', ls=':', alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f'{v:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fp = os.path.join(BASE, 'model_b_slope_contribution.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {fp}")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    if best_match:
        slope_total = 0.075 + best_match[2]
        gap = 0.166 - slope_total
        sigma = abs(gap) / 0.041
        print(f"  C15 predicted slope:    +0.075")
        print(f"  Model B contribution:   {best_match[2]:+.4f}")
        print(f"  C15 + Model B:          {slope_total:+.4f}")
        print(f"  HSC observed:           +0.166 +/- 0.041")
        print(f"  Residual:               {gap:+.4f} ({sigma:.1f} sigma)")
        if sigma < 1.0:
            print("  -> Tension resolved below 1 sigma with Model B")
        elif sigma < 2.0:
            print("  -> Tension partially relieved")
        else:
            print("  -> Model B alone insufficient, additional mechanism needed")


if __name__ == '__main__':
    main()
