#!/usr/bin/env python3
"""model_b_slope_contribution_v2.py — rs_tanh corrected"""

import os
import sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TA3  = os.path.join(BASE, 'TA3_gc_independent.csv')
MRT  = os.path.join(BASE, 'SPARC_Lelli2016c.mrt')

for fp in ['C:/Windows/Fonts/ipag.ttf',
           '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf']:
    if os.path.exists(fp):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break


def load_sparc():
    ta3 = {}
    with open(TA3, 'r') as f:
        hdr = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            row = dict(zip(hdr, vals))
            name = row['galaxy']
            try:
                ta3[name] = {
                    'gc_a0': float(row['gc_over_a0']),
                    'ud': float(row['upsilon_d']),
                    'rs_tanh': float(row['rs_tanh']) if 'rs_tanh' in row else 0.0,
                }
            except (ValueError, KeyError):
                continue

    # MRT whitespace split: 0:Galaxy 1:T 7:L36 9:Reff 11:Rdisk 13:MHI 15:Vflat
    mrt = {}
    with open(MRT, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                name  = parts[0]
                T     = float(parts[1])
                L36   = float(parts[7])
                eff   = float(parts[9])
                vflat = float(parts[15])
            except (ValueError, IndexError):
                continue
            mrt[name] = {'T': T, 'L36': L36, 'r_eff': eff, 'v_flat': vflat}

    galaxies = []
    n_rs_ta3 = 0
    for name in ta3:
        if name not in mrt:
            continue
        m = mrt[name]
        t = ta3[name]
        Mstar = t['ud'] * m['L36'] * 1e9
        if Mstar <= 0 or m['v_flat'] <= 0:
            continue
        hR = m['r_eff'] / 1.678 if m['r_eff'] > 0 else 1.0

        if t['rs_tanh'] > 0.1:
            r_s = t['rs_tanh']
            n_rs_ta3 += 1
        else:
            r_s = hR * 2.0

        galaxies.append({
            'name': name, 'gc_a0': t['gc_a0'],
            'Mstar': Mstar, 'logMstar': np.log10(Mstar),
            'v_flat': m['v_flat'], 'hR': hR, 'r_s': r_s,
            'T_type': m['T'],
        })

    rs_arr = [g['r_s'] for g in galaxies]
    print(f"SPARC: {len(galaxies)} galaxies, rs_tanh={n_rs_ta3}")
    print(f"  r_s: median={np.median(rs_arr):.2f}, "
          f"range={min(rs_arr):.2f}-{max(rs_arr):.2f} kpc")
    return galaxies


def T_transition(r, r_s, w=3.0):
    arg = np.clip(w * (r / r_s - 1.0), -20, 20)
    return 0.5 * (1.0 + np.tanh(arg))


def compute_kappa_ratio(r_s_kpc, k_cross, f_xi, f_delta=0.70,
                         alpha=2.0, beta_AB=0.503,
                         R_min=30.0, R_max=2000.0, N_R=12):
    xi = f_xi * r_s_kpc
    delta = f_delta * xi
    if xi < 0.01:
        return 1.0

    R_arr = np.logspace(np.log10(R_min), np.log10(R_max), N_R)
    ratios = []
    for R in R_arr:
        N_pts = 150
        l_arr = np.linspace(-3000.0, 3000.0, N_pts)
        dl = l_arr[1] - l_arr[0]
        sum_rho = 0.0
        sum_rho_P = 0.0
        for l in l_arr:
            r = np.sqrt(R**2 + l**2)
            if r < 0.1:
                continue
            rho = 1.0 / r**2
            T_val = T_transition(r, r_s_kpc)
            n_fold = k_cross * (T_val / r)**2
            P_rw = 1.0 - np.exp(-np.pi * (2.0 * delta)**2 * n_fold)
            sum_rho += rho * dl
            sum_rho_P += rho * P_rw * dl

        q = sum_rho_P / sum_rho if sum_rho > 1e-30 else 0.0
        attn = np.exp(-delta / xi) if xi > 0 else 1.0
        ratio = (1.0 + alpha * beta_AB * q) * (q + (1.0 - q) * attn)
        ratios.append(ratio)

    weights = 1.0 / R_arr
    return np.average(ratios, weights=weights)


def main():
    galaxies = load_sparc()

    alpha = 2.0
    beta_AB = 0.503
    f_delta = 0.70

    k_cross_values = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 403.4]
    f_xi_values = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

    mstar_bins = [
        (8.5, 10.3, 'Bin1'),
        (10.3, 10.6, 'Bin2'),
        (10.6, 10.8, 'Bin3'),
        (10.8, 11.0, 'Bin4'),
    ]

    print("\n" + "="*70)
    print("Model B slope contribution scan [v2: rs_tanh]")
    print("="*70)
    print(f"target Delta_slope = +0.091 (HSC +0.166 - C15 +0.075)")
    print(f"alpha={alpha}, beta_AB={beta_AB}, f_delta={f_delta}")
    print(f"scan: {len(k_cross_values)} x {len(f_xi_values)} = "
          f"{len(k_cross_values)*len(f_xi_values)}")
    print()

    results_all = []
    best_match = None
    best_delta = 999.0

    print(f"{'k_cross':>8s} {'f_xi':>6s} | "
          f"{'Bin1':>8s} {'Bin2':>8s} {'Bin3':>8s} {'Bin4':>8s} | "
          f"{'slope':>8s} {'gap':>8s} {'sig':>6s}")
    print("-"*80)

    for k_cross in k_cross_values:
        for f_xi in f_xi_values:
            for g in galaxies:
                g['kappa_ratio'] = compute_kappa_ratio(
                    g['r_s'], k_cross, f_xi, f_delta, alpha, beta_AB, N_R=8)

            bin_logM = []
            bin_log_kr = []
            for lo, hi, label in mstar_bins:
                subset = [g for g in galaxies if lo <= g['logMstar'] < hi]
                if len(subset) < 3:
                    bin_logM.append(0.5*(lo+hi))
                    bin_log_kr.append(0.0)
                    continue
                bin_logM.append(np.mean([g['logMstar'] for g in subset]))
                bin_log_kr.append(np.mean([np.log10(g['kappa_ratio']) for g in subset]))

            bin_logM = np.array(bin_logM)
            bin_log_kr = np.array(bin_log_kr)
            coeffs = np.polyfit(bin_logM, bin_log_kr, 1)
            slope_kr = coeffs[0]

            gap = slope_kr - 0.091
            sigma = abs(0.166 - (0.075 + slope_kr)) / 0.041

            if abs(gap) < abs(best_delta):
                best_delta = gap
                best_match = {
                    'k_cross': k_cross, 'f_xi': f_xi,
                    'slope': slope_kr, 'sigma': sigma,
                    'bin_logM': bin_logM.copy(),
                    'bin_log_kr': bin_log_kr.copy(),
                }

            kr_strs = [f"{10**v:.4f}" for v in bin_log_kr]
            flag = '***' if abs(gap) < 0.01 else ('**' if abs(gap) < 0.02 else
                   ('*' if abs(gap) < 0.04 else ''))
            print(f"{k_cross:8.1f} {f_xi:6.1f} | "
                  f"{kr_strs[0]:>8s} {kr_strs[1]:>8s} {kr_strs[2]:>8s} {kr_strs[3]:>8s} | "
                  f"{slope_kr:+8.4f} {gap:+8.4f} {sigma:6.2f} {flag}")

            results_all.append({
                'k_cross': k_cross, 'f_xi': f_xi,
                'slope': slope_kr, 'gap': gap, 'sigma': sigma,
            })

    print("\n" + "="*70)
    print("Best match")
    print("="*70)
    bm = best_match
    print(f"  k_cross = {bm['k_cross']:.1f}")
    print(f"  f_xi    = {bm['f_xi']:.1f}")
    print(f"  slope   = {bm['slope']:+.4f}")
    print(f"  C15+B   = {0.075 + bm['slope']:+.4f}")
    print(f"  sigma   = {bm['sigma']:.2f}")
    for i in range(4):
        kr = 10**bm['bin_log_kr'][i]
        print(f"    Bin{i+1}: logM={bm['bin_logM'][i]:.1f}, "
              f"kr={kr:.4f} ({(kr-1)*100:+.1f}%)")

    rs_med = np.median([g['r_s'] for g in galaxies])
    xi_med = bm['f_xi'] * rs_med
    delta_med = f_delta * xi_med
    print(f"\n  r_s median = {rs_med:.2f} kpc -> xi={xi_med:.2f}, delta={delta_med:.2f}")

    print("\n" + "="*70)
    print("Individual galaxies at best params")
    print("="*70)
    logM_all = []
    logkr_all = []
    for g in galaxies:
        g['kappa_ratio_best'] = compute_kappa_ratio(
            g['r_s'], bm['k_cross'], bm['f_xi'], f_delta, alpha, beta_AB, N_R=10)
        logM_all.append(g['logMstar'])
        logkr_all.append(np.log10(g['kappa_ratio_best']))

    logM_all = np.array(logM_all)
    logkr_all = np.array(logkr_all)
    coeffs_all = np.polyfit(logM_all, logkr_all, 1)
    if np.std(logkr_all) > 1e-9:
        rho, pval = spearmanr(logM_all, logkr_all)
    else:
        rho, pval = 0, 1
    print(f"  all-gal slope = {coeffs_all[0]:+.4f}")
    print(f"  Spearman rho = {rho:+.3f} (p={pval:.2e})")

    print("\n" + "="*70)
    print("Top-5 + max slope")
    print("="*70)
    sorted_results = sorted(results_all, key=lambda r: abs(r['gap']))
    print(f"  {'#':>3s} {'k_cross':>8s} {'f_xi':>6s} {'slope':>8s} {'C15+B':>8s} {'sigma':>6s}")
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1:>3d} {r['k_cross']:8.1f} {r['f_xi']:6.1f} "
              f"{r['slope']:+8.4f} {0.075+r['slope']:+8.4f} {r['sigma']:6.2f}")

    max_slope = max(r['slope'] for r in results_all)
    max_r = [r for r in results_all if r['slope'] == max_slope][0]
    print(f"\n  max slope = {max_slope:+.4f} @ k={max_r['k_cross']}, f_xi={max_r['f_xi']}")
    print(f"  C15+max = {0.075 + max_slope:+.4f} (sigma={max_r['sigma']:.2f})")

    if max_slope < 0.091:
        deficit = 0.091 - max_slope
        print(f"  >>> Model B 最大でも target に {deficit:+.4f} 不足")

    # figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    k_arr = sorted(set(r['k_cross'] for r in results_all))
    f_arr = sorted(set(r['f_xi'] for r in results_all))
    Z = np.full((len(f_arr), len(k_arr)), np.nan)
    for r in results_all:
        i = f_arr.index(r['f_xi'])
        j = k_arr.index(r['k_cross'])
        Z[i, j] = r['slope']
    im = ax.imshow(Z, aspect='auto', origin='lower',
                   extent=[0, len(k_arr), 0, len(f_arr)],
                   cmap='RdYlBu_r', vmin=-0.05, vmax=0.15)
    ax.set_xticks(np.arange(len(k_arr)) + 0.5)
    ax.set_xticklabels([f'{k:.0f}' if k >= 1 else f'{k:.1f}' for k in k_arr], fontsize=7)
    ax.set_yticks(np.arange(len(f_arr)) + 0.5)
    ax.set_yticklabels([f'{f:.1f}' for f in f_arr], fontsize=7)
    ax.set_xlabel('k_cross'); ax.set_ylabel('f_xi')
    ax.set_title('(a) Model B slope contribution [v2]')
    plt.colorbar(im, ax=ax, label='slope')
    for r in results_all:
        i = f_arr.index(r['f_xi'])
        j = k_arr.index(r['k_cross'])
        color = 'white' if abs(r['slope']) > 0.08 else 'black'
        ax.text(j+0.5, i+0.5, f"{r['slope']:+.3f}", ha='center', va='center',
                fontsize=5, color=color)

    ax = axes[0, 1]
    kr_vals = [10**v for v in bm['bin_log_kr']]
    colors_bin = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    ax.bar(range(4), kr_vals, color=colors_bin, alpha=0.8, edgecolor='black', lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'Bin{i+1}\n{bm["bin_logM"][i]:.1f}' for i in range(4)], fontsize=9)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_ylabel('kappa_total / kappa_A')
    ax.set_title(f'(b) Best: k={bm["k_cross"]:.0f}, f_xi={bm["f_xi"]:.1f}, s={bm["sigma"]:.2f}')
    for i, kr in enumerate(kr_vals):
        ax.text(i, kr + 0.003, f'{(kr-1)*100:+.1f}%', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    sc = ax.scatter(logM_all, logkr_all, c=[g['T_type'] for g in galaxies],
                    cmap='coolwarm', s=20, alpha=0.7, edgecolors='gray', lw=0.3)
    plt.colorbar(sc, ax=ax, label='T-type')
    x_fit = np.linspace(7.5, 11.5, 100)
    ax.plot(x_fit, np.polyval(coeffs_all, x_fit), 'k-', lw=2,
            label=f'slope={coeffs_all[0]:+.4f}')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('log10(M*/Msun)'); ax.set_ylabel('log10(kappa_total/kappa_A)')
    ax.set_title(f'(c) Individual galaxies (rho={rho:+.3f})')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    slope_B = bm['slope']
    labels = ['C15', 'Model B v2', 'C15+B', 'HSC', 'Gap']
    gap_val = 0.091 - slope_B
    vals = [0.075, slope_B, 0.075+slope_B, 0.166, gap_val]
    errs = [0, 0, 0, 0.041, 0]
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#F44336', '#FF9800']
    ax.bar(range(5), vals, color=colors, alpha=0.8,
           yerr=errs, capsize=5, edgecolor='black', lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('gc-M* slope'); ax.set_title('(d) Slope budget')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f'{v:+.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    figpath = os.path.join(BASE, 'model_b_slope_contribution_v2.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nfig: {figpath}")

    print("\n" + "="*70)
    print("Summary [v2: rs_tanh]")
    print("="*70)
    print(f"  C15 prediction:  +0.075")
    print(f"  Model B best:    {slope_B:+.4f}")
    print(f"  C15+B:           {0.075 + slope_B:+.4f}")
    print(f"  HSC observed:    +0.166 +/- 0.041")
    print(f"  residual:        {bm['sigma']:.2f} sigma")
    if max_slope < 0.091:
        print(f"  max reachable:   {max_slope:+.4f}")
        print(f"  deficit:         {0.091 - max_slope:+.4f}")
    print(f"\n  v1 (r_s=Rdisk/2, bug): slope=+0.089, sigma=0.05 (coincidence)")
    print(f"  v2 (rs_tanh):          slope={slope_B:+.4f}, sigma={bm['sigma']:.2f}")


if __name__ == '__main__':
    main()
