#!/usr/bin/env python3
"""
model_b_kcross_origin.py — rs_tanh 版
"""

import os
import sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TA3  = os.path.join(BASE, 'TA3_gc_independent.csv')
MRT  = os.path.join(BASE, 'SPARC_Lelli2016c.mrt')


def load_sparc():
    ta3 = {}
    with open(TA3, 'r') as f:
        hdr = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            row = dict(zip(hdr, vals))
            name = row['galaxy']
            try:
                rs_val = float(row['rs_tanh']) if 'rs_tanh' in row else 0.0
                ta3[name] = {
                    'gc_a0': float(row['gc_over_a0']),
                    'ud': float(row['upsilon_d']),
                    'rs_tanh': rs_val,
                }
            except (ValueError, KeyError):
                continue

    # MRT (whitespace split): 0:Galaxy 1:T 7:L[3.6] 9:Reff 11:Rdisk 13:MHI 15:Vflat
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
                MHI   = float(parts[13])
                vflat = float(parts[15])
            except (ValueError, IndexError):
                continue
            mrt[name] = {'T': T, 'L36': L36, 'r_eff': eff,
                         'v_flat': vflat, 'MHI': MHI}

    galaxies = []
    n_rs_ta3 = 0
    n_rs_fallback = 0
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
            n_rs_fallback += 1

        galaxies.append({
            'name': name,
            'gc_a0': t['gc_a0'],
            'Mstar': Mstar,
            'logMstar': np.log10(Mstar),
            'v_flat': m['v_flat'],
            'hR': hR,
            'r_s': r_s,
            'T_type': m['T'],
            'Sigma_dyn': m['v_flat']**2 / hR,
        })
    print(f"SPARC: {len(galaxies)} galaxies loaded")
    print(f"  r_s source: rs_tanh={n_rs_ta3}, fallback(hR*2)={n_rs_fallback}")
    print(f"  r_s median = {np.median([g['r_s'] for g in galaxies]):.2f} kpc")
    print(f"  r_s range: {min(g['r_s'] for g in galaxies):.2f} - "
          f"{max(g['r_s'] for g in galaxies):.2f} kpc")
    return galaxies


def T_transition(r, r_s, w=3.0):
    arg = np.clip(w * (r / r_s - 1.0), -20, 20)
    return 0.5 * (1.0 + np.tanh(arg))


def compute_kappa_ratio(r_s_kpc, k_cross, f_xi, f_delta=0.70,
                         alpha=2.0, beta_AB=0.503,
                         R_min=30.0, R_max=2000.0, N_R=12):
    xi = f_xi * r_s_kpc
    delta = f_delta * xi
    R_arr = np.logspace(np.log10(R_min), np.log10(R_max), N_R)
    ratios = []

    for R in R_arr:
        L_max = 3000.0
        N_pts = 150
        l_arr = np.linspace(-L_max, L_max, N_pts)
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


def compute_slope(galaxies, key='kappa_ratio'):
    MSTAR_BINS = [(8.5, 10.3), (10.3, 10.6), (10.6, 10.8), (10.8, 11.0)]
    bin_logM = []
    bin_log_kr = []
    for lo, hi in MSTAR_BINS:
        subset = [g for g in galaxies if lo <= g['logMstar'] < hi]
        if len(subset) < 3:
            return np.nan
        mean_logM = np.mean([g['logMstar'] for g in subset])
        mean_log_kr = np.mean([np.log10(g[key]) for g in subset])
        bin_logM.append(mean_logM)
        bin_log_kr.append(mean_log_kr)

    coeffs = np.polyfit(bin_logM, bin_log_kr, 1)
    return coeffs[0]


def main():
    galaxies = load_sparc()

    print("\n" + "="*70)
    print("Part 1: dimensional analysis")
    print("="*70)

    f_delta = 0.70
    k_cross_cl = 403.4
    k_eff_gal = f_delta**2 * 0.5**2 * 10.0
    print(f"  k_eff(galaxy) = {f_delta}^2 * 0.5^2 * 10 = {k_eff_gal:.3f}")
    P_rw_gal = 1 - np.exp(-np.pi * k_eff_gal)
    print(f"  P_rewire(r_s, galaxy) = {P_rw_gal:.3f}")
    f_xi_cl = np.sqrt(k_eff_gal / (f_delta**2 * k_cross_cl))
    print(f"  f_xi(cluster) under k_eff invariance = {f_xi_cl:.4f}")
    print(f"  r_s=300 kpc -> xi = {f_xi_cl*300:.1f} kpc")

    print("\n" + "="*70)
    print("Part 2: scaling candidates scan")
    print("="*70)

    cluster = {
        'r_s': 300.0, 'v_char': 800.0, 'M_total': 5e14,
        'Sigma_dyn': 800**2 / 300, 'k_cross': 403.4,
    }

    r_s_med = np.median([g['r_s'] for g in galaxies])
    v_med = np.median([g['v_flat'] for g in galaxies])
    M_med = np.median([g['Mstar'] for g in galaxies])
    Sig_med = np.median([g['Sigma_dyn'] for g in galaxies])

    print(f"\n  SPARC medians: r_s={r_s_med:.1f} kpc, v={v_med:.0f} km/s, "
          f"logM={np.log10(M_med):.1f}, Sigma={Sig_med:.0f}")
    print(f"  Cluster: r_s={cluster['r_s']:.0f}, v={cluster['v_char']:.0f}, "
          f"logM={np.log10(cluster['M_total']):.1f}")

    scalings = [
        ('k prop r_s',      lambda g: g['r_s'],                  cluster['r_s']),
        ('k prop r_s^1.15', lambda g: g['r_s']**1.15,            cluster['r_s']**1.15),
        ('k prop r_s^2',    lambda g: g['r_s']**2,               cluster['r_s']**2),
        ('k prop v^2',      lambda g: g['v_flat']**2,            cluster['v_char']**2),
        ('k prop M^0.5',    lambda g: g['Mstar']**0.5,           cluster['M_total']**0.5),
        ('k prop Sig_dyn',  lambda g: g['Sigma_dyn'],            cluster['Sigma_dyn']),
        ('k prop r_s*v',    lambda g: g['r_s'] * g['v_flat'],    cluster['r_s'] * cluster['v_char']),
        ('k = const',       lambda g: 1.0,                       1.0),
    ]

    f_xi_fixed = 0.5
    target_slope = 0.091

    print(f"\n  {'Scaling':>20s} {'k_gal(med)':>10s} {'Ratio':>8s} "
          f"{'Slope':>8s} {'Gap':>8s} {'Score':>6s}")
    print("  " + "-"*66)

    best_scaling = None
    best_gap = 999
    best_k_med = 0

    for sname, sfunc, cl_val in scalings:
        C = cluster['k_cross'] / cl_val
        for g in galaxies:
            g['k_cross_pred'] = C * sfunc(g)
            kr = compute_kappa_ratio(g['r_s'], g['k_cross_pred'],
                                     f_xi_fixed, N_R=8)
            g['kappa_ratio'] = kr

        k_gal_med = np.median([g['k_cross_pred'] for g in galaxies])
        ratio = k_gal_med / cluster['k_cross']
        slope = compute_slope(galaxies)
        gap = slope - target_slope if np.isfinite(slope) else 999

        score = '***' if abs(gap) < 0.02 else ('**' if abs(gap) < 0.04 else
                ('*' if abs(gap) < 0.08 else ''))
        print(f"  {sname:>20s} {k_gal_med:>10.1f} {ratio:>8.4f} "
              f"{slope:>+8.4f} {gap:>+8.4f} {score:>6s}")

        if abs(gap) < abs(best_gap):
            best_gap = gap
            best_scaling = sname
            best_k_med = k_gal_med

    print(f"\n  Best: {best_scaling}, k_gal_med={best_k_med:.1f}")

    print("\n" + "="*70)
    print(f"Part 3: detailed analysis of {best_scaling}")
    print("="*70)

    for sname, sfunc, cl_val in scalings:
        if sname == best_scaling:
            C = cluster['k_cross'] / cl_val
            for g in galaxies:
                g['k_cross_best'] = C * sfunc(g)
                g['kappa_ratio_best'] = compute_kappa_ratio(
                    g['r_s'], g['k_cross_best'], f_xi_fixed, N_R=10)
            break

    k_arr = np.array([g['k_cross_best'] for g in galaxies])
    print(f"  k_cross: median={np.median(k_arr):.1f}, "
          f"16-84%={np.percentile(k_arr,16):.1f}-{np.percentile(k_arr,84):.1f}")

    log_k = np.log10(k_arr)
    logM = np.array([g['logMstar'] for g in galaxies])
    props = [
        ('log r_s',       np.log10([g['r_s'] for g in galaxies])),
        ('log v_flat',    np.log10([g['v_flat'] for g in galaxies])),
        ('log M*',        logM),
        ('T_type',        [g['T_type'] for g in galaxies]),
        ('log Sigma_dyn', np.log10([g['Sigma_dyn'] for g in galaxies])),
    ]
    print("\n  k_cross vs properties:")
    for pname, parr in props:
        if np.std(parr) < 1e-9 or np.std(log_k) < 1e-9:
            print(f"    {pname:>15s}: (constant — skipped)")
            continue
        rho, pval = spearmanr(parr, log_k)
        print(f"    {pname:>15s}: rho={rho:+.3f} (p={pval:.2e})")

    slope_final = compute_slope(galaxies, key='kappa_ratio_best')
    total_slope = 0.075 + slope_final
    residual = (0.166 - total_slope) / 0.041
    print(f"\n  Model B slope: {slope_final:+.4f}")
    print(f"  C15 + B: {total_slope:+.4f} vs HSC +0.166±0.041")
    print(f"  Residual: {residual:.2f} sigma")

    print("\n" + "="*70)
    print("Part 4: k_eff invariance test")
    print("="*70)

    log_rs = np.log10([g['r_s'] for g in galaxies])
    log_k_best = np.log10([g['k_cross_best'] for g in galaxies])
    if np.std(log_k_best) > 1e-9:
        p_fit = np.polyfit(log_rs, log_k_best, 1)[0]
    else:
        p_fit = float('nan')
    p_2pt = (np.log10(cluster['k_cross']) - np.log10(best_k_med)) / \
            (np.log10(cluster['r_s']) - np.log10(r_s_med))
    q_xi = (2 - p_2pt) / 2
    print(f"  p (SPARC-only): {p_fit:.3f}")
    print(f"  p (2-point):    {p_2pt:.3f}")
    print(f"  q = (2-p)/2 = {q_xi:.3f} -> xi prop r_s^{q_xi:.2f}")

    xi_gal = 0.5 * r_s_med
    xi_cl = f_xi_cl * cluster['r_s']
    print(f"\n  xi: galaxy={xi_gal:.1f} kpc, cluster={xi_cl:.1f} kpc")

    print("\n" + "="*70)
    print("Part 5: predictive scaling law")
    print("="*70)
    p_exact = (np.log10(403.4) - np.log10(10)) / \
              (np.log10(cluster['r_s']) - np.log10(r_s_med))
    A_exact = 10 / r_s_med**p_exact
    print(f"  k_cross = {A_exact:.4f} * r_s^{p_exact:.3f}")
    print(f"    r_s={r_s_med:.1f} -> k={A_exact*r_s_med**p_exact:.1f}")
    print(f"    r_s={cluster['r_s']:.0f}  -> k={A_exact*cluster['r_s']**p_exact:.1f}")

    if np.std(log_k_best) > 1e-9:
        rho_rs, pv_rs = spearmanr(log_rs, log_k_best)
        print(f"  SPARC Spearman(log k, log r_s) = {rho_rs:+.3f} (p={pv_rs:.2e})")

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    sc = ax.scatter([g['r_s'] for g in galaxies],
               [g['k_cross_best'] for g in galaxies],
               c=[g['T_type'] for g in galaxies], cmap='coolwarm',
               s=20, alpha=0.6, edgecolors='gray', linewidths=0.3)
    ax.scatter([cluster['r_s']], [cluster['k_cross']], marker='*',
               s=300, c='red', edgecolors='black', zorder=10, label='Cluster')
    r_plot = np.logspace(0, 3, 100)
    ax.plot(r_plot, A_exact * r_plot**p_exact, 'k-', lw=2,
            label=f'k = {A_exact:.3f} * r_s^{p_exact:.2f}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('r_s [kpc]'); ax.set_ylabel('k_cross')
    ax.set_title('(a) k_cross scaling')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax).set_label('T-type')

    ax = axes[0, 1]
    for mb, (lo, hi) in enumerate([(8.5,10.3),(10.3,10.6),(10.6,10.8),(10.8,11.0)]):
        subset = [g['kappa_ratio_best'] for g in galaxies if lo <= g['logMstar'] < hi]
        if subset:
            ax.hist(np.log10(subset), bins=20, alpha=0.5,
                    label=f'Bin{mb+1} (N={len(subset)})')
    ax.set_xlabel('log10(kappa_total / kappa_A)')
    ax.set_ylabel('Count')
    ax.set_title('(b) kappa_ratio by M*')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    r_s_arr = np.logspace(0, 3, 100)
    for f_xi_test in [0.3, 0.5, 0.8]:
        P_arr = [1 - np.exp(-np.pi * f_delta**2 * f_xi_test**2
                             * A_exact * rs**p_exact) for rs in r_s_arr]
        ax.plot(r_s_arr, P_arr, lw=2, label=f'f_xi={f_xi_test}')
    ax.axvline(r_s_med, color='blue', ls='--', lw=1, alpha=0.5, label='SPARC')
    ax.axvline(cluster['r_s'], color='red', ls='--', lw=1, alpha=0.5, label='Cluster')
    ax.set_xscale('log')
    ax.set_xlabel('r_s [kpc]'); ax.set_ylabel('P_rewire(r_s)')
    ax.set_title('(c) P_rewire at r_s')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = ['C15', 'Model B', 'C15+B', 'HSC']
    vals = [0.075, slope_final, 0.075 + slope_final, 0.166]
    errs = [0, 0, 0, 0.041]
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#F44336']
    ax.bar(range(4), vals, color=colors, alpha=0.8,
           yerr=errs, capsize=5, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('gc-M* slope'); ax.set_title('(d) Slope decomposition')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f'{v:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    figpath = os.path.join(BASE, 'model_b_kcross_origin.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {figpath}")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"  k_cross = {A_exact:.4f} * r_s^{p_exact:.3f}")
    print(f"  slope contribution: {slope_final:+.4f}")
    print(f"  C15+B: {0.075+slope_final:+.4f} vs HSC +0.166±0.041")
    print(f"  Residual: {residual:.2f} sigma")


if __name__ == '__main__':
    main()
