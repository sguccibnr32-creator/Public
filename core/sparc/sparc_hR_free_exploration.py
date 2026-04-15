#!/usr/bin/env python3
"""
sparc_hR_free_exploration.py — h_R非依存のg_c-物性関係の探索
"""
import os, sys, warnings
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

TA3_FILE = "TA3_gc_independent.csv"
MASTER_FILE = "SPARC_Lelli2016c.mrt"
a0 = 1.2e-10


def load_ta3(filepath):
    data = {}
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            name = parts[0].strip()
            try:
                data[name] = {
                    'gc_over_a0': float(parts[1]),
                    'v_flat': float(parts[2]),
                    'rs_tanh': float(parts[3]),
                    'upsilon_d': float(parts[4]),
                    'chi2_dof': float(parts[5]),
                }
            except ValueError:
                continue
    return data


def load_sparc_master(filepath):
    """Fixed-width MRT parser for SPARC_Lelli2016c.mrt"""
    props = {}
    if not os.path.exists(filepath):
        return props

    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    n_dash = 0
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('---'):
            n_dash += 1
            if n_dash >= 3:
                data_start_idx = i + 1
                break

    if data_start_idx is None:
        return props

    # split-based parser. Field order:
    # [0]Galaxy [1]T [2]D [3]e_D [4]f_D [5]Inc [6]e_Inc [7]L36 [8]e_L36
    # [9]Reff [10]SBeff [11]Rdisk [12]SBdisk [13]MHI [14]RHI [15]Vflat
    for line in lines[data_start_idx:]:
        parts = line.split()
        if len(parts) < 16:
            continue
        try:
            name = parts[0].strip()
            info = {'T_type': float(parts[1])}
            info['D_Mpc'] = float(parts[2])
            info['Inc'] = float(parts[5])
            info['L36'] = float(parts[7])
            info['Reff'] = float(parts[9])
            info['SBeff'] = float(parts[10])
            info['Rdisk'] = float(parts[11])
            info['SBdisk'] = float(parts[12])
            info['Vf'] = float(parts[15])
            props[name] = info
        except (ValueError, IndexError):
            continue
    return props


def fit_power_law(x, y):
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept, r, p, se = stats.linregress(lx, ly)
    resid_std = np.std(ly - (intercept + slope * lx))
    return {
        'slope': slope, 'intercept': intercept, 'r2': r**2,
        'r': r, 'p': p, 'se': se, 'resid_std': resid_std,
        'N': len(x)
    }


def test_alpha05_analogy(slope, se, N):
    t = (slope - 0.5) / se
    p = 2 * stats.t.sf(abs(t), df=N - 2)
    return p


def main():
    print("=" * 70)
    print("Track 2: h_R-free g_c-property relation exploration")
    print("=" * 70)

    print("\n[1] Loading data...")
    ta3 = load_ta3(TA3_FILE)
    print(f"  TA3: {len(ta3)} galaxies")
    master = load_sparc_master(MASTER_FILE)
    print(f"  Master: {len(master)} galaxies")

    galaxies = []
    for name in ta3:
        d = ta3[name]
        gc = d['gc_over_a0'] * a0
        vflat = d['v_flat']
        Yd = d['upsilon_d']

        if gc <= 0 or vflat <= 0:
            continue

        entry = {
            'name': name, 'gc': gc, 'gc_a0': d['gc_over_a0'],
            'vflat': vflat, 'Yd': Yd, 'rs': d['rs_tanh'],
        }

        if name in master:
            m = master[name]
            entry['T_type'] = m.get('T_type', np.nan)
            entry['L36'] = m.get('L36', np.nan)
            entry['SBdisk'] = m.get('SBdisk', np.nan)
            entry['Rdisk'] = m.get('Rdisk', np.nan)

            if not np.isnan(m.get('L36', np.nan)):
                entry['Mstar'] = Yd * m['L36'] * 1e9
            else:
                entry['Mstar'] = np.nan

            if not np.isnan(m.get('Rdisk', np.nan)) and m['Rdisk'] > 0:
                hR_kpc = m['Rdisk']
                entry['GS0'] = (vflat * 1e3)**2 / (hR_kpc * 3.086e19)
                entry['hR'] = hR_kpc
            else:
                entry['GS0'] = np.nan
                entry['hR'] = np.nan
        else:
            for k in ['T_type', 'L36', 'SBdisk', 'Rdisk', 'Mstar', 'GS0', 'hR']:
                entry[k] = np.nan

        galaxies.append(entry)

    print(f"  Combined: {len(galaxies)} galaxies")

    gc_a0 = np.array([g['gc_a0'] for g in galaxies])
    gc = np.array([g['gc'] for g in galaxies])
    vflat = np.array([g['vflat'] for g in galaxies])
    Yd = np.array([g['Yd'] for g in galaxies])
    T_type = np.array([g['T_type'] for g in galaxies])
    L36 = np.array([g['L36'] for g in galaxies])
    SBdisk = np.array([g['SBdisk'] for g in galaxies])
    Rdisk = np.array([g['Rdisk'] for g in galaxies])
    Mstar = np.array([g['Mstar'] for g in galaxies])
    GS0 = np.array([g['GS0'] for g in galaxies])

    print("\n[2] Testing h_R-free proxies...")
    print(f"\n  {'Proxy':<40} {'slope':>7} {'SE':>6} {'r^2':>6} {'sigma':>6} {'p(0.5)':>8} {'N':>4}")
    print("  " + "-" * 77)

    results = {}

    mask = np.isfinite(GS0) & (GS0 > 0) & (gc > 0)
    if mask.sum() > 10:
        r = fit_power_law(GS0[mask], gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['GS0_standard'] = r
        print(f"  {'(REF) G*Sigma0 = vflat^2/hR':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = (vflat > 0) & (gc > 0)
    if mask.sum() > 10:
        r = fit_power_law(vflat[mask], gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['vflat_only'] = r
        print(f"  {'v_flat only':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = (vflat > 0) & (gc > 0)
    if mask.sum() > 10:
        r = fit_power_law(vflat[mask]**2, gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['vflat2'] = r
        print(f"  {'v_flat^2':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(Mstar) & (Mstar > 0) & (gc > 0)
    if mask.sum() > 10:
        r = fit_power_law(Mstar[mask], gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['Mstar'] = r
        print(f"  {'M_star = Yd * L[3.6]':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(L36) & (L36 > 0) & (gc > 0)
    if mask.sum() > 10:
        r = fit_power_law(L36[mask], gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['L36_only'] = r
        print(f"  {'L[3.6] only (Yd-free!)':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(SBdisk) & (SBdisk > 0) & (gc > 0)
    if mask.sum() > 10:
        SB_flux = 10**(-0.4 * (SBdisk[mask] - 20))
        r = fit_power_law(SB_flux, gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['SBdisk'] = r
        print(f"  {'SB_disk (flux, h_R-free)':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
    if mask.sum() > 10:
        proxy = vflat[mask] * np.sqrt(L36[mask])
        r = fit_power_law(proxy, gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['vflat_sqrtL'] = r
        print(f"  {'v_flat * sqrt(L[3.6])':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
    if mask.sum() > 10:
        proxy = vflat[mask]**4 / L36[mask]
        r = fit_power_law(proxy, gc[mask])
        p05 = test_alpha05_analogy(r['slope'], r['se'], r['N'])
        results['vflat4_L'] = r
        print(f"  {'v_flat^4 / L[3.6]':<40} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>8.4f} {r['N']:>4}")

    mask = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
    if mask.sum() > 20:
        log_gc = np.log10(gc[mask] / a0)
        log_vf = np.log10(vflat[mask])
        log_L = np.log10(L36[mask])

        X = np.column_stack([np.ones(mask.sum()), log_vf, log_L])
        beta = np.linalg.lstsq(X, log_gc, rcond=None)[0]
        pred = X @ beta
        resid_std = np.std(log_gc - pred)

        ss_res = np.sum((log_gc - pred)**2)
        ss_tot = np.sum((log_gc - log_gc.mean())**2)
        r2_2var = 1 - ss_res / ss_tot

        results['vflat_L_2var'] = {
            'a_vflat': beta[1], 'b_L36': beta[2], 'intercept': beta[0],
            'r2': r2_2var, 'resid_std': resid_std, 'N': mask.sum()
        }
        print(f"\n  {'Optimal 2-var: vflat^a * L^b':<40} a={beta[1]:.3f} b={beta[2]:.3f} r2={r2_2var:.3f} sigma={resid_std:.3f}  N={mask.sum()}")

        proxy_gm = vflat[mask]**2 / np.sqrt(L36[mask])
        r_gm = fit_power_law(proxy_gm, gc[mask])
        p05_gm = test_alpha05_analogy(r_gm['slope'], r_gm['se'], r_gm['N'])
        results['vflat2_sqrtL'] = r_gm
        print(f"  {'v_flat^2 / sqrt(L[3.6])':<40} {r_gm['slope']:>7.3f} {r_gm['se']:>6.3f} {r_gm['r2']:>6.3f} {r_gm['resid_std']:>6.3f} {p05_gm:>8.4f} {r_gm['N']:>4}")

    print("\n[3] Spearman correlations with log(gc/a0)...")
    log_gc = np.log10(gc_a0)

    correlations = []
    for label, arr in [('log(v_flat)', np.log10(vflat)),
                        ('T_type', T_type),
                        ('log(L[3.6])', np.log10(np.where(L36 > 0, L36, np.nan))),
                        ('SBdisk', SBdisk),
                        ('log(Rdisk)', np.log10(np.where(Rdisk > 0, Rdisk, np.nan))),
                        ('log(Yd)', np.log10(Yd)),
                        ('log(M_star)', np.log10(np.where(Mstar > 0, Mstar, np.nan)))]:
        mask_ok = np.isfinite(arr) & np.isfinite(log_gc)
        if mask_ok.sum() > 10:
            rho, p = stats.spearmanr(arr[mask_ok], log_gc[mask_ok])
            correlations.append((label, rho, p, mask_ok.sum()))

    print(f"\n  {'Variable':<25} {'rho':>7} {'p':>12} {'N':>5}")
    print("  " + "-" * 50)
    for label, rho, p, n in sorted(correlations, key=lambda x: -abs(x[1])):
        print(f"  {label:<25} {rho:>+7.3f} {p:>12.2e} {n:>5}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    ref_sigma = results.get('GS0_standard', {}).get('resid_std', 0.313)
    print(f"\n  Reference: G*Sigma0 (with h_R) sigma = {ref_sigma:.3f} dex")

    print("\n  h_R-free proxy ranking (by residual scatter):")
    proxy_ranking = []
    for name, r in results.items():
        if name == 'GS0_standard' or name == 'vflat_L_2var':
            continue
        if 'resid_std' in r:
            proxy_ranking.append((name, r['resid_std'], r.get('r2', 0)))

    for name, sigma, r2 in sorted(proxy_ranking, key=lambda x: x[1]):
        penalty = (sigma - ref_sigma) / ref_sigma * 100
        verdict = "GOOD" if sigma < ref_sigma * 1.3 else "MARGINAL" if sigma < ref_sigma * 1.6 else "POOR"
        print(f"    {name:<30} sigma={sigma:.3f} r2={r2:.3f} penalty={penalty:+.0f}% [{verdict}]")

    if 'vflat_L_2var' in results:
        r = results['vflat_L_2var']
        print(f"\n  Best 2-variable model (h_R-free):")
        print(f"    log(gc/a0) = {r['intercept']:.3f} + {r['a_vflat']:.3f}*log(vflat) + {r['b_L36']:.3f}*log(L36)")
        print(f"    r2 = {r['r2']:.3f}, sigma = {r['resid_std']:.3f} dex")
        print(f"    vs G*Sigma0: penalty = {(r['resid_std']-ref_sigma)/ref_sigma*100:+.0f}%")

    print("\n  Practical implications:")
    best_hRfree = min(proxy_ranking, key=lambda x: x[1]) if proxy_ranking else None
    if best_hRfree and best_hRfree[1] < ref_sigma * 1.5:
        print(f"    [OK] {best_hRfree[0]} achieves sigma={best_hRfree[1]:.3f} (< 1.5x reference)")
        print(f"    -> MaNGA/PROBES verification is FEASIBLE with this proxy")
    else:
        print(f"    [!] All h_R-free proxies have >50% scatter penalty")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        ax = axes[0, 0]
        mask = np.isfinite(GS0) & (GS0 > 0) & (gc > 0)
        if mask.sum() > 0:
            ax.scatter(np.log10(GS0[mask]/a0), np.log10(gc[mask]/a0), s=8, alpha=0.4)
            r = results.get('GS0_standard', {})
            if r:
                xr = np.linspace(-1.5, 1.5, 50)
                ax.plot(xr, r['intercept'] + r['slope']*xr, 'r-', label=f"slope={r['slope']:.3f}")
                ax.plot(xr, r['intercept'] + 0.5*xr, 'g--', alpha=0.5, label='slope=0.5')
            ax.set_xlabel('log(G*Sigma0/a0)')
            ax.set_ylabel('log(gc/a0)')
            ax.set_title(f'REF: G*Sigma0 (with h_R)\nsigma={r.get("resid_std",0):.3f}')
            ax.legend(fontsize=8)

        ax = axes[0, 1]
        mask = (vflat > 0) & (gc > 0)
        ax.scatter(np.log10(vflat[mask]), np.log10(gc[mask]/a0), s=8, alpha=0.4)
        r = results.get('vflat_only', {})
        if r:
            xr = np.linspace(1.0, 2.6, 50)
            ax.plot(xr, r['intercept'] + r['slope']*xr, 'r-', label=f"slope={r['slope']:.3f}")
        ax.set_xlabel('log(v_flat [km/s])')
        ax.set_ylabel('log(gc/a0)')
        ax.set_title(f'v_flat only (h_R-FREE)\nsigma={r.get("resid_std",0):.3f}')
        ax.legend(fontsize=8)

        ax = axes[0, 2]
        mask = np.isfinite(L36) & (L36 > 0) & (gc > 0)
        if mask.sum() > 0:
            ax.scatter(np.log10(L36[mask]), np.log10(gc[mask]/a0), s=8, alpha=0.4)
            r = results.get('L36_only', {})
            if r:
                xr = np.linspace(-3, 2, 50)
                ax.plot(xr, r['intercept'] + r['slope']*xr, 'r-', label=f"slope={r['slope']:.3f}")
            ax.set_xlabel('log(L[3.6] [10^9 Lsun])')
            ax.set_ylabel('log(gc/a0)')
            ax.set_title(f'L[3.6] only (Yd-FREE, h_R-FREE)\nsigma={r.get("resid_std",0):.3f}')
            ax.legend(fontsize=8)

        ax = axes[1, 0]
        mask = np.isfinite(Mstar) & (Mstar > 0) & (gc > 0)
        if mask.sum() > 0:
            ax.scatter(np.log10(Mstar[mask]), np.log10(gc[mask]/a0), s=8, alpha=0.4)
            r = results.get('Mstar', {})
            if r:
                xr = np.linspace(6, 12, 50)
                ax.plot(xr, r['intercept'] + r['slope']*xr, 'r-', label=f"slope={r['slope']:.3f}")
            ax.set_xlabel('log(M_star [Msun])')
            ax.set_ylabel('log(gc/a0)')
            ax.set_title(f'M_star (h_R-FREE)\nsigma={r.get("resid_std",0):.3f}')
            ax.legend(fontsize=8)

        ax = axes[1, 1]
        mask = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
        if mask.sum() > 0:
            proxy = vflat[mask]**2 / np.sqrt(L36[mask])
            ax.scatter(np.log10(proxy), np.log10(gc[mask]/a0), s=8, alpha=0.4)
            r = results.get('vflat2_sqrtL', {})
            if r:
                xr = np.linspace(1, 6, 50)
                ax.plot(xr, r['intercept'] + r['slope']*xr, 'r-', label=f"slope={r['slope']:.3f}")
                ax.plot(xr, r['intercept'] + 0.5*xr, 'g--', alpha=0.5, label='slope=0.5')
            ax.set_xlabel('log(v_flat^2 / sqrt(L[3.6]))')
            ax.set_ylabel('log(gc/a0)')
            ax.set_title(f'v_flat^2/sqrt(L) (h_R-FREE)\nsigma={r.get("resid_std",0):.3f}')
            ax.legend(fontsize=8)

        ax = axes[1, 2]
        names_bar = ['GS0_standard', 'vflat_only', 'L36_only', 'Mstar', 'vflat2_sqrtL', 'vflat_sqrtL']
        labels_bar = ['G*S0\n(with h_R)', 'v_flat\nonly', 'L[3.6]\nonly', 'M_star', 'vf^2/sqrtL', 'vf*sqrtL']
        sigmas = []
        colors_bar = []
        for n in names_bar:
            if n in results:
                s = results[n]['resid_std']
                sigmas.append(s)
                colors_bar.append('green' if s < ref_sigma * 1.3 else 'orange' if s < ref_sigma * 1.6 else 'red')
            else:
                sigmas.append(0)
                colors_bar.append('gray')
        ax.bar(range(len(sigmas)), sigmas, color=colors_bar, alpha=0.7)
        ax.axhline(ref_sigma, color='blue', ls='--', label=f'ref={ref_sigma:.3f}')
        ax.axhline(ref_sigma * 1.3, color='orange', ls=':', alpha=0.5, label='1.3x ref')
        ax.set_xticks(range(len(labels_bar)))
        ax.set_xticklabels(labels_bar, fontsize=8)
        ax.set_ylabel('Residual scatter [dex]')
        ax.set_title('Proxy comparison')
        ax.legend(fontsize=8)

        plt.suptitle('h_R-free proxy exploration for g_c prediction', fontsize=14)
        plt.tight_layout()
        outpath = "sparc_hR_free_exploration.png"
        plt.savefig(outpath, dpi=150)
        print(f"\n  Figure saved: {outpath}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available)")

    import json
    out = {}
    for name, r in results.items():
        out[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in r.items()}
    json_path = "sparc_hR_free_exploration_results.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
