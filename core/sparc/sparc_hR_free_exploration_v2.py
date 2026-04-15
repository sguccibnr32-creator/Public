#!/usr/bin/env python3
"""
sparc_hR_free_exploration_v2.py — h_R非依存プロキシ探索（修正版）
"""
import os, sys, glob, warnings, json
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

SPARC_DIR = "Rotmod_LTG"
TA3_FILE = "TA3_gc_independent.csv"
MASTER_FILE = "SPARC_Lelli2016c.mrt"
PHASE1_FILE = "phase1/sparc_results.csv"
a0 = 1.2e-10


def load_phase1(filepath):
    """phase1/sparc_results.csv: galaxy, ud, vflat (real values, not floored)"""
    data = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')
        try:
            i_name = header.index('galaxy')
            i_ud = header.index('ud')
            i_vf = header.index('vflat')
        except ValueError:
            return data
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(i_name, i_ud, i_vf):
                continue
            try:
                name = parts[i_name].strip()
                ud = float(parts[i_ud])
                vf = float(parts[i_vf])
                if name and ud > 0 and vf > 0:
                    data[normalize_name(name)] = {'ud': ud, 'vflat': vf}
            except ValueError:
                continue
    return data


def normalize_name(name):
    return name.strip().replace(' ', '').replace('_', '').replace('-', '').upper()


def load_ta3(filepath):
    data = {}
    with open(filepath, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            name = parts[0].strip()
            try:
                data[normalize_name(name)] = {
                    'name_orig': name,
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
    """split-based parser"""
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

    for line in lines[data_start_idx:]:
        parts = line.split()
        if len(parts) < 16:
            continue
        try:
            name = parts[0].strip()
            info = {'name_orig': name, 'T_type': float(parts[1])}
            info['L36'] = float(parts[7])
            info['Rdisk_master'] = float(parts[11])
            info['SBdisk'] = float(parts[12])
            info['Vf_master'] = float(parts[15])
            props[normalize_name(name)] = info
        except (ValueError, IndexError):
            continue
    return props


def load_rotmod(filepath):
    R, Vobs, errV, Vgas, Vdisk, Vbul = [], [], [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    R.append(float(parts[0]))
                    Vobs.append(float(parts[1]))
                    errV.append(max(float(parts[2]), 1.0))
                    Vgas.append(float(parts[3]))
                    Vdisk.append(float(parts[4]))
                    Vbul.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    return {k: np.array(v) for k, v in
            [('R', R), ('Vobs', Vobs), ('errV', errV),
             ('Vgas', Vgas), ('Vdisk', Vdisk), ('Vbul', Vbul)]}


def compute_hR_pipeline(rotmod, Yd):
    R = rotmod['R']
    Vdisk = rotmod['Vdisk']
    # Match original pipeline: filter R > 0.01 first
    mask = R > 0.01
    R_m = R[mask]
    Vd_m = Vdisk[mask]
    if len(R_m) < 5:
        return None
    profile = np.sqrt(max(Yd, 0.01)) * np.abs(Vd_m)
    i_pk = np.argmax(profile)
    r_pk = R_m[i_pk]
    if r_pk >= 0.9 * R_m.max() or r_pk < 0.01:
        return None
    hR = r_pk / 2.15
    return hR if hR > 0 else None


def fit_power_law(x, y):
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept, r, p, se = stats.linregress(lx, ly)
    resid = np.std(ly - (intercept + slope * lx))
    return {'slope': slope, 'intercept': intercept, 'r2': r**2,
            'r': r, 'p_slope': p, 'se': se, 'resid_std': resid, 'N': len(x)}


def p_test(slope, se, N, target=0.5):
    t = (slope - target) / se
    return 2 * stats.t.sf(abs(t), df=N - 2)


def main():
    print("=" * 70)
    print("Track 2 v2: h_R-free proxy exploration (fixed)")
    print("=" * 70)

    print("\n[1] Loading...")
    ta3 = load_ta3(TA3_FILE)
    print(f"  TA3: {len(ta3)}")
    master = load_sparc_master(MASTER_FILE)
    print(f"  Master: {len(master)}")
    phase1 = load_phase1(PHASE1_FILE)
    print(f"  Phase1: {len(phase1)}")

    rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    if not rotmod_files:
        rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*.dat"))
    file_map = {}
    for fp in rotmod_files:
        gname = os.path.basename(fp).replace('_rotmod.dat', '').replace('.dat', '')
        file_map[normalize_name(gname)] = fp
    print(f"  Rotmod: {len(file_map)}")

    print("\n[2] Building galaxy dataset...")
    data = []

    for nkey in ta3:
        if nkey not in file_map:
            continue
        d = ta3[nkey]
        if d['gc_over_a0'] <= 0:
            continue

        # Use phase1 vflat/ud if available (real values, not TA3-floored)
        p1 = phase1.get(nkey, {})
        vf_use = p1.get('vflat', d['v_flat'])
        Yd_use = p1.get('ud', d['upsilon_d'])
        if vf_use <= 5.0:
            continue

        rotmod = load_rotmod(file_map[nkey])
        if len(rotmod['R']) < 5:
            continue

        Yd = Yd_use
        hR = compute_hR_pipeline(rotmod, Yd)
        if hR is None:
            continue

        gc = d['gc_over_a0'] * a0
        vf = vf_use
        GS0 = (vf * 1e3)**2 / (hR * 3.086e19)

        entry = {
            'name': nkey, 'gc': gc, 'gc_a0': d['gc_over_a0'],
            'vflat': vf, 'Yd': Yd, 'hR': hR, 'GS0': GS0,
        }

        if nkey in master:
            m = master[nkey]
            entry['T_type'] = m.get('T_type', np.nan)
            entry['L36'] = m.get('L36', np.nan)
            entry['SBdisk'] = m.get('SBdisk', np.nan)
            L_val = m.get('L36', np.nan)
            entry['Mstar'] = Yd * L_val * 1e9 if not np.isnan(L_val) else np.nan
        else:
            entry['T_type'] = np.nan
            entry['L36'] = np.nan
            entry['SBdisk'] = np.nan
            entry['Mstar'] = np.nan

        data.append(entry)

    print(f"  Dataset: {len(data)} galaxies (with pipeline h_R)")

    gc = np.array([d['gc'] for d in data])
    gc_a0 = np.array([d['gc_a0'] for d in data])
    vflat = np.array([d['vflat'] for d in data])
    Yd = np.array([d['Yd'] for d in data])
    hR = np.array([d['hR'] for d in data])
    GS0 = np.array([d['GS0'] for d in data])
    T_type = np.array([d['T_type'] for d in data])
    L36 = np.array([d['L36'] for d in data])
    SBdisk = np.array([d['SBdisk'] for d in data])
    Mstar = np.array([d['Mstar'] for d in data])

    print("\n[3] Testing proxies...")
    print(f"\n  {'Proxy':<35} {'slope':>7} {'SE':>6} {'r2':>6} {'sigma':>6} {'p(0.5)':>9} {'N':>4}")
    print("  " + "-" * 73)

    results = {}

    mask = (GS0 > 0) & (gc > 0)
    r = fit_power_law(GS0[mask], gc[mask])
    p05 = p_test(r['slope'], r['se'], r['N'])
    results['GS0_pipeline'] = {**r, 'p_05': p05}
    ref_sigma = r['resid_std']
    print(f"  {'(REF) G*S0 = vf^2/(Vdpk/2.15)':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    mask = (vflat > 0) & (gc > 0)
    r = fit_power_law(vflat[mask], gc[mask])
    p05 = p_test(r['slope'], r['se'], r['N'])
    results['vflat'] = {**r, 'p_05': p05}
    print(f"  {'v_flat only':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    r = fit_power_law(vflat[mask]**2, gc[mask])
    p05 = p_test(r['slope'], r['se'], r['N'])
    results['vflat2'] = {**r, 'p_05': p05}
    print(f"  {'v_flat^2':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    mask_L = np.isfinite(L36) & (L36 > 0) & (gc > 0)
    if mask_L.sum() > 10:
        r = fit_power_law(L36[mask_L], gc[mask_L])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['L36'] = {**r, 'p_05': p05}
        print(f"  {'L[3.6] only (Yd-free, hR-free)':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    mask_M = np.isfinite(Mstar) & (Mstar > 0) & (gc > 0)
    if mask_M.sum() > 10:
        r = fit_power_law(Mstar[mask_M], gc[mask_M])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['Mstar'] = {**r, 'p_05': p05}
        print(f"  {'M_star = Yd * L[3.6]':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    mask_SB = np.isfinite(SBdisk) & (SBdisk > 0) & (gc > 0)
    if mask_SB.sum() > 10:
        SB_flux = 10**(-0.4 * (SBdisk[mask_SB] - 20))
        r = fit_power_law(SB_flux, gc[mask_SB])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['SBdisk'] = {**r, 'p_05': p05}
        print(f"  {'SB_disk (flux, hR-free)':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    mask_vL = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
    if mask_vL.sum() > 10:
        proxy = vflat[mask_vL] * np.sqrt(L36[mask_vL])
        r = fit_power_law(proxy, gc[mask_vL])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['vflat_sqrtL'] = {**r, 'p_05': p05}
        print(f"  {'v_flat * sqrt(L[3.6])':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    if mask_vL.sum() > 10:
        proxy = vflat[mask_vL]**2 / np.sqrt(L36[mask_vL])
        r = fit_power_law(proxy, gc[mask_vL])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['vflat2_sqrtL'] = {**r, 'p_05': p05}
        print(f"  {'v_flat^2 / sqrt(L[3.6])':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    if mask_vL.sum() > 10:
        proxy = vflat[mask_vL]**4 / L36[mask_vL]
        r = fit_power_law(proxy, gc[mask_vL])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results['vflat4_L'] = {**r, 'p_05': p05}
        print(f"  {'v_flat^4 / L[3.6]':<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    if mask_vL.sum() > 20:
        log_gc = np.log10(gc[mask_vL] / a0)
        log_vf = np.log10(vflat[mask_vL])
        log_L = np.log10(L36[mask_vL])

        X = np.column_stack([np.ones(mask_vL.sum()), log_vf, log_L])
        beta = np.linalg.lstsq(X, log_gc, rcond=None)[0]
        pred = X @ beta
        resid_std_2v = np.std(log_gc - pred)
        ss_res = np.sum((log_gc - pred)**2)
        ss_tot = np.sum((log_gc - log_gc.mean())**2)
        r2_2v = 1 - ss_res / ss_tot

        results['2var_vf_L'] = {
            'a_vflat': float(beta[1]), 'b_L36': float(beta[2]),
            'intercept': float(beta[0]), 'r2': float(r2_2v),
            'resid_std': float(resid_std_2v), 'N': int(mask_vL.sum())
        }
        print(f"\n  2-var optimal: log(gc/a0) = {beta[0]:.2f} + {beta[1]:.3f}*log(vf) + {beta[2]:.3f}*log(L)")
        print(f"    r2={r2_2v:.3f}, sigma={resid_std_2v:.3f}, N={mask_vL.sum()}")

    mask_3v = mask_vL & np.isfinite(T_type)
    if mask_3v.sum() > 20:
        log_gc = np.log10(gc[mask_3v] / a0)
        log_vf = np.log10(vflat[mask_3v])
        log_L = np.log10(L36[mask_3v])
        T = T_type[mask_3v]

        X = np.column_stack([np.ones(mask_3v.sum()), log_vf, log_L, T])
        beta = np.linalg.lstsq(X, log_gc, rcond=None)[0]
        pred = X @ beta
        resid_std_3v = np.std(log_gc - pred)
        r2_3v = 1 - np.sum((log_gc - pred)**2) / np.sum((log_gc - log_gc.mean())**2)

        results['3var_vf_L_T'] = {
            'a_vflat': float(beta[1]), 'b_L36': float(beta[2]),
            'c_T': float(beta[3]), 'intercept': float(beta[0]),
            'r2': float(r2_3v), 'resid_std': float(resid_std_3v),
            'N': int(mask_3v.sum())
        }
        print(f"\n  3-var: + {beta[3]:.3f}*T_type")
        print(f"    r2={r2_3v:.3f}, sigma={resid_std_3v:.3f}, N={mask_3v.sum()}")

    print("\n[4] Spearman correlations with log(gc/a0)...")
    log_gc_all = np.log10(gc_a0)
    print(f"\n  {'Variable':<25} {'rho':>7} {'p':>12} {'N':>5}")
    print("  " + "-" * 50)
    for label, arr in [('log(M_star)', np.log10(np.where(Mstar > 0, Mstar, np.nan))),
                        ('log(L[3.6])', np.log10(np.where(L36 > 0, L36, np.nan))),
                        ('T_type', T_type),
                        ('log(v_flat)', np.log10(vflat)),
                        ('SBdisk [mag]', SBdisk),
                        ('log(h_R)', np.log10(hR)),
                        ('log(Yd)', np.log10(Yd)),
                        ('log(G*S0)', np.log10(GS0))]:
        m = np.isfinite(arr) & np.isfinite(log_gc_all)
        if m.sum() > 10:
            rho, p = stats.spearmanr(arr[m], log_gc_all[m])
            print(f"  {label:<25} {rho:>+7.3f} {p:>12.2e} {m.sum():>5}")

    print("\n[5] Geometric mean relation check...")
    mask_hL = np.isfinite(L36) & (L36 > 0) & (hR > 0)
    if mask_hL.sum() > 10:
        r_hL = fit_power_law(L36[mask_hL], hR[mask_hL])
        print(f"  h_R vs L[3.6]: slope={r_hL['slope']:.3f}, r2={r_hL['r2']:.3f}")

        r_vL = fit_power_law(L36[mask_hL], vflat[mask_hL])
        print(f"  v_flat vs L[3.6]: slope={r_vL['slope']:.3f}, r2={r_vL['r2']:.3f}")

        gs0_pred_slope = 2 * r_vL['slope'] - r_hL['slope']
        print(f"  -> G*S0 ~ L^(2*{r_vL['slope']:.2f} - {r_hL['slope']:.2f}) = L^{gs0_pred_slope:.2f}")

        alpha_GS0 = results.get('GS0_pipeline', {}).get('slope', 0.545)
        gc_L_predicted = alpha_GS0 * gs0_pred_slope
        gc_L_observed = results.get('L36', {}).get('slope', 0)
        print(f"  -> If alpha(GS0)={alpha_GS0:.3f}, predict gc ~ L^{gc_L_predicted:.3f}")
        print(f"     Observed: gc ~ L^{gc_L_observed:.3f}")
        match = 'YES' if abs(gc_L_predicted - gc_L_observed) < 0.05 else 'PARTIAL' if abs(gc_L_predicted - gc_L_observed) < 0.1 else 'NO'
        print(f"     Match: {match}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    print(f"\n  Reference: G*S0 (pipeline h_R) sigma={ref_sigma:.3f}, N={results['GS0_pipeline']['N']}")
    print(f"  Reference alpha(GS0) = {results['GS0_pipeline']['slope']:.3f}")

    print(f"\n  h_R-free proxy ranking:")
    ranking = []
    for name in ['L36', 'Mstar', 'vflat_sqrtL', 'vflat2_sqrtL', 'SBdisk', 'vflat', 'vflat2', 'vflat4_L']:
        if name in results:
            r = results[name]
            penalty = (r['resid_std'] - ref_sigma) / ref_sigma * 100
            ranking.append((name, r['resid_std'], r['r2'], penalty))
    for name, sigma, r2, penalty in sorted(ranking, key=lambda x: x[1]):
        tag = "BETTER" if penalty < -10 else "~SAME" if abs(penalty) < 15 else "WORSE"
        print(f"    {name:<25} sigma={sigma:.3f} r2={r2:.3f} penalty={penalty:+.0f}% [{tag}]")

    if '2var_vf_L' in results:
        r = results['2var_vf_L']
        penalty = (r['resid_std'] - ref_sigma) / ref_sigma * 100
        print(f"    {'2var(vf,L)':<25} sigma={r['resid_std']:.3f} r2={r['r2']:.3f} penalty={penalty:+.0f}%")

    print(f"\n  Practical verdict for MaNGA/PROBES extension:")
    best = min(ranking, key=lambda x: x[1]) if ranking else None
    if best and best[1] < ref_sigma * 1.5:
        print(f"    [OK] {best[0]} (sigma={best[1]:.3f}) enables independent verification")
    else:
        print(f"    [!] No h_R-free proxy matches G*S0 precision")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        def plot_proxy(ax, x, y, res, xlabel, title):
            ax.scatter(np.log10(x), np.log10(y / a0), s=8, alpha=0.4)
            if res:
                xr = np.linspace(np.log10(x).min() - 0.2, np.log10(x).max() + 0.2, 50)
                ax.plot(xr, res['intercept'] + res['slope'] * xr, 'r-',
                        label=f"slope={res['slope']:.3f}")
                ax.set_title(f"{title}\nsigma={res.get('resid_std', 0):.3f}, r2={res.get('r2', 0):.3f}")
            else:
                ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('log(gc/a0)')
            ax.legend(fontsize=8)

        plot_proxy(axes[0, 0], GS0, gc, results.get('GS0_pipeline'),
                   'log(G*Sigma0)', 'REF: G*Sigma0 (pipeline h_R)')
        if 'GS0_pipeline' in results:
            xr = np.linspace(-1.5, 1.5, 50)
            axes[0, 0].plot(xr, results['GS0_pipeline']['intercept'] + 0.5 * xr,
                            'g--', alpha=0.5, label='alpha=0.5')
            axes[0, 0].legend(fontsize=8)

        plot_proxy(axes[0, 1], vflat, gc, results.get('vflat'),
                   'log(v_flat [km/s])', 'v_flat only')

        if mask_L.sum() > 0:
            plot_proxy(axes[0, 2], L36[mask_L], gc[mask_L], results.get('L36'),
                       'log(L[3.6])', 'L[3.6] only (Yd-/hR-free)')

        if mask_M.sum() > 0:
            plot_proxy(axes[1, 0], Mstar[mask_M], gc[mask_M], results.get('Mstar'),
                       'log(M_star)', 'M_star (hR-free)')

        if mask_vL.sum() > 0:
            proxy = vflat[mask_vL]**2 / np.sqrt(L36[mask_vL])
            plot_proxy(axes[1, 1], proxy, gc[mask_vL], results.get('vflat2_sqrtL'),
                       'log(v_flat^2 / sqrt(L))', 'v_flat^2/sqrt(L) (hR-free)')

        ax = axes[1, 2]
        bar_names = ['GS0_pipeline', 'L36', 'Mstar', 'vflat_sqrtL', 'vflat', 'SBdisk']
        bar_labels = ['G*S0\n(+hR)', 'L[3.6]', 'M*', 'vf*sqrtL', 'vflat', 'SBdisk']
        sigmas = []
        bar_colors = []
        for n in bar_names:
            if n in results:
                s = results[n]['resid_std']
                sigmas.append(s)
                bar_colors.append('green' if s <= ref_sigma else 'orange' if s < ref_sigma * 1.3 else 'red')
            else:
                sigmas.append(0)
                bar_colors.append('gray')
        ax.bar(range(len(sigmas)), sigmas, color=bar_colors, alpha=0.7)
        ax.axhline(ref_sigma, color='blue', ls='--', lw=1.5, label=f'G*S0 ref={ref_sigma:.3f}')
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel('Residual scatter [dex]')
        ax.set_title('Proxy comparison')
        ax.legend(fontsize=8)

        plt.suptitle('h_R-free proxy exploration v2', fontsize=13)
        plt.tight_layout()
        plt.savefig("sparc_hR_free_v2.png", dpi=150)
        print(f"\n  Figure: sparc_hR_free_v2.png")
        plt.close()
    except ImportError:
        pass

    out = {}
    for name, r in results.items():
        out[name] = {k: float(v) if isinstance(v, (np.floating, float, np.integer)) else v for k, v in r.items()}
    with open("sparc_hR_free_v2_results.json", 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  JSON: sparc_hR_free_v2_results.json")
    print("\n[DONE]")


if __name__ == '__main__':
    main()
