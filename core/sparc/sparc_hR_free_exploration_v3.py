#!/usr/bin/env python3
"""
sparc_hR_free_exploration_v3.py — h_R非依存プロキシ探索（direct gc版）
"""
import os, sys, glob, warnings, json
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

SPARC_DIR = "Rotmod_LTG"
MASTER_FILE = "SPARC_Lelli2016c.mrt"
a0 = 1.2e-10


def normalize_name(name):
    return name.strip().replace(' ', '').replace('_', '').replace('-', '').upper()


def load_sparc_master(filepath):
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
            info['Rdisk'] = float(parts[11])
            info['SBdisk'] = float(parts[12])
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


def sps_steep_Yd(T):
    return float(np.clip(0.85 - 0.06 * T, 0.25, 0.90))


def process_galaxy(rotmod, Yd):
    R = rotmod['R']
    Vobs = rotmod['Vobs']
    Vgas = rotmod['Vgas']
    Vdisk = rotmod['Vdisk']
    Vbul = rotmod['Vbul']

    # Filter R > 0.01
    mask_R = R > 0.01
    R = R[mask_R]
    Vobs = Vobs[mask_R]
    Vgas = Vgas[mask_R]
    Vdisk = Vdisk[mask_R]
    Vbul = Vbul[mask_R]

    n = len(R)
    if n < 5:
        return None

    R_m = R * 3.086e19
    g_obs = (Vobs * 1e3)**2 / R_m
    Vbar2 = Vgas**2 + Yd * Vdisk**2 + 0.5 * Vbul**2
    g_bar = np.maximum(Vbar2, 0) * 1e6 / R_m

    i_start = 2 * n // 3
    if i_start >= n - 1:
        i_start = max(0, n - 3)

    resid = g_obs[i_start:] - g_bar[i_start:]
    gc = np.median(resid)
    if gc <= 0:
        return None

    vflat = np.median(Vobs[i_start:])
    if vflat <= 5.0:
        return None

    profile = np.sqrt(max(Yd, 0.01)) * np.abs(Vdisk)
    if profile.max() <= 0:
        return None
    i_pk = np.argmax(profile)
    r_pk = R[i_pk]
    if r_pk >= 0.9 * R.max():
        return None
    hR = r_pk / 2.15
    if hR <= 0:
        return None

    GS0 = (vflat * 1e3)**2 / (hR * 3.086e19)

    return {
        'gc': gc, 'gc_a0': gc / a0, 'vflat': vflat,
        'hR': hR, 'GS0': GS0, 'Yd': Yd,
    }


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
    print("Track 2 v3: h_R-free proxy exploration (direct gc, SPS steep Yd)")
    print("  gc = direct median(g_obs - g_bar)  [TA3-independent]")
    print("  Yd = 0.85 - 0.06*T  [SPS steep]")
    print("  vflat = Vobs outer 1/3 median  [TA3-independent]")
    print("=" * 70)

    print("\n[1] Loading...")
    master = load_sparc_master(MASTER_FILE)
    print(f"  Master: {len(master)} galaxies")

    rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    if not rotmod_files:
        rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*.dat"))
    file_map = {}
    for fp in rotmod_files:
        gname = os.path.basename(fp).replace('_rotmod.dat', '').replace('.dat', '')
        file_map[normalize_name(gname)] = fp
    print(f"  Rotmod: {len(file_map)}")

    print("\n[2] Processing galaxies (SPS steep Yd, direct gc)...")
    galaxies = []

    for nkey, fpath in file_map.items():
        T = master.get(nkey, {}).get('T_type', 5.0)
        has_T = nkey in master and 'T_type' in master[nkey]

        Yd = sps_steep_Yd(T)

        rotmod = load_rotmod(fpath)
        result = process_galaxy(rotmod, Yd)
        if result is None:
            continue

        result['name'] = nkey
        result['T_type'] = T if has_T else np.nan
        result['has_T'] = has_T
        result['L36'] = master.get(nkey, {}).get('L36', np.nan)
        result['SBdisk'] = master.get(nkey, {}).get('SBdisk', np.nan)

        L = result['L36']
        result['Mstar'] = Yd * L * 1e9 if np.isfinite(L) and L > 0 else np.nan

        galaxies.append(result)

    N_total = len(galaxies)
    N_with_T = sum(1 for g in galaxies if g['has_T'])
    print(f"  Total: {N_total} galaxies ({N_with_T} with real T-type)")

    gc = np.array([g['gc'] for g in galaxies])
    gc_a0 = np.array([g['gc_a0'] for g in galaxies])
    vflat = np.array([g['vflat'] for g in galaxies])
    hR = np.array([g['hR'] for g in galaxies])
    GS0 = np.array([g['GS0'] for g in galaxies])
    Yd_arr = np.array([g['Yd'] for g in galaxies])
    T_type = np.array([g['T_type'] for g in galaxies])
    L36 = np.array([g['L36'] for g in galaxies])
    SBdisk = np.array([g['SBdisk'] for g in galaxies])
    Mstar = np.array([g['Mstar'] for g in galaxies])

    print("\n[3] Independence check: gc vs v_flat...")
    rho_gv, p_gv = stats.spearmanr(np.log10(vflat), np.log10(gc_a0))
    print(f"  Spearman log(gc/a0) vs log(vflat): rho={rho_gv:+.3f}, p={p_gv:.2e}")
    if abs(rho_gv) > 0.99:
        print(f"  [WARNING] gc and vflat still nearly perfectly correlated")
    else:
        print(f"  [OK] Direct gc removes structural dependency")

    print("\n[4] Testing proxies...")
    print(f"\n  {'Proxy':<35} {'slope':>7} {'SE':>6} {'r2':>6} {'sigma':>6} {'p(0.5)':>9} {'N':>4}")
    print("  " + "-" * 73)

    results = {}

    def test_proxy(name, label, x, y, mask=None):
        if mask is None:
            mask = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        if mask.sum() < 10:
            return
        r = fit_power_law(x[mask], y[mask])
        p05 = p_test(r['slope'], r['se'], r['N'])
        results[name] = {**r, 'p_05': p05, 'label': label}
        print(f"  {label:<35} {r['slope']:>7.3f} {r['se']:>6.3f} {r['r2']:>6.3f} {r['resid_std']:>6.3f} {p05:>9.4f} {r['N']:>4}")

    test_proxy('GS0', '(REF) G*S0 = vf^2/(Vdpk/2.15)', GS0, gc)

    ref_sigma = results.get('GS0', {}).get('resid_std', 0.3)
    ref_alpha = results.get('GS0', {}).get('slope', 0.5)

    test_proxy('vflat', 'v_flat only', vflat, gc)
    test_proxy('vflat2', 'v_flat^2', vflat**2, gc)

    mask_L = np.isfinite(L36) & (L36 > 0) & (gc > 0)
    test_proxy('L36', 'L[3.6] only (Yd-free, hR-free)', L36, gc, mask_L)

    mask_M = np.isfinite(Mstar) & (Mstar > 0) & (gc > 0)
    test_proxy('Mstar', 'M_star (hR-free)', Mstar, gc, mask_M)

    mask_SB = np.isfinite(SBdisk) & (SBdisk > 0) & (gc > 0)
    if mask_SB.sum() > 10:
        SB_flux = 10**(-0.4 * (SBdisk - 20))
        test_proxy('SBdisk', 'SB_disk flux (hR-free)', SB_flux, gc, mask_SB)

    mask_vL = np.isfinite(L36) & (L36 > 0) & (vflat > 0) & (gc > 0)
    if mask_vL.sum() > 10:
        test_proxy('vflat_sqrtL', 'v_flat * sqrt(L[3.6])',
                   vflat * np.sqrt(np.where(L36 > 0, L36, 1)), gc, mask_vL)
        test_proxy('vflat2_sqrtL', 'v_flat^2 / sqrt(L) (geom-type)',
                   vflat**2 / np.sqrt(np.where(L36 > 0, L36, 1)), gc, mask_vL)
        test_proxy('vflat4_L', 'v_flat^4 / L[3.6]',
                   vflat**4 / np.where(L36 > 0, L36, 1), gc, mask_vL)

    test_proxy('hR', 'h_R only', hR, gc)

    if mask_vL.sum() > 20:
        log_gc = np.log10(gc[mask_vL] / a0)
        log_vf = np.log10(vflat[mask_vL])
        log_L = np.log10(L36[mask_vL])

        X = np.column_stack([np.ones(mask_vL.sum()), log_vf, log_L])
        beta = np.linalg.lstsq(X, log_gc, rcond=None)[0]
        pred = X @ beta
        resid_2v = np.std(log_gc - pred)
        r2_2v = 1 - np.sum((log_gc - pred)**2) / np.sum((log_gc - log_gc.mean())**2)

        results['2var_vf_L'] = {
            'a_vflat': float(beta[1]), 'b_L36': float(beta[2]),
            'intercept': float(beta[0]), 'r2': float(r2_2v),
            'resid_std': float(resid_2v), 'N': int(mask_vL.sum())
        }
        print(f"\n  2-var: log(gc/a0) = {beta[0]:.2f} + {beta[1]:.3f}*log(vf) + {beta[2]:.3f}*log(L)")
        print(f"    r2={r2_2v:.3f}, sigma={resid_2v:.3f}")

    print("\n[5] Spearman correlations with log(gc/a0)...")
    log_gc_all = np.log10(gc_a0)
    print(f"\n  {'Variable':<25} {'rho':>7} {'p':>12} {'N':>5}")
    print("  " + "-" * 50)
    corr_data = []
    for label, arr in [('log(v_flat)', np.log10(vflat)),
                        ('log(G*S0)', np.log10(GS0)),
                        ('log(h_R)', np.log10(hR)),
                        ('log(M_star)', np.log10(np.where(Mstar > 0, Mstar, np.nan))),
                        ('log(L[3.6])', np.log10(np.where(L36 > 0, L36, np.nan))),
                        ('T_type', T_type),
                        ('SBdisk', SBdisk),
                        ('log(Yd)', np.log10(Yd_arr))]:
        m = np.isfinite(arr) & np.isfinite(log_gc_all)
        if m.sum() > 10:
            rho, p = stats.spearmanr(arr[m], log_gc_all[m])
            corr_data.append((label, rho, p, m.sum()))
    for label, rho, p, n in sorted(corr_data, key=lambda x: -abs(x[1])):
        print(f"  {label:<25} {rho:>+7.3f} {p:>12.2e} {n:>5}")

    print("\n[6] Geometric mean consistency check...")
    mask_all = np.isfinite(L36) & (L36 > 0) & (hR > 0) & (vflat > 0)
    if mask_all.sum() > 10:
        r_hL = fit_power_law(L36[mask_all], hR[mask_all])
        r_vL = fit_power_law(L36[mask_all], vflat[mask_all])
        print(f"  Scaling relations:")
        print(f"    h_R ~ L^{r_hL['slope']:.3f} (r2={r_hL['r2']:.3f})")
        print(f"    v_flat ~ L^{r_vL['slope']:.3f} (r2={r_vL['r2']:.3f}) [Tully-Fisher]")

        gs0_L_slope = 2 * r_vL['slope'] - r_hL['slope']
        print(f"    G*S0 ~ L^{gs0_L_slope:.3f}")

        gc_L_obs = results.get('L36', {}).get('slope', np.nan)
        gc_L_pred_from_GS0 = ref_alpha * gs0_L_slope
        print(f"\n  Consistency test:")
        print(f"    alpha(G*S0) = {ref_alpha:.3f}")
        print(f"    Predicted: gc ~ L^{gc_L_pred_from_GS0:.3f}")
        print(f"    Observed:  gc ~ L^{gc_L_obs:.3f}")
        diff = abs(gc_L_pred_from_GS0 - gc_L_obs)
        if diff < 0.03:
            print(f"    [MATCH] diff={diff:.3f}")
        elif diff < 0.08:
            print(f"    [PARTIAL] diff={diff:.3f}")
        else:
            print(f"    [MISMATCH] diff={diff:.3f}")

        gc_L_pred_05 = 0.5 * gs0_L_slope
        print(f"\n    If alpha=0.5: predict gc ~ L^{gc_L_pred_05:.3f}")
        print(f"    Observed:     gc ~ L^{gc_L_obs:.3f}")
        diff_05 = abs(gc_L_pred_05 - gc_L_obs)
        if diff_05 < 0.05:
            print(f"    [OK] alpha=0.5 within {diff_05:.3f}")
        else:
            print(f"    [NOTE] gap={diff_05:.3f}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    print(f"\n  Reference: G*S0 alpha={ref_alpha:.3f}, sigma={ref_sigma:.3f}, N={results['GS0']['N']}")
    print(f"\n  Independence: gc vs vflat rho={rho_gv:+.3f}")

    print(f"\n  h_R-free proxy ranking (by sigma):")
    ranking = []
    for name in ['L36', 'Mstar', 'vflat_sqrtL', 'vflat2_sqrtL', 'SBdisk',
                  'vflat', 'vflat2', 'vflat4_L', 'hR']:
        if name in results:
            r = results[name]
            penalty = (r['resid_std'] - ref_sigma) / ref_sigma * 100
            ranking.append((name, r['resid_std'], r['r2'], penalty, r.get('label', name)))

    for name, sigma, r2, penalty, label in sorted(ranking, key=lambda x: x[1]):
        tag = "BETTER" if penalty < -10 else "~SAME" if abs(penalty) < 15 else "WORSE"
        print(f"    {label[:33]:<33} sigma={sigma:.3f} r2={r2:.3f} penalty={penalty:+.0f}% [{tag}]")

    if '2var_vf_L' in results:
        r = results['2var_vf_L']
        penalty = (r['resid_std'] - ref_sigma) / ref_sigma * 100
        print(f"    {'2var(vf,L)':<33} sigma={r['resid_std']:.3f} r2={r['r2']:.3f} penalty={penalty:+.0f}%")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        def plot_xy(ax, x, y, res, xlabel, title):
            ax.scatter(np.log10(x), np.log10(y / a0), s=8, alpha=0.4)
            if res:
                xr = np.linspace(np.log10(x).min()-0.3, np.log10(x).max()+0.3, 50)
                ax.plot(xr, res['intercept'] + res['slope']*xr, 'r-',
                        label=f"slope={res['slope']:.3f}")
                ax.plot(xr, res['intercept'] + 0.5*xr, 'g--', alpha=0.5, label='slope=0.5')
                ax.set_title(f"{title}\nsigma={res.get('resid_std', 0):.3f}")
            else:
                ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('log(gc/a0)')
            ax.legend(fontsize=7)

        plot_xy(axes[0,0], GS0, gc, results.get('GS0'),
                'log(GxS0) [pipeline h_R]', 'REF: GxS0 (with h_R)')

        plot_xy(axes[0,1], vflat, gc, results.get('vflat'),
                'log(v_flat)', 'v_flat only (hR-free)')

        if mask_L.sum() > 0:
            plot_xy(axes[0,2], L36[mask_L], gc[mask_L], results.get('L36'),
                    'log(L[3.6])', 'L[3.6] only (Yd/hR-free)')

        if mask_M.sum() > 0:
            plot_xy(axes[1,0], Mstar[mask_M], gc[mask_M], results.get('Mstar'),
                    'log(M_star)', 'M_star (hR-free)')

        if mask_vL.sum() > 0:
            proxy_g = (vflat[mask_vL]**2) / np.sqrt(L36[mask_vL])
            plot_xy(axes[1,1], proxy_g, gc[mask_vL], results.get('vflat2_sqrtL'),
                    'log(v^2/sqrtL)', 'v_flat^2/sqrtL')

        ax = axes[1,2]
        bar_items = ['GS0', 'L36', 'Mstar', 'vflat_sqrtL', 'vflat', 'vflat2_sqrtL']
        bar_labels = ['G*S0\n(+hR)', 'L[3.6]', 'M*', 'vf*sqrtL', 'vflat', 'vf^2/sqrtL']
        sigmas, bcolors = [], []
        for n in bar_items:
            if n in results:
                s = results[n]['resid_std']
                sigmas.append(s)
                bcolors.append('green' if s <= ref_sigma else 'orange' if s < ref_sigma*1.3 else 'red')
            else:
                sigmas.append(0)
                bcolors.append('gray')
        ax.bar(range(len(sigmas)), sigmas, color=bcolors, alpha=0.7)
        ax.axhline(ref_sigma, color='blue', ls='--', label=f'ref={ref_sigma:.3f}')
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel('sigma [dex]')
        ax.set_title('Proxy comparison')
        ax.legend(fontsize=8)

        plt.suptitle('h_R-free v3: direct gc, SPS steep Yd, TA3-independent', fontsize=12)
        plt.tight_layout()
        plt.savefig("sparc_hR_free_v3.png", dpi=150)
        print(f"\n  Figure: sparc_hR_free_v3.png")
        plt.close()
    except ImportError:
        pass

    out = {}
    for name, r in results.items():
        out[name] = {k: float(v) if isinstance(v, (np.floating, float, np.integer)) else v for k, v in r.items()}
    with open("sparc_hR_free_v3_results.json", 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  JSON: sparc_hR_free_v3_results.json")
    print("\n[DONE]")


if __name__ == '__main__':
    main()
