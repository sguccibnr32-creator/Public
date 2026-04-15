#!/usr/bin/env python3
"""
sparc_sps_Yd_test.py — 色-M/L関係に基づく独立Υ_d推定によるα検証
"""
import os, sys, glob, warnings
import numpy as np
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

SPARC_DIR = "Rotmod_LTG"
TA3_FILE = "TA3_gc_independent.csv"
MASTER_FILE = "SPARC_Lelli2016c.mrt"
a0 = 1.2e-10

SPS_MODELS = {
    "SPS_steep": {
        "desc": "SPS steep: Sa=0.80 -> Im=0.25",
        "func": lambda T: np.clip(0.85 - 0.06 * T, 0.20, 0.90)
    },
    "SPS_moderate": {
        "desc": "SPS moderate: Sa=0.70 -> Im=0.35",
        "func": lambda T: np.clip(0.75 - 0.04 * T, 0.30, 0.80)
    },
    "SPS_shallow": {
        "desc": "SPS shallow: Sa=0.65 -> Im=0.45",
        "func": lambda T: np.clip(0.67 - 0.02 * T, 0.40, 0.70)
    },
    "flat_05": {
        "desc": "Flat Yd=0.5",
        "func": lambda T: np.full_like(T, 0.5, dtype=float)
    },
}


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
    print(f"  TA3: {len(data)} galaxies loaded")
    return data


def load_sparc_master(filepath):
    props = {}
    if not os.path.exists(filepath):
        print(f"  [!] {filepath} not found. Trying download...")
        try:
            import urllib.request
            url = "http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt"
            urllib.request.urlretrieve(url, filepath)
            print(f"  [OK] Downloaded {filepath}")
        except Exception as e:
            print(f"  [FAIL] Download failed: {e}")
            return props

    # Fixed-width MRT format: bytes 1-11 Galaxy, 12-13 T, 14-19 D,
    # 35-41 L36, 62-66 Rdisk, 67-74 SBdisk, 87-91 Vflat
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    # Find data start: lines after the second '---' divider
    n_dash = 0
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('---'):
            n_dash += 1
            if n_dash >= 3:  # third dash divider before data
                data_start_idx = i + 1
                break

    if data_start_idx is None:
        # Fallback: find first line that parses as data
        for i, line in enumerate(lines):
            if len(line) > 90:
                try:
                    int(line[11:13].strip())
                    data_start_idx = i
                    break
                except (ValueError, IndexError):
                    continue

    if data_start_idx is None:
        print("  [!] Could not find data section")
        return props

    for line in lines[data_start_idx:]:
        if len(line) < 90:
            continue
        try:
            name = line[0:11].strip()
            # Note: MRT doc says bytes 12-13 but file has extra space; use 12:14
            T = int(line[12:14].strip())
            if not name:
                continue
            props[name] = {'T_type': float(T)}
            try:
                props[name]['L36'] = float(line[34:42].strip())
            except (ValueError, IndexError):
                pass
            try:
                props[name]['Rdisk'] = float(line[61:67].strip())
            except (ValueError, IndexError):
                pass
            try:
                props[name]['SBdisk'] = float(line[66:74].strip())
            except (ValueError, IndexError):
                pass
            try:
                props[name]['Vflat'] = float(line[86:92].strip())
            except (ValueError, IndexError):
                pass
        except (ValueError, IndexError):
            continue

    print(f"  Master table: {len(props)} galaxies with T-type")
    return props


def load_rotmod(filepath):
    R, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk = [], [], [], [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('R'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    Vobs.append(float(parts[1]))
                    errV.append(float(parts[2]))
                    Vgas.append(float(parts[3]))
                    Vdisk.append(float(parts[4]))
                    Vbul.append(float(parts[5]) if len(parts) > 5 else 0.0)
                    SBdisk.append(float(parts[6]) if len(parts) > 6 else 0.0)
                except ValueError:
                    continue
    return {
        'R': np.array(R), 'Vobs': np.array(Vobs), 'errV': np.array(errV),
        'Vgas': np.array(Vgas), 'Vdisk': np.array(Vdisk), 'Vbul': np.array(Vbul)
    }


def compute_gc_and_proxy(rotmod, Yd, Yb=0.5):
    R = rotmod['R']
    Vobs = rotmod['Vobs']
    errV = rotmod['errV']
    Vgas = rotmod['Vgas']
    Vdisk = rotmod['Vdisk']
    Vbul = rotmod['Vbul']

    if len(R) < 5:
        return None, None

    Vbar2 = Vgas**2 + Yd * Vdisk**2 + Yb * Vbul**2
    Vbar = np.sqrt(np.maximum(Vbar2, 0))

    R_m = R * 3.086e19
    g_obs = Vobs**2 * 1e6 / R_m
    g_bar = Vbar2 * 1e6 / R_m

    n = len(R)
    outer_start = 2 * n // 3
    if outer_start >= n - 1:
        outer_start = max(0, n - 3)

    g_residual = g_obs[outer_start:] - g_bar[outer_start:]
    gc = np.median(g_residual)

    if gc <= 0:
        return None, None

    v_flat = np.median(Vobs[outer_start:])

    abs_Vdisk = np.abs(Vdisk)
    if abs_Vdisk.max() > 0:
        r_pk = R[np.argmax(np.sqrt(Yd) * abs_Vdisk)]
        h_R = r_pk / 2.15
    else:
        h_R = R[n // 2]

    if h_R <= 0:
        return None, None

    GS0 = (v_flat * 1e3)**2 / (h_R * 3.086e19)

    return gc, GS0


def fit_alpha(gc_arr, GS0_arr):
    x = np.log10(GS0_arr / a0)
    y = np.log10(gc_arr / a0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    t_stat = (slope - 0.5) / std_err
    p_05 = 2 * stats.t.sf(abs(t_stat), df=len(x) - 2)

    return {
        'alpha': slope, 'se': std_err, 'r2': r_value**2,
        'p_05': p_05, 'N': len(x), 'intercept': intercept,
        'residual_std': np.std(y - (intercept + slope * x))
    }


def main():
    print("=" * 70)
    print("Track 1: SPS Yd independent estimation -> alpha verification")
    print("=" * 70)

    print("\n[1] Data loading...")
    ta3 = load_ta3(TA3_FILE)
    if not ta3:
        print("ERROR: TA3 file not found")
        sys.exit(1)

    master = load_sparc_master(MASTER_FILE)

    rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    if not rotmod_files:
        rotmod_files = glob.glob(os.path.join(SPARC_DIR, "*.dat"))
    print(f"  Rotmod files: {len(rotmod_files)}")

    if not master:
        print("\n  [!] Master table unavailable.")
        print("  Fallback: using TA3 Yd to infer T-type proxy")
        for name, d in ta3.items():
            Yd = d['upsilon_d']
            T_est = (0.75 - np.clip(Yd, 0.30, 0.80)) / 0.04
            master[name] = {'T_type': T_est, 'T_source': 'inferred'}

    def get_galaxy_name(filepath):
        fname = os.path.basename(filepath)
        return fname.replace('_rotmod.dat', '').replace('.dat', '')

    print("\n[2] Computing g_c with different Yd models...")

    results = {}

    for model_name, model in SPS_MODELS.items():
        gc_list = []
        GS0_list = []
        gal_names = []

        for fpath in rotmod_files:
            gname = get_galaxy_name(fpath)

            if gname not in master:
                continue
            T = master[gname]['T_type']

            Yd_sps = model['func'](np.array([T]))[0]

            rotmod = load_rotmod(fpath)
            if len(rotmod['R']) < 5:
                continue

            gc, GS0 = compute_gc_and_proxy(rotmod, Yd_sps)
            if gc is not None and GS0 is not None and gc > 0 and GS0 > 0:
                gc_list.append(gc)
                GS0_list.append(GS0)
                gal_names.append(gname)

        if len(gc_list) < 10:
            print(f"  {model_name}: N={len(gc_list)} (too few, skipped)")
            continue

        gc_arr = np.array(gc_list)
        GS0_arr = np.array(GS0_list)
        res = fit_alpha(gc_arr, GS0_arr)
        results[model_name] = res
        results[model_name]['desc'] = model['desc']

    gc_ta3, GS0_ta3 = [], []
    for fpath in rotmod_files:
        gname = get_galaxy_name(fpath)
        if gname not in ta3 or gname not in master:
            continue
        Yd_opt = ta3[gname]['upsilon_d']
        rotmod = load_rotmod(fpath)
        if len(rotmod['R']) < 5:
            continue
        gc, GS0 = compute_gc_and_proxy(rotmod, Yd_opt)
        if gc is not None and GS0 is not None and gc > 0 and GS0 > 0:
            gc_ta3.append(gc)
            GS0_ta3.append(GS0)

    if len(gc_ta3) >= 10:
        res_ta3 = fit_alpha(np.array(gc_ta3), np.array(GS0_ta3))
        results['TA3_optimal'] = res_ta3
        results['TA3_optimal']['desc'] = "TA3 RC-optimized Yd (reference)"

    print("\n[3] TA3 Yd vs T-type correlation...")
    Yd_ta3_vals, T_vals = [], []
    for gname in ta3:
        if gname in master:
            Yd_ta3_vals.append(ta3[gname]['upsilon_d'])
            T_vals.append(master[gname]['T_type'])
    if len(Yd_ta3_vals) > 10:
        r_YdT, p_YdT = stats.spearmanr(T_vals, Yd_ta3_vals)
        slope_YdT, intercept_YdT, _, _, _ = stats.linregress(T_vals, Yd_ta3_vals)
        print(f"  Spearman r(Yd, T) = {r_YdT:.3f}, p = {p_YdT:.2e}")
        print(f"  Linear fit: Yd = {intercept_YdT:.3f} + {slope_YdT:.4f} * T")
        print(f"  -> Predicted range: T=2: {intercept_YdT+2*slope_YdT:.3f}, T=10: {intercept_YdT+10*slope_YdT:.3f}")

        def ta3_fit_func(T, a=slope_YdT, b=intercept_YdT):
            return np.clip(b + a * T, 0.20, 1.50)

        gc_fitYd, GS0_fitYd = [], []
        for fpath in rotmod_files:
            gname = get_galaxy_name(fpath)
            if gname not in master:
                continue
            T = master[gname]['T_type']
            Yd_fit = ta3_fit_func(np.array([T]))[0]
            rotmod = load_rotmod(fpath)
            if len(rotmod['R']) < 5:
                continue
            gc, GS0 = compute_gc_and_proxy(rotmod, Yd_fit)
            if gc is not None and GS0 is not None and gc > 0 and GS0 > 0:
                gc_fitYd.append(gc)
                GS0_fitYd.append(GS0)

        if len(gc_fitYd) >= 10:
            res_fit = fit_alpha(np.array(gc_fitYd), np.array(GS0_fitYd))
            results['TA3_Yd_linfit'] = res_fit
            results['TA3_Yd_linfit']['desc'] = f"Yd from TA3 linear fit: {intercept_YdT:.3f}+{slope_YdT:.4f}*T"

    print("\n" + "=" * 70)
    print("RESULTS: alpha comparison across Yd models")
    print("=" * 70)
    print(f"{'Model':<25} {'N':>4} {'alpha':>8} {'SE':>7} {'p(0.5)':>10} {'sigma':>7} {'Verdict':>14}")
    print("-" * 80)

    for name in ['flat_05', 'SPS_shallow', 'SPS_moderate', 'SPS_steep', 'TA3_Yd_linfit', 'TA3_optimal']:
        if name not in results:
            continue
        r = results[name]
        verdict = "REJECT" if r['p_05'] < 0.05 else "OK (not rej.)"
        print(f"  {r['desc'][:23]:<23} {r['N']:>4} {r['alpha']:>8.3f} {r['se']:>7.3f} {r['p_05']:>10.4f} {r['residual_std']:>7.3f} {verdict:>14}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if 'TA3_optimal' in results:
        alpha_ta3 = results['TA3_optimal']['alpha']
    else:
        alpha_ta3 = 0.545

    best_sps = None
    best_diff = 999
    for name in ['SPS_shallow', 'SPS_moderate', 'SPS_steep']:
        if name in results:
            diff = abs(results[name]['alpha'] - alpha_ta3)
            if diff < best_diff:
                best_diff = diff
                best_sps = name

    if best_sps and best_sps in results:
        r = results[best_sps]
        print(f"\n  Best SPS model: {r['desc']}")
        print(f"  alpha = {r['alpha']:.3f} +/- {r['se']:.3f}")
        print(f"  Distance from TA3 (alpha={alpha_ta3:.3f}): Delta = {r['alpha']-alpha_ta3:+.3f}")
        if r['p_05'] >= 0.05:
            print(f"  alpha=0.5 is NOT rejected (p={r['p_05']:.3f})")
            print(f"  -> SPS Yd independently reproduces alpha ~ 0.5")
        else:
            print(f"  alpha=0.5 IS rejected (p={r['p_05']:.4f})")
            print(f"  -> SPS Yd partially reproduces but not exactly 0.5")

    if 'TA3_Yd_linfit' in results:
        r = results['TA3_Yd_linfit']
        print(f"\n  TA3 Yd linear fit model: alpha = {r['alpha']:.3f} +/- {r['se']:.3f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        names_plot = [n for n in ['flat_05', 'SPS_shallow', 'SPS_moderate', 'SPS_steep', 'TA3_Yd_linfit', 'TA3_optimal'] if n in results]
        alphas = [results[n]['alpha'] for n in names_plot]
        errors = [results[n]['se'] for n in names_plot]
        labels = [results[n]['desc'][:20] for n in names_plot]
        colors = ['gray', 'skyblue', 'steelblue', 'navy', 'orange', 'red'][:len(names_plot)]
        y_pos = range(len(names_plot))
        ax.barh(y_pos, alphas, xerr=errors, color=colors[:len(names_plot)], alpha=0.7)
        ax.axvline(0.5, color='green', ls='--', label='alpha=0.5')
        ax.axvline(0.545, color='red', ls=':', label='TA3=0.545')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('alpha')
        ax.set_title('alpha vs Yd model')
        ax.legend(fontsize=8)

        ax = axes[1]
        if len(Yd_ta3_vals) > 0:
            ax.scatter(T_vals, Yd_ta3_vals, s=10, alpha=0.5, label='TA3 optimized')
            T_range = np.linspace(1, 10, 50)
            for mname, model in SPS_MODELS.items():
                if mname != 'flat_05':
                    ax.plot(T_range, model['func'](T_range), '--', label=mname, alpha=0.7)
            ax.set_xlabel('T-type')
            ax.set_ylabel('Upsilon_d')
            ax.set_title('Yd: TA3 optimal vs SPS models')
            ax.legend(fontsize=7)

        ax = axes[2]
        if best_sps and best_sps in results:
            gc_plot, GS0_plot = [], []
            for fpath in rotmod_files:
                gname = get_galaxy_name(fpath)
                if gname not in master:
                    continue
                T = master[gname]['T_type']
                Yd_sps = SPS_MODELS[best_sps]['func'](np.array([T]))[0]
                rotmod = load_rotmod(fpath)
                if len(rotmod['R']) < 5:
                    continue
                gc, GS0 = compute_gc_and_proxy(rotmod, Yd_sps)
                if gc is not None and GS0 is not None and gc > 0 and GS0 > 0:
                    gc_plot.append(gc)
                    GS0_plot.append(GS0)
            gc_plot = np.array(gc_plot)
            GS0_plot = np.array(GS0_plot)
            ax.scatter(np.log10(GS0_plot/a0), np.log10(gc_plot/a0), s=10, alpha=0.5)
            x_fit = np.linspace(-1.5, 1.5, 50)
            r = results[best_sps]
            ax.plot(x_fit, r['intercept'] + r['alpha'] * x_fit, 'r-',
                    label=f"alpha={r['alpha']:.3f}")
            ax.plot(x_fit, r['intercept'] + 0.5 * x_fit, 'g--', label='alpha=0.5')
            ax.set_xlabel('log(G*Sigma0/a0)')
            ax.set_ylabel('log(gc/a0)')
            ax.set_title(f'Best SPS: {best_sps}')
            ax.legend(fontsize=8)

        plt.tight_layout()
        outpath = "sparc_sps_Yd_test.png"
        plt.savefig(outpath, dpi=150)
        print(f"\n  Figure saved: {outpath}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping figure)")

    print("\n  Results JSON:")
    import json
    out = {}
    for name, r in results.items():
        out[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in r.items()}
    json_path = "sparc_sps_Yd_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
