#!/usr/bin/env python3
"""
sparc_alpha_reproduce.py
========================
gc_geometric_mean_test.py の手法を完全に再現し、alpha=0.545を復元する。
"""

import numpy as np
import csv
import sys
import json
from pathlib import Path
from scipy.stats import linregress, t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _fp in ['/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
            '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf']:
    try: _fm.fontManager.addfont(_fp)
    except: pass
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# === パス設定 ===
BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
SPARC_DIR = BASE / "Rotmod_LTG"
TA3_PATH = BASE / "TA3_gc_independent.csv"
PHASE1_PATH = BASE / "phase1" / "sparc_results.csv"

if not SPARC_DIR.exists():
    SPARC_DIR = Path("Rotmod_LTG")
if not TA3_PATH.exists():
    TA3_PATH = Path("TA3_gc_independent.csv")
if not PHASE1_PATH.exists():
    PHASE1_PATH = Path("phase1/sparc_results.csv")

OUTDIR = Path("alpha_reproduce_output")
OUTDIR.mkdir(exist_ok=True)

a0 = 1.2e-10  # m/s^2
kpc = 3.086e19  # m

N_BOOT = 1000
RNG_SEED = 42


def name_norm(n):
    import re
    return re.sub(r'[\s\-_]', '', n.upper().replace('_ROTMOD', '').replace('_LSBALL', ''))


def load_ta3():
    data = {}
    with open(TA3_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get('galaxy', '').strip()
                gc_a0 = float(row.get('gc_over_a0', row.get(' gc_over_a0', '0')).strip())
                vf_ta3 = float(row.get('v_flat', row.get(' v_flat', '0')).strip())
                ud_ta3 = float(row.get('upsilon_d', row.get(' upsilon_d', '0.5')).strip())
                if name and gc_a0 > 0:
                    data[name_norm(name)] = {
                        'gc_a0': gc_a0,
                        'gc': gc_a0 * a0,
                        'vflat_ta3': vf_ta3,
                        'ud_ta3': ud_ta3,
                        'name_orig': name,
                    }
            except (ValueError, KeyError):
                pass
    print(f'TA3: {len(data)} galaxies')
    return data


def load_phase1():
    data = {}
    if not PHASE1_PATH.exists():
        print(f'WARNING: phase1 not found: {PHASE1_PATH}')
        return None

    with open(PHASE1_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f'phase1 headers: {headers}')
        for row in reader:
            try:
                name = ''
                for key in ['galaxy', 'Galaxy', 'name', 'Name', 'GALAXY']:
                    if key in row and row[key].strip():
                        name = row[key].strip()
                        break
                if not name:
                    name = list(row.values())[0].strip()

                vflat = None
                for key in ['vflat', 'v_flat', 'Vflat', 'V_flat', 'vflat_kms']:
                    if key in row:
                        try:
                            vflat = float(row[key].strip())
                            break
                        except ValueError:
                            pass

                ud = None
                for key in ['ud', 'upsilon_d', 'Upsilon_d', 'Yd', 'ud_opt',
                            'upsilon_disk', 'Ud']:
                    if key in row:
                        try:
                            ud = float(row[key].strip())
                            break
                        except ValueError:
                            pass

                if name and vflat and vflat > 0:
                    data[name_norm(name)] = {
                        'vflat_p1': vflat,
                        'ud_p1': ud if ud else 0.5,
                        'name_orig': name,
                    }
            except (ValueError, KeyError):
                pass
    print(f'phase1: {len(data)} galaxies')
    return data


def load_sparc(filepath):
    try:
        d = np.loadtxt(filepath, comments='#')
    except:
        return None
    if d.ndim != 2 or d.shape[1] < 5:
        return None
    return {
        'R': d[:, 0], 'Vo': d[:, 1], 'eV': d[:, 2],
        'Vg': d[:, 3], 'Vd': d[:, 4],
        'Vb': d[:, 5] if d.shape[1] > 5 else np.zeros(len(d)),
    }


def alpha_fit(gc_arr, gs0_arr):
    log_gc = np.log10(gc_arr)
    log_gs = np.log10(gs0_arr)
    log_a0 = np.log10(a0)

    x = log_gs - log_a0
    y = log_gc

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return None

    sl, ic, r, p, se = linregress(x, y)
    t_stat = (sl - 0.5) / se
    p05 = 2 * tdist.sf(abs(t_stat), df=len(x)-2)
    return {'alpha': sl, 'e_alpha': se, 'p05': p05, 'r': r, 'N': len(x)}


def main():
    print('=' * 70)
    print('alpha=0.545 Exact Reproduction')
    print('=' * 70)

    ta3 = load_ta3()
    phase1 = load_phase1()

    files = sorted(SPARC_DIR.glob('*.dat'))
    if not files:
        files = sorted(SPARC_DIR.glob('*_rotmod.dat'))
    sparc = {}
    for f in files:
        g = load_sparc(f)
        if g:
            sparc[name_norm(f.stem)] = g
    print(f'SPARC: {len(sparc)} galaxies')

    matched = []
    for nn, t in ta3.items():
        if nn not in sparc:
            continue
        gal = sparc[nn]
        R = gal['R']
        Vd = gal['Vd']
        n = len(R)
        if n < 5:
            continue

        p1 = phase1.get(nn, {}) if phase1 else {}
        vflat_p1 = p1.get('vflat_p1', t['vflat_ta3'])
        ud_p1 = p1.get('ud_p1', t['ud_ta3'])

        entry = {
            'nn': nn, 'name': t['name_orig'], 'gc': t['gc'], 'gc_a0': t['gc_a0'],
            'vflat_ta3': t['vflat_ta3'], 'ud_ta3': t['ud_ta3'],
            'vflat_p1': vflat_p1, 'ud_p1': ud_p1,
            'R': R, 'Vd': Vd, 'Vo': gal['Vo'], 'Vg': gal['Vg'], 'Vb': gal['Vb'],
        }
        matched.append(entry)

    print(f'Matched: {len(matched)}')

    for m in matched:
        R = m['R']
        Vd = m['Vd']
        n = len(R)

        # 元パイプライン: r > 0.01 mask first
        mask = R > 0.01
        R_m = R[mask]
        Vd_m = Vd[mask]

        if len(R_m) < 5:
            m['hR_original'] = None
            m['r_pk_at_edge'] = True
            m['hR_v2'] = None
            m['vflat_obs'] = None
            continue

        vds_p1 = np.sqrt(m['ud_p1']) * np.abs(Vd_m)
        i_pk_p1 = np.argmax(vds_p1)
        r_pk_p1 = R_m[i_pk_p1]
        m['r_pk_original'] = r_pk_p1
        m['hR_original'] = r_pk_p1 / 2.15 if r_pk_p1 > 0 else None
        m['r_pk_at_edge'] = (r_pk_p1 >= R_m.max() * 0.9) or (r_pk_p1 < 0.01)

        vds_ta3 = np.sqrt(m['ud_ta3']) * np.abs(Vd_m)
        i_pk_ta3 = np.argmax(vds_ta3)
        r_pk_ta3 = R_m[i_pk_ta3]
        m['hR_ta3_ud'] = r_pk_ta3 / 2.15 if r_pk_ta3 > 0 else None
        m['r_pk_ta3_edge'] = (r_pk_ta3 >= R_m.max() * 0.9) or (r_pk_ta3 < 0.01)

        i_pk_v2 = max(np.argmax(np.abs(Vd)), 1)
        r_pk_v2 = R[i_pk_v2]
        m['hR_v2'] = r_pk_v2 / 2.2 if r_pk_v2 > 0 else None

        outer = slice(2*n//3, n)
        m['vflat_obs'] = np.median(m['Vo'][outer]) if n >= 5 else None

    print(f'\n{"="*70}')
    print('Combination Matrix')
    print(f'{"="*70}')

    def compute_alpha(label, gc_func, gs0_func, filter_func):
        gc_list, gs0_list = [], []
        n_filtered = 0
        for m in matched:
            if not filter_func(m):
                n_filtered += 1
                continue
            gc = gc_func(m)
            gs0 = gs0_func(m)
            if gc and gs0 and gc > 0 and gs0 > 0:
                gc_list.append(gc)
                gs0_list.append(gs0)

        gc_arr = np.array(gc_list)
        gs0_arr = np.array(gs0_list)
        fit = alpha_fit(gc_arr, gs0_arr)
        if fit:
            ok = 'YES' if fit['p05'] > 0.05 else 'no'
            print(f'{label:<55} N={fit["N"]:3d} (filt={n_filtered:2d}) '
                  f'a={fit["alpha"]:.3f}+/-{fit["e_alpha"]:.3f} '
                  f'p={fit["p05"]:.4f} {ok}')
            return fit
        else:
            print(f'{label:<55} FAILED')
            return None

    def filt_original(m):
        return not m.get('r_pk_at_edge', True) and m.get('hR_original') and m['hR_original'] > 0
    def filt_none(m):
        return m.get('hR_original') is not None or m.get('hR_v2') is not None

    def gs0_orig(m):
        if m.get('hR_original') and m['hR_original'] > 0:
            return (m['vflat_p1'] * 1e3)**2 / (m['hR_original'] * kpc)
        return None
    def gs0_orig_ta3vf(m):
        if m.get('hR_original') and m['hR_original'] > 0:
            return (m['vflat_ta3'] * 1e3)**2 / (m['hR_original'] * kpc)
        return None
    def gs0_orig_obsvf(m):
        if m.get('hR_original') and m['hR_original'] > 0 and m.get('vflat_obs'):
            return (m['vflat_obs'] * 1e3)**2 / (m['hR_original'] * kpc)
        return None
    def gs0_v2(m):
        if m.get('hR_v2') and m['hR_v2'] > 0 and m.get('vflat_obs'):
            return (m['vflat_obs'] * 1e3)**2 / (m['hR_v2'] * kpc)
        return None
    def gs0_hybrid_215(m):
        if m.get('hR_v2') and m['hR_v2'] > 0 and m.get('vflat_obs'):
            hR_215 = m['hR_v2'] * 2.2 / 2.15
            return (m['vflat_obs'] * 1e3)**2 / (hR_215 * kpc)
        return None

    def gc_ta3(m):
        return m['gc']
    def gc_direct_05(m):
        R, Vo, Vg, Vd, Vb = m['R'], m['Vo'], m['Vg'], m['Vd'], m['Vb']
        n = len(R)
        if n < 5: return None
        Vbar = np.sqrt(np.maximum(Vg**2 + 0.5*Vd**2 + 0.7*Vb**2, 0))
        s = slice(2*n//3, n)
        Rm = R[s] * kpc
        if len(Rm) < 2: return None
        gc = np.median((Vo[s]*1e3)**2/Rm - (Vbar[s]*1e3)**2/Rm)
        return gc if gc > 0 else None
    def gc_direct_opt(m):
        R, Vo, Vg, Vd, Vb = m['R'], m['Vo'], m['Vg'], m['Vd'], m['Vb']
        ud = m['ud_p1']
        n = len(R)
        if n < 5: return None
        Vbar = np.sqrt(np.maximum(Vg**2 + ud*Vd**2 + 0.7*Vb**2, 0))
        s = slice(2*n//3, n)
        Rm = R[s] * kpc
        if len(Rm) < 2: return None
        gc = np.median((Vo[s]*1e3)**2/Rm - (Vbar[s]*1e3)**2/Rm)
        return gc if gc > 0 else None

    print('\n--- Step 1: Exact Reproduction ---')
    fit_exact = compute_alpha(
        'EXACT: gc=TA3, GS0=vflat_p1^2/(rpk_p1/2.15), edge_filter',
        gc_ta3, gs0_orig, filt_original)

    print('\n--- Step 2: Element Substitution ---')

    compute_alpha('vflat: p1 -> TA3 (rest=original)',
                  gc_ta3, gs0_orig_ta3vf, filt_original)
    compute_alpha('vflat: p1 -> obs_median (rest=original)',
                  gc_ta3, gs0_orig_obsvf, filt_original)
    compute_alpha('divisor: 2.15 -> 2.2 (rest=v2-style hR_v2)',
                  gc_ta3, gs0_hybrid_215, filt_original)
    compute_alpha('filter: edge_filter OFF (rest=original)',
                  gc_ta3, gs0_orig, filt_none)

    def gs0_v2peak_p1vf(m):
        if m.get('hR_v2') and m['hR_v2'] > 0:
            return (m['vflat_p1'] * 1e3)**2 / (m['hR_v2'] * kpc)
        return None
    compute_alpha('r_pk: sqrt(ud)*Vd -> simple Vd (rest=original)',
                  gc_ta3, gs0_v2peak_p1vf, filt_original)

    compute_alpha('gc: TA3 -> direct(Yd=0.5) (rest=original)',
                  gc_direct_05, gs0_orig, filt_original)
    compute_alpha('gc: TA3 -> direct(Yd=p1_opt) (rest=original)',
                  gc_direct_opt, gs0_orig, filt_original)

    print('\n--- Step 3: Full v2 Method ---')
    compute_alpha('FULL_V2: gc=direct(0.5), GS0=obs^2/(Vd_pk/2.2), no_filter',
                  gc_direct_05, gs0_v2, filt_none)

    print(f'\n{"="*70}')
    print('v_flat Comparison')
    print(f'{"="*70}')

    vf_p1_list, vf_ta3_list, vf_obs_list = [], [], []
    for m in matched:
        if m.get('vflat_obs') and m['vflat_p1'] > 0:
            vf_p1_list.append(m['vflat_p1'])
            vf_ta3_list.append(m['vflat_ta3'])
            vf_obs_list.append(m['vflat_obs'])

    vf_p1 = np.array(vf_p1_list)
    vf_ta3 = np.array(vf_ta3_list)
    vf_obs = np.array(vf_obs_list)

    if len(vf_p1) > 0:
        print(f'  N = {len(vf_p1)}')
        print(f'  phase1 vs TA3:   median ratio = {np.median(vf_p1/vf_ta3):.3f}, '
              f'corr = {np.corrcoef(vf_p1, vf_ta3)[0,1]:.4f}')
        print(f'  phase1 vs obs:   median ratio = {np.median(vf_p1/vf_obs):.3f}, '
              f'corr = {np.corrcoef(vf_p1, vf_obs)[0,1]:.4f}')
        print(f'  TA3 vs obs:      median ratio = {np.median(vf_ta3/vf_obs):.3f}, '
              f'corr = {np.corrcoef(vf_ta3, vf_obs)[0,1]:.4f}')

    if phase1:
        print(f'\n{"="*70}')
        print('Upsilon_d Comparison (phase1 vs TA3)')
        print(f'{"="*70}')
        ud_p1_list, ud_ta3_list = [], []
        for m in matched:
            ud_p1_list.append(m['ud_p1'])
            ud_ta3_list.append(m['ud_ta3'])
        ud_p1_arr = np.array(ud_p1_list)
        ud_ta3_arr = np.array(ud_ta3_list)
        print(f'  phase1: median={np.median(ud_p1_arr):.3f}, '
              f'mean={np.mean(ud_p1_arr):.3f}, range=[{np.min(ud_p1_arr):.3f},{np.max(ud_p1_arr):.3f}]')
        print(f'  TA3:    median={np.median(ud_ta3_arr):.3f}, '
              f'mean={np.mean(ud_ta3_arr):.3f}, range=[{np.min(ud_ta3_arr):.3f},{np.max(ud_ta3_arr):.3f}]')
        print(f'  corr:   {np.corrcoef(ud_p1_arr, ud_ta3_arr)[0,1]:.4f}')

    print(f'\n{"="*70}')
    print('Edge Filter Analysis')
    print(f'{"="*70}')
    n_edge = sum(1 for m in matched if m.get('r_pk_at_edge'))
    n_ok = sum(1 for m in matched if not m.get('r_pk_at_edge'))
    print(f'  Edge filtered: {n_edge}/{len(matched)} ({100*n_edge/len(matched):.1f}%)')
    print(f'  Kept: {n_ok}')

    gc_edge = [m['gc_a0'] for m in matched if m.get('r_pk_at_edge')]
    gc_ok = [m['gc_a0'] for m in matched if not m.get('r_pk_at_edge')]
    if gc_edge:
        print(f'  Edge galaxies gc/a0: median={np.median(gc_edge):.3f}, '
              f'mean={np.mean(gc_edge):.3f}')
    if gc_ok:
        print(f'  Kept galaxies gc/a0: median={np.median(gc_ok):.3f}, '
              f'mean={np.mean(gc_ok):.3f}')

    # 図
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    if len(vf_p1) > 0:
        ax.scatter(vf_obs, vf_p1, s=15, alpha=0.5, label='phase1', color='#1a1a2e')
        ax.scatter(vf_obs, vf_ta3, s=15, alpha=0.5, label='TA3', color='#e94560')
        lim = [0, max(max(vf_obs), max(vf_p1), max(vf_ta3)) * 1.05]
        ax.plot(lim, lim, 'k--', alpha=0.3)
        ax.set_xlabel('V_flat (obs median) [km/s]')
        ax.set_ylabel('V_flat (phase1 / TA3) [km/s]')
        ax.set_title('(a) V_flat comparison')
        ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    hR_orig_p = [m['hR_original'] for m in matched if m.get('hR_original') and m.get('hR_v2')]
    hR_v2_p = [m['hR_v2'] for m in matched if m.get('hR_original') and m.get('hR_v2')]
    if hR_orig_p and hR_v2_p:
        colors_scat = ['#e94560' if m['r_pk_at_edge'] else '#1a1a2e'
                       for m in matched if m.get('hR_original') and m.get('hR_v2')]
        ax.scatter(hR_v2_p, hR_orig_p, s=15, alpha=0.5, c=colors_scat)
        lim = [0, max(max(hR_v2_p), max(hR_orig_p)) * 1.05]
        ax.plot(lim, lim, 'k--', alpha=0.3, label='1:1')
        ax.set_xlabel('h_R (v2: Vd_peak/2.2) [kpc]')
        ax.set_ylabel('h_R (original: sqrt(ud)*Vd_peak/2.15) [kpc]')
        ax.set_title('(b) h_R comparison (red=edge-filtered)')
        ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if phase1:
        ax.hist(ud_p1_arr, bins=20, alpha=0.6, label='phase1', color='#1a1a2e')
        ax.hist(ud_ta3_arr, bins=20, alpha=0.6, label='TA3', color='#e94560')
        ax.axvline(0.5, color='green', ls='--', lw=2, label='Y_d=0.5')
        ax.set_xlabel('Y_d')
        ax.set_title('(c) Y_d: phase1 vs TA3')
        ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    gc_plot, gs0_plot, edge_plot = [], [], []
    for m in matched:
        gs0 = gs0_orig(m)
        if gs0 and gs0 > 0 and m['gc'] > 0:
            gc_plot.append(m['gc'])
            gs0_plot.append(gs0)
            edge_plot.append(m.get('r_pk_at_edge', False))

    if gc_plot:
        gc_p = np.array(gc_plot)
        gs0_p = np.array(gs0_plot)
        edge_p = np.array(edge_plot)

        ax.scatter(np.log10(gs0_p[~edge_p]/a0), np.log10(gc_p[~edge_p]),
                   s=15, alpha=0.5, color='#1a1a2e', label='kept')
        if np.any(edge_p):
            ax.scatter(np.log10(gs0_p[edge_p]/a0), np.log10(gc_p[edge_p]),
                       s=15, alpha=0.5, color='#e94560', label='edge-filtered')

        x_kept = np.log10(gs0_p[~edge_p]/a0)
        y_kept = np.log10(gc_p[~edge_p])
        mk = np.isfinite(x_kept) & np.isfinite(y_kept)
        if np.sum(mk) > 10:
            sl, ic, _, _, _ = linregress(x_kept[mk], y_kept[mk])
            x_line = np.linspace(np.min(x_kept[mk]), np.max(x_kept[mk]), 50)
            ax.plot(x_line, sl*x_line + ic, 'g-', lw=2, label=f'alpha={sl:.3f}')

        ax.set_xlabel('log(G*Sigma_0 / a_0)')
        ax.set_ylabel('log(g_c)')
        ax.set_title('(d) Regression (kept only)')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('alpha=0.545 Reproduction & Decomposition', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = OUTDIR / 'alpha_reproduce.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure: {fig_path}')

    summary = {'n_matched': len(matched), 'n_edge_filtered': n_edge}
    with open(OUTDIR / 'alpha_reproduce.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == '__main__':
    main()
