#!/usr/bin/env python3
"""
sparc_alpha_decomposition.py
============================
TA3_gc_independent.csv を直接読み込み、alpha=0.545 を復元し、
各方法論的要素の寄与を分離する。

分離する要素:
  (1) gc_over_a0 の定義 (TA3 vs 直接計算)
  (2) Υ_d の最適化 (個別 vs 0.5固定)
  (3) h_R の定義 (rs_tanh vs V_disk peak/2.2)
  (4) v_flat の定義 (tanh fit vs 外側median)
  (5) 下限打切り (gc_over_a0=0.1 or v_flat=5.0)
  (6) gc_circular vs gc_over_a0

テスト:
  A) TA3の全パラメータ再現 -> alpha確認
  B) 要素を1つずつv2方式に置換 -> alphaの変化を測定
  C) v2の全パラメータで計算 -> alpha=0.670確認
  -> 差分がどの要素に帰属するかを定量化

実行: uv run --with scipy --with matplotlib python sparc_alpha_decomposition.py

前提: TA3_gc_independent.csv がカレントまたは指定パスに存在すること
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

# === 設定 ===
SPARC_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG")
if not SPARC_DIR.exists():
    SPARC_DIR = Path("Rotmod_LTG")
if not SPARC_DIR.exists():
    SPARC_DIR = Path(".")

TA3_CANDIDATES = [
    Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\TA3_gc_independent.csv"),
    Path("TA3_gc_independent.csv"),
    Path("../TA3_gc_independent.csv"),
]

OUTDIR = Path("alpha_decomposition_output")
OUTDIR.mkdir(exist_ok=True)

# 物理定数
G_SI = 6.674e-11
Msun = 1.989e30
pc = 3.086e16
kpc = 1e3 * pc
a0 = 1.2e-10  # m/s^2

N_BOOT = 1000
RNG_SEED = 42


def load_ta3():
    """TA3 CSV読み込み"""
    for p in TA3_CANDIDATES:
        if p.exists():
            print(f'TA3 found: {p}')
            data = []
            with open(p, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        name = row.get('galaxy', '').strip()
                        vflat = float(row.get('v_flat', row.get(' v_flat', '0')).strip())
                        rs = float(row.get('rs_tanh', row.get(' rs_tanh', '0')).strip())
                        ud = float(row.get('upsilon_d', row.get(' upsilon_d', '0.5')).strip())
                        gc_a0 = float(row.get('gc_over_a0', row.get(' gc_over_a0', '0')).strip())
                        gc_err = float(row.get('gc_err', row.get(' gc_err', '0')).strip())
                        chi2 = float(row.get('chi2_dof', row.get(' chi2_dof', '99')).strip())
                        gc_circ_str = row.get('gc_circular', row.get(' gc_circular', '')).strip()
                        gc_circ = float(gc_circ_str) if gc_circ_str else np.nan

                        if name and vflat > 0 and gc_a0 > 0:
                            data.append({
                                'name': name,
                                'v_flat_ta3': vflat,        # km/s
                                'rs_tanh': rs,               # kpc
                                'upsilon_d': ud,
                                'gc_over_a0': gc_a0,
                                'gc_ta3': gc_a0 * a0,        # m/s^2
                                'gc_err': gc_err,
                                'chi2_dof': chi2,
                                'gc_circular': gc_circ * a0 if np.isfinite(gc_circ) else np.nan,
                            })
                    except (ValueError, KeyError):
                        pass
            print(f'  {len(data)} galaxies loaded')
            return data
    print('ERROR: TA3_gc_independent.csv not found')
    sys.exit(1)


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
        'name': filepath.stem,
    }


def name_norm(n):
    import re
    return re.sub(r'[\s\-_]', '', n.upper().replace('_ROTMOD', '').replace('_LSBALL', ''))


def gc_direct(gal, Yd, Yb=0.7):
    """直接計算: median(g_obs - g_bar) in outer 1/3"""
    R, Vo, Vg, Vd, Vb = gal['R'], gal['Vo'], gal['Vg'], gal['Vd'], gal['Vb']
    n = len(R)
    if n < 5: return None
    Vbar = np.sqrt(np.maximum(Vg**2 + Yd*Vd**2 + Yb*Vb**2, 0))
    s = slice(2*n//3, n)
    Rm = R[s] * kpc
    if len(Rm) < 2 or np.any(Rm <= 0): return None
    gc = np.median((Vo[s]*1e3)**2/Rm - (Vbar[s]*1e3)**2/Rm)
    return gc if gc > 0 else None


def hR_vdisk_peak(gal):
    """h_R = V_disk peak / 2.2"""
    Vd = gal['Vd']
    R = gal['R']
    idx = max(np.argmax(np.abs(Vd)), 1)
    h = R[idx] / 2.2
    return h if h > 0.01 else None


def vflat_obs(gal):
    """外側1/3のV_obs中央値"""
    n = len(gal['R'])
    if n < 5: return None
    return np.median(gal['Vo'][2*n//3:])


def alpha_fit(gc_arr, gs0_arr):
    x = np.log10(gs0_arr)
    y = np.log10(gc_arr)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10: return None
    sl, ic, r, p, se = linregress(x, y)
    t_stat = (sl - 0.5) / se
    p05 = 2 * tdist.sf(abs(t_stat), df=len(x)-2)
    return {'alpha': sl, 'e_alpha': se, 'p05': p05, 'r': r, 'N': len(x)}


def bootstrap_alpha(gc_arr, gs0_arr, n_boot=N_BOOT):
    rng = np.random.RandomState(RNG_SEED)
    N = len(gc_arr)
    alphas = []
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        fit = alpha_fit(gc_arr[idx], gs0_arr[idx])
        if fit: alphas.append(fit['alpha'])
    return np.array(alphas)


def main():
    print('=' * 70)
    print('alpha=0.545 Decomposition Analysis')
    print('=' * 70)

    # === データ読み込み ===
    ta3 = load_ta3()

    files = sorted(SPARC_DIR.glob('*.dat'))
    if not files:
        files = sorted(SPARC_DIR.glob('*_rotmod.dat'))
    sparc = {}
    for f in files:
        g = load_sparc(f)
        if g: sparc[name_norm(g['name'])] = g
    print(f'SPARC: {len(sparc)} galaxies')

    # === TA3-SPARC クロスマッチ ===
    matched = []
    for t in ta3:
        nn = name_norm(t['name'])
        if nn in sparc:
            gal = sparc[nn]
            hR = hR_vdisk_peak(gal)
            vf = vflat_obs(gal)
            gc_v2_05 = gc_direct(gal, Yd=0.5)
            gc_v2_opt = gc_direct(gal, Yd=t['upsilon_d'])
            if hR and vf and gc_v2_05:
                matched.append({
                    **t,
                    'gal': gal,
                    'hR_vdpeak': hR,
                    'vflat_obs': vf,
                    'gc_v2_Yd05': gc_v2_05,
                    'gc_v2_Ydopt': gc_v2_opt,
                })
    print(f'Matched: {len(matched)}')

    # === 下限打切りの確認 ===
    gc_a0_vals = [m['gc_over_a0'] for m in matched]
    vflat_vals = [m['v_flat_ta3'] for m in matched]
    n_gc_floor = sum(1 for v in gc_a0_vals if v <= 0.101)
    n_vf_floor = sum(1 for v in vflat_vals if v <= 5.1)
    print(f'\nFloor values:')
    print(f'  gc_over_a0 <= 0.101: {n_gc_floor}/{len(matched)} ({100*n_gc_floor/len(matched):.1f}%)')
    print(f'  v_flat <= 5.1: {n_vf_floor}/{len(matched)} ({100*n_vf_floor/len(matched):.1f}%)')

    # === Υ_d分布 ===
    ud_vals = np.array([m['upsilon_d'] for m in matched])
    print(f'\nUpsilon_d distribution:')
    print(f'  median = {np.median(ud_vals):.3f}')
    print(f'  mean   = {np.mean(ud_vals):.3f}')
    print(f'  std    = {np.std(ud_vals):.3f}')
    print(f'  range  = [{np.min(ud_vals):.3f}, {np.max(ud_vals):.3f}]')
    print(f'  IQR    = [{np.percentile(ud_vals,25):.3f}, {np.percentile(ud_vals,75):.3f}]')

    # ================================================================
    # Alpha測定: 組み合わせ分離テスト
    # ================================================================
    print(f'\n{"="*70}')
    print('Alpha Decomposition: Element-by-Element Substitution')
    print(f'{"="*70}')

    # G*Sigma_0 の3つの定義
    def gs0_ta3_rs(m):
        """TA3: v_flat^2 / rs_tanh"""
        if m['rs_tanh'] > 0:
            return (m['v_flat_ta3']*1e3)**2 / (m['rs_tanh']*kpc)
        return None

    def gs0_ta3_vf_v2_hr(m):
        """TA3 v_flat + v2 h_R"""
        return (m['v_flat_ta3']*1e3)**2 / (m['hR_vdpeak']*kpc)

    def gs0_v2(m):
        """v2: vflat_obs^2 / hR_vdpeak"""
        return (m['vflat_obs']*1e3)**2 / (m['hR_vdpeak']*kpc)

    # 組み合わせテスト
    tests = [
        # (label, gc_source, gs0_func, filter_desc)
        ('A: TA3全再現\n  gc=TA3, GS0=vflat_ta3^2/rs_tanh',
         lambda m: m['gc_ta3'], gs0_ta3_rs, 'none'),

        ('B: TA3 gc + v2のGS0\n  gc=TA3, GS0=vflat_obs^2/hR_vdpeak',
         lambda m: m['gc_ta3'], gs0_v2, 'none'),

        ('C: TA3 gc + TA3 vflat + v2 hR\n  gc=TA3, GS0=vflat_ta3^2/hR_vdpeak',
         lambda m: m['gc_ta3'], gs0_ta3_vf_v2_hr, 'none'),

        ('D: v2 gc(Yd=opt) + TA3のGS0\n  gc=direct(Yd_opt), GS0=vflat_ta3^2/rs_tanh',
         lambda m: m['gc_v2_Ydopt'], gs0_ta3_rs, 'none'),

        ('E: v2 gc(Yd=0.5) + TA3のGS0\n  gc=direct(Yd=0.5), GS0=vflat_ta3^2/rs_tanh',
         lambda m: m['gc_v2_Yd05'], gs0_ta3_rs, 'none'),

        ('F: v2全パラメータ\n  gc=direct(Yd=0.5), GS0=vflat_obs^2/hR_vdpeak',
         lambda m: m['gc_v2_Yd05'], gs0_v2, 'none'),

        ('G: v2 gc(Yd=opt) + v2のGS0\n  gc=direct(Yd_opt), GS0=vflat_obs^2/hR_vdpeak',
         lambda m: m.get('gc_v2_Ydopt'), gs0_v2, 'none'),

        ('H: gc_circular + TA3のGS0\n  gc=gc_circular, GS0=vflat_ta3^2/rs_tanh',
         lambda m: m.get('gc_circular'), gs0_ta3_rs, 'none'),
    ]

    # 下限除外テスト
    tests.append(
        ('A2: TA3全再現 (floor除外)\n  gc_over_a0>0.101 & vflat>5.1',
         lambda m: m['gc_ta3'], gs0_ta3_rs, 'no_floor'),
    )

    print(f'\n{"Label":<50} {"N":>5} {"alpha":>7} {"SE":>7} {"p(0.5)":>9} {"0.5?":>5}')
    print('-' * 85)

    decomp_results = {}
    for label, gc_func, gs0_func, filt in tests:
        gc_list, gs0_list = [], []
        for m in matched:
            # フィルタ
            if filt == 'no_floor':
                if m['gc_over_a0'] <= 0.101 or m['v_flat_ta3'] <= 5.1:
                    continue

            gc = gc_func(m)
            gs0 = gs0_func(m)
            if gc and gs0 and gc > 0 and gs0 > 0 and np.isfinite(gc) and np.isfinite(gs0):
                gc_list.append(gc)
                gs0_list.append(gs0)

        gc_arr = np.array(gc_list)
        gs0_arr = np.array(gs0_list)
        fit = alpha_fit(gc_arr, gs0_arr)

        label_short = label.split('\n')[0]
        if fit:
            ok = 'YES' if fit['p05'] > 0.05 else 'no'
            print(f'{label_short:<50} {fit["N"]:5d} {fit["alpha"]:7.3f} {fit["e_alpha"]:7.3f} '
                  f'{fit["p05"]:9.4f} {ok:>5}')
            decomp_results[label_short] = fit
        else:
            print(f'{label_short:<50}  -- failed --')

    # ================================================================
    # 差分分析: どの要素がどれだけalphaを動かすか
    # ================================================================
    print(f'\n{"="*70}')
    print('Element Attribution (delta-alpha)')
    print(f'{"="*70}')

    def get_alpha(key):
        for k, v in decomp_results.items():
            if k.startswith(key):
                return v['alpha']
        return None

    a_A = get_alpha('A:')
    a_B = get_alpha('B:')
    a_C = get_alpha('C:')
    a_D = get_alpha('D:')
    a_E = get_alpha('E:')
    a_F = get_alpha('F:')
    a_G = get_alpha('G:')

    if all(v is not None for v in [a_A, a_B, a_C, a_D, a_E, a_F]):
        print(f'\n  Base (A: TA3 full):        alpha = {a_A:.3f}')
        print(f'  Target (F: v2 full):       alpha = {a_F:.3f}')
        print(f'  Total gap:                 Delta = {a_F - a_A:+.3f}')
        print()

        # GS0の影響 (A->B: rs_tanh -> hR_vdpeak, keeping gc=TA3)
        print(f'  A->B (GS0: rs_tanh -> hR_vdpeak):  {a_B - a_A:+.3f}')

        # vflatの影響 (B->C or equivalent)
        if a_C is not None:
            print(f'  A->C (hR only: rs_tanh -> hR_vdpeak, keep vflat_ta3): {a_C - a_A:+.3f}')
            print(f'  C->B (vflat: ta3 -> obs, keep hR_vdpeak):  {a_B - a_C:+.3f}')

        # gc定義の影響 (A->D: TA3 gc -> direct gc with Yd_opt)
        print(f'  A->D (gc: TA3 -> direct(Yd_opt)):  {a_D - a_A:+.3f}')

        # Yd最適化の影響 (D->E: Yd_opt -> Yd=0.5)
        print(f'  D->E (Yd: optimal -> 0.5):         {a_E - a_D:+.3f}')

        # 残差
        print(f'  E->F (GS0: ta3 -> v2):             {a_F - a_E:+.3f}')

        print(f'\n  Decomposition check: A + sum = {a_A + (a_B-a_A) + (a_D-a_A) + (a_E-a_D) + (a_F-a_E):.3f} (should be ~{a_F:.3f})')
        print(f'  Note: elements are not strictly additive due to interactions')

    # ================================================================
    # 図の生成
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) alpha by combination
    ax = axes[0, 0]
    labels_short = [k.split(':')[0] for k in decomp_results.keys()]
    alphas_plot = [v['alpha'] for v in decomp_results.values()]
    errors_plot = [v['e_alpha'] for v in decomp_results.values()]
    y_pos = np.arange(len(labels_short))
    colors_bar = ['#1a1a2e'] * len(labels_short)
    ax.barh(y_pos, alphas_plot, xerr=errors_plot, color=colors_bar, alpha=0.7, capsize=3)
    ax.axvline(0.5, color='#e94560', ls='--', lw=2, label='alpha=0.5')
    ax.axvline(0.545, color='green', ls=':', lw=2, label='alpha=0.545 (original)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_short, fontsize=9)
    ax.set_xlabel('alpha')
    ax.set_title('(a) alpha by parameter combination')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # (b) Υ_d分布
    ax = axes[0, 1]
    ax.hist(ud_vals, bins=20, color='#1a1a2e', alpha=0.7, edgecolor='white')
    ax.axvline(0.5, color='#e94560', ls='--', lw=2, label='Y_d=0.5 (fixed)')
    ax.axvline(np.median(ud_vals), color='green', ls=':', lw=2,
               label=f'median={np.median(ud_vals):.2f}')
    ax.set_xlabel('Y_d (TA3 optimized)')
    ax.set_ylabel('Count')
    ax.set_title('(b) TA3 per-galaxy Y_d distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) gc_ta3 vs gc_v2(Yd=0.5)
    ax = axes[1, 0]
    gc_ta3_plot = [m['gc_ta3'] for m in matched if m['gc_v2_Yd05']]
    gc_v2_plot = [m['gc_v2_Yd05'] for m in matched if m['gc_v2_Yd05']]
    ax.scatter(gc_ta3_plot, gc_v2_plot, s=15, alpha=0.5, color='#1a1a2e')
    lim = [min(min(gc_ta3_plot), min(gc_v2_plot)),
           max(max(gc_ta3_plot), max(gc_v2_plot))]
    ax.plot(lim, lim, 'k--', alpha=0.3, label='1:1')
    ax.set_xlabel('g_c (TA3: tanh fit, Yd optimized)')
    ax.set_ylabel('g_c (v2: median, Yd=0.5)')
    ax.set_title(f'(c) g_c comparison (N={len(gc_ta3_plot)})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) gc_ta3 vs gc_v2(Yd=opt)
    ax = axes[1, 1]
    gc_vopt = [m['gc_v2_Ydopt'] for m in matched if m['gc_v2_Ydopt'] and m['gc_v2_Ydopt'] > 0]
    gc_ta3_opt = [m['gc_ta3'] for m in matched if m['gc_v2_Ydopt'] and m['gc_v2_Ydopt'] > 0]
    if gc_vopt:
        ax.scatter(gc_ta3_opt, gc_vopt, s=15, alpha=0.5, color='#e94560')
        lim2 = [min(min(gc_ta3_opt), min(gc_vopt)),
                max(max(gc_ta3_opt), max(gc_vopt))]
        ax.plot(lim2, lim2, 'k--', alpha=0.3, label='1:1')
        ax.set_xlabel('g_c (TA3)')
        ax.set_ylabel('g_c (v2: median, Yd=TA3 optimized)')
        ax.set_title(f'(d) g_c with matched Yd (N={len(gc_vopt)})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('alpha=0.545 Decomposition Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = OUTDIR / 'alpha_decomposition.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure: {fig_path}')

    # JSON
    summary = {
        'n_matched': len(matched),
        'n_gc_floor': n_gc_floor,
        'n_vf_floor': n_vf_floor,
        'Yd_stats': {
            'median': float(np.median(ud_vals)),
            'mean': float(np.mean(ud_vals)),
            'std': float(np.std(ud_vals)),
        },
        'decomposition': {k: v for k, v in decomp_results.items()},
    }
    with open(OUTDIR / 'alpha_decomposition.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "alpha_decomposition.json"}')


if __name__ == '__main__':
    main()
