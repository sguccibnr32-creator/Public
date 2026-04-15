#!/usr/bin/env python3
"""
sparc_gc_method_comparison.py
=============================
元パイプライン (TA3_gc_independent.csv) と本スクリプトの
g_c測定手法の差を系統的に分析する。

核心的問い:
  同じSPARCデータ、同じΥ_d=0.5、同じG×Σ₀定義で、
  なぜ alpha が 0.545 (TA3) vs 0.670 (v2) と 0.125 異なるのか？

分析項目:
  (1) TA3_gc_independent.csv を読み込み、g_c値を銀河ごとに比較
  (2) g_c測定手法の違いを特定:
      - TA3: どのような最適化/フィッティングでg_cを決めているか？
      - v2: median(g_obs - g_bar) in outer 1/3
  (3) g_cの差分分析: どの銀河でどの程度ずれているか
  (4) g_c定義の変種テスト:
      a) 外側1/3の中央値 (v2のデフォルト)
      b) 外側1/3の平均値
      c) 外側1/2の中央値
      d) 最外点のみ
      e) フラット領域検出 (V_obs変動 < 10%)
      f) V_obsの2乗/R at R_last
  (5) 各定義でのalpha測定 -> どの定義がTA3のalpha=0.545に最も近いか
  (6) Bootstrap 95%CIで手法差の有意性評価

実行: uv run --with scipy --with matplotlib python sparc_gc_method_comparison.py

前提:
  - SPARC Rotmod_LTG/ ディレクトリ
  - TA3_gc_independent.csv（元パイプライン出力）
    存在しない場合はスキップし、g_c定義変種テストのみ実施
"""

import numpy as np
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

# TA3ファイルの候補パス
TA3_CANDIDATES = [
    Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\TA3_gc_independent.csv"),
    Path("TA3_gc_independent.csv"),
    Path("../TA3_gc_independent.csv"),
]

OUTDIR = Path("gc_method_comparison_output")
OUTDIR.mkdir(exist_ok=True)

# 物理定数
G_SI = 6.674e-11
Msun = 1.989e30
pc = 3.086e16
kpc = 1e3 * pc
a0 = 1.2e-10

YD = 0.5
YB = 0.7
N_BOOT = 1000
RNG_SEED = 42


def load_sparc(filepath):
    try:
        data = np.loadtxt(filepath, comments='#')
    except:
        return None
    if data.ndim != 2 or data.shape[1] < 5:
        return None
    return {
        'R': data[:, 0], 'Vo': data[:, 1], 'eV': data[:, 2],
        'Vg': data[:, 3], 'Vd': data[:, 4],
        'Vb': data[:, 5] if data.shape[1] > 5 else np.zeros(len(data)),
        'name': filepath.stem,
    }


def vbar(gal, Yd=YD, Yb=YB):
    return np.sqrt(np.maximum(
        gal['Vg']**2 + Yd * gal['Vd']**2 + Yb * gal['Vb']**2, 0))


def observables(gal):
    """V_flat, h_R, G*Sigma_0 (Yd-independent)"""
    R, Vo, Vd = gal['R'], gal['Vo'], gal['Vd']
    n = len(R)
    if n < 5:
        return None
    outer = slice(2*n//3, n)
    V_flat = np.median(Vo[outer])
    idx_pk = max(np.argmax(np.abs(Vd)), 1)
    h_R = R[idx_pk] / 2.2
    if V_flat <= 0 or h_R <= 0.01:
        return None
    GS0 = (V_flat * 1e3)**2 / (h_R * kpc)
    return {'V_flat': V_flat, 'h_R': h_R, 'GS0': GS0}


# ================================================================
# g_c測定の6つの変種
# ================================================================
def gc_outer_third_median(gal):
    """v2デフォルト: 外側1/3の中央値"""
    R, Vo, Vb_ = gal['R'], gal['Vo'], vbar(gal)
    n = len(R)
    if n < 5: return None
    s = slice(2*n//3, n)
    Rm = R[s] * kpc
    gc = np.median((Vo[s]*1e3)**2/Rm - (Vb_[s]*1e3)**2/Rm)
    return gc if gc > 0 else None


def gc_outer_third_mean(gal):
    """外側1/3の平均値"""
    R, Vo, Vb_ = gal['R'], gal['Vo'], vbar(gal)
    n = len(R)
    if n < 5: return None
    s = slice(2*n//3, n)
    Rm = R[s] * kpc
    gc = np.mean((Vo[s]*1e3)**2/Rm - (Vb_[s]*1e3)**2/Rm)
    return gc if gc > 0 else None


def gc_outer_half_median(gal):
    """外側1/2の中央値"""
    R, Vo, Vb_ = gal['R'], gal['Vo'], vbar(gal)
    n = len(R)
    if n < 5: return None
    s = slice(n//2, n)
    Rm = R[s] * kpc
    gc = np.median((Vo[s]*1e3)**2/Rm - (Vb_[s]*1e3)**2/Rm)
    return gc if gc > 0 else None


def gc_last_point(gal):
    """最外点のみ"""
    R, Vo, Vb_ = gal['R'], gal['Vo'], vbar(gal)
    n = len(R)
    if n < 3: return None
    Rm = R[-1] * kpc
    if Rm <= 0: return None
    gc = (Vo[-1]*1e3)**2/Rm - (Vb_[-1]*1e3)**2/Rm
    return gc if gc > 0 else None


def gc_flat_region(gal):
    """フラット領域検出: V_obsの変動が10%以内の最長連続区間"""
    R, Vo, Vb_ = gal['R'], gal['Vo'], vbar(gal)
    n = len(R)
    if n < 5: return None

    # フラット領域: |V(i)-V(i-1)|/V(i) < 0.1 が連続する区間
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 1
    for i in range(1, n):
        if Vo[i] > 0 and abs(Vo[i] - Vo[i-1]) / Vo[i] < 0.10:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = i, 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    if best_len < 3:
        return None

    s = slice(best_start, best_start + best_len)
    Rm = R[s] * kpc
    gc = np.median((Vo[s]*1e3)**2/Rm - (Vb_[s]*1e3)**2/Rm)
    return gc if gc > 0 else None


def gc_weighted_outer(gal):
    """誤差重み付き外側1/3"""
    R, Vo, eV, Vb_ = gal['R'], gal['Vo'], gal['eV'], vbar(gal)
    n = len(R)
    if n < 5: return None
    s = slice(2*n//3, n)
    Rm = R[s] * kpc
    gc_pts = (Vo[s]*1e3)**2/Rm - (Vb_[s]*1e3)**2/Rm
    w = 1.0 / np.maximum(eV[s], 1.0)**2
    gc = np.average(gc_pts, weights=w)
    return gc if gc > 0 else None


GC_METHODS = {
    'outer_1/3_median': gc_outer_third_median,
    'outer_1/3_mean':   gc_outer_third_mean,
    'outer_1/2_median': gc_outer_half_median,
    'last_point':       gc_last_point,
    'flat_region':      gc_flat_region,
    'weighted_outer':   gc_weighted_outer,
}


def alpha_fit(gc_arr, GS0_arr):
    x = np.log10(GS0_arr)
    y = np.log10(gc_arr)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10: return None
    sl, ic, r, p, se = linregress(x, y)
    t_stat = (sl - 0.5) / se
    p05 = 2 * tdist.sf(abs(t_stat), df=len(x)-2)
    return {'alpha': sl, 'e_alpha': se, 'p05': p05, 'r': r, 'N': len(x)}


def main():
    print('=' * 70)
    print('SPARC g_c Method Comparison')
    print('=' * 70)

    # === 銀河読み込み ===
    files = sorted(SPARC_DIR.glob('*.dat'))
    if not files:
        files = sorted(SPARC_DIR.glob('*_rotmod.dat'))
    if not files:
        print(f'ERROR: {SPARC_DIR}')
        sys.exit(1)

    galaxies = []
    obs_list = []
    for f in files:
        g = load_sparc(f)
        if g is None: continue
        o = observables(g)
        if o is None: continue
        galaxies.append(g)
        obs_list.append(o)
    print(f'{len(galaxies)} galaxies loaded')

    GS0_all = np.array([o['GS0'] for o in obs_list])

    # === TA3比較 (ファイルが存在する場合) ===
    ta3_path = None
    for p in TA3_CANDIDATES:
        if p.exists():
            ta3_path = p
            break

    ta3_gc = {}
    matched = []
    if ta3_path:
        print(f'\nTA3 found: {ta3_path}')
        import csv
        with open(ta3_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('Name', row.get('name', row.get('galaxy', ''))).strip()
                # gc_over_a0 形式 or 直接 gc
                gc_val = (row.get('gc') or row.get('g_c') or row.get('gc_indep')
                          or row.get('gc_over_a0'))
                try:
                    val = float(gc_val)
                    # gc_over_a0 形式なら a0を掛ける
                    if 'gc_over_a0' in row and gc_val == row.get('gc_over_a0'):
                        val = val * a0
                    ta3_gc[name] = val
                except (ValueError, TypeError):
                    pass
        print(f'  {len(ta3_gc)} galaxies in TA3')

        # 銀河ごとのg_c比較
        for gal, obs in zip(galaxies, obs_list):
            name = gal['name']
            ta3_name = name.replace('_rotmod', '').replace('_Rotmod', '')
            gc_ta3 = ta3_gc.get(name) or ta3_gc.get(ta3_name)
            if gc_ta3 is None:
                continue
            gc_v2 = gc_outer_third_median(gal)
            if gc_v2 is None:
                continue
            matched.append({
                'name': name,
                'gc_ta3': gc_ta3,
                'gc_v2': gc_v2,
                'ratio': gc_v2 / gc_ta3 if gc_ta3 > 0 else np.nan,
                'GS0': obs['GS0'],
            })

        print(f'  Matched: {len(matched)} galaxies')

        if len(matched) > 10:
            ratios = np.array([m['ratio'] for m in matched if np.isfinite(m['ratio'])])
            print(f'  g_c(v2) / g_c(TA3):')
            print(f'    median = {np.median(ratios):.3f}')
            print(f'    mean   = {np.mean(ratios):.3f}')
            print(f'    std    = {np.std(ratios):.3f}')
            print(f'    range  = [{np.min(ratios):.3f}, {np.max(ratios):.3f}]')

            # TA3のg_cでalpha測定
            gc_ta3_arr = np.array([m['gc_ta3'] for m in matched])
            gs0_ta3_arr = np.array([m['GS0'] for m in matched])
            fit_ta3 = alpha_fit(gc_ta3_arr, gs0_ta3_arr)
            if fit_ta3:
                print(f'\n  TA3 g_c -> alpha = {fit_ta3["alpha"]:.3f} +/- {fit_ta3["e_alpha"]:.3f}, '
                      f'p(0.5) = {fit_ta3["p05"]:.4f}, N={fit_ta3["N"]}')

            gc_v2_arr = np.array([m['gc_v2'] for m in matched])
            fit_v2m = alpha_fit(gc_v2_arr, gs0_ta3_arr)
            if fit_v2m:
                print(f'  v2 g_c  -> alpha = {fit_v2m["alpha"]:.3f} +/- {fit_v2m["e_alpha"]:.3f}, '
                      f'p(0.5) = {fit_v2m["p05"]:.4f}, N={fit_v2m["N"]}')

            if fit_ta3 and fit_v2m:
                z = (fit_v2m['alpha'] - fit_ta3['alpha']) / np.sqrt(
                    fit_v2m['e_alpha']**2 + fit_ta3['e_alpha']**2)
                print(f'  Delta-alpha z-score: {z:.2f}')
    else:
        print('\nTA3_gc_independent.csv not found. Skipping direct comparison.')
        print('  Searched:', [str(p) for p in TA3_CANDIDATES])

    # ================================================================
    # g_c定義変種テスト
    # ================================================================
    print(f'\n{"="*70}')
    print('g_c Definition Variants Test (Y_d=0.5, Y_b=0.7)')
    print(f'{"="*70}')
    print(f'{"Method":<22} {"N":>5} {"alpha":>8} {"SE":>8} {"p(0.5)":>10} {"r":>7} {"reject":>7}')
    print('-' * 70)

    variant_results = {}
    for method_name, method_func in GC_METHODS.items():
        gc_list = []
        gs0_list = []
        for gal, obs in zip(galaxies, obs_list):
            gc = method_func(gal)
            if gc is not None:
                gc_list.append(gc)
                gs0_list.append(obs['GS0'])

        fit = alpha_fit(np.array(gc_list), np.array(gs0_list))
        if fit:
            reject = 'YES' if fit['p05'] < 0.05 else 'no'
            print(f'{method_name:<22} {fit["N"]:5d} {fit["alpha"]:8.3f} {fit["e_alpha"]:8.3f} '
                  f'{fit["p05"]:10.4f} {fit["r"]:7.3f} {reject:>7}')
            variant_results[method_name] = fit
        else:
            print(f'{method_name:<22}   -- fit failed --')

    # ================================================================
    # Bootstrap各手法
    # ================================================================
    print(f'\n{"="*70}')
    print(f'Bootstrap 95% CI (N_boot={N_BOOT})')
    print(f'{"="*70}')

    rng = np.random.RandomState(RNG_SEED)

    for method_name, method_func in GC_METHODS.items():
        gc_list, gs0_list = [], []
        for gal, obs in zip(galaxies, obs_list):
            gc = method_func(gal)
            if gc is not None:
                gc_list.append(gc)
                gs0_list.append(obs['GS0'])
        gc_arr = np.array(gc_list)
        gs0_arr = np.array(gs0_list)
        N = len(gc_arr)
        if N < 20:
            continue

        alphas = []
        for _ in range(N_BOOT):
            idx = rng.choice(N, N, replace=True)
            fit = alpha_fit(gc_arr[idx], gs0_arr[idx])
            if fit:
                alphas.append(fit['alpha'])
        alphas = np.array(alphas)
        ci = np.percentile(alphas, [2.5, 97.5])
        has_05 = ci[0] <= 0.5 <= ci[1]
        print(f'  {method_name:<22}: {np.mean(alphas):.3f} +/- {np.std(alphas):.3f}, '
              f'95%CI=[{ci[0]:.3f}, {ci[1]:.3f}], 0.5 in CI: {"YES" if has_05 else "NO"}')

    # ================================================================
    # 図の生成
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) alpha by method
    ax = axes[0]
    methods = list(variant_results.keys())
    alphas = [variant_results[m]['alpha'] for m in methods]
    errors = [variant_results[m]['e_alpha'] for m in methods]
    y_pos = np.arange(len(methods))

    ax.barh(y_pos, alphas, xerr=errors, color='#1a1a2e', alpha=0.7, capsize=3)
    ax.axvline(0.5, color='#e94560', ls='--', lw=2, label='alpha=0.5')
    ax.axvline(0.545, color='green', ls=':', lw=2, label='alpha=0.545 (TA3)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('alpha')
    ax.set_title('(a) alpha by g_c definition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    # (b) TA3 vs v2 scatter (if available)
    ax = axes[1]
    if ta3_path and len(matched) > 10:
        gc_ta3_plot = [m['gc_ta3'] for m in matched]
        gc_v2_plot = [m['gc_v2'] for m in matched]
        ax.scatter(gc_ta3_plot, gc_v2_plot, s=15, alpha=0.5, color='#1a1a2e')
        lim = [min(min(gc_ta3_plot), min(gc_v2_plot)),
               max(max(gc_ta3_plot), max(gc_v2_plot))]
        ax.plot(lim, lim, 'k--', alpha=0.3, label='1:1')
        ax.set_xlabel('g_c (TA3)')
        ax.set_ylabel('g_c (v2: outer 1/3 median)')
        ax.set_title(f'(b) g_c comparison (N={len(matched)})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'TA3 data not available\n\nRun gc_geometric_mean_test.py\n'
                'to generate TA3_gc_independent.csv\nthen re-run this script',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title('(b) g_c: TA3 vs v2 (requires TA3 file)')

    plt.tight_layout()
    fig_path = OUTDIR / 'gc_method_comparison.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\nFigure: {fig_path}')

    # JSON
    summary = {
        'variant_results': {k: v for k, v in variant_results.items()},
        'ta3_found': ta3_path is not None,
        'ta3_matched': len(matched) if ta3_path else 0,
    }
    with open(OUTDIR / 'gc_method_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "gc_method_comparison.json"}')

    # ================================================================
    # 結果の解釈ガイド
    # ================================================================
    print(f'\n{"="*70}')
    print('Interpretation Guide')
    print(f'{"="*70}')
    print('''
If all methods give alpha ~ 0.65-0.70:
  -> The 0.125 gap vs TA3 is due to g_c definition, not Y_d.
  -> TA3 likely uses a different approach (e.g., RC fitting, not simple subtraction).
  -> Next step: read TA3 source code to understand its g_c definition.

If some method gives alpha ~ 0.54:
  -> That method matches TA3. The gap is explained by g_c definition choice.
  -> Document which definition is physically most appropriate.

If TA3 g_c values are systematically lower than v2:
  -> TA3 may account for adiabatic contraction or non-circular motions,
     which reduce g_c and push alpha toward 0.5.
''')


if __name__ == '__main__':
    main()
