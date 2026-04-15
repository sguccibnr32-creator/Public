#!/usr/bin/env python3
"""
sparc_Yd_sensitivity.py
=======================
SPARC 175銀河でΥ_dを0.1-1.5の範囲で系統的に変化させ、
alphaの応答を測定する感度テスト。

核心的問い:
  PROBESではΥ_d増加 -> alpha減少の単調トレンドが確認された。
  SPARCでも同じトレンドが存在するか？
  alpha=0.5はΥ_d=0.5の「帰結」か、それとも「物理」か？

テスト内容:
  (1) Υ_d = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5
      でalphaを測定（Υ_bul=0.7固定）
  (2) Υ_bulも同時に振る2Dグリッド（Υ_d x Υ_b）
  (3) dalpha/dYd の数値微分 -> Υ_d=0.5近傍での感度
  (4) alpha=0.5となるΥ_dの逆算 -> 天文学的標準値と一致するか
  (5) MOND比較: g_c=a0 (alpha=1) が成立するΥ_dの特定

実行: uv run --with scipy --with matplotlib python sparc_Yd_sensitivity.py

前提: SPARCデータが Rotmod_LTG/ ディレクトリに存在すること。
"""

import numpy as np
import sys
import json
from pathlib import Path
from scipy.stats import linregress, t as tdist
from scipy.optimize import brentq
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

OUTDIR = Path("sparc_Yd_sensitivity_output")
OUTDIR.mkdir(exist_ok=True)

# 物理定数
G_SI = 6.674e-11
Msun = 1.989e30
pc = 3.086e16
kpc = 1e3 * pc
a0 = 1.2e-10  # m/s^2

# Υ_dグリッド
YD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
YB_DEFAULT = 0.7


def load_sparc_galaxy(filepath):
    """SPARC Rotmod_LTGファイル読み込み"""
    try:
        data = np.loadtxt(filepath, comments='#')
    except:
        return None
    if data.ndim != 2 or data.shape[1] < 5:
        return None
    return {
        'R_kpc': data[:, 0],
        'V_obs': data[:, 1],
        'e_V':   data[:, 2],
        'V_gas': data[:, 3],
        'V_disk': data[:, 4],
        'V_bul': data[:, 5] if data.shape[1] > 5 else np.zeros(len(data)),
        'name':  filepath.stem,
    }


def compute_gc_sigma0(gal, Yd, Yb):
    """
    g_cとSigma_0を測定。
    V_bar^2 = V_gas^2 + Yd*V_disk^2 + Yb*V_bul^2
    """
    R = gal['R_kpc']
    Vo = gal['V_obs']
    Vg = gal['V_gas']
    Vd = gal['V_disk']
    Vb = gal['V_bul']

    n = len(R)
    if n < 5:
        return None

    V_bar2 = Vg**2 + Yd * Vd**2 + Yb * Vb**2
    V_bar = np.sqrt(np.maximum(V_bar2, 0))

    # 外側1/3でg_c測定
    outer = slice(2*n//3, n)
    R_m = R[outer] * kpc
    if len(R_m) < 2 or np.any(R_m <= 0):
        return None

    g_obs = (Vo[outer] * 1e3)**2 / R_m
    g_bar = (V_bar[outer] * 1e3)**2 / R_m

    gc_vals = g_obs - g_bar
    gc = np.median(gc_vals)

    if gc <= 0:
        return None

    # Sigma_0推定: V_disk peakからスケール長を推定
    idx_peak = np.argmax(np.abs(Vd))
    if idx_peak == 0:
        idx_peak = 1
    R_peak = R[idx_peak]
    h_kpc = R_peak / 2.2
    V_d_peak = abs(Vd[idx_peak])

    if h_kpc <= 0 or V_d_peak <= 0:
        return None

    h_m = h_kpc * kpc
    # Sigma_0 = Yd * V_d_peak^2 / (2*pi*G*h)  [SI] -> [M_sun/pc^2]
    Sigma0_SI = Yd * (V_d_peak * 1e3)**2 / (2 * np.pi * G_SI * h_m)
    Sigma0 = Sigma0_SI * pc**2 / Msun

    if Sigma0 <= 0:
        return None

    return {'gc': gc, 'Sigma0': Sigma0}


def alpha_fit(gc_arr, S0_arr):
    """線形回帰 log(g_c) = log(eta) + alpha * log(a0*G*Sigma_0)"""
    x = np.log10(a0 * G_SI * S0_arr * Msun / pc**2)
    y = np.log10(gc_arr)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        return None

    slope, intercept, r, p, se = linregress(x, y)
    t_stat = (slope - 0.5) / se
    p_05 = 2 * tdist.sf(abs(t_stat), df=len(x) - 2)

    return {
        'alpha': slope, 'e_alpha': se, 'p05': p_05,
        'r': r, 'N': len(x), 'intercept': intercept,
    }


def main():
    print('=' * 60)
    print('SPARC Y_d 感度テスト')
    print('=' * 60)

    # 全銀河読み込み
    files = sorted(SPARC_DIR.glob('*.dat'))
    if not files:
        files = sorted(SPARC_DIR.glob('*_rotmod.dat'))
    if not files:
        print(f'ERROR: .datファイルが見つかりません: {SPARC_DIR}')
        sys.exit(1)

    galaxies = []
    for f in files:
        gal = load_sparc_galaxy(f)
        if gal is not None:
            galaxies.append(gal)
    print(f'{len(galaxies)} 銀河読み込み')

    # ================================================================
    # テスト1: Υ_d 1Dスキャン (Υ_b=0.7固定)
    # ================================================================
    print(f'\n--- テスト1: Y_d 1Dスキャン (Y_b={YB_DEFAULT}) ---')
    print(f'{"Y_d":>6} {"N":>5} {"alpha":>8} {"SE":>8} {"p(0.5)":>10} {"r":>7}')
    print('-' * 50)

    results_1d = []
    for Yd in YD_GRID:
        gc_list = []
        S0_list = []
        for gal in galaxies:
            res = compute_gc_sigma0(gal, Yd, YB_DEFAULT)
            if res:
                gc_list.append(res['gc'])
                S0_list.append(res['Sigma0'])

        gc_arr = np.array(gc_list)
        S0_arr = np.array(S0_list)
        fit = alpha_fit(gc_arr, S0_arr)

        if fit:
            print(f'{Yd:6.2f} {fit["N"]:5d} {fit["alpha"]:8.3f} {fit["e_alpha"]:8.3f} '
                  f'{fit["p05"]:10.4f} {fit["r"]:7.3f}')
            results_1d.append({'Yd': Yd, **fit})
        else:
            print(f'{Yd:6.2f}   -- fit failed --')

    # ================================================================
    # テスト2: Υ_d x Υ_b 2Dグリッド
    # ================================================================
    print(f'\n--- テスト2: Y_d x Y_b 2Dグリッド ---')
    YD_2D = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    YB_2D = [0.3, 0.5, 0.7, 0.9, 1.1]

    print(f'{"":>6}', end='')
    for Yb in YB_2D:
        print(f' Yb={Yb:.1f}', end='')
    print()
    print('-' * (6 + 9 * len(YB_2D)))

    results_2d = {}
    for Yd in YD_2D:
        print(f'Yd={Yd:.1f}', end='')
        for Yb in YB_2D:
            gc_list, S0_list = [], []
            for gal in galaxies:
                res = compute_gc_sigma0(gal, Yd, Yb)
                if res:
                    gc_list.append(res['gc'])
                    S0_list.append(res['Sigma0'])
            fit = alpha_fit(np.array(gc_list), np.array(S0_list))
            if fit:
                print(f'  {fit["alpha"]:6.3f}', end='')
                results_2d[(Yd, Yb)] = fit
            else:
                print(f'    --  ', end='')
        print()

    # ================================================================
    # テスト3: dalpha/dYd (数値微分)
    # ================================================================
    print(f'\n--- テスト3: dalpha/dYd (Y_d=0.5近傍) ---')
    if len(results_1d) >= 3:
        Yd_arr = np.array([r['Yd'] for r in results_1d])
        alpha_arr = np.array([r['alpha'] for r in results_1d])

        # Υ_d=0.5の前後で数値微分
        idx_05 = np.argmin(np.abs(Yd_arr - 0.5))
        if 0 < idx_05 < len(Yd_arr) - 1:
            dYd = Yd_arr[idx_05 + 1] - Yd_arr[idx_05 - 1]
            dalpha = alpha_arr[idx_05 + 1] - alpha_arr[idx_05 - 1]
            sensitivity = dalpha / dYd
            print(f'  dalpha/dYd at Y_d=0.5: {sensitivity:.3f}')
            print(f'  -> Y_dを0.1変えるとalphaが{abs(sensitivity)*0.1:.3f}変化')
            print(f'  -> Y_dの10%不確かさ(+/-0.05)でalphaは+/-{abs(sensitivity)*0.05:.3f}')

        # 全範囲の勾配
        from scipy.stats import linregress as lr
        sl, ic, _, _, _ = lr(Yd_arr, alpha_arr)
        print(f'  全範囲線形近似: alpha = {sl:.3f} x Y_d + {ic:.3f}')

    # ================================================================
    # テスト4: alpha=0.5を与えるΥ_dの逆算
    # ================================================================
    print(f'\n--- テスト4: alpha=0.5を与えるY_dの逆算 ---')
    if len(results_1d) >= 3:
        Yd_arr = np.array([r['Yd'] for r in results_1d])
        alpha_arr = np.array([r['alpha'] for r in results_1d])

        # 補間でalpha=0.5となるΥ_dを探す
        from scipy.interpolate import interp1d
        try:
            f_interp = interp1d(Yd_arr, alpha_arr - 0.5, kind='linear')
            # alpha-0.5がゼロになる点を探す
            for i in range(len(Yd_arr) - 1):
                if (alpha_arr[i] - 0.5) * (alpha_arr[i+1] - 0.5) < 0:
                    Yd_cross = brentq(f_interp, Yd_arr[i], Yd_arr[i+1])
                    print(f'  alpha=0.5 at Y_d = {Yd_cross:.3f}')
                    if 0.3 <= Yd_cross <= 0.9:
                        print(f'  -> 天文学的標準範囲 (0.3-0.9) 内: YES')
                    else:
                        print(f'  -> 天文学的標準範囲 (0.3-0.9) 内: NO')
                    break
            else:
                # 交差しない場合
                if np.all(alpha_arr > 0.5):
                    print(f'  alpha > 0.5 for all Y_d in [{Yd_arr[0]:.1f}, {Yd_arr[-1]:.1f}]')
                    print(f'  alpha=0.5には Y_d > {Yd_arr[-1]:.1f} が必要')
                elif np.all(alpha_arr < 0.5):
                    print(f'  alpha < 0.5 for all Y_d in [{Yd_arr[0]:.1f}, {Yd_arr[-1]:.1f}]')
        except Exception as e:
            print(f'  補間エラー: {e}')

    # ================================================================
    # テスト5: MOND比較 (alpha=1を与えるΥ_d)
    # ================================================================
    print(f'\n--- テスト5: alpha=1.0 (MOND相当) を与えるY_d ---')
    if len(results_1d) >= 3:
        try:
            f_interp1 = interp1d(Yd_arr, alpha_arr - 1.0, kind='linear')
            for i in range(len(Yd_arr) - 1):
                if (alpha_arr[i] - 1.0) * (alpha_arr[i+1] - 1.0) < 0:
                    Yd_mond = brentq(f_interp1, Yd_arr[i], Yd_arr[i+1])
                    print(f'  alpha=1.0 at Y_d = {Yd_mond:.3f}')
                    break
            else:
                if np.all(alpha_arr < 1.0):
                    print(f'  alpha < 1.0 for all Y_d -> MOND (alpha=1) は全Y_dで棄却')
                elif np.all(alpha_arr > 1.0):
                    print(f'  alpha > 1.0 for all Y_d -> 予想外の結果')
        except Exception as e:
            print(f'  補間エラー: {e}')

    # ================================================================
    # 図の生成
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) alpha vs Y_d (1D)
    ax = axes[0, 0]
    Yd_plot = [r['Yd'] for r in results_1d]
    alpha_plot = [r['alpha'] for r in results_1d]
    e_alpha_plot = [r['e_alpha'] for r in results_1d]
    ax.errorbar(Yd_plot, alpha_plot, yerr=e_alpha_plot, fmt='o-', color='#1a1a2e',
                capsize=3, markersize=6)
    ax.axhline(0.5, color='#e94560', ls='--', lw=2, label='alpha=0.5')
    ax.axhline(1.0, color='blue', ls=':', lw=1, label='alpha=1.0 (MOND)')
    ax.axvline(0.5, color='grey', ls=':', alpha=0.5, label='Y_d=0.5 (standard)')
    ax.axvspan(0.3, 0.9, alpha=0.1, color='green', label='standard range')
    ax.set_xlabel('Y_disk [M_sun/L_sun]')
    ax.set_ylabel('alpha')
    ax.set_title('(a) alpha vs Y_d (SPARC, Y_b=0.7)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) p(alpha=0.5) vs Y_d
    ax = axes[0, 1]
    p05_plot = [r['p05'] for r in results_1d]
    ax.semilogy(Yd_plot, p05_plot, 'o-', color='#1a1a2e', markersize=6)
    ax.axhline(0.05, color='#e94560', ls='--', label='p=0.05')
    ax.axvline(0.5, color='grey', ls=':', alpha=0.5)
    ax.set_xlabel('Y_disk [M_sun/L_sun]')
    ax.set_ylabel('p(alpha=0.5)')
    ax.set_title('(b) p(alpha=0.5) vs Y_d')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # p>0.05の領域をマーク
    for i, (yd, pv) in enumerate(zip(Yd_plot, p05_plot)):
        if pv > 0.05:
            ax.plot(yd, pv, 'o', color='#d4edda', markersize=12, zorder=0)

    # (c) N vs Y_d
    ax = axes[1, 0]
    N_plot = [r['N'] for r in results_1d]
    ax.bar(Yd_plot, N_plot, width=0.07, color='#1a1a2e', alpha=0.7)
    ax.set_xlabel('Y_disk [M_sun/L_sun]')
    ax.set_ylabel('N (valid galaxies)')
    ax.set_title('(c) Number of valid galaxies vs Y_d')
    ax.grid(True, alpha=0.3, axis='y')

    # (d) 2Dヒートマップ (Y_d x Y_b)
    ax = axes[1, 1]
    alpha_2d = np.full((len(YD_2D), len(YB_2D)), np.nan)
    for i, Yd in enumerate(YD_2D):
        for j, Yb in enumerate(YB_2D):
            if (Yd, Yb) in results_2d:
                alpha_2d[i, j] = results_2d[(Yd, Yb)]['alpha']

    im = ax.imshow(alpha_2d, aspect='auto', origin='lower',
                   extent=[YB_2D[0]-0.1, YB_2D[-1]+0.1, YD_2D[0]-0.05, YD_2D[-1]+0.05],
                   cmap='RdYlGn_r', vmin=0.3, vmax=0.8)
    ax.set_xlabel('Y_bul [M_sun/L_sun]')
    ax.set_ylabel('Y_disk [M_sun/L_sun]')
    ax.set_title('(d) alpha(Y_d, Y_b) 2D map')
    plt.colorbar(im, ax=ax, label='alpha')
    # alpha=0.5等高線
    try:
        ax.contour(np.linspace(YB_2D[0], YB_2D[-1], len(YB_2D)),
                   np.linspace(YD_2D[0], YD_2D[-1], len(YD_2D)),
                   alpha_2d, levels=[0.5], colors='#e94560', linewidths=2)
    except:
        pass

    plt.tight_layout()
    fig_path = OUTDIR / 'sparc_Yd_sensitivity.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\n figure: {fig_path}')

    # === 結果JSON保存 ===
    summary = {
        'test1_1d': results_1d,
        'test2_2d': {f'Yd={k[0]}_Yb={k[1]}': v for k, v in results_2d.items()},
    }
    with open(OUTDIR / 'Yd_sensitivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'results: {OUTDIR / "Yd_sensitivity_summary.json"}')


if __name__ == '__main__':
    main()
