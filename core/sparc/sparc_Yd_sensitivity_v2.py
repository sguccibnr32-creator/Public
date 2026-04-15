#!/usr/bin/env python3
"""
sparc_Yd_sensitivity_v2.py
==========================
修正版: 元パイプラインと同一の回帰変数定義を使用。

核心的違い:
  旧スクリプト (v1): Sigma_0 = Yd * V_peak^2 / (2*pi*G*h) -> Yd依存
  本スクリプト (v2): G*Sigma_0 = V_flat^2 / h_R           -> Yd非依存 (観測量のみ)

これにより、alpha vs Yd の真の依存性を測定できる。
g_c = g_obs - g_bar のみがΥ_dの影響を受ける。

テスト:
  (1) Υ_d = 0.1-1.5 でalpha測定 (G*Sigma_0はΥ_d固定)
  (2) dalpha/dYd の数値微分
  (3) alpha=0.5を与えるΥ_dの逆算
  (4) g_cが負になる（V_bar > V_obs）Υ_dの閾値
  (5) Bootstrap信頼区間つきのΥ_dスキャン

実行: uv run --with scipy --with matplotlib python sparc_Yd_sensitivity_v2.py
"""

import numpy as np
import sys
import json
from pathlib import Path
from scipy.stats import linregress, t as tdist
from scipy.optimize import brentq
from scipy.interpolate import interp1d
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
a0 = 1.2e-10

# Υ_dグリッド (細かく)
YD_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
YB_DEFAULT = 0.7
N_BOOT = 500
RNG_SEED = 42


def load_sparc_galaxy(filepath):
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


def measure_observables(gal):
    """
    Υ_d非依存の観測量を測定:
    - V_flat: 外側1/3のV_obsの中央値 [km/s]
    - h_R: V_diskピーク位置 / 2.2 [kpc] (指数ディスクのピーク)
    - G*Sigma_0 = V_flat^2 / h_R [m/s^2 * m = m^2/s^2 ... ]
      正確には: G*Sigma_0 ~ V_flat^2 / (h_R [m]) [m/s^2]
    """
    R = gal['R_kpc']
    Vo = gal['V_obs']
    Vd = gal['V_disk']
    n = len(R)
    if n < 5:
        return None

    # V_flat (外側1/3中央値)
    outer = slice(2*n//3, n)
    V_flat = np.median(Vo[outer])  # km/s
    if V_flat <= 0:
        return None

    # h_R from V_disk peak
    idx_peak = np.argmax(np.abs(Vd))
    if idx_peak == 0:
        idx_peak = min(1, n-1)
    R_peak = R[idx_peak]
    h_R_kpc = R_peak / 2.2
    if h_R_kpc <= 0.01:
        return None

    h_R_m = h_R_kpc * kpc

    # G*Sigma_0 = V_flat^2 / h_R [SI: m/s^2]
    # これは加速度の次元を持ち、a_0と直接比較可能
    G_Sigma0 = (V_flat * 1e3)**2 / h_R_m  # m/s^2

    return {
        'V_flat': V_flat,
        'h_R_kpc': h_R_kpc,
        'G_Sigma0': G_Sigma0,
    }


def measure_gc(gal, Yd, Yb):
    """
    g_c = g_obs - g_bar (外側1/3の中央値)
    g_barのみΥ_dに依存する。
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

    outer = slice(2*n//3, n)
    R_m = R[outer] * kpc
    if len(R_m) < 2 or np.any(R_m <= 0):
        return None

    g_obs = (Vo[outer] * 1e3)**2 / R_m
    g_bar = (V_bar[outer] * 1e3)**2 / R_m

    gc_vals = g_obs - g_bar
    gc = np.median(gc_vals)

    # g_c < 0 の場合（V_bar > V_obs: oversubtraction）
    n_negative = np.sum(gc_vals < 0)

    return {
        'gc': gc,
        'n_negative': n_negative,
        'n_points': len(gc_vals),
    }


def alpha_fit(gc_arr, GS0_arr):
    """
    log(g_c) = log(eta) + alpha * log(a_0 * G*Sigma_0)
    ここでG*Sigma_0は既にG込みの値なので、a_0 * G*Sigma_0 を横軸にする。

    注意: 元パイプラインの正確な定義を確認する必要あり。
    ここでは x = log10(G*Sigma_0) として回帰する。
    a_0との積は定数オフセットなのでalphaには影響しない。
    """
    x = np.log10(GS0_arr)
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
        'r': r, 'N': len(x),
    }


def main():
    print('=' * 65)
    print('SPARC Y_d 感度テスト v2')
    print('G*Sigma_0 = V_flat^2/h_R (Y_d非依存)')
    print('=' * 65)

    # 全銀河読み込み
    files = sorted(SPARC_DIR.glob('*.dat'))
    if not files:
        files = sorted(SPARC_DIR.glob('*_rotmod.dat'))
    if not files:
        print(f'ERROR: {SPARC_DIR}')
        sys.exit(1)

    galaxies = []
    for f in files:
        gal = load_sparc_galaxy(f)
        if gal is not None:
            galaxies.append(gal)
    print(f'{len(galaxies)} galaxies loaded')

    # === 事前計算: Υ_d非依存の観測量 ===
    obs_data = []
    valid_galaxies = []
    for gal in galaxies:
        obs = measure_observables(gal)
        if obs is not None:
            obs_data.append(obs)
            valid_galaxies.append(gal)
    print(f'{len(valid_galaxies)} galaxies with valid observables')

    GS0_all = np.array([o['G_Sigma0'] for o in obs_data])

    # ================================================================
    # テスト1: Υ_d 1Dスキャン
    # ================================================================
    print(f'\n{"="*65}')
    print('Test 1: Y_d scan (G*Sigma_0 = V_flat^2/h_R, Y_d-independent)')
    print(f'{"="*65}')
    print(f'{"Y_d":>6} {"N":>5} {"N_gc>0":>7} {"alpha":>8} {"SE":>8} '
          f'{"p(0.5)":>10} {"r":>7} {"reject":>7}')
    print('-' * 65)

    results_1d = []
    for Yd in YD_GRID:
        gc_list = []
        gs0_list = []
        n_gc_pos = 0
        n_gc_neg = 0

        for gal, obs in zip(valid_galaxies, obs_data):
            res = measure_gc(gal, Yd, YB_DEFAULT)
            if res is None:
                continue
            if res['gc'] > 0:
                gc_list.append(res['gc'])
                gs0_list.append(obs['G_Sigma0'])
                n_gc_pos += 1
            else:
                n_gc_neg += 1

        gc_arr = np.array(gc_list)
        gs0_arr = np.array(gs0_list)
        fit = alpha_fit(gc_arr, gs0_arr)

        entry = {
            'Yd': Yd,
            'N_total': n_gc_pos + n_gc_neg,
            'N_gc_pos': n_gc_pos,
            'N_gc_neg': n_gc_neg,
        }

        if fit:
            entry.update(fit)
            reject = 'YES' if fit['p05'] < 0.05 else 'no'
            print(f'{Yd:6.2f} {fit["N"]:5d} {n_gc_pos:7d} {fit["alpha"]:8.3f} '
                  f'{fit["e_alpha"]:8.3f} {fit["p05"]:10.4f} {fit["r"]:7.3f} {reject:>7}')
        else:
            print(f'{Yd:6.2f}   -- fit failed (N_pos={n_gc_pos}, N_neg={n_gc_neg}) --')

        results_1d.append(entry)

    # ================================================================
    # テスト2: dalpha/dYd
    # ================================================================
    print(f'\n{"="*65}')
    print('Test 2: dalpha/dYd')
    print(f'{"="*65}')

    valid_1d = [r for r in results_1d if 'alpha' in r]
    if len(valid_1d) >= 3:
        Yd_arr = np.array([r['Yd'] for r in valid_1d])
        alpha_arr = np.array([r['alpha'] for r in valid_1d])

        # 中心差分
        for target_Yd in [0.3, 0.5, 0.7]:
            idx = np.argmin(np.abs(Yd_arr - target_Yd))
            if 0 < idx < len(Yd_arr) - 1:
                dYd = Yd_arr[idx+1] - Yd_arr[idx-1]
                dalpha = alpha_arr[idx+1] - alpha_arr[idx-1]
                sens = dalpha / dYd
                print(f'  dalpha/dYd at Y_d={target_Yd:.1f}: {sens:.3f}')
                print(f'    -> Y_d +/- 0.05 (10%) -> alpha +/- {abs(sens)*0.05:.4f}')

        # 全範囲線形フィット
        sl, ic, _, _, _ = linregress(Yd_arr, alpha_arr)
        print(f'\n  Linear fit: alpha = {sl:.4f} * Y_d + {ic:.4f}')
        print(f'  Overall slope: {sl:.4f} per unit Y_d')

    # ================================================================
    # テスト3: alpha=0.5を与えるΥ_d
    # ================================================================
    print(f'\n{"="*65}')
    print('Test 3: Y_d that gives alpha=0.5')
    print(f'{"="*65}')

    if len(valid_1d) >= 3:
        try:
            f_interp = interp1d(Yd_arr, alpha_arr - 0.5, kind='linear',
                               fill_value='extrapolate')
            # 交差点を探す
            found_crossing = False
            for i in range(len(Yd_arr) - 1):
                v0 = alpha_arr[i] - 0.5
                v1 = alpha_arr[i+1] - 0.5
                if v0 * v1 < 0:
                    Yd_cross = brentq(f_interp, Yd_arr[i], Yd_arr[i+1])
                    print(f'  alpha=0.5 at Y_d = {Yd_cross:.4f}')
                    in_range = 0.3 <= Yd_cross <= 0.9
                    print(f'  Standard range (0.3-0.9): {"YES" if in_range else "NO"}')
                    found_crossing = True
                    break
            if not found_crossing:
                if np.all(alpha_arr > 0.5):
                    print(f'  alpha > 0.5 for all Y_d in [{Yd_arr[0]:.2f}, {Yd_arr[-1]:.2f}]')
                    print(f'  -> alpha=0.5 requires Y_d > {Yd_arr[-1]:.1f} (extrapolation)')
                    # 外挿
                    Yd_extrap = (0.5 - ic) / sl
                    print(f'  -> Linear extrapolation: Y_d ~= {Yd_extrap:.2f}')
                elif np.all(alpha_arr < 0.5):
                    print(f'  alpha < 0.5 for all Y_d')
                    print(f'  -> alpha=0.5 is never reached (alpha always below)')
        except Exception as e:
            print(f'  Error: {e}')

    # ================================================================
    # テスト4: g_c < 0 の割合 vs Υ_d
    # ================================================================
    print(f'\n{"="*65}')
    print('Test 4: Oversubtraction (g_c < 0) fraction vs Y_d')
    print(f'{"="*65}')
    print(f'{"Y_d":>6} {"N_pos":>6} {"N_neg":>6} {"frac_neg":>9}')
    for r in results_1d:
        n_pos = r.get('N_gc_pos', 0)
        n_neg = r.get('N_gc_neg', 0)
        total = n_pos + n_neg
        frac = n_neg / total if total > 0 else 0
        print(f'{r["Yd"]:6.2f} {n_pos:6d} {n_neg:6d} {frac:9.3f}')

    # ================================================================
    # テスト5: Bootstrap Υ_dスキャン (主要3点)
    # ================================================================
    print(f'\n{"="*65}')
    print(f'Test 5: Bootstrap Y_d scan (N_boot={N_BOOT})')
    print(f'{"="*65}')

    rng = np.random.RandomState(RNG_SEED)

    for Yd_target in [0.3, 0.5, 0.7]:
        # 全銀河のg_c計算
        gc_full = []
        gs0_full = []
        for gal, obs in zip(valid_galaxies, obs_data):
            res = measure_gc(gal, Yd_target, YB_DEFAULT)
            if res and res['gc'] > 0:
                gc_full.append(res['gc'])
                gs0_full.append(obs['G_Sigma0'])
        gc_full = np.array(gc_full)
        gs0_full = np.array(gs0_full)
        N = len(gc_full)

        alpha_boot = []
        for _ in range(N_BOOT):
            idx = rng.choice(N, N, replace=True)
            fit = alpha_fit(gc_full[idx], gs0_full[idx])
            if fit:
                alpha_boot.append(fit['alpha'])
        alpha_boot = np.array(alpha_boot)

        ci = np.percentile(alpha_boot, [2.5, 97.5])
        contains_05 = ci[0] <= 0.5 <= ci[1]
        print(f'  Y_d={Yd_target:.1f}: alpha={np.mean(alpha_boot):.3f} +/- {np.std(alpha_boot):.3f}, '
              f'95%CI=[{ci[0]:.3f}, {ci[1]:.3f}], alpha=0.5 in CI: {"YES" if contains_05 else "NO"}')

    # ================================================================
    # 図の生成
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    valid_1d = [r for r in results_1d if 'alpha' in r]
    Yd_plot = [r['Yd'] for r in valid_1d]
    alpha_plot = [r['alpha'] for r in valid_1d]
    e_alpha_plot = [r['e_alpha'] for r in valid_1d]
    p05_plot = [r['p05'] for r in valid_1d]

    # (a) alpha vs Y_d
    ax = axes[0, 0]
    ax.errorbar(Yd_plot, alpha_plot, yerr=e_alpha_plot, fmt='o-', color='#1a1a2e',
                capsize=3, markersize=6, lw=2)
    ax.axhline(0.5, color='#e94560', ls='--', lw=2, label='alpha=0.5')
    ax.axhline(1.0, color='blue', ls=':', lw=1, label='alpha=1.0 (MOND)')
    ax.axvline(0.5, color='grey', ls=':', alpha=0.5)
    ax.axvspan(0.3, 0.9, alpha=0.08, color='green', label='standard Y_d range')
    ax.set_xlabel('Y_disk [M_sun/L_sun]', fontsize=11)
    ax.set_ylabel('alpha', fontsize=11)
    ax.set_title('(a) alpha vs Y_d\n(G*Sigma_0 = V_flat^2/h_R: Y_d-independent)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(alpha_plot)-0.1, top=max(max(alpha_plot)+0.1, 0.7))

    # (b) p(alpha=0.5) vs Y_d
    ax = axes[0, 1]
    ax.semilogy(Yd_plot, [max(p, 1e-10) for p in p05_plot], 'o-', color='#1a1a2e',
                markersize=6, lw=2)
    ax.axhline(0.05, color='#e94560', ls='--', lw=2, label='p=0.05')
    ax.axvline(0.5, color='grey', ls=':', alpha=0.5)
    ax.set_xlabel('Y_disk [M_sun/L_sun]', fontsize=11)
    ax.set_ylabel('p(alpha=0.5)', fontsize=11)
    ax.set_title('(b) p(alpha=0.5) vs Y_d', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # p>0.05をマーク
    for yd, pv in zip(Yd_plot, p05_plot):
        if pv > 0.05:
            ax.plot(yd, max(pv, 1e-10), 'o', color='#d4edda', markersize=14, zorder=0)

    # (c) oversubtraction割合
    ax = axes[1, 0]
    Yd_os = [r['Yd'] for r in results_1d]
    frac_neg = [r.get('N_gc_neg',0)/(r.get('N_gc_pos',0)+r.get('N_gc_neg',1))
                for r in results_1d]
    ax.bar(Yd_os, frac_neg, width=0.04, color='#e94560', alpha=0.7)
    ax.set_xlabel('Y_disk [M_sun/L_sun]', fontsize=11)
    ax.set_ylabel('Oversubtraction fraction (g_c < 0)', fontsize=11)
    ax.set_title('(c) V_bar > V_obs fraction vs Y_d', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(0.5, color='grey', ls=':', alpha=0.5)

    # (d) dalpha/dYd (数値微分)
    ax = axes[1, 1]
    if len(Yd_plot) >= 3:
        Yd_mid = []
        dalpha_dYd = []
        for i in range(1, len(Yd_plot)-1):
            Yd_mid.append(Yd_plot[i])
            da = (alpha_plot[i+1] - alpha_plot[i-1]) / (Yd_plot[i+1] - Yd_plot[i-1])
            dalpha_dYd.append(da)
        ax.plot(Yd_mid, dalpha_dYd, 'o-', color='#1a1a2e', markersize=6, lw=2)
        ax.axhline(0, color='grey', ls='-', alpha=0.3)
        ax.axvline(0.5, color='grey', ls=':', alpha=0.5)
    ax.set_xlabel('Y_disk [M_sun/L_sun]', fontsize=11)
    ax.set_ylabel('dalpha/dY_d', fontsize=11)
    ax.set_title('(d) Sensitivity dalpha/dY_d', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('SPARC Y_d Sensitivity Test v2\n'
                 '(G*Sigma_0 = V_flat^2/h_R: Y_d-independent regressor)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    fig_path = OUTDIR / 'sparc_Yd_sensitivity_v2.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure: {fig_path}')

    # JSON保存
    summary = {'results_1d': valid_1d}
    with open(OUTDIR / 'Yd_sensitivity_v2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "Yd_sensitivity_v2_summary.json"}')


if __name__ == '__main__':
    main()
