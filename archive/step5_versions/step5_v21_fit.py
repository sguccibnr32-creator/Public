# -*- coding: utf-8 -*-
"""
Step 5: v2.1 モデル（P(y)=y² 理論移行関数）での回転曲線フィット
v_flat 1パラメータのみで全回転曲線を再現できるか検証

使用例:
  python -X utf8 step5_v21_fit.py --rotmod-dir Rotmod_LTG
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, curve_fit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# --- 定数 ---
G_c      = 1.2e-10    # m/s² (a₀)
kpc_to_m = 3.086e19
x50      = 1.628      # tanh→P(y²) 変換係数

# --- モデル関数 ---

def rs_from_vflat(vflat_kms):
    """r_s = 0.814 × v_flat² / g_c [kpc]"""
    v_ms = vflat_kms * 1e3
    return 0.814 * v_ms**2 / G_c / kpc_to_m

def F_theory(x):
    """
    P(y)=y² から導出した移行関数
    x > 1：F = 12×[1/(3x³) - 1/(4x⁴)]
    x ≤ 1：F = 1
    """
    x = np.atleast_1d(np.float64(x))
    F = np.ones_like(x)
    mask = x > 1
    xi = x[mask]
    F[mask] = 12.0 * (1.0/(3.0*xi**3) - 1.0/(4.0*xi**4))
    return np.clip(F, 0, 1)

def v_model(r_kpc, vflat_kms, v_disk, v_gas, v_bul):
    """
    v2.1 モデルの回転速度
    r_s は v_flat から自動計算
    """
    rs = rs_from_vflat(vflat_kms)
    x  = r_kpc / rs
    Fx = F_theory(x)
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat_kms**2 * Fx
    return np.sqrt(np.clip(v2, 0, None))

# --- フィット ---

def fit_galaxy(df, name, vflat_range=(20, 350)):
    """v_flat 1パラメータで最小化"""
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd   = df['Vdisk'].values
    vg   = df['Vgas'].values
    vb   = df['Vbul'].values

    def chi2(vflat):
        vmod = v_model(r, vflat, vd, vg, vb)
        return np.sum(((vobs - vmod) / verr)**2)

    res = minimize_scalar(
        chi2,
        bounds=vflat_range,
        method='bounded',
        options={'xatol': 0.1}
    )
    vflat_best = res.x
    rs_best    = rs_from_vflat(vflat_best)
    chi2_best  = res.fun
    dof        = max(len(r) - 1, 1)

    # tanh モデルとの比較用（2パラメータ: vflat, rs）
    def v_tanh(r_kpc, vflat_t, rs_t, w=1.5):
        x  = r_kpc / rs_t
        Fx = 0.5 * (1.0 + np.tanh(w * (x - 1.0)))
        return np.sqrt(np.clip(
            vd_interp(r_kpc)**2 + vg_interp(r_kpc)**2
            + vb_interp(r_kpc)**2 + vflat_t**2 * Fx,
            0, None))

    # interp を使うため別に定義
    from scipy.interpolate import interp1d
    vd_interp = interp1d(r, vd, fill_value='extrapolate')
    vg_interp = interp1d(r, vg, fill_value='extrapolate')
    vb_interp = interp1d(r, vb, fill_value='extrapolate')

    try:
        def v_tanh_fit(r_kpc, vflat_t, rs_t):
            x  = r_kpc / rs_t
            Fx = 0.5 * (1.0 + np.tanh(1.5 * (x - 1.0)))
            return np.sqrt(np.clip(
                vd**2 + vg**2 + vb**2 + vflat_t**2 * Fx,
                0, None))

        popt, _ = curve_fit(
            v_tanh_fit, r, vobs,
            p0=[vflat_best, rs_best],
            bounds=([10, 0.1], [400, 100]),
            sigma=verr, maxfev=5000)
        vf_tanh, rs_tanh = popt
        vmod_tanh = v_tanh_fit(r, vf_tanh, rs_tanh)
        chi2_tanh = np.sum(((vobs - vmod_tanh) / verr)**2) / dof
    except Exception:
        vf_tanh   = np.nan
        rs_tanh   = np.nan
        chi2_tanh = np.nan

    # ΔAIC = AIC_v21 - AIC_tanh
    # AIC = chi2 + 2k
    # v2.1: k=1, tanh: k=2
    chi2_tanh_total = chi2_tanh * dof if not np.isnan(chi2_tanh) else np.nan
    delta_aic = (chi2_best + 2*1) - (chi2_tanh_total + 2*2) if not np.isnan(chi2_tanh_total) else np.nan

    return {
        'galaxy':        name,
        'n_pts':         len(r),
        'vflat_v21':     round(vflat_best, 1),
        'rs_v21':        round(rs_best, 2),
        'chi2dof_v21':   round(chi2_best / dof, 3),
        'vflat_tanh':    round(vf_tanh, 1) if not np.isnan(vf_tanh) else np.nan,
        'rs_tanh':       round(rs_tanh, 2) if not np.isnan(rs_tanh) else np.nan,
        'chi2dof_tanh':  round(chi2_tanh, 3) if not np.isnan(chi2_tanh) else np.nan,
        'dAIC':          round(delta_aic, 1) if not np.isnan(delta_aic) else np.nan,
    }

# --- プロット（1銀河） ---

def plot_galaxy(df, result, ax):
    name = result['galaxy']
    vf   = result['vflat_v21']
    rs   = result['rs_v21']
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = df['errV'].values
    vd   = df['Vdisk'].values
    vg   = df['Vgas'].values
    vb   = df['Vbul'].values

    r_plot = np.linspace(0.01, r.max() * 1.1, 300)
    vd_i = np.interp(r_plot, r, vd)
    vg_i = np.interp(r_plot, r, vg)
    vb_i = np.interp(r_plot, r, vb)

    vmod  = v_model(r_plot, vf, vd_i, vg_i, vb_i)
    x     = r_plot / rs
    Fx    = F_theory(x)
    v_dm  = np.sqrt(np.clip(vf**2 * Fx, 0, None))
    v_bar = np.sqrt(np.clip(vd_i**2 + vg_i**2 + vb_i**2, 0, None))

    ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=4, label='data')
    ax.plot(r_plot, vmod,  'r-',  lw=2, label='v2.1 total')
    ax.plot(r_plot, v_bar, 'b--', lw=1.2, label='baryon')
    ax.plot(r_plot, v_dm,  'g-.', lw=1.2, label='membrane')
    ax.axvline(rs, color='purple', ls=':', lw=0.8, label=f'rs={rs:.1f}')

    chi2str = f"{result['chi2dof_v21']:.2f}"
    ax.set_title(f"{name}  vf={vf:.0f}  rs={rs:.1f}  chi2={chi2str}", fontsize=9)
    ax.set_xlabel('r [kpc]', fontsize=8)
    ax.set_ylabel('v [km/s]', fontsize=8)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.2)

# --- メイン ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotmod-dir', default='Rotmod_LTG')
    args = parser.parse_args()

    data_dir = Path(args.rotmod_dir)
    if not data_dir.exists():
        alt = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
        if alt.exists():
            data_dir = alt
        else:
            print(f"ERROR: {data_dir} が見つかりません"); return

    targets = [
        'NGC3198',      # v2.0 基準
        'NGC2403',      # Sc 中型
        'NGC2841',      # Sb 高表面密度
        'NGC7331',      # Sb 高表面密度
        'NGC6503',      # Sc 中型
        'DDO154',       # Im 矮小（境界検証）
        'NGC1705',      # 矮小（ratio≈1）
        'NGC4214',      # 矮小（ratio≈1）
        'ESO444-G084',  # 矮小（ratio≈1）
    ]

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}

    records = []
    plot_data = []

    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found")
            continue

        cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']
        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)

        result = fit_galaxy(df, name)
        records.append(result)
        plot_data.append((df, result))

        daic_str = f"{result['dAIC']:.1f}" if not np.isnan(result.get('dAIC', np.nan)) else "N/A"
        print(f"  {name:15s} vflat={result['vflat_v21']:6.1f}  "
              f"rs={result['rs_v21']:5.2f}kpc  "
              f"chi2/dof(v21)={result['chi2dof_v21']:.3f}  "
              f"chi2/dof(tanh)={result['chi2dof_tanh']}  "
              f"dAIC={daic_str}")

    if not records:
        print("解析対象なし"); return

    # --- 9パネルプロット ---
    n = len(plot_data)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, (df, result) in enumerate(plot_data):
        plot_galaxy(df, result, axes[i])

    # 余ったパネルを非表示
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_fig = 'v21_multifit.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"\n図を {out_fig} に保存しました。")

    # --- サマリーテーブル ---
    df_res = pd.DataFrame(records)
    df_res.to_csv('v21_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n{'='*90}")
    print("v2.1 フィット結果サマリー")
    print(f"{'='*90}")
    print(df_res[['galaxy','n_pts','vflat_v21','rs_v21',
                  'chi2dof_v21','chi2dof_tanh','dAIC']].to_string(index=False))

    # 判定
    print(f"\n--- 判定 ---")
    for _, row in df_res.iterrows():
        name = row['galaxy']
        c21  = row['chi2dof_v21']
        daic = row['dAIC']
        vf   = row['vflat_v21']
        verdict = []
        if c21 < 2.0:
            verdict.append("chi2<2 OK")
        else:
            verdict.append(f"chi2={c21:.1f} NG")
        if not np.isnan(daic):
            if daic < 2:
                verdict.append("dAIC<2 OK (v2.1 competitive)")
            elif daic < 10:
                verdict.append(f"dAIC={daic:.0f} marginal")
            else:
                verdict.append(f"dAIC={daic:.0f} NG")
        boundary = "above" if vf >= 60 else "below"
        print(f"  {name:15s} [{boundary:5s} 60km/s]: {', '.join(verdict)}")

if __name__ == '__main__':
    main()
