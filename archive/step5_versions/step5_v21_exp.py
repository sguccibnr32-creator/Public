# -*- coding: utf-8 -*-
"""
Step 5 v2: 修正 v2.1 モデル（F_exp 指数飽和型）での回転曲線フィット
F(r) = 1 - exp(-r/r_s)
1パラメータ(vflat) vs 2パラメータ(vflat,rs) vs tanh(2param)

使用例:
  python -X utf8 step5_v21_exp.py --rotmod-dir Rotmod_LTG
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
G_c      = 1.2e-10    # m/s²
kpc_to_m = 3.086e19

# --- モデル関数 ---

def F_exp(r_kpc, rs_kpc):
    """指数飽和型移行関数: F(r) = 1 - exp(-r/r_s)"""
    return 1.0 - np.exp(-r_kpc / rs_kpc)

def rs_predict(vflat_kms):
    """r_s = v_flat² / g_c [kpc]"""
    v_ms = vflat_kms * 1e3
    return v_ms**2 / G_c / kpc_to_m

def v_model_v21(r_kpc, vflat_kms, v_disk, v_gas, v_bul, rs_kpc=None):
    if rs_kpc is None:
        rs_kpc = rs_predict(vflat_kms)
    Fx = F_exp(r_kpc, rs_kpc)
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat_kms**2 * Fx
    return np.sqrt(np.clip(v2, 0, None))

# --- フィット ---

def fit_1param(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def chi2(vflat):
        vm = v_model_v21(r, vflat, vd, vg, vb)
        return np.sum(((vobs - vm) / verr)**2)

    res = minimize_scalar(chi2, bounds=(10, 400), method='bounded',
                          options={'xatol': 0.1})
    vf = res.x
    rs = rs_predict(vf)
    c2 = res.fun / max(len(r) - 1, 1)
    return vf, rs, c2

def fit_2param(df, vf0=None, rs0=None):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, vflat, rs):
        return v_model_v21(r, vflat, vd, vg, vb, rs_kpc=rs)

    if vf0 is None:
        vf0, rs0, _ = fit_1param(df)
    try:
        popt, _ = curve_fit(model, r, vobs, p0=[vf0, rs0],
                           bounds=([10, 0.1], [400, 200]),
                           sigma=verr, maxfev=5000)
        vm = model(r, *popt)
        c2 = np.sum(((vobs - vm) / verr)**2) / max(len(r) - 2, 1)
        return popt[0], popt[1], c2
    except Exception:
        return np.nan, np.nan, np.nan

def fit_tanh(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, vflat, rs):
        x  = r / rs
        Fx = 0.5 * (1.0 + np.tanh(1.5 * (x - 1.0)))
        return np.sqrt(np.clip(vd**2 + vg**2 + vb**2 + vflat**2 * Fx, 0, None))

    try:
        popt, _ = curve_fit(model, r, vobs, p0=[150, 7],
                           bounds=([10, 0.1], [400, 200]),
                           sigma=verr, maxfev=5000)
        vm = model(r, *popt)
        c2 = np.sum(((vobs - vm) / verr)**2) / max(len(r) - 2, 1)
        return popt[0], popt[1], c2
    except Exception:
        return np.nan, np.nan, np.nan

# --- プロット ---

def plot_comparison(df, name, vf1, rs1, c1, vf2, rs2, c2, vft, rst, ct, out_dir='.'):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = df['errV'].values
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    rp  = np.linspace(0.01, r.max() * 1.1, 300)
    vdi = np.interp(rp, r, vd)
    vgi = np.interp(rp, r, vg)
    vbi = np.interp(rp, r, vb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(name, fontsize=13)

    configs = [
        (vf1, rs1, f'v2.1 1p: vf={vf1:.1f} rs={rs1:.1f} chi2={c1:.2f}', c1),
        (vf2, rs2, f'v2.1 2p: vf={vf2:.1f} rs={rs2:.1f} chi2={c2:.2f}', c2),
    ]

    for ax, (vf, rs, label, c2_val) in zip(axes, configs):
        vm   = v_model_v21(rp, vf, vdi, vgi, vbi, rs_kpc=rs)
        vbar = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2, 0, None))
        vdm  = np.sqrt(np.clip(vf**2 * F_exp(rp, rs), 0, None))

        # tanh comparison
        if not np.isnan(vft):
            x_t  = rp / rst
            Ft   = 0.5 * (1.0 + np.tanh(1.5 * (x_t - 1.0)))
            vtanh = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2 + vft**2 * Ft, 0, None))
        else:
            vtanh = None

        ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=5, label='data')
        ax.plot(rp, vm,   'r-',  lw=2, label=label)
        if vtanh is not None:
            ax.plot(rp, vtanh, 'm--', lw=1.5, label=f'tanh: chi2={ct:.2f}', alpha=0.7)
        ax.plot(rp, vbar, 'b:',  lw=1.2, label='baryon')
        ax.plot(rp, vdm,  'g-.', lw=1.2, label='membrane')
        ax.axvline(rs, color='purple', ls=':', lw=0.8, label=f'rs={rs:.1f}')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('v [km/s]')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/{name}_v21_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

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

    targets = ['NGC3198', 'NGC2403', 'NGC2841', 'DDO154', 'NGC4214']

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}

    print(f"{'galaxy':15s} {'vf_1p':>7} {'rs_1p':>7} {'chi2_1p':>8} "
          f"{'vf_2p':>7} {'rs_2p':>7} {'chi2_2p':>8} "
          f"{'chi2_th':>8} {'dAIC':>7}")
    print("-" * 85)

    records = []
    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found"); continue

        cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']
        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)

        vf1, rs1, c1 = fit_1param(df)
        vf2, rs2, c2 = fit_2param(df, vf1, rs1)
        vft, rst, ct = fit_tanh(df)

        # ΔAIC: v2.1(1p) vs tanh(2p)
        n = len(df)
        if not np.isnan(ct):
            aic_v21  = c1 * max(n-1, 1) + 2 * 1
            aic_tanh = ct * max(n-2, 1) + 2 * 2
            daic = aic_v21 - aic_tanh
        else:
            daic = np.nan

        print(f"{name:15s} {vf1:7.1f} {rs1:7.2f} {c1:8.3f} "
              f"{vf2:7.1f} {rs2:7.2f} {c2:8.3f} "
              f"{ct:8.3f} {daic:7.1f}"
              if not np.isnan(ct) else
              f"{name:15s} {vf1:7.1f} {rs1:7.2f} {c1:8.3f} "
              f"{vf2:7.1f} {rs2:7.2f} {c2:8.3f} "
              f"{'N/A':>8} {'N/A':>7}")

        records.append({
            'galaxy':     name,
            'n_pts':      n,
            'vf_1p':      round(vf1, 1),
            'rs_1p':      round(rs1, 2),
            'rs_pred':    round(rs_predict(vf1), 2),
            'chi2_1p':    round(c1, 3),
            'vf_2p':      round(vf2, 1) if not np.isnan(vf2) else np.nan,
            'rs_2p':      round(rs2, 2) if not np.isnan(rs2) else np.nan,
            'chi2_2p':    round(c2, 3) if not np.isnan(c2) else np.nan,
            'vf_tanh':    round(vft, 1) if not np.isnan(vft) else np.nan,
            'rs_tanh':    round(rst, 2) if not np.isnan(rst) else np.nan,
            'chi2_tanh':  round(ct, 3) if not np.isnan(ct) else np.nan,
            'dAIC':       round(daic, 1) if not np.isnan(daic) else np.nan,
        })

        plot_comparison(df, name, vf1, rs1, c1, vf2, rs2, c2, vft, rst, ct)

    if not records:
        print("解析対象なし"); return

    df_res = pd.DataFrame(records)
    df_res.to_csv('v21_exp_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n{'='*85}")
    print("サマリー")
    print(f"{'='*85}")
    print(df_res[['galaxy','n_pts','vf_1p','rs_1p','rs_pred',
                  'chi2_1p','vf_2p','rs_2p','chi2_2p',
                  'chi2_tanh','dAIC']].to_string(index=False))

    # 判定
    print(f"\n--- 判定 ---")
    for _, row in df_res.iterrows():
        name = row['galaxy']
        c1 = row['chi2_1p']
        c2 = row['chi2_2p']
        ct = row['chi2_tanh']
        daic = row['dAIC']
        rs_ratio = row['rs_1p'] / row['rs_pred'] if row['rs_pred'] > 0 else np.nan

        parts = []
        if c1 < 3:
            parts.append(f"1p chi2={c1:.2f} OK")
        else:
            parts.append(f"1p chi2={c1:.1f} NG")
        if not np.isnan(c2):
            if c2 < 2:
                parts.append(f"2p chi2={c2:.2f} OK")
            else:
                parts.append(f"2p chi2={c2:.1f}")
        if not np.isnan(daic):
            if daic < 2:
                parts.append(f"dAIC={daic:.0f} competitive")
            else:
                parts.append(f"dAIC={daic:.0f}")
        parts.append(f"rs_1p/pred={rs_ratio:.2f}")
        print(f"  {name:15s}: {', '.join(parts)}")

    # NGC3198 の個別プロット確認
    print(f"\n個別プロットを保存: NGC3198_v21_comparison.png 等")

if __name__ == '__main__':
    main()
