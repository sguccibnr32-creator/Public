# -*- coding: utf-8 -*-
"""
Step 5 v2.5: 次元切り替え移行関数 H(x)
H(x) = (1-exp(-x)) + max(0, ln(x))
1パラメータ (vflat) で rs = vflat²/g_c を自動計算

使用例:
  python -X utf8 step5_v25.py --rotmod-dir Rotmod_LTG
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, curve_fit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

G_c      = 1.2e-10
kpc_to_m = 3.086e19

# ============================================================
# モデル
# ============================================================
def H_v25(x):
    x = np.atleast_1d(np.float64(x))
    Fe = 1.0 - np.exp(-x)
    Fl = np.where(x > 1, np.log(x), 0.0)
    return Fe + Fl

def rs_v25(vflat_kms):
    return (vflat_kms * 1e3)**2 / G_c / kpc_to_m

def v_model_v25(r_kpc, vflat, v_disk, v_gas, v_bul):
    rs = rs_v25(vflat)
    x  = r_kpc / rs
    H  = H_v25(x)
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * H
    return np.sqrt(np.clip(v2, 0, None))

def v_model_tanh(r_kpc, vflat, rs, v_disk, v_gas, v_bul):
    x  = r_kpc / rs
    Fx = 0.5 * (1.0 + np.tanh(1.5 * (x - 1.0)))
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * Fx
    return np.sqrt(np.clip(v2, 0, None))

# ============================================================
# フィット
# ============================================================
def fit_v25(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def chi2(vflat):
        vm = v_model_v25(r, vflat, vd, vg, vb)
        return np.sum(((vobs - vm) / verr)**2)

    res = minimize_scalar(chi2, bounds=(10, 400), method='bounded',
                          options={'xatol': 0.1})
    vf  = res.x
    rs  = rs_v25(vf)
    dof = max(len(r) - 1, 1)
    return vf, rs, res.fun / dof, res.fun

def fit_tanh(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, vflat, rs):
        return v_model_tanh(r, vflat, rs, vd, vg, vb)

    try:
        popt, _ = curve_fit(model, r, vobs, p0=[150, 7],
                           bounds=([10, 0.1], [400, 200]),
                           sigma=verr, maxfev=5000)
        vm = model(r, *popt)
        dof = max(len(r) - 2, 1)
        chi2 = np.sum(((vobs - vm) / verr)**2)
        return popt[0], popt[1], chi2 / dof, chi2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# ============================================================
# プロット（3パネル）
# ============================================================
def plot_v25(df, name, vf, rs, c2, vft, rst, c2t):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = df['errV'].values
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    rp  = np.linspace(0.01, r.max() * 1.2, 500)
    vdi = np.interp(rp, r, vd)
    vgi = np.interp(rp, r, vg)
    vbi = np.interp(rp, r, vb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name}  v2.5 vs tanh", fontsize=13)

    # Panel 1: rotation curves
    ax = axes[0]
    vm25  = v_model_v25(rp, vf, vdi, vgi, vbi)
    vbar  = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2, 0, None))
    xp    = rp / rs
    vdm25 = np.sqrt(np.clip(vf**2 * H_v25(xp), 0, None))

    if not np.isnan(vft):
        xpt   = rp / rst
        Ft    = 0.5 * (1.0 + np.tanh(1.5 * (xpt - 1.0)))
        vtanh = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2 + vft**2 * Ft, 0, None))
        ax.plot(rp, vtanh, 'm--', lw=1.5, label=f'tanh chi2={c2t:.2f}', alpha=0.7)

    ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=5, label='data')
    ax.plot(rp, vm25,  'r-',  lw=2.5, label=f'v2.5 chi2={c2:.2f}')
    ax.plot(rp, vbar,  'b--', lw=1.2, label='baryon', alpha=0.7)
    ax.plot(rp, vdm25, 'g-.', lw=1.2, label='membrane', alpha=0.7)
    ax.axvline(rs, color='purple', ls=':', lw=1, label=f'rs={rs:.1f}')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.set_title('Rotation Curve')
    ax.legend(fontsize=7)
    ax.set_ylim(0, None)

    # Panel 2: H(x) shape
    ax = axes[1]
    xarr = np.linspace(0.01, 10, 500)
    Harr = H_v25(xarr)
    Tarr = 0.5 * (1.0 + np.tanh(1.5 * (xarr - 1.0)))
    ax.plot(xarr, Harr, 'r-',  lw=2, label='H(x) v2.5')
    ax.plot(xarr, Tarr, 'm--', lw=1.5, label='tanh', alpha=0.7)
    ax.axvline(1, color='gray', ls=':', lw=0.8)
    ax.axhline(1, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('x = r/r_s')
    ax.set_ylabel('H(x) or F(x)')
    ax.set_title('Transfer function')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # Panel 3: residuals
    ax = axes[2]
    vm25_data = v_model_v25(r, vf, vd, vg, vb)
    res25 = (vobs - vm25_data) / verr
    if not np.isnan(vft):
        vmt_data = v_model_tanh(r, vft, rst, vd, vg, vb)
        rest = (vobs - vmt_data) / verr
        ax.plot(r, rest, 'm--o', ms=4, label='tanh residual', alpha=0.7)
    ax.plot(r, res25, 'ro-', ms=4, label='v2.5 residual')
    ax.axhline(0,  color='k', ls='-', lw=0.8)
    ax.axhline(+2, color='gray', ls=':', lw=0.8)
    ax.axhline(-2, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('residual / sigma')
    ax.set_title('Residuals')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = f'{name}_v25.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# メイン
# ============================================================
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

    targets = ['NGC3198', 'NGC2841', 'NGC2403', 'NGC4214',
               'DDO154', 'NGC1705', 'NGC6503', 'NGC7331']

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}

    cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

    print(f"{'galaxy':15s} {'vf_v25':>8} {'rs_v25':>7} "
          f"{'chi2_v25':>9} {'chi2_tanh':>10} {'dAIC':>7}")
    print("-" * 65)

    records = []
    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found"); continue

        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)

        vf, rs, c2, chi2_raw = fit_v25(df)
        vft, rst, c2t, chi2t_raw = fit_tanh(df)

        n = len(df)
        if not np.isnan(chi2t_raw):
            daic = (chi2_raw + 2*1) - (chi2t_raw + 2*2)
        else:
            daic = np.nan

        da_str = f"{daic:7.1f}" if not np.isnan(daic) else "    N/A"
        c2t_str = f"{c2t:10.3f}" if not np.isnan(c2t) else "       N/A"
        print(f"{name:15s} {vf:8.1f} {rs:7.2f} {c2:9.3f} {c2t_str} {da_str}")

        records.append({
            'galaxy':     name,
            'n_pts':      n,
            'vf_v25':     round(vf, 1),
            'rs_v25':     round(rs, 2),
            'chi2_v25':   round(c2, 3),
            'vf_tanh':    round(vft, 1) if not np.isnan(vft) else np.nan,
            'rs_tanh':    round(rst, 2) if not np.isnan(rst) else np.nan,
            'chi2_tanh':  round(c2t, 3) if not np.isnan(c2t) else np.nan,
            'dAIC':       round(daic, 1) if not np.isnan(daic) else np.nan,
        })

        plot_v25(df, name, vf, rs, c2, vft, rst, c2t)

    if not records:
        return

    df_res = pd.DataFrame(records)
    df_res.to_csv('v25_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n{'='*75}")
    print("v2.5 サマリー")
    print(f"{'='*75}")
    print(df_res[['galaxy','vf_v25','rs_v25','chi2_v25',
                  'vf_tanh','rs_tanh','chi2_tanh','dAIC']].to_string(index=False))

    # 判定
    print(f"\n--- 判定 ---")
    for _, row in df_res.iterrows():
        name = row['galaxy']
        c25  = row['chi2_v25']
        daic = row['dAIC']
        parts = []
        if c25 < 2:    parts.append(f"chi2={c25:.2f} OK")
        elif c25 < 5:  parts.append(f"chi2={c25:.1f} marginal")
        elif c25 < 20: parts.append(f"chi2={c25:.0f} improved?")
        else:          parts.append(f"chi2={c25:.0f} NG")

        if not np.isnan(daic):
            if abs(daic) < 2:   parts.append("~tanh")
            elif daic < 0:      parts.append(f"dAIC={daic:.0f} v25 WINS")
            else:               parts.append(f"dAIC={daic:.0f}")

        # rs comparison
        rs_ratio = row['rs_v25'] / row['rs_tanh'] if (not np.isnan(row['rs_tanh']) and row['rs_tanh'] > 0) else np.nan
        if not np.isnan(rs_ratio):
            parts.append(f"rs_ratio={rs_ratio:.2f}")

        print(f"  {name:15s}: {', '.join(parts)}")

    # H(x) の値を表示
    print(f"\n--- H(x) reference values ---")
    for x_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        print(f"  x={x_val:4.1f}: H={H_v25(x_val)[0]:.3f}")

    print(f"\n個別プロットを保存: NGC3198_v25.png 等")

if __name__ == '__main__':
    main()
