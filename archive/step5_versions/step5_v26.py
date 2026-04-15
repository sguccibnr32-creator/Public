# -*- coding: utf-8 -*-
"""
Step 5 v2.6: 自己無撞着 r_s + tanh 形状
rs = [v_disk²(rs) + 0.5×v_flat²] / g_c を数値的に解く
F(x) = 0.5*(1+tanh(w*(x-1)))  x = r/rs

使用例:
  python -X utf8 step5_v26.py --rotmod-dir Rotmod_LTG
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, curve_fit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

G_c      = 1.2e-10
kpc_to_m = 3.086e19

# ============================================================
# 自己無撞着 r_s
# ============================================================
def rs_selfconsistent(vflat_kms, r_kpc, v_disk):
    """rs = [v_disk²(rs) + 0.5×v_flat²] / g_c を brentq で解く"""
    vf2 = (vflat_kms * 1e3)**2

    def equation(rs):
        vd_rs = np.interp(rs, r_kpc, v_disk)
        rhs = ((vd_rs * 1e3)**2 + 0.5 * vf2) / G_c / kpc_to_m
        return rs - rhs

    r_min, r_max = 0.01, 200.0
    try:
        f_min = equation(r_min)
        f_max = equation(r_max)
        if f_min * f_max > 0:
            return (0.5 * vf2) / G_c / kpc_to_m
        return brentq(equation, r_min, r_max, xtol=0.01)
    except Exception:
        return (0.5 * vf2) / G_c / kpc_to_m

# ============================================================
# モデル
# ============================================================
def v_model_v26(r_kpc, vflat, w, v_disk, v_gas, v_bul, rs_fixed=None):
    if rs_fixed is None:
        rs = rs_selfconsistent(vflat, r_kpc, v_disk)
    else:
        rs = rs_fixed
    x  = r_kpc / rs
    Fx = 0.5 * (1.0 + np.tanh(w * (x - 1.0)))
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * Fx
    return np.sqrt(np.clip(v2, 0, None)), rs

def v_model_tanh(r_kpc, vflat, rs, v_disk, v_gas, v_bul):
    x  = r_kpc / rs
    Fx = 0.5 * (1.0 + np.tanh(1.5 * (x - 1.0)))
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * Fx
    return np.sqrt(np.clip(v2, 0, None))

# ============================================================
# フィット
# ============================================================
def fit_v26_1param(df, w=1.5):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def chi2(vflat):
        vm, _ = v_model_v26(r, vflat, w, vd, vg, vb)
        return np.sum(((vobs - vm) / verr)**2)

    res = minimize_scalar(chi2, bounds=(10, 400), method='bounded',
                          options={'xatol': 0.1})
    vf = res.x
    vm, rs = v_model_v26(r, vf, w, vd, vg, vb)
    dof = max(len(r) - 1, 1)
    return vf, rs, res.fun / dof, res.fun

def fit_v26_2param(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, vflat, w):
        vm, _ = v_model_v26(r, vflat, w, vd, vg, vb)
        return vm

    best_chi2 = np.inf
    best_popt = None
    best_rs = None
    for vf0 in [50, 100, 150, 200]:
        for w0 in [0.5, 1.0, 1.5, 2.5]:
            try:
                popt, _ = curve_fit(model, r, vobs, p0=[vf0, w0],
                                   bounds=([10, 0.1], [400, 5.0]),
                                   sigma=verr, maxfev=5000)
                vm = model(r, *popt)
                chi2 = np.sum(((vobs - vm) / verr)**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_popt = popt
                    _, rs = v_model_v26(r, popt[0], popt[1], vd, vg, vb)
                    best_rs = rs
            except Exception:
                continue

    if best_popt is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    vf2, w2 = best_popt
    dof = max(len(r) - 2, 1)
    return vf2, w2, best_rs, best_chi2 / dof, best_chi2

def fit_tanh_free(df):
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
# プロット
# ============================================================
def plot_v26(df, name, vf1, rs1, c1, vf2, w2, rs2, c2, vft, rst, ct):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = df['errV'].values
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    rp  = np.linspace(0.01, r.max() * 1.2, 500)
    vdi = np.interp(rp, r, vd)
    vgi = np.interp(rp, r, vg)
    vbi = np.interp(rp, r, vb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name}  v2.6 (self-consistent rs)", fontsize=13)

    # Panel 1: 1-param
    ax = axes[0]
    vm1 = v_model_v26(rp, vf1, 1.5, vdi, vgi, vbi, rs_fixed=rs1)[0]
    vbar = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2, 0, None))
    x1 = rp / rs1
    Fx1 = 0.5 * (1.0 + np.tanh(1.5 * (x1 - 1.0)))
    vdm1 = np.sqrt(np.clip(vf1**2 * Fx1, 0, None))

    if not np.isnan(vft):
        xt = rp / rst
        Ft = 0.5 * (1.0 + np.tanh(1.5 * (xt - 1.0)))
        vtanh = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2 + vft**2 * Ft, 0, None))
        ax.plot(rp, vtanh, 'm--', lw=1.5, alpha=0.7, label=f'tanh chi2={ct:.2f}')

    ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=5, label='data')
    ax.plot(rp, vm1,  'r-',  lw=2.5, label=f'v2.6 1p chi2={c1:.2f}')
    ax.plot(rp, vbar, 'b--', lw=1.2, alpha=0.7, label='baryon')
    ax.plot(rp, vdm1, 'g-.', lw=1.2, alpha=0.7, label='membrane')
    ax.axvline(rs1, color='purple', ls=':', lw=1, label=f'rs_sc={rs1:.1f}')
    if not np.isnan(rst):
        ax.axvline(rst, color='magenta', ls=':', lw=0.6, alpha=0.5, label=f'rs_tanh={rst:.1f}')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.set_title(f'1-param: vf={vf1:.0f} rs={rs1:.1f}', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2)

    # Panel 2: 2-param
    ax = axes[1]
    if not np.isnan(vf2):
        vm2 = v_model_v26(rp, vf2, w2, vdi, vgi, vbi, rs_fixed=rs2)[0]
        x2 = rp / rs2
        Fx2 = 0.5 * (1.0 + np.tanh(w2 * (x2 - 1.0)))
        vdm2 = np.sqrt(np.clip(vf2**2 * Fx2, 0, None))

        ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=5, label='data')
        ax.plot(rp, vm2,  'r-',  lw=2.5, label=f'v2.6 2p chi2={c2:.2f}')
        if not np.isnan(vft):
            ax.plot(rp, vtanh, 'm--', lw=1.5, alpha=0.7, label=f'tanh chi2={ct:.2f}')
        ax.plot(rp, vbar, 'b--', lw=1.2, alpha=0.7, label='baryon')
        ax.plot(rp, vdm2, 'g-.', lw=1.2, alpha=0.7, label='membrane')
        ax.axvline(rs2, color='purple', ls=':', lw=1, label=f'rs_sc={rs2:.1f}')
        ax.set_title(f'2-param: vf={vf2:.0f} w={w2:.2f} rs={rs2:.1f}', fontsize=9)
    else:
        ax.set_title('2-param: failed')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2)

    # Panel 3: residuals (1-param)
    ax = axes[2]
    vm1_data = v_model_v26(r, vf1, 1.5, vd, vg, vb, rs_fixed=rs1)[0]
    res1 = (vobs - vm1_data) / verr
    ax.plot(r, res1, 'ro-', ms=4, label='v2.6 1p')
    if not np.isnan(vft):
        vmt_data = v_model_tanh(r, vft, rst, vd, vg, vb)
        rest = (vobs - vmt_data) / verr
        ax.plot(r, rest, 'm--o', ms=4, alpha=0.7, label='tanh')
    ax.axhline(0,  color='k', ls='-', lw=0.8)
    ax.axhline(+2, color='gray', ls=':', lw=0.8)
    ax.axhline(-2, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('residual / sigma')
    ax.set_title('Residuals (1-param)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{name}_v26.png', dpi=150, bbox_inches='tight')
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
               'DDO154', 'NGC1705', 'NGC6503']

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}
    cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

    print(f"{'galaxy':12s} {'vf_1p':>7} {'rs_sc':>7} {'chi2_1p':>8} "
          f"{'vf_2p':>7} {'w_2p':>5} {'rs_2p':>7} {'chi2_2p':>8} "
          f"{'chi2_th':>8} {'rs_th':>7} {'dAIC':>7}")
    print("-" * 100)

    records = []
    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found"); continue

        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)

        vf1, rs1, c1, chi2_1 = fit_v26_1param(df, w=1.5)
        vf2, w2, rs2, c2, chi2_2 = fit_v26_2param(df)
        vft, rst, ct, chi2_t = fit_tanh_free(df)

        n = len(df)
        if not np.isnan(chi2_t):
            daic = (chi2_1 + 2*1) - (chi2_t + 2*2)
        else:
            daic = np.nan

        w2s = f"{w2:5.2f}" if not np.isnan(w2) else "  N/A"
        rs2s = f"{rs2:7.2f}" if not np.isnan(rs2) else "    N/A"
        c2s = f"{c2:8.3f}" if not np.isnan(c2) else "     N/A"
        cts = f"{ct:8.3f}" if not np.isnan(ct) else "     N/A"
        rsts = f"{rst:7.2f}" if not np.isnan(rst) else "    N/A"
        das = f"{daic:7.1f}" if not np.isnan(daic) else "    N/A"

        print(f"{name:12s} {vf1:7.1f} {rs1:7.2f} {c1:8.3f} "
              f"{vf2:7.1f} {w2s} {rs2s} {c2s} {cts} {rsts} {das}")

        records.append({
            'galaxy':     name,
            'n_pts':      n,
            'vf_1p':      round(vf1, 1),
            'rs_sc':      round(rs1, 2),
            'chi2_1p':    round(c1, 3),
            'vf_2p':      round(vf2, 1) if not np.isnan(vf2) else np.nan,
            'w_2p':       round(w2, 3) if not np.isnan(w2) else np.nan,
            'rs_2p':      round(rs2, 2) if not np.isnan(rs2) else np.nan,
            'chi2_2p':    round(c2, 3) if not np.isnan(c2) else np.nan,
            'vf_tanh':    round(vft, 1) if not np.isnan(vft) else np.nan,
            'rs_tanh':    round(rst, 2) if not np.isnan(rst) else np.nan,
            'chi2_tanh':  round(ct, 3) if not np.isnan(ct) else np.nan,
            'dAIC_1p':    round(daic, 1) if not np.isnan(daic) else np.nan,
        })

        plot_v26(df, name, vf1, rs1, c1,
                 vf2, w2, rs2, c2 if not np.isnan(c2) else 0,
                 vft, rst, ct if not np.isnan(ct) else 0)

    if not records:
        return

    df_res = pd.DataFrame(records)
    df_res.to_csv('v26_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n{'='*100}")
    print("v2.6 サマリー")
    print(f"{'='*100}")
    print(df_res[['galaxy','vf_1p','rs_sc','chi2_1p',
                  'vf_2p','w_2p','rs_2p','chi2_2p',
                  'rs_tanh','chi2_tanh','dAIC_1p']].to_string(index=False))

    # rs 比較
    print(f"\n--- rs_sc vs rs_tanh ---")
    for _, row in df_res.iterrows():
        rs_ratio = row['rs_sc'] / row['rs_tanh'] if (not np.isnan(row.get('rs_tanh', np.nan)) and row['rs_tanh'] > 0) else np.nan
        match = "MATCH" if (not np.isnan(rs_ratio) and 0.5 < rs_ratio < 2.0) else ""
        print(f"  {row['galaxy']:12s}: rs_sc={row['rs_sc']:6.2f}, "
              f"rs_tanh={row['rs_tanh'] if not np.isnan(row.get('rs_tanh', np.nan)) else 'N/A':>6}, "
              f"ratio={rs_ratio:.2f}  {match}"
              if not np.isnan(rs_ratio) else
              f"  {row['galaxy']:12s}: rs_sc={row['rs_sc']:6.2f}, rs_tanh=N/A")

    # w の普遍性
    valid_w = df_res.dropna(subset=['w_2p'])
    if len(valid_w) >= 3:
        print(f"\n--- w の普遍性 ---")
        print(f"w 中央値: {valid_w['w_2p'].median():.3f}")
        print(f"w 平均:   {valid_w['w_2p'].mean():.3f}")
        print(f"w std:    {valid_w['w_2p'].std():.3f}")
        print(f"w 範囲:   {valid_w['w_2p'].min():.3f} ~ {valid_w['w_2p'].max():.3f}")

    # 判定
    print(f"\n--- 判定 ---")
    for _, row in df_res.iterrows():
        name = row['galaxy']
        c1 = row['chi2_1p']
        daic = row['dAIC_1p']
        parts = []
        if c1 < 2:     parts.append(f"chi2_1p={c1:.2f} OK")
        elif c1 < 5:   parts.append(f"chi2_1p={c1:.1f} marginal")
        elif c1 < 20:  parts.append(f"chi2_1p={c1:.0f} improved")
        else:           parts.append(f"chi2_1p={c1:.0f} NG")

        if not np.isnan(daic):
            if abs(daic) < 2:   parts.append("~tanh")
            elif daic < 0:      parts.append(f"dAIC={daic:.0f} v26 WINS")
            else:               parts.append(f"dAIC={daic:.0f}")

        print(f"  {name:12s}: {', '.join(parts)}")

    print(f"\n個別プロット保存完了")

if __name__ == '__main__':
    main()
