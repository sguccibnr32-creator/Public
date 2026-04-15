# -*- coding: utf-8 -*-
"""
Step 5 v2.3: 弾性相 + 塑性相（対数成長）モデル
v² = v_bar² + v_e²·(1-exp(-r/rs)) + v_p²·ln(1+r/rs)
rs = v_e²/g_c

使用例:
  python -X utf8 step5_v23.py --rotmod-dir Rotmod_LTG
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
def v_model_v23(r_kpc, v_e, v_p, v_disk, v_gas, v_bul):
    """v2.3: 弾性相 + 塑性相（対数成長）"""
    rs = max((v_e * 1e3)**2 / G_c / kpc_to_m, 0.01)
    Fe = 1.0 - np.exp(-r_kpc / rs)
    Fp = np.log1p(r_kpc / rs)
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + v_e**2 * Fe + v_p**2 * Fp
    return np.sqrt(np.clip(v2, 0, None))

def v_model_tanh(r_kpc, vflat, rs, v_disk, v_gas, v_bul):
    x  = r_kpc / rs
    Fx = 0.5 * (1.0 + np.tanh(1.5 * (x - 1.0)))
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * Fx
    return np.sqrt(np.clip(v2, 0, None))

# ============================================================
# フィット
# ============================================================
def fit_v23(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, v_e, v_p):
        return v_model_v23(r, v_e, v_p, vd, vg, vb)

    best_chi2 = np.inf
    best_popt = None
    for ve0 in [20, 50, 80, 120, 180, 250]:
        for vp0 in [0, 10, 30, 60, 100]:
            try:
                popt, _ = curve_fit(
                    model, r, vobs,
                    p0=[ve0, vp0],
                    bounds=([10, 0], [300, 200]),
                    sigma=verr, maxfev=5000)
                vm = model(r, *popt)
                chi2 = np.sum(((vobs - vm) / verr)**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_popt = popt
            except Exception:
                continue

    if best_popt is None:
        return None

    v_e, v_p = best_popt
    rs = max((v_e * 1e3)**2 / G_c / kpc_to_m, 0.01)
    vflat_eff = np.sqrt(v_e**2 + v_p**2)
    dof = max(len(r) - 2, 1)

    return {
        'v_e': v_e, 'v_p': v_p,
        'vflat': vflat_eff,
        'plastic_ratio': v_p / vflat_eff if vflat_eff > 0 else 0,
        'rs': rs,
        'chi2dof': best_chi2 / dof,
        'chi2_total': best_chi2,
    }

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
        return popt[0], popt[1], chi2 / dof
    except Exception:
        return np.nan, np.nan, np.nan

# ============================================================
# プロット
# ============================================================
def plot_galaxy(df, name, result, vft, rst, ct, ax):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = df['errV'].values
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    rp  = np.linspace(0.01, r.max() * 1.1, 300)
    vdi = np.interp(rp, r, vd)
    vgi = np.interp(rp, r, vg)
    vbi = np.interp(rp, r, vb)

    v_e, v_p, rs = result['v_e'], result['v_p'], result['rs']

    vm   = v_model_v23(rp, v_e, v_p, vdi, vgi, vbi)
    vbar = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2, 0, None))
    Fe   = 1.0 - np.exp(-rp / rs)
    Fp   = np.log1p(rp / rs)
    v_el = np.sqrt(np.clip(v_e**2 * Fe, 0, None))
    v_pl = np.sqrt(np.clip(v_p**2 * Fp, 0, None))

    if not np.isnan(vft):
        x_t = rp / rst
        Ft  = 0.5 * (1.0 + np.tanh(1.5 * (x_t - 1.0)))
        vtanh = np.sqrt(np.clip(vdi**2 + vgi**2 + vbi**2 + vft**2 * Ft, 0, None))
    else:
        vtanh = None

    ax.errorbar(r, vobs, yerr=verr, fmt='k.', ms=4, label='data')
    ax.plot(rp, vm,    'r-',  lw=2.0, label=f'v2.3 chi2={result["chi2dof"]:.2f}')
    if vtanh is not None:
        ax.plot(rp, vtanh, 'm--', lw=1.5, alpha=0.7, label=f'tanh chi2={ct:.2f}')
    ax.plot(rp, vbar,  'b:',  lw=1.2, label='baryon')
    ax.plot(rp, v_el,  'g-.', lw=1.2, label=f'elastic ve={v_e:.0f}')
    ax.plot(rp, v_pl,  'c--', lw=1.2, label=f'plastic vp={v_p:.0f}')
    ax.axvline(rs, color='purple', ls=':', lw=0.8)

    pr = result['plastic_ratio']
    ax.set_title(f"{name}  ve={v_e:.0f} vp={v_p:.0f} pr={pr:.2f}  "
                 f"chi2={result['chi2dof']:.2f}", fontsize=9)
    ax.set_xlabel('r [kpc]', fontsize=8)
    ax.set_ylabel('v [km/s]', fontsize=8)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2)

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

    targets = ['NGC3198', 'NGC2403', 'NGC2841', 'NGC7331', 'NGC6503',
               'DDO154', 'NGC1705', 'NGC4214']

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}

    records = []
    plot_data = []

    print(f"{'galaxy':15s} {'v_e':>6} {'v_p':>6} {'vflat':>6} {'pr':>5} "
          f"{'rs':>6} {'chi2_23':>8} {'chi2_th':>8} {'dAIC':>7}")
    print("-" * 80)

    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found"); continue

        cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']
        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)

        result = fit_v23(df)
        if result is None:
            print(f"  {name}: fit failed"); continue

        vft, rst, ct = fit_tanh(df)

        n = len(df)
        if not np.isnan(ct):
            daic = result['chi2_total'] - ct * max(n-2, 1)  # same k=2
        else:
            daic = np.nan

        pr = result['plastic_ratio']
        ct_str = f"{ct:8.3f}" if not np.isnan(ct) else "     N/A"
        da_str = f"{daic:7.1f}" if not np.isnan(daic) else "    N/A"
        print(f"{name:15s} {result['v_e']:6.1f} {result['v_p']:6.1f} "
              f"{result['vflat']:6.1f} {pr:5.2f} {result['rs']:6.1f} "
              f"{result['chi2dof']:8.3f} {ct_str} {da_str}")

        records.append({
            'galaxy':         name,
            'n_pts':          n,
            'v_e':            round(result['v_e'], 1),
            'v_p':            round(result['v_p'], 1),
            'vflat':          round(result['vflat'], 1),
            'plastic_ratio':  round(pr, 3),
            'rs':             round(result['rs'], 2),
            'chi2dof_v23':    round(result['chi2dof'], 3),
            'vf_tanh':        round(vft, 1) if not np.isnan(vft) else np.nan,
            'rs_tanh':        round(rst, 2) if not np.isnan(rst) else np.nan,
            'chi2dof_tanh':   round(ct, 3) if not np.isnan(ct) else np.nan,
            'dAIC':           round(daic, 1) if not np.isnan(daic) else np.nan,
        })
        plot_data.append((df, name, result, vft, rst, ct))

    if not records:
        return

    df_res = pd.DataFrame(records)
    df_res.to_csv('v23_results.csv', index=False, encoding='utf-8-sig')

    # プロット
    n = len(plot_data)
    ncols = 3
    nrows = max((n + ncols - 1) // ncols, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (df, name, result, vft, rst, ct) in enumerate(plot_data):
        plot_galaxy(df, name, result, vft, rst, ct, axes[i])
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('v23_multifit.png', dpi=150, bbox_inches='tight')

    # サマリー
    print(f"\n{'='*95}")
    print("v2.3 サマリー")
    print(f"{'='*95}")
    print(df_res[['galaxy','v_e','v_p','vflat','plastic_ratio',
                  'rs','chi2dof_v23','chi2dof_tanh','dAIC']].to_string(index=False))

    # 判定
    print(f"\n--- 判定 ---")
    for _, row in df_res.iterrows():
        name = row['galaxy']
        c23  = row['chi2dof_v23']
        daic = row['dAIC']
        pr   = row['plastic_ratio']
        vf   = row['vflat']

        parts = []
        if c23 < 2:    parts.append(f"chi2={c23:.2f} OK")
        elif c23 < 5:  parts.append(f"chi2={c23:.1f} marginal")
        else:          parts.append(f"chi2={c23:.0f} NG")

        if not np.isnan(daic):
            if abs(daic) < 2:   parts.append("~tanh")
            elif daic < 0:      parts.append(f"dAIC={daic:.0f} v23 WINS")
            else:               parts.append(f"dAIC={daic:.0f}")

        if pr > 0.01:  parts.append(f"pr={pr:.2f} PLASTIC")
        else:          parts.append(f"pr={pr:.2f}")
        print(f"  {name:15s}: {', '.join(parts)}")

    # plastic_ratio サマリー
    print(f"\n--- plastic_ratio ---")
    pr_all = df_res['plastic_ratio']
    print(f"中央値: {pr_all.median():.3f}")
    print(f"範囲:   {pr_all.min():.3f} ~ {pr_all.max():.3f}")
    plastic_galaxies = df_res[df_res['plastic_ratio'] > 0.01]
    if len(plastic_galaxies) > 0:
        print(f"塑性相が有意な銀河: {plastic_galaxies['galaxy'].tolist()}")

    print(f"\n図を v23_multifit.png に保存しました。")

if __name__ == '__main__':
    main()
