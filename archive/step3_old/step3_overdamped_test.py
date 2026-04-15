# -*- coding: utf-8 -*-
"""
Step 3: 過減衰仮説検定
Sa/S0 (gc_a0 > 1) vs Im/IB (gc_a0 <= 0.3) で
Model A（膜式）vs Model B（g_obs = g_N）を AIC 比較

使用例:
  python step3_overdamped_test.py
  python step3_overdamped_test.py --rotmod-dir Rotmod_LTG
  python step3_overdamped_test.py --rotmod-dir Rotmod_LTG --csv sparc167_galaxy_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import argparse

# ============================================================
# 定数
# ============================================================
a0       = 1.2e-10       # m/s²
kpc_m    = 3.0857e19     # 1 kpc → m
kms_m    = 1e3           # 1 km/s → m/s
UPSILON      = 0.5       # ディスク恒星質量光度比
UPSILON_BUL  = 0.7       # バルジ

# ============================================================
# SPARC .dat 読み込み
# ============================================================
def read_sparc_dat(filepath):
    """
    列：Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    """
    rows = []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append([float(p) for p in parts[:8]])
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.array(rows)
    ncol = arr.shape[1]
    df = pd.DataFrame()
    df['R']     = arr[:, 0]
    df['Vobs']  = arr[:, 1]
    df['errV']  = arr[:, 2]
    df['Vgas']  = arr[:, 3]
    df['Vdisk'] = arr[:, 4]
    df['Vbul']  = arr[:, 5] if ncol > 5 else 0.0
    return df

def compute_g(df):
    R_m      = df['R'].values    * kpc_m
    Vobs_ms  = df['Vobs'].values * kms_m
    errV_ms  = df['errV'].values * kms_m
    Vgas_ms  = df['Vgas'].values * kms_m
    Vdisk_ms = df['Vdisk'].values * kms_m
    Vbul_ms  = df['Vbul'].values * kms_m

    def signed_sq(V):
        return np.sign(V) * V**2

    g_bar = (signed_sq(Vgas_ms)
             + UPSILON     * signed_sq(Vdisk_ms)
             + UPSILON_BUL * signed_sq(Vbul_ms)) / R_m
    g_obs  = signed_sq(Vobs_ms) / R_m
    g_err  = 2.0 * np.abs(Vobs_ms) * errV_ms / R_m
    return g_bar, g_obs, g_err

# ============================================================
# モデル
# ============================================================
def model_A(g_N, gc_a0):
    disc = g_N**2 + 4.0 * gc_a0 * a0 * g_N
    disc = np.where(disc < 0, 0.0, disc)
    return (g_N + np.sqrt(disc)) / 2.0

def model_B(g_N):
    return g_N.copy()

def compute_aic(g_pred, g_obs_data, g_err, k):
    mask = g_err > 0
    if mask.sum() < 2:
        return np.nan, np.nan
    res  = (g_pred[mask] - g_obs_data[mask]) / g_err[mask]
    chi2 = float(np.sum(res**2))
    return chi2, chi2 + 2.0 * k

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Step 3 過減衰仮説検定')
    parser.add_argument('--rotmod-dir', default=None,
                        help='SPARC _rotmod.dat ファイルのディレクトリ')
    parser.add_argument('--csv', default='sparc167_galaxy_data.csv',
                        help='サマリー CSV ファイル名')
    args = parser.parse_args()

    # ---- CSV 読み込み ----
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} が見つかりません")
        return
    summary = pd.read_csv(args.csv)
    print(f"サマリー CSV 列名: {summary.columns.tolist()}")
    print(f"総銀河数: {len(summary)}")

    # gc_a0 列の特定
    gc_col = None
    for col in summary.columns:
        if 'gc_a0' in col or 'gc/a0' in col or 'gca0' in col:
            gc_col = col
            break
    if gc_col is None:
        print("ERROR: gc_a0 列が見つかりません")
        print(summary.columns.tolist())
        return

    galaxy_col = summary.columns[0]

    # ---- .dat ファイル検索 ----
    search_dirs = []
    if args.rotmod_dir:
        search_dirs.append(args.rotmod_dir)
    search_dirs += [
        r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG',
        './Rotmod_LTG', '.',
    ]

    dat_files = {}
    for d in search_dirs:
        for fp in glob.glob(os.path.join(d, '*.dat')):
            name = os.path.basename(fp).replace('_rotmod.dat', '').replace('.dat', '')
            dat_files[name] = fp
        if dat_files:
            print(f"→ .dat ファイルを {d} で発見")
            break

    print(f"SPARC .dat ファイル数: {len(dat_files)}")
    if len(dat_files) == 0:
        print("ERROR: .dat ファイルが見つかりません")
        return

    # ---- 群の定義 ----
    sa_s0_df = summary[summary[gc_col] >  1.0]
    im_ib_df = summary[summary[gc_col] <= 0.3]
    sa_s0_names = sa_s0_df[galaxy_col].values
    im_ib_names = im_ib_df[galaxy_col].values
    print(f"\nSa/S0 (gc_a0 > 1.0) : N = {len(sa_s0_names)}")
    print(f"Im/IB (gc_a0 <= 0.3): N = {len(im_ib_names)}")

    gc_lookup = summary[[galaxy_col, gc_col]]

    # ---- 解析関数 ----
    def analyze_group(names, label):
        results, skipped = [], []
        for name in names:
            fp = dat_files.get(name)
            if fp is None:
                matches = [k for k in dat_files if name in k or k in name]
                fp = dat_files[matches[0]] if matches else None
            if fp is None:
                skipped.append(name)
                continue

            df = read_sparc_dat(fp)
            if df is None or len(df) < 3:
                skipped.append(name)
                continue

            g_bar, g_obs_data, g_err = compute_g(df)
            mask = (g_bar > 0) & (g_obs_data > 0) & (g_err > 0)
            if mask.sum() < 3:
                skipped.append(name)
                continue

            g_bar_u = g_bar[mask]
            g_obs_u = g_obs_data[mask]
            g_err_u = g_err[mask]

            gc_a0 = float(gc_lookup[gc_lookup[galaxy_col] == name][gc_col].values[0])

            chi2_A, AIC_A = compute_aic(model_A(g_bar_u, gc_a0), g_obs_u, g_err_u, k=1)
            chi2_B, AIC_B = compute_aic(model_B(g_bar_u),         g_obs_u, g_err_u, k=0)

            if np.isnan(AIC_A) or np.isnan(AIC_B):
                skipped.append(name)
                continue

            dAIC = AIC_B - AIC_A
            results.append({
                'galaxy': name, 'gc_a0': round(gc_a0, 4),
                'n_pts':  int(mask.sum()),
                'chi2_A': round(chi2_A, 3), 'chi2_B': round(chi2_B, 3),
                'AIC_A':  round(AIC_A,  3), 'AIC_B':  round(AIC_B,  3),
                'dAIC':   round(dAIC,   3),
                'winner': 'B(g=gN)' if dAIC < 0 else 'A(膜式)',
                'label':  label,
            })

        if skipped:
            print(f"  スキップ {len(skipped)} 件: {skipped[:8]}{'...' if len(skipped)>8 else ''}")
        return pd.DataFrame(results)

    print("\n--- Sa/S0 解析中 ---")
    df_sa = analyze_group(sa_s0_names, 'Sa/S0')
    print("--- Im/IB 解析中 ---")
    df_im = analyze_group(im_ib_names, 'Im/IB')

    # ---- 結果表示 ----
    for label, df_g in [('Sa/S0', df_sa), ('Im/IB', df_im)]:
        if len(df_g) == 0:
            print(f"\n{label}: 解析対象なし")
            continue
        n_B, n_tot = (df_g['dAIC'] < 0).sum(), len(df_g)
        print(f"\n{'='*55}")
        print(f"{label}  N={n_tot}")
        print(f"{'='*55}")
        print(df_g[['galaxy','gc_a0','n_pts','chi2_A','chi2_B','dAIC','winner']]
              .to_string(index=False))
        print(f"\nModel B 優勢 (g=g_N) : {n_B}/{n_tot} ({100*n_B/n_tot:.0f}%)")
        print(f"Model A 優勢 (膜式)  : {n_tot-n_B}/{n_tot} ({100*(n_tot-n_B)/n_tot:.0f}%)")
        print(f"ΔAIC 中央値 : {df_g['dAIC'].median():.3f}")
        print(f"ΔAIC 平均値 : {df_g['dAIC'].mean():.3f}")
        if n_tot >= 5:
            stat_w, p_w = stats.wilcoxon(df_g['dAIC'].values, alternative='less')
            print(f"Wilcoxon (ΔAIC < 0) : W={stat_w:.1f}, p={p_w:.4f}")

    # ---- 群間比較 ----
    if len(df_sa) >= 5 and len(df_im) >= 5:
        print(f"\n{'='*55}")
        print("群間比較 Mann-Whitney U（片側：Sa/S0 ΔAIC < Im/IB ΔAIC）")
        print(f"{'='*55}")
        stat_mw, p_mw = stats.mannwhitneyu(
            df_sa['dAIC'].values, df_im['dAIC'].values, alternative='less')
        print(f"U={stat_mw:.1f}, p={p_mw:.4f}")
        if   p_mw < 0.01: print("→ p < 0.01：Sa/S0 で有意に Model B 優勢（過減衰仮説支持）")
        elif p_mw < 0.05: print("→ p < 0.05：Sa/S0 で Model B 優勢傾向")
        else:             print("→ 有意差なし")

    # ---- CSV 保存 ----
    all_res = pd.concat([df_sa, df_im], ignore_index=True)
    out = 'step3_result.csv'
    all_res.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n結果を {out} に保存しました。")

if __name__ == '__main__':
    main()
