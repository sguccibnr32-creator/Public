# -*- coding: utf-8 -*-
"""
Step 3 v2: 過減衰仮説検定（循環論理修正版）
gc_a0 を自由パラメータとして各銀河で再フィットし
Model A vs Model B を公平に AIC 比較する

使用例:
  python -X utf8 step3_overdamped_v2.py --rotmod-dir Rotmod_LTG --csv sparc167_galaxy_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import os, glob, argparse

# ============================================================
# 定数
# ============================================================
a0          = 1.2e-10   # m/s²
kpc_m       = 3.0857e19 # 1 kpc → m
kms_m       = 1e3       # 1 km/s → m/s
UPSILON     = 0.5       # ディスク Υ（固定・将来改善余地あり）
UPSILON_BUL = 0.7       # バルジ Υ

# ============================================================
# SPARC .dat 読み込み
# ============================================================
def read_sparc_dat(filepath):
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
    df = pd.DataFrame()
    df['R']     = arr[:, 0]
    df['Vobs']  = arr[:, 1]
    df['errV']  = arr[:, 2]
    df['Vgas']  = arr[:, 3]
    df['Vdisk'] = arr[:, 4]
    df['Vbul']  = arr[:, 5] if arr.shape[1] > 5 else 0.0
    return df

def compute_g(df):
    R_m      = df['R'].values    * kpc_m
    Vobs_ms  = df['Vobs'].values * kms_m
    errV_ms  = df['errV'].values * kms_m
    Vgas_ms  = df['Vgas'].values * kms_m
    Vdisk_ms = df['Vdisk'].values * kms_m
    Vbul_ms  = df['Vbul'].values * kms_m

    def ssq(V): return np.sign(V) * V**2

    g_bar = (ssq(Vgas_ms)
             + UPSILON     * ssq(Vdisk_ms)
             + UPSILON_BUL * ssq(Vbul_ms)) / R_m
    g_obs = ssq(Vobs_ms) / R_m
    g_err = 2.0 * np.abs(Vobs_ms) * errV_ms / R_m
    return g_bar, g_obs, g_err

# ============================================================
# モデル
# ============================================================
def model_A(g_N, gc_a0):
    disc = g_N**2 + 4.0 * gc_a0 * a0 * g_N
    disc = np.where(disc < 0, 0.0, disc)
    return (g_N + np.sqrt(disc)) / 2.0

def chi2_func(g_pred, g_obs_data, g_err):
    mask = g_err > 0
    if mask.sum() < 2:
        return np.nan
    return float(np.sum(((g_pred[mask] - g_obs_data[mask]) / g_err[mask])**2))

# ============================================================
# Model A：gc_a0 を自由パラメータとして最適化（循環論理なし）
# ============================================================
def fit_model_A(g_bar, g_obs_data, g_err):
    """
    gc_a0 in (1e-4, 100) を log スケールで最適化
    返り値：(gc_a0_best, chi2_min)
    """
    def objective(log_gc):
        gc = 10.0**log_gc
        return chi2_func(model_A(g_bar, gc), g_obs_data, g_err)

    res = optimize.minimize_scalar(objective, bounds=(-4, 2), method='bounded')
    gc_best  = 10.0**res.x
    chi2_min = res.fun
    return gc_best, chi2_min

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotmod-dir', default=None)
    parser.add_argument('--csv', default='sparc167_galaxy_data.csv')
    args = parser.parse_args()

    # ---- CSV ----
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} が見つかりません"); return
    summary = pd.read_csv(args.csv)
    print(f"サマリー CSV 列名: {summary.columns.tolist()}")
    print(f"総銀河数: {len(summary)}")

    gc_col = None
    for col in summary.columns:
        if 'gc_a0' in col or 'gc/a0' in col or 'gca0' in col:
            gc_col = col; break
    if gc_col is None:
        print("ERROR: gc_a0 列が見つかりません"); return
    galaxy_col = summary.columns[0]

    # ---- .dat 検索 ----
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
            name = os.path.basename(fp).replace('_rotmod.dat','').replace('.dat','')
            dat_files[name] = fp
        if dat_files:
            print(f"→ .dat を {d} で発見 ({len(dat_files)} 件)")
            break
    if not dat_files:
        print("ERROR: .dat ファイルが見つかりません"); return

    # ---- 群定義 ----
    sa_names = summary[summary[gc_col] >  1.0][galaxy_col].values
    im_names = summary[summary[gc_col] <= 0.3][galaxy_col].values
    print(f"\nSa/S0 (gc_a0 > 1.0) : N = {len(sa_names)}")
    print(f"Im/IB (gc_a0 <= 0.3): N = {len(im_names)}")

    # ---- 解析 ----
    def analyze_group(names, label):
        results, skipped = [], []
        for name in names:
            fp = dat_files.get(name)
            if fp is None:
                m = [k for k in dat_files if name in k or k in name]
                fp = dat_files[m[0]] if m else None
            if fp is None:
                skipped.append(name); continue

            df = read_sparc_dat(fp)
            if df is None or len(df) < 3:
                skipped.append(name); continue

            g_bar, g_obs_data, g_err = compute_g(df)
            mask = (g_bar > 0) & (g_obs_data > 0) & (g_err > 0)
            if mask.sum() < 3:
                skipped.append(name); continue

            gb = g_bar[mask]; go = g_obs_data[mask]; ge = g_err[mask]

            # Model B（パラメータなし）
            chi2_B = chi2_func(gb, go, ge)
            AIC_B  = chi2_B                    # k=0

            # Model A（gc_a0 を新規フィット・循環なし）
            gc_fit, chi2_A = fit_model_A(gb, go, ge)
            AIC_A  = chi2_A + 2.0              # k=1

            if np.isnan(AIC_A) or np.isnan(AIC_B):
                skipped.append(name); continue

            # CSV の gc_a0（参考値）
            gc_csv = float(summary[summary[galaxy_col]==name][gc_col].values[0])

            dAIC = AIC_B - AIC_A
            results.append({
                'galaxy':   name,
                'gc_csv':   round(gc_csv, 4),   # 元の CSV 値（参考）
                'gc_fit':   round(gc_fit, 4),   # 今回フィット値（正式）
                'n_pts':    int(mask.sum()),
                'chi2_A':   round(chi2_A, 2),
                'chi2_B':   round(chi2_B, 2),
                'dAIC':     round(dAIC,   2),
                'winner':   'B(g=gN)' if dAIC < 0 else 'A(膜式)',
                'label':    label,
            })

        if skipped:
            print(f"  スキップ {len(skipped)} 件: {skipped[:8]}"
                  f"{'...' if len(skipped)>8 else ''}")
        return pd.DataFrame(results)

    print("\n--- Sa/S0 解析中（gc_a0 再フィット）---")
    df_sa = analyze_group(sa_names, 'Sa/S0')
    print("--- Im/IB 解析中（gc_a0 再フィット）---")
    df_im = analyze_group(im_names, 'Im/IB')

    # ---- 結果表示 ----
    for label, df_g in [('Sa/S0', df_sa), ('Im/IB', df_im)]:
        if len(df_g) == 0:
            print(f"\n{label}: 解析対象なし"); continue
        n_B = (df_g['dAIC'] < 0).sum(); n_tot = len(df_g)
        print(f"\n{'='*60}")
        print(f"{label}  N={n_tot}  [gc_a0 再フィット版・循環なし]")
        print(f"{'='*60}")
        print(df_g[['galaxy','gc_csv','gc_fit','n_pts',
                    'chi2_A','chi2_B','dAIC','winner']].to_string(index=False))
        print(f"\nModel B 優勢 (g=g_N): {n_B}/{n_tot} ({100*n_B/n_tot:.0f}%)")
        print(f"Model A 優勢 (膜式) : {n_tot-n_B}/{n_tot} ({100*(n_tot-n_B)/n_tot:.0f}%)")
        print(f"ΔAIC 中央値: {df_g['dAIC'].median():.2f}")
        print(f"ΔAIC 平均値: {df_g['dAIC'].mean():.2f}")
        if n_tot >= 5:
            stat_w, p_w = stats.wilcoxon(df_g['dAIC'].values, alternative='less')
            print(f"Wilcoxon (ΔAIC < 0): W={stat_w:.1f}, p={p_w:.4f}")

    # ---- 群間比較 ----
    if len(df_sa) >= 5 and len(df_im) >= 5:
        print(f"\n{'='*60}")
        print("群間比較 Mann-Whitney U（片側：Sa/S0 ΔAIC < Im/IB ΔAIC）")
        print(f"{'='*60}")
        stat_mw, p_mw = stats.mannwhitneyu(
            df_sa['dAIC'].values, df_im['dAIC'].values, alternative='less')
        print(f"U={stat_mw:.1f}, p={p_mw:.4f}")
        if   p_mw < 0.01: print("→ p<0.01：Sa/S0 で有意に Model B 優勢（過減衰仮説支持）")
        elif p_mw < 0.05: print("→ p<0.05：Sa/S0 で Model B 優勢傾向")
        else:             print("→ 有意差なし")

        # 追加：gc_fit の群間比較（gc_csv との乖離も確認）
        print(f"\n--- gc_fit 分布比較（再フィット値） ---")
        print(f"Sa/S0 gc_fit 中央値: {df_sa['gc_fit'].median():.4f}")
        print(f"Im/IB gc_fit 中央値: {df_im['gc_fit'].median():.4f}")
        stat_gc, p_gc = stats.mannwhitneyu(
            df_sa['gc_fit'].values, df_im['gc_fit'].values, alternative='greater')
        print(f"gc_fit 群間 Mann-Whitney: U={stat_gc:.1f}, p={p_gc:.4f}")

    # ---- CSV 保存 ----
    out = 'step3_v2_result.csv'
    pd.concat([df_sa, df_im], ignore_index=True).to_csv(
        out, index=False, encoding='utf-8-sig')
    print(f"\n結果を {out} に保存しました。")

if __name__ == '__main__':
    main()
