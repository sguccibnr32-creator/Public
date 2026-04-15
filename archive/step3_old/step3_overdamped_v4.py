# -*- coding: utf-8 -*-
"""
Step 3 v4: 過減衰仮説検定（per-galaxy Υ 固定版）
CSV の ML(Υ) 列を銀河ごとに使用し gc_a0 のみ自由パラメータ（k=1）
縮退問題を回避した最もクリーンな設計

使用例:
  python -X utf8 step3_overdamped_v4.py --rotmod-dir Rotmod_LTG --csv sparc167_galaxy_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import os, glob, argparse

# ============================================================
# 定数
# ============================================================
a0      = 1.2e-10
kpc_m   = 3.0857e19
kms_m   = 1e3
UPSILON_BUL = 0.7    # バルジは固定

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

def compute_g(df, upsilon):
    """per-galaxy Υ を使って g_bar・g_obs・g_err を計算"""
    R_m      = df['R'].values    * kpc_m
    Vobs_ms  = df['Vobs'].values * kms_m
    errV_ms  = df['errV'].values * kms_m
    Vgas_ms  = df['Vgas'].values * kms_m
    Vdisk_ms = df['Vdisk'].values * kms_m
    Vbul_ms  = df['Vbul'].values * kms_m

    def ssq(V): return np.sign(V) * V**2

    g_bar = (ssq(Vgas_ms)
             + upsilon      * ssq(Vdisk_ms)
             + UPSILON_BUL  * ssq(Vbul_ms)) / R_m
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

def chi2_val(g_pred, g_obs_data, g_err):
    mask = g_err > 0
    if mask.sum() < 2:
        return np.nan
    return float(np.sum(((g_pred[mask] - g_obs_data[mask]) / g_err[mask])**2))

def fit_model_A(g_bar, g_obs_data, g_err):
    """gc_a0 のみ最適化（Υ は固定済み・k=1）"""
    def obj(log_gc):
        gc = 10.0**log_gc
        mask = (g_bar > 0) & (g_obs_data > 0) & (g_err > 0)
        if mask.sum() < 2:
            return 1e9
        return chi2_val(model_A(g_bar[mask], gc), g_obs_data[mask], g_err[mask])

    res = optimize.minimize_scalar(obj, bounds=(-4, 2), method='bounded')
    return 10.0**res.x, float(res.fun)

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotmod-dir', default=None)
    parser.add_argument('--csv', default='sparc167_galaxy_data.csv')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} が見つかりません"); return

    summary = pd.read_csv(args.csv)
    print(f"CSV 列名: {summary.columns.tolist()}")

    # ---- 列の特定 ----
    gc_col = next((c for c in summary.columns
                   if 'gc_a0' in c or 'gc/a0' in c or 'gca0' in c), None)
    if gc_col is None:
        print("ERROR: gc_a0 列なし"); return

    # ML(Υ) 列の候補を探す
    ml_col = next((c for c in summary.columns
                   if 'ML' in c or 'ml' in c or 'upsilon' in c.lower()
                   or 'Upsilon' in c or 'Y_' in c or '_Y' in c), None)
    if ml_col is None:
        print("WARNING: ML(Υ) 列が見つかりません。")
        print("利用可能な列:", summary.columns.tolist())
        print("--ml-col オプションで列名を指定してください。")
        return
    print(f"使用する Υ 列: '{ml_col}'")
    print(f"Υ の範囲: {summary[ml_col].min():.3f} ~ {summary[ml_col].max():.3f}")
    print(f"Υ 中央値: {summary[ml_col].median():.3f}")

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
        print("ERROR: .dat なし"); return

    # ---- 群定義 ----
    sa_names = summary[summary[gc_col] >  1.0][galaxy_col].values
    im_names = summary[summary[gc_col] <= 0.3][galaxy_col].values
    print(f"\nSa/S0 (gc_a0>1.0) : N={len(sa_names)}")
    print(f"Im/IB (gc_a0≤0.3) : N={len(im_names)}")

    # ---- 解析 ----
    def analyze_group(names, label):
        results, skipped = [], []
        total = len(names)
        for i, name in enumerate(names):
            print(f"  [{i+1:3d}/{total}] {name:<20s}", end='\r')

            row = summary[summary[galaxy_col] == name]
            if len(row) == 0:
                skipped.append(name); continue

            gc_csv  = float(row[gc_col].values[0])
            upsilon = float(row[ml_col].values[0])
            if np.isnan(upsilon) or upsilon <= 0:
                upsilon = 0.5   # フォールバック

            fp = dat_files.get(name)
            if fp is None:
                m = [k for k in dat_files if name in k or k in name]
                fp = dat_files[m[0]] if m else None
            if fp is None:
                skipped.append(name); continue

            df = read_sparc_dat(fp)
            if df is None or len(df) < 3:
                skipped.append(name); continue

            g_bar, g_obs_data, g_err = compute_g(df, upsilon)
            mask = (g_bar > 0) & (g_obs_data > 0) & (g_err > 0)
            if mask.sum() < 3:
                skipped.append(name); continue

            gb = g_bar[mask]; go = g_obs_data[mask]; ge = g_err[mask]

            # Model B：g_obs = g_N（k=0）
            chi2_B = chi2_val(gb, go, ge)
            AIC_B  = chi2_B

            # Model A：gc_a0 フィット（k=1）
            gc_fit, chi2_A = fit_model_A(gb, go, ge)
            AIC_A = chi2_A + 2.0

            if np.isnan(AIC_A) or np.isnan(AIC_B):
                skipped.append(name); continue

            dAIC = AIC_B - AIC_A
            results.append({
                'galaxy':   name,
                'gc_csv':   round(gc_csv,  4),
                'gc_fit':   round(gc_fit,  4),
                'upsilon':  round(upsilon, 3),
                'n_pts':    int(mask.sum()),
                'chi2_A':   round(chi2_A,  2),
                'chi2_B':   round(chi2_B,  2),
                'dAIC':     round(dAIC,    2),
                'winner':   'B(g=gN)' if dAIC < 0 else 'A(膜式)',
                'label':    label,
            })

        print()
        if skipped:
            print(f"  スキップ {len(skipped)} 件: {skipped[:8]}"
                  f"{'...' if len(skipped)>8 else ''}")
        return pd.DataFrame(results)

    print("\n--- Sa/S0 解析中 ---")
    df_sa = analyze_group(sa_names, 'Sa/S0')
    print("--- Im/IB 解析中 ---")
    df_im = analyze_group(im_names, 'Im/IB')

    # ---- 結果表示 ----
    for label, df_g in [('Sa/S0', df_sa), ('Im/IB', df_im)]:
        if len(df_g) == 0:
            print(f"\n{label}: 解析対象なし"); continue
        n_B = (df_g['dAIC'] < 0).sum(); n_tot = len(df_g)
        print(f"\n{'='*65}")
        print(f"{label}  N={n_tot}  [per-galaxy Υ 固定・gc_a0 のみフィット]")
        print(f"{'='*65}")
        print(df_g[['galaxy','gc_csv','gc_fit','upsilon',
                    'n_pts','chi2_A','chi2_B','dAIC','winner']].to_string(index=False))
        print(f"\nModel B 優勢 (g=g_N): {n_B}/{n_tot} ({100*n_B/n_tot:.0f}%)")
        print(f"Model A 優勢 (膜式) : {n_tot-n_B}/{n_tot} ({100*(n_tot-n_B)/n_tot:.0f}%)")
        print(f"ΔAIC  中央値: {df_g['dAIC'].median():.2f}")
        print(f"gc_fit 中央値: {df_g['gc_fit'].median():.4f}")
        print(f"Υ     中央値: {df_g['upsilon'].median():.3f}")
        if n_tot >= 5:
            stat_w, p_w = stats.wilcoxon(df_g['dAIC'].values, alternative='less')
            print(f"Wilcoxon (ΔAIC<0): W={stat_w:.1f}, p={p_w:.4f}")

    # ---- 群間比較 ----
    if len(df_sa) >= 5 and len(df_im) >= 5:
        print(f"\n{'='*65}")
        print("群間比較（片側：Sa/S0 ΔAIC < Im/IB ΔAIC）")
        stat_mw, p_mw = stats.mannwhitneyu(
            df_sa['dAIC'].values, df_im['dAIC'].values, alternative='less')
        print(f"ΔAIC Mann-Whitney: U={stat_mw:.1f}, p={p_mw:.4f}")
        if   p_mw < 0.01: print("→ p<0.01：Sa/S0 で有意に Model B 優勢（過減衰仮説支持）")
        elif p_mw < 0.05: print("→ p<0.05：Sa/S0 で Model B 優勢傾向")
        else:             print("→ 有意差なし")

        print(f"\n--- gc_fit 群間比較（Υ補正後）---")
        print(f"Sa/S0 gc_fit 中央値: {df_sa['gc_fit'].median():.4f}")
        print(f"Im/IB gc_fit 中央値: {df_im['gc_fit'].median():.4f}")
        stat_gc, p_gc = stats.mannwhitneyu(
            df_sa['gc_fit'].values, df_im['gc_fit'].values, alternative='greater')
        print(f"gc_fit 群間 Mann-Whitney: U={stat_gc:.1f}, p={p_gc:.4f}")

    # ---- CSV 保存 ----
    out = 'step3_v4_result.csv'
    pd.concat([df_sa, df_im], ignore_index=True).to_csv(
        out, index=False, encoding='utf-8-sig')
    print(f"\n結果を {out} に保存しました。")

if __name__ == '__main__':
    main()
