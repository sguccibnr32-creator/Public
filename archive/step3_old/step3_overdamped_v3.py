# -*- coding: utf-8 -*-
"""
Step 3 v3: 過減衰仮説検定（Υ 自由パラメータ版）
Model A2: gc_a0 + Υ を同時フィット（k=2）
Model B2: Υ のみフィット（k=1）
ΔAIC = AIC_B2 - AIC_A2 で公平比較

使用例:
  python -X utf8 step3_overdamped_v3.py --rotmod-dir Rotmod_LTG --csv sparc167_galaxy_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import os, glob, argparse

# ============================================================
# 定数
# ============================================================
a0      = 1.2e-10   # m/s²
kpc_m   = 3.0857e19
kms_m   = 1e3
# Υ の探索範囲（型に応じた物理的上限）
UPSILON_MIN = 0.2
UPSILON_MAX = 3.0
UPSILON_BUL = 0.7   # バルジは固定（観測的制約が強い）

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

def compute_components(df):
    """速度成分を m/s 単位で返す（g_bar は Υ に依存するので分離）"""
    R_m      = df['R'].values    * kpc_m
    Vobs_ms  = df['Vobs'].values * kms_m
    errV_ms  = df['errV'].values * kms_m
    Vgas_ms  = df['Vgas'].values * kms_m
    Vdisk_ms = df['Vdisk'].values * kms_m
    Vbul_ms  = df['Vbul'].values * kms_m

    def ssq(V): return np.sign(V) * V**2

    # g_bar を Υ から分離して返す
    g_gas  = ssq(Vgas_ms)  / R_m                    # Υ 依存なし
    g_disk = ssq(Vdisk_ms) / R_m                    # × Υ
    g_bul  = UPSILON_BUL * ssq(Vbul_ms) / R_m       # 固定
    g_obs  = ssq(Vobs_ms) / R_m
    g_err  = 2.0 * np.abs(Vobs_ms) * errV_ms / R_m
    return g_gas, g_disk, g_bul, g_obs, g_err

def g_bar_from(g_gas, g_disk, g_bul, upsilon):
    return g_gas + upsilon * g_disk + g_bul

# ============================================================
# モデル
# ============================================================
def model_A2(g_N, gc_a0):
    disc = g_N**2 + 4.0 * gc_a0 * a0 * g_N
    disc = np.where(disc < 0, 0.0, disc)
    return (g_N + np.sqrt(disc)) / 2.0

def chi2_val(g_pred, g_obs_data, g_err):
    mask = g_err > 0
    if mask.sum() < 2:
        return np.nan
    return float(np.sum(((g_pred[mask] - g_obs_data[mask]) / g_err[mask])**2))

# ============================================================
# Model B2：Υ のみフィット（k=1）
# ============================================================
def fit_model_B2(g_gas, g_disk, g_bul, g_obs_data, g_err):
    """g_obs = g_bar(Υ) で Υ を最適化"""
    def obj(upsilon):
        gb = g_bar_from(g_gas, g_disk, g_bul, upsilon)
        mask = (gb > 0) & (g_obs_data > 0) & (g_err > 0)
        if mask.sum() < 2:
            return 1e9
        return chi2_val(gb[mask], g_obs_data[mask], g_err[mask])

    res = optimize.minimize_scalar(obj,
                                   bounds=(UPSILON_MIN, UPSILON_MAX),
                                   method='bounded')
    upsilon_best = res.x
    chi2_min     = res.fun
    return upsilon_best, chi2_min

# ============================================================
# Model A2：gc_a0 + Υ を同時フィット（k=2）
# ============================================================
def fit_model_A2(g_gas, g_disk, g_bul, g_obs_data, g_err):
    """gc_a0 と Υ を境界付き differential_evolution で同時最適化"""
    def obj(params):
        log_gc, upsilon = params
        gc = 10.0**log_gc
        gb = g_bar_from(g_gas, g_disk, g_bul, upsilon)
        mask = (gb > 0) & (g_obs_data > 0) & (g_err > 0)
        if mask.sum() < 2:
            return 1e9
        g_pred = model_A2(gb[mask], gc)
        return chi2_val(g_pred, g_obs_data[mask], g_err[mask])

    bounds = [(-4, 2), (UPSILON_MIN, UPSILON_MAX)]
    res = optimize.differential_evolution(
        obj, bounds, seed=42, maxiter=1000, tol=1e-6,
        workers=1, polish=True)
    gc_best      = 10.0**res.x[0]
    upsilon_best = float(res.x[1])
    chi2_min     = float(res.fun)
    return gc_best, upsilon_best, chi2_min

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
    print(f"サマリー CSV 列名: {summary.columns.tolist()}")

    gc_col = None
    for col in summary.columns:
        if 'gc_a0' in col or 'gc/a0' in col or 'gca0' in col:
            gc_col = col; break
    if gc_col is None:
        print("ERROR: gc_a0 列なし"); return
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
            print(f"  [{i+1}/{total}] {name}", end='\r')
            fp = dat_files.get(name)
            if fp is None:
                m = [k for k in dat_files if name in k or k in name]
                fp = dat_files[m[0]] if m else None
            if fp is None:
                skipped.append(name); continue

            df = read_sparc_dat(fp)
            if df is None or len(df) < 3:
                skipped.append(name); continue

            g_gas, g_disk, g_bul, g_obs_data, g_err = compute_components(df)
            # 基本マスク（errV>0・Vobs>0）
            mask0 = (g_obs_data > 0) & (g_err > 0)
            if mask0.sum() < 3:
                skipped.append(name); continue

            g_gas  = g_gas[mask0];  g_disk = g_disk[mask0]
            g_bul  = g_bul[mask0];  g_obs_data = g_obs_data[mask0]
            g_err  = g_err[mask0]

            gc_csv = float(summary[summary[galaxy_col]==name][gc_col].values[0])

            # Model B2（Υ フィット・k=1）
            up_B, chi2_B = fit_model_B2(g_gas, g_disk, g_bul, g_obs_data, g_err)
            AIC_B = chi2_B + 2.0 * 1

            # Model A2（gc_a0 + Υ フィット・k=2）
            gc_fit, up_A, chi2_A = fit_model_A2(g_gas, g_disk, g_bul, g_obs_data, g_err)
            AIC_A = chi2_A + 2.0 * 2

            if np.isnan(AIC_A) or np.isnan(AIC_B):
                skipped.append(name); continue

            dAIC = AIC_B - AIC_A  # >0 → A優勢、<0 → B優勢

            results.append({
                'galaxy':  name,
                'gc_csv':  round(gc_csv, 4),
                'gc_fit':  round(gc_fit, 4),
                'up_A':    round(up_A,   3),   # Model A2 の最適 Υ
                'up_B':    round(up_B,   3),   # Model B2 の最適 Υ
                'n_pts':   int(mask0.sum()),
                'chi2_A':  round(chi2_A, 2),
                'chi2_B':  round(chi2_B, 2),
                'dAIC':    round(dAIC,   2),
                'winner':  'B(g=gN)' if dAIC < 0 else 'A(膜式)',
                'label':   label,
            })

        print()
        if skipped:
            print(f"  スキップ {len(skipped)} 件: {skipped[:8]}"
                  f"{'...' if len(skipped)>8 else ''}")
        return pd.DataFrame(results)

    print("\n--- Sa/S0 解析中（gc_a0 + Υ 同時フィット）---")
    df_sa = analyze_group(sa_names, 'Sa/S0')
    print("--- Im/IB 解析中（gc_a0 + Υ 同時フィット）---")
    df_im = analyze_group(im_names, 'Im/IB')

    # ---- 結果表示 ----
    for label, df_g in [('Sa/S0', df_sa), ('Im/IB', df_im)]:
        if len(df_g) == 0:
            print(f"\n{label}: 解析対象なし"); continue
        n_B = (df_g['dAIC'] < 0).sum(); n_tot = len(df_g)
        print(f"\n{'='*65}")
        print(f"{label}  N={n_tot}  [Υ 自由パラメータ版]")
        print(f"{'='*65}")
        print(df_g[['galaxy','gc_csv','gc_fit','up_A','up_B',
                    'chi2_A','chi2_B','dAIC','winner']].to_string(index=False))
        print(f"\nModel B2 優勢 (Υフィット・g=gN): {n_B}/{n_tot} ({100*n_B/n_tot:.0f}%)")
        print(f"Model A2 優勢 (膜式+Υフィット) : {n_tot-n_B}/{n_tot} ({100*(n_tot-n_B)/n_tot:.0f}%)")
        print(f"ΔAIC 中央値: {df_g['dAIC'].median():.2f}")
        print(f"gc_fit 中央値: {df_g['gc_fit'].median():.4f}")
        print(f"up_A  中央値: {df_g['up_A'].median():.3f}")
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
        if   p_mw < 0.01: print("→ p<0.01：Sa/S0 で有意に Model B2 優勢（過減衰仮説支持）")
        elif p_mw < 0.05: print("→ p<0.05：Sa/S0 で Model B2 優勢傾向")
        else:             print("→ 有意差なし")

        # gc_fit 群間比較
        print(f"\n--- gc_fit 群間比較（Υ補正後）---")
        print(f"Sa/S0 gc_fit 中央値: {df_sa['gc_fit'].median():.4f}")
        print(f"Im/IB gc_fit 中央値: {df_im['gc_fit'].median():.4f}")
        stat_gc, p_gc = stats.mannwhitneyu(
            df_sa['gc_fit'].values, df_im['gc_fit'].values, alternative='greater')
        print(f"gc_fit 群間 Mann-Whitney: U={stat_gc:.1f}, p={p_gc:.4f}")

        # Υ 群間比較（副次的確認）
        print(f"\n--- up_A 群間比較（最適 Υ）---")
        print(f"Sa/S0 up_A 中央値: {df_sa['up_A'].median():.3f}")
        print(f"Im/IB up_A 中央値: {df_im['up_A'].median():.3f}")
        stat_up, p_up = stats.mannwhitneyu(
            df_sa['up_A'].values, df_im['up_A'].values, alternative='greater')
        print(f"up_A 群間 Mann-Whitney: U={stat_up:.1f}, p={p_up:.4f}")

    # ---- CSV 保存 ----
    out = 'step3_v3_result.csv'
    pd.concat([df_sa, df_im], ignore_index=True).to_csv(
        out, index=False, encoding='utf-8-sig')
    print(f"\n結果を {out} に保存しました。")

if __name__ == '__main__':
    main()
