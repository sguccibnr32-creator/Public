# -*- coding: utf-8 -*-
"""
検証A：gc_fit vs Σ_bar の相関検証
理論予測：c_obs = 1 - [-K·Σ² + √(K²·Σ⁴ + 4(1-c_bare))] / 2

入力：
  step3_v4_result.csv（gc_fit 列）
  SPARC _rotmod.dat（SBdisk 列から Σ_bar を計算）

使用例：
  python -X utf8 step3_verifyA.py --rotmod-dir Rotmod_LTG
    --v4csv step3_v4_result.csv
    --sparc-csv sparc167_galaxy_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import os, glob, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 定数
# ============================================================
kpc_m   = 3.0857e19
Lsun_W  = 3.828e26
pc_m    = 3.0857e16

# ============================================================
# Σ_bar の計算
# SBdisk [L_sun/pc²] × Υ [M_sun/L_sun] → Σ_bar [M_sun/pc²]
# ============================================================
def compute_sigma_bar(filepath, upsilon):
    """
    _rotmod.dat の SBdisk 列から代表的な Σ_bar を計算
    代表値：r_s 付近の中央値（なければ全体中央値）
    列：Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul
    """
    rows = []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(p) for p in parts[:8]])
            except ValueError:
                continue
    if not rows:
        return np.nan
    arr   = np.array(rows)
    # SBdisk は列インデックス 6
    SBdisk = arr[:, 6]          # [L_sun/pc²]
    SBgas  = arr[:, 3]**2 / (arr[:, 0] * kpc_m / pc_m)  # 近似は使わない
    # Σ_disk [M_sun/pc²] = Υ × SBdisk
    sigma_disk = upsilon * SBdisk
    # ガス寄与は小さいので今回は省略（Σ_bar ≈ Σ_disk）
    sigma_bar  = sigma_disk
    # 外れ値を除いた中央値
    valid = sigma_bar[(sigma_bar > 0) & np.isfinite(sigma_bar)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))

# ============================================================
# 理論曲線
# ============================================================
def c_obs_theory(sigma, K, c_bare):
    """
    c_obs = 1 - x
    x = [-K·Σ² + √(K²·Σ⁴ + 4(1-c_bare))] / 2
    """
    s2 = sigma**2
    x  = (-K * s2 + np.sqrt(K**2 * s2**2 + 4*(1-c_bare))) / 2.0
    return 1.0 - x

def fit_theory(sigma_arr, gc_arr):
    """K と c_bare を最小二乗フィット"""
    def residuals(params):
        K, c_bare = params
        if K <= 0 or c_bare <= 0 or c_bare >= 1:
            return 1e9
        pred = c_obs_theory(sigma_arr, K, c_bare)
        return np.sum((pred - gc_arr)**2)

    res = optimize.differential_evolution(
        residuals,
        bounds=[(1e-6, 10.0), (0.01, 0.99)],
        seed=42, maxiter=2000, tol=1e-8, polish=True
    )
    return res.x[0], res.x[1], res.fun

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotmod-dir',  default='Rotmod_LTG')
    parser.add_argument('--v4csv',       default='step3_v4_result.csv')
    parser.add_argument('--sparc-csv',   default='sparc167_galaxy_data.csv')
    args = parser.parse_args()

    # ---- v4 結果読み込み ----
    if not os.path.exists(args.v4csv):
        print(f"ERROR: {args.v4csv} なし"); return
    v4 = pd.read_csv(args.v4csv)
    print(f"v4 結果: {len(v4)} 銀河")
    print(f"列: {v4.columns.tolist()}")

    # ---- SPARC サマリー（Υ 取得用）----
    if not os.path.exists(args.sparc_csv):
        print(f"ERROR: {args.sparc_csv} なし"); return
    summary = pd.read_csv(args.sparc_csv)
    galaxy_col = summary.columns[0]
    ml_col = next((c for c in summary.columns
                   if 'ML' in c or 'ml' in c or 'upsilon' in c.lower()), None)
    if ml_col is None:
        print("WARNING: ML(Υ) 列なし → Υ=0.5 固定")
        summary['_upsilon'] = 0.5
        ml_col = '_upsilon'
    print(f"Υ 列: '{ml_col}'")

    # ---- .dat 検索 ----
    dat_files = {}
    search_dirs = [args.rotmod_dir,
                   r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG',
                   './Rotmod_LTG', '.']
    for d in search_dirs:
        for fp in glob.glob(os.path.join(d, '*.dat')):
            name = os.path.basename(fp).replace('_rotmod.dat','').replace('.dat','')
            dat_files[name] = fp
        if dat_files:
            print(f"→ .dat を {d} で発見 ({len(dat_files)} 件)")
            break

    # ---- Σ_bar 計算 ----
    sigma_list = []
    for _, row in v4.iterrows():
        name = row['galaxy']
        fp   = dat_files.get(name)
        if fp is None:
            m = [k for k in dat_files if name in k or k in name]
            fp = dat_files[m[0]] if m else None

        up_row = summary[summary[galaxy_col] == name]
        upsilon = float(up_row[ml_col].values[0]) if len(up_row) > 0 else 0.5

        if fp:
            sig = compute_sigma_bar(fp, upsilon)
        else:
            sig = np.nan
        sigma_list.append(sig)

    v4['sigma_bar'] = sigma_list
    v4['log_sigma'] = np.log10(v4['sigma_bar'].clip(lower=1e-3))

    # 有効データのみ
    valid = v4[np.isfinite(v4['sigma_bar']) & np.isfinite(v4['gc_fit'])
               & (v4['sigma_bar'] > 0)].copy()
    print(f"\n有効サンプル: {len(valid)} / {len(v4)}")
    print(f"Σ_bar 範囲: {valid['sigma_bar'].min():.1f} ~ {valid['sigma_bar'].max():.1f} M_sun/pc²")

    # ============================================================
    # 相関検定
    # ============================================================
    print(f"\n{'='*55}")
    print("gc_fit vs Σ_bar 相関検定")
    print(f"{'='*55}")

    # Spearman（全体）
    r_all, p_all = stats.spearmanr(valid['sigma_bar'], valid['gc_fit'])
    print(f"全体  Spearman r={r_all:.3f}, p={p_all:.4f}  (N={len(valid)})")

    # 群別
    for label in ['Sa/S0', 'Im/IB']:
        sub = valid[valid['label'] == label]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(sub['sigma_bar'], sub['gc_fit'])
        print(f"{label:6s} Spearman r={r:.3f}, p={p:.4f}  (N={len(sub)})")

    # log(Σ_bar) vs gc_fit の Pearson（理論曲線が飽和形なので log スケールで線形化）
    r_log, p_log = stats.pearsonr(valid['log_sigma'], valid['gc_fit'])
    print(f"\nlog(Σ_bar) vs gc_fit Pearson r={r_log:.3f}, p={p_log:.4f}")

    # ============================================================
    # 理論曲線フィット
    # ============================================================
    print(f"\n{'='*55}")
    print("理論曲線 c_obs(Σ) フィット")
    print(f"{'='*55}")
    sig_arr = valid['sigma_bar'].values
    gc_arr  = valid['gc_fit'].values.clip(0, 0.9999)

    K_fit, c_bare_fit, res_val = fit_theory(sig_arr, gc_arr)
    print(f"K      = {K_fit:.5f}  [pc²/M_sun]²")
    print(f"c_bare = {c_bare_fit:.4f}  （Im/IB 代表値との比較：{valid[valid['label']=='Im/IB']['gc_fit'].median():.4f}）")
    rmse   = np.sqrt(res_val / len(valid))
    print(f"RMSE   = {rmse:.4f}")

    # ============================================================
    # 図
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('gc_fit vs Σ_bar：熱力学的論証の検証', fontsize=13)

    colors = {'Sa/S0': '#e94560', 'Im/IB': '#1a6fb5', 'Sd/Sc': '#2ca02c',
              'Sb/Sbc': '#ff7f0e'}

    # ---- 左：散布図 + 理論曲線 ----
    ax = axes[0]
    for label, grp in valid.groupby('label'):
        ax.scatter(grp['sigma_bar'], grp['gc_fit'],
                   label=label, alpha=0.7, s=40,
                   color=colors.get(label, 'grey'))

    sig_plot = np.logspace(np.log10(valid['sigma_bar'].min()),
                           np.log10(valid['sigma_bar'].max()), 300)
    c_plot   = c_obs_theory(sig_plot, K_fit, c_bare_fit)
    ax.plot(sig_plot, c_plot, 'k-', lw=2,
            label=f'理論 (K={K_fit:.4f}, c_bare={c_bare_fit:.3f})')
    ax.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.5)
    ax.axhline(c_bare_fit, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Σ_bar [M_sun/pc²]', fontsize=11)
    ax.set_ylabel('gc_fit (= c_obs)', fontsize=11)
    ax.set_title(f'全体 Spearman r={r_all:.3f}, p={p_all:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.5)

    # ---- 右：型別箱ひげ図 ----
    ax2 = axes[1]
    order = ['Im/IB', 'Sd/Sc', 'Sb/Sbc', 'Sa/S0']
    data_box  = [valid[valid['label']==lb]['gc_fit'].values
                 for lb in order if lb in valid['label'].values]
    labels_box = [lb for lb in order if lb in valid['label'].values]
    bp = ax2.boxplot(data_box, labels=labels_box, patch_artist=True)
    for patch, lb in zip(bp['boxes'], labels_box):
        patch.set_facecolor(colors.get(lb, 'grey'))
        patch.set_alpha(0.7)
    ax2.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.5, label='c=1 臨界点')
    ax2.set_ylabel('gc_fit (= c_obs)', fontsize=11)
    ax2.set_title('型別 gc_fit 分布', fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_fig = 'verifyA_gc_sigma.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"\n図を {out_fig} に保存しました。")

    # ---- CSV 保存 ----
    out_csv = 'verifyA_result.csv'
    valid[['galaxy','label','gc_fit','gc_csv','sigma_bar','upsilon']].to_csv(
        out_csv, index=False, encoding='utf-8-sig')
    print(f"結果を {out_csv} に保存しました。")

    # ---- サマリー ----
    print(f"\n{'='*55}")
    print("理論予測との整合性まとめ")
    print(f"{'='*55}")
    print(f"予測1：gc_fit と Σ_bar に正の相関")
    print(f"  → Spearman r={r_all:.3f}, p={p_all:.4f}  "
          f"{'支持' if p_all < 0.05 else '非支持'}")
    print(f"予測2：c_obs(Σ) = 1 - [-KΣ²+√(K²Σ⁴+4(1-c_bare))]/2 が成立")
    print(f"  → RMSE={rmse:.4f}  "
          f"{'支持（RMSE<0.15）' if rmse < 0.15 else '要検討（RMSE≥0.15）'}")
    print(f"予測3：高密度（Sa/S0）で c_obs → 1⁻")
    sa_med = valid[valid['label']=='Sa/S0']['gc_fit'].median()
    im_med = valid[valid['label']=='Im/IB']['gc_fit'].median()
    print(f"  → Sa/S0 中央値={sa_med:.3f}, Im/IB 中央値={im_med:.3f}  "
          f"{'支持' if sa_med > im_med else '非支持'}")

if __name__ == '__main__':
    main()
