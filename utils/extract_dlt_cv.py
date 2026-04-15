"""
dlt の z 方向 CV（変動係数）の抽出スクリプト
各ピクセルで dlt{z}_r10（z=3〜9）の z 方向の変動を計算
CV = std(dlt across z bins) / mean(|dlt| across z bins)
→ 過密度が赤方偏移で大きく変化するピクセルを検出
→ T-7 の c（弾性係数）の空間変動の proxy 候補
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"   # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

z_bins = list('3456789')

# ── dlt の z 方向 CV の計算 ──────────────────────────────
# 各ピクセルについて z=3〜9 の dlt 値の分布を計算
for ap in ['r10', 'r30']:
    dlt_cols = [f'dlt{z}_{ap}' for z in z_bins if f'dlt{z}_{ap}' in df.columns]
    if not dlt_cols: continue
    dlt_v = df[dlt_cols].values.astype(float)   # shape: (n_pixels, 7)

    # z 方向の統計（各行 = 1ピクセル、列 = z bin）
    dlt_std  = np.nanstd(dlt_v,  axis=1)         # z 方向の std
    dlt_mean = np.nanmean(dlt_v, axis=1)          # z 方向の mean
    dlt_abs  = np.nanmean(np.abs(dlt_v), axis=1)  # z 方向の |mean|
    dlt_max  = np.nanmax(dlt_v,  axis=1)          # z 方向の max
    dlt_min  = np.nanmin(dlt_v,  axis=1)          # z 方向の min
    dlt_range = dlt_max - dlt_min                  # z 方向のレンジ

    # CV = std / |mean|（分母がゼロのピクセルは NaN）
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(dlt_abs > 1e-6, dlt_std / dlt_abs, np.nan)

    # 正規化レンジ = range / |mean|
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_range = np.where(dlt_abs > 1e-6, dlt_range / dlt_abs, np.nan)

    # z 方向の単調性（z=3→9 で増加 or 減少）
    # 線形フィットの傾き
    z_centers = np.array([0.35,0.45,0.55,0.65,0.75,0.85,0.95])
    slopes = np.full(len(df), np.nan)
    valid_mask = np.all(np.isfinite(dlt_v), axis=1)
    if valid_mask.sum() > 0:
        X = z_centers - z_centers.mean()
        denom = (X**2).sum()
        slopes[valid_mask] = (dlt_v[valid_mask] @ X) / denom

    # ピーク位置（最大 dlt を持つ z bin）
    peak_z = np.full(len(df), np.nan)
    peak_z[valid_mask] = z_centers[np.argmax(dlt_v[valid_mask], axis=1)]

    suffix = f'_{ap}'
    df[f'dlt_cv{suffix}']         = cv
    df[f'dlt_std{suffix}']        = dlt_std
    df[f'dlt_mean{suffix}']       = dlt_mean
    df[f'dlt_range{suffix}']      = dlt_range
    df[f'dlt_norm_range{suffix}'] = norm_range
    df[f'dlt_slope{suffix}']      = slopes    # z 増加方向の傾き
    df[f'dlt_peak_z{suffix}']     = peak_z   # 最大 dlt の z

# eff_z 加重平均の dlt も再計算（クロス相関用）
eff_cols = [f'eff_z{z}' for z in z_bins if f'eff_z{z}' in df.columns]
eff_w    = df[eff_cols].values.copy().astype(float)
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0
dlt_r10 = [f'dlt{z}_r10' for z in z_bins if f'dlt{z}_r10' in df.columns]
dlt_v2  = df[dlt_r10].values.astype(float)
w2      = eff_w[:, :len(dlt_r10)]
z_dlt   = np.sum(dlt_v2 * w2, axis=1) / np.maximum(w2.sum(axis=1), 1e-9)
z_dlt[zero_mask] = np.nan
df['z_dlt_r10'] = z_dlt

# los_grad
if 'nd3_r10' in df.columns and 'nd9_r10' in df.columns:
    total = (df['nd3_r10'] + df['nd9_r10']).replace(0, np.nan)
    df['los_grad'] = (df['nd9_r10'] - df['nd3_r10']) / total

# ── 出力 CSV ─────────────────────────────────────────────
out_cols = ['ra','dec','field','skymap_id','eff_r10',
            'dlt_cv_r10','dlt_std_r10','dlt_mean_r10',
            'dlt_range_r10','dlt_norm_range_r10',
            'dlt_slope_r10','dlt_peak_z_r10',
            'dlt_cv_r30','dlt_std_r30',
            'z_dlt_r10','los_grad']
out_cols = [c for c in out_cols if c in df.columns]
df[out_cols].to_csv('sss2_dlt_cv.csv', index=False)
print(f"\n保存完了: sss2_dlt_cv.csv ({len(df)} 行, {len(out_cols)} 列)")

# ── 統計サマリー ──────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print("【dlt_cv_r10（z 方向変動係数）の基本統計】")
for col in ['dlt_cv_r10','dlt_std_r10','dlt_mean_r10',
            'dlt_range_r10','dlt_norm_range_r10']:
    s = df[col].dropna()
    print(f"  {col:25s}: n={len(s)}, mean={s.mean():.5f}, "
          f"std={s.std():.5f}, min={s.min():.5f}, max={s.max():.5f}")

print()
print("【dlt_cv の分位数】")
s = df['dlt_cv_r10'].dropna()
for q in [0.05,0.10,0.25,0.50,0.75,0.90,0.95]:
    print(f"  {int(q*100):3d}%ile: {s.quantile(q):.5f}")

print()
print("【dlt_slope の分布（z 方向の単調性）】")
sl = df['dlt_slope_r10'].dropna()
print(f"  slope mean={sl.mean():.6f}, std={sl.std():.5f}")
print(f"  slope > 0（遠方ほど過密度大）: {(sl>0).mean():.3f} ({(sl>0).mean()*100:.1f}%)")
print(f"  slope < 0（近傍ほど過密度大）: {(sl<0).mean():.3f} ({(sl<0).mean()*100:.1f}%)")

print()
print("【dlt_peak_z の分布（最大過密度の z 位置）】")
pk = df['dlt_peak_z_r10'].dropna()
for z_c in [0.35,0.45,0.55,0.65,0.75,0.85,0.95]:
    frac = (np.abs(pk - z_c) < 0.05).mean()
    print(f"  z={z_c:.2f}: {frac:.3f} ({frac*100:.1f}%)")

print()
print("【クロス相関（Spearman r）】")
pairs = [
    ('dlt_cv_r10',         'z_dlt_r10',  'CV vs 平均過密度'),
    ('dlt_std_r10',        'z_dlt_r10',  'std vs 平均過密度'),
    ('dlt_norm_range_r10', 'z_dlt_r10',  '正規化レンジ vs 過密度'),
    ('dlt_slope_r10',      'z_dlt_r10',  '傾き vs 過密度'),
    ('dlt_cv_r10',         'los_grad',   'CV vs 視線勾配'),
    ('dlt_cv_r10',         'dlt_cv_r30', 'CV r10 vs r30（一致確認）'),
]
for a,b,label in pairs:
    if a not in df.columns or b not in df.columns: continue
    mask = df[a].notna() & df[b].notna()
    r, p = spearmanr(df.loc[mask,a], df.loc[mask,b])
    print(f"  {label}")
    print(f"    {a} vs {b}: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【field 別 dlt_cv_r10】")
for fid in sorted(df['field'].dropna().unique()):
    s = df.loc[df['field']==fid, 'dlt_cv_r10'].dropna()
    high_cv = (s > s.quantile(0.9)).mean()
    print(f"  field={int(fid)}: mean={s.mean():.5f}, "
          f"std={s.std():.5f}, 上位10%: {high_cv:.3f}")

print()
print("【高 CV ピクセルの特性（上位 5%）】")
thresh_hi = df['dlt_cv_r10'].quantile(0.95)
hi_cv = df[df['dlt_cv_r10'] >= thresh_hi]
print(f"  閾値 CV > {thresh_hi:.4f}: {len(hi_cv)} ピクセル")
for col in ['z_dlt_r10','los_grad']:
    if col in hi_cv.columns:
        s = hi_cv[col].dropna()
        s_all = df[col].dropna()
        print(f"  高CV の {col}: mean={s.mean():.5f}（全体 {s_all.mean():.5f}）")