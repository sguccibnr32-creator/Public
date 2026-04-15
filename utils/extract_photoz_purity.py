"""
photo-z 純度 p_high の抽出スクリプト v3
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"   # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

z_bins = list('3456789')

# ── photo-z 純度の計算 ──────────────────────────────────────
print("photo-z 純度を計算中...")
new_cols = {}

for ap in ['r10', 'r30']:
    for z in z_bins:
        nd_col = f'nd{z}_{ap}'
        n80_col = f'n{z}_80_{ap}'
        n68_col = f'n{z}_68_{ap}'
        n24_col = f'n{z}_24_{ap}'
        n02_col = f'n{z}_02_{ap}'
        if nd_col not in df.columns or n80_col not in df.columns:
            continue
        denom = df[nd_col].replace(0, np.nan)
        new_cols[f'p_high{z}_{ap}'] = (df[n80_col] / denom).values
        new_cols[f'p_peak{z}_{ap}'] = ((df[n68_col] + df[n80_col]) / denom).values
        new_cols[f'p_low{z}_{ap}']  = ((df[n02_col] + df[n24_col]) / denom).values

# ── eff_z 加重平均 ────────────────────────────────────────────
eff_cols = [f'eff_z{z}' for z in z_bins if f'eff_z{z}' in df.columns]
eff_w    = df[eff_cols].values.copy().astype(float)
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0

def wt_avg(col_list):
    valid = [c for c in col_list if c in new_cols]   # new_cols のみ参照
    if not valid:
        return np.full(len(df), np.nan)
    vals = np.column_stack([new_cols[c] for c in valid])
    w    = eff_w[:, :len(valid)]
    out  = np.average(vals.astype(float), weights=w, axis=1)
    out[zero_mask] = np.nan
    return out

for ap in ['r10', 'r30']:
    new_cols[f'p_high_wt_{ap}'] = wt_avg([f'p_high{z}_{ap}' for z in z_bins])
    new_cols[f'p_peak_wt_{ap}'] = wt_avg([f'p_peak{z}_{ap}' for z in z_bins])
    new_cols[f'p_low_wt_{ap}']  = wt_avg([f'p_low{z}_{ap}'  for z in z_bins])

new_cols['p_high_eff_r10'] = new_cols['p_high_wt_r10'] * df['eff_r10'].values

# z_dlt_r10 も計算
dlt_r10 = [f'dlt{z}_r10' for z in z_bins if f'dlt{z}_r10' in df.columns]
dlt_v   = df[dlt_r10].values.astype(float)
w2      = eff_w[:, :len(dlt_r10)]
z_dlt   = np.sum(dlt_v * w2, axis=1) / np.maximum(w2.sum(axis=1), 1e-9)
z_dlt[zero_mask] = np.nan
new_cols['z_dlt_r10'] = z_dlt

# ── 出力 CSV ─────────────────────────────────────────────────
out_df = pd.DataFrame({
    'ra': df['ra'], 'dec': df['dec'],
    'field': df['field'], 'skymap_id': df['skymap_id'],
    'eff_r10': df['eff_r10']
})
for k, v in new_cols.items():
    if '_r30' not in k:   # r10 のみ保存（軽量化）
        out_df[k] = v

out_df.to_csv('sss2_photoz_purity.csv', index=False)
print(f"\n保存完了: sss2_photoz_purity.csv ({len(out_df)} 行, {len(out_df.columns)} 列)")

# ── 統計サマリー ──────────────────────────────────────────────
print("\n=== 統計サマリー ===\n")

print("【z bin 別 p_high_r10（高確率銀河の割合）】")
print(f"{'z bin':6s}  {'mean':7s}  {'std':7s}  {'min':7s}  {'max':7s}")
print("-"*42)
for z in z_bins:
    s = pd.Series(new_cols.get(f'p_high{z}_r10', [])).dropna()
    if len(s):
        print(f"z={z}     {s.mean():.4f}   {s.std():.4f}   "
              f"{s.min():.4f}   {s.max():.4f}")

print()
print("【z bin 別 p_low_r10（汚染率）】")
print(f"{'z bin':6s}  {'mean':7s}  {'std':7s}")
print("-"*24)
for z in z_bins:
    s = pd.Series(new_cols.get(f'p_low{z}_r10', [])).dropna()
    if len(s):
        print(f"z={z}     {s.mean():.4f}   {s.std():.4f}")

print()
print("【加重平均 proxy】")
for col in ['p_high_wt_r10','p_peak_wt_r10','p_low_wt_r10','p_high_eff_r10']:
    s = pd.Series(new_cols.get(col, [])).dropna()
    if len(s):
        print(f"{col}: n={len(s)}, mean={s.mean():.5f}, "
              f"std={s.std():.5f}, min={s.min():.5f}, max={s.max():.5f}")

print()
print("【p_high_wt_r10 vs z_dlt_r10 の相関】")
ph = pd.Series(new_cols['p_high_wt_r10'])
zd = pd.Series(new_cols['z_dlt_r10'])
mask = ph.notna() & zd.notna()
r, p = spearmanr(ph[mask], zd[mask])
print(f"  r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【field 別 p_high_wt_r10】")
for fid in sorted(df['field'].dropna().unique()):
    s = pd.Series(new_cols['p_high_wt_r10'])[df['field'].values == fid]
    s = s.dropna()
    print(f"  field={int(fid)}: mean={s.mean():.4f}, std={s.std():.4f}, n={len(s)}")