"""
aperture 差 sgm_c10 - sgm_r10 の抽出スクリプト
円形(c10) vs 矩形(r10) の sigma の差 = 外縁部の過密度 proxy
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"   # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

z_bins = list('3456789')
new_cols = {}

# ── aperture 差の計算 ─────────────────────────────────────
# ap_diff{z} = sgm{z}_c10 - sgm{z}_r10
# > 0: 円形(c10)の方が sigma 大 → 外縁部に過密度構造
# < 0: r10 の方が sigma 大 → 中心部に集中
# |ap_diff| 大 → aperture 形状への強い依存性
for z in z_bins:
    sc = f'sgm{z}_c10'; sr = f'sgm{z}_r10'
    if sc not in df.columns or sr not in df.columns:
        continue
    diff = (df[sc] - df[sr]).values
    new_cols[f'ap_diff{z}']     = diff
    new_cols[f'ap_diff_abs{z}'] = np.abs(diff)
    # 比（c10/r10）：1より大きければ外縁優位
    denom = df[sr].replace(0, np.nan)
    new_cols[f'ap_ratio{z}'] = (df[sc] / denom).values

# ── eff_z 加重平均 ────────────────────────────────────────
eff_cols = [f'eff_z{z}' for z in z_bins if f'eff_z{z}' in df.columns]
eff_w    = df[eff_cols].values.copy().astype(float)
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0

def wt_avg(col_list):
    valid = [c for c in col_list if c in new_cols]
    if not valid: return np.full(len(df), np.nan)
    vals = np.column_stack([new_cols[c] for c in valid])
    w    = eff_w[:, :len(valid)]
    out  = np.average(vals.astype(float), weights=w, axis=1)
    out[zero_mask] = np.nan
    return out

new_cols['ap_diff_wt']     = wt_avg([f'ap_diff{z}'     for z in z_bins])
new_cols['ap_diff_abs_wt'] = wt_avg([f'ap_diff_abs{z}' for z in z_bins])
new_cols['ap_ratio_wt']    = wt_avg([f'ap_ratio{z}'    for z in z_bins])

# 高純度 bin（z=6,9）のみ
new_cols['ap_diff_pure']   = wt_avg(['ap_diff6','ap_diff9'])
new_cols['ap_ratio_pure']  = wt_avg(['ap_ratio6','ap_ratio9'])

# 外縁優位フラグ（ap_ratio > 1.1）
new_cols['ap_outer_flag'] = (new_cols['ap_ratio_wt'] > 1.1).astype(float)
new_cols['ap_inner_flag'] = (new_cols['ap_ratio_wt'] < 0.9).astype(float)

# z_dlt と los_grad も再計算（クロス相関用）
dlt_r10 = [f'dlt{z}_r10' for z in z_bins if f'dlt{z}_r10' in df.columns]
dlt_v   = df[dlt_r10].values.astype(float)
w2      = eff_w[:, :len(dlt_r10)]
z_dlt   = np.sum(dlt_v * w2, axis=1) / np.maximum(w2.sum(axis=1), 1e-9)
z_dlt[zero_mask] = np.nan
new_cols['z_dlt_r10'] = z_dlt

nd3 = df.get('nd3_r10'); nd9 = df.get('nd9_r10')
if nd3 is not None and nd9 is not None:
    total = (nd3 + nd9).replace(0, np.nan)
    new_cols['los_grad'] = ((nd9 - nd3) / total).values

# ── 出力 CSV ──────────────────────────────────────────────
out_df = pd.DataFrame({
    'ra': df['ra'], 'dec': df['dec'],
    'field': df['field'], 'skymap_id': df['skymap_id'],
    'eff_r10': df['eff_r10'],
})
for k, v in new_cols.items():
    out_df[k] = v

out_df.to_csv('sss2_aperture_diff.csv', index=False)
print(f"\n保存完了: sss2_aperture_diff.csv ({len(out_df)} 行, {len(out_df.columns)} 列)")

# ── 統計サマリー ──────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print("【z bin 別 aperture 差 sgm_c10 - sgm_r10】")
print(f"{'z bin':6s}  {'mean':8s}  {'std':8s}  {'min':8s}  {'max':8s}  {'>0 割合'}")
print("-"*58)
for z in z_bins:
    s = pd.Series(new_cols.get(f'ap_diff{z}',[])).dropna()
    if len(s):
        pos_frac = (s > 0).mean()
        print(f"z={z}     {s.mean():+8.4f}  {s.std():8.4f}  "
              f"{s.min():8.4f}  {s.max():8.4f}  {pos_frac:.3f}")

print()
print("【z bin 別 aperture 比 sgm_c10 / sgm_r10】")
print(f"{'z bin':6s}  {'mean':8s}  {'std':8s}  {'比>1.1':8s}  {'比<0.9':8s}")
print("-"*45)
for z in z_bins:
    s = pd.Series(new_cols.get(f'ap_ratio{z}',[])).dropna()
    if len(s):
        outer = (s > 1.1).mean(); inner = (s < 0.9).mean()
        print(f"z={z}     {s.mean():8.4f}  {s.std():8.4f}  "
              f"{outer:.3f}     {inner:.3f}")

print()
print("【加重平均 proxy の統計】")
for col in ['ap_diff_wt','ap_diff_abs_wt','ap_ratio_wt','ap_diff_pure','ap_ratio_pure']:
    s = pd.Series(new_cols.get(col,[])).dropna()
    if len(s):
        print(f"  {col:20s}: n={len(s)}, mean={s.mean():+.5f}, "
              f"std={s.std():.5f}, min={s.min():.5f}, max={s.max():.5f}")

print()
n_outer = int(pd.Series(new_cols['ap_outer_flag']).sum())
n_inner = int(pd.Series(new_cols['ap_inner_flag']).sum())
n_total = pd.Series(new_cols['ap_outer_flag']).notna().sum()
print(f"【外縁優位 / 内側優位の割合】")
print(f"  外縁優位（ratio > 1.1）: {n_outer:7d} ({n_outer/n_total*100:.1f}%)")
print(f"  内側優位（ratio < 0.9）: {n_inner:7d} ({n_inner/n_total*100:.1f}%)")
print(f"  中間（0.9〜1.1）       : {n_total-n_outer-n_inner:7d} "
      f"({(n_total-n_outer-n_inner)/n_total*100:.1f}%)")

print()
print("【クロス相関（Spearman r）】")
pairs = [
    ('ap_diff_wt',     'z_dlt_r10',  'aperture 差 vs 過密度'),
    ('ap_diff_abs_wt', 'z_dlt_r10',  '|aperture 差| vs 過密度'),
    ('ap_ratio_wt',    'z_dlt_r10',  'aperture 比 vs 過密度'),
    ('ap_diff_wt',     'los_grad',   'aperture 差 vs 視線勾配'),
    ('ap_diff_pure',   'ap_diff_wt', '高純度 vs 全体（一致確認）'),
]
for a, b, label in pairs:
    va = pd.Series(new_cols.get(a,[])); vb = pd.Series(new_cols.get(b,[]))
    mask = va.notna() & vb.notna()
    if mask.sum() < 100: continue
    r, p = spearmanr(va[mask], vb[mask])
    print(f"  {label}")
    print(f"    {a} vs {b}: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【field 別 ap_diff_wt】")
for fid in sorted(df['field'].dropna().unique()):
    s = pd.Series(new_cols['ap_diff_wt'])[df['field'].values == fid].dropna()
    pos = (s > 0).mean()
    print(f"  field={int(fid)}: mean={s.mean():+.5f}, std={s.std():.5f}, "
          f">0: {pos:.3f} ({pos*100:.1f}%)")