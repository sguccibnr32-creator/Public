"""
集中度 nd_r30/nd_r10 の抽出スクリプト
銀河団・密集領域の検出に使う
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"    # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

z_bins = list('3456789')

# ── 集中度の計算 ──────────────────────────────────────────
# conc{z} = nd{z}_r30 / nd{z}_r10
# 大きいほど銀河が中心に集中（銀河団・密集領域）
print("集中度を計算中...")
new_cols = {}

for z in z_bins:
    r10 = f'nd{z}_r10'
    r30 = f'nd{z}_r30'
    if r10 not in df.columns or r30 not in df.columns:
        continue
    denom = df[r10].replace(0, np.nan)
    new_cols[f'conc{z}'] = (df[r30] / denom).values

# ── eff_z 加重平均 ─────────────────────────────────────────
eff_cols = [f'eff_z{z}' for z in z_bins if f'eff_z{z}' in df.columns]
eff_w    = df[eff_cols].values.copy().astype(float)
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0

def wt_avg(col_list):
    valid = [c for c in col_list if c in new_cols]
    if not valid:
        return np.full(len(df), np.nan)
    vals  = np.column_stack([new_cols[c] for c in valid])
    w     = eff_w[:, :len(valid)]
    out   = np.average(vals.astype(float), weights=w, axis=1)
    out[zero_mask] = np.nan
    return out

new_cols['conc_wt']   = wt_avg([f'conc{z}' for z in z_bins])

# 高純度 bin（z=6,9）の集中度
new_cols['conc_pure'] = wt_avg(['conc6', 'conc9'])

# 集中度の z 方向勾配（z=9 vs z=3：遠方 vs 近傍）
if 'conc3' in new_cols and 'conc9' in new_cols:
    c3 = new_cols['conc3']; c9 = new_cols['conc9']
    denom = (np.abs(c3) + np.abs(c9) + 1e-6)
    new_cols['conc_grad'] = (c9 - c3) / denom

# z_dlt も同時計算（クロス相関用）
dlt_r10 = [f'dlt{z}_r10' for z in z_bins if f'dlt{z}_r10' in df.columns]
dlt_v   = df[dlt_r10].values.astype(float)
w2      = eff_w[:, :len(dlt_r10)]
z_dlt   = np.sum(dlt_v * w2, axis=1) / np.maximum(w2.sum(axis=1), 1e-9)
z_dlt[zero_mask] = np.nan
new_cols['z_dlt_r10'] = z_dlt

# ── 高集中度ピクセルの抽出（銀河団候補）──────────────────
conc_wt = pd.Series(new_cols['conc_wt'])
thresh_3sig = conc_wt.mean() + 3 * conc_wt.std()
thresh_5sig = conc_wt.mean() + 5 * conc_wt.std()
high_mask = conc_wt >= thresh_3sig

# ── 出力 CSV ─────────────────────────────────────────────
out_df = pd.DataFrame({
    'ra':        df['ra'],
    'dec':       df['dec'],
    'field':     df['field'],
    'skymap_id': df['skymap_id'],
    'eff_r10':   df['eff_r10'],
})
for k, v in new_cols.items():
    out_df[k] = v

out_df.to_csv('sss2_concentration.csv', index=False)
print(f"\n保存完了: sss2_concentration.csv ({len(out_df)} 行, {len(out_df.columns)} 列)")

# 銀河団候補を別ファイルに保存
candidates = out_df[high_mask].copy()
candidates.to_csv('sss2_cluster_candidates.csv', index=False)
print(f"銀河団候補: sss2_cluster_candidates.csv ({len(candidates)} 行)")

# ── 統計サマリー ──────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print("【z bin 別 集中度 conc{z} = nd_r30/nd_r10】")
print(f"{'z bin':6s}  {'mean':7s}  {'std':7s}  {'min':7s}  {'max':7s}")
print("-"*42)
for z in z_bins:
    s = pd.Series(new_cols.get(f'conc{z}', [])).dropna()
    if len(s):
        print(f"z={z}     {s.mean():.4f}   {s.std():.4f}   "
              f"{s.min():.4f}   {s.max():.4f}")

print()
print("【加重平均 集中度 conc_wt】")
for col in ['conc_wt', 'conc_pure', 'conc_grad']:
    s = pd.Series(new_cols.get(col, [])).dropna()
    if len(s):
        print(f"{col:15s}: n={len(s)}, mean={s.mean():.5f}, "
              f"std={s.std():.5f}, min={s.min():.5f}, max={s.max():.5f}")

print()
cw = pd.Series(new_cols['conc_wt']).dropna()
thresh3 = cw.mean() + 3*cw.std()
thresh5 = cw.mean() + 5*cw.std()
print(f"【高集中度ピクセル（銀河団候補）】")
print(f"  3σ 閾値 = {thresh3:.4f}: {(cw >= thresh3).sum()} ピクセル")
print(f"  5σ 閾値 = {thresh5:.4f}: {(cw >= thresh5).sum()} ピクセル")

print()
print("【集中度 vs z_dlt の相関】")
for col in ['conc_wt', 'conc_pure']:
    cv = pd.Series(new_cols[col])
    zd = pd.Series(new_cols['z_dlt_r10'])
    mask = cv.notna() & zd.notna()
    r, p = spearmanr(cv[mask], zd[mask])
    print(f"  {col} vs z_dlt_r10: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【field 別 conc_wt】")
for fid in sorted(df['field'].dropna().unique()):
    s = pd.Series(new_cols['conc_wt'])[df['field'].values == fid].dropna()
    print(f"  field={int(fid)}: mean={s.mean():.4f}, std={s.std():.4f}, "
          f"n_3sig={int((s >= thresh3).sum())}")

print()
print("【銀河団候補（3σ以上）の上位10ピクセル】")
top = out_df[high_mask].nlargest(10, 'conc_wt')[
    ['ra','dec','field','conc_wt','z_dlt_r10']
]
print(top.to_string(index=False, float_format='%.4f'))