"""
視線方向勾配 (nd9-nd3)/(nd9+nd3) の抽出スクリプト
z=0.3 と z=0.9 の銀河数差 = 視線方向の n_fold 積分の proxy
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"   # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

# ── 視線方向勾配の計算 ────────────────────────────────────
# grad = (nd9 - nd3) / (nd9 + nd3)
# 値域: -1〜+1
# grad > 0: 遠方(z=0.9)の銀河が近傍(z=0.3)より多い → 遠方に構造
# grad < 0: 近傍(z=0.3)の銀河が遠方(z=0.9)より多い → 近傍に構造（今回の主成分）
# |grad| 大: 視線方向に強い非対称構造（膜の折り目が特定 z に集中）

new_cols = {}

for ap in ['r10', 'r30']:
    nd3 = f'nd3_{ap}'; nd9 = f'nd9_{ap}'
    if nd3 not in df.columns or nd9 not in df.columns:
        continue
    total = (df[nd3] + df[nd9]).replace(0, np.nan)
    new_cols[f'los_grad_{ap}']     = ((df[nd9] - df[nd3]) / total).values
    new_cols[f'los_grad_abs_{ap}'] = np.abs(new_cols[f'los_grad_{ap}'])

    # 近傍優位 vs 遠方優位の分類
    new_cols[f'los_near_{ap}'] = (new_cols[f'los_grad_{ap}'] < -0.2).astype(float)
    new_cols[f'los_far_{ap}']  = (new_cols[f'los_grad_{ap}'] >  0.2).astype(float)

# 中間 z ビン（z=5,6）との比較
for ap in ['r10']:
    nd5 = f'nd5_{ap}'; nd6 = f'nd6_{ap}'
    if nd5 in df.columns and nd6 in df.columns:
        # 中間 z の銀河数と端の差
        mid = (df[nd5] + df[nd6]) / 2
        ends = (df['nd3_r10'] + df['nd9_r10']) / 2
        total2 = (mid + ends).replace(0, np.nan)
        new_cols['los_curvature'] = ((ends - mid) / total2).values

# z bin ごとの eff_z 加重
eff_cols = [f'eff_z{z}' for z in '3456789' if f'eff_z{z}' in df.columns]
eff_w = df[eff_cols].values.copy().astype(float)
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0

# z_dlt も再計算（クロス相関用）
dlt_r10 = [f'dlt{z}_r10' for z in '3456789' if f'dlt{z}_r10' in df.columns]
dlt_v = df[dlt_r10].values.astype(float)
w2    = eff_w[:, :len(dlt_r10)]
z_dlt = np.sum(dlt_v * w2, axis=1) / np.maximum(w2.sum(axis=1), 1e-9)
z_dlt[zero_mask] = np.nan
new_cols['z_dlt_r10'] = z_dlt

# ── 出力 CSV ─────────────────────────────────────────────
out_df = pd.DataFrame({
    'ra': df['ra'], 'dec': df['dec'],
    'field': df['field'], 'skymap_id': df['skymap_id'],
    'eff_r10': df['eff_r10'],
    'nd3_r10': df['nd3_r10'], 'nd9_r10': df['nd9_r10'],
})
for k, v in new_cols.items():
    out_df[k] = v

out_df.to_csv('sss2_los_gradient.csv', index=False)
print(f"\n保存完了: sss2_los_gradient.csv ({len(out_df)} 行, {len(out_df.columns)} 列)")

# ── 統計サマリー ──────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print("【視線方向勾配 los_grad_r10 = (nd9-nd3)/(nd9+nd3) の基本統計】")
for col in ['los_grad_r10','los_grad_r30','los_grad_abs_r10']:
    s = pd.Series(new_cols.get(col,[])).dropna()
    if len(s):
        print(f"  {col}: n={len(s)}, mean={s.mean():.5f}, std={s.std():.5f}, "
              f"min={s.min():.5f}, max={s.max():.5f}")

print()
print("【近傍優位 / 遠方優位 / 中間の割合（|grad|>0.2 を閾値）】")
g = pd.Series(new_cols.get('los_grad_r10',[])).dropna()
n_near = (g < -0.2).sum(); n_far = (g > 0.2).sum()
n_mid  = ((g >= -0.2) & (g <= 0.2)).sum()
print(f"  近傍優位（grad < -0.2）: {n_near:7d} ({n_near/len(g)*100:.1f}%)")
print(f"  中間（|grad| ≦ 0.2）  : {n_mid:7d} ({n_mid/len(g)*100:.1f}%)")
print(f"  遠方優位（grad > +0.2）: {n_far:7d} ({n_far/len(g)*100:.1f}%)")

print()
print("【分布の分位数（構造の非対称性の把握）】")
for q in [0.05,0.10,0.25,0.50,0.75,0.90,0.95]:
    print(f"  {int(q*100):3d}%ile: {g.quantile(q):.5f}")

print()
print("【los_grad vs z_dlt の相関】")
zd = pd.Series(new_cols['z_dlt_r10'])
for col in ['los_grad_r10','los_grad_abs_r10']:
    cv = pd.Series(new_cols.get(col,[]))
    mask = cv.notna() & zd.notna()
    r, p = spearmanr(cv[mask], zd[mask])
    print(f"  {col} vs z_dlt_r10: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【los_grad vs eff_r10 の相関（感度依存性）】")
for col in ['los_grad_r10','los_grad_abs_r10']:
    cv = pd.Series(new_cols.get(col,[]))
    eff = df['eff_r10']
    mask = cv.notna() & eff.notna()
    r, p = spearmanr(cv[mask], eff[mask])
    print(f"  {col} vs eff_r10: r={r:.4f}, p={p:.2e}")

print()
print("【los_grad_r10 vs los_grad_r30 の一致確認（aperture 依存性）】")
v1 = pd.Series(new_cols.get('los_grad_r10',[]))
v2 = pd.Series(new_cols.get('los_grad_r30',[]))
mask = v1.notna() & v2.notna()
r, p = spearmanr(v1[mask], v2[mask])
print(f"  r10 vs r30: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【field 別 los_grad_r10 統計】")
for fid in sorted(df['field'].dropna().unique()):
    s = pd.Series(new_cols['los_grad_r10'])[df['field'].values == fid]
    s = s.dropna()
    n_near_f = (s < -0.2).sum()
    print(f"  field={int(fid)}: mean={s.mean():+.5f}, std={s.std():.5f}, "
          f"near:{n_near_f}({n_near_f/len(s)*100:.1f}%)")

print()
print("【T-7 との接続：z_eff の視線積分への寄与】")
print("  los_grad ≈ (nd_far - nd_near)/(nd_far + nd_near)")
print("  これは視線方向の銀河数の非対称性")
print("  T-7 の z_eff = α × n_fold × ξ²")
print("  n_fold は視線方向に積分されるため、")
print("  |los_grad| 大 → 特定 z に折り目が集中 → z_eff が大きい可能性")