"""
sss2 全体データから z_eff proxy を計算して結果を保存するスクリプト v3
重みゼロ行の対処を追加
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = 'C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv'   # ← 実際のファイル名に変更（拡張子も確認）

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

z_bins  = list('3456789')
eff_cols = [f'eff_z{z}' for z in z_bins if f'eff_z{z}' in df.columns]
dlt_r10  = [f'dlt{z}_r10' for z in z_bins if f'dlt{z}_r10' in df.columns]
dlt_r30  = [f'dlt{z}_r30' for z in z_bins if f'dlt{z}_r30' in df.columns]
sgm_r10  = [f'sgm{z}_r10' for z in z_bins if f'sgm{z}_r10' in df.columns]
dlt_c10  = [f'dlt{z}_c10' for z in z_bins if f'dlt{z}_c10' in df.columns]

eff_w = df[eff_cols].values.copy()

# 重みが全ゼロの行を特定 → 均等重みに置換（NaN を出さない）
zero_mask = eff_w.sum(axis=1) == 0
eff_w[zero_mask] = 1.0
n_zero = zero_mask.sum()
print(f"重みゼロ行: {n_zero} 行（均等重みに置換）")

def wt_avg(cols):
    w = eff_w[:, :len(cols)]
    vals = df[cols].values.astype(float)
    result = np.average(vals, weights=w, axis=1)
    result[zero_mask] = np.nan   # 重みゼロ行は NaN に
    return result

print("proxy 計算中...")
df['z_dlt_r10'] = wt_avg(dlt_r10)
df['z_dlt_r30'] = wt_avg(dlt_r30)
df['z_sgm_r10'] = wt_avg(sgm_r10)
if dlt_c10:
    df['z_dlt_c10'] = wt_avg(dlt_c10)

# Dec 系統除去（NaN を除いて回帰）
for col in ['z_dlt_r10','z_dlt_r30','z_sgm_r10']:
    valid = df[col].notna()
    coef  = np.polyfit(df.loc[valid,'dec'], df.loc[valid, col], 1)
    df[col+'_resid'] = df[col] - np.polyval(coef, df['dec'])

# 出力
out_cols = ['ra','dec','field','skymap_id','eff_r10','eff_r30',
            'z_dlt_r10','z_dlt_r30','z_sgm_r10',
            'z_dlt_r10_resid','z_dlt_r30_resid','z_sgm_r10_resid']
if 'z_dlt_c10' in df.columns:
    out_cols += ['z_dlt_c10']
out_cols = [c for c in out_cols if c in df.columns]

df[out_cols].to_csv('sss2_zeff_proxy.csv', index=False)
print(f"\n保存完了: sss2_zeff_proxy.csv ({len(df)} 行, {len(out_cols)} 列)")

# ── 統計サマリー（コピペ用）──────────────────────────────
print("\n=== 統計サマリー ===")
for col in ['z_dlt_r10','z_dlt_r10_resid','z_sgm_r10','z_sgm_r10_resid']:
    if col in df.columns:
        s = df[col].dropna()
        print(f"{col}: n={len(s)}, mean={s.mean():.5f}, std={s.std():.5f}, "
              f"min={s.min():.5f}, max={s.max():.5f}")

print("\n=== proxy 間相関 ===")
pairs = [('z_dlt_r10_resid','z_dlt_r30_resid'),
         ('z_dlt_r10_resid','z_sgm_r10_resid'),
         ('z_dlt_r10','z_dlt_r30')]
for a,b in pairs:
    if a in df.columns and b in df.columns:
        mask = df[a].notna() & df[b].notna()
        r,p  = spearmanr(df.loc[mask,a], df.loc[mask,b])
        print(f"{a} vs {b}: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print("\n=== Dec 系統 ===")
for col in ['z_dlt_r10','z_dlt_r10_resid']:
    if col in df.columns:
        mask = df[col].notna()
        r,_ = spearmanr(df.loc[mask,'dec'], df.loc[mask,col])
        print(f"{col} vs dec: r={r:.4f}")

print("\n=== field 別行数 ===")
if 'field' in df.columns:
    print(df['field'].value_counts().to_string())