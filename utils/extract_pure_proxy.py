"""
z=6 と z=9 の高純度 bin だけで z_dlt を再計算するスクリプト
汚染を除いた純粋な n_fold proxy を抽出する
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

filename = "C:\\Users\\sgucc\\Downloads\\hscssp_pdr2_wide_densitymap.csv"    # ← 実際のファイル名に変更

print("データ読み込み中...")
df = pd.read_csv(filename)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

# ── 重み設定 ───────────────────────────────────────────────
eff_w_all = df[[f'eff_z{z}' for z in '3456789']].values.astype(float)
zero_mask = eff_w_all.sum(axis=1) == 0
eff_w_all[zero_mask] = 1.0

def wt_avg_dlt(z_list, ap='r10'):
    cols = [f'dlt{z}_{ap}' for z in z_list if f'dlt{z}_{ap}' in df.columns]
    z_idx = [int(z)-3 for z in z_list]  # eff_z の列インデックス（z3=0, z9=6）
    vals = df[cols].values.astype(float)
    w    = eff_w_all[:, z_idx]
    w_sum = w.sum(axis=1)
    out  = np.where(w_sum > 0,
                    np.sum(vals * w, axis=1) / w_sum,
                    np.nan)
    out[zero_mask] = np.nan
    return out

def wt_avg_sgm(z_list, ap='r10'):
    cols = [f'sgm{z}_{ap}' for z in z_list if f'sgm{z}_{ap}' in df.columns]
    z_idx = [int(z)-3 for z in z_list]
    vals = df[cols].values.astype(float)
    w    = eff_w_all[:, z_idx]
    w_sum = w.sum(axis=1)
    out  = np.where(w_sum > 0,
                    np.sum(vals * w, axis=1) / w_sum,
                    np.nan)
    out[zero_mask] = np.nan
    return out

print("proxy 計算中...")

# ── 1. 全 z bin（従来通り）────────────────────────────────
z_all  = list('3456789')
z_pure = ['6','9']       # 高純度 bin のみ
z_low  = ['5','7','8']   # 低純度 bin のみ
z_mid  = ['3','4']       # 中程度 bin

proxies = {
    # 従来
    'z_dlt_all_r10':   wt_avg_dlt(z_all,  'r10'),
    'z_dlt_all_r30':   wt_avg_dlt(z_all,  'r30'),
    # 高純度 bin のみ（z=6,9）
    'z_dlt_pure_r10':  wt_avg_dlt(z_pure, 'r10'),
    'z_dlt_pure_r30':  wt_avg_dlt(z_pure, 'r30'),
    # 低純度 bin のみ（z=5,7,8：比較用）
    'z_dlt_low_r10':   wt_avg_dlt(z_low,  'r10'),
    # 中程度 bin（z=3,4）
    'z_dlt_mid_r10':   wt_avg_dlt(z_mid,  'r10'),
    # sgm 版（pure）
    'z_sgm_pure_r10':  wt_avg_sgm(z_pure, 'r10'),
    'z_sgm_pure_r30':  wt_avg_sgm(z_pure, 'r30'),
    # 高純度 bin の c10 版
    'z_dlt_pure_c10':  wt_avg_dlt(z_pure, 'c10') if 'dlt6_c10' in df.columns else np.full(len(df), np.nan),
}

# ── 2. 出力 CSV ───────────────────────────────────────────
out_df = pd.DataFrame({
    'ra': df['ra'], 'dec': df['dec'],
    'field': df['field'], 'skymap_id': df['skymap_id'],
    'eff_r10': df['eff_r10'],
})
for k, v in proxies.items():
    out_df[k] = v

out_df.to_csv('sss2_pure_proxy.csv', index=False)
print(f"\n保存完了: sss2_pure_proxy.csv ({len(out_df)} 行, {len(out_df.columns)} 列)")

# ── 3. 統計サマリー ────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print("【各 proxy の基本統計】")
print(f"{'proxy':25s}  {'n':7s}  {'mean':8s}  {'std':8s}  {'min':8s}  {'max':8s}")
print("-"*72)
for k, v in proxies.items():
    s = pd.Series(v).dropna()
    print(f"{k:25s}  {len(s):7d}  {s.mean():8.5f}  {s.std():8.5f}  "
          f"{s.min():8.5f}  {s.max():8.5f}")

print()
print("【proxy 間の相関（Spearman r）】")
print("目的：高純度 proxy と全 z 平均の一致確認、低純度との分離確認")
print()
pairs = [
    ('z_dlt_pure_r10', 'z_dlt_all_r10',  '純粋 vs 全体（一致確認）'),
    ('z_dlt_pure_r10', 'z_dlt_pure_r30', '純粋 r10 vs r30（両 proxy 一致）'),
    ('z_dlt_pure_r10', 'z_sgm_pure_r10', '純粋 dlt vs sgm（定義の違い）'),
    ('z_dlt_pure_r10', 'z_dlt_low_r10',  '純粋 vs 低純度（汚染の影響）'),
    ('z_dlt_pure_r10', 'z_dlt_pure_c10', '純粋 r10 vs c10'),
]
for a, b, label in pairs:
    va = pd.Series(proxies[a]); vb = pd.Series(proxies[b])
    mask = va.notna() & vb.notna()
    if mask.sum() < 10:
        print(f"  {label}: データ不足")
        continue
    r, p = spearmanr(va[mask], vb[mask])
    print(f"  {label}")
    print(f"    {a} vs {b}: r={r:.4f}, p={p:.2e}, n={mask.sum()}")

print()
print("【aperture 間一致：全体 vs 高純度の比較】")
# 全体の r10 vs r30
va = pd.Series(proxies['z_dlt_all_r10'])
vb = pd.Series(proxies['z_dlt_all_r30'])
mask = va.notna() & vb.notna()
r_all, _ = spearmanr(va[mask], vb[mask])
# 高純度の r10 vs r30
va = pd.Series(proxies['z_dlt_pure_r10'])
vb = pd.Series(proxies['z_dlt_pure_r30'])
mask = va.notna() & vb.notna()
r_pure, _ = spearmanr(va[mask], vb[mask])
print(f"  全体（z=3〜9）r10 vs r30: r={r_all:.4f}")
print(f"  高純度（z=6,9）r10 vs r30: r={r_pure:.4f}")
print(f"  差分: Δr={r_pure-r_all:+.4f}")

print()
print("【field 別 std の比較（純度による変化）】")
for label, key in [('全体', 'z_dlt_all_r10'), ('高純度', 'z_dlt_pure_r10')]:
    print(f"  {label}:")
    for fid in sorted(df['field'].dropna().unique()):
        s = pd.Series(proxies[key])[df['field'].values == fid]
        s = s.dropna()
        print(f"    field={int(fid)}: std={s.std():.5f}, mean={s.mean():+.5f}")