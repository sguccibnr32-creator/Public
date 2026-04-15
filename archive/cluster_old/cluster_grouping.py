"""
5σ 候補（176 ピクセル）を銀河団としてグループ化し中心座標を確定
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

filename = 'sss2_concentration.csv'

print("データ読み込み中...")
df = pd.read_csv(filename).dropna(subset=['conc_wt','ra','dec'])

thresh5 = df['conc_wt'].mean() + 5 * df['conc_wt'].std()
thresh3 = df['conc_wt'].mean() + 3 * df['conc_wt'].std()
cands5  = df[df['conc_wt'] >= thresh5].copy().reset_index(drop=True)
print(f"5σ 候補ピクセル数: {len(cands5)}")
print(f"5σ 閾値: conc_wt > {thresh5:.4f}")

# ── グループ化（リンク半径 = 3 arcmin ≈ 0.05 deg）──────────
link_deg = 0.05   # 3 arcmin

coords = cands5[['ra','dec']].values
tree   = cKDTree(coords)
visited = np.zeros(len(cands5), dtype=bool)
groups  = []

for i in range(len(cands5)):
    if visited[i]: continue
    # このピクセルから link_deg 以内の全ピクセルを再帰的に収集
    members = set()
    queue   = [i]
    while queue:
        idx = queue.pop()
        if visited[idx]: continue
        visited[idx] = True
        members.add(idx)
        neighbors = tree.query_ball_point(coords[idx], link_deg)
        for nb in neighbors:
            if not visited[nb]:
                queue.append(nb)
    groups.append(list(members))

print(f"\nグループ数: {len(groups)}")

# ── 各グループの中心座標と統計 ────────────────────────────
results = []
for g in groups:
    sub = cands5.iloc[g]
    # conc_wt で重み付けした中心
    w    = sub['conc_wt'].values
    ra_c = np.average(sub['ra'].values,  weights=w)
    dc_c = np.average(sub['dec'].values, weights=w)
    results.append({
        'ra_center':   ra_c,
        'dec_center':  dc_c,
        'n_pixels':    len(g),
        'conc_max':    sub['conc_wt'].max(),
        'conc_mean':   sub['conc_wt'].mean(),
        'z_dlt_mean':  sub['z_dlt_r10'].mean() if 'z_dlt_r10' in sub.columns else np.nan,
        'field':       sub['field'].mode()[0] if 'field' in sub.columns else np.nan,
        'ra_min': sub['ra'].min(),  'ra_max': sub['ra'].max(),
        'dec_min': sub['dec'].min(),'dec_max': sub['dec'].max(),
    })

clusters = pd.DataFrame(results).sort_values('conc_max', ascending=False).reset_index(drop=True)
clusters.index += 1   # 1 始まり

# 単独ピクセル vs 複数ピクセルのクラスター
multi  = clusters[clusters['n_pixels'] >= 2]
single = clusters[clusters['n_pixels'] == 1]
print(f"複数ピクセル銀河団: {len(multi)}")
print(f"単独ピクセル:       {len(single)}")

# ── 出力 ─────────────────────────────────────────────────
clusters.to_csv('cluster_catalog.csv', index_label='cluster_id')
multi.to_csv('cluster_catalog_multi.csv', index_label='cluster_id')
print(f"\n保存: cluster_catalog.csv（全 {len(clusters)} クラスター）")
print(f"保存: cluster_catalog_multi.csv（複数ピクセル {len(multi)} クラスター）")

# ── 統計サマリー ──────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

print(f"【グループ化パラメータ】")
print(f"  リンク半径 = {link_deg} deg = {link_deg*60:.1f} arcmin")
print(f"  5σ 閾値   = {thresh5:.4f}")
print(f"  3σ 閾値   = {thresh3:.4f}")

print()
print(f"【クラスター統計】")
print(f"  総グループ数: {len(clusters)}")
print(f"  複数ピクセル: {len(multi)}  （2 ピクセル以上）")
print(f"  単独ピクセル: {len(single)}")
print(f"  最大サイズ:   {clusters['n_pixels'].max()} ピクセル")
print(f"  n_pixels 分布: {dict(clusters['n_pixels'].value_counts().sort_index())}")

print()
print(f"【上位20銀河団（conc_max の高い順）】")
print(f"{'ID':4s}  {'RA':9s}  {'Dec':8s}  {'n_pix':6s}  "
      f"{'conc_max':9s}  {'conc_mean':10s}  {'z_dlt':8s}  {'field':5s}")
print("-"*75)
for idx, row in clusters.head(20).iterrows():
    print(f"{idx:4d}  {row['ra_center']:9.4f}  {row['dec_center']:8.4f}  "
          f"{int(row['n_pixels']):6d}  {row['conc_max']:9.4f}  "
          f"{row['conc_mean']:10.4f}  {row['z_dlt_mean']:8.4f}  "
          f"{int(row['field']):5d}")

print()
print(f"【field 別クラスター数（複数ピクセル）】")
if len(multi):
    for fid in sorted(multi['field'].dropna().unique()):
        n = (multi['field'] == fid).sum()
        print(f"  field={int(fid)}: {n} クラスター")

print()
print(f"【Miyaoka 2018 とのクロスマッチ準備】")
print(f"  クラスター中心座標のうち RA=0〜60° または RA=130〜160° のもの:")
m18_range = multi[(multi['ra_center'] < 60) |
                  ((multi['ra_center'] > 130) & (multi['ra_center'] < 160))]
print(f"  {len(m18_range)} クラスター（XMM-LSS/COSMOS フィールド候補）")
if len(m18_range):
    for _, row in m18_range.iterrows():
        print(f"    RA={row['ra_center']:.3f}, Dec={row['dec_center']:.3f}, "
              f"conc={row['conc_max']:.3f}, field={int(row['field'])}")