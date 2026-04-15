#!/usr/bin/env python3
"""
HSC photo-z 分布解析 → クラスター赤方偏移の決定
"""
import numpy as np,pandas as pd,sys,os,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))

CLUSTERS=[
    ("cl1",140.45,-0.25,"GAMA09H #1"),
    ("cl2",142.21,-0.12,"GAMA09H #2"),
    ("cl3",140.30,-0.35,"GAMA09H #3"),
    ("cl4",182.68,-0.28,"WIDE12H"),
    ("cl5",216.75,-0.20,"GAMA15H"),
    ("cl6",335.40,-0.95,"VVDS #1"),
    ("cl7",342.43,-0.57,"VVDS #2"),
    ("cl8",139.00,-0.41,"GAMA09H #4"),
]

print("="*70)
print("HSC photo-z 分布解析 → クラスター赤方偏移決定")
print("="*70)

# データ読み込み
dfs={}
for name,ra,dec,label in CLUSTERS:
    fpath=os.path.join(base_dir,f"hsc_photoz_{name}.csv")
    if os.path.exists(fpath) and os.path.getsize(fpath)>1000:
        # ヘッダーが "# col1,col2,..." の形式なので # を除去して読む
        with open(fpath,'r',encoding='utf-8') as fh:
            first=fh.readline()
        if first.startswith('# '):
            # ヘッダーの # を除去
            df=pd.read_csv(fpath,header=0)
            df.columns=[c.replace('# ','').strip() for c in df.columns]
        else:
            df=pd.read_csv(fpath)
        if 'photoz_best' in df.columns and len(df)>50:
            dfs[name]=(df,ra,dec,label)
            print(f"  {name} ({label}): {len(df)} 行")

if not dfs:
    # jobファイルも試す
    for f in sorted(os.listdir(base_dir)):
        if f.startswith('hsc_photoz_job') and f.endswith('.csv'):
            df=pd.read_csv(os.path.join(base_dir,f),comment='#')
            if 'photoz_best' in df.columns and len(df)>100:
                jid=f.replace('hsc_photoz_job','').replace('.csv','')
                dfs[f'job{jid}']=(df,0,0,f'Job {jid}')
                print(f"  {f}: {len(df)} 行")

print(f"\n  有効データ: {len(dfs)} クラスター")

if len(dfs)==0:
    print("  データなし。終了。"); sys.exit(0)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] 各クラスターの photo-z 分布")
print("="*70)

z_bins=np.arange(0,2.01,0.02)
z_cen=0.5*(z_bins[:-1]+z_bins[1:])

cluster_z={}

for name,(df,ra,dec,label) in dfs.items():
    z=df['photoz_best'].dropna().values
    z=z[(z>0.01)&(z<2.0)]

    # ヒストグラム
    hist,_=np.histogram(z,bins=z_bins)

    # フィールド密度推定: 外側リング の銀河密度
    if 'ra' in df.columns and 'dec' in df.columns:
        dra=(df['ra'].values-ra)*np.cos(np.radians(dec))
        ddec=df['dec'].values-dec
        dist_am=np.sqrt(dra**2+ddec**2)*60  # arcmin
        r_max=np.percentile(dist_am,95)  # 実際の検索範囲
        r_mid=r_max*0.4  # 内側/外側の境界
        outer=(dist_am>r_mid)&(dist_am<=r_max)
        inner=(dist_am<=r_mid)
        area_inner=np.pi*r_mid**2
        area_outer=np.pi*(r_max**2-r_mid**2)

        z_inner=df.loc[inner,'photoz_best'].dropna().values
        z_outer=df.loc[outer,'photoz_best'].dropna().values
        z_inner=z_inner[(z_inner>0.01)&(z_inner<2.0)]
        z_outer=z_outer[(z_outer>0.01)&(z_outer<2.0)]

        h_inner,_=np.histogram(z_inner,bins=z_bins)
        h_outer,_=np.histogram(z_outer,bins=z_bins)

        # 面積補正した過剰密度
        bg=h_outer*area_inner/area_outer
        excess=h_inner-bg
    else:
        excess=hist.astype(float)

    # ピーク検出: 過剰密度の最大
    # 平滑化 (3ビン移動平均)
    kernel=np.ones(3)/3
    excess_smooth=np.convolve(excess,kernel,mode='same')

    peak_idx=np.argmax(excess_smooth)
    z_peak=z_cen[peak_idx]
    excess_peak=excess_smooth[peak_idx]

    # ピーク周辺 (±0.05) の重みつき平均
    mask_peak=np.abs(z_cen-z_peak)<0.05
    if excess_smooth[mask_peak].sum()>0:
        z_cluster=np.average(z_cen[mask_peak],weights=np.maximum(excess_smooth[mask_peak],0))
    else:
        z_cluster=z_peak

    # S/N: ピーク / バックグラウンドの標準偏差
    bg_std=np.std(excess_smooth[(z_cen<z_peak-0.2)|(z_cen>z_peak+0.2)])
    sn_z=excess_peak/bg_std if bg_std>0 else 0

    cluster_z[name]={'z':z_cluster,'z_peak':z_peak,'sn':sn_z,'n_total':len(z),
                     'excess_peak':excess_peak,'label':label}

    print(f"\n  {name} ({label}):")
    print(f"    全銀河: {len(z)}, z中央値: {np.median(z):.3f}")
    if 'z_inner' in dir():
        print(f"    内側(r<3'): {len(z_inner)}, 外側(r>3'): {len(z_outer)}")
    print(f"    過剰密度ピーク: z = {z_peak:.3f}")
    print(f"    重みつき z_cluster = {z_cluster:.3f}")
    print(f"    ピーク S/N = {sn_z:.1f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] クラスター赤方偏移サマリー")
print("="*70)

print(f"\n  {'名前':<8s} {'ラベル':<15s} {'z_cluster':>10s} {'S/N':>6s} {'N_gal':>7s}")
print(f"  {'-'*50}")
for name in sorted(cluster_z.keys()):
    cz=cluster_z[name]
    print(f"  {name:<8s} {cz['label']:<15s} {cz['z']:>10.3f} {cz['sn']:>6.1f} {cz['n_total']:>7d}")

# 全クラスターの平均赤方偏移
z_vals=[cz['z'] for cz in cluster_z.values() if cz['sn']>2]
if z_vals:
    z_mean=np.mean(z_vals)
    z_std=np.std(z_vals)
    print(f"\n  有意なクラスター (S/N>2) の z_cluster:")
    print(f"    平均 = {z_mean:.3f} +/- {z_std:.3f}")
    print(f"    範囲 = [{min(z_vals):.3f}, {max(z_vals):.3f}]")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] プロット")
print("="*70)

ncl=len(dfs)
ncols=min(4,ncl); nrows=(ncl+ncols-1)//ncols
fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,3.5*nrows),squeeze=False)

for idx,(name,(df,ra,dec,label)) in enumerate(dfs.items()):
    ax=axes[idx//ncols][idx%ncols]
    z=df['photoz_best'].dropna().values
    z=z[(z>0.01)&(z<2.0)]

    # 全体ヒストグラム
    ax.hist(z,bins=z_bins,alpha=0.5,color='steelblue',label='All')

    # 内側 (赤)
    if 'ra' in df.columns:
        dra=(df['ra'].values-ra)*np.cos(np.radians(dec))
        ddec=df['dec'].values-dec
        dist_am=np.sqrt(dra**2+ddec**2)*60
        dra2=(df['ra'].values-ra)*np.cos(np.radians(dec))
        ddec2=df['dec'].values-dec
        dist_am2=np.sqrt(dra2**2+ddec2**2)*60
        r_mid2=np.percentile(dist_am2,95)*0.4
        z_in=df.loc[dist_am2<=r_mid2,'photoz_best'].dropna().values
        z_in=z_in[(z_in>0.01)&(z_in<2.0)]
        if len(z_in)>5:
            ax.hist(z_in,bins=z_bins,alpha=0.7,color='coral',label="r<2'")

    cz=cluster_z.get(name,{})
    if cz:
        ax.axvline(cz['z'],color='red',ls='--',lw=2,label=f"z={cz['z']:.3f}")

    ax.set_xlabel('photo-z'); ax.set_ylabel('N')
    ax.set_title(f"{name} ({label})\nz={cz.get('z',0):.3f}, S/N={cz.get('sn',0):.1f}",fontsize=9)
    ax.legend(fontsize=7); ax.set_xlim(0,1.5)

# 余った軸を消す
for idx in range(len(dfs),nrows*ncols):
    axes[idx//ncols][idx%ncols].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'hsc_photoz_distributions.png'),dpi=150)
plt.close()
print("  -> hsc_photoz_distributions.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[結論]")
print("="*70)

if z_vals:
    print(f"""
  photo-z 解析結果:
    有効クラスター数: {len(z_vals)}
    推定赤方偏移: z = {z_mean:.3f} +/- {z_std:.3f}

  NFW/膜モデル比較の更新:
    現在の仮定: z_lens = 0.35
    推定値:     z_lens = {z_mean:.3f}
    -> hsc_nfw_membrane_compare.py の Z_L を更新して再実行
""")
else:
    print("  有意なクラスターなし")

print("完了。")
