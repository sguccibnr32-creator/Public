#!/usr/bin/env python3
"""
HSC-SSP Wide デンシティマップ解析 + クラスター探索
"""
import numpy as np,pandas as pd,sys,os,io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
DENSITY_FILE=Path(r"E:\スバル望遠鏡データ\hscssp_pdr2_wide_densitymap.csv")
SHAPE_FILE=Path(os.path.join(base_dir,"931720.csv.gz.1"))

print("="*70)
print("HSC-SSP Wide デンシティマップ解析 + クラスター探索")
print("="*70)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] デンシティマップの構造確認")
print("="*70)

if not DENSITY_FILE.exists():
    print(f"  !!! ファイルなし: {DENSITY_FILE}"); sys.exit(1)

size_mb=DENSITY_FILE.stat().st_size/(1024*1024)
print(f"  ファイル: {DENSITY_FILE.name}")
print(f"  サイズ: {size_mb:.1f} MB")

# 先頭のみでカラム確認
df_density=pd.read_csv(DENSITY_FILE,nrows=100)
print(f"  カラム数: {len(df_density.columns)}")
print(f"\n  カラム一覧:")
print(f"  {'#':>3s} {'カラム名':<30s} {'型':<12s} {'min':>12s} {'max':>12s}")
print(f"  {'-'*72}")
for i,col in enumerate(df_density.columns):
    dtype=str(df_density[col].dtype)
    if pd.api.types.is_numeric_dtype(df_density[col]):
        mn=f"{df_density[col].min():.4g}"; mx=f"{df_density[col].max():.4g}"
    else:
        mn=str(df_density[col].iloc[0])[:10]; mx=str(df_density[col].iloc[-1])[:10]
    print(f"  {i:>3d} {col:<30s} {dtype:<12s} {mn:>12s} {mx:>12s}")

# RA/Dec/Density カラム特定
ra_col=None; dec_col=None; density_col=None
for c in df_density.columns:
    cl=c.lower()
    if 'ra' in cl and ra_col is None: ra_col=c
    if 'dec' in cl and dec_col is None: dec_col=c
    if 'dens' in cl or 'count' in cl or 'n_' in cl or 'ngal' in cl:
        density_col=c

if ra_col is None or dec_col is None:
    for c in df_density.columns:
        if pd.api.types.is_numeric_dtype(df_density[c]):
            vmin,vmax=df_density[c].min(),df_density[c].max()
            if 0<=vmin and vmax<=360 and vmax>100 and ra_col is None: ra_col=c
            elif -90<=vmin and vmax<=90 and abs(vmax-vmin)<50 and dec_col is None: dec_col=c

print(f"\n  特定カラム: RA={ra_col}, Dec={dec_col}, Density={density_col}")

# 全データ読み込み
print("\n  全データ読み込み中...")
df_density=pd.read_csv(DENSITY_FILE)
print(f"  行数: {len(df_density):,}")

if ra_col and dec_col:
    ra=df_density[ra_col].values; dec=df_density[dec_col].values
    print(f"  座標範囲: RA=[{ra.min():.3f}, {ra.max():.3f}], Dec=[{dec.min():.3f}, {dec.max():.3f}]")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] HSC-SSP Wide フィールド特定")
print("="*70)

hsc_fields={
    'XMM-LSS':  {'ra':(29,40),  'dec':(-7,-1)},
    'GAMA09H':  {'ra':(129,142),'dec':(-3,3)},
    'WIDE12H':  {'ra':(175,190),'dec':(-3,3)},
    'GAMA15H':  {'ra':(210,225),'dec':(-3,3)},
    'HECTOMAP': {'ra':(240,250),'dec':(42,45)},
    'VVDS':     {'ra':(333,345),'dec':(-2,3)},
}

if ra_col and dec_col:
    print(f"\n  {'フィールド':<12s} {'ピクセル数':>10s} {'RA範囲':>20s} {'Dec範囲':>20s}")
    print(f"  {'-'*65}")
    for fn,fs in hsc_fields.items():
        m=(ra>=fs['ra'][0])&(ra<=fs['ra'][1])&(dec>=fs['dec'][0])&(dec<=fs['dec'][1])
        n=int(m.sum())
        if n>0:
            print(f"  {fn:<12s} {n:>10,} [{ra[m].min():.1f},{ra[m].max():.1f}] [{dec[m].min():.1f},{dec[m].max():.1f}]")
        else:
            print(f"  {fn:<12s} {'---':>10s}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] 既知クラスターとのクロスマッチ")
print("="*70)

# Miyaoka + PSZ2 クラスター
clusters=[
    ("A68",9.278,9.157,0.255,"Miyaoka"),
    ("A209",22.969,-13.609,0.206,"Miyaoka"),
    ("A267",28.175,1.006,0.230,"Miyaoka"),
    ("A383",42.014,-3.529,0.187,"Miyaoka"),
    ("A521",73.529,-10.224,0.247,"Miyaoka"),
    ("A586",113.084,31.633,0.171,"Miyaoka"),
    ("A611",120.237,36.057,0.288,"Miyaoka"),
    ("A697",130.739,36.365,0.282,"Miyaoka"),
    ("A963",154.264,39.047,0.206,"Miyaoka"),
    ("A1689",197.873,-1.341,0.183,"Miyaoka"),
    ("A1835",210.259,2.878,0.253,"Miyaoka"),
    ("A2204",248.196,5.576,0.151,"Miyaoka"),
    ("A2261",260.613,32.133,0.224,"Miyaoka"),
    ("RXJ1347",206.878,-11.753,0.451,"Miyaoka"),
    ("MS1358",209.960,62.515,0.329,"Miyaoka"),
    ("ZW3146",155.916,4.186,0.291,"Miyaoka"),
    ("PSZ2_G186",135.0,-1.5,0.23,"PSZ2"),
    ("PSZ2_G212",185.0,-2.0,0.19,"PSZ2"),
]

if ra_col and dec_col:
    pix_deg=1.5/60.0  # 1.5 arcmin
    in_fp=[]; y3_cl=[]

    print(f"\n  {'名前':<15s} {'RA':>8s} {'Dec':>8s} {'z':>6s} {'HSC内':>6s} {'Y3内':>5s}")
    print(f"  {'-'*55}")

    for cl in clusters:
        name,ra_cl,dec_cl,z_cl,src=cl
        dist=np.sqrt(((ra-ra_cl)*np.cos(np.radians(dec_cl)))**2+(dec-dec_cl)**2)
        is_in=dist.min()<3*pix_deg
        is_y3=is_in and (-5.9<=dec_cl<=0)
        hsc_s="YES" if is_in else "---"
        y3_s="YES" if is_y3 else "---"
        print(f"  {name:<15s} {ra_cl:>8.3f} {dec_cl:>8.3f} {z_cl:>6.3f} {hsc_s:>6s} {y3_s:>5s}")
        if is_in: in_fp.append(cl)
        if is_y3: y3_cl.append(cl)

    print(f"\n  HSC-SSP内: {len(in_fp)}/{len(clusters)}")
    print(f"  Y3シェイプ重複: {len(y3_cl)}")

    # Y3 の Dec 範囲外だがHSC内のクラスター
    print(f"\n  HSC内だがY3外 (PDR2/PDR3シェイプが必要):")
    for cl in in_fp:
        if cl not in y3_cl:
            print(f"    {cl[0]}: Dec={cl[2]:.3f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] フットプリント可視化")
print("="*70)

if ra_col and dec_col:
    fig,axes=plt.subplots(1,2,figsize=(18,6))

    # (a) 全天
    ax=axes[0]
    ra_p=ra.copy(); ra_p[ra_p>180]-=360
    if density_col and density_col in df_density.columns:
        dens=df_density[density_col].values
        sc=ax.scatter(ra_p,dec,c=np.log10(np.maximum(dens,1)),s=0.1,alpha=0.3,
                     cmap='viridis',rasterized=True)
        plt.colorbar(sc,ax=ax,label='log10(density)',shrink=0.8)
    else:
        ax.scatter(ra_p,dec,s=0.1,alpha=0.3,c='steelblue',rasterized=True)

    for cl in clusters:
        name,ra_cl,dec_cl,z_cl,src=cl
        rp=ra_cl-360 if ra_cl>180 else ra_cl
        is_in=cl in in_fp
        col='red' if is_in else 'gray'
        mk='*' if is_in else 'x'; ms=15 if is_in else 8
        ax.plot(rp,dec_cl,mk,color=col,ms=ms,mew=1.5)
        if is_in:
            ax.annotate(name,(rp,dec_cl),fontsize=6,xytext=(3,3),
                       textcoords='offset points',color='red')

    ax.set_xlabel('RA [deg]'); ax.set_ylabel('Dec [deg]')
    ax.set_title(f'HSC-SSP Wide ({len(df_density):,} pixels)')
    ax.invert_xaxis()

    # (b) Y3領域拡大
    ax=axes[1]
    m_y3=(dec>=-7)&(dec<=1)
    if m_y3.sum()>0:
        if density_col and density_col in df_density.columns:
            sc2=ax.scatter(ra_p[m_y3],dec[m_y3],c=np.log10(np.maximum(dens[m_y3],1)),
                          s=1,alpha=0.5,cmap='viridis',rasterized=True)
        else:
            ax.scatter(ra_p[m_y3],dec[m_y3],s=1,alpha=0.5,c='steelblue',rasterized=True)

    ax.axhline(-5.9,color='red',ls='--',alpha=0.5,label='Y3 Dec range')
    ax.axhline(0,color='red',ls='--',alpha=0.5)

    for cl in y3_cl:
        name,ra_cl,dec_cl,z_cl,src=cl
        rp=ra_cl-360 if ra_cl>180 else ra_cl
        ax.plot(rp,dec_cl,'*',color='red',ms=15,mew=1.5)
        ax.annotate(name,(rp,dec_cl),fontsize=8,fontweight='bold',
                   xytext=(5,5),textcoords='offset points',color='red')

    ax.set_xlabel('RA [deg]'); ax.set_ylabel('Dec [deg]')
    ax.set_title('Y3 shape catalog overlap (Dec: -5.9 ~ 0)')
    ax.set_ylim(-7,2); ax.invert_xaxis(); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'hsc_footprint_clusters.png'),dpi=150)
    plt.close()
    print("  -> hsc_footprint_clusters.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] サマリー")
print("="*70)

print(f"""
  HSC-SSP Wide デンシティマップ:
    ピクセル数: {len(df_density):,}
    推定ピクセルサイズ: ~1.5 arcmin

  クラスター:
    全数: {len(clusters)}
    HSC-SSP内: {len(in_fp)}
    Y3シェイプ重複: {len(y3_cl)}
""")

if len(y3_cl)>0:
    print("  Y3で解析可能なクラスター:")
    for cl in y3_cl:
        print(f"    {cl[0]}: RA={cl[1]:.3f}, Dec={cl[2]:.3f}, z={cl[3]:.3f}")
    print(f"""
  次のステップ:
    1. Y3シェイプカタログからクラスター周辺天体を抽出
    2. 接線剪断プロファイル gamma_t(R) を測定
    3. NFW / 膜モデルとの比較""")
elif len(in_fp)>0:
    print("  HSC内クラスターはあるがY3 Dec範囲(-5.9~0)外")
    print("  -> PDR2/PDR3シェイプカタログの使用を検討")
    print(f"\n  代替案: 銀河-銀河レンズ")
    print(f"    1. SDSS spectroscopic銀河をHSC Y3フットプリントで選択")
    print(f"    2. G*Sigma0 ビン別 DeltaSigma(R) 測定")
    print(f"    3. g_c = eta*sqrt(a0*G*Sigma0) vs g_c=a0 比較")
else:
    print("  フットプリント内クラスターなし -> 銀河-銀河レンズへ")

print("\n完了。")
