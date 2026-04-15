#!/usr/bin/env python3
"""
HSC-SSP デンシティマップからのクラスター候補スクリーニング
"""
import numpy as np,pandas as pd,sys,os,io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
DENSITY_FILE=Path(r"E:\スバル望遠鏡データ\hscssp_pdr2_wide_densitymap.csv")
Y3_DEC_MIN=-5.9; Y3_DEC_MAX=0.0
SGM_THRESHOLD=4.0
MIN_PIXELS=3
GROUPING_RADIUS_ARCMIN=5.0

print("="*70)
print("HSC-SSP デンシティマップ クラスター候補スクリーニング")
print("="*70)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] データ読み込み")
print("="*70)

df=pd.read_csv(DENSITY_FILE)
print(f"  行数: {len(df):,}, カラム数: {len(df.columns)}")

ra_col='ra'; dec_col='dec'
if ra_col not in df.columns or dec_col not in df.columns:
    for c in df.columns:
        if 'ra' in c.lower() and ra_col not in df.columns: ra_col=c
        if 'dec' in c.lower() and dec_col not in df.columns: dec_col=c

ra_vals=df[ra_col].values.copy()
dec_vals=df[dec_col].values
ra_vals[ra_vals>360]-=360
print(f"  RA=[{ra_vals.min():.3f}, {ra_vals.max():.3f}], Dec=[{dec_vals.min():.3f}, {dec_vals.max():.3f}]")

sgm_cols=sorted([c for c in df.columns if c.startswith('sgm')])
dlt_cols=sorted([c for c in df.columns if c.startswith('dlt')])
nd_cols=sorted([c for c in df.columns if c.startswith('nd')])
print(f"  sgm: {len(sgm_cols)}, dlt: {len(dlt_cols)}, nd: {len(nd_cols)} カラム")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] 過剰密度カラム調査")
print("="*70)

print(f"\n  {'カラム':<20s} {'平均':>8s} {'std':>8s} {'max':>8s} {'N(>4s)':>8s} {'N(>3s)':>8s} {'適性':>6s}")
print(f"  {'-'*64}")

best_col=None; best_score=0
for c in sgm_cols:
    valid=df[c].notna()&np.isfinite(df[c])
    if valid.sum()==0: continue
    vals=df.loc[valid,c]
    n4=int((vals>4.0).sum()); n3=int((vals>3.0).sum())
    is_r10='r10' in c or 'c10' in c
    is_mid_z=any(f'{n}' in c for n in [3,4,5])
    suit=""
    if is_r10 and is_mid_z: suit="[**]"
    elif is_r10 or is_mid_z: suit="[*]"
    print(f"  {c:<20s} {vals.mean():>8.2f} {vals.std():>8.2f} {vals.max():>8.1f} {n4:>8d} {n3:>8d} {suit:>6s}")
    score=n3*(2 if is_r10 else 1)*(2 if is_mid_z else 1)
    if score>best_score:
        best_score=score; best_col=c

# 全 sgm の最大値も計算
sgm_max=df[sgm_cols].max(axis=1).fillna(0).values
n4_max=int((sgm_max>4.0).sum()); n3_max=int((sgm_max>3.0).sum())
print(f"  {'sgm_max (全カラム)':<20s} {'':>8s} {'':>8s} {sgm_max.max():>8.1f} {n4_max:>8d} {n3_max:>8d}")

print(f"\n  選択: {best_col}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] ピーク検出")
print("="*70)

# best_col と sgm_max の両方を試す
for label,sgm_use in [(best_col,df[best_col].fillna(0).values),
                       ('sgm_max',sgm_max)]:
    for thresh in [4.0,3.5,3.0,2.5]:
        mask=sgm_use>thresh
        n_pk=int(mask.sum())
        if n_pk>=5:
            print(f"  {label}, 閾値={thresh}sigma: {n_pk} ピクセル")
            if label==best_col and thresh<=SGM_THRESHOLD:
                break
    else:
        continue
    break

# 最終的に使うもの: best_col で十分なら best_col、なければ sgm_max
sgm_use=df[best_col].fillna(0).values
for thresh in [4.0,3.5,3.0,2.5]:
    mask_peak=sgm_use>thresh
    if mask_peak.sum()>=5:
        SGM_THRESHOLD=thresh; break
else:
    sgm_use=sgm_max
    for thresh in [4.0,3.5,3.0,2.5]:
        mask_peak=sgm_use>thresh
        if mask_peak.sum()>=5:
            SGM_THRESHOLD=thresh; best_col='sgm_max'; break

peak_idx=np.where(mask_peak)[0]
print(f"\n  最終: カラム={best_col}, 閾値={SGM_THRESHOLD}sigma, ピクセル={len(peak_idx)}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] ピクセルグルーピング（クラスター同定）")
print("="*70)

if len(peak_idx)>0:
    p_ra=ra_vals[peak_idx]; p_dec=dec_vals[peak_idx]; p_sgm=sgm_use[peak_idx]
    group_r=GROUPING_RADIUS_ARCMIN/60.0
    visited=np.zeros(len(peak_idx),dtype=bool)
    clusters=[]
    order=np.argsort(-p_sgm)

    for seed in order:
        if visited[seed]: continue
        dra=(p_ra-p_ra[seed])*np.cos(np.radians(p_dec[seed]))
        ddec=p_dec-p_dec[seed]
        dist=np.sqrt(dra**2+ddec**2)
        members=np.where((dist<group_r)&(~visited))[0]

        if len(members)>=MIN_PIXELS:
            w=p_sgm[members]
            clusters.append({
                'ra':np.average(p_ra[members],weights=w),
                'dec':np.average(p_dec[members],weights=w),
                'sgm_peak':p_sgm[members].max(),
                'sgm_mean':np.average(p_sgm[members],weights=w),
                'n_pixels':len(members),
                'radius_arcmin':dist[members].max()*60,
            })
            visited[members]=True
        else:
            visited[seed]=True

    print(f"  クラスター候補: {len(clusters)}")
    df_cl=pd.DataFrame(clusters).sort_values('sgm_peak',ascending=False).reset_index(drop=True)

    # 各候補の最寄りピクセルから z ビン別 sgm を取得
    for ci in range(min(len(df_cl),50)):
        rc=df_cl.loc[ci,'ra']; dc=df_cl.loc[ci,'dec']
        dra=(ra_vals-rc)*np.cos(np.radians(dc))
        ddec=dec_vals-dc
        dist=np.sqrt(dra**2+ddec**2)
        nearest=np.argmin(dist)

        z_sgm={}
        for sc in sgm_cols:
            val=df.iloc[nearest][sc]
            if pd.notna(val) and np.isfinite(val): z_sgm[sc]=val
        if z_sgm:
            bz=max(z_sgm,key=z_sgm.get)
            df_cl.loc[ci,'best_zbin']=bz
            df_cl.loc[ci,'best_zbin_sgm']=z_sgm[bz]

        # nd（銀河数）も取得
        for nc in nd_cols:
            val=df.iloc[nearest][nc]
            if pd.notna(val): df_cl.loc[ci,f'near_{nc}']=val
else:
    df_cl=pd.DataFrame()

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] Y3フィルタと品質スコア")
print("="*70)

if len(df_cl)>0:
    df_cl['in_y3']=(df_cl['dec']>=Y3_DEC_MIN)&(df_cl['dec']<=Y3_DEC_MAX)
    df_cl['quality']=df_cl['sgm_peak']*np.sqrt(df_cl['n_pixels'])
    n_y3=int(df_cl['in_y3'].sum())
    print(f"  全候補: {len(df_cl)}")
    print(f"  Y3重複 (Dec {Y3_DEC_MIN}~{Y3_DEC_MAX}): {n_y3}")

    print(f"\n  === 全候補 上位20 ===")
    print(f"  {'#':>3s} {'RA':>9s} {'Dec':>8s} {'sgm_pk':>7s} {'n_pix':>6s} {'qual':>7s} {'Y3':>3s} {'best_zbin':<20s}")
    print(f"  {'-'*68}")
    for i in range(min(20,len(df_cl))):
        r=df_cl.iloc[i]
        y3="Y" if r.get('in_y3',False) else ""
        zb=str(r.get('best_zbin',''))[:19]
        print(f"  {i+1:>3d} {r['ra']:>9.3f} {r['dec']:>8.3f} {r['sgm_peak']:>7.1f} {int(r['n_pixels']):>6d} {r['quality']:>7.1f} {y3:>3s} {zb:<20s}")

    df_y3=df_cl[df_cl['in_y3']].copy().reset_index(drop=True)
    if len(df_y3)>0:
        print(f"\n  === Y3重複候補 上位20 ===")
        print(f"  {'#':>3s} {'RA':>9s} {'Dec':>8s} {'sgm_pk':>7s} {'n_pix':>6s} {'qual':>7s} {'best_zbin':<20s}")
        print(f"  {'-'*62}")
        for i in range(min(20,len(df_y3))):
            r=df_y3.iloc[i]
            zb=str(r.get('best_zbin',''))[:19]
            print(f"  {i+1:>3d} {r['ra']:>9.3f} {r['dec']:>8.3f} {r['sgm_peak']:>7.1f} {int(r['n_pixels']):>6d} {r['quality']:>7.1f} {zb:<20s}")

        df_y3.to_csv(os.path.join(base_dir,'hsc_cluster_candidates_y3.csv'),index=False)
        print(f"\n  -> hsc_cluster_candidates_y3.csv ({len(df_y3)} 候補)")

    df_cl.to_csv(os.path.join(base_dir,'hsc_cluster_candidates_all.csv'),index=False)
    print(f"  -> hsc_cluster_candidates_all.csv ({len(df_cl)} 候補)")
else:
    df_y3=pd.DataFrame()

# ====================================================================
print(f"\n{'='*70}")
print("[Step 6] プロット生成")
print("="*70)

if len(df_cl)>0:
    fig,axes=plt.subplots(2,2,figsize=(16,12))

    # (a) 全天
    ax=axes[0,0]
    ax.scatter(ra_vals,dec_vals,s=0.05,alpha=0.1,c='lightgray',rasterized=True)
    sc=ax.scatter(df_cl['ra'],df_cl['dec'],c=df_cl['sgm_peak'],
                 s=df_cl['n_pixels']*5,cmap='hot_r',alpha=0.7,
                 edgecolors='black',linewidths=0.5,vmin=SGM_THRESHOLD)
    plt.colorbar(sc,ax=ax,label='Peak sigma',shrink=0.8)
    ax.axhline(Y3_DEC_MIN,color='blue',ls='--',alpha=0.5)
    ax.axhline(Y3_DEC_MAX,color='blue',ls='--',alpha=0.5)
    ax.set_xlabel('RA [deg]'); ax.set_ylabel('Dec [deg]')
    ax.set_title(f'Cluster candidates (N={len(df_cl)}, thr={SGM_THRESHOLD}s)')
    ax.invert_xaxis()

    # (b) シグマ分布
    ax=axes[0,1]
    ax.hist(df_cl['sgm_peak'],bins=30,color='steelblue',alpha=0.7,edgecolor='white')
    ax.axvline(SGM_THRESHOLD,color='red',ls='--',label=f'thr={SGM_THRESHOLD}')
    ax.set_xlabel('Peak sigma'); ax.set_ylabel('Count')
    ax.set_title('Peak sigma distribution'); ax.legend()

    # (c) Y3拡大
    ax=axes[1,0]
    m_y3=(dec_vals>=Y3_DEC_MIN-1)&(dec_vals<=Y3_DEC_MAX+1)
    if m_y3.sum()>0:
        ax.scatter(ra_vals[m_y3],dec_vals[m_y3],s=0.1,alpha=0.2,c='lightgray',rasterized=True)
    if len(df_y3)>0:
        sc2=ax.scatter(df_y3['ra'],df_y3['dec'],c=df_y3['sgm_peak'],
                      s=df_y3['n_pixels']*10,cmap='hot_r',alpha=0.8,
                      edgecolors='black',linewidths=1,vmin=SGM_THRESHOLD)
        plt.colorbar(sc2,ax=ax,label='Peak sigma',shrink=0.8)
        for i in range(min(10,len(df_y3))):
            r=df_y3.iloc[i]
            ax.annotate(f"#{i+1}",(r['ra'],r['dec']),fontsize=8,fontweight='bold',
                       color='red',xytext=(3,3),textcoords='offset points')
    ax.axhline(Y3_DEC_MIN,color='blue',ls='--',alpha=0.7,label='Y3 range')
    ax.axhline(Y3_DEC_MAX,color='blue',ls='--',alpha=0.7)
    ax.set_xlabel('RA [deg]'); ax.set_ylabel('Dec [deg]')
    ax.set_title(f'Y3 overlap (N={len(df_y3)})')
    ax.set_ylim(Y3_DEC_MIN-1,Y3_DEC_MAX+1); ax.invert_xaxis(); ax.legend(fontsize=8)

    # (d) 品質
    ax=axes[1,1]
    if len(df_y3)>0:
        ax.scatter(df_y3['n_pixels'],df_y3['sgm_peak'],s=50,c='coral',alpha=0.7,
                  edgecolors='black',label='Y3')
    not_y3=df_cl[~df_cl['in_y3']]
    if len(not_y3)>0:
        ax.scatter(not_y3['n_pixels'],not_y3['sgm_peak'],s=20,c='gray',alpha=0.3,label='Other')
    ax.set_xlabel('N pixels'); ax.set_ylabel('Peak sigma')
    ax.set_title('Candidate quality'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'hsc_cluster_screening.png'),dpi=150)
    plt.close()
    print("  -> hsc_cluster_screening.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[サマリー]")
print("="*70)

if len(df_cl)>0:
    print(f"""
  クラスター候補探索結果:
    使用カラム: {best_col}
    閾値: {SGM_THRESHOLD} sigma
    全候補: {len(df_cl)}
    Y3重複: {len(df_y3)}
""")
    if len(df_y3)>0:
        print("  Y3重複の上位候補:")
        for i in range(min(10,len(df_y3))):
            r=df_y3.iloc[i]
            print(f"    #{i+1}: RA={r['ra']:.3f}, Dec={r['dec']:.3f}, sgm={r['sgm_peak']:.1f}")
        print(f"""
  次のステップ:
    1. 上位候補の座標で Y3 シェイプカタログから天体抽出
    2. z ビン番号 -> 赤方偏移対応 (HSC Y3: bin1~4 -> z=0.3-1.5)
    3. 接線剪断プロファイル gamma_t(R) 測定
    4. NFW / 膜モデル比較""")
    else:
        print("  Y3重複候補なし -> 銀河-銀河レンズ (Stage 2) へ")
else:
    print("  候補なし。閾値またはデータ要確認。")

print("\n完了。")
