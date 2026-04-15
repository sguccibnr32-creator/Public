#!/usr/bin/env python3
"""
HSC Y3 接線剪断プロファイル測定
9.6GB シェイプカタログをチャンク処理して全クラスターを同時抽出
"""
import numpy as np,pandas as pd,sys,os,io,time
from scipy import stats,integrate
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
SHAPE_FILE=Path(os.path.join(base_dir,"931720.csv.gz.1"))
CLUSTER_FILE=Path(os.path.join(base_dir,"hsc_cluster_candidates_y3.csv"))

# 解析パラメータ
EXTRACT_RADIUS_ARCMIN=30.0
R_MIN_ARCMIN=0.5; R_MAX_ARCMIN=25.0; N_BINS=12
CHUNKSIZE=1_000_000

# 宇宙論
H0=70.0; Om=0.3; OL=0.7; c_light=2.998e5
Z_CLUSTER=0.35  # 暫定

# 実カラム名（inspect結果から）
COL_RA='i_ra'; COL_DEC='i_dec'
COL_E1='i_hsmshaperegauss_e1'; COL_E2='i_hsmshaperegauss_e2'
COL_W='i_hsmshaperegauss_derived_weight'
COL_M='i_hsmshaperegauss_derived_shear_bias_m'
COL_C1='i_hsmshaperegauss_derived_shear_bias_c1'
COL_C2='i_hsmshaperegauss_derived_shear_bias_c2'
COL_ZBIN='hsc_y3_zbin'
COL_BMASK='b_mode_mask'

print("="*70)
print("HSC Y3 接線剪断プロファイル測定")
print("="*70)

# ====================================================================
# 宇宙論関数
# ====================================================================
def comoving_dist(z):
    f=lambda zz:1.0/np.sqrt(Om*(1+zz)**3+OL)
    d,_=integrate.quad(f,0,z)
    return c_light/H0*d

def ang_diam_dist(z):
    return comoving_dist(z)/(1+z)

def sigma_cr(z_l,z_s):
    D_l=ang_diam_dist(z_l); D_s=ang_diam_dist(z_s)
    D_ls=(comoving_dist(z_s)-comoving_dist(z_l))/(1+z_s)
    if D_ls<=0: return np.inf
    G_pc=4.301e-3; Mpc_pc=1e6
    return c_light**2/(4*np.pi*G_pc)*(D_s)/(D_l*D_ls)*1e-6  # Msun/pc^2

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] クラスター候補読み込み")
print("="*70)

if CLUSTER_FILE.exists():
    df_cl=pd.read_csv(CLUSTER_FILE)
    print(f"  候補数: {len(df_cl)}")
else:
    print("  CSVなし -> ハードコード上位10候補")
    df_cl=pd.DataFrame([
        {'ra':140.450,'dec':-0.251,'sgm_peak':6.4,'n_pixels':37},
        {'ra':335.403,'dec':-0.947,'sgm_peak':5.9,'n_pixels':22},
        {'ra':140.509,'dec':-0.349,'sgm_peak':5.8,'n_pixels':21},
        {'ra':140.503,'dec':-0.441,'sgm_peak':5.5,'n_pixels':17},
        {'ra':140.337,'dec':-0.249,'sgm_peak':5.4,'n_pixels':23},
        {'ra':342.433,'dec':-0.572,'sgm_peak':5.2,'n_pixels':26},
        {'ra':220.153,'dec':-1.691,'sgm_peak':5.2,'n_pixels':33},
        {'ra':138.943,'dec':-0.413,'sgm_peak':5.1,'n_pixels':26},
        {'ra':139.109,'dec':-0.444,'sgm_peak':5.1,'n_pixels':27},
        {'ra':182.638,'dec':-0.304,'sgm_peak':5.1,'n_pixels':30},
    ])

# 近接候補のマージ（10'以内）
merge_r=10.0/60.0
merged=[]; used=set()
for i in range(len(df_cl)):
    if i in used: continue
    group=[i]
    for j in range(i+1,len(df_cl)):
        if j in used: continue
        ri=df_cl.iloc[i]; rj=df_cl.iloc[j]
        dra=(ri['ra']-rj['ra'])*np.cos(np.radians(ri['dec']))
        ddec=ri['dec']-rj['dec']
        if np.sqrt(dra**2+ddec**2)<merge_r:
            group.append(j); used.add(j)
    used.add(i)
    sub=df_cl.iloc[group]; w=sub['sgm_peak'].values
    merged.append({
        'ra':np.average(sub['ra'].values,weights=w),
        'dec':np.average(sub['dec'].values,weights=w),
        'sgm_peak':sub['sgm_peak'].max(),
        'n_pixels':sub['n_pixels'].sum(),
        'n_merged':len(group),
    })

df_m=pd.DataFrame(merged).sort_values('sgm_peak',ascending=False).reset_index(drop=True)
NC=len(df_m)
print(f"  マージ: {len(df_cl)} -> {NC} 独立構造")
for i in range(min(NC,10)):
    r=df_m.iloc[i]
    print(f"    #{i+1}: RA={r['ra']:.3f}, Dec={r['dec']:.3f}, sgm={r['sgm_peak']:.1f}, merge={int(r['n_merged'])}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] Y3シェイプカタログ チャンク処理")
print("="*70)

ext_r_deg=EXTRACT_RADIUS_ARCMIN/60.0
cl_ra=df_m['ra'].values; cl_dec=df_m['dec'].values
cos_dec=np.cos(np.radians(cl_dec))

# 各クラスターのソース蓄積
src_lists=[[] for _ in range(NC)]

if not SHAPE_FILE.exists():
    print(f"  !!! ファイルなし: {SHAPE_FILE}")
    sys.exit(1)

print(f"  ファイル: {SHAPE_FILE.name} ({SHAPE_FILE.stat().st_size/1e9:.1f} GB)")
print(f"  クラスター数: {NC}, 抽出半径: {EXTRACT_RADIUS_ARCMIN}'")
print(f"  チャンクサイズ: {CHUNKSIZE:,}")

# 使用カラムのみ読み込み（メモリ節約）
use_cols=[COL_RA,COL_DEC,COL_E1,COL_E2,COL_W,COL_M,COL_C1,COL_C2,COL_ZBIN,COL_BMASK]

t0=time.time()
total_rows=0; total_ext=0

for chunk in pd.read_csv(SHAPE_FILE,chunksize=CHUNKSIZE,usecols=use_cols):
    total_rows+=len(chunk)

    # b_mode_mask == 1 のみ使用（品質フラグ）
    if COL_BMASK in chunk.columns:
        chunk=chunk[chunk[COL_BMASK]==1]

    src_ra=chunk[COL_RA].values
    src_dec=chunk[COL_DEC].values

    # 全クラスター同時マッチ
    for ic in range(NC):
        # 矩形プレフィルタ
        dra_raw=np.abs(src_ra-cl_ra[ic])
        dra_raw=np.minimum(dra_raw,360-dra_raw)  # RA wrap
        pre_mask=(dra_raw<ext_r_deg/cos_dec[ic]+0.01)&(np.abs(src_dec-cl_dec[ic])<ext_r_deg+0.01)
        if pre_mask.sum()==0: continue

        sub_ra=src_ra[pre_mask]; sub_dec=src_dec[pre_mask]
        dra=(sub_ra-cl_ra[ic])*cos_dec[ic]
        ddec=sub_dec-cl_dec[ic]
        dist2=dra**2+ddec**2
        circle=dist2<ext_r_deg**2
        if circle.sum()==0: continue

        extracted=chunk[pre_mask][circle].copy()
        extracted['_dist_deg']=np.sqrt(dist2[circle])
        extracted['_phi']=np.arctan2(ddec[circle],dra[circle])
        src_lists[ic].append(extracted)
        total_ext+=circle.sum()

    if total_rows%5_000_000==0:
        el=time.time()-t0
        pct=total_rows/35_800_000*100
        print(f"    {total_rows/1e6:.0f}M行 ({pct:.0f}%), 抽出: {total_ext:,}, {el:.0f}秒",flush=True)

el=time.time()-t0
print(f"\n  完了: {total_rows:,}行, 抽出: {total_ext:,}天体, {el:.0f}秒")

# 結合
for ic in range(NC):
    src_lists[ic]=pd.concat(src_lists[ic],ignore_index=True) if src_lists[ic] else pd.DataFrame()

n_per=[len(s) for s in src_lists]
print(f"\n  クラスター別ソース数:")
for ic in range(NC):
    print(f"    #{ic+1} (RA={cl_ra[ic]:.2f}, Dec={cl_dec[ic]:.2f}): {n_per[ic]:,} 天体")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] 接線剪断プロファイル測定")
print("="*70)

r_bins=np.logspace(np.log10(R_MIN_ARCMIN),np.log10(R_MAX_ARCMIN),N_BINS+1)
r_cen=np.sqrt(r_bins[:-1]*r_bins[1:])

D_A=ang_diam_dist(Z_CLUSTER)
arcmin_Mpc=D_A*np.pi/(180*60)
r_Mpc=r_cen*arcmin_Mpc

profiles=[]
for ic in range(NC):
    df_s=src_lists[ic]
    if len(df_s)<50:
        profiles.append(None); continue

    e1=df_s[COL_E1].values-df_s[COL_C1].values  # 加法的バイアス補正
    e2=df_s[COL_E2].values-df_s[COL_C2].values
    w=df_s[COL_W].values
    m=df_s[COL_M].values
    phi=df_s['_phi'].values
    dist_am=df_s['_dist_deg'].values*60.0

    # 接線・交差剪断
    gt=-(e1*np.cos(2*phi)+e2*np.sin(2*phi))
    gx= (e1*np.sin(2*phi)-e2*np.cos(2*phi))

    gt_p=np.full(N_BINS,np.nan); gx_p=np.full(N_BINS,np.nan)
    gt_e=np.full(N_BINS,np.nan); n_s=np.zeros(N_BINS,dtype=int)

    for ib in range(N_BINS):
        mk=(dist_am>=r_bins[ib])&(dist_am<r_bins[ib+1])
        n_s[ib]=mk.sum()
        if n_s[ib]<5: continue
        wb=w[mk]; ws=wb.sum(); mm=np.average(m[mk],weights=wb)
        gt_p[ib]=np.sum(wb*gt[mk])/ws/(1+mm)
        gx_p[ib]=np.sum(wb*gx[mk])/ws/(1+mm)
        sig=np.sqrt(np.sum(wb*gt[mk]**2)/ws-gt_p[ib]**2)
        gt_e[ib]=sig/np.sqrt(n_s[ib])

    v=~np.isnan(gt_p)&(gt_e>0)
    sn=np.sqrt(np.sum((gt_p[v]/gt_e[v])**2)) if v.sum()>0 else 0

    profiles.append({'r_am':r_cen,'r_Mpc':r_Mpc,'gt':gt_p,'gx':gx_p,
                     'gt_err':gt_e,'n_src':n_s,'sn':sn})
    if sn>1:
        print(f"  #{ic+1}: S/N={sn:.1f}, N={n_s.sum():,}, gt_max={np.nanmax(gt_p):.5f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] スタッキング")
print("="*70)

gt_st=np.zeros(N_BINS); gx_st=np.zeros(N_BINS)
gt_st_e=np.zeros(N_BINS); w_st=np.zeros(N_BINS)
n_st=np.zeros(N_BINS,dtype=int); n_used=0

for ic,pr in enumerate(profiles):
    if pr is None: continue
    v=~np.isnan(pr['gt'])&(pr['gt_err']>0)
    if v.sum()==0: continue
    n_used+=1
    for ib in range(N_BINS):
        if not v[ib]: continue
        iv=1.0/pr['gt_err'][ib]**2
        gt_st[ib]+=pr['gt'][ib]*iv
        gx_st[ib]+=pr['gx'][ib]*iv
        w_st[ib]+=iv; n_st[ib]+=1

for ib in range(N_BINS):
    if w_st[ib]>0:
        gt_st[ib]/=w_st[ib]; gx_st[ib]/=w_st[ib]
        gt_st_e[ib]=1.0/np.sqrt(w_st[ib])
    else:
        gt_st[ib]=gx_st[ib]=gt_st_e[ib]=np.nan

vs=~np.isnan(gt_st)&(gt_st_e>0)
sn_st=np.sqrt(np.sum((gt_st[vs]/gt_st_e[vs])**2)) if vs.sum()>0 else 0

print(f"  スタック使用: {n_used} クラスター")
print(f"  スタック S/N: {sn_st:.1f}")
if vs.sum()>0:
    print(f"  gamma_t 最大: {np.nanmax(gt_st):.5f}")
    print(f"  gamma_t 最内ビン: {gt_st[vs][0]:.5f} +/- {gt_st_e[vs][0]:.5f}")

# プロファイル表
print(f"\n  {'R[arcmin]':>10s} {'R[Mpc]':>8s} {'gamma_t':>10s} {'+/-':>10s} {'gamma_x':>10s} {'N_cl':>5s}")
print(f"  {'-'*56}")
for ib in range(N_BINS):
    if np.isnan(gt_st[ib]): continue
    print(f"  {r_cen[ib]:>10.2f} {r_Mpc[ib]:>8.3f} {gt_st[ib]:>10.5f} {gt_st_e[ib]:>10.5f} {gx_st[ib]:>10.5f} {n_st[ib]:>5d}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] プロット生成")
print("="*70)

fig,axes=plt.subplots(2,2,figsize=(14,12))

# (a) スタッキング
ax=axes[0,0]
vp=~np.isnan(gt_st)
if vp.sum()>0:
    ax.errorbar(r_cen[vp],gt_st[vp]*1e3,yerr=gt_st_e[vp]*1e3,
               fmt='o',color='steelblue',ms=6,capsize=3,label=r'$\gamma_t$ (stack)')
    ax.errorbar(r_cen[vp],gx_st[vp]*1e3,yerr=gt_st_e[vp]*1e3,
               fmt='s',color='coral',ms=4,capsize=2,alpha=0.5,label=r'$\gamma_\times$ (null)')
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]')
ax.set_ylabel(r'$\gamma \times 10^3$')
ax.set_title(f'Stacked shear (N={n_used}, S/N={sn_st:.1f})')
ax.legend(fontsize=9)

# (b) 個別S/N
ax=axes[0,1]
sn_vals=[p['sn'] if p else 0 for p in profiles]
ax.barh(range(len(sn_vals)),sn_vals,color='steelblue',alpha=0.7)
ax.axvline(3,color='red',ls='--',alpha=0.5,label='S/N=3')
ax.set_xlabel('Total S/N'); ax.set_ylabel('Cluster #')
ax.set_title('Individual S/N'); ax.legend(fontsize=8)

# (c) 上位個別
ax=axes[1,0]
cols_plt=['steelblue','coral','green','purple','orange']
plotted=0
for ic,pr in enumerate(profiles):
    if pr is None or pr['sn']<1 or plotted>=5: continue
    v=~np.isnan(pr['gt'])
    if v.sum()<3: continue
    ax.errorbar(pr['r_am'][v],pr['gt'][v]*1e3,yerr=pr['gt_err'][v]*1e3,
               fmt='o-',ms=4,capsize=2,color=cols_plt[plotted%5],
               label=f'#{ic+1} (S/N={pr["sn"]:.1f})',alpha=0.7)
    plotted+=1
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]')
ax.set_ylabel(r'$\gamma_t \times 10^3$')
ax.set_title('Top individual profiles'); ax.legend(fontsize=7)

# (d) ビン参加数
ax=axes[1,1]
ax.bar(range(N_BINS),n_st,color='steelblue',alpha=0.7)
ax.set_xticks(range(N_BINS))
ax.set_xticklabels([f'{r:.1f}' for r in r_cen],rotation=45,fontsize=7)
ax.set_xlabel('R [arcmin]'); ax.set_ylabel('N clusters')
ax.set_title('Bin participation')

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'hsc_y3_shear_profiles.png'),dpi=150)
plt.close()
print("  -> hsc_y3_shear_profiles.png 保存完了")

# 結果CSV保存
pd.DataFrame({
    'r_arcmin':r_cen,'r_Mpc':r_Mpc,
    'gamma_t':gt_st,'gamma_x':gx_st,'gamma_t_err':gt_st_e,'n_clusters':n_st
}).to_csv(os.path.join(base_dir,'hsc_y3_stacked_shear.csv'),index=False)

summary=[]
for ic in range(NC):
    row=df_m.iloc[ic].to_dict()
    row['sn_total']=profiles[ic]['sn'] if profiles[ic] else 0
    row['n_sources']=n_per[ic]
    summary.append(row)
pd.DataFrame(summary).to_csv(os.path.join(base_dir,'hsc_y3_cluster_shear_summary.csv'),index=False)
print("  -> hsc_y3_stacked_shear.csv, hsc_y3_cluster_shear_summary.csv 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[サマリー]")
print("="*70)

print(f"""
  接線剪断プロファイル測定結果:
    クラスター数: {NC} (マージ後)
    スタック使用: {n_used}
    スタック S/N: {sn_st:.1f}
    仮定 z_cluster: {Z_CLUSTER}
""")

if sn_st>5:
    print("  判定: S/N>5 強いシグナル検出。NFWフィット+膜モデル比較可能。")
elif sn_st>3:
    print("  判定: S/N>3 有意なシグナル。追加候補で改善可能。")
elif sn_st>0:
    print("  判定: S/N<3 弱い。銀河-銀河レンズ(Stage 2)検討。")
else:
    print("  判定: シグナルなし。データ確認要。")

print("\n完了。")
