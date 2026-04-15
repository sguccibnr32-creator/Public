#!/usr/bin/env python3
"""
N-2 Phase 2: LITTLE THINGS 回転曲線からの g_c 独立測定
Oh+2015 VizieR データ使用
"""
import numpy as np,pandas as pd,sys,os,io
from scipy import stats,optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
a0=1.2e-10; G_SI=6.674e-11; kms=1e3; kpc_m=3.086e19

# SPARC外の銀河リスト
SPARC_GALAXIES={"CVnIdwA","DDO43","DDO_43","DDO46","DDO_46","DDO47","DDO_47",
                "DDO50","DDO_50","DDO52","DDO_52","DDO87","DDO_87",
                "DDO126","DDO_126","DDO133","DDO_133","DDO154","DDO_154",
                "DDO168","DDO_168","F564-V3","IC1613","IC_1613",
                "NGC2366","NGC_2366","UGC8508","UGC_8508","WLM"}

print("="*70)
print("N-2 Phase 2: LITTLE THINGS g_c 独立測定")
print("="*70)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] データ読み込み")
print("="*70)

rot_file=os.path.join(base_dir,"littlethings_rotcurves","oh2015_rotdmbar.csv")
gal_file=os.path.join(base_dir,"littlethings_rotcurves","oh2015_galaxies.csv")

df_rot=pd.read_csv(rot_file)
df_gal=pd.read_csv(gal_file)

# 最初の行（単位行）を除去
df_rot=df_rot[df_rot['Name'].str.strip()!=''].copy()
df_rot=df_rot[~df_rot['R'].apply(lambda x: str(x).strip()=='' or 'kpc' in str(x))].copy()
df_gal=df_gal[df_gal['Name'].str.strip()!=''].copy()
df_gal=df_gal[~df_gal['Dist'].apply(lambda x: 'Mpc' in str(x))].copy()

# 数値変換
for c in ['R','V','e_V','R0.3','V0.3']:
    df_rot[c]=pd.to_numeric(df_rot[c],errors='coerce')

df_rot['Name']=df_rot['Name'].str.strip()
df_gal['Name']=df_gal['Name'].str.strip()

galaxies=df_rot['Name'].unique()
galaxies=[g for g in galaxies if g and len(g)>1]
print(f"  回転曲線: {len(df_rot)} 点, {len(galaxies)} 銀河")
print(f"  銀河一覧: {galaxies[:15]}")

# SPARC重複判定
for g in galaxies:
    g_clean=g.replace('_','').replace(' ','')
    is_sparc=any(g_clean.upper()==s.replace('_','').upper() for s in SPARC_GALAXIES)
    tag=" [SPARC]" if is_sparc else " [独立]"
    # この銀河のデータ点数
    n_pts=len(df_rot[df_rot['Name']==g])
    print(f"    {g:<12s}: {n_pts:>3d} 点{tag}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] g_c 測定（RAR フィット）")
print("="*70)

def fit_gc(R_kpc,V_obs,V_err,V_bar=None):
    """RAR フィットで g_c を測定"""
    R_m=R_kpc*kpc_m
    g_obs=(V_obs*kms)**2/R_m

    if V_bar is not None:
        g_N=(V_bar*kms)**2/R_m
    else:
        # rotdmbar の Type='Data' は total observed、'Model' はフィット
        # V_bar がない場合は MOND deep regime で推定
        # ここでは g_N << g_obs を仮定 (矮小銀河)
        g_N=0.1*g_obs  # 粗い近似

    valid=(g_N>0)&(g_obs>0)&np.isfinite(g_N)&np.isfinite(g_obs)&(R_kpc>0)
    if valid.sum()<3: return None,None,None,None

    go=g_obs[valid]; gn=g_N[valid]
    ge=2*V_obs[valid]*kms*np.maximum(V_err[valid],0.5)*kms/R_m[valid] if V_err is not None else 0.2*go
    ge=np.maximum(ge,0.1*go)

    def chi2(lg):
        gc=10**lg
        gm=0.5*(gn+np.sqrt(gn**2+4*gc*gn))
        return np.sum(((go-gm)/ge)**2)

    gc_g=np.logspace(-12,-8,100)
    c2=[chi2(np.log10(g)) for g in gc_g]
    bi=np.argmin(c2); gc_b=gc_g[bi]; c2_b=c2[bi]

    try:
        res=optimize.minimize_scalar(chi2,bounds=(np.log10(gc_b)-1,np.log10(gc_b)+1),method='bounded')
        if res.fun<c2_b: gc_b=10**res.x; c2_b=res.fun
    except: pass

    dc2=np.array(c2)-c2_b
    m68=dc2<1
    lo=gc_g[m68].min() if m68.sum()>0 else gc_b
    hi=gc_g[m68].max() if m68.sum()>0 else gc_b
    dof=valid.sum()-1
    return gc_b,(lo,hi),c2_b/max(dof,1),valid.sum()

results=[]
for gname in galaxies:
    if not gname or len(gname)<2: continue

    sub=df_rot[(df_rot['Name']==gname)&(df_rot['Type'].str.strip()=='Data')].copy()
    if len(sub)<5: continue

    R=sub['R'].values; V=sub['V'].values; eV=sub['e_V'].values
    valid=np.isfinite(R)&np.isfinite(V)&(R>0)&(V>0)
    if valid.sum()<5: continue
    R=R[valid]; V=V[valid]; eV=eV[valid]

    # rotdmbar にはバリオン回転速度が含まれていないので、
    # galaxies テーブルの Mgas, Mstar から推定
    gal_row=df_gal[df_gal['Name']==gname]
    if len(gal_row)>0:
        Mgas=pd.to_numeric(gal_row.iloc[0].get('Mgas','nan'),errors='coerce')*1e7  # 10^7 Msun
        Mstar=pd.to_numeric(gal_row.iloc[0].get('MstarK','nan'),errors='coerce')*1e7
        if np.isnan(Mstar):
            Mstar=pd.to_numeric(gal_row.iloc[0].get('MstarSED','nan'),errors='coerce')*1e7
        Rmax=pd.to_numeric(gal_row.iloc[0].get('Rmax','nan'),errors='coerce')
        Vmax=pd.to_numeric(gal_row.iloc[0].get('V(Rmax)','nan'),errors='coerce')
    else:
        Mgas=Mstar=Rmax=Vmax=np.nan

    # バリオン回転速度の推定（点質量近似）
    M_bar=(Mgas+Mstar)*1.989e30 if np.isfinite(Mgas) and np.isfinite(Mstar) else None
    if M_bar is not None and M_bar>0:
        R_m=R*kpc_m
        V_bar_est=np.sqrt(G_SI*M_bar*np.minimum(R/max(R)*1.0,1.0)/R_m)/kms  # 粗い近似
        V_bar_est=np.clip(V_bar_est,0.5,V.max())
    else:
        V_bar_est=None

    gc,gc_ci,c2dof,npts=fit_gc(R,V,eV,V_bar_est)
    if gc is None: continue

    # SPARC重複判定
    g_clean=gname.replace('_','').replace(' ','')
    is_sparc=any(g_clean.upper()==s.replace('_','').upper() for s in SPARC_GALAXIES)

    # G*Sigma0
    vf=V[-3:].mean() if len(V)>=3 else V[-1]  # 最外3点の平均
    hR=Rmax/4 if np.isfinite(Rmax) else R.max()/4  # 近似
    G_Sig0=(vf*kms)**2/(hR*kpc_m)
    gc_gm=np.sqrt(a0*G_Sig0)

    results.append({
        'galaxy':gname,'v_flat':vf,'h_R':hR,
        'gc':gc,'gc_lo':gc_ci[0],'gc_hi':gc_ci[1],
        'gc_a0':gc/a0,'gc_gm':gc_gm,'gc_gm_a0':gc_gm/a0,
        'G_Sig0':G_Sig0,'chi2dof':c2dof,'n_pts':npts,
        'in_sparc':is_sparc,'Mgas':Mgas/1e7 if np.isfinite(Mgas) else np.nan,
        'Mstar':Mstar/1e7 if np.isfinite(Mstar) else np.nan,
    })

df_r=pd.DataFrame(results)
df_indep=df_r[~df_r['in_sparc']].copy().reset_index(drop=True)
df_sparc=df_r[df_r['in_sparc']].copy().reset_index(drop=True)

print(f"\n  測定完了: {len(df_r)} 銀河 (SPARC内{len(df_sparc)}, 独立{len(df_indep)})")

print(f"\n  {'銀河':<12s} {'v_flat':>6s} {'gc/a0':>7s} {'gc_gm/a0':>9s} {'比率':>6s} {'chi2':>6s} {'N':>4s} {'SPARC':>6s}")
print(f"  {'-'*60}")
for _,row in df_r.iterrows():
    tag="Y" if row['in_sparc'] else ""
    ratio=row['gc']/row['gc_gm'] if row['gc_gm']>0 else np.nan
    print(f"  {row['galaxy']:<12s} {row['v_flat']:>6.0f} {row['gc_a0']:>7.3f} {row['gc_gm_a0']:>9.3f} {ratio:>6.2f} {row['chi2dof']:>6.2f} {int(row['n_pts']):>4d} {tag:>6s}")

# ====================================================================
if len(df_indep)>=3:
    print(f"\n{'='*70}")
    print("[Step 3] 幾何平均法則の独立検証")
    print("="*70)

    log_gc=np.log10(df_indep['gc'].values)
    log_GS=np.log10(df_indep['G_Sig0'].values)
    x=log_GS-np.log10(a0)
    y=log_gc-np.log10(a0)

    sl,ic,r,p,se=stats.linregress(x,y)
    print(f"  alpha(独立) = {sl:.3f} +/- {se:.3f}, r={r:.3f}, p={p:.4f}")

    # alpha=0.5 検定
    t05=(sl-0.5)/se; p05=2*stats.t.sf(abs(t05),len(x)-2)
    print(f"  alpha=0.5: t={t05:.2f}, p={p05:.4f} -> {'整合' if p05>0.05 else '不整合'}")

    # alpha=0.545 (SPARC)
    tsp=(sl-0.545)/se; psp=2*stats.t.sf(abs(tsp),len(x)-2)
    print(f"  alpha=0.545(SPARC): t={tsp:.2f}, p={psp:.4f}")

    # g_c/g_c_geomean 比較
    ratio=df_indep['gc'].values/df_indep['gc_gm'].values
    print(f"\n  gc(meas)/gc(geomean): med={np.median(ratio):.3f}, mean={np.mean(ratio):.3f}")

    # MOND比較
    gc_a0=df_indep['gc_a0'].values
    print(f"  gc/a0: med={np.median(gc_a0):.3f}, mean={np.mean(gc_a0):.3f}")

    # 残差比較
    r_gm=y-(0.5*x+np.median(y-0.5*x))
    r_mond=y-0
    print(f"\n  残差std: MOND={np.std(r_mond):.3f}, 幾何平均={np.std(r_gm):.3f}")
    imp=(1-np.std(r_gm)/np.std(r_mond))*100
    print(f"  改善: {imp:.1f}%")

# SPARC重複銀河の整合性
if len(df_sparc)>=3:
    print(f"\n  SPARC重複銀河の整合性:")
    x_s=np.log10(df_sparc['G_Sig0'].values)-np.log10(a0)
    y_s=np.log10(df_sparc['gc'].values)-np.log10(a0)
    sl_s,_,r_s,_,se_s=stats.linregress(x_s,y_s)
    print(f"    alpha(SPARC重複) = {sl_s:.3f} +/- {se_s:.3f}, r={r_s:.3f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] プロット")
print("="*70)

fig,axes=plt.subplots(1,3,figsize=(18,6))

# (a) gc vs G*Sigma0
ax=axes[0]
if len(df_indep)>0:
    ax.scatter(np.log10(df_indep['G_Sig0']/a0),np.log10(df_indep['gc']/a0),
              s=60,c='coral',edgecolors='black',zorder=5,label=f'Independent (N={len(df_indep)})')
if len(df_sparc)>0:
    ax.scatter(np.log10(df_sparc['G_Sig0']/a0),np.log10(df_sparc['gc']/a0),
              s=40,c='steelblue',edgecolors='black',alpha=0.5,label=f'SPARC overlap (N={len(df_sparc)})')
xr=np.linspace(-3,3,100)
ax.plot(xr,0.5*xr+np.median(y-0.5*x) if len(df_indep)>=3 else 0*xr,'g-',lw=2,label='alpha=0.5')
ax.axhline(0,color='blue',ls=':',alpha=0.5,label='MOND')
if len(df_indep)>=3:
    ax.plot(xr,sl*xr+ic,'r--',lw=2,label=f'fit: alpha={sl:.2f}')
ax.set_xlabel('log(G*Sigma0/a0)'); ax.set_ylabel('log(gc/a0)')
ax.set_title('(a) Geometric mean law'); ax.legend(fontsize=7)

# (b) gc(meas) vs gc(pred)
ax=axes[1]
if len(df_indep)>0:
    ax.scatter(np.log10(df_indep['gc_gm']/a0),np.log10(df_indep['gc']/a0),
              s=60,c='coral',edgecolors='black')
dg=np.linspace(-2,2,100); ax.plot(dg,dg,'k--',alpha=0.3)
ax.set_xlabel('log(gc predicted/a0)'); ax.set_ylabel('log(gc measured/a0)')
ax.set_title('(b) Predicted vs Measured')

# (c) 残差
ax=axes[2]
if len(df_indep)>=3:
    ax.hist(r_mond,bins=10,alpha=0.5,color='blue',label=f'MOND ({np.std(r_mond):.2f})')
    ax.hist(r_gm,bins=10,alpha=0.5,color='green',label=f'Geomean ({np.std(r_gm):.2f})')
    ax.set_title(f'(c) Residuals (improv: {imp:.0f}%)')
else:
    ax.set_title('(c) Insufficient data')
ax.set_xlabel('Residual [dex]'); ax.set_ylabel('Count'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'littlethings_verification.png'),dpi=150)
plt.close()

df_r.to_csv(os.path.join(base_dir,'littlethings_gc_results.csv'),index=False)
print("  -> littlethings_verification.png, littlethings_gc_results.csv 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[最終サマリー]")
print("="*70)
print(f"""
  LITTLE THINGS g_c 独立測定結果:
    全銀河: {len(df_r)}, SPARC重複: {len(df_sparc)}, 独立: {len(df_indep)}
""")
if len(df_indep)>=3:
    print(f"""    alpha(独立) = {sl:.3f} +/- {se:.3f}
    alpha=0.5 検定: p={p05:.4f} -> {'整合' if p05>0.05 else '不整合'}
    gc/a0 中央値(独立) = {np.median(gc_a0):.3f}
    残差改善(MOND比) = {imp:.1f}%
""")
print("完了。")
