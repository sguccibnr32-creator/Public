#!/usr/bin/env python3
"""
cl1 単体解析: プロファイル抽出 + Abel 変換 NFW/膜比較
cl1: RA=140.45, Dec=-0.25, z_spec=0.313, sigma_v=527 km/s
"""
import numpy as np,pandas as pd,sys,os,io,time
from scipy import optimize,integrate,interpolate
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))

SHAPE_FILE=Path(os.path.join(base_dir,"931720.csv.gz.1"))
CL1_RA=140.450; CL1_DEC=-0.251
Z_L=0.313; Z_S=1.0; SIGMA_V=527
EXT_R_AM=30.0; R_MIN=0.3; R_MAX=25.0; N_BINS=12; CHUNKSIZE=1_000_000

H0=70.0; Om=0.3; OL=0.7; c_light=2.998e5
G_Mpc=4.301e-9; G_SI=6.674e-11; Msun=1.989e30; Mpc_m=3.086e22; a0=1.2e-10

# 実カラム名
COL_RA='i_ra'; COL_DEC='i_dec'
COL_E1='i_hsmshaperegauss_e1'; COL_E2='i_hsmshaperegauss_e2'
COL_W='i_hsmshaperegauss_derived_weight'
COL_M='i_hsmshaperegauss_derived_shear_bias_m'
COL_C1='i_hsmshaperegauss_derived_shear_bias_c1'
COL_C2='i_hsmshaperegauss_derived_shear_bias_c2'
COL_BMASK='b_mode_mask'

print("="*70)
print("cl1 単体解析: 個別プロファイル + Abel 変換")
print("="*70)
print(f"  cl1: RA={CL1_RA}, Dec={CL1_DEC}, z_spec={Z_L}, sigma_v={SIGMA_V}")

# 宇宙論
def E(z): return np.sqrt(Om*(1+z)**3+OL)
def comoving(z):
    d,_=integrate.quad(lambda zz:c_light/(H0*E(zz)),0,z); return d
def D_A(z): return comoving(z)/(1+z)
def D_A2(z1,z2): return (comoving(z2)-comoving(z1))/(1+z2)
def rho_cr(z): return 3*(H0*E(z))**2/(8*np.pi*G_Mpc)
def Sigma_cr(zl,zs):
    Dl=D_A(zl); Ds=D_A(zs); Dls=D_A2(zl,zs)
    if Dls<=0: return np.inf
    return c_light**2/(4*np.pi*G_Mpc)*Ds/(Dl*Dls)

Scr=Sigma_cr(Z_L,Z_S); Dl=D_A(Z_L)
am2mpc=Dl*np.pi/(180*60)
h=H0/100; M200_sv=1e15/h*(SIGMA_V/1082.9)**(1/0.3361)
print(f"  D_A={Dl:.1f} Mpc, Sig_cr={Scr:.3e}, 1'={am2mpc*1000:.1f} kpc")
print(f"  M200(sigma_v) ~ {M200_sv:.2e} Msun")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] cl1 ソース銀河抽出")
print("="*70)

cl1_file=Path(os.path.join(base_dir,"cl1_sources.csv"))
ext_r_deg=EXT_R_AM/60.0; cos_dec=np.cos(np.radians(CL1_DEC))

if cl1_file.exists() and cl1_file.stat().st_size>10000:
    df_src=pd.read_csv(cl1_file)
    print(f"  既存: {cl1_file.name} ({len(df_src):,} 天体)")
else:
    if not SHAPE_FILE.exists():
        print(f"  !!! {SHAPE_FILE} なし"); sys.exit(1)
    print(f"  {SHAPE_FILE.name} からチャンク抽出中...")
    use_cols=[COL_RA,COL_DEC,COL_E1,COL_E2,COL_W,COL_M,COL_C1,COL_C2,COL_BMASK]
    t0=time.time(); chunks=[]; total=0
    for chunk in pd.read_csv(SHAPE_FILE,chunksize=CHUNKSIZE,usecols=use_cols):
        total+=len(chunk)
        if COL_BMASK in chunk.columns: chunk=chunk[chunk[COL_BMASK]==1]
        sra=chunk[COL_RA].values; sdec=chunk[COL_DEC].values
        dra=(sra-CL1_RA)*cos_dec; ddec=sdec-CL1_DEC
        d2=dra**2+ddec**2; m=d2<ext_r_deg**2
        if m.sum()>0:
            sub=chunk[m].copy()
            sub['_dist_deg']=np.sqrt(d2[m])
            sub['_phi']=np.arctan2(ddec[m],dra[m])
            chunks.append(sub)
        if total%5_000_000==0:
            ne=sum(len(c) for c in chunks)
            print(f"    {total/1e6:.0f}M行, 抽出:{ne}, {time.time()-t0:.0f}s")
    df_src=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
    print(f"  完了: {total:,}行, {len(df_src):,}天体, {time.time()-t0:.0f}s")
    df_src.to_csv(cl1_file,index=False)
    print(f"  -> {cl1_file.name}")

if len(df_src)==0: print("  ソースなし"); sys.exit(1)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] cl1 接線剪断プロファイル")
print("="*70)

e1=df_src[COL_E1].values-df_src[COL_C1].values
e2=df_src[COL_E2].values-df_src[COL_C2].values
w=df_src[COL_W].values; m_vals=df_src[COL_M].values
phi=df_src['_phi'].values; dist_am=df_src['_dist_deg'].values*60.0

gt_raw=-(e1*np.cos(2*phi)+e2*np.sin(2*phi))
gx_raw= (e1*np.sin(2*phi)-e2*np.cos(2*phi))

r_bins=np.logspace(np.log10(R_MIN),np.log10(R_MAX),N_BINS+1)
r_cen=np.sqrt(r_bins[:-1]*r_bins[1:])

gt_p=np.full(N_BINS,np.nan); gx_p=np.full(N_BINS,np.nan)
gt_e=np.full(N_BINS,np.nan); n_s=np.zeros(N_BINS,dtype=int)

for ib in range(N_BINS):
    mk=(dist_am>=r_bins[ib])&(dist_am<r_bins[ib+1]); n_s[ib]=mk.sum()
    if n_s[ib]<10: continue
    wb=w[mk]; ws=wb.sum(); mm=np.average(m_vals[mk],weights=wb)
    gt_p[ib]=np.sum(wb*gt_raw[mk])/ws/(1+mm)
    gx_p[ib]=np.sum(wb*gx_raw[mk])/ws/(1+mm)
    sig=np.sqrt(np.sum(wb*gt_raw[mk]**2)/ws-gt_p[ib]**2)
    gt_e[ib]=sig/np.sqrt(n_s[ib])

v=~np.isnan(gt_p)&(gt_e>0)
sn=np.sqrt(np.sum((gt_p[v]/gt_e[v])**2)) if v.sum()>0 else 0
gx_mean=np.nanmean(gx_p[v])

print(f"  ソース: {len(df_src):,}, 有効ビン: {v.sum()}/{N_BINS}")
print(f"  S/N: {sn:.1f}, gamma_x平均: {gx_mean:.5f} ({'PASS' if abs(gx_mean)<0.005 else '要注意'})")
print(f"\n  {'R[arcmin]':>10s} {'R[kpc]':>8s} {'gamma_t':>10s} {'err':>10s} {'gamma_x':>10s} {'N':>7s}")
print(f"  {'-'*58}")
for ib in range(N_BINS):
    if np.isnan(gt_p[ib]): continue
    print(f"  {r_cen[ib]:>10.2f} {r_cen[ib]*am2mpc*1000:>8.0f} {gt_p[ib]:>10.5f} {gt_e[ib]:>10.5f} {gx_p[ib]:>10.5f} {n_s[ib]:>7d}")

rf_=r_cen[v]; gf=gt_p[v]; ef=gt_e[v]; nd=len(rf_)

pd.DataFrame({'r_arcmin':r_cen,'r_kpc':r_cen*am2mpc*1000,
              'gamma_t':gt_p,'gamma_t_err':gt_e,'gamma_x':gx_p,'n_sources':n_s
}).to_csv(os.path.join(base_dir,'cl1_individual_shear.csv'),index=False)
print(f"  -> cl1_individual_shear.csv")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] NFW フィット")
print("="*70)

def nfw_params(M200,c200):
    rc=rho_cr(Z_L); r200=(3*M200/(4*np.pi*200*rc))**(1./3.); rs=r200/c200
    dc=200./3.*c200**3/(np.log(1+c200)-c200/(1+c200)); return rs,dc*rc,r200

def nfw_enclosed(r,M200,c200):
    rs,rho_s,_=nfw_params(M200,c200); x=r/rs
    return 4*np.pi*rho_s*rs**3*(np.log(1+x)-x/(1+x))

def nfw_gamma(R_am,M200,c200):
    rs,rho_s,_=nfw_params(M200,c200)
    R_Mpc=np.maximum(R_am*am2mpc,1e-6); x=R_Mpc/rs
    S=np.zeros_like(x); Sm=np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi<1e-6: continue
        elif abs(xi-1)<1e-4: S[i]=2*rs*rho_s/3; Sm[i]=4*rs*rho_s*(1+np.log(0.5))
        elif xi<1:
            sq=np.sqrt(1-xi**2)
            S[i]=2*rs*rho_s/(xi**2-1)*(1-np.log((1+sq)/xi)/sq)
            Sm[i]=4*rs*rho_s*(np.log(xi/2)+np.log((1+sq)/xi)/sq)/xi**2
        else:
            sq=np.sqrt(xi**2-1)
            S[i]=2*rs*rho_s/(xi**2-1)*(1-np.arctan(sq)/sq)
            Sm[i]=4*rs*rho_s*(np.log(xi/2)+np.arctan(sq)/sq)/xi**2
    return (Sm-S)/Scr

def c2n(p):
    M,c=10**p[0],10**p[1]
    if c<1 or c>20 or M<1e12 or M>1e16: return 1e20
    try: return np.sum(((gf-nfw_gamma(rf_,M,c))/ef)**2)
    except: return 1e20

bc=np.inf; bp=(14,0.5)
for lm in np.linspace(13,15.5,30):
    for lc in np.linspace(0,1.2,20):
        c2=c2n((lm,lc))
        if c2<bc: bc=c2; bp=(lm,lc)
try:
    res=optimize.minimize(c2n,bp,method='Nelder-Mead',options={'xatol':0.0005,'fatol':0.005})
    if res.fun<bc: bc=res.fun; bp=res.x
except: pass

M200=10**bp[0]; c200=10**bp[1]
rs_n,_,r200_n=nfw_params(M200,c200)
gt_nfw=nfw_gamma(rf_,M200,c200); c2_nfw=bc; dof_n=nd-2
print(f"  M200={M200:.2e}, c200={c200:.2f}, r200={r200_n*1000:.0f} kpc")
print(f"  chi2/dof={c2_nfw:.1f}/{dof_n}={c2_nfw/dof_n:.2f}")
print(f"  M200(lens)/M200(sig_v)={M200/M200_sv:.2f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] Abel 変換 MOND/膜")
print("="*70)

def g_mond_f(gN,gc):
    return 0.5*(gN+np.sqrt(gN**2+4*gc*gN)) if gN>0 else 0

def mond_rho(r_arr,M200,c200,gc,fb=0.17):
    n=len(r_arr); Me=np.zeros(n)
    for i,r in enumerate(r_arr):
        Mb=nfw_enclosed(r,M200,c200)*fb; rm=r*Mpc_m
        gN=G_SI*Mb*Msun/rm**2 if rm>1e10 else 0
        Me[i]=g_mond_f(gN,gc)*rm**2/G_SI/Msun
    rho=np.zeros(n)
    for i in range(1,n-1):
        dM=(Me[i+1]-Me[i-1])/(r_arr[i+1]-r_arr[i-1])
        rho[i]=max(dM/(4*np.pi*r_arr[i]**2),0)
    rho[0]=rho[1]; rho[-1]=rho[-2]
    return rho,Me

def abel_proj(R_arr,r_f,rho_f):
    lr=np.log10(r_f); lrho=np.log10(np.maximum(rho_f,1e-50))
    fi=interpolate.interp1d(lr,lrho,fill_value=-50,bounds_error=False)
    S=np.zeros(len(R_arr)); rmax=r_f.max()
    for i,R in enumerate(R_arr):
        if R<=0: continue
        um=np.sqrt(max(rmax**2-R**2,0))
        if um<=0: continue
        def f(u): return 10**fi(np.log10(np.sqrt(u**2+R**2)))
        try: val,_=integrate.quad(f,0,um,limit=200,epsrel=1e-4)
        except: val=0
        S[i]=2*val
    return S

def comp_DS(R_arr,Sig):
    sf=interpolate.interp1d(R_arr,Sig,fill_value=0,bounds_error=False)
    Sm=np.zeros(len(R_arr))
    for i,R in enumerate(R_arr):
        if R<=0: continue
        try: val,_=integrate.quad(lambda Rp:sf(Rp)*Rp,R_arr[0]*0.1,R,limit=100)
        except: val=0
        Sm[i]=2*val/R**2
    return Sm-Sig

def mond_gt_abel(R_am,M200,c200,gc,fb=0.17):
    R_Mpc=np.maximum(R_am*am2mpc,1e-4)
    r_f=np.logspace(np.log10(R_Mpc.min()*0.1),np.log10(R_Mpc.max()*5),400)
    rho,Me=mond_rho(r_f,M200,c200,gc,fb)
    Sig=abel_proj(R_Mpc,r_f,rho)
    DS=comp_DS(R_Mpc,Sig)
    return DS/Scr,rho,Me,r_f

gc_gr=np.logspace(-12,-8,80)
c2_gc=np.full(len(gc_gr),np.inf)
print(f"  g_c スキャン ({len(gc_gr)} 点)...")
t0=time.time()
for ig,gc in enumerate(gc_gr):
    try:
        gt_m,_,_,_=mond_gt_abel(rf_,M200,c200,gc)
        if np.all(np.isfinite(gt_m)): c2_gc[ig]=np.sum(((gf-gt_m)/ef)**2)
    except: pass
    if (ig+1)%20==0:
        bi=np.argmin(c2_gc[:ig+1])
        print(f"    {ig+1}/{len(gc_gr)}: best={gc_gr[bi]/a0:.3f}a0, chi2={c2_gc[bi]:.1f}, {time.time()-t0:.0f}s")

bi=np.argmin(c2_gc); gc_b=gc_gr[bi]; c2_b=c2_gc[bi]
ai=np.argmin(np.abs(gc_gr-a0)); c2_a0=c2_gc[ai]; dof_a=nd-2

gt_ab,rho_b,Me_b,r_fn=mond_gt_abel(rf_,M200,c200,gc_b)
gt_mo,_,_,_=mond_gt_abel(rf_,M200,c200,a0)

dc2=c2_gc-c2_b; m68=dc2<1; m95=dc2<4
lo68=gc_gr[m68].min()/a0 if m68.sum()>0 else gc_b/a0
hi68=gc_gr[m68].max()/a0 if m68.sum()>0 else gc_b/a0
lo95=gc_gr[m95].min()/a0 if m95.sum()>0 else gc_b/a0
hi95=gc_gr[m95].max()/a0 if m95.sum()>0 else gc_b/a0
a0_68=lo68<=1.0<=hi68; a0_95=lo95<=1.0<=hi95

print(f"\n  結果:")
print(f"    g_c = {gc_b/a0:.3f} a0, 68%:[{lo68:.3f},{hi68:.3f}], 95%:[{lo95:.3f},{hi95:.3f}]")
print(f"    a0 in 68%: {'YES' if a0_68 else 'NO'}, 95%: {'YES' if a0_95 else 'NO'}")
print(f"    chi2(best)/dof={c2_b:.1f}/{dof_a}={c2_b/dof_a:.2f}")
print(f"    chi2(a0)/dof={c2_a0:.1f}/{dof_a}={c2_a0/dof_a:.2f}")
print(f"    chi2(NFW)/dof={c2_nfw:.1f}/{dof_n}={c2_nfw/dof_n:.2f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] モデル比較")
print("="*70)

aic_n=c2_nfw+4; aic_mo=c2_a0+2; aic_me=c2_b+4
print(f"\n  {'モデル':<40s} {'chi2':>7s} {'dof':>4s} {'chi2/dof':>9s} {'AIC':>7s} {'dAIC':>7s}")
print(f"  {'-'*70}")
print(f"  {'NFW':<40s} {c2_nfw:>7.1f} {dof_n:>4d} {c2_nfw/dof_n:>9.2f} {aic_n:>7.1f} {0:>+7.1f}")
print(f"  {'MOND g_c=a0 (Abel)':<40s} {c2_a0:>7.1f} {dof_a:>4d} {c2_a0/dof_a:>9.2f} {aic_mo:>7.1f} {aic_mo-aic_n:>+7.1f}")
print(f"  {'膜 g_c={0:.3f}a0 (Abel)'.format(gc_b/a0):<40s} {c2_b:>7.1f} {dof_a:>4d} {c2_b/dof_a:>9.2f} {aic_me:>7.1f} {aic_me-aic_n:>+7.1f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 6] プロット")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(f'cl1 Individual (z={Z_L}, $\\sigma_v$={SIGMA_V}, S/N={sn:.1f})',fontsize=13,fontweight='bold')

ax=axes[0,0]
ax.errorbar(rf_,gf*1e3,yerr=ef*1e3,fmt='o',color='black',ms=6,capsize=3,label='cl1',zorder=5)
ax.plot(rf_,gt_nfw*1e3,'r-',lw=2,label=f'NFW (M={M200:.1e})')
ax.plot(rf_,gt_mo*1e3,'b--',lw=2,label='MOND Abel ($g_c=a_0$)')
ax.plot(rf_,gt_ab*1e3,'g-.',lw=2,label=f'Membrane ({gc_b/a0:.2f}$a_0$)')
gx_fit=gx_p[v]
ax.errorbar(rf_,gx_fit*1e3,yerr=ef*1e3,fmt='x',color='gray',ms=4,alpha=0.5,label='$\\gamma_\\times$')
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma \times 10^3$')
ax.set_title('(a) Shear profile'); ax.legend(fontsize=6)

ax=axes[0,1]
ax.errorbar(rf_,(gf-gt_nfw)/ef,fmt='o',color='red',ms=4,label='NFW')
ax.errorbar(rf_*1.03,(gf-gt_mo)/ef,fmt='s',color='blue',ms=4,label='MOND')
ax.errorbar(rf_*1.06,(gf-gt_ab)/ef,fmt='^',color='green',ms=4,label='Membrane')
ax.axhline(0,color='k',ls='--',alpha=0.3); ax.axhspan(-2,2,alpha=0.05,color='gray')
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'Resid/$\sigma$')
ax.set_title('(b) Residuals'); ax.legend(fontsize=7)

ax=axes[0,2]
fin=np.isfinite(dc2)
ax.plot(gc_gr[fin]/a0,dc2[fin],'b-',lw=2)
ax.axhline(1,color='orange',ls='--',alpha=0.5,label='68%')
ax.axhline(4,color='red',ls='--',alpha=0.5,label='95%')
ax.axvline(1.0,color='gray',ls=':',alpha=0.7,label='$a_0$')
ax.axvline(gc_b/a0,color='green',ls='-',lw=2,label=f'best={gc_b/a0:.2f}$a_0$')
ax.axvline(0.825,color='orange',ls='-.',alpha=0.5,label='SPARC')
ax.set_xscale('log'); ax.set_xlabel('$g_c/a_0$'); ax.set_ylabel(r'$\Delta\chi^2$')
ax.set_title('(c) $g_c$ constraint'); ax.legend(fontsize=6)
ax.set_ylim(0,min(30,dc2[fin].max() if fin.sum()>0 else 30))

ax=axes[1,0]
ax.bar(range(N_BINS),n_s,color='steelblue',alpha=0.7)
ax.set_xticks(range(N_BINS))
ax.set_xticklabels([f'{r:.1f}' for r in r_cen],rotation=45,fontsize=7)
ax.set_xlabel('R [arcmin]'); ax.set_ylabel('N sources')
ax.set_title(f'(d) Source count ({n_s.sum():,})')

ax=axes[1,1]
Mb=np.array([nfw_enclosed(r,M200,c200)*0.17 for r in r_fn])
Mn=np.array([nfw_enclosed(r,M200,c200) for r in r_fn])
ax.loglog(r_fn*1000,Me_b,'g-',lw=2,label='$M_{eff}$')
ax.loglog(r_fn*1000,Mb,'b--',lw=2,label='$M_{bar}$')
ax.loglog(r_fn*1000,Mn,'r:',lw=2,label='$M_{NFW}$')
ax.set_xlabel('r [kpc]'); ax.set_ylabel('M(<r) [Msun]')
ax.set_title('(e) Enclosed mass'); ax.legend(fontsize=8)

ax=axes[1,2]
nm=['NFW','MOND\n(Abel)','Membrane\n(Abel)']
vl=[c2_nfw/dof_n,c2_a0/dof_a,c2_b/dof_a]; cl=['red','blue','green']
bars=ax.bar(nm,vl,color=cl,alpha=0.7,edgecolor='black')
ax.axhline(1,color='k',ls='--',alpha=0.3)
for b,val in zip(bars,vl): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{val:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.set_ylabel(r'$\chi^2$/dof'); ax.set_title('(f) Model comparison')

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'cl1_individual_abel.png'),dpi=150)
plt.close()
print("  -> cl1_individual_abel.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[最終サマリー]")
print("="*70)
print(f"""
  cl1 単体 (z_spec={Z_L}, sig_v={SIGMA_V}):
    ソース: {len(df_src):,}, S/N={sn:.1f}, gamma_x={gx_mean:.5f}

  NFW: M200={M200:.2e}, c200={c200:.2f}, chi2/dof={c2_nfw/dof_n:.2f}
  MOND (Abel): chi2/dof={c2_a0/dof_a:.2f}, dAIC={aic_mo-aic_n:+.1f}
  膜 (Abel): g_c={gc_b/a0:.3f}a0, chi2/dof={c2_b/dof_a:.2f}, dAIC={aic_me-aic_n:+.1f}

  g_c = {gc_b/a0:.3f} a0, 68%:[{lo68:.3f},{hi68:.3f}], 95%:[{lo95:.3f},{hi95:.3f}]
  a0 in 95%: {'YES' if a0_95 else 'NO'}
  SPARC (0.825) in 95%: {'YES' if lo95<=0.825<=hi95 else 'NO'}

  Level: {'A (a0棄却)' if not a0_95 else 'B (a0棄却できない)'}
""")
print("完了。")
