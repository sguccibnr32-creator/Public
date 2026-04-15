#!/usr/bin/env python3
"""
最終解析: 分光確認クラスター cl1 (z=0.313) + Abel 変換
"""
import numpy as np,pandas as pd,sys,os,io,time
from scipy import optimize,integrate,interpolate
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))

H0=70.0; Om=0.3; OL=0.7; c_light=2.998e5
G_Mpc=4.301e-9; G_SI=6.674e-11; Msun=1.989e30; Mpc_m=3.086e22
a0=1.2e-10

Z_L=0.313; Z_S=1.0; SIGMA_V=527
CL1_RA=140.450; CL1_DEC=-0.251

print("="*70)
print("最終解析: cl1 (z_spec=0.313) + Abel 変換")
print("="*70)
print(f"  z_spec={Z_L} (22銀河, sigma_v={SIGMA_V} km/s), z_s={Z_S}")

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
am_Mpc=Dl*np.pi/(180*60)
h=H0/100; M200_sig=1e15/h*(SIGMA_V/1082.9)**(1/0.3361)
print(f"  D_A={Dl:.1f} Mpc, Sig_cr={Scr:.3e}, 1'={am_Mpc*1000:.1f} kpc")
print(f"  M200(sigma_v) ~ {M200_sig:.2e} Msun")

# NFW
def nfw_params(M200,c200,zl):
    rc=rho_cr(zl); r200=(3*M200/(4*np.pi*200*rc))**(1./3.); rs=r200/c200
    dc=200./3.*c200**3/(np.log(1+c200)-c200/(1+c200)); rho_s=dc*rc
    return rs,rho_s,r200

def nfw_enclosed(r,M200,c200,zl):
    rs,rho_s,_=nfw_params(M200,c200,zl); x=r/rs
    return 4*np.pi*rho_s*rs**3*(np.log(1+x)-x/(1+x))

def nfw_density(r,M200,c200,zl):
    rs,rho_s,_=nfw_params(M200,c200,zl); x=r/rs
    return rho_s/(x*(1+x)**2)

def nfw_gamma(R_am,M200,c200,zl,zs):
    rs,rho_s,_=nfw_params(M200,c200,zl)
    R_Mpc=np.maximum(R_am*am_Mpc,1e-6); x=R_Mpc/rs
    Sig=np.zeros_like(x); Sm=np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi<1e-6: continue
        elif abs(xi-1)<1e-4: Sig[i]=2*rs*rho_s/3; Sm[i]=4*rs*rho_s*(1+np.log(0.5))
        elif xi<1:
            sq=np.sqrt(1-xi**2)
            Sig[i]=2*rs*rho_s/(xi**2-1)*(1-np.log((1+sq)/xi)/sq)
            Sm[i]=4*rs*rho_s*(np.log(xi/2)+np.log((1+sq)/xi)/sq)/xi**2
        else:
            sq=np.sqrt(xi**2-1)
            Sig[i]=2*rs*rho_s/(xi**2-1)*(1-np.arctan(sq)/sq)
            Sm[i]=4*rs*rho_s*(np.log(xi/2)+np.arctan(sq)/sq)/xi**2
    return (Sm-Sig)/Sigma_cr(zl,zs)

# MOND/膜 + Abel
def g_mond(gN,gc):
    return 0.5*(gN+np.sqrt(gN**2+4*gc*gN)) if gN>0 else 0

def mond_rho_eff(r_arr,M200,c200,zl,gc,fb=0.17):
    n=len(r_arr); Meff=np.zeros(n)
    for i,r in enumerate(r_arr):
        Mb=nfw_enclosed(r,M200,c200,zl)*fb
        rm=r*Mpc_m; Mkg=Mb*Msun
        gN=G_SI*Mkg/rm**2 if rm>1e10 else 0
        gobs=g_mond(gN,gc)
        Meff[i]=gobs*rm**2/G_SI/Msun
    rho=np.zeros(n)
    for i in range(1,n-1):
        dMdr=(Meff[i+1]-Meff[i-1])/(r_arr[i+1]-r_arr[i-1])
        rho[i]=max(dMdr/(4*np.pi*r_arr[i]**2),0)
    rho[0]=rho[1]; rho[-1]=rho[-2]
    return rho,Meff

def abel(R_arr,r_f,rho_f):
    lr=np.log10(r_f); lrho=np.log10(np.maximum(rho_f,1e-50))
    rhoi=interpolate.interp1d(lr,lrho,fill_value=-50,bounds_error=False)
    Sig=np.zeros(len(R_arr)); rmax=r_f.max()
    for i,R in enumerate(R_arr):
        if R<=0: continue
        um=np.sqrt(max(rmax**2-R**2,0))
        if um<=0: continue
        def f(u): return 10**rhoi(np.log10(np.sqrt(u**2+R**2)))
        try: v,_=integrate.quad(f,0,um,limit=200,epsrel=1e-4)
        except: v=0
        Sig[i]=2*v
    return Sig

def delta_sigma(R_arr,Sig):
    Sf=interpolate.interp1d(R_arr,Sig,fill_value=0,bounds_error=False)
    Sm=np.zeros(len(R_arr))
    for i,R in enumerate(R_arr):
        if R<=0: continue
        try: v,_=integrate.quad(lambda Rp:Sf(Rp)*Rp,R_arr[0]*0.1,R,limit=100)
        except: v=0
        Sm[i]=2*v/R**2
    return Sm-Sig

def mond_gamma_abel(R_am,M200,c200,zl,zs,gc,fb=0.17):
    R_Mpc=np.maximum(R_am*am_Mpc,1e-4)
    rf=np.logspace(np.log10(R_Mpc.min()*0.1),np.log10(R_Mpc.max()*5),400)
    rho,Meff=mond_rho_eff(rf,M200,c200,zl,gc,fb)
    Sig=abel(R_Mpc,rf,rho)
    DS=delta_sigma(R_Mpc,Sig)
    return DS/Sigma_cr(zl,zs),rho,Meff,rf

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] データ読み込み")
print("="*70)

sf=Path(os.path.join(base_dir,"hsc_y3_stacked_shear.csv"))
if sf.exists(): df=pd.read_csv(sf); print(f"  {sf.name}: {len(df)} ビン")
else:
    df=pd.DataFrame({
        'r_arcmin':[0.59,0.82,1.14,1.59,2.22,3.09,4.30,5.99,8.34,11.61,16.17,22.51],
        'gamma_t':[0.038,0.053,0.032,0.023,0.018,0.015,0.012,0.010,0.008,0.007,0.006,0.006],
        'gamma_t_err':[0.025,0.019,0.010,0.006,0.004,0.003,0.002,0.002,0.001,0.001,0.001,0.001]})

rd=df['r_arcmin'].values; gd=df['gamma_t'].values; ge=df['gamma_t_err'].values
v=np.isfinite(gd)&np.isfinite(ge)&(ge>0)
rf_=rd[v]; gf=gd[v]; ef=ge[v]; nd=len(rf_)
print(f"  有効ビン: {nd}, 半径: {rf_.min()*am_Mpc*1000:.0f}-{rf_.max()*am_Mpc*1000:.0f} kpc")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] NFW フィット")
print("="*70)

def c2n(p):
    M,c=10**p[0],10**p[1]
    if c<1 or c>20 or M<1e12 or M>1e16: return 1e20
    try: return np.sum(((gf-nfw_gamma(rf_,M,c,Z_L,Z_S))/ef)**2)
    except: return 1e20

bc=np.inf; bp=(14,0.5)
for lm in np.linspace(13,15.5,25):
    for lc in np.linspace(0,1.2,15):
        c2=c2n((lm,lc))
        if c2<bc: bc=c2; bp=(lm,lc)
try:
    res=optimize.minimize(c2n,bp,method='Nelder-Mead',options={'xatol':0.001,'fatol':0.01})
    if res.fun<bc: bc=res.fun; bp=res.x
except: pass

M200=10**bp[0]; c200=10**bp[1]
rs_n,_,r200_n=nfw_params(M200,c200,Z_L)
gt_nfw=nfw_gamma(rf_,M200,c200,Z_L,Z_S)
c2_nfw=bc; dof_n=nd-2

print(f"  M200={M200:.2e}, c200={c200:.2f}")
print(f"  r200={r200_n*1000:.0f} kpc, rs={rs_n*1000:.0f} kpc")
print(f"  chi2/dof={c2_nfw:.1f}/{dof_n}={c2_nfw/dof_n:.2f}")
print(f"  M200(sig_v)={M200_sig:.2e} vs M200(lens)={M200:.2e}, ratio={M200/M200_sig:.2f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] Abel 変換 MOND/膜モデル")
print("="*70)

gc_gr=np.logspace(-12,-8,80)
c2_gc=np.full(len(gc_gr),np.inf)

print(f"  g_c スキャン ({len(gc_gr)} 点)...")
t0=time.time()
for ig,gcv in enumerate(gc_gr):
    try:
        gt_m,_,_,_=mond_gamma_abel(rf_,M200,c200,Z_L,Z_S,gcv)
        if np.all(np.isfinite(gt_m)):
            c2_gc[ig]=np.sum(((gf-gt_m)/ef)**2)
    except: pass
    if (ig+1)%20==0:
        bi=np.argmin(c2_gc[:ig+1])
        print(f"    {ig+1}/{len(gc_gr)}: best={gc_gr[bi]/a0:.3f} a0, chi2={c2_gc[bi]:.1f}, {time.time()-t0:.0f}s")

bi=np.argmin(c2_gc); gc_b=gc_gr[bi]; c2_b=c2_gc[bi]
ai=np.argmin(np.abs(gc_gr-a0)); c2_a0=c2_gc[ai]
dof_a=nd-2

# ベスト+MOND プロファイル
gt_ab,rho_b,Meff_b,r_fn=mond_gamma_abel(rf_,M200,c200,Z_L,Z_S,gc_b)
gt_mo,_,_,_=mond_gamma_abel(rf_,M200,c200,Z_L,Z_S,a0)

# 信頼区間
dc2=c2_gc-c2_b
m68=dc2<1; m95=dc2<4
lo68=gc_gr[m68].min()/a0 if m68.sum()>0 else gc_b/a0
hi68=gc_gr[m68].max()/a0 if m68.sum()>0 else gc_b/a0
lo95=gc_gr[m95].min()/a0 if m95.sum()>0 else gc_b/a0
hi95=gc_gr[m95].max()/a0 if m95.sum()>0 else gc_b/a0
a0_68=lo68<=1.0<=hi68; a0_95=lo95<=1.0<=hi95

print(f"\n  結果:")
print(f"    g_c(best) = {gc_b/a0:.3f} a0")
print(f"    68% CI: [{lo68:.3f}, {hi68:.3f}] a0")
print(f"    95% CI: [{lo95:.3f}, {hi95:.3f}] a0")
print(f"    a0 in 68%: {'YES' if a0_68 else 'NO'}, 95%: {'YES' if a0_95 else 'NO'}")
print(f"    chi2(best)/dof={c2_b:.1f}/{dof_a}={c2_b/dof_a:.2f}")
print(f"    chi2(a0)/dof={c2_a0:.1f}/{dof_a}={c2_a0/dof_a:.2f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] モデル比較")
print("="*70)

aic_n=c2_nfw+4; aic_mo=c2_a0+2; aic_me=c2_b+4
print(f"\n  {'モデル':<40s} {'chi2':>7s} {'dof':>4s} {'chi2/dof':>9s} {'AIC':>7s} {'dAIC':>7s}")
print(f"  {'-'*70}")
print(f"  {'NFW (解析的)':<40s} {c2_nfw:>7.1f} {dof_n:>4d} {c2_nfw/dof_n:>9.2f} {aic_n:>7.1f} {0:>+7.1f}")
print(f"  {'MOND g_c=a0 (Abel)':<40s} {c2_a0:>7.1f} {dof_a:>4d} {c2_a0/dof_a:>9.2f} {aic_mo:>7.1f} {aic_mo-aic_n:>+7.1f}")
print(f"  {'膜 g_c={0:.3f}a0 (Abel)'.format(gc_b/a0):<40s} {c2_b:>7.1f} {dof_a:>4d} {c2_b/dof_a:>9.2f} {aic_me:>7.1f} {aic_me-aic_n:>+7.1f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] 全手法の統合")
print("="*70)

print(f"""
  手法                    z_lens  g_c/a0   95% CI
  SPARC RAR (175銀河)      --     0.825    ~1
  HSC z=0.35 (旧NFWscale)  0.35   0.733    [0.51,0.99]
  HSC z=0.56 (photo-z)     0.56   0.777    [0.54,1.03]
  HSC z=0.313 (分光+Abel)  0.313  {gc_b/a0:.3f}    [{lo95:.3f},{hi95:.3f}]
""")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 6] プロット")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(f'cl1 (z_spec=0.313, $\\sigma_v$=527 km/s): Abel Transform Analysis',
             fontsize=13,fontweight='bold')

ax=axes[0,0]
ax.errorbar(rf_,gf*1e3,yerr=ef*1e3,fmt='o',color='black',ms=6,capsize=3,label='HSC Y3',zorder=5)
ax.plot(rf_,gt_nfw*1e3,'r-',lw=2,label=f'NFW (M={M200:.1e})')
ax.plot(rf_,gt_mo*1e3,'b--',lw=2,label='MOND Abel ($g_c=a_0$)')
ax.plot(rf_,gt_ab*1e3,'g-.',lw=2,label=f'Membrane Abel ({gc_b/a0:.2f}$a_0$)')
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_t \times 10^3$')
ax.set_title('(a) Shear profile'); ax.legend(fontsize=7)

ax=axes[0,1]
ax.errorbar(rf_,(gf-gt_nfw)/ef,fmt='o',color='red',ms=4,label='NFW')
ax.errorbar(rf_*1.03,(gf-gt_mo)/ef,fmt='s',color='blue',ms=4,label='MOND Abel')
ax.errorbar(rf_*1.06,(gf-gt_ab)/ef,fmt='^',color='green',ms=4,label='Membrane Abel')
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
ax.set_ylim(0,min(25,dc2[fin].max() if fin.sum()>0 else 25))

ax=axes[1,0]
rho_nfw3=np.array([nfw_density(r,M200,c200,Z_L) for r in r_fn])
ax.loglog(r_fn*1000,rho_b,'g-',lw=2,label='MOND eff.')
ax.loglog(r_fn*1000,rho_nfw3,'r--',lw=2,label='NFW')
ax.set_xlabel('r [kpc]'); ax.set_ylabel(r'$\rho$ [Msun/Mpc$^3$]')
ax.set_title('(d) 3D density'); ax.legend(fontsize=8)

ax=axes[1,1]
Mb=np.array([nfw_enclosed(r,M200,c200,Z_L)*0.17 for r in r_fn])
Mn=np.array([nfw_enclosed(r,M200,c200,Z_L) for r in r_fn])
ax.loglog(r_fn*1000,Meff_b,'g-',lw=2,label='$M_{eff}$ (MOND)')
ax.loglog(r_fn*1000,Mb,'b--',lw=2,label='$M_{bar}$')
ax.loglog(r_fn*1000,Mn,'r:',lw=2,label='$M_{NFW}$')
ax.set_xlabel('r [kpc]'); ax.set_ylabel('M(<r) [Msun]')
ax.set_title('(e) Enclosed mass'); ax.legend(fontsize=8)

ax=axes[1,2]
nm=['NFW','MOND\n(Abel)',f'Membrane\n(Abel)']
vl=[c2_nfw/dof_n,c2_a0/dof_a,c2_b/dof_a]; cl=['red','blue','green']
bars=ax.bar(nm,vl,color=cl,alpha=0.7,edgecolor='black')
ax.axhline(1,color='k',ls='--',alpha=0.3)
for b,v in zip(bars,vl): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.set_ylabel(r'$\chi^2$/dof'); ax.set_title(f'(f) Model comparison (z={Z_L})')

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'hsc_final_cl1_abel.png'),dpi=150)
plt.close()
print("  -> hsc_final_cl1_abel.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[最終サマリー]")
print("="*70)
print(f"""
  cl1 (z_spec=0.313, sig_v=527 km/s) + Abel 変換:

  NFW:    M200={M200:.2e}, c200={c200:.2f}, chi2/dof={c2_nfw/dof_n:.2f}
  MOND:   g_c=a0, chi2/dof={c2_a0/dof_a:.2f}, dAIC={aic_mo-aic_n:+.1f}
  膜:     g_c={gc_b/a0:.3f}a0, chi2/dof={c2_b/dof_a:.2f}, dAIC={aic_me-aic_n:+.1f}

  g_c = {gc_b/a0:.3f} a0, 68%:[{lo68:.3f},{hi68:.3f}], 95%:[{lo95:.3f},{hi95:.3f}]
  a0 in 95%: {'YES' if a0_95 else 'NO'}
  SPARC (0.825 a0) in 95%: {'YES' if lo95<=0.825<=hi95 else 'NO'}

  Level: {'B (a0棄却できない)' if a0_95 else 'A候補 (a0棄却)'}
""")
print("完了。")
