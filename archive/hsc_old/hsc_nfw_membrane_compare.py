#!/usr/bin/env python3
"""
NFW フィット + 膜モデル比較
入力: hsc_y3_stacked_shear.csv
"""
import numpy as np,pandas as pd,sys,os,io
from scipy import optimize,integrate
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))

# 定数
H0=70.0; Om=0.3; OL=0.7; c_light=2.998e5
G_Mpc=4.301e-9  # (km/s)^2 Mpc/Msun
G_SI=6.674e-11; Msun_kg=1.989e30; pc_m=3.086e16; Mpc_m=3.086e22
a0=1.2e-10  # m/s^2
f_bar=0.17
Z_L=0.35; Z_S=0.75

print("="*70)
print("NFW フィット + 膜モデル比較")
print("="*70)
print(f"  z_lens={Z_L}, z_source={Z_S}")

# ====================================================================
# 宇宙論
# ====================================================================
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

S_cr=Sigma_cr(Z_L,Z_S); D_l=D_A(Z_L)
print(f"  D_A(z_l) = {D_l:.1f} Mpc")
print(f"  Sigma_cr = {S_cr:.3e} Msun/Mpc^2 = {S_cr*1e-12:.1f} Msun/pc^2")

# ====================================================================
# NFW (Wright & Brainerd 2000)
# ====================================================================
def nfw_params(M200,c200,zl):
    rc=rho_cr(zl)
    r200=(3*M200/(4*np.pi*200*rc))**(1./3.)
    rs=r200/c200
    dc=200./3.*c200**3/(np.log(1+c200)-c200/(1+c200))
    rho_s=dc*rc
    return rs,rho_s,r200

def nfw_sigma(R,rs,rho_s):
    x=R/rs; out=np.zeros_like(x,dtype=float)
    for i,xi in enumerate(x):
        if xi<1e-6: out[i]=0
        elif abs(xi-1)<1e-6: out[i]=2*rs*rho_s/3.0
        elif xi<1:
            sq=np.sqrt(1-xi**2)
            out[i]=2*rs*rho_s/(xi**2-1)*(1-np.log((1+sq)/xi)/sq)
        else:
            sq=np.sqrt(xi**2-1)
            out[i]=2*rs*rho_s/(xi**2-1)*(1-np.arctan(sq)/sq)
    return out

def nfw_sigma_mean(R,rs,rho_s):
    x=R/rs; out=np.zeros_like(x,dtype=float)
    for i,xi in enumerate(x):
        if xi<1e-6: out[i]=0
        elif abs(xi-1)<1e-6: out[i]=4*rs*rho_s*(1+np.log(0.5))
        elif xi<1:
            sq=np.sqrt(1-xi**2)
            g=np.log(xi/2)+np.log((1+sq)/xi)/sq
            out[i]=4*rs*rho_s*g/xi**2
        else:
            sq=np.sqrt(xi**2-1)
            g=np.log(xi/2)+np.arctan(sq)/sq
            out[i]=4*rs*rho_s*g/xi**2
    return out

def nfw_delta_sigma(R,M200,c200,zl):
    rs,rho_s,r200=nfw_params(M200,c200,zl)
    return nfw_sigma_mean(R,rs,rho_s)-nfw_sigma(R,rs,rho_s)

def nfw_gamma_t(R_am,M200,c200,zl,zs):
    R_Mpc=R_am*np.pi/(180*60)*D_A(zl)
    ds=nfw_delta_sigma(R_Mpc,M200,c200,zl)
    return ds/Sigma_cr(zl,zs)

# ====================================================================
# MOND/膜剪断モデル
# ====================================================================
def baryon_mass_enc(R_Mpc,M_bar,c200,zl):
    rc=rho_cr(zl)
    r200=(M_bar/f_bar/(4./3.*np.pi*200*rc))**(1./3.)
    rs=r200/c200; x=R_Mpc/rs
    dc=200./3.*c200**3/(np.log(1+c200)-c200/(1+c200))
    rho_s=dc*rc
    return 4*np.pi*rho_s*rs**3*(np.log(1+x)-x/(1+x))*f_bar

def mond_gamma_t(R_am,M_bar,c200,zl,zs,gc_val):
    Dl=D_A(zl); Scr=Sigma_cr(zl,zs)
    R_Mpc=R_am*np.pi/(180*60)*Dl
    M_enc=baryon_mass_enc(R_Mpc,M_bar,c200,zl)
    R_m=R_Mpc*Mpc_m; M_kg=M_enc*Msun_kg
    g_N=G_SI*M_kg/R_m**2
    g_obs=0.5*(g_N+np.sqrt(g_N**2+4*gc_val*g_N))
    M_eff=g_obs*R_m**2/G_SI/Msun_kg
    # NFW形状をスケーリング
    M_nfw=baryon_mass_enc(R_Mpc,M_bar,c200,zl)/f_bar
    scaling=np.clip(M_eff/np.maximum(M_nfw,1e5),0.01,100)
    gt_nfw=nfw_gamma_t(R_am,M_bar/f_bar,c200,zl,zs)
    return gt_nfw*scaling

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] データ読み込み")
print("="*70)

data_file=Path(os.path.join(base_dir,"hsc_y3_stacked_shear.csv"))
if data_file.exists():
    df=pd.read_csv(data_file)
    print(f"  {data_file.name}: {len(df)} ビン")
else:
    print("  CSVなし -> ハードコード")
    df=pd.DataFrame({
        'r_arcmin':[0.59,0.82,1.14,1.59,2.22,3.09,4.30,5.99,8.34,11.61,16.17,22.51],
        'gamma_t':[0.038,0.053,0.032,0.023,0.018,0.015,0.012,0.010,0.008,0.007,0.006,0.006],
        'gamma_t_err':[0.025,0.019,0.010,0.006,0.004,0.003,0.002,0.002,0.001,0.001,0.001,0.001],
    })

r_d=df['r_arcmin'].values; gt_d=df['gamma_t'].values; gt_e=df['gamma_t_err'].values
gx_d=df['gamma_x'].values if 'gamma_x' in df.columns else np.zeros_like(gt_d)

v=np.isfinite(gt_d)&np.isfinite(gt_e)&(gt_e>0)
r_f=r_d[v]; gt_f=gt_d[v]; err_f=gt_e[v]
print(f"  有効ビン: {v.sum()}, gamma_t: [{gt_f.min():.5f}, {gt_f.max():.5f}]")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] NFW フィット")
print("="*70)

def chi2_nfw(p):
    lm,lc=p; M=10**lm; c=10**lc
    if c<1 or c>20 or M<1e12 or M>1e16: return 1e20
    try:
        gt_m=nfw_gamma_t(r_f,M,c,Z_L,Z_S)
        return np.sum(((gt_f-gt_m)/err_f)**2)
    except: return 1e20

best_c2=np.inf; best_p=(14.0,0.5)
for lm in np.linspace(13.0,15.5,25):
    for lc in np.linspace(0.0,1.2,20):
        c2=chi2_nfw((lm,lc))
        if c2<best_c2: best_c2=c2; best_p=(lm,lc)

try:
    res=optimize.minimize(chi2_nfw,best_p,method='Nelder-Mead',
                         options={'xatol':0.001,'fatol':0.01})
    if res.fun<best_c2: best_p=res.x; best_c2=res.fun
except: pass

M200=10**best_p[0]; c200=10**best_p[1]
rs_f,rho_s_f,r200_f=nfw_params(M200,c200,Z_L)
dof_nfw=len(r_f)-2

print(f"  M_200 = {M200:.2e} Msun")
print(f"  c_200 = {c200:.2f}")
print(f"  r_200 = {r200_f:.2f} Mpc, r_s = {rs_f:.3f} Mpc")
print(f"  chi2/dof = {best_c2:.1f}/{dof_nfw} = {best_c2/dof_nfw:.2f}")

gt_nfw=nfw_gamma_t(r_f,M200,c200,Z_L,Z_S)

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] MOND / 膜モデル")
print("="*70)

M_bar=f_bar*M200
print(f"  M_bar = {f_bar} x M_200 = {M_bar:.2e} Msun")

# MOND (g_c=a0)
gt_mond=mond_gamma_t(r_f,M_bar,c200,Z_L,Z_S,a0)
chi2_mond=np.sum(((gt_f-gt_mond)/err_f)**2)
dof_mond=len(r_f)-1

print(f"\n  MOND (g_c=a0={a0:.1e}):")
print(f"    chi2/dof = {chi2_mond:.1f}/{dof_mond} = {chi2_mond/dof_mond:.2f}")

# g_c フリーフィット
def chi2_gc(lg):
    try:
        gt_m=mond_gamma_t(r_f,M_bar,c200,Z_L,Z_S,10**lg)
        return np.sum(((gt_f-gt_m)/err_f)**2)
    except: return 1e20

gc_grid=np.logspace(-12,-8,200)
c2g=np.array([chi2_gc(np.log10(g)) for g in gc_grid])
idx=np.argmin(c2g); gc_best=gc_grid[idx]; c2_best=c2g[idx]

try:
    res2=optimize.minimize_scalar(chi2_gc,bounds=(np.log10(gc_best)-1,np.log10(gc_best)+1),method='bounded')
    if res2.fun<c2_best: gc_best=10**res2.x; c2_best=res2.fun
except: pass

gt_mem=mond_gamma_t(r_f,M_bar,c200,Z_L,Z_S,gc_best)
dof_gc=len(r_f)-2

print(f"\n  膜モデル (g_c free):")
print(f"    g_c = {gc_best:.2e} m/s^2 = {gc_best/a0:.3f} a0")
print(f"    chi2/dof = {c2_best:.1f}/{dof_gc} = {c2_best/dof_gc:.2f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] モデル比較")
print("="*70)

aic_nfw=best_c2+2*2; aic_mond=chi2_mond+2*1; aic_mem=c2_best+2*2

print(f"\n  {'モデル':<35s} {'chi2':>8s} {'dof':>4s} {'chi2/dof':>9s} {'AIC':>8s} {'dAIC':>8s}")
print(f"  {'-'*70}")
print(f"  {'NFW (M200, c200)':<35s} {best_c2:>8.1f} {dof_nfw:>4d} {best_c2/dof_nfw:>9.2f} {aic_nfw:>8.1f} {0:>+8.1f}")
print(f"  {'MOND (g_c=a0)':<35s} {chi2_mond:>8.1f} {dof_mond:>4d} {chi2_mond/dof_mond:>9.2f} {aic_mond:>8.1f} {aic_mond-aic_nfw:>+8.1f}")
print(f"  {'膜モデル (g_c free)':<35s} {c2_best:>8.1f} {dof_gc:>4d} {c2_best/dof_gc:>9.2f} {aic_mem:>8.1f} {aic_mem-aic_nfw:>+8.1f}")

# 幾何平均法則
Sigma0_cl=M_bar/(np.pi*(rs_f*1e6)**2)  # Msun/pc^2
G_Sig0=G_SI*Sigma0_cl*Msun_kg/pc_m**2
gc_gm=np.sqrt(a0*G_Sig0)

print(f"\n  幾何平均法則との比較:")
print(f"    Sigma0 = {Sigma0_cl:.1f} Msun/pc^2")
print(f"    G*Sigma0 = {G_Sig0:.2e} m/s^2")
print(f"    g_c(geomean) = sqrt(a0*G*Sig0) = {gc_gm:.2e} m/s^2 = {gc_gm/a0:.2f} a0")
print(f"    g_c(fit) = {gc_best:.2e} m/s^2 = {gc_best/a0:.3f} a0")
print(f"    ratio = {gc_best/gc_gm:.3f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] g_c 信頼区間")
print("="*70)

gc_scan=np.logspace(np.log10(gc_best)-2,np.log10(gc_best)+2,500)
c2_scan=np.array([chi2_gc(np.log10(g)) for g in gc_scan])
dc2=c2_scan-c2_best

m68=dc2<1.0; m95=dc2<4.0
if m68.sum()>0:
    lo68=gc_scan[m68].min(); hi68=gc_scan[m68].max()
    print(f"  68% CI: [{lo68:.2e}, {hi68:.2e}] = [{lo68/a0:.3f}, {hi68/a0:.3f}] a0")
if m95.sum()>0:
    lo95=gc_scan[m95].min(); hi95=gc_scan[m95].max()
    print(f"  95% CI: [{lo95:.2e}, {hi95:.2e}] = [{lo95/a0:.3f}, {hi95/a0:.3f}] a0")

a0_in68=(lo68<=a0<=hi68) if m68.sum()>0 else False
a0_in95=(lo95<=a0<=hi95) if m95.sum()>0 else False
print(f"\n  a0={a0:.1e} は 68%CI内: {'YES' if a0_in68 else 'NO'}, 95%CI内: {'YES' if a0_in95 else 'NO'}")

gm_in68=(lo68<=gc_gm<=hi68) if m68.sum()>0 else False
gm_in95=(lo95<=gc_gm<=hi95) if m95.sum()>0 else False
print(f"  geomean={gc_gm:.2e} は 68%CI内: {'YES' if gm_in68 else 'NO'}, 95%CI内: {'YES' if gm_in95 else 'NO'}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 6] プロット")
print("="*70)

fig,axes=plt.subplots(2,2,figsize=(14,12))

# (a) プロファイル
ax=axes[0,0]
ax.errorbar(r_f,gt_f*1e3,yerr=err_f*1e3,fmt='o',color='black',ms=6,capsize=3,
           label='HSC Y3 stacked',zorder=5)
ax.plot(r_f,gt_nfw*1e3,'r-',lw=2,label=f'NFW (M={M200:.1e}, c={c200:.1f})')
ax.plot(r_f,gt_mond*1e3,'b--',lw=2,label=f'MOND ($g_c=a_0$)')
ax.plot(r_f,gt_mem*1e3,'g-.',lw=2,label=f'Membrane ($g_c$={gc_best/a0:.2f}$a_0$)')
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_t \times 10^3$')
ax.set_title('Shear profile: NFW vs MOND vs Membrane'); ax.legend(fontsize=8)

# (b) 残差
ax=axes[0,1]
ax.errorbar(r_f,(gt_f-gt_nfw)/err_f,fmt='o',color='red',ms=5,label='NFW')
ax.errorbar(r_f*1.05,(gt_f-gt_mond)/err_f,fmt='s',color='blue',ms=5,label='MOND')
ax.errorbar(r_f*1.1,(gt_f-gt_mem)/err_f,fmt='^',color='green',ms=5,label='Membrane')
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.axhline(2,color='gray',ls=':',alpha=0.3); ax.axhline(-2,color='gray',ls=':',alpha=0.3)
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'Residual/$\sigma$')
ax.set_title('Residuals'); ax.legend(fontsize=8)

# (c) g_c chi2プロファイル
ax=axes[1,0]
ax.plot(gc_scan/a0,dc2,'b-',lw=2)
ax.axhline(1,color='orange',ls='--',alpha=0.5,label='68% CL')
ax.axhline(4,color='red',ls='--',alpha=0.5,label='95% CL')
ax.axvline(1.0,color='gray',ls=':',alpha=0.5,label='$g_c=a_0$ (MOND)')
ax.axvline(gc_best/a0,color='green',ls='-',alpha=0.7,label=f'best={gc_best/a0:.2f}$a_0$')
if gc_gm>0:
    ax.axvline(gc_gm/a0,color='purple',ls='--',alpha=0.5,label=f'geomean={gc_gm/a0:.2f}$a_0$')
ax.set_xlabel(r'$g_c/a_0$'); ax.set_ylabel(r'$\Delta\chi^2$')
ax.set_title(r'$g_c$ constraint'); ax.set_xscale('log')
ax.set_ylim(0,min(20,dc2.max())); ax.legend(fontsize=7)

# (d) モデル比較
ax=axes[1,1]
names=['NFW','MOND\n($g_c=a_0$)','Membrane\n($g_c$ free)']
vals=[best_c2/dof_nfw,chi2_mond/dof_mond,c2_best/dof_gc]
cols=['red','blue','green']
bars=ax.bar(names,vals,color=cols,alpha=0.7,edgecolor='black')
ax.axhline(1,color='k',ls='--',alpha=0.3)
ax.set_ylabel(r'$\chi^2$/dof'); ax.set_title('Model comparison')
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.2f}',ha='center',fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'hsc_nfw_membrane_comparison.png'),dpi=150)
plt.close()
print("  -> hsc_nfw_membrane_comparison.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[サマリー]")
print("="*70)

print(f"""
  HSC Y3 スタック接線剪断プロファイル解析:

  NFW:     M_200={M200:.2e} Msun, c_200={c200:.2f}
           chi2/dof={best_c2:.1f}/{dof_nfw}={best_c2/dof_nfw:.2f}

  MOND:    g_c=a0={a0:.1e} m/s^2
           chi2/dof={chi2_mond:.1f}/{dof_mond}={chi2_mond/dof_mond:.2f}
           dAIC={aic_mond-aic_nfw:+.1f}

  膜モデル: g_c={gc_best:.2e} m/s^2 = {gc_best/a0:.3f} a0
           chi2/dof={c2_best:.1f}/{dof_gc}={c2_best/dof_gc:.2f}
           dAIC={aic_mem-aic_nfw:+.1f}

  幾何平均: g_c(geomean)={gc_gm:.2e} m/s^2 = {gc_gm/a0:.2f} a0
           g_c(fit)/g_c(geomean) = {gc_best/gc_gm:.3f}

  g_c 信頼区間:""")
if m68.sum()>0: print(f"    68%: [{lo68/a0:.3f}, {hi68/a0:.3f}] a0")
if m95.sum()>0: print(f"    95%: [{lo95/a0:.3f}, {hi95/a0:.3f}] a0")
print(f"    a0 は 95%CI内: {'YES' if a0_in95 else 'NO'}")
print(f"    geomean は 95%CI内: {'YES' if gm_in95 else 'NO'}")

print(f"""
  注意:
    z_lens={Z_L} は暫定。分光赤方偏移が必要。
    バリオン質量は f_bar={f_bar} x M_200 から推定。
    MOND剪断はNFWスケーリング近似。正確な投影計算は未実装。
""")
print("完了。")
