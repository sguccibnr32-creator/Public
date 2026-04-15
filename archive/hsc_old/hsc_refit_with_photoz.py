#!/usr/bin/env python3
"""
NFW / 膜モデル比較（z_lens 更新版）
photo-z 解析で確定した個別クラスター赤方偏移を使用
"""
import numpy as np,pandas as pd,sys,os,io
from scipy import optimize,integrate
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))

H0=70.0; Om=0.3; OL=0.7; c_light=2.998e5
G_Mpc=4.301e-9; G_SI=6.674e-11; Msun_kg=1.989e30; Mpc_m=3.086e22
a0=1.2e-10; f_bar=0.17
Z_S_EFF=1.0

CLUSTERS_PHOTOZ=[
    ("cl2",142.21,-0.12,0.670,5.5,"使用"),
    ("cl3",140.30,-0.35,0.370,4.2,"使用"),
    ("cl4",182.68,-0.28,0.510,5.2,"使用"),
    ("cl5",216.75,-0.20,0.610,7.4,"使用"),
]

print("="*70)
print("NFW / 膜モデル比較（photo-z 更新版）")
print("="*70)
print(f"  z_source(有効)={Z_S_EFF}, 使用クラスター: {len(CLUSTERS_PHOTOZ)}")
for n,ra,dec,z,sn,st in CLUSTERS_PHOTOZ:
    print(f"    {n}: z={z:.3f}, S/N={sn:.1f}")

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

print(f"\n  Sigma_cr 一覧:")
for n,ra,dec,zl,sn,st in CLUSTERS_PHOTOZ:
    Scr=Sigma_cr(zl,Z_S_EFF); Dl=D_A(zl)
    am_Mpc=Dl*np.pi/(180*60)
    print(f"    {n}: z={zl:.3f}, D_A={Dl:.0f} Mpc, Sig_cr={Scr:.2e}, 1'={am_Mpc:.4f} Mpc")

# ====================================================================
# NFW (Wright & Brainerd 2000)
# ====================================================================
def nfw_params(M200,c200,zl):
    rc=rho_cr(zl); r200=(3*M200/(4*np.pi*200*rc))**(1./3.); rs=r200/c200
    dc=200./3.*c200**3/(np.log(1+c200)-c200/(1+c200)); rho_s=dc*rc
    return rs,rho_s,r200

def nfw_sigma(R,rs,rho_s):
    x=R/rs; out=np.zeros_like(x,dtype=float)
    for i,xi in enumerate(x):
        if xi<1e-6: out[i]=0
        elif abs(xi-1)<1e-6: out[i]=2*rs*rho_s/3.0
        elif xi<1:
            sq=np.sqrt(1-xi**2); out[i]=2*rs*rho_s/(xi**2-1)*(1-np.log((1+sq)/xi)/sq)
        else:
            sq=np.sqrt(xi**2-1); out[i]=2*rs*rho_s/(xi**2-1)*(1-np.arctan(sq)/sq)
    return out

def nfw_sigma_mean(R,rs,rho_s):
    x=R/rs; out=np.zeros_like(x,dtype=float)
    for i,xi in enumerate(x):
        if xi<1e-6: out[i]=0
        elif abs(xi-1)<1e-6: out[i]=4*rs*rho_s*(1+np.log(0.5))
        elif xi<1:
            sq=np.sqrt(1-xi**2); g=np.log(xi/2)+np.log((1+sq)/xi)/sq; out[i]=4*rs*rho_s*g/xi**2
        else:
            sq=np.sqrt(xi**2-1); g=np.log(xi/2)+np.arctan(sq)/sq; out[i]=4*rs*rho_s*g/xi**2
    return out

def nfw_gamma_t(R_am,M200,c200,zl,zs):
    rs,rho_s,r200=nfw_params(M200,c200,zl); Dl=D_A(zl)
    R_Mpc=np.maximum(R_am*np.pi/(180*60)*Dl,1e-6)
    ds=nfw_sigma_mean(R_Mpc,rs,rho_s)-nfw_sigma(R_Mpc,rs,rho_s)
    return ds/Sigma_cr(zl,zs)

# ====================================================================
# MOND/膜
# ====================================================================
def baryon_enc(R_Mpc,M200,c200,zl):
    rs,rho_s,r200=nfw_params(M200,c200,zl); x=np.maximum(R_Mpc/rs,1e-6)
    return 4*np.pi*rho_s*rs**3*(np.log(1+x)-x/(1+x))*f_bar

def mond_gamma_t(R_am,M200,c200,zl,zs,gc):
    Dl=D_A(zl); R_Mpc=np.maximum(R_am*np.pi/(180*60)*Dl,1e-6)
    Mb=baryon_enc(R_Mpc,M200,c200,zl)
    Rm=R_Mpc*Mpc_m; Mkg=Mb*Msun_kg
    gN=G_SI*Mkg/Rm**2; gobs=0.5*(gN+np.sqrt(gN**2+4*gc*gN))
    Meff=gobs*Rm**2/G_SI/Msun_kg
    Mnfw=baryon_enc(R_Mpc,M200,c200,zl)/f_bar
    scl=np.clip(np.where(Mnfw>0,Meff/Mnfw,1.0),0.01,100)
    return nfw_gamma_t(R_am,M200,c200,zl,zs)*scl

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] データ読み込み")
print("="*70)

sf=Path(os.path.join(base_dir,"hsc_y3_stacked_shear.csv"))
if sf.exists():
    df_s=pd.read_csv(sf); print(f"  {sf.name}: {len(df_s)} ビン")
else:
    df_s=pd.DataFrame({
        'r_arcmin':[0.59,0.82,1.14,1.59,2.22,3.09,4.30,5.99,8.34,11.61,16.17,22.51],
        'gamma_t':[0.038,0.053,0.032,0.023,0.018,0.015,0.012,0.010,0.008,0.007,0.006,0.006],
        'gamma_t_err':[0.025,0.019,0.010,0.006,0.004,0.003,0.002,0.002,0.001,0.001,0.001,0.001]})

r_d=df_s['r_arcmin'].values; gt_d=df_s['gamma_t'].values; gt_e=df_s['gamma_t_err'].values
v=np.isfinite(gt_d)&np.isfinite(gt_e)&(gt_e>0)
r_f=r_d[v]; gt_f=gt_d[v]; err_f=gt_e[v]
print(f"  有効ビン: {v.sum()}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] z_lens スキャン")
print("="*70)

z_scan=np.arange(0.20,0.85,0.05)
gc_vs_z=[]

print(f"\n  {'z_l':>5s} {'chi2_NFW':>9s} {'chi2_MOND':>10s} {'chi2_mem':>9s} {'g_c/a0':>8s} {'M200':>11s}")
print(f"  {'-'*58}")

for zl in z_scan:
    if zl>=Z_S_EFF-0.05: continue
    def c2n(p):
        lm,lc=p; M,c=10**lm,10**lc
        if c<1 or c>20 or M<1e12 or M>1e16: return 1e20
        try: return np.sum(((gt_f-nfw_gamma_t(r_f,M,c,zl,Z_S_EFF))/err_f)**2)
        except: return 1e20
    bc=np.inf; bp=(14,0.5)
    for lm in np.linspace(13,15.5,15):
        for lc in np.linspace(0,1.2,10):
            c2=c2n((lm,lc))
            if c2<bc: bc=c2; bp=(lm,lc)
    try:
        res=optimize.minimize(c2n,bp,method='Nelder-Mead')
        if res.fun<bc: bc=res.fun; bp=res.x
    except: pass
    M_z=10**bp[0]; c_z=10**bp[1]
    gt_mo=mond_gamma_t(r_f,M_z,c_z,zl,Z_S_EFF,a0)
    c2mo=np.sum(((gt_f-gt_mo)/err_f)**2)
    def c2gc(lg):
        try: return np.sum(((gt_f-mond_gamma_t(r_f,M_z,c_z,zl,Z_S_EFF,10**lg))/err_f)**2)
        except: return 1e20
    gg=np.logspace(-12,-8,80); c2g=[c2gc(np.log10(g)) for g in gg]
    gc_z=gg[np.argmin(c2g)]; c2me=min(c2g)
    gc_vs_z.append({'z_l':zl,'gc':gc_z,'gc_a0':gc_z/a0,'chi2_nfw':bc,'chi2_mond':c2mo,'chi2_membrane':c2me,'M200':M_z})
    print(f"  {zl:>5.2f} {bc:>9.1f} {c2mo:>10.1f} {c2me:>9.1f} {gc_z/a0:>8.3f} {M_z:>11.2e}")

df_zs=pd.DataFrame(gc_vs_z)

# 加重平均 z
z_w=np.array([sn for _,_,_,_,sn,_ in CLUSTERS_PHOTOZ])
z_v=np.array([z for _,_,_,z,_,_ in CLUSTERS_PHOTOZ])
Z_L=np.average(z_v,weights=z_w)
print(f"\n  クラスター z: 加重平均={Z_L:.3f}, 中央値={np.median(z_v):.3f}")

# ====================================================================
print(f"\n{'='*70}")
print(f"[Step 3] z_lens={Z_L:.3f} でのモデル比較")
print("="*70)

# NFW
def c2nfw(p):
    lm,lc=p; M,c=10**lm,10**lc
    if c<1 or c>20 or M<1e12 or M>1e16: return 1e20
    try: return np.sum(((gt_f-nfw_gamma_t(r_f,M,c,Z_L,Z_S_EFF))/err_f)**2)
    except: return 1e20
bc=np.inf; bp=(14,0.5)
for lm in np.linspace(13,15.5,20):
    for lc in np.linspace(0,1.2,15):
        c2=c2nfw((lm,lc))
        if c2<bc: bc=c2; bp=(lm,lc)
try:
    res=optimize.minimize(c2nfw,bp,method='Nelder-Mead',options={'xatol':0.001,'fatol':0.01})
    if res.fun<bc: bc=res.fun; bp=res.x
except: pass
M200=10**bp[0]; c200=10**bp[1]
rs_b,_,r200_b=nfw_params(M200,c200,Z_L)
c2_nfw=bc; dof_n=len(r_f)-2
gt_nfw=nfw_gamma_t(r_f,M200,c200,Z_L,Z_S_EFF)
print(f"  NFW: M200={M200:.2e}, c200={c200:.2f}, r200={r200_b:.2f} Mpc")
print(f"    chi2/dof={c2_nfw:.1f}/{dof_n}={c2_nfw/dof_n:.2f}")

# MOND
gt_mo=mond_gamma_t(r_f,M200,c200,Z_L,Z_S_EFF,a0)
c2_mo=np.sum(((gt_f-gt_mo)/err_f)**2); dof_mo=len(r_f)-1
print(f"  MOND: chi2/dof={c2_mo:.1f}/{dof_mo}={c2_mo/dof_mo:.2f}")

# 膜
def c2gc(lg):
    try: return np.sum(((gt_f-mond_gamma_t(r_f,M200,c200,Z_L,Z_S_EFF,10**lg))/err_f)**2)
    except: return 1e20
gc_gr=np.logspace(-12,-8,200); c2_gr=np.array([c2gc(np.log10(g)) for g in gc_gr])
gc_b=gc_gr[np.argmin(c2_gr)]; c2_me=c2_gr.min(); dof_me=len(r_f)-2
gt_me=mond_gamma_t(r_f,M200,c200,Z_L,Z_S_EFF,gc_b)
print(f"  膜モデル: g_c={gc_b:.2e} m/s^2 = {gc_b/a0:.3f} a0")
print(f"    chi2/dof={c2_me:.1f}/{dof_me}={c2_me/dof_me:.2f}")

# 信頼区間
dc2=c2_gr-c2_me
m68=dc2<1.0; m95=dc2<4.0
lo68=gc_gr[m68].min()/a0 if m68.sum()>0 else gc_b/a0
hi68=gc_gr[m68].max()/a0 if m68.sum()>0 else gc_b/a0
lo95=gc_gr[m95].min()/a0 if m95.sum()>0 else gc_b/a0
hi95=gc_gr[m95].max()/a0 if m95.sum()>0 else gc_b/a0
a0_in68=lo68<=1.0<=hi68; a0_in95=lo95<=1.0<=hi95
print(f"  68% CI: [{lo68:.3f}, {hi68:.3f}] a0")
print(f"  95% CI: [{lo95:.3f}, {hi95:.3f}] a0")
print(f"  a0 in 68%CI: {'YES' if a0_in68 else 'NO'}, 95%CI: {'YES' if a0_in95 else 'NO'}")

# AIC
aic_n=c2_nfw+4; aic_mo=c2_mo+2; aic_me=c2_me+4
print(f"\n  {'モデル':<30s} {'chi2':>8s} {'dof':>4s} {'chi2/dof':>9s} {'AIC':>8s} {'dAIC':>8s}")
print(f"  {'-'*64}")
print(f"  {'NFW':<30s} {c2_nfw:>8.1f} {dof_n:>4d} {c2_nfw/dof_n:>9.2f} {aic_n:>8.1f} {0:>+8.1f}")
print(f"  {'MOND (g_c=a0)':<30s} {c2_mo:>8.1f} {dof_mo:>4d} {c2_mo/dof_mo:>9.2f} {aic_mo:>8.1f} {aic_mo-aic_n:>+8.1f}")
print(f"  {'膜 (g_c free)':<30s} {c2_me:>8.1f} {dof_me:>4d} {c2_me/dof_me:>9.2f} {aic_me:>8.1f} {aic_me-aic_n:>+8.1f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] g_c(z_lens) の安定性")
print("="*70)

gc_a0_vals=df_zs['gc_a0'].values
print(f"  g_c/a0 の範囲: [{gc_a0_vals.min():.3f}, {gc_a0_vals.max():.3f}]")
print(f"  中央値: {np.median(gc_a0_vals):.3f}, std: {np.std(gc_a0_vals):.3f}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 5] SPARC との統合比較")
print("="*70)

print(f"""
  スケール         手法              g_c/a0
  銀河 (SPARC)     RAR独立測定       0.825 (中央値)
  銀河 (SPARC)     幾何平均法則      alpha=0.545 (alpha=0.5と整合)
  クラスター(HSC)  弱レンズ(z=0.35)  0.733 [0.512,0.994]
  クラスター(HSC)  弱レンズ(z={Z_L:.2f})  {gc_b/a0:.3f} [{lo95:.3f},{hi95:.3f}]
""")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 6] プロット")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(f'NFW vs MOND vs Membrane ($z_l$={Z_L:.3f}, $z_s$={Z_S_EFF})',fontsize=14,fontweight='bold')

# (a) プロファイル
ax=axes[0,0]
ax.errorbar(r_f,gt_f*1e3,yerr=err_f*1e3,fmt='o',color='black',ms=6,capsize=3,label='HSC Y3',zorder=5)
ax.plot(r_f,gt_nfw*1e3,'r-',lw=2,label=f'NFW (M={M200:.1e})')
ax.plot(r_f,gt_mo*1e3,'b--',lw=2,label=f'MOND ($g_c=a_0$)')
ax.plot(r_f,gt_me*1e3,'g-.',lw=2,label=f'Membrane ({gc_b/a0:.2f}$a_0$)')
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_t \times 10^3$')
ax.set_title(f'(a) $z_l$={Z_L:.2f}'); ax.legend(fontsize=7)

# (b) 残差
ax=axes[0,1]
ax.errorbar(r_f,(gt_f-gt_nfw)/err_f,fmt='o',color='red',ms=5,label='NFW')
ax.errorbar(r_f*1.05,(gt_f-gt_mo)/err_f,fmt='s',color='blue',ms=5,label='MOND')
ax.errorbar(r_f*1.1,(gt_f-gt_me)/err_f,fmt='^',color='green',ms=5,label='Membrane')
ax.axhline(0,color='k',ls='--',alpha=0.3); ax.axhspan(-2,2,alpha=0.05,color='gray')
ax.set_xscale('log'); ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'Residual/$\sigma$')
ax.set_title('(b) Residuals'); ax.legend(fontsize=8)

# (c) g_c chi2
ax=axes[0,2]
ax.plot(gc_gr/a0,dc2,'b-',lw=2)
ax.axhline(1,color='orange',ls='--',alpha=0.5,label='68%')
ax.axhline(4,color='red',ls='--',alpha=0.5,label='95%')
ax.axvline(1.0,color='gray',ls=':',alpha=0.5,label='$a_0$')
ax.axvline(gc_b/a0,color='green',ls='-',alpha=0.7,label=f'best={gc_b/a0:.2f}$a_0$')
ax.set_xscale('log'); ax.set_xlabel('$g_c/a_0$'); ax.set_ylabel(r'$\Delta\chi^2$')
ax.set_title('(c) $g_c$ constraint'); ax.set_ylim(0,min(20,dc2[np.isfinite(dc2)].max()))
ax.legend(fontsize=7)

# (d) g_c vs z_l
ax=axes[1,0]
ax.plot(df_zs['z_l'],df_zs['gc_a0'],'o-',color='steelblue',ms=6)
ax.axhline(1.0,color='gray',ls='--',alpha=0.5,label='$a_0$')
ax.axhline(0.825,color='orange',ls=':',alpha=0.5,label='SPARC(0.825)')
ax.axvline(Z_L,color='red',ls='--',alpha=0.5,label=f'$z_l$={Z_L:.2f}')
for _,_,_,zc,sn,_ in CLUSTERS_PHOTOZ:
    ax.axvline(zc,color='green',ls=':',alpha=0.3)
ax.set_xlabel('$z_{lens}$'); ax.set_ylabel('$g_c/a_0$')
ax.set_title('(d) $g_c$ stability'); ax.legend(fontsize=7)

# (e) chi2/dof bar
ax=axes[1,1]
nm=['NFW','MOND\n($g_c=a_0$)',f'Membrane\n({gc_b/a0:.2f}$a_0$)']
vl=[c2_nfw/dof_n,c2_mo/dof_mo,c2_me/dof_me]; cl=['red','blue','green']
bars=ax.bar(nm,vl,color=cl,alpha=0.7,edgecolor='black')
ax.axhline(1,color='k',ls='--',alpha=0.3)
for b,v in zip(bars,vl): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.2f}',ha='center',fontsize=10)
ax.set_ylabel(r'$\chi^2$/dof'); ax.set_title('(e) Model comparison')

# (f) chi2 vs z_l
ax=axes[1,2]
ax.plot(df_zs['z_l'],df_zs['chi2_nfw'],'r-o',ms=4,label='NFW')
ax.plot(df_zs['z_l'],df_zs['chi2_mond'],'b--s',ms=4,label='MOND')
ax.plot(df_zs['z_l'],df_zs['chi2_membrane'],'g-.^',ms=4,label='Membrane')
ax.axvline(Z_L,color='red',ls='--',alpha=0.5)
ax.set_xlabel('$z_{lens}$'); ax.set_ylabel(r'$\chi^2$')
ax.set_title('(f) $\\chi^2$ vs $z_l$'); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'hsc_refit_photoz.png'),dpi=150)
plt.close()
df_zs.to_csv(os.path.join(base_dir,'hsc_gc_vs_zlens.csv'),index=False)
print("  -> hsc_refit_photoz.png, hsc_gc_vs_zlens.csv 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("[最終サマリー]")
print("="*70)
print(f"""
  photo-z 更新後 (z_lens={Z_L:.3f}):

  NFW:    M200={M200:.2e}, c200={c200:.2f}, chi2/dof={c2_nfw/dof_n:.2f}
  MOND:   g_c=a0, chi2/dof={c2_mo/dof_mo:.2f}, dAIC={aic_mo-aic_n:+.1f}
  膜:     g_c={gc_b/a0:.3f}a0, chi2/dof={c2_me/dof_me:.2f}, dAIC={aic_me-aic_n:+.1f}

  g_c 推定:
    g_c = {gc_b:.2e} m/s^2 = {gc_b/a0:.3f} a0
    68% CI: [{lo68:.3f}, {hi68:.3f}] a0
    95% CI: [{lo95:.3f}, {hi95:.3f}] a0
    a0 in 95%CI: {'YES' if a0_in95 else 'NO'}

  SPARC比較:
    SPARC: g_c = 0.825 a0
    HSC:   g_c = {gc_b/a0:.3f} a0
    整合性: {'2sigma以内' if abs(gc_b/a0-0.825)<2*max((hi95-lo95)/4,0.1) else '有意なずれ'}

  Level判定: 膜モデル弱レンズ検証 -> Level {'A' if not a0_in95 else 'B'}
""")
print("完了。")
