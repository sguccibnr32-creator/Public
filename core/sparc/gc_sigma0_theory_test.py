#!/usr/bin/env python3
"""
g_c ∝ G·Sigma0·F(c) theory verification
=========================================
Theory:  g_c = eta · G · Sigma0 · h_R · 2(1 - sqrt(c))
         Sigma0 ∝ v_flat^2 / h_R

Tests:
  (1) g_c vs G·Sigma0 correlation
  (2) F(c) correction effect (continuous c estimation)
  (3) c(Sigma0) relation
  (4) Power index decomposition: g_c ∝ v_flat^a · h_R^b
"""
import csv,os,sys,numpy as np
from scipy.stats import linregress,spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
rotmod_dir=os.path.join(base_dir,'Rotmod_LTG')
csv_path=os.path.join(base_dir,'phase1','sparc_results.csv')

a0_unit=3703.0  # (km/s)^2/kpc
a0_si=1.2e-10   # m/s^2
G_si=6.674e-11  # m^3/(kg·s^2)
kpc_to_m=3.086e19
kms_to_ms=1e3
Msun=1.989e30

def load_csv(path):
    with open(path,"r",encoding="utf-8-sig") as fh:
        reader=csv.DictReader(fh); return reader.fieldnames,list(reader)

def read_dat(name):
    path=os.path.join(rotmod_dir,f"{name}_rotmod.dat")
    if os.path.exists(path):
        try: return np.loadtxt(path,comments='#')
        except: pass
    return None

# ============================================================
# Step 0: Load data
# ============================================================
print("="*70)
print("Step 0: Data loading")
print("="*70)

_,src_rows=load_csv(csv_path)
gal_info={}
for row in src_rows:
    name=row.get('galaxy','').strip()
    if not name: continue
    ud=float(row.get('ud','nan')); vf=float(row.get('vflat','nan'))
    if np.isfinite(ud) and np.isfinite(vf) and ud>0 and vf>0:
        gal_info[name]={'ud':ud,'vflat':vf}

gc_dict={}
gcp=os.path.join(base_dir,"TA3_gc_independent.csv")
if os.path.exists(gcp):
    _,gcr=load_csv(gcp)
    for row in gcr:
        n=row.get('galaxy','').strip()
        gc=float(row.get('gc_over_a0','nan'))
        if n and np.isfinite(gc) and gc>0: gc_dict[n]=gc

print(f"  sparc_results: {len(gal_info)} galaxies")
print(f"  TA3 gc: {len(gc_dict)} galaxies")

# ============================================================
# Step 1: Build physical quantities from rotation curves
# ============================================================
print(f"\n{'='*70}")
print("Step 1: Building physical quantities")
print("="*70)

results=[]
for name,info in gal_info.items():
    if name not in gc_dict: continue
    data=read_dat(name)
    if data is None or data.shape[0]<5: continue
    rad=data[:,0]; v_obs=data[:,1]; v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=info['ud']

    mask=rad>0.01; r=rad[mask]
    vd=v_disk[mask]; vg=v_gas[mask]; vb=v_bul[mask]
    if len(r)<5: continue

    # h_R from disk peak
    vds=np.sqrt(ud)*np.abs(vd)
    V_pk=np.max(vds); i_pk=np.argmax(vds); r_pk=r[i_pk]
    if r_pk>=r.max()*0.9 or r_pk<0.01: continue  # unreliable h_R
    h_R=r_pk/2.15

    # baryonic quantities
    vbar2=ud*np.sign(vd)*vd**2+np.sign(vg)*vg**2+np.sign(vb)*vb**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))
    V_pk_b=np.max(v_bar)
    disk_dom=V_pk/info['vflat']
    compact=V_pk_b/h_R

    results.append({
        'name':name,'gc_a0':gc_dict[name],
        'vflat':info['vflat'],'ud':ud,'h_R':h_R,
        'V_peak_disk':V_pk,'V_peak_bar':V_pk_b,
        'disk_dom':disk_dom,'compact':compact,
    })

N=len(results)
print(f"  Valid galaxies: {N}")

# Extract arrays
gc_a0=np.array([d['gc_a0'] for d in results])
gc_si=gc_a0*a0_si
vflat_kms=np.array([d['vflat'] for d in results])
vflat_ms=vflat_kms*kms_to_ms
hR_kpc=np.array([d['h_R'] for d in results])
hR_m=hR_kpc*kpc_to_m
ud_arr=np.array([d['ud'] for d in results])
disk_dom=np.array([d['disk_dom'] for d in results])

# ============================================================
# Step 2: Compute Sigma0 proxy and G·Sigma0
# ============================================================
print(f"\n{'='*70}")
print("Step 2: Physical quantities")
print("="*70)

# Sigma0 ∝ v_flat^2 / (G·h_R)
Sigma0_proxy=vflat_ms**2/(G_si*hR_m)  # [kg/m^2]
Sigma0_Msun_pc2=Sigma0_proxy/(Msun/(3.086e16)**2)

# G · Sigma0 [m/s^2] = v_flat^2 / h_R
G_Sigma0=G_si*Sigma0_proxy  # = v_flat^2 / h_R in SI

print(f"  Sigma0 proxy median: {np.median(Sigma0_Msun_pc2):.1f} Msun/pc^2")
print(f"  G·Sigma0 median: {np.median(G_Sigma0):.2e} m/s^2")
print(f"  g_c median: {np.median(gc_si):.2e} m/s^2")
print(f"  g_c/(G·Sigma0) median: {np.median(gc_si/G_Sigma0):.4f}")

# ============================================================
# Step 3: Test (1) - g_c vs G·Sigma0 basic correlation
# ============================================================
print(f"\n{'='*70}")
print("Step 3: g_c vs G·Sigma0 basic correlation")
print("="*70)

log_gc=np.log10(gc_si)
log_GSigma0=np.log10(G_Sigma0)

slope,intercept,r_val,p_val,se=linregress(log_GSigma0,log_gc)
print(f"  log(g_c) = {slope:.3f} * log(G·Sigma0) + {intercept:.3f}")
print(f"  r = {r_val:.3f}, R2 = {r_val**2:.3f}, p = {p_val:.2e}")
print(f"  Power index = {slope:.3f} (theory: 1.0)")

# Also in a0 units for comparison
log_gc_a0=np.log10(gc_a0)
log_vf=np.log10(vflat_kms)
log_hR=np.log10(hR_kpc)
# G·Sigma0 in galactic units = vflat^2/h_R
log_GSig_gal=2*log_vf-log_hR
slope_g,intercept_g,r_g,p_g,_=linregress(log_GSig_gal,log_gc_a0)
print(f"\n  Galactic units: log(gc/a0) = {slope_g:.3f} * log(v^2/h_R) + {intercept_g:.3f}")
print(f"  r = {r_g:.3f}, R2 = {r_g**2:.3f}")

rho_sp,p_sp=spearmanr(log_GSigma0,log_gc)
print(f"\n  Spearman: rho = {rho_sp:.3f}, p = {p_sp:.2e}")

# Compare with v^2/h_R^1.3
log_proxy13=np.log10(vflat_ms**2/hR_m**1.3)
_,_,r13,_,_=linregress(log_proxy13,log_gc)
print(f"  v^2/h_R^1.3: r = {r13:.3f}, R2 = {r13**2:.3f}")

# ============================================================
# Step 4: Test (2) - F(c) correction via continuous c estimation
# ============================================================
print(f"\n{'='*70}")
print("Step 4: F(c) correction (continuous c estimation)")
print("="*70)

# g_c = eta · G · Sigma0 · 2(1-sqrt(c))
# ratio = g_c / (G·Sigma0)
ratio=gc_si/G_Sigma0
eta_scale=np.percentile(ratio,95)/2  # calibrate from upper envelope
print(f"  eta_scale = {eta_scale:.4f} (from 95th percentile / 2)")

# Infer effective c: F(c_eff) = ratio / eta_scale
F_eff=ratio/eta_scale
F_eff_clipped=np.clip(F_eff,0.01,1.99)
c_eff=(1-F_eff_clipped/2)**2

print(f"  c_eff median = {np.median(c_eff):.3f}")
print(f"  c_eff mean = {np.mean(c_eff):.3f}")
print(f"  c_eff std = {np.std(c_eff):.3f}")
print(f"  c_eff [5th,95th] = [{np.percentile(c_eff,5):.3f}, {np.percentile(c_eff,95):.3f}]")

# F(c) corrected prediction
F_c_est=2*(1-np.sqrt(c_eff))
theory_Fc=G_Sigma0*F_c_est
log_theory_Fc=np.log10(np.maximum(theory_Fc,1e-20))
slope_fc,intercept_fc,r_fc,p_fc,_=linregress(log_theory_Fc,log_gc)
print(f"\n  F(c)-corrected: r = {r_fc:.3f}, R2 = {r_fc**2:.3f} (trivially perfect by construction)")

# Instead: test if c_eff correlates with observables
# This would indicate F(c) has physical meaning
log_Sigma0=np.log10(Sigma0_Msun_pc2)
rho_cS,p_cS=spearmanr(log_Sigma0,c_eff)
r_cS,p_cS_p=pearsonr(log_Sigma0,c_eff)
slope_cS,intercept_cS,_,_,_=linregress(log_Sigma0,c_eff)
print(f"\n  c_eff vs log(Sigma0):")
print(f"    Pearson r = {r_cS:.3f}, p = {p_cS_p:.2e}")
print(f"    Spearman rho = {rho_cS:.3f}, p = {p_cS:.2e}")
print(f"    c_eff = {slope_cS:.4f} * log(Sigma0) + {intercept_cS:.4f}")

# c_eff vs disk dominance
rho_cd,p_cd=spearmanr(disk_dom,c_eff)
print(f"  c_eff vs disk_dom: rho = {rho_cd:.3f}, p = {p_cd:.2e}")

# c_eff vs v_flat
rho_cv,p_cv=spearmanr(log_vf,c_eff)
print(f"  c_eff vs log(v_flat): rho = {rho_cv:.3f}, p = {p_cv:.2e}")

# c_eff vs Upsilon_d
rho_cu,p_cu=spearmanr(np.log10(ud_arr),c_eff)
print(f"  c_eff vs log(Ud): rho = {rho_cu:.3f}, p = {p_cu:.2e}")

# ============================================================
# Step 5: Test (3) - Better eta calibration via bins
# ============================================================
print(f"\n{'='*70}")
print("Step 5: Sigma0 bin analysis")
print("="*70)

sort_S=np.argsort(log_Sigma0); nper=max(N//5,1)
print(f"\n{'Sigma0 bin':>14s}  {'N':>4s}  {'gc/GSig':>10s}  {'c_eff':>8s}  {'gc_pred r':>10s}")
for i in range(5):
    i0=i*nper; i1=(i+1)*nper if i<4 else N; idx=sort_S[i0:i1]
    label=f"{10**log_Sigma0[idx].min():.0f}-{10**log_Sigma0[idx].max():.0f}"
    med_ratio=np.median(ratio[idx])
    med_c=np.median(c_eff[idx])
    # Within-bin prediction quality
    if len(idx)>5:
        rb,_=pearsonr(log_GSigma0[idx],log_gc[idx])
    else:
        rb=np.nan
    print(f"{label:>14s}  {len(idx):4d}  {med_ratio:10.4f}  {med_c:8.3f}  {rb:10.3f}")

# ============================================================
# Step 6: Test (4) - Power index decomposition
# ============================================================
print(f"\n{'='*70}")
print("Step 6: Power index decomposition")
print("="*70)

# log(gc/a0) = a*log(v_flat) + b*log(h_R) + const
X=np.column_stack([log_vf,log_hR,np.ones(N)])
beta,_,_,_=np.linalg.lstsq(X,log_gc_a0,rcond=None)
y_pred=X@beta
ss_res=np.sum((log_gc_a0-y_pred)**2)
ss_tot=np.sum((log_gc_a0-np.mean(log_gc_a0))**2)
R2_2var=1-ss_res/ss_tot

print(f"  log(gc/a0) = {beta[0]:.3f}*log(v_flat) + {beta[1]:.3f}*log(h_R) + {beta[2]:.3f}")
print(f"  R2 = {R2_2var:.3f}")
print(f"  -> gc ∝ v_flat^{beta[0]:.2f} · h_R^{beta[1]:.2f}")
print(f"  Theory (G·Sigma0 = v^2/h_R): v_flat^2.0 · h_R^-1.0")
print(f"  v_flat exponent deviation: {beta[0]-2:.3f}")
print(f"  h_R exponent deviation: {beta[1]-(-1):.3f}")

# Effective combined exponent for v^2/h_R
# If gc ∝ (v^2/h_R)^gamma, then a=2*gamma, b=-gamma
gamma_v=beta[0]/2
gamma_h=-beta[1]
print(f"\n  Implied gamma from v_flat: {gamma_v:.3f}")
print(f"  Implied gamma from h_R:    {gamma_h:.3f}")
print(f"  gamma mismatch: {abs(gamma_v-gamma_h):.3f}")
if abs(gamma_v-gamma_h)<0.1:
    print(f"  -> Consistent with gc ∝ (v^2/h_R)^{np.mean([gamma_v,gamma_h]):.2f}")
else:
    print(f"  -> v_flat and h_R have INDEPENDENT contributions (not pure Sigma0)")

# Add Ud as 3rd variable
X3=np.column_stack([log_vf,log_hR,np.log10(ud_arr),np.ones(N)])
beta3,_,_,_=np.linalg.lstsq(X3,log_gc_a0,rcond=None)
y_pred3=X3@beta3
R2_3var=1-np.sum((log_gc_a0-y_pred3)**2)/ss_tot

print(f"\n  3-variable (+ Ud):")
print(f"  log(gc/a0) = {beta3[0]:.3f}*log(vf) + {beta3[1]:.3f}*log(hR) + {beta3[2]:.3f}*log(Ud) + {beta3[3]:.3f}")
print(f"  R2 = {R2_3var:.3f} (dR2 = {R2_3var-R2_2var:+.4f})")

# ============================================================
# Step 7: Comparison with empirical 6-var model
# ============================================================
print(f"\n{'='*70}")
print("Step 7: Theory vs empirical prediction")
print("="*70)

# Empirical model (from gc_predictive_model.py)
COEFFS={'intercept':-2.175,'log_vflat':2.015,'log_Vpeak':-0.046,
        'log_hR':-1.294,'disk_dom':0.138,'log_compact':-1.024,'log_Ud':-0.217}

V_pk_arr=np.array([d['V_peak_disk'] for d in results])
compact_arr=np.array([d['compact'] for d in results])
gc_emp=10**(COEFFS['intercept']+COEFFS['log_vflat']*log_vf
            +COEFFS['log_Vpeak']*np.log10(np.maximum(V_pk_arr,0.1))
            +COEFFS['log_hR']*log_hR
            +COEFFS['disk_dom']*disk_dom
            +COEFFS['log_compact']*np.log10(np.maximum(compact_arr,0.01))
            +COEFFS['log_Ud']*np.log10(np.maximum(ud_arr,0.01)))

r_emp,_=pearsonr(np.log10(gc_emp),log_gc_a0)
err_emp=np.abs(np.log10(gc_emp)-log_gc_a0)

# G·Sigma0 model (1-variable)
gc_GSig=10**(slope_g*log_GSig_gal+intercept_g)
r_GSig,_=pearsonr(np.log10(gc_GSig),log_gc_a0)
err_GSig=np.abs(np.log10(gc_GSig)-log_gc_a0)

# 2-variable model
gc_2var=10**y_pred
r_2var,_=pearsonr(np.log10(gc_2var),log_gc_a0)
err_2var=np.abs(np.log10(gc_2var)-log_gc_a0)

# g_c=a0 baseline
err_a0=np.abs(log_gc_a0)

print(f"  {'Model':>25s}  {'r':>6s}  {'err_med':>8s}  {'params':>6s}")
print(f"  {'g_c=a0 (MOND)':>25s}  {'--':>6s}  {np.median(err_a0):8.3f}  {'0':>6s}")
print(f"  {'G·Sigma0 (1-var)':>25s}  {r_GSig:6.3f}  {np.median(err_GSig):8.3f}  {'1':>6s}")
print(f"  {'v_flat + h_R (2-var)':>25s}  {r_2var:6.3f}  {np.median(err_2var):8.3f}  {'2':>6s}")
print(f"  {'Empirical (6-var)':>25s}  {r_emp:6.3f}  {np.median(err_emp):8.3f}  {'6':>6s}")

# Is G·Sigma0 enough, or is the 6-var model necessary?
print(f"\n  G·Sigma0 captures {(1-np.median(err_GSig)/np.median(err_a0))*100:.0f}% of a0 error")
print(f"  6-var captures    {(1-np.median(err_emp)/np.median(err_a0))*100:.0f}% of a0 error")
print(f"  2-var captures    {(1-np.median(err_2var)/np.median(err_a0))*100:.0f}% of a0 error")

# ============================================================
# Step 8: Plots
# ============================================================
print(f"\n{'='*70}")
print("Step 8: Generating plots")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(r'$g_c \propto G \cdot \Sigma_0 \cdot F(c)$ Theory Verification',fontsize=14,fontweight='bold')

# (a) g_c vs G·Sigma0
ax=axes[0,0]
ax.scatter(log_GSigma0,log_gc,s=15,alpha=0.5,c='steelblue')
x_range=np.linspace(log_GSigma0.min(),log_GSigma0.max(),100)
ax.plot(x_range,slope*x_range+intercept,'r-',lw=2,
        label=f'fit: slope={slope:.2f}, r={r_val:.3f}')
ax.plot(x_range,x_range+np.log10(np.median(ratio)),'k--',lw=1,alpha=0.5,
        label='slope=1 ref')
ax.set_xlabel(r'$\log(G \cdot \Sigma_0)$ [m/s$^2$]')
ax.set_ylabel(r'$\log(g_c)$ [m/s$^2$]')
ax.set_title(f'(a) $g_c$ vs $G \\cdot \\Sigma_0$ (r={r_val:.3f})')
ax.legend(fontsize=8)

# (b) Residuals vs Sigma0
ax=axes[0,1]
resid=log_gc-(slope*log_GSigma0+intercept)
ax.scatter(log_Sigma0,resid,s=15,alpha=0.5,c='steelblue')
ax.axhline(0,color='k',ls='--',alpha=0.3)
rho_res,_=spearmanr(log_Sigma0,resid)
ax.set_xlabel(r'$\log(\Sigma_0)$ [Msun/pc$^2$]')
ax.set_ylabel('Residual [dex]')
ax.set_title(f'(b) Residuals vs $\\Sigma_0$ (rho={rho_res:.3f}, std={np.std(resid):.3f})')

# (c) c_eff vs Sigma0
ax=axes[0,2]
sc=ax.scatter(log_Sigma0,c_eff,s=15,alpha=0.5,c=log_vf,cmap='viridis')
plt.colorbar(sc,ax=ax,label=r'$\log(v_{flat})$')
x_fit=np.linspace(log_Sigma0.min(),log_Sigma0.max(),100)
ax.plot(x_fit,slope_cS*x_fit+intercept_cS,'r-',lw=2,alpha=0.7)
ax.set_xlabel(r'$\log(\Sigma_0)$ [Msun/pc$^2$]')
ax.set_ylabel(r'$c_{eff}$ (inferred)')
ax.set_title(f'(c) $c(\\Sigma_0)$ (rho={rho_cS:.3f})')

# (d) Power index: observed vs theory
ax=axes[1,0]
# Show g_c vs v^2/h_R in galactic units
ax.scatter(log_GSig_gal,log_gc_a0,s=15,alpha=0.5,c='steelblue')
x_r=np.linspace(log_GSig_gal.min(),log_GSig_gal.max(),100)
ax.plot(x_r,slope_g*x_r+intercept_g,'r-',lw=2,label=f'slope={slope_g:.2f}')
ax.plot(x_r,1.0*x_r+(intercept_g+(slope_g-1)*np.mean(log_GSig_gal)),'k--',lw=1,alpha=0.5,label='slope=1')
ax.set_xlabel(r'$\log(v_{flat}^2/h_R)$ [(km/s)$^2$/kpc]')
ax.set_ylabel(r'$\log(g_c/a_0)$')
ax.set_title(f'(d) Galactic units (slope={slope_g:.2f})')
ax.legend(fontsize=8)

# (e) Prediction comparison
ax=axes[1,1]
models=[
    ('$G \\cdot \\Sigma_0$',np.log10(gc_GSig),'steelblue',r_GSig),
    ('2-var',np.log10(gc_2var),'green',r_2var),
    ('6-var emp.',np.log10(gc_emp),'coral',r_emp),
]
for label,pred,col,rv in models:
    ax.scatter(pred,log_gc_a0,s=10,alpha=0.3,c=col,label=f'{label} (r={rv:.3f})')
diag=np.linspace(log_gc_a0.min(),log_gc_a0.max(),100)
ax.plot(diag,diag,'k--',alpha=0.3)
ax.set_xlabel(r'$\log(g_c/a_0)$ predicted')
ax.set_ylabel(r'$\log(g_c/a_0)$ observed')
ax.set_title('(e) Prediction comparison')
ax.legend(fontsize=7)

# (f) Error distributions
ax=axes[1,2]
ax.hist(err_a0,bins=30,alpha=0.4,color='gray',label=f'$g_c=a_0$ ({np.median(err_a0):.3f})')
ax.hist(err_GSig,bins=30,alpha=0.4,color='steelblue',label=f'$G\\cdot\\Sigma_0$ ({np.median(err_GSig):.3f})')
ax.hist(err_2var,bins=30,alpha=0.4,color='green',label=f'2-var ({np.median(err_2var):.3f})')
ax.hist(err_emp,bins=30,alpha=0.4,color='coral',label=f'6-var ({np.median(err_emp):.3f})')
ax.set_xlabel('|error| [dex]')
ax.set_ylabel('Count')
ax.set_title('(f) Prediction error distribution')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'gc_sigma0_theory_verification.png'),dpi=150)
plt.close()
print("  -> gc_sigma0_theory_verification.png saved")

# ============================================================
# Step 9: Summary
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY: g_c ∝ G·Sigma0·F(c) verification")
print("="*70)

print(f"""
Test (1): g_c vs G·Sigma0 basic correlation
  Pearson r  = {r_val:.3f}
  Spearman rho = {rho_sp:.3f}
  Power index = {slope:.3f} (theory: 1.0)
  -> {'Strong correlation' if abs(r_val)>0.6 else 'Moderate' if abs(r_val)>0.4 else 'Weak'}

Test (2): c_eff correlates with observables?
  c_eff vs Sigma0:  rho = {rho_cS:.3f}, p = {p_cS:.2e}
  c_eff vs disk_dom: rho = {rho_cd:.3f}, p = {p_cd:.2e}
  c_eff vs v_flat:  rho = {rho_cv:.3f}, p = {p_cv:.2e}
  c_eff vs Ud:      rho = {rho_cu:.3f}, p = {p_cu:.2e}

Test (3): Power index decomposition
  Observed: g_c ∝ v_flat^{beta[0]:.2f} · h_R^{beta[1]:.2f}
  Theory:   g_c ∝ v_flat^2.0 · h_R^-1.0
  gamma from v_flat: {gamma_v:.3f}
  gamma from h_R:    {gamma_h:.3f}

Test (4): Model hierarchy
  g_c=a0:      err_med = {np.median(err_a0):.3f} dex
  G·Sigma0:    err_med = {np.median(err_GSig):.3f} dex (captures {(1-np.median(err_GSig)/np.median(err_a0))*100:.0f}%)
  2-var:       err_med = {np.median(err_2var):.3f} dex (captures {(1-np.median(err_2var)/np.median(err_a0))*100:.0f}%)
  6-var emp:   err_med = {np.median(err_emp):.3f} dex (captures {(1-np.median(err_emp)/np.median(err_a0))*100:.0f}%)
""")

# Key conclusion
if abs(gamma_v-gamma_h)<0.15 and abs(r_val)>0.6:
    print("  CONCLUSION: g_c ∝ G·Sigma0 is a good first-order description.")
    print(f"              Effective exponent gamma={np.mean([gamma_v,gamma_h]):.2f} (theory: 1.0)")
    if abs(np.mean([gamma_v,gamma_h])-1.0)<0.15:
        print("              Power index consistent with theory at ~15% level.")
    else:
        print(f"              Power index deviates from 1.0 by {abs(np.mean([gamma_v,gamma_h])-1.0):.2f}")
elif abs(r_val)>0.6:
    print("  CONCLUSION: g_c correlates with G·Sigma0 but v_flat and h_R")
    print("              have independent contributions beyond pure Sigma0.")
else:
    print("  CONCLUSION: g_c vs G·Sigma0 correlation is weak.")
    print("              Theory requires additional physical mechanisms.")

print("\nDone.")
