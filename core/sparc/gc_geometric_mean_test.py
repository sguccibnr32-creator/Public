#!/usr/bin/env python3
"""
Geometric mean hypothesis verification:
  g_c = eta * a0^(1-alpha) * (G*Sigma0)^alpha
"""
import csv,os,sys,numpy as np
from scipy.stats import linregress,spearmanr,pearsonr,t as t_dist
from scipy.stats import chi2 as chi2_dist
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
kpc_m=3.086e19
kms_ms=1e3

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
# Data loading
# ============================================================
print("="*70)
print("Geometric mean: g_c = eta * a0^(1-alpha) * (G*Sigma0)^alpha")
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

print(f"  sparc: {len(gal_info)}, gc: {len(gc_dict)}")

# Build physical quantities from rotation curves
results=[]
for name,info in gal_info.items():
    if name not in gc_dict: continue
    data=read_dat(name)
    if data is None or data.shape[0]<5: continue
    rad=data[:,0]; v_disk=data[:,4]; v_gas=data[:,3]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=info['ud']

    mask=rad>0.01; r=rad[mask]
    vd=v_disk[mask]; vg=v_gas[mask]; vb=v_bul[mask]
    if len(r)<5: continue

    vds=np.sqrt(ud)*np.abs(vd)
    V_pk=np.max(vds); i_pk=np.argmax(vds); r_pk=r[i_pk]
    if r_pk>=r.max()*0.9 or r_pk<0.01: continue
    h_R=r_pk/2.15

    vbar2=ud*np.sign(vd)*vd**2+np.sign(vg)*vg**2+np.sign(vb)*vb**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))
    V_pk_b=np.max(v_bar)
    disk_dom=V_pk/info['vflat']

    results.append({
        'name':name,'gc_a0':gc_dict[name],
        'vflat':info['vflat'],'ud':ud,'h_R':h_R,
        'V_peak_disk':V_pk,'disk_dom':disk_dom,
        'compact':V_pk_b/h_R,
    })

n=len(results)
print(f"  Valid galaxies: {n}")

# Arrays
gc_a0=np.array([d['gc_a0'] for d in results])
gc_si=gc_a0*a0_si
vflat_kms=np.array([d['vflat'] for d in results])
vflat_ms=vflat_kms*kms_ms
hR_kpc=np.array([d['h_R'] for d in results])
hR_m=hR_kpc*kpc_m
ud_arr=np.array([d['ud'] for d in results])
disk_dom=np.array([d['disk_dom'] for d in results])

# G*Sigma0 = v_flat^2/h_R [SI]
G_S0=vflat_ms**2/hR_m
log_gc=np.log10(gc_si)
log_GS=np.log10(G_S0)
log_a0v=np.log10(a0_si)
log_vf=np.log10(vflat_kms)
log_hR=np.log10(hR_kpc)
log_gc_a0=np.log10(gc_a0)

# ====================================================================
print(f"\n{'='*70}")
print("Test 1: Optimal alpha and confidence interval")
print("="*70)

# g_c = eta * a0^(1-alpha) * (G*S0)^alpha
# log(g_c) = log(eta) + (1-alpha)*log(a0) + alpha*log(G*S0)
# log(g_c) = [log(eta) + log(a0)] + alpha*[log(G*S0) - log(a0)]
# y = intercept + alpha * x,  where x = log(G*S0/a0)

x=log_GS-log_a0v  # log(G*Sigma0/a0)
sl,ic,r_x,p_x,se_x=linregress(x,log_gc)
alpha_opt=sl
eta_opt=10**(ic-log_a0v)

resid_B=log_gc-(sl*x+ic)
s2=np.sum(resid_B**2)/(n-2)
se_a=np.sqrt(s2/np.sum((x-np.mean(x))**2))
tc=t_dist.ppf(0.975,n-2)
ci=(alpha_opt-tc*se_a,alpha_opt+tc*se_a)

print(f"  Optimal alpha = {alpha_opt:.4f} +/- {se_a:.4f}")
print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  eta = {eta_opt:.4f}")
print(f"  r = {r_x:.4f}, R2 = {r_x**2:.4f}")
print(f"  Residual std = {np.std(resid_B):.4f} dex")

# Test specific alpha values
for label,a_test in [("0.0 (MOND: g_c=const)",0.0),
                      ("0.5 (geometric mean)",0.5),
                      ("1.0 (pure G*Sigma0)",1.0)]:
    t_s=(alpha_opt-a_test)/se_a
    p_v=2*t_dist.sf(abs(t_s),n-2)
    judge="cannot reject" if p_v>0.05 else "REJECTED (p<0.05)"
    print(f"\n  alpha = {label}:")
    print(f"    t = {t_s:.3f}, p = {p_v:.4f} -> {judge}")

p00=2*t_dist.sf(abs((alpha_opt-0.0)/se_a),n-2)
p05=2*t_dist.sf(abs((alpha_opt-0.5)/se_a),n-2)
p10=2*t_dist.sf(abs((alpha_opt-1.0)/se_a),n-2)

# ====================================================================
print(f"\n{'='*70}")
print("Test 2: Model comparison (AIC)")
print("="*70)

# A: MOND (g_c=a0, 0 params)
pred_A=np.full(n,log_a0v)
ss_A=np.sum((log_gc-pred_A)**2)
aic_A=n*np.log(ss_A/n)

# B: geometric mean alpha-free (2 params: alpha, eta)
pred_B=sl*x+ic
ss_B=np.sum(resid_B**2)
aic_B=n*np.log(ss_B/n)+2*2

# C: exact geometric mean alpha=0.5 fixed (1 param: eta)
pC_raw=0.5*log_GS+0.5*log_a0v
eC=np.median(log_gc-pC_raw)
pred_C=pC_raw+eC
ss_C=np.sum((log_gc-pred_C)**2)
aic_C=n*np.log(ss_C/n)+2*1

# D: free power g_c=(G*S0)^s (2 params: s, intercept)
sD,iD,rD,_,_=linregress(log_GS,log_gc)
pred_D=sD*log_GS+iD
ss_D=np.sum((log_gc-pred_D)**2)
aic_D=n*np.log(ss_D/n)+2*2

# E: 2-variable empirical (vflat, hR) + intercept = 3 params
X2=np.column_stack([log_vf,log_hR,np.ones(n)])
beta2,_,_,_=np.linalg.lstsq(X2,log_gc_a0,rcond=None)
pred_E_a0=X2@beta2
pred_E=pred_E_a0+log_a0v  # convert to SI
ss_E=np.sum((log_gc-pred_E)**2)
aic_E=n*np.log(ss_E/n)+2*3

models=[
    ("A: MOND (g_c=a0, 0p)",0,ss_A,aic_A),
    ("B: geom mean alpha-free (2p)",2,ss_B,aic_B),
    ("C: geom mean alpha=0.5 (1p)",1,ss_C,aic_C),
    ("D: free power (G*S0)^s (2p)",2,ss_D,aic_D),
    ("E: 2-var empirical (3p)",3,ss_E,aic_E),
]
print(f"\n  {'Model':<38s} {'k':>3s} {'RSS':>9s} {'AIC':>9s} {'dAIC':>7s}")
print(f"  {'-'*70}")
for nm,k,ss,aic in models:
    print(f"  {nm:<38s} {k:>3d} {ss:>9.2f} {aic:>9.1f} {aic-aic_A:>+7.1f}")
print(f"\n  dAIC < 0 -> better than MOND")

# ====================================================================
print(f"\n{'='*70}")
print("Test 3: Galaxy property bins (universality)")
print("="*70)

# Use v_flat bins as proxy for galaxy type
sort_vf=np.argsort(vflat_kms)
nper=max(n//5,1)
bin_labels=[]; bin_alphas=[]; bin_ns=[]

print(f"\n  {'v_flat bin':>14s}  {'N':>4s}  {'alpha':>7s}  {'+/-se':>7s}  {'r':>7s}")
print(f"  {'-'*45}")
for i in range(5):
    i0=i*nper; i1=(i+1)*nper if i<4 else n; idx=sort_vf[i0:i1]
    vmin,vmax=vflat_kms[idx].min(),vflat_kms[idx].max()
    label=f"{vmin:.0f}-{vmax:.0f}"
    if len(idx)<5: continue
    s_,i_,r_,_,se_=linregress(x[idx],log_gc[idx])
    print(f"  {label:>14s}  {len(idx):4d}  {s_:7.3f}  {se_:7.3f}  {r_:7.3f}")
    bin_labels.append(label); bin_alphas.append(s_); bin_ns.append(len(idx))

if len(bin_alphas)>2:
    av=np.std(bin_alphas)
    print(f"\n  alpha inter-bin std = {av:.3f}")
    if av<0.1:
        print("  -> alpha is approximately universal (no v_flat dependence)")
    else:
        print("  -> alpha shows v_flat dependence")

# Also bin by disk dominance
sort_dd=np.argsort(disk_dom)
print(f"\n  {'disk_dom bin':>14s}  {'N':>4s}  {'alpha':>7s}  {'+/-se':>7s}  {'r':>7s}")
print(f"  {'-'*45}")
dd_alphas=[]
for i in range(5):
    i0=i*nper; i1=(i+1)*nper if i<4 else n; idx=sort_dd[i0:i1]
    dmin,dmax=disk_dom[idx].min(),disk_dom[idx].max()
    label=f"{dmin:.2f}-{dmax:.2f}"
    if len(idx)<5: continue
    s_,i_,r_,_,se_=linregress(x[idx],log_gc[idx])
    print(f"  {label:>14s}  {len(idx):4d}  {s_:7.3f}  {se_:7.3f}  {r_:7.3f}")
    dd_alphas.append(s_)

if len(dd_alphas)>2:
    print(f"  alpha inter-bin std = {np.std(dd_alphas):.3f}")

# ====================================================================
print(f"\n{'='*70}")
print("Test 4: Residual structure")
print("="*70)

for lab,vals in [("log(v_flat)",log_vf),("log(h_R)",log_hR),
                  ("log(G*S0)",log_GS),("log(Ud)",np.log10(ud_arr)),
                  ("disk_dom",disk_dom)]:
    rho,p=spearmanr(vals,resid_B)
    sig=" *significant*" if p<0.05 else ""
    print(f"  resid vs {lab:<12s}: rho={rho:+.3f}, p={p:.2e}{sig}")

# ====================================================================
print(f"\n{'='*70}")
print("Test 5: Exact geometric mean g_c = eta*sqrt(a0*G*S0)")
print("="*70)

gc_gm=np.sqrt(a0_si*G_S0)
eta_gm=np.median(gc_si/gc_gm)
lgm=np.log10(eta_gm*gc_gm)
rgm=log_gc-lgm
r_gm=np.corrcoef(lgm,log_gc)[0,1]
r_mond=log_gc-log_a0v

print(f"  alpha=0.5 fixed:")
print(f"    eta = {eta_gm:.4f}")
print(f"    r = {r_gm:.4f}")
print(f"    residual std = {np.std(rgm):.4f} dex")
print(f"    |error| median = {np.median(np.abs(rgm)):.4f} dex")

print(f"\n  Residual std comparison:")
print(f"    MOND (g_c=a0):            {np.std(r_mond):.4f} dex")
print(f"    Geometric mean (a=0.5):   {np.std(rgm):.4f} dex")
print(f"    Optimal (a={alpha_opt:.3f}):      {np.std(resid_B):.4f} dex")

i05=(1-np.std(rgm)/np.std(r_mond))*100
ifr=(1-np.std(resid_B)/np.std(r_mond))*100
print(f"\n  Improvement over MOND:")
print(f"    alpha=0.5 fixed: {i05:.1f}%")
print(f"    alpha free:      {ifr:.1f}%")
print(f"    Extra from freeing alpha: {ifr-i05:.1f}%")

# ====================================================================
print(f"\n{'='*70}")
print("Generating plots...")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(r'$g_c = \eta\,a_0^{1-\alpha}\,(G\Sigma_0)^{\alpha}$  Verification',
             fontsize=14,fontweight='bold')

# (a) Main scatter with fits
ax=axes[0,0]
la0=log_gc_a0; lGa=np.log10(G_S0/a0_si)
ax.scatter(lGa,la0,s=15,alpha=0.5,c='steelblue')
xr=np.linspace(lGa.min(),lGa.max(),100)
ax.plot(xr,alpha_opt*xr+np.log10(eta_opt),'r-',lw=2,label=f'$\\alpha$={alpha_opt:.3f}')
ax.plot(xr,0.5*xr+eC-log_a0v,'g--',lw=2,alpha=0.7,label='$\\alpha$=0.5')
ax.plot(xr,xr,'k:',lw=1,alpha=0.3,label='$\\alpha$=1.0')
ax.axhline(0,color='gray',ls='--',lw=1,alpha=0.3,label='MOND ($g_c=a_0$)')
ax.set_xlabel(r'$\log(G\Sigma_0/a_0)$')
ax.set_ylabel(r'$\log(g_c/a_0)$')
ax.set_title(f'(a) $\\alpha$={alpha_opt:.3f}$\\pm${se_a:.3f}')
ax.legend(fontsize=8)

# (b) chi2 profile for alpha
ax=axes[0,1]
ar=np.linspace(0,1,200)
c2_arr=np.array([np.sum((log_gc-(a_*x+(np.mean(log_gc)-a_*np.mean(x))))**2) for a_ in ar])
ax.plot(ar,c2_arr-c2_arr.min(),'b-',lw=2)
ax.axhline(chi2_dist.ppf(0.95,1),color='r',ls='--',alpha=0.5,label='95% CL')
ax.axvline(alpha_opt,color='k',ls='-',alpha=0.3,label=f'best={alpha_opt:.3f}')
ax.axvline(0.5,color='g',ls='--',alpha=0.5,label='0.5')
ax.axvline(1.0,color='gray',ls=':',alpha=0.5,label='1.0')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$\Delta\chi^2$')
ax.set_title(r'(b) $\alpha$ confidence')
ax.set_xlim(0,1); ax.legend(fontsize=8)

# (c) v_flat bin alpha values
ax=axes[0,2]
if len(bin_alphas)>0:
    cols=plt.cm.viridis(np.linspace(0.2,0.8,len(bin_alphas)))
    bars=ax.bar(range(len(bin_labels)),bin_alphas,color=cols,alpha=0.7)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels,rotation=30,fontsize=8)
    ax.axhline(alpha_opt,color='r',ls='--',lw=2,label=f'global={alpha_opt:.3f}')
    ax.axhline(0.5,color='g',ls=':',lw=1.5,label='0.5')
    for i_,(b_,nn_) in enumerate(zip(bars,bin_ns)):
        ax.text(b_.get_x()+b_.get_width()/2,b_.get_height()+0.01,
                f'N={nn_}',ha='center',fontsize=7)
    ax.set_ylabel(r'$\alpha$')
    ax.set_title('(c) v_flat bin dependence')
    ax.legend(fontsize=8)

# (d) Observed vs predicted
ax=axes[1,0]
ax.scatter(pred_A,log_gc,s=10,alpha=0.2,c='gray',label='MOND')
ax.scatter(pred_C,log_gc,s=10,alpha=0.4,c='green',label='$\\alpha$=0.5')
ax.scatter(pred_B,log_gc,s=12,alpha=0.5,c='coral',label=f'$\\alpha$={alpha_opt:.3f}')
dg=np.linspace(log_gc.min(),log_gc.max(),100)
ax.plot(dg,dg,'k--',alpha=0.3)
ax.set_xlabel('predicted log($g_c$)')
ax.set_ylabel('observed log($g_c$)')
ax.set_title('(d) Obs vs Pred')
ax.legend(fontsize=8)

# (e) Residual distributions
ax=axes[1,1]
bins=np.linspace(-1,1,40)
ax.hist(r_mond,bins=bins,alpha=0.3,color='gray',
        label=f'MOND ({np.std(r_mond):.3f})')
ax.hist(rgm,bins=bins,alpha=0.3,color='green',
        label=f'a=0.5 ({np.std(rgm):.3f})')
ax.hist(resid_B,bins=bins,alpha=0.5,color='coral',
        label=f'a={alpha_opt:.3f} ({np.std(resid_B):.3f})')
ax.set_xlabel('Residual [dex]'); ax.set_ylabel('Count')
ax.set_title('(e) Residual distribution')
ax.legend(fontsize=8)

# (f) Residual structure
ax=axes[1,2]
rv,_=spearmanr(log_vf,resid_B)
rh,_=spearmanr(log_hR,resid_B)
ax.scatter(log_vf,resid_B,s=15,alpha=0.5,c='steelblue',
           label=f'v_flat ($\\rho$={rv:.3f})')
ax.scatter(log_hR,resid_B,s=15,alpha=0.3,c='coral',
           label=f'h_R ($\\rho$={rh:.3f})')
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xlabel('log(v_flat) or log(h_R)')
ax.set_ylabel('Residual [dex]')
ax.set_title('(f) Residual structure')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'gc_geometric_mean_verification.png'),dpi=150)
plt.close()
print("  -> gc_geometric_mean_verification.png saved")

# ====================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print(f"""
{'='*60}
  Hypothesis: g_c = eta * a0^(1-alpha) * (G*Sigma0)^alpha
{'='*60}

  alpha determination:
    Optimal = {alpha_opt:.4f} +/- {se_a:.4f}
    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]

  Hypothesis tests:
    alpha=0 (MOND):         p={p00:.2e} -> {'REJECTED' if p00<0.05 else 'cannot reject'}
    alpha=0.5 (geom mean):  p={p05:.4f} -> {'cannot reject' if p05>0.05 else 'REJECTED'}
    alpha=1.0 (G*S0 only):  p={p10:.2e} -> {'REJECTED' if p10<0.05 else 'cannot reject'}

  Improvement over MOND:
    Residual std: {np.std(r_mond):.3f} -> {np.std(resid_B):.3f} dex ({ifr:.1f}% improvement)
""")

# Conclusion
if p00<0.05:
    print("  [ESTABLISHED] MOND (g_c=const) is rejected. g_c depends on galaxy parameters.")
if p10<0.05:
    print("  [ESTABLISHED] Pure g_c ∝ G*Sigma0 (alpha=1) is rejected. a0 contribution persists.")
if p05>0.05:
    print("  [ESTABLISHED] Exact geometric mean g_c = eta*sqrt(a0*G*Sigma0) is consistent.")
    print("                -> Membrane critical acceleration is a geometric mix of")
    print("                   universal scale a0 and local surface density G*Sigma0.")
else:
    d="local baryon structure slightly dominant" if alpha_opt>0.5 else "membrane intrinsic scale slightly dominant"
    print(f"  [INDICATION] alpha={alpha_opt:.3f}: {d}")

if len(bin_alphas)>2:
    av=np.std(bin_alphas)
    if av<0.1:
        print(f"  [ESTABLISHED] alpha is universal (inter-bin std={av:.3f}).")
    else:
        print(f"  [INDICATION] alpha shows v_flat dependence (inter-bin std={av:.3f}).")

print(f"""
  U(epsilon;c) connection:
    Membrane effective potential U_eff = U_membrane + U_coupling(Sigma0)
    Critical point condition determines alpha.
    alpha~0.5 -> equipartition of membrane intrinsic and baryon coupling energy.
""")
print("Done.")
