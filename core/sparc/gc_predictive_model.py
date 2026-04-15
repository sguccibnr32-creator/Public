#!/usr/bin/env python3
"""
g_c predictive model from galaxy properties (no r_s used)
"""
import csv,os,sys,numpy as np
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
rotmod_dir=os.path.join(base_dir,'Rotmod_LTG')
csv_path=os.path.join(base_dir,'phase1','sparc_results.csv')

a0_unit=3703.0

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
# Data collection
# ============================================================
print("="*70)
print("Data collection")
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

results=[]
for name,info in gal_info.items():
    if name not in gc_dict: continue
    data=read_dat(name)
    if data is None or data.shape[0]<5: continue
    rad=data[:,0]; v_obs=data[:,1]; v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=info['ud']

    mask=rad>0.01; r=rad[mask]
    vd=np.sqrt(ud)*np.abs(v_disk[mask]); vg=np.abs(v_gas[mask])
    vbar2=ud*np.sign(v_disk[mask])*v_disk[mask]**2+np.sign(v_gas[mask])*v_gas[mask]**2+np.sign(v_bul[mask])*v_bul[mask]**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))

    V_pk_d=np.max(vd); i_pk=np.argmax(vd); r_pk=r[i_pk]
    if r_pk>=r.max()*0.9 or r_pk<0.01: continue
    h_R=r_pk/2.15
    V_pk_b=np.max(v_bar)
    Sigma0=V_pk_d**2/h_R
    gas_frac=np.mean(vg**2/np.maximum(v_bar**2,0.01))
    M_bar=info['vflat']**4/a0_unit
    disk_dom=V_pk_d/info['vflat']
    compact=V_pk_b/h_R
    gN_max=np.max(v_bar**2/r)/a0_unit

    results.append({
        'name':name,'gc':gc_dict[name],'log_gc':np.log10(gc_dict[name]),
        'vflat':info['vflat'],'ud':ud,'h_R':h_R,
        'V_peak_disk':V_pk_d,'V_peak_bar':V_pk_b,
        'Sigma0':Sigma0,'gas_frac':gas_frac,'M_bar':M_bar,
        'disk_dom':disk_dom,'compactness':compact,'gN_max':gN_max,
    })

N=len(results)
print(f"Analyzed: {N}")

gc=np.array([d['gc'] for d in results])
log_gc=np.array([d['log_gc'] for d in results])

predictors={
    'log_vflat':np.log10(np.array([d['vflat'] for d in results])),
    'log_hR':np.log10(np.array([d['h_R'] for d in results])),
    'log_Sigma0':np.log10(np.array([d['Sigma0'] for d in results])),
    'log_Vpeak':np.log10(np.array([d['V_peak_disk'] for d in results])),
    'gas_frac':np.array([d['gas_frac'] for d in results]),
    'log_Mbar':np.log10(np.array([d['M_bar'] for d in results])),
    'log_Ud':np.log10(np.array([d['ud'] for d in results])),
    'disk_dom':np.array([d['disk_dom'] for d in results]),
    'log_compact':np.log10(np.array([d['compactness'] for d in results])),
    'log_gNmax':np.log10(np.maximum(np.array([d['gN_max'] for d in results]),1e-4)),
}

# ============================================================
# Single-variable correlations
# ============================================================
print(f"\n{'='*70}")
print("Single-variable correlations")
print(f"{'='*70}")
print(f"\n{'Variable':>15s}  {'Pearson r':>10s}  {'p':>10s}")
single_results={}
for pname,parr in predictors.items():
    mask=np.isfinite(parr)&np.isfinite(log_gc)
    if mask.sum()<20: continue
    rp,pp=pearsonr(parr[mask],log_gc[mask])
    sig='***' if pp<0.001 else '**' if pp<0.01 else '*' if pp<0.05 else ''
    print(f"{pname:>15s}  {rp:+10.3f}  {pp:10.2e} {sig}")
    single_results[pname]={'r':rp,'p':pp}

# Best single
best_single=max(single_results.items(),key=lambda x:abs(x[1]['r']))
best_name=best_single[0]; best_arr=predictors[best_name]
mask_bs=np.isfinite(best_arr)&np.isfinite(log_gc)
p_best=np.polyfit(best_arr[mask_bs],log_gc[mask_bs],1)
pred_best=np.polyval(p_best,best_arr[mask_bs])
R2_best=1-np.sum((log_gc[mask_bs]-pred_best)**2)/np.sum((log_gc[mask_bs]-np.mean(log_gc[mask_bs]))**2)
print(f"\nBest single: {best_name} (r={best_single[1]['r']:.3f}, R2={R2_best:.4f})")

# ============================================================
# Stepwise regression
# ============================================================
print(f"\n{'='*70}")
print("Stepwise regression")
print(f"{'='*70}")

sorted_preds=sorted(single_results.items(),key=lambda x:abs(x[1]['r']),reverse=True)
pred_names_sorted=[x[0] for x in sorted_preds]

common_mask=np.ones(N,dtype=bool)
for pname in pred_names_sorted[:6]:
    common_mask&=np.isfinite(predictors[pname])
common_mask&=np.isfinite(log_gc)
N_common=common_mask.sum()
y=log_gc[common_mask]

print(f"\nCommon valid: {N_common}/{N}")
print(f"\n{'nvar':>5s}  {'added':>15s}  {'R2':>8s}  {'dR2':>8s}  {'BIC':>10s}  {'std':>8s}")

best_bic=np.inf; best_model=None; prev_R2=0; best_combo=None

for n_vars in range(1,min(7,len(pred_names_sorted)+1)):
    if n_vars==1:
        bc=None; br2=-1
        for pname in pred_names_sorted:
            X=np.column_stack([np.ones(N_common),predictors[pname][common_mask]])
            b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
            pred=X@b; R2=1-np.sum((y-pred)**2)/np.sum((y-np.mean(y))**2)
            if R2>br2: br2=R2; bc=[pname]; bb=b; bp=pred
    else:
        pv=best_combo.copy(); br2=-1
        for pname in pred_names_sorted:
            if pname in pv: continue
            tv=pv+[pname]
            X=np.column_stack([np.ones(N_common)]+[predictors[v][common_mask] for v in tv])
            b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
            pred=X@b; R2=1-np.sum((y-pred)**2)/np.sum((y-np.mean(y))**2)
            if R2>br2: br2=R2; bc=tv; bb=b; bp=pred

    best_combo=bc; k=n_vars+1
    rss=np.sum((y-bp)**2); bic=N_common*np.log(rss/N_common)+k*np.log(N_common)
    rstd=np.std(y-bp); dR2=br2-prev_R2
    added=bc[-1] if n_vars>0 else bc[0]
    print(f"{n_vars:5d}  {added:>15s}  {br2:8.4f}  {dR2:+8.4f}  {bic:10.2f}  {rstd:8.4f}")

    if bic<best_bic:
        best_bic=bic
        best_model={'vars':bc.copy(),'coeffs':bb.copy(),'R2':br2,'bic':bic,'residual_std':rstd,'pred':bp.copy()}
    prev_R2=br2

bm=best_model
print(f"\nBest model (BIC): vars={bm['vars']}, R2={bm['R2']:.4f}, BIC={bm['bic']:.2f}")
eq=f"log(g_c/a0) = {bm['coeffs'][0]:.3f}"
for i,v in enumerate(bm['vars']): eq+=f" {bm['coeffs'][i+1]:+.3f}*{v}"
print(f"  {eq}")

# vs g_c=a0
rss_const=np.sum(y**2); bic_const=N_common*np.log(rss_const/N_common)
rss_model=np.sum((y-bm['pred'])**2)
print(f"\ng_c=a0: RSS={rss_const:.2f}, BIC={bic_const:.2f}")
print(f"Model:  RSS={rss_model:.2f}, BIC={bm['bic']:.2f}")
print(f"dBIC = {bm['bic']-bic_const:.2f}")

err_model=np.abs(y-bm['pred']); err_a0=np.abs(y)
print(f"\nPrediction error: model={np.median(err_model):.3f} dex, a0={np.median(err_a0):.3f} dex")
print(f"Improvement: {(1-np.median(err_model)/np.median(err_a0))*100:.0f}%")

# ============================================================
# Plots
# ============================================================
fig,axes=plt.subplots(2,3,figsize=(16,10))

ax=axes[0,0]
ba=best_arr[mask_bs]
ax.scatter(ba,log_gc[mask_bs],s=10,alpha=0.4,c='steelblue')
xx=np.linspace(ba.min(),ba.max(),100)
ax.plot(xx,np.polyval(p_best,xx),'r-',lw=1.5)
ax.set_xlabel(best_name); ax.set_ylabel('log(g_c/a0)')
ax.set_title(f'(a) Best single: {best_name} (r={best_single[1]["r"]:.3f})')

ax=axes[0,1]
ax.scatter(bm['pred'],y,s=10,alpha=0.4,c='green')
ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
ax.set_title(f'(b) Best model (R2={bm["R2"]:.3f})')

ax=axes[0,2]
vf_cm=np.array([d['vflat'] for d in results])[common_mask]
resid=y-bm['pred']
ax.scatter(vf_cm,resid,s=10,alpha=0.4,c='orange')
ax.axhline(0,color='black',ls='--')
rho_r,_=spearmanr(vf_cm,resid)
ax.set_xlabel('v_flat'); ax.set_ylabel('Residual')
ax.set_title(f'(c) Residual vs v_flat (rho={rho_r:.3f})')

ax=axes[1,0]
gc_cm=gc[common_mask]; gc_pred=10**bm['pred']
ax.scatter(vf_cm,gc_cm,s=10,alpha=0.4,c='steelblue',label='Measured')
ax.scatter(vf_cm,gc_pred,s=10,alpha=0.4,c='red',label='Predicted')
ax.axhline(1.0,color='gray',ls='--',label='g_c=a0')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('v_flat'); ax.set_ylabel('g_c/a0')
ax.set_title('(d) g_c: measured vs predicted vs a0'); ax.legend(fontsize=7)

ax=axes[1,1]
sr=sorted(single_results.items(),key=lambda x:abs(x[1]['r']),reverse=True)[:8]
names_b=[x[0] for x in sr]; r_vals=[x[1]['r'] for x in sr]
colors_b=['steelblue' if r>0 else 'coral' for r in r_vals]
ax.barh(range(len(names_b)),r_vals,color=colors_b)
ax.set_yticks(range(len(names_b))); ax.set_yticklabels(names_b,fontsize=8)
ax.set_xlabel('Pearson r'); ax.set_title('(e) Predictor ranking')
ax.axvline(0,color='black',lw=0.5)

ax=axes[1,2]
ax.hist(err_model,bins=30,alpha=0.5,color='green',label=f'Model (med={np.median(err_model):.3f})')
ax.hist(err_a0,bins=30,alpha=0.5,color='gray',label=f'g_c=a0 (med={np.median(err_a0):.3f})')
ax.set_xlabel('|error| [dex]'); ax.set_ylabel('Count')
ax.set_title('(f) Prediction error'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'gc_predictive_model.png'),dpi=150)
plt.close()
print(f"\n[SAVED] gc_predictive_model.png")

# Final
print(f"\n{'='*70}")
print("Conclusion")
print(f"{'='*70}")
print(f"""
  Best predictor: {best_name} (r={best_single[1]['r']:.3f})
  Best model: {eq}
  R2={bm['R2']:.4f}, residual std={bm['residual_std']:.4f} dex
  dBIC vs g_c=a0: {bm['bic']-bic_const:.2f}
  Error improvement: {(1-np.median(err_model)/np.median(err_a0))*100:.0f}%
""")
