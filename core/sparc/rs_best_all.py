#!/usr/bin/env python3
"""
Global chi2 profile scan for all 175 galaxies -> rs_best
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

def load_csv(path):
    with open(path,"r",encoding="utf-8-sig") as fh:
        reader=csv.DictReader(fh); return reader.fieldnames,list(reader)

def read_dat(name):
    path=os.path.join(rotmod_dir,f"{name}_rotmod.dat")
    if os.path.exists(path):
        try: return np.loadtxt(path,comments='#')
        except: pass
    return None

_,src_rows=load_csv(csv_path)
galaxies=[]
for row in src_rows:
    name=row.get('galaxy','').strip()
    if not name: continue
    rs1=float(row.get('rs1','nan')); rs2=float(row.get('rs2','nan'))
    vf=float(row.get('vflat','nan')); ud=float(row.get('ud','nan'))
    if all(np.isfinite([rs1,rs2,vf])) and vf>0:
        if not np.isfinite(ud) or ud<=0: ud=0.5
        galaxies.append({'name':name,'rs1':rs1,'rs2':rs2,'vflat':vf,'upsilon_d':ud})

print(f"[INFO] {len(galaxies)} galaxies")

def chi2_at_rs(rs,rad,v_obs,err_v,v_bar):
    T=np.ones_like(rad) if rs<=0.001 else 0.5*(1+np.tanh((rad-rs)/max(rs,0.001)))
    w=1.0/np.maximum(err_v,1.0)**2
    vf_max=max(v_obs.max()*1.5,50)
    best_c2,best_vf=np.inf,0
    # Coarse grid
    for vf_t in np.linspace(1,vf_max,80):
        vm=np.sqrt(np.maximum(v_bar**2+vf_t**2*T,0.01))
        c2=np.sum(w*(v_obs-vm)**2)
        if c2<best_c2: best_c2=c2; best_vf=vf_t
    # Refine
    for vf_t in np.linspace(max(best_vf-vf_max/40,0.1),best_vf+vf_max/40,40):
        vm=np.sqrt(np.maximum(v_bar**2+vf_t**2*T,0.01))
        c2=np.sum(w*(v_obs-vm)**2)
        if c2<best_c2: best_c2=c2; best_vf=vf_t
    return best_c2,best_vf

def find_rs_best(rad,v_obs,err_v,v_bar,R_max):
    rs_grid=np.sort(np.unique(np.concatenate([
        np.array([0.001,0.005,0.01,0.02,0.05,0.1]),
        np.linspace(0.2,min(R_max*1.5,80),80),
    ])))
    chi2_l,vf_l=[],[]
    for rs in rs_grid:
        c2,vf=chi2_at_rs(rs,rad,v_obs,err_v,v_bar)
        chi2_l.append(c2); vf_l.append(vf)
    chi2_a=np.array(chi2_l); vf_a=np.array(vf_l)
    i_min=np.argmin(chi2_a)
    rs_b=rs_grid[i_min]; c2_min=chi2_a[i_min]; vf_b=vf_a[i_min]
    # Refine
    if 1<i_min<len(rs_grid)-1:
        rs_fine=np.linspace(rs_grid[max(i_min-3,0)],rs_grid[min(i_min+3,len(rs_grid)-1)],30)
        for rs in rs_fine:
            c2,vf=chi2_at_rs(rs,rad,v_obs,err_v,v_bar)
            if c2<c2_min: c2_min=c2; rs_b=rs; vf_b=vf
    c2_0,_=chi2_at_rs(0.001,rad,v_obs,err_v,v_bar)
    return {
        'rs_best':rs_b,'vf_best':vf_b,'chi2_min':c2_min,
        'chi2_dof':c2_min/max(len(rad)-2,1),'chi2_at_0':c2_0,
        'improvement':(c2_0-c2_min)/max(c2_0,0.1),'N_pts':len(rad),
    }

# ============================================================
# Main loop
# ============================================================
print(f"\nScanning all galaxies...")
results=[]
for i,gal in enumerate(galaxies):
    data=read_dat(gal['name'])
    if data is None or data.shape[0]<3: continue
    rad=data[:,0]; v_obs=data[:,1]; err_v=data[:,2]
    v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=gal['upsilon_d']
    vbar2=ud*np.sign(v_disk)*v_disk**2+np.sign(v_gas)*v_gas**2+np.sign(v_bul)*v_bul**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))
    R_max=rad.max(); r_min=rad[rad>0].min() if np.any(rad>0) else 0.01
    fit=find_rs_best(rad,v_obs,err_v,v_bar,R_max)
    results.append({
        'name':gal['name'],'vflat':gal['vflat'],
        'rs1':gal['rs1'],'rs2':gal['rs2'],'upsilon_d':ud,
        'R_max':R_max,'r_min':r_min,**fit,
    })
    if (i+1)%25==0: print(f"  {i+1}/{len(galaxies)}...")

N=len(results)
print(f"\nDone: {N} galaxies")

# Arrays
rs_best=np.array([d['rs_best'] for d in results])
rs1=np.array([d['rs1'] for d in results])
rs2=np.array([d['rs2'] for d in results])
vf=np.array([d['vflat'] for d in results])
Rmax=np.array([d['R_max'] for d in results])
improve=np.array([d['improvement'] for d in results])

# ============================================================
# Scaling analysis
# ============================================================
print(f"\n{'='*70}")
print("rs_best scaling")
print(f"{'='*70}")

print(f"\nrs_best: median={np.median(rs_best):.3f}, mean={np.mean(rs_best):.3f}, CV={np.std(rs_best)/np.mean(rs_best):.3f}")

mask_pos=(rs_best>0.01)&(vf>0)
lv=np.log10(vf[mask_pos])
lrb=np.log10(rs_best[mask_pos])
lrs1=np.log10(np.maximum(rs1[mask_pos],0.001))
lrs2=np.log10(np.maximum(rs2[mask_pos],0.001))

p_best=np.polyfit(lv,lrb,1); r_best,_=pearsonr(lv,lrb)
rho_best,_=spearmanr(vf[mask_pos],rs_best[mask_pos])
p_rs1=np.polyfit(lv,lrs1,1); r_rs1,_=pearsonr(lv,lrs1)
p_rs2=np.polyfit(lv,lrs2,1); r_rs2,_=pearsonr(lv,lrs2)

print(f"\n{'Metric':>12s}  {'alpha':>7s}  {'r':>7s}  {'median[kpc]':>12s}")
print(f"{'rs_best':>12s}  {p_best[0]:7.3f}  {r_best:7.3f}  {np.median(rs_best):12.3f}")
print(f"{'rs1':>12s}  {p_rs1[0]:7.3f}  {r_rs1:7.3f}  {np.median(rs1):12.3f}")
print(f"{'rs2':>12s}  {p_rs2[0]:7.3f}  {r_rs2:7.3f}  {np.median(rs2):12.3f}")

rho_Rmax,p_Rmax=spearmanr(rs_best[mask_pos],Rmax[mask_pos])
print(f"\nrs_best vs R_max: rho={rho_Rmax:.3f} (p={p_Rmax:.2e})")
print(f"rs_best/R_max median: {np.median(rs_best/Rmax):.3f}")

# Comparison
diff1=np.abs(rs_best-rs1); diff2=np.abs(rs_best-rs2)
close1=np.sum(diff1<diff2); close2=np.sum(diff2<diff1)
print(f"\nrs_best closer to rs1: {close1} ({100*close1/N:.0f}%)")
print(f"rs_best closer to rs2: {close2} ({100*close2/N:.0f}%)")

m1=rs1>0.01; m2=rs2>0.01
print(f"rs_best/rs1 median: {np.median(rs_best[m1]/rs1[m1]):.3f}")
print(f"rs_best/rs2 median: {np.median(rs_best[m2]/rs2[m2]):.3f}")

r_b1,_=pearsonr(np.log10(np.maximum(rs_best,0.01)),np.log10(np.maximum(rs1,0.001)))
r_b2,_=pearsonr(np.log10(np.maximum(rs_best,0.01)),np.log10(np.maximum(rs2,0.001)))
print(f"log corr rs_best vs rs1: {r_b1:.3f}")
print(f"log corr rs_best vs rs2: {r_b2:.3f}")

print(f"\nImprovement over rs=0:")
print(f"  median: {np.median(improve)*100:.1f}%")
print(f"  >10%: {np.sum(improve>0.1)} ({100*np.sum(improve>0.1)/N:.0f}%)")
print(f"  >30%: {np.sum(improve>0.3)} ({100*np.sum(improve>0.3)/N:.0f}%)")
print(f"  <1%:  {np.sum(improve<0.01)} ({100*np.sum(improve<0.01)/N:.0f}%)")

# ============================================================
# Plots
# ============================================================
fig,axes=plt.subplots(2,3,figsize=(16,10))

ax=axes[0,0]
ax.scatter(lv,lrb,s=10,alpha=0.4,c='blue',label='rs_best')
vv=np.linspace(lv.min(),lv.max(),100)
ax.plot(vv,np.polyval(p_best,vv),'b-',lw=2,label=f'rs_best: a={p_best[0]:.3f}')
ax.plot(vv,np.polyval(p_rs1,vv),'gray',ls='--',label=f'rs1: a={p_rs1[0]:.3f}')
ax.plot(vv,np.polyval(p_rs2,vv),'r',ls=':',label=f'rs2: a={p_rs2[0]:.3f}')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s)')
ax.set_title(f'(a) Scaling (N={mask_pos.sum()})'); ax.legend(fontsize=7)

ax=axes[0,1]
ax.scatter(rs1,rs_best,s=10,alpha=0.4,c='steelblue')
mx=max(rs1.max(),rs_best.max())
ax.plot([0.001,mx],[0.001,mx],'k--',alpha=0.3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('rs1'); ax.set_ylabel('rs_best')
ax.set_title(f'(b) rs_best vs rs1 (r={r_b1:.3f})')

ax=axes[0,2]
ax.scatter(rs2,rs_best,s=10,alpha=0.4,c='coral')
mx=max(rs2.max(),rs_best.max())
ax.plot([0.001,mx],[0.001,mx],'k--',alpha=0.3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('rs2'); ax.set_ylabel('rs_best')
ax.set_title(f'(c) rs_best vs rs2 (r={r_b2:.3f})')

ax=axes[1,0]
ax.scatter(Rmax,rs_best,s=10,alpha=0.4,c='green')
ax.plot([0.1,Rmax.max()],[0.1,Rmax.max()],'k--',alpha=0.2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('R_max'); ax.set_ylabel('rs_best')
ax.set_title(f'(d) rs_best vs R_max (rho={rho_Rmax:.3f})')

ax=axes[1,1]
ax.hist(np.log10(np.maximum(rs_best,0.001)),bins=40,color='blue',alpha=0.5,label='rs_best')
ax.hist(np.log10(np.maximum(rs1,0.001)),bins=40,color='gray',alpha=0.3,label='rs1')
ax.hist(np.log10(np.maximum(rs2,0.001)),bins=40,color='red',alpha=0.3,label='rs2')
ax.set_xlabel('log(r_s)'); ax.set_ylabel('Count')
ax.set_title('(e) Distribution comparison'); ax.legend(fontsize=8)

ax=axes[1,2]
ax.scatter(vf,improve*100,s=10,alpha=0.4,c='purple')
ax.axhline(10,color='red',ls='--',alpha=0.5,label='10%')
ax.set_xlabel('v_flat'); ax.set_ylabel('Improvement over rs=0 [%]')
ax.set_title('(f) chi2 improvement'); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'rs_best_all.png'),dpi=150)
plt.close()
print(f"\n[SAVED] rs_best_all.png")

# CSV
out=os.path.join(base_dir,"rs_best_all.csv")
with open(out,"w",newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['galaxy','v_flat','rs1','rs2','rs_best','vf_best','chi2_dof','improvement_pct','R_max','rs_best_over_Rmax','upsilon_d'])
    for d in sorted(results,key=lambda x:x['vflat']):
        w.writerow([d['name'],f"{d['vflat']:.1f}",f"{d['rs1']:.4f}",f"{d['rs2']:.3f}",
                    f"{d['rs_best']:.4f}",f"{d['vf_best']:.1f}",f"{d['chi2_dof']:.4f}",
                    f"{d['improvement']*100:.1f}",f"{d['R_max']:.2f}",
                    f"{d['rs_best']/d['R_max']:.4f}",f"{d['upsilon_d']:.3f}"])
print(f"[SAVED] {out}")

# Final
print(f"\n{'='*70}")
print("rs_best scaling conclusion")
print(f"{'='*70}")
print(f"""
  rs_best: alpha={p_best[0]:.3f} (r={r_best:.3f})
  rs1:     alpha={p_rs1[0]:.3f} (r={r_rs1:.3f})
  rs2:     alpha={p_rs2[0]:.3f} (r={r_rs2:.3f})

  rs_best closer to rs2: {close2}/{N} ({100*close2/N:.0f}%)
  rs_best vs R_max: rho={rho_Rmax:.3f}
""")
