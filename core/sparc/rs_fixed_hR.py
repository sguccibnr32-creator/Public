#!/usr/bin/env python3
"""
rs=h_R fixed model comparison: MOND vs tanh(rs=hR) vs constant vs tanh+MOND
"""
import csv,os,sys,numpy as np
from scipy.optimize import minimize_scalar
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

def mond_gobs(gN,gc):
    return (gN+np.sqrt(gN**2+4*gc*gN))/2

_,src_rows=load_csv(csv_path)
galaxies=[]
for row in src_rows:
    name=row.get('galaxy','').strip()
    if not name: continue
    ud=float(row.get('ud','nan')); vf=float(row.get('vflat','nan'))
    if np.isfinite(ud) and np.isfinite(vf) and ud>0 and vf>0:
        galaxies.append({'name':name,'ud':ud,'vflat_orig':vf})

print(f"[INFO] {len(galaxies)} galaxies")

def fit_1p_grid(func,prange,*args):
    best=(1e30,prange[0])
    for p in prange:
        c=func(p,*args)
        if c<best[0]: best=(c,p)
    dp=(prange[1]-prange[0])*2
    fine=np.linspace(max(best[1]-dp,prange[0]),min(best[1]+dp,prange[-1]),40)
    for p in fine:
        c=func(p,*args)
        if c<best[0]: best=(c,p)
    return best[1],best[0]

print("\nFitting all galaxies...")
results=[]

for i,gal in enumerate(galaxies):
    data=read_dat(gal['name'])
    if data is None or data.shape[0]<5: continue
    rad=data[:,0]; v_obs=data[:,1]; err_v=data[:,2]
    v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=gal['ud']

    mask=rad>0.01; r=rad[mask]; vo=v_obs[mask]; ev=err_v[mask]
    vd=v_disk[mask]; vg=v_gas[mask]; vb=v_bul[mask]
    if len(r)<5: continue

    N_pts=len(r)
    vbar2=ud*np.sign(vd)*vd**2+np.sign(vg)*vg**2+np.sign(vb)*vb**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))
    gN=np.maximum(vbar2/r,1e-10)

    vds=np.sqrt(ud)*np.abs(vd)
    i_pk=np.argmax(vds); r_pk=r[i_pk]
    h_R=r_pk/2.15 if r_pk<r.max()*0.95 and r_pk>0.01 else np.median(r)/2.15
    rs_fixed=2.15*h_R

    w=1.0/np.maximum(ev,1.0)**2

    # Model A: MOND
    def chi2_A(gc_a0):
        gc=gc_a0*a0_unit; gm=mond_gobs(gN,gc)
        vm=np.sqrt(np.maximum(r*gm,0.01))
        return np.sum(w*(vo-vm)**2)
    gc_best,c2_A=fit_1p_grid(chi2_A,np.logspace(-1.5,1.5,60))

    # Model B: tanh rs=hR
    def chi2_B(vf):
        T=0.5*(1+np.tanh((r-rs_fixed)/max(rs_fixed,0.01)))
        vm=np.sqrt(np.maximum(v_bar**2+vf**2*T,0.01))
        return np.sum(w*(vo-vm)**2)
    vf_B,c2_B=fit_1p_grid(chi2_B,np.linspace(1,max(vo)*2,60))

    # Model C: constant
    def chi2_C(vf):
        vm=np.sqrt(np.maximum(v_bar**2+vf**2,0.01))
        return np.sum(w*(vo-vm)**2)
    vf_C,c2_C=fit_1p_grid(chi2_C,np.linspace(1,max(vo)*2,60))

    # Model D: tanh+MOND (2p) - simplified
    best_D=(1e30,0,0)
    for gc_t in np.logspace(-1,1,15):
        for vf_t in np.linspace(max(1,vf_B*0.5),vf_B*1.5,15):
            gm=mond_gobs(gN,gc_t*a0_unit)
            vm_mond=np.sqrt(np.maximum(r*gm,0.01))
            T=0.5*(1+np.tanh((r-rs_fixed)/max(rs_fixed,0.01)))
            vm=np.sqrt(np.maximum(v_bar**2+(vm_mond**2-v_bar**2)*T,0.01))
            c2=np.sum(w*(vo-vm)**2)
            if c2<best_D[0]: best_D=(c2,gc_t,vf_t)
    c2_D=best_D[0]; gc_D=best_D[1]

    dof1=max(N_pts-1,1); dof2=max(N_pts-2,1)
    aic_A=c2_A+2; aic_B=c2_B+2; aic_C=c2_C+2; aic_D=c2_D+4

    results.append({
        'name':gal['name'],'vflat':gal['vflat_orig'],'h_R':h_R,'N_pts':N_pts,
        'chi2_A':c2_A/dof1,'chi2_B':c2_B/dof1,'chi2_C':c2_C/dof1,'chi2_D':c2_D/dof2,
        'gc_A':gc_best,'vf_B':vf_B,'vf_C':vf_C,'gc_D':gc_D,
        'daic_BA':aic_B-aic_A,'daic_BC':aic_B-aic_C,'daic_DA':aic_D-aic_A,
    })
    if (i+1)%25==0: print(f"  {i+1}/{len(galaxies)}...")

N=len(results)
print(f"\nDone: {N}")

# Arrays
chi2_A=np.array([d['chi2_A'] for d in results])
chi2_B=np.array([d['chi2_B'] for d in results])
chi2_C=np.array([d['chi2_C'] for d in results])
chi2_D=np.array([d['chi2_D'] for d in results])
daic_BA=np.array([d['daic_BA'] for d in results])
daic_BC=np.array([d['daic_BC'] for d in results])
daic_DA=np.array([d['daic_DA'] for d in results])
vf_arr=np.array([d['vflat'] for d in results])

# ============================================================
# Results
# ============================================================
print(f"\n{'='*70}")
print(f"Model comparison (N={N})")
print(f"{'='*70}")
print(f"\n{'Model':>25s}  {'chi2/dof med':>14s}")
print(f"{'A: MOND':>25s}  {np.median(chi2_A):14.3f}")
print(f"{'B: tanh rs=hR':>25s}  {np.median(chi2_B):14.3f}")
print(f"{'C: constant':>25s}  {np.median(chi2_C):14.3f}")
print(f"{'D: tanh+MOND':>25s}  {np.median(chi2_D):14.3f}")

print(f"\n--- B vs A ---")
print(f"  dAIC(B-A) median: {np.median(daic_BA):+.2f}")
print(f"  B better (dAIC<-2): {np.sum(daic_BA<-2)}/{N} ({100*np.sum(daic_BA<-2)/N:.0f}%)")
print(f"  A better (dAIC>+2): {np.sum(daic_BA>2)}/{N} ({100*np.sum(daic_BA>2)/N:.0f}%)")

print(f"\n--- B vs C ---")
print(f"  dAIC(B-C) median: {np.median(daic_BC):+.2f}")
print(f"  B better (dAIC<-2): {np.sum(daic_BC<-2)}/{N} ({100*np.sum(daic_BC<-2)/N:.0f}%)")

# v_flat bins
sort_vf=np.argsort(vf_arr); nper=max(N//5,1)
print(f"\n{'v_flat':>12s}  {'N':>4s}  {'c2_A':>6s}  {'c2_B':>6s}  {'c2_C':>6s}  {'B<A%':>5s}  {'B<C%':>5s}")
for i in range(5):
    i0=i*nper; i1=(i+1)*nper if i<4 else N; idx=sort_vf[i0:i1]
    vf_b=vf_arr[idx]; label=f"{vf_b.min():.0f}-{vf_b.max():.0f}"
    ba=100*np.sum(daic_BA[idx]<0)/len(idx); bc=100*np.sum(daic_BC[idx]<0)/len(idx)
    print(f"{label:>12s}  {len(idx):4d}  {np.median(chi2_A[idx]):6.2f}  {np.median(chi2_B[idx]):6.2f}  {np.median(chi2_C[idx]):6.2f}  {ba:4.0f}%  {bc:4.0f}%")

# ============================================================
# Plots
# ============================================================
fig,axes=plt.subplots(2,3,figsize=(16,10))

ax=axes[0,0]
bp=ax.boxplot([chi2_A,chi2_B,chi2_C,chi2_D],
              labels=['A:MOND','B:tanh\nrs=hR','C:const','D:tanh\n+MOND'],
              showfliers=False,patch_artist=True)
for patch,c in zip(bp['boxes'],['lightcoral','lightblue','lightgray','lightgreen']):
    patch.set_facecolor(c)
ax.set_ylabel('chi2/dof'); ax.set_title('(a) Model comparison')

ax=axes[0,1]
ax.hist(daic_BA,bins=50,color='steelblue',edgecolor='white',alpha=0.7)
ax.axvline(0,color='red',ls='--',lw=2)
ax.axvline(np.median(daic_BA),color='black',ls='-',label=f'Med={np.median(daic_BA):+.1f}')
ax.set_xlabel('dAIC (B-A)'); ax.set_title('(b) tanh(rs=hR) vs MOND'); ax.legend(fontsize=8)

ax=axes[0,2]
ax.hist(daic_BC,bins=50,color='coral',edgecolor='white',alpha=0.7)
ax.axvline(0,color='red',ls='--',lw=2)
ax.axvline(np.median(daic_BC),color='black',ls='-',label=f'Med={np.median(daic_BC):+.1f}')
ax.set_xlabel('dAIC (B-C)'); ax.set_title('(c) tanh(rs=hR) vs constant'); ax.legend(fontsize=8)

ax=axes[1,0]
mx=max(np.percentile(chi2_A,95),np.percentile(chi2_B,95))
ax.scatter(chi2_A,chi2_B,s=10,alpha=0.4,c='steelblue')
ax.plot([0,mx],[0,mx],'k--',alpha=0.3)
ax.set_xlabel('chi2/dof (A:MOND)'); ax.set_ylabel('chi2/dof (B:tanh)')
ax.set_title('(d) Per-galaxy B vs A'); ax.set_xlim(0,mx); ax.set_ylim(0,mx)

ax=axes[1,1]
ax.scatter(vf_arr,daic_BA,s=10,alpha=0.4,c='green')
ax.axhline(0,color='red',ls='--'); ax.axhline(-2,color='gray',ls=':',alpha=0.5)
ax.axhline(2,color='gray',ls=':',alpha=0.5)
ax.set_xlabel('v_flat'); ax.set_ylabel('dAIC (B-A)'); ax.set_title('(e) dAIC vs v_flat')

ax=axes[1,2]
for idx,col in [(np.argmin(daic_BA),'blue'),(np.argmin(np.abs(daic_BA)),'gray'),(np.argmax(daic_BA),'red')]:
    d=results[idx]; data=read_dat(d['name'])
    if data is None: continue
    rad=data[:,0]; v_obs=data[:,1]; vmax=max(v_obs)
    ax.plot(rad,v_obs/vmax,color=col,alpha=0.7,label=f"{d['name']} (dAIC={d['daic_BA']:+.0f})")
ax.set_xlabel('r [kpc]'); ax.set_ylabel('v/v_max')
ax.set_title('(f) Representative galaxies'); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'rs_fixed_hR.png'),dpi=150)
plt.close()
print(f"\n[SAVED] rs_fixed_hR.png")

# CSV
out=os.path.join(base_dir,"rs_fixed_hR_comparison.csv")
with open(out,"w",newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['galaxy','v_flat','h_R','N_pts','chi2_A','chi2_B','chi2_C','chi2_D',
                'daic_BA','daic_BC','daic_DA','gc_A','vf_B','vf_C','gc_D'])
    for d in sorted(results,key=lambda x:x['daic_BA']):
        w.writerow([d['name'],f"{d['vflat']:.1f}",f"{d['h_R']:.3f}",d['N_pts'],
                    f"{d['chi2_A']:.4f}",f"{d['chi2_B']:.4f}",f"{d['chi2_C']:.4f}",f"{d['chi2_D']:.4f}",
                    f"{d['daic_BA']:.2f}",f"{d['daic_BC']:.2f}",f"{d['daic_DA']:.2f}",
                    f"{d['gc_A']:.4f}",f"{d['vf_B']:.1f}",f"{d['vf_C']:.1f}",f"{d['gc_D']:.4f}"])
print(f"[SAVED] {out}")

# Final
print(f"\n{'='*70}")
print("Conclusion")
print(f"{'='*70}")
print(f"""
  A(MOND) chi2/dof median: {np.median(chi2_A):.3f}
  B(tanh rs=hR) chi2/dof:  {np.median(chi2_B):.3f}
  C(constant) chi2/dof:    {np.median(chi2_C):.3f}

  B vs A: dAIC median={np.median(daic_BA):+.2f}
    B sig better: {np.sum(daic_BA<-2)}/{N} ({100*np.sum(daic_BA<-2)/N:.0f}%)
    A sig better: {np.sum(daic_BA>2)}/{N} ({100*np.sum(daic_BA>2)/N:.0f}%)
  B vs C: dAIC median={np.median(daic_BC):+.2f}
    B sig better: {np.sum(daic_BC<-2)}/{N} ({100*np.sum(daic_BC<-2)/N:.0f}%)
""")
