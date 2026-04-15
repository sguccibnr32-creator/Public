#!/usr/bin/env python3
"""
P3: Z2 SSB Domain Wall hypothesis -- gc prediction.
Uses TA3+phase1 instead of sparc_gc.csv.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import csv, sys, glob, warnings
warnings.filterwarnings('ignore')

a0=1.2e-10; kpc_m=3.0857e19
BASE=Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
ROTMOD_DIR=BASE/"Rotmod_LTG"; PHASE1=BASE/"phase1"/"sparc_results.csv"; TA3=BASE/"TA3_gc_independent.csv"
for p,l in [(ROTMOD_DIR,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not p.exists(): print(f'[ERROR] {l}: {p}'); sys.exit(1)

def load_pipeline():
    data={}
    with open(PHASE1,'r',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        for row in reader:
            name=row.get('galaxy','').strip()
            try: data[name]={'vflat':float(row.get('vflat','0')),'Yd':float(row.get('ud','0.5'))}
            except: pass
    with open(TA3,'r',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        for row in reader:
            name=row.get('galaxy','').strip()
            try:
                gc_a0=float(row.get('gc_over_a0','0'))
                if name in data and gc_a0>0:
                    data[name]['gc_obs']=gc_a0*a0; data[name]['gc_a0']=gc_a0
            except: pass
    return {k:v for k,v in data.items() if 'gc_obs' in v and v['vflat']>0}

def load_rotmod(name):
    fpath=ROTMOD_DIR/f"{name}_rotmod.dat"
    if not fpath.exists(): return None
    r,vobs,vgas,vdisk,vbul=[],[],[],[],[]
    with open(fpath,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                r.append(float(parts[0])); vobs.append(float(parts[1]))
                vgas.append(float(parts[3])); vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
            except: continue
    if len(r)<5: return None
    return np.array(r),np.array(vobs),np.array(vgas),np.array(vdisk),np.array(vbul)

def compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=0.5,Yb=0.7):
    conv=1e6/(r_kpc*kpc_m)
    return np.abs(Vgas)**2*conv+Yd*np.abs(Vdisk)**2*conv+Yb*np.abs(Vbul)**2*conv

def epsilon_eq(x,c):
    if np.isscalar(x):
        disc=(x+2)**2-4*c
        return (-x+np.sqrt(disc))/2 if disc>=0 else np.nan
    disc=(x+2)**2-4*c; result=np.full_like(x,np.nan,dtype=float)
    v=disc>=0; result[v]=(-x[v]+np.sqrt(disc[v]))/2; return result

def compute_epsilon_profile(r_kpc,gN,c):
    gc=c*a0; x=gN/gc; eps=epsilon_eq(x,c); return eps,np.isfinite(eps)

def measure_kink(r_kpc,eps,valid):
    if valid.sum()<5: return np.nan,np.nan,np.nan,np.nan
    rv=r_kpc[valid]; ev=eps[valid]
    if len(rv)<3: return np.nan,np.nan,np.nan,np.nan
    dr=np.diff(rv); de=np.diff(ev); ok=dr>0
    if ok.sum()<2: return np.nan,np.nan,np.nan,np.nan
    dedr=de[ok]/dr[ok]; rmid=(rv[:-1]+rv[1:])[ok]/2
    E_kink=np.trapezoid(dedr**2,rmid)
    r_wall=rmid[np.argmax(np.abs(dedr))]
    e_max=np.max(ev); e_min=np.min(ev); e_range=e_max-e_min
    if e_range<1e-10: return np.nan,r_wall,E_kink,np.nan
    delta=e_range/(np.max(np.abs(dedr))+1e-30)
    return delta,r_wall,E_kink,np.max(np.abs(dedr))

def predict_c_delta(r_kpc,gN,hR,beta,c_min=1.001,c_max=100):
    def res(c):
        eps,v=compute_epsilon_profile(r_kpc,gN,c)
        d,_,_,_=measure_kink(r_kpc,eps,v)
        return (d/hR-beta) if np.isfinite(d) and d>0 else 1e10
    try:
        if res(c_min)*res(c_max)>0: return np.nan
        return brentq(res,c_min,c_max,xtol=1e-6,maxiter=100)
    except: return np.nan

def predict_c_energy(r_kpc,gN,E_target,c_min=1.001,c_max=100):
    def res(c):
        eps,v=compute_epsilon_profile(r_kpc,gN,c)
        _,_,E,_=measure_kink(r_kpc,eps,v)
        return (np.log10(E)-np.log10(E_target)) if np.isfinite(E) and E>0 else 1e10
    try:
        if res(c_min)*res(c_max)>0: return np.nan
        return brentq(res,c_min,c_max,xtol=1e-6,maxiter=100)
    except: return np.nan

def main():
    print("="*70)
    print("P3: Z2 SSB Domain Wall -- gc prediction")
    print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)} galaxies")

    records=[]
    for name,info in pipe.items():
        rot=load_rotmod(name)
        if rot is None: continue
        r_kpc,Vobs,Vgas,Vdisk,Vbul=rot
        Yd=info['Yd']
        vds=np.sqrt(max(Yd,0.01))*np.abs(Vdisk)
        rpk=r_kpc[np.argmax(vds)]
        if rpk<0.01 or rpk>=r_kpc.max()*0.9: continue
        hR=rpk/2.15
        gN=compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=Yd)
        gc_val=info['gc_obs']; c_obs=gc_val/a0
        if c_obs<=1: continue
        eps,valid=compute_epsilon_profile(r_kpc,gN,c_obs)
        delta,r_wall,E_kink,max_grad=measure_kink(r_kpc,eps,valid)
        if np.isfinite(delta) and delta>0 and hR>0:
            records.append({'name':name,'gc_obs':gc_val,'c_obs':c_obs,'hR':hR,
                'vflat':info['vflat'],'Yd':Yd,'delta':delta,'r_wall':r_wall,
                'E_kink':E_kink,'max_grad':max_grad,'delta_over_hR':delta/hR,
                'r_wall_over_hR':r_wall/hR if np.isfinite(r_wall) else np.nan,
                'gN_profile':gN,'r_profile':r_kpc})

    N=len(records); print(f"Kink measured: {N} galaxies")
    if N<10: print("Too few."); sys.exit(1)

    dh=np.array([r['delta_over_hR'] for r in records])
    dh_f=dh[np.isfinite(dh)&(dh>0)&(dh<100)]
    c_arr=np.array([r['c_obs'] for r in records])
    vf_arr=np.array([r['vflat'] for r in records])
    hR_arr=np.array([r['hR'] for r in records])
    E_arr=np.array([r['E_kink'] for r in records])
    rw_hR=np.array([r['r_wall_over_hR'] for r in records])

    print("\n"+"="*50); print("Test 1: delta/hR distribution"); print("="*50)
    print(f"  N={len(dh_f)}")
    print(f"  median={np.median(dh_f):.3f}, mean={np.mean(dh_f):.3f}, std={np.std(dh_f):.3f}")
    print(f"  CV={np.std(dh_f)/np.mean(dh_f):.3f}")
    universal = np.std(dh_f)/np.mean(dh_f) < 0.3

    print("\n"+"="*50); print("Test 2: delta/hR drivers"); print("="*50)
    mask=np.isfinite(dh)&(dh>0)&(dh<100)
    for lb,arr in [("c_obs",c_arr),("vflat",vf_arr),("hR",hR_arr),("log E",np.log10(E_arr+1e-30))]:
        m=mask&np.isfinite(arr)
        if m.sum()>10:
            rho,p=pearsonr(arr[m],dh[m])
            print(f"  delta/hR vs {lb:12s}: rho={rho:+.4f}, p={p:.3e}")

    print("\n"+"="*50); print("Test 3: r_wall/hR"); print("="*50)
    rw_f=rw_hR[np.isfinite(rw_hR)&(rw_hR>0)]
    print(f"  N={len(rw_f)}, median={np.median(rw_f):.2f}, mean={np.mean(rw_f):.2f}")

    print("\n"+"="*50); print("Test 4: E_kink constraint"); print("="*50)
    E_valid=E_arr[mask&np.isfinite(E_arr)&(E_arr>0)]
    E_med=np.median(E_valid); E_cv=np.std(E_valid)/np.mean(E_valid)
    print(f"  E_kink: median={E_med:.4e}, CV={E_cv:.3f}")

    gc_pE,gc_oE=[],[]
    for rec in records:
        c_p=predict_c_energy(rec['r_profile'],rec['gN_profile'],E_med)
        if np.isfinite(c_p) and c_p>0: gc_pE.append(c_p*a0); gc_oE.append(rec['gc_obs'])
    print(f"  Converged: {len(gc_pE)}/{N}")
    if len(gc_pE)>10:
        lp=np.log10(np.array(gc_pE)); lo=np.log10(np.array(gc_oE))
        m=np.isfinite(lp)&np.isfinite(lo)
        if m.sum()>5:
            rho,_=pearsonr(lp[m],lo[m])
            R2=1-np.sum((lo[m]-lp[m])**2)/np.sum((lo[m]-lo[m].mean())**2)
            print(f"  rho={rho:.4f}, R^2(1:1)={R2:.4f}, ratio median={np.median(np.array(gc_pE)[m]/np.array(gc_oE)[m]):.3f}")

    print("\n"+"="*50); print("Test 5: delta=beta*hR constraint"); print("="*50)
    beta_med=np.median(dh_f); print(f"  beta={beta_med:.3f}")
    gc_pD,gc_oD=[],[]
    for rec in records:
        c_p=predict_c_delta(rec['r_profile'],rec['gN_profile'],rec['hR'],beta_med)
        if np.isfinite(c_p) and c_p>0: gc_pD.append(c_p*a0); gc_oD.append(rec['gc_obs'])
    print(f"  Converged: {len(gc_pD)}/{N}")
    if len(gc_pD)>10:
        lp=np.log10(np.array(gc_pD)); lo=np.log10(np.array(gc_oD))
        m=np.isfinite(lp)&np.isfinite(lo)
        if m.sum()>5:
            rho,_=pearsonr(lp[m],lo[m])
            R2=1-np.sum((lo[m]-lp[m])**2)/np.sum((lo[m]-lo[m].mean())**2)
            print(f"  rho={rho:.4f}, R^2(1:1)={R2:.4f}")

    print("\n"+"="*50); print("Test 6: Scaling"); print("="*50)
    log_gc=np.log10(np.array([r['gc_obs'] for r in records]))
    log_GS0=np.log10(np.array([r['vflat']**2/(r['hR']*kpc_m) for r in records]))
    log_delta=np.log10(np.array([r['delta'] for r in records]))
    log_E=np.log10(E_arr+1e-30)

    m1=np.isfinite(log_delta)&np.isfinite(log_GS0)
    if m1.sum()>10:
        sl=np.polyfit(log_GS0[m1],log_delta[m1],1)
        rho,_=pearsonr(log_GS0[m1],log_delta[m1])
        print(f"  log(delta) vs log(GS0): slope={sl[0]:.3f}, rho={rho:.3f}")
    m2=np.isfinite(log_E)&np.isfinite(log_gc)&(log_E>-20)
    if m2.sum()>10:
        sl=np.polyfit(log_gc[m2],log_E[m2],1)
        rho,_=pearsonr(log_gc[m2],log_E[m2])
        print(f"  log(E_kink) vs log(gc): slope={sl[0]:.3f}, rho={rho:.3f}")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(16,10))
        fig.suptitle(f"P3: Domain Wall (N={N})",fontsize=14)

        axes[0,0].hist(dh_f,bins=30,edgecolor='black',alpha=0.7)
        axes[0,0].axvline(np.median(dh_f),color='red',ls='--',label=f'med={np.median(dh_f):.2f}')
        axes[0,0].set_xlabel('delta/hR'); axes[0,0].set_title('(a) Kink width/hR'); axes[0,0].legend()

        axes[0,1].hist(rw_f,bins=30,edgecolor='black',alpha=0.7)
        axes[0,1].axvline(np.median(rw_f),color='red',ls='--',label=f'med={np.median(rw_f):.1f}')
        axes[0,1].set_xlabel('r_wall/hR'); axes[0,1].set_title('(b) Wall position'); axes[0,1].legend()

        m=np.isfinite(dh)&(dh>0)&(dh<100)
        axes[0,2].scatter(c_arr[m],dh[m],s=8,alpha=0.5)
        axes[0,2].set_xlabel('c_obs'); axes[0,2].set_ylabel('delta/hR'); axes[0,2].set_title('(c) delta/hR vs c')

        m3=np.isfinite(log_E)&np.isfinite(log_gc)&(log_E>-20)
        if m3.sum()>0:
            axes[1,0].scatter(log_gc[m3],log_E[m3],s=8,alpha=0.5)
            axes[1,0].set_xlabel('log(gc)'); axes[1,0].set_ylabel('log(E_kink)'); axes[1,0].set_title('(d) E_kink vs gc')

        if len(gc_pE)>5:
            lp=np.log10(np.array(gc_pE)); lo=np.log10(np.array(gc_oE))
            m=np.isfinite(lp)&np.isfinite(lo)
            axes[1,1].scatter(lo[m],lp[m],s=8,alpha=0.5)
            lim=[min(lo[m].min(),lp[m].min())-0.1,max(lo[m].max(),lp[m].max())+0.1]
            axes[1,1].plot(lim,lim,'k--')
            axes[1,1].set_xlabel('log(gc_obs)'); axes[1,1].set_ylabel('log(gc_pred)')
            axes[1,1].set_title('(e) E constraint')

        if len(gc_pD)>5:
            lp=np.log10(np.array(gc_pD)); lo=np.log10(np.array(gc_oD))
            m=np.isfinite(lp)&np.isfinite(lo)
            axes[1,2].scatter(lo[m],lp[m],s=8,alpha=0.5)
            lim=[min(lo[m].min(),lp[m].min())-0.1,max(lo[m].max(),lp[m].max())+0.1]
            axes[1,2].plot(lim,lim,'k--')
            axes[1,2].set_xlabel('log(gc_obs)'); axes[1,2].set_ylabel('log(gc_pred)')
            axes[1,2].set_title('(f) delta constraint')

        plt.tight_layout()
        fig.savefig(BASE/"P3_domainwall_results.png",dpi=150)
        print(f"\nFigure: P3_domainwall_results.png")
    except Exception as e: print(f"Plot: {e}")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"  delta/hR: median={np.median(dh_f):.3f}, CV={np.std(dh_f)/np.mean(dh_f):.3f}")
    print(f"  r_wall/hR: median={np.median(rw_f):.2f}")
    print(f"  {'Universal (CV<0.3)' if universal else 'Non-universal (CV>=0.3)'}")
    print("\n[DONE]")

if __name__=="__main__": main()
