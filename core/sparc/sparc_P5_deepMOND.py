#!/usr/bin/env python3
"""
P5: Deep MOND perturbation -- does alpha=0.5 emerge?
Uses TA3+phase1 instead of sparc_gc.csv.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr, linregress
from pathlib import Path
import csv, sys, warnings
warnings.filterwarnings('ignore')

a0=1.2e-10; kpc_m=3.0857e19
BASE=Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
ROTMOD_DIR=BASE/"Rotmod_LTG"; PHASE1=BASE/"phase1"/"sparc_results.csv"; TA3=BASE/"TA3_gc_independent.csv"
for p,l in [(ROTMOD_DIR,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not p.exists(): print(f'[ERROR] {l}: {p}'); sys.exit(1)

def load_pipeline():
    data={}
    with open(PHASE1,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name=row.get('galaxy','').strip()
            try: data[name]={'vflat':float(row.get('vflat','0')),'Yd':float(row.get('ud','0.5'))}
            except: pass
    with open(TA3,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name=row.get('galaxy','').strip()
            try:
                gc_a0=float(row.get('gc_over_a0','0'))
                if name in data and gc_a0>0: data[name]['gc_obs']=gc_a0*a0; data[name]['gc_a0']=gc_a0
            except: pass
    return {k:v for k,v in data.items() if 'gc_obs' in v and v['vflat']>0}

def load_rotmod(name):
    fpath=ROTMOD_DIR/f"{name}_rotmod.dat"
    if not fpath.exists(): return None
    r,vobs,errv,vgas,vdisk,vbul=[],[],[],[],[],[]
    with open(fpath,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                r.append(float(parts[0])); vobs.append(float(parts[1])); errv.append(max(float(parts[2]),1.0))
                vgas.append(float(parts[3])); vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
            except: continue
    if len(r)<8: return None
    return np.array(r),np.array(vobs),np.array(errv),np.array(vgas),np.array(vdisk),np.array(vbul)

def compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=0.5,Yb=0.7):
    conv=1e6/(r_kpc*kpc_m)
    return np.abs(Vgas)**2*conv+Yd*np.abs(Vdisk)**2*conv+Yb*np.abs(Vbul)**2*conv

def V_deepMOND(r_kpc,gN,gc):
    r_m=r_kpc*kpc_m; g_eff=np.sqrt(gN*gc)
    return np.sqrt(np.abs(r_m*g_eff))*1e-3

def V_fullMOND(r_kpc,gN,gc):
    r_m=r_kpc*kpc_m; x=gN/gc
    g_eff=gc*(x+np.sqrt(x**2+4*x))/2
    return np.sqrt(np.abs(r_m*g_eff))*1e-3

def fit_gc(r_kpc,gN,Vobs,errV,mode="full"):
    N=len(r_kpc); n_flat=max(5,N//2); idx=slice(N-n_flat,N)
    rf,Vf,ef,gNf=r_kpc[idx],Vobs[idx],errV[idx],gN[idx]
    valid=ef>0
    if valid.sum()<3: ef=np.maximum(ef,np.median(Vobs)*0.1); valid=np.ones(len(ef),dtype=bool)
    def chi2(log_gc):
        gc=10**log_gc
        Vp=V_deepMOND(rf[valid],gNf[valid],gc) if mode=="deep" else V_fullMOND(rf[valid],gNf[valid],gc)
        return np.sum(((Vf[valid]-Vp)/ef[valid])**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method="bounded")
    return 10**res.x,res.fun,valid.sum()-1

def main():
    print("="*70)
    print("P5: Deep MOND Perturbation -- alpha=0.5 derivation")
    print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)} galaxies")

    records=[]; skipped=0
    for name,info in pipe.items():
        rot=load_rotmod(name)
        if rot is None: skipped+=1; continue
        r_kpc,Vobs,errV,Vgas,Vdisk,Vbul=rot
        Yd=info['Yd']
        vds=np.sqrt(max(Yd,0.01))*np.abs(Vdisk)
        rpk=r_kpc[np.argmax(vds)]
        if rpk<0.01 or rpk>=r_kpc.max()*0.9: skipped+=1; continue
        hR=rpk/2.15
        gN=compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=Yd)
        gc_obs=info['gc_obs']; vflat=info['vflat']

        gc_deep,chi2d,_=fit_gc(r_kpc,gN,Vobs,errV,mode="deep")
        gc_full,chi2f,_=fit_gc(r_kpc,gN,Vobs,errV,mode="full")
        if not(np.isfinite(gc_deep) and np.isfinite(gc_full)): skipped+=1; continue

        # 1st order correction
        x=gN/gc_full; g_full=gc_full*(x+np.sqrt(x**2+4*x))/2; g_deep=np.sqrt(gN*gc_full); g_corr=g_full-g_deep
        n_out=max(3,len(r_kpc)//2)
        rel_corr=np.median(np.abs(g_corr[-n_out:])/(g_deep[-n_out:]+1e-30))
        x_outer=np.median(gN[-n_out:]/gc_full)
        v90=0.9*vflat; ridx=np.where(Vobs>=v90)[0]
        r_flat=r_kpc[ridx[0]] if len(ridx)>0 else r_kpc[-1]
        GS0=(vflat*1e3)**2/(hR*kpc_m)

        records.append({
            'name':name,'gc_obs':gc_obs,'gc_deep':gc_deep,'gc_full':gc_full,
            'chi2_deep':chi2d,'chi2_full':chi2f,
            'hR':hR,'vflat':vflat,'Yd':Yd,'GS0':GS0,
            'rel_corr':rel_corr,'x_outer':x_outer,'r_flat_hR':r_flat/hR,
            'delta_gc':np.log10(gc_full)-np.log10(gc_deep),
        })

    N=len(records); print(f"Processed: {N}, Skipped: {skipped}")
    if N<10: print("Too few."); sys.exit(1)

    gc_obs=np.array([r['gc_obs'] for r in records])
    gc_deep=np.array([r['gc_deep'] for r in records])
    gc_full=np.array([r['gc_full'] for r in records])
    hR_arr=np.array([r['hR'] for r in records])
    vflat_arr=np.array([r['vflat'] for r in records])
    GS0_arr=np.array([r['GS0'] for r in records])
    x_outer=np.array([r['x_outer'] for r in records])
    delta_gc=np.array([r['delta_gc'] for r in records])

    log_gcd=np.log10(gc_deep); log_gcf=np.log10(gc_full); log_gco=np.log10(gc_obs)
    log_GS0=np.log10(GS0_arr/a0); log_vf=np.log10(vflat_arr); log_hR=np.log10(hR_arr)

    # Test 1: gc_deep scaling
    print("\n"+"="*50); print("Test 1: gc_deep scaling"); print("="*50)
    m=np.isfinite(log_gcd)&np.isfinite(log_GS0)
    sl=linregress(log_GS0[m],log_gcd[m])
    from scipy.stats import t as tdist
    p05=2*tdist.sf(abs(sl.slope-0.5)/sl.stderr,df=m.sum()-2)
    print(f"  slope={sl.slope:.4f}+/-{sl.stderr:.4f}, R^2={sl.rvalue**2:.4f}, p(0.5)={p05:.4f}")
    X=np.column_stack([log_vf[m],log_hR[m],np.ones(m.sum())]); y=log_gcd[m]
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    R2mv=1-np.sum((y-X@coef)**2)/np.sum((y-y.mean())**2)
    print(f"  Multivar: vflat^{coef[0]:.3f} * hR^{coef[1]:.3f}, R^2={R2mv:.4f}")

    # Test 2: gc_full scaling
    print("\n"+"="*50); print("Test 2: gc_full scaling"); print("="*50)
    m2=np.isfinite(log_gcf)&np.isfinite(log_GS0)
    sl2=linregress(log_GS0[m2],log_gcf[m2])
    p05f=2*tdist.sf(abs(sl2.slope-0.5)/sl2.stderr,df=m2.sum()-2)
    print(f"  slope={sl2.slope:.4f}+/-{sl2.stderr:.4f}, R^2={sl2.rvalue**2:.4f}, p(0.5)={p05f:.4f}")
    X2=np.column_stack([log_vf[m2],log_hR[m2],np.ones(m2.sum())]); y2=log_gcf[m2]
    coef2=np.linalg.lstsq(X2,y2,rcond=None)[0]
    R2mv2=1-np.sum((y2-X2@coef2)**2)/np.sum((y2-y2.mean())**2)
    print(f"  Multivar: vflat^{coef2[0]:.3f} * hR^{coef2[1]:.3f}, R^2={R2mv2:.4f}")

    # Test 3: gc predictions vs gc_obs
    print("\n"+"="*50); print("Test 3: gc predictions vs gc_obs"); print("="*50)
    gc_geom=np.sqrt(a0*GS0_arr*a0)  # eta=1 geom mean
    for lb,gcp in [("gc_deep",gc_deep),("gc_full",gc_full),("gc_geom",gc_geom)]:
        lp=np.log10(gcp); m=np.isfinite(log_gco)&np.isfinite(lp)
        if m.sum()<10: continue
        rho,_=pearsonr(lp[m],log_gco[m])
        R2=1-np.sum((log_gco[m]-lp[m])**2)/np.sum((log_gco[m]-log_gco[m].mean())**2)
        ratio=gcp[m]/gc_obs[m]
        print(f"  {lb:10s}: rho={rho:.4f}, R^2(1:1)={R2:.4f}, ratio med={np.median(ratio):.3f}")

    # Test 4: 1st order correction structure
    print("\n"+"="*50); print("Test 4: 1st order correction"); print("="*50)
    m4=np.isfinite(delta_gc)
    print(f"  Delta_gc: median={np.median(delta_gc[m4]):.4f}, std={np.std(delta_gc[m4]):.4f}")
    for lb,arr in [("log(hR)",log_hR),("log(vflat)",log_vf),("x_outer",x_outer)]:
        mm=np.isfinite(delta_gc)&np.isfinite(arr)
        if mm.sum()>10:
            rho,p=pearsonr(arr[mm],delta_gc[mm])
            print(f"  Delta vs {lb:12s}: rho={rho:+.4f}, p={p:.3e}")

    # Test 5: x_outer distribution
    print("\n"+"="*50); print("Test 5: Deep MOND degree"); print("="*50)
    xf=x_outer[np.isfinite(x_outer)]
    print(f"  median(x_outer)={np.median(xf):.3f}")
    print(f"  frac x<0.1: {np.mean(xf<0.1):.1%}, x<0.3: {np.mean(xf<0.3):.1%}, x<1: {np.mean(xf<1):.1%}")

    for xcut in [0.1,0.3,0.5,1.0]:
        mx=(x_outer<xcut)&np.isfinite(log_gcf)&np.isfinite(log_GS0)
        if mx.sum()>10:
            sx=linregress(log_GS0[mx],log_gcf[mx])
            print(f"  x<{xcut}: N={mx.sum()}, alpha={sx.slope:.4f}+/-{sx.stderr:.4f}")

    # Test 6: vflat vs hR
    print("\n"+"="*50); print("Test 6: Self-consistency (vflat vs hR)"); print("="*50)
    m6=np.isfinite(log_vf)&np.isfinite(log_hR)
    sl6=linregress(log_hR[m6],log_vf[m6])
    print(f"  log(vflat) vs log(hR): slope={sl6.slope:.4f}+/-{sl6.stderr:.4f}, R^2={sl6.rvalue**2:.4f}")
    print(f"  Self-consist prediction: slope=0.5, deviation={abs(sl6.slope-0.5)/sl6.stderr:.1f}sigma")

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(16,10))
        fig.suptitle(f"P5: Deep MOND Perturbation (N={N})",fontsize=14)

        ax=axes[0,0]
        m=np.isfinite(log_gcf)&np.isfinite(log_gco)
        ax.scatter(log_gco[m],log_gcf[m],s=8,alpha=0.5)
        lim=[min(log_gco[m].min(),log_gcf[m].min())-0.1,max(log_gco[m].max(),log_gcf[m].max())+0.1]
        ax.plot(lim,lim,'k--'); ax.set_xlabel('log(gc_obs)'); ax.set_ylabel('log(gc_full)')
        ax.set_title('(a) gc_full vs gc_obs'); ax.set_aspect('equal')

        ax=axes[0,1]
        m=np.isfinite(log_gcf)&np.isfinite(log_GS0)
        ax.scatter(log_GS0[m],log_gcf[m],s=8,alpha=0.5)
        xl=np.linspace(log_GS0[m].min(),log_GS0[m].max(),50)
        ax.plot(xl,sl2.slope*xl+sl2.intercept,'r-',label=f'slope={sl2.slope:.3f}')
        ax.plot(xl,0.5*xl+np.median(log_gcf[m]-0.5*log_GS0[m]),'g--',label='slope=0.5')
        ax.set_xlabel('log(GS0/a0)'); ax.set_ylabel('log(gc_full)')
        ax.set_title('(b) gc_full scaling'); ax.legend(fontsize=9)

        ax=axes[0,2]
        m=np.isfinite(log_gcd)&np.isfinite(log_gcf)
        ax.scatter(log_gcf[m],log_gcd[m],s=8,alpha=0.5)
        lim2=[min(log_gcf[m].min(),log_gcd[m].min())-0.1,max(log_gcf[m].max(),log_gcd[m].max())+0.1]
        ax.plot(lim2,lim2,'k--'); ax.set_xlabel('log(gc_full)'); ax.set_ylabel('log(gc_deep)')
        ax.set_title('(c) Deep vs Full'); ax.set_aspect('equal')

        ax=axes[1,0]
        m=np.isfinite(delta_gc)&np.isfinite(x_outer)
        ax.scatter(x_outer[m],delta_gc[m],s=8,alpha=0.5)
        ax.axhline(0,color='k',ls='--'); ax.set_xlabel('x_outer'); ax.set_ylabel('Delta log(gc)')
        ax.set_title('(d) 1st order correction')

        ax=axes[1,1]
        ax.hist(xf,bins=30,edgecolor='black',alpha=0.7)
        ax.axvline(np.median(xf),color='r',ls='--',label=f'med={np.median(xf):.2f}')
        ax.axvline(1,color='blue',ls=':',label='x=1')
        ax.set_xlabel('x_outer'); ax.set_title('(e) Deep MOND degree'); ax.legend()

        ax=axes[1,2]
        m=np.isfinite(log_vf)&np.isfinite(log_hR)
        ax.scatter(log_hR[m],log_vf[m],s=8,alpha=0.5)
        xl=np.linspace(log_hR[m].min(),log_hR[m].max(),50)
        ax.plot(xl,sl6.slope*xl+sl6.intercept,'r-',label=f'slope={sl6.slope:.3f}')
        ax.plot(xl,0.5*xl+np.median(log_vf[m]-0.5*log_hR[m]),'g--',label='slope=0.5')
        ax.set_xlabel('log(hR)'); ax.set_ylabel('log(vflat)'); ax.set_title('(f) vflat vs hR')
        ax.legend()

        plt.tight_layout()
        fig.savefig(BASE/"P5_deepMOND_results.png",dpi=150)
        print(f"\nFigure: P5_deepMOND_results.png")
    except Exception as e: print(f"Plot: {e}")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"  gc_deep: alpha={sl.slope:.4f}+/-{sl.stderr:.4f}, p(0.5)={p05:.4f}")
    print(f"  gc_full: alpha={sl2.slope:.4f}+/-{sl2.stderr:.4f}, p(0.5)={p05f:.4f}")
    print(f"  gc_full multivar: vflat^{coef2[0]:.3f} * hR^{coef2[1]:.3f}")
    print(f"  x_outer median={np.median(xf):.3f}")
    print(f"  vflat vs hR: slope={sl6.slope:.4f}")
    if abs(sl2.slope-0.5)<2*sl2.stderr:
        print("\n  * gc_full alpha~0.5: geometric mean law reproduced from full MOND!")
    else:
        print(f"\n  gc_full alpha={sl2.slope:.3f} != 0.5: MOND alone insufficient")
    print("\n[DONE]")

if __name__=="__main__": main()
