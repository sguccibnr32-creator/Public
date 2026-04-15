#!/usr/bin/env python3
"""
Tautology check: does E_grad_norm's hR factor drive the 26% effect?
Uses TA3+phase1.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from numpy.linalg import lstsq
from pathlib import Path
import csv, sys, warnings
warnings.filterwarnings('ignore')

a0=1.2e-10; kpc_m=3.0857e19; Yd_default=0.5; Yb_default=0.7

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
                if name in data and gc_a0>0: data[name]['gc_obs']=gc_a0*a0
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

def partial_corr(x,y,controls):
    if len(controls)==0: return pearsonr(x,y)
    C=np.column_stack(controls+[np.ones(len(x))])
    return pearsonr(x-C@lstsq(C,x,rcond=None)[0],y-C@lstsq(C,y,rcond=None)[0])

def fit_gc_deep(r_kpc,gN,Vobs,errV):
    e=np.where(errV>0,errV,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc
        Vp=np.sqrt(r_kpc*kpc_m*np.sqrt(gN*gc))*1e-3
        return np.sum(((Vobs-Vp)/e)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x

def compute_strain_energies(r_kpc,gN,gc,hR):
    eps=np.sqrt(gN/gc); N=len(r_kpc)
    result={"E_local":np.nan,"E_grad":np.nan,"E_grad_norm":np.nan}
    if N<4: return result
    result["E_local"]=np.trapezoid(eps**2*r_kpc,r_kpc)
    dr=np.diff(r_kpc); deps=np.diff(eps); mask=dr>0
    if mask.sum()<2: return result
    dedr=deps[mask]/dr[mask]
    r_mid=(r_kpc[:-1]+r_kpc[1:])[mask]/2
    result["E_grad"]=np.trapezoid(dedr**2*r_mid,r_mid)
    dedr_n=dedr*hR
    result["E_grad_norm"]=np.trapezoid(dedr_n**2*r_mid,r_mid)
    return result

def main():
    print("="*70); print("Tautology check: E_grad_norm hR factor"); print("="*70)
    pipe=load_pipeline()

    records=[]
    for name,info in pipe.items():
        rot=load_rotmod(name)
        if rot is None: continue
        r_kpc,Vobs,errV,Vgas,Vdisk,Vbul=rot
        Yd=info['Yd']; vflat=info['vflat']; gc_obs_ms2=info['gc_obs']
        vds=np.sqrt(max(Yd,0.01))*np.abs(Vdisk)
        rpk=r_kpc[np.argmax(vds)]
        if rpk<0.01 or rpk>=r_kpc.max()*0.9: continue
        hR=rpk/2.15
        gN=compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=Yd)
        gc_deep=fit_gc_deep(r_kpc,gN,Vobs,errV)
        strain=compute_strain_energies(r_kpc,gN,gc_deep,hR)
        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan
        records.append({
            'gc_obs':gc_obs_ms2,'gc_deep':gc_deep,'hR':hR,'vflat':vflat,
            'x_outer':x_outer,'E_local':strain['E_local'],
            'E_grad':strain['E_grad'],'E_grad_norm':strain['E_grad_norm'],
        })

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); hR_a=arr("hR"); vflat_a=arr("vflat")
    x_outer=arr("x_outer")
    E_local=arr("E_local"); E_grad=arr("E_grad"); E_grad_norm=arr("E_grad_norm")

    log_gc=np.log10(gc_obs); log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a)
    log_x=np.log10(x_outer+1e-10)

    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)&np.isfinite(log_x)
       &np.isfinite(E_grad)&(E_grad>0)&np.isfinite(E_local)&(E_local>0)
       &np.isfinite(E_grad_norm)&(E_grad_norm>0))
    print(f"Valid: {m.sum()}")

    base=[log_vf[m],log_x[m]]
    rho_base,p_base=partial_corr(log_hR[m],log_gc[m],base)
    print(f"\nBaseline |vflat,x: rho={rho_base:+.4f}")

    # Test 1
    print("\n"+"="*50); print("Test 1: E_grad (not normalized, no hR)"); print("="*50)
    log_Eg=np.log10(E_grad[m])
    rho_Eg,p_Eg=partial_corr(log_hR[m],log_gc[m],base+[log_Eg])
    print(f"  |vflat,x,E_grad: rho={rho_Eg:+.4f}, p={p_Eg:.3e}")
    print(f"  d|rho|={abs(rho_Eg)-abs(rho_base):+.4f}")
    log_Egn=np.log10(E_grad_norm[m])
    rho_Egn,p_Egn=partial_corr(log_hR[m],log_gc[m],base+[log_Egn])
    print(f"  |vflat,x,E_grad_norm: rho={rho_Egn:+.4f}, p={p_Egn:.3e}")
    print(f"  d|rho|={abs(rho_Egn)-abs(rho_base):+.4f}")

    if abs(rho_Eg)>abs(rho_base)*0.9:
        print("  >> E_grad (no hR) DOES NOT reduce hR corr")
        print("  >> E_grad_norm effect depends on hR^2 factor")
    else:
        pct=(abs(rho_base)-abs(rho_Eg))/abs(rho_base)*100
        print(f"  >> E_grad (no hR) reduces by {pct:.1f}%")

    # Test 2
    print("\n"+"="*50); print("Test 2: Placebo E_local * hR^2"); print("="*50)
    p1=E_local[m]*hR_a[m]**2
    log_p1=np.log10(p1+1e-30)
    rho_p1,p_p1=partial_corr(log_hR[m],log_gc[m],base+[log_p1])
    print(f"  |vflat,x,E_local*hR^2: rho={rho_p1:+.4f}")
    print(f"  d|rho|={abs(rho_p1)-abs(rho_base):+.4f}")

    log_hR2=2*log_hR[m]
    rho_hR2,p_hR2=partial_corr(log_hR[m],log_gc[m],base+[log_hR2])
    print(f"\n  |vflat,x,hR^2: rho={rho_hR2:+.4f}")
    print(f"  (hR^2 is monotone in hR, same as controlling for hR)")

    print(f"\n  E_local * hR^n scan:")
    for n in [0,0.5,1.0,1.5,2.0,2.5,3.0]:
        pn=E_local[m]*hR_a[m]**n
        log_pn=np.log10(pn+1e-30)
        rho_pn,_=partial_corr(log_hR[m],log_gc[m],base+[log_pn])
        tag=" <-- E_grad_norm equiv" if abs(n-2)<0.01 else ""
        print(f"    n={n:.1f}: rho={rho_pn:+.4f}{tag}")

    # Test 3
    print("\n"+"="*50); print("Test 3: Random * hR^2 placebo"); print("="*50)
    np.random.seed(42)
    rho_randoms=[]
    for i in range(5):
        rq=np.random.lognormal(0,1,m.sum())
        pr=rq*hR_a[m]**2
        log_pr=np.log10(pr+1e-30)
        rho_r,_=partial_corr(log_hR[m],log_gc[m],base+[log_pr])
        rho_randoms.append(rho_r)
        print(f"  Trial {i+1}: rho={rho_r:+.4f}")
    print(f"  Mean: {np.mean(rho_randoms):+.4f}, std: {np.std(rho_randoms):.4f}")
    print(f"  E_grad_norm: {rho_Egn:+.4f}")

    if abs(np.mean(rho_randoms))<abs(rho_base)*0.3:
        print("  >> Random * hR^2 also kills the correlation")
        print("  >> TAUTOLOGY: hR^2 factor is what matters")
        tautology_random=True
    else:
        print("  >> Random * hR^2 keeps correlation")
        print("  >> E_grad_norm content contributes")
        tautology_random=False

    # Test 4
    print("\n"+"="*50); print("Test 4: E_grad_norm = hR^2 * G(r) decomposition"); print("="*50)
    print("  E_grad_norm = hR^2 * integral((de/dr)^2 r dr) = hR^2 * E_grad")
    print(f"  E_grad alone: rho={rho_Eg:+.4f}")
    print(f"  E_grad_norm: rho={rho_Egn:+.4f}")
    rho_both,p_both=partial_corr(log_hR[m],log_gc[m],base+[log_Eg,log_hR2])
    print(f"  |vflat,x,E_grad,hR^2: rho={rho_both:+.4f}")

    # Test 5
    print("\n"+"="*50); print("Test 5: hR shuffle test (1000 trials)"); print("="*50)
    np.random.seed(123)
    n_shuffle=1000
    rho_sh=[]
    for _ in range(n_shuffle):
        hR_s=hR_a[m].copy()
        np.random.shuffle(hR_s)
        Egn_s=E_grad[m]*hR_s**2
        log_s=np.log10(Egn_s+1e-30)
        rho_s,_=partial_corr(log_hR[m],log_gc[m],base+[log_s])
        rho_sh.append(rho_s)
    rho_sh=np.array(rho_sh)
    print(f"  Shuffled: mean={np.mean(rho_sh):+.4f}, std={np.std(rho_sh):.4f}")
    print(f"  5-95%: [{np.percentile(rho_sh,5):+.4f}, {np.percentile(rho_sh,95):+.4f}]")
    print(f"  Real E_grad_norm: {rho_Egn:+.4f}")
    pct=np.mean(rho_sh<=rho_Egn)*100
    print(f"  Real percentile: {pct:.1f}%")

    if pct>5 and pct<95:
        print("  >> Real is indistinguishable from shuffled")
        print("  >> TAUTOLOGY confirmed")
        tautology_shuffle=True
    else:
        print(f"  >> Real is outside shuffled distribution")
        tautology_shuffle=False

    # Test 6
    print("\n"+"="*50); print("Test 6: E_grad/E_local (no hR) precise eval"); print("="*50)
    ratio=E_grad[m]/E_local[m]
    log_ratio=np.log10(ratio+1e-30)
    rho_ratio,p_ratio=partial_corr(log_hR[m],log_gc[m],base+[log_ratio])
    print(f"  |vflat,x,E_grad/E_local: rho={rho_ratio:+.4f}, p={p_ratio:.3e}")
    pct_r=(abs(rho_base)-abs(rho_ratio))/0.472*100
    print(f"  Overall explanation: {pct_r:.1f}%")

    rho_r_gc,p_rg=partial_corr(log_ratio,log_gc[m],[log_hR[m],log_vf[m]])
    print(f"  rho(ratio,gc|hR,vf)={rho_r_gc:+.4f}, p={p_rg:.3e}")
    rho_r_hR,_=pearsonr(log_hR[m],log_ratio)
    print(f"  rho(ratio,hR)={rho_r_hR:+.4f}")
    rho_r_x,_=pearsonr(log_x[m],log_ratio)
    print(f"  rho(ratio,x)={rho_r_x:+.4f}")

    # Test 7
    print("\n"+"="*50); print("Test 7: E_grad * hR^alpha optimal"); print("="*50)
    alphas=np.arange(-2.0,4.1,0.25)
    rho_alpha=[]
    for a in alphas:
        qty=E_grad[m]*hR_a[m]**a
        lq=np.log10(qty+1e-30)
        rho_a,_=partial_corr(log_hR[m],log_gc[m],base+[lq])
        rho_alpha.append(rho_a)
    rho_alpha=np.array(rho_alpha)
    best_idx=np.argmin(np.abs(rho_alpha))
    best_alpha=alphas[best_idx]; best_rho=rho_alpha[best_idx]
    print(f"  Best alpha={best_alpha:.2f}: rho={best_rho:+.4f}")
    rho_0=rho_alpha[np.argmin(np.abs(alphas-0))]
    rho_2=rho_alpha[np.argmin(np.abs(alphas-2))]
    print(f"  alpha=0 (E_grad): rho={rho_0:+.4f}")
    print(f"  alpha=2 (E_grad_norm): rho={rho_2:+.4f}")
    if abs(best_alpha)<0.5:
        print("  >> Best near 0: E_grad has physical power")
    elif abs(best_alpha-2)<0.5:
        print("  >> Best near 2: hR^2 normalization is essential (tautology-like)")
    else:
        print(f"  >> Intermediate: alpha={best_alpha:.1f}")

    # Test 8
    print("\n"+"="*50); print("Test 8: Direct hR control"); print("="*50)
    rho_hc,p_hc=partial_corr(log_hR[m],log_gc[m],base+[log_hR[m]])
    print(f"  |vflat,x,hR: rho={rho_hc:+.4f}")
    print(f"  (Tautological: controlling for hR gives 0)")

    # Summary
    print("\n"+"="*70); print("SUMMARY: Tautology Verdict"); print("="*70)
    print(f"\n  Partial corr comparison:")
    print(f"    |vflat,x:               rho={rho_base:+.4f}")
    print(f"    |vflat,x,E_grad:        rho={rho_Eg:+.4f} (no hR)")
    print(f"    |vflat,x,E_grad/E_loc:  rho={rho_ratio:+.4f} (no hR)")
    print(f"    |vflat,x,E_local*hR^2:  rho={rho_p1:+.4f} (placebo)")
    print(f"    |vflat,x,E_grad_norm:   rho={rho_Egn:+.4f} (with hR^2)")
    print(f"    |vflat,x,hR:            rho={rho_hc:+.4f} (direct)")

    print(f"\n  Shuffle test: pct={pct:.1f}%")
    print(f"  Best alpha={best_alpha:.2f}")

    Eg_eff=abs(rho_Eg)<abs(rho_base)*0.7
    ratio_eff=abs(rho_ratio)<abs(rho_base)*0.7
    placebo_works=abs(rho_p1)<abs(rho_base)*0.3
    shuffle_indist=(pct>5 and pct<95)

    print(f"\n  VERDICT:")
    if placebo_works and shuffle_indist:
        print("  >> TAUTOLOGY CONFIRMED")
        print("    E_local*hR^2 placebo has same effect")
        print("    Shuffle indistinguishable")
        print("    E_grad_norm's 26% is hR^2 normalization artifact")
        if ratio_eff:
            pct_real=(abs(rho_base)-abs(rho_ratio))/0.472*100
            print(f"    But E_grad/E_local (no hR): rho={rho_ratio:+.4f}")
            print(f"    -> Real physical effect: {pct_real:.1f}% (grad/local ratio)")
    elif not placebo_works and not shuffle_indist:
        print("  >> PHYSICAL EFFECT CONFIRMED")
        print("    Placebo fails, shuffle distinguishes")
    else:
        print("  >> PARTIAL TAUTOLOGY")
        print(f"    Placebo: {'works' if placebo_works else 'fails'}")
        print(f"    Shuffle: {'indist' if shuffle_indist else 'distinct'}")
        if Eg_eff:
            pct_Eg=(abs(rho_base)-abs(rho_Eg))/0.472*100
            print(f"    E_grad (no hR) physical: {pct_Eg:.1f}%")
        if ratio_eff:
            pct_real=(abs(rho_base)-abs(rho_ratio))/0.472*100
            print(f"    E_grad/E_local (no hR) physical: {pct_real:.1f}%")

    print(f"\n  hR partial corr FINAL decomposition:")
    print(f"    x_outer (MOND): 69%")
    if ratio_eff:
        pct_real=(abs(rho_base)-abs(rho_ratio))/0.472*100
        print(f"    E_grad/E_local (no hR): {pct_real:.1f}%")
        print(f"    Residual: {100-69-pct_real:.1f}%")
    elif Eg_eff:
        pct_Eg=(abs(rho_base)-abs(rho_Eg))/0.472*100
        print(f"    E_grad (no hR): {pct_Eg:.1f}%")
        print(f"    Residual: {100-69-pct_Eg:.1f}%")
    else:
        print("    Gradient (no hR): no effect")
        print("    Residual: 31% (unexplained)")

if __name__=="__main__": main()
