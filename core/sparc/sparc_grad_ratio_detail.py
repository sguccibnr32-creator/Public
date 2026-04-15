#!/usr/bin/env python3
"""
E_grad/E_local precision evaluation.
Uses TA3+phase1.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr, linregress
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

def partial_corr_residuals(x,y,controls):
    if len(controls)==0: return x,y
    C=np.column_stack(controls+[np.ones(len(x))])
    return x-C@lstsq(C,x,rcond=None)[0],y-C@lstsq(C,y,rcond=None)[0]

def fit_gc_deep(r_kpc,gN,Vobs,errV):
    e=np.where(errV>0,errV,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc
        Vp=np.sqrt(r_kpc*kpc_m*np.sqrt(gN*gc))*1e-3
        return np.sum(((Vobs-Vp)/e)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x

def compute_strain(r_kpc,gN,gc,hR):
    eps=np.sqrt(gN/gc); N=len(r_kpc)
    if N<4: return np.nan,np.nan,np.nan,None,None
    E_local=np.trapezoid(eps**2*r_kpc,r_kpc)
    dr=np.diff(r_kpc); deps=np.diff(eps); mask=dr>0
    if mask.sum()<2: return E_local,np.nan,np.nan,None,None
    dedr=deps[mask]/dr[mask]
    r_mid=(r_kpc[:-1]+r_kpc[1:])[mask]/2
    E_grad=np.trapezoid(dedr**2*r_mid,r_mid)
    ratio=E_grad/E_local if E_local>0 else np.nan
    return E_local,E_grad,ratio,dedr,r_mid

def main():
    print("="*70); print("E_grad/E_local precision evaluation"); print("="*70)
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
        E_local,E_grad,ratio,dedr,r_mid=compute_strain(r_kpc,gN,gc_deep,hR)
        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan
        rmax_hR=r_kpc[-1]/hR if hR>0 else np.nan

        grad_shape={}
        if dedr is not None and len(dedr)>4:
            abs_d=np.abs(dedr); peak_i=np.argmax(abs_d)
            grad_shape["grad_peak_frac"]=abs_d[peak_i]/(np.mean(abs_d)+1e-30)
            n_h=len(dedr)//2
            if n_h>=2:
                grad_shape["grad_asymm"]=np.mean(abs_d[:n_h])/(np.mean(abs_d[n_h:])+1e-30)
            else: grad_shape["grad_asymm"]=np.nan
            grad_shape["grad_peak_r_hR"]=r_mid[peak_i]/hR if hR>0 else np.nan
            total=np.sum(abs_d)
            if total>0:
                r_mean=np.sum(abs_d*r_mid)/total
                r_var=np.sum(abs_d*(r_mid-r_mean)**2)/total
                grad_shape["grad_width_hR"]=np.sqrt(r_var)/hR if hR>0 else np.nan
            else: grad_shape["grad_width_hR"]=np.nan
        else:
            grad_shape={"grad_peak_frac":np.nan,"grad_asymm":np.nan,
                        "grad_peak_r_hR":np.nan,"grad_width_hR":np.nan}

        records.append({
            'name':name,'gc_obs':gc_obs_ms2,'gc_deep':gc_deep,
            'hR':hR,'vflat':vflat,'x_outer':x_outer,'rmax_hR':rmax_hR,
            'E_local':E_local,'E_grad':E_grad,'ratio':ratio,
            **grad_shape,
        })

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); gc_deep=arr("gc_deep")
    hR_a=arr("hR"); vflat_a=arr("vflat")
    x_outer=arr("x_outer"); rmax_hR=arr("rmax_hR")
    E_local=arr("E_local"); E_grad=arr("E_grad"); ratio=arr("ratio")
    grad_pf=arr("grad_peak_frac"); grad_as=arr("grad_asymm")
    grad_pr=arr("grad_peak_r_hR"); grad_wd=arr("grad_width_hR")

    log_gc=np.log10(gc_obs); log_gcd=np.log10(gc_deep)
    log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a)
    log_x=np.log10(x_outer+1e-10); log_ratio=np.log10(ratio+1e-30)

    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)&np.isfinite(log_x)
       &np.isfinite(log_ratio)&np.isfinite(E_grad)&(E_grad>0)&np.isfinite(E_local)&(E_local>0))
    print(f"Valid: {m.sum()}")

    base=[log_vf[m],log_x[m]]
    rho_base,p_base=partial_corr(log_hR[m],log_gc[m],base)
    print(f"Baseline |vflat,x: rho={rho_base:+.4f}, p={p_base:.3e}")

    print("\n"+"="*50); print("Test 1: ratio partial corr detail"); print("="*50)
    rho_r,p_r=partial_corr(log_hR[m],log_gc[m],base+[log_ratio[m]])
    print(f"  |vflat,x,ratio: rho={rho_r:+.4f}, p={p_r:.3e}")
    delta_pct=(abs(rho_base)-abs(rho_r))/0.472*100
    print(f"  Reduction of total hR corr: {delta_pct:.1f}%")

    rx_hR,rx_gc=partial_corr_residuals(log_hR[m],log_gc[m],base)
    rho_sp_b,p_sp_b=spearmanr(rx_hR,rx_gc)
    rx2,rgc2=partial_corr_residuals(log_hR[m],log_gc[m],base+[log_ratio[m]])
    rho_sp_r,p_sp_r=spearmanr(rx2,rgc2)
    print(f"\n  Spearman (residuals):")
    print(f"    base: rho_s={rho_sp_b:+.4f}")
    print(f"    +ratio: rho_s={rho_sp_r:+.4f}")

    real_delta=abs(rho_base)-abs(rho_r)
    print(f"\n  Effect size:")
    print(f"    partial R^2 base: {rho_base**2:.4f}")
    print(f"    partial R^2 +ratio: {rho_r**2:.4f}")
    print(f"    dR^2: {rho_base**2-rho_r**2:.4f}")

    print("\n"+"="*50); print("Test 2: rmax/hR subsets"); print("="*50)
    for rcut in [3,5,8,10]:
        mc=m&(rmax_hR>rcut)
        if mc.sum()<15: continue
        rb,pb=partial_corr(log_hR[mc],log_gc[mc],[log_vf[mc],log_x[mc]])
        rr,pr=partial_corr(log_hR[mc],log_gc[mc],[log_vf[mc],log_x[mc],log_ratio[mc]])
        d=abs(rb)-abs(rr)
        print(f"  rmax/hR>{rcut:2d} (N={mc.sum():3d}): base={rb:+.4f}, +ratio={rr:+.4f}, d={d:+.4f}")

    print("\n"+"="*50); print("Test 3: Bootstrap CI (5000)"); print("="*50)
    np.random.seed(42)
    n_boot=5000
    rho_bb=[]; rho_br=[]; delta_b=[]
    n_samp=m.sum()
    lhRm=log_hR[m]; lgcm=log_gc[m]; lvfm=log_vf[m]; lxm=log_x[m]; lratm=log_ratio[m]
    for _ in range(n_boot):
        idx=np.random.choice(n_samp,n_samp,replace=True)
        try:
            rb,_=partial_corr(lhRm[idx],lgcm[idx],[lvfm[idx],lxm[idx]])
            rr,_=partial_corr(lhRm[idx],lgcm[idx],[lvfm[idx],lxm[idx],lratm[idx]])
            rho_bb.append(rb); rho_br.append(rr); delta_b.append(abs(rb)-abs(rr))
        except: pass
    rho_bb=np.array(rho_bb); rho_br=np.array(rho_br); delta_b=np.array(delta_b)
    print(f"  Success: {len(delta_b)}/{n_boot}")
    print(f"  base 95% CI: [{np.percentile(rho_bb,2.5):.4f}, {np.percentile(rho_bb,97.5):.4f}]")
    print(f"  +ratio 95% CI: [{np.percentile(rho_br,2.5):.4f}, {np.percentile(rho_br,97.5):.4f}]")
    print(f"\n  delta distribution:")
    print(f"    mean={np.mean(delta_b):.4f}, median={np.median(delta_b):.4f}")
    print(f"    95% CI=[{np.percentile(delta_b,2.5):.4f}, {np.percentile(delta_b,97.5):.4f}]")
    print(f"    P(d>0)={np.mean(delta_b>0):.3f}")
    boot_p_pos=np.mean(delta_b>0)

    print("\n"+"="*50); print("Test 4: ratio independence"); print("="*50)
    rho_rh,p_rh=pearsonr(log_hR[m],log_ratio[m])
    rho_rx,p_rx=pearsonr(log_x[m],log_ratio[m])
    rho_rv,p_rv=pearsonr(log_vf[m],log_ratio[m])
    rho_rhx,_=partial_corr(log_hR[m],log_ratio[m],[log_x[m]])
    rho_rxh,_=partial_corr(log_x[m],log_ratio[m],[log_hR[m]])
    print(f"  ratio vs hR: rho={rho_rh:+.4f}")
    print(f"  ratio vs x: rho={rho_rx:+.4f}")
    print(f"  ratio vs vflat: rho={rho_rv:+.4f}")
    print(f"  ratio vs hR|x: rho={rho_rhx:+.4f}")
    print(f"  ratio vs x|hR: rho={rho_rxh:+.4f}")
    rho_rgc,p_rgc=partial_corr(log_ratio[m],log_gc[m],[log_hR[m],log_vf[m]])
    print(f"\n  ratio vs gc|hR,vflat: rho={rho_rgc:+.4f}, p={p_rgc:.3e}")
    rho_rgcx,p_rgcx=partial_corr(log_ratio[m],log_gc[m],[log_hR[m],log_vf[m],log_x[m]])
    print(f"  ratio vs gc|hR,vflat,x: rho={rho_rgcx:+.4f}, p={p_rgcx:.3e}")

    print("\n"+"="*50); print("Test 5: What drives ratio"); print("="*50)
    sl=linregress(log_hR[m],log_ratio[m])
    print(f"  ratio vs hR: slope={sl.slope:.4f}+/-{sl.stderr:.4f}")
    X=np.column_stack([log_hR[m],log_x[m],np.ones(m.sum())])
    coef=lstsq(X,log_ratio[m],rcond=None)[0]
    R2_hx=1-np.sum((log_ratio[m]-X@coef)**2)/np.sum((log_ratio[m]-log_ratio[m].mean())**2)
    print(f"  R^2(ratio ~ hR + x): {R2_hx:.4f}")
    print(f"  ratio ~ hR^{coef[0]:.3f} * x^{coef[1]:.3f}")
    print(f"  -> {R2_hx*100:.1f}% explained by hR and x")
    print(f"  -> {(1-R2_hx)*100:.1f}% is 'independent physical info'")

    print("\n"+"="*50); print("Test 6: gc residual vs ratio"); print("="*50)
    resid=log_gc[m]-log_gcd[m]
    rho_r_r,p_r_r=pearsonr(log_ratio[m],resid)
    print(f"  raw: rho(resid,ratio)={rho_r_r:+.4f}, p={p_r_r:.3e}")
    rho_r_rx,p_r_rx=partial_corr(log_ratio[m],resid,[log_x[m]])
    print(f"  |x: rho={rho_r_rx:+.4f}, p={p_r_rx:.3e}")
    rho_r_rxh,p_r_rxh=partial_corr(log_ratio[m],resid,[log_x[m],log_hR[m]])
    print(f"  |x,hR: rho={rho_r_rxh:+.4f}, p={p_r_rxh:.3e}")
    rho_r_full,p_r_full=partial_corr(log_ratio[m],resid,[log_x[m],log_hR[m],log_vf[m]])
    print(f"  |x,hR,vf: rho={rho_r_full:+.4f}, p={p_r_full:.3e}")

    print("\n"+"="*50); print("Test 7: Gradient shape params"); print("="*50)
    shape_p={"grad_peak_frac":grad_pf,"grad_asymm":grad_as,
             "grad_peak_r/hR":grad_pr,"grad_width/hR":grad_wd}
    for name,sp in shape_p.items():
        v=m&np.isfinite(sp)&(sp>0)
        if v.sum()<20: continue
        log_sp=np.log10(sp[v]+1e-10)
        rho_s,p_s=partial_corr(log_hR[v],log_gc[v],[log_vf[v],log_x[v],log_sp])
        d=abs(rho_s)-abs(rho_base)
        resid_v=log_gc[v]-log_gcd[v]
        rho_sr,_=partial_corr(log_sp,resid_v,[log_x[v],log_hR[v]])
        print(f"  {name:20s}: hR partial rho={rho_s:+.4f} (d={d:+.4f}), resid|x,hR: rho={rho_sr:+.4f}")

    print("\n"+"="*50); print("Test 8: Permutation test (2000)"); print("="*50)
    np.random.seed(777)
    n_perm=2000
    perm_deltas=[]
    for _ in range(n_perm):
        rs=log_ratio[m].copy()
        np.random.shuffle(rs)
        try:
            rho_p,_=partial_corr(log_hR[m],log_gc[m],base+[rs])
            perm_deltas.append(abs(rho_base)-abs(rho_p))
        except: pass
    perm_deltas=np.array(perm_deltas)
    p_perm=np.mean(perm_deltas>=real_delta)
    print(f"  Real delta={real_delta:.4f}")
    print(f"  Perm: mean={np.mean(perm_deltas):.4f}, std={np.std(perm_deltas):.4f}")
    print(f"  Perm p-value={p_perm:.4f}")
    print(f"  95th pct={np.percentile(perm_deltas,95):.4f}")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  Partial corr:")
    print(f"    base: rho={rho_base:+.4f}")
    print(f"    +ratio: rho={rho_r:+.4f}")
    print(f"    d|rho|={real_delta:+.4f} ({delta_pct:.1f}%)")
    print(f"    Pearson p={p_r:.3e}")
    print(f"    Perm p={p_perm:.4f}")
    print(f"\n  Bootstrap:")
    print(f"    P(d>0)={boot_p_pos:.3f}")
    print(f"    95% CI d=[{np.percentile(delta_b,2.5):.4f}, {np.percentile(delta_b,97.5):.4f}]")
    print(f"\n  Independence:")
    print(f"    ratio {R2_hx*100:.1f}% from hR+x")
    print(f"    ratio vs gc|hR,vf,x: rho={rho_rgcx:+.4f}")

    print(f"\n  VERDICT:")
    if p_perm<0.05 and boot_p_pos>0.9:
        print("  >> Statistically significant physical effect")
        print(f"  >> Adds {delta_pct:.0f}% to hR decomposition")
    elif p_perm<0.10 or boot_p_pos>0.7:
        print("  >> Suggestive but not decisive")
        print(f"  >> Direction correct, underpowered N={m.sum()}")
    else:
        print("  >> Noise only")
        print("  >> Gradient hypothesis not supported")

    print(f"\n  FINAL hR decomposition:")
    print(f"    x_outer (MOND): 69%")
    if p_perm<0.10:
        print(f"    E_grad/E_local: {delta_pct:.0f}% (p_perm={p_perm:.3f})")
        print(f"    Residual: {100-69-delta_pct:.0f}%")
    else:
        print(f"    E_grad/E_local: unclear ({delta_pct:.0f}%, p={p_perm:.3f})")
        print(f"    Residual: 31%")

if __name__=="__main__": main()
