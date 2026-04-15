#!/usr/bin/env python3
"""
gc method systematics: TA3 tanh gc_obs vs deep/full MOND gc.
Uses TA3+phase1.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, linregress
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
    res_x=x-C@lstsq(C,x,rcond=None)[0]
    res_y=y-C@lstsq(C,y,rcond=None)[0]
    return pearsonr(res_x,res_y)

def fit_gc_deep(r_kpc,gN,Vobs,errV,r_range="all"):
    N=len(r_kpc)
    if r_range=="outer50": sl=slice(N-max(5,N//2),N)
    elif r_range=="outer30": sl=slice(N-max(5,N//3),N)
    elif r_range=="inner50": sl=slice(0,max(5,N//2))
    else: sl=slice(0,N)
    rf=r_kpc[sl]; Vf=Vobs[sl]; ef=errV[sl]; gNf=gN[sl]
    ef=np.where(ef>0,ef,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc; g_eff=np.sqrt(gNf*gc)
        Vp=np.sqrt(rf*kpc_m*g_eff)*1e-3
        return np.sum(((Vf-Vp)/ef)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x,res.fun

def fit_gc_full(r_kpc,gN,Vobs,errV,r_range="all"):
    N=len(r_kpc)
    if r_range=="outer50": sl=slice(N-max(5,N//2),N)
    elif r_range=="outer30": sl=slice(N-max(5,N//3),N)
    elif r_range=="inner50": sl=slice(0,max(5,N//2))
    else: sl=slice(0,N)
    rf=r_kpc[sl]; Vf=Vobs[sl]; ef=errV[sl]; gNf=gN[sl]
    ef=np.where(ef>0,ef,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc; x=gNf/gc
        g_eff=gc*(x+np.sqrt(x**2+4*x))/2
        Vp=np.sqrt(rf*kpc_m*g_eff)*1e-3
        return np.sum(((Vf-Vp)/ef)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x,res.fun

def fit_gc_tanh(r_kpc,Vobs,errV,vflat):
    Vn=Vobs/vflat; en=errV/vflat
    en=np.where(en>0,en,0.1)
    def chi2(lrt):
        rt=10**lrt; Vp=np.tanh(r_kpc/rt)
        return np.sum(((Vn-Vp)/en)**2)
    res=minimize_scalar(chi2,bounds=(-2,2),method='bounded')
    rt=10**res.x
    return (vflat*1e3)**2/(rt*kpc_m),rt,res.fun

def main():
    print("="*70); print("gc method systematics: TA3 vs deep vs full MOND"); print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)}")

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

        gc_deep_all,_=fit_gc_deep(r_kpc,gN,Vobs,errV,"all")
        gc_deep_out,_=fit_gc_deep(r_kpc,gN,Vobs,errV,"outer50")
        gc_deep_in,_=fit_gc_deep(r_kpc,gN,Vobs,errV,"inner50")
        gc_deep_o30,_=fit_gc_deep(r_kpc,gN,Vobs,errV,"outer30")
        gc_full_all,_=fit_gc_full(r_kpc,gN,Vobs,errV,"all")
        gc_full_out,_=fit_gc_full(r_kpc,gN,Vobs,errV,"outer50")
        gc_tanh,rt_tanh,_=fit_gc_tanh(r_kpc,Vobs,errV,vflat)

        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep_all) if gc_deep_all>0 else np.nan
        rmax_hR=r_kpc[-1]/hR if hR>0 else np.nan
        proxy=(vflat*1e3)**2/(hR*kpc_m)
        gc_ratio_io=gc_deep_in/gc_deep_out if gc_deep_out>0 else np.nan

        records.append({
            'name':name,'gc_obs':gc_obs_ms2,'gc_deep_all':gc_deep_all,
            'gc_deep_out':gc_deep_out,'gc_deep_in':gc_deep_in,'gc_deep_o30':gc_deep_o30,
            'gc_full_all':gc_full_all,'gc_full_out':gc_full_out,'gc_tanh':gc_tanh,
            'rt_tanh':rt_tanh,'gc_ratio_io':gc_ratio_io,
            'hR':hR,'vflat':vflat,'proxy':proxy,'x_outer':x_outer,'rmax_hR':rmax_hR,
        })

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); gc_deep=arr("gc_deep_all"); gc_deep_o=arr("gc_deep_out")
    gc_deep_i=arr("gc_deep_in"); gc_deep_o3=arr("gc_deep_o30")
    gc_full=arr("gc_full_all"); gc_full_o=arr("gc_full_out"); gc_tanh=arr("gc_tanh")
    hR_a=arr("hR"); vflat_a=arr("vflat"); proxy_a=arr("proxy")
    x_outer=arr("x_outer"); rmax_hR=arr("rmax_hR"); rt=arr("rt_tanh")

    log_gco=np.log10(gc_obs); log_gcd=np.log10(gc_deep); log_gcdo=np.log10(gc_deep_o)
    log_gcdi=np.log10(gc_deep_i); log_gcf=np.log10(gc_full); log_gcfo=np.log10(gc_full_o)
    log_gct=np.log10(gc_tanh); log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a)
    log_pr=np.log10(proxy_a); log_x=np.log10(x_outer+1e-10)

    m=(np.isfinite(log_gco)&np.isfinite(log_gcd)&np.isfinite(log_hR)&np.isfinite(log_vf)
       &np.isfinite(log_x)&np.isfinite(log_gcdo)&np.isfinite(log_gcdi)
       &np.isfinite(log_gcf)&np.isfinite(log_gct)&np.isfinite(log_gcfo)&np.isfinite(rmax_hR))
    print(f"Valid: {m.sum()}")

    print("\n"+"="*50); print("Test 1: delta = log(gc_obs/gc_X) vs hR"); print("="*50)
    gc_methods={
        "gc_deep(all)":log_gcd,"gc_deep(out50)":log_gcdo,"gc_deep(out30)":np.log10(gc_deep_o3),
        "gc_deep(in50)":log_gcdi,"gc_full(all)":log_gcf,"gc_full(out50)":log_gcfo,
        "gc_tanh":log_gct,
    }
    print(f"  {'method':<18s} {'dmed':>8s} {'rho(d,hR)':>12s} {'p':>12s} {'rho|vf':>10s} {'p':>12s}")
    print("  "+"-"*78)
    for name,lgx in gc_methods.items():
        mx=m&np.isfinite(lgx)
        if mx.sum()<20: continue
        d=log_gco[mx]-lgx[mx]
        rh,ph=pearsonr(log_hR[mx],d)
        rhv,phv=partial_corr(log_hR[mx],d,[log_vf[mx]])
        print(f"  {name:<18s} {np.median(d):+.4f} {rh:+.4f}   p={ph:.3e}  {rhv:+.4f}  p={phv:.3e}")

    print("\n"+"="*50); print("Test 2: hR partial corr by method |vflat,x"); print("="*50)
    all_methods={
        "gc_obs (TA3)":log_gco,"gc_deep(all)":log_gcd,"gc_deep(out50)":log_gcdo,
        "gc_deep(in50)":log_gcdi,"gc_full(all)":log_gcf,"gc_full(out50)":log_gcfo,
        "gc_tanh":log_gct,
    }
    print(f"  {'method':<18s} {'rho|vf':>8s} {'rho|vf,x':>10s} {'alpha':>8s} {'R^2':>8s}")
    print("  "+"-"*56)
    for name,lgx in all_methods.items():
        mx=m&np.isfinite(lgx)
        if mx.sum()<20: continue
        rho_vf,_=partial_corr(log_hR[mx],lgx[mx],[log_vf[mx]])
        rho_vfx,_=partial_corr(log_hR[mx],lgx[mx],[log_vf[mx],log_x[mx]])
        sl=linregress(log_pr[mx],lgx[mx])
        print(f"  {name:<18s} {rho_vf:+.4f}   {rho_vfx:+.4f}   {sl.slope:+.4f}   {sl.rvalue**2:.4f}")

    print("\n"+"="*50); print("Test 3: rmax/hR subsets"); print("="*50)
    for rcut in [5,8,10,15]:
        mc=m&(rmax_hR>rcut)
        if mc.sum()<15: continue
        print(f"\n  rmax/hR>{rcut} (N={mc.sum()}):")
        for name,lgx in [("gc_obs",log_gco),("gc_deep(all)",log_gcd),("gc_full(all)",log_gcf),("gc_tanh",log_gct)]:
            mx=mc&np.isfinite(lgx)
            if mx.sum()<10: continue
            rho_vf,p_vf=partial_corr(log_hR[mx],lgx[mx],[log_vf[mx]])
            rho_vfx,p_vfx=partial_corr(log_hR[mx],lgx[mx],[log_vf[mx],log_x[mx]])
            print(f"    {name:<16s}: |vf={rho_vf:+.4f}  |vf,x={rho_vfx:+.4f}")

    print("\n"+"="*50); print("Test 4: Geometric mean fit per method"); print("="*50)
    for name,lgx in all_methods.items():
        mx=m&np.isfinite(lgx)
        if mx.sum()<20: continue
        sl=linregress(log_pr[mx],lgx[mx])
        X=np.column_stack([log_vf[mx],log_hR[mx],np.ones(mx.sum())])
        coef=lstsq(X,lgx[mx],rcond=None)[0]
        R2=1-np.sum((lgx[mx]-X@coef)**2)/np.sum((lgx[mx]-lgx[mx].mean())**2)
        print(f"  {name:<20s}: alpha={sl.slope:.4f}+/-{sl.stderr:.4f}  vf^{coef[0]:.3f}*hR^{coef[1]:.3f}  R^2={R2:.4f}")

    print("\n"+"="*50); print("Test 5: Inner vs outer fit difference"); print("="*50)
    d_io=log_gcdi[m]-log_gcdo[m]
    rio_h,pio_h=pearsonr(log_hR[m],d_io)
    rio_hv,pio_hv=partial_corr(log_hR[m],d_io,[log_vf[m]])
    rio_hvx,pio_hvx=partial_corr(log_hR[m],d_io,[log_vf[m],log_x[m]])
    print(f"  Delta(in-out) vs hR: raw={rio_h:+.4f}, |vf={rio_hv:+.4f}, |vf,x={rio_hvx:+.4f}")
    print(f"  Stats: median={np.median(d_io):.4f}, std={np.std(d_io):.4f}")

    d_obs_all=np.abs(log_gco[m]-log_gcd[m])
    d_obs_out=np.abs(log_gco[m]-log_gcdo[m])
    d_obs_in=np.abs(log_gco[m]-log_gcdi[m])
    print(f"  |gc_obs - gc_deep(all)|  median={np.median(d_obs_all):.4f}")
    print(f"  |gc_obs - gc_deep(out)|  median={np.median(d_obs_out):.4f}")
    print(f"  |gc_obs - gc_deep(in)|   median={np.median(d_obs_in):.4f}")

    print("\n"+"="*50); print("Test 6: Optimal gc weighted combination"); print("="*50)
    def rho_weight(w):
        lgw=w*log_gcdo[m]+(1-w)*log_gcdi[m]
        try:
            rho,_=partial_corr(log_hR[m],lgw,[log_vf[m],log_x[m]])
            return abs(rho)
        except: return 1.0
    rw=minimize_scalar(rho_weight,bounds=(-1,2),method='bounded')
    w_opt=rw.x; rho_opt=rw.fun
    lg_opt=w_opt*log_gcdo[m]+(1-w_opt)*log_gcdi[m]
    rho_act,p_act=partial_corr(log_hR[m],lg_opt,[log_vf[m],log_x[m]])
    print(f"  Optimal w_outer={w_opt:.4f}, min |rho|={rho_opt:.4f} (signed={rho_act:+.4f})")

    Xw=np.column_stack([log_gcdo[m],log_gcdi[m],np.ones(m.sum())])
    cw=lstsq(Xw,log_gco[m],rcond=None)[0]
    print(f"  gc_obs decomp: gc_out^{cw[0]:.3f} * gc_in^{cw[1]:.3f}")

    print("\n"+"="*50); print("Test 7: tanh rt/hR structure"); print("="*50)
    log_rt=np.log10(rt); rt_hR=rt/hR_a
    mr=m&np.isfinite(log_rt)&np.isfinite(rt_hR)
    if mr.sum()>20:
        print(f"  rt/hR: median={np.median(rt_hR[mr]):.3f}, std={np.std(rt_hR[mr]):.3f}")
        rho_rth,p_rth=pearsonr(log_hR[mr],np.log10(rt_hR[mr]))
        print(f"  rho(log(rt/hR),log(hR))={rho_rth:+.4f}, p={p_rth:.3e}")
        rho_t,p_t=partial_corr(log_hR[mr],log_gco[mr],[log_vf[mr],log_x[mr],np.log10(rt_hR[mr])])
        rho_base_r,_=partial_corr(log_hR[mr],log_gco[mr],[log_vf[mr],log_x[mr]])
        print(f"  rho(gc_obs,hR|vf,x,rt/hR)={rho_t:+.4f} (base={rho_base_r:+.4f})")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  hR partial corr by gc method (|vflat,x_outer):")
    for label,lgx in all_methods.items():
        mx=m&np.isfinite(lgx)
        if mx.sum()>20:
            rho,p=partial_corr(log_hR[mx],lgx[mx],[log_vf[mx],log_x[mx]])
            print(f"    {label:<20s}: rho={rho:+.4f} (p={p:.3e})")

    print(f"\n  Optimal weighted: rho={rho_opt:.4f}")

    print(f"\n  Key question: is 31% systematic or physical?")
    rho_db,p_db=partial_corr(log_hR[m],log_gcd[m],[log_vf[m],log_x[m]])
    rho_ob,p_ob=partial_corr(log_hR[m],log_gco[m],[log_vf[m],log_x[m]])
    print(f"    gc_deep(all) |vf,x: {rho_db:+.4f}")
    print(f"    gc_obs (TA3) |vf,x: {rho_ob:+.4f}")
    if abs(rho_db)<abs(rho_ob)*0.5:
        print(f"\n  >> gc_deep shows much less hR dependence")
        print(f"  >> 31% is mostly METHOD-dependent (TA3 systematics)")
    elif abs(rho_db)>abs(rho_ob)*0.8:
        print(f"\n  >> gc_deep retains similar hR dependence")
        print(f"  >> 31% is method-independent (physical)")
    else:
        pct=(abs(rho_ob)-abs(rho_db))/abs(rho_ob)*100
        print(f"\n  >> Partial: method={pct:.0f}%, physical={100-pct:.0f}%")

if __name__=="__main__": main()
