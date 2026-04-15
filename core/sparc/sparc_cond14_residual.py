#!/usr/bin/env python3
"""
Condition-14 (plastic region) -- does it explain hR residual 31%?
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
    return pearsonr(x-C@lstsq(C,x,rcond=None)[0],y-C@lstsq(C,y,rcond=None)[0])

def fit_gc_deep(r_kpc,gN,Vobs,errV,r_range="all"):
    N=len(r_kpc)
    if r_range=="outer50": sl=slice(N-max(5,N//2),N)
    elif r_range=="inner50": sl=slice(0,max(5,N//2))
    else: sl=slice(0,N)
    rf,Vf,ef,gNf=r_kpc[sl],Vobs[sl],errV[sl],gN[sl]
    ef=np.where(ef>0,ef,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc
        Vp=np.sqrt(rf*kpc_m*np.sqrt(gNf*gc))*1e-3
        return np.sum(((Vf-Vp)/ef)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x

def compute_plasticity(r_kpc,gN,gc_obs,hR):
    p={}
    p["s_c"]=np.max(gN)/gc_obs if gc_obs>0 else np.nan
    p["r_peak_hR"]=r_kpc[np.argmax(gN)]/hR if hR>0 else np.nan

    x_arr=gN/gc_obs
    if len(r_kpc)>2:
        p["E_strain"]=np.trapezoid(x_arr**2*r_kpc,r_kpc)
    else: p["E_strain"]=np.nan

    above=r_kpc[gN>gc_obs]
    p["r_plastic_hR"]=above[-1]/hR if len(above)>0 and hR>0 else 0.0

    total=np.trapezoid(gN*r_kpc,r_kpc) if len(r_kpc)>2 else 0
    if total>0:
        mp=gN>gc_obs
        if mp.any():
            p["f_p_gc"]=np.trapezoid(gN[mp]*r_kpc[mp],r_kpc[mp])/total
        else: p["f_p_gc"]=0.0
    else: p["f_p_gc"]=np.nan

    inner=r_kpc<hR
    if inner.sum()>0 and len(r_kpc)>2:
        gi=np.trapezoid(gN[inner]*r_kpc[inner],r_kpc[inner])
        gt=np.trapezoid(gN*r_kpc,r_kpc)
        p["central_conc"]=gi/gt if gt>0 else np.nan
    else: p["central_conc"]=np.nan

    r_norm=r_kpc/hR
    mn=np.abs(r_norm-1.0)<0.5
    if mn.sum()>=3:
        sl=linregress(r_norm[mn],x_arr[mn])
        p["strain_gradient"]=sl.slope
    else: p["strain_gradient"]=np.nan

    excess=np.maximum(0,gN-gc_obs)
    if len(r_kpc)>2 and gc_obs>0 and hR>0:
        Ee=np.trapezoid(excess*r_kpc,r_kpc)
        p["excess_strain"]=Ee/(gc_obs*hR**2)
    else: p["excess_strain"]=np.nan
    return p

def main():
    print("="*70); print("Condition-14 plastic region vs hR residual 31%"); print("="*70)
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

        gc_deep=fit_gc_deep(r_kpc,gN,Vobs,errV,"all")
        gc_deep_in=fit_gc_deep(r_kpc,gN,Vobs,errV,"inner50")
        gc_deep_out=fit_gc_deep(r_kpc,gN,Vobs,errV,"outer50")

        plast=compute_plasticity(r_kpc,gN,gc_obs_ms2,hR)
        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan
        rmax_hR=r_kpc[-1]/hR if hR>0 else np.nan
        delta_gc=np.log10(gc_obs_ms2)-np.log10(gc_deep) if gc_deep>0 else np.nan
        delta_io=np.log10(gc_deep_in)-np.log10(gc_deep_out) if gc_deep_out>0 else np.nan

        records.append({
            'name':name,'gc_obs':gc_obs_ms2,'gc_deep':gc_deep,
            'gc_deep_in':gc_deep_in,'gc_deep_out':gc_deep_out,
            'delta_gc':delta_gc,'delta_io':delta_io,
            'hR':hR,'vflat':vflat,'x_outer':x_outer,'rmax_hR':rmax_hR,
            **plast,
        })

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); gc_deep=arr("gc_deep")
    hR_a=arr("hR"); vflat_a=arr("vflat")
    x_outer=arr("x_outer"); rmax_hR=arr("rmax_hR")
    delta_gc=arr("delta_gc"); delta_io=arr("delta_io")
    s_c=arr("s_c"); r_peak_hR=arr("r_peak_hR"); E_strain=arr("E_strain")
    r_plastic_hR=arr("r_plastic_hR"); f_p_gc=arr("f_p_gc")
    central_conc=arr("central_conc"); strain_grad=arr("strain_gradient")
    excess_strain=arr("excess_strain")

    log_gc=np.log10(gc_obs); log_gcd=np.log10(gc_deep)
    log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a)
    log_x=np.log10(x_outer+1e-10); log_sc=np.log10(s_c+1e-10)

    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)
       &np.isfinite(log_x)&np.isfinite(log_gcd)&np.isfinite(log_sc)&np.isfinite(delta_gc))
    print(f"Valid: {m.sum()}")

    plast_params={"s_c":s_c,"r_peak/hR":r_peak_hR,"E_strain":E_strain,
                  "r_plastic/hR":r_plastic_hR,"f_p(gc)":f_p_gc,
                  "central_conc":central_conc,"strain_grad":strain_grad,
                  "excess_strain":excess_strain}

    print("\n"+"="*50); print("Test 1: plasticity parameters"); print("="*50)
    for name,ap in plast_params.items():
        v=np.isfinite(ap)&m
        if v.sum()>10:
            print(f"  {name:20s}: N={v.sum():3d} median={np.median(ap[v]):.4f} std={np.std(ap[v]):.4f}")

    print("\n"+"="*50); print("Test 2: s_c vs gc_in - gc_out"); print("="*50)
    m_io=m&np.isfinite(delta_io)
    if m_io.sum()>10:
        rho_sio,p_sio=pearsonr(log_sc[m_io],delta_io[m_io])
        print(f"  rho(s_c, in-out) = {rho_sio:+.4f}, p = {p_sio:.3e}")
        q25,q75=np.percentile(s_c[m_io],[25,75])
        lo=s_c[m_io]<q25; hi=s_c[m_io]>q75
        print(f"  s_c<Q25 ({q25:.2f}): delta_io median = {np.median(delta_io[m_io][lo]):.4f}")
        print(f"  s_c>Q75 ({q75:.2f}): delta_io median = {np.median(delta_io[m_io][hi]):.4f}")

        print(f"\n  All plasticity params vs delta_io:")
        for name,ap in plast_params.items():
            v=m_io&np.isfinite(ap)
            if v.sum()>15:
                if np.min(ap[v])>0:
                    rho,p=pearsonr(np.log10(ap[v]+1e-10),delta_io[v])
                else:
                    rho,p=pearsonr(ap[v],delta_io[v])
                print(f"    {name:20s}: rho={rho:+.4f}, p={p:.3e}")

    print("\n"+"="*50); print("Test 3: plast params vs hR, gc"); print("="*50)
    print(f"  {'param':<20s} {'rho(hR)':>10s} {'p(hR)':>12s} {'rho(gc)':>10s} {'p(gc)':>12s}")
    print("  "+"-"*68)
    for name,ap in plast_params.items():
        v=m&np.isfinite(ap)
        if v.sum()>15:
            apx=np.log10(ap[v]+1e-10) if np.min(ap[v])>0 else ap[v]
            rh,ph=pearsonr(log_hR[v],apx); rg,pg=pearsonr(log_gc[v],apx)
            print(f"  {name:<20s} {rh:+.4f}   p={ph:.3e}   {rg:+.4f}   p={pg:.3e}")

    print("\n"+"="*50); print("Test 4: hR partial corr with plasticity controls"); print("="*50)
    base=[log_vf[m],log_x[m]]
    rho_base,_=partial_corr(log_hR[m],log_gc[m],base)
    print(f"  baseline |vflat,x: rho={rho_base:+.4f}")
    print(f"\n  {'added':<20s} {'rho':>8s} {'d|rho|':>10s}")
    print("  "+"-"*40)
    best_delta=0; best_name=""
    for name,ap in plast_params.items():
        v=m&np.isfinite(ap)
        if v.sum()<m.sum()*0.7: continue
        af=ap.copy()
        af[~np.isfinite(af)]=np.nanmedian(ap)
        ctrl=np.log10(af[m]+1e-10) if np.min(af[m])>0 else af[m]
        try:
            rn,pn=partial_corr(log_hR[m],log_gc[m],base+[ctrl])
            d=abs(rn)-abs(rho_base)
            print(f"  {name:<20s} {rn:+.4f}   {d:+.4f}")
            if d<best_delta: best_delta=d; best_name=name
        except: pass
    if best_delta<-0.01: print(f"\n  Best: {best_name} (delta={best_delta:+.4f})")
    else: print(f"\n  No plasticity param significantly reduces hR partial corr")

    print("\n"+"="*50); print("Test 5: Plasticity correction model"); print("="*50)
    ratio=gc_obs[m]/gc_deep[m]
    log_ratio=np.log10(ratio)
    sl_sc=linregress(log_sc[m],log_ratio)
    print(f"  log(gc_obs/gc_deep) vs log(s_c):")
    print(f"    slope={sl_sc.slope:.4f}+/-{sl_sc.stderr:.4f}, R^2={sl_sc.rvalue**2:.4f}")
    print(f"    p={sl_sc.pvalue:.3e}")
    gc_corr=gc_deep[m]*(s_c[m]**sl_sc.slope)*10**sl_sc.intercept
    log_gcc=np.log10(gc_corr)
    rho_cc,p_cc=partial_corr(log_hR[m],log_gcc,[log_vf[m],log_x[m]])
    print(f"\n  Corrected gc hR partial corr: rho={rho_cc:+.4f} (base={rho_base:+.4f})")
    print(f"  delta={abs(rho_cc)-abs(rho_base):+.4f}")

    print("\n"+"="*50); print("Test 6: rmax/hR > 8 subset"); print("="*50)
    m8=m&(rmax_hR>8)
    if m8.sum()>15:
        rho_b8,_=partial_corr(log_hR[m8],log_gc[m8],[log_vf[m8],log_x[m8]])
        print(f"  baseline (rmax/hR>8): rho={rho_b8:+.4f} (N={m8.sum()})")
        for name,ap in [("s_c",s_c),("excess_strain",excess_strain),("f_p(gc)",f_p_gc),("central_conc",central_conc)]:
            v8=m8&np.isfinite(ap)
            if v8.sum()<10: continue
            af=ap.copy(); af[~np.isfinite(af)]=np.nanmedian(ap)
            ctrl=np.log10(af[m8]+1e-10) if np.min(af[m8])>0 else af[m8]
            try:
                rho8,_=partial_corr(log_hR[m8],log_gc[m8],[log_vf[m8],log_x[m8],ctrl])
                d8=abs(rho8)-abs(rho_b8)
                print(f"  +{name:20s}: rho={rho8:+.4f}, d={d8:+.4f}")
            except: pass

    print("\n"+"="*50); print("Test 7: x_outer and s_c independence"); print("="*50)
    rho_xs,p_xs=pearsonr(log_x[m],log_sc[m])
    print(f"  rho(x_outer, s_c)={rho_xs:+.4f}, p={p_xs:.3e}")
    rho_sh,p_sh=partial_corr(log_hR[m],log_gc[m],[log_vf[m],log_x[m],log_sc[m]])
    print(f"  rho(gc,hR|vf,x,s_c)={rho_sh:+.4f}, p={p_sh:.3e}")
    d_sc=abs(rho_sh)-abs(rho_base)
    print(f"  delta={d_sc:+.4f}")
    if d_sc<-0.02:
        pct=-d_sc/0.472*100
        print(f"  -> s_c explains additional {pct:.1f}% of hR dependence")
    else:
        print(f"  -> s_c has no independent explanatory power")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  hR partial corr:")
    print(f"    |vflat:        rho=-0.472")
    print(f"    |vflat,x:      rho={rho_base:+.4f} (x explains 69%)")
    print(f"    |vflat,x,s_c:  rho={rho_sh:+.4f} (s_c adds {d_sc:+.4f})")
    if m_io.sum()>10: print(f"\n  s_c vs gc_in-gc_out: rho={rho_sio:+.4f}")
    print(f"  s_c vs x_outer:      rho={rho_xs:+.4f}")
    print(f"\n  Condition-14 evaluation:")
    if d_sc<-0.03:
        print(f"  >> Plasticity params partially explain residual hR")
        print(f"  >> Condition-14 may function as 'small correction'")
    elif abs(d_sc)<0.03:
        print(f"  >> Plasticity params are redundant with x_outer")
        print(f"  >> Condition-14 does NOT explain residual 31%")
    else:
        print(f"  >> Plasticity params INCREASE hR correlation")
        print(f"  >> Condition-14 unrelated to residual 31%")

if __name__=="__main__": main()
