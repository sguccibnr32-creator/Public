#!/usr/bin/env python3
"""
hR residual 31% decomposition via high-order shape parameters.
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

def compute_shape_params(r_kpc,Vobs,errV,Vgas,Vdisk,Vbul,vflat,hR,gc_obs_ms2,Yd=0.5):
    gN=compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=Yd)
    N=len(r_kpc); params={}
    params["rmax_hR"]=r_kpc[-1]/hR if hR>0 else np.nan

    log_r=np.log10(r_kpc); log_gN=np.log10(gN+1e-30)
    valid=np.isfinite(log_gN)&np.isfinite(log_r)&(gN>0)
    if valid.sum()>=5:
        dr=np.diff(log_r[valid]); dg=np.diff(log_gN[valid])
        mdr=np.abs(dr)>1e-10
        if mdr.sum()>=3:
            grad=dg[mdr]/dr[mdr]
            n_out=max(1,len(grad)//2)
            params["grad_outer"]=np.median(grad[-n_out:])
            params["grad_inner"]=np.median(grad[:n_out])
            params["grad_ratio"]=params["grad_inner"]/params["grad_outer"] if abs(params["grad_outer"])>0.01 else np.nan
            params["grad_std"]=np.std(grad)
        else:
            for k in ["grad_outer","grad_inner","grad_ratio","grad_std"]: params[k]=np.nan
    else:
        for k in ["grad_outer","grad_inner","grad_ratio","grad_std"]: params[k]=np.nan

    gc=gc_obs_ms2
    if gc>0:
        x_arr=gN/gc; r_lo=np.nan; r_hi=np.nan
        for i in range(N-1):
            if x_arr[i]>=3.0 and x_arr[i+1]<3.0:
                frac=(3.0-x_arr[i+1])/(x_arr[i]-x_arr[i+1])
                r_lo=r_kpc[i+1]+frac*(r_kpc[i]-r_kpc[i+1]); break
        for i in range(N-1):
            if x_arr[i]>=0.3 and x_arr[i+1]<0.3:
                frac=(0.3-x_arr[i+1])/(x_arr[i]-x_arr[i+1])
                r_hi=r_kpc[i+1]+frac*(r_kpc[i]-r_kpc[i+1]); break
        if np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi>r_lo:
            params["transition_width"]=(r_hi-r_lo)/hR
            params["transition_center"]=(r_hi+r_lo)/2/hR
        else:
            params["transition_width"]=np.nan; params["transition_center"]=np.nan
    else:
        params["transition_width"]=np.nan; params["transition_center"]=np.nan

    Vd_abs=np.abs(Vdisk)
    if np.max(Vd_abs)>0:
        idx=np.argmax(Vd_abs); r_peak=r_kpc[idx]
        params["peak_ratio"]=r_peak/(2.2*hR)
        if r_kpc[-1]>hR and r_kpc[0]<hR:
            V_at_hR=np.interp(hR,r_kpc,Vd_abs)
            params["shape_conc"]=V_at_hR/np.max(Vd_abs)
        else:
            params["shape_conc"]=np.nan
    else:
        params["peak_ratio"]=np.nan; params["shape_conc"]=np.nan

    def fit_gc_deep(r,gN_,Vo,eV):
        def chi2(lgc):
            gc_=10**lgc; g_eff=np.sqrt(gN_*gc_)
            Vp=np.sqrt(r*kpc_m*g_eff)*1e-3
            e=np.where(eV>0,eV,np.median(Vo)*0.1)
            return np.sum(((Vo-Vp)/e)**2)
        res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
        return 10**res.x
    gc_deep=fit_gc_deep(r_kpc,gN,Vobs,errV)
    params["gc_deep"]=gc_deep

    g_eff_d=np.sqrt(gN*gc_deep); V_d=np.sqrt(r_kpc*kpc_m*g_eff_d)*1e-3
    e=np.where(errV>0,errV,np.median(Vobs)*0.1)
    frac_r=(Vobs-V_d)/e
    n_half=N//2
    if n_half>=3:
        params["resid_inner"]=np.median(frac_r[:n_half])
        params["resid_outer"]=np.median(frac_r[n_half:])
        params["resid_asymm"]=params["resid_inner"]-params["resid_outer"]
    else:
        for k in ["resid_inner","resid_outer","resid_asymm"]: params[k]=np.nan

    n_out=max(3,N//3)
    params["x_outer"]=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan

    if valid.sum()>=7:
        try: params["gN_curvature"]=np.polyfit(log_r[valid],log_gN[valid],2)[0]
        except: params["gN_curvature"]=np.nan
    else: params["gN_curvature"]=np.nan

    Vbar_sq=np.abs(Vgas)*Vgas+Yd*np.abs(Vdisk)*Vdisk+Yb_default*np.abs(Vbul)*Vbul
    Vbar_abs=np.sqrt(np.abs(Vbar_sq))
    params["Vbar_Vobs_outer"]=np.median(Vbar_abs[-n_out:]/(Vobs[-n_out:]+1e-10))
    V_out=Vobs[-n_out:]
    params["flatness"]=np.std(V_out)/np.mean(V_out) if np.mean(V_out)>0 else np.nan

    va=np.abs(Vbar_sq); v=va>0
    params["f_bul"]=np.median(Yb_default*Vbul[v]**2/va[v]) if v.sum()>0 else 0
    params["f_gas"]=np.median(Vgas[v]**2/va[v]) if v.sum()>0 else 0
    params["disk_conc"]=np.max(np.abs(Vdisk))*np.sqrt(Yd)/vflat if vflat>0 else np.nan
    return params

def main():
    print("="*70)
    print("hR residual 31%: high-order shape parameter decomposition")
    print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)} galaxies")

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

        params=compute_shape_params(r_kpc,Vobs,errV,Vgas,Vdisk,Vbul,vflat,hR,gc_obs_ms2,Yd=Yd)
        proxy=(vflat*1e3)**2/(hR*kpc_m)
        gc_geom=np.sqrt(a0*proxy)
        eta=gc_obs_ms2/gc_geom if gc_geom>0 else np.nan

        records.append({'name':name,'gc_obs':gc_obs_ms2,'hR':hR,'vflat':vflat,'eta':eta,**params})

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); gc_deep=arr("gc_deep"); hR_a=arr("hR"); vflat_a=arr("vflat"); eta_a=arr("eta")
    log_gc=np.log10(gc_obs); log_gcd=np.log10(gc_deep)
    log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a); log_eta=np.log10(np.abs(eta_a))

    x_outer=arr("x_outer"); rmax_hR=arr("rmax_hR")
    grad_outer=arr("grad_outer"); grad_inner=arr("grad_inner")
    grad_ratio=arr("grad_ratio"); grad_std=arr("grad_std")
    trans_width=arr("transition_width"); trans_center=arr("transition_center")
    peak_ratio=arr("peak_ratio"); shape_conc=arr("shape_conc")
    resid_asymm=arr("resid_asymm"); gN_curvature=arr("gN_curvature")
    Vbar_Vobs=arr("Vbar_Vobs_outer"); flatness=arr("flatness")

    log_x=np.log10(x_outer+1e-10)
    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)&np.isfinite(log_x)
       &np.isfinite(log_gcd)&(eta_a>0)&np.isfinite(log_eta))
    print(f"Valid: {m.sum()}")

    new_params={"rmax/hR":rmax_hR,"grad_outer":grad_outer,"grad_inner":grad_inner,
                "grad_ratio":grad_ratio,"grad_std":grad_std,"trans_width":trans_width,
                "trans_center":trans_center,"peak_ratio":peak_ratio,"shape_conc":shape_conc,
                "resid_asymm":resid_asymm,"gN_curvature":gN_curvature,
                "Vbar/Vobs_out":Vbar_Vobs,"flatness":flatness}

    print("\n"+"="*50); print("Test 1: parameter statistics"); print("="*50)
    for name,ap in new_params.items():
        v=np.isfinite(ap)&m
        if v.sum()>10: print(f"  {name:20s}: N={v.sum():3d}, median={np.median(ap[v]):+.3f}, std={np.std(ap[v]):.3f}")

    print("\n"+"="*50); print("Test 2: correlations with hR, gc"); print("="*50)
    print(f"  {'param':<20s} {'rho(hR)':>10s} {'p(hR)':>12s} {'rho(gc)':>10s} {'p(gc)':>12s}")
    print("  "+"-"*68)
    for name,ap in new_params.items():
        v=np.isfinite(ap)&m
        if v.sum()>15:
            rh,ph=pearsonr(log_hR[v],ap[v]); rg,pg=pearsonr(log_gc[v],ap[v])
            print(f"  {name:<20s} {rh:+.4f}   p={ph:.3e}  {rg:+.4f}   p={pg:.3e}")

    print("\n"+"="*50); print("Test 3: sequential partial correlations"); print("="*50)
    base=[log_vf[m],log_x[m]]
    rho_base,p_base=partial_corr(log_hR[m],log_gc[m],base)
    print(f"  baseline |vflat,x_outer: rho={rho_base:+.4f}, p={p_base:.3e}")
    print()
    print(f"  {'added param':<20s} {'rho':>8s} {'p':>12s} {'d|rho|':>10s} {'N':>5s}")
    print("  "+"-"*60)
    results=[]
    for name,ap in new_params.items():
        v=np.isfinite(ap)&m
        if v.sum()<30: continue
        af=ap.copy(); nm=~np.isfinite(af)
        if nm.sum()>0: af[nm]=np.nanmedian(ap)
        try:
            rho_new,p_new=partial_corr(log_hR[m],log_gc[m],base+[af[m]])
            delta=abs(rho_new)-abs(rho_base)
            print(f"  {name:<20s} {rho_new:+.4f}   p={p_new:.3e}   {delta:+.4f}   {v.sum():>5d}")
            results.append((name,rho_new,delta,af))
        except Exception as e: print(f"  {name:<20s} ERROR")

    print("\n"+"="*50); print("Test 4: top 3 combined"); print("="*50)
    results.sort(key=lambda x:x[2])
    if len(results)>=3:
        top3=results[:3]
        print("  Top 3 (most reducing |rho|):")
        for n,r,d,_ in top3: print(f"    {n}: delta={d:+.4f}, rho={r:+.4f}")
        ctrls=base+[r[3][m] for r in top3]
        rho_t3,p_t3=partial_corr(log_hR[m],log_gc[m],ctrls)
        print(f"\n  Top 3 combined: rho={rho_t3:+.4f}, p={p_t3:.3e}")
        print(f"  Reduction from base: {abs(rho_base)-abs(rho_t3):.4f}")

    print("\n"+"="*50); print("Test 5: all parameters"); print("="*50)
    ctrls_all=base.copy()
    for _,_,_,af in results: ctrls_all.append(af[m])
    try:
        rho_all,p_all=partial_corr(log_hR[m],log_gc[m],ctrls_all)
        print(f"  rho={rho_all:+.4f}, p={p_all:.3e}")
        print(f"  Reduction from base: {abs(rho_base)-abs(rho_all):.4f}")
        base_from_vf=0.472  # from previous analysis
        print(f"  From |vflat baseline (0.472):")
        print(f"    x_outer alone: {(0.472-abs(rho_base))/0.472*100:.1f}%")
        print(f"    + all new: {(0.472-abs(rho_all))/0.472*100:.1f}%")
    except Exception as e: print(f"  ERROR: {e}")

    print("\n"+"="*50); print("Test 6: observational systematics"); print("="*50)
    resid=log_gc[m]-log_gcd[m]
    rmv=np.isfinite(rmax_hR[m])
    if rmv.sum()>10:
        rho_rm,p_rm=pearsonr(rmax_hR[m][rmv],resid[rmv])
        print(f"  rho(gc_resid, rmax/hR)={rho_rm:+.4f}, p={p_rm:.3e}")
        rmm=np.median(rmax_hR[m][rmv])
        hi=rmax_hR[m][rmv]>rmm; lo=~hi
        print(f"  rmax/hR>{rmm:.1f}: resid median={np.median(resid[rmv][hi]):.4f}")
        print(f"  rmax/hR<{rmm:.1f}: resid median={np.median(resid[rmv][lo]):.4f}")

    print("\n"+"="*50); print("Test 7: rmax/hR cut"); print("="*50)
    for rmax_cut in [5,8,10,15]:
        mc=m&(rmax_hR>rmax_cut)
        if mc.sum()>20:
            rho_c,p_c=partial_corr(log_hR[mc],log_gc[mc],[log_vf[mc],log_x[mc]])
            print(f"  rmax/hR>{rmax_cut:2d}: N={mc.sum():3d}, rho={rho_c:+.4f}, p={p_c:.3e}")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  |vflat: rho=-0.472")
    print(f"  |vflat,x_outer: rho={rho_base:+.4f} (x_outer explains 69%)")
    if len(results)>=3: print(f"  |vflat,x,top3: rho={rho_t3:+.4f}")
    try: print(f"  |all params: rho={rho_all:+.4f}")
    except: pass

    print(f"\n  Residual 31% breakdown:")
    for name,rho,delta,_ in results[:5]:
        pct=-delta/0.472*100
        print(f"    {name:<20s}: {pct:+.1f}%")

if __name__=="__main__": main()
