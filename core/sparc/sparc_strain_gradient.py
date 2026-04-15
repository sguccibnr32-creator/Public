#!/usr/bin/env python3
"""
Strain gradient energy -- does (d_epsilon/dr)^2 explain residual 31%?
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

def fit_gc_deep(r_kpc,gN,Vobs,errV):
    e=np.where(errV>0,errV,np.median(Vobs)*0.1)
    def chi2(lgc):
        gc=10**lgc
        Vp=np.sqrt(r_kpc*kpc_m*np.sqrt(gN*gc))*1e-3
        return np.sum(((Vobs-Vp)/e)**2)
    res=minimize_scalar(chi2,bounds=(-12,-8),method='bounded')
    return 10**res.x

def compute_strain_energies(r_kpc,gN,gc,hR):
    x=gN/gc; eps=np.sqrt(x)
    N=len(r_kpc); result={}
    if N>2: result["E_local"]=np.trapezoid(eps**2*r_kpc,r_kpc)
    else: result["E_local"]=np.nan
    if N>3:
        dr=np.diff(r_kpc); deps=np.diff(eps); mask=dr>0
        if mask.sum()>2:
            dedr=deps[mask]/dr[mask]
            r_mid=(r_kpc[:-1]+r_kpc[1:])[mask]/2
            result["E_grad"]=np.trapezoid(dedr**2*r_mid,r_mid)
            dedr_n=dedr*hR
            result["E_grad_norm"]=np.trapezoid(dedr_n**2*r_mid,r_mid)
            result["grad_peak"]=np.max(np.abs(dedr))
            result["grad_peak_norm"]=np.max(np.abs(dedr_n))
            idx=np.argmax(np.abs(dedr))
            result["r_grad_peak_hR"]=r_mid[idx]/hR
            n_h=len(dedr)//2
            if n_h>=2:
                result["grad_outer_inner"]=np.median(np.abs(dedr[n_h:]))/(np.median(np.abs(dedr[:n_h]))+1e-30)
            else: result["grad_outer_inner"]=np.nan
        else:
            for k in ["E_grad","E_grad_norm","grad_peak","grad_peak_norm","r_grad_peak_hR","grad_outer_inner"]:
                result[k]=np.nan
    else:
        for k in ["E_grad","E_grad_norm","grad_peak","grad_peak_norm","r_grad_peak_hR","grad_outer_inner"]:
            result[k]=np.nan
    if np.isfinite(result.get("E_grad",np.nan)) and result.get("E_local",0)>0:
        result["grad_local_ratio"]=result["E_grad"]/result["E_local"]
    else: result["grad_local_ratio"]=np.nan
    gN_peak=np.max(gN)
    if np.isfinite(result.get("E_grad",np.nan)):
        result["e_grad_density"]=result["E_grad"]/((hR)**2*gc**2+1e-30)
    else: result["e_grad_density"]=np.nan
    return result

def main():
    print("="*70); print("Strain gradient energy for hR residual 31%"); print("="*70)
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
        gc_deep=fit_gc_deep(r_kpc,gN,Vobs,errV)
        strain=compute_strain_energies(r_kpc,gN,gc_deep,hR)
        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan
        rmax_hR=r_kpc[-1]/hR if hR>0 else np.nan
        records.append({
            'name':name,'gc_obs':gc_obs_ms2,'gc_deep':gc_deep,
            'hR':hR,'vflat':vflat,'x_outer':x_outer,'rmax_hR':rmax_hR,
            'E_grad_d':strain.get("E_grad",np.nan),
            'E_grad_norm_d':strain.get("E_grad_norm",np.nan),
            'E_local_d':strain.get("E_local",np.nan),
            'grad_peak_d':strain.get("grad_peak",np.nan),
            'grad_peak_norm_d':strain.get("grad_peak_norm",np.nan),
            'grad_local_ratio_d':strain.get("grad_local_ratio",np.nan),
            'r_grad_peak_hR_d':strain.get("r_grad_peak_hR",np.nan),
            'grad_oi_d':strain.get("grad_outer_inner",np.nan),
            'e_grad_density_d':strain.get("e_grad_density",np.nan),
        })

    N=len(records); print(f"Processed: {N}")
    def arr(k): return np.array([r[k] for r in records],dtype=float)

    gc_obs=arr("gc_obs"); gc_deep=arr("gc_deep")
    hR_a=arr("hR"); vflat_a=arr("vflat")
    x_outer=arr("x_outer"); rmax_hR=arr("rmax_hR")
    E_grad=arr("E_grad_d"); E_grad_norm=arr("E_grad_norm_d"); E_local=arr("E_local_d")
    grad_peak=arr("grad_peak_d"); grad_peak_norm=arr("grad_peak_norm_d")
    grad_local=arr("grad_local_ratio_d"); r_grad_hR=arr("r_grad_peak_hR_d")
    grad_oi=arr("grad_oi_d"); e_grad_dens=arr("e_grad_density_d")

    log_gc=np.log10(gc_obs); log_gcd=np.log10(gc_deep)
    log_hR=np.log10(hR_a); log_vf=np.log10(vflat_a); log_x=np.log10(x_outer+1e-10)

    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)&np.isfinite(log_x)
       &np.isfinite(log_gcd)&np.isfinite(E_grad)&(E_grad>0)&np.isfinite(E_local)&(E_local>0))
    print(f"Valid: {m.sum()}")

    grad_params={"E_grad":E_grad,"E_grad_norm":E_grad_norm,"E_local":E_local,
                 "grad_peak":grad_peak,"grad_peak_norm":grad_peak_norm,
                 "E_grad/E_local":grad_local,"r_grad_peak/hR":r_grad_hR,
                 "grad_outer/inner":grad_oi,"e_grad_density":e_grad_dens}

    print("\n"+"="*50); print("Test 1: parameter statistics"); print("="*50)
    for name,ap in grad_params.items():
        v=np.isfinite(ap)&m&(ap>0)
        if v.sum()>10:
            print(f"  {name:20s}: N={v.sum():3d} median={np.median(ap[v]):.4e} std/mean={np.std(ap[v])/np.mean(ap[v]):.3f}")

    print("\n"+"="*50); print("Test 2: corr with hR, x, gc"); print("="*50)
    print(f"  {'param':<20s} {'rho(hR)':>10s} {'rho(x)':>10s} {'rho(gc)':>10s} {'rho(hR|x)':>10s}")
    print("  "+"-"*64)
    for name,ap in grad_params.items():
        v=np.isfinite(ap)&m&(ap>0)
        if v.sum()<20: continue
        log_ap=np.log10(ap[v]+1e-30)
        rh,_=pearsonr(log_hR[v],log_ap)
        rx,_=pearsonr(log_x[v],log_ap)
        rg,_=pearsonr(log_gc[v],log_ap)
        rhx,_=partial_corr(log_hR[v],log_ap,[log_x[v]])
        print(f"  {name:<20s} {rh:+.4f}     {rx:+.4f}     {rg:+.4f}     {rhx:+.4f}")

    print("\n"+"="*50); print("Test 3: hR partial corr with grad params"); print("="*50)
    base=[log_vf[m],log_x[m]]
    rho_base,_=partial_corr(log_hR[m],log_gc[m],base)
    print(f"  baseline |vflat,x: rho={rho_base:+.4f}")
    print(f"\n  {'added':<20s} {'rho':>8s} {'d|rho|':>10s} {'p':>12s}")
    print("  "+"-"*52)
    best_delta=0; best_name=""; best_ctrl=None
    for name,ap in grad_params.items():
        v=np.isfinite(ap)&m&(ap>0)
        if v.sum()<m.sum()*0.7: continue
        af=ap.copy(); af[~(np.isfinite(ap)&(ap>0))]=np.nanmedian(ap[ap>0])
        ctrl=np.log10(af[m]+1e-30)
        try:
            rn,pn=partial_corr(log_hR[m],log_gc[m],base+[ctrl])
            d=abs(rn)-abs(rho_base)
            print(f"  {name:<20s} {rn:+.4f}   {d:+.4f}   p={pn:.3e}")
            if d<best_delta: best_delta=d; best_name=name; best_ctrl=ctrl
        except: pass
    if best_delta<-0.01:
        print(f"\n  Best: {best_name} (delta={best_delta:+.4f})")
        print(f"  -> Adds {-best_delta/0.472*100:.1f}% explanation")
    else:
        print(f"\n  No grad param reduces hR partial corr")

    print("\n"+"="*50); print("Test 4: E_grad/E_local independence"); print("="*50)
    v_glr=np.isfinite(grad_local)&m&(grad_local>0)
    if v_glr.sum()>20:
        log_glr_v=np.log10(grad_local[v_glr])
        rho_glx,p_glx=pearsonr(log_x[v_glr],log_glr_v)
        print(f"  rho(ratio, x_outer)={rho_glx:+.4f}, p={p_glx:.3e}")
        rho_glh,p_glh=partial_corr(log_hR[v_glr],log_glr_v,[log_x[v_glr]])
        print(f"  rho(ratio, hR|x)={rho_glh:+.4f}, p={p_glh:.3e}")
        sl=linregress(log_hR[v_glr],log_glr_v)
        print(f"  log(ratio) vs log(hR): slope={sl.slope:.4f}+/-{sl.stderr:.4f} (predict -2)")

    print("\n"+"="*50); print("Test 5: gc residual vs grad params"); print("="*50)
    for name,ap in grad_params.items():
        v=np.isfinite(ap)&m&(ap>0)
        if v.sum()<20: continue
        log_ap=np.log10(ap[v]+1e-30)
        resid_v=(log_gc[v]-log_gcd[v])
        rho_r,_=pearsonr(log_ap,resid_v)
        rho_rx,_=partial_corr(log_ap,resid_v,[log_x[v]])
        print(f"  {name:20s}: rho(resid)={rho_r:+.4f}  rho(resid|x)={rho_rx:+.4f}")

    print("\n"+"="*50); print("Test 6: rmax/hR>8 subset"); print("="*50)
    m8=m&(rmax_hR>8)
    if m8.sum()>15:
        rho_b8,_=partial_corr(log_hR[m8],log_gc[m8],[log_vf[m8],log_x[m8]])
        print(f"  baseline (rmax/hR>8,N={m8.sum()}): rho={rho_b8:+.4f}")
        for name,ap in [("E_grad",E_grad),("E_grad/E_local",grad_local),
                        ("grad_peak_norm",grad_peak_norm),("E_grad_norm",E_grad_norm)]:
            v8=m8&np.isfinite(ap)&(ap>0)
            if v8.sum()<10: continue
            af=ap.copy(); af[~(np.isfinite(ap)&(ap>0))]=np.nanmedian(ap[ap>0])
            ctrl=np.log10(af[m8]+1e-30)
            try:
                rho8,_=partial_corr(log_hR[m8],log_gc[m8],[log_vf[m8],log_x[m8],ctrl])
                d8=abs(rho8)-abs(rho_b8)
                print(f"  +{name:20s}: rho={rho8:+.4f}, d={d8:+.4f}")
            except: pass

    print("\n"+"="*50); print("Test 7: Sequential partial corr"); print("="*50)
    glr_f=grad_local.copy()
    glr_f[~(np.isfinite(glr_f)&(glr_f>0))]=np.nanmedian(grad_local[np.isfinite(grad_local)&(grad_local>0)])
    log_glr_all=np.log10(glr_f[m]+1e-30)

    steps=[
        ("raw",[]),
        ("|vflat",[log_vf[m]]),
        ("|vflat,x",[log_vf[m],log_x[m]]),
        ("|vflat,x,E_grad/E_local",[log_vf[m],log_x[m],log_glr_all]),
    ]
    if best_ctrl is not None and best_name!="E_grad/E_local":
        steps.append((f"|vflat,x,{best_name}",[log_vf[m],log_x[m],best_ctrl]))
        steps.append((f"|vflat,x,ratio,{best_name}",[log_vf[m],log_x[m],log_glr_all,best_ctrl]))

    print(f"  {'controls':<40s} {'rho':>8s} {'p':>12s}")
    print("  "+"-"*62)
    for lb,ctrls in steps:
        try:
            rho_s,p_s=partial_corr(log_hR[m],log_gc[m],ctrls)
            print(f"  {lb:<40s} {rho_s:+.4f}   p={p_s:.3e}")
        except Exception as e: print(f"  {lb:<40s} ERROR")

    print("\n"+"="*50); print("Test 8: Theoretical scaling"); print("="*50)
    v_eg=np.isfinite(E_grad)&m&(E_grad>0)
    if v_eg.sum()>20:
        sl_eg=linregress(log_hR[v_eg],np.log10(E_grad[v_eg]))
        print(f"  E_grad vs hR: slope={sl_eg.slope:.4f}+/-{sl_eg.stderr:.4f}")
    v_el=np.isfinite(E_local)&m&(E_local>0)
    if v_el.sum()>20:
        sl_el=linregress(log_hR[v_el],np.log10(E_local[v_el]))
        print(f"  E_local vs hR: slope={sl_el.slope:.4f}+/-{sl_el.stderr:.4f}")

    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  hR partial corr decomposition:")
    print(f"    |vflat:          rho=-0.472")
    print(f"    |vflat,x:        rho={rho_base:+.4f}")
    if best_delta<-0.01:
        rho_best,_=partial_corr(log_hR[m],log_gc[m],base+[best_ctrl])
        print(f"    |vflat,x,{best_name}: rho={rho_best:+.4f}")

    if best_delta<-0.02:
        print(f"\n  >> Grad energy partially explains residual hR [*]")
        print(f"  >> Physical mechanism: membrane strain gradient")
    elif best_delta<-0.01:
        print(f"\n  >> Slight effect ({-best_delta/0.472*100:.1f}%) but not decisive")
    else:
        print(f"\n  >> Gradient term does NOT explain residual 31%")

if __name__=="__main__": main()
