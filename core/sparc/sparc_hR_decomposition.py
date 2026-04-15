#!/usr/bin/env python3
"""
hR partial correlation decomposition: rho=-0.312 origin.
Uses TA3+phase1.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr, linregress
from numpy.linalg import lstsq
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

def partial_corr(x,y,controls):
    if len(controls)==0: return pearsonr(x,y)
    C=np.column_stack(controls+[np.ones(len(x))])
    res_x=x-C@lstsq(C,x,rcond=None)[0]
    res_y=y-C@lstsq(C,y,rcond=None)[0]
    return pearsonr(res_x,res_y)

def main():
    print("="*70)
    print("hR partial correlation decomposition: rho=-0.312 origin")
    print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)} galaxies")

    records=[]; skipped=0
    for name,info in pipe.items():
        rot=load_rotmod(name)
        if rot is None: skipped+=1; continue
        r_kpc,Vobs,errV,Vgas,Vdisk,Vbul=rot
        Yd=info['Yd']; vflat=info['vflat']; gc_obs=info['gc_obs']

        vds=np.sqrt(max(Yd,0.01))*np.abs(Vdisk)
        rpk=r_kpc[np.argmax(vds)]
        if rpk<0.01 or rpk>=r_kpc.max()*0.9: skipped+=1; continue
        hR=rpk/2.15; hR_m=hR*kpc_m
        gN=compute_gN(r_kpc,Vgas,Vdisk,Vbul,Yd=Yd)

        # Baryon fractions
        Vbar2=np.abs(Vgas)**2+Yd*np.abs(Vdisk)**2+0.7*np.abs(Vbul)**2
        valid=Vbar2>0
        f_bul=np.median(0.7*Vbul[valid]**2/Vbar2[valid]) if valid.sum()>0 else 0
        f_gas=np.median(Vgas[valid]**2/Vbar2[valid]) if valid.sum()>0 else 0
        disk_conc=np.max(np.sqrt(Yd)*np.abs(Vdisk))/vflat if vflat>0 else np.nan

        # Deep MOND fit
        def chi2_deep(lgc):
            gc=10**lgc; g_eff=np.sqrt(gN*gc); r_m=r_kpc*kpc_m
            Vp=np.sqrt(np.abs(r_m*g_eff))*1e-3
            nf=max(5,len(r_kpc)//2); Vf=Vobs[-nf:]; Vpr=Vp[-nf:]; ef=errV[-nf:]
            return np.sum(((Vf-Vpr)/ef)**2)
        res=minimize_scalar(chi2_deep,bounds=(-12,-8),method='bounded')
        gc_deep=10**res.x

        # x_outer
        n_out=max(3,len(r_kpc)//3)
        x_outer=np.median(gN[-n_out:]/gc_deep) if gc_deep>0 else np.nan

        # Proxy & eta
        proxy=(vflat*1e3)**2/(hR_m)
        gc_geom=np.sqrt(a0*proxy)
        eta=gc_obs/gc_geom if gc_geom>0 else np.nan

        records.append({
            'name':name,'gc_obs':gc_obs,'gc_deep':gc_deep,
            'hR':hR,'vflat':vflat,'Yd':Yd,
            'f_bul':f_bul,'f_gas':f_gas,'disk_conc':disk_conc,
            'x_outer':x_outer,'proxy':proxy,'eta':eta,
        })

    N=len(records); print(f"Processed: {N}, Skipped: {skipped}")
    if N<10: print("Too few."); sys.exit(1)

    gc_obs=np.array([r['gc_obs'] for r in records])
    gc_deep=np.array([r['gc_deep'] for r in records])
    hR=np.array([r['hR'] for r in records])
    vflat=np.array([r['vflat'] for r in records])
    Yd=np.array([r['Yd'] for r in records])
    f_bul=np.array([r['f_bul'] for r in records])
    f_gas=np.array([r['f_gas'] for r in records])
    disk_conc=np.array([r['disk_conc'] for r in records])
    x_outer=np.array([r['x_outer'] for r in records])
    eta=np.array([r['eta'] for r in records])

    log_gc=np.log10(gc_obs); log_gcd=np.log10(gc_deep)
    log_hR=np.log10(hR); log_vf=np.log10(vflat); log_Yd=np.log10(Yd)
    log_eta=np.log10(np.abs(eta)); log_proxy=np.log10(np.array([r['proxy'] for r in records]))
    log_x=np.log10(x_outer+1e-10)

    m=(np.isfinite(log_gc)&np.isfinite(log_hR)&np.isfinite(log_vf)&np.isfinite(log_gcd)
       &np.isfinite(log_eta)&np.isfinite(log_Yd)&np.isfinite(x_outer)
       &np.isfinite(f_bul)&np.isfinite(f_gas)&np.isfinite(disk_conc)&(eta>0))
    print(f"Valid: {m.sum()}")

    # Test 0: Baseline
    print("\n"+"="*50); print("Test 0: Baseline partial correlations"); print("="*50)
    rho_raw,p_raw=pearsonr(log_hR[m],log_gc[m])
    print(f"  raw: rho(hR,gc)={rho_raw:+.4f}, p={p_raw:.3e}")
    rho_pv,p_pv=partial_corr(log_hR[m],log_gc[m],[log_vf[m]])
    print(f"  |vflat: rho={rho_pv:+.4f}, p={p_pv:.3e}")
    rho_pvy,p_pvy=partial_corr(log_hR[m],log_gc[m],[log_vf[m],log_Yd[m]])
    print(f"  |vflat,Yd: rho={rho_pvy:+.4f}, p={p_pvy:.3e}")

    # Test 1: Yd vs hR
    print("\n"+"="*50); print("Test 1: Yd vs hR"); print("="*50)
    rho_yh,p_yh=pearsonr(log_hR[m],log_Yd[m])
    print(f"  rho(Yd,hR)={rho_yh:+.4f}, p={p_yh:.3e}")
    Yd_uniq=np.unique(np.round(Yd[m],4))
    print(f"  Yd unique values: {len(Yd_uniq)} (range [{Yd[m].min():.3f},{Yd[m].max():.3f}])")

    # Test 2: eta vs hR
    print("\n"+"="*50); print("Test 2: eta vs hR"); print("="*50)
    rho_eh,p_eh=pearsonr(log_hR[m],log_eta[m])
    print(f"  raw: rho(eta,hR)={rho_eh:+.4f}, p={p_eh:.3e}")
    rho_ehv,p_ehv=partial_corr(log_hR[m],log_eta[m],[log_vf[m]])
    print(f"  |vflat: rho={rho_ehv:+.4f}, p={p_ehv:.3e}")
    rho_ehvx,p_ehvx=partial_corr(log_hR[m],log_eta[m],[log_vf[m],log_x[m]])
    print(f"  |vflat,x: rho={rho_ehvx:+.4f}, p={p_ehvx:.3e}")

    # Test 3: gc residual (obs-deep) decomposition
    print("\n"+"="*50); print("Test 3: gc residual = log(gc_obs/gc_deep)"); print("="*50)
    resid=log_gc[m]-log_gcd[m]
    print(f"  median={np.median(resid):.4f}, std={np.std(resid):.4f}")
    for lb,arr in [("hR",log_hR[m]),("vflat",log_vf[m]),("x_outer",log_x[m]),
                    ("f_bul",f_bul[m]),("f_gas",f_gas[m]),("disk_conc",disk_conc[m])]:
        rho,p=pearsonr(arr,resid)
        print(f"  rho(resid,{lb:12s})={rho:+.4f}, p={p:.3e}")
    rho_rhv,p_rhv=partial_corr(log_hR[m],resid,[log_vf[m]])
    print(f"  rho(resid,hR|vflat)={rho_rhv:+.4f}, p={p_rhv:.3e}")

    # Test 4: Exponential disk deviation
    print("\n"+"="*50); print("Test 4: Exponential disk deviation"); print("="*50)
    vf_ms=vflat[m]*1e3; hR_m_arr=hR[m]*kpc_m
    gc_ideal=vf_ms**2/(2*np.pi*hR_m_arr)
    dev=np.log10(gc_deep[m]/gc_ideal)
    print(f"  log(gc_deep/gc_ideal): median={np.median(dev):.4f}, std={np.std(dev):.4f}")
    rho_dh,p_dh=pearsonr(log_hR[m],dev)
    print(f"  rho(deviation,hR)={rho_dh:+.4f}, p={p_dh:.3e}")

    # Test 5: Bulge/gas effects
    print("\n"+"="*50); print("Test 5: Bulge/gas effects"); print("="*50)
    rho_bh,p_bh=pearsonr(log_hR[m],f_bul[m])
    print(f"  rho(f_bul,hR)={rho_bh:+.4f}, p={p_bh:.3e}")
    rho_gh,p_gh=pearsonr(log_hR[m],f_gas[m])
    print(f"  rho(f_gas,hR)={rho_gh:+.4f}, p={p_gh:.3e}")
    rho_pvb,p_pvb=partial_corr(log_hR[m],log_gc[m],[log_vf[m],f_bul[m]])
    print(f"  rho(gc,hR|vflat,f_bul)={rho_pvb:+.4f}, p={p_pvb:.3e}")
    rho_pvg,p_pvg=partial_corr(log_hR[m],log_gc[m],[log_vf[m],f_gas[m]])
    print(f"  rho(gc,hR|vflat,f_gas)={rho_pvg:+.4f}, p={p_pvg:.3e}")

    # Test 6: Sequential partial correlations
    print("\n"+"="*50); print("Test 6: Sequential partial correlations"); print("="*50)
    steps=[
        ("raw",[]),
        ("|vflat",[log_vf[m]]),
        ("|vflat,x_outer",[log_vf[m],log_x[m]]),
        ("|vflat,x,f_bul",[log_vf[m],log_x[m],f_bul[m]]),
        ("|vflat,x,f_bul,f_gas",[log_vf[m],log_x[m],f_bul[m],f_gas[m]]),
        ("|vflat,x,f_bul,f_gas,disk_conc",[log_vf[m],log_x[m],f_bul[m],f_gas[m],disk_conc[m]]),
        ("|vflat,x,f_bul,f_gas,dc,Yd",[log_vf[m],log_x[m],f_bul[m],f_gas[m],disk_conc[m],log_Yd[m]]),
    ]
    print(f"  {'Controls':<45s} {'rho':>8s} {'p':>12s} {'d|rho|':>8s}")
    print("  "+"-"*75)
    prev=None
    for lb,ctrls in steps:
        rho,p=partial_corr(log_hR[m],log_gc[m],ctrls)
        ch=f"{abs(rho)-abs(prev):+.4f}" if prev is not None else "--"
        print(f"  {lb:<45s} {rho:+.4f}   p={p:.3e}   {ch:>8s}")
        prev=rho

    # Test 7: BTFR alpha vs deep MOND alpha
    print("\n"+"="*50); print("Test 7: hR exponent theory vs obs"); print("="*50)
    # gc_btfr (large r)
    gc_btfr_arr=[]
    for rec in records:
        rot=load_rotmod(rec['name'])
        if rot is None: gc_btfr_arr.append(np.nan); continue
        r_i,_,_,Vg_i,Vd_i,Vb_i=rot
        gN_i=compute_gN(r_i,Vg_i,Vd_i,Vb_i,Yd=rec['Yd'])
        vf_ms=rec['vflat']*1e3; no=max(3,len(r_i)//3)
        ro=r_i[-no:]*kpc_m; gNo=gN_i[-no:]; vo=gNo>0
        gc_btfr_arr.append(np.median(vf_ms**4/(ro[vo]**2*gNo[vo])) if vo.sum()>0 else np.nan)
    gc_btfr=np.array(gc_btfr_arr)
    log_gcb=np.log10(gc_btfr)
    mb=m&np.isfinite(log_gcb)
    if mb.sum()>10:
        sl_b=linregress(log_proxy[mb],log_gcb[mb])
        sl_d=linregress(log_proxy[m],log_gcd[m])
        print(f"  gc_btfr (large r) vs proxy: alpha={sl_b.slope:.4f}+/-{sl_b.stderr:.4f}")
        print(f"  gc_deep (full RC) vs proxy: alpha={sl_d.slope:.4f}+/-{sl_d.stderr:.4f}")
        print(f"  Difference: {sl_b.slope-sl_d.slope:+.4f}")
        # Multivariate
        Xb=np.column_stack([log_vf[mb],log_hR[mb],np.ones(mb.sum())])
        cb=lstsq(Xb,log_gcb[mb],rcond=None)[0]
        Xd=np.column_stack([log_vf[m],log_hR[m],np.ones(m.sum())])
        cd=lstsq(Xd,log_gcd[m],rcond=None)[0]
        print(f"  gc_btfr multivar: vflat^{cb[0]:.3f} * hR^{cb[1]:.3f}")
        print(f"  gc_deep multivar: vflat^{cd[0]:.3f} * hR^{cd[1]:.3f}")
        print(f"  hR exponent: BTFR={cb[1]:.3f} vs deep={cd[1]:.3f} (diff={cb[1]-cd[1]:+.3f})")

    # Summary
    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\n  Baseline: rho(gc,hR|vflat)={rho_pv:+.4f}")
    print(f"  After all controls: rho={rho:+.4f}")
    print(f"  Total reduction: {abs(rho_pv)-abs(rho):.4f}")
    print(f"\n  gc residual (obs-deep) vs hR: {rho_rhv:+.4f}")
    print(f"  eta vs hR: {rho_eh:+.4f}")
    print(f"  eta|vflat vs hR: {rho_ehv:+.4f}")

    if abs(rho_rhv)<0.1 and abs(rho_ehv)<0.1:
        print("\n  >> hR effect fully explained by deep MOND structure")
    elif abs(rho_rhv)>0.1:
        print(f"\n  >> Residual hR effect in gc_obs-gc_deep: {rho_rhv:+.3f}")
    if abs(rho_eh)>0.15:
        print(f"  >> eta has hR dependence: {rho_eh:+.3f}")

    print("\n[DONE]")

if __name__=="__main__": main()
