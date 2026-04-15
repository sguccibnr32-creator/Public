#!/usr/bin/env python3
"""
Membrane stiffness kappa estimation.
Uses TA3+phase1.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import spearmanr, pearsonr
import csv, warnings
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
for p,l in [(ROTMOD,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not os.path.exists(p): print(f'[ERROR] {l}: {p}'); sys.exit(1)

a0 = 1.2e-10
kpc_m = 3.086e19

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

def load_rotcurve(gname):
    fname=os.path.join(ROTMOD,f"{gname}_rotmod.dat")
    if not os.path.exists(fname): return None
    rad,vobs,vgas,vdisk,vbul=[],[],[],[],[]
    with open(fname,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                rad.append(float(parts[0])); vobs.append(float(parts[1]))
                vgas.append(float(parts[3])); vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
            except: continue
    if len(rad)<5: return None
    return {'r':np.array(rad),'vobs':np.array(vobs),'vgas':np.array(vgas),
            'vdisk':np.array(vdisk),'vbul':np.array(vbul)}

def compute_strain_quantities(rc,gc_deep,Yd=0.5):
    r_m=rc['r']*kpc_m
    v_bar2=(Yd*rc['vdisk']**2+rc['vgas']**2+rc['vbul']**2)*1e6
    gN=v_bar2/r_m
    if gc_deep<=0: return None
    gN_pos=np.maximum(gN,0)
    eps=np.sqrt(gN_pos/gc_deep)
    if len(r_m)<3: return None
    deps=np.zeros_like(eps)
    deps[0]=(eps[1]-eps[0])/(r_m[1]-r_m[0])
    deps[-1]=(eps[-1]-eps[-2])/(r_m[-1]-r_m[-2])
    for i in range(1,len(eps)-1):
        deps[i]=(eps[i+1]-eps[i-1])/(r_m[i+1]-r_m[i-1])
    E_grad=np.mean(deps**2); E_local=np.mean(eps**2)
    if E_local==0: return None
    return {'E_grad':E_grad,'E_local':E_local,'ratio':E_grad/E_local,
            'mean_eps':np.mean(eps)}

def compute_gc_deep(rc,Yd=0.5):
    r_m=rc['r']*kpc_m
    v_bar2=(Yd*rc['vdisk']**2+rc['vgas']**2+rc['vbul']**2)*1e6
    v_obs2=(rc['vobs']*1e3)**2
    gN=v_bar2/r_m; gobs=v_obs2/r_m
    mask=gN>0
    gc_pts=gobs[mask]**2/gN[mask]
    if len(gc_pts)<3: return None
    gc_med=np.median(gc_pts)
    if gc_med<=0: return None
    x=gN[mask]/gc_med; deep=x<1.0
    if np.sum(deep)<3: return gc_med
    return np.median(gc_pts[deep])

def compute_hR(rc,Yd):
    vds=np.sqrt(max(Yd,0.01))*np.abs(rc['vdisk'])
    rpk=rc['r'][np.argmax(vds)]
    if rpk<0.01 or rpk>=rc['r'].max()*0.9: return None
    return rpk/2.15

def main():
    print("="*70); print("kappa (membrane stiffness) estimation"); print("="*70)
    pipe=load_pipeline(); print(f"Pipeline: {len(pipe)} galaxies")

    results=[]; n_loaded=0; n_skipped=0
    for gname,info in pipe.items():
        gc_obs=info['gc_obs']; vflat=info['vflat']; Yd=info['Yd']
        if gc_obs<=0 or vflat<=0: n_skipped+=1; continue
        rc=load_rotcurve(gname)
        if rc is None: n_skipped+=1; continue
        hR=compute_hR(rc,Yd)
        if hR is None: n_skipped+=1; continue
        gc_deep=compute_gc_deep(rc,Yd=Yd)
        if gc_deep is None or gc_deep<=0: n_skipped+=1; continue
        sq=compute_strain_quantities(rc,gc_deep,Yd=Yd)
        if sq is None: n_skipped+=1; continue
        results.append({'name':gname,'gc_obs':gc_obs,'gc_deep':gc_deep,
            'vflat':vflat,'hR':hR,'Yd':Yd,'ratio':sq['ratio'],
            'E_grad':sq['E_grad'],'E_local':sq['E_local'],
            'rmax_hR':rc['r'][-1]/hR if hR>0 else 0})
        n_loaded+=1

    print(f"Loaded: {n_loaded}, Skipped: {n_skipped}")
    if n_loaded<20: print("ERROR"); sys.exit(1)

    gc_obs=np.array([r['gc_obs'] for r in results])
    gc_deep=np.array([r['gc_deep'] for r in results])
    ratio=np.array([r['ratio'] for r in results])
    vflat=np.array([r['vflat'] for r in results])
    hR=np.array([r['hR'] for r in results])
    Yd=np.array([r['Yd'] for r in results])
    rmax_hR=np.array([r['rmax_hR'] for r in results])

    N=len(gc_obs)
    log_excess=np.log10(gc_obs/gc_deep)
    print(f"\ngc_obs/gc_deep: median={np.median(gc_obs/gc_deep):.3f}, std(log)={np.std(log_excess):.3f} dex")
    print(f"ratio: median={np.median(ratio):.4e}, range=[{np.min(ratio):.4e},{np.max(ratio):.4e}]")

    # (1) kappa MLE
    print("\n"+"="*70); print("(1) kappa MLE"); print("="*70)
    def chi2_kappa(k):
        model=np.log10(np.maximum(1+k*ratio,1e-10))
        return np.sum((log_excess-model)**2)
    ratio_scale=np.median(ratio)
    kappa_max=10.0/ratio_scale if ratio_scale>0 else 1e20
    res=minimize_scalar(chi2_kappa,bounds=(-kappa_max*2,kappa_max*2),method='bounded')
    kappa_best=res.x; chi2_best=res.fun
    chi2_null=np.sum(log_excess**2)
    dAIC=chi2_best-chi2_null+2
    R2=1-chi2_best/chi2_null
    print(f"kappa_best={kappa_best:.6e}")
    print(f"kappa*median(ratio)={kappa_best*ratio_scale:.4f}")
    print(f"chi2(k=0)={chi2_null:.3f}, chi2(k_best)={chi2_best:.3f}")
    print(f"dAIC={dAIC:.1f}, R^2={R2:.4f}")

    gc_corr=gc_deep*(1+kappa_best*ratio)
    res_uncorr=log_excess
    res_corr=np.log10(np.maximum(gc_obs/gc_corr,1e-30))
    print(f"residual std: uncorr={np.std(res_uncorr):.4f}, corr={np.std(res_corr):.4f}")

    # Bootstrap
    print("\n--- Bootstrap (2000) ---")
    np.random.seed(42)
    kappa_boot=[]
    for _ in range(2000):
        idx=np.random.randint(0,N,N)
        le=log_excess[idx]; ra=ratio[idx]
        def chi2_b(k):
            return np.sum((le-np.log10(np.maximum(1+k*ra,1e-10)))**2)
        try:
            rb=minimize_scalar(chi2_b,bounds=(-kappa_max*2,kappa_max*2),method='bounded')
            kappa_boot.append(rb.x)
        except: pass
    kappa_boot=np.array(kappa_boot)
    ci_lo,ci_hi=np.percentile(kappa_boot,[2.5,97.5])
    print(f"95% CI=[{ci_lo:.4e}, {ci_hi:.4e}]")
    print(f"CI includes 0? {'YES' if ci_lo<=0<=ci_hi else 'NO'}")

    # (2) Per-galaxy kappa
    print("\n"+"="*70); print("(2) kappa universality"); print("="*70)
    mask_r=ratio>0
    kappa_i=np.full(N,np.nan)
    kappa_i[mask_r]=(gc_obs[mask_r]/gc_deep[mask_r]-1)/ratio[mask_r]
    valid=np.isfinite(kappa_i)
    ki_v=kappa_i[valid]
    med_ki=np.median(ki_v); mad_ki=np.median(np.abs(ki_v-med_ki))
    clip=np.abs(ki_v-med_ki)<10*(mad_ki+1e-30)
    ki_c=ki_v[clip]
    print(f"kappa_i: N={len(ki_c)}, median={np.median(ki_c):.4e}, MAD={mad_ki:.4e}")
    print(f"  CV={mad_ki/(abs(med_ki)+1e-30):.2f}")

    idx_v=np.where(valid)[0][clip]
    print("\nkappa_i vs galaxy params (Spearman):")
    for pname,pv in [('log(vflat)',np.log10(vflat[idx_v])),
                      ('log(hR)',np.log10(hR[idx_v])),
                      ('log(Yd)',np.log10(np.maximum(Yd[idx_v],0.01))),
                      ('rmax/hR',rmax_hR[idx_v]),
                      ('log(gc_deep)',np.log10(gc_deep[idx_v]))]:
        finite=np.isfinite(pv)&np.isfinite(ki_c)
        if finite.sum()<10: continue
        rho,p=spearmanr(pv[finite],ki_c[finite])
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"  {pname:15s}: rho={rho:+.3f}, p={p:.4f} {sig}")

    # 2-param model
    print("\n--- Galaxy-dependent kappa models ---")
    for dn,dv in [('log_vflat',np.log10(vflat)),
                  ('log_hR',np.log10(hR)),
                  ('log_Yd',np.log10(np.maximum(Yd,0.01)))]:
        def chi2_2p(p):
            k0,k1=p; kv=k0+k1*dv
            m=np.log10(np.maximum(1+kv*ratio,1e-10))
            return np.sum((log_excess-m)**2)
        r2p=minimize(chi2_2p,[kappa_best,0],method='Nelder-Mead')
        dAIC_2p=r2p.fun-chi2_best+2
        print(f"  k=k0+k1*{dn}: k0={r2p.x[0]:.4e}, k1={r2p.x[1]:.4e}, dAIC={dAIC_2p:+.1f}")

    # (3) hR partial corr
    print("\n"+"="*70); print("(3) hR partial correlation"); print("="*70)
    log_gc_o=np.log10(gc_obs); log_gc_d=np.log10(gc_deep)
    log_gc_c=np.log10(np.maximum(gc_corr,1e-30))
    log_vf=np.log10(vflat); log_hR=np.log10(hR)

    def pcorr(x,y,z):
        cx=np.polyfit(z,x,1); cy=np.polyfit(z,y,1)
        return spearmanr(x-np.polyval(cx,z),y-np.polyval(cy,z))

    rho_o,p_o=pcorr(log_gc_o,log_hR,log_vf)
    rho_d,p_d=pcorr(log_gc_d,log_hR,log_vf)
    rho_c,p_c=pcorr(log_gc_c,log_hR,log_vf)
    print(f"rho(gc_obs,hR|vf)       = {rho_o:+.3f} (p={p_o:.4f})")
    print(f"rho(gc_deep,hR|vf)      = {rho_d:+.3f} (p={p_d:.4f})")
    print(f"rho(gc_corrected,hR|vf) = {rho_c:+.3f} (p={p_c:.4f})")
    if abs(rho_o)>0:
        red=(1-abs(rho_c)/abs(rho_o))*100
        print(f"Improvement: |rho| {abs(rho_o):.3f} -> {abs(rho_c):.3f} ({red:.1f}% reduction)")

    # (4) Circular check
    print("\n"+"="*70); print("(4) Circular reference check"); print("="*70)
    rho_rh,p_rh=spearmanr(ratio,hR)
    print(f"rho(ratio,hR)={rho_rh:+.3f}, p={p_rh:.4f}")
    rho_rg,p_rg=spearmanr(ratio,gc_obs)
    print(f"rho(ratio,gc_obs)={rho_rg:+.3f}, p={p_rg:.4f}")
    print("(E_grad/E_local doesn't directly contain hR - not tautology)")

    # Shuffle test
    np.random.seed(42)
    rho_sh=[]
    for _ in range(1000):
        rs=np.random.permutation(ratio)
        gc_s=gc_deep*(1+kappa_best*rs)
        ls=np.log10(np.maximum(gc_s,1e-30))
        rhs,_=pcorr(ls,log_hR,log_vf)
        rho_sh.append(rhs)
    rho_sh=np.array(rho_sh)
    pct=np.mean(rho_sh<=rho_c)*100
    print(f"\nShuffle test: real rho={rho_c:+.3f}, shuffle mean={np.mean(rho_sh):+.3f}, pct={pct:.1f}%")

    # Summary
    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"\nkappa_best = {kappa_best:.4e}")
    print(f"95% CI = [{ci_lo:.4e}, {ci_hi:.4e}]")
    print(f"CI includes 0: {'YES' if ci_lo<=0<=ci_hi else 'NO'}")
    print(f"dAIC vs k=0: {dAIC:+.1f}")
    print(f"R^2 improvement: {R2:.4f}")
    if abs(rho_o)>0:
        print(f"hR partial corr: |rho| {abs(rho_o):.3f} -> {abs(rho_c):.3f}")

    print(f"\nkappa * median(ratio) = {kappa_best*ratio_scale:.4f}")
    print(f"-> Typical galaxy: {abs(kappa_best*ratio_scale)*100:.1f}% gc correction")

    verdict="unclear"
    if ci_lo<=0<=ci_hi:
        verdict="CI includes 0 - kappa not significantly non-zero"
    elif kappa_best>0:
        verdict="kappa>0 significant - gradient term stiffens membrane"
    else:
        verdict="kappa<0 significant - gradient softens membrane"
    print(f"\nVERDICT: {verdict}")

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,10))

        ax=axes[0,0]
        ax.scatter(np.log10(gc_obs),np.log10(gc_corr),s=10,alpha=0.5,c='steelblue')
        lim=[np.log10(gc_obs).min()-0.1,np.log10(gc_obs).max()+0.1]
        ax.plot(lim,lim,'k--'); ax.set_xlabel('log(gc_obs)'); ax.set_ylabel('log(gc_corr)')
        ax.set_title(f'kappa={kappa_best:.2e}')

        ax=axes[0,1]
        ax.hist(ki_c,bins=40,alpha=0.7,color='steelblue',edgecolor='white')
        ax.axvline(kappa_best,color='red',lw=2,label=f'best={kappa_best:.2e}')
        ax.axvline(0,color='grey',ls='--')
        ax.set_xlabel('kappa_i'); ax.set_title('Per-galaxy kappa distribution'); ax.legend()

        ax=axes[1,0]
        ax.scatter(ratio,log_excess,s=10,alpha=0.5,c='steelblue')
        rs=np.sort(ratio)
        ax.plot(rs,np.log10(np.maximum(1+kappa_best*rs,1e-10)),'r-',lw=2)
        ax.set_xlabel('E_grad/E_local'); ax.set_ylabel('log(gc_obs/gc_deep)')
        ax.set_title('ratio vs excess')

        ax=axes[1,1]
        ax.hist(kappa_boot,bins=50,alpha=0.7,color='steelblue',edgecolor='white')
        ax.axvline(kappa_best,color='red',lw=2,label=f'best')
        ax.axvline(ci_lo,color='orange',ls='--',label='95% CI')
        ax.axvline(ci_hi,color='orange',ls='--')
        ax.axvline(0,color='grey',ls='--')
        ax.set_xlabel('kappa (bootstrap)'); ax.set_title('Bootstrap'); ax.legend()

        plt.tight_layout()
        fp=os.path.join(BASE,'kappa_estimation_results.png')
        plt.savefig(fp,dpi=150)
        print(f"\nFigure: {fp}")
    except Exception as e: print(f"Plot: {e}")

    print("\n[DONE]")

if __name__=='__main__': main()
