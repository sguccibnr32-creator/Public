#!/usr/bin/env python3
"""
P2b: Self-consistent critical deformation — gc prediction.
Modified to use TA3+phase1 instead of sparc_gc.csv.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr, linregress
from pathlib import Path
import csv, sys, glob, warnings
warnings.filterwarnings('ignore')

a0 = 1.2e-10
kpc_m = 3.0857e19

BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
ROTMOD_DIR = BASE / "Rotmod_LTG"
PHASE1 = BASE / "phase1" / "sparc_results.csv"
TA3 = BASE / "TA3_gc_independent.csv"

for p,l in [(ROTMOD_DIR,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not p.exists(): print(f'[ERROR] {l}: {p}'); sys.exit(1)


def load_pipeline():
    """Load TA3 + phase1 → {name: {gc_obs, vflat, Yd}}"""
    data = {}
    # phase1
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('galaxy','').strip()
            try:
                data[name] = {
                    'vflat': float(row.get('vflat','0')),
                    'Yd': float(row.get('ud','0.5')),
                }
            except: pass
    # TA3
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('galaxy','').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0','0'))
                if name in data and gc_a0 > 0:
                    data[name]['gc_obs'] = gc_a0 * a0
                    data[name]['gc_a0'] = gc_a0
            except: pass
    return {k:v for k,v in data.items() if 'gc_obs' in v and v['vflat']>0}


def load_rotmod(name):
    fpath = ROTMOD_DIR / f"{name}_rotmod.dat"
    if not fpath.exists(): return None
    r,vobs,errv,vgas,vdisk,vbul = [],[],[],[],[],[]
    with open(fpath,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                r.append(float(parts[0])); vobs.append(float(parts[1]))
                errv.append(float(parts[2])); vgas.append(float(parts[3]))
                vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
            except: continue
    if len(r)<5: return None
    return np.array(r),np.array(vobs),np.array(vgas),np.array(vdisk),np.array(vbul)


def compute_gN(r_kpc, Vgas, Vdisk, Vbul, Yd=0.5, Yb=0.7):
    conv = 1e6 / (r_kpc * kpc_m)  # (km/s)^2 / (kpc->m)
    gN = np.abs(Vgas)**2 * conv + Yd * np.abs(Vdisk)**2 * conv + Yb * np.abs(Vbul)**2 * conv
    return gN


def g_eff_over_gc(x):
    return (x + np.sqrt(x**2 + 4*x)) / 2


def critical_RHS(c):
    return 2.0 * (np.sqrt(c) - 1.0)


def solve_c(gN_profile, r_kpc):
    idx_max = np.argmax(gN_profile)
    gN_max = gN_profile[idx_max]
    r_max = r_kpc[idx_max]
    if gN_max <= 0: return np.nan, np.nan, np.nan, False

    def residual(c):
        gc = c * a0
        x_max = gN_max / gc
        return g_eff_over_gc(x_max) - critical_RHS(c)

    c_min, c_max = 1.001, 50.0
    try:
        r_lo = residual(c_min); r_hi = residual(c_max)
    except: return np.nan,np.nan,np.nan,False

    if r_lo * r_hi > 0:
        for c_try in [100,500,2000]:
            try:
                if residual(c_min)*residual(c_try)<0: c_max=c_try; break
            except: continue
        else: return np.nan,np.nan,np.nan,False

    try: c_sol = brentq(residual, c_min, c_max, xtol=1e-8)
    except: return np.nan,np.nan,np.nan,False

    gc_pred = c_sol * a0
    threshold = critical_RHS(c_sol)
    gc = gc_pred; x_arr = gN_profile/gc
    geff_gc = g_eff_over_gc(x_arr)
    r_c = r_max
    for i in range(len(r_kpc)-1,0,-1):
        if geff_gc[i] >= threshold: r_c=r_kpc[i]; break
    return c_sol, gc_pred, r_c, True


def main():
    print("="*70)
    print("P2b: Self-consistent Critical Deformation -- gc prediction")
    print("="*70)

    pipe = load_pipeline()
    print(f"Pipeline: {len(pipe)} galaxies")

    results = []
    skipped = 0

    for name, info in pipe.items():
        rot = load_rotmod(name)
        if rot is None: skipped+=1; continue
        r_kpc, Vobs, Vgas, Vdisk, Vbul = rot

        # hR from Vdisk peak (pipeline convention)
        Yd = info['Yd']
        vds = np.sqrt(max(Yd,0.01))*np.abs(Vdisk)
        rpk = r_kpc[np.argmax(vds)]
        if rpk<0.01 or rpk>=r_kpc.max()*0.9: skipped+=1; continue
        hR = rpk/2.15

        gN = compute_gN(r_kpc, Vgas, Vdisk, Vbul, Yd=Yd)
        c_sol, gc_pred, r_c, ok = solve_c(gN, r_kpc)
        if not ok: skipped+=1; continue

        vflat = info['vflat']
        GS0 = (vflat*1e3)**2 / (hR*kpc_m)

        results.append({
            'name':name, 'gc_obs':info['gc_obs'], 'gc_a0':info.get('gc_a0',0),
            'gc_pred':gc_pred, 'c_sol':c_sol, 'r_c':r_c,
            'hR':hR, 'vflat':vflat, 'Yd':Yd,
            'gN_max':np.max(gN), 'GS0':GS0,
            'r_c_over_hR':r_c/hR if hR>0 else np.nan,
        })

    N=len(results)
    print(f"Converged: {N}, Skipped: {skipped}")
    if N<10: print("Too few."); sys.exit(1)

    gc_obs=np.array([r['gc_obs'] for r in results])
    gc_pred=np.array([r['gc_pred'] for r in results])
    c_arr=np.array([r['c_sol'] for r in results])
    hR_arr=np.array([r['hR'] for r in results])
    vflat_arr=np.array([r['vflat'] for r in results])
    rc_hR=np.array([r['r_c_over_hR'] for r in results])
    GS0_arr=np.array([r['GS0'] for r in results])

    # Test 1
    print("\n"+"="*50); print("Test 1: gc_pred vs gc_obs"); print("="*50)
    log_obs=np.log10(gc_obs); log_pred=np.log10(gc_pred)
    m=np.isfinite(log_obs)&np.isfinite(log_pred)
    rp,pp=pearsonr(log_pred[m],log_obs[m])
    rs,ps=spearmanr(log_pred[m],log_obs[m])
    ratio=gc_pred[m]/gc_obs[m]
    ss_res=np.sum((log_obs[m]-log_pred[m])**2)
    ss_tot=np.sum((log_obs[m]-log_obs[m].mean())**2)
    R2=1-ss_res/ss_tot if ss_tot>0 else np.nan
    sl1,it1,_,_,se1=linregress(log_pred[m],log_obs[m])
    print(f"  N={m.sum()}")
    print(f"  Pearson rho={rp:.4f}, Spearman rho={rs:.4f}")
    print(f"  gc_pred/gc_obs: median={np.median(ratio):.3f}, std={np.std(ratio):.3f}")
    print(f"  log offset: median={np.median(log_pred[m]-log_obs[m]):.3f}")
    print(f"  R^2(direct 1:1)={R2:.4f}")
    print(f"  Fit: log(gc_obs)={sl1:.3f}*log(gc_pred)+{it1:.3f}")

    # Test 2
    print("\n"+"="*50); print("Test 2: gc_pred scaling"); print("="*50)
    log_GS0=np.log10(GS0_arr/a0)
    log_pred_all=np.log10(gc_pred)
    m2=np.isfinite(log_GS0)&np.isfinite(log_pred_all)
    sl2,it2,r2,_,se2=linregress(log_GS0[m2],log_pred_all[m2])
    print(f"  log(gc_pred) vs log(GS0/a0): slope={sl2:.4f}+/-{se2:.4f} (expect 0.5)")
    print(f"  R^2={r2**2:.4f}")
    from scipy.stats import t as tdist
    t2=abs(sl2-0.5)/se2; p2s=2*tdist.sf(t2,df=m2.sum()-2)
    print(f"  p(slope=0.5)={p2s:.4f}")

    # Test 3
    print("\n"+"="*50); print("Test 3: Residual structure"); print("="*50)
    delta=log_pred[m]-log_obs[m]
    for lab,arr in [('log(hR)',np.log10(hR_arr[m])),('log(vflat)',np.log10(vflat_arr[m])),
                    ('log(gN_max)',np.log10(np.array([r['gN_max'] for r in results])[m]))]:
        rr,pr=pearsonr(arr,delta)
        print(f"  Δ vs {lab}: rho={rr:.4f}, p={pr:.3e}")

    # Test 4
    print("\n"+"="*50); print("Test 4: r_c/hR distribution"); print("="*50)
    rcf=rc_hR[np.isfinite(rc_hR)]
    print(f"  median={np.median(rcf):.2f}, mean={np.mean(rcf):.2f}, std={np.std(rcf):.2f}")
    print(f"  frac r_c/hR<3: {np.mean(rcf<3):.1%}, <5: {np.mean(rcf<5):.1%}")

    # Test 5
    print("\n"+"="*50); print("Test 5: c distribution"); print("="*50)
    print(f"  median={np.median(c_arr):.3f}, mean={np.mean(c_arr):.3f}")
    print(f"  range=[{np.min(c_arr):.3f},{np.max(c_arr):.3f}]")

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,10))

        ax=axes[0,0]
        ax.scatter(log_obs[m],log_pred[m],s=8,alpha=0.5)
        lim=[min(log_obs[m].min(),log_pred[m].min())-0.1,max(log_obs[m].max(),log_pred[m].max())+0.1]
        ax.plot(lim,lim,'k--',lw=1,label='1:1')
        ax.set_xlabel('log(gc_obs)'); ax.set_ylabel('log(gc_pred)')
        ax.set_title(f'(a) gc_pred vs gc_obs R^2={R2:.3f}')
        ax.legend(); ax.set_aspect('equal')

        ax=axes[0,1]
        gc_geom=np.sqrt(a0*GS0_arr)
        log_geom=np.log10(gc_geom)
        m2p=np.isfinite(log_geom)&np.isfinite(log_pred_all)
        ax.scatter(log_geom[m2p],log_pred_all[m2p],s=8,alpha=0.5)
        ax.plot(lim,lim,'k--',lw=1)
        ax.set_xlabel('log(gc_geom)'); ax.set_ylabel('log(gc_pred)')
        ax.set_title('(b) gc_pred vs geometric mean')

        ax=axes[1,0]
        ax.scatter(np.log10(hR_arr[m]),delta,s=8,alpha=0.5)
        ax.axhline(0,color='k',ls='--',lw=1)
        ax.set_xlabel('log(hR)'); ax.set_ylabel('Δlog(gc)')
        ax.set_title(f'(c) Residual vs hR')

        ax=axes[1,1]
        ax.hist(rcf,bins=30,edgecolor='black',alpha=0.7)
        ax.axvline(np.median(rcf),color='red',ls='--',label=f'median={np.median(rcf):.1f}')
        ax.set_xlabel('r_c/hR'); ax.set_ylabel('Count')
        ax.set_title('(d) Critical radius'); ax.legend()

        plt.suptitle(f'P2b: Critical Deformation (N={N})',fontsize=14)
        plt.tight_layout()
        fig.savefig(BASE/'P2b_critical_results.png',dpi=150)
        print(f"\nFigure: P2b_critical_results.png")
    except Exception as e:
        print(f"Plot error: {e}")

    # Summary
    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"  gc_pred vs gc_obs: rho={rp:.4f}, R^2={R2:.4f}")
    print(f"  Scaling: alpha={sl2:.4f} (expect 0.5), p(0.5)={p2s:.4f}")
    print(f"  r_c/hR median={np.median(rcf):.2f}")
    if abs(sl2-0.5)<0.1: print("  → alpha≈0.5: geometric mean law reproduced ✓")
    else: print(f"  → alpha={sl2:.3f}: deviation from geometric mean")
    if R2>0.3: print(f"  → R^2={R2:.3f}: P2b predicts gc significantly ✓")
    else: print(f"  → R^2={R2:.3f}: weak prediction")

    print("\n[DONE]")

if __name__=="__main__": main()
