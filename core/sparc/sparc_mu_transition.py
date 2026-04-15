#!/usr/bin/env python3
"""
sparc_mu_transition.py (TA3+phase1+MRT adapted)
mu(x) correction and alpha(x) transition function.
"""
import os, csv, warnings, glob
import numpy as np
from scipy import stats, optimize
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.0857e19

def load_pipeline():
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try: data[n] = {'vflat': float(row.get('vflat', '0')), 'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                g = float(row.get('gc_over_a0', '0'))
                if n in data and g > 0: data[n]['gc'] = g * a0
            except: pass
    return {k: v for k, v in data.items() if 'gc' in v and v['vflat'] > 0}

def parse_mrt():
    data = {}; in_data = False; sep = 0
    with open(MRT, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4: in_data = True
                continue
            if not in_data: continue
            p = line.split()
            if len(p) < 18: continue
            try: data[p[0]] = {'Rdisk': float(p[11])}
            except: continue
    return data

def compute_hR_pipe(gname, Yd):
    fp = os.path.join(ROTMOD, f"{gname}_rotmod.dat")
    if not os.path.exists(fp): return None
    r, vdisk = [], []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            pp = line.split()
            if len(pp) >= 6:
                try: r.append(float(pp[0])); vdisk.append(float(pp[4]))
                except: continue
    if len(r) < 5: return None
    r = np.array(r); vdisk = np.array(vdisk)
    vds = np.sqrt(max(Yd, 0.01)) * np.abs(vdisk)
    rpk = r[np.argmax(vds)]
    if rpk < 0.01 or rpk >= r.max() * 0.9: return None
    return rpk / 2.15

def main():
    print("=" * 70)
    print("mu(x) transition: alpha(x) formulation")
    print("=" * 70)

    pipe = load_pipeline()
    mrt_d = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt_d)}")

    names, gc_a, vf_a, hR_a, Yd_a = [], [], [], [], []
    for n in sorted(pipe.keys()):
        gd = pipe[n]; gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vf <= 0: continue
        m = mrt_d.get(n, {})
        hR = m.get('Rdisk', 0)
        if hR <= 0: continue
        names.append(n); gc_a.append(gc); vf_a.append(vf); hR_a.append(hR); Yd_a.append(Yd)

    gc = np.array(gc_a); vf = np.array(vf_a); hR = np.array(hR_a); Yd = np.array(Yd_a)
    N = len(gc); log_gc = np.log10(gc); log_vf = np.log10(vf)
    Sd = (vf*1e3)**2 / (hR*kpc_m); log_Sd = np.log10(Sd)
    x_gc = gc / a0; log_x = np.log10(x_gc)
    log_a0Sd = np.log10(a0 * Sd)
    print(f"N = {N}")

    # Baseline
    sl0, ic0, r0, _, se0 = linregress(log_Sd, log_gc)
    pred0 = ic0 + sl0*log_Sd; resid0 = log_gc - pred0
    scat0 = np.std(resid0); R2_0 = r0**2
    SS_tot = np.sum((log_gc - log_gc.mean())**2)
    print(f"\nBaseline: alpha={sl0:.3f}+/-{se0:.3f}, R2={R2_0:.3f}, scatter={scat0:.3f}")

    # T1: binned alpha(x)
    print("\n" + "=" * 70)
    print("T1: Binned alpha(x)")
    print("=" * 70)
    n_bins = 8; sort_idx = np.argsort(log_x); bin_size = N // n_bins
    bx, ba, bae = [], [], []
    print(f"\n  {'bin':>3s} {'<x>':>8s} {'N':>4s} {'alpha':>10s} {'R2':>7s} {'sc':>7s}")
    for b in range(n_bins):
        i0 = b*bin_size; i1 = (b+1)*bin_size if b < n_bins-1 else N
        idx = sort_idx[i0:i1]
        xm = np.median(x_gc[idx])
        s, i_, r_, _, se_ = linregress(log_Sd[idx], log_gc[idx])
        res_ = log_gc[idx] - (i_ + s*log_Sd[idx])
        bx.append(np.log10(xm)); ba.append(s); bae.append(se_)
        print(f"  {b+1:3d} {xm:8.2f} {len(idx):4d} {s:7.3f}+/-{se_:.3f} {r_**2:7.3f} {np.std(res_):7.3f}")
    bx = np.array(bx); ba = np.array(ba); bae = np.array(bae)
    rho_ax, _ = spearmanr(bx, ba)
    print(f"  Trend: rho(log x, alpha)={rho_ax:+.3f}")

    # Local alpha (sliding window)
    window = 15
    alpha_loc = np.full(N, np.nan)
    for i in range(N):
        dist = np.abs(log_x - log_x[i])
        near = np.argsort(dist)[:2*window+1]
        if len(near) < 10: continue
        s, _, _, _, se = linregress(log_Sd[near], log_gc[near])
        alpha_loc[i] = s
    valid = ~np.isnan(alpha_loc)

    # T2: transition function fits
    print("\n" + "=" * 70)
    print("T2: alpha(x) transition functions")
    print("=" * 70)
    xv = log_x[valid]; av = alpha_loc[valid]

    def mA(lx, ad, xc, gam):
        return ad / (1 + (10**lx / xc)**gam)
    def mB(lx, ad, xc):
        return ad * xc / (10**lx + xc)
    def mC(lx, ad, xc, aN):
        return aN + (ad - aN) / (1 + 10**lx / xc)
    def mD(lx, ad, xc):
        return ad * (1 - np.tanh(np.log10(10**lx / xc))) / 2
    def mE(lx, ad, beta):
        return ad * (10**lx)**(-beta)

    models = {'A: ad/(1+(x/xc)^g)': (mA, [0.7, 1.0, 1.0]),
              'B: ad*xc/(x+xc)': (mB, [0.7, 1.0]),
              'C: aN+(ad-aN)/(1+x/xc)': (mC, [0.7, 1.0, 0.1]),
              'D: ad*(1-tanh)/2': (mD, [0.7, 1.0]),
              'E: ad*x^(-b)': (mE, [0.7, 0.3])}

    results = {}
    print(f"\n  {'model':<30s} {'params':>25s} {'R2':>7s} {'RMSE':>7s} {'AIC':>8s}")
    for name, (func, p0) in models.items():
        try:
            popt, _ = curve_fit(func, xv, av, p0=p0, maxfev=10000)
            ap = func(xv, *popt)
            ss = np.sum((av - ap)**2); st = np.sum((av - av.mean())**2)
            r2 = 1 - ss/st; rmse = np.sqrt(ss/len(av))
            aic = len(av)*np.log(ss/len(av)) + 2*len(popt)
            ps = ','.join([f'{v:.3f}' for v in popt])
            print(f"  {name:<30s} [{ps:>22s}] {r2:7.3f} {rmse:7.4f} {aic:8.1f}")
            results[name] = {'popt': popt, 'func': func, 'r2': r2, 'aic': aic}
        except Exception as e:
            print(f"  {name:<30s} FAILED: {e}")

    if not results:
        print("No models converged."); return
    best = min(results, key=lambda k: results[k]['aic'])
    print(f"\n  Best (AIC): {best}")
    bf = results[best]['func']; bp = results[best]['popt']

    # T3: modified geometric mean law
    print("\n" + "=" * 70)
    print("T3: Modified gc prediction with alpha(x)")
    print("=" * 70)

    # Method 1: direct (x=gc_obs, upper bound)
    alpha_per = bf(log_x, *bp)
    alpha_per = np.clip(alpha_per, 0.01, 2.0)
    log_eta_arr = log_gc - alpha_per * log_a0Sd
    log_eta_med = np.median(log_eta_arr)
    pred1 = log_eta_med + alpha_per * log_a0Sd
    resid1 = log_gc - pred1; scat1 = np.std(resid1)
    R2_1 = 1 - np.sum(resid1**2)/SS_tot
    print(f"  Direct (x=gc_obs): R2={R2_1:.4f}, scatter={scat1:.4f} (dR2={R2_1-R2_0:+.4f})")

    # Method 2: iterative
    gc_iter = 10**(ic0 + sl0*log_Sd)
    for it in range(20):
        xi = np.log10(np.clip(gc_iter/a0, 1e-5, None))
        ai = np.clip(bf(xi, *bp), 0.01, 2.0)
        gc_iter = 10**(log_eta_med + ai*log_a0Sd)
        delta = np.max(np.abs(np.log10(gc_iter) - (log_eta_med + ai*log_a0Sd)))
        if delta < 1e-6:
            print(f"  Iterative: converged at {it+1} iter")
            break
    pred2 = np.log10(gc_iter); resid2 = log_gc - pred2
    scat2 = np.std(resid2); R2_2 = 1 - np.sum(resid2**2)/SS_tot
    print(f"  Iterative: R2={R2_2:.4f}, scatter={scat2:.4f}")

    # Method 3: joint optimization
    def obj3(params):
        le = params[0]; mp = params[1:]
        a = np.clip(bf(log_x, *mp), 0.01, 2.0)
        return np.sum((log_gc - le - a*log_a0Sd)**2)
    x0_3 = np.concatenate([[log_eta_med], bp])
    res3 = minimize(obj3, x0_3, method='Nelder-Mead', options={'maxiter': 50000})
    le3 = res3.x[0]; mp3 = res3.x[1:]
    a3 = np.clip(bf(log_x, *mp3), 0.01, 2.0)
    pred3 = le3 + a3*log_a0Sd; resid3 = log_gc - pred3
    scat3 = np.std(resid3); R2_3 = 1 - np.sum(resid3**2)/SS_tot
    print(f"  Joint opt: R2={R2_3:.4f}, scatter={scat3:.4f} (dR2={R2_3-R2_0:+.4f})")
    print(f"    eta={10**le3:.4f}, params={mp3}")
    for xtest in [0.1, 0.5, 1.0, 5.0, 10.0]:
        print(f"    alpha(x={xtest})={bf(np.log10(xtest), *mp3):.3f}")

    best_resid = resid3; best_scat = scat3; best_R2 = R2_3

    # T4: residual improvement
    print("\n" + "=" * 70)
    print("T4: Residual improvement")
    print("=" * 70)
    rho_old, _ = spearmanr(log_gc, resid0)
    rho_new, _ = spearmanr(log_gc, best_resid)
    print(f"  gc correlation: {rho_old:+.3f} -> {rho_new:+.3f}")
    rvf_o, _ = spearmanr(log_vf, resid0); rvf_n, _ = spearmanr(log_vf, best_resid)
    print(f"  vflat corr: {rvf_o:+.3f} -> {rvf_n:+.3f}")

    # T5: per-bin scatter
    print("\n" + "=" * 70)
    print("T5: Per-bin scatter comparison")
    print("=" * 70)
    print(f"  {'bin':>3s} {'<x>':>8s} {'sc_base':>8s} {'sc_new':>8s} {'improv':>8s}")
    for b in range(n_bins):
        i0 = b*bin_size; i1 = (b+1)*bin_size if b < n_bins-1 else N
        idx = sort_idx[i0:i1]
        xm = np.median(x_gc[idx])
        so = np.std(resid0[idx]); sn = np.std(best_resid[idx])
        print(f"  {b+1:3d} {xm:8.2f} {so:8.4f} {sn:8.4f} {(so-sn)/so*100:+8.1f}%")

    # T6: alpha(x) + eta(Yd)
    print("\n" + "=" * 70)
    print("T6: alpha(x) + eta(Yd) combined")
    print("=" * 70)
    log_Yd = np.log10(Yd)
    def obj6(params):
        le0, bYd = params[0], params[1]; mp = params[2:]
        a = np.clip(bf(log_x, *mp), 0.01, 2.0)
        le = le0 + bYd*log_Yd
        return np.sum((log_gc - le - a*log_a0Sd)**2)
    x0_6 = np.concatenate([[le3, -0.3], mp3])
    res6 = minimize(obj6, x0_6, method='Nelder-Mead', options={'maxiter': 100000})
    p6 = res6.x; le6 = p6[0]; bYd6 = p6[1]; mp6 = p6[2:]
    a6 = np.clip(bf(log_x, *mp6), 0.01, 2.0)
    pred6 = (le6 + bYd6*log_Yd) + a6*log_a0Sd
    resid6 = log_gc - pred6; scat6 = np.std(resid6)
    R2_6 = 1 - np.sum(resid6**2)/SS_tot
    rho6, _ = spearmanr(log_gc, resid6)
    print(f"  eta0={10**le6:.4f}, beta_Yd={bYd6:.3f} (eta prop Yd^{bYd6:.2f})")
    print(f"  R2={R2_6:.4f}, scatter={scat6:.4f}")
    print(f"  gc-resid rho={rho6:+.3f}")
    print(f"  Improvement: {scat0:.3f} -> {scat6:.3f} ({(scat0-scat6)/scat0*100:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Baseline:      alpha=const={sl0:.3f}, R2={R2_0:.3f}, scatter={scat0:.3f}
  alpha(x):      R2={R2_3:.3f}, scatter={scat3:.3f} (dR2={R2_3-R2_0:+.3f})
  alpha(x)+Yd:   R2={R2_6:.3f}, scatter={scat6:.3f} (dR2={R2_6-R2_0:+.3f})

  gc-resid rho:  {rho_old:+.3f} -> {rho_new:+.3f} -> {rho6:+.3f}
  Best model: {best}
  eta(Yd): eta0={10**le6:.3f} * Yd^({bYd6:.2f})
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        ax.errorbar(bx, ba, yerr=bae, fmt='ko', capsize=3)
        xp = np.linspace(log_x.min(), log_x.max(), 200)
        ax.plot(xp, bf(xp, *mp3), 'r-', lw=2, label=best[:20])
        ax.axhline(0.5, color='blue', ls='--', alpha=0.5)
        ax.set_xlabel('log(gc/a0)'); ax.set_ylabel('alpha')
        ax.set_title('(a) alpha(x)'); ax.legend(fontsize=7)

        ax = axes[0, 1]
        if valid.any():
            sc_ = ax.scatter(log_x[valid], alpha_loc[valid], c=log_vf[valid], cmap='viridis', s=8, alpha=0.4)
            plt.colorbar(sc_, ax=ax, label='log vf')
        ax.set_xlabel('log(gc/a0)'); ax.set_ylabel('alpha_local')
        ax.set_title('(b) local alpha')

        ax = axes[0, 2]
        ax.scatter(log_gc, resid0, s=10, alpha=0.3, label=f'base s={scat0:.3f}')
        ax.scatter(log_gc, resid6, s=10, alpha=0.3, label=f'a(x)+Yd s={scat6:.3f}')
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel('log gc'); ax.set_ylabel('resid')
        ax.set_title('(c) residual comparison'); ax.legend(fontsize=7)

        ax = axes[1, 0]
        xph = np.logspace(-2, 2, 500)
        ax.plot(xph, bf(np.log10(xph), *mp3), 'r-', lw=2)
        ax.axvline(1, color='gray', ls='--', alpha=0.5)
        ax.axhline(0.5, color='blue', ls='--', alpha=0.5)
        ax.set_xscale('log'); ax.set_xlabel('x=gc/a0'); ax.set_ylabel('alpha(x)')
        ax.set_title('(d) transition function'); ax.set_ylim(0, 1)

        ax = axes[1, 1]
        ax.scatter(log_Yd, np.log10(gc/np.sqrt(a0*Sd)), s=10, alpha=0.5)
        sl_ey, ic_ey, _, _, _ = linregress(log_Yd, np.log10(gc/np.sqrt(a0*Sd)))
        xl = np.linspace(log_Yd.min(), log_Yd.max(), 50)
        ax.plot(xl, ic_ey+sl_ey*xl, 'r-', lw=2, label=f'eta~Yd^{sl_ey:.2f}')
        ax.set_xlabel('log Yd'); ax.set_ylabel('log eta')
        ax.set_title('(e) eta vs Yd'); ax.legend()

        ax = axes[1, 2]
        ax.scatter(pred6, log_gc, s=10, alpha=0.5, c='steelblue')
        lim = [log_gc.min()-0.1, log_gc.max()+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('predicted'); ax.set_ylabel('observed')
        ax.set_title(f'(f) final R2={R2_6:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'mu_transition.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
