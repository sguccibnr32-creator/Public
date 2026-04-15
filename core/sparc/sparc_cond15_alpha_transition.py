#!/usr/bin/env python3
"""sparc_cond15_alpha_transition.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, curve_fit, differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.0857e19
ETA0 = 0.584; BETA_YD = -0.361

def Sd_SI(vf, hR): return (vf*1e3)**2 / (hR*kpc_m)
def gc_c15(vf, hR, Yd, alpha=0.5): return ETA0 * Yd**BETA_YD * (a0*Sd_SI(vf, hR))**alpha
def alpha_tanh(lgc, al, ah, x0, w): return al + (ah-al)*0.5*(1+np.tanh((lgc-x0)/w))

def gc_iter(vf, hR, Yd, afunc, ap, maxit=20):
    gc = gc_c15(vf, hR, Yd, 0.5)
    for it in range(maxit):
        a = np.clip(afunc(np.log10(gc), *ap), 0.01, 0.99)
        gc_new = gc_c15(vf, hR, Yd, a)
        if np.max(np.abs(np.log10(gc_new)-np.log10(gc))) < 1e-6: return gc_new, a, it+1
        gc = gc_new
    return gc, a, maxit

def load_data():
    pipe = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try: pipe[n] = {'vflat': float(row.get('vflat', '0')), 'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                g = float(row.get('gc_over_a0', '0'))
                if n in pipe and g > 0: pipe[n]['gc'] = g * a0
            except: pass
    mrt = {}; in_data = False; sep = 0
    with open(MRT, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4: in_data = True
                continue
            if not in_data: continue
            p = line.split()
            if len(p) < 18: continue
            try: mrt[p[0]] = {'Rdisk': float(p[11])}
            except: continue
    return pipe, mrt

def main():
    print("=" * 70)
    print("C15 alpha(gc) transition test")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")
    gc_obs = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    hR = np.array([mrt[n]['Rdisk'] for n in names])
    lgc = np.log10(gc_obs); log_Sd = np.log10(Sd_SI(vf, hR))

    # Baseline
    gc_base = gc_c15(vf, hR, Yd, 0.5)
    rb = lgc - np.log10(gc_base)
    print(f"Baseline (a=0.5): bias={np.mean(rb):+.3f}, scatter={np.std(rb):.3f}")

    # T1: Sliding window alpha
    print("\n" + "=" * 60)
    print("T1: Sliding window alpha(gc)")
    print("=" * 60)
    si = np.argsort(lgc); w = min(40, N//4)
    sg, sa = [], []
    for s in range(0, N-w+1, max(1, w//5)):
        idx = si[s:s+w]
        sl, _, _, _, _ = stats.linregress(log_Sd[idx], lgc[idx])
        sg.append(np.median(lgc[idx])); sa.append(sl)
    sg = np.array(sg); sa = np.array(sa)
    print(f"  {len(sg)} windows, alpha range [{sa.min():.3f}, {sa.max():.3f}]")

    # Fit tanh
    try:
        p0 = [0.3, 0.05, np.median(sg), 0.3]
        popt, _ = curve_fit(alpha_tanh, sg, sa, p0=p0, maxfev=10000,
                             bounds=([0, -0.5, sg.min(), 0.01], [1, 1, sg.max(), 3]))
        print(f"  Tanh: al={popt[0]:.3f}, ah={popt[1]:.3f}, x0={popt[2]:.3f}, w={popt[3]:.3f}")
        # Diagnostic
        ad = alpha_tanh(lgc, *popt)
        gd = gc_c15(vf, hR, Yd, ad)
        rd = lgc - np.log10(gd)
        print(f"  Diagnostic: bias={np.mean(rd):+.3f}, scatter={np.std(rd):.3f} ({(1-np.std(rd)/np.std(rb))*100:.1f}%)")
        tanh_p = popt
    except Exception as e:
        print(f"  Tanh failed: {e}"); tanh_p = None

    # T2: Iterative
    print("\n" + "=" * 60)
    print("T2: Iterative convergence")
    print("=" * 60)
    if tanh_p is not None:
        gi, ai, ni = gc_iter(vf, hR, Yd, alpha_tanh, tanh_p)
        ri = lgc - np.log10(gi)
        print(f"  Converged in {ni} iter, alpha [{ai.min():.3f},{ai.max():.3f}]")
        print(f"  bias={np.mean(ri):+.3f}, scatter={np.std(ri):.3f}")

    # T3: Proxy mode
    print("\n" + "=" * 60)
    print("T3: Non-circular proxy alpha")
    print("=" * 60)
    alpha_eff = (lgc - np.log10(ETA0*Yd**BETA_YD)) / np.log10(a0*Sd_SI(vf, hR))
    av = np.isfinite(alpha_eff) & (alpha_eff > -1) & (alpha_eff < 2)
    for pn, pv in [("log_vflat", np.log10(vf)), ("log_Sd", log_Sd), ("log_Yd", np.log10(Yd))]:
        r_, p_ = stats.pearsonr(pv[av], alpha_eff[av])
        sl_, il_, _, _, _ = stats.linregress(pv[av], alpha_eff[av])
        ap_ = np.clip(sl_*pv + il_, 0.01, 0.99)
        gp = gc_c15(vf, hR, Yd, ap_)
        rp = lgc - np.log10(gp)
        slb, _, _, _, _ = stats.linregress(lgc, rp)
        print(f"  {pn:10s}: r={r_:+.3f}, scatter={np.std(rp):.3f}, bias_slope={slb:+.3f}")

    # T4: Optimize
    print("\n" + "=" * 60)
    print("T4: Parameter optimization")
    print("=" * 60)
    res_c = minimize_scalar(lambda a: np.std(lgc - np.log10(gc_c15(vf, hR, Yd, a))),
                             bounds=(0.01, 0.99), method='bounded')
    aopt = res_c.x
    gc_oc = gc_c15(vf, hR, Yd, aopt)
    roc = lgc - np.log10(gc_oc)
    print(f"  Optimal const alpha={aopt:.4f}, scatter={np.std(roc):.4f}")

    opt_tp = tanh_p
    if tanh_p is not None:
        def obj(p):
            try:
                g, _, _ = gc_iter(vf, hR, Yd, alpha_tanh, p, 30)
                return np.std(lgc - np.log10(g))
            except: return 1e10
        try:
            res_de = differential_evolution(obj, [(0, 1), (-0.3, 0.8),
                (lgc.min()-1, lgc.max()+1), (0.05, 2)], seed=42, maxiter=100)
            opt_tp = res_de.x
            go, ao, _ = gc_iter(vf, hR, Yd, alpha_tanh, opt_tp, 30)
            ro = lgc - np.log10(go)
            print(f"  Optimized tanh: al={opt_tp[0]:.3f}, ah={opt_tp[1]:.3f}, x0={opt_tp[2]:.3f}, w={opt_tp[3]:.3f}")
            print(f"  scatter={np.std(ro):.4f}, delta vs const={np.std(ro)-np.std(roc):+.4f}")
        except Exception as e:
            print(f"  DE failed: {e}")

    # T5: LOO
    print("\n" + "=" * 60)
    print("T5: LOO cross-validation")
    print("=" * 60)
    loo_f, loo_c, loo_t = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False
        loo_f[i] = lgc[i] - np.log10(gc_c15(vf[i], hR[i], Yd[i], 0.5))
        rt = minimize_scalar(lambda a: np.std(lgc[tr]-np.log10(gc_c15(vf[tr], hR[tr], Yd[tr], a))),
                              bounds=(0.01, 0.99), method='bounded')
        loo_c[i] = lgc[i] - np.log10(gc_c15(vf[i], hR[i], Yd[i], rt.x))
        if opt_tp is not None:
            gt, _, _ = gc_iter(np.array([vf[i]]), np.array([hR[i]]), np.array([Yd[i]]),
                                alpha_tanh, opt_tp, 30)
            loo_t[i] = lgc[i] - np.log10(gt[0])
    print(f"  a=0.5:    {np.std(loo_f):.4f}")
    print(f"  a_opt:    {np.std(loo_c):.4f}")
    if opt_tp is not None:
        print(f"  a(gc):    {np.std(loo_t):.4f}")

    # T6: AIC
    print("\n" + "=" * 60)
    print("T6: AIC comparison")
    print("=" * 60)
    mods = [("a=0.5", 1, np.std(rb)), ("a_opt", 2, np.std(roc))]
    if opt_tp is not None: mods.append(("a(gc)", 5, np.std(ro)))
    aic0 = None
    print(f"  {'model':<12s} {'k':>3s} {'sc':>8s} {'AIC':>8s} {'dAIC':>8s}")
    for nm, k, sc in mods:
        aic = N*np.log(N*sc**2/N) + 2*k
        if aic0 is None: aic0 = aic
        print(f"  {nm:<12s} {k:3d} {sc:8.4f} {aic:8.1f} {aic-aic0:+8.1f}")

    # T7: Binned bias
    print("\n" + "=" * 60)
    print("T7: Binned bias")
    print("=" * 60)
    pcts = np.percentile(lgc, np.linspace(0, 100, 6))
    slb0, _, _, _, _ = stats.linregress(lgc, rb)
    slbc, _, _, _, _ = stats.linregress(lgc, roc)
    print(f"  {'bin':>3s} {'a=0.5':>8s} {'a_opt':>8s}" + (f" {'a(gc)':>8s}" if opt_tp is not None else ""))
    for i in range(5):
        m = (lgc >= pcts[i]) & (lgc <= pcts[i+1] + (0.001 if i == 4 else 0))
        if m.sum() < 3: continue
        s = f"  {i+1:3d} {np.mean(rb[m]):+8.3f} {np.mean(roc[m]):+8.3f}"
        if opt_tp is not None: s += f" {np.mean(ro[m]):+8.3f}"
        print(s)
    print(f"\n  Bias slope: a=0.5={slb0:+.3f}, a_opt={slbc:+.3f}", end="")
    if opt_tp is not None:
        slbt, _, _, _, _ = stats.linregress(lgc, ro)
        print(f", a(gc)={slbt:+.3f} ({(1-abs(slbt)/abs(slb0))*100:.0f}% reduction)")
    else:
        print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  a=0.5: scatter={np.std(rb):.4f}, LOO={np.std(loo_f):.4f}")
    print(f"  a_opt={aopt:.3f}: scatter={np.std(roc):.4f}, LOO={np.std(loo_c):.4f}")
    if opt_tp is not None:
        print(f"  a(gc) tanh: scatter={np.std(ro):.4f}, LOO={np.std(loo_t):.4f}")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        ax.scatter(sg, sa, s=20, alpha=0.6, c='steelblue')
        gp_ = np.linspace(sg.min(), sg.max(), 200)
        if tanh_p is not None: ax.plot(gp_, alpha_tanh(gp_, *tanh_p), 'r-', lw=2, label='data fit')
        if opt_tp is not None: ax.plot(gp_, alpha_tanh(gp_, *opt_tp), 'g--', lw=2, label='optimized')
        ax.axhline(0.5, ls=':', color='grey'); ax.axhline(aopt, ls='--', color='orange')
        ax.set_xlabel('log gc'); ax.set_ylabel('alpha'); ax.set_title('(a) alpha(gc)'); ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.scatter(lgc, rb, s=12, alpha=0.4, c='steelblue', label=f'a=0.5 s={np.std(rb):.3f}')
        if opt_tp is not None:
            ax.scatter(lgc, ro, s=12, alpha=0.4, c='coral', label=f'a(gc) s={np.std(ro):.3f}')
        ax.axhline(0, ls='--', color='grey')
        ax.set_xlabel('log gc'); ax.set_ylabel('resid'); ax.set_title('(b) residuals'); ax.legend(fontsize=7)

        ax = axes[0, 2]
        if opt_tp is not None:
            ax.hist(ao, bins=30, color='coral', alpha=0.7)
        ax.axvline(0.5, ls='--', color='blue'); ax.axvline(aopt, ls='--', color='orange')
        ax.set_xlabel('alpha'); ax.set_title('(c) per-galaxy alpha')

        ax = axes[1, 0]
        ax.scatter(lgc, np.log10(gc_base), s=12, alpha=0.4, c='steelblue', label='a=0.5')
        if opt_tp is not None:
            ax.scatter(lgc, np.log10(go), s=12, alpha=0.4, c='coral', label='a(gc)')
        lim = [lgc.min()-0.1, lgc.max()+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('obs'); ax.set_ylabel('pred'); ax.set_title('(d) pred vs obs'); ax.legend(fontsize=7)

        ax = axes[1, 1]
        labs = ['a=0.5', 'a_opt']
        scs = [np.std(loo_f), np.std(loo_c)]
        cols = ['steelblue', 'orange']
        if opt_tp is not None: labs.append('a(gc)'); scs.append(np.std(loo_t)); cols.append('coral')
        ax.bar(labs, scs, color=cols, alpha=0.7)
        ax.set_ylabel('LOO scatter'); ax.set_title('(e) LOO')
        for i, s in enumerate(scs): ax.text(i, s+0.001, f'{s:.4f}', ha='center', fontsize=9)

        ax = axes[1, 2]
        if opt_tp is not None:
            aobs = alpha_tanh(lgc, *opt_tp)
            apred = alpha_tanh(np.log10(go), *opt_tp)
            ax.scatter(aobs, apred, s=12, alpha=0.5)
            lim2 = [min(aobs.min(), apred.min())-0.01, max(aobs.max(), apred.max())+0.01]
            ax.plot(lim2, lim2, 'k--')
            r_c, _ = stats.pearsonr(aobs, apred)
            ax.set_xlabel('alpha(gc_obs)'); ax.set_ylabel('alpha(gc_pred)')
            ax.set_title(f'(f) circularity r={r_c:.3f}')

        plt.suptitle('C15 alpha(gc) transition', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'cond15_alpha_transition.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == "__main__":
    main()
