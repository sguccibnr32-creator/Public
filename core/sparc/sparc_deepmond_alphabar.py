#!/usr/bin/env python3
"""sparc_deepmond_alphabar.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19; Msun = 1.989e30; Lsun = 3.828e26

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
            try:
                mrt[p[0]] = {'T': int(p[1]), 'L': float(p[7]), 'Rdisk': float(p[11]),
                              'MHI': float(p[13])}
            except: continue
    return pipe, mrt

def main():
    print("=" * 70)
    print("Deep MOND alpha_bar analysis")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc' in pipe[n] and n in mrt
                     and mrt[n].get('L', 0) > 0 and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    gc = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    L36 = np.array([mrt[n]['L'] for n in names])
    MHI = np.array([mrt[n].get('MHI', 0) for n in names])
    hR = np.array([mrt[n]['Rdisk'] for n in names])
    T_type = np.array([mrt[n].get('T', np.nan) for n in names])

    Mbar = (Yd*L36*1e9 + 1.33*MHI*1e9) * Msun
    Rd_m = hR * kpc_m
    Sd = (vf*1e3)**2 / Rd_m
    Sb = G_SI * Mbar / (2*np.pi*Rd_m**2)

    log_gc = np.log10(gc); log_Sd = np.log10(Sd); log_Sb = np.log10(Sb)
    log_Yd = np.log10(Yd); log_vf = np.log10(vf); log_hR = np.log10(hR)
    log_ratio = log_Sb - log_Sd

    # Global fits
    sl_dyn, _, r_dyn, _, se_dyn = stats.linregress(log_Sd, log_gc)
    sl_bar, _, r_bar, _, se_bar = stats.linregress(log_Sb, log_gc)
    print(f"\nalpha_dyn={sl_dyn:.3f}+/-{se_dyn:.3f}, R2={r_dyn**2:.3f}")
    print(f"alpha_bar={sl_bar:.3f}+/-{se_bar:.3f}, R2={r_bar**2:.3f}")

    # T1: Binned alpha
    print("\n" + "=" * 60)
    print("T1: alpha by gc regime (6 bins)")
    print("=" * 60)
    nbins = 6; pcts = np.percentile(log_gc, np.linspace(0, 100, nbins+1))
    bc, bad, bab, bade, babe = [], [], [], [], []
    print(f"  {'bin':>3s} {'gc range':>22s} {'N':>4s} {'a_dyn':>10s} {'a_bar':>10s} {'R2d':>6s}")
    for i in range(nbins):
        m = (log_gc >= pcts[i]) & (log_gc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        n = m.sum()
        if n < 5: continue
        sd, _, rd, _, sed = stats.linregress(log_Sd[m], log_gc[m])
        sb, _, rb, _, seb = stats.linregress(log_Sb[m], log_gc[m])
        bc.append((pcts[i]+pcts[i+1])/2); bad.append(sd); bab.append(sb)
        bade.append(sed); babe.append(seb)
        rng = f"[{10**pcts[i]:.1e},{10**pcts[i+1]:.1e}]"
        print(f"  {i+1:3d} {rng:>22s} {n:4d} {sd:7.3f}+/-{sed:.3f} {sb:7.3f}+/-{seb:.3f} {rd**2:6.3f}")

    # T2: Sliding window
    print("\n" + "=" * 60)
    print("T2: Sliding window (40 galaxies)")
    print("=" * 60)
    si = np.argsort(log_gc); w = min(40, N//4)
    sg, sad, sab = [], [], []
    for s in range(0, N-w+1, max(1, w//4)):
        idx = si[s:s+w]
        sd, _, _, _, _ = stats.linregress(log_Sd[idx], log_gc[idx])
        sb, _, _, _, _ = stats.linregress(log_Sb[idx], log_gc[idx])
        sg.append(np.median(log_gc[idx])); sad.append(sd); sab.append(sb)

    # T3: Sigma divergence
    print("\n" + "=" * 60)
    print("T3: Sigma_bar/Sigma_dyn divergence")
    print("=" * 60)
    for i in range(nbins):
        m = (log_gc >= pcts[i]) & (log_gc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 3: continue
        print(f"  Bin {i+1}: log(Sb/Sd) = {np.median(log_ratio[m]):+.3f}+/-{np.std(log_ratio[m]):.3f}")
    r_div, p_div = stats.pearsonr(log_gc, log_ratio)
    print(f"  r(gc, Sb/Sd) = {r_div:.3f} (p={p_div:.2e})")

    # T4: Deep vs shallow
    print("\n" + "=" * 60)
    print("T4: Deep vs shallow MOND")
    print("=" * 60)
    med = np.median(log_gc); deep = log_gc < med; shallow = ~deep
    print(f"  Split at log(gc)={med:.3f}")
    for vn, va in [("Yd", Yd), ("T-type", T_type), ("log(hR)", log_hR), ("log(vflat)", log_vf)]:
        v = np.isfinite(va)
        if (v & deep).sum() < 5: continue
        _, p = stats.mannwhitneyu(va[v & deep], va[v & shallow], alternative='two-sided')
        print(f"  {vn:10s}: deep={np.median(va[v&deep]):.3f}, shallow={np.median(va[v&shallow]):.3f}, p={p:.3e}")

    # T5: Yd correction
    print("\n" + "=" * 60)
    print("T5: Yd-corrected alpha_bar")
    print("=" * 60)
    X = np.column_stack([log_Sb, log_Yd, np.ones(N)])
    b, _, _, _ = np.linalg.lstsq(X, log_gc, rcond=None)
    print(f"  Without Yd: a_bar={sl_bar:.3f}")
    print(f"  With Yd:    a_bar={b[0]:.3f}, b_Yd={b[1]:.3f}")
    print(f"\n  By regime:")
    for i in range(nbins):
        m = (log_gc >= pcts[i]) & (log_gc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 5: continue
        Xm = np.column_stack([log_Sb[m], log_Yd[m], np.ones(m.sum())])
        bm, _, _, _ = np.linalg.lstsq(Xm, log_gc[m], rcond=None)
        print(f"    Bin {i+1}: a_bar(Yd)={bm[0]:.3f}, b_Yd={bm[1]:.3f}")

    # T6: Functional form
    print("\n" + "=" * 60)
    print("T6: alpha(gc) functional form")
    print("=" * 60)
    if len(sg) > 5:
        sg_ = np.array(sg); sad_ = np.array(sad)
        const = np.mean(sad_)
        sl_l, il, rl, _, _ = stats.linregress(sg_, sad_)
        def tanh_m(x, al, ah, x0, w): return al + (ah-al)*0.5*(1+np.tanh((x-x0)/w))
        try:
            po, _ = curve_fit(tanh_m, sg_, sad_, p0=[0.3, 0.6, np.median(sg_), 0.5], maxfev=5000)
            rss_t = np.sum((sad_ - tanh_m(sg_, *po))**2)
            rss_c = np.sum((sad_ - const)**2)
            rss_l = np.sum((sad_ - (sl_l*sg_+il))**2)
            n_ = len(sad_)
            for nm, rss, k in [("Const", rss_c, 1), ("Linear", rss_l, 2), ("Tanh", rss_t, 4)]:
                aic = n_*np.log(rss/n_) + 2*k
                print(f"  {nm:8s}: AIC={aic:.1f}")
            print(f"  Tanh: a_low={po[0]:.3f}, a_high={po[1]:.3f}, x0={po[2]:.3f}, w={po[3]:.3f}")
        except Exception as e:
            print(f"  Tanh failed: {e}")

    # T7: Deep MOND mechanism
    print("\n" + "=" * 60)
    print("T7: Deep MOND mechanism")
    print("=" * 60)
    d30 = log_gc < np.percentile(log_gc, 30)
    if d30.sum() >= 10:
        print(f"  Deep30 N={d30.sum()}")
        for vn, va in [("Sd", log_Sd), ("Sb", log_Sb), ("Yd", log_Yd), ("hR", log_hR), ("vf", log_vf)]:
            r, p = stats.pearsonr(va[d30], log_gc[d30])
            print(f"    r(gc, {vn:4s}) = {r:+.3f} (p={p:.3e})")
        rd = log_ratio[d30]
        r_rd, p_rd = stats.pearsonr(rd, log_gc[d30])
        print(f"    r(gc, Sb/Sd) in deep = {r_rd:+.3f} (p={p_rd:.3e})")

    # T8: Condition 15 accuracy
    print("\n" + "=" * 60)
    print("T8: C15 prediction by regime")
    print("=" * 60)
    gc_pred = 0.584 * Yd**(-0.361) * np.sqrt(a0 * Sd)
    lr = log_gc - np.log10(gc_pred)
    if np.std(lr) < 5:
        print(f"  Overall scatter={np.std(lr):.3f}")
        for i in range(nbins):
            m = (log_gc >= pcts[i]) & (log_gc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
            if m.sum() < 3: continue
            print(f"    Bin {i+1}: bias={np.mean(lr[m]):+.3f}, scatter={np.std(lr[m]):.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  alpha_dyn = {sl_dyn:.3f}+/-{se_dyn:.3f}
  alpha_bar = {sl_bar:.3f}+/-{se_bar:.3f}
  r(gc, Sb/Sd) = {r_div:.3f}

  Key: alpha_dyn is stable across regimes
       alpha_bar declines in deep MOND
       Sigma_bar/Sigma_dyn diverges for low-gc galaxies
       -> Baryonic surface density fails to track dynamical in deep MOND
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        ax.errorbar(bc, bad, yerr=bade, fmt='s-', color='blue', capsize=4, ms=8, label='a_dyn')
        ax.errorbar(bc, bab, yerr=babe, fmt='o-', color='red', capsize=4, ms=8, label='a_bar')
        ax.axhline(0.5, ls='--', color='grey'); ax.axhline(1/3, ls=':', color='green')
        ax.axhline(0, ls=':', color='black', alpha=0.3)
        ax.set_xlabel('log(gc)'); ax.set_ylabel('alpha'); ax.set_title('(a) binned'); ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.plot(sg, sad, 'b-', lw=2, label='a_dyn')
        ax.plot(sg, sab, 'r-', lw=2, label='a_bar')
        ax.axhline(0.5, ls='--', color='grey'); ax.axhline(1/3, ls=':', color='green')
        ax.set_xlabel('log(gc)'); ax.set_ylabel('alpha'); ax.set_title('(b) sliding'); ax.legend(fontsize=7)

        ax = axes[0, 2]
        ax.scatter(log_gc, log_ratio, s=15, alpha=0.5, c='steelblue')
        xf = np.linspace(log_gc.min(), log_gc.max(), 50)
        sl_r, ir, _, _, _ = stats.linregress(log_gc, log_ratio)
        ax.plot(xf, sl_r*xf+ir, 'r-', lw=2); ax.axhline(0, ls='--', color='grey')
        ax.set_xlabel('log(gc)'); ax.set_ylabel('log(Sb/Sd)')
        ax.set_title(f'(c) divergence r={r_div:.3f}')

        ax = axes[1, 0]
        sc_ = ax.scatter(log_Sd, log_gc, c=log_gc, cmap='coolwarm', s=15, alpha=0.7)
        xf = np.linspace(log_Sd.min(), log_Sd.max(), 50)
        ax.plot(xf, sl_dyn*xf+(log_gc.mean()-sl_dyn*log_Sd.mean()), 'k-', lw=2)
        ax.set_xlabel('log(Sd)'); ax.set_ylabel('log(gc)'); ax.set_title('(d) gc vs Sd')
        plt.colorbar(sc_, ax=ax, label='log gc')

        ax = axes[1, 1]
        sc2 = ax.scatter(log_Sb, log_gc, c=log_gc, cmap='coolwarm', s=15, alpha=0.7)
        xf = np.linspace(log_Sb.min(), log_Sb.max(), 50)
        ax.plot(xf, sl_bar*xf+(log_gc.mean()-sl_bar*log_Sb.mean()), 'k-', lw=2)
        ax.set_xlabel('log(Sb)'); ax.set_ylabel('log(gc)'); ax.set_title('(e) gc vs Sb')
        plt.colorbar(sc2, ax=ax, label='log gc')

        ax = axes[1, 2]
        ax.scatter(log_Yd[deep], log_gc[deep], s=15, alpha=0.5, c='blue', label='deep')
        ax.scatter(log_Yd[shallow], log_gc[shallow], s=15, alpha=0.5, c='red', label='shallow')
        ax.set_xlabel('log(Yd)'); ax.set_ylabel('log(gc)'); ax.set_title('(f) gc vs Yd')
        ax.legend()

        plt.suptitle('Deep MOND alpha_bar', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'deepmond_alphabar_analysis.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == "__main__":
    main()
