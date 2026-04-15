#!/usr/bin/env python3
"""sparc_btfr_withinbin.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy import stats
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
    print("Membrane BTFR Within-Bin Slope Analysis")
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
    Rdisk = np.array([mrt[n]['Rdisk'] for n in names])
    T_type = np.array([mrt[n].get('T', np.nan) for n in names])

    Mbar = (Yd*L36*1e9 + 1.33*MHI*1e9) * Msun
    hR = Rdisk

    log_vf = np.log10(vf); log_gc = np.log10(gc); log_mb = np.log10(Mbar/Msun)
    log_hr = np.log10(hR)
    y = 4*np.log10(vf*1e3)
    x_mem = np.log10(G_SI*Mbar*gc)
    x_std = np.log10(G_SI*Mbar*a0)

    # T1: Overall
    print("\n" + "=" * 60)
    print("T1: Overall slopes")
    print("=" * 60)
    for lbl, x in [("Membrane", x_mem), ("Standard", x_std)]:
        sl, _, r, _, se = stats.linregress(x, y)
        print(f"  {lbl}: slope={sl:.3f}+/-{se:.3f}, R2={r**2:.3f}")

    # T2: Bin width dependence
    print("\n" + "=" * 60)
    print("T2: Within-bin slope vs bin count")
    print("=" * 60)
    results_bw = []
    for nbins in [3, 4, 5, 6, 8, 10]:
        pcts = np.percentile(log_vf, np.linspace(0, 100, nbins+1))
        slopes, ns = [], []
        for i in range(nbins):
            m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
            if m.sum() < 5 or np.std(x_mem[m]) < 1e-10: continue
            sl, _, _, _, _ = stats.linregress(x_mem[m], y[m])
            slopes.append(sl); ns.append(m.sum())
        if slopes:
            w = np.array(ns, dtype=float)
            wm = np.average(slopes, weights=w)
            ws = np.sqrt(np.average((np.array(slopes)-wm)**2, weights=w))
            results_bw.append((nbins, wm, ws))
            print(f"  {nbins:2d} bins: within slope={wm:.3f}+/-{ws:.3f}")

    # T3: Detailed 5-bin
    print("\n" + "=" * 60)
    print("T3: 5-bin detail")
    print("=" * 60)
    nbins = 5; pcts = np.percentile(log_vf, np.linspace(0, 100, nbins+1))
    bin_slopes, bin_centers, bin_errors = [], [], []
    print(f"  {'bin':>3s} {'vf range':>16s} {'N':>4s} {'slope':>10s} {'R2':>7s}")
    for i in range(nbins):
        m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 5: continue
        sl, _, r, _, se = stats.linregress(x_mem[m], y[m])
        bin_slopes.append(sl); bin_centers.append((pcts[i]+pcts[i+1])/2); bin_errors.append(se)
        print(f"  {i+1:3d} [{10**pcts[i]:.0f},{10**pcts[i+1]:.0f}] {m.sum():4d} {sl:7.3f}+/-{se:.3f} {r**2:7.3f}")

    # T4: Separating gc vs Mbar
    print("\n" + "=" * 60)
    print("T4: gc vs Mbar within bins")
    print("=" * 60)
    for i in range(nbins):
        m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 5: continue
        X = np.column_stack([log_mb[m], log_gc[m], np.ones(m.sum())])
        b, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
        print(f"  Bin {i+1}: b_Mbar={b[0]:.3f}, b_gc={b[1]:.3f}")

    # T5: Theoretical prediction
    print("\n" + "=" * 60)
    print("T5: Var(logvf)/Var(x) prediction")
    print("=" * 60)
    print(f"  {'bin':>3s} {'ratio':>8s} {'4*ratio':>8s} {'observed':>8s}")
    for i in range(nbins):
        m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 5: continue
        vr = np.var(log_vf[m]); vx = np.var(x_mem[m])
        ratio = vr/vx if vx > 0 else 0
        sl, _, _, _, _ = stats.linregress(x_mem[m], y[m])
        print(f"  {i+1:3d} {ratio:8.4f} {4*ratio:8.3f} {sl:8.3f}")

    # T6: Galaxy type
    print("\n" + "=" * 60)
    print("T6: By galaxy type")
    print("=" * 60)
    for lbl, mask in [("Early T<=3", T_type <= 3), ("Late T=4-7", (T_type >= 4) & (T_type <= 7)),
                       ("Irr T>=8", T_type >= 8)]:
        mt = mask & np.isfinite(T_type)
        if mt.sum() < 10: continue
        sl_a, _, _, _, se_a = stats.linregress(x_mem[mt], y[mt])
        p3 = np.percentile(log_vf[mt], [0, 33, 67, 100])
        wbs = []
        for j in range(3):
            mb2 = mt & (log_vf >= p3[j]) & (log_vf <= p3[j+1] + 0.01)
            if mb2.sum() >= 5:
                s, _, _, _, _ = stats.linregress(x_mem[mb2], y[mb2])
                wbs.append(s)
        wm = np.mean(wbs) if wbs else np.nan
        print(f"  {lbl}: N={mt.sum()}, overall={sl_a:.3f}+/-{se_a:.3f}, within={wm:.3f}")

    # T7: Bootstrap
    print("\n" + "=" * 60)
    print("T7: Bootstrap (1000x)")
    print("=" * 60)
    np.random.seed(42)
    boot = []
    for _ in range(1000):
        idx = np.random.choice(N, N, replace=True)
        p5 = np.percentile(log_vf[idx], np.linspace(0, 100, 6))
        sl_list = []
        for i in range(5):
            m = (log_vf[idx] >= p5[i]) & (log_vf[idx] <= p5[i+1] + 0.001)
            if m.sum() >= 5 and np.std(x_mem[idx][m]) > 1e-10:
                s, _, _, _, _ = stats.linregress(x_mem[idx][m], y[idx][m])
                sl_list.append(s)
        if sl_list: boot.append(np.mean(sl_list))
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"  Mean={np.mean(boot):.3f}+/-{np.std(boot):.3f}, 95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")
    for v, l in [(0.5, "0.5"), (1.0, "1.0"), (0.0, "0.0")]:
        p = 2*min(np.mean(boot > v), np.mean(boot < v))
        print(f"  p(slope={l}): {p:.3f}")

    # T8: Correlation structure
    print("\n" + "=" * 60)
    print("T8: Within-bin correlations")
    print("=" * 60)
    print(f"  {'bin':>3s} {'r(y,Mb)':>8s} {'r(y,gc)':>8s} {'r(y,hR)':>8s}")
    for i in range(nbins):
        m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 5: continue
        rmb = np.corrcoef(y[m], log_mb[m])[0, 1]
        rgc = np.corrcoef(y[m], log_gc[m])[0, 1]
        rhr = np.corrcoef(y[m], log_hr[m])[0, 1]
        print(f"  {i+1:3d} {rmb:+8.3f} {rgc:+8.3f} {rhr:+8.3f}")

    # Summary
    sl_all, _, _, _, se_all = stats.linregress(x_mem, y)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Overall slope: {sl_all:.3f}+/-{se_all:.3f}
  Within-bin slope: {np.mean(boot):.3f}+/-{np.std(boot):.3f}
  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]
  Simpson gap: {sl_all - np.mean(boot):+.3f}

  p(within=0.5): {2*min(np.mean(boot>0.5), np.mean(boot<0.5)):.3f}
  p(within=1.0): {2*min(np.mean(boot>1.0), np.mean(boot<1.0)):.3f}
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        sc_ = ax.scatter(x_mem, y, c=log_vf, cmap='viridis', s=15, alpha=0.7)
        xf = np.linspace(x_mem.min(), x_mem.max(), 50)
        sl_a, ic_a, _, _, _ = stats.linregress(x_mem, y)
        ax.plot(xf, sl_a*xf+ic_a, 'r-', lw=2, label=f'overall {sl_a:.2f}')
        ax.plot(xf, xf+np.mean(y-x_mem), 'k--', lw=1, label='slope=1')
        cols = plt.cm.tab10(np.linspace(0, 0.5, nbins))
        for i in range(nbins):
            m = (log_vf >= pcts[i]) & (log_vf <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
            if m.sum() < 5: continue
            sl_b, ic_b, _, _, _ = stats.linregress(x_mem[m], y[m])
            xr = np.array([x_mem[m].min(), x_mem[m].max()])
            ax.plot(xr, sl_b*xr+ic_b, '-', color=cols[i], lw=1.5)
        ax.set_xlabel('log(GMbar*gc)'); ax.set_ylabel('log(vf^4)')
        ax.set_title("(a) Simpson's Paradox"); ax.legend(fontsize=7)
        plt.colorbar(sc_, ax=ax, label='log vf')

        ax = axes[0, 1]
        if results_bw:
            nbs = [r[0] for r in results_bw]
            sls = [r[1] for r in results_bw]
            ers = [r[2] for r in results_bw]
            ax.errorbar(nbs, sls, yerr=ers, fmt='ko-', capsize=4)
            ax.axhline(0.5, color='blue', ls='--', alpha=0.7, label='0.5')
            ax.axhline(1.0, color='red', ls='--', alpha=0.7, label='1.0')
        ax.set_xlabel('N bins'); ax.set_ylabel('within-bin slope')
        ax.set_title('(b) vs bin width'); ax.legend()

        ax = axes[1, 0]
        if bin_centers:
            ax.errorbar(bin_centers, bin_slopes, yerr=bin_errors, fmt='s-', color='darkblue', capsize=4, ms=8)
            ax.axhline(0.5, color='blue', ls='--'); ax.axhline(1.0, color='red', ls='--')
            ax.axhline(np.mean(bin_slopes), color='green', ls=':', label=f'mean={np.mean(bin_slopes):.3f}')
        ax.set_xlabel('log(vflat) center'); ax.set_ylabel('slope')
        ax.set_title('(c) per-bin slope'); ax.legend()

        ax = axes[1, 1]
        ax.hist(boot, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(boot), color='red', lw=2, label=f'mean={np.mean(boot):.3f}')
        ax.axvline(0.5, color='blue', ls='--'); ax.axvline(1.0, color='orange', ls='--')
        ax.axvline(ci[0], color='grey', ls=':'); ax.axvline(ci[1], color='grey', ls=':')
        ax.set_xlabel('within-bin slope'); ax.set_title('(d) bootstrap')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'btfr_withinbin_analysis.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == "__main__":
    main()
