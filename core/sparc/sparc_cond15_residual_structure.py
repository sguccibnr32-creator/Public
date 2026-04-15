#!/usr/bin/env python3
"""sparc_cond15_residual_structure.py (TA3+phase1+MRT+Rotmod adapted)"""
import os, csv, warnings
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD_DIR = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.086e19; Msun = 1.989e30; Lsun = 3.828e26
ETA0 = 0.584; BETA = -0.361

def load_all():
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
                mrt[p[0]] = {'T': int(p[1]), 'D': float(p[2]), 'Inc': float(p[5]),
                              'L': float(p[7]), 'Rdisk': float(p[11]), 'SBdisk0': float(p[12]),
                              'MHI': float(p[13]), 'Vf_mrt': float(p[15]),
                              'eVf': float(p[16]), 'Q': int(p[17])}
            except: continue
    return pipe, mrt

def main():
    print("=" * 70)
    print("C15 residual structure analysis")
    print("=" * 70)
    pipe, mrt = load_all()
    names = sorted([n for n in pipe if 'gc' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    results = []
    for gname in names:
        gd = pipe[gname]; gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        m = mrt[gname]; hR = m['Rdisk']
        if gc <= 0 or vf <= 0 or hR <= 0: continue
        gc_pred = ETA0 * Yd**BETA * np.sqrt(a0 * (vf*1e3)**2 / (hR*kpc_m))
        if gc_pred <= 0: continue
        delta = np.log10(gc / gc_pred)
        # Rotmod I0
        I0 = np.nan
        fp = os.path.join(ROTMOD_DIR, f"{gname}_rotmod.dat")
        if os.path.exists(fp):
            try:
                d = np.loadtxt(fp, comments='#')
                if d.ndim == 2 and d.shape[1] >= 7:
                    sb = d[:, 6]; r_ = d[:, 0]
                    inner = (r_ > 0.1) & (sb > 0)
                    if inner.sum() > 0:
                        I0_est = sb[inner][:5] * np.exp(r_[inner][:5] / hR)
                        I0 = np.median(I0_est)
            except: pass
        L = m.get('L', np.nan); MHI = m.get('MHI', 0)
        Mstar = Yd * L * 1e9 * Msun if L > 0 else np.nan
        Mgas = 1.33 * MHI * 1e9 * Msun
        Mbar = (Mstar + Mgas) if np.isfinite(Mstar) else np.nan
        f_gas = Mgas / Mbar if np.isfinite(Mbar) and Mbar > 0 else np.nan
        results.append({'name': gname, 'gc': gc, 'gc_pred': gc_pred, 'delta': delta,
                        'vflat': vf, 'hR': hR, 'Yd': Yd, 'T': m.get('T', np.nan),
                        'D': m.get('D', np.nan), 'Inc': m.get('Inc', np.nan),
                        'Q': m.get('Q', np.nan), 'eVf': m.get('eVf', np.nan),
                        'I0': I0, 'SBdisk0': m.get('SBdisk0', np.nan),
                        'f_gas': f_gas, 'Mbar': Mbar/Msun if np.isfinite(Mbar) else np.nan,
                        'Sigma_dyn': vf**2/hR})
    print(f"Valid: {len(results)}")
    if len(results) < 20: print("Too few."); return

    delta = np.array([r['delta'] for r in results])

    # T1
    print("\n" + "=" * 60)
    print("T1: Residual statistics")
    print("=" * 60)
    print(f"  mean={np.mean(delta):+.4f}, std={np.std(delta):.4f}, median={np.median(delta):+.4f}")
    w, pw = stats.shapiro(delta[:min(50, len(delta))])
    print(f"  Shapiro: W={w:.4f}, p={pw:.4f}")
    lgc = np.log10(np.array([r['gc'] for r in results]))
    pcts = np.percentile(lgc, [20, 40, 60, 80])
    print(f"\n  gc quintile bias:")
    for i, (lo, hi, lbl) in enumerate(zip([-99]+list(pcts), list(pcts)+[99],
                                            ['Q1','Q2','Q3','Q4','Q5'])):
        m = (lgc >= lo) & (lgc < hi)
        if m.sum() > 0: print(f"    {lbl}: bias={np.mean(delta[m]):+.4f} N={m.sum()}")

    # T2: I0
    print("\n" + "=" * 60)
    print("T2: Delta vs I0 (central SB)")
    print("=" * 60)
    I0 = np.array([r['I0'] for r in results])
    mi = np.isfinite(I0) & (I0 > 0)
    if mi.sum() > 10:
        sl, _, rv, pv, se = stats.linregress(np.log10(I0[mi]), delta[mi])
        print(f"  N={mi.sum()}, slope={sl:.4f}+/-{se:.4f}, R2={rv**2:.3f}, p={pv:.2e}")
        if pv < 0.05:
            dc = delta[mi] - (sl*np.log10(I0[mi]) + np.mean(delta[mi]) - sl*np.mean(np.log10(I0[mi])))
            print(f"  scatter: {np.std(delta[mi]):.4f} -> {np.std(dc):.4f}")
    else:
        print(f"  I0 data insufficient ({mi.sum()})")
        SBd = np.array([r['SBdisk0'] for r in results])
        ms = np.isfinite(SBd) & (SBd > 0)
        if ms.sum() > 10:
            sl, _, rv, pv, se = stats.linregress(np.log10(SBd[ms]), delta[ms])
            print(f"  SBdisk0 fallback: N={ms.sum()}, slope={sl:.4f}, R2={rv**2:.3f}, p={pv:.2e}")

    # T3: f_gas
    print("\n" + "=" * 60)
    print("T3: Delta vs f_gas")
    print("=" * 60)
    fg = np.array([r['f_gas'] for r in results])
    mf = np.isfinite(fg) & (fg >= 0) & (fg <= 1)
    if mf.sum() > 10:
        sl, _, rv, pv, se = stats.linregress(fg[mf], delta[mf])
        print(f"  N={mf.sum()}, slope={sl:.4f}+/-{se:.4f}, R2={rv**2:.3f}, p={pv:.2e}")
    else:
        print(f"  f_gas data insufficient ({mf.sum()})")

    # T4: Ttype
    print("\n" + "=" * 60)
    print("T4: Delta vs Ttype")
    print("=" * 60)
    T = np.array([r['T'] for r in results])
    mt = np.isfinite(T)
    if mt.sum() > 10:
        sl, _, rv, pv, se = stats.linregress(T[mt], delta[mt])
        print(f"  slope={sl:.4f}+/-{se:.4f}, R2={rv**2:.3f}, p={pv:.2e}")
        groups = {}
        for lbl, lo, hi in [("T<=3",-2,3),("T=4-7",4,7),("T>=8",8,12)]:
            m2 = mt & (T >= lo) & (T <= hi)
            if m2.sum() >= 3:
                groups[lbl] = delta[m2]
                print(f"    {lbl}: N={m2.sum()}, mean={np.mean(delta[m2]):+.4f}")
        if len(groups) >= 2:
            H, pk = stats.kruskal(*groups.values())
            print(f"  KW: H={H:.3f}, p={pk:.4f}")

    # T5: D, Inc, Q, eVf
    print("\n" + "=" * 60)
    print("T5: Delta vs D, Inc, Q, eVf")
    print("=" * 60)
    for vn, arr in [("log D", np.log10(np.maximum(np.array([r['D'] for r in results]), 0.01))),
                     ("Inc", np.array([r['Inc'] for r in results])),
                     ("Q", np.array([r['Q'] for r in results])),
                     ("eVf", np.array([r['eVf'] for r in results]))]:
        m = np.isfinite(arr)
        if m.sum() < 10: continue
        sl, _, rv, pv, _ = stats.linregress(arr[m], delta[m])
        sig = "*" if pv < 0.05 else ""
        print(f"  {vn:8s}: N={m.sum()}, slope={sl:+.4f}, R2={rv**2:.3f}, p={pv:.2e} {sig}")

    # T6: Multivariate
    print("\n" + "=" * 60)
    print("T6: Multivariate regression")
    print("=" * 60)
    cands = {}
    for vn, ext in [("T", lambda r: r['T']), ("f_gas", lambda r: r['f_gas']),
                      ("log_I0", lambda r: np.log10(r['I0']) if r['I0'] > 0 else np.nan),
                      ("log_D", lambda r: np.log10(r['D']) if r['D'] > 0 else np.nan),
                      ("Inc", lambda r: r['Inc']), ("Q", lambda r: float(r['Q'])),
                      ("log_Sd", lambda r: np.log10(r['Sigma_dyn']) if r['Sigma_dyn'] > 0 else np.nan)]:
        va = np.array([ext(r) for r in results])
        if np.isfinite(va).sum() > 30: cands[vn] = va

    mask_all = np.ones(len(results), dtype=bool)
    for va in cands.values(): mask_all &= np.isfinite(va)
    nv = mask_all.sum()
    if nv > 20:
        X = np.column_stack([cands[k][mask_all] for k in cands] + [np.ones(nv)])
        y = delta[mask_all]
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b; res = y - pred
        R2 = 1 - np.sum(res**2)/np.sum((y-y.mean())**2)
        print(f"  N={nv}, R2={R2:.4f}, scatter: {np.std(y):.4f} -> {np.std(res):.4f}")
        for i, k in enumerate(cands):
            print(f"    {k:10s}: {b[i]:+.4f}")

    # T7: Stepwise
    print("\n" + "=" * 60)
    print("T7: Stepwise selection")
    print("=" * 60)
    selected = []; remaining = list(cands.keys()); cur_sc = np.std(delta)
    print(f"  Step 0: scatter={cur_sc:.4f}")
    for step in range(min(5, len(remaining))):
        best_v, best_sc = None, cur_sc
        for k in remaining:
            ma = np.isfinite(cands[k])
            for sk in selected: ma &= np.isfinite(cands[sk])
            if ma.sum() < 20: continue
            Xs = np.column_stack([cands[s][ma] for s in selected] + [cands[k][ma], np.ones(ma.sum())])
            bs, _, _, _ = np.linalg.lstsq(Xs, delta[ma], rcond=None)
            sc = np.std(delta[ma] - Xs @ bs)
            if sc < best_sc: best_sc = sc; best_v = k
        if best_v is None: break
        red = (1 - best_sc/cur_sc)*100
        print(f"  Step {step+1}: +{best_v:10s} scatter={best_sc:.4f} ({red:+.1f}%)")
        selected.append(best_v); remaining.remove(best_v); cur_sc = best_sc
    print(f"  Selected: {selected}")

    # T8: LOO
    print("\n" + "=" * 60)
    print("T8: LOO cross-validation")
    print("=" * 60)
    if selected:
        ma = np.ones(len(results), dtype=bool)
        for k in selected: ma &= np.isfinite(cands[k])
        idx = np.where(ma)[0]; nv = len(idx)
        loo = np.zeros(nv)
        for i in range(nv):
            tr = np.delete(idx, i)
            Xtr = np.column_stack([cands[k][tr] for k in selected] + [np.ones(len(tr))])
            btr, _, _, _ = np.linalg.lstsq(Xtr, delta[tr], rcond=None)
            Xte = np.array([cands[k][idx[i]] for k in selected] + [1.0])
            loo[i] = delta[idx[i]] - Xte @ btr
        Xall = np.column_stack([cands[k][idx] for k in selected] + [np.ones(nv)])
        ball, _, _, _ = np.linalg.lstsq(Xall, delta[idx], rcond=None)
        sc_base = np.std(delta[idx]); sc_full = np.std(delta[idx] - Xall @ ball); sc_loo = np.std(loo)
        print(f"  baseline={sc_base:.4f}, full={sc_full:.4f} ({(1-sc_full/sc_base)*100:+.1f}%), LOO={sc_loo:.4f} ({(1-sc_loo/sc_base)*100:+.1f}%)")
        deg = (sc_loo - sc_full)/sc_full*100
        print(f"  LOO degradation: {deg:+.1f}% -> {'OK' if deg < 5 else 'overfitting risk'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  scatter = {np.std(delta):.4f} dex")
    if selected:
        print(f"  Best stepwise: {selected} -> scatter {cur_sc:.4f}")
    print(f"  Stepwise reduction: {(1-cur_sc/np.std(delta))*100:.1f}%")

    # Figure
    try:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        ax = axes[0, 0]
        ax.hist(delta, bins=25, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', ls='--')
        ax.set_xlabel('Delta'); ax.set_title(f'(1) scatter={np.std(delta):.3f}')

        ax = axes[0, 1]
        ax.scatter(lgc, delta, s=10, alpha=0.5); ax.axhline(0, color='red', ls='--')
        ax.set_xlabel('log gc'); ax.set_ylabel('Delta'); ax.set_title('(2) bias pattern')

        ax = axes[0, 2]
        if mi.sum() > 10:
            ax.scatter(np.log10(I0[mi]), delta[mi], s=10, alpha=0.5)
            ax.set_xlabel('log I0')
        ax.axhline(0, color='red', ls='--'); ax.set_ylabel('Delta'); ax.set_title('(3) vs I0')

        ax = axes[0, 3]
        if mf.sum() > 10:
            ax.scatter(fg[mf], delta[mf], s=10, alpha=0.5)
            ax.set_xlabel('f_gas')
        ax.axhline(0, color='red', ls='--'); ax.set_ylabel('Delta'); ax.set_title('(4) vs f_gas')

        ax = axes[1, 0]
        if mt.sum() > 10:
            bx = []
            for lo, hi in [(-2,3),(4,7),(8,12)]:
                m2 = mt & (T>=lo) & (T<=hi)
                if m2.sum() >= 3: bx.append(delta[m2])
            if bx: ax.boxplot(bx, labels=['T<=3','4-7','>=8'])
        ax.axhline(0, color='red', ls='--'); ax.set_title('(5) by type')

        ax = axes[1, 1]
        D = np.array([r['D'] for r in results])
        md = np.isfinite(D) & (D > 0)
        if md.sum() > 10: ax.scatter(np.log10(D[md]), delta[md], s=10, alpha=0.5)
        ax.axhline(0, color='red', ls='--'); ax.set_xlabel('log D'); ax.set_title('(6) vs dist')

        ax = axes[1, 2]
        Inc = np.array([r['Inc'] for r in results])
        mi2 = np.isfinite(Inc)
        if mi2.sum() > 10: ax.scatter(Inc[mi2], delta[mi2], s=10, alpha=0.5)
        ax.axhline(0, color='red', ls='--'); ax.set_xlabel('Inc'); ax.set_title('(7) vs inc')

        ax = axes[1, 3]
        gcp = np.array([r['gc_pred'] for r in results])
        gco = np.array([r['gc'] for r in results])
        ax.scatter(np.log10(gcp), np.log10(gco), s=10, alpha=0.5)
        lim = [min(np.log10(gcp).min(), np.log10(gco).min())-0.2,
               max(np.log10(gcp).max(), np.log10(gco).max())+0.2]
        ax.plot(lim, lim, 'r-')
        ax.set_xlabel('log gc_pred'); ax.set_ylabel('log gc_obs'); ax.set_title('(8) pred vs obs')

        plt.suptitle('C15 Residual Structure', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'cond15_residual_structure.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
