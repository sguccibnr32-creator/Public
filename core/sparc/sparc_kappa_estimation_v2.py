#!/usr/bin/env python3
"""sparc_kappa_estimation_v2.py (TA3+phase1+MRT adapted) - full version with figures"""
import os, csv, warnings
import numpy as np
from scipy import stats
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
ROTMOD = BASE / "Rotmod_LTG"
PHASE1 = BASE / "phase1" / "sparc_results.csv"
TA3 = BASE / "TA3_gc_independent.csv"
MRT = BASE / "SPARC_Lelli2016c.mrt"
OUT = BASE / "kappa_output"
OUT.mkdir(exist_ok=True)
a0 = 1.2e-10; kpc_m = 3.086e19
C14_A = 0.12; C14_B = 0.51; C14_TAU = 4.7

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
                if n in pipe and g > 0: pipe[n]['gc_a0'] = g
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
            try: mrt[p[0]] = {'Rdisk': float(p[11]), 'T': int(p[1])}
            except: continue
    return pipe, mrt

def U_pp(eps, c):
    eps = np.clip(eps, 1e-10, 1-1e-10)
    return -1 + c/(1-eps)**2

def main():
    print("=" * 70)
    print("R-6: Membrane rigidity kappa estimation (full)")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc_a0' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    results = []
    for gname in names:
        gd = pipe[gname]; gc_a0 = gd['gc_a0']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        hR = mrt[gname]['Rdisk']; T = mrt[gname].get('T', np.nan)
        fp = ROTMOD / f"{gname}_rotmod.dat"
        if not fp.exists(): continue
        try: d = np.loadtxt(fp, comments='#')
        except: continue
        if d.ndim != 2 or d.shape[1] < 6 or len(d) < 5: continue
        R = d[:, 0]; Vdisk = d[:, 4]; Vgas = d[:, 3]; Vbul = d[:, 5]
        Rm = R * kpc_m; valid = Rm > 0
        if valid.sum() < 4: continue
        Rm = Rm[valid]; R_ = R[valid]
        Vb2 = (Yd*np.sign(Vdisk[valid])*Vdisk[valid]**2 + np.sign(Vgas[valid])*Vgas[valid]**2
               + np.sign(Vbul[valid])*Vbul[valid]**2) * 1e6
        gN = np.maximum(np.abs(Vb2)/Rm, 0)
        gc_si = gc_a0*a0; eps = np.sqrt(np.maximum(gN/gc_si, 0))
        v2 = eps > 0
        if v2.sum() < 4: continue
        r = Rm[v2]; ep = eps[v2]
        E_local = np.trapezoid(ep**2*r, r)
        deps = np.gradient(ep, r)
        E_grad = np.trapezoid(deps**2*r, r)
        if E_local <= 0: continue
        ratio = E_grad/E_local
        eps_med = np.median(ep)
        if eps_med >= 0.95 or eps_med <= 0: continue
        Upp = U_pp(eps_med, gc_a0)
        if Upp <= 0: continue
        hR_m = hR*kpc_m
        kappa_over_Upp = ratio*hR_m**2
        kappa = kappa_over_Upp*Upp
        kappa_kpc2 = kappa/kpc_m**2
        l_kappa = np.sqrt(kappa_over_Upp)/kpc_m if kappa_over_Upp > 0 else 0
        # C14 prediction: B ~ 2*kappa/(U''*hR^2)
        B_pred = 2*kappa/(Upp*hR_m**2) if Upp > 0 else np.nan
        results.append({'galaxy': gname, 'gc_a0': gc_a0, 'hR': hR, 'vflat': vf, 'Yd': Yd, 'T': T,
                        'E_local': E_local, 'E_grad': E_grad, 'ratio': ratio,
                        'log_ratio': np.log10(ratio), 'kappa_kpc2': kappa_kpc2,
                        'l_kappa_kpc': l_kappa, 'Upp': Upp, 'eps_med': eps_med,
                        'B_pred': B_pred, 'n_pts': int(v2.sum())})

    print(f"Results: {len(results)} galaxies")
    if len(results) < 5: print("Too few."); return

    # Arrays
    kp = np.array([r['kappa_kpc2'] for r in results])
    hR_ = np.array([r['hR'] for r in results])
    vf_ = np.array([r['vflat'] for r in results])
    gc_ = np.array([r['gc_a0'] for r in results])
    T_ = np.array([r['T'] for r in results])
    lk = np.log10(kp[kp > 0])
    Bp = np.array([r['B_pred'] for r in results])
    lr = np.array([r['log_ratio'] for r in results])
    Upp_ = np.array([r['Upp'] for r in results])
    l_kappa = np.array([r['l_kappa_kpc'] for r in results])

    # T1-T2
    print("\n" + "=" * 60)
    print("T1-T2: kappa statistics")
    print("=" * 60)
    print(f"  log10(kappa/kpc^2): median={np.median(lk):.3f}, std={np.std(lk):.3f}")
    print(f"  l_kappa median={np.median(l_kappa):.4f} kpc")
    print(f"  U'' median={np.median(Upp_):.4f}")

    # T5: Type
    print("\n" + "=" * 60)
    print("T5: By galaxy type")
    print("=" * 60)
    for lbl, lo, hi in [("T<=3", -2, 3), ("T=4-7", 4, 7), ("T>=8", 8, 12)]:
        m = (T_ >= lo) & (T_ <= hi) & np.isfinite(T_)
        if m.sum() < 3: continue
        print(f"  {lbl}: N={m.sum()}, kappa_med={np.median(kp[m]):.3e}, l_kappa={np.median(l_kappa[m]):.4f}")
    # KW test
    groups = []
    for lo, hi in [(-2, 3), (4, 7), (8, 12)]:
        m = (T_ >= lo) & (T_ <= hi) & np.isfinite(T_) & (kp > 0)
        if m.sum() >= 3: groups.append(np.log10(kp[m]))
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        print(f"  KW test: H={H:.2f}, p={p:.4f}")

    # T6: Correlations
    print("\n" + "=" * 60)
    print("T6: kappa correlations")
    print("=" * 60)
    for vn, va in [("hR", np.log10(hR_)), ("vflat", np.log10(vf_)), ("gc", np.log10(gc_)),
                    ("Sigma_dyn", np.log10(vf_**2/hR_))]:
        m = np.isfinite(va) & (kp > 0)
        if m.sum() < 10: continue
        rho, p = stats.spearmanr(va[m], np.log10(kp[m]))
        print(f"  kappa vs {vn:10s}: rho={rho:+.3f} (p={p:.2e})")
    # kappa/hR^2 residual
    kn = kp/hR_**2
    for vn, va in [("vflat", np.log10(vf_)), ("gc", np.log10(gc_))]:
        m = np.isfinite(va) & (kn > 0)
        if m.sum() < 10: continue
        rho, p = stats.spearmanr(va[m], np.log10(kn[m]))
        print(f"  kappa/hR^2 vs {vn:10s}: rho={rho:+.3f} (p={p:.2e})")

    # T7: Universality
    print("\n" + "=" * 60)
    print("T7: Universality test")
    print("=" * 60)
    cv_raw = np.std(kp)/np.mean(kp)
    cv_norm = np.std(kn)/np.mean(kn)
    sl, _, rval, pval, se = stats.linregress(np.log10(hR_), np.log10(kp))
    print(f"  CV(kappa) = {cv_raw:.3f}")
    print(f"  CV(kappa/hR^2) = {cv_norm:.3f}")
    print(f"  kappa vs hR: slope={sl:.3f}+/-{se:.3f}, R2={rval**2:.3f}")
    t_n2 = (sl-2)/se; p_n2 = 2*stats.t.sf(abs(t_n2), len(kp)-2)
    print(f"  p(slope=2) = {p_n2:.4f}")
    print(f"  ratio vs hR: slope={(stats.linregress(np.log10(hR_), lr))[0]:.3f}")

    # T8: C14 prediction
    print("\n" + "=" * 60)
    print("T8: C14 parameter prediction")
    print("=" * 60)
    Bv = Bp[np.isfinite(Bp) & (Bp > 0) & (Bp < 10)]
    if len(Bv) > 5:
        print(f"  B_predicted: median={np.median(Bv):.4f} (obs={C14_B})")
        print(f"  B ratio pred/obs = {np.median(Bv)/C14_B:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  kappa: median = 10^{np.median(lk):.1f} kpc^2, scatter = {np.std(lk):.2f} dex
  kappa is {'universal' if np.std(lk)<0.5 else 'NOT universal'} (scatter {'<' if np.std(lk)<0.5 else '>'} 0.5 dex)
  kappa/hR^2: CV = {cv_norm:.3f}
  kappa vs hR slope = {sl:.2f} (n=2 {'not rejected' if p_n2>0.05 else 'rejected'})
  kappa vs gc: rho = {stats.spearmanr(np.log10(gc_), np.log10(kp))[0]:+.3f} (gc-independent)
  B_predicted = {np.median(Bv):.3f} vs B_obs = {C14_B} (ratio {np.median(Bv)/C14_B:.2f})
""")

    # Figure (8 panels)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    ax = axes[0, 0]
    ax.hist(lk, bins=25, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(lk), color='red', ls='--')
    ax.set_xlabel('log10(kappa/kpc^2)'); ax.set_title('(1) kappa dist')

    ax = axes[0, 1]
    ax.scatter(np.log10(hR_), np.log10(kp), s=15, alpha=0.5)
    xf = np.linspace(np.log10(hR_).min(), np.log10(hR_).max(), 50)
    ax.plot(xf, sl*xf+(np.mean(np.log10(kp))-sl*np.mean(np.log10(hR_))), 'r-', lw=2)
    ax.set_xlabel('log hR'); ax.set_ylabel('log kappa'); ax.set_title(f'(2) slope={sl:.2f}')

    ax = axes[0, 2]
    ax.hist(l_kappa[l_kappa > 0], bins=25, alpha=0.7, color='coral')
    ax.set_xlabel('l_kappa (kpc)'); ax.set_title('(3) rigidity length')

    ax = axes[0, 3]
    ax.scatter(np.log10(gc_), np.log10(kp), s=15, alpha=0.5, c='green')
    ax.set_xlabel('log gc/a0'); ax.set_ylabel('log kappa'); ax.set_title('(4) kappa vs gc')

    ax = axes[1, 0]
    ax.scatter(np.log10(vf_), np.log10(kp), s=15, alpha=0.5, c='purple')
    ax.set_xlabel('log vflat'); ax.set_ylabel('log kappa'); ax.set_title('(5) kappa vs vflat')

    ax = axes[1, 1]
    ax.scatter(np.log10(hR_), lr, s=15, alpha=0.5, c='teal')
    sl_r, _, _, _, se_r = stats.linregress(np.log10(hR_), lr)
    xf = np.linspace(np.log10(hR_).min(), np.log10(hR_).max(), 50)
    ax.plot(xf, sl_r*xf+(np.mean(lr)-sl_r*np.mean(np.log10(hR_))), 'r-', lw=2)
    ax.set_xlabel('log hR'); ax.set_ylabel('log(E_grad/E_local)'); ax.set_title(f'(6) ratio slope={sl_r:.2f}')

    ax = axes[1, 2]
    if len(Bv) > 5:
        ax.hist(Bv, bins=20, alpha=0.7)
        ax.axvline(C14_B, color='red', ls='--', label=f'obs B={C14_B}')
        ax.axvline(np.median(Bv), color='blue', ls=':', label=f'pred={np.median(Bv):.3f}')
        ax.set_xlabel('B predicted'); ax.set_title('(7) C14 B prediction'); ax.legend(fontsize=7)

    ax = axes[1, 3]
    gps = []
    for lo, hi in [(-2, 3), (4, 7), (8, 12)]:
        m = (T_ >= lo) & (T_ <= hi) & np.isfinite(T_) & (kp > 0)
        if m.sum() >= 3: gps.append(np.log10(kp[m]))
    if gps:
        bp = ax.boxplot(gps, labels=['T<=3', 'T=4-7', 'T>=8'], patch_artist=True)
        for patch, c in zip(bp['boxes'], ['#ff9999', '#99ff99', '#9999ff']): patch.set_facecolor(c)
    ax.set_ylabel('log kappa'); ax.set_title('(8) by type')

    plt.suptitle('Membrane Rigidity kappa from SPARC', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / 'kappa_full_results.png', dpi=150)
    print(f"Figure saved: {OUT/'kappa_full_results.png'}")

    # Save CSV
    import pandas as pd
    pd.DataFrame(results).to_csv(OUT / 'kappa_results.csv', index=False)
    print(f"CSV saved: {OUT/'kappa_results.csv'}")
    print("\nDone.")

if __name__ == '__main__':
    main()
