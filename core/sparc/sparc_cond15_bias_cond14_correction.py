#!/usr/bin/env python3
"""sparc_cond15_bias_cond14_correction.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19

ETA0 = 0.584; BETA_YD = -0.361
C14_A = 0.12; C14_B = 0.51; C14_TAU = 4.7

def gc_c15(vf, hR, Yd):
    Sd = (vf*1e3)**2 / (hR*kpc_m)
    return ETA0 * Yd**BETA_YD * np.sqrt(a0 * Sd)

def f14(r_kpc, hR_kpc):
    return C14_A + C14_B * np.exp(-r_kpc / (C14_TAU * hR_kpc))

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
            try: mrt[p[0]] = {'T': int(p[1]), 'Rdisk': float(p[11])}
            except: continue
    return pipe, mrt

def main():
    print("=" * 70)
    print("C15 bias + C14 correction test")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    gc_obs = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    hR = np.array([mrt[n]['Rdisk'] for n in names])
    T_type = np.array([mrt[n].get('T', np.nan) for n in names])

    gc_pred = gc_c15(vf, hR, Yd)
    lgc = np.log10(gc_obs); log_resid = lgc - np.log10(gc_pred)

    # T1
    print("\n" + "=" * 60)
    print("T1: C15 bias")
    print("=" * 60)
    print(f"  Overall: bias={np.mean(log_resid):+.3f}, scatter={np.std(log_resid):.3f}")
    nbins = 5; pcts = np.percentile(lgc, np.linspace(0, 100, nbins+1))
    bb, bbc = [], []
    print(f"  {'bin':>3s} {'N':>4s} {'bias':>8s} {'scatter':>8s}")
    for i in range(nbins):
        m = (lgc >= pcts[i]) & (lgc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 3: continue
        bi = np.mean(log_resid[m])
        bb.append(bi)
        print(f"  {i+1:3d} {m.sum():4d} {bi:+8.3f} {np.std(log_resid[m]):8.3f}")

    # T2
    print("\n" + "=" * 60)
    print("T2: C14 at representative radii")
    print("=" * 60)
    best_r, best_abs = None, 999
    print(f"  {'r/hR':>5s} {'f14':>6s} {'bias':>8s} {'scatter':>8s}")
    for rm in [1, 2, 3, 4, 5, 6]:
        f = f14(rm*hR, hR)
        ge = gc_pred * f
        res = lgc - np.log10(ge)
        bi = np.mean(res)
        print(f"  {rm:5d} {np.median(f):6.3f} {bi:+8.3f} {np.std(res):8.3f}")
        if abs(bi) < best_abs: best_abs = abs(bi); best_r = rm
    print(f"  Best: r/hR={best_r}")

    # T3
    print("\n" + "=" * 60)
    print("T3: Standard radii")
    print("=" * 60)
    for lbl, rv in [("2.2hR", 2.2*hR), ("3.0hR", 3.0*hR), ("4.0hR", 4.0*hR)]:
        f = f14(rv, hR); ge = gc_pred*f; res = lgc - np.log10(ge)
        print(f"  {lbl:8s}: bias={np.mean(res):+.3f}, scatter={np.std(res):.3f}")

    # T4
    print("\n" + "=" * 60)
    print("T4: Bias gc-dependence before/after")
    print("=" * 60)
    rb = best_r*hR; fb = f14(rb, hR); geb = gc_pred*fb
    resb = lgc - np.log10(geb)
    sl_bf, _, r_bf, _, _ = stats.linregress(lgc, log_resid)
    sl_af, _, r_af, _, _ = stats.linregress(lgc, resb)
    print(f"  C15: slope={sl_bf:+.3f} (r={r_bf:.3f})")
    print(f"  C14+15: slope={sl_af:+.3f} (r={r_af:.3f})")
    print(f"  Reduction: {(1-abs(sl_af)/abs(sl_bf))*100:.1f}%")

    print(f"\n  {'bin':>3s} {'C15':>8s} {'C14+15':>8s} {'improved':>9s}")
    n_imp = 0
    for i in range(nbins):
        m = (lgc >= pcts[i]) & (lgc <= pcts[i+1] + (0.001 if i == nbins-1 else 0))
        if m.sum() < 3: continue
        b1 = np.mean(log_resid[m]); b2 = np.mean(resb[m])
        imp = abs(b2) < abs(b1)
        if imp: n_imp += 1
        bbc.append(b2)
        print(f"  {i+1:3d} {b1:+8.3f} {b2:+8.3f} {'Yes' if imp else 'No':>9s}")
    print(f"  Improved: {n_imp}/{nbins}")

    # T5: r_eff
    print("\n" + "=" * 60)
    print("T5: Optimal r_eff per galaxy")
    print("=" * 60)
    ratio = gc_obs / gc_pred
    arg = (ratio - C14_A) / C14_B
    valid_r = arg > 0
    r_eff = np.full(N, np.nan)
    r_eff[valid_r] = -C14_TAU * hR[valid_r] * np.log(arg[valid_r])
    reh = r_eff / hR
    vm = np.isfinite(reh) & (reh > 0) & (reh < 20)
    print(f"  Valid: {vm.sum()}/{N}")
    if vm.sum() > 10:
        print(f"  r_eff/hR: median={np.median(reh[vm]):.2f}, mean={np.mean(reh[vm]):.2f}, std={np.std(reh[vm]):.2f}")
        for lbl, va in [("gc", lgc), ("vflat", np.log10(vf)), ("hR", np.log10(hR)), ("Yd", Yd)]:
            r, p = stats.pearsonr(va[vm], reh[vm])
            print(f"    r(r_eff/hR, {lbl:6s}) = {r:+.3f} (p={p:.3e})")

    # T6: Rotmod validation
    print("\n" + "=" * 60)
    print("T6: Rotmod validation")
    print("=" * 60)
    gc_local_l, gc_eff_l, gc_gl_l, gc_obs_l = [], [], [], []
    for j in range(N):
        fp = os.path.join(ROTMOD, f"{names[j]}_rotmod.dat")
        if not os.path.exists(fp): continue
        try:
            d = np.loadtxt(fp, comments='#')
            if d.ndim != 2 or d.shape[1] < 6: continue
        except: continue
        R = d[:, 0]; Vobs = d[:, 1]; Vgas = d[:, 3]; Vdisk = d[:, 4]; Vbul = d[:, 5]
        Rm = R * kpc_m; valid = Rm > 0
        if valid.sum() < 3: continue
        gobs = (Vobs[valid]*1e3)**2 / Rm[valid]
        Vb2 = (np.sign(Vgas[valid])*Vgas[valid]**2 + Yd[j]*np.sign(Vdisk[valid])*Vdisk[valid]**2
               + 0.7*np.sign(Vbul[valid])*Vbul[valid]**2) * 1e6
        gN = np.abs(Vb2) / Rm[valid]
        m2 = (gN > 0) & (gobs > gN)
        if m2.sum() < 2: continue
        gc_local = gobs[m2] * (gobs[m2] - gN[m2]) / gN[m2]
        gc_gl = gc_c15(vf[j], hR[j], Yd[j])
        gc_eff_r = gc_gl * f14(R[valid][m2], hR[j])
        outer = R[valid][m2] > 2*hR[j]
        if outer.sum() < 2: outer = np.ones(m2.sum(), dtype=bool)
        gc_local_l.append(np.median(gc_local[outer]))
        gc_eff_l.append(np.median(gc_eff_r[outer]))
        gc_gl_l.append(gc_gl); gc_obs_l.append(gc_obs[j])

    if len(gc_local_l) >= 10:
        for lbl, gt in [("C15", np.array(gc_gl_l)), ("C14+15", np.array(gc_eff_l)),
                         ("Rotmod local", np.array(gc_local_l))]:
            res = np.log10(np.array(gc_obs_l)) - np.log10(gt)
            v = np.isfinite(res)
            print(f"  {lbl:15s}: bias={np.mean(res[v]):+.3f}, scatter={np.std(res[v]):.3f} (N={v.sum()})")
    else:
        print(f"  Only {len(gc_local_l)} galaxies with Rotmod")

    # T7: Galaxy type
    print("\n" + "=" * 60)
    print("T7: By galaxy type")
    print("=" * 60)
    for lbl, mask in [("Sa-Sb T<=3", T_type <= 3), ("Sbc-Sd 4-7", (T_type >= 4) & (T_type <= 7)),
                       ("Sdm-Im T>=8", T_type >= 8)]:
        mt = mask & np.isfinite(T_type)
        if mt.sum() < 5: continue
        b1 = np.mean(log_resid[mt]); s1 = np.std(log_resid[mt])
        b2 = np.mean(resb[mt]); s2 = np.std(resb[mt])
        print(f"  {lbl}: N={mt.sum()}")
        print(f"    C15: bias={b1:+.3f} sc={s1:.3f}  C14+15: bias={b2:+.3f} sc={s2:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  C15: bias={np.mean(log_resid):+.3f}, scatter={np.std(log_resid):.3f}, slope={sl_bf:+.3f}
  C14+15 (r/{best_r}hR): bias={np.mean(resb):+.3f}, scatter={np.std(resb):.3f}, slope={sl_af:+.3f}
  Slope reduction: {(1-abs(sl_af)/abs(sl_bf))*100:.1f}%
  Bins improved: {n_imp}/{nbins}
  r_eff/hR median: {f'{np.median(reh[vm]):.2f}' if vm.sum()>0 else 'N/A'}
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        ax.scatter(lgc, log_resid, s=15, alpha=0.5, c='steelblue', label='C15')
        ax.scatter(lgc, resb, s=15, alpha=0.5, c='coral', label='C14+15')
        ax.axhline(0, ls='--', color='grey')
        xf = np.linspace(lgc.min(), lgc.max(), 50)
        ax.plot(xf, sl_bf*xf+(np.mean(log_resid)-sl_bf*lgc.mean()), 'b-', alpha=0.7)
        ax.plot(xf, sl_af*xf+(np.mean(resb)-sl_af*lgc.mean()), 'r-', alpha=0.7)
        ax.set_xlabel('log gc'); ax.set_ylabel('residual'); ax.set_title('(a) bias vs gc')
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        x_pos = np.arange(len(bb))
        w = 0.35
        ax.bar(x_pos-w/2, bb, w, color='steelblue', alpha=0.7, label='C15')
        ax.bar(x_pos+w/2, bbc, w, color='coral', alpha=0.7, label='C14+15')
        ax.axhline(0, ls='--', color='grey')
        ax.set_xticks(x_pos); ax.set_xticklabels([f'B{i+1}' for i in range(len(bb))])
        ax.set_ylabel('bias'); ax.set_title('(b) binned bias'); ax.legend(fontsize=7)

        ax = axes[0, 2]
        rh = np.linspace(0, 12, 200)
        ax.plot(rh, C14_A+C14_B*np.exp(-rh/C14_TAU), 'b-', lw=2)
        ax.axhline(C14_A, ls=':', color='grey'); ax.axvline(best_r, ls='--', color='green')
        ax.set_xlabel('r/hR'); ax.set_ylabel('f14'); ax.set_title('(c) C14 factor')

        ax = axes[1, 0]
        if vm.sum() > 5:
            ax.hist(reh[vm], bins=30, color='steelblue', alpha=0.7)
            ax.axvline(np.median(reh[vm]), color='red', lw=2)
            ax.set_xlabel('r_eff/hR'); ax.set_title('(d) optimal r_eff')

        ax = axes[1, 1]
        if vm.sum() > 5:
            ax.scatter(lgc[vm], reh[vm], s=15, alpha=0.5)
            sl_r, ir, rr, pr, _ = stats.linregress(lgc[vm], reh[vm])
            xf = np.linspace(lgc[vm].min(), lgc[vm].max(), 50)
            ax.plot(xf, sl_r*xf+ir, 'r-', label=f'r={rr:.3f}')
            ax.set_xlabel('log gc'); ax.set_ylabel('r_eff/hR'); ax.set_title('(e) r_eff vs gc')
            ax.legend()

        ax = axes[1, 2]
        ax.scatter(lgc, np.log10(gc_pred), s=12, alpha=0.4, c='steelblue', label='C15')
        ax.scatter(lgc, np.log10(geb), s=12, alpha=0.4, c='coral', label='C14+15')
        lim = [lgc.min()-0.1, lgc.max()+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('log gc obs'); ax.set_ylabel('log gc pred'); ax.set_title('(f) pred vs obs')
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'cond15_bias_cond14_correction.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == "__main__":
    main()
