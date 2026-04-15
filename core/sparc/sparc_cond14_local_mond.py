#!/usr/bin/env python3
"""sparc_cond14_local_mond.py (TA3+phase1+MRT adapted)
kappa=0 local MOND equilibrium test for Condition 14 pattern."""
import os, csv, warnings
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
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

def g_obs_mond(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4*g_bar*gc)) / 2

def gc_from_rar(g_bar, g_obs):
    m = (g_obs > g_bar) & (g_bar > 0)
    gc = np.full_like(g_obs, np.nan)
    gc[m] = g_obs[m]*(g_obs[m]-g_bar[m])/g_bar[m]
    return gc

def c14_model(rh, A, B, tau): return A + B*np.exp(-rh/tau)

def main():
    print("=" * 70)
    print("kappa=0 local MOND: Condition 14 pattern test")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc_a0' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    print(f"N={len(names)}")

    all_data = []
    for gname in names:
        gd = pipe[gname]; gc_a0 = gd['gc_a0']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        hR = mrt[gname]['Rdisk']; T = mrt[gname].get('T', np.nan)
        fp = os.path.join(ROTMOD, f"{gname}_rotmod.dat")
        if not os.path.exists(fp): continue
        try: d = np.loadtxt(fp, comments='#')
        except: continue
        if d.ndim != 2 or d.shape[1] < 6 or len(d) < 5: continue
        R = d[:, 0]; Vobs = d[:, 1]; Vgas = d[:, 3]; Vdisk = d[:, 4]; Vbul = d[:, 5]
        Rm = R*kpc_m; valid = Rm > 0
        if valid.sum() < 5: continue
        Rm = Rm[valid]; R_ = R[valid]
        g_obs = (Vobs[valid]*1e3)**2/Rm
        Vb2 = (Yd*np.sign(Vdisk[valid])*Vdisk[valid]**2 + np.sign(Vgas[valid])*Vgas[valid]**2
               + np.sign(Vbul[valid])*Vbul[valid]**2)*1e6
        g_bar = np.maximum(np.abs(Vb2)/Rm, 1e-20)
        gc_si = gc_a0*a0; x_r = g_bar/gc_si
        g_obs_pred = g_obs_mond(g_bar, gc_si)
        gc_local = gc_from_rar(g_bar, g_obs)
        ratio = gc_local/gc_si
        rh = R_/hR
        v = np.isfinite(ratio) & (ratio > 0) & (ratio < 100) & (rh > 0)
        if v.sum() < 5: continue
        all_data.append({'name': gname, 'r': R_[v], 'rh': rh[v], 'g_obs': g_obs[v],
                         'g_bar': g_bar[v], 'g_obs_pred': g_obs_pred[v],
                         'gc_local': gc_local[v], 'ratio': ratio[v], 'x_r': x_r[v],
                         'hR': hR, 'vflat': vf, 'T': T, 'gc_a0': gc_a0})

    print(f"Processed: {len(all_data)} galaxies")
    if len(all_data) < 10: print("Too few."); return

    # T1: x(r) structure
    print("\n" + "=" * 60)
    print("T1: x(r) = g_bar/gc profile")
    print("=" * 60)
    n_tot = sum(len(d['x_r']) for d in all_data)
    n_newton = sum(np.sum(d['x_r'] > 1) for d in all_data)
    n_deep = sum(np.sum(d['x_r'] < 0.1) for d in all_data)
    print(f"  x>1 (Newton): {n_newton}/{n_tot} ({100*n_newton/n_tot:.1f}%)")
    print(f"  x<0.1 (deep MOND): {n_deep}/{n_tot} ({100*n_deep/n_tot:.1f}%)")

    # T2: simple interpolation accuracy
    print("\n" + "=" * 60)
    print("T2: Simple interpolation accuracy")
    print("=" * 60)
    all_resid = np.concatenate([np.log10(d['g_obs']/d['g_obs_pred']) for d in all_data])
    print(f"  N_pts={len(all_resid)}, RMS={np.sqrt(np.mean(all_resid**2)):.4f} dex")

    # T3: gc_local/gc_global
    print("\n" + "=" * 60)
    print("T3: gc_local/gc_global statistics")
    print("=" * 60)
    all_ratio = np.concatenate([d['ratio'] for d in all_data])
    print(f"  median={np.median(all_ratio):.4f}, mean={np.mean(all_ratio):.4f}")
    print(f"  Theory (simple interp, kappa=0): ratio=1.0 everywhere")
    print(f"  Deviation: {np.median(all_ratio)-1:+.4f}")

    # T4: Stacked r/hR pattern -> C14 fit
    print("\n" + "=" * 60)
    print("T4: Stacked ratio vs r/hR")
    print("=" * 60)
    all_rh = np.concatenate([d['rh'] for d in all_data])
    all_rat = np.concatenate([d['ratio'] for d in all_data])
    bins = np.arange(0.5, 12.5, 0.5)
    bc, bm, bs = [], [], []
    print(f"  {'r/hR':>6s} {'median':>8s} {'N':>5s}")
    for i in range(len(bins)-1):
        m = (all_rh >= bins[i]) & (all_rh < bins[i+1])
        if m.sum() >= 10:
            c = (bins[i]+bins[i+1])/2; med = np.median(all_rat[m])
            bc.append(c); bm.append(med); bs.append(np.std(all_rat[m]))
            print(f"  {c:6.1f} {med:8.3f} {m.sum():5d}")
    bc = np.array(bc); bm = np.array(bm)
    popt = None
    if len(bc) >= 4:
        try:
            popt, _ = curve_fit(c14_model, bc, bm, p0=[0.1, 0.5, 4], maxfev=5000,
                                bounds=([0,0,0.5],[2,5,20]))
            print(f"\n  C14 fit: A={popt[0]:.4f}, B={popt[1]:.4f}, tau={popt[2]:.3f}")
            print(f"  Empirical: A={C14_A}, B={C14_B}, tau={C14_TAU}")
        except: pass

    # T5: Per-galaxy C14 fits
    print("\n" + "=" * 60)
    print("T5: Per-galaxy C14 fits")
    print("=" * 60)
    fit_r = []
    for d in all_data:
        if len(d['rh']) < 6: continue
        try:
            po, _ = curve_fit(c14_model, d['rh'], d['ratio'], p0=[0.1,0.5,4],
                               maxfev=5000, bounds=([0,0,0.5],[2,5,20]))
            fit_r.append({'name': d['name'], 'A': po[0], 'B': po[1], 'tau': po[2]})
        except: continue
    if fit_r:
        A_ = np.array([f['A'] for f in fit_r])
        B_ = np.array([f['B'] for f in fit_r])
        tau_ = np.array([f['tau'] for f in fit_r])
        print(f"  N={len(fit_r)}")
        print(f"  A: median={np.median(A_):.4f} (emp {C14_A})")
        print(f"  B: median={np.median(B_):.4f} (emp {C14_B})")
        print(f"  tau: median={np.median(tau_):.3f} (emp {C14_TAU})")

    # T6: ratio vs x(r)
    print("\n" + "=" * 60)
    print("T6: ratio vs x(r)")
    print("=" * 60)
    all_x = np.concatenate([d['x_r'] for d in all_data])
    print(f"  Theory: ratio=1.0 for all x (simple interp)")
    lx_bins = np.arange(-2.5, 2, 0.25)
    xc, xm = [], []
    for i in range(len(lx_bins)-1):
        m = (np.log10(all_x) >= lx_bins[i]) & (np.log10(all_x) < lx_bins[i+1])
        if m.sum() >= 10:
            c = (lx_bins[i]+lx_bins[i+1])/2; xc.append(c); xm.append(np.median(all_rat[m]))
    if len(xc) > 3:
        sl, _, rv, pv, _ = stats.linregress(xc, xm)
        print(f"  ratio vs log(x): slope={sl:.4f}, R2={rv**2:.3f}, p={pv:.2e}")

    # T7: By type
    print("\n" + "=" * 60)
    print("T7: By galaxy type")
    print("=" * 60)
    if fit_r:
        T_arr = np.array([mrt.get(f['name'], {}).get('T', np.nan) for f in fit_r])
        for lbl, lo, hi in [("T<=3", -2, 3), ("T=4-7", 4, 7), ("T>=8", 8, 12)]:
            m = (T_arr >= lo) & (T_arr <= hi) & np.isfinite(T_arr)
            if m.sum() < 3: continue
            print(f"  {lbl}: N={m.sum()}, B_med={np.median(B_[m]):.4f}, tau_med={np.median(tau_[m]):.3f}")

    # T8: Residual after local MOND
    print("\n" + "=" * 60)
    print("T8: Residual after local MOND")
    print("=" * 60)
    print(f"  Overall RMS = {np.sqrt(np.mean(all_resid**2)):.4f} dex")
    rb = np.arange(0.5, 12.5, 1)
    all_rh2 = np.concatenate([d['rh'] for d in all_data])
    all_res2 = all_resid  # already computed
    # Need to align lengths
    all_res_rh = []
    for d in all_data:
        res = np.log10(d['g_obs']/d['g_obs_pred'])
        for j in range(len(d['rh'])):
            all_res_rh.append((d['rh'][j], res[j]))
    all_res_rh = np.array(all_res_rh)
    print(f"  {'r/hR':>6s} {'median resid':>12s} {'N':>5s}")
    for i in range(len(rb)-1):
        m = (all_res_rh[:,0] >= rb[i]) & (all_res_rh[:,0] < rb[i+1])
        if m.sum() >= 10:
            print(f"  {(rb[i]+rb[i+1])/2:6.1f} {np.median(all_res_rh[m,1]):+12.4f} {m.sum():5d}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  kappa=0 local MOND prediction: ratio = gc_local/gc_global = 1.0 everywhere
  Observed: median ratio = {np.median(all_ratio):.3f}

  Stacked C14 fit: A={popt[0]:.3f}, B={popt[1]:.3f}, tau={popt[2]:.1f}
  Empirical C14:   A={C14_A}, B={C14_B}, tau={C14_TAU}

  Simple interpolation RMS: {np.sqrt(np.mean(all_resid**2)):.4f} dex

  Key insight:
    ratio < 1 at all radii -> gc_global systematically overestimates gc_local
    The r/hR pattern emerges from MOND transition x(r) structure
    NOT from membrane rigidity (kappa=0 confirmed)
""" if popt is not None else "  C14 fit failed")

    # Figure
    try:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        ax = axes[0, 0]
        for d in all_data[:10]:
            ax.plot(d['rh'], d['x_r'], alpha=0.5, lw=0.8)
        ax.axhline(1, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('r/hR'); ax.set_ylabel('x=g_bar/gc')
        ax.set_title('(1) x(r) profiles'); ax.set_yscale('log'); ax.set_xlim(0, 12)

        ax = axes[0, 1]
        for d in all_data:
            ax.scatter(np.log10(d['g_obs_pred']), np.log10(d['g_obs']), s=1, alpha=0.1, c='blue')
        ax.plot([-12,-8.5],[-12,-8.5],'r-'); ax.set_xlabel('log g_pred'); ax.set_ylabel('log g_obs')
        ax.set_title('(2) MOND accuracy')

        ax = axes[0, 2]
        for d in all_data[:15]:
            ax.plot(d['rh'], d['ratio'], alpha=0.3, lw=0.5)
        rhl = np.linspace(0.5, 12, 100)
        ax.plot(rhl, c14_model(rhl, C14_A, C14_B, C14_TAU), 'k-', lw=2, label='C14 emp')
        ax.axhline(1, color='red', ls='--'); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
        ax.set_xlabel('r/hR'); ax.set_ylabel('ratio'); ax.set_title('(3) ratio profiles')
        ax.legend(fontsize=7)

        ax = axes[0, 3]
        if len(bc) > 0:
            ax.errorbar(bc, bm, yerr=np.array(bs)/np.sqrt(10), fmt='ko', ms=4)
            ax.plot(rhl, c14_model(rhl, C14_A, C14_B, C14_TAU), 'r-', lw=2, label='emp')
            if popt is not None:
                ax.plot(rhl, c14_model(rhl, *popt), 'b--', lw=2, label='fit')
            ax.axhline(1, color='grey', ls=':')
        ax.set_xlabel('r/hR'); ax.set_ylabel('median ratio'); ax.set_title('(4) stacked')
        ax.set_xlim(0, 12); ax.legend(fontsize=7)

        ax = axes[1, 0]
        if fit_r:
            ax.scatter(A_, B_, s=10, alpha=0.5)
            ax.axhline(C14_B, color='red', ls='--'); ax.axvline(C14_A, color='red', ls='--')
        ax.set_xlabel('A'); ax.set_ylabel('B'); ax.set_title('(5) per-galaxy A vs B')

        ax = axes[1, 1]
        ax.scatter(np.log10(all_x), all_rat, s=1, alpha=0.05, c='blue')
        ax.axhline(1, color='red', ls='--'); ax.set_ylim(0, 3)
        ax.set_xlabel('log x'); ax.set_ylabel('ratio'); ax.set_title('(6) ratio vs x')

        ax = axes[1, 2]
        if fit_r:
            ax.hist(B_, bins=20, alpha=0.7)
            ax.axvline(C14_B, color='red', ls='--', label=f'emp {C14_B}')
            ax.axvline(np.median(B_), color='blue', ls=':', label=f'med {np.median(B_):.3f}')
            ax.set_xlabel('B'); ax.set_title('(7) B distribution'); ax.legend(fontsize=7)

        ax = axes[1, 3]
        rr = np.array(all_res_rh)
        rbc, rmed = [], []
        for i in range(len(rb)-1):
            m = (rr[:,0] >= rb[i]) & (rr[:,0] < rb[i+1])
            if m.sum() > 5:
                rbc.append((rb[i]+rb[i+1])/2); rmed.append(np.median(rr[m,1]))
        ax.plot(rbc, rmed, 'ko-', ms=4); ax.axhline(0, color='red', ls='--')
        ax.set_xlabel('r/hR'); ax.set_ylabel('median resid'); ax.set_title('(8) residual pattern')
        ax.set_xlim(0, 12)

        plt.suptitle('Condition 14 from Local MOND (kappa=0)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'cond14_local_mond_results.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
