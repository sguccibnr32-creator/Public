#!/usr/bin/env python3
"""sparc_cond14_15_integration.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings, glob
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr
from scipy.optimize import minimize, curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19

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

def load_rotmod(gname):
    fp = os.path.join(ROTMOD, f"{gname}_rotmod.dat")
    if not os.path.exists(fp): return None
    try:
        d = np.loadtxt(fp, comments='#')
        if d.ndim == 2 and d.shape[1] >= 6:
            return {'Rad': d[:,0], 'Vobs': d[:,1], 'errV': d[:,2],
                    'Vgas': d[:,3], 'Vdisk': d[:,4], 'Vbul': d[:,5]}
    except: pass
    return None

def main():
    print("=" * 70)
    print("Condition 14+15 Integration")
    print("=" * 70)
    pipe = load_pipeline(); mrt_d = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt_d)}")

    galaxy_data = []
    for n in sorted(pipe.keys()):
        gd = pipe[n]; gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vf <= 0: continue
        m = mrt_d.get(n, {}); hR = m.get('Rdisk', 0)
        if hR <= 0: continue
        rm = load_rotmod(n)
        if rm is None or len(rm['Rad']) < 5: continue

        Rad_m = rm['Rad'] * kpc_m
        v2gas = np.sign(rm['Vgas']) * rm['Vgas']**2 * 1e6
        v2disk = np.sign(rm['Vdisk']) * rm['Vdisk']**2 * 1e6
        v2bul = np.sign(rm['Vbul']) * rm['Vbul']**2 * 1e6
        valid = Rad_m > 0
        gbar = np.zeros_like(Rad_m)
        gbar[valid] = np.abs(v2gas[valid]/Rad_m[valid] + Yd*v2disk[valid]/Rad_m[valid] + 0.7*v2bul[valid]/Rad_m[valid])
        gobs = np.zeros_like(Rad_m)
        gobs[valid] = (rm['Vobs'][valid]*1e3)**2 / Rad_m[valid]
        eps = np.zeros_like(gobs)
        m2 = valid & (gbar > 0)
        eps[m2] = gobs[m2]/gbar[m2] - 1

        vv = valid & (gbar > 0) & (gobs > 0) & np.isfinite(eps)
        if vv.sum() < 3: continue
        galaxy_data.append({
            'name': n, 'gc': gc, 'vflat': vf, 'hR': hR, 'Yd': Yd,
            'Rad': rm['Rad'][vv], 'Vobs': rm['Vobs'][vv], 'errV': rm['errV'][vv],
            'gbar': gbar[vv], 'gobs': gobs[vv], 'eps': eps[vv],
            'r_norm': rm['Rad'][vv] / hR,
        })

    N_gal = len(galaxy_data)
    print(f"Loaded: {N_gal} galaxies")

    gc_arr = np.array([g['gc'] for g in galaxy_data])
    vf_arr = np.array([g['vflat'] for g in galaxy_data])
    hR_arr = np.array([g['hR'] for g in galaxy_data])
    Yd_arr = np.array([g['Yd'] for g in galaxy_data])
    Sd = (vf_arr*1e3)**2 / (hR_arr*kpc_m)
    log_gc = np.log10(gc_arr); log_Sd = np.log10(Sd); log_Yd = np.log10(Yd_arr)
    log_a0Sd = np.log10(a0*Sd)

    # T1: Condition 15
    print("\n" + "=" * 70)
    print("T1: Condition 15")
    print("=" * 70)
    X15 = np.column_stack([np.ones(N_gal), log_a0Sd, log_Yd])
    b15 = np.linalg.lstsq(X15, log_gc, rcond=None)[0]
    pred15 = X15 @ b15; resid15 = log_gc - pred15
    R2_15 = 1 - np.sum(resid15**2)/np.sum((log_gc - log_gc.mean())**2)
    gc_global = 10**pred15
    print(f"  alpha={b15[1]:.4f}, beta={b15[2]:.4f}, eta0={10**b15[0]:.4f}")
    print(f"  R2={R2_15:.4f}, scatter={np.std(resid15):.4f}")

    # T2: epsilon profiles
    print("\n" + "=" * 70)
    print("T2: epsilon(r) statistics")
    print("=" * 70)
    all_rn, all_eps, all_gi = [], [], []
    for i, gd in enumerate(galaxy_data):
        for j in range(len(gd['Rad'])):
            all_rn.append(gd['r_norm'][j]); all_eps.append(gd['eps'][j]); all_gi.append(i)
    all_rn = np.array(all_rn); all_eps = np.array(all_eps)
    print(f"  N_pts={len(all_rn)}, eps: mean={np.mean(all_eps):.3f}, median={np.median(all_eps):.3f}, std={np.std(all_eps):.3f}")
    r_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 999]
    print(f"\n  {'r/hR':>10s} {'N':>5s} {'<eps>':>8s} {'median':>8s}")
    for k in range(len(r_bins)-1):
        m = (all_rn >= r_bins[k]) & (all_rn < r_bins[k+1])
        if m.sum() < 5: continue
        lbl = f"[{r_bins[k]:.1f},{r_bins[k+1]:.1f})" if r_bins[k+1] < 100 else f">{r_bins[k]:.1f}"
        print(f"  {lbl:>10s} {m.sum():5d} {np.mean(all_eps[m]):8.3f} {np.median(all_eps[m]):8.3f}")
    rho_er, _ = spearmanr(all_rn, all_eps)
    print(f"  rho(eps, r/hR) = {rho_er:+.3f}")

    # T3: Condition 14 fit
    print("\n" + "=" * 70)
    print("T3: Condition 14 fit")
    print("=" * 70)
    all_gr, all_rn2 = [], []
    for i, gd in enumerate(galaxy_data):
        ge = gd['gobs'] - gd['gbar']; m = ge > 0
        if m.sum() < 2: continue
        for j in np.where(m)[0]:
            all_gr.append(ge[j]/gc_global[i]); all_rn2.append(gd['r_norm'][j])
    all_gr = np.array(all_gr); all_rn2 = np.array(all_rn2)
    print(f"  g_excess/gc_global: N={len(all_gr)}, mean={np.mean(all_gr):.3f}, median={np.median(all_gr):.3f}")
    rho_gr, _ = spearmanr(all_rn2, all_gr)
    print(f"  rho(ratio, r/hR) = {rho_gr:+.3f}")

    bin_r, bin_ratio = [], []
    print(f"\n  {'r/hR':>8s} {'N':>5s} {'<ratio>':>8s} {'std':>8s}")
    for k in range(len(r_bins)-1):
        m = (all_rn2 >= r_bins[k]) & (all_rn2 < r_bins[k+1])
        if m.sum() < 5: continue
        rm_ = (r_bins[k]+min(r_bins[k+1], all_rn2[m].max()))/2
        bin_r.append(rm_); bin_ratio.append(np.median(all_gr[m]))
        lbl = f"{r_bins[k]:.1f}-{r_bins[k+1]:.1f}" if r_bins[k+1]<100 else f">{r_bins[k]:.1f}"
        print(f"  {lbl:>8s} {m.sum():5d} {np.mean(all_gr[m]):8.3f} {np.std(all_gr[m]):8.3f}")

    def model_c14(rn, A, B, rs): return A + B*np.exp(-rn/rs)
    popt = None
    if len(bin_r) >= 3:
        try:
            popt, _ = curve_fit(model_c14, np.array(bin_r), np.array(bin_ratio), p0=[1,1,2], maxfev=10000)
            print(f"\n  C14 fit: A={popt[0]:.4f}, B={popt[1]:.4f}, r_scale={popt[2]:.4f}")
            print(f"  center(r=0): {popt[0]+popt[1]:.4f}, outer: {popt[0]:.4f}")
            if popt[1] > 0:
                print(f"  -> Center excess: {popt[1]/popt[0]*100:.0f}%")
        except Exception as e:
            print(f"  Fit failed: {e}")

    # T4: C15 vs C14+15
    print("\n" + "=" * 70)
    print("T4: C15 only vs C14+15")
    print("=" * 70)
    chi2_c15, chi2_c14, beta14 = [], [], []
    for i, gd in enumerate(galaxy_data):
        gobs = gd['gobs']; gbar = gd['gbar']; errV = gd['errV']
        gc_gl = gc_global[i]; rn = gd['r_norm']
        Vm = gd['Vobs']*1e3; Rm = gd['Rad']*kpc_m
        err_g = np.where(Rm > 0, 2*Vm*errV*1e3/Rm, 1e-10)
        err_g = np.maximum(err_g, 1e-12)
        g15 = gbar + gc_gl
        c2a = np.sum(((gobs-g15)/err_g)**2)
        def obj(p):
            B, rs = p
            if rs < 0.1: return 1e20
            return np.sum(((gobs - gbar - gc_gl*(1+B*np.exp(-rn/rs)))/err_g)**2)
        try:
            res = minimize(obj, [0.5, 2.0], method='Nelder-Mead', options={'maxiter': 5000})
            c2b = res.fun; Bf = res.x[0]
        except: c2b = c2a; Bf = 0
        n = len(gobs)
        chi2_c15.append(c2a/max(n-1,1)); chi2_c14.append(c2b/max(n-3,1)); beta14.append(Bf)
    chi2_c15 = np.array(chi2_c15); chi2_c14 = np.array(chi2_c14); beta14 = np.array(beta14)
    improved = chi2_c14 < chi2_c15
    delta_chi2 = chi2_c15 - chi2_c14
    print(f"  Improved: {improved.sum()}/{N_gal} ({100*improved.mean():.0f}%)")
    print(f"  median chi2_red: C15={np.median(chi2_c15):.3f}, C14+15={np.median(chi2_c14):.3f}")
    print(f"  median delta_chi2={np.median(delta_chi2):+.3f}")
    print(f"\n  beta14: mean={np.mean(beta14):.3f}, median={np.median(beta14):.3f}, positive={np.sum(beta14>0)}/{N_gal}")

    # T5: beta14 correlations
    print("\n" + "=" * 70)
    print("T5: beta14 correlations")
    print("=" * 70)
    for vn, va in [("vflat", np.log10(vf_arr)), ("gc", log_gc), ("Yd", log_Yd), ("Sd", log_Sd)]:
        rho, p = spearmanr(va, beta14)
        print(f"  vs {vn:6s}: rho={rho:+.3f} (p={p:.2e})")

    # T6: stacked signal
    print("\n" + "=" * 70)
    print("T6: Stacked C14 signal")
    print("=" * 70)
    r_grid = np.linspace(0, 8, 20); stacked = []
    for k in range(len(r_grid)-1):
        m = (all_rn2 >= r_grid[k]) & (all_rn2 < r_grid[k+1])
        if m.sum() > 5:
            stacked.append(((r_grid[k]+r_grid[k+1])/2, np.median(all_gr[m]),
                           np.std(all_gr[m])/np.sqrt(m.sum())))
    if stacked:
        print(f"  {'r/hR':>6s} {'ratio':>8s} {'err':>8s}")
        for rm, rat, err in stacked:
            mk = " *" if abs(rat-1) > 2*err else ""
            print(f"  {rm:6.2f} {rat:8.3f} {err:8.3f}{mk}")

    # T7: top improved
    print("\n" + "=" * 70)
    print("T7: Top improved galaxies")
    print("=" * 70)
    top = np.argsort(delta_chi2)[-10:][::-1]
    print(f"  {'galaxy':<15s} {'chi2_C15':>8s} {'chi2_C14':>8s} {'delta':>8s} {'beta14':>8s}")
    for idx in top:
        print(f"  {galaxy_data[idx]['name']:<15s} {chi2_c15[idx]:8.2f} {chi2_c14[idx]:8.2f} {delta_chi2[idx]:+8.2f} {beta14[idx]:8.3f}")

    # T8: Summary
    print("\n" + "=" * 70)
    print("T8: Integration summary")
    print("=" * 70)
    print(f"""
  C15: alpha={b15[1]:.4f}, beta={b15[2]:.4f}, eta0={10**b15[0]:.4f}, R2={R2_15:.4f}
  C14: <beta14>={np.median(beta14):.3f}, r_scale={'N/A' if popt is None else f'{popt[2]:.2f}'}
  Improvement: {improved.sum()}/{N_gal} galaxies, median delta_chi2={np.median(delta_chi2):+.3f}
  beta14 positive: {np.sum(beta14>0)}/{N_gal}

  gc_eff(r) = eta0 * Yd^beta * sqrt(a0*vflat^2/hR) * [1 + beta14*exp(-r/(rs*hR))]
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        ax.scatter(all_rn, all_eps, s=2, alpha=0.1, c='gray')
        if stacked:
            rs_, rats_, errs_ = zip(*stacked)
            ax.errorbar(rs_, [r-1 for r in rats_], yerr=errs_, fmt='ro-', capsize=3)
        ax.axhline(0, color='k', ls='--'); ax.set_xlim(0, 10)
        ax.set_xlabel('r/hR'); ax.set_ylabel('eps'); ax.set_title('(a) eps profile')

        ax = axes[0, 1]
        ax.scatter(all_rn2, all_gr, s=2, alpha=0.1, c='gray')
        if popt is not None:
            rp = np.linspace(0, 8, 200)
            ax.plot(rp, model_c14(rp, *popt), 'r-', lw=2)
        ax.axhline(1, color='k', ls='--'); ax.set_xlim(0, 8)
        ax.set_ylim(0, np.percentile(all_gr, 95)*1.2)
        ax.set_xlabel('r/hR'); ax.set_ylabel('g_exc/gc_global'); ax.set_title('(b) C14 signal')

        ax = axes[0, 2]
        ax.scatter(chi2_c15, chi2_c14, s=15, alpha=0.5)
        lim = [0, min(np.percentile(chi2_c15, 95), np.percentile(chi2_c14, 95))*1.1]
        ax.plot(lim, lim, 'r--')
        ax.set_xlabel('chi2 C15'); ax.set_ylabel('chi2 C14+15')
        ax.set_title(f'(c) {improved.sum()}/{N_gal} improved')

        ax = axes[1, 0]
        ax.hist(beta14, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', ls='--')
        ax.axvline(np.median(beta14), color='blue', label=f'med={np.median(beta14):.2f}')
        ax.set_xlabel('beta14'); ax.set_title('(d) beta14 dist'); ax.legend()

        ax = axes[1, 1]
        ax.scatter(np.log10(vf_arr), beta14, s=15, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        rho_bv, _ = spearmanr(np.log10(vf_arr), beta14)
        ax.set_xlabel('log vflat'); ax.set_ylabel('beta14')
        ax.set_title(f'(e) rho={rho_bv:+.3f}')

        ax = axes[1, 2]
        if len(top) > 0:
            best = top[0]; gd = galaxy_data[best]
            ax.errorbar(gd['Rad'], gd['Vobs'], yerr=gd['errV'], fmt='ko', capsize=2, ms=3, label='Vobs')
            Rm = gd['Rad']*kpc_m
            V15 = np.sqrt((gd['gbar']+gc_global[best])*Rm)/1e3
            V14 = np.sqrt((gd['gbar']+gc_global[best]*(1+beta14[best]*np.exp(-gd['r_norm']/2)))*Rm)/1e3
            ax.plot(gd['Rad'], V15, 'b-', lw=1.5, label='C15')
            ax.plot(gd['Rad'], V14, 'r-', lw=1.5, label='C14+15')
            ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]')
            ax.set_title(f"(f) {gd['name']}"); ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'cond14_15_integration.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
