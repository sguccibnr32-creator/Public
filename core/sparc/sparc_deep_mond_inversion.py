#!/usr/bin/env python3
"""
sparc_deep_mond_inversion.py (TA3+phase1 adapted)
Deep-MOND inversion approach for gc extraction.
"""
import os, csv, warnings, glob
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
a0 = 1.2e-10
kpc_m = 3.0857e19

def load_pipeline():
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                data[name] = {'vflat': float(row.get('vflat', '0')),
                              'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                if name in data and gc_a0 > 0:
                    data[name]['gc'] = gc_a0 * a0
            except: pass
    return {k: v for k, v in data.items() if 'gc' in v and v['vflat'] > 0}

def load_rotcurve(fpath):
    r, vobs, errv, vgas, vdisk, vbul = [], [], [], [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split()
            if len(p) < 6: continue
            try:
                r.append(float(p[0])); vobs.append(float(p[1]))
                errv.append(float(p[2])); vgas.append(float(p[3]))
                vdisk.append(float(p[4])); vbul.append(float(p[5]))
            except: continue
    return (np.array(r), np.array(vobs), np.array(errv),
            np.array(vgas), np.array(vdisk), np.array(vbul))

def compute_hR(rad, vdisk, Yd):
    vds = np.sqrt(max(Yd, 0.01)) * np.abs(vdisk)
    if len(vds) == 0: return None
    rpk = rad[np.argmax(vds)]
    if rpk < 0.01 or rpk >= rad.max() * 0.9: return None
    return rpk / 2.15

def fit_alpha(log_gc, g_proxy, n_boot=500):
    def resid(p, gp, lgc):
        return p[0] + p[1]*np.log10(a0*gp) - lgc
    res = optimize.least_squares(resid, [0, 0.5], args=(g_proxy, log_gc),
                                  bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha = res.x[1]; eta = 10**res.x[0]
    pred = res.x[0] + res.x[1]*np.log10(a0*g_proxy)
    R2 = 1 - np.sum((log_gc - pred)**2)/np.sum((log_gc - log_gc.mean())**2)
    boots = []
    ng = len(log_gc)
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(ng, ng, replace=True)
        try:
            rb = optimize.least_squares(resid, res.x, args=(g_proxy[idx], log_gc[idx]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            boots.append(rb.x[1])
        except: pass
    err = np.std(boots) if boots else np.nan
    z = (alpha - 0.5)/err if err > 0 else np.inf
    p05 = 2 * stats.norm.sf(abs(z))
    return alpha, err, eta, R2, p05

def main():
    print("=" * 70)
    print("S-2 Approach A: Deep-MOND Inversion")
    print("=" * 70)

    pipe = load_pipeline()
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Pipeline: {len(pipe)}, Rotmod: {len(rotmod_files)}")

    records = []
    for fpath in rotmod_files:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        if gname not in pipe: continue
        gd = pipe[gname]
        gc_known = gd['gc']; vflat = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc_known <= 0 or vflat <= 0: continue
        r_kpc, vobs, errv, vgas, vdisk, vbul = load_rotcurve(fpath)
        if len(r_kpc) < 5: continue
        hR = compute_hR(r_kpc, vdisk, Yd)
        if hR is None: continue

        mask = (r_kpc > 0) & (vobs > 0)
        r_m = r_kpc[mask] * kpc_m
        vobs_ms = vobs[mask] * 1e3
        gobs = vobs_ms**2 / r_m
        vbar2 = (Yd * vdisk[mask]**2 + vgas[mask]**2 * np.sign(vgas[mask]) +
                 Yd * vbul[mask]**2 * np.sign(vbul[mask])) * 1e6
        gbar = np.abs(vbar2) / r_m
        x = gbar / gc_known
        deep = x < 1.0
        if deep.sum() < 3: continue

        records.append({
            "name": gname, "gc_known": gc_known, "vflat": vflat, "hR": hR,
            "Yd": Yd, "r_kpc": r_kpc[mask], "gobs": gobs, "gbar": gbar,
            "x_ratio": x, "n_deep": int(deep.sum()),
            "n_very_deep": int((x < 0.3).sum()),
        })

    N = len(records)
    print(f"Valid (deep>=3): {N}")
    print(f"Total deep pts: {sum(r['n_deep'] for r in records)}")

    gc_known = np.array([r["gc_known"] for r in records])
    log_gc = np.log10(gc_known)
    vflat = np.array([r["vflat"] for r in records])
    hR = np.array([r["hR"] for r in records])
    g_proxy = (vflat * 1e3)**2 / (hR * kpc_m)

    # T1: MOND assumption
    print("\n" + "=" * 70)
    print("T1: MOND assumption (gc=a0)")
    print("=" * 70)
    g_sigma0_mond = []
    for rec in records:
        deep = rec["x_ratio"] < 1.0
        gbar_mond = rec["gobs"][deep]**2 / a0
        g_sigma0_mond.append(np.median(gbar_mond))
    g_sigma0_mond = np.array(g_sigma0_mond)
    v1 = (g_sigma0_mond > 0) & np.isfinite(g_sigma0_mond)
    if v1.sum() > 20:
        a1, e1, eta1, r2_1, p05_1 = fit_alpha(log_gc[v1], g_sigma0_mond[v1])
        z0 = a1/e1 if e1 > 0 else np.inf
        p0 = 2 * stats.norm.sf(abs(z0))
        print(f"  alpha={a1:.3f}+/-{e1:.3f}, R2={r2_1:.3f}")
        print(f"  p(alpha=0)={p0:.2e} {'MOND rejected' if p0 < 0.05 else 'not rejected'}")
        print(f"  p(alpha=0.5)={p05_1:.4f}")

    # T2: membrane assumption
    print("\n" + "=" * 70)
    print("T2: Membrane assumption (gc=gc_known)")
    print("=" * 70)
    g_sigma0_memb = []
    for rec in records:
        deep = rec["x_ratio"] < 1.0
        gbar_m = rec["gobs"][deep]**2 / rec["gc_known"]
        g_sigma0_memb.append(np.median(gbar_m))
    g_sigma0_memb = np.array(g_sigma0_memb)
    v2 = (g_sigma0_memb > 0) & np.isfinite(g_sigma0_memb)
    if v2.sum() > 20:
        a2, e2, eta2, r2_2, p05_2 = fit_alpha(log_gc[v2], g_sigma0_memb[v2])
        print(f"  alpha={a2:.3f}+/-{e2:.3f}, R2={r2_2:.3f}")
        print(f"  self-consistent: {'YES' if abs(a2-0.5) < 2*e2 else 'NO'}")

    # T3: gc_deep from gobs^2/gbar
    print("\n" + "=" * 70)
    print("T3: gc_deep = median(gobs^2/gbar) at deep-MOND")
    print("=" * 70)
    gc_deep = []
    for rec in records:
        deep = rec["x_ratio"] < 1.0
        gc_pts = rec["gobs"][deep]**2 / rec["gbar"][deep]
        gc_deep.append(np.median(gc_pts))
    gc_deep = np.array(gc_deep)
    v3 = np.isfinite(gc_deep) & (gc_deep > 0)
    if v3.sum() > 20:
        log_gcd = np.log10(gc_deep[v3])
        rho, p = stats.pearsonr(log_gc[v3], log_gcd)
        scatter = np.std(log_gcd - log_gc[v3])
        ratio = np.median(gc_deep[v3] / gc_known[v3])
        a3, e3, eta3, r2_3, p05_3 = fit_alpha(log_gcd, g_proxy[v3])
        a_ref, e_ref, _, r2_ref, _ = fit_alpha(log_gc[v3], g_proxy[v3])
        print(f"  gc_deep vs gc_known: r={rho:.3f}, scatter={scatter:.3f} dex, "
              f"median ratio={ratio:.3f}")
        print(f"  alpha(gc_deep)={a3:.3f}+/-{e3:.3f}, p(0.5)={p05_3:.3f}")
        print(f"  alpha(gc_known)={a_ref:.3f}+/-{e_ref:.3f} (ref)")

    # T4: depth dependence
    print("\n" + "=" * 70)
    print("T4: Depth-dependent gc precision")
    print("=" * 70)
    x_cuts = [1.0, 0.5, 0.3, 0.1]
    alphas_d, errs_d, ns_d, scatters_d = [], [], [], []
    for xc in x_cuts:
        gc_c = []
        for rec in records:
            d = rec["x_ratio"] < xc
            if d.sum() < 2: gc_c.append(np.nan); continue
            gc_c.append(np.median(rec["gobs"][d]**2 / rec["gbar"][d]))
        gc_c = np.array(gc_c)
        v = np.isfinite(gc_c) & (gc_c > 0) & np.isfinite(log_gc)
        ns_d.append(v.sum())
        if v.sum() > 15:
            sc = np.std(np.log10(gc_c[v]) - log_gc[v])
            scatters_d.append(sc)
            ac, ec, _, _, pc = fit_alpha(np.log10(gc_c[v]), g_proxy[v])
            alphas_d.append(ac); errs_d.append(ec)
            print(f"  x<{xc}: N={v.sum():3d}, scatter={sc:.3f}, "
                  f"alpha={ac:.3f}+/-{ec:.3f}, p(0.5)={pc:.3f}")
        else:
            alphas_d.append(np.nan); errs_d.append(np.nan); scatters_d.append(np.nan)
            print(f"  x<{xc}: N={v.sum():3d} (too few)")

    # T5: independence
    print("\n" + "=" * 70)
    print("T5: Independence test")
    print("=" * 70)
    if v3.sum() > 20:
        rho_i, p_i = stats.pearsonr(np.log10(g_proxy[v3]), log_gcd)
        rho_r, _ = stats.pearsonr(np.log10(g_proxy[v3]), log_gc[v3])
        print(f"  r(gc_deep, vflat^2/hR) = {rho_i:.3f} (p={p_i:.2e})")
        print(f"  r(gc_known, vflat^2/hR) = {rho_r:.3f} (ref)")
        print(f"  NOTE: gc_deep = gobs^2/gbar is noisy estimate of gc_known")
        print(f"        No independent info (deep-MOND identity)")

    # T6: principle
    print("\n" + "=" * 70)
    print("T6: Vobs-only fundamental limitation")
    print("=" * 70)
    print("""  Deep-MOND inversion gc = gobs^2/gbar requires gbar.
  Without Vbar decomposition:
    (a) gc=a0 (MOND assumption) -> circular
    (b) gc free + gbar free -> 1 eq, 2 unknowns
    (c) inner Newtonian fit needs Yd
  => Not truly Vobs-only.

  T1 remains useful: MOND (gc=a0) vs membrane (gc=galaxy-dependent)
  discrimination test.
""")

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    if v1.sum() > 20:
        print(f"  T1 MOND: alpha={a1:.3f}+/-{e1:.3f}, p(a=0)={p0:.2e}")
        if p0 < 0.05:
            print(f"       -> MOND rejected. gc is galaxy-dependent.")
        else:
            print(f"       -> MOND not rejected.")
    if v3.sum() > 20:
        print(f"  T3 gc_deep: r={rho:.3f}, scatter={scatter:.3f}, "
              f"alpha={a3:.3f}+/-{e3:.3f}")
    print(f"""
  All S-2 approaches:
    A: Deep-MOND -> requires gbar (Vbar dependent)
    B: Shape -> r_t/v_t circularity
    C: Photometric -> Yd-alpha degeneracy

  Independent routes remaining:
    1. HSC weak lensing / cluster profiles
    2. Condition-14 epsilon threshold
    3. PROBES/MaNGA for precision vflat+hR
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if v3.sum() > 10:
            ax = axes[0, 0]
            ax.scatter(log_gc[v3], log_gcd, s=10, alpha=0.5, c='steelblue')
            lim = [log_gc[v3].min()-0.2, log_gc[v3].max()+0.2]
            ax.plot(lim, lim, 'k--')
            ax.set_xlabel('log gc_known'); ax.set_ylabel('log gc_deep')
            ax.set_title(f'(a) r={rho:.3f}, sc={scatter:.3f}')

            ax = axes[0, 1]
            lx = np.log10(a0 * g_proxy[v3])
            ax.scatter(lx, log_gcd, s=10, alpha=0.5, c='steelblue', label='gc_deep')
            ax.scatter(lx, log_gc[v3], s=10, alpha=0.3, c='gray', label='gc_known')
            xf = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(xf, np.log10(eta3)+a3*xf, 'r-', lw=2)
            ax.set_xlabel('log(a0*vflat^2/hR)'); ax.set_ylabel('log gc')
            ax.set_title(f'(b) alpha={a3:.3f}+/-{e3:.3f}')
            ax.legend(fontsize=7)

        ax = axes[1, 0]
        valid_idx = [i for i, a in enumerate(alphas_d) if not np.isnan(a)]
        if valid_idx:
            ax.errorbar(valid_idx, [alphas_d[i] for i in valid_idx],
                        yerr=[errs_d[i] for i in valid_idx],
                        fmt='o-', color='steelblue', capsize=5)
        ax.axhline(0.5, color='red', ls='--')
        ax.set_xticks(range(len(x_cuts)))
        ax.set_xticklabels([f'x<{xc}' for xc in x_cuts])
        ax.set_ylabel('alpha'); ax.set_title('(c) depth dependence')

        ax = axes[1, 1]
        valid_idx2 = [i for i, s in enumerate(scatters_d) if not np.isnan(s)]
        if valid_idx2:
            ax.plot(valid_idx2, [scatters_d[i] for i in valid_idx2], 'o-', c='steelblue')
        ax.set_xticks(range(len(x_cuts)))
        ax.set_xticklabels([f'x<{xc}' for xc in x_cuts])
        ax.set_ylabel('scatter (dex)'); ax.set_title('(d) scatter vs depth')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'deep_mond_inversion.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
