#!/usr/bin/env python3
"""
sparc_gc_photometric.py (TA3+phase1 adapted)
Photometric surface density + free Yd simultaneous fit.
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

def main():
    print("=" * 70)
    print("S-2 Approach C: Photometric + free Yd")
    print("=" * 70)
    pipe = load_pipeline()
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Pipeline: {len(pipe)}, Rotmod: {len(rotmod_files)}")

    records = []
    for fpath in rotmod_files:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        if gname not in pipe: continue
        gd = pipe[gname]
        gc_known = gd['gc']; vflat = gd['vflat']; Yd_sps = gd.get('Yd', 0.5)
        if gc_known <= 0 or vflat <= 0: continue
        r, vobs, errv, vgas, vdisk, vbul = load_rotcurve(fpath)
        if len(r) < 5: continue
        hR = compute_hR(r, vdisk, Yd_sps)
        if hR is None: continue

        outer = r > 2 * hR
        if outer.sum() < 3:
            outer = np.zeros(len(r), dtype=bool)
            outer[-min(3, len(r)):] = True
        vdisk_flat = np.median(np.abs(vdisk[outer]))
        vgas_flat = np.median(np.abs(vgas[outer]))
        vbul_flat = np.median(np.abs(vbul[outer]))
        if vdisk_flat <= 0: continue

        g_star = (vdisk_flat * 1e3)**2 / (hR * kpc_m)
        g_gas = (vgas_flat * 1e3)**2 / (hR * kpc_m) if vgas_flat > 0 else 0
        g_bul = (vbul_flat * 1e3)**2 / (hR * kpc_m) if vbul_flat > 0 else 0
        g_obs = (vflat * 1e3)**2 / (hR * kpc_m)

        records.append({"name": gname, "gc_known": gc_known, "vflat": vflat,
                        "hR": hR, "vdisk_flat": vdisk_flat, "vgas_flat": vgas_flat,
                        "vbul_flat": vbul_flat, "g_star": g_star, "g_gas": g_gas,
                        "g_bul": g_bul, "g_obs": g_obs})

    N = len(records)
    print(f"Valid: {N}")
    if N < 30: print("Too few."); return

    gc_known = np.array([r["gc_known"] for r in records])
    log_gc = np.log10(gc_known)
    g_star = np.array([r["g_star"] for r in records])
    g_gas = np.array([r["g_gas"] for r in records])
    g_bul = np.array([r["g_bul"] for r in records])
    g_obs = np.array([r["g_obs"] for r in records])
    vflat = np.array([r["vflat"] for r in records])
    hR = np.array([r["hR"] for r in records])

    print("\n" + "=" * 70)
    print("Method 1: Yd + alpha simultaneous fit")
    print("=" * 70)

    def model_1(params, gs, gg, gb):
        log_eta, alpha, log_Yd = params
        Yd = 10**log_Yd
        g_bar = np.maximum(Yd * gs + gg + Yd * gb, 1e-15)
        return log_eta + alpha * np.log10(a0 * g_bar)

    def resid_1(p, gs, gg, gb, y):
        return model_1(p, gs, gg, gb) - y

    res1 = optimize.least_squares(resid_1, [0, 0.5, np.log10(0.5)],
                                    args=(g_star, g_gas, g_bul, log_gc),
                                    bounds=([-5, 0.01, -1], [5, 2.0, 1.0]))
    eta_1 = 10**res1.x[0]; alpha_1 = res1.x[1]; Yd_1 = 10**res1.x[2]
    pred_1 = model_1(res1.x, g_star, g_gas, g_bul)
    R2_1 = 1 - np.sum((log_gc - pred_1)**2)/np.sum((log_gc - log_gc.mean())**2)
    scatter_1 = np.std(log_gc - pred_1)
    print(f"  alpha={alpha_1:.4f}, Yd={Yd_1:.4f}, eta={eta_1:.4f}")
    print(f"  R2={R2_1:.4f}, scatter={scatter_1:.4f} dex")

    np.random.seed(42)
    boots_1 = {"alpha": [], "Yd": [], "eta": []}
    for _ in range(1000):
        idx = np.random.choice(N, N, replace=True)
        try:
            rb = optimize.least_squares(resid_1, res1.x,
                args=(g_star[idx], g_gas[idx], g_bul[idx], log_gc[idx]),
                bounds=([-5, 0.01, -1], [5, 2.0, 1.0]))
            boots_1["alpha"].append(rb.x[1])
            boots_1["Yd"].append(10**rb.x[2])
            boots_1["eta"].append(10**rb.x[0])
        except: pass
    a_err_1 = np.std(boots_1["alpha"])
    Yd_err_1 = np.std(boots_1["Yd"])
    z1 = (alpha_1 - 0.5) / a_err_1 if a_err_1 > 0 else np.inf
    p05_1 = 2 * stats.norm.sf(abs(z1))
    print(f"  alpha={alpha_1:.3f}+/-{a_err_1:.3f}, p(0.5)={p05_1:.4f}")
    print(f"  Yd={Yd_1:.3f}+/-{Yd_err_1:.3f}")

    print("\n" + "=" * 70)
    print("Method 2: gas-rich/poor split")
    print("=" * 70)
    gas_frac = g_gas / (g_star + g_gas + g_bul + 1e-15)
    med_gf = np.median(gas_frac)
    gas_rich = gas_frac > med_gf
    gas_poor = ~gas_rich
    print(f"  gas fraction median: {med_gf:.3f}")
    for label, mask in [("gas_rich", gas_rich), ("gas_poor", gas_poor)]:
        if mask.sum() < 20: continue
        try:
            res = optimize.least_squares(resid_1, res1.x,
                args=(g_star[mask], g_gas[mask], g_bul[mask], log_gc[mask]),
                bounds=([-5, 0.01, -1], [5, 2.0, 1.0]))
            a_sub = res.x[1]; Yd_sub = 10**res.x[2]
            pred_sub = model_1(res.x, g_star[mask], g_gas[mask], g_bul[mask])
            R2_sub = 1 - np.sum((log_gc[mask]-pred_sub)**2)/np.sum((log_gc[mask]-log_gc[mask].mean())**2)
            print(f"  {label}: alpha={a_sub:.3f}, Yd={Yd_sub:.3f}, R2={R2_sub:.3f}")
        except: pass

    print("\n" + "=" * 70)
    print("Method 3: Vdisk^2/hR only (Yd absorbed, gas ignored)")
    print("=" * 70)

    def model_3(p, gs): return p[0] + p[1]*np.log10(a0*gs)
    def resid_3(p, gs, y): return model_3(p, gs) - y

    res3 = optimize.least_squares(resid_3, [0, 0.5], args=(g_star, log_gc),
                                    bounds=([-5, 0.01], [5, 2.0]))
    alpha_3 = res3.x[1]; eta_3 = 10**res3.x[0]
    pred_3 = model_3(res3.x, g_star)
    R2_3 = 1 - np.sum((log_gc - pred_3)**2)/np.sum((log_gc - log_gc.mean())**2)
    scatter_3 = np.std(log_gc - pred_3)

    np.random.seed(42)
    boots_3 = []
    for _ in range(1000):
        idx = np.random.choice(N, N, replace=True)
        try:
            rb = optimize.least_squares(resid_3, res3.x,
                args=(g_star[idx], log_gc[idx]),
                bounds=([-5, 0.01], [5, 2.0]))
            boots_3.append(rb.x[1])
        except: pass
    a_err_3 = np.std(boots_3)
    z3 = (alpha_3 - 0.5) / a_err_3 if a_err_3 > 0 else np.inf
    p05_3 = 2 * stats.norm.sf(abs(z3))
    print(f"  alpha={alpha_3:.3f}+/-{a_err_3:.3f}, p(0.5)={p05_3:.4f}")
    print(f"  eta'={eta_3:.3f}, R2={R2_3:.3f}, scatter={scatter_3:.3f}")

    print("\n" + "=" * 70)
    print("Method 4: Vobs proxy (reference)")
    print("=" * 70)
    res4 = optimize.least_squares(resid_3, [0, 0.5], args=(g_obs, log_gc),
                                    bounds=([-5, 0.01], [5, 2.0]))
    alpha_4 = res4.x[1]
    pred_4 = model_3(res4.x, g_obs)
    R2_4 = 1 - np.sum((log_gc - pred_4)**2)/np.sum((log_gc - log_gc.mean())**2)
    np.random.seed(42)
    boots_4 = []
    for _ in range(1000):
        idx = np.random.choice(N, N, replace=True)
        try:
            rb = optimize.least_squares(resid_3, res4.x,
                args=(g_obs[idx], log_gc[idx]),
                bounds=([-5, 0.01], [5, 2.0]))
            boots_4.append(rb.x[1])
        except: pass
    a_err_4 = np.std(boots_4)
    print(f"  alpha={alpha_4:.3f}+/-{a_err_4:.3f}, R2={R2_4:.3f}")

    print("\n" + "=" * 70)
    print("Summary comparison")
    print("=" * 70)
    print(f"\n  {'method':<35s} {'alpha':>7s} {'err':>7s} {'R2':>6s} {'scatter':>8s}")
    print(f"  {'Vbar (prior)':<35s} {'0.545':>7s} {'0.041':>7s} {'---':>6s} {'---':>8s}")
    print(f"  {'M1 Yd+alpha+star+gas+bul':<35s} {alpha_1:>7.3f} {a_err_1:>7.3f} "
          f"{R2_1:>6.3f} {scatter_1:>8.3f}")
    print(f"  {'M3 Vdisk^2/hR only':<35s} {alpha_3:>7.3f} {a_err_3:>7.3f} "
          f"{R2_3:>6.3f} {scatter_3:>8.3f}")
    print(f"  {'M4 Vobs proxy':<35s} {alpha_4:>7.3f} {a_err_4:>7.3f} "
          f"{R2_4:>6.3f} {'---':>8s}")

    # 50/50 generalization
    print("\n" + "=" * 70)
    print("50/50 split generalization (20x)")
    print("=" * 70)
    gen = {"M1": {"alpha": [], "Yd": []}, "M3": {"alpha": []}, "M4": {"alpha": []}}
    for seed in range(20):
        np.random.seed(seed*11 + 7)
        idx = np.arange(N); np.random.shuffle(idx)
        half = N // 2
        tr, te = idx[:half], idx[half:]
        try:
            rb = optimize.least_squares(resid_1, res1.x,
                args=(g_star[tr], g_gas[tr], g_bul[tr], log_gc[tr]),
                bounds=([-5, 0.01, -1], [5, 2.0, 1.0]))
            def rfy(p, gs, gg, gb, y, lYd):
                Yd = 10**lYd
                g = np.maximum(Yd*gs + gg + Yd*gb, 1e-15)
                return p[0] + p[1]*np.log10(a0*g) - y
            rb_te = optimize.least_squares(rfy, [rb.x[0], rb.x[1]],
                args=(g_star[te], g_gas[te], g_bul[te], log_gc[te], rb.x[2]),
                bounds=([-5, 0.01], [5, 2.0]))
            gen["M1"]["alpha"].append(rb_te.x[1])
            gen["M1"]["Yd"].append(10**rb.x[2])
        except: pass
        try:
            rb = optimize.least_squares(resid_3, res3.x, args=(g_star[tr], log_gc[tr]),
                bounds=([-5, 0.01], [5, 2.0]))
            rb_te = optimize.least_squares(resid_3, rb.x, args=(g_star[te], log_gc[te]),
                bounds=([-5, 0.01], [5, 2.0]))
            gen["M3"]["alpha"].append(rb_te.x[1])
        except: pass
        try:
            rb = optimize.least_squares(resid_3, res4.x, args=(g_obs[tr], log_gc[tr]),
                bounds=([-5, 0.01], [5, 2.0]))
            rb_te = optimize.least_squares(resid_3, rb.x, args=(g_obs[te], log_gc[te]),
                bounds=([-5, 0.01], [5, 2.0]))
            gen["M4"]["alpha"].append(rb_te.x[1])
        except: pass

    print(f"\n  {'method':<10s} {'a_mean':>8s} {'a_std':>8s} {'covers 0.5':>12s}")
    for mname in gen:
        if gen[mname]["alpha"]:
            at = np.array(gen[mname]["alpha"])
            cov = abs(at.mean() - 0.5) < 2*at.std()
            print(f"  {mname:<10s} {at.mean():>8.3f} {at.std():>8.3f} "
                  f"{'YES' if cov else 'NO':>12s}")
            if "Yd" in gen[mname] and gen[mname]["Yd"]:
                yd = np.array(gen[mname]["Yd"])
                print(f"  {' '*10} Yd: {yd.mean():.3f}+/-{yd.std():.3f}")

    print("\n" + "=" * 70)
    print("Independence: g_star vs g_obs")
    print("=" * 70)
    log_gs = np.log10(g_star); log_go = np.log10(g_obs)
    rho, p = stats.pearsonr(log_gs, log_go)
    print(f"  Pearson r = {rho:.4f} (p={p:.2e})")
    if rho > 0.95:
        print("  WARNING: strong correlation - g_star ~ g_obs")
    elif rho > 0.8:
        print("  Moderate correlation - partial independence")
    else:
        print("  OK: weak correlation")

    vd_flat_arr = np.array([r["vdisk_flat"] for r in records])
    ratio = vd_flat_arr / vflat
    print(f"  Vdisk_flat/Vobs_flat: median={np.median(ratio):.3f}, "
          f"std={np.std(ratio):.3f}, range=[{np.min(ratio):.3f},{np.max(ratio):.3f}]")

    # Summary
    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)
    print(f"\n  M1: alpha={alpha_1:.3f}+/-{a_err_1:.3f}, Yd={Yd_1:.3f}, p(0.5)={p05_1:.4f}")
    print(f"  M3: alpha={alpha_3:.3f}+/-{a_err_3:.3f}, p(0.5)={p05_3:.4f}")
    print(f"  M4: alpha={alpha_4:.3f}+/-{a_err_4:.3f}")
    print(f"  r(g_star, g_obs) = {rho:.3f}")
    print(f"  Vdisk/Vobs ratio = {np.median(ratio):.3f}")

    if p05_1 > 0.05 and abs(alpha_1 - 0.5) < 2*a_err_1:
        if rho < 0.95:
            print("\n  A: alpha=0.5 not rejected AND g_star partially independent")
            print("     -> Photometric + Yd fit VIABLE for PROBES/MaNGA")
        else:
            print("\n  B: alpha=0.5 not rejected BUT g_star ~ g_obs")
            print("     -> Vbar-free but not Vobs-free. Limited independence")
    elif p05_1 > 0.01:
        print("\n  C: alpha=0.5 weakly supported")
    else:
        print("\n  D: alpha=0.5 rejected tendency")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        lx = np.log10(a0 * (Yd_1*g_star + g_gas + Yd_1*g_bul))
        ax.scatter(lx, log_gc, s=10, alpha=0.5, c='steelblue')
        xf = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xf, res1.x[0] + alpha_1*xf, 'r-', lw=2,
                label=f"a={alpha_1:.3f}+/-{a_err_1:.3f}\nYd={Yd_1:.3f}")
        ax.set_xlabel('log(a0*G*Sigma_bar)'); ax.set_ylabel('log gc')
        ax.set_title('(a) M1 fit')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.hist(boots_1["alpha"], bins=40, alpha=0.5, color='steelblue', label='M1')
        ax.hist(boots_3, bins=40, alpha=0.5, color='orange', label='M3')
        ax.axvline(0.5, color='red', ls='--')
        ax.axvline(0.545, color='blue', ls=':', label='Vbar 0.545')
        ax.set_xlabel('alpha'); ax.set_title('(b) bootstrap')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.scatter(log_go, log_gs, s=10, alpha=0.5, c='steelblue')
        lim = [min(log_go.min(), log_gs.min())-0.1, max(log_go.max(), log_gs.max())+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('log(Vobs^2/hR)'); ax.set_ylabel('log(Vdisk^2/hR)')
        ax.set_title(f'(c) r={rho:.3f}')

        ax = axes[1, 1]
        cols = ['steelblue', 'orange', 'gray']
        for i, (m, c) in enumerate(zip(gen, cols)):
            if gen[m]["alpha"]:
                at = np.array(gen[m]["alpha"])
                ax.scatter([i]*len(at), at, c=c, s=30, alpha=0.5)
                ax.errorbar(i, at.mean(), yerr=at.std(), color=c, capsize=5)
        ax.axhline(0.5, color='red', ls='--')
        ax.axhline(0.545, color='blue', ls=':')
        ax.set_xticks(range(len(gen)))
        ax.set_xticklabels(list(gen.keys()))
        ax.set_ylabel('alpha')
        ax.set_title('(d) 50/50 gen')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'gc_photometric.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
