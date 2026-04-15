#!/usr/bin/env python3
"""
sparc_sbdisk_photometric.py (TA3+phase1 adapted)
Approach C redesigned: use SBdisk(r) profile from Rotmod col 7.
"""
import os, csv, warnings, glob
import numpy as np
from scipy import optimize, stats
from numpy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10
G_SI = 6.674e-11
kpc_m = 3.0857e19
pc_m = 3.0857e16
Msun = 1.989e30

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

def parse_mrt():
    """Parse SPARC_Lelli2016c.mrt for hR_phot and SBdisk_central."""
    data = {}
    in_data = False; sep = 0
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
                name = p[0]
                data[name] = {
                    'Rdisk': float(p[11]),    # kpc photometric hR
                    'SBdisk': float(p[12]),   # L/pc^2 central SB
                    'MHI': float(p[13]),      # 10^9 Msun
                    'RHI': float(p[14]),      # kpc
                    'Vflat_mrt': float(p[15]),
                }
            except: continue
    return data

def load_rotcurve_full(fpath):
    r, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = \
        [], [], [], [], [], [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split()
            if len(p) >= 7:
                try:
                    r.append(float(p[0])); vobs.append(float(p[1]))
                    errv.append(float(p[2])); vgas.append(float(p[3]))
                    vdisk.append(float(p[4])); vbul.append(float(p[5]))
                    sbdisk.append(float(p[6]))
                    sbbul.append(float(p[7]) if len(p) >= 8 else 0.0)
                except: continue
    return (np.array(r), np.array(vobs), np.array(errv), np.array(vgas),
            np.array(vdisk), np.array(vbul), np.array(sbdisk), np.array(sbbul))

def fit_alpha_yd(log_gc, sigma_star, gas_ratio=None, n_boot=1000):
    if gas_ratio is None: gas_ratio = np.zeros(len(sigma_star))
    def resid(p, ss, gr, lgc):
        Yd = 10**p[2]
        gb = G_SI * Yd * ss * (1+gr) * Msun / pc_m**2
        gb = np.maximum(gb, 1e-20)
        return p[0] + p[1]*np.log10(a0*gb) - lgc
    res = optimize.least_squares(resid, [0, 0.5, np.log10(0.5)],
                                  args=(sigma_star, gas_ratio, log_gc),
                                  bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
    alpha = res.x[1]; eta = 10**res.x[0]; Yd = 10**res.x[2]
    pred = resid(res.x, sigma_star, gas_ratio, log_gc) + log_gc
    R2 = 1 - np.sum((log_gc-pred)**2)/np.sum((log_gc-log_gc.mean())**2)
    sc = np.std(log_gc - pred)
    np.random.seed(42)
    ba, by = [], []
    ng = len(log_gc)
    for _ in range(n_boot):
        idx = np.random.choice(ng, ng, replace=True)
        try:
            rb = optimize.least_squares(resid, res.x,
                args=(sigma_star[idx], gas_ratio[idx], log_gc[idx]),
                bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
            ba.append(rb.x[1]); by.append(10**rb.x[2])
        except: pass
    ae = np.std(ba) if ba else np.nan
    ye = np.std(by) if by else np.nan
    z = (alpha-0.5)/ae if ae > 0 else np.inf
    p05 = 2*stats.norm.sf(abs(z))
    return alpha, ae, Yd, ye, eta, R2, sc, p05

def fit_alpha_simple(log_gc, g_proxy, n_boot=500):
    def resid(p, gp, lgc):
        return p[0] + p[1]*np.log10(a0*gp) - lgc
    res = optimize.least_squares(resid, [0, 0.5], args=(g_proxy, log_gc),
                                  bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha = res.x[1]; eta = 10**res.x[0]
    pred = res.x[0] + res.x[1]*np.log10(a0*g_proxy)
    R2 = 1 - np.sum((log_gc-pred)**2)/np.sum((log_gc-log_gc.mean())**2)
    np.random.seed(42)
    boots = []
    ng = len(log_gc)
    for _ in range(n_boot):
        idx = np.random.choice(ng, ng, replace=True)
        try:
            rb = optimize.least_squares(resid, res.x, args=(g_proxy[idx], log_gc[idx]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            boots.append(rb.x[1])
        except: pass
    err = np.std(boots) if boots else np.nan
    z = (alpha-0.5)/err if err > 0 else np.inf
    p05 = 2*stats.norm.sf(abs(z))
    return alpha, err, eta, R2, p05

def main():
    print("=" * 70)
    print("Approach C v3: SBdisk(r) from Rotmod + MRT photometric hR")
    print("=" * 70)

    pipe = load_pipeline()
    mrt = parse_mrt()
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}, Rotmod: {len(rotmod_files)}")

    # Sample check
    print("\nSample SBdisk profiles:")
    for fpath in rotmod_files[:3]:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        r, _, _, _, _, _, sb, _ = load_rotcurve_full(fpath)
        m = sb > 0
        if m.sum() > 0:
            print(f"  {gname}: SBdisk range [{sb[m].min():.1f}, {sb[m].max():.1f}] L/pc^2, N={m.sum()}")

    records = []
    for fpath in rotmod_files:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        if gname not in pipe: continue
        gd = pipe[gname]
        gc = gd['gc']; vflat = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vflat <= 0: continue

        # Use MRT photometric hR if available, else compute from Vdisk
        if gname in mrt and mrt[gname]['Rdisk'] > 0:
            hR = mrt[gname]['Rdisk']
        else:
            continue  # require photometric hR

        r_kpc, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = load_rotcurve_full(fpath)
        if len(r_kpc) < 5: continue
        mask_sb = sbdisk > 0
        if mask_sb.sum() < 3: continue

        # Central SBdisk (innermost point)
        sb_central = sbdisk[mask_sb][0]
        r_central = r_kpc[mask_sb][0]

        # Exponential fit for Sigma_0
        sb_v = sbdisk[mask_sb]; r_v = r_kpc[mask_sb]
        n_inner = max(4, len(sb_v) // 2)
        try:
            ln_sb = np.log(sb_v[:n_inner])
            slope, intercept, _, _, _ = stats.linregress(r_v[:n_inner], ln_sb)
            sigma0_fit = np.exp(intercept)
            hR_fit = -1.0/slope if slope < 0 else np.nan
        except:
            sigma0_fit = sb_central; hR_fit = np.nan

        # Gas fraction proxy
        vgas_v = np.abs(vgas[mask_sb])
        vdisk_v = np.abs(vdisk[mask_sb])
        n3 = max(2, len(vgas_v)//3)
        vg_in = np.median(vgas_v[:n3])
        vd_in = np.median(vdisk_v[:n3])
        gas_ratio = (vg_in / max(vd_in, 1.0))**2 if vd_in > 0 else 0

        # MRT gas info
        if gname in mrt:
            MHI = mrt[gname]['MHI']; RHI = mrt[gname]['RHI']
            if MHI > 0 and RHI > 0:
                sigma_gas = (MHI * 1.33e9) / (np.pi * (RHI * 1e3)**2)  # Msun/pc^2
            else:
                sigma_gas = 0
        else:
            sigma_gas = 0

        g_obs = (vflat * 1e3)**2 / (hR * kpc_m)

        records.append({
            'name': gname, 'gc': gc, 'vflat': vflat, 'hR': hR, 'Yd': Yd,
            'sb_central': sb_central, 'sigma0_fit': sigma0_fit,
            'hR_fit': hR_fit, 'gas_ratio': gas_ratio, 'sigma_gas': sigma_gas,
            'g_obs': g_obs,
        })

    N = len(records)
    print(f"\nValid galaxies: {N}")

    gc_arr = np.array([r['gc'] for r in records])
    log_gc = np.log10(gc_arr)
    vflat = np.array([r['vflat'] for r in records])
    hR = np.array([r['hR'] for r in records])
    sb_central = np.array([r['sb_central'] for r in records])
    sigma0_fit = np.array([r['sigma0_fit'] for r in records])
    gas_ratio = np.array([r['gas_ratio'] for r in records])
    sigma_gas = np.array([r['sigma_gas'] for r in records])
    g_obs = np.array([r['g_obs'] for r in records])
    Yd_arr = np.array([r['Yd'] for r in records])

    g_sigma_central = G_SI * sb_central * Msun / pc_m**2
    g_sigma_fit = G_SI * sigma0_fit * Msun / pc_m**2

    # T1: central SBdisk + Yd
    print("\n" + "=" * 70)
    print("T1: SBdisk central + Yd simultaneous")
    print("=" * 70)
    v1 = (sb_central > 0) & np.isfinite(sb_central)
    a1, ae1, yd1, yde1, eta1, r2_1, sc1, p1 = fit_alpha_yd(
        log_gc[v1], sb_central[v1], gas_ratio[v1])
    print(f"  alpha={a1:.3f}+/-{ae1:.3f}, Yd={yd1:.3f}+/-{yde1:.3f}")
    print(f"  eta={eta1:.3f}, R2={r2_1:.3f}, scatter={sc1:.3f}, p(0.5)={p1:.4f}")

    # T1b: with MRT gas
    print("\n  T1b: with MRT Sigma_gas:")
    v1b = v1 & (sigma_gas >= 0)
    # Sigma_bar = Yd * SBdisk + sigma_gas [Msun/pc^2]
    def resid_1b(p, sb, sg, lgc):
        Yd = 10**p[2]
        Sigma_bar = Yd * sb + sg
        gb = G_SI * np.maximum(Sigma_bar, 1e-10) * Msun / pc_m**2
        return p[0] + p[1]*np.log10(a0*gb) - lgc
    res1b = optimize.least_squares(resid_1b, [0, 0.5, np.log10(0.5)],
                                     args=(sb_central[v1b], sigma_gas[v1b], log_gc[v1b]),
                                     bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
    a1b = res1b.x[1]; yd1b = 10**res1b.x[2]
    pred1b = resid_1b(res1b.x, sb_central[v1b], sigma_gas[v1b], log_gc[v1b]) + log_gc[v1b]
    R2_1b = 1 - np.sum((log_gc[v1b]-pred1b)**2)/np.sum((log_gc[v1b]-log_gc[v1b].mean())**2)
    print(f"  alpha={a1b:.3f}, Yd={yd1b:.3f}, R2={R2_1b:.3f}")

    # T2: exp fit
    print("\n" + "=" * 70)
    print("T2: Exponential fit Sigma_0 + Yd")
    print("=" * 70)
    v2 = (sigma0_fit > 0) & np.isfinite(sigma0_fit)
    a2, ae2, yd2, yde2, eta2, r2_2, sc2, p2 = fit_alpha_yd(
        log_gc[v2], sigma0_fit[v2], gas_ratio[v2])
    print(f"  alpha={a2:.3f}+/-{ae2:.3f}, Yd={yd2:.3f}+/-{yde2:.3f}")
    print(f"  R2={r2_2:.3f}, scatter={sc2:.3f}, p(0.5)={p2:.4f}")

    # T3: Yd absorbed
    print("\n" + "=" * 70)
    print("T3: Yd absorbed (alpha only)")
    print("=" * 70)
    for label, gs, v in [("SBdisk central", g_sigma_central, v1),
                          ("Exp fit Sigma0", g_sigma_fit, v2),
                          ("Vobs proxy", g_obs, np.ones(N, dtype=bool))]:
        vv = v & (gs > 0) & np.isfinite(gs)
        if vv.sum() > 20:
            a, e, eta, r2, p05 = fit_alpha_simple(log_gc[vv], gs[vv])
            print(f"  {label:25s}: alpha={a:.3f}+/-{e:.3f}, R2={r2:.3f}, p(0.5)={p05:.4f}")

    # T4: independence
    print("\n" + "=" * 70)
    print("T4: Independence")
    print("=" * 70)
    v4 = v1 & (g_obs > 0)
    log_gsb = np.log10(g_sigma_central[v4])
    log_gobs = np.log10(g_obs[v4])
    rho_i, p_i = stats.pearsonr(log_gsb, log_gobs)
    print(f"  r(G*Sigma_SBdisk, vflat^2/hR) = {rho_i:.3f}")

    X = np.column_stack([np.ones(v4.sum()), log_gobs])
    b1, _, _, _ = lstsq(X, log_gc[v4], rcond=None)
    b2, _, _, _ = lstsq(X, log_gsb, rcond=None)
    resid_gc = log_gc[v4] - X @ b1
    resid_sb = log_gsb - X @ b2
    rho_p, p_p = stats.pearsonr(resid_gc, resid_sb)
    print(f"  Partial rho(gc, SBdisk | vflat^2/hR) = {rho_p:.3f} (p={p_p:.2e})")
    if abs(rho_p) > 0.2:
        print(f"  -> SBdisk has independent info beyond vflat^2/hR")
    else:
        print(f"  -> SBdisk is redundant with vflat^2/hR")

    # T5: 50/50
    print("\n" + "=" * 70)
    print("T5: 50/50 split (20x)")
    print("=" * 70)
    gen = {"SBdisk_Yd": [], "Sigma0_Yd": [], "Vobs": []}
    for seed in range(20):
        np.random.seed(seed*13+5)
        idx = np.arange(N); np.random.shuffle(idx)
        half = N // 2; tr = idx[:half]; te = idx[half:]
        try:
            vt = v1[tr]; ve = v1[te]
            if vt.sum() > 15 and ve.sum() > 15:
                ti = tr[vt]; ei = te[ve]
                at, _, ydt, _, _, _, _, _ = fit_alpha_yd(
                    log_gc[ti], sb_central[ti], gas_ratio[ti], n_boot=50)
                gt = G_SI * ydt * sb_central[ei] * (1+gas_ratio[ei]) * Msun / pc_m**2
                gt = np.maximum(gt, 1e-20)
                ate, _, _, _, _ = fit_alpha_simple(log_gc[ei], gt, n_boot=50)
                gen["SBdisk_Yd"].append(ate)
        except: pass
        try:
            vt = v2[tr]; ve = v2[te]
            if vt.sum() > 15 and ve.sum() > 15:
                ti = tr[vt]; ei = te[ve]
                at, _, ydt, _, _, _, _, _ = fit_alpha_yd(
                    log_gc[ti], sigma0_fit[ti], gas_ratio[ti], n_boot=50)
                gt = G_SI * ydt * sigma0_fit[ei] * (1+gas_ratio[ei]) * Msun / pc_m**2
                gt = np.maximum(gt, 1e-20)
                ate, _, _, _, _ = fit_alpha_simple(log_gc[ei], gt, n_boot=50)
                gen["Sigma0_Yd"].append(ate)
        except: pass
        try:
            ate, _, _, _, _ = fit_alpha_simple(log_gc[te], g_obs[te], n_boot=50)
            gen["Vobs"].append(ate)
        except: pass

    print(f"\n  {'model':<15s} {'a_test':>8s} {'std':>8s} {'covers 0.5':>12s}")
    for m in gen:
        if gen[m]:
            at = np.array(gen[m])
            cov = abs(at.mean()-0.5) < 2*at.std()
            print(f"  {m:<15s} {at.mean():>8.3f} {at.std():>8.3f} {'YES' if cov else 'NO':>12s}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  {'model':<35s} {'alpha':>7s} {'err':>7s} {'R2':>6s} {'p(0.5)':>8s}")
    print(f"  {'Vbar prior':<35s} {'0.545':>7s} {'0.041':>7s} {'---':>6s} {'0.273':>8s}")
    print(f"  {'T1 SBdisk_central+Yd':<35s} {a1:>7.3f} {ae1:>7.3f} {r2_1:>6.3f} {p1:>8.4f}")
    print(f"  {'T1b SBdisk+MRT_gas+Yd':<35s} {a1b:>7.3f} {'---':>7s} {R2_1b:>6.3f} {'---':>8s}")
    print(f"  {'T2 ExpFit_Sigma0+Yd':<35s} {a2:>7.3f} {ae2:>7.3f} {r2_2:>6.3f} {p2:>8.4f}")
    print(f"  {'Old C: Vdisk_outer+Yd':<35s} {'0.382':>7s} {'0.085':>7s} {'0.089':>6s} {'0.168':>8s}")
    print(f"\n  Independence: r={rho_i:.3f}, partial={rho_p:.3f}")
    print(f"  T1 Yd={yd1:.3f}+/-{yde1:.3f}")

    if abs(a1 - 0.5) < 2*ae1 and 0.2 < yd1 < 1.0:
        print("\n  Verdict A: alpha~0.5 + physical Yd -> SBdisk photometry WORKS")
    elif abs(a1 - 0.5) < 3*ae1:
        print("\n  Verdict B: alpha~0.5 within 3sigma")
    elif yd1 < 0.15:
        print("\n  Verdict C: Yd at lower bound -> degeneracy persists")
    else:
        print(f"\n  Verdict D: alpha={a1:.3f} systematically off")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        for i, rec in enumerate(records[:3]):
            fp = os.path.join(ROTMOD, f"{rec['name']}_rotmod.dat")
            if os.path.exists(fp):
                r, _, _, _, _, _, sb, _ = load_rotcurve_full(fp)
                m = sb > 0
                if m.sum() > 3:
                    ax.semilogy(r[m], sb[m], 'o-', ms=3, label=rec['name'])
        ax.set_xlabel('r [kpc]'); ax.set_ylabel('SBdisk [L/pc^2]')
        ax.set_title('(a) SBdisk profiles'); ax.legend(fontsize=7)

        if v1.sum() > 20:
            ax = axes[0, 1]
            gb = G_SI * yd1 * sb_central[v1] * (1+gas_ratio[v1]) * Msun / pc_m**2
            gb = np.maximum(gb, 1e-20)
            lx = np.log10(a0 * gb)
            ax.scatter(lx, log_gc[v1], s=10, alpha=0.5, c='steelblue')
            xf = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(xf, np.log10(eta1)+a1*xf, 'r-', lw=2,
                    label=f'a={a1:.3f}, Yd={yd1:.3f}')
            ax.set_xlabel('log(a0*G*Yd*SBdisk)'); ax.set_ylabel('log gc')
            ax.set_title(f'(b) T1 R2={r2_1:.3f}'); ax.legend()

        ax = axes[1, 0]
        if v4.sum() > 10:
            ax.scatter(log_gobs, log_gsb, s=10, alpha=0.5, c='steelblue')
            lim = [min(log_gobs.min(), log_gsb.min())-0.2,
                   max(log_gobs.max(), log_gsb.max())+0.2]
            ax.plot(lim, lim, 'k--')
            ax.set_xlabel('log(vflat^2/hR)'); ax.set_ylabel('log(G*SBdisk)')
            ax.set_title(f'(c) r={rho_i:.3f}')

        ax = axes[1, 1]
        cols = ['steelblue', 'orange', 'gray']
        for i, (m, c) in enumerate(zip(gen, cols)):
            if gen[m]:
                at = np.array(gen[m])
                ax.scatter([i]*len(at), at, c=c, s=30, alpha=0.5)
                ax.errorbar(i, at.mean(), yerr=at.std(), color=c, capsize=5)
        ax.axhline(0.5, color='red', ls='--')
        ax.set_xticks(range(len(gen)))
        ax.set_xticklabels(list(gen.keys()), fontsize=7)
        ax.set_ylabel('alpha test'); ax.set_title('(d) 50/50')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'sbdisk_photometric.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
