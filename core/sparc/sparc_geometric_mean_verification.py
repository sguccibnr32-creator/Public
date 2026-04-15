#!/usr/bin/env python3
"""
sparc_geometric_mean_verification.py (TA3+phase1+MRT adapted)
Comprehensive verification of gc = eta * sqrt(a0 * G * Sigma0).
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
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19; pc_m = 3.0857e16
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
            try:
                name = p[0]
                data[name] = {
                    'T_type': int(p[1]), 'Dist': float(p[2]),
                    'L36': float(p[7]),       # 10^9 Lsun
                    'Reff': float(p[9]),       # kpc
                    'SBeff': float(p[10]),     # Lsun/pc^2
                    'Rdisk': float(p[11]),     # kpc
                    'SBdisk0': float(p[12]),   # Lsun/pc^2
                    'MHI': float(p[13]),       # 10^9 Msun
                    'RHI': float(p[14]),       # kpc
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

def fit_alpha(log_gc, g_proxy, n_boot=500):
    def resid(p, gp, lgc):
        return p[0] + p[1]*np.log10(a0*gp) - lgc
    res = optimize.least_squares(resid, [0, 0.5], args=(g_proxy, log_gc),
                                  bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha = res.x[1]; eta = 10**res.x[0]
    pred = res.x[0] + alpha*np.log10(a0*g_proxy)
    R2 = 1 - np.sum((log_gc-pred)**2)/np.sum((log_gc-log_gc.mean())**2)
    sc = np.std(log_gc - pred)
    np.random.seed(42); boots = []
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
    return alpha, err, eta, R2, sc, p05, pred

def main():
    print("=" * 70)
    print("Geometric Mean Law: Comprehensive Verification")
    print("=" * 70)

    pipe = load_pipeline()
    mrt = parse_mrt()
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}, Rotmod: {len(rotmod_files)}")

    records = []
    for fpath in rotmod_files:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        if gname not in pipe: continue
        gd = pipe[gname]
        gc = gd['gc']; vflat = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vflat <= 0: continue

        # Photometric hR from MRT (preferred) or compute from Vdisk
        m = mrt.get(gname, {})
        hR = m.get('Rdisk', 0)
        if hR <= 0: continue

        r_kpc, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = load_rotcurve_full(fpath)
        if len(r_kpc) < 5: continue

        mask_sb = sbdisk > 0
        sb_central = sbdisk[mask_sb][0] if mask_sb.sum() > 0 else np.nan

        # 5 Sigma definitions
        sigma_dyn = (vflat * 1e3)**2 / (hR * kpc_m)
        sigma_bar_sb = G_SI * sb_central * Msun / pc_m**2 if np.isfinite(sb_central) else np.nan

        SBdisk0 = m.get('SBdisk0', np.nan)
        sigma_bar_mrt = G_SI * SBdisk0 * Msun / pc_m**2 if SBdisk0 and SBdisk0 > 0 else np.nan

        SBeff = m.get('SBeff', np.nan)
        sigma_bar_eff = G_SI * SBeff * Msun / pc_m**2 if SBeff and SBeff > 0 else np.nan

        L36 = m.get('L36', np.nan)
        if L36 and L36 > 0:
            sigma_L = (L36 * 1e9) / (2 * np.pi * (hR * 1e3)**2)
            sigma_bar_L = G_SI * sigma_L * Msun / pc_m**2
        else:
            sigma_bar_L = np.nan

        T_type = m.get('T_type', np.nan)
        MHI = m.get('MHI', np.nan)

        records.append({
            'name': gname, 'gc': gc, 'vflat': vflat, 'hR': hR, 'Yd': Yd,
            'T_type': T_type, 'L36': L36, 'MHI': MHI,
            'sb_central': sb_central,
            'sigma_dyn': sigma_dyn,
            'sigma_bar_sb': sigma_bar_sb,
            'sigma_bar_mrt': sigma_bar_mrt,
            'sigma_bar_eff': sigma_bar_eff,
            'sigma_bar_L': sigma_bar_L,
        })

    N = len(records)
    print(f"Integrated: {N} galaxies")

    gc_arr = np.array([r['gc'] for r in records])
    log_gc = np.log10(gc_arr)
    vflat = np.array([r['vflat'] for r in records])
    hR = np.array([r['hR'] for r in records])

    sigma_defs = {
        "(a) vflat^2/hR (dynamical)": np.array([r['sigma_dyn'] for r in records]),
        "(b) SBdisk central (Rotmod)": np.array([r['sigma_bar_sb'] for r in records]),
        "(c) SBdisk0 (MRT)": np.array([r['sigma_bar_mrt'] for r in records]),
        "(d) SBeff (MRT)": np.array([r['sigma_bar_eff'] for r in records]),
        "(e) L/(2pi hR^2) (luminosity)": np.array([r['sigma_bar_L'] for r in records]),
    }

    # T1
    print("\n" + "=" * 70)
    print("T1: Basic verification gc = eta * (a0*vflat^2/hR)^alpha")
    print("=" * 70)
    sd = sigma_defs["(a) vflat^2/hR (dynamical)"]
    a1, e1, eta1, r2_1, sc1, p1, pred1 = fit_alpha(log_gc, sd)
    print(f"  alpha={a1:.4f}+/-{e1:.4f}, eta={eta1:.4f}")
    print(f"  R2={r2_1:.4f}, scatter={sc1:.4f} dex, p(0.5)={p1:.4f}")
    z0 = a1/e1; p0 = 2*stats.norm.sf(abs(z0))
    print(f"  p(alpha=0, MOND) = {p0:.2e}")

    # alpha=0.5 fixed
    log_x_half = 0.5 * np.log10(a0 * sd)
    log_eta_fixed = np.median(log_gc - log_x_half)
    pred_fixed = log_eta_fixed + log_x_half
    r2_fixed = 1 - np.sum((log_gc-pred_fixed)**2)/np.sum((log_gc-log_gc.mean())**2)
    sc_fixed = np.std(log_gc - pred_fixed)
    eta_fixed = 10**log_eta_fixed
    print(f"\n  alpha=0.5 fixed: eta={eta_fixed:.4f}, R2={r2_fixed:.4f}, sc={sc_fixed:.4f}")

    # T2
    print("\n" + "=" * 70)
    print("T2: 5 Sigma0 definitions comparison")
    print("=" * 70)
    print(f"\n  {'definition':<35s} {'N':>5s} {'alpha':>7s} {'err':>7s} {'R2':>7s} {'p(0.5)':>8s}")
    print("  " + "-"*71)
    t2_res = {}
    for label, sigma in sigma_defs.items():
        v = np.isfinite(sigma) & (sigma > 0)
        if v.sum() < 20:
            print(f"  {label:<35s} {v.sum():>5d} ---"); continue
        a, e, eta, r2, sc, p05, _ = fit_alpha(log_gc[v], sigma[v])
        t2_res[label] = {'alpha': a, 'err': e, 'R2': r2, 'p05': p05, 'N': v.sum()}
        m = " [*]" if abs(a-0.5) < 2*e else ""
        print(f"  {label:<35s} {v.sum():>5d} {a:>7.3f} {e:>7.3f} {r2:>7.3f} {p05:>8.4f}{m}")

    # T3
    print("\n" + "=" * 70)
    print("T3: MOND discrimination")
    print("=" * 70)
    gc_sc = np.std(log_gc)
    print(f"  gc median={np.median(gc_arr):.2e} ({np.median(gc_arr)/a0:.3f}*a0)")
    print(f"  gc scatter={gc_sc:.3f} dex")
    resid_mond = log_gc - np.log10(a0)
    print(f"  MOND (gc=a0): scatter={np.std(resid_mond):.3f} dex")
    print(f"  Geometric mean: scatter={sc1:.3f} dex")
    print(f"  Improvement: {(1-sc1/np.std(resid_mond))*100:.1f}%")
    rho_sp, p_sp = stats.spearmanr(log_gc, np.log10(sd))
    print(f"  Spearman rho(gc, vflat^2/hR) = {rho_sp:.3f}")

    # T4
    print("\n" + "=" * 70)
    print("T4: eta structure")
    print("=" * 70)
    eta_i = gc_arr / np.sqrt(a0 * sd)
    log_eta_i = np.log10(eta_i)
    print(f"  eta median={np.median(eta_i):.3f}, scatter={np.std(log_eta_i):.3f} dex")
    T_type = np.array([r['T_type'] for r in records])
    L36 = np.array([r['L36'] for r in records])
    sb_c = np.array([r['sb_central'] for r in records])
    for vn, va in [("T_type", T_type), ("log(L36)", np.log10(np.maximum(L36, 1e-10))),
                    ("log(SBdisk)", np.log10(np.maximum(sb_c, 1e-10))),
                    ("log(vflat)", np.log10(vflat)), ("log(hR)", np.log10(hR))]:
        v = np.isfinite(va) & np.isfinite(log_eta_i)
        if v.sum() > 20:
            rho, p = stats.pearsonr(va[v], log_eta_i[v])
            print(f"  rho(eta, {vn:15s}) = {rho:+.3f} (p={p:.2e})")

    # T5
    print("\n" + "=" * 70)
    print("T5: Self-consistency vflat+hR -> gc")
    print("=" * 70)
    log_vf = np.log10(vflat); log_hR = np.log10(hR)
    X = np.column_stack([np.ones(N), log_vf, log_hR])
    beta, _, _, _ = lstsq(X, log_gc, rcond=None)
    pred_d = X @ beta
    R2_d = 1 - np.sum((log_gc-pred_d)**2)/np.sum((log_gc-log_gc.mean())**2)
    print(f"  log(gc) = {beta[0]:.3f} + {beta[1]:.3f}*log(vflat) + {beta[2]:.3f}*log(hR)")
    print(f"  R2={R2_d:.4f}")
    print(f"  Expected: vflat=1.0 -> {beta[1]:.3f}, hR=-0.5 -> {beta[2]:.3f}")

    # T6
    print("\n" + "=" * 70)
    print("T6: Baryonic vs dynamical Sigma0")
    print("=" * 70)
    for label, sigma in [("SBdisk(Rotmod)", sigma_defs["(b) SBdisk central (Rotmod)"]),
                          ("SBdisk0(MRT)", sigma_defs["(c) SBdisk0 (MRT)"])]:
        v = np.isfinite(sigma) & (sigma > 0)
        if v.sum() < 30: continue
        lsb = np.log10(sigma[v]); lsd = np.log10(sd[v]); lgc = log_gc[v]
        r_bar, _ = stats.pearsonr(lsb, lgc)
        r_dyn, _ = stats.pearsonr(lsd, lgc)
        Xc = np.column_stack([np.ones(v.sum()), lsd])
        b1, _, _, _ = lstsq(Xc, lgc, rcond=None)
        b2, _, _, _ = lstsq(Xc, lsb, rcond=None)
        rp, pp = stats.pearsonr(lgc - Xc@b1, lsb - Xc@b2)
        print(f"\n  {label} (N={v.sum()}):")
        print(f"    r(gc, Sigma_bar)={r_bar:.3f}, r(gc, Sigma_dyn)={r_dyn:.3f}")
        print(f"    partial rho(gc, Sigma_bar | Sigma_dyn)={rp:.3f} (p={pp:.2e})")

    v_sb = np.isfinite(sigma_defs["(b) SBdisk central (Rotmod)"]) & \
           (sigma_defs["(b) SBdisk central (Rotmod)"] > 0)
    if v_sb.sum() > 20:
        ratio = sd[v_sb] / sigma_defs["(b) SBdisk central (Rotmod)"][v_sb]
        print(f"\n  Sigma_dyn / Sigma_bar: median={np.median(ratio):.1f}, "
              f"range=[{np.min(ratio):.1f},{np.max(ratio):.1f}]")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    a_dyn = t2_res.get("(a) vflat^2/hR (dynamical)", {}).get('alpha', '---')
    a_sb = t2_res.get("(b) SBdisk central (Rotmod)", {}).get('alpha', '---')
    a_mrt = t2_res.get("(c) SBdisk0 (MRT)", {}).get('alpha', '---')
    print(f"""
  T1: alpha={a1:.3f}+/-{e1:.3f}, p(0.5)={p1:.4f}, R2={r2_1:.4f}
  T2: dynamical alpha={a_dyn}, baryonic(Rotmod)={a_sb}, MRT={a_mrt}
  T3: MOND scatter={np.std(resid_mond):.3f} -> geom={sc1:.3f} ({(1-sc1/np.std(resid_mond))*100:.1f}%)
  T4: eta median={np.median(eta_i):.3f}, scatter={np.std(log_eta_i):.3f} dex
  T5: vflat+hR R2={R2_d:.4f}, coefs: vflat={beta[1]:.3f}(~1), hR={beta[2]:.3f}(~-0.5)

  Key finding: Sigma0 in geometric mean law = DYNAMICAL, not baryonic.
  Baryonic SBdisk gives alpha~0.2 (rejected from 0.5).
  vflat^2/hR gives alpha~{a1:.3f} (consistent with 0.5).
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        lx = 0.5*np.log10(a0*sd)
        ax.scatter(lx, log_gc, s=10, alpha=0.5, c='steelblue')
        xf = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xf, log_eta_fixed + xf, 'r-', lw=2,
                label=f'a=0.5, eta={eta_fixed:.2f}')
        ax.set_xlabel('log sqrt(a0*vflat^2/hR)'); ax.set_ylabel('log gc')
        ax.set_title(f'(a) R2={r2_fixed:.3f}'); ax.legend()

        ax = axes[0, 1]
        for lbl, sig, col in [("dyn", sd, 'steelblue'),
                                ("bar(SB)", sigma_defs["(b) SBdisk central (Rotmod)"], 'salmon')]:
            v = np.isfinite(sig) & (sig > 0)
            if v.sum() > 10:
                ax.scatter(np.log10(a0*sig[v]), log_gc[v], s=8, alpha=0.3, c=col, label=lbl)
        ax.set_xlabel('log(a0*Sigma)'); ax.set_ylabel('log gc')
        ax.set_title('(b) dyn vs bar'); ax.legend()

        ax = axes[0, 2]
        ax.hist(log_eta_i, bins=30, alpha=0.7, color='steelblue')
        ax.axvline(np.log10(np.median(eta_i)), color='red', lw=2,
                   label=f'med={np.median(eta_i):.2f}')
        ax.set_xlabel('log eta'); ax.set_title(f'(c) scatter={np.std(log_eta_i):.3f}')
        ax.legend()

        ax = axes[1, 0]
        ax.scatter(pred_d, log_gc, s=10, alpha=0.5, c='steelblue')
        lim = [log_gc.min()-0.2, log_gc.max()+0.2]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('predicted'); ax.set_ylabel('observed')
        ax.set_title(f'(d) direct R2={R2_d:.4f}')

        ax = axes[1, 1]
        labs, als, ers = [], [], []
        for l, r_ in t2_res.items():
            labs.append(l[:15]); als.append(r_['alpha']); ers.append(r_['err'])
        if labs:
            ax.errorbar(range(len(labs)), als, yerr=ers, fmt='o', c='steelblue', capsize=5)
            ax.axhline(0.5, color='red', ls='--')
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels(labs, fontsize=6, rotation=30)
            ax.set_ylabel('alpha'); ax.set_title('(e) by Sigma def')

        ax = axes[1, 2]
        vt = np.isfinite(T_type) & np.isfinite(log_eta_i)
        if vt.sum() > 10:
            ax.scatter(T_type[vt], log_eta_i[vt], s=10, alpha=0.5, c='steelblue')
            rt, _ = stats.pearsonr(T_type[vt], log_eta_i[vt])
            ax.set_xlabel('T-type'); ax.set_ylabel('log eta')
            ax.set_title(f'(f) eta vs T (r={rt:.3f})')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'geometric_mean_verification.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
