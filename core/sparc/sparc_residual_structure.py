#!/usr/bin/env python3
"""
sparc_residual_structure.py (TA3+phase1+MRT adapted)
Residual structure analysis of geometric mean law R^2~0.55.
"""
import os, csv, warnings, glob
import numpy as np
from scipy import stats, optimize
from scipy.stats import spearmanr, pearsonr, shapiro, linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.0857e19

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
            try:
                data[p[0]] = {
                    'T': int(p[1]), 'D': float(p[2]), 'Inc': float(p[5]),
                    'L': float(p[7]), 'Reff': float(p[9]), 'SBeff': float(p[10]),
                    'Rdisk': float(p[11]), 'SBdisk0': float(p[12]),
                    'MHI': float(p[13]), 'RHI': float(p[14]),
                    'Vf': float(p[15]), 'Q': int(p[17]),
                }
            except: continue
    return data

def compute_hR_pipe(gname, Yd):
    fp = os.path.join(ROTMOD, f"{gname}_rotmod.dat")
    if not os.path.exists(fp): return None
    r, vdisk = [], []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            pp = line.split()
            if len(pp) >= 6:
                try: r.append(float(pp[0])); vdisk.append(float(pp[4]))
                except: continue
    if len(r) < 5: return None
    r = np.array(r); vdisk = np.array(vdisk)
    vds = np.sqrt(max(Yd, 0.01)) * np.abs(vdisk)
    rpk = r[np.argmax(vds)]
    if rpk < 0.01 or rpk >= r.max() * 0.9: return None
    return rpk / 2.15

def partial_corr(x, y, z):
    if z.ndim == 1: z = z.reshape(-1, 1)
    Z = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(x - Z@bx, y - Z@by)

def main():
    print("=" * 70)
    print("Residual structure of geometric mean law (R^2~0.55)")
    print("=" * 70)

    pipe = load_pipeline()
    mrt = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}")

    # Build dataset with both hR definitions
    names, gc_a, vf_a, hR_p_a, hR_m_a, Yd_a = [], [], [], [], [], []
    mrt_props = {}  # name -> mrt dict
    for n in sorted(pipe.keys()):
        gd = pipe[n]; gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vf <= 0: continue
        m = mrt.get(n, {})
        hR_m = m.get('Rdisk', 0)
        if hR_m <= 0: continue
        hR_p = compute_hR_pipe(n, Yd)
        if hR_p is None: continue
        names.append(n); gc_a.append(gc); vf_a.append(vf)
        hR_p_a.append(hR_p); hR_m_a.append(hR_m); Yd_a.append(Yd)
        if n in mrt: mrt_props[n] = mrt[n]

    gc = np.array(gc_a); vf = np.array(vf_a)
    hR_p = np.array(hR_p_a); hR_m = np.array(hR_m_a); Yd = np.array(Yd_a)
    N = len(gc); log_gc = np.log10(gc)
    log_vf = np.log10(vf); log_hRp = np.log10(hR_p); log_hRm = np.log10(hR_m)
    print(f"Matched: {N}")

    # Use MRT hR for baseline (more standard)
    Sd_m = (vf*1e3)**2 / (hR_m*kpc_m)
    log_Sd = np.log10(Sd_m)
    sl, ic, r_val, p_val, se = linregress(log_Sd, log_gc)
    pred = ic + sl*log_Sd
    resid = log_gc - pred
    scatter = np.std(resid); R2 = r_val**2
    print(f"\nBaseline (MRT hR): alpha={sl:.4f}+/-{se:.4f}, R2={R2:.4f}, scatter={scatter:.4f}")

    # Also pipeline hR
    Sd_p = (vf*1e3)**2 / (hR_p*kpc_m)
    sl_p, ic_p, r_p, _, se_p = linregress(np.log10(Sd_p), log_gc)
    print(f"Pipeline hR:       alpha={sl_p:.4f}+/-{se_p:.4f}, R2={r_p**2:.4f}")

    # T1: residual diagnostics
    print("\n" + "=" * 70)
    print("T1: Residual diagnostics")
    print("=" * 70)
    sw, sw_p = shapiro(resid)
    skew = stats.skew(resid); kurt = stats.kurtosis(resid)
    print(f"  Shapiro-Wilk: W={sw:.4f}, p={sw_p:.4e}")
    print(f"  Skewness={skew:.3f}, Kurtosis={kurt:.3f}")
    r_het, p_het = spearmanr(pred, np.abs(resid))
    print(f"  Heteroscedasticity: rho(pred,|resid|)={r_het:.3f}, p={p_het:.3e}")
    r_gc, p_gc = spearmanr(log_gc, resid)
    r_vf, p_vf = spearmanr(log_vf, resid)
    r_hr, p_hr = spearmanr(log_hRm, resid)
    print(f"  rho(gc, resid)   = {r_gc:+.3f} (p={p_gc:.3e})")
    print(f"  rho(vflat, resid)= {r_vf:+.3f} (p={p_vf:.3e})")
    print(f"  rho(hR, resid)   = {r_hr:+.3f} (p={p_hr:.3e})")

    # T2: residual vs MRT properties
    print("\n" + "=" * 70)
    print("T2: Residual vs MRT galaxy properties")
    print("=" * 70)
    prop_names = ['T', 'D', 'Inc', 'L', 'SBeff', 'SBdisk0', 'MHI', 'RHI', 'Vf', 'Q']
    print(f"\n  {'prop':<12s} {'N':>4s} {'rho':>7s} {'p':>10s} {'r_part':>8s} {'p_part':>10s}")
    sig_props = []
    for prop in prop_names:
        vals, idx = [], []
        for i, n in enumerate(names):
            if n in mrt_props and prop in mrt_props[n]:
                v = mrt_props[n][prop]
                if np.isfinite(v) and v > -90:
                    if prop in ('L', 'MHI', 'RHI', 'D', 'SBeff', 'SBdisk0') and v <= 0: continue
                    vals.append(np.log10(v) if prop in ('L', 'MHI', 'RHI', 'D', 'SBeff', 'SBdisk0') else v)
                    idx.append(i)
        if len(vals) < 20: continue
        vals = np.array(vals); idx = np.array(idx)
        rho, p_s = spearmanr(vals, resid[idx])
        try: rp, pp = partial_corr(resid[idx], vals, log_Sd[idx])
        except: rp, pp = np.nan, np.nan
        sig = " ***" if p_s < 0.001 else (" **" if p_s < 0.01 else (" *" if p_s < 0.05 else ""))
        print(f"  {prop:<12s} {len(vals):>4d} {rho:>+7.3f} {p_s:>10.3e} {rp:>+8.3f} {pp:>10.3e}{sig}")
        if p_s < 0.05: sig_props.append((prop, rho, p_s, rp, pp))

    # T3: per-galaxy eta
    print("\n" + "=" * 70)
    print("T3: Per-galaxy eta distribution")
    print("=" * 70)
    eta = gc / np.sqrt(a0 * Sd_m)
    log_eta = np.log10(eta)
    print(f"  eta (alpha=0.5): median={np.median(eta):.4f}, geom_mean={10**np.mean(log_eta):.4f}")
    print(f"  scatter(log eta) = {np.std(log_eta):.4f} dex")
    print(f"\n  eta vs basic quantities:")
    for vn, va in [("vflat", log_vf), ("hR(MRT)", log_hRm), ("Yd", np.log10(Yd))]:
        rho, p = spearmanr(va, log_eta)
        print(f"    {vn:10s}: rho={rho:+.3f} (p={p:.3e})")

    print(f"\n  eta vs MRT properties:")
    for prop in ['T', 'L', 'MHI', 'Inc', 'Q', 'SBdisk0']:
        vals, idx = [], []
        for i, n in enumerate(names):
            if n in mrt_props and prop in mrt_props[n]:
                v = mrt_props[n][prop]
                if np.isfinite(v) and v > -90:
                    if prop in ('L', 'MHI', 'SBdisk0') and v <= 0: continue
                    vals.append(np.log10(v) if prop in ('L', 'MHI', 'SBdisk0') else v)
                    idx.append(i)
        if len(vals) < 20: continue
        rho, p = spearmanr(np.array(vals), log_eta[np.array(idx)])
        sig = " *" if p < 0.05 else (" **" if p < 0.01 else "")
        print(f"    {prop:10s}: rho={rho:+.3f} (p={p:.3e}){sig}")

    # T4: subgroup fits
    print("\n" + "=" * 70)
    print("T4: Subgroup fits")
    print("=" * 70)
    T_arr = np.array([mrt_props.get(n, {}).get('T', np.nan) for n in names])
    v_T = np.isfinite(T_arr)
    if v_T.sum() > 30:
        print(f"\n  T-type subgroups:")
        print(f"  {'group':<18s} {'N':>4s} {'alpha':>10s} {'R2':>7s} {'sc':>7s} {'<eta>':>7s}")
        for label, mask in [("Early T<=3", T_arr[v_T]<=3), ("Mid 3<T<=6", (T_arr[v_T]>3)&(T_arr[v_T]<=6)),
                             ("Late T>6", T_arr[v_T]>6)]:
            idx_g = np.where(v_T)[0][mask]
            if len(idx_g) < 10: print(f"  {label:<18s} {len(idx_g):4d} (few)"); continue
            s, i_, r_, _, se_ = linregress(log_Sd[idx_g], log_gc[idx_g])
            res_ = log_gc[idx_g] - (i_ + s*log_Sd[idx_g])
            print(f"  {label:<18s} {len(idx_g):4d} {s:7.3f}+/-{se_:.3f} {r_**2:7.3f} {np.std(res_):7.3f} {np.median(eta[idx_g]):7.3f}")

    # vflat split
    vf_med = np.median(log_vf)
    print(f"\n  vflat split (median={10**vf_med:.1f} km/s):")
    for label, mask in [("Low vflat", log_vf < vf_med), ("High vflat", log_vf >= vf_med)]:
        s, i_, r_, _, se_ = linregress(log_Sd[mask], log_gc[mask])
        res_ = log_gc[mask] - (i_ + s*log_Sd[mask])
        print(f"  {label:<18s} {mask.sum():4d} {s:7.3f}+/-{se_:.3f} {r_**2:7.3f} {np.std(res_):7.3f}")

    # T5: multivariate improvement
    print("\n" + "=" * 70)
    print("T5: Multivariate improvement")
    print("=" * 70)
    full_props = ['T', 'L', 'Inc', 'MHI', 'Q']
    rows = []
    for i, n in enumerate(names):
        if n not in mrt_props: continue
        m = mrt_props[n]
        row = [log_gc[i], log_Sd[i], log_vf[i], log_hRm[i]]
        ok = True
        for prop in full_props:
            if prop not in m: ok = False; break
            v = m[prop]
            if prop in ('L', 'MHI'):
                if v <= 0: ok = False; break
                row.append(np.log10(v))
            else:
                row.append(v)
        if ok: rows.append(row)
    if len(rows) > 30:
        D = np.array(rows); y = D[:, 0]; Nf = len(y)
        SS_tot = np.sum((y - y.mean())**2)
        # Baseline
        X0 = np.column_stack([np.ones(Nf), D[:, 1]])
        p0 = np.linalg.lstsq(X0, y, rcond=None)[0]
        R2_0 = 1 - np.sum((y - X0@p0)**2)/SS_tot
        print(f"  Baseline (Sd only): R2={R2_0:.4f}, N={Nf}")
        # +vflat+hR
        X2 = np.column_stack([np.ones(Nf), D[:, 2], D[:, 3]])
        p2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        R2_2 = 1 - np.sum((y - X2@p2)**2)/SS_tot
        print(f"  +vflat+hR: R2={R2_2:.4f}")
        # +all
        Xa = np.column_stack([np.ones(Nf), D[:, 1:]])
        pa = np.linalg.lstsq(Xa, y, rcond=None)[0]
        R2_a = 1 - np.sum((y - Xa@pa)**2)/SS_tot
        sc_a = np.std(y - Xa@pa)
        print(f"  +all MRT ({'+'.join(['Sd','vf','hR']+full_props)}): R2={R2_a:.4f}, sc={sc_a:.4f}")
        vn = ['const', 'Sd', 'vf', 'hR'] + full_props
        for vi, bi in zip(vn, pa):
            print(f"    {vi:8s}: {bi:+.4f}")

        # Stepwise
        print(f"\n  Forward stepwise from Sd:")
        added = []; remaining = list(range(2, D.shape[1]))
        an = ['vf', 'hR'] + full_props
        prev_r2 = R2_0
        for step in range(min(5, len(remaining))):
            best_r2 = prev_r2; best_j = None
            for j in remaining:
                Xt = np.column_stack([X0] + [D[:, c] for c in added] + [D[:, j]])
                pt = np.linalg.lstsq(Xt, y, rcond=None)[0]
                r2t = 1 - np.sum((y - Xt@pt)**2)/SS_tot
                if r2t > best_r2: best_r2 = r2t; best_j = j
            if best_j is None: break
            added.append(best_j); remaining.remove(best_j)
            Xn = np.column_stack([X0] + [D[:, c] for c in added])
            pn = np.linalg.lstsq(Xn, y, rcond=None)[0]
            sc_n = np.std(y - Xn@pn)
            print(f"    +{an[best_j-2]:8s}: R2={best_r2:.4f} (dR2={best_r2-prev_r2:+.4f}, sc={sc_n:.4f})")
            prev_r2 = best_r2

    # T6: partial correlations
    print("\n" + "=" * 70)
    print("T6: Partial correlations")
    print("=" * 70)
    rp_vf, pp_vf = partial_corr(log_gc, log_vf, log_Sd)
    rp_hr, pp_hr = partial_corr(log_gc, log_hRm, log_Sd)
    rp_vf2, pp_vf2 = partial_corr(log_gc, log_vf, log_hRm)
    rp_hr2, pp_hr2 = partial_corr(log_gc, log_hRm, log_vf)
    print(f"  r(gc, vflat | Sd)  = {rp_vf:+.3f} (p={pp_vf:.3e})")
    print(f"  r(gc, hR | Sd)     = {rp_hr:+.3f} (p={pp_hr:.3e})")
    print(f"  r(gc, vflat | hR)  = {rp_vf2:+.3f} (p={pp_vf2:.3e})")
    print(f"  r(gc, hR | vflat)  = {rp_hr2:+.3f} (p={pp_hr2:.3e})")
    # Yd
    rp_yd, pp_yd = partial_corr(log_gc, np.log10(Yd), log_Sd)
    print(f"  r(gc, Yd | Sd)     = {rp_yd:+.3f} (p={pp_yd:.3e})")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Baseline: alpha={sl:.3f}+/-{se:.3f}, R2={R2:.3f}, scatter={scatter:.3f} dex
  Residual: skew={skew:.3f}, kurt={kurt:.3f}, {'normal' if sw_p > 0.05 else 'non-normal'}
  Dependencies: gc rho={r_gc:+.3f}, vflat rho={r_vf:+.3f}, hR rho={r_hr:+.3f}
  Partial:  r(gc,vflat|Sd)={rp_vf:+.3f}, r(gc,hR|Sd)={rp_hr:+.3f}
  eta: median={np.median(eta):.3f}, scatter={np.std(log_eta):.3f} dex
  Significant MRT props: {[p[0] for p in sig_props]}
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        sc_ = ax.scatter(log_Sd, log_gc, c=log_vf, cmap='viridis', s=12, alpha=0.6)
        xf = np.linspace(log_Sd.min(), log_Sd.max(), 50)
        ax.plot(xf, ic+sl*xf, 'r-', lw=2, label=f'a={sl:.3f}')
        ax.set_xlabel('log Sd'); ax.set_ylabel('log gc')
        ax.set_title(f'(a) R2={R2:.3f}'); ax.legend(); plt.colorbar(sc_, ax=ax, label='log vf')

        ax = axes[0, 1]
        ax.scatter(log_gc, resid, s=12, alpha=0.5); ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('log gc'); ax.set_ylabel('resid'); ax.set_title(f'(b) rho={r_gc:+.3f}')

        ax = axes[0, 2]
        ax.scatter(log_vf, resid, s=12, alpha=0.5); ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('log vflat'); ax.set_ylabel('resid'); ax.set_title(f'(c) rho={r_vf:+.3f}')

        ax = axes[1, 0]
        ax.scatter(log_hRm, resid, s=12, alpha=0.5); ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('log hR'); ax.set_ylabel('resid'); ax.set_title(f'(d) rho={r_hr:+.3f}')

        ax = axes[1, 1]
        ax.hist(log_eta, bins=25, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(log_eta), color='r'); ax.axvline(np.log10(0.729), color='b', ls='--')
        ax.set_xlabel('log eta'); ax.set_title(f'(e) scatter={np.std(log_eta):.3f}')

        ax = axes[1, 2]
        ax.hist(resid, bins=25, density=True, alpha=0.7, edgecolor='black')
        xn = np.linspace(resid.min(), resid.max(), 100)
        ax.plot(xn, stats.norm.pdf(xn, 0, scatter), 'r-', lw=2)
        ax.set_xlabel('resid [dex]'); ax.set_title('(f) residual dist')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'residual_structure.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
