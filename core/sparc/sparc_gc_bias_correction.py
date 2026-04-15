#!/usr/bin/env python3
"""
sparc_gc_bias_correction.py (TA3+phase1 adapted)
Bias correction model for shape-based gc.
"""
import os, csv, warnings, glob
from itertools import combinations
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

def extract_gc_shape(r_kpc, vobs, beta_threshold=0.25, min_points=5):
    if len(r_kpc) < min_points: return None
    mask = (r_kpc > 0) & (vobs > 0)
    r = r_kpc[mask]; v = vobs[mask]
    if len(r) < min_points: return None
    ln_r = np.log(r); ln_v = np.log(v)
    if len(ln_v) >= 5:
        ln_v_s = np.convolve(ln_v, np.ones(3)/3, mode='valid')
        ln_r_m = np.convolve(ln_r, np.ones(3)/3, mode='valid')
        beta = np.gradient(ln_v_s, ln_r_m)
        r_beta = np.exp(ln_r_m)
    else:
        beta = np.gradient(ln_v, ln_r); r_beta = r
    idx = None
    for i in range(1, len(beta)):
        if beta[i-1] >= beta_threshold and beta[i] < beta_threshold:
            idx = i; break
    if idx is None:
        if np.all(beta < beta_threshold): idx = 0
        else: return None
    if idx > 0 and idx < len(beta):
        b0, b1 = beta[idx-1], beta[idx]
        r0, r1 = r_beta[idx-1], r_beta[idx]
        if b0 != b1:
            frac = (beta_threshold - b0) / (b1 - b0)
            r_t = r0 + frac * (r1 - r0)
        else:
            r_t = r_beta[idx]
    else:
        r_t = r_beta[idx]
    try:
        v_interp = np.interp(r_t, r, v)
    except: return None
    if r_t <= 0 or v_interp <= 0: return None
    gc_shape = (v_interp * 1e3)**2 / (r_t * kpc_m)
    return gc_shape, r_t, v_interp, beta, r_beta

def extract_curve_features(r_kpc, vobs):
    mask = (r_kpc > 0) & (vobs > 0)
    r = r_kpc[mask]; v = vobs[mask]
    if len(r) < 5: return None
    f = {}
    f["N_pts"] = len(r); f["r_max"] = r[-1]; f["r_min"] = r[0]
    f["v_max"] = np.max(v); f["v_last"] = v[-1]
    f["v_max_r"] = r[np.argmax(v)]
    n3 = max(3, len(r) // 3)
    if len(r) > n3:
        so, _, _, _, _ = stats.linregress(r[-n3:], v[-n3:])
        si, _, _, _, _ = stats.linregress(r[:n3], v[:n3])
        f["slope_outer"] = so; f["slope_inner"] = si
    else:
        f["slope_outer"] = 0.0; f["slope_inner"] = 0.0
    f["v_ratio"] = np.max(v) / v[-1] if v[-1] > 0 else np.nan
    ln_r = np.log(r); ln_v = np.log(v)
    b = np.gradient(ln_v, ln_r)
    f["beta_median"] = np.median(b); f["beta_std"] = np.std(b); f["beta_min"] = np.min(b)
    return f

def main():
    print("=" * 70)
    print("S-2 bias correction model")
    print("=" * 70)
    pipe = load_pipeline()
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Pipeline: {len(pipe)}, Rotmod: {len(rotmod_files)}")

    beta_t = 0.25
    records = []
    for fpath in rotmod_files:
        gname = os.path.basename(fpath).replace('_rotmod.dat', '')
        if gname not in pipe: continue
        gd = pipe[gname]
        gc_known = gd['gc']; vflat = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc_known <= 0 or vflat <= 0: continue
        r, vobs, errv, vgas, vdisk, vbul = load_rotcurve(fpath)
        if len(r) < 5: continue
        hR = compute_hR(r, vdisk, Yd)
        if hR is None: continue
        out = extract_gc_shape(r, vobs, beta_threshold=beta_t)
        if out is None: continue
        gc_shape, r_t, v_t, _, _ = out
        feats = extract_curve_features(r, vobs)
        if feats is None: continue
        rec = {"name": gname, "gc_known": gc_known, "gc_shape": gc_shape,
               "r_t": r_t, "v_t": v_t, "vflat": vflat, "hR": hR,
               "r_max_over_hR": feats["r_max"] / hR if hR > 0 else np.nan}
        rec.update(feats)
        records.append(rec)

    N = len(records)
    print(f"Valid galaxies: {N}")
    if N < 30: print("Too few."); return

    delta = np.array([np.log10(r['gc_shape']/r['gc_known']) for r in records])
    print(f"\nT1: delta driver analysis")
    print(f"  delta: mean={np.mean(delta):.3f}, std={np.std(delta):.3f}")

    drivers = {
        "log_vflat": np.log10(np.array([r["vflat"] for r in records])),
        "log_hR": np.log10(np.array([r["hR"] for r in records])),
        "log_N_pts": np.log10(np.array([r["N_pts"] for r in records])),
        "r_max_over_hR": np.array([r["r_max_over_hR"] for r in records]),
        "slope_outer": np.array([r["slope_outer"] for r in records]),
        "v_ratio": np.array([r["v_ratio"] for r in records]),
        "beta_median": np.array([r["beta_median"] for r in records]),
        "beta_std": np.array([r["beta_std"] for r in records]),
        "log_r_t": np.log10(np.array([r["r_t"] for r in records])),
        "log_v_t": np.log10(np.array([r["v_t"] for r in records])),
    }

    print(f"\n  {'driver':<20s} {'pearson':>10s} {'p':>12s} {'spearman':>12s}")
    print("  " + "-"*56)
    driver_corrs = {}
    for dn, dv in drivers.items():
        valid = np.isfinite(dv) & np.isfinite(delta)
        if valid.sum() < 20: continue
        pr, pp = stats.pearsonr(dv[valid], delta[valid])
        sr, sp = stats.spearmanr(dv[valid], delta[valid])
        driver_corrs[dn] = (pr, pp, sr, sp, valid)
        m = "*" if abs(pr) > 0.3 else " "
        print(f"  {dn:<20s} {pr:>+10.3f} {pp:>12.2e} {sr:>+12.3f} {m}")

    sig_names = [n for n, (pr, *_) in driver_corrs.items() if abs(pr) > 0.15]
    print(f"\nSignificant drivers (|r|>0.15): {sig_names}")

    all_valid = np.ones(N, dtype=bool)
    for dn in sig_names:
        all_valid &= np.isfinite(drivers[dn])
    all_valid &= np.isfinite(delta)
    n_v = all_valid.sum()
    print(f"All valid: {n_v}")
    delta_v = delta[all_valid]

    def build_X(dnames):
        cols = [np.ones(n_v)]
        for dn in dnames: cols.append(drivers[dn][all_valid])
        return np.column_stack(cols)

    def ols_fit(X, y):
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        yp = X @ beta
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - y.mean())**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        n, k = X.shape
        AICc = n*np.log(ss_res/n) + 2*k + (2*k*(k+1))/max(n-k-1, 1)
        return beta, R2, AICc, np.std(y - yp)

    def cv_score(dnames):
        n = len(delta_v); idx = np.arange(n)
        np.random.seed(42); np.random.shuffle(idx)
        folds = np.array_split(idx, 5)
        oof = np.full(n, np.nan)
        full = {dn: drivers[dn][all_valid] for dn in dnames}
        for i in range(5):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(5) if j != i])
            Xtr = [np.ones(len(tr))]; Xte = [np.ones(len(te))]
            for dn in dnames:
                Xtr.append(full[dn][tr]); Xte.append(full[dn][te])
            Xtr = np.column_stack(Xtr); Xte = np.column_stack(Xte)
            bt, _, _, _ = np.linalg.lstsq(Xtr, delta_v[tr], rcond=None)
            oof[te] = Xte @ bt
        res = delta_v - oof
        return np.std(res[np.isfinite(res)])

    print("\nT2: model selection")
    X0 = build_X([]); _, R2_0, _, sc0 = ols_fit(X0, delta_v)
    cv0 = np.std(delta_v)
    print(f"  M0 const: sc={sc0:.3f} cv={cv0:.3f}")

    single = {}
    for dn in sig_names:
        X1 = build_X([dn]); _, r2, _, sc = ols_fit(X1, delta_v)
        cv = cv_score([dn])
        single[dn] = (r2, sc, cv)
        print(f"  M1 {dn:<18s}: sc={sc:.3f} R2={r2:.3f} cv={cv:.3f}")

    best_single = min(single, key=lambda k: single[k][2]) if single else None

    pair_results = {}
    if len(sig_names) >= 2:
        for pair in combinations(sig_names, 2):
            X2 = build_X(list(pair)); _, r2, _, sc = ols_fit(X2, delta_v)
            cv = cv_score(list(pair))
            pair_results[pair] = (r2, sc, cv)
        sorted_pairs = sorted(pair_results.items(), key=lambda x: x[1][2])
        print("\n  Top 5 pairs by CV:")
        for pair, (r2, sc, cv) in sorted_pairs[:5]:
            print(f"    {pair[0]:>15s}+{pair[1]:<15s}: sc={sc:.3f} R2={r2:.3f} cv={cv:.3f}")
        best_pair = sorted_pairs[0][0]
    else:
        best_pair = None

    if len(sig_names) >= 3:
        X_all = build_X(sig_names); _, r2_all, _, sc_all = ols_fit(X_all, delta_v)
        cv_all = cv_score(sig_names)
        print(f"\n  M_all ({len(sig_names)} var): sc={sc_all:.3f} R2={r2_all:.3f} cv={cv_all:.3f}")
    else:
        cv_all = 999; sc_all = 999

    cands = {"M0": ([], cv0, sc0)}
    if best_single: cands[f"M1_{best_single}"] = ([best_single], single[best_single][2], single[best_single][1])
    if best_pair: cands[f"M2_{best_pair[0]}+{best_pair[1]}"] = (list(best_pair), pair_results[best_pair][2], pair_results[best_pair][1])
    if len(sig_names) >= 3: cands[f"M_all"] = (sig_names, cv_all, sc_all)

    print("\nT3: best model selection")
    print(f"  {'model':<40s} {'cv':>8s} {'train':>8s}")
    for mname, (dv, cv, sc) in cands.items():
        m = " *" if cv == min(c[1] for c in cands.values()) else ""
        print(f"  {mname:<40s} {cv:>8.3f} {sc:>8.3f}{m}")

    best_mname = min(cands, key=lambda k: cands[k][1])
    best_vars, _, _ = cands[best_mname]
    print(f"\n  Selected: {best_mname}")

    if best_vars:
        X_best = build_X(best_vars)
        beta_best, _, _, _ = ols_fit(X_best, delta_v)
        delta_pred = X_best @ beta_best
        delta_corr = delta_v - delta_pred
        print(f"  intercept={beta_best[0]:.4f}")
        for i, vn in enumerate(best_vars):
            print(f"    {vn}: {beta_best[i+1]:.4f}")
        gc_shape_v = np.array([r["gc_shape"] for r in records])[all_valid]
        gc_corr = gc_shape_v * 10**(-delta_pred)
    else:
        beta_best = None
        delta_corr = delta_v - np.mean(delta_v)
        gc_corr = np.array([r["gc_shape"] for r in records])[all_valid]

    gc_known_v = np.array([r["gc_known"] for r in records])[all_valid]
    print(f"\n  scatter: {np.std(delta_v):.3f} -> {np.std(delta_corr):.3f} dex "
          f"({(1-np.std(delta_corr)/np.std(delta_v))*100:.1f}% improvement)")

    # Alpha refit
    vflat_v = np.array([r["vflat"] for r in records])[all_valid]
    hR_v = np.array([r["hR"] for r in records])[all_valid]
    mfit = (vflat_v > 0) & (hR_v > 0)
    print(f"\nAlpha fit valid: {mfit.sum()}")

    if mfit.sum() < 20: print("Too few."); return

    g_proxy = (vflat_v[mfit] * 1e3)**2 / (hR_v[mfit] * kpc_m)
    log_gc_known = np.log10(gc_known_v[mfit])
    log_gc_raw = np.log10(np.array([r["gc_shape"] for r in records])[all_valid][mfit])
    log_gc_corr = np.log10(gc_corr[mfit])

    def model(p, gp): return p[0] + p[1]*np.log10(a0*gp)
    def resf(p, gp, y): return model(p, gp) - y

    results_alpha = {}
    for label, lg in [("gc_known", log_gc_known), ("gc_shape_raw", log_gc_raw),
                      ("gc_shape_corr", log_gc_corr)]:
        res = optimize.least_squares(resf, [0, 0.5], args=(g_proxy, lg),
                                      bounds=([-np.inf, 0.01], [np.inf, 2.0]))
        alpha = res.x[1]; eta = 10**res.x[0]
        pred = model(res.x, g_proxy)
        R2 = 1 - np.sum((lg-pred)**2)/np.sum((lg-lg.mean())**2)
        np.random.seed(42)
        boots = []
        ng = mfit.sum()
        for _ in range(500):
            idx = np.random.choice(ng, ng, replace=True)
            try:
                rb = optimize.least_squares(resf, res.x, args=(g_proxy[idx], lg[idx]),
                                             bounds=([-np.inf, 0.01], [np.inf, 2.0]))
                boots.append(rb.x[1])
            except: pass
        err = np.std(boots)
        z = (alpha - 0.5)/err if err > 0 else np.inf
        p05 = 2 * stats.norm.sf(abs(z))
        results_alpha[label] = {'alpha': alpha, 'err': err, 'eta': eta,
                                 'R2': R2, 'p_05': p05}

    print(f"\n  {'source':<18s} {'alpha':>7s} {'err':>7s} {'p(0.5)':>8s} {'R2':>7s} {'eta':>8s}")
    for lbl, r in results_alpha.items():
        print(f"  {lbl:<18s} {r['alpha']:>7.3f} {r['err']:>7.3f} "
              f"{r['p_05']:>8.3f} {r['R2']:>7.3f} {r['eta']:>8.3f}")

    # 5-fold CV for alpha
    print("\nT4: 5-fold CV alpha stability")
    n_g = mfit.sum()
    idx = np.arange(n_g)
    np.random.seed(123); np.random.shuffle(idx)
    folds = np.array_split(idx, 5)
    cv_alphas = {"gc_known": [], "gc_shape_raw": [], "gc_shape_corr": []}
    for i in range(5):
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        for lbl, lg in [("gc_known", log_gc_known), ("gc_shape_raw", log_gc_raw),
                        ("gc_shape_corr", log_gc_corr)]:
            try:
                r_ = optimize.least_squares(resf, [0, 0.5],
                                             args=(g_proxy[tr], lg[tr]),
                                             bounds=([-np.inf, 0.01], [np.inf, 2.0]))
                cv_alphas[lbl].append(r_.x[1])
            except: pass
    print(f"  {'source':<18s} {'mean':>7s} {'std':>7s} {'range':>16s}")
    for lbl, a in cv_alphas.items():
        if a:
            arr = np.array(a)
            print(f"  {lbl:<18s} {arr.mean():>7.3f} {arr.std():>7.3f} "
                  f"[{arr.min():.3f},{arr.max():.3f}]")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    sb = np.std(delta_v); sa = np.std(delta_corr)
    imp = (1 - sa/sb) * 100
    ac = results_alpha["gc_shape_corr"]
    print(f"  scatter: {sb:.3f} -> {sa:.3f} dex ({imp:.1f}%)")
    print(f"  alpha corrected: {ac['alpha']:.3f}+/-{ac['err']:.3f}, p(0.5)={ac['p_05']:.3f}")
    if sa < 0.4 and abs(ac['alpha']-0.5) < 2*ac['err']:
        print("  Verdict A: PROBES/MaNGA ready")
    elif sa < 0.5:
        print("  Verdict B: improved, binning needed")
    else:
        print("  Verdict C: limited")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        if sig_names:
            top = max(driver_corrs, key=lambda k: abs(driver_corrs[k][0]))
            dv = drivers[top]
            v = np.isfinite(dv) & np.isfinite(delta)
            ax.scatter(dv[v], delta[v], s=10, alpha=0.5, c='steelblue')
            ax.axhline(0, color='k', ls='--')
            ax.set_xlabel(top); ax.set_ylabel('delta')
            ax.set_title(f"(a) top driver {top} r={driver_corrs[top][0]:.3f}")

        ax = axes[0, 1]
        ax.hist(delta_v, bins=25, alpha=0.5, color='salmon',
                label=f'raw s={np.std(delta_v):.3f}')
        ax.hist(delta_corr, bins=25, alpha=0.5, color='steelblue',
                label=f'corr s={np.std(delta_corr):.3f}')
        ax.axvline(0, color='k', ls='--')
        ax.set_xlabel('delta')
        ax.set_title('(b) distribution')
        ax.legend()

        ax = axes[1, 0]
        lx = np.log10(a0 * g_proxy)
        ax.scatter(lx, log_gc_corr, s=10, alpha=0.3, c='steelblue')
        xf = np.linspace(lx.min(), lx.max(), 50)
        for lbl, col, lss in [("gc_known", 'k', '-'), ("gc_shape_raw", 'salmon', '--'),
                               ("gc_shape_corr", 'steelblue', '-')]:
            ra = results_alpha[lbl]
            ax.plot(xf, np.log10(ra['eta']) + ra['alpha']*xf, color=col, ls=lss, lw=2,
                    label=f"{lbl}: a={ra['alpha']:.3f}")
        ax.set_xlabel('log(a0*vflat^2/hR)'); ax.set_ylabel('log gc')
        ax.set_title('(c) alpha comparison')
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        pos = [1, 2, 3]
        lbls = ["gc_known", "gc_shape_raw", "gc_shape_corr"]
        cols = ['k', 'salmon', 'steelblue']
        for p_, l, c in zip(pos, lbls, cols):
            if cv_alphas[l]:
                arr = np.array(cv_alphas[l])
                ax.scatter([p_]*len(arr), arr, c=c, s=40)
                ax.errorbar(p_, arr.mean(), yerr=arr.std(), color=c, capsize=5)
        ax.axhline(0.5, color='red', ls='--')
        ax.set_xticks(pos); ax.set_xticklabels(['known', 'raw', 'corr'], fontsize=8)
        ax.set_ylabel('alpha'); ax.set_title('(d) CV alpha')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'gc_bias_correction.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()
