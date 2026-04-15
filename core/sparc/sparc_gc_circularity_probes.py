#!/usr/bin/env python3
"""
sparc_gc_circularity_probes.py (TA3+phase1 adapted)
Circularity test + PROBES/MaNGA generalization.
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
    return gc_shape, r_t, v_interp

def extract_features(r_kpc, vobs, hR):
    mask = (r_kpc > 0) & (vobs > 0)
    r = r_kpc[mask]; v = vobs[mask]
    if len(r) < 5: return None
    f = {"N_pts": len(r), "r_max": r[-1], "v_max": np.max(v), "v_last": v[-1],
         "v_max_r": r[np.argmax(v)], "r_max_over_hR": r[-1]/hR if hR > 0 else np.nan}
    n3 = max(3, len(r) // 3)
    if len(r) > n3:
        sl, _, _, _, _ = stats.linregress(r[-n3:], v[-n3:])
        f["slope_outer"] = sl
        sl2, _, _, _, _ = stats.linregress(r[:n3], v[:n3])
        f["slope_inner"] = sl2
    else:
        f["slope_outer"] = 0.0; f["slope_inner"] = 0.0
    f["v_ratio"] = np.max(v) / v[-1] if v[-1] > 0 else np.nan
    ln_r = np.log(r); ln_v = np.log(v)
    b = np.gradient(ln_v, ln_r)
    f["beta_median"] = np.median(b); f["beta_std"] = np.std(b); f["beta_min"] = np.min(b)
    return f

def ols(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return beta, R2, np.std(y - pred), pred

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

def cv_predict(dnames, ddict, y, n_folds=5, seed=42):
    n = len(y)
    np.random.seed(seed)
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    oof = np.full(n, np.nan)
    for i in range(n_folds):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        Xtr = np.column_stack([np.ones(len(tr))] + [ddict[d][tr] for d in dnames])
        Xte = np.column_stack([np.ones(len(te))] + [ddict[d][te] for d in dnames])
        b, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        oof[te] = Xte @ b
    return oof

def main():
    print("=" * 70)
    print("S-2 Circularity + PROBES readiness")
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
        r, vobs, errv, vgas, vdisk, vbul = load_rotcurve(fpath)
        if len(r) < 5: continue
        hR = compute_hR(r, vdisk, Yd)
        if hR is None: continue
        out = extract_gc_shape(r, vobs, beta_threshold=0.25)
        if out is None: continue
        gc_shape, r_t, v_t = out
        feats = extract_features(r, vobs, hR)
        if feats is None: continue
        rec = {"name": gname, "gc_known": gc_known, "gc_shape": gc_shape,
               "r_t": r_t, "v_t": v_t, "vflat": vflat, "hR": hR}
        rec.update(feats)
        records.append(rec)

    N = len(records)
    print(f"Valid galaxies: {N}")
    if N < 30: print("Too few."); return

    gc_known = np.array([r["gc_known"] for r in records])
    gc_shape = np.array([r["gc_shape"] for r in records])
    log_gc_known = np.log10(gc_known); log_gc_shape = np.log10(gc_shape)
    delta = log_gc_shape - log_gc_known
    vflat = np.array([r["vflat"] for r in records])
    hR = np.array([r["hR"] for r in records])
    g_proxy = (vflat * 1e3)**2 / (hR * kpc_m)

    drivers = {
        "log_vflat": np.log10(vflat), "log_hR": np.log10(hR),
        "log_N_pts": np.log10(np.array([r["N_pts"] for r in records])),
        "v_ratio": np.array([r["v_ratio"] for r in records]),
        "beta_median": np.array([r["beta_median"] for r in records]),
        "beta_std": np.array([r["beta_std"] for r in records]),
        "log_r_t": np.log10(np.array([r["r_t"] for r in records])),
        "log_v_t": np.log10(np.array([r["v_t"] for r in records])),
        "slope_outer": np.array([r["slope_outer"] for r in records]),
        "slope_inner": np.array([r["slope_inner"] for r in records]),
        "r_max_over_hR": np.array([r["r_max_over_hR"] for r in records]),
        "beta_min": np.array([r["beta_min"] for r in records]),
    }
    all_finite = np.ones(N, dtype=bool)
    for dn, dv in drivers.items():
        all_finite &= np.isfinite(dv)
    print(f"All valid: {all_finite.sum()}/{N}")

    mask = all_finite
    delta_m = delta[mask]
    log_gc_known_m = log_gc_known[mask]
    log_gc_shape_m = log_gc_shape[mask]
    g_proxy_m = g_proxy[mask]
    drivers_m = {k: v[mask] for k, v in drivers.items()}
    N_m = mask.sum()

    full_8 = ["log_vflat", "log_hR", "log_N_pts", "v_ratio",
              "beta_median", "beta_std", "log_r_t", "log_v_t"]
    no_rt_vt = ["log_vflat", "log_hR", "log_N_pts", "v_ratio",
                "beta_median", "beta_std"]
    only_rt_vt = ["log_r_t", "log_v_t"]
    minimal_3 = ["log_vflat", "log_hR", "v_ratio"]

    print("\n" + "=" * 70)
    print("Test 1: Circularity")
    print("=" * 70)
    models = {
        "M_full_8var": full_8,
        "M_no_rt_vt_6var": no_rt_vt,
        "M_only_rt_vt_2var": only_rt_vt,
        "M_minimal_3var": minimal_3,
        "M_vflat_hR_2var": ["log_vflat", "log_hR"],
    }
    print(f"\n  {'model':<25s} {'train':>8s} {'cv':>8s} {'R2':>6s}")
    print("  " + "-"*50)
    model_results = {}
    for mname, dvars in models.items():
        X = np.column_stack([np.ones(N_m)] + [drivers_m[d] for d in dvars])
        beta, R2, scatter, pred = ols(X, delta_m)
        oof = cv_predict(dvars, drivers_m, delta_m)
        cv_sc = np.std(delta_m - oof)
        model_results[mname] = {"vars": dvars, "beta": beta, "R2": R2,
                                "scatter": scatter, "cv_scatter": cv_sc, "pred": pred}
        print(f"  {mname:<25s} {scatter:>8.3f} {cv_sc:>8.3f} {R2:>6.3f}")

    sc_full = model_results["M_full_8var"]["cv_scatter"]
    sc_no = model_results["M_no_rt_vt_6var"]["cv_scatter"]
    sc_only = model_results["M_only_rt_vt_2var"]["cv_scatter"]
    sc_raw = np.std(delta_m)
    print(f"\n  raw = {sc_raw:.3f}")
    print(f"  r_t+v_t only = {sc_only:.3f} ({(1-sc_only/sc_raw)*100:.1f}%)")
    print(f"  no r_t/v_t   = {sc_no:.3f} ({(1-sc_no/sc_raw)*100:.1f}%)")
    print(f"  full 8       = {sc_full:.3f} ({(1-sc_full/sc_raw)*100:.1f}%)")
    circ = sc_only < sc_no
    if circ:
        print("  WARNING: r_t/v_t dominates -> circularity suspected")
    else:
        print("  OK: shape features have independent info")

    print("\n" + "=" * 70)
    print("Test 2: Direct prediction")
    print("=" * 70)
    direct_models = {
        "D_base_2var": ["log_vflat", "log_hR"],
        "D_min_4var": ["log_vflat", "log_hR", "v_ratio", "beta_median"],
        "D_full_8var": ["log_vflat", "log_hR", "log_N_pts", "v_ratio",
                        "beta_median", "beta_std", "slope_outer", "r_max_over_hR"],
    }
    print(f"\n  {'model':<20s} {'train':>8s} {'cv':>8s} {'R2':>6s}")
    direct_results = {}
    for mname, dvars in direct_models.items():
        X = np.column_stack([np.ones(N_m)] + [drivers_m[d] for d in dvars])
        beta, R2, scatter, pred = ols(X, log_gc_known_m)
        oof = cv_predict(dvars, drivers_m, log_gc_known_m)
        cv_sc = np.std(log_gc_known_m - oof)
        direct_results[mname] = {"vars": dvars, "beta": beta, "R2": R2,
                                  "scatter": scatter, "cv_scatter": cv_sc, "pred": pred}
        print(f"  {mname:<20s} {scatter:>8.3f} {cv_sc:>8.3f} {R2:>6.3f}")

    print("\n" + "=" * 70)
    print("Test 3: Alpha fits")
    print("=" * 70)
    print(f"\n  {'method':<40s} {'alpha':>7s} {'err':>7s} {'p(0.5)':>8s} {'R2':>6s}")
    a, e, _, r2, p05 = fit_alpha(log_gc_known_m, g_proxy_m)
    print(f"  {'gc_known (ref)':<40s} {a:>7.3f} {e:>7.3f} {p05:>8.3f} {r2:>6.3f}")
    a, e, _, r2, p05 = fit_alpha(log_gc_shape_m, g_proxy_m)
    print(f"  {'gc_shape (raw)':<40s} {a:>7.3f} {e:>7.3f} {p05:>8.3f} {r2:>6.3f}")
    for mname, mres in model_results.items():
        gc_corr_log = log_gc_shape_m - mres["pred"]
        a, e, _, r2, p05 = fit_alpha(gc_corr_log, g_proxy_m)
        print(f"  {'shape+'+mname:<40s} {a:>7.3f} {e:>7.3f} {p05:>8.3f} {r2:>6.3f}")
    for mname, mres in direct_results.items():
        a, e, _, r2, p05 = fit_alpha(mres["pred"], g_proxy_m)
        print(f"  {'direct:'+mname:<40s} {a:>7.3f} {e:>7.3f} {p05:>8.3f} {r2:>6.3f}")

    print("\n" + "=" * 70)
    print("Test 4: 50/50 split generalization")
    print("=" * 70)
    test_models = {
        "6var_no_rt_vt": no_rt_vt,
        "3var_minimal": minimal_3,
        "2var_base": ["log_vflat", "log_hR"],
    }
    n_splits = 20
    split_results = {m: {"alpha_test": [], "scatter_test": [], "alpha_train": []}
                     for m in test_models}
    for seed in range(n_splits):
        np.random.seed(seed*7 + 13)
        idx = np.arange(N_m); np.random.shuffle(idx)
        half = N_m // 2
        tr = idx[:half]; te = idx[half:]
        for mname, dvars in test_models.items():
            X_tr = np.column_stack([np.ones(len(tr))] + [drivers_m[d][tr] for d in dvars])
            X_te = np.column_stack([np.ones(len(te))] + [drivers_m[d][te] for d in dvars])
            b_tr, _, _, _ = np.linalg.lstsq(X_tr, delta_m[tr], rcond=None)
            delta_pred_te = X_te @ b_tr
            gc_corr_te = log_gc_shape_m[te] - delta_pred_te
            try:
                a_te, _, _, _, _ = fit_alpha(gc_corr_te, g_proxy_m[te], n_boot=100)
                a_tr, _, _, _, _ = fit_alpha(log_gc_shape_m[tr] - (X_tr @ b_tr),
                                              g_proxy_m[tr], n_boot=100)
                sc_te = np.std(gc_corr_te - log_gc_known_m[te])
                split_results[mname]["alpha_test"].append(a_te)
                split_results[mname]["alpha_train"].append(a_tr)
                split_results[mname]["scatter_test"].append(sc_te)
            except: pass

    print(f"\n  {'model':<20s} {'a_test':>8s} {'a_std':>8s} {'sc_test':>9s} {'a_train':>9s}")
    for mname in test_models:
        sr = split_results[mname]
        if sr["alpha_test"]:
            at = np.array(sr["alpha_test"])
            atr = np.array(sr["alpha_train"])
            st = np.array(sr["scatter_test"])
            print(f"  {mname:<20s} {at.mean():>8.3f} {at.std():>8.3f} "
                  f"{st.mean():>9.3f} {atr.mean():>9.3f}")
            cov = np.mean(np.abs(at - 0.5) < 2*at.std())
            print(f"  {' '*20} 2sigma coverage of 0.5: {cov*100:.0f}%")

    print("\n" + "=" * 70)
    print("Test 5: Sample size requirements")
    print("=" * 70)
    best_gen = min(split_results,
                   key=lambda m: np.mean(split_results[m]["scatter_test"])
                   if split_results[m]["scatter_test"] else 999)
    print(f"  Best: {best_gen}")
    sr = split_results[best_gen]
    if sr["alpha_test"]:
        at = np.array(sr["alpha_test"]); st = np.array(sr["scatter_test"])
        print(f"  gen alpha: {at.mean():.3f} +/- {at.std():.3f}")
        print(f"  gen scatter: {st.mean():.3f} dex")
        dr = np.std(np.log10(a0 * g_proxy_m))
        for te in [0.05, 0.03, 0.02]:
            Nn = int((st.mean() / (te * dr))**2)
            print(f"  alpha +/-{te} needs N ~ {Nn}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Circularity: {'YES' if circ else 'NO'}")
    best = min(split_results,
               key=lambda m: abs(np.mean(split_results[m]["alpha_test"]) - 0.5)
               if split_results[m]["alpha_test"] else 999)
    sr = split_results[best]
    if sr["alpha_test"]:
        am = np.mean(sr["alpha_test"]); asd = np.std(sr["alpha_test"])
        sm = np.mean(sr["scatter_test"])
        print(f"  Best gen ({best}): alpha={am:.3f}+/-{asd:.3f} sc={sm:.3f}")
        if abs(am - 0.5) < 2*asd and sm < 0.5:
            print("  Verdict: **GO** for PROBES/MaNGA")
        elif abs(am - 0.5) < 3*asd:
            print("  Verdict: **Conditional GO**")
        else:
            print("  Verdict: HOLD")

    # figure (simplified)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        names = list(models.keys())
        cvs = [model_results[m]["cv_scatter"] for m in names]
        ax.bar(range(len(names)), cvs, color=['steelblue','seagreen','salmon','orange','gray'])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=30)
        ax.axhline(sc_raw, color='k', ls='--', label='raw')
        ax.set_ylabel('CV scatter')
        ax.set_title('(a) Bias correction')
        ax.legend()

        ax = axes[0, 1]
        labs = ['raw', 'r_t+v_t', 'no r_t/v_t', 'full']
        vals = [sc_raw, sc_only, sc_no, sc_full]
        ax.bar(range(4), vals, color=['gray','salmon','seagreen','steelblue'])
        ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel('CV scatter')
        ax.set_title('(b) Circularity')
        msg = "CIRCULAR" if circ else "OK"
        col = 'red' if circ else 'green'
        ax.text(0.5, 0.95, msg, transform=ax.transAxes, ha='center', va='top',
                fontsize=12, color=col, fontweight='bold')

        ax = axes[1, 0]
        for i, (mname, color) in enumerate(zip(test_models, ['seagreen','orange','gray'])):
            sr = split_results[mname]
            if sr["alpha_test"]:
                at = np.array(sr["alpha_test"])
                ax.scatter([i]*len(at), at, c=color, s=30, alpha=0.5, label=mname)
                ax.errorbar(i, at.mean(), yerr=at.std(), color=color, capsize=5, lw=2)
        ax.axhline(0.5, color='red', ls='--')
        ax.set_xticks(range(len(test_models)))
        ax.set_xticklabels(list(test_models.keys()), fontsize=7, rotation=15)
        ax.set_ylabel('alpha test'); ax.set_title('(c) 50/50 split')
        ax.legend(fontsize=6)

        ax = axes[1, 1]
        target_errs = np.linspace(0.01, 0.10, 50)
        dr_val = np.std(np.log10(a0 * g_proxy_m))
        for mname, color in zip(test_models, ['seagreen','orange','gray']):
            sr = split_results[mname]
            if sr["scatter_test"]:
                sc = np.mean(sr["scatter_test"])
                Nn = (sc/(target_errs*dr_val))**2
                ax.plot(target_errs, Nn, color=color, label=mname)
        ax.axhline(175, color='blue', ls='--', label='SPARC')
        ax.axhline(1500, color='purple', ls=':', label='PROBES')
        ax.axhline(10000, color='red', ls=':', label='MaNGA')
        ax.set_xlabel('target alpha err'); ax.set_ylabel('N needed')
        ax.set_yscale('log'); ax.set_ylim(10, 1e5)
        ax.set_title('(d) Sample size')
        ax.legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'gc_circularity_probes.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
