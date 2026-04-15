#!/usr/bin/env python3
"""
sparc_gc_shape.py (TA3+phase1 adapted)
Shape-based gc extraction from rotation curve alone.
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
            parts = line.split()
            if len(parts) >= 6:
                try:
                    r.append(float(parts[0])); vobs.append(float(parts[1]))
                    errv.append(float(parts[2])); vgas.append(float(parts[3]))
                    vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
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
    beta_val = beta[idx] if idx < len(beta) else np.nan
    return gc_shape, r_t, v_interp, beta_val

def main():
    print("=" * 70)
    print("S-2 Shape-based gc extraction (Vobs only)")
    print("=" * 70)

    pipe = load_pipeline()
    print(f"Pipeline galaxies: {len(pipe)}")
    rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, "*_rotmod.dat")))
    print(f"Rotmod files: {len(rotmod_files)}")

    beta_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    results_by_beta = {}
    for bt in beta_thresholds:
        results = []
        for fpath in rotmod_files:
            gname = os.path.basename(fpath).replace('_rotmod.dat', '')
            if gname not in pipe: continue
            gd = pipe[gname]
            gc_known = gd['gc']
            vflat = gd['vflat']
            Yd = gd.get('Yd', 0.5)
            if gc_known <= 0 or vflat <= 0: continue
            r, vobs, errv, vgas, vdisk, vbul = load_rotcurve(fpath)
            if len(r) < 5: continue
            hR = compute_hR(r, vdisk, Yd)
            if hR is None: continue
            out = extract_gc_shape(r, vobs, beta_threshold=bt)
            if out is None: continue
            gc_shape, r_t, v_t, bv = out
            results.append({'name': gname, 'gc_known': gc_known, 'gc_shape': gc_shape,
                            'r_t': r_t, 'v_t': v_t, 'vflat': vflat, 'hR': hR})
        results_by_beta[bt] = results
        n = len(results)
        if n > 10:
            lk = np.log10([r['gc_known'] for r in results])
            ls = np.log10([r['gc_shape'] for r in results])
            rho, p = stats.pearsonr(lk, ls)
            ratio = np.median(np.array([r['gc_shape'] for r in results]) /
                              np.array([r['gc_known'] for r in results]))
            print(f"beta_t={bt:.2f}: N={n}, r={rho:.3f} (p={p:.2e}), "
                  f"median(shape/known)={ratio:.3f}")

    best_bt, best_r = None, -1
    for bt, res in results_by_beta.items():
        if len(res) < 20: continue
        lk = np.log10([r['gc_known'] for r in res])
        ls = np.log10([r['gc_shape'] for r in res])
        rho, _ = stats.pearsonr(lk, ls)
        if rho > best_r:
            best_r = rho; best_bt = bt
    if best_bt is None:
        best_bt = max(results_by_beta, key=lambda bt: len(results_by_beta[bt]))
    results = results_by_beta[best_bt]
    print(f"\nBest beta_t={best_bt:.2f} (N={len(results)}, r={best_r:.3f})")

    gc_k = np.array([r['gc_known'] for r in results])
    gc_s = np.array([r['gc_shape'] for r in results])
    lk = np.log10(gc_k); ls = np.log10(gc_s)
    vflat_a = np.array([r['vflat'] for r in results])
    hR_a = np.array([r['hR'] for r in results])

    print("\n" + "=" * 70)
    print("T1: gc_shape vs gc_known")
    print("=" * 70)
    slope, intercept, r_val, p_val, se = stats.linregress(lk, ls)
    print(f"  log(gc_shape) = {slope:.3f}*log(gc_known) + {intercept:.3f}")
    print(f"  Pearson r={r_val:.3f}, p={p_val:.2e}")
    print(f"  Spearman rho={stats.spearmanr(lk, ls).statistic:.3f}")
    print(f"  scatter dex={np.std(ls - lk):.3f}")
    print(f"  median ratio={np.median(gc_s/gc_k):.3f}")

    print("\n" + "=" * 70)
    print("T2: alpha fit (multivariate)")
    print("=" * 70)
    mask = (vflat_a > 0) & (hR_a > 0)
    print(f"  valid: {mask.sum()}/{len(mask)}")
    if mask.sum() < 20:
        print("Too few valid."); return
    log_vf = np.log10(vflat_a[mask])
    log_hR = np.log10(hR_a[mask])
    log_gc_k = lk[mask]; log_gc_s = ls[mask]
    X = np.column_stack([np.ones(mask.sum()), log_vf, log_hR])
    beta_k, _, _, _ = np.linalg.lstsq(X, log_gc_k, rcond=None)
    beta_s, _, _, _ = np.linalg.lstsq(X, log_gc_s, rcond=None)
    R2_k = 1 - np.sum((log_gc_k - X @ beta_k)**2) / np.sum((log_gc_k - log_gc_k.mean())**2)
    R2_s = 1 - np.sum((log_gc_s - X @ beta_s)**2) / np.sum((log_gc_s - log_gc_s.mean())**2)
    print(f"  gc_known: {beta_k[0]:.3f} + {beta_k[1]:.3f}*log(vflat) + "
          f"{beta_k[2]:.3f}*log(hR)  R2={R2_k:.3f}")
    print(f"  gc_shape: {beta_s[0]:.3f} + {beta_s[1]:.3f}*log(vflat) + "
          f"{beta_s[2]:.3f}*log(hR)  R2={R2_s:.3f}")
    print(f"  alpha_known: vflat={beta_k[1]/2:.3f}, hR={-beta_k[2]:.3f}")
    print(f"  alpha_shape: vflat={beta_s[1]/2:.3f}, hR={-beta_s[2]:.3f}")

    print("\n" + "=" * 70)
    print("T3: direct alpha fit")
    print("=" * 70)
    g_proxy = (vflat_a[mask] * 1e3)**2 / (hR_a[mask] * kpc_m)

    def model(params, gp):
        return params[0] + params[1] * np.log10(a0 * gp)

    def resf(params, gp, y):
        return model(params, gp) - y

    res_k = optimize.least_squares(resf, [0, 0.5], args=(g_proxy, log_gc_k),
                                    bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    res_s = optimize.least_squares(resf, [0, 0.5], args=(g_proxy, log_gc_s),
                                    bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha_k = res_k.x[1]; alpha_s = res_s.x[1]
    eta_k = 10**res_k.x[0]; eta_s = 10**res_s.x[0]
    pred_k = model(res_k.x, g_proxy)
    pred_s = model(res_s.x, g_proxy)
    R2k = 1 - np.sum((log_gc_k - pred_k)**2)/np.sum((log_gc_k - log_gc_k.mean())**2)
    R2s = 1 - np.sum((log_gc_s - pred_s)**2)/np.sum((log_gc_s - log_gc_s.mean())**2)

    np.random.seed(42)
    ab_k, ab_s = [], []
    n_g = mask.sum()
    for _ in range(500):
        idx = np.random.choice(n_g, n_g, replace=True)
        try:
            rb = optimize.least_squares(resf, res_k.x, args=(g_proxy[idx], log_gc_k[idx]),
                                        bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            ab_k.append(rb.x[1])
        except: pass
        try:
            rb = optimize.least_squares(resf, res_s.x, args=(g_proxy[idx], log_gc_s[idx]),
                                        bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            ab_s.append(rb.x[1])
        except: pass
    err_k = np.std(ab_k); err_s = np.std(ab_s)
    z_k = (alpha_k - 0.5) / err_k if err_k > 0 else np.inf
    z_s = (alpha_s - 0.5) / err_s if err_s > 0 else np.inf
    p05_k = 2 * stats.norm.sf(abs(z_k))
    p05_s = 2 * stats.norm.sf(abs(z_s))
    print(f"  gc_known: alpha={alpha_k:.3f}+/-{err_k:.3f}, eta={eta_k:.3f}, "
          f"R2={R2k:.3f}, p(0.5)={p05_k:.3f}")
    print(f"  gc_shape: alpha={alpha_s:.3f}+/-{err_s:.3f}, eta={eta_s:.3f}, "
          f"R2={R2s:.3f}, p(0.5)={p05_s:.3f}")

    print("\n" + "=" * 70)
    print("T4: Vobs proxy problem")
    print("=" * 70)
    print(f"  Vbar-based (prev): alpha ~ 0.545+/-0.041")
    print(f"  Vobs-proxy gc_known: alpha={alpha_k:.3f}+/-{err_k:.3f}")
    print(f"  Vobs-proxy gc_shape: alpha={alpha_s:.3f}+/-{err_s:.3f}")
    if abs(alpha_k - 1.0) < 3*err_k:
        print("  -> Vobs proxy gives alpha~1, as expected problem")
    elif abs(alpha_k - 0.5) < 3*err_k:
        print("  -> Vobs proxy gives alpha~0.5, unexpectedly good")

    print("\n" + "=" * 70)
    print("T5: transition acceleration distribution")
    print("=" * 70)
    print(f"  gc_shape: median={np.median(gc_s):.2e}, scatter={np.std(ls):.3f} dex")
    print(f"  gc_known: median={np.median(gc_k):.2e}, scatter={np.std(lk):.3f} dex")
    print(f"  a0={a0:.2e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  shape scatter: {np.std(ls-lk):.3f} dex")
    print(f"  alpha (Vobs proxy): {alpha_s:.3f}+/-{err_s:.3f}")
    if abs(alpha_s - 0.5) < 2*err_s:
        verdict = "A: alpha~0.5 recovered. PROBES/MaNGA promising"
    elif abs(alpha_s - 0.5) < 3*err_s:
        verdict = "B: alpha=0.5 not rejected but biased"
    elif abs(alpha_s - 1.0) < 2*err_s:
        verdict = "C: alpha~1.0, Vobs proxy problem persists"
    else:
        verdict = f"D: alpha={alpha_s:.3f}, partial"
    print(f"  Verdict: {verdict}")

    # figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        ax.scatter(lk, ls, s=10, alpha=0.5, c='steelblue')
        lim = [min(lk.min(), ls.min())-0.1, max(lk.max(), ls.max())+0.1]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.plot(lim, [slope*x+intercept for x in lim], 'r-', lw=1,
                label=f"slope={slope:.2f}")
        ax.set_xlabel('log gc_known'); ax.set_ylabel('log gc_shape')
        ax.set_title(f"(a) r={r_val:.3f}, N={len(results)}")
        ax.legend()

        ax = axes[0, 1]
        ax.scatter(log_vf, log_gc_s - log_gc_k, s=10, alpha=0.5, c='steelblue')
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel('log vflat'); ax.set_ylabel('log(shape/known)')
        ax.set_title('(b) residual vs vflat')

        ax = axes[1, 0]
        lx = np.log10(a0 * g_proxy)
        ax.scatter(lx, log_gc_s, s=10, alpha=0.5, c='steelblue')
        xf = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xf, res_s.x[0] + res_s.x[1]*xf, 'r-', lw=2,
                label=f'alpha={alpha_s:.3f}+/-{err_s:.3f}')
        ax.set_xlabel('log(a0 * vflat^2/hR)'); ax.set_ylabel('log gc_shape')
        ax.set_title(f'(c) R2={R2s:.3f}')
        ax.legend()

        ax = axes[1, 1]
        bts = sorted(results_by_beta.keys())
        ns = [len(results_by_beta[bt]) for bt in bts]
        rs = []
        for bt in bts:
            r_ = results_by_beta[bt]
            if len(r_) > 10:
                lk2 = np.log10([r['gc_known'] for r in r_])
                ls2 = np.log10([r['gc_shape'] for r in r_])
                rs.append(stats.pearsonr(lk2, ls2).statistic)
            else:
                rs.append(np.nan)
        ax.plot(bts, rs, 'o-', c='steelblue')
        ax2 = ax.twinx()
        ax2.bar(bts, ns, width=0.03, alpha=0.3, color='orange')
        ax.set_xlabel('beta threshold'); ax.set_ylabel('Pearson r')
        ax2.set_ylabel('N')
        ax.set_title('(d) beta sensitivity')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'gc_shape_extraction.png'), dpi=150)
        print("\nFigure saved: gc_shape_extraction.png")
    except Exception as e:
        print(f"Figure error: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()
