#!/usr/bin/env python3
"""
sparc_hR_definition_check.py
hR definition comparison: pipeline (Vdisk peak/2.15) vs MRT (photometric Rdisk)
"""
import os, csv, warnings, glob
import numpy as np
from scipy import stats, optimize
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
            try: data[p[0]] = {'Rdisk': float(p[11]), 'Vflat_mrt': float(p[15])}
            except: continue
    return data

def compute_hR_pipeline(gname, Yd):
    """Pipeline hR: Vdisk peak / 2.15"""
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

def main():
    print("=" * 70)
    print("hR definition check: pipeline vs MRT photometric")
    print("=" * 70)

    pipe = load_pipeline()
    mrt = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}")

    # Build matched dataset
    names, gc_arr, vf_arr = [], [], []
    hR_pipe_arr, hR_mrt_arr = [], []

    for n in sorted(pipe.keys()):
        if n not in mrt: continue
        gd = pipe[n]; m = mrt[n]
        gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        hR_m = m['Rdisk']
        if gc <= 0 or vf <= 0 or hR_m <= 0: continue
        hR_p = compute_hR_pipeline(n, Yd)
        if hR_p is None: continue
        names.append(n); gc_arr.append(gc); vf_arr.append(vf)
        hR_pipe_arr.append(hR_p); hR_mrt_arr.append(hR_m)

    gc = np.array(gc_arr); vf = np.array(vf_arr)
    hR_p = np.array(hR_pipe_arr); hR_m = np.array(hR_mrt_arr)
    N = len(gc); log_gc = np.log10(gc)
    print(f"Matched: {N}")

    # T1: hR comparison
    print("\n" + "=" * 70)
    print("T1: hR comparison")
    print("=" * 70)
    ratio = hR_p / hR_m
    log_ratio = np.log10(ratio)
    print(f"  hR_pipe / hR_mrt:")
    print(f"    mean={np.mean(ratio):.4f}, median={np.median(ratio):.4f}, std={np.std(ratio):.4f}")
    print(f"    log: mean={np.mean(log_ratio):+.4f}, std={np.std(log_ratio):.4f} dex")
    print(f"  hR_pipe range: [{hR_p.min():.3f}, {hR_p.max():.3f}] kpc")
    print(f"  hR_mrt  range: [{hR_m.min():.3f}, {hR_m.max():.3f}] kpc")
    r_hR = np.corrcoef(np.log10(hR_p), np.log10(hR_m))[0, 1]
    print(f"  log-log r = {r_hR:.4f}")

    close1 = np.mean(np.abs(ratio - 1) < 0.01)
    close5 = np.mean(np.abs(ratio - 1) < 0.05)
    close10 = np.mean(np.abs(ratio - 1) < 0.10)
    print(f"  Within 1%: {close1*100:.1f}%, 5%: {close5*100:.1f}%, 10%: {close10*100:.1f}%")

    t_stat, p_val = stats.ttest_1samp(log_ratio, 0)
    print(f"  t-test (log_ratio=0): t={t_stat:.3f}, p={p_val:.2e}")

    # T2: outliers
    print("\n" + "=" * 70)
    print("T2: Outliers (|log ratio| > 2*std)")
    print("=" * 70)
    outlier = np.abs(log_ratio - np.mean(log_ratio)) > 2 * np.std(log_ratio)
    n_out = outlier.sum()
    print(f"  {n_out} outliers")
    if n_out <= 20:
        for i in np.where(outlier)[0]:
            print(f"    {names[i]:15s} pipe={hR_p[i]:.3f} mrt={hR_m[i]:.3f} ratio={ratio[i]:.3f}")

    # T3: gc prediction R^2
    print("\n" + "=" * 70)
    print("T3: gc prediction R^2 comparison")
    print("=" * 70)

    Sd_p = (vf * 1e3)**2 / (hR_p * kpc_m)
    Sd_m = (vf * 1e3)**2 / (hR_m * kpc_m)

    # Single variable: gc ~ Sigma_dyn^alpha
    def fit_a(sd):
        def r(p, x, y): return p[0] + p[1]*np.log10(a0*x) - y
        res = optimize.least_squares(r, [0, 0.5], args=(sd, log_gc),
                                      bounds=([-np.inf, 0.01], [np.inf, 2.0]))
        pred = res.x[0] + res.x[1]*np.log10(a0*sd)
        R2 = 1 - np.sum((log_gc-pred)**2)/np.sum((log_gc-log_gc.mean())**2)
        return res.x[1], R2, np.std(log_gc - pred)

    a_p, r2_p, sc_p = fit_a(Sd_p)
    a_m, r2_m, sc_m = fit_a(Sd_m)
    print(f"  Pipeline hR: alpha={a_p:.4f}, R2={r2_p:.4f}, scatter={sc_p:.4f}")
    print(f"  MRT Rdisk:   alpha={a_m:.4f}, R2={r2_m:.4f}, scatter={sc_m:.4f}")
    print(f"  Delta R2: {r2_p - r2_m:+.4f}")

    # 2-variable: gc ~ vflat + hR
    print("\n  2-variable: log gc = a + b1*log(vflat) + b2*log(hR)")
    from numpy.linalg import lstsq
    log_vf = np.log10(vf)
    Xp = np.column_stack([np.ones(N), log_vf, np.log10(hR_p)])
    Xm = np.column_stack([np.ones(N), log_vf, np.log10(hR_m)])
    bp, _, _, _ = lstsq(Xp, log_gc, rcond=None)
    bm, _, _, _ = lstsq(Xm, log_gc, rcond=None)
    pred_p = Xp @ bp; pred_m = Xm @ bm
    SS_tot = np.sum((log_gc - log_gc.mean())**2)
    R2_2p = 1 - np.sum((log_gc - pred_p)**2) / SS_tot
    R2_2m = 1 - np.sum((log_gc - pred_m)**2) / SS_tot
    print(f"  Pipeline: {bp[0]:.3f} + {bp[1]:.3f}*log(vf) + {bp[2]:.3f}*log(hR)  R2={R2_2p:.4f}")
    print(f"  MRT:      {bm[0]:.3f} + {bm[1]:.3f}*log(vf) + {bm[2]:.3f}*log(hR)  R2={R2_2m:.4f}")
    print(f"  Delta R2_2var: {R2_2p - R2_2m:+.4f}")

    # T4: circularity check
    print("\n" + "=" * 70)
    print("T4: Circularity check - is gc computed from vflat+hR?")
    print("=" * 70)
    # If gc = eta * sqrt(a0 * vflat^2/hR_pipe), then R2 should be ~1.0
    for eta_test in [0.729, 1.0, 1.24]:
        gc_calc = eta_test * np.sqrt(a0 * (vf*1e3)**2 / (hR_p * kpc_m))
        log_calc = np.log10(gc_calc)
        R2_calc = 1 - np.sum((log_gc - log_calc)**2) / SS_tot
        sc_calc = np.std(log_gc - log_calc)
        print(f"  eta={eta_test:.3f}: gc_calc R2={R2_calc:.6f}, scatter={sc_calc:.4f}")

    # With MRT hR
    print()
    for eta_test in [0.729, 1.0, 1.24]:
        gc_calc = eta_test * np.sqrt(a0 * (vf*1e3)**2 / (hR_m * kpc_m))
        log_calc = np.log10(gc_calc)
        R2_calc = 1 - np.sum((log_gc - log_calc)**2) / SS_tot
        sc_calc = np.std(log_gc - log_calc)
        print(f"  MRT eta={eta_test:.3f}: gc_calc R2={R2_calc:.6f}, scatter={sc_calc:.4f}")

    # T5: What is gc in TA3?
    print("\n" + "=" * 70)
    print("T5: TA3 gc definition check")
    print("=" * 70)
    # TA3 gc is from tanh fit to rotation curve, independent of hR
    # Check: gc vs vflat^4/(G*Mbar) relationship
    # If gc comes from BTFR-like: gc = vflat^4 / (G*Mbar*a0) -> gc depends on vflat only
    # Or if gc comes from RAR fit: gc_obs = gobs^2/gbar at each r -> median

    # Check correlation structure
    r_gc_vf = stats.pearsonr(log_gc, log_vf)[0]
    r_gc_hp = stats.pearsonr(log_gc, np.log10(hR_p))[0]
    r_gc_hm = stats.pearsonr(log_gc, np.log10(hR_m))[0]
    r_gc_sdp = stats.pearsonr(log_gc, np.log10(Sd_p))[0]
    r_gc_sdm = stats.pearsonr(log_gc, np.log10(Sd_m))[0]
    print(f"  r(gc, vflat)       = {r_gc_vf:.4f}")
    print(f"  r(gc, hR_pipe)     = {r_gc_hp:.4f}")
    print(f"  r(gc, hR_mrt)      = {r_gc_hm:.4f}")
    print(f"  r(gc, Sd_pipe)     = {r_gc_sdp:.4f}")
    print(f"  r(gc, Sd_mrt)      = {r_gc_sdm:.4f}")

    # Partial: gc ~ hR | vflat
    Xv = np.column_stack([np.ones(N), log_vf])
    b1, _, _, _ = lstsq(Xv, log_gc, rcond=None)
    b2p, _, _, _ = lstsq(Xv, np.log10(hR_p), rcond=None)
    b2m, _, _, _ = lstsq(Xv, np.log10(hR_m), rcond=None)
    rp_p = stats.pearsonr(log_gc - Xv@b1, np.log10(hR_p) - Xv@b2p)[0]
    rp_m = stats.pearsonr(log_gc - Xv@b1, np.log10(hR_m) - Xv@b2m)[0]
    print(f"\n  Partial r(gc, hR | vflat):")
    print(f"    pipeline hR: {rp_p:.4f}")
    print(f"    MRT Rdisk:   {rp_m:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  hR comparison:
    pipe/mrt: mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}
    log-log r = {r_hR:.4f}

  gc prediction:
    1-var: pipe R2={r2_p:.4f} (a={a_p:.3f}) vs mrt R2={r2_m:.4f} (a={a_m:.3f})
    2-var: pipe R2={R2_2p:.4f} vs mrt R2={R2_2m:.4f}

  Circularity (pipe hR):
    eta=0.729 R2 vs gc: see T4

  Partial correlation gc-hR|vflat:
    pipe: {rp_p:.4f}, mrt: {rp_m:.4f}
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        ax.scatter(hR_m, hR_p, s=10, alpha=0.5)
        lim = [min(hR_m.min(), hR_p.min())*0.8, max(hR_m.max(), hR_p.max())*1.2]
        ax.plot(lim, lim, 'r--')
        ax.set_xlabel('hR MRT [kpc]'); ax.set_ylabel('hR pipeline [kpc]')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'(a) hR comparison r={r_hR:.3f}')

        ax = axes[0, 1]
        ax.hist(log_ratio, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', ls='--')
        ax.axvline(np.mean(log_ratio), color='blue', label=f'mean={np.mean(log_ratio):+.3f}')
        ax.set_xlabel('log(hR_pipe/hR_mrt)'); ax.set_title('(b) ratio dist')
        ax.legend()

        ax = axes[1, 0]
        ax.scatter(np.log10(Sd_p), log_gc, s=10, alpha=0.4, c='steelblue', label=f'pipe R2={r2_p:.3f}')
        ax.scatter(np.log10(Sd_m), log_gc, s=10, alpha=0.4, c='salmon', label=f'mrt R2={r2_m:.3f}')
        ax.set_xlabel('log Sigma_dyn'); ax.set_ylabel('log gc')
        ax.set_title('(c) gc vs Sigma_dyn'); ax.legend()

        ax = axes[1, 1]
        ax.scatter(pred_p, log_gc, s=10, alpha=0.4, c='steelblue', label=f'pipe R2={R2_2p:.4f}')
        ax.scatter(pred_m, log_gc, s=10, alpha=0.4, c='salmon', label=f'mrt R2={R2_2m:.4f}')
        lim2 = [log_gc.min()-0.2, log_gc.max()+0.2]
        ax.plot(lim2, lim2, 'k--')
        ax.set_xlabel('predicted'); ax.set_ylabel('observed')
        ax.set_title('(d) 2-var prediction'); ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'hR_definition_check.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
