#!/usr/bin/env python3
"""f2h_empirical_slope_v2.py — reproduce official slope with Step 3 spec"""

import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar, brentq

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

G_SI = 6.67430e-11
MSUN = 1.98892e30
a0_SI = 1.2e-10
kpc_m = 3.08568e19
LN10 = np.log(10.0)

BINS = {
    1: {"label": "Bin1 [8.5,10.3)", "logM_lo": 8.5, "logM_hi": 10.3},
    2: {"label": "Bin2 [10.3,10.6)", "logM_lo": 10.3, "logM_hi": 10.6},
    3: {"label": "Bin3 [10.6,10.8)", "logM_lo": 10.6, "logM_hi": 10.8},
    4: {"label": "Bin4 [10.8,11.0)", "logM_lo": 10.8, "logM_hi": 11.0},
}
for b in BINS.values():
    b["logM_mid"] = 0.5 * (b["logM_lo"] + b["logM_hi"])
    b["M_gal"] = 10**b["logM_mid"] * MSUN


def rar_model_SI(g_bar, gc_SI):
    x = np.sqrt(g_bar / gc_SI)
    x = np.clip(x, 1e-10, 500.0)
    return g_bar / (1.0 - np.exp(-x))


def fit_gc_step3(g_bar, g_obs, g_obs_err):
    def chi2_func(log_gc):
        gc = 10**log_gc
        model = rar_model_SI(g_bar, gc)
        return np.sum(((g_obs - model) / g_obs_err)**2)

    res = minimize_scalar(chi2_func, bounds=(-14, -8), method='bounded')
    log_gc_best = res.x
    gc_best = 10**log_gc_best
    chi2_min = res.fun
    dof = len(g_bar) - 1

    target = chi2_min + 1.0
    try:
        log_lo = brentq(lambda lg: chi2_func(lg) - target, -14, log_gc_best, xtol=1e-6)
        gc_err_lo = gc_best - 10**log_lo
    except (ValueError, RuntimeError):
        gc_err_lo = 0.1 * gc_best
    try:
        log_hi = brentq(lambda lg: chi2_func(lg) - target, log_gc_best, -8, xtol=1e-6)
        gc_err_hi = 10**log_hi - gc_best
    except (ValueError, RuntimeError):
        gc_err_hi = 0.1 * gc_best

    gc_err = 0.5 * (gc_err_lo + gc_err_hi)
    return gc_best, gc_err, chi2_min, dof


def wls_slope(logM, gc_vals, gc_errs):
    y = np.log10(gc_vals)
    ye = gc_errs / (gc_vals * LN10)
    w = 1.0 / ye**2

    S = np.sum(w)
    Sx = np.sum(w * logM)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * logM**2)
    Sxy = np.sum(w * logM * y)

    det = S * Sxx - Sx**2
    slope = (S * Sxy - Sx * Sy) / det
    slope_err = np.sqrt(S / det)
    return slope, slope_err


def g_bar_to_R_kpc(g_bar, M_gal):
    R_m = np.sqrt(G_SI * M_gal / g_bar)
    return R_m / kpc_m


def main():
    basedir = os.path.dirname(os.path.abspath(__file__))

    data = {}
    for bnum in [1, 2, 3, 4]:
        fname = os.path.join(basedir, f"phase_b_combined_mbin_{bnum}.txt")
        arr = np.loadtxt(fname)
        g_bar = arr[:, 0]
        g_obs = arr[:, 1]
        g_err = arr[:, 2]
        n_pairs = arr[:, 3]

        mask = (np.isfinite(g_obs) & (g_err > 0) & np.isfinite(g_bar)
                & (g_bar > 0) & (n_pairs > 30))
        mask_nf = (np.isfinite(g_obs) & (g_err > 0)
                   & np.isfinite(g_bar) & (g_bar > 0))

        data[bnum] = {
            "g_bar": g_bar[mask], "g_obs": g_obs[mask],
            "g_err": g_err[mask], "n_pairs": n_pairs[mask],
            "R_kpc": g_bar_to_R_kpc(g_bar[mask], BINS[bnum]["M_gal"]),
            "g_bar_nf": g_bar[mask_nf], "g_obs_nf": g_obs[mask_nf],
            "g_err_nf": g_err[mask_nf], "n_pairs_nf": n_pairs[mask_nf],
            "R_kpc_nf": g_bar_to_R_kpc(g_bar[mask_nf], BINS[bnum]["M_gal"]),
        }
        print(f"Bin {bnum}: {mask_nf.sum()} valid -> {mask.sum()} after npair>30, "
              f"R={data[bnum]['R_kpc'].min():.0f}-{data[bnum]['R_kpc'].max():.0f} kpc")

    logM = np.array([BINS[b]["logM_mid"] for b in [1, 2, 3, 4]])

    print("\n" + "=" * 80)
    print("PART 1: Reproduce official slope +0.166 ± 0.041")
    print("=" * 80)

    print("\n--- (A) Step 3 spec: npair>30, SI gc, WLS ---")
    gc_A = {}
    for bnum in [1, 2, 3, 4]:
        d = data[bnum]
        gc, gce, chi2, dof = fit_gc_step3(d["g_bar"], d["g_obs"], d["g_err"])
        gc_A[bnum] = (gc, gce)
        print(f"  Bin{bnum}: gc={gc/a0_SI:.3f}±{gce/a0_SI:.3f} a0, "
              f"chi2/dof={chi2:.1f}/{dof}")

    gc_vals_A = np.array([gc_A[b][0] for b in [1,2,3,4]])
    gc_errs_A = np.array([gc_A[b][1] for b in [1,2,3,4]])
    slope_A, slope_A_err = wls_slope(logM, gc_vals_A, gc_errs_A)
    print(f"  -> slope_A = {slope_A:+.4f} ± {slope_A_err:.4f}")

    print("\n--- (B) No filter + SI gc ---")
    gc_B = {}
    for bnum in [1, 2, 3, 4]:
        d = data[bnum]
        gc, gce, _, _ = fit_gc_step3(d["g_bar_nf"], d["g_obs_nf"], d["g_err_nf"])
        gc_B[bnum] = (gc, gce)
        print(f"  Bin{bnum}: gc={gc/a0_SI:.3f}±{gce/a0_SI:.3f} a0")

    gc_vals_B = np.array([gc_B[b][0] for b in [1,2,3,4]])
    gc_errs_B = np.array([gc_B[b][1] for b in [1,2,3,4]])
    log_gc_B = np.log10(gc_vals_B)
    slope_B_uw = np.polyfit(logM, log_gc_B, 1)[0]
    slope_B_w, slope_B_w_err = wls_slope(logM, gc_vals_B, gc_errs_B)
    print(f"  -> polyfit:  {slope_B_uw:+.4f}")
    print(f"  -> WLS:      {slope_B_w:+.4f} ± {slope_B_w_err:.4f}")

    print("\n--- (C) npair>30 + polyfit ---")
    log_gc_A = np.log10(gc_vals_A)
    slope_C_uw = np.polyfit(logM, log_gc_A, 1)[0]
    print(f"  -> polyfit: {slope_C_uw:+.4f}")

    print("\n" + "=" * 80)
    print("ATTRIBUTION")
    print("=" * 80)
    print(f"  Official Step 3:         +0.1660 ± 0.0410")
    print(f"  (A) Full Step 3 repro:   {slope_A:+.4f} ± {slope_A_err:.4f}")
    print(f"  (B) No filter polyfit:   {slope_B_uw:+.4f}  (= v1)")
    print(f"  (B) No filter WLS:       {slope_B_w:+.4f} ± {slope_B_w_err:.4f}")
    print(f"  (C) npair>30 polyfit:    {slope_C_uw:+.4f}")
    print(f"  npair effect (A-B_WLS):  {slope_A - slope_B_w:+.4f}")
    print(f"  WLS effect (A-C_poly):   {slope_A - slope_C_uw:+.4f}")

    print("\n" + "=" * 80)
    print("PART 2: 2-halo slope (Step 3 spec, npair>30)")
    print("=" * 80)

    R_cuts = [100, 150, 200, 250, 300, 400, 500]
    results = {}

    for R_cut in R_cuts:
        gc_inner = {}
        n_inner = {}

        for bnum in [1, 2, 3, 4]:
            d = data[bnum]
            mask_in = d["R_kpc"] < R_cut
            n_in = mask_in.sum()
            n_inner[bnum] = n_in

            if n_in >= 3:
                gc, gce, _, _ = fit_gc_step3(
                    d["g_bar"][mask_in], d["g_obs"][mask_in], d["g_err"][mask_in])
                gc_inner[bnum] = (gc, gce)
            else:
                gc_inner[bnum] = (np.nan, np.nan)

        gc_i_vals = np.array([gc_inner[b][0] for b in [1,2,3,4]])
        gc_i_errs = np.array([gc_inner[b][1] for b in [1,2,3,4]])
        valid = np.isfinite(gc_i_vals) & (gc_i_vals > 0) & (gc_i_errs > 0)

        if valid.sum() >= 3:
            slope_inner, slope_inner_err = wls_slope(
                logM[valid], gc_i_vals[valid], gc_i_errs[valid])
        else:
            slope_inner = slope_inner_err = float('nan')

        delta_slope = slope_A - slope_inner
        delta_err = np.sqrt(slope_A_err**2 + slope_inner_err**2)

        results[R_cut] = {
            "gc_inner": gc_inner, "n_inner": n_inner,
            "slope_inner": slope_inner, "slope_inner_err": slope_inner_err,
            "delta_slope": delta_slope, "delta_err": delta_err,
        }

        sig = abs(delta_slope) / max(delta_err, 1e-10)
        print(f"  R_cut={R_cut:4d}: slope_in={slope_inner:+.4f}±{slope_inner_err:.4f}, "
              f"Δ={delta_slope:+.4f}±{delta_err:.4f} ({sig:.1f}σ), "
              f"n={[n_inner[b] for b in [1,2,3,4]]}")

    R_ref = 200
    print("\n" + "=" * 80)
    print(f"PART 3: f_2h(R) profile (R_cut={R_ref} kpc)")
    print("=" * 80)
    for bnum in [1, 2, 3, 4]:
        d = data[bnum]
        gc_ref = results[R_ref]["gc_inner"][bnum][0]
        if not np.isfinite(gc_ref):
            print(f"  Bin{bnum}: gc_inner undefined")
            continue
        model_1h = rar_model_SI(d["g_bar"], gc_ref)
        excess = d["g_obs"] - model_1h
        f_2h = excess / d["g_obs"]
        print(f"\n  Bin{bnum} (logM*={BINS[bnum]['logM_mid']:.1f}, "
              f"gc_1h={gc_ref/a0_SI:.3f} a0):")
        print(f"  {'R(kpc)':>8s} {'g_obs':>12s} {'model_1h':>12s} {'f_2h':>7s} {'npair':>8s}")
        for i in range(len(d["R_kpc"])):
            print(f"  {d['R_kpc'][i]:8.0f} {d['g_obs'][i]:12.4e} "
                  f"{model_1h[i]:12.4e} {f_2h[i]:7.3f} {d['n_pairs'][i]:8.0f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors_bin = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

        ax = axes[0, 0]
        ax.errorbar(logM, gc_vals_A/a0_SI, yerr=gc_errs_A/a0_SI, fmt='o-', color='red',
                    label=f'npair>30 WLS: {slope_A:+.3f}', capsize=4, ms=8)
        ax.errorbar(logM+0.02, gc_vals_B/a0_SI, yerr=gc_errs_B/a0_SI,
                    fmt='s--', color='blue',
                    label=f'no filter poly: {slope_B_uw:+.3f}', capsize=4, ms=8, alpha=0.7)
        ax.set_xlabel('log M*'); ax.set_ylabel('gc (a0)')
        ax.set_title('(a) Slope reproduction'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        rcs = list(results.keys())
        ds = [results[rc]["delta_slope"] for rc in rcs]
        dse = [results[rc]["delta_err"] for rc in rcs]
        ax.errorbar(rcs, ds, yerr=dse, fmt='o-', color='green', capsize=4, ms=8)
        ax.axhline(0.091, color='red', ls='--', lw=1.5, label='Target +0.091')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('R_cut (kpc)'); ax.set_ylabel('Delta slope')
        ax.set_title('(b) 2-halo contribution (Step 3 spec)')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for idx, bnum in enumerate([1, 2, 3, 4]):
            d = data[bnum]
            gc_ref = results[R_ref]["gc_inner"][bnum][0]
            if not np.isfinite(gc_ref):
                continue
            model_1h = rar_model_SI(d["g_bar"], gc_ref)
            f_2h = (d["g_obs"] - model_1h) / d["g_obs"]
            ax.plot(d["R_kpc"], f_2h, 'o-', color=colors_bin[idx], ms=5,
                    label=f'Bin{bnum}')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('R (kpc)'); ax.set_ylabel('f_2h')
        ax.set_title('(c) 2-halo fraction f_2h(R)')
        ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for idx, bnum in enumerate([1, 2, 3, 4]):
            d = data[bnum]
            ax.semilogy(d["R_kpc_nf"], d["n_pairs_nf"], 'o', color=colors_bin[idx],
                        ms=5, alpha=0.7, label=f'Bin{bnum}')
        ax.axhline(30, color='red', ls='--', lw=1.5, label='npair=30')
        ax.set_xlabel('R (kpc)'); ax.set_ylabel('N_pairs')
        ax.set_title('(d) N_pairs vs R')
        ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfig = os.path.join(basedir, "f2h_empirical_slope_v2.png")
        plt.savefig(outfig, dpi=150)
        print(f"\nFig: {outfig}")
    except ImportError:
        pass

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Official Step 3:   +0.166 ± 0.041")
    print(f"  (A) Full repro:    {slope_A:+.4f} ± {slope_A_err:.4f}")
    print(f"  (B) no filter:     {slope_B_uw:+.4f}")
    print(f"  Gap (A-official):  {slope_A - 0.166:+.4f}")
    print(f"\n  slope_all:    {slope_A:+.4f} ± {slope_A_err:.4f}")
    print(f"  slope_inner:  {results[200]['slope_inner']:+.4f} ± {results[200]['slope_inner_err']:.4f}")
    print(f"  Delta_slope:  {results[200]['delta_slope']:+.4f} ± {results[200]['delta_err']:.4f}")


if __name__ == "__main__":
    main()
