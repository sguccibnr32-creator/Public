#!/usr/bin/env python3
"""f2h_empirical_slope.py — empirical 2-halo slope from HSC Phase B data"""

import os
import sys
import numpy as np
from scipy.optimize import minimize

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

G_SI = 6.67430e-11
MSUN = 1.98892e30
a0_SI = 1.2e-10
kpc_m = 3.08568e19

BINS = {
    1: {"label": "Bin1 [8.5,10.3)", "logM_lo": 8.5, "logM_hi": 10.3},
    2: {"label": "Bin2 [10.3,10.6)", "logM_lo": 10.3, "logM_hi": 10.6},
    3: {"label": "Bin3 [10.6,10.8)", "logM_lo": 10.6, "logM_hi": 10.8},
    4: {"label": "Bin4 [10.8,11.0)", "logM_lo": 10.8, "logM_hi": 11.0},
}
for b in BINS.values():
    b["logM_mid"] = 0.5 * (b["logM_lo"] + b["logM_hi"])
    b["M_gal"] = 10**b["logM_mid"] * MSUN


def rar_model(g_bar, gc_a0):
    gc_SI = gc_a0 * a0_SI
    x = np.sqrt(g_bar / gc_SI)
    x = np.clip(x, 1e-10, 500.0)
    return g_bar / (1.0 - np.exp(-x))


def fit_gc(g_bar, g_obs, g_obs_err, gc_init=1.0):
    def chi2_func(params):
        gc_a0 = params[0]
        if gc_a0 <= 0.01 or gc_a0 > 100:
            return 1e20
        model = rar_model(g_bar, gc_a0)
        resid = (g_obs - model) / g_obs_err
        return np.sum(resid**2)

    result = minimize(chi2_func, [gc_init], method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 5000})
    gc_best = result.x[0]
    chi2_val = result.fun

    eps = 0.01
    c0 = chi2_func([gc_best])
    cp = chi2_func([gc_best + eps])
    cm = chi2_func([gc_best - eps])
    d2 = (cp + cm - 2*c0) / eps**2
    gc_err = np.sqrt(2.0 / max(d2, 1e-20)) if d2 > 0 else 0.5

    return gc_best, gc_err, chi2_val, len(g_bar) - 1


def g_bar_to_R_kpc(g_bar, M_gal):
    R_m = np.sqrt(G_SI * M_gal / g_bar)
    return R_m / kpc_m


def main():
    basedir = os.path.dirname(os.path.abspath(__file__))

    data = {}
    for bnum in [1, 2, 3, 4]:
        fname = os.path.join(basedir, f"phase_b_combined_mbin_{bnum}.txt")
        if not os.path.isfile(fname):
            print(f"ERROR: {fname} not found")
            return
        arr = np.loadtxt(fname)
        g_bar = arr[:, 0]
        g_obs = arr[:, 1]
        g_err = arr[:, 2]
        n_pairs = arr[:, 3]
        mask = (g_err > 0) & np.isfinite(g_obs) & np.isfinite(g_bar) & (g_bar > 0)
        data[bnum] = {
            "g_bar": g_bar[mask],
            "g_obs": g_obs[mask],
            "g_err": g_err[mask],
            "n_pairs": n_pairs[mask],
            "R_kpc": g_bar_to_R_kpc(g_bar[mask], BINS[bnum]["M_gal"]),
        }
        print(f"Bin {bnum}: {mask.sum()} pts, "
              f"R = {data[bnum]['R_kpc'].min():.0f}-{data[bnum]['R_kpc'].max():.0f} kpc")

    R_cuts = [100, 150, 200, 250, 300, 400, 500]

    print("\n" + "="*80)
    print("R_cut scan")
    print("="*80)

    results = {}
    for R_cut in R_cuts:
        gc_inner = {}
        gc_all = {}
        n_inner = {}

        for bnum in [1, 2, 3, 4]:
            d = data[bnum]
            gc_a, gc_a_e, _, _ = fit_gc(d["g_bar"], d["g_obs"], d["g_err"])
            gc_all[bnum] = (gc_a, gc_a_e)

            mask_in = d["R_kpc"] < R_cut
            n_in = mask_in.sum()
            n_inner[bnum] = n_in
            if n_in >= 3:
                gc_i, gc_i_e, _, _ = fit_gc(
                    d["g_bar"][mask_in], d["g_obs"][mask_in], d["g_err"][mask_in])
                gc_inner[bnum] = (gc_i, gc_i_e)
            else:
                gc_inner[bnum] = (np.nan, np.nan)

        logM = np.array([BINS[b]["logM_mid"] for b in [1,2,3,4]])
        gc_a_vals = np.array([gc_all[b][0] for b in [1,2,3,4]])
        gc_a_errs = np.array([gc_all[b][1] for b in [1,2,3,4]])
        gc_i_vals = np.array([gc_inner[b][0] for b in [1,2,3,4]])
        gc_i_errs = np.array([gc_inner[b][1] for b in [1,2,3,4]])

        valid = np.isfinite(gc_i_vals) & (gc_i_vals > 0)
        if valid.sum() >= 3 and np.all(gc_a_vals > 0):
            log_gc_a = np.log10(gc_a_vals)
            log_gc_i = np.log10(gc_i_vals[valid])
            logM_v = logM[valid]
            slope_all, _ = np.polyfit(logM, log_gc_a, 1)
            slope_inner, _ = np.polyfit(logM_v, log_gc_i, 1)

            n_boot = 2000
            slopes_all_boot = []
            slopes_inner_boot = []
            for _ in range(n_boot):
                gc_a_pert = gc_a_vals + np.random.normal(0, gc_a_errs)
                gc_a_pert = np.clip(gc_a_pert, 0.1, 100)
                s_a, _ = np.polyfit(logM, np.log10(gc_a_pert), 1)
                slopes_all_boot.append(s_a)

                gc_i_pert = gc_i_vals[valid] + np.random.normal(0, gc_i_errs[valid])
                gc_i_pert = np.clip(gc_i_pert, 0.1, 100)
                s_i, _ = np.polyfit(logM_v, np.log10(gc_i_pert), 1)
                slopes_inner_boot.append(s_i)

            slope_all_err = np.std(slopes_all_boot)
            slope_inner_err = np.std(slopes_inner_boot)
            delta_slope = slope_all - slope_inner
            delta_slope_err = np.sqrt(slope_all_err**2 + slope_inner_err**2)
        else:
            slope_all = slope_inner = delta_slope = np.nan
            slope_all_err = slope_inner_err = delta_slope_err = np.nan

        results[R_cut] = {
            "gc_all": gc_all, "gc_inner": gc_inner, "n_inner": n_inner,
            "slope_all": slope_all, "slope_all_err": slope_all_err,
            "slope_inner": slope_inner, "slope_inner_err": slope_inner_err,
            "delta_slope": delta_slope, "delta_slope_err": delta_slope_err,
        }

        print(f"\n--- R_cut = {R_cut} kpc ---")
        for bnum in [1,2,3,4]:
            gc_a, gc_a_e = gc_all[bnum]
            gc_i, gc_i_e = gc_inner[bnum]
            n_in = n_inner[bnum]
            f2h = (gc_a - gc_i) / gc_a if np.isfinite(gc_i) and gc_a > 0 else np.nan
            print(f"  Bin{bnum}: gc_all={gc_a:.3f}±{gc_a_e:.3f}, "
                  f"gc_inner={gc_i:.3f}±{gc_i_e:.3f} ({n_in} pts), "
                  f"f_2h={f2h:.3f}")
        print(f"  slope_all    = {slope_all:+.4f} ± {slope_all_err:.4f}")
        print(f"  slope_inner  = {slope_inner:+.4f} ± {slope_inner_err:.4f}")
        sig = abs(delta_slope) / max(delta_slope_err, 1e-10)
        print(f"  Delta_slope  = {delta_slope:+.4f} ± {delta_slope_err:.4f} "
              f"({sig:.1f} sigma)")

    print("\n" + "="*80)
    print("f_2h(R) profile per M* bin (gc_inner at R_cut=200 kpc)")
    print("="*80)
    R_ref = 200
    for bnum in [1, 2, 3, 4]:
        d = data[bnum]
        gc_ref = results[R_ref]["gc_inner"][bnum][0]
        if not np.isfinite(gc_ref):
            print(f"  Bin{bnum}: gc_inner undefined")
            continue
        model_1h = rar_model(d["g_bar"], gc_ref)
        excess = d["g_obs"] - model_1h
        f_2h = excess / d["g_obs"]
        print(f"\n  Bin{bnum} (logM*={BINS[bnum]['logM_mid']:.1f}, gc_1h={gc_ref:.3f} a0):")
        print(f"  {'R(kpc)':>10s}  {'g_obs':>12s}  {'model_1h':>12s}  {'excess':>12s}  {'f_2h':>8s}")
        for i in range(len(d["R_kpc"])):
            print(f"  {d['R_kpc'][i]:10.0f}  {d['g_obs'][i]:12.4e}  "
                  f"{model_1h[i]:12.4e}  {excess[i]:12.4e}  {f_2h[i]:8.3f}")

    print("\n" + "="*80)
    print("SUMMARY: 2-halo slope contribution")
    print("="*80)
    print(f"\n{'R_cut':>8s}  {'slope_all':>12s}  {'slope_inner':>12s}  "
          f"{'Delta_slope':>20s}  {'sig':>6s}")
    for R_cut in R_cuts:
        r = results[R_cut]
        sig = abs(r["delta_slope"]) / max(r["delta_slope_err"], 1e-10)
        print(f"{R_cut:8d}  {r['slope_all']:+12.4f}  {r['slope_inner']:+12.4f}  "
              f"{r['delta_slope']:+8.4f}±{r['delta_slope_err']:.4f}  {sig:6.1f}")

    print(f"\nHSC: +0.166 ± 0.041, C15: +0.075, Target Delta_slope: +0.091, Model B: +0.017")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        r200 = results[200]
        logM_arr = [BINS[b]["logM_mid"] for b in [1,2,3,4]]
        gc_a = [r200["gc_all"][b][0] for b in [1,2,3,4]]
        gc_a_e = [r200["gc_all"][b][1] for b in [1,2,3,4]]
        gc_i = [r200["gc_inner"][b][0] for b in [1,2,3,4]]
        gc_i_e = [r200["gc_inner"][b][1] for b in [1,2,3,4]]
        ax.errorbar(logM_arr, gc_a, yerr=gc_a_e, fmt='o-', color='red',
                     label=f'all (slope={r200["slope_all"]:+.3f})', capsize=3)
        ax.errorbar(logM_arr, gc_i, yerr=gc_i_e, fmt='s-', color='blue',
                     label=f'inner R<200 (slope={r200["slope_inner"]:+.3f})', capsize=3)
        ax.set_xlabel('log M*'); ax.set_ylabel('gc (a0)')
        ax.set_title('gc vs M*: all vs inner')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        r_cuts_arr = list(results.keys())
        ds = [results[rc]["delta_slope"] for rc in r_cuts_arr]
        ds_e = [results[rc]["delta_slope_err"] for rc in r_cuts_arr]
        ax.errorbar(r_cuts_arr, ds, yerr=ds_e, fmt='o-', color='green', capsize=3)
        ax.axhline(0.091, color='red', ls='--', label='Target +0.091')
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R_cut (kpc)'); ax.set_ylabel('Delta slope')
        ax.set_title('2-halo contribution vs R_cut')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        colors = ['blue', 'green', 'orange', 'red']
        for idx, bnum in enumerate([1, 2, 3, 4]):
            d = data[bnum]
            gc_ref = r200["gc_inner"][bnum][0]
            if not np.isfinite(gc_ref):
                continue
            model_1h = rar_model(d["g_bar"], gc_ref)
            f_2h = (d["g_obs"] - model_1h) / d["g_obs"]
            ax.plot(d["R_kpc"], f_2h, 'o-', color=colors[idx], markersize=4,
                    label=f'Bin{bnum}')
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R (kpc)'); ax.set_ylabel('f_2h')
        ax.set_title('2-halo fraction f_2h(R)')
        ax.set_xscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for idx, bnum in enumerate([1, 2, 3, 4]):
            d = data[bnum]
            gc_ref = r200["gc_inner"][bnum][0]
            ax.errorbar(d["g_bar"], d["g_obs"], yerr=d["g_err"],
                        fmt='o', color=colors[idx], markersize=4, alpha=0.7, capsize=2,
                        label=f'Bin{bnum}')
            if np.isfinite(gc_ref):
                gb_fine = np.logspace(np.log10(d["g_bar"].min()),
                                      np.log10(d["g_bar"].max()), 100)
                ax.plot(gb_fine, rar_model(gb_fine, gc_ref),
                        '-', color=colors[idx], alpha=0.5)
        gb_range = np.logspace(-16, -10, 100)
        ax.plot(gb_range, gb_range, 'k--', alpha=0.3, label='g_obs=g_bar')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('g_bar'); ax.set_ylabel('g_obs')
        ax.set_title('RAR + 1h model')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfig = os.path.join(basedir, "f2h_empirical_slope.png")
        plt.savefig(outfig, dpi=150)
        print(f"\nFigure: {outfig}")
    except ImportError:
        pass

    print("\n" + "="*80)
    print("CONCLUSION (R_cut=200 kpc)")
    print("="*80)
    r200 = results[200]
    print(f"  slope_all    = {r200['slope_all']:+.4f} ± {r200['slope_all_err']:.4f}")
    print(f"  slope_inner  = {r200['slope_inner']:+.4f} ± {r200['slope_inner_err']:.4f}")
    print(f"  Delta_slope  = {r200['delta_slope']:+.4f} ± {r200['delta_slope_err']:.4f}")
    print(f"  Target:        +0.091")


if __name__ == "__main__":
    main()
