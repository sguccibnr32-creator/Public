# -*- coding: utf-8 -*-
"""
brouwer_kids_c15_radial.py
Follow-up: Radial dependence of gc in KiDS lensing RAR
"""

import os, math
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

a0 = 1.2e-10
G_SI = 6.674e-11
Msun = 1.989e30
pc_m = 3.0857e16
kpc_m = 3.0857e19
Mpc_m = 3.0857e22

BASE = Path(os.path.dirname(os.path.abspath(__file__)))


def load_esd(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'col1': data[:, 0], 'ESD_t': data[:, 1], 'ESD_x': data[:, 2],
        'error': data[:, 3], 'bias': data[:, 4], 'n': len(data),
    }


def esd_to_gobs(ESD_t, bias):
    return 5.580e-13 * ESD_t / bias


def esd_to_gobs_err(error, bias):
    return 5.580e-13 * error / bias


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4*gc*g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def fit_gc(g_bar, g_obs, g_err):
    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(g_bar, gc)
        res = np.log10(g_obs) - np.log10(gp)
        w = 1.0 / np.maximum((g_err / (g_obs * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)
    result = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**result.x, result.fun


def esd_to_vcirc(ESD_t, bias, R_Mpc):
    v2 = 4 * 4.52e-30 * (ESD_t / bias) * R_Mpc * 1e6 * (3.086e13)**2
    return np.sqrt(np.maximum(v2, 0))


def gobs_from_vcirc(v_kms, R_Mpc):
    v_m = v_kms * 1e3
    R_m = R_Mpc * Mpc_m
    return v_m**2 / R_m


def gbar_point_mass(Mstar_Msun, R_Mpc):
    R_m = R_Mpc * Mpc_m
    return G_SI * Mstar_Msun * Msun / R_m**2


MSTAR_BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
MSTAR_BIN_LOG_CENTERS = [0.5*(MSTAR_BIN_EDGES[i]+MSTAR_BIN_EDGES[i+1]) for i in range(4)]
MSTAR_BIN_CENTERS_MSUN = [10**lm for lm in MSTAR_BIN_LOG_CENTERS]


def main():
    print("=" * 70)
    print("Brouwer+2021 KiDS: Radial gc Analysis & Hot Gas Check")
    print("=" * 70)

    print("\n--- Q1: HOT GAS CORRECTION ---")
    f_no_hg = BASE / "Fig-4-5-C1_KiDS-isolated_Nobins.txt"
    f_hg = BASE / "Fig-4_KiDS-isolated_hotgas_Nobins.txt"

    for label, fpath in [("No hot gas", f_no_hg), ("With hot gas", f_hg)]:
        if not fpath.exists():
            print(f"  [{label}] File not found: {fpath.name}")
            continue
        d = load_esd(fpath)
        gb = d['col1']
        go = esd_to_gobs(d['ESD_t'], d['bias'])
        ge = esd_to_gobs_err(d['error'], d['bias'])
        valid = (gb > 0) & (go > 0) & np.isfinite(go) & (ge > 0)
        gb, go, ge = gb[valid], go[valid], ge[valid]
        gc_fit, chi2_val = fit_gc(gb, go, ge)
        res_mond = np.log10(go / mond_rar(gb))
        res_memb = np.log10(go / membrane_rar(gb, gc_fit))
        print(f"\n  {label} ({fpath.name}):")
        print(f"    N = {len(gb)}, g_bar range: [{gb.min():.2e}, {gb.max():.2e}]")
        print(f"    gc_fit = {gc_fit/a0:.4f} a0")
        print(f"    MOND residual:     median={np.median(res_mond):+.4f}, std={np.std(res_mond):.4f}")
        print(f"    Membrane residual: median={np.median(res_memb):+.4f}, std={np.std(res_memb):.4f}")

    print("\n" + "=" * 70)
    print("Q2: g_bar-LIMITED gc FIT (= RADIAL CUT PROXY)")
    print("=" * 70)
    print("Higher g_bar = smaller R (1-halo); Lower g_bar = larger R (2-halo)")

    if f_no_hg.exists():
        d = load_esd(f_no_hg)
        gb_all = d['col1']
        go_all = esd_to_gobs(d['ESD_t'], d['bias'])
        ge_all = esd_to_gobs_err(d['error'], d['bias'])
        valid = (gb_all > 0) & (go_all > 0) & np.isfinite(go_all) & (ge_all > 0)
        gb_all, go_all, ge_all = gb_all[valid], go_all[valid], ge_all[valid]
        idx = np.argsort(gb_all)
        gb_s, go_s, ge_s = gb_all[idx], go_all[idx], ge_all[idx]

        print(f"\nAll {len(gb_s)} points sorted by g_bar:")
        for i in range(len(gb_s)):
            print(f"  Point {i+1:2d}: g_bar = {gb_s[i]:.2e} m/s^2 "
                  f"({gb_s[i]/a0:.2e} a0)  g_obs = {go_s[i]:.2e}")

        print(f"\n{'Cut':>20s} {'N':>3s} {'gc/a0':>8s} {'Note':>25s}")
        gbar_cuts = [1e-12, 5e-13, 1e-13, 5e-14, 1e-14, 5e-15, 0]
        for gcut in gbar_cuts:
            mask = gb_all >= gcut
            if mask.sum() < 3:
                continue
            gc_cut, _ = fit_gc(gb_all[mask], go_all[mask], ge_all[mask])
            if gcut > 1e-12:
                regime = "near-galaxy (<100kpc)"
            elif gcut > 1e-13:
                regime = "halo (100-500kpc)"
            elif gcut > 1e-14:
                regime = "group (0.5-2Mpc)"
            else:
                regime = "all radii"
            print(f"  g_bar > {gcut:.0e}  {mask.sum():3d}  {gc_cut/a0:8.4f}  {regime:>25s}")

        print(f"\n--- Sliding window gc (5-point windows) ---")
        print(f"{'Window':>12s} {'g_bar_mid':>12s} {'g_bar_mid/a0':>12s} {'gc/a0':>8s}")
        n_pts = len(gb_s)
        window = min(5, n_pts - 1)
        for start in range(n_pts - window + 1):
            end = start + window
            gb_w = gb_s[start:end]
            go_w = go_s[start:end]
            ge_w = ge_s[start:end]
            if len(gb_w) >= 3:
                gc_w, _ = fit_gc(gb_w, go_w, ge_w)
                gb_mid = np.exp(np.mean(np.log(gb_w)))
                print(f"  pts {start+1:2d}-{end:2d}  {gb_mid:12.2e}  {gb_mid/a0:12.2e}  {gc_w/a0:8.4f}")

    print("\n" + "=" * 70)
    print("Q3: ROTATION CURVE FILES (EXPLICIT R [Mpc])")
    print("=" * 70)

    rc_results = {}
    for i in range(4):
        fname = BASE / f"Fig-3_KiDS-isolated_Massbin-{i+1}.txt"
        if not fname.exists():
            fname = BASE / f"Fig-3_Lensing-rotation-curves_Massbin-{i+1}.txt"
        if not fname.exists():
            print(f"  Bin {i+1}: RC file not found")
            continue

        d = load_esd(fname)
        R_Mpc = d['col1']
        ESD_t = d['ESD_t']
        bias = d['bias']
        error = d['error']

        v_circ = esd_to_vcirc(ESD_t, bias, R_Mpc)
        g_obs_r = gobs_from_vcirc(v_circ, R_Mpc)
        Mstar = MSTAR_BIN_CENTERS_MSUN[i]
        g_bar_r = gbar_point_mass(Mstar, R_Mpc)
        v_err = v_circ * (error / (2 * np.maximum(ESD_t, 1e-10)))
        g_err_r = 2 * g_obs_r * v_err / np.maximum(v_circ, 1e-10)

        valid = (R_Mpc > 0) & (g_obs_r > 0) & np.isfinite(g_obs_r) & (g_bar_r > 0)

        print(f"\n  Bin {i+1} (logM* = {MSTAR_BIN_LOG_CENTERS[i]:.1f}):")
        print(f"    R range: [{R_Mpc[valid].min():.3f}, {R_Mpc[valid].max():.3f}] Mpc "
              f"= [{R_Mpc[valid].min()*1000:.0f}, {R_Mpc[valid].max()*1000:.0f}] kpc")
        print(f"    v_circ range: [{v_circ[valid].min():.1f}, {v_circ[valid].max():.1f}] km/s")

        R_cuts_kpc = [50, 100, 200, 300, 500, 1000, 3000]
        print(f"    {'R_max [kpc]':>12s} {'N':>3s} {'gc/a0':>8s} {'regime':>15s}")
        for Rmax_kpc in R_cuts_kpc:
            Rmax_Mpc = Rmax_kpc / 1000.0
            mask = valid & (R_Mpc <= Rmax_Mpc)
            if mask.sum() < 3:
                continue
            gb_m = g_bar_r[mask]
            go_m = g_obs_r[mask]
            ge_m = np.maximum(g_err_r[mask], go_m * 0.1)
            gc_r, _ = fit_gc(gb_m, go_m, ge_m)
            regime = "1-halo" if Rmax_kpc <= 300 else ("transition" if Rmax_kpc <= 1000 else "2-halo incl.")
            print(f"    {Rmax_kpc:12d} {mask.sum():3d} {gc_r/a0:8.4f}  {regime:>15s}")

        rc_results[i] = {
            'R': R_Mpc[valid], 'g_obs': g_obs_r[valid],
            'g_bar': g_bar_r[valid], 'v_circ': v_circ[valid],
        }

    print("\n" + "=" * 70)
    print("Q4: LOCAL gc AT EACH RADIUS (per-point inversion)")
    print("=" * 70)
    print("From g_obs = (g_bar + sqrt(g_bar^2 + 4*gc*g_bar))/2, solve for gc:")
    print("gc = (g_obs - g_bar)^2 / g_bar  [valid when g_obs > g_bar]")

    if f_no_hg.exists():
        d = load_esd(f_no_hg)
        gb = d['col1']
        go = esd_to_gobs(d['ESD_t'], d['bias'])
        ge = esd_to_gobs_err(d['error'], d['bias'])
        valid = (gb > 0) & (go > gb) & np.isfinite(go)
        gb, go, ge = gb[valid], go[valid], ge[valid]
        gc_local = (go - gb)**2 / gb

        print(f"\n{'#':>3s} {'g_bar/a0':>12s} {'g_obs/a0':>12s} {'gc_local/a0':>12s} {'regime':>15s}")
        for j in range(len(gb)):
            regime = "deep MOND" if gb[j] < 0.01*a0 else ("MOND trans." if gb[j] < a0 else "Newtonian")
            print(f"  {j+1:2d}  {gb[j]/a0:12.2e}  {go[j]/a0:12.4f}  {gc_local[j]/a0:12.4f}  {regime:>15s}")

        print(f"\n  gc_local statistics:")
        print(f"    median = {np.median(gc_local)/a0:.4f} a0")
        print(f"    mean   = {np.mean(gc_local)/a0:.4f} a0")
        print(f"    std    = {np.std(gc_local)/a0:.4f} a0")
        print(f"    range  = [{gc_local.min()/a0:.4f}, {gc_local.max()/a0:.4f}] a0")

        log_gb = np.log10(gb)
        log_gc = np.log10(gc_local)
        if len(log_gb) >= 4:
            coeffs = np.polyfit(log_gb, log_gc, 1)
            print(f"\n  log(gc_local) vs log(g_bar) slope: {coeffs[0]:+.3f}")
            if abs(coeffs[0]) < 0.1:
                print(f"    -> gc_local approximately constant (MOND-like)")
            else:
                print(f"    -> gc_local varies with scale (non-trivial)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        ax = axs[0, 0]
        for label, fpath, col, mk in [
            ("No hot gas", f_no_hg, 'black', 'o'),
            ("With hot gas", f_hg, 'red', 's')
        ]:
            if not fpath.exists():
                continue
            d = load_esd(fpath)
            gb = d['col1']
            go = esd_to_gobs(d['ESD_t'], d['bias'])
            ge = esd_to_gobs_err(d['error'], d['bias'])
            v = (gb > 0) & (go > 0) & np.isfinite(go)
            ax.errorbar(gb[v]/a0, go[v]/a0, yerr=ge[v]/a0, fmt=mk, color=col,
                        ms=5, capsize=2, label=label)
        gb_r = np.logspace(-6, 0, 200) * a0
        ax.loglog(gb_r/a0, mond_rar(gb_r)/a0, 'g-', lw=1.5, label='MOND')
        ax.loglog(gb_r/a0, gb_r/a0, 'k:', lw=0.5)
        ax.set_xlabel('g_bar / a0')
        ax.set_ylabel('g_obs / a0')
        ax.set_title('Q1: Hot Gas Correction Effect')
        ax.legend(fontsize=7)
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(1e-3, 10)
        ax.grid(True, which='both', alpha=0.2)

        ax = axs[0, 1]
        if f_no_hg.exists():
            d = load_esd(f_no_hg)
            gb = d['col1']
            go = esd_to_gobs(d['ESD_t'], d['bias'])
            v = (gb > 0) & (go > gb) & np.isfinite(go)
            gb, go = gb[v], go[v]
            gc_loc = (go - gb)**2 / gb
            ax.semilogx(gb/a0, gc_loc/a0, 'ko-', ms=6)
            ax.axhline(1.0, color='g', ls='--', lw=1, label='MOND (gc=a0)')
            ax.axhline(0.24, color='gray', ls=':', lw=1, label='SPARC median')
            ax.set_xlabel('g_bar / a0')
            ax.set_ylabel('gc_local / a0')
            ax.set_title('Q4: Local gc at Each Acceleration Scale')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        ax = axs[1, 0]
        if rc_results:
            colors = ['blue', 'green', 'orange', 'red']
            for i, res in rc_results.items():
                R_kpc = res['R'] * 1000
                ax.loglog(R_kpc, res['g_obs']/a0, 'o-', color=colors[i], ms=4,
                          label=f'Bin {i+1} (logM*={MSTAR_BIN_LOG_CENTERS[i]:.1f})')
            ax.set_xlabel('R [kpc]')
            ax.set_ylabel('g_obs / a0')
            ax.set_title('Q3: Radial g_obs Profiles')
            ax.axvline(300, color='gray', ls='--', lw=1, alpha=0.5)
            ax.text(310, 0.5, '1-halo/2-halo', fontsize=7, color='gray')
            ax.legend(fontsize=7)
            ax.grid(True, which='both', alpha=0.2)
        else:
            ax.text(0.5, 0.5, 'RC files not found', ha='center', va='center',
                    transform=ax.transAxes)

        ax = axs[1, 1]
        if f_no_hg.exists():
            d = load_esd(f_no_hg)
            gb = d['col1']
            go = esd_to_gobs(d['ESD_t'], d['bias'])
            v = (gb > 0) & (go > gb) & np.isfinite(go)
            gb, go = gb[v], go[v]
            gc_loc = (go - gb)**2 / gb
            Mstar_typ = 10**10.5 * Msun
            R_approx_kpc = np.sqrt(G_SI * Mstar_typ / gb) / kpc_m
            ax.semilogx(R_approx_kpc, gc_loc/a0, 'ko-', ms=6)
            ax.axhline(1.0, color='g', ls='--', lw=1, label='MOND')
            ax.axhline(0.24, color='gray', ls=':', lw=1, label='SPARC median')
            ax.axvline(30, color='blue', ls=':', alpha=0.5)
            ax.text(32, 0.5, 'SPARC\nrange', fontsize=7, color='blue')
            ax.axvline(300, color='red', ls=':', alpha=0.5)
            ax.text(320, 0.5, '1h/2h', fontsize=7, color='red')
            ax.set_xlabel('Approximate R [kpc]')
            ax.set_ylabel('gc_local / a0')
            ax.set_title('gc(R) Transition Profile')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = BASE / "brouwer_kids_c15_radial.png"
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved: {out_png}")
    except Exception as e:
        print(f"\n[Plot error: {e}]")


if __name__ == '__main__':
    main()
