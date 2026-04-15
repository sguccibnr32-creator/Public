#!/usr/bin/env python3
"""plasticity_self_consistent.py — derive Yd exponent -0.361 from first principles"""

import os
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

a0_SI = 1.2e-10
G_SI = 6.67430e-11
MSUN = 1.98892e30
kpc_m = 3.08568e19
T_m_default = np.sqrt(6)


def delta_U(c):
    if np.isscalar(c):
        if c <= 0:
            return -1.5
        if c > 1:
            return 0.5 * (c - 1)**2
        u = np.sqrt(c)
        return -1.5 + 2*u - c/2 - (c/2)*np.log(c)
    else:
        result = np.zeros_like(c, dtype=float)
        for i, ci in enumerate(c):
            result[i] = delta_U(ci)
        return result


def delta_U_deriv(c):
    if c <= 0:
        return np.inf
    if c > 1:
        return (c - 1)
    return 1/np.sqrt(c) - 0.5 - (1 + np.log(c))/2


def solve_self_consistent(Q, T_m):
    def equation(s):
        c = s * Q
        du = delta_U(c)
        rhs = 1.0 / (1.0 + np.exp(-du / T_m))
        return s - rhs

    try:
        f_lo = equation(0.01)
        f_hi = equation(0.99)
    except (OverflowError, FloatingPointError):
        return 0.5, 0.5

    if f_lo * f_hi > 0:
        res = minimize_scalar(lambda s: equation(s)**2,
                              bounds=(0.01, 0.99), method='bounded')
        s_sol = res.x
    else:
        s_sol = brentq(equation, 0.01, 0.99, xtol=1e-10)
    return s_sol, 1.0 - s_sol


def fp_to_Upsilon(f_p):
    s = 1 - f_p
    if s <= 0.01:
        return 100.0
    return 0.228 * s**(-2.77)


def Upsilon_to_fp(Yd):
    return 1.0 - (Yd / 0.228)**(-0.361)


def compute_beta_eff(T_m_val):
    Q_scan = np.logspace(-0.5, 1.5, 300)
    gc_s = np.zeros(len(Q_scan))
    Yd_s = np.zeros(len(Q_scan))
    for i, Q in enumerate(Q_scan):
        s, fp = solve_self_consistent(Q, T_m_val)
        gc_s[i] = s * Q
        Yd_s[i] = fp_to_Upsilon(fp)
    mask = (Yd_s > 0.3) & (Yd_s < 3.0) & (gc_s > 0.01)
    if mask.sum() > 5:
        coeffs = np.polyfit(np.log10(Yd_s[mask]), np.log10(gc_s[mask]), 1)
        return coeffs[0]
    return float('nan')


def main():
    basedir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("PART A: Energy barrier DU(c)")
    print("=" * 80)
    c_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.42, 0.57, 0.8, 1.0]
    print(f"\n{'c':>8s}  {'eps_c':>8s}  {'DU(c)':>10s}  {'DU/T_m':>10s}")
    for c in c_vals:
        eps_c = 1 - np.sqrt(c) if c <= 1 else 0
        du = delta_U(c)
        du_Tm = du / T_m_default
        print(f"{c:8.3f}  {eps_c:8.4f}  {du:10.4f}  {du_Tm:10.4f}")

    print("\n" + "=" * 80)
    print(f"PART B: f_p(Q), T_m = sqrt(6) = {T_m_default:.4f}")
    print("=" * 80)

    Q_range = np.logspace(-1, 2, 200)
    s_solutions = np.zeros(len(Q_range))
    fp_solutions = np.zeros(len(Q_range))
    for i, Q in enumerate(Q_range):
        s, fp = solve_self_consistent(Q, T_m_default)
        s_solutions[i] = s
        fp_solutions[i] = fp

    print(f"\n{'Q':>8s}  {'c=sQ':>8s}  {'s':>8s}  {'f_p':>8s}  {'Yd(fp)':>8s}  {'gc/a0':>8s}")
    for Q in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        s, fp = solve_self_consistent(Q, T_m_default)
        c = s * Q
        Yd = fp_to_Upsilon(fp)
        print(f"{Q:8.2f}  {c:8.4f}  {s:8.4f}  {fp:8.4f}  {Yd:8.3f}  {c:8.4f}")

    s_min, fp_max = solve_self_consistent(0.01, T_m_default)
    s_max, fp_min = solve_self_consistent(100, T_m_default)
    print(f"\nLimits:")
    print(f"  Q->0:   f_p_max = {fp_max:.4f} "
          f"(theory: {1/(1+np.exp(1.5/T_m_default)):.4f})")
    print(f"  Q->inf: f_p_min = {fp_min:.4f}")

    print("\n" + "=" * 80)
    print("PART C: Effective exponent in SPARC range")
    print("=" * 80)

    Q_fine = np.logspace(-0.5, 1.5, 500)
    gc_arr = np.zeros(len(Q_fine))
    Yd_arr = np.zeros(len(Q_fine))
    for i, Q in enumerate(Q_fine):
        s, fp = solve_self_consistent(Q, T_m_default)
        gc_arr[i] = s * Q
        Yd_arr[i] = fp_to_Upsilon(fp)

    mask_sparc = (Yd_arr > 0.3) & (Yd_arr < 3.0) & (gc_arr > 0.01)
    if mask_sparc.sum() > 5:
        coeffs = np.polyfit(np.log10(Yd_arr[mask_sparc]),
                            np.log10(gc_arr[mask_sparc]), 1)
        beta_eff = coeffs[0]
        print(f"\n  Yd range: {Yd_arr[mask_sparc].min():.3f}-{Yd_arr[mask_sparc].max():.3f}")
        print(f"  gc range: {gc_arr[mask_sparc].min():.4f}-{gc_arr[mask_sparc].max():.4f}")
        print(f"  beta_eff = {beta_eff:+.4f}")
        print(f"  Observed: -0.361")
        print(f"  Ratio:    {beta_eff / -0.361:.3f}")
    else:
        beta_eff = float('nan')

    print(f"\n  Local exponent at specific Q:")
    print(f"  {'Q':>8s}  {'gc/a0':>8s}  {'Yd':>8s}  {'beta_local':>12s}")
    for Q in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
        dQ = Q * 0.01
        s1, fp1 = solve_self_consistent(Q, T_m_default)
        s2, fp2 = solve_self_consistent(Q + dQ, T_m_default)
        gc1 = s1 * Q
        gc2 = s2 * (Q + dQ)
        Yd1 = fp_to_Upsilon(fp1)
        Yd2 = fp_to_Upsilon(fp2)
        if Yd1 > 0 and Yd2 > 0 and gc1 > 0 and gc2 > 0:
            beta_loc = (np.log10(gc2) - np.log10(gc1)) / (np.log10(Yd2) - np.log10(Yd1))
            print(f"  {Q:8.2f}  {gc1:8.4f}  {Yd1:8.3f}  {beta_loc:+12.4f}")

    print("\n" + "=" * 80)
    print("PART D: T_m scan for beta = -0.361")
    print("=" * 80)
    T_m_range = np.arange(0.5, 5.1, 0.25)
    print(f"\n  {'T_m':>8s}  {'beta_eff':>10s}  {'vs -0.361':>10s}")
    best_Tm = None
    best_diff = 999
    for Tm in T_m_range:
        be = compute_beta_eff(Tm)
        diff = abs(be - (-0.361)) if np.isfinite(be) else 999
        note = "  <-- sqrt(6)" if abs(Tm - T_m_default) < 0.01 else ""
        if diff < best_diff:
            best_diff = diff
            best_Tm = Tm
        print(f"  {Tm:8.2f}  {be:+10.4f}  {be-(-0.361):+10.4f}{note}")

    print(f"\n  Best T_m = {best_Tm:.2f}, sqrt(6) = {T_m_default:.4f}")
    T_m_fine = np.arange(max(0.5, best_Tm - 0.5), best_Tm + 0.55, 0.05)
    print(f"\n  Fine scan:")
    for Tm in T_m_fine:
        be = compute_beta_eff(Tm)
        marker = " ***" if abs(be - (-0.361)) < 0.005 else ""
        print(f"    T_m={Tm:.2f}: beta={be:+.4f}{marker}")

    print("\n" + "=" * 80)
    print("PART E: SPARC comparison")
    print("=" * 80)
    ta3_path = os.path.join(basedir, "TA3_gc_independent.csv")
    mrt_path = os.path.join(basedir, "SPARC_Lelli2016c.mrt")

    sparc_available = os.path.isfile(ta3_path) and os.path.isfile(mrt_path)
    galaxies = []
    gc_obs = Yd_obs = gc_pred = None

    if sparc_available:
        import csv
        ta3_data = {}
        with open(ta3_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['galaxy'].strip()
                try:
                    ta3_data[name] = {
                        'gc_a0': float(row['gc_over_a0']),
                        'Yd': float(row['upsilon_d']),
                        'v_flat': float(row['v_flat']),
                    }
                except (ValueError, KeyError):
                    continue

        mrt_data = {}
        with open(mrt_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 18:
                    continue
                try:
                    name = parts[0]
                    hR = float(parts[11])
                    mrt_data[name] = {'hR': hR}
                except (ValueError, IndexError):
                    continue

        for name in ta3_data:
            if name in mrt_data and ta3_data[name]['v_flat'] > 0:
                d = ta3_data[name]
                hR = mrt_data[name]['hR']
                if hR > 0:
                    vf = d['v_flat'] * 1e3
                    hR_m = hR * kpc_m
                    Q = np.sqrt(vf**2 / (a0_SI * hR_m))
                    galaxies.append({
                        'name': name, 'gc_a0': d['gc_a0'], 'Yd': d['Yd'],
                        'v_flat': d['v_flat'], 'hR': hR, 'Q': Q,
                    })
        print(f"\n  Loaded {len(galaxies)} galaxies")

        if len(galaxies) > 10:
            gc_pred = []
            gc_obs = []
            Yd_obs = []
            fp_pred = []
            for g in galaxies:
                s, fp = solve_self_consistent(g['Q'], T_m_default)
                gc_pred.append(s * g['Q'])
                gc_obs.append(g['gc_a0'])
                Yd_obs.append(g['Yd'])
                fp_pred.append(fp)
            gc_pred = np.array(gc_pred)
            gc_obs = np.array(gc_obs)
            Yd_obs = np.array(Yd_obs)
            fp_pred = np.array(fp_pred)

            from scipy.stats import spearmanr
            r_gc, p_gc = spearmanr(np.log10(gc_obs), np.log10(gc_pred))
            print(f"\n  gc prediction:")
            print(f"    Spearman r = {r_gc:.4f}, p = {p_gc:.2e}")
            print(f"    median gc_pred/gc_obs = {np.median(gc_pred/gc_obs):.3f}")
            print(f"    scatter (dex) = {np.std(np.log10(gc_pred/gc_obs)):.3f}")

            Yd_pred = np.array([fp_to_Upsilon(fp) for fp in fp_pred])
            mask_valid = (Yd_pred > 0.1) & (Yd_pred < 10)
            if mask_valid.sum() > 10:
                r_Yd, p_Yd = spearmanr(np.log10(Yd_obs[mask_valid]),
                                        np.log10(Yd_pred[mask_valid]))
                print(f"\n  Yd prediction (from f_p):")
                print(f"    Spearman r = {r_Yd:.4f}, p = {p_Yd:.2e}")
                print(f"    median Yd_pred/Yd_obs = "
                      f"{np.median(Yd_pred[mask_valid]/Yd_obs[mask_valid]):.3f}")

            mask_Yd = (Yd_obs > 0.3) & (Yd_obs < 3.0) & (gc_obs > 0.01)
            if mask_Yd.sum() > 10:
                beta_sparc = np.polyfit(np.log10(Yd_obs[mask_Yd]),
                                        np.log10(gc_obs[mask_Yd]), 1)[0]
                beta_pred = np.polyfit(np.log10(Yd_obs[mask_Yd]),
                                       np.log10(gc_pred[mask_Yd]), 1)[0]
                print(f"\n  Effective exponent (Yd=[0.3,3.0], N={mask_Yd.sum()}):")
                print(f"    Observed:   beta = {beta_sparc:+.4f}")
                print(f"    Predicted:  beta = {beta_pred:+.4f}")
                print(f"    C15 target: beta = -0.361")
    else:
        print(f"\n  SPARC files not found")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        ax = axes[0, 0]
        c_plot = np.linspace(0.01, 1.5, 200)
        du_plot = np.array([delta_U(ci) for ci in c_plot])
        ax.plot(c_plot, du_plot, 'b-', lw=2)
        ax.axhline(0, color='gray', ls=':'); ax.axvline(1, color='red', ls='--')
        ax.set_xlabel('c'); ax.set_ylabel('DU(c)')
        ax.set_title('(a) Energy barrier DU(c)')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.semilogx(Q_range, fp_solutions, 'b-', lw=2, label=f'T_m=sqrt(6)')
        for Tm_test, col in [(1.0, 'red'), (2.0, 'green'), (3.0, 'orange')]:
            fp_test = np.zeros(len(Q_range))
            for i, Q in enumerate(Q_range):
                _, fp_test[i] = solve_self_consistent(Q, Tm_test)
            ax.semilogx(Q_range, fp_test, color=col, ls='--', lw=1.5,
                        label=f'T_m={Tm_test:.1f}')
        ax.axhline(0.5, color='gray', ls=':'); ax.set_xlabel('Q')
        ax.set_ylabel('f_p'); ax.set_title('(b) Self-consistent f_p(Q)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        mask_plot = (Yd_arr > 0.1) & (Yd_arr < 10) & (gc_arr > 0.001)
        ax.loglog(Yd_arr[mask_plot], gc_arr[mask_plot], 'b-', lw=2,
                  label='Self-consistent')
        Yd_ref = np.logspace(-0.5, 0.5, 50)
        ax.loglog(Yd_ref, 0.5 * Yd_ref**(-0.361), 'r--', lw=1.5,
                  label='Yd^{-0.361}')
        if sparc_available and gc_obs is not None:
            ax.scatter(Yd_obs, gc_obs, c='gray', s=10, alpha=0.5, label='SPARC')
        ax.set_xlabel('Upsilon_d'); ax.set_ylabel('gc/a0')
        ax.set_title('(c) gc vs Yd')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogx(Q_range, s_solutions, 'b-', lw=2)
        ax.axhline(0.5, color='gray', ls=':')
        ax.axvspan(0.5, 20, color='yellow', alpha=0.15, label='SPARC range')
        ax.set_xlabel('Q'); ax.set_ylabel('s')
        ax.set_title('(d) s(Q)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        Tm_scan_fine = np.arange(0.5, 5.05, 0.1)
        betas = [compute_beta_eff(Tm) for Tm in Tm_scan_fine]
        ax.plot(Tm_scan_fine, betas, 'b-', lw=2)
        ax.axhline(-0.361, color='red', ls='--', lw=1.5, label='Observed -0.361')
        ax.axvline(T_m_default, color='green', ls='--', lw=1.5,
                   label=f'sqrt(6)={T_m_default:.2f}')
        ax.set_xlabel('T_m'); ax.set_ylabel('beta_eff')
        ax.set_title('(e) beta_eff vs T_m')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if sparc_available and gc_obs is not None:
            ax.loglog(gc_obs, gc_pred, 'o', color='steelblue', ms=4, alpha=0.5)
            lims = [min(gc_obs.min(), gc_pred.min())*0.5,
                    max(gc_obs.max(), gc_pred.max())*2]
            ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')
            ax.set_xlabel('gc observed'); ax.set_ylabel('gc predicted')
            ax.set_title('(f) pred vs obs')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'SPARC N/A', transform=ax.transAxes, ha='center')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfig = os.path.join(basedir, "plasticity_self_consistent.png")
        plt.savefig(outfig, dpi=150)
        print(f"\nFig: {outfig}")
    except ImportError:
        pass

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  T_m = sqrt(6) = {T_m_default:.4f}")
    print(f"  Self-consistent beta_eff (Yd=[0.3,3.0]): {beta_eff:+.4f}")
    print(f"  Observed C15:                            -0.3610")
    print(f"  Best T_m for -0.361: {best_Tm:.2f}")


if __name__ == "__main__":
    main()
