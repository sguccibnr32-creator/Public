#!/usr/bin/env python3
"""plasticity_direct_gc.py — direct gc(Q) prediction vs SPARC"""

import os
import sys
import csv
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
T_m_Z2 = np.sqrt(6)


def delta_U(c):
    if c <= 1e-10:
        return -1.5
    if c > 1:
        return (c - 1) * 0.5
    u = np.sqrt(c)
    return -1.5 + 2*u - c/2.0 - (c/2.0)*np.log(c)


def solve_sc(Q, T_m):
    def residual(s):
        c = s * Q
        du = delta_U(c)
        arg = -du / T_m
        arg = np.clip(arg, -50, 50)
        rhs = 1.0 / (1.0 + np.exp(arg))
        return s - rhs

    s_lo, s_hi = 0.001, 0.999
    try:
        f_lo = residual(s_lo)
        f_hi = residual(s_hi)
    except Exception:
        return 0.5

    if f_lo * f_hi > 0:
        res = minimize_scalar(lambda s: residual(s)**2,
                              bounds=(0.001, 0.999), method='bounded')
        return res.x
    return brentq(residual, s_lo, s_hi, xtol=1e-12)


def load_sparc(basedir):
    ta3_path = os.path.join(basedir, "TA3_gc_independent.csv")
    mrt_path = os.path.join(basedir, "SPARC_Lelli2016c.mrt")

    if not os.path.isfile(ta3_path) or not os.path.isfile(mrt_path):
        return None

    ta3 = {}
    with open(ta3_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['galaxy'].strip()
            try:
                ta3[name] = {
                    'gc_a0': float(row['gc_over_a0']),
                    'Yd': float(row['upsilon_d']),
                    'v_flat': float(row['v_flat']),
                }
            except (ValueError, KeyError):
                continue

    mrt = {}
    with open(mrt_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                name = parts[0]
                hR = float(parts[11])
                mrt[name] = hR
            except (ValueError, IndexError):
                continue

    galaxies = []
    for name in ta3:
        if name not in mrt:
            continue
        d = ta3[name]
        hR = mrt[name]
        if hR <= 0 or d['v_flat'] <= 0 or d['gc_a0'] <= 0:
            continue
        vf_SI = d['v_flat'] * 1e3
        hR_SI = hR * kpc_m
        Q = np.sqrt(vf_SI**2 / (a0_SI * hR_SI))
        galaxies.append({
            'name': name, 'gc_a0': d['gc_a0'], 'Yd': d['Yd'],
            'v_flat': d['v_flat'], 'hR': hR, 'Q': Q, 'logQ': np.log10(Q),
        })

    print(f"Loaded {len(galaxies)} galaxies")
    return galaxies


def main():
    basedir = os.path.dirname(os.path.abspath(__file__))
    galaxies = load_sparc(basedir)
    if galaxies is None:
        print("Files not found")
        return

    gc_obs = np.array([g['gc_a0'] for g in galaxies])
    Q_arr = np.array([g['Q'] for g in galaxies])
    Yd_arr = np.array([g['Yd'] for g in galaxies])
    log_gc_obs = np.log10(gc_obs)
    log_Q = np.log10(Q_arr)
    log_Yd = np.log10(Yd_arr)
    N = len(galaxies)

    print("=" * 80)
    print("PART 0: Q distribution")
    print("=" * 80)
    print(f"  N={N}, Q: {Q_arr.min():.2f}-{Q_arr.max():.2f}, "
          f"median={np.median(Q_arr):.2f}")
    print(f"  gc/a0: {gc_obs.min():.4f}-{gc_obs.max():.3f}, median={np.median(gc_obs):.3f}")

    resid_baseline = log_gc_obs - log_Q
    print(f"\n  Baseline gc=Q*a0 (s=1): bias={np.median(resid_baseline):+.3f}, "
          f"scatter={np.std(resid_baseline):.3f} dex")

    print("\n" + "=" * 80)
    print("PART 1: Self-consistent model gc = A*s(Q,T_m)*Q")
    print("=" * 80)

    def evaluate_model(T_m, A=None):
        s_arr = np.array([solve_sc(Q, T_m) for Q in Q_arr])
        gc_model = s_arr * Q_arr
        log_gc_model = np.log10(np.clip(gc_model, 1e-6, 1e6))
        if A is None:
            offset = np.median(log_gc_obs - log_gc_model)
            A_opt = 10**offset
        else:
            A_opt = A
            offset = np.log10(A)
        log_gc_pred = log_gc_model + offset
        resid = log_gc_obs - log_gc_pred
        return {
            'T_m': T_m, 'A': A_opt, 'scatter': np.std(resid),
            'bias': np.median(resid), 'chi2': np.sum(resid**2),
            's_arr': s_arr, 'gc_pred': gc_model * A_opt,
            'log_gc_pred': log_gc_pred, 'resid': resid,
        }

    print(f"\n  {'T_m':>6s}  {'A':>8s}  {'scatter':>8s}  {'bias':>8s}  "
          f"{'s_med':>7s}  {'note':>12s}")
    T_m_grid = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.45, 3.0, 4.0, 5.0,
                7.0, 10.0, 15.0, 20.0, 50.0, 100.0]
    best_scatter = 999
    best_Tm = None
    for Tm in T_m_grid:
        r = evaluate_model(Tm)
        note = "<-- sqrt(6)" if abs(Tm - T_m_Z2) < 0.05 else ""
        if r['scatter'] < best_scatter:
            best_scatter = r['scatter']
            best_Tm = Tm
        s_med = np.median(r['s_arr'])
        print(f"  {Tm:6.2f}  {r['A']:8.4f}  {r['scatter']:8.4f}  "
              f"{r['bias']:+8.4f}  {s_med:7.4f}  {note}")
    print(f"\n  Best T_m = {best_Tm}")

    if best_Tm < 200:
        Tm_lo = max(0.1, best_Tm * 0.3)
        Tm_hi = best_Tm * 3
    else:
        Tm_lo = 20; Tm_hi = 500

    Tm_fine = np.linspace(Tm_lo, Tm_hi, 100)
    scatters = []
    As = []
    for Tm in Tm_fine:
        r = evaluate_model(Tm)
        scatters.append(r['scatter'])
        As.append(r['A'])
    scatters = np.array(scatters)
    As = np.array(As)
    idx_best = np.argmin(scatters)
    Tm_opt = Tm_fine[idx_best]
    A_opt = As[idx_best]

    print(f"\n  Fine optimal: T_m = {Tm_opt:.3f}, A = {A_opt:.4f}")
    print(f"  Min scatter = {scatters[idx_best]:.4f} dex")
    print(f"  sqrt(6) = {T_m_Z2:.4f}, ratio = {Tm_opt/T_m_Z2:.3f}")

    print("\n" + "=" * 80)
    print(f"PART 2: Optimal model analysis")
    print("=" * 80)
    r_opt = evaluate_model(Tm_opt)
    from scipy.stats import spearmanr
    rho, p = spearmanr(log_gc_obs, r_opt['log_gc_pred'])
    print(f"  Spearman r={rho:.4f} (p={p:.2e})")
    print(f"  Scatter={r_opt['scatter']:.4f} dex, bias={r_opt['bias']:+.4f}")

    gc_C15 = 0.584 * Yd_arr**(-0.361) * Q_arr
    log_gc_C15 = np.log10(gc_C15)
    resid_C15 = log_gc_obs - log_gc_C15
    scatter_C15 = np.std(resid_C15)
    rho_C15, _ = spearmanr(log_gc_obs, log_gc_C15)

    print(f"\n  C15 benchmark (0.584*Yd^-0.361*Q):")
    print(f"    Spearman r={rho_C15:.4f}, scatter={scatter_C15:.4f}, "
          f"bias={np.median(resid_C15):+.4f}")
    print(f"\n  Delta scatter (SC - C15) = {r_opt['scatter']-scatter_C15:+.4f} dex")

    print("\n" + "=" * 80)
    print("PART 3: Residual structure")
    print("=" * 80)
    rho_resid_Yd, p_resid_Yd = spearmanr(log_Yd, r_opt['resid'])
    print(f"  Residual vs log(Yd): r={rho_resid_Yd:+.4f} (p={p_resid_Yd:.2e})")
    if abs(rho_resid_Yd) > 0.2 and p_resid_Yd < 0.01:
        print(f"  -> SIGNIFICANT Yd dependence in residuals")
        beta_resid = np.polyfit(log_Yd, r_opt['resid'], 1)[0]
        print(f"  Residual slope vs log(Yd): {beta_resid:+.4f}")
        print(f"  C15 partial exponent:      -0.361")

    vf_arr = np.array([g['v_flat'] for g in galaxies])
    hR_arr = np.array([g['hR'] for g in galaxies])
    for varname, var in [('log(v_flat)', np.log10(vf_arr)),
                          ('log(hR)', np.log10(hR_arr)),
                          ('log(Q)', log_Q)]:
        rho_v, p_v = spearmanr(var, r_opt['resid'])
        print(f"  Residual vs {varname}: r={rho_v:+.4f} (p={p_v:.2e})")

    print("\n" + "=" * 80)
    print("PART 4: A=1 fixed")
    print("=" * 80)
    r_A1 = evaluate_model(Tm_opt, A=1.0)
    r_A1_Z2 = evaluate_model(T_m_Z2, A=1.0)
    print(f"  T_m={Tm_opt:.3f}, A=1: scatter={r_A1['scatter']:.4f}, "
          f"bias={r_A1['bias']:+.4f}")
    print(f"  T_m=sqrt(6), A=1:    scatter={r_A1_Z2['scatter']:.4f}, "
          f"bias={r_A1_Z2['bias']:+.4f}")

    print("\n" + "=" * 80)
    print("PART 5: Extended model gc = A*s(Q,T_m)*Q*Yd^beta")
    print("=" * 80)

    def fit_extended(T_m_val):
        s_arr = np.array([solve_sc(Q, T_m_val) for Q in Q_arr])
        log_sQ = np.log10(np.clip(s_arr * Q_arr, 1e-6, 1e6))
        y = log_gc_obs - log_sQ
        coeffs = np.polyfit(log_Yd, y, 1)
        beta_fit = coeffs[0]
        A_fit = 10**coeffs[1]
        resid = y - (coeffs[1] + beta_fit * log_Yd)
        return A_fit, beta_fit, np.std(resid)

    print(f"\n  {'T_m':>8s}  {'A':>8s}  {'beta':>8s}  {'scatter':>8s}  {'note':>20s}")
    for Tm in [0.5, 1.0, 1.5, 2.0, T_m_Z2, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        A, beta, scat = fit_extended(Tm)
        note = ""
        if abs(Tm - T_m_Z2) < 0.05:
            note = "<-- sqrt(6)"
        if abs(beta - (-0.361)) < 0.02:
            note += " beta~C15!"
        print(f"  {Tm:8.2f}  {A:8.4f}  {beta:+8.4f}  {scat:8.4f}  {note}")

    A_inf, beta_inf, scat_inf = fit_extended(1000.0)
    print(f"\n  T_m=1000 limit: A={A_inf:.4f}, beta={beta_inf:+.4f}, "
          f"scatter={scat_inf:.4f}")
    print(f"  C15:             beta=-0.361, scatter={scatter_C15:.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        ax = axes[0, 0]
        ax.loglog(gc_obs, r_opt['gc_pred'], 'o', color='steelblue', ms=4, alpha=0.5)
        ax.plot([0.01, 20], [0.01, 20], 'k--', alpha=0.5)
        ax.set_xlabel('gc obs'); ax.set_ylabel('gc pred')
        ax.set_title(f'(a) SC: T_m={Tm_opt:.2f}, A={A_opt:.3f}, '
                     f'scatter={r_opt["scatter"]:.3f}')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.loglog(gc_obs, gc_C15, 'o', color='red', ms=4, alpha=0.5)
        ax.plot([0.01, 20], [0.01, 20], 'k--', alpha=0.5)
        ax.set_xlabel('gc obs'); ax.set_ylabel('gc C15')
        ax.set_title(f'(b) C15: scatter={scatter_C15:.3f}')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(Tm_fine, scatters, 'b-', lw=2)
        ax.axvline(T_m_Z2, color='green', ls='--', label=f'sqrt(6)')
        ax.axvline(Tm_opt, color='red', ls='--', label=f'opt={Tm_opt:.2f}')
        ax.axhline(scatter_C15, color='orange', ls=':', label=f'C15={scatter_C15:.3f}')
        ax.set_xlabel('T_m'); ax.set_ylabel('scatter (dex)')
        ax.set_title('(c) Scatter vs T_m')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.scatter(Yd_arr, r_opt['resid'], c='steelblue', s=15, alpha=0.5)
        ax.axhline(0, color='gray', ls=':'); ax.set_xscale('log')
        ax.set_xlabel('Yd'); ax.set_ylabel('Residual (dex)')
        ax.set_title(f'(d) Residual vs Yd (r={rho_resid_Yd:+.3f})')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        Q_plot = np.logspace(-1, 2, 300)
        for Tm, col, lab in [(Tm_opt, 'blue', f'opt T_m={Tm_opt:.1f}'),
                              (T_m_Z2, 'green', 'sqrt(6)')]:
            s_plot = [solve_sc(Q, Tm) for Q in Q_plot]
            ax.semilogx(Q_plot, s_plot, color=col, lw=2, label=lab)
        ax.scatter(Q_arr, r_opt['s_arr'], c='gray', s=10, alpha=0.3, label='SPARC')
        ax.axhline(0.5, color='gray', ls=':')
        ax.set_xlabel('Q'); ax.set_ylabel('s')
        ax.set_title('(e) s(Q)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        s_empirical = gc_obs / Q_arr
        sc = ax.scatter(Q_arr, s_empirical, c=np.log10(Yd_arr), cmap='coolwarm',
                        s=20, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='log(Yd)')
        s_curve = [solve_sc(Q, Tm_opt) for Q in Q_plot]
        ax.semilogx(Q_plot, s_curve, 'k-', lw=2, label='model')
        ax.set_xlabel('Q'); ax.set_ylabel('gc/(a0*Q)')
        ax.set_title('(f) Empirical s by Yd')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 3.0)

        plt.tight_layout()
        outfig = os.path.join(basedir, "plasticity_direct_gc.png")
        plt.savefig(outfig, dpi=150)
        print(f"\nFig: {outfig}")
    except ImportError:
        pass

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Optimal: T_m={Tm_opt:.3f}, A={A_opt:.4f}, scatter={r_opt['scatter']:.4f}")
    print(f"  sqrt(6)={T_m_Z2:.4f}, ratio={Tm_opt/T_m_Z2:.3f}")
    print(f"  C15 scatter={scatter_C15:.4f}")
    print(f"  Residual-Yd corr: r={rho_resid_Yd:+.4f} (p={p_resid_Yd:.2e})")


if __name__ == "__main__":
    main()
