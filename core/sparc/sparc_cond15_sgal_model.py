#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_cond15_sgal_model.py
Condition 15 improved model test: adding S_gal as a parameter

Tests whether S_gal (galaxy-level plasticity index) reduces C15 scatter
beyond what eta(Yd) already captures.

Models:
  M0: log(gc) = a + beta*log(Yd) + 0.5*log(vflat^2/hR)    [current C15]
  M1: log(gc) = a + beta*log(Yd) + gamma*S_gal + 0.5*[...]  [C15 + S_gal]
  M2: log(gc) = a + beta*log(Yd) + g1*f_gas + 0.5*[...]     [C15 + f_gas]
  M3: log(gc) = a + beta*log(Yd) + g1*f_gas + g2*T + 0.5*[...] [C15 + f_gas + T]
  M4: full free alpha with S_gal
"""

import numpy as np
import os
import sys
from scipy import stats
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 3-file loader
# ============================================================
def load_mrt():
    """Load SPARC MRT (split-based parser, established pattern)."""
    candidates = [
        os.path.join(BASE, "SPARC_Lelli2016c.mrt"),
        os.path.join(BASE, "MRT", "table2.dat"),
        os.path.join(BASE, "table2.dat"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        print("  ERROR: MRT file not found")
        print(f"  Searched: {candidates}")
        return {}

    galaxies = {}
    in_data = False
    sep = 0
    with open(path, 'r', encoding='ascii', errors='replace') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4:
                    in_data = True
                continue
            if not in_data:
                continue
            p = line.split()
            if len(p) < 18:
                continue
            try:
                galaxies[p[0]] = {
                    'T': int(p[1]),
                    'L36': float(p[7]),
                    'Reff': float(p[9]),
                    'SBeff': float(p[10]),
                    'Rdisk': float(p[11]),
                    'SBdisk0': float(p[12]),
                    'MHI': float(p[13]),
                    'RHI': float(p[14]),
                    'Vflat': float(p[15]),
                    'Q': float(p[17]),
                }
            except (ValueError, IndexError):
                continue

    print(f"  MRT: {len(galaxies)} galaxies from {path}")
    return galaxies


def load_ta3():
    """Load TA3_gc_independent.csv."""
    path = os.path.join(BASE, "TA3_gc_independent.csv")
    galaxies = {}
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return galaxies
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')
        gc_col = header.index('gc_over_a0')
        name_col = header.index('galaxy')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) <= max(gc_col, name_col):
                continue
            name = vals[name_col].strip()
            try:
                gc_val = float(vals[gc_col].strip())
                if gc_val > 0:
                    galaxies[name] = {'gc_over_a0': gc_val}
            except ValueError:
                continue
    print(f"  TA3: {len(galaxies)} galaxies from {path}")
    return galaxies


def load_phase1():
    """Load phase1/sparc_results.csv (ud = Upsilon_d, vflat)."""
    path = os.path.join(BASE, "phase1", "sparc_results.csv")
    galaxies = {}
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return galaxies
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')
        ud_col = header.index('ud')
        name_col = header.index('galaxy')
        vf_col = header.index('vflat')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) <= max(ud_col, name_col, vf_col):
                continue
            name = vals[name_col].strip()
            try:
                galaxies[name] = {
                    'ud': float(vals[ud_col].strip()),
                    'vflat_p1': float(vals[vf_col].strip()),
                }
            except ValueError:
                continue
    print(f"  phase1: {len(galaxies)} galaxies from {path}")
    return galaxies


ROTMOD_DIR = os.path.join(BASE, "Rotmod_LTG")


def load_rotmod(galaxy_name):
    fname = os.path.join(ROTMOD_DIR, f"{galaxy_name}_rotmod.dat")
    if not os.path.exists(fname):
        return None
    rows = []
    with open(fname, 'r', encoding='ascii', errors='replace') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    rows.append({
                        'r': float(parts[0]),
                        'vobs': float(parts[1]),
                        'evobs': float(parts[2]),
                        'vgas': float(parts[3]),
                        'vdisk': float(parts[4]),
                        'vbul': float(parts[5]),
                        'SBdisk': float(parts[6])
                    })
                except ValueError:
                    continue
    return rows if rows else None


# ============================================================
# Merge and compute derived quantities
# ============================================================
def merge_data(mrt, ta3, phase1):
    galaxies = []
    matched = 0

    for name, m in mrt.items():
        if name not in ta3 or 'gc_over_a0' not in ta3[name]:
            continue
        gc_a0 = ta3[name]['gc_over_a0']
        if gc_a0 <= 0:
            continue

        p1 = phase1.get(name, {})
        ud = p1.get('ud', 0.5)
        if ud <= 0:
            ud = 0.5
        # Prefer MRT Vflat, fallback to phase1
        Vflat = m.get('Vflat', np.nan)
        if np.isnan(Vflat) or Vflat <= 0:
            Vflat = p1.get('vflat_p1', np.nan)

        Rdisk = m.get('Rdisk', np.nan)
        T = m.get('T', np.nan)
        L36 = m.get('L36', np.nan)
        MHI = m.get('MHI', np.nan)
        SBdisk0 = m.get('SBdisk0', np.nan)
        Q = m.get('Q', np.nan)

        if np.isnan(Vflat) or np.isnan(Rdisk) or Rdisk <= 0 or Vflat <= 0:
            continue

        # Mgas = 1.33 * MHI (MHI in 1e9 Msun in SPARC MRT)
        Mgas = 1.33 * MHI if not np.isnan(MHI) else np.nan
        # Mstar = ud * L36 (L36 in 1e9 Lsun)
        Mstar = ud * L36 if not np.isnan(L36) and L36 > 0 else np.nan

        if not np.isnan(Mgas) and not np.isnan(Mstar) and (Mstar + Mgas) > 0:
            f_gas = Mgas / (Mstar + Mgas)
        else:
            f_gas = np.nan

        Sigma_dyn = Vflat**2 / Rdisk

        # Rotmod
        compact = np.nan
        gas_extent = np.nan
        rotmod = load_rotmod(name)
        if rotmod and len(rotmod) >= 3:
            vgas = np.array([r['vgas'] for r in rotmod])
            vdisk = np.array([r['vdisk'] for r in rotmod])
            vbul = np.array([r['vbul'] for r in rotmod])
            r_arr = np.array([r['r'] for r in rotmod])
            v_bar2 = (np.sqrt(ud) * vdisk)**2 + vgas**2 + vbul**2
            v_bar = np.sqrt(np.maximum(v_bar2, 0))
            if len(v_bar) > 0 and Rdisk > 0:
                compact = np.max(v_bar) / Rdisk
            gas_mask = np.abs(vgas) > 1.0
            if np.sum(gas_mask) > 2:
                gas_extent = np.max(r_arr[gas_mask]) / Rdisk

        matched += 1
        galaxies.append({
            'name': name,
            'gc_a0': gc_a0,
            'log_gc': np.log10(gc_a0),
            'Vflat': Vflat,
            'Rdisk': Rdisk,
            'ud': ud,
            'T': float(T) if not np.isnan(T) else np.nan,
            'f_gas': f_gas,
            'SBdisk0': SBdisk0,
            'Sigma_dyn': Sigma_dyn,
            'compact': compact,
            'gas_extent': gas_extent,
            'Q': Q,
        })

    print(f"  Merged: {matched} galaxies")
    return galaxies


# ============================================================
# S_gal computation
# ============================================================
def normalize_rank(arr):
    valid = np.isfinite(arr)
    result = np.full_like(arr, np.nan, dtype=float)
    if np.sum(valid) > 2:
        ranks = stats.rankdata(arr[valid])
        result[valid] = (ranks - 1) / (np.sum(valid) - 1)
    return result


def compute_sgal(galaxies):
    N = len(galaxies)
    f_gas = np.array([g['f_gas'] for g in galaxies])
    ud = np.array([g['ud'] for g in galaxies])
    SBdisk0 = np.array([g['SBdisk0'] for g in galaxies])
    T = np.array([g['T'] for g in galaxies])

    fg_norm = normalize_rank(f_gas)
    ud_norm = normalize_rank(ud)
    sb_norm = normalize_rank(SBdisk0)  # high mag = low brightness = plastic
    tt_norm = normalize_rank(T)

    for i in range(N):
        components = [v for v in [fg_norm[i], ud_norm[i], sb_norm[i], tt_norm[i]]
                      if np.isfinite(v)]
        galaxies[i]['S_gal'] = np.mean(components) if len(components) >= 2 else np.nan

    n_valid = sum(1 for g in galaxies if np.isfinite(g['S_gal']))
    print(f"  S_gal computed: {n_valid}/{N}")


# ============================================================
# Model fitting
# ============================================================
def fit_model(y, X, names):
    N = len(y)
    k = X.shape[1]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    resid = y - y_pred
    scatter = np.std(resid)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res / ss_tot
    AIC = N * np.log(ss_res / N) + 2 * k
    return coeffs, scatter, R2, AIC, resid


def loo_scatter(y, X):
    N = len(y)
    loo_resid = np.zeros(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        try:
            c = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            loo_resid[i] = y[i] - X[i] @ c
        except Exception:
            loo_resid[i] = np.nan
    valid = np.isfinite(loo_resid)
    return np.std(loo_resid[valid])


# ============================================================
# Tests
# ============================================================
def test_models(galaxies):
    print("\n" + "=" * 70)
    print("C15 + S_gal model comparison")
    print("=" * 70)

    log_gc = np.array([g['log_gc'] for g in galaxies])
    log_vf = np.log10(np.array([g['Vflat'] for g in galaxies]))
    log_hR = np.log10(np.array([g['Rdisk'] for g in galaxies]))
    log_ud = np.array([np.log10(g['ud']) if g['ud'] > 0 else np.nan
                       for g in galaxies])
    S_gal = np.array([g['S_gal'] for g in galaxies])
    f_gas = np.array([g['f_gas'] for g in galaxies])
    T = np.array([g['T'] for g in galaxies])
    log_Sd = np.log10(np.array([g['Sigma_dyn'] for g in galaxies]))

    # ---- M0: Current C15 (alpha=0.5 fixed, eta(Yd)) ----
    mask0 = np.isfinite(log_gc) & np.isfinite(log_ud) & np.isfinite(log_Sd)
    N0 = np.sum(mask0)
    print(f"\n  M0: C15 baseline (alpha=0.5 fixed + Yd)")
    print(f"  N = {N0}")
    if N0 < 20:
        print("  Insufficient data")
        return

    y0 = log_gc[mask0] - 0.5 * log_Sd[mask0]
    X0 = np.column_stack([log_ud[mask0], np.ones(N0)])
    c0, scat0, R2_0, AIC_0, res0 = fit_model(y0, X0, ['beta', 'const'])
    loo0 = loo_scatter(y0, X0)

    X_adj0 = np.column_stack([log_ud[mask0], log_Sd[mask0], np.ones(N0)])
    c_full0, scat_full0, R2_full0, AIC_full0, res_full0 = fit_model(
        log_gc[mask0], X_adj0, ['beta', 'alpha', 'const'])

    print(f"  beta = {c0[0]:.3f}")
    print(f"  scatter = {scat0:.4f} dex (alpha=0.5 fixed)")
    print(f"  Full scatter = {scat_full0:.4f} dex")
    print(f"  R2 = {R2_full0:.4f}")
    print(f"  alpha_fit = {c_full0[1]:.3f} (should be ~0.5)")
    print(f"  LOO scatter = {loo0:.4f} dex")

    scat1 = loo1 = AIC_1 = None
    scat2 = loo2 = AIC_2 = None
    scat3 = loo3 = AIC_3 = None
    scat4 = loo4 = AIC_4 = None
    scat5 = loo5 = AIC_5 = None

    # ---- M1: C15 + S_gal ----
    mask1 = mask0 & np.isfinite(S_gal)
    N1 = np.sum(mask1)
    print(f"\n  M1: C15 + S_gal")
    print(f"  N = {N1}")
    if N1 >= 20:
        y1 = log_gc[mask1] - 0.5 * log_Sd[mask1]
        X1 = np.column_stack([log_ud[mask1], S_gal[mask1], np.ones(N1)])
        c1, scat1, R2_1, AIC_1, res1 = fit_model(y1, X1, ['beta', 'gamma_S', 'const'])
        loo1 = loo_scatter(y1, X1)
        print(f"  beta = {c1[0]:.3f}, gamma_S = {c1[1]:.3f}")
        print(f"  scatter = {scat1:.4f} dex, LOO = {loo1:.4f}")
        print(f"  dAIC vs M0 = {AIC_1 - AIC_0:.2f}")
        print(f"  scatter improvement: {100*(1-scat1/scat0):+.1f}% (LOO: {100*(1-loo1/loo0):+.1f}%)")
        mse1 = np.sum(res1**2) / (N1 - 3)
        try:
            cov1 = mse1 * np.linalg.inv(X1.T @ X1)
            se_gamma = np.sqrt(cov1[1, 1])
            t_gamma = c1[1] / se_gamma
            p_gamma = 2 * stats.t.sf(abs(t_gamma), df=N1 - 3)
            print(f"  gamma_S: t = {t_gamma:.2f}, p = {p_gamma:.4f}")
        except Exception:
            pass

    # ---- M2: C15 + f_gas ----
    mask2 = mask0 & np.isfinite(f_gas)
    N2 = np.sum(mask2)
    print(f"\n  M2: C15 + f_gas")
    print(f"  N = {N2}")
    if N2 >= 20:
        y2 = log_gc[mask2] - 0.5 * log_Sd[mask2]
        X2 = np.column_stack([log_ud[mask2], f_gas[mask2], np.ones(N2)])
        c2, scat2, R2_2, AIC_2, res2 = fit_model(y2, X2, ['beta', 'g_fgas', 'const'])
        loo2 = loo_scatter(y2, X2)
        print(f"  beta = {c2[0]:.3f}, g_fgas = {c2[1]:.3f}")
        print(f"  scatter = {scat2:.4f} dex, LOO = {loo2:.4f}")
        print(f"  dAIC vs M0 = {AIC_2 - AIC_0:.2f}")
        print(f"  improvement: {100*(1-scat2/scat0):+.1f}% (LOO: {100*(1-loo2/loo0):+.1f}%)")

    # ---- M3: C15 + f_gas + Ttype ----
    mask3 = mask0 & np.isfinite(f_gas) & np.isfinite(T)
    N3 = np.sum(mask3)
    print(f"\n  M3: C15 + f_gas + Ttype")
    print(f"  N = {N3}")
    if N3 >= 20:
        y3 = log_gc[mask3] - 0.5 * log_Sd[mask3]
        X3 = np.column_stack([log_ud[mask3], f_gas[mask3], T[mask3], np.ones(N3)])
        c3, scat3, R2_3, AIC_3, res3 = fit_model(y3, X3, ['beta', 'g_fgas', 'g_T', 'const'])
        loo3 = loo_scatter(y3, X3)
        print(f"  beta={c3[0]:.3f}, g_fgas={c3[1]:.3f}, g_T={c3[2]:.4f}")
        print(f"  scatter = {scat3:.4f} dex, LOO = {loo3:.4f}")
        print(f"  dAIC vs M0 = {AIC_3 - AIC_0:.2f}")
        print(f"  improvement: {100*(1-scat3/scat0):+.1f}% (LOO: {100*(1-loo3/loo0):+.1f}%)")

    # ---- M4: Free alpha + S_gal ----
    mask4 = mask1.copy()
    N4 = np.sum(mask4)
    print(f"\n  M4: Free alpha + S_gal")
    print(f"  N = {N4}")
    if N4 >= 20:
        y4 = log_gc[mask4]
        X4 = np.column_stack([log_ud[mask4], S_gal[mask4], log_Sd[mask4], np.ones(N4)])
        c4, scat4, R2_4, AIC_4, res4 = fit_model(y4, X4, ['beta', 'gamma_S', 'alpha', 'const'])
        loo4 = loo_scatter(y4, X4)
        print(f"  beta={c4[0]:.3f}, gamma_S={c4[1]:.3f}, alpha={c4[2]:.3f}")
        print(f"  scatter = {scat4:.4f} dex, LOO = {loo4:.4f}")
        print(f"  dAIC vs M0 = {AIC_4 - AIC_0:.2f}")

    # ---- M5: Maximal ----
    mask5 = mask0 & np.isfinite(f_gas) & np.isfinite(T) & np.isfinite(S_gal)
    N5 = np.sum(mask5)
    print(f"\n  M5: Free alpha + Yd + f_gas + T + S_gal (maximal)")
    print(f"  N = {N5}")
    if N5 >= 20:
        y5 = log_gc[mask5]
        X5 = np.column_stack([log_ud[mask5], f_gas[mask5], T[mask5],
                              S_gal[mask5], log_Sd[mask5], np.ones(N5)])
        c5, scat5, R2_5, AIC_5, res5 = fit_model(y5, X5,
            ['beta', 'g_fgas', 'g_T', 'gamma_S', 'alpha', 'const'])
        loo5 = loo_scatter(y5, X5)
        print(f"  beta={c5[0]:.3f}, g_fgas={c5[1]:.3f}, g_T={c5[2]:.4f}, "
              f"gamma_S={c5[3]:.3f}, alpha={c5[4]:.3f}")
        print(f"  scatter = {scat5:.4f} dex, LOO = {loo5:.4f}")
        print(f"  dAIC vs M0 = {AIC_5 - AIC_0:.2f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  {'Model':<35} {'k':>2} {'N':>4} {'scatter':>8} {'LOO':>8} {'dAIC':>8} {'improv':>8}")
    print(f"  {'-'*80}")
    print(f"  {'M0: C15 (Yd, alpha=0.5)':<35} {2:>2} {N0:>4} {scat0:>8.4f} {loo0:>8.4f} {'0':>8} {'--':>8}")
    if scat1 is not None:
        print(f"  {'M1: C15 + S_gal':<35} {3:>2} {N1:>4} {scat1:>8.4f} {loo1:>8.4f} {AIC_1-AIC_0:>+8.1f} {100*(1-loo1/loo0):>+7.1f}%")
    if scat2 is not None:
        print(f"  {'M2: C15 + f_gas':<35} {3:>2} {N2:>4} {scat2:>8.4f} {loo2:>8.4f} {AIC_2-AIC_0:>+8.1f} {100*(1-loo2/loo0):>+7.1f}%")
    if scat3 is not None:
        print(f"  {'M3: C15 + f_gas + T':<35} {4:>2} {N3:>4} {scat3:>8.4f} {loo3:>8.4f} {AIC_3-AIC_0:>+8.1f} {100*(1-loo3/loo0):>+7.1f}%")
    if scat4 is not None:
        print(f"  {'M4: free alpha + S_gal':<35} {4:>2} {N4:>4} {scat4:>8.4f} {loo4:>8.4f} {AIC_4-AIC_0:>+8.1f} {100*(1-loo4/loo0):>+7.1f}%")
    if scat5 is not None:
        print(f"  {'M5: maximal (all vars)':<35} {6:>2} {N5:>4} {scat5:>8.4f} {loo5:>8.4f} {AIC_5-AIC_0:>+8.1f} {100*(1-loo5/loo0):>+7.1f}%")

    print(f"\n  Judgment criteria:")
    print(f"    LOO improvement > 5%  -> adopt")
    print(f"    dAIC < -10            -> strong support")
    print(f"    dAIC < -5             -> moderate support")
    print(f"    dAIC > -2             -> no support")

    print(f"\n  C15 scatter baseline: {scat0:.4f} dex")
    for label, scat, loo, aic in [('M1 (S_gal)', scat1, loo1, AIC_1),
                                  ('M2 (f_gas)', scat2, loo2, AIC_2),
                                  ('M3 (f+T)', scat3, loo3, AIC_3),
                                  ('M4 (alpha+S)', scat4, loo4, AIC_4),
                                  ('M5 (max)', scat5, loo5, AIC_5)]:
        if scat is None:
            continue
        adopt = "ADOPT" if (1 - loo/loo0) > 0.05 and (aic - AIC_0) < -5 else "REJECT"
        print(f"  {label:<14} LOO improv = {100*(1-loo/loo0):+.1f}%, dAIC = {aic-AIC_0:+.1f} -> {adopt}")


# ============================================================
# Figure
# ============================================================
def make_figure(galaxies):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("\nWARNING: matplotlib not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('C15 + S_gal Model Comparison', fontsize=14, fontweight='bold')

    log_gc = np.array([g['log_gc'] for g in galaxies])
    log_ud = np.array([np.log10(g['ud']) if g['ud'] > 0 else np.nan for g in galaxies])
    log_Sd = np.log10(np.array([g['Sigma_dyn'] for g in galaxies]))
    S_gal = np.array([g['S_gal'] for g in galaxies])
    f_gas = np.array([g['f_gas'] for g in galaxies])
    T = np.array([g['T'] for g in galaxies])

    # P1: M0 residual vs S_gal
    ax = axes[0, 0]
    mask = np.isfinite(log_gc) & np.isfinite(log_ud) & np.isfinite(log_Sd) & np.isfinite(S_gal)
    if np.sum(mask) > 5:
        y = log_gc[mask] - 0.5 * log_Sd[mask]
        X = np.column_stack([log_ud[mask], np.ones(np.sum(mask))])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ c
        ax.scatter(S_gal[mask], resid, s=10, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        z = np.polyfit(S_gal[mask], resid, 1)
        xl = np.linspace(np.min(S_gal[mask]), np.max(S_gal[mask]), 100)
        ax.plot(xl, np.polyval(z, xl), 'b-', alpha=0.7)
    ax.set_xlabel('S_gal')
    ax.set_ylabel('M0 residual [dex]')
    ax.set_title('M0 residual vs S_gal')

    # P2: M0 residual vs f_gas
    ax = axes[0, 1]
    mask2 = np.isfinite(log_gc) & np.isfinite(log_ud) & np.isfinite(log_Sd) & np.isfinite(f_gas)
    if np.sum(mask2) > 5:
        y2 = log_gc[mask2] - 0.5 * log_Sd[mask2]
        X2 = np.column_stack([log_ud[mask2], np.ones(np.sum(mask2))])
        c2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        resid2 = y2 - X2 @ c2
        ax.scatter(f_gas[mask2], resid2, s=10, alpha=0.5, c='green')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('f_gas')
    ax.set_ylabel('M0 residual [dex]')
    ax.set_title('M0 residual vs f_gas')

    # P3: M0 residual vs Ttype
    ax = axes[0, 2]
    mask3 = np.isfinite(log_gc) & np.isfinite(log_ud) & np.isfinite(log_Sd) & np.isfinite(T)
    if np.sum(mask3) > 5:
        y3 = log_gc[mask3] - 0.5 * log_Sd[mask3]
        X3 = np.column_stack([log_ud[mask3], np.ones(np.sum(mask3))])
        c3 = np.linalg.lstsq(X3, y3, rcond=None)[0]
        resid3 = y3 - X3 @ c3
        ax.scatter(T[mask3], resid3, s=10, alpha=0.5, c='orange')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('T-type')
    ax.set_ylabel('M0 residual [dex]')
    ax.set_title('M0 residual vs Ttype')

    # P4: Observed vs Predicted (M0 full)
    ax = axes[1, 0]
    mask4 = np.isfinite(log_gc) & np.isfinite(log_ud) & np.isfinite(log_Sd)
    if np.sum(mask4) > 5:
        y4 = log_gc[mask4]
        X4 = np.column_stack([log_ud[mask4], log_Sd[mask4], np.ones(np.sum(mask4))])
        c4 = np.linalg.lstsq(X4, y4, rcond=None)[0]
        pred4 = X4 @ c4
        ax.scatter(pred4, y4, s=8, alpha=0.4)
        lims = [min(pred4.min(), y4.min())-0.2, max(pred4.max(), y4.max())+0.2]
        ax.plot(lims, lims, 'r-')
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Predicted log(gc/a0)')
    ax.set_ylabel('Observed log(gc/a0)')
    ax.set_title('M0: Observed vs Predicted')

    # P5: Observed vs Predicted (M1 with S_gal, free alpha)
    ax = axes[1, 1]
    if np.sum(mask) > 5:
        y5 = log_gc[mask]
        X5 = np.column_stack([log_ud[mask], S_gal[mask], log_Sd[mask], np.ones(np.sum(mask))])
        c5 = np.linalg.lstsq(X5, y5, rcond=None)[0]
        pred5 = X5 @ c5
        ax.scatter(pred5, y5, s=8, alpha=0.4, c='blue')
        lims = [min(pred5.min(), y5.min())-0.2, max(pred5.max(), y5.max())+0.2]
        ax.plot(lims, lims, 'r-')
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Predicted log(gc/a0)')
    ax.set_ylabel('Observed log(gc/a0)')
    ax.set_title('M1: C15 + S_gal')

    # P6: placeholder
    ax = axes[1, 2]
    ax.text(0.5, 0.5, 'See console output\nfor model comparison',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('Model Comparison')

    plt.tight_layout()
    out = os.path.join(BASE, 'cond15_sgal_model.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {out}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("C15 + S_gal Model Test")
    print("=" * 70)

    print("\n--- Loading data ---")
    mrt = load_mrt()
    ta3 = load_ta3()
    phase1 = load_phase1()

    if not mrt:
        print("FATAL: No MRT data"); sys.exit(1)
    if not ta3:
        print("FATAL: No TA3 data"); sys.exit(1)

    print("\n--- Merging ---")
    galaxies = merge_data(mrt, ta3, phase1)
    if len(galaxies) < 20:
        print(f"FATAL: Only {len(galaxies)} galaxies after merge"); sys.exit(1)

    print("\n--- Computing S_gal ---")
    compute_sgal(galaxies)

    test_models(galaxies)
    make_figure(galaxies)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == '__main__':
    main()
