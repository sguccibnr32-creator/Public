# -*- coding: utf-8 -*-
"""
sparc_gc_ttype_partial_corr.py

Test whether T-type has an independent effect on gc beyond Sigma_dyn.

Key question: Is gc(Red)/gc(Blue) ~ 2.5 in KiDS fully explained by
Sigma_dyn differences, or does morphological type carry independent
information about gc?

Partial correlations:
  r(gc, T-type | Sigma_dyn) = 0   -> (B)+(C) confirmed
  r(gc, T-type | Sigma_dyn) != 0  -> (D) has room

Uses TA3 + phase1 + MRT 3-file merge (standard pipeline).

Usage: uv run --with scipy python sparc_gc_ttype_partial_corr.py
"""

import os
import csv
import numpy as np
from pathlib import Path

a0 = 1.2e-10  # m/s^2

BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
TA3_CSV = BASE / "TA3_gc_independent.csv"
PHASE1_CSV = BASE / "phase1" / "sparc_results.csv"
MRT_FILE = BASE / "SPARC_Lelli2016c.mrt"


# ====================================================================
#  DATA LOADING (standard pipeline: TA3 csv + phase1 csv + MRT split)
# ====================================================================

def load_ta3():
    galaxies = {}
    with open(TA3_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
            except ValueError:
                continue
            if name and gc_a0 > 0:
                galaxies[name] = {'gc_a0': gc_a0}
    return galaxies


def load_phase1():
    galaxies = {}
    with open(PHASE1_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                vflat = float(row.get('vflat', '0'))
                Yd = float(row.get('ud', '0.5'))
            except ValueError:
                continue
            if name and vflat > 0:
                galaxies[name] = {'vflat': vflat, 'Yd': Yd}
    return galaxies


def load_mrt():
    """Parse SPARC MRT using split() pattern (sep>=4 => in_data).

    Columns (after split): [0]Galaxy [1]T [7]L36 [9]Reff [10]SBeff
    [11]Rdisk [12]SBdisk0 [13]MHI [14]RHI [15]Vflat [17]Q
    """
    galaxies = {}
    in_data = False
    sep = 0
    with open(MRT_FILE, 'r') as f:
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
                    'Rdisk': float(p[11]),   # disk scale length hR (kpc)
                    'Vflat': float(p[15]),
                }
            except (ValueError, IndexError):
                continue
    return galaxies


# ====================================================================
#  STATISTICS
# ====================================================================

def partial_corr(x, y, z):
    """Spearman partial r(x, y | z) via residual method."""
    from scipy import stats
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    sxz, ixz, _, _, _ = stats.linregress(z, x)
    res_x = x - (sxz * z + ixz)
    syz, iyz, _, _, _ = stats.linregress(z, y)
    res_y = y - (syz * z + iyz)
    r, p = stats.spearmanr(res_x, res_y)
    return r, p, n


def partial_corr_pearson(x, y, z):
    """Pearson partial r(x, y | z) via formula."""
    from scipy import stats
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    numer = r_xy - r_xz * r_yz
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return np.nan, np.nan, n
    rp = numer / denom
    t_stat = rp * np.sqrt((n - 3) / max(1 - rp**2, 1e-30))
    p_val = 2 * stats.t.sf(abs(t_stat), df=n - 3)
    return rp, p_val, n


# ====================================================================
#  MAIN
# ====================================================================

def main():
    from scipy import stats
    from numpy.linalg import lstsq

    print("=" * 72)
    print("  SPARC PARTIAL CORRELATION TEST")
    print("  gc vs T-type, controlling for Sigma_dyn / Yd")
    print("=" * 72)

    ta3 = load_ta3()
    ph1 = load_phase1()
    mrt = load_mrt()
    print(f"\n  TA3:    {len(ta3):4d} galaxies")
    print(f"  Phase1: {len(ph1):4d} galaxies")
    print(f"  MRT:    {len(mrt):4d} galaxies")

    common = set(ta3) & set(ph1) & set(mrt)
    print(f"  Merged: {len(common):4d} galaxies")

    names = sorted(common)
    gc_a0 = np.array([ta3[n]['gc_a0'] for n in names])
    vflat = np.array([ph1[n]['vflat'] for n in names])
    hR    = np.array([mrt[n]['Rdisk'] for n in names])
    Yd    = np.array([ph1[n]['Yd'] for n in names])
    T     = np.array([mrt[n]['T'] for n in names], dtype=float)

    valid = (gc_a0 > 0) & (vflat > 0) & (hR > 0) & (Yd > 0) & np.isfinite(T)
    gc_a0, vflat, hR, Yd, T = gc_a0[valid], vflat[valid], hR[valid], Yd[valid], T[valid]
    N = len(gc_a0)
    print(f"  After quality cut: {N} galaxies")

    log_gc = np.log10(gc_a0)
    Sigma_dyn = vflat**2 / hR
    log_Sigma = np.log10(Sigma_dyn)
    log_Yd = np.log10(np.maximum(Yd, 0.01))

    # ---------- Raw correlations ----------
    print(f"\n{'='*72}")
    print(f"  RAW CORRELATIONS (Spearman)")
    print(f"{'='*72}")
    pairs = [
        ('log gc', 'T-type', log_gc, T),
        ('log gc', 'log Sigma_dyn', log_gc, log_Sigma),
        ('log gc', 'log Yd', log_gc, log_Yd),
        ('T-type', 'log Sigma_dyn', T, log_Sigma),
        ('T-type', 'log Yd', T, log_Yd),
        ('log Sigma_dyn', 'log Yd', log_Sigma, log_Yd),
    ]
    print(f"\n  {'X':>16s}  {'Y':>16s}  {'rho':>8s}  {'p-value':>12s}")
    for lx, ly, x, y in pairs:
        r, p = stats.spearmanr(x, y)
        print(f"  {lx:>16s}  {ly:>16s}  {r:+8.3f}  {p:12.2e}")

    # ---------- Partial correlations ----------
    print(f"\n{'='*72}")
    print(f"  PARTIAL CORRELATIONS")
    print(f"{'='*72}")
    tests = [
        (log_gc, T, log_Sigma, 'log gc', 'T-type', 'log Sigma_dyn'),
        (log_gc, log_Sigma, T, 'log gc', 'log Sigma_dyn', 'T-type'),
        (log_gc, T, log_Yd, 'log gc', 'T-type', 'log Yd'),
        (log_gc, log_Yd, T, 'log gc', 'log Yd', 'T-type'),
        (log_gc, log_Yd, log_Sigma, 'log gc', 'log Yd', 'log Sigma_dyn'),
        (log_gc, log_Sigma, log_Yd, 'log gc', 'log Sigma_dyn', 'log Yd'),
    ]

    print(f"\n  Spearman partial (residual method):")
    for x, y, z, lx, ly, lz in tests:
        r, p, n = partial_corr(x, y, z)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    r({lx:>14s}, {ly:>14s} | {lz:>14s}) = {r:+.3f}  p={p:.2e}  {sig}")

    print(f"\n  Pearson partial (formula method):")
    for x, y, z, lx, ly, lz in tests:
        r, p, n = partial_corr_pearson(x, y, z)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    r({lx:>14s}, {ly:>14s} | {lz:>14s}) = {r:+.3f}  p={p:.2e}  {sig}")

    # ---------- Multiple regression ----------
    print(f"\n{'='*72}")
    print(f"  MULTIPLE REGRESSION: log gc = a + b1*log Sigma + b2*log Yd + b3*T")
    print(f"{'='*72}")

    def fit_model(X):
        b, _, _, _ = lstsq(X, log_gc, rcond=None)
        pred = X @ b
        ss_res = np.sum((log_gc - pred)**2)
        ss_tot = np.sum((log_gc - np.mean(log_gc))**2)
        r2 = 1 - ss_res / ss_tot
        return b, r2, ss_res

    ones = np.ones(N)
    b1, r2_1, ss1 = fit_model(np.column_stack([ones, log_Sigma]))
    b2, r2_2, ss2 = fit_model(np.column_stack([ones, log_Sigma, T]))
    b3, r2_3, ss3 = fit_model(np.column_stack([ones, log_Sigma, log_Yd]))
    b4, r2_4, ss4 = fit_model(np.column_stack([ones, log_Sigma, log_Yd, T]))

    def aic(ss, n, k):
        return n * np.log(ss / n) + 2 * k

    a1 = aic(ss1, N, 2); a2 = aic(ss2, N, 3); a3 = aic(ss3, N, 3); a4 = aic(ss4, N, 4)

    print(f"\n  {'Model':>35s}  {'R2':>7s}  {'dAIC':>8s}  Coefficients")
    print(f"  {'gc = f(Sigma)':>35s}  {r2_1:7.3f}  {0.0:+8.1f}  "
          f"b_Sig={b1[1]:+.3f}")
    print(f"  {'gc = f(Sigma, T)':>35s}  {r2_2:7.3f}  {a2-a1:+8.1f}  "
          f"b_Sig={b2[1]:+.3f}, b_T={b2[2]:+.4f}")
    print(f"  {'gc = f(Sigma, Yd) [~C15]':>35s}  {r2_3:7.3f}  {a3-a1:+8.1f}  "
          f"b_Sig={b3[1]:+.3f}, b_Yd={b3[2]:+.3f}")
    print(f"  {'gc = f(Sigma, Yd, T)':>35s}  {r2_4:7.3f}  {a4-a1:+8.1f}  "
          f"b_Sig={b4[1]:+.3f}, b_Yd={b4[2]:+.3f}, b_T={b4[3]:+.4f}")

    print(f"\n  Marginal Delta-R2 from adding T-type:")
    print(f"    Over Sigma alone:  {r2_2 - r2_1:+.4f}  "
          f"({'significant' if r2_2-r2_1 > 0.01 else 'negligible'})")
    print(f"    Over C15 (Sig+Yd): {r2_4 - r2_3:+.4f}  "
          f"({'significant' if r2_4-r2_3 > 0.01 else 'negligible'})")

    # ---------- Interpretation ----------
    print(f"\n{'='*72}")
    print(f"  INTERPRETATION")
    print(f"{'='*72}")

    r_key, p_key, _ = partial_corr(log_gc, T, log_Sigma)
    r_key_p, p_key_p, _ = partial_corr_pearson(log_gc, T, log_Sigma)

    print(f"\n  KEY: r(log gc, T-type | log Sigma_dyn)")
    print(f"    Spearman: {r_key:+.3f}  (p={p_key:.2e})")
    print(f"    Pearson:  {r_key_p:+.3f}  (p={p_key_p:.2e})")

    if abs(r_key) < 0.15 and p_key > 0.05:
        print(f"\n  -> T-type has NO independent effect on gc beyond Sigma_dyn.")
        print(f"     KiDS Fig-8 gc ordering is fully explained by Sigma_dyn differences")
        print(f"     between color/Sersic bins. Yd^-0.361 is a conversion correction,")
        print(f"     NOT an independent physical variable. (B)+(C) CONFIRMED.")
    elif abs(r_key) < 0.3:
        print(f"\n  -> T-type has WEAK independent effect on gc beyond Sigma_dyn.")
        print(f"     (B)+(C) dominant but (D) cannot be excluded.")
    else:
        print(f"\n  -> T-type has STRONG independent effect. (D) has room.")

    # Multi-variable partial (proper)
    X_yz = np.column_stack([log_Sigma, log_Yd])
    b_x = lstsq(X_yz, log_gc, rcond=None)[0]
    b_t = lstsq(X_yz, T, rcond=None)[0]
    res_gc = log_gc - X_yz @ b_x
    res_T = T - X_yz @ b_t
    r_final, p_final = stats.spearmanr(res_gc, res_T)

    print(f"\n  SECONDARY: r(log gc, T-type | log Sigma_dyn, log Yd) = {r_final:+.3f}  (p={p_final:.2e})")
    if abs(r_final) < 0.15 and p_final > 0.05:
        print(f"  -> After C15 variables (Sigma + Yd), T-type fully absorbed.")
        print(f"     C15 is sufficient; no independent morphological signal.")
    else:
        print(f"  -> T-type retains information beyond C15 variables.")

    print(f"\n{'='*72}\n  ANALYSIS COMPLETE\n{'='*72}")


if __name__ == '__main__':
    main()
