#!/usr/bin/env python3
"""
sparc_plasticity_analysis.py
SPARC 175 galaxies: plasticity region distribution integrated analysis

Purpose:
  Independently verify S_plastic correlation established by
  Miyaoka 2018 (16 clusters) at SPARC galaxy scale.

Tests (T1-T8):
  T1: Build galaxy-level plasticity indicator S_gal
  T2: S_gal vs gc/a0 correlation
  T3: Morphological bridge to Miyaoka S_plastic
  T4: C15 residual vs S_gal
  T5: Deep-MOND amplification
  T6: PCA of plasticity indicators
  T7: Im-type irregular galaxy analysis
  T8: kappa=0 physical consistency

Data: TA3_gc_independent.csv + phase1/sparc_results.csv + SPARC_Lelli2016c.mrt
"""

import csv
import numpy as np
import os
import sys
from pathlib import Path
from scipy import stats

# ============================================================
# Paths and constants
# ============================================================
BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
ROTMOD = BASE / "Rotmod_LTG"
PHASE1 = BASE / "phase1" / "sparc_results.csv"
TA3    = BASE / "TA3_gc_independent.csv"
MRT    = BASE / "SPARC_Lelli2016c.mrt"

a0 = 1.2e-10   # m/s^2
kpc_m = 3.086e19
ETA0 = 0.584
BETA = -0.361

# ============================================================
# Standard loaders (TA3 + phase1 + MRT)
# ============================================================
def load_pipeline():
    """Load gc from TA3 and Yd/vflat from phase1."""
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                data[n] = {
                    'vflat': float(row.get('vflat', '0')),
                    'Yd': float(row.get('ud', '0.5')),
                }
            except Exception:
                pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                if n in data and gc_a0 > 0:
                    data[n]['gc_a0'] = gc_a0
                    data[n]['gc'] = gc_a0 * a0
            except Exception:
                pass
    return {k: v for k, v in data.items() if 'gc' in v and v['vflat'] > 0}


def parse_mrt():
    """Parse SPARC_Lelli2016c.mrt."""
    data = {}
    in_data = False
    sep = 0
    with open(MRT, 'r') as f:
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
                data[p[0]] = {
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
            except Exception:
                continue
    return data


def load_rotmod(galaxy_name):
    """Load rotation curve from Rotmod_LTG."""
    fname = ROTMOD / f"{galaxy_name}_rotmod.dat"
    if not fname.exists():
        return None
    rows = []
    with open(fname, 'r') as f:
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
                        'SBdisk': float(parts[6]),
                    })
                except ValueError:
                    continue
    return rows if rows else None


# ============================================================
# Build galaxy parameter table
# ============================================================
def build_galaxies():
    """Merge all sources into galaxy parameter list."""
    pipe = load_pipeline()   # {name: {vflat, Yd, gc_a0, gc}}
    mrt  = parse_mrt()       # {name: {T, L36, Rdisk, SBdisk0, ...}}

    galaxies = []
    for name, p in pipe.items():
        m = mrt.get(name, {})
        if not m:
            continue

        gc_a0 = p['gc_a0']
        vflat = p['vflat']
        Yd    = p['Yd']
        hR    = m.get('Rdisk', np.nan)       # kpc
        Ttype = m.get('T', np.nan)
        L36   = m.get('L36', np.nan)         # 10^36 W (3.6um)
        SBdisk0 = m.get('SBdisk0', np.nan)   # mag/arcsec^2
        SBeff   = m.get('SBeff', np.nan)
        MHI   = m.get('MHI', np.nan)         # 10^9 Msun
        Q     = m.get('Q', np.nan)

        if np.isnan(hR) or hR <= 0:
            continue

        # --- f_gas ---
        f_gas = np.nan
        if np.isfinite(MHI) and np.isfinite(L36) and L36 > 0:
            # M_gas = 1.33 * MHI (He correction), MHI in 1e9 Msun
            Mgas = 1.33 * MHI * 1e9   # Msun
            # L36 in 1e36 W -> L_sun at 3.6um
            # Use Yd directly: Mstar = Yd * L_3.6
            # L36 is L[3.6] in units of 10^36 W; convert:
            # L_sun(3.6um) ~ 1.7e26 W  =>  L_3.6/L_sun = L36*1e36/1.7e26
            L_Lsun = L36 * 1e36 / 1.7e26
            Yd_use = Yd if np.isfinite(Yd) and Yd > 0 else 0.5
            Mstar = Yd_use * L_Lsun
            Mbar = Mstar + Mgas
            if Mbar > 0:
                f_gas = Mgas / Mbar

        # --- compact, gas_extent, vbar_asym from rotmod ---
        compact = np.nan
        gas_extent = np.nan
        vbar_asym = np.nan
        rotmod = load_rotmod(name)
        if rotmod and len(rotmod) > 3:
            Yd_use = Yd if np.isfinite(Yd) and Yd > 0 else 0.5
            r_arr  = np.array([row['r'] for row in rotmod])
            vgas   = np.array([row['vgas'] for row in rotmod])
            vdisk  = np.array([row['vdisk'] for row in rotmod])
            vbul   = np.array([row['vbul'] for row in rotmod])

            v_bar2 = (np.sqrt(Yd_use) * vdisk)**2 + vgas**2 + vbul**2
            v_bar  = np.sqrt(np.maximum(v_bar2, 0))

            if len(v_bar) > 0 and hR > 0:
                compact = np.max(v_bar) / hR

            gas_mask = np.abs(vgas) > 1.0
            if np.sum(gas_mask) > 2:
                gas_extent = np.max(r_arr[gas_mask]) / hR

            if len(v_bar) >= 6:
                n_half = len(v_bar) // 2
                inner_mean = np.mean(v_bar[:n_half])
                outer_mean = np.mean(v_bar[n_half:])
                if inner_mean > 0:
                    vbar_asym = (outer_mean - inner_mean) / inner_mean

        # --- C15 prediction and residual ---
        Yd_use = Yd if np.isfinite(Yd) and Yd > 0 else 0.5
        vflat_ms = vflat * 1e3        # km/s -> m/s
        hR_m     = hR * kpc_m         # kpc -> m
        gc_pred  = ETA0 * Yd_use**BETA * np.sqrt(a0 * vflat_ms**2 / hR_m)
        gc_obs   = gc_a0 * a0
        delta_gc = np.log10(gc_obs / gc_pred) if gc_pred > 0 else np.nan

        galaxies.append({
            'name': name,
            'gc_a0': gc_a0,
            'log_gc_a0': np.log10(gc_a0) if gc_a0 > 0 else np.nan,
            'vflat': vflat,
            'hR': hR,
            'Yd': Yd,
            'Ttype': float(Ttype) if np.isfinite(Ttype) else np.nan,
            'f_gas': f_gas,
            'SBdisk': SBdisk0,
            'SBeff': SBeff,
            'compact': compact,
            'gas_extent': gas_extent,
            'vbar_asym': vbar_asym,
            'Sigma_dyn': (vflat * 1e3)**2 / hR_m,
            'delta_gc': delta_gc,
            'quality': Q,
        })

    return galaxies


# ============================================================
# Rank-based normalisation [0, 1]
# ============================================================
def normalize_rank(arr):
    valid = np.isfinite(arr)
    result = np.full_like(arr, np.nan)
    if np.sum(valid) > 2:
        ranks = stats.rankdata(arr[valid])
        result[valid] = (ranks - 1) / (np.sum(valid) - 1)
    return result


# ============================================================
# T1: Build S_gal
# ============================================================
def test_T1(galaxies):
    print("=" * 70)
    print("T1: Galaxy-level plasticity indicator S_gal")
    print("=" * 70)

    N = len(galaxies)
    print(f"\n  N galaxies: {N}")

    f_gas   = np.array([g['f_gas'] for g in galaxies])
    Yd      = np.array([g['Yd'] for g in galaxies])
    SBdisk  = np.array([g['SBdisk'] for g in galaxies])
    Ttype   = np.array([g['Ttype'] for g in galaxies])
    compact = np.array([g['compact'] for g in galaxies])
    gas_ext = np.array([g['gas_extent'] for g in galaxies])
    log_gc  = np.array([g['log_gc_a0'] for g in galaxies])

    print(f"\n  Available counts:")
    for label, arr in [('f_gas', f_gas), ('Yd', Yd), ('SBdisk', SBdisk),
                       ('Ttype', Ttype), ('compact', compact),
                       ('gas_extent', gas_ext)]:
        n = np.sum(np.isfinite(arr))
        print(f"    {label:<15}: {n}/{N} ({100*n/N:.0f}%)")

    # Individual correlations with gc/a0
    print(f"\n  Individual correlations vs log(gc/a0):")
    print(f"  {'Indicator':<15} {'rho':>8} {'p':>12} {'Direction':>16} {'Miyaoka analog'}")
    print(f"  {'-'*70}")

    correlations = {}
    for label, arr, miy in [
        ('f_gas',       f_gas,    'fgas(r500) r=+0.71'),
        ('Yd',          Yd,       'T_ratio(inv) N/A'),
        ('log(SBdisk)', np.log10(np.maximum(SBdisk, 1e-10)), 'C_mass r=+0.02'),
        ('Ttype',       Ttype,    'N/A'),
        ('compact',     compact,  'N/A'),
        ('gas_extent',  gas_ext,  'Dfgas analog'),
    ]:
        mask = np.isfinite(arr) & np.isfinite(log_gc)
        if np.sum(mask) > 10:
            r, p = stats.spearmanr(arr[mask], log_gc[mask])
            dirn = 'plastic->gc low' if r < 0 else 'elastic->gc high'
            print(f"  {label:<15} {r:>+8.3f} {p:>12.2e} {dirn:>16}  {miy}")
            correlations[label] = (r, p)
        else:
            print(f"  {label:<15}  insufficient data")

    # Build S_gal (equal weight, rank-based)
    print(f"\n  S_gal construction:")
    print(f"  Miyaoka:  S_plastic = 0.4*norm(fgas) + 0.3*norm(Dfgas) + 0.3*(1-norm(T_ratio))")
    print("  Galaxy:   S_gal = mean of available [norm(f_gas), norm(Yd), 1-norm(SBdisk), norm(Ttype)]")

    fg_norm = normalize_rank(f_gas)
    yd_norm = normalize_rank(Yd)
    sb_norm = 1.0 - normalize_rank(SBdisk)   # low SB -> high score
    tt_norm = normalize_rank(Ttype)           # late type -> high score

    S_gal = np.full(N, np.nan)
    for i in range(N):
        vals = [v for v in [fg_norm[i], yd_norm[i], sb_norm[i], tt_norm[i]]
                if np.isfinite(v)]
        if len(vals) >= 2:
            S_gal[i] = np.mean(vals)

    n_valid = np.sum(np.isfinite(S_gal))
    print(f"\n  S_gal (equal weight): {n_valid}/{N} galaxies computed")
    if n_valid > 0:
        print(f"    median = {np.nanmedian(S_gal):.3f}")
        print(f"    range  = [{np.nanmin(S_gal):.3f}, {np.nanmax(S_gal):.3f}]")

    # Correlation-based weights
    weights = {}
    total_abs_r = 0
    for label, (r, p) in correlations.items():
        if p < 0.1:
            weights[label] = abs(r)
            total_abs_r += abs(r)
    if total_abs_r > 0:
        for k in weights:
            weights[k] /= total_abs_r
        print(f"\n  Correlation-based weights: { {k: round(v,3) for k,v in weights.items()} }")

    # Store
    for i, g in enumerate(galaxies):
        g['S_gal']   = S_gal[i]
        g['fg_norm'] = fg_norm[i]
        g['yd_norm'] = yd_norm[i]
        g['sb_norm'] = sb_norm[i]
        g['tt_norm'] = tt_norm[i]

    return correlations


# ============================================================
# T2: S_gal vs gc/a0
# ============================================================
def test_T2(galaxies):
    print("\n" + "=" * 70)
    print("T2: S_gal vs gc/a0 correlation")
    print("=" * 70)

    S  = np.array([g['S_gal'] for g in galaxies])
    gc = np.array([g['log_gc_a0'] for g in galaxies])
    mask = np.isfinite(S) & np.isfinite(gc)
    n = np.sum(mask)

    print(f"\n  Valid galaxies: {n}")
    if n < 10:
        print("  Insufficient data")
        return

    r, p = stats.spearmanr(S[mask], gc[mask])
    print(f"  Spearman: rho = {r:+.3f}, p = {p:.2e}")

    slope, intercept, rval, pval, stderr = stats.linregress(S[mask], gc[mask])
    print(f"  OLS: slope = {slope:.3f}+/-{stderr:.3f}, R2 = {rval**2:.3f}")

    # Miyaoka comparison
    print(f"\n  Miyaoka 2018 comparison:")
    print(f"    Miyaoka: S_plastic vs fgas(r500) r=+0.709 (N=16)")
    print(f"    SPARC:   S_gal vs log(gc/a0)    r={r:+.3f} (N={n})")

    # S_gal quintiles
    quintiles = np.percentile(S[mask], [20, 40, 60, 80])
    bins = [(-np.inf, quintiles[0]), (quintiles[0], quintiles[1]),
            (quintiles[1], quintiles[2]), (quintiles[2], quintiles[3]),
            (quintiles[3], np.inf)]
    labels = ['Q1(elastic)', 'Q2', 'Q3', 'Q4', 'Q5(plastic)']

    print(f"\n  S_gal quintile gc/a0:")
    print(f"  {'Quintile':<14} {'median gc/a0':>14} {'N':>4}")
    for label, (lo, hi) in zip(labels, bins):
        m = mask & (S >= lo) & (S < hi)
        if np.sum(m) > 0:
            med_gc = np.median(10**gc[m])
            print(f"  {label:<14} {med_gc:>14.3f} {np.sum(m):>4}")


# ============================================================
# T3: Morphology bridge
# ============================================================
def test_T3(galaxies):
    print("\n" + "=" * 70)
    print("T3: Morphology bridge -- SPARC galaxies vs Miyaoka clusters")
    print("=" * 70)

    miyaoka_bridge = [
        ('IC 2574',  'Sm', 0.060, 'J1023/J1217', 0.85),
        ('NGC 0300', 'Sd', 0.206, 'J1311/J1115', 0.73),
        ('NGC 3198', 'Sc', 0.397, 'J0231/J1258', 0.53),
        ('NGC 2841', 'Sb', 2.970, 'J1415',       0.20),
    ]

    print(f"\n  Miyaoka reference (v3.7 sec 8-6):")
    print(f"  {'SPARC galaxy':<12} {'morph':<5} {'gc/a0':>8} {'Miyaoka type':<14} {'S_plastic':>10}")
    for name, morph, gc, miy, sp in miyaoka_bridge:
        print(f"  {name:<12} {morph:<5} {gc:>8.3f} {miy:<14} {sp:>10.2f}")

    Ttype = np.array([g['Ttype'] for g in galaxies])
    S     = np.array([g['S_gal'] for g in galaxies])
    gc    = np.array([g['gc_a0'] for g in galaxies])

    type_bins = [
        ('Sa-Sb (T<=3)',    -2, 3),
        ('Sbc-Sc (T=4-5)',   4, 5),
        ('Scd-Sd (T=6-7)',   6, 7),
        ('Sdm-Sm (T=8-9)',   8, 9),
        ('Im (T>=10)',      10, 12),
    ]

    print(f"\n  SPARC by morphology:")
    print(f"  {'Type':<18} {'N':>4} {'med S_gal':>10} {'med gc/a0':>10} {'Miyaoka analog'}")
    print(f"  {'-'*65}")

    for label, tlo, thi in type_bins:
        mask = np.isfinite(Ttype) & np.isfinite(S) & (Ttype >= tlo) & (Ttype <= thi)
        if np.sum(mask) >= 3:
            med_S  = np.median(S[mask])
            med_gc = np.median(gc[mask])
            if med_S > 0.7:
                miy = 'J1023 (S~0.85)'
            elif med_S > 0.5:
                miy = 'J1311 (S~0.73)'
            elif med_S > 0.3:
                miy = 'J0231 (S~0.53)'
            else:
                miy = 'J1415 (S~0.20)'
            print(f"  {label:<18} {np.sum(mask):>4} {med_S:>10.3f} {med_gc:>10.3f}  {miy}")

    # Monotonicity
    mask_all = np.isfinite(Ttype) & np.isfinite(S)
    if np.sum(mask_all) > 10:
        r_TS, p_TS = stats.spearmanr(Ttype[mask_all], S[mask_all])
        print(f"\n  Ttype vs S_gal: rho={r_TS:+.3f}, p={p_TS:.2e}")
        if r_TS > 0:
            print(f"  -> Monotonic: later type = more plastic")
        else:
            print(f"  -> Non-monotonic")


# ============================================================
# T4: C15 residual vs S_gal
# ============================================================
def test_T4(galaxies):
    print("\n" + "=" * 70)
    print("T4: C15 residual Delta(gc) vs S_gal")
    print("=" * 70)

    delta = np.array([g['delta_gc'] for g in galaxies])
    S     = np.array([g['S_gal'] for g in galaxies])
    mask  = np.isfinite(delta) & np.isfinite(S)
    n = np.sum(mask)

    print(f"\n  Valid galaxies: {n}")
    if n < 10:
        print("  Insufficient data")
        return

    r, p = stats.spearmanr(S[mask], delta[mask])
    print(f"  Spearman: rho = {r:+.3f}, p = {p:.2e}")

    slope, intercept, rval, pval, stderr = stats.linregress(S[mask], delta[mask])
    print(f"  OLS: slope = {slope:.3f}+/-{stderr:.3f}, R2 = {rval**2:.3f}")

    if abs(r) < 0.1 and p > 0.1:
        print(f"\n  -> S_gal is uncorrelated with C15 residual")
        print(f"     Implication: C15 eta(Yd) already absorbs plasticity effect")
    else:
        print(f"\n  -> S_gal correlates with C15 residual")
        print(f"     Implication: plasticity component exists beyond eta(Yd)")


# ============================================================
# T5: Deep-MOND amplification
# ============================================================
def test_T5(galaxies):
    print("\n" + "=" * 70)
    print("T5: Deep-MOND amplification of plasticity effect")
    print("=" * 70)

    gc     = np.array([g['gc_a0'] for g in galaxies])
    S      = np.array([g['S_gal'] for g in galaxies])
    log_gc = np.array([g['log_gc_a0'] for g in galaxies])

    deep_mask   = np.isfinite(S) & np.isfinite(log_gc) & (gc < 0.3)
    mid_mask    = np.isfinite(S) & np.isfinite(log_gc) & (gc >= 0.3) & (gc < 2.0)
    newton_mask = np.isfinite(S) & np.isfinite(log_gc) & (gc >= 2.0)

    print(f"\n  Regime classification:")
    for label, mask in [('Deep MOND (gc<0.3a0)', deep_mask),
                        ('Transition (0.3-2.0a0)', mid_mask),
                        ('Newtonian (gc>2a0)', newton_mask)]:
        n = np.sum(mask)
        if n > 5:
            r, p = stats.spearmanr(S[mask], log_gc[mask])
            print(f"  {label:<28} N={n:>4}, rho(S,gc)={r:+.3f}, p={p:.2e}")
        else:
            print(f"  {label:<28} N={n:>4} (insufficient)")

    print(f"\n  S_gal variance by gc regime:")
    for label, mask in [('Deep MOND', deep_mask),
                        ('Transition', mid_mask),
                        ('Newtonian', newton_mask)]:
        n = np.sum(mask)
        if n > 5:
            std_S  = np.std(S[mask])
            std_gc = np.std(log_gc[mask])
            print(f"  {label:<14} std(S_gal)={std_S:.3f}, std(log gc)={std_gc:.3f}")


# ============================================================
# T6: PCA
# ============================================================
def test_T6(galaxies):
    print("\n" + "=" * 70)
    print("T6: PCA -- independent dimensions of plasticity indicators")
    print("=" * 70)

    fg = np.array([g['fg_norm'] for g in galaxies])
    yd = np.array([g['yd_norm'] for g in galaxies])
    sb = np.array([g['sb_norm'] for g in galaxies])
    tt = np.array([g['tt_norm'] for g in galaxies])

    mask = np.isfinite(fg) & np.isfinite(yd) & np.isfinite(sb) & np.isfinite(tt)
    n = np.sum(mask)
    print(f"\n  All 4 indicators valid: {n} galaxies")

    if n < 15:
        print("  Insufficient for PCA")
        return

    X = np.column_stack([fg[mask], yd[mask], sb[mask], tt[mask]])
    X_centered = X - X.mean(axis=0)

    cov_mat = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    var_ratio = eigenvalues / np.sum(eigenvalues)

    labels_pc = ['f_gas', 'Yd', '1-SBdisk', 'Ttype']
    print(f"\n  Principal components:")
    for i in range(4):
        loadings = ', '.join(f'{v:.3f}' for v in eigenvectors[:, i])
        print(f"  PC{i+1}: var={100*var_ratio[i]:.1f}%  loadings=[{loadings}]")

    print(f"\n  PC1+PC2 cumulative: {100*(var_ratio[0]+var_ratio[1]):.1f}%")

    # PC1 vs gc
    pc1 = X_centered @ eigenvectors[:, 0]
    log_gc = np.array([g['log_gc_a0'] for g in galaxies])
    r, p = stats.spearmanr(pc1, log_gc[mask])
    print(f"  PC1 vs log(gc/a0): rho={r:+.3f}, p={p:.2e}")


# ============================================================
# T7: Im-type irregulars
# ============================================================
def test_T7(galaxies):
    print("\n" + "=" * 70)
    print("T7: Im-type irregular galaxies (T>=10)")
    print("=" * 70)

    Ttype   = np.array([g['Ttype'] for g in galaxies])
    gc      = np.array([g['gc_a0'] for g in galaxies])
    S       = np.array([g['S_gal'] for g in galaxies])
    f_gas   = np.array([g['f_gas'] for g in galaxies])
    Yd      = np.array([g['Yd'] for g in galaxies])
    delta   = np.array([g['delta_gc'] for g in galaxies])
    vbar_a  = np.array([g['vbar_asym'] for g in galaxies])
    gas_ext = np.array([g['gas_extent'] for g in galaxies])

    im_mask    = np.isfinite(Ttype) & (Ttype >= 10)
    other_mask = np.isfinite(Ttype) & (Ttype < 10)

    n_im    = np.sum(im_mask)
    n_other = np.sum(other_mask)
    print(f"\n  Im-type: {n_im} galaxies,  Other: {n_other} galaxies")

    if n_im < 5:
        print("  Insufficient Im-type data")
        return

    print(f"\n  {'Indicator':<15} {'Im median':>12} {'Other med':>12} {'p (MW)':>12}")
    print(f"  {'-'*55}")

    for label, arr in [('gc/a0', gc), ('S_gal', S), ('f_gas', f_gas),
                       ('Yd', Yd), ('Delta(gc)', delta),
                       ('gas_extent', gas_ext), ('vbar_asym', vbar_a)]:
        m_im = im_mask & np.isfinite(arr)
        m_ot = other_mask & np.isfinite(arr)
        if np.sum(m_im) >= 3 and np.sum(m_ot) >= 3:
            med_im = np.median(arr[m_im])
            med_ot = np.median(arr[m_ot])
            U, p = stats.mannwhitneyu(arr[m_im], arr[m_ot], alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {label:<15} {med_im:>12.3f} {med_ot:>12.3f} {p:>12.2e} {sig}")

    im_deep = im_mask & (gc < 0.1)
    print(f"\n  Im-type with gc < 0.1a0: {np.sum(im_deep)} "
          f"({100*np.sum(im_deep)/max(n_im,1):.0f}%)")


# ============================================================
# T8: kappa=0 physical consistency
# ============================================================
def test_T8(galaxies):
    print("\n" + "=" * 70)
    print("T8: kappa=0 plasticity consistency check")
    print("=" * 70)

    S     = np.array([g['S_gal'] for g in galaxies])
    gc    = np.array([g['log_gc_a0'] for g in galaxies])
    delta = np.array([g['delta_gc'] for g in galaxies])
    Yd    = np.array([g['Yd'] for g in galaxies])

    mask = np.isfinite(S) & np.isfinite(gc) & np.isfinite(delta)
    n = np.sum(mask)
    print(f"\n  Valid galaxies: {n}")

    # Partial correlation: S_gal vs gc | Yd
    mask2 = mask & np.isfinite(Yd)
    if np.sum(mask2) > 20:
        sl1, in1, _, _, _ = stats.linregress(Yd[mask2], S[mask2])
        sl2, in2, _, _, _ = stats.linregress(Yd[mask2], gc[mask2])
        res_S  = S[mask2]  - (sl1 * Yd[mask2] + in1)
        res_gc = gc[mask2] - (sl2 * Yd[mask2] + in2)
        r_part, p_part = stats.spearmanr(res_S, res_gc)
        print(f"\n  Partial correlation S_gal vs gc | Yd:")
        print(f"    rho = {r_part:+.3f}, p = {p_part:.2e}")
        if p_part < 0.05:
            print(f"    -> After controlling Yd, S_gal-gc correlation survives")
            print(f"       Implication: plasticity has component independent of eta(Yd)")
        else:
            print(f"    -> After controlling Yd, S_gal-gc correlation vanishes")
            print(f"       Implication: S_gal information is fully absorbed by eta(Yd)")

    print(f"\n  kappa=0 physical interpretation:")
    print(f"    Old (v3.7): plasticity creates r-dependent gc(r) via C14")
    print(f"    New (v4.7): plasticity appears as inter-galaxy gc variation (C15 scatter)")
    print(f"    kappa=0 means no spatial coupling within membrane,")
    print(f"    but allows different global f_p per galaxy.")
    print(f"    S_gal is an observational proxy for this f_p.")


# ============================================================
# Figure
# ============================================================
def make_figure(galaxies):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWARNING: matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('SPARC Plasticity Analysis (kappa=0)', fontsize=14, fontweight='bold')

    S      = np.array([g['S_gal'] for g in galaxies])
    gc     = np.array([g['log_gc_a0'] for g in galaxies])
    Ttype  = np.array([g['Ttype'] for g in galaxies])
    f_gas  = np.array([g['f_gas'] for g in galaxies])
    delta  = np.array([g['delta_gc'] for g in galaxies])
    gc_a0  = np.array([g['gc_a0'] for g in galaxies])

    # P1: S_gal distribution
    ax = axes[0, 0]
    m = np.isfinite(S)
    if np.sum(m) > 0:
        ax.hist(S[m], bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('S_gal')
    ax.set_ylabel('Count')
    ax.set_title('T1: S_gal distribution')

    # P2: S_gal vs gc
    ax = axes[0, 1]
    m = np.isfinite(S) & np.isfinite(gc)
    if np.sum(m) > 0:
        ax.scatter(S[m], gc[m], s=10, alpha=0.5)
        if np.sum(m) > 5:
            z = np.polyfit(S[m], gc[m], 1)
            x_line = np.linspace(np.min(S[m]), np.max(S[m]), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r-', alpha=0.7)
    ax.set_xlabel('S_gal')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('T2: S_gal vs gc')

    # P3: Ttype vs S_gal
    ax = axes[0, 2]
    m = np.isfinite(Ttype) & np.isfinite(S)
    if np.sum(m) > 0:
        ax.scatter(Ttype[m], S[m], s=10, alpha=0.5)
    ax.set_xlabel('T-type')
    ax.set_ylabel('S_gal')
    ax.set_title('T3: Morphology bridge')

    # P4: f_gas vs gc
    ax = axes[0, 3]
    m = np.isfinite(f_gas) & np.isfinite(gc)
    if np.sum(m) > 0:
        ax.scatter(f_gas[m], gc[m], s=10, alpha=0.5, c='green')
    ax.set_xlabel('f_gas')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('T1: f_gas vs gc (Miyaoka analog)')

    # P5: Delta(gc) vs S_gal
    ax = axes[1, 0]
    m = np.isfinite(S) & np.isfinite(delta)
    if np.sum(m) > 0:
        ax.scatter(S[m], delta[m], s=10, alpha=0.5, c='orange')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('S_gal')
    ax.set_ylabel('Delta(gc) [C15 residual]')
    ax.set_title('T4: C15 residual vs S_gal')

    # P6: Deep MOND coloring
    ax = axes[1, 1]
    m = np.isfinite(S) & np.isfinite(gc)
    if np.sum(m) > 0:
        colors_arr = ['blue' if g < -0.5 else 'green' if g < 0.3 else 'red'
                      for g in gc[m]]
        ax.scatter(S[m], gc[m], s=10, alpha=0.5, c=colors_arr)
    ax.set_xlabel('S_gal')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('T5: Deep MOND (blue) vs Newton (red)')

    # P7: Im type highlight
    ax = axes[1, 2]
    m_im = np.isfinite(S) & np.isfinite(gc) & np.isfinite(Ttype) & (Ttype >= 10)
    m_ot = np.isfinite(S) & np.isfinite(gc) & np.isfinite(Ttype) & (Ttype < 10)
    if np.sum(m_ot) > 0:
        ax.scatter(S[m_ot], gc[m_ot], s=8, alpha=0.3, c='grey', label='T<10')
    if np.sum(m_im) > 0:
        ax.scatter(S[m_im], gc[m_im], s=25, alpha=0.8, c='red', marker='*', label='Im (T>=10)')
    ax.set_xlabel('S_gal')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('T7: Im-type galaxies')
    ax.legend(fontsize=7)

    # P8: Miyaoka bridge overlay
    ax = axes[1, 3]
    m = np.isfinite(Ttype) & np.isfinite(S) & np.isfinite(gc)
    for tlo, thi, color, label in [(0, 3, 'red', 'Sa-Sb'),
                                   (4, 5, 'orange', 'Sbc-Sc'),
                                   (6, 7, 'green', 'Scd-Sd'),
                                   (8, 9, 'blue', 'Sdm-Sm'),
                                   (10, 12, 'purple', 'Im')]:
        tm = m & (Ttype >= tlo) & (Ttype <= thi)
        if np.sum(tm) > 0:
            ax.scatter(np.median(S[tm]), np.median(gc[tm]),
                       s=100, c=color, label=label, zorder=5)
    # Miyaoka reference
    miy_gc = [np.log10(2.97), np.log10(0.397), np.log10(0.206), np.log10(0.060)]
    miy_sp = [0.20, 0.53, 0.73, 0.85]
    ax.scatter(miy_sp, miy_gc, s=80, marker='D', c='black', zorder=10,
               label='Miyaoka ref')
    ax.set_xlabel('S_gal / S_plastic')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('T3: SPARC vs Miyaoka bridge')
    ax.legend(fontsize=6)

    plt.tight_layout()
    out = str(BASE / 'sparc_plasticity_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {out}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("SPARC 175 galaxies: Plasticity region integrated analysis")
    print("Miyaoka 2018 independent verification + kappa=0 interpretation")
    print("=" * 70)

    galaxies = build_galaxies()
    print(f"\n  Galaxies loaded: {len(galaxies)}")

    if len(galaxies) < 10:
        print("ERROR: too few galaxies loaded")
        sys.exit(1)

    # Run all tests
    correlations = test_T1(galaxies)
    test_T2(galaxies)
    test_T3(galaxies)
    test_T4(galaxies)
    test_T5(galaxies)
    test_T6(galaxies)
    test_T7(galaxies)
    test_T8(galaxies)

    # Figure
    make_figure(galaxies)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
  This analysis independently tests the plasticity region hypothesis
  (Miyaoka 2018, 16 clusters) at SPARC galaxy scale (175 galaxies).

  S_gal (galaxy-level plasticity indicator):
    f_gas:    gas fraction (plastic -> high)
    Yd:       mass-to-light ratio (plastic -> high)
    1-SBdisk: inverse surface brightness (plastic -> low SB)
    Ttype:    morphological type (plastic -> late type)

  Under kappa=0 (v4.7):
    Plasticity appears as inter-galaxy gc variation (C15 scatter 0.286 dex),
    NOT as radial gc(r) dependence (C14).
    S_gal is the observational proxy for f_p (plasticity fraction per galaxy).

  Miyaoka bridge:
    If S_plastic (cluster) and S_gal (galaxy) reflect the same physics
    (membrane elastic/plastic two-phase structure at different scales),
    a monotonic correspondence via morphological type is expected.
    """)


if __name__ == '__main__':
    main()
