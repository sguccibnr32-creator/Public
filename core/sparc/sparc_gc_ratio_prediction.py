# -*- coding: utf-8 -*-
"""
sparc_gc_ratio_prediction.py

Test 2: Does C15 quantitatively predict the KiDS Fig-8 gc ratio?

Approach:
  1. Split SPARC by T-type as proxy for color/Sersic
     - "Red/High-n" proxy: T <= 3 (E, S0, Sa, Sab, Sb)
     - "Blue/Low-n" proxy: T > 3 (Sbc, Sc, Scd, Sd, Sdm, Sm, Im, BCD)
  2. Compute median vflat, hR, Yd for each group
  3. Compute C15 gc for each group
  4. Compare predicted ratio with KiDS observed (2.3-2.8)

Usage: python sparc_gc_ratio_prediction.py
"""

import csv
import numpy as np
from pathlib import Path
from scipy import stats

a0 = 1.2e-10
kpc_m = 3.0857e19

BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
TA3_CSV = BASE / "TA3_gc_independent.csv"
PHASE1_CSV = BASE / "phase1" / "sparc_results.csv"
MRT_FILE = BASE / "SPARC_Lelli2016c.mrt"


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
    """Parse SPARC MRT: sep>=4 then split(). Columns per memory."""
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
                    'Rdisk': float(p[11]),
                    'Vflat': float(p[15]),
                }
            except (ValueError, IndexError):
                continue
    return galaxies


def c15_gc(Yd, vflat_kms, hR_kpc):
    """C15: gc = 0.584 * Yd^{-0.361} * sqrt(a0 * vflat^2 / hR)"""
    return 0.584 * Yd**(-0.361) * np.sqrt(a0 * (vflat_kms * 1e3)**2 / (hR_kpc * kpc_m))


def main():
    print("=" * 72)
    print("  TEST 2: C15 PREDICTED gc RATIO vs KiDS OBSERVED")
    print("=" * 72)

    ta3 = load_ta3()
    p1 = load_phase1()
    mrt = load_mrt()

    common = sorted(set(ta3) & set(p1) & set(mrt))
    print(f"\n  Merged: {len(common)} galaxies")

    gc_a0 = np.array([ta3[n]['gc_a0'] for n in common])
    vflat = np.array([p1[n]['vflat'] for n in common])
    hR    = np.array([mrt[n]['Rdisk'] for n in common])
    Yd    = np.array([p1[n]['Yd'] for n in common])
    T     = np.array([mrt[n]['T'] for n in common], dtype=float)

    valid = (gc_a0 > 0) & (vflat > 0) & (hR > 0) & (Yd > 0) & np.isfinite(T)
    gc_a0, vflat, hR, Yd, T = gc_a0[valid], vflat[valid], hR[valid], Yd[valid], T[valid]
    N = len(gc_a0)
    print(f"  After quality cut: {N}")

    gc_c15 = c15_gc(Yd, vflat, hR) / a0
    Sigma_dyn = vflat**2 / hR

    # ==============================================================
    print(f"\n{'='*72}")
    print(f"  T-TYPE SPLIT ANALYSIS")
    print(f"{'='*72}")

    t_cuts = [
        (3, 'T<=3 (E-Sb)', 'T>3 (Sbc-Im)'),
        (5, 'T<=5 (E-Sc)', 'T>5 (Scd-Im)'),
        (1, 'T<=1 (E-Sa)', 'T>1 (Sab-Im)'),
    ]

    for t_cut, label_early, label_late in t_cuts:
        early = T <= t_cut
        late = T > t_cut
        n_e, n_l = int(early.sum()), int(late.sum())

        if n_e < 5 or n_l < 5:
            print(f"\n  Cut T={t_cut}: skipped (n_early={n_e}, n_late={n_l})")
            continue

        print(f"\n  --- Cut T={t_cut}: {label_early} (N={n_e}) vs {label_late} (N={n_l}) ---")

        print(f"\n  {'Parameter':>14s}  {'Early':>12s}  {'Late':>12s}  {'Ratio(E/L)':>12s}")
        for lbl, arr in [('vflat (km/s)', vflat), ('hR (kpc)', hR),
                         ('Yd', Yd), ('Sigma_dyn', Sigma_dyn)]:
            med_e = np.median(arr[early])
            med_l = np.median(arr[late])
            ratio = med_e / med_l if med_l > 0 else np.nan
            print(f"  {lbl:>14s}  {med_e:12.3f}  {med_l:12.3f}  {ratio:12.3f}")

        gc_med_e = np.median(gc_a0[early])
        gc_med_l = np.median(gc_a0[late])
        ratio_obs = gc_med_e / gc_med_l

        gc_c15_e = np.median(gc_c15[early])
        gc_c15_l = np.median(gc_c15[late])
        ratio_c15 = gc_c15_e / gc_c15_l

        gc_from_med_e = c15_gc(np.median(Yd[early]), np.median(vflat[early]),
                               np.median(hR[early])) / a0
        gc_from_med_l = c15_gc(np.median(Yd[late]), np.median(vflat[late]),
                               np.median(hR[late])) / a0
        ratio_from_med = gc_from_med_e / gc_from_med_l

        print(f"\n  {'gc measure':>25s}  {'Early':>10s}  {'Late':>10s}  {'Ratio':>10s}")
        print(f"  {'Observed (TA3 median)':>25s}  {gc_med_e:10.3f}  {gc_med_l:10.3f}  {ratio_obs:10.2f}")
        print(f"  {'C15 median(gc_i)':>25s}  {gc_c15_e:10.3f}  {gc_c15_l:10.3f}  {ratio_c15:10.2f}")
        print(f"  {'C15 from median params':>25s}  {gc_from_med_e:10.3f}  {gc_from_med_l:10.3f}  {ratio_from_med:10.2f}")
        print(f"  {'KiDS Color obs':>25s}  {'---':>10s}  {'---':>10s}  {'2.3-2.8':>10s}")
        print(f"  {'KiDS Sersic obs':>25s}  {'---':>10s}  {'---':>10s}  {'2.4-2.5':>10s}")

        Yd_e, Yd_l = np.median(Yd[early]), np.median(Yd[late])
        Sig_e, Sig_l = np.median(Sigma_dyn[early]), np.median(Sigma_dyn[late])
        yd_factor = (Yd_e / Yd_l)**(-0.361)
        sig_factor = (Sig_e / Sig_l)**0.5
        combined = yd_factor * sig_factor

        print(f"\n  Ratio decomposition (C15 structure):")
        print(f"    Yd factor:    ({Yd_e:.3f}/{Yd_l:.3f})^-0.361 = {yd_factor:.3f}")
        print(f"    Sigma factor: ({Sig_e:.1f}/{Sig_l:.1f})^0.5  = {sig_factor:.3f}")
        print(f"    Combined:     {yd_factor:.3f} x {sig_factor:.3f} = {combined:.3f}")
        print(f"    Actual ratio: {ratio_obs:.3f}")

        ks, ks_p = stats.ks_2samp(gc_a0[early], gc_a0[late])
        mw, mw_p = stats.mannwhitneyu(gc_a0[early], gc_a0[late], alternative='greater')
        print(f"\n  Statistical tests (gc_early > gc_late):")
        print(f"    KS test: D={ks:.3f}, p={ks_p:.2e}")
        print(f"    Mann-Whitney (one-sided): U={mw:.0f}, p={mw_p:.2e}")

    # ==============================================================
    print(f"\n{'='*72}")
    print(f"  CONTINUOUS: gc vs T-type in 3 bins")
    print(f"{'='*72}")

    bins = [(-5, 2, 'E-Sab'), (2, 5, 'Sb-Sc'), (5, 11, 'Scd-Im')]
    print(f"\n  {'Bin':>10s}  {'N':>5s}  {'gc/a0':>8s}  {'gc_C15':>8s}  "
          f"{'vflat':>8s}  {'hR':>8s}  {'Yd':>7s}  {'Sigma':>10s}")
    for t_lo, t_hi, label in bins:
        mask = (T >= t_lo) & (T < t_hi)
        n = int(mask.sum())
        if n < 3:
            continue
        print(f"  {label:>10s}  {n:>5d}  {np.median(gc_a0[mask]):8.3f}  "
              f"{np.median(gc_c15[mask]):8.3f}  "
              f"{np.median(vflat[mask]):8.1f}  "
              f"{np.median(hR[mask]):8.2f}  "
              f"{np.median(Yd[mask]):7.3f}  "
              f"{np.median(Sigma_dyn[mask]):10.1f}")

    # ==============================================================
    print(f"\n{'='*72}")
    print(f"  GRAND COMPARISON: SPARC vs KiDS")
    print(f"{'='*72}")

    early = T <= 3
    late = T > 3
    gc_med_e = np.median(gc_a0[early])
    gc_med_l = np.median(gc_a0[late])
    gc_c15_e = np.median(gc_c15[early])
    gc_c15_l = np.median(gc_c15[late])

    print(f"\n  SPARC (T<=3 vs T>3):")
    print(f"    Observed gc ratio:   {gc_med_e/gc_med_l:.2f}")
    print(f"    C15 predicted ratio: {gc_c15_e/gc_c15_l:.2f}")
    print(f"\n  KiDS (Brouwer+2021 Fig-8):")
    print(f"    Color (Red/Blue):    2.3-2.8")
    print(f"    Sersic (High/Low):   2.4-2.5")

    gap = (gc_med_e/gc_med_l) - 2.5
    print(f"\n  Gap (SPARC obs - KiDS mid 2.5):  {gap:+.2f}")

    if abs(gap) < 0.5:
        print(f"  -> SPARC and KiDS gc ratios are CONSISTENT")
    elif gap > 0:
        print(f"  -> SPARC shows LARGER ratio than KiDS")
    else:
        print(f"  -> SPARC shows SMALLER ratio than KiDS")

    print(f"\n{'='*72}\n  ANALYSIS COMPLETE\n{'='*72}")


if __name__ == '__main__':
    main()
