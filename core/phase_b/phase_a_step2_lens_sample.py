# -*- coding: utf-8 -*-
"""
phase_a_step2_lens_sample.py

Step 2: Construct isolated lens sample from GAMA DR4.
"""

import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table, join
from scipy.spatial import cKDTree

GAMA_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\GAMA_DR4")
OUTPUT_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase_a_output")

FILES = {
    'science':   GAMA_DIR / 'gkvScienceCatv02.fits',
    'masses':    GAMA_DIR / 'StellarMassesGKVv24.fits',
    'matches':   GAMA_DIR / 'gkvGamaIIMatchesv01.fits',
    'groups':    GAMA_DIR / 'G3CGalv10.fits',
}

# Column names verified from Step 1 inspection
COLS = {
    'sci_id':     'uberID',
    'sci_ra':     'RAcen',
    'sci_dec':    'Deccen',
    'sci_z':      'Z',
    'sci_nq':     'NQ',
    'sci_mag':    'mag',         # GKV r-band Kron
    'sci_uclass': 'uberclass',
    'mass_id':    'uberID',
    'mass_logm':  'logmstar',
    'match_uber': 'uberID',
    'match_cata': 'CATAID',
    'grp_id':     'CATAID',
    'grp_group':  'GroupID',     # 0 = not in a group (isolated)
    'grp_rank':   'RankIterCen', # -999 = not in group, 1 = BCG
}

# Brouwer+2021 selections
Z_MIN, Z_MAX = 0.1, 0.5
NQ_MIN = 3
MAG_R_MAX = 20.0
LOGMSTAR_MIN, LOGMSTAR_MAX = 8.5, 11.5
MSTAR_BINS = [8.5, 10.3, 10.6, 10.8, 11.0, 11.5]

# G12 field
G12_RA_MIN, G12_RA_MAX = 174.0, 186.0
G12_DEC_MIN, G12_DEC_MAX = -3.0, 2.0

# Isolation
ISO_PROJ_MPC = 3.0
ISO_DLOGM_MAX = 0.5
ISO_DZ_MAX = 0.01

H0 = 70.0
c_km = 3e5


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def load_table(key):
    path = FILES[key]
    print(f"  Loading {key}: {path.name} ...", end='', flush=True)
    t = Table.read(path)
    print(f" {len(t):,} rows")
    return t


def comoving_to_angular(R_mpc, z):
    D_A = c_km * z / H0
    return np.degrees(R_mpc / max(D_A, 1.0))


def main():
    section("STEP 2: GAMA LENS SAMPLE CONSTRUCTION")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("Loading GAMA tables")
    sci = load_table('science')
    mass = load_table('masses')
    matches = load_table('matches')
    grp = load_table('groups')

    # ----- Join 1: ScienceCat x StellarMasses (uberID) -----
    section("Join 1: ScienceCat x StellarMasses (uberID)")
    sci_cols = [COLS['sci_id'], COLS['sci_ra'], COLS['sci_dec'],
                COLS['sci_z'], COLS['sci_nq'], COLS['sci_mag'],
                COLS['sci_uclass']]
    mass_cols = [COLS['mass_id'], COLS['mass_logm']]

    sci_sub = sci[sci_cols]
    mass_sub = mass[mass_cols]

    # Rename mass Z to avoid collision (both have Z)
    merged = join(sci_sub, mass_sub, keys=COLS['sci_id'], join_type='inner')
    print(f"  Result: {len(merged):,} rows")

    # ----- Quality cuts -----
    section("Quality cuts")
    n0 = len(merged)

    uclass = merged[COLS['sci_uclass']]
    # From sample: uberclass=2 for galaxies. gkvInputCat had class=[4,1,1]
    # gkvScienceCat sample [2,2,2] → galaxies only in ScienceCat, but let's check range
    uc_unique = np.unique(uclass)
    print(f"  uberclass values in ScienceCat: {uc_unique}")
    # Conservative: keep uclass==1 (primary galaxy) and 2 (galaxy)
    mask_gal = np.isin(uclass, [1, 2])
    merged = merged[mask_gal]
    print(f"  Galaxy class (1 or 2): {n0:,} -> {len(merged):,}")

    n1 = len(merged)
    merged = merged[merged[COLS['sci_nq']] >= NQ_MIN]
    print(f"  NQ >= {NQ_MIN}: {n1:,} -> {len(merged):,}")

    n2 = len(merged)
    merged = merged[(merged[COLS['sci_z']] >= Z_MIN) &
                    (merged[COLS['sci_z']] <= Z_MAX)]
    print(f"  z in [{Z_MIN}, {Z_MAX}]: {n2:,} -> {len(merged):,}")

    n3 = len(merged)
    merged = merged[merged[COLS['sci_mag']] <= MAG_R_MAX]
    print(f"  mag_r < {MAG_R_MAX}: {n3:,} -> {len(merged):,}")

    n4 = len(merged)
    merged = merged[(merged[COLS['mass_logm']] >= LOGMSTAR_MIN) &
                    (merged[COLS['mass_logm']] <= LOGMSTAR_MAX)]
    print(f"  logM* in [{LOGMSTAR_MIN}, {LOGMSTAR_MAX}]: {n4:,} -> {len(merged):,}")

    # ----- G12 field cut -----
    section("G12 field selection")
    n5 = len(merged)
    merged = merged[(merged[COLS['sci_ra']] >= G12_RA_MIN) &
                    (merged[COLS['sci_ra']] <= G12_RA_MAX) &
                    (merged[COLS['sci_dec']] >= G12_DEC_MIN) &
                    (merged[COLS['sci_dec']] <= G12_DEC_MAX)]
    print(f"  G12 (RA 174-186, Dec -3-+2): {n5:,} -> {len(merged):,}")

    # ----- Join 2: uberID -> CATAID -----
    section("Join 2: Add CATAID via GamaIIMatches")
    # Use only unique best matches (nmatch=1 or take first per uberID)
    match_sub = matches[[COLS['match_uber'], COLS['match_cata']]]
    # Keep only CATAID > 0 to avoid null joins
    match_sub = match_sub[match_sub[COLS['match_cata']] > 0]
    # Deduplicate on uberID (first occurrence)
    _, unique_idx = np.unique(np.array(match_sub[COLS['match_uber']]),
                              return_index=True)
    match_sub = match_sub[sorted(unique_idx)]
    print(f"  Matches table (unique uberID with CATAID>0): {len(match_sub):,}")

    merged = join(merged, match_sub, keys=COLS['match_uber'], join_type='inner')
    print(f"  With CATAID: {len(merged):,}")

    # ----- Join 3: CATAID -> G3CGal group info -----
    section("Join 3: Group info via G3CGal")
    grp_cols = [COLS['grp_id'], COLS['grp_group'], COLS['grp_rank']]
    grp_sub = grp[grp_cols]
    # G3CGal may have duplicate CATAID (one galaxy can be in multiple groups)
    _, unique_idx = np.unique(np.array(grp_sub[COLS['grp_id']]),
                              return_index=True)
    grp_sub = grp_sub[sorted(unique_idx)]
    print(f"  G3CGal (unique CATAID): {len(grp_sub):,}")

    merged = join(merged, grp_sub, keys=COLS['grp_id'], join_type='left')

    # ----- Isolation -----
    section("Isolation criterion (GAMA G3C groups)")

    # Diagnostic
    group_col = COLS['grp_group']
    rank_col = COLS['grp_rank']

    if hasattr(merged[group_col], 'mask'):
        group_val = np.array(merged[group_col].filled(0))
    else:
        group_val = np.array(merged[group_col])
    if hasattr(merged[rank_col], 'mask'):
        rank_val = np.array(merged[rank_col].filled(-999))
    else:
        rank_val = np.array(merged[rank_col])

    print(f"  GroupID == 0 (not in a G3C group): {(group_val == 0).sum():,}")
    print(f"  RankIterCen == 1 (BCG of group): {(rank_val == 1).sum():,}")
    print(f"  RankIterCen == -999 (no group info): {(rank_val == -999).sum():,}")
    print(f"  In a group, not BCG: {((group_val != 0) & (rank_val != 1) & (rank_val != -999)).sum():,}")

    # Brouwer-like isolation:
    #   isolated = GroupID == 0 (no group detected by G3C FoF with M*>M*_thresh)
    #   OR BCG (dominant central)
    # Note: G3C FoF threshold is r_Petro<=19.4 for groups, so some of our
    # "isolated" may still have faint neighbors. Augment with KDTree check
    # for M*-ratio neighbors.

    mask_group_iso = (group_val == 0) | (rank_val == 1)
    merged_giso = merged[mask_group_iso]
    print(f"  Group-isolated: {len(merged):,} -> {len(merged_giso):,}")

    # ----- Additional projected isolation (M*-ratio neighbor check) -----
    section("Additional projected M*-neighbor check (3 Mpc/h, dlogM*<0.5)")

    ra = np.array(merged_giso[COLS['sci_ra']], dtype=float)
    dec = np.array(merged_giso[COLS['sci_dec']], dtype=float)
    z_arr = np.array(merged_giso[COLS['sci_z']], dtype=float)
    logm = np.array(merged_giso[COLS['mass_logm']], dtype=float)
    N = len(merged_giso)

    # Approx equirectangular for tree
    cos_dec = np.cos(np.radians(np.median(dec)))
    xy = np.column_stack([ra * cos_dec, dec])
    tree = cKDTree(xy)

    isolated = np.ones(N, dtype=bool)
    report_step = max(1000, N // 20)

    for i in range(N):
        theta_max = comoving_to_angular(ISO_PROJ_MPC, z_arr[i])
        neighbors = tree.query_ball_point(xy[i], theta_max)
        for j in neighbors:
            if j == i:
                continue
            if abs(z_arr[j] - z_arr[i]) > ISO_DZ_MAX:
                continue
            if logm[j] > logm[i] - ISO_DLOGM_MAX:
                # brighter-or-comparable neighbor exists
                isolated[i] = False
                break
        if (i + 1) % report_step == 0:
            print(f"    ...{i+1:,}/{N:,} ({isolated[:i+1].sum():,} still isolated)")

    merged_iso = merged_giso[isolated]
    print(f"  After M*-neighbor check: {N:,} -> {len(merged_iso):,}")

    # ----- Summary -----
    section("LENS SAMPLE SUMMARY")
    lenses = merged_iso
    z_arr = np.array(lenses[COLS['sci_z']])
    logm = np.array(lenses[COLS['mass_logm']])

    print(f"  Total lenses: {len(lenses):,}")
    print(f"  Redshift: median={np.median(z_arr):.3f}, "
          f"range=[{np.min(z_arr):.3f}, {np.max(z_arr):.3f}]")
    print(f"  log M*: median={np.median(logm):.2f}, "
          f"range=[{np.min(logm):.2f}, {np.max(logm):.2f}]")

    print(f"\n  M* bin distribution:")
    for i in range(len(MSTAR_BINS) - 1):
        mask = (logm >= MSTAR_BINS[i]) & (logm < MSTAR_BINS[i + 1])
        print(f"    [{MSTAR_BINS[i]:.1f}, {MSTAR_BINS[i+1]:.1f}): "
              f"{mask.sum():,} lenses")

    # ----- Save -----
    out_full = OUTPUT_DIR / 'gama_lenses_g12_isolated.fits'
    lenses.write(out_full, overwrite=True)
    print(f"\n  Saved: {out_full} ({out_full.stat().st_size/1e6:.1f} MB)")

    key_cols = [COLS['sci_id'], COLS['sci_ra'], COLS['sci_dec'],
                COLS['sci_z'], COLS['mass_logm']]
    lenses_lite = lenses[key_cols]
    out_lite = OUTPUT_DIR / 'gama_lenses_g12_lite.fits'
    lenses_lite.write(out_lite, overwrite=True)
    print(f"  Lite: {out_lite} ({out_lite.stat().st_size/1e6:.1f} MB)")

    section("STEP 2 COMPLETE")


if __name__ == '__main__':
    main()
