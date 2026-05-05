#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
per_filter_sensitivity_extension.py (v2)
==========================================
Cross-method aggregation/imputation extension for Lesson 94 establishment.
Anchor 21 v0.2 (axis_1 full operational closure) - refinement 3.

v2 corrections (vs v1)
----------------------
v1 had two structural mismatches with v0.1.4 J4 canonical, both detected
during precondition fail (b_alpha measured 1.216 vs expected 0.108).

  Mismatch 1 (PRIMARY, b_alpha 11x off):
    v1 used 2-feature simple OLS (y = b*x + a, x=log10(g_C15), y=log10(g_obs))
    producing RAR-like slope ~1.2.
    v2 uses 3-feature partial OLS (canonical / J4 structure):
      target   = delta_primary = log10(g_obs) - log10(g_C15)
      features = [1, 2*log10(rho_gal), log10(r_h)]
      b_alpha  = beta[1]
    where b_alpha is the rho_gal^2 scaling exponent (universal coupling
    slope), with log_r_h partialled out via multiple regression.

  Mismatch 2 (SECONDARY, NaN count 3 vs 11):
    v1 applied Q<3 / vflat>0 filters BEFORE g_obs aggregation, so only
    3 NaN g_obs surfaced (the rest were filtered out by Q / vflat first).
    v2 reorders to canonical:
      bridge filter -> sparc_171 -> aggregate g_obs (-> 11 NaN) ->
      impute (per method) -> Q<3 / vflat>0 / finite delta_primary ->
      partial OLS sample.

Methods (unchanged)
-------------------
A0_baseline    : mean(V^2/r) for r > 2*hR, NaN g_obs filter exclusion (canonical)
A1_median_V2r  : median(V^2/r), NaN g_obs filter exclusion
A2_min_fill    : mean(V^2/r), NaN g_obs at sparc_171 -> min(valid g_obs)
                 [log-finite, no extrapolation, refinement Q2 corrected from zero_fill]
A3_mean_fill   : mean(V^2/r), NaN g_obs at sparc_171 -> mean(valid g_obs)
A4_knn_impute  : mean(V^2/r), NaN g_obs at sparc_171 -> k=5 KDTree imputation
                 features: [log Mstar, log Mgas, log hR, log SBdisk0]
                 distance: Euclidean, scaling: zero-mean unit-variance
                 (sklearn-free via scipy.spatial.KDTree)

Forensic compliance
-------------------
rule 1   : anchor IMMUTABLE (no modification of v0.1.4 anchor files)
rule #26 : multi-route minimum (standalone Q-intact loader + 3-feature partial
                                OLS replicating v0.1.4 J4 canonical structure)
rule 92  : parsimony first (5-method comparison in single script)

Provenance
----------
parent commit : 5783bef (anchor 21 v0.1.4)
parent tag    : companion-v0.4a-validation-2026-05-05
references    : phase_c3_step3_reference.py
                  (J4 standalone Q-intact loader + canonical OLS structure)
                per_filter_sensitivity.csv (J3 filter 3 establishment)

Output
------
per_filter_sensitivity_extension.csv (8-column schema, unchanged from v1)
columns: scenario, n_sample, n_imputed, b_alpha, SE_b_alpha,
         delta_vs_baseline_abs, delta_vs_baseline_rel, notes

Note on n_imputed:
  Counts imputed galaxies that pass all downstream filters and enter
  the OLS sample. Pool-level imputation count (sparc_171 NaN, ~11 per
  v0.1.4 J3 filter 3) is reported in the notes column.

Usage
-----
  uv run --with scipy --with numpy --with pandas \\
      python per_filter_sensitivity_extension.py \\
      --ta3 /path/to/Table_A3.dat \\
      --phase1 /path/to/phase1_TA1_TA2.csv \\
      --mrt /path/to/MRT_SPARC_175.txt \\
      --rotcurve_dir /path/to/SPARC_rotation_curves/ \\
      --output ./per_filter_sensitivity_extension.csv

USER-FILL sections
------------------
Five loader/derivation functions are marked USER-FILL. v1 carried
these from phase_c3_step3_reference.py successfully (loaders verified
correct via Phase 1-3 of v1 run). v2 retains them unchanged:
  - load_TA3, load_phase1, load_MRT_Q_intact, load_rotation_curves
  - compute_g_C15  (anchor 19 sec 1.5 formula)
  - aggregate_g_obs

NEW v2 functions (verify against canonical):
  - derive_partial_OLS_features: r_h = R_H_FACTOR * Rdisk,
                                 rho_gal = M_bar / ((4/3)*pi*r_h^3)
  - compute_partial_OLS:         3-feature partial OLS

If any v2 NEW function diverges from phase_c3_step3_reference.py
canonical convention, precondition_A0_check (Q5 beta) will catch it.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ============================================================================
# Constants (anchor 21 v0.1.4 reference values, IMMUTABLE)
# ============================================================================

EXPECTED_B_ALPHA = 0.108442979149252  # G1 / J4 (anchor 21 v0.1.4)
EXPECTED_SE      = 0.0153             # J1b (anchor 21 v0.1.4, partial OLS SE)
B_ALPHA_TOL      = 1e-10              # bit-exact (J4 strict)
SE_TOL           = 1e-3               # OLS SE tolerance (Q5 beta)

# k-NN imputation (refinement Q3)
KNN_K            = 5
KNN_FEATURE_COLS = ["log_Mstar", "log_Mgas", "log_hR", "log_SBdisk0"]

# C15 prediction constants (anchor 19 sec 1.5)
C15_COEF        = 0.584
C15_UPSILON_EXP = -0.361
A_0             = 1.2e-10                       # m/s^2
A0_KPC          = A_0 * 3.086e19 / 1e6          # ~ 3702 (km/s)^2/kpc

# Half-mass radius coefficient (exponential disk convention)
# r_h = R_H_FACTOR * Rdisk
# USER-FILL: verify R_H_FACTOR against phase_c3_step3_reference.py
R_H_FACTOR      = 1.68

# Bridge galaxies (anchor 19 sec 1.5)
BRIDGE_4 = [
    "NGC3741",
    "NGC2915",
    "ESO444-G084",
    "NGC1705",
]


# ============================================================================
# Standalone Q-intact loaders (J4 pattern, no canonical cdca6afd import)
# Identical to v1 (verified correct in v1 run Phase 1-3)
# ============================================================================

def load_TA3(path):
    """Load TA3_gc_independent.csv -- gc_over_a0 in units of a0."""
    df = pd.read_csv(path)
    df["galaxy"] = df["galaxy"].astype(str).str.strip()
    return df[["galaxy", "gc_over_a0"]]


def load_phase1(path):
    """Load phase1/sparc_results.csv -- mass-to-light ratio Upsilon_d (column 'ud')."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    if "galaxy" not in df.columns:
        for alt in ("Galaxy", "name", "Name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "galaxy"})
                break
    if "ud" not in df.columns:
        for alt in ("Ud", "Upsilon_d"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ud"})
                break
    df["galaxy"] = df["galaxy"].astype(str).str.strip()
    return df[["galaxy", "ud"]]


def load_MRT_Q_intact(path):
    """Load MRT_SPARC_175.txt with Q column intact (J4 pattern, 18-col schema)."""
    raw_names = ["galaxy", "T", "D", "e_D", "f_D", "Inc", "e_Inc",
                 "L36", "e_L36", "Reff", "SBeff", "Rdisk", "SBdisk0",
                 "MHI", "RHI", "Vflat", "e_Vflat", "Q"]
    df = pd.read_csv(path, sep=r"\s+", skiprows=98, header=None,
                     names=raw_names, usecols=range(len(raw_names)),
                     engine="python")
    df["galaxy"] = df["galaxy"].astype(str).str.strip()
    return df[["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat", "Q"]]


def load_rotation_curves(rotcurve_dir, galaxy_names):
    """Load per-radius rotation curves. Returns dict[galaxy] -> DataFrame ['r', 'V']."""
    rc_dir = Path(rotcurve_dir)
    rotcurves = {}
    for galaxy_name in galaxy_names:
        p = rc_dir / f"{galaxy_name}_rotmod.dat"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(
                p, sep=r"\s+", comment="#", header=None,
                names=["Rad", "Vobs", "errV", "Vgas", "Vdisk",
                       "Vbul", "SBdisk", "SBbul"],
                engine="python",
            )
            rotcurves[galaxy_name] = pd.DataFrame({
                "r": df["Rad"].values,
                "V": df["Vobs"].values,
            })
        except Exception:
            pass
    return rotcurves


def compute_g_C15(galaxy_df):
    """
    Compute g_C15 per galaxy from canonical formula (anchor 19 sec 1.5):
        gc_C15 = C15_COEF * Upsilon_d^C15_UPSILON_EXP
                 * sqrt(A0_KPC * Vflat^2 / Rdisk)
    """
    return pd.Series(
        C15_COEF * galaxy_df["ud"] ** C15_UPSILON_EXP
        * np.sqrt(A0_KPC * galaxy_df["Vflat"] ** 2 / galaxy_df["Rdisk"]),
        index=galaxy_df.index,
        name="g_C15"
    )


# ============================================================================
# Per-galaxy g_obs aggregation (mean and median variants, identical to v1)
# ============================================================================

def aggregate_g_obs(rotcurves, galaxy_df, aggregation="mean"):
    """
    Compute per-galaxy g_obs from rotation curve points with r > 2*hR.
    g_obs(r) = V(r)^2 / r
    NaN g_obs occurs when no radius points satisfy r > 2*hR for a galaxy.
    """
    g_obs_series = {}
    for galaxy_name in galaxy_df["galaxy"].values:
        if galaxy_name not in rotcurves:
            g_obs_series[galaxy_name] = np.nan
            continue
        rc = rotcurves[galaxy_name]
        hR = galaxy_df.loc[galaxy_df["galaxy"] == galaxy_name, "Rdisk"].values[0]
        outer = rc[rc["r"] > 2.0 * hR]
        if len(outer) == 0:
            g_obs_series[galaxy_name] = np.nan
            continue
        g_per_r = outer["V"].values ** 2 / outer["r"].values
        if aggregation == "mean":
            g_obs_series[galaxy_name] = float(np.mean(g_per_r))
        elif aggregation == "median":
            g_obs_series[galaxy_name] = float(np.median(g_per_r))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return pd.Series(g_obs_series, name="g_obs")


# ============================================================================
# Partial OLS feature derivation (v2 NEW)
# ============================================================================

def derive_partial_OLS_features(galaxy_df):
    """
    Add partial OLS feature columns to galaxy_df:
      r_h         = R_H_FACTOR * Rdisk        (half-mass radius, R_H_FACTOR = 1.68)
      log_r_h     = log10(r_h)
      M_bar       = Mstar + Mgas              (baryonic mass)
      rho_gal     = M_bar / ((4/3) * pi * r_h^3)
      log_rho_gal = log10(rho_gal)

    USER-FILL: verify R_H_FACTOR (= 1.68) and rho_gal volume convention
               (= 4*pi/3 * r_h^3 sphere) against phase_c3_step3_reference.py.
               If canonical uses different convention, precondition_A0_check
               will FAIL and surface the deviation.
    """
    df = galaxy_df.copy()
    df["r_h"] = R_H_FACTOR * df["Rdisk"]
    df["log_r_h"] = np.log10(df["r_h"])
    df["M_bar"] = df["Mstar"] + df["Mgas"]
    df["rho_gal"] = df["M_bar"] / ((4.0 / 3.0) * np.pi * df["r_h"] ** 3)
    df["log_rho_gal"] = np.log10(df["rho_gal"])
    return df


# ============================================================================
# 3-feature partial OLS (v2 NEW, replaces v1 simple OLS)
# ============================================================================

def compute_partial_OLS(delta_primary, log_rho_gal, log_r_h):
    """
    3-feature partial OLS (canonical / v0.1.4 J4 structure):
      target   = delta_primary = log10(g_obs) - log10(g_C15)
      features = [1, 2*log10(rho_gal), log10(r_h)]
      b_alpha  = beta[1]   (coefficient of 2*log10(rho_gal))

    The factor of 2 in feature column 1 makes b_alpha represent the
    rho_gal^2 scaling exponent (universal coupling slope). log_r_h is
    partialled out via multiple regression.

    Returns:
      b_alpha    : float, partial slope coefficient (beta[1])
      SE_b_alpha : float, standard error of beta[1]
      n          : int, sample size after finite-value filter
    """
    delta_primary = np.asarray(delta_primary, dtype=float)
    log_rho_gal = np.asarray(log_rho_gal, dtype=float)
    log_r_h = np.asarray(log_r_h, dtype=float)
    valid = (np.isfinite(delta_primary)
             & np.isfinite(log_rho_gal)
             & np.isfinite(log_r_h))
    delta_primary = delta_primary[valid]
    log_rho_gal = log_rho_gal[valid]
    log_r_h = log_r_h[valid]
    n = len(delta_primary)
    if n < 4:
        return float("nan"), float("nan"), n

    # Design matrix: [intercept, 2*log10(rho_gal), log10(r_h)]
    X = np.column_stack([
        np.ones(n),
        2.0 * log_rho_gal,
        log_r_h,
    ])
    y = delta_primary

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_alpha = float(beta[1])

    # Standard error of beta[1]: SE = sqrt(sigma^2 * (X'X)^{-1}[1,1])
    y_pred = X @ beta
    resid = y - y_pred
    dof = n - 3  # 3 predictors including intercept
    sigma2 = float(np.sum(resid ** 2) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    SE_b = float(np.sqrt(sigma2 * XtX_inv[1, 1]))

    return b_alpha, SE_b, n


# ============================================================================
# Imputation methods (refinement Q2 / Q3, unchanged from v1)
# ============================================================================

def impute_min_fill(g_obs):
    """A2: NaN g_obs -> min(valid g_obs). Log-finite, no extrapolation."""
    g = np.asarray(g_obs, dtype=float).copy()
    valid = ~np.isnan(g)
    fill_value = float(np.min(g[valid]))
    g_filled = np.where(valid, g, fill_value)
    n_imp = int(np.sum(~valid))
    return g_filled, n_imp, {"fill_value": fill_value}


def impute_mean_fill(g_obs):
    """A3: NaN g_obs -> mean(valid g_obs)."""
    g = np.asarray(g_obs, dtype=float).copy()
    valid = ~np.isnan(g)
    fill_value = float(np.mean(g[valid]))
    g_filled = np.where(valid, g, fill_value)
    n_imp = int(np.sum(~valid))
    return g_filled, n_imp, {"fill_value": fill_value}


def impute_knn(g_obs, features_df, k=KNN_K, feature_cols=KNN_FEATURE_COLS):
    """A4: k-NN imputation via scipy.spatial.KDTree (sklearn-free)."""
    g = np.asarray(g_obs, dtype=float).copy()
    valid = ~np.isnan(g)

    F = features_df[feature_cols].values.astype(float)
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0, ddof=0)
    F_std = np.where(F_std == 0, 1.0, F_std)
    F_scaled = (F - F_mean) / F_std

    F_valid = F_scaled[valid]
    g_valid = g[valid]
    if len(F_valid) < k:
        raise ValueError(
            f"k-NN: insufficient valid samples ({len(F_valid)}) for k={k}"
        )

    tree = KDTree(F_valid)
    F_invalid = F_scaled[~valid]
    if len(F_invalid) == 0:
        return g, 0, {"k": k, "feature_cols": feature_cols, "n_imputed": 0,
                      "mean_distance_to_nn": 0.0}

    distances, indices = tree.query(F_invalid, k=k)
    if k == 1:
        imputed_values = g_valid[indices]
    else:
        imputed_values = np.mean(g_valid[indices], axis=1)

    g_filled = g.copy()
    g_filled[~valid] = imputed_values
    n_imp = int(np.sum(~valid))
    info = {
        "k": k,
        "feature_cols": feature_cols,
        "scaling": "zero-mean unit-variance",
        "tree": "scipy.spatial.KDTree",
        "n_imputed": n_imp,
        "mean_distance_to_nn": float(np.mean(distances)),
    }
    return g_filled, n_imp, info


# ============================================================================
# Method runners (v2: 3-feature partial OLS + canonical filter ordering)
# All runners receive sparc_171 with g_obs aggregated + features derived.
# Each runner applies remaining filters (Q<3, vflat>0, finite delta_primary)
# internally before partial OLS.
# ============================================================================

def _apply_remaining_filters(df):
    """Apply Q<3, vflat>0, finite delta_primary filter chain."""
    valid = (
        (df["Q"] < 3)
        & (df["Vflat"] > 0)
        & np.isfinite(df["delta_primary"])
        & np.isfinite(df["log_rho_gal"])
        & np.isfinite(df["log_r_h"])
    )
    return df[valid].reset_index(drop=True)


def run_A0_baseline(df_171):
    """A0: canonical mean(V^2/r), NaN g_obs filter exclusion (no imputation)."""
    df = df_171.copy()
    df["delta_primary"] = np.log10(df["g_obs_mean"]) - np.log10(df["g_C15"])
    df_v = _apply_remaining_filters(df)
    b, SE, n = compute_partial_OLS(
        df_v["delta_primary"].values,
        df_v["log_rho_gal"].values,
        df_v["log_r_h"].values,
    )
    return {
        "scenario": "A0_baseline",
        "n_sample": n,
        "n_imputed": 0,
        "b_alpha": b,
        "SE_b_alpha": SE,
        "notes": "canonical mean(V^2/r) aggregation, NaN g_obs excluded by "
                 "filter 3 (finite delta_primary); 3-feature partial OLS "
                 "(precondition Q5 beta target b_alpha=0.108442979149252, SE=0.0153)",
    }


def run_A1_median_V2r(df_171):
    """A1: median(V^2/r) aggregation, NaN g_obs filter exclusion."""
    df = df_171.copy()
    df["delta_primary"] = np.log10(df["g_obs_median"]) - np.log10(df["g_C15"])
    df_v = _apply_remaining_filters(df)
    b, SE, n = compute_partial_OLS(
        df_v["delta_primary"].values,
        df_v["log_rho_gal"].values,
        df_v["log_r_h"].values,
    )
    return {
        "scenario": "A1_median_V2r",
        "n_sample": n,
        "n_imputed": 0,
        "b_alpha": b,
        "SE_b_alpha": SE,
        "notes": "median(V^2/r) aggregation alternative to mean, "
                 "NaN g_obs filter 3 exclusion preserved",
    }


def _run_imputed(df_171, scenario_name, impute_fn, impute_label, fill_info_str,
                 features_for_knn=None):
    """
    Common runner for A2/A3/A4: impute at sparc_171 level, then filter, then OLS.
    Reports both pool-level imputed count and in-OLS imputed count.
    """
    df = df_171.copy()
    g_obs = df["g_obs_mean"].values
    was_imputed_mask = np.isnan(g_obs)

    if features_for_knn is not None:
        # k-NN needs features attached
        df["log_Mstar"] = np.log10(df["Mstar"].values)
        df["log_Mgas"] = np.log10(df["Mgas"].values)
        df["log_hR"] = np.log10(df["Rdisk"].values)
        df["log_SBdisk0"] = np.log10(df["SBdisk0"].values)
        g_filled, n_imp_pool, info = impute_fn(g_obs, df, k=KNN_K, feature_cols=KNN_FEATURE_COLS)
    else:
        g_filled, n_imp_pool, info = impute_fn(g_obs)

    df["g_obs_imputed"] = g_filled
    df["was_imputed"] = was_imputed_mask
    df["delta_primary"] = np.log10(df["g_obs_imputed"]) - np.log10(df["g_C15"])

    df_v = _apply_remaining_filters(df)
    n_imp_OLS = int(df_v["was_imputed"].sum())

    b, SE, n = compute_partial_OLS(
        df_v["delta_primary"].values,
        df_v["log_rho_gal"].values,
        df_v["log_r_h"].values,
    )

    notes = (f"NaN g_obs (sparc_171) -> {impute_label}; "
             f"pool-level imputed n={n_imp_pool}, "
             f"in-OLS-sample imputed n={n_imp_OLS}; "
             f"{fill_info_str}")

    return {
        "scenario": scenario_name,
        "n_sample": n,
        "n_imputed": n_imp_OLS,
        "n_imputed_pool": n_imp_pool,  # extra for log; popped before csv
        "b_alpha": b,
        "SE_b_alpha": SE,
        "notes": notes,
    }


def run_A2_min_fill(df_171):
    """A2: NaN g_obs (sparc_171) -> min(valid g_obs)."""
    g_obs = df_171["g_obs_mean"].values
    valid = ~np.isnan(g_obs)
    min_val = float(np.min(g_obs[valid]))
    return _run_imputed(
        df_171, "A2_min_fill", impute_min_fill,
        f"min(valid)={min_val:.6e}",
        "log-finite, no extrapolation (refinement Q2 corrected from zero_fill)",
    )


def run_A3_mean_fill(df_171):
    """A3: NaN g_obs (sparc_171) -> mean(valid g_obs)."""
    g_obs = df_171["g_obs_mean"].values
    valid = ~np.isnan(g_obs)
    mean_val = float(np.mean(g_obs[valid]))
    return _run_imputed(
        df_171, "A3_mean_fill", impute_mean_fill,
        f"mean(valid)={mean_val:.6e}",
        "neutral aggregate baseline imputation",
    )


def run_A4_knn_impute(df_171):
    """A4: k=5 KDTree imputation on log-feature space (refinement Q3)."""
    df = df_171.copy()
    df["log_Mstar"] = np.log10(df["Mstar"].values)
    df["log_Mgas"] = np.log10(df["Mgas"].values)
    df["log_hR"] = np.log10(df["Rdisk"].values)
    df["log_SBdisk0"] = np.log10(df["SBdisk0"].values)

    g_obs = df["g_obs_mean"].values
    was_imputed_mask = np.isnan(g_obs)
    g_filled, n_imp_pool, info = impute_knn(
        g_obs, df, k=KNN_K, feature_cols=KNN_FEATURE_COLS
    )
    df["g_obs_imputed"] = g_filled
    df["was_imputed"] = was_imputed_mask
    df["delta_primary"] = np.log10(df["g_obs_imputed"]) - np.log10(df["g_C15"])

    df_v = _apply_remaining_filters(df)
    n_imp_OLS = int(df_v["was_imputed"].sum())

    b, SE, n = compute_partial_OLS(
        df_v["delta_primary"].values,
        df_v["log_rho_gal"].values,
        df_v["log_r_h"].values,
    )

    notes = (f"k={KNN_K} KDTree imputation on [{','.join(KNN_FEATURE_COLS)}], "
             f"sklearn-free (scipy.spatial.KDTree), "
             f"mean_dist_to_nn={info['mean_distance_to_nn']:.4f}; "
             f"pool-level imputed n={n_imp_pool}, in-OLS-sample imputed n={n_imp_OLS}; "
             f"k-sensitivity falsification path: vary k in {{3,5,10}}")

    return {
        "scenario": "A4_knn_impute",
        "n_sample": n,
        "n_imputed": n_imp_OLS,
        "n_imputed_pool": n_imp_pool,
        "b_alpha": b,
        "SE_b_alpha": SE,
        "notes": notes,
    }


# ============================================================================
# Precondition: A0 two-axis bit-exact + SE check (refinement Q5 beta)
# ============================================================================

def precondition_A0_check(result_A0):
    """
    Two-axis precondition (Q5 beta):
      (a) b_alpha bit-exact match (J4 strict, < B_ALPHA_TOL)
      (b) OLS SE match (J1b reference, +/- SE_TOL)

    Aborts with sys.exit(1) if either axis fails.
    """
    delta_b = abs(result_A0["b_alpha"] - EXPECTED_B_ALPHA)
    delta_SE = abs(result_A0["SE_b_alpha"] - EXPECTED_SE)
    bit_exact_b = delta_b < B_ALPHA_TOL
    SE_match = delta_SE < SE_TOL

    print("=" * 72)
    print("A0 baseline two-axis precondition check (refinement Q5 beta)")
    print("=" * 72)
    print(f"  axis (a) b_alpha bit-exact:")
    print(f"    measured  = {result_A0['b_alpha']:.18g}")
    print(f"    expected  = {EXPECTED_B_ALPHA:.18g}  (J4 / G1, anchor 21 v0.1.4)")
    print(f"    delta     = {delta_b:.6e}  (tol = {B_ALPHA_TOL:.0e})")
    print(f"    status    = {'PASS' if bit_exact_b else 'FAIL'}")
    print(f"  axis (b) OLS SE match:")
    print(f"    measured  = {result_A0['SE_b_alpha']:.6f}")
    print(f"    expected  = {EXPECTED_SE:.6f}  (J1b, anchor 21 v0.1.4)")
    print(f"    delta     = {delta_SE:.6e}  (tol = {SE_TOL:.0e})")
    print(f"    status    = {'PASS' if SE_match else 'FAIL'}")
    overall = bit_exact_b and SE_match
    print(f"  overall    = {'PASS' if overall else 'FAIL'}")
    print("=" * 72)

    if not overall:
        print("\nABORT: precondition FAIL -- A1-A4 results would not be"
              " interpretable\n       as pure method-change effects.")
        print("       v2 candidates for review (vs phase_c3_step3_reference.py):")
        print("         - R_H_FACTOR (= 1.68): canonical r_h convention?")
        print("         - rho_gal volume formula: (4/3)*pi*r_h^3 sphere?")
        print("         - C15_COEF / C15_UPSILON_EXP / A0_KPC: anchor 19 sec 1.5?")
        print("         - design matrix order: [1, 2*log10(rho_gal), log10(r_h)]?")
        print("         - factor of 2 on log10(rho_gal): rho_gal^2 vs rho_gal?")
        print("         - filter 3 application: NaN exclusion via finite delta_primary?")
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="per_filter_sensitivity_extension v2 -- Lesson 94 cross-method"
    )
    parser.add_argument("--ta3",          required=True, help="Path to Table_A3.dat")
    parser.add_argument("--phase1",       required=True, help="Path to phase1_TA1_TA2.csv")
    parser.add_argument("--mrt",          required=True, help="Path to MRT_SPARC_175.txt")
    parser.add_argument("--rotcurve_dir", required=True, help="Directory of per-galaxy rotation curve files")
    parser.add_argument("--output", default="per_filter_sensitivity_extension.csv",
                        help="Output csv path")
    args = parser.parse_args()

    print("=" * 72)
    print("per_filter_sensitivity_extension.py v2")
    print("(3-feature partial OLS + canonical sparc_171 ordering)")
    print("=" * 72)

    print("\nPhase 1: load three-table data ...")
    ta3 = load_TA3(args.ta3)
    phase1 = load_phase1(args.phase1)
    mrt = load_MRT_Q_intact(args.mrt)
    print(f"  TA3 n={len(ta3)}, phase1 n={len(phase1)}, MRT n={len(mrt)}")

    df_merged = mrt.merge(phase1, on="galaxy", how="inner")
    df_merged = df_merged.merge(ta3, on="galaxy", how="inner")
    df_merged["Mgas"]  = 1.33 * df_merged["MHI"]
    df_merged["Mstar"] = df_merged["ud"] * df_merged["L36"]
    print(f"  merged 3-way n={len(df_merged)}")

    print("\nPhase 2: bridge filter (filter 2 only) -> sparc_171 ...")
    n_pre_bridge = len(df_merged)
    df_171 = df_merged[~df_merged["galaxy"].isin(BRIDGE_4)].copy().reset_index(drop=True)
    print(f"  pre-bridge n={n_pre_bridge}")
    print(f"  bridge excluded n={n_pre_bridge - len(df_171)}")
    print(f"  sparc_171 n={len(df_171)}")
    print(f"  (Q<3 / vflat>0 / finite delta_primary applied per-method downstream)")

    print("\nPhase 3: load rotation curves + aggregate g_obs at sparc_171 level ...")
    rotcurves = load_rotation_curves(args.rotcurve_dir, df_171["galaxy"].tolist())
    df_171["g_obs_mean"]   = aggregate_g_obs(rotcurves, df_171, "mean").values
    df_171["g_obs_median"] = aggregate_g_obs(rotcurves, df_171, "median").values

    n_nan_mean = int(df_171["g_obs_mean"].isna().sum())
    n_nan_median = int(df_171["g_obs_median"].isna().sum())
    print(f"  sparc_171 NaN g_obs (mean):   {n_nan_mean}")
    print(f"  sparc_171 NaN g_obs (median): {n_nan_median}")
    print(f"  expected NaN count (per v0.1.4 J3 filter 3): 11")

    print("\nPhase 4: compute g_C15 + partial OLS features ...")
    df_171["g_C15"] = compute_g_C15(df_171).values
    df_171 = derive_partial_OLS_features(df_171)
    print(f"  derived: r_h, log_r_h, M_bar, rho_gal, log_rho_gal")
    print(f"  R_H_FACTOR={R_H_FACTOR}, rho_gal volume = (4/3)*pi*r_h^3")

    print("\nPhase 5: method runs (5 scenarios, 3-feature partial OLS) ...")

    result_A0 = run_A0_baseline(df_171)
    print(f"  A0 done: b_alpha={result_A0['b_alpha']:.15f}, "
          f"SE={result_A0['SE_b_alpha']:.6f}, n={result_A0['n_sample']}")

    # Precondition check (Q5 beta) -- aborts on FAIL
    precondition_A0_check(result_A0)

    result_A1 = run_A1_median_V2r(df_171)
    print(f"  A1 done: b_alpha={result_A1['b_alpha']:.15f}, "
          f"SE={result_A1['SE_b_alpha']:.6f}, n={result_A1['n_sample']}")

    result_A2 = run_A2_min_fill(df_171)
    print(f"  A2 done: b_alpha={result_A2['b_alpha']:.15f}, "
          f"SE={result_A2['SE_b_alpha']:.6f}, n={result_A2['n_sample']}, "
          f"imputed_in_OLS={result_A2['n_imputed']} "
          f"(pool={result_A2.get('n_imputed_pool', 'n/a')})")

    result_A3 = run_A3_mean_fill(df_171)
    print(f"  A3 done: b_alpha={result_A3['b_alpha']:.15f}, "
          f"SE={result_A3['SE_b_alpha']:.6f}, n={result_A3['n_sample']}, "
          f"imputed_in_OLS={result_A3['n_imputed']} "
          f"(pool={result_A3.get('n_imputed_pool', 'n/a')})")

    result_A4 = run_A4_knn_impute(df_171)
    print(f"  A4 done: b_alpha={result_A4['b_alpha']:.15f}, "
          f"SE={result_A4['SE_b_alpha']:.6f}, n={result_A4['n_sample']}, "
          f"imputed_in_OLS={result_A4['n_imputed']} "
          f"(pool={result_A4.get('n_imputed_pool', 'n/a')})")

    print("\nPhase 6: aggregate to csv ...")
    rows = [result_A0, result_A1, result_A2, result_A3, result_A4]
    baseline_b = result_A0["b_alpha"]
    for r in rows:
        r["delta_vs_baseline_abs"] = r["b_alpha"] - baseline_b
        r["delta_vs_baseline_rel"] = (r["b_alpha"] - baseline_b) / baseline_b
        r.pop("n_imputed_pool", None)  # csv strips this; already in notes

    df_out = pd.DataFrame(rows, columns=[
        "scenario", "n_sample", "n_imputed",
        "b_alpha", "SE_b_alpha",
        "delta_vs_baseline_abs", "delta_vs_baseline_rel",
        "notes"
    ])
    df_out.to_csv(args.output, index=False, encoding="utf-8")

    print("\n" + "=" * 72)
    print("Summary table (per_filter_sensitivity_extension v2)")
    print("=" * 72)
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(df_out.to_string(index=False))
    print(f"\nWritten: {args.output}")
    print("\nLesson 94 evidence base now at rule #26 multi-route minimum compliance.")


if __name__ == "__main__":
    main()
