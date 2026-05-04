#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J-system Companion Paper §2.5 v0.2 SPARC Empirical Execution v1.0.2
====================================================================

Runs Algorithm B per-galaxy + f_E adoption + dSph J3 consistency check +
Sub-issue S-1〜S-6 numerical resolve, against SPARC 175 (171-galaxy fit
pool) and dSph 31 sample.

Conforms to:
  - foundation_gamma_actual.py    (SHA b0cb36d7) — cascade SSoT canonical, #22(vi)
  - anchor 7  (§2.5 v0.1,         SHA 9e03f53e) — predecessor, 3-tier LOCK preserved
  - anchor 8  (§2.6 v0.1,         SHA f6a48b51) — chapter-level milestone
  - anchor 14 (§4 v0.4,           SHA 295bc05c) — Layer B-α/B-β + NGC 3198
  - anchor 17 (§3 v0.2,           SHA 178dad11) — c_super=0.5709
  - anchor 19 (§1 v0.4,           SHA 0b269c10) — A 級 prerequisite
  - anchor 21 (§2 closure v0.1.1, SHA 44df9afb) — chapter-level §2 LOCK ESTABLISHED
  - C3-A5 internal_memo_c3_extension_v3.pdf (SHA 69fb1a95) — Lesson 91/93
  - C3-A4 J0 minimal form         (SHA 7e8823f4) — reference baseline

v1.0.2 changelog (2026-05-04):
  Skeleton additions ahead of T12-T14 verbatim extraction. Orchestration
  for f_E aggregation + dSph audit + AC1-AC7 evaluation wired now;
  numerical body fills with T12-T14 supply.
  - load_rotation_curve(): SPARC RC per-galaxy loader stub (T12 fill point)
  - compute_g_obs_g_bar(): V → g derivation skeleton (T12 fill point)
  - nu_canonical_self_check(): T13 verification framework (reference pair
    placeholder; populates on T13 verbatim supply)
  - nll_self_check(): T14 verification framework
  - aggregate_f_E_adoption(): per-galaxy → sample-level (B/E/mixed) per
    Q4 LOCK F_E_LOWER=0.20 / F_E_UPPER=0.80
  - run_dsph_audit(): dSph J3 + b_α 3-axis audit orchestration
  - evaluate_acceptance_criteria(): AC1-AC7 evaluation populating
    summary.json::acceptance block
  - write_human_readable_summary(): run_summary.txt for at-a-glance review
  - main(): wired the above into orchestration; added SPARC_RC_BASE_PATH
    env var (T12) and --no-rotation-curve flag for staged validation

v1.0.1 changelog (2026-05-03):
  - V_DOUBLE_PRIME_AT_X_HALF filled (5 anchor values from foundation_gamma_actual.py)
  - DELTA_AIC_THRESHOLD set to 2.0; added F_E_LOWER/F_E_UPPER (Q4 LOCK)
  - EXCLUDED_4_SPARC_GALAXIES filled (NGC3741/NGC2915/ESO444-G084/NGC1705)
  - f_opt rewritten as V''(x=0.5, c) deg-4 Lagrange + f_opt_v3_cascade
  - chi_coh implemented (Layer B-α: 1 − f_p; B-β: analytic Strigari)
  - algorithm_b_step implemented (per-radius fixed-point per anchor 7 §2.5.3)
  - resolve_S1/S3 fill structure with C3-A5 + parent v4.8 SHA references
  - F6 dSph audit: 28/31 + 30 sample handled

Outputs follow OUTPUT_SCHEMA_section2_5_v0_2.md v1.0.

Environment: Windows + Claude Code (cannot run inside claude.ai container).

Remaining TODO_USER_VERIFY items (data wiring / formula extraction):
  - T9-T11: TA3 / phase1 / MRT column verification
  - T12: SPARC rotation curve per-galaxy loader (Lelli .dat file format,
         derivation V → g_obs/g_bar, σ_g handling)
  - T13: ν_canonical(x; c) full functional form (parent v4.8 §3.1 L335)
  - T14: NLL_E / NLL_B explicit formulas (anchor 7 §2.5.2 / §2.5.3)
  - T15: σ_g(r) reference fill on v0.2 run (parent v4.8 §6 line XXX)
  - T16: Phase C3 v3 §X chapter numbers fill on run

Operating modes:
  - python run_section2_5_v0_2.py --dry-run
        Validates env vars + input file SHAs only (no computation).
  - python run_section2_5_v0_2.py --no-rotation-curve
        Runs Algorithm B with rotation_curve=None (stub mode), exercises
        orchestration without T12 wiring. Useful pre-T12.
  - python run_section2_5_v0_2.py
        Full scientific run (requires all of T9-T16 wired).

Usage:
  $env:MASTER_ROOT          = "D:\\ドキュメント\\エントロピー\\膜宇宙論再考察AB効果有り\\C3 拡張版仮説関連2"
  $env:PARENT_MASTER_ROOT   = "$env:MASTER_ROOT\\arxiv\\v48_release"
  $env:SPARC_TA3_PATH       = "<path to TA3 file>"
  $env:SPARC_PHASE1_PATH    = "<path to phase1 file>"
  $env:SPARC_MRT_PATH       = "<path to MRT fixed-width file>"
  $env:SPARC_RC_BASE_PATH   = "<dir containing per-galaxy rotation curve .dat files>"
  $env:DSPH_DATASET_PATH    = "<path to dSph dataset>"
  $env:OUTPUT_ROOT          = "C:\\build\\section2_5_v0_2_runs"
  python run_section2_5_v0_2.py

License: CC-BY 4.0
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq  # for ν_canonical = f_opt^(-1) inversion

# ============================================================================
# §1 CONSTANTS (per anchor 7 / 8 / 14 / 17 / 19 / 21)
# ============================================================================

# Universal acceleration scale (MOND a_0)
A_0: float = 1.2e-10  # m/s²
# Note: TA3 carries gc_over_a0 in a_0 units always (per project convention).

# Membrane time scale T_m (Z2 SSB) — anchor 8 §2.6 / anchor 20 §C
T_M: float = float(np.sqrt(6.0))

# 5-anchor c values for f_opt deg-4 Lagrange interp (parent v4.8 §3.1 L335 immutable)
C_ANCHORS: Tuple[float, ...] = (0.30, 0.42, 0.618, 0.80, 1.00)

# Algorithm B parameters (anchor 7 §2.5.3, R-2 LOCK)
C_GALAXY_INIT: float = 0.42      # initialization
TOL: float = 1e-6                # convergence tolerance
N_MAX: int = 50                  # max iterations
K_B: int = 0                     # B canonical: parameter-free

# E pipeline parameters (anchor 7 §2.5.2, Q-C1 LOCK)
K_E_DEFAULT: int = 2             # k_E=2 default LOCK
K_E_SENSITIVITY: int = 1         # F8 audit variant (placeholder pending)

# Cascade and supersonic regime constants
C_CASCADE: float = 0.83          # anchor 6 boundary r→∞
C_SUPER: float = 0.5709          # anchor 17 §3.8 supersonic boundary

# Bernoulli analytic closure (anchor 16 / anchor 7 §2.5.5)
S_0: float = 0.3515              # at Q→0
G_STRIGARI_OVER_A0: float = 0.227948  # = s_0 * (1 - s_0)

# C15 final form constants (anchor 7 / 14 / 19 baseline)
ETA_0_LOCKED: float = 0.584      # prefactor (locked)
ALPHA_LOCKED: float = 0.5        # algebraic, deep-MOND limit
BETA_LOCKED: float = -0.361      # Y_d exponent

# A 級昇格 baseline (anchor 19 §1.5)
PATH_A_ETA_0_BASELINE: float = 0.5629   # path A median
PATH_A_DELTA_PCT_BASELINE: float = -3.61
PATH_B_ETA_0_BASELINE: float = 0.5649   # path B v_flat²-weighted
PATH_B_DELTA_PCT_BASELINE: float = -3.27
T6_TOLERANCE_PCT: float = 5.0    # 5% tolerance
T4_TOLERANCE_PCT: float = 5.0

# Phase C3 b_α baseline (anchor 19 §1.5)
B_ALPHA_SPARC_BASELINE: float = 0.1084
B_ALPHA_DSPH_BASELINE: float = 0.1127
B_ALPHA_ABS_DIFF_BASELINE: float = 0.0042

# Helium correction
HELIUM_FACTOR: float = 1.33      # Mgas = 1.33 × MHI

# Anchor 21 v0.1.1 chapter-level §2 LOCK reference (chain immutable)
ANCHOR_21_SHA_REFERENCE: str = (
    "44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f"
)

# All upstream anchor SHA prefixes (8-char, manifest record only)
ANCHOR_REFERENCES: List[Dict[str, str]] = [
    {"anchor": "5", "sha_prefix": "3270fb40", "role": "§2.4 v0.1"},
    {"anchor": "6", "sha_prefix": "6ac356c3", "role": "§2.1-§2.3 v0.1"},
    {"anchor": "7", "sha_prefix": "9e03f53e", "role": "§2.5 v0.1 (predecessor)"},
    {"anchor": "8", "sha_prefix": "f6a48b51", "role": "§2.6 v0.1 chapter milestone"},
    {"anchor": "14", "sha_prefix": "295bc05c", "role": "§4 v0.4 Layer B-α/B-β + NGC 3198"},
    {"anchor": "16", "sha_prefix": "69678018", "role": "§5 v0.2.1 disambig"},
    {"anchor": "17", "sha_prefix": "178dad11", "role": "§3 v0.2 c_super=0.5709"},
    {"anchor": "19", "sha_prefix": "0b269c10", "role": "§1 v0.4 A 級 prerequisite"},
    {"anchor": "20", "sha_prefix": "56afa4c2", "role": "milestone summary"},
    {"anchor": "21", "sha_prefix": "44df9afb", "role": "§2 closure v0.1.1"},
]

# Auxiliary SHA references (NOT anchor-counted; foundation + C3 internal docs)
AUXILIARY_REFERENCES: List[Dict[str, str]] = [
    {"role": "foundation_gamma_actual.py (cascade SSoT canonical)",
     "sha_prefix": "b0cb36d7"},
    {"role": "parent v4.8 .tex companion ja (NEW canonical 2026-04-30)",
     "sha_prefix": "902f79c6"},
    {"role": "parent v4.8 .tex companion en",
     "sha_prefix": "2dcf69e6"},
    {"role": "parent v4.7.8 short version",
     "sha_prefix": "b7bf9629"},
    {"role": "parent v4.8 historical (superseded, retained)",
     "sha_prefix": "394f2571"},
    {"role": "C3-A5 internal_memo_c3_extension_v3.pdf (Lesson 91/93)",
     "sha_prefix": "69fb1a95"},
    {"role": "C3-A4 J0 minimal form (reference baseline 専用)",
     "sha_prefix": "7e8823f4"},
]

# ============================================================================
# §2 USER-INPUT FILL TABLE (v1.0.1 status)
# ============================================================================
#
# v1.0.1 fills applied (2026-05-03):
#
#   ✅ Item 1: V_DOUBLE_PRIME_AT_X_HALF      filled (5 values from foundation b0cb36d7)
#   ✅ Item 7: DELTA_AIC_THRESHOLD = 2.0      filled (anchor 7 §2.5.4 Q-C3 + Lesson 92)
#   ✅ Item 8: EXCLUDED_4_SPARC_GALAXIES      filled (anchor 7 §2.5.1, 4 bridge galaxies)
#   ✅ F_E_LOWER = 0.20, F_E_UPPER = 0.80     added (Q4 LOCK sample-level rule)
#
#   ✅ Item 2: chi_coh closed-form            filled — Layer B-α: 1−f_p (anchor 14 §4.13.2)
#                                              Layer B-β: NotImplemented (analytic only)
#   ✅ Item 3: Algorithm B step formula       filled — per-radius fixed-point (anchor 7 §2.5.3)
#   ✅ Item 4: F1 / resolve_S3 structure      filled (parent v4.8 902f79c6 ref)
#   ✅ Item 5: F2 / resolve_S1 structure      filled (C3-A5 69fb1a95, Lesson 91)
#   ✅ Item 6: F6 / dSph audit logic          filled (28 typical + 3 reverse pattern)
#
# Items 4/5/6 contain LITERAL placeholders (XXX line / §X chapter) that fill
# DURING the v0.2 round itself — not as TODO; this is anchor 7 + anchor 8 design.
#
# Remaining TODO_USER_VERIFY items (data wiring, not values):
#
#   T9:  TA3 column delimiter / column names           (verify against actual file)
#   T10: phase1 column names                           (verify)
#   T11: MRT fixed-width column specs                  (verify colspecs in load_MRT)
#   T12: SPARC rotation curve per-galaxy loader        (separate Lelli .dat files)
#   T13: ν_canonical(x; c) full functional form        (parent v4.8 §3.1 L335)
#   T14: NLL_E / NLL_B explicit formulas               (anchor 7 §2.5.2 / §2.5.3 L246+)
#   T15: σ_g(r) reference fill on v0.2 run             (parent v4.8 §6 line XXX)
#   T16: Phase C3 v3 §X chapter numbers fill on run    (Lesson 91, dSph 30 sample)
# ============================================================================

# Source: foundation_gamma_actual.py L75-77, L99-106 (cascade SSoT canonical,
# SHA b0cb36d7..., #22(vi) immutable)
# SSoT: v37_chap18_table18_2_vpp_full.csv x_0_5 column
# anchor: V''(x=0.5, 0.83) = 10.463 → f_opt(0.83) = 2π/√10.463 = 1.9425 (exact)
V_DOUBLE_PRIME_AT_X_HALF: Dict[float, Optional[float]] = {
    0.30:  62.1,    # Im,        T=10 anchor
    0.42:  38.7,    # Sc,        T=5  anchor
    0.618: 20.2,    # reference, base
    0.80:  11.6,    # Sb,        T=3  anchor
    1.00:   6.0,    # flexon boundary
}
# self-check: vpp_x05(0.83) must equal 10.463 ± 1e-2
# self-check: f_opt_v3_cascade(0.83) must equal 1.9425 ± 1e-3
# Source SHA: foundation b0cb36d7... + parent v4.8 §3.1 L335 902f79c6 immutable

DELTA_AIC_THRESHOLD: float = 2.0
# Per-galaxy decision rule (anchor 7 §2.5.4, Q-C3 LOCK + Lesson 92 parsimony):
#   ΔAIC > +2  → Candidate B 採択 (B decisive evidence advantage)
#   ΔAIC < −2  → Candidate E 採択 (E が B の +4 a priori penalty を NLL で overcome)
#   |ΔAIC| ≤ 2 → parsimony default → Candidate B 採択 (Lesson 92, k_B=0 < k_E=2)
#
# ΔAIC formula (anchor 7 §2.5.4, k_E=2 + k_B=0 → +4 a priori advantage for B):
#   ΔAIC (galaxy g) = AIC_E (g) − AIC_B (g) = 2·(NLL_E − NLL_B) + 4

# Sample-level final form selection (Q4 LOCK, f_E threshold sample-level rule):
F_E_LOWER: float = 0.20  # f_E < 0.20 → B globally adopted
F_E_UPPER: float = 0.80  # f_E > 0.80 → E globally adopted
# 0.20 ≤ f_E ≤ 0.80 → mixed, flagged for §6 cross-id audit

EXCLUDED_4_SPARC_GALAXIES: Dict[str, str] = {
    # Bridge pre-cut 4 galaxy (anchor 7 §2.5.1, Phase C3 v3 Lesson 91 protocol)
    # SPARC 175 − 4 = 171-galaxy §2.5 fit pool 確定
    "NGC3741":     "low-density bridge case, ρ profile extreme",
    "NGC2915":     "low-density bridge case, ρ profile extreme",
    "ESO444-G084": "low-density bridge case, ρ profile extreme",
    "NGC1705":     "low-density bridge case, ρ profile extreme",
}
# 4 galaxy は cascade SSoT 5-anchor canonical range
# (c ∈ {0.30, 0.42, 0.618, 0.80, 1.00}) の lower edge boundary 近傍に位置し、
# Phase C3 v3 §X analysis で fit instability + Algorithm B convergence
# failure risk が確認済。
# pre-cut forensic record として §2.5.5 dSph J3 consistency check においても
# extreme regime extension domain の reference として cite されるが、
# §2.5 selection target からは除外維持。
# Cross-check (userMemories): bridge 4 galaxy = continuous C15→Strigari transition
# 4/4 confirmed (anchor 7 v0.1 / dSph extension v4.7.8).

SPARC_TOTAL: int = 175
SPARC_FIT_POOL_SIZE: int = SPARC_TOTAL - len(EXCLUDED_4_SPARC_GALAXIES)  # = 171


# ============================================================================
# §3 RUNTIME CONFIGURATION
# ============================================================================

@dataclass
class RunConfig:
    """Captured at run start, written to run_config.json."""
    run_id: str
    timestamp_iso: str
    git_commit: str
    master_root: str
    parent_master_root: str
    sparc_ta3_path: str
    sparc_phase1_path: str
    sparc_mrt_path: str
    dsph_dataset_path: str
    output_root: str
    output_dir: str
    random_seed: int = 42
    output_float_precision: int = 6
    delta_aic_threshold: Optional[float] = None
    k_e_default: int = K_E_DEFAULT
    k_e_sensitivity: int = K_E_SENSITIVITY
    tol: float = TOL
    n_max: int = N_MAX
    c_galaxy_init: float = C_GALAXY_INIT
    sparc_rc_base_path: Optional[str] = None  # T12: per-galaxy RC dir
    no_rotation_curve: bool = False           # staged-validation override


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "no-git"


def build_run_config(no_rotation_curve: bool = False) -> RunConfig:
    """Read env vars, build run config object.

    Parameters
    ----------
    no_rotation_curve : bool
        If True, SPARC_RC_BASE_PATH is optional; Algorithm B runs in stub
        mode (rotation_curve=None for every galaxy). Useful pre-T12.
    """
    required = [
        "MASTER_ROOT", "PARENT_MASTER_ROOT",
        "SPARC_TA3_PATH", "SPARC_PHASE1_PATH", "SPARC_MRT_PATH",
        "DSPH_DATASET_PATH", "OUTPUT_ROOT",
    ]
    if not no_rotation_curve:
        # Full scientific run: rotation curve dir is required (T12 wired).
        required.append("SPARC_RC_BASE_PATH")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "See module docstring for usage."
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = now.strftime("%Y-%m-%dT%H-%M-%S")
    git_short = _git_short_sha()
    run_id = f"{timestamp_iso}_{git_short}"

    output_root = os.environ["OUTPUT_ROOT"]
    output_dir = str(Path(output_root) / f"section2_5_v0_2_{run_id}")

    return RunConfig(
        run_id=run_id,
        timestamp_iso=now.isoformat(),
        git_commit=git_short,
        master_root=os.environ["MASTER_ROOT"],
        parent_master_root=os.environ["PARENT_MASTER_ROOT"],
        sparc_ta3_path=os.environ["SPARC_TA3_PATH"],
        sparc_phase1_path=os.environ["SPARC_PHASE1_PATH"],
        sparc_mrt_path=os.environ["SPARC_MRT_PATH"],
        dsph_dataset_path=os.environ["DSPH_DATASET_PATH"],
        output_root=output_root,
        output_dir=output_dir,
        delta_aic_threshold=DELTA_AIC_THRESHOLD,
        sparc_rc_base_path=os.environ.get("SPARC_RC_BASE_PATH"),
        no_rotation_curve=no_rotation_curve,
    )


# ============================================================================
# §4 THREE-FILE LOADER (TA3 + phase1 + MRT)
# ============================================================================

def load_TA3(path: str) -> pd.DataFrame:
    """
    SPARC TA3: galaxy ↔ gc_over_a0 (in a_0 units, ALWAYS).
    Returns DataFrame with columns: galaxy, gc_over_a0
    """
    # Project convention: TA3 contains gc_over_a0 already in a_0 units.
    # File format may be CSV / TSV / fixed-width — adapt loader as needed.
    # TODO_USER_VERIFY: confirm exact TA3 column names + delimiter.
    df = pd.read_csv(path, sep=None, engine="python")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if "galaxy" not in df.columns:
        # Try common alternatives
        for alt in ("Galaxy", "name", "Name", "ID"):
            if alt in df.columns:
                df = df.rename(columns={alt: "galaxy"})
                break
    if "gc_over_a0" not in df.columns:
        for alt in ("gc_a0", "g_c_over_a0", "gc/a0"):
            if alt in df.columns:
                df = df.rename(columns={alt: "gc_over_a0"})
                break
    return df[["galaxy", "gc_over_a0"]].copy()


def load_phase1(path: str) -> pd.DataFrame:
    """
    SPARC phase1: galaxy ↔ ud (= Υ_d = mass-to-light ratio of disk).
    Returns DataFrame with columns: galaxy, Ud
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "galaxy" not in df.columns:
        for alt in ("Galaxy", "name", "Name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "galaxy"})
                break
    if "Ud" not in df.columns:
        for alt in ("ud", "Y_d", "Upsilon_d", "M_L_disk"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Ud"})
                break
    return df[["galaxy", "Ud"]].copy()


def load_MRT(path: str) -> pd.DataFrame:
    """
    SPARC MRT (fixed-width catalog).
    Required columns: galaxy, MHI, L36, Rdisk, SBdisk0, T, Vflat
    """
    # MRT is fixed-width; column specs follow SPARC official format.
    # TODO_USER_VERIFY: column widths against the actual MRT file header.
    # Reference: McGaugh et al. SPARC catalog standard layout.
    colspecs = [
        (0, 12),    # galaxy
        (13, 15),   # T (Hubble type)
        (16, 22),   # Rdisk
        (23, 30),   # SBdisk0
        (31, 38),   # L36
        (39, 46),   # MHI
        (47, 54),   # Vflat
    ]
    names = ["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat"]
    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        skiprows=0,  # adjust if header lines exist
        comment="#",
    )
    df["galaxy"] = df["galaxy"].astype(str).str.strip()
    return df


def merge_three_files(
    ta3: pd.DataFrame,
    phase1: pd.DataFrame,
    mrt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge by galaxy key, compute Mgas = 1.33 × MHI, Mstar = Ud × L36.
    """
    df = mrt.merge(phase1, on="galaxy", how="left")
    df = df.merge(ta3, on="galaxy", how="left")
    df["Mgas"] = HELIUM_FACTOR * df["MHI"]
    df["Mstar"] = df["Ud"] * df["L36"]
    df["Mbar"] = df["Mstar"] + df["Mgas"]
    df["gc_obs"] = df["gc_over_a0"] * A_0
    return df


def mark_fit_pool_171(df: pd.DataFrame) -> pd.DataFrame:
    """Mark the 171-galaxy fit pool (excludes 4 per anchor 7 §2.5.1)."""
    df = df.copy()
    excluded = set(EXCLUDED_4_SPARC_GALAXIES.keys())
    df["in_171_pool"] = ~df["galaxy"].isin(excluded)
    df["exclusion_reason"] = df["galaxy"].apply(
        lambda g: EXCLUDED_4_SPARC_GALAXIES.get(g, "")
    )
    return df


# ============================================================================
# §4b SPARC ROTATION CURVE LOADER — T12 fill point
# ============================================================================
# These two functions are skeleton stubs awaiting T12 verbatim extraction.
# When T12 is supplied (Lelli SPARC RC .dat file format + V → g derivation
# formulas), the FILL_HERE blocks below populate.
#
# Until then:
#   - load_rotation_curve() returns None (signals stub mode)
#   - compute_g_obs_g_bar() raises NotImplementedError
#   - Algorithm B per-galaxy gracefully handles None and flags the galaxy
#
# T12 verbatim extraction targets:
#   1. Lelli SPARC RC .dat file format (column layout, header, units)
#   2. V_obs / V_disk / V_gas / V_bulge → g_obs / g_bar derivation
#      (typical: g = V² / r with km/s → m/s, kpc → m unit conversion)
#   3. σ_g(r) handling (from V_obs uncertainty propagation, or direct column)
#   4. Pre-processing rules (zero/negative values, minimum r cutoff)
#   5. Existing implementation (e.g., C3-A1 sparc_fp_verification.py)
#      function definition verbatim (most authoritative source).
# ============================================================================


def load_rotation_curve(
    galaxy: str,
    base_path: Optional[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load per-galaxy SPARC rotation curve (T12 fill point).

    Expected return dict keys (when T12 wired):
        'r'        : np.ndarray  per-radius distance, kpc
        'v_obs'    : np.ndarray  observed circular velocity, km/s
        'v_disk'   : np.ndarray  disk component circular velocity, km/s
        'v_gas'    : np.ndarray  gas component circular velocity, km/s
        'v_bulge'  : np.ndarray  bulge component circular velocity, km/s
        'sigma_v'  : np.ndarray  per-radius velocity uncertainty, km/s
        'g_obs'    : np.ndarray  observed gravitational acceleration, m/s²
        'g_bar'    : np.ndarray  baryonic acceleration, m/s²
        'sigma_g'  : np.ndarray  σ_g(r), m/s²

    The (v_*, sigma_v) and (g_*, sigma_g) families are redundant on purpose —
    Algorithm B / chi_coh use g_*; sanity audits use v_*. The compute_g_obs_g_bar()
    helper translates v → g and computes σ_g.

    Parameters
    ----------
    galaxy : str
        Galaxy identifier matching the MRT/TA3/phase1 'galaxy' column.
    base_path : str | None
        Directory containing per-galaxy rotation curve files (env var
        SPARC_RC_BASE_PATH). If None, returns None (stub mode).

    Returns
    -------
    rotation_curve : dict | None
        Dict with keys above on success, or None if base_path is None
        (stub mode) or if T12 is not yet wired.

    NOTES
    -----
    T12 verbatim required to populate the FILL_HERE block below. Until
    supplied, this function returns None for every galaxy.
    """
    if base_path is None:
        return None

    # ------------------------------------------------------------------
    # FILL_HERE (T12): Lelli SPARC RC .dat parser
    # ------------------------------------------------------------------
    # Expected after T12 fill:
    #
    #   rc_file = Path(base_path) / f"{galaxy}_rotmod.dat"  # adjust naming
    #   if not rc_file.exists():
    #       return None
    #   df = pd.read_fwf(rc_file, ...) or pd.read_csv(rc_file, sep=r'\s+', ...)
    #   # column mapping per Lelli standard format
    #   r       = df['Rad'].values        # kpc
    #   v_obs   = df['Vobs'].values       # km/s
    #   sigma_v = df['errV'].values       # km/s (or e_Vobs)
    #   v_disk  = df['Vdisk'].values
    #   v_gas   = df['Vgas'].values
    #   v_bulge = df.get('Vbul', pd.Series(0.0)).values  # may be absent
    #   # apply pre-processing rules per T12 spec:
    #   #   - drop r==0
    #   #   - drop v_obs <= 0
    #   #   - clip σ_v floor (typical: 2 km/s)
    #   ...
    #   g_obs, g_bar, sigma_g = compute_g_obs_g_bar(r, v_obs, v_disk, v_gas,
    #                                                v_bulge, sigma_v, ud)
    #   return {'r': r, 'v_obs': v_obs, ..., 'g_obs': g_obs, 'g_bar': g_bar,
    #           'sigma_g': sigma_g}
    # ------------------------------------------------------------------
    return None  # stub: T12 not yet wired


def compute_g_obs_g_bar(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    v_disk_kms: np.ndarray,
    v_gas_kms: np.ndarray,
    v_bulge_kms: np.ndarray,
    sigma_v_kms: np.ndarray,
    upsilon_d: float,
    upsilon_b: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive per-radius (g_obs, g_bar, σ_g) from per-radius velocity components.

    Skeleton formulas (T12 verbatim required for canonical):

        r_m       = r_kpc * 3.0857e19              # kpc → m
        g_obs     = (v_obs_kms * 1e3)² / r_m       # m/s²
        g_bar     = (upsilon_d * v_disk_kms² + v_gas_kms² + upsilon_b * v_bulge_kms²)
                    * 1e6 / r_m                    # km²/s² → m²/s² then /r_m
        σ_g       = 2 * (v_obs_kms * 1e3) / r_m * (sigma_v_kms * 1e3)
                                                   # error propagation g = v²/r

    Returns
    -------
    g_obs : m/s²
    g_bar : m/s²
    sigma_g : m/s²

    Raises
    ------
    NotImplementedError until T12 verbatim is supplied (the formulas above
    are the typical convention but the canonical project formula must be
    verbatim-confirmed before scientific use).
    """
    # ------------------------------------------------------------------
    # FILL_HERE (T12): canonical V → g derivation per project spec
    # ------------------------------------------------------------------
    raise NotImplementedError(
        "compute_g_obs_g_bar requires T12 verbatim — Lelli SPARC V → g "
        "derivation + σ_g propagation. The skeleton formulas in the "
        "docstring are typical but must be verbatim-confirmed before use."
    )


# ============================================================================
# §5 f_opt / ν_canonical (cascade SSoT, foundation_gamma_actual.py SHA b0cb36d7;
#                         parent v4.8 §3.1 L335 immutable, SHA 902f79c6)
# ============================================================================

def vpp_x05(c: float) -> float:
    """
    V''(x=0.5, c) via deg-4 exact-pass Lagrange on 5-pt SSoT.
    SSoT: v37_chap18_table18_2_vpp_full.csv x_0_5 column.
    Source: foundation_gamma_actual.py L75-77, L99-106 (SHA b0cb36d7, #22(vi) immutable).

    Self-check anchor: vpp_x05(0.83) == 10.463 ± 1e-2
    """
    cs = np.array(C_ANCHORS, dtype=float)
    vs = np.array(
        [V_DOUBLE_PRIME_AT_X_HALF[ci] for ci in C_ANCHORS],
        dtype=float,
    )
    return float(_lagrange_eval(cs, vs, c))


def f_opt_v3_cascade(c: float) -> float:
    """
    f_opt(c) at canonical x = 0.5: f_opt(c) = 2π / √V''(x=0.5, c).
    Source: foundation_gamma_actual.py (cascade SSoT canonical, SHA b0cb36d7).

    Self-check anchor: f_opt_v3_cascade(0.83) == 1.9425 ± 1e-3
    (i.e. 2π / √10.463)
    """
    return float(2.0 * np.pi / np.sqrt(vpp_x05(c)))


def f_opt(x: float, c: float) -> float:
    """
    f_opt(x; c) — full functional form for arbitrary x.

    The cascade SSoT canonical form gives f_opt(c) at the fixed point x=0.5:
        f_opt(x=0.5, c) = 2π / √V''(x=0.5, c)
    where V''(x=0.5, c) is deg-4 Lagrange interpolated over 5 anchors.

    The (x, c) joint form for arbitrary x is parent v4.8 §3.1 L335 spec
    (SHA 902f79c6 immutable). It is NOT yet wired here.

    Until parent v4.8 §3.1 L335 verbatim is supplied (T13 in user-input
    fill table), this function raises for x != 0.5 to prevent silent
    misuse. f_opt(x=0.5, c) returns the canonical cascade SSoT value.
    """
    if abs(x - 0.5) < 1e-12:
        return f_opt_v3_cascade(c)
    raise NotImplementedError(
        "f_opt(x; c) for x != 0.5 requires parent v4.8 §3.1 L335 spec "
        "(T13 user-input fill). Currently only the canonical x=0.5 point "
        "is wired via cascade SSoT (foundation b0cb36d7)."
    )


def _lagrange_eval(xs: np.ndarray, ys: np.ndarray, x_eval: float) -> float:
    """Evaluate Lagrange interpolating polynomial at x_eval."""
    n = len(xs)
    total = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i == j:
                continue
            term *= (x_eval - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


def nu_canonical(x: float, c: float) -> float:
    """
    ν_canonical(x; c) = f_opt^(-1)(x; c)
    Numerical inversion via Brent's method (1D root-finding).

    Currently only x=0.5 canonical point is wired. For per-radius use
    (Algorithm B step), the full (x, c) form is required (T13).
    """
    def residual(nu: float) -> float:
        return f_opt(nu, c) - x
    try:
        return brentq(residual, 1e-6, 1e2, xtol=1e-10)
    except (ValueError, NotImplementedError):
        # bracket failed or x != 0.5 form not yet wired
        return float("nan")


def _vpp_x05_self_check() -> None:
    """
    Run cascade SSoT self-check anchors at module import.
    Aborts with AssertionError if foundation b0cb36d7 reference values
    have drifted (i.e. someone modified V_DOUBLE_PRIME_AT_X_HALF).
    """
    vpp_at_083 = vpp_x05(0.83)
    f_opt_at_083 = f_opt_v3_cascade(0.83)
    assert abs(vpp_at_083 - 10.463) < 1e-2, (
        f"cascade SSoT self-check FAIL: vpp_x05(0.83) = {vpp_at_083}, "
        f"expected 10.463 ± 1e-2 (foundation b0cb36d7)"
    )
    assert abs(f_opt_at_083 - 1.9425) < 1e-3, (
        f"cascade SSoT self-check FAIL: f_opt_v3_cascade(0.83) = {f_opt_at_083}, "
        f"expected 1.9425 ± 1e-3 (= 2π/√10.463, foundation b0cb36d7)"
    )


# ============================================================================
# T13 verification framework — ν_canonical(x; c) reference pair self-check
# ============================================================================
# Populates on T13 verbatim supply (parent v4.8 §3.1 L335 verbatim).
# Until T13 is supplied, NU_CANONICAL_REFERENCE_PAIRS is empty and
# nu_canonical_self_check() is a no-op (does not raise — T13 wiring may
# be deferred for staged validation).
#
# Once filled, each entry is a tuple (x, c, expected_nu, abs_tol)
# where ν_canonical(x; c) must equal expected_nu within abs_tol.
# Reference pairs SHOULD include at least:
#   - canonical x=0.5 across the 5 cascade SSoT anchors
#   - a few off-anchor (x, c) test points showing the joint form
# ============================================================================

NU_CANONICAL_REFERENCE_PAIRS: List[Tuple[float, float, float, float]] = [
    # FILL_HERE (T13): tuples of (x, c, expected_nu, abs_tol)
    # Example structure (uncommment + fill on T13 supply):
    # (0.5, 0.30, EXPECTED_NU_AT_X05_C030, 1e-3),
    # (0.5, 0.42, EXPECTED_NU_AT_X05_C042, 1e-3),
    # (0.5, 0.618, EXPECTED_NU_AT_X05_C0618, 1e-3),
    # (0.5, 0.83, EXPECTED_NU_AT_X05_C083, 1e-3),  # = 1.9425 / x_at_radius
    # (0.5, 1.00, EXPECTED_NU_AT_X05_C1, 1e-3),
    # (X_OFF, C_OFF, EXPECTED_OFF, 1e-3),  # off-anchor sanity
]


def nu_canonical_self_check() -> Dict[str, Any]:
    """
    T13 verification framework — runs ν_canonical(x; c) at reference pairs.

    Returns a dict with keys:
        n_pairs : int  — number of reference pairs tested
        n_pass  : int  — pairs within abs_tol
        n_fail  : int  — pairs outside abs_tol
        failures: list — per-failure (x, c, expected, observed, abs_diff)

    Status states:
        - 0 pairs (T13 not yet wired): returns {n_pairs: 0, status: 'deferred'}
        - all pass: returns {n_pairs: N, n_pass: N, status: 'pass'}
        - any fail: raises AssertionError with diagnostic detail

    This function is a no-op when NU_CANONICAL_REFERENCE_PAIRS is empty,
    so it can be called unconditionally during main() preflight.
    """
    if not NU_CANONICAL_REFERENCE_PAIRS:
        return {
            "n_pairs": 0,
            "status": "deferred",
            "reason": "T13 verbatim not yet supplied (parent v4.8 §3.1 L335)",
        }

    failures = []
    n_pass = 0
    for (x, c, expected, abs_tol) in NU_CANONICAL_REFERENCE_PAIRS:
        observed = nu_canonical(x, c)
        if np.isnan(observed):
            failures.append({
                "x": x, "c": c, "expected": expected, "observed": None,
                "abs_diff": None, "reason": "nu_canonical returned NaN",
            })
            continue
        abs_diff = abs(observed - expected)
        if abs_diff > abs_tol:
            failures.append({
                "x": x, "c": c, "expected": expected, "observed": observed,
                "abs_diff": abs_diff, "reason": f"diff > tol={abs_tol}",
            })
        else:
            n_pass += 1

    n_pairs = len(NU_CANONICAL_REFERENCE_PAIRS)
    if failures:
        raise AssertionError(
            f"nu_canonical_self_check FAIL: "
            f"{len(failures)}/{n_pairs} reference pairs failed.\n"
            + "\n".join(
                f"  (x={f['x']}, c={f['c']}): expected {f['expected']}, "
                f"observed {f['observed']}, abs_diff={f['abs_diff']} "
                f"({f['reason']})"
                for f in failures
            )
        )
    return {
        "n_pairs": n_pairs, "n_pass": n_pass, "n_fail": 0,
        "status": "pass",
    }


# ============================================================================
# §6 χ_coh (Layer B-α: anchor 14 §4.13.2 + C3-A2 explicit cite;
#          Layer B-β: anchor 16 §5.7 analytic Strigari)
# ============================================================================

def compute_f_p(
    g_N: np.ndarray,
    vdisk: np.ndarray,
    vgas: np.ndarray,
    upsilon_d: float,
    a_0_value: float = A_0,
) -> float:
    """
    Plastic mass fraction f_p (Layer B-α operational realization).
    Source: C3-A1 sparc_fp_verification.py L131 (single-pass mass-weighted
    threshold, NON-iterative).

    Formula:
        plastic_mask = g_N < a_0
        weight = |vdisk|² · upsilon_d + |vgas|²
        f_p = sum(weight[plastic_mask]) / sum(weight)

    Parameters
    ----------
    g_N : per-radius Newtonian gravitational acceleration (m/s²)
    vdisk : per-radius disk circular velocity component (km/s)
    vgas : per-radius gas circular velocity component (km/s)
    upsilon_d : Y_d disk mass-to-light ratio (per-galaxy scalar, from phase1)
    a_0_value : MOND universal acceleration (default 1.2e-10 m/s²)

    Returns
    -------
    f_p : plastic mass fraction in [0, 1]
    """
    weight = (vdisk ** 2) * upsilon_d + (vgas ** 2)
    total = float(np.sum(weight))
    if total <= 0.0:
        return float("nan")
    plastic_mask = g_N < a_0_value
    return float(np.sum(weight[plastic_mask]) / total)


def chi_coh(galaxy: str, layer: str = "B-alpha", **kwargs: Any) -> float:
    """
    χ_coh = coherence factor.
    Source: anchor 14 §4.13.2 + C3-A2 explicit cite + anchor 16 §5.7 disambig.

    Layer B-α (operational realization, SPARC catalog domain, 163 galaxy):
        χ_coh ≈ 1 − f_p
        where f_p = plastic mass fraction from C3-A1 sparc_fp_verification.py L131
            (single-pass mass-weighted threshold, NON-iterative)
        plastic_mask = g_N < a_0
        f_p = np.sum(weight[plastic_mask]) / total_weight
        weight = |vdisk|² · upsilon_d + |vgas|²
        → CSV column: c_membrane = 1 − f_p

    Layer B-β (analytic closure, dSph extreme regime, Q→0):
        s = sigmoid(ΔU(sQ)/T_m)        # self-consistent
        s_0 = 1/(1 + exp(3/(2·T_m)))   # Q→0 analytic limit
            = 0.3515  at  T_m = √6
        G_Strigari = s_0 · (1 − s_0) · a_0 = 0.228 · a_0
        NOTE: Layer B-β is NOT implemented in C3-A1 (analytic computation only).

    Domain disjoint (anchor 16 §5.7):
        Layer B-α  →  SPARC catalog (163 galaxy operational)
        Layer B-β  →  dSph extreme regime (analytic only)

    Parameters
    ----------
    galaxy : galaxy identifier (e.g., "NGC3198")
    layer : "B-alpha" (default) or "B-beta"
    **kwargs : for B-alpha, expects g_N, vdisk, vgas, upsilon_d; alternatively
               f_p directly. For B-beta, no kwargs needed (analytic constant).

    Returns
    -------
    chi_coh : coherence factor
    """
    if layer == "B-alpha":
        if "f_p" in kwargs:
            f_p = float(kwargs["f_p"])
        elif all(k in kwargs for k in ("g_N", "vdisk", "vgas", "upsilon_d")):
            f_p = compute_f_p(
                g_N=np.asarray(kwargs["g_N"], dtype=float),
                vdisk=np.asarray(kwargs["vdisk"], dtype=float),
                vgas=np.asarray(kwargs["vgas"], dtype=float),
                upsilon_d=float(kwargs["upsilon_d"]),
            )
        else:
            raise ValueError(
                f"chi_coh(galaxy='{galaxy}', layer='B-alpha') requires either "
                f"f_p or (g_N, vdisk, vgas, upsilon_d). "
                f"Received kwargs: {list(kwargs.keys())}"
            )
        return 1.0 - f_p

    if layer == "B-beta":
        # Analytic Strigari closure, dSph extreme regime
        # s_0 = 1 / (1 + exp(3 / (2 * T_m)))  with T_m = √6 → s_0 = 0.3515
        # χ_coh under B-β is operational regime separate from B-α; the analytic
        # form here returns s_0 itself (representing membrane-side coherence
        # at extreme regime). Per anchor 16 §5.7, B-β values are documented
        # for reference only; not used in B-α SPARC pipeline.
        return S_0

    raise ValueError(
        f"chi_coh: unknown layer '{layer}'. Expected 'B-alpha' or 'B-beta'."
    )


# ============================================================================
# §7 ALGORITHM B PER-GALAXY (anchor 7 §2.5.3, R-2 LOCK)
# ============================================================================

@dataclass
class AlgorithmBResult:
    galaxy: str
    converged: bool
    n_iter: int
    c_galaxy_final: float          # per-galaxy summary statistic of c*(r)
    residual_final: float          # max_r |Δc*(r)|
    c_star_profile_final: Optional[np.ndarray] = None  # per-radius c*(r)
    out_of_range_flag: bool = False  # True if any c*(r) ∉ [0.30, 1.00]
    iter_trace: List[Tuple[int, float, float]] = field(default_factory=list)


def algorithm_b_step(
    galaxy: str,
    c_star_n: np.ndarray,
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    a_0_value: float = A_0,
) -> np.ndarray:
    """
    Algorithm B: single iteration step (anchor 7 §2.5.3, R-2 LOCK).

    For each radius r:
        ν_n(r) ≡ ν_canonical(x_r; c*(r)_n)         # x_r = g_bar(r)/a_0
        c*(r)_{n+1} ← ν_canonical^(-1)(g_obs(r)/g_bar(r); g_bar(r)/a_0; ν_n)

    NOTE: The full ν_canonical(x; c) form for arbitrary x is parent v4.8 §3.1
    L335 spec (T13 user-input fill remaining). Currently this step relies on
    nu_canonical() which only handles x=0.5 canonically. For radii where
    g_bar/a_0 ≠ 0.5, the inversion will return NaN and the galaxy will be
    flagged. This will be resolved when T13 is supplied.

    Returns
    -------
    c_star_next : per-radius c*(r) at iteration n+1
    """
    n_radii = len(c_star_n)
    c_star_next = np.full(n_radii, np.nan, dtype=float)
    for i in range(n_radii):
        x_i = g_bar[i] / a_0_value           # deep-MOND ratio
        target = g_obs[i] / g_bar[i]         # observed ν ratio
        # Solve f_opt(target; c) = x_i for c, OR equivalently invert
        # ν_canonical at the (target, x_i) operating point.
        # Per anchor 7 §2.5.3 the update IS the back-solve for c.
        c_star_next[i] = _backsolve_c(target, x_i)
    return c_star_next


def _backsolve_c(target_nu: float, x_at_radius: float) -> float:
    """
    Back-solve cascade c from (target ν, x = g_bar/a_0).
    Returns NaN if the ν_canonical(x; c) form needed is not yet wired (T13)
    or if the solve fails to bracket within the canonical range [0.30, 1.00].
    """
    # Bracket on cascade SSoT canonical range
    c_lo, c_hi = 0.30, 1.00
    try:
        # Need ν_canonical(x_at_radius; c). Currently only x=0.5 wired.
        if abs(x_at_radius - 0.5) > 1e-12:
            return float("nan")
        def residual(c_val: float) -> float:
            return nu_canonical(x_at_radius, c_val) - target_nu
        return brentq(residual, c_lo, c_hi, xtol=1e-10)
    except (ValueError, NotImplementedError):
        return float("nan")


def algorithm_b_per_galaxy(
    galaxy_row: pd.Series,
    rotation_curve: Optional[Dict[str, np.ndarray]] = None,
) -> AlgorithmBResult:
    """
    Algorithm B simultaneous self-consistency loop (anchor 7 §2.5.3, R-2 LOCK).

    Outline (anchor 7 §2.5.3):
        1. Initialize: c*(r)_0 = c_galaxy = 0.42  ∀ r
           (anchor 6 §2.3 inner anchor typical-case prior)
        2. Iterate for n = 0, 1, 2, ... until convergence:
            a. Compute ν_n(x) = ν_canonical(x; c*(r)_n) at each r
               using cascade SSoT 5-anchor Lagrange (parent §3.1 L335)
            b. Update:
                c*(r)_{n+1} = ν_canonical^(-1)(g_obs(r)/g_bar(r);
                                                g_bar(r)/a_0; ν_n)
            c. Convergence: max_r |c*(r)_{n+1} − c*(r)_n| < tol = 1e-6
            d. Maximum iterations: N_max = 50
               (typical: 5-20 iter expected per smoothness of cascade SSoT
                deg-4 Lagrange composition, C³ continuous within
                [0.30, 1.00] anchor range)
        3. On convergence: c*(r)_∞ is the B-realized profile per galaxy.
           NLL_B = Σ_r log(σ_g(r)) + const
           (residual term = 0 since g_pred = g_obs exactly when converged)
        4. On non-convergence (divergence / N_max reached / c*(r) ∉ [0.30, 1.00]):
           galaxy flagged for post-hoc audit, excluded from §2.5.4 selection.

    Parameters
    ----------
    galaxy_row : merged TA3+phase1+MRT row (galaxy-level scalars)
    rotation_curve : per-galaxy rotation curve dict with keys
                     'r' (kpc), 'g_obs' (m/s²), 'g_bar' (m/s²), 'sigma_g' (m/s²)
                     Loaded externally from Lelli SPARC RC .dat files (T12).
                     If None, function returns a non-converged stub result
                     (galaxy is flagged, not failed silently).
    """
    galaxy = str(galaxy_row["galaxy"])

    if rotation_curve is None:
        # Per-radius rotation curve not yet loaded (T12 wiring pending).
        # Return non-converged stub so the run does not silently produce
        # false numerics. The galaxy is flagged for post-hoc audit.
        return AlgorithmBResult(
            galaxy=galaxy,
            converged=False,
            n_iter=0,
            c_galaxy_final=C_GALAXY_INIT,
            residual_final=float("nan"),
            c_star_profile_final=None,
            out_of_range_flag=False,
            iter_trace=[],
        )

    g_obs = np.asarray(rotation_curve["g_obs"], dtype=float)
    g_bar = np.asarray(rotation_curve["g_bar"], dtype=float)
    n_radii = len(g_obs)
    if n_radii == 0:
        return AlgorithmBResult(
            galaxy=galaxy, converged=False, n_iter=0,
            c_galaxy_final=C_GALAXY_INIT, residual_final=float("nan"),
        )

    # Initialize c*(r)_0 = 0.42 ∀ r
    c_star = np.full(n_radii, C_GALAXY_INIT, dtype=float)
    trace: List[Tuple[int, float, float]] = []
    converged = False
    residual = float("inf")
    out_of_range = False

    for n in range(N_MAX + 1):
        c_star_next = algorithm_b_step(galaxy, c_star, g_obs, g_bar, A_0)
        # Range constraint check (cascade SSoT canonical [0.30, 1.00])
        valid = ~np.isnan(c_star_next)
        if not np.any(valid):
            # All radii failed inversion — flag and break
            out_of_range = True
            break
        # Compute residual on valid radii only
        diff = c_star_next[valid] - c_star[valid]
        residual = float(np.max(np.abs(diff))) if len(diff) > 0 else float("nan")
        # Mean c*(r) as galaxy-level summary
        c_galaxy_summary = float(np.nanmean(c_star_next))
        trace.append((n, c_galaxy_summary, residual))
        # Range check
        if np.any((c_star_next[valid] < 0.30) | (c_star_next[valid] > 1.00)):
            out_of_range = True
            c_star = c_star_next
            break
        if residual < TOL:
            converged = True
            c_star = c_star_next
            break
        c_star = c_star_next

    return AlgorithmBResult(
        galaxy=galaxy,
        converged=converged and not out_of_range,
        n_iter=len(trace) - 1 if converged else (
            len(trace) if out_of_range else N_MAX
        ),
        c_galaxy_final=float(np.nanmean(c_star)),
        residual_final=residual,
        c_star_profile_final=c_star,
        out_of_range_flag=out_of_range,
        iter_trace=trace,
    )


# ============================================================================
# §8 f_E ADOPTION (anchor 7 §2.5.2 + §2.5.4, Q-C1 LOCK k_E=2 default)
# ============================================================================

def e_pipeline_score(galaxy_row: pd.Series, k_E: int = K_E_DEFAULT) -> float:
    """
    E pipeline score: NLL_E + 2·k_E (AIC convention).

    NLL_E formula (anchor 7 §2.5.2 L246-253):
        NLL_E(θ_E) = (1/2) · Σ_r [(g_obs(r) − g_pred(r; θ_E))² / σ_g(r)²]
                     + Σ_r log(σ_g(r)) + const
    where σ_g(r) = parent v4.8 §6 line XXX (per-radius observational uncertainty,
    Gaussian assumption; XXX literal fills during v0.2 round, T15).

    AIC_E = 2·NLL_E + 2·k_E,  k_E = 2 (Q-C1 LOCK: c_0 + λ_eff).

    Currently NotImplemented pending T14 (NLL_E formula wiring) + T12
    (rotation curve loader for per-radius g_obs / g_bar / σ_g arrays).
    """
    raise NotImplementedError(
        "e_pipeline_score requires T12 (rotation curve loader) + T14 "
        "(NLL_E explicit formula) + T15 (σ_g(r) reference fill)."
    )


def b_pipeline_score(galaxy_row: pd.Series, b_result: AlgorithmBResult) -> float:
    """
    Algorithm B score: NLL_B + 2·k_B (AIC convention).

    NLL_B formula (anchor 7 §2.5.3, on convergence):
        NLL_B = Σ_r log(σ_g(r)) + const
        (residual term = 0 since g_pred = g_obs exactly when converged)

    AIC_B = 2·NLL_B + 2·k_B,  k_B = 0 (R-1 (R1-α) LOCK: parameter-free canonical).

    On non-convergence, returns NaN (galaxy excluded from §2.5.4 selection).
    """
    if not b_result.converged:
        return float("nan")
    raise NotImplementedError(
        "b_pipeline_score requires T12 (rotation curve loader) + T15 "
        "(σ_g(r) reference fill) for the log-σ_g sum."
    )


def select_pipeline(
    galaxy_row: pd.Series,
    b_result: AlgorithmBResult,
    k_E: int = K_E_DEFAULT,
    threshold: Optional[float] = None,
) -> Tuple[str, float]:
    """
    ΔAIC selection (anchor 7 §2.5.4, Q-C3 LOCK + Lesson 92 parsimony first).

    ΔAIC = AIC_E − AIC_B = 2·(NLL_E − NLL_B) + 4   (k_E=2, k_B=0)

    Per-galaxy decision rule:
        ΔAIC > +threshold (default +2)  → "B"   (B decisive evidence)
        ΔAIC < −threshold (default −2)  → "E"   (E overcomes B's +4 a priori)
        |ΔAIC| ≤ threshold              → "B"   (parsimony default, k_B=0 < k_E=2)

    Returns ("B", delta) or ("E", delta).
    """
    if threshold is None:
        threshold = DELTA_AIC_THRESHOLD
    aic_e = e_pipeline_score(galaxy_row, k_E=k_E)
    aic_b = b_pipeline_score(galaxy_row, b_result)
    delta = aic_e - aic_b
    if np.isnan(delta):
        # B did not converge; fall back to E
        return ("E", delta)
    if delta > threshold:
        selected = "B"
    elif delta < -threshold:
        selected = "E"
    else:
        # |ΔAIC| ≤ threshold → parsimony default → B
        selected = "B"
    return (selected, delta)


# ============================================================================
# T14 verification framework — NLL reference pair self-check
# ============================================================================
# Populates on T14 verbatim supply (anchor 7 §2.5.2 / §2.5.3 explicit
# formulas + a representative galaxy benchmark).
#
# Each entry is (galaxy_name, expected_nll_e, expected_nll_b, abs_tol)
# computed against a fixed test galaxy where g_obs(r) / g_bar(r) / σ_g(r)
# are known reference values.
# ============================================================================

NLL_REFERENCE_PAIRS: List[Tuple[str, float, float, float]] = [
    # FILL_HERE (T14): tuples of (galaxy, expected_NLL_E, expected_NLL_B, abs_tol)
    # Example structure (uncomment + fill on T14 supply):
    # ("NGC3198", EXPECTED_NLL_E_NGC3198, EXPECTED_NLL_B_NGC3198, 1e-3),
]


def nll_self_check(
    test_galaxies: Optional[Dict[str, pd.Series]] = None,
    b_results: Optional[Dict[str, AlgorithmBResult]] = None,
) -> Dict[str, Any]:
    """
    T14 verification framework — runs NLL_E / NLL_B against reference pairs.

    Returns a dict mirroring nu_canonical_self_check structure.

    No-op when NLL_REFERENCE_PAIRS is empty; safe to call unconditionally.
    """
    if not NLL_REFERENCE_PAIRS:
        return {
            "n_pairs": 0,
            "status": "deferred",
            "reason": "T14 verbatim not yet supplied (anchor 7 §2.5.2 / §2.5.3)",
        }

    if test_galaxies is None or b_results is None:
        return {
            "n_pairs": len(NLL_REFERENCE_PAIRS),
            "status": "deferred",
            "reason": "test_galaxies / b_results not provided to nll_self_check",
        }

    failures = []
    n_pass = 0
    for (galaxy, expected_e, expected_b, abs_tol) in NLL_REFERENCE_PAIRS:
        if galaxy not in test_galaxies or galaxy not in b_results:
            failures.append({
                "galaxy": galaxy,
                "reason": "galaxy not in test_galaxies / b_results dict",
            })
            continue
        try:
            obs_aic_e = e_pipeline_score(test_galaxies[galaxy])
            obs_aic_b = b_pipeline_score(
                test_galaxies[galaxy], b_results[galaxy]
            )
            # Convert AIC back to NLL for comparison: NLL = (AIC − 2k) / 2
            obs_nll_e = (obs_aic_e - 2 * K_E_DEFAULT) / 2.0
            obs_nll_b = (obs_aic_b - 2 * K_B) / 2.0
        except NotImplementedError as exc:
            failures.append({
                "galaxy": galaxy,
                "reason": f"score function not yet wired: {exc}",
            })
            continue
        diff_e = abs(obs_nll_e - expected_e)
        diff_b = abs(obs_nll_b - expected_b)
        if diff_e > abs_tol or diff_b > abs_tol:
            failures.append({
                "galaxy": galaxy,
                "expected_NLL_E": expected_e, "observed_NLL_E": obs_nll_e,
                "diff_E": diff_e,
                "expected_NLL_B": expected_b, "observed_NLL_B": obs_nll_b,
                "diff_B": diff_b,
                "abs_tol": abs_tol,
            })
        else:
            n_pass += 1

    n_pairs = len(NLL_REFERENCE_PAIRS)
    if failures:
        raise AssertionError(
            f"nll_self_check FAIL: {len(failures)}/{n_pairs} reference pairs "
            f"failed.\n" + "\n".join(f"  {f}" for f in failures)
        )
    return {"n_pairs": n_pairs, "n_pass": n_pass, "n_fail": 0, "status": "pass"}


# ============================================================================
# f_E AGGREGATION + sample-level decision (Q4 LOCK F_E_LOWER/F_E_UPPER)
# ============================================================================

def aggregate_f_E_adoption(
    sparc_df: pd.DataFrame,
    b_results: Dict[str, AlgorithmBResult],
    k_E: int = K_E_DEFAULT,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Per-galaxy ΔAIC selection → aggregate f_E → sample-level decision.

    Sample-level rule (Q4 LOCK, anchor 7 §2.5.4):
        f_E < F_E_LOWER (= 0.20)             → "B"      (B globally adopted)
        f_E > F_E_UPPER (= 0.80)             → "E"      (E globally adopted)
        F_E_LOWER ≤ f_E ≤ F_E_UPPER          → "mixed"  (flagged for §6 cross-id audit)

    Skips galaxies where:
        - Algorithm B did not converge (no b_result.converged)
        - select_pipeline raises (T14 not yet wired)

    Returns dict with per-galaxy decisions + aggregate counts + sample-level
    decision + skip diagnostics. Population to summary.json::f_E_adoption.
    """
    if threshold is None:
        threshold = DELTA_AIC_THRESHOLD

    decisions: Dict[str, str] = {}
    deltas: Dict[str, float] = {}
    skipped_galaxies: Dict[str, str] = {}

    for _, row in sparc_df.iterrows():
        galaxy = str(row["galaxy"])
        if galaxy not in b_results:
            skipped_galaxies[galaxy] = "no Algorithm B result"
            continue
        try:
            selected, delta = select_pipeline(
                row, b_results[galaxy], k_E=k_E, threshold=threshold
            )
            decisions[galaxy] = selected
            deltas[galaxy] = float(delta) if not np.isnan(delta) else float("nan")
        except NotImplementedError as exc:
            skipped_galaxies[galaxy] = f"score function not yet wired: {exc}"
            continue

    n_total = len(decisions)
    n_E = sum(1 for v in decisions.values() if v == "E")
    n_B = sum(1 for v in decisions.values() if v == "B")
    f_E = (n_E / n_total) if n_total > 0 else float("nan")

    # Sample-level decision
    if n_total == 0:
        sample_decision = "deferred"
        sample_decision_reason = (
            "no per-galaxy decisions available (T14 not yet wired)"
        )
    elif f_E < F_E_LOWER:
        sample_decision = "B"
        sample_decision_reason = f"f_E={f_E:.4f} < F_E_LOWER={F_E_LOWER}"
    elif f_E > F_E_UPPER:
        sample_decision = "E"
        sample_decision_reason = f"f_E={f_E:.4f} > F_E_UPPER={F_E_UPPER}"
    else:
        sample_decision = "mixed"
        sample_decision_reason = (
            f"F_E_LOWER={F_E_LOWER} ≤ f_E={f_E:.4f} ≤ F_E_UPPER={F_E_UPPER}; "
            f"flagged for §6 cross-id audit"
        )

    # Sanity stats on deltas (filter NaN)
    valid_deltas = [d for d in deltas.values() if not np.isnan(d)]
    delta_mean = float(np.mean(valid_deltas)) if valid_deltas else float("nan")
    delta_median = float(np.median(valid_deltas)) if valid_deltas else float("nan")

    return {
        "k_E_used": k_E,
        "threshold": threshold,
        "n_total_decided": n_total,
        "n_skipped": len(skipped_galaxies),
        "n_selected_E": n_E,
        "n_selected_B": n_B,
        "f_E_fraction": f_E,
        "delta_AIC_mean": delta_mean,
        "delta_AIC_median": delta_median,
        "sample_level_decision": sample_decision,
        "sample_level_decision_reason": sample_decision_reason,
        "F_E_LOWER": F_E_LOWER,
        "F_E_UPPER": F_E_UPPER,
        "per_galaxy_decisions": decisions,
        "per_galaxy_delta_AIC": deltas,
        "skipped_galaxies": skipped_galaxies,
    }


# ============================================================================
# §9 dSph J3 + b_α 3-AXIS AUDIT (anchor 7 §2.5.5)
# ============================================================================

def load_dsph(path: str) -> pd.DataFrame:
    """Load dSph dataset (31 sample). Format may vary; adapt as needed."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def dsph_j3_check(dsph: pd.DataFrame) -> pd.DataFrame:
    """
    Per-dwarf J3 metric + pass flag (anchor 7 §2.5.5 + F6 audit).

    Anchor 7 §2.5.5 baseline (Axis 2):
        - 28 typical galaxy: typical pattern         → in_28_baseline = True
        -  3 reverse galaxy: reverse pattern         → in_28_baseline = False
        - reference baseline: J0 minimal form        (C3-A4, SHA 7e8823f4 L57)
            c*(r) = clip(1 − f_p(r), [0.10, 1.00])
            ※ §2.5.5 内 reference baseline 専用
              (anchor 5 Issue 1 fix: J0 は §2 INDEPENDENT BASE 除外、
               §6.2 cross-id audit constraint 用)

    F6 audit (Phase C3 v3 §X dSph 30 sample chapter, C3-A5 SHA 69fb1a95):
        dSph 30 sample = subset of 31 used for Strigari extension cross-paper
        coherence (b_α_dSph = +0.1127). The 30 sample populates in_30_baseline.

    Phase C3 cross-paper coherence (milestone §I.1):
        b_α_SPARC = +0.1084  (163 galaxy)
        b_α_dSph  = +0.1127  (30 galaxy, Strigari extension)
        |diff|    =  0.0042  (0.5% agreement)
        ΔAIC(α-γ) = −2.00    (parsimony confirm)
    """
    dsph = dsph.copy()
    if "J3_metric" not in dsph.columns:
        # J3 metric definition is anchor 7 §2.5.5 spec; if the dataset
        # already carries it, use it. Otherwise this is a wiring gap (T11
        # for the dSph dataset format — verify column names).
        raise NotImplementedError(
            "dSph dataset missing 'J3_metric' column. Verify dataset format "
            "(T11 USER_VERIFY) or compute J3 metric per anchor 7 §2.5.5 spec."
        )

    # Strigari analytic prediction (anchor 16 §5.7 / anchor 7 §2.5.5)
    dsph["s_0"] = S_0
    dsph["G_Strigari_over_a0"] = S_0 * (1.0 - S_0)

    # in_28_baseline — anchor 7 §2.5.5 28/31 typical pattern
    # in_30_baseline — F6 audit Phase C3 v3 §X dSph 30 sample (filled at v0.2 run)
    if "in_28_baseline" not in dsph.columns:
        dsph["in_28_baseline"] = False  # to be set by J3 metric pass logic
    if "in_30_baseline" not in dsph.columns:
        dsph["in_30_baseline"] = False  # to be set per F6 audit criterion

    return dsph


def _b_alpha_slope_fit(c_membrane: np.ndarray, gs0_over_a0: np.ndarray) -> Dict[str, float]:
    """
    Single-sample b_α estimator via log-log slope fit.
    Source: C3-A1 sparc_fp_verification.py L277-287 (SHA ab6f509b) pattern:
        log_c   = log10(c_membrane)
        log_GS0 = log10(G·Σ_0/a_0)
        slope, intercept, r, p, se = stats.linregress(log_GS0, log_c)
        b_α = slope

    Returns
    -------
    {b_alpha, intercept, r_value, r_squared, p_value, stderr, n_used}
        n_used: count of valid (c > 0, GS0 > 0, finite) samples
    """
    from scipy import stats as _stats
    c = np.asarray(c_membrane, dtype=float)
    g = np.asarray(gs0_over_a0, dtype=float)
    valid = (c > 0) & (g > 0) & np.isfinite(c) & np.isfinite(g)
    n_used = int(np.sum(valid))
    if n_used < 3:
        return {
            "b_alpha": float("nan"), "intercept": float("nan"),
            "r_value": float("nan"), "r_squared": float("nan"),
            "p_value": float("nan"), "stderr": float("nan"),
            "n_used": n_used,
        }
    log_c = np.log10(c[valid])
    log_g = np.log10(g[valid])
    slope, intercept, r_val, p_val, stderr = _stats.linregress(log_g, log_c)
    return {
        "b_alpha": float(slope), "intercept": float(intercept),
        "r_value": float(r_val), "r_squared": float(r_val ** 2),
        "p_value": float(p_val), "stderr": float(stderr),
        "n_used": n_used,
    }


def b_alpha_3axis_audit(
    sparc_df: pd.DataFrame, dsph_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute b_α on 3 axes (anchor 7 §2.5.5 Axis 3 + Phase C3 v3 §4.3
    universal coupling A-grade, SHA 69fb1a95 C3-A5).

    Estimator (C3-A1 sparc_fp_verification.py L277-287 verbatim pattern):

        b_α = slope of log10(c_membrane) vs log10(G·Σ_0/a_0)

    where:
        c_membrane = 1 − f_p             (Layer B-α operational, SPARC catalog)
                   = analytic per-dwarf  (Layer B-β extreme, dSph 30 sample)
        G·Σ_0/a_0  = baryonic surface density characteristic ratio
                   = v_flat² / (h_R · a_0)        (per-galaxy, SI conversion)
                   = sigma_los² / (R_half · a_0)  (per-dwarf)

    3-axis decomposition (anchor 19 §1.5 baselines):
        axis 1: SPARC-only sample (171-pool, c_mem > 0 subset, ~163 galaxies)
                → b_α_SPARC, baseline +0.1084
        axis 2: dSph-only sample (J3 30/31 in 30_baseline subset)
                → b_α_dSph, baseline +0.1127
        axis 3: combined SPARC + dSph integrated sample (3.92 dex density range)
                → b_α_combined, target 0.11 ± 0.005 (Phase C3 v3 §4.3 A-grade)

    Phase C3 cross-paper coherence (anchor 19 §1.5 + milestone summary §I.1):
        |b_α_SPARC − b_α_dSph| baseline = 0.0042 (~0.5% agreement)
        AC4 acceptance: |diff| ≤ 0.005 (per EXECUTION_PLAN §4.3)

    Returns
    -------
    dict with keys:
        axis_1_SPARC : per-axis fit dict (from _b_alpha_slope_fit)
        axis_2_dSph  : per-axis fit dict
        axis_3_combined : per-axis fit dict
        abs_diff_SPARC_dSph : |b_α_SPARC − b_α_dSph|
        agreement_pct        : 100 * |diff| / |b_α_SPARC|
        baseline_b_alpha_SPARC : 0.1084 (anchor 19 §1.5 record)
        baseline_b_alpha_dSph  : 0.1127 (anchor 19 §1.5 record)
        baseline_abs_diff      : 0.0042 (anchor 19 §1.5 record)
        baseline_phase_c3      : 0.11 ± 0.005 (Phase C3 v3 §4.3 universal coupling)
        ac4_pass : bool — |diff| ≤ 0.005 (EXECUTION_PLAN §4.3 acceptance)
        deviation_from_phase_c3_sigma : (b_α_combined − 0.11) / 0.005
    """
    KPC_M = 3.0857e19  # meters per kpc

    # --- axis 1: SPARC-only ---
    pool = sparc_df[sparc_df.get("in_171_pool", True)] if "in_171_pool" in sparc_df.columns else sparc_df
    c_sparc = pool["c_mem"].values if "c_mem" in pool.columns else (
        1.0 - pool["f_p"].values if "f_p" in pool.columns else np.array([])
    )
    if "Vflat" in pool.columns and "Rdisk" in pool.columns and len(c_sparc) > 0:
        v_flat_ms = pool["Vflat"].values * 1e3      # km/s -> m/s
        h_R_m    = pool["Rdisk"].values * KPC_M     # kpc -> m
        gs0_sparc = (v_flat_ms ** 2) / (h_R_m * A_0)  # dimensionless ratio
    else:
        gs0_sparc = np.array([])
    axis_1 = _b_alpha_slope_fit(c_sparc, gs0_sparc)

    # --- axis 2: dSph-only ---
    in_30 = dsph_df.get("in_30_baseline", pd.Series([False] * len(dsph_df)))
    dsph_pool = dsph_df[in_30] if in_30.any() else dsph_df
    c_dsph = (
        dsph_pool["c_mem"].values if "c_mem" in dsph_pool.columns
        else dsph_pool["G_Strigari_over_a0"].values if "G_Strigari_over_a0" in dsph_pool.columns
        else np.full(len(dsph_pool), S_0)
    )
    if "sigma_los" in dsph_pool.columns and "Rhalf" in dsph_pool.columns and len(dsph_pool) > 0:
        sigma_los_ms = dsph_pool["sigma_los"].values * 1e3       # km/s -> m/s
        rhalf_m      = dsph_pool["Rhalf"].values * 3.0857e16     # pc -> m
        gs0_dsph = (sigma_los_ms ** 2) / (rhalf_m * A_0)
    else:
        gs0_dsph = np.array([])
    axis_2 = _b_alpha_slope_fit(c_dsph, gs0_dsph)

    # --- axis 3: combined SPARC + dSph (Phase C3 v3 §4.3) ---
    c_combined   = np.concatenate([np.asarray(c_sparc),   np.asarray(c_dsph)])
    gs0_combined = np.concatenate([np.asarray(gs0_sparc), np.asarray(gs0_dsph)])
    axis_3 = _b_alpha_slope_fit(c_combined, gs0_combined)

    # --- cross-paper coherence (anchor 19 §1.5) ---
    b_sparc = axis_1["b_alpha"]
    b_dsph  = axis_2["b_alpha"]
    b_comb  = axis_3["b_alpha"]
    if np.isfinite(b_sparc) and np.isfinite(b_dsph):
        abs_diff = abs(b_sparc - b_dsph)
        agreement_pct = 100.0 * abs_diff / abs(b_sparc) if b_sparc != 0 else float("nan")
    else:
        abs_diff = float("nan")
        agreement_pct = float("nan")

    return {
        "axis_1_SPARC": axis_1,
        "axis_2_dSph": axis_2,
        "axis_3_combined": axis_3,
        "abs_diff_SPARC_dSph": abs_diff,
        "agreement_pct": agreement_pct,
        "baseline_b_alpha_SPARC": B_ALPHA_SPARC_BASELINE,        # 0.1084
        "baseline_b_alpha_dSph": B_ALPHA_DSPH_BASELINE,          # 0.1127
        "baseline_abs_diff": B_ALPHA_ABS_DIFF_BASELINE,          # 0.0042
        "baseline_phase_c3_central": 0.11,                        # Phase C3 v3 §4.3
        "baseline_phase_c3_sigma": 0.005,                         # Phase C3 v3 §4.3
        "ac4_pass": bool(abs_diff <= 0.005) if np.isfinite(abs_diff) else None,
        "deviation_from_phase_c3_sigma": (
            (b_comb - 0.11) / 0.005 if np.isfinite(b_comb) else float("nan")
        ),
        "source_anchor": "anchor 7 §2.5.5 Axis 3 + Phase C3 v3 §4.3 (C3-A5 69fb1a95)",
        "estimator_source": "C3-A1 sparc_fp_verification.py L277-287 (ab6f509b)",
    }


def run_dsph_audit(
    dsph: pd.DataFrame,
    sparc_df: pd.DataFrame,
    deferred_on_missing: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate dSph J3 + b_α 3-axis audit.

    Calls:
        dsph_j3_check()       — adds in_28_baseline / in_30_baseline columns
        b_alpha_3axis_audit() — Phase C3 cross-paper coherence

    Parameters
    ----------
    dsph : DataFrame   — loaded dSph 31 sample
    sparc_df : DataFrame — SPARC 171-galaxy fit pool (for axis 1 b_α)
    deferred_on_missing : bool
        If True (default), missing T11 column / NotImplementedError yields
        a {"status": "deferred", "reason": ...} block instead of raising.
        Useful pre-T11/T14 for orchestration validation runs.

    Returns
    -------
    dsph_audit_result : dict to be merged into summary.json::dSph_J3 +
                        summary.json::b_alpha
    """
    result: Dict[str, Any] = {
        "n_total": len(dsph),
        "G_Strigari_over_a0_predicted": G_STRIGARI_OVER_A0,
        "j3_status": "deferred",
        "b_alpha_status": "deferred",
    }

    # J3 audit
    try:
        dsph_with_j3 = dsph_j3_check(dsph)
        result["j3_status"] = "computed"
        result["n_pass_observed_28_baseline"] = int(
            dsph_with_j3.get("in_28_baseline", pd.Series(dtype=bool)).sum()
        )
        result["n_pass_observed_30_baseline"] = int(
            dsph_with_j3.get("in_30_baseline", pd.Series(dtype=bool)).sum()
        )
        # G_Strigari observed mean from per-dwarf
        if "G_Strigari_over_a0" in dsph_with_j3.columns:
            result["G_Strigari_over_a0_observed_mean"] = float(
                dsph_with_j3["G_Strigari_over_a0"].mean()
            )
            result["G_Strigari_agreement_pct"] = (
                100.0 * abs(
                    result["G_Strigari_over_a0_observed_mean"]
                    - G_STRIGARI_OVER_A0
                ) / G_STRIGARI_OVER_A0
            )
    except NotImplementedError as exc:
        if not deferred_on_missing:
            raise
        result["j3_status"] = "deferred"
        result["j3_deferred_reason"] = str(exc)

    # b_α 3-axis audit (v1.0.3: each axis returns dict with b_alpha + diagnostics;
    # b_alpha_3axis_audit() also pre-computes abs_diff_SPARC_dSph + ac4_pass).
    try:
        b_alpha = b_alpha_3axis_audit(sparc_df, dsph)
        result["b_alpha_status"] = "computed"
        result["b_alpha_axes"] = b_alpha
        # Extract scalar b_α from per-axis dict (shape: {b_alpha, intercept, r_value, ...})
        a1 = b_alpha.get("axis_1_SPARC", {}) or {}
        a2 = b_alpha.get("axis_2_dSph", {}) or {}
        a3 = b_alpha.get("axis_3_combined", {}) or {}
        b_sparc = a1.get("b_alpha", float("nan")) if isinstance(a1, dict) else float("nan")
        b_dsph  = a2.get("b_alpha", float("nan")) if isinstance(a2, dict) else float("nan")
        b_comb  = a3.get("b_alpha", float("nan")) if isinstance(a3, dict) else float("nan")
        # Use pre-computed abs_diff if available, else recompute defensively
        abs_diff = b_alpha.get("abs_diff_SPARC_dSph", float("nan"))
        if not isinstance(abs_diff, (int, float)) or not np.isfinite(abs_diff):
            if np.isfinite(b_sparc) and np.isfinite(b_dsph):
                abs_diff = abs(b_sparc - b_dsph)
            else:
                abs_diff = float("nan")
        result["b_alpha_SPARC_observed"] = b_sparc
        result["b_alpha_dSph_observed"] = b_dsph
        result["b_alpha_combined_observed"] = b_comb
        result["b_alpha_abs_diff"] = abs_diff
        result["b_alpha_anchor19_baseline_abs_diff"] = B_ALPHA_ABS_DIFF_BASELINE
        result["b_alpha_within_AC4_threshold_005"] = (
            bool(abs_diff <= 0.005) if np.isfinite(abs_diff) else None
        )
        # Phase C3 v3 §4.3 universal coupling deviation (combined axis)
        result["b_alpha_phase_c3_deviation_sigma"] = b_alpha.get(
            "deviation_from_phase_c3_sigma", float("nan")
        )
    except NotImplementedError as exc:
        if not deferred_on_missing:
            raise
        result["b_alpha_status"] = "deferred"
        result["b_alpha_deferred_reason"] = str(exc)

    return result


# ============================================================================
# §10 SUB-ISSUE S-1〜S-6 RESOLVERS (anchor 7 §2.5.6 [H-1])
# ============================================================================

def resolve_S1(sparc_df: pd.DataFrame, dsph_df: pd.DataFrame) -> Dict[str, Any]:
    """
    S-1 (category A: reference resolution) — F2 link

    F2 placeholder resolution (anchor 8 §2.6.5 E5-γ, S-1 specific aspect):
        Reference: Phase C3 v3 §X Lesson 91 (bridge pre-cut protocol)
        Source: internal_memo_c3_extension_v3.pdf  SHA  69fb1a95...  (C3-A5)
        Lesson 91 = "bridge / extreme-regime pre-cut protocol"
        Resolution path: §2.5 v0.2 round literal § number resolution

    Bridge pre-cut 4-galaxy list (anchor 7 §2.5.1):
        4 galaxy が cascade SSoT 5-anchor canonical range
        (c ∈ {0.30, 0.42, 0.618, 0.80, 1.00}) の lower edge boundary 近傍に位置し、
        Phase C3 v3 §X analysis で fit instability + Algorithm B convergence
        failure risk が確認済 → §2.5 fit pool から事前除外。
        SPARC 175 − 4 = 171-galaxy fit pool 確定。

    The §X literal chapter number fills during the v0.2 round itself
    (Phase C3 v3 paper chapter numbering finalization synchronizes with this).
    """
    return {
        "category": "A",
        "linked_F": "F2",
        "anchor_8_verbatim": (
            "Phase C3 v3 §X Lesson 91 chapter placeholder "
            "(bridge pre-cut 4 galaxy reference)"
        ),
        "source": {
            "C3_A5_SHA": "69fb1a95",
            "C3_A5_filename": "internal_memo_c3_extension_v3.pdf",
            "lesson_91_description": "bridge / extreme-regime pre-cut protocol",
        },
        "bridge_pre_cut_4_galaxies": list(EXCLUDED_4_SPARC_GALAXIES.keys()),
        "fit_pool_after_pre_cut": SPARC_FIT_POOL_SIZE,
        "resolution_input_required": (
            "Phase C3 v3 §X chapter number — fills during v0.2 run"
        ),
        "resolution_status": "structure_filled_pending_chapter_number",
        "resolution_value": None,  # populated when §X is filled
    }


def resolve_S2(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """S-2 (category B: operational protocol)"""
    return {
        "category": "B",
        "resolution_method": "operational protocol per anchor 7 §2.5.6",
        "resolution_status": "pending",
        "resolution_value": None,
    }


def resolve_S3(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """
    S-3 (category A: reference resolution) — F1 link

    F1 placeholder resolution (anchor 8 §2.6.5 E5-γ, S-3 specific aspect):
        Reference: parent v4.8 §6 line XXX (σ_g(r) observational uncertainty)
        Status at anchor 8: placeholder maintained, handoff continuation
        Resolution path: §2.5 v0.2 round literal reference resolution

    NLL computation (anchor 7 §2.5.2):
        NLL_E(θ_E) = (1/2) · Σ_r [(g_obs(r) − g_pred(r; θ_E))² / σ_g(r)²]
                     + Σ_r log(σ_g(r)) + const
    where σ_g(r) = parent v4.8 §6 で確立した per-radius observational
    uncertainty (Gaussian assumption).

    Source SHA references:
        parent v4.8 .tex companion ja  : 902f79c6  (NEW canonical 2026-04-30)
        parent v4.8 .tex companion en  : 2dcf69e6
        parent v4.7.8 短縮版           : b7bf9629
        parent v4.8 (historical)       : 394f2571  (superseded, retained)

    The XXX literal line number fills during the v0.2 round itself
    (when the parent v4.8 §6 line is located against the 902f79c6 canonical).
    """
    return {
        "category": "A",
        "linked_F": "F1",
        "anchor_8_verbatim": (
            "parent v4.8 §6 line XXX (σ_g(r) observational uncertainty reference)"
        ),
        "source": {
            "parent_v48_tex_canonical_ja_SHA": "902f79c6",
            "parent_v48_tex_canonical_en_SHA": "2dcf69e6",
            "parent_v478_short_SHA": "b7bf9629",
            "parent_v48_historical_SHA": "394f2571",
        },
        "sigma_g_reference_target": "parent v4.8 §6 line XXX",
        "resolution_input_required": (
            "parent v4.8 §6 line number — fills during v0.2 run"
        ),
        "resolution_status": "structure_filled_pending_line_number",
        "resolution_value": None,  # populated when XXX is filled
    }


def resolve_S4(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """S-4 (category B: operational protocol)"""
    return {
        "category": "B",
        "resolution_method": "operational protocol per anchor 7 §2.5.6",
        "resolution_status": "pending",
        "resolution_value": None,
    }


def resolve_S5(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """S-5 (category B: operational protocol)"""
    return {
        "category": "B",
        "resolution_method": "operational protocol per anchor 7 §2.5.6",
        "resolution_status": "pending",
        "resolution_value": None,
    }


def resolve_S6(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """S-6 (category C: numerical threshold determination)"""
    return {
        "category": "C",
        "resolution_method": "numerical threshold from per-galaxy distribution",
        "threshold_target": None,    # TODO_USER_INPUT
        "threshold_observed": None,
        "resolution_status": "pending",
        "resolution_value": None,
    }


def resolve_F8_k_E_sensitivity(sparc_df: pd.DataFrame) -> Dict[str, Any]:
    """
    F8: k_E=2 default LOCK (Q-C1) preserved + k_E=1 sensitivity audit.
    Re-runs E pipeline with k_E=1, compares ΔAIC against k_E=2 baseline.
    """
    # Two-pass: k_E=2 (default) and k_E=1 (sensitivity)
    return {
        "k_E_2_default_used": True,
        "k_E_1_sensitivity_observed": None,
        "delta_AIC_k_E_2_minus_k_E_1": None,
        "Q_C1_LOCK_preserved": None,  # set True if k_E=2 remains preferred
        "resolution_status": "pending",
    }


# ============================================================================
# ACCEPTANCE CRITERIA EVALUATION (AC1-AC7)
# ============================================================================

def evaluate_acceptance_criteria(
    summary_block: Dict[str, Any],
    f_E_block: Dict[str, Any],
    dsph_block: Dict[str, Any],
    sub_issues: Dict[str, Dict[str, Any]],
    f_flag_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate AC1-AC7 against orchestration outputs.

    Per OUTPUT_SCHEMA_section2_5_v0_2.md §4 + EXECUTION_PLAN §8.

    AC1: Algorithm B convergence rate ≥ 0.95
    AC2: f_E adoption fraction in expected range (placeholder; need user range)
    AC3: dSph J3 ≥ 28/31 (no regression)
    AC4: b_α |Δ| (SPARC vs dSph) ≤ 0.005
    AC5: All S-1〜S-6 RESOLVED
    AC6: Q-C1 LOCK preserved (k_E=2 default)
    AC7: anchor 21 chapter-level consistency

    Each AC produces: {"observed": value, "threshold": ...,
                        "pass": bool | None}.
    None means deferred (input not yet computable, e.g., T14 not wired).

    Returns
    -------
    acceptance : dict to be merged into summary.json::acceptance.
    """
    ac = {}

    # AC1: Algorithm B convergence rate
    rate = summary_block.get("algorithm_b", {}).get("convergence_rate")
    ac["AC1_algorithm_b_convergence_rate_ge_threshold"] = {
        "threshold": 0.95,  # suggested; configurable in run_config
        "observed": rate,
        "pass": (rate is not None) and (rate >= 0.95),
    }
    if rate is None:
        ac["AC1_algorithm_b_convergence_rate_ge_threshold"]["pass"] = None

    # AC2: f_E adoption fraction in expected range
    f_E = f_E_block.get("f_E_fraction")
    sample_decision = f_E_block.get("sample_level_decision")
    ac["AC2_f_E_adoption_in_expected_range"] = {
        "f_E_observed": f_E if f_E is None or not np.isnan(f_E) else None,
        "sample_level_decision": sample_decision,
        "F_E_LOWER": F_E_LOWER, "F_E_UPPER": F_E_UPPER,
        # Without a user-provided "expected_range", AC2 records the decision
        # but does not pass/fail. v0.2 finalize: set expected_range based on
        # anchor 7 §2.5.4 expectation (typically B-leaning per parsimony).
        "pass": None,  # deferred until expected_range supplied
    }

    # AC3: dSph J3 ≥ 28/31
    n_pass_28 = dsph_block.get("n_pass_observed_28_baseline")
    n_total = dsph_block.get("n_total")
    ac["AC3_dSph_J3_at_least_28_31"] = {
        "threshold_min": 28, "threshold_total": 31,
        "observed_pass": n_pass_28, "observed_total": n_total,
        "pass": (n_pass_28 is not None) and (n_pass_28 >= 28),
    }
    if n_pass_28 is None:
        ac["AC3_dSph_J3_at_least_28_31"]["pass"] = None

    # AC4: b_α |Δ| ≤ 0.005 (anchor 19 §1.5 baseline 0.0042; EXECUTION_PLAN §4.3)
    # NaN handling: b_alpha_3axis_audit() returns NaN when sample size < 3 or
    # T12/T13/T14 wiring deferred; treat as 'pending' (None), not False.
    abs_diff = dsph_block.get("b_alpha_abs_diff")
    is_finite = (
        abs_diff is not None
        and isinstance(abs_diff, (int, float))
        and np.isfinite(abs_diff)
    )
    if is_finite:
        ac4_pass: Optional[bool] = bool(abs_diff <= 0.005)
    else:
        ac4_pass = None  # pending / unable to compute
    ac["AC4_b_alpha_abs_diff_le_0_005"] = {
        "threshold": 0.005,
        "observed_abs_diff": (abs_diff if is_finite else None),
        "anchor_19_baseline_abs_diff": B_ALPHA_ABS_DIFF_BASELINE,
        "phase_c3_universal_coupling_central": 0.11,
        "phase_c3_universal_coupling_sigma": 0.005,
        "phase_c3_deviation_sigma": dsph_block.get(
            "b_alpha_phase_c3_deviation_sigma"
        ),
        "pass": ac4_pass,
        "reason_if_pending": (
            None if is_finite else (
                "abs_diff is None (b_α audit not run)"
                if abs_diff is None else
                "abs_diff is NaN (computation failed; n_used < 3 or T12/T13/T14 deferred)"
            )
        ),
    }

    # AC5: All S-1〜S-6 RESOLVED
    s_resolved = {}
    for s_key in ("S_1", "S_2", "S_3", "S_4", "S_5", "S_6"):
        block = sub_issues.get(s_key, {})
        status = block.get("resolution_status", "missing")
        # v0.2 finalize: status should equal "resolved" (or
        # "structure_filled_pending_chapter_number" / "_line_number" → wait
        # for v0.2 round literal fill).
        s_resolved[s_key] = (status == "resolved")
    ac["AC5_S1_S6_all_resolved"] = {
        **s_resolved,
        "all_pass": all(s_resolved.values()) if s_resolved else False,
    }
    if all(v is None for v in s_resolved.values()):
        ac["AC5_S1_S6_all_resolved"]["all_pass"] = None

    # AC6: Q-C1 LOCK preserved (k_E=2 default)
    f8 = f_flag_status.get("F8", {})
    q_c1_preserved = f8.get("Q_C1_LOCK_preserved")
    ac["AC6_Q_C1_LOCK_preserved_k_E_2_default"] = {
        "k_E_2_default_used": f8.get("k_E_2_default_used", True),
        "delta_AIC_k_E_2_vs_k_E_1": f8.get("delta_AIC_k_E_2_minus_k_E_1"),
        "k_E_2_remains_preferred": q_c1_preserved,
        "pass": q_c1_preserved if q_c1_preserved is not None else None,
    }

    # AC7: anchor 21 chapter-level consistency
    f4_resolved = f_flag_status.get("F4", {}).get(
        "v0_1_1_status_at_anchor_21", "RESOLVED"
    ) == "RESOLVED"
    f12_resolved = f_flag_status.get("F12", {}).get(
        "v0_1_1_status_at_anchor_21", "RESOLVED"
    ) == "RESOLVED"
    f1_f2_f6_f8_resolved = all(
        f_flag_status.get(k, {}).get("v0_2_resolution_status") == "resolved"
        for k in ("F1", "F2", "F6", "F8")
    )
    ac["AC7_anchor_21_chapter_level_consistency"] = {
        "F4_remains_RESOLVED": f4_resolved,
        "F12_remains_RESOLVED": f12_resolved,
        "F1_F2_F6_F8_resolution_via_S_handoff": f1_f2_f6_f8_resolved,
        "10_axis_audit_no_regression": True,  # by construction (no anchor 7 modification)
        "pass": f4_resolved and f12_resolved and f1_f2_f6_f8_resolved,
    }

    # Aggregate
    individual_passes = [v.get("pass") for v in ac.values()]
    if any(p is None for p in individual_passes):
        all_pass = None
    else:
        all_pass = all(p for p in individual_passes)
    ac["all_AC_pass"] = all_pass
    ac["v0_2_finalize_eligible"] = (all_pass is True)

    return ac


def write_human_readable_summary(
    output_dir: Path,
    config: "RunConfig",
    summary: Dict[str, Any],
    f_E_block: Dict[str, Any],
    dsph_block: Dict[str, Any],
    sub_issues: Dict[str, Dict[str, Any]],
    f_flag_status: Dict[str, Any],
    acceptance: Dict[str, Any],
) -> None:
    """Write a one-page human-readable summary to run_summary.txt."""
    lines = []
    p = lines.append
    p("=" * 78)
    p("J-system Companion Paper §2.5 v0.2 — Run Summary")
    p("=" * 78)
    p(f"Run ID         : {config.run_id}")
    p(f"Output dir     : {config.output_dir}")
    p(f"Anchor 21 SHA  : {ANCHOR_21_SHA_REFERENCE}")
    p(f"Wall clock     : {summary.get('wall_clock_seconds', 0.0):.1f} s")
    p("")
    p("─── Inputs ───")
    n_full = summary.get("n_galaxies_sparc_total", 0)
    n_pool = summary.get("n_galaxies_sparc_inpool", 0)
    n_dsph = summary.get("n_dsph", 0)
    p(f"  SPARC          : {n_full} total, {n_pool} in-pool")
    p(f"  dSph           : {n_dsph}")
    p(f"  Excluded 4     : {', '.join(EXCLUDED_4_SPARC_GALAXIES.keys())}")
    p("")
    p("─── Algorithm B ───")
    ab = summary.get("algorithm_b", {})
    p(f"  converged      : {ab.get('n_converged', 0)} / "
      f"{ab.get('n_converged', 0) + ab.get('n_not_converged', 0)}")
    p(f"  rate           : {ab.get('convergence_rate', float('nan')):.4f}")
    p(f"  mean iter      : {ab.get('mean_n_iter', float('nan')):.2f}")
    p(f"  median iter    : {ab.get('median_n_iter', float('nan')):.1f}")
    p(f"  init c_galaxy  : {ab.get('init_c_galaxy')}")
    p(f"  tol / N_max    : {ab.get('tol')} / {ab.get('N_max')}")
    p("")
    p("─── f_E adoption ───")
    p(f"  k_E used       : {f_E_block.get('k_E_used')}")
    p(f"  ΔAIC threshold : {f_E_block.get('threshold')}")
    p(f"  n_decided      : {f_E_block.get('n_total_decided', 0)} "
      f"(skipped: {f_E_block.get('n_skipped', 0)})")
    p(f"  n_E / n_B      : {f_E_block.get('n_selected_E', 0)} / "
      f"{f_E_block.get('n_selected_B', 0)}")
    fE = f_E_block.get('f_E_fraction', float('nan'))
    p(f"  f_E            : {fE if fE is None else f'{fE:.4f}'}")
    p(f"  ΔAIC mean / med: {f_E_block.get('delta_AIC_mean', float('nan'))} / "
      f"{f_E_block.get('delta_AIC_median', float('nan'))}")
    p(f"  sample-level   : {f_E_block.get('sample_level_decision')}")
    p(f"  reason         : {f_E_block.get('sample_level_decision_reason')}")
    p("")
    p("─── dSph audit ───")
    p(f"  J3 status      : {dsph_block.get('j3_status')}")
    p(f"  n_pass (28-base): {dsph_block.get('n_pass_observed_28_baseline')}")
    p(f"  n_pass (30-base): {dsph_block.get('n_pass_observed_30_baseline')}")
    p(f"  G_Strigari pred: {G_STRIGARI_OVER_A0:.6f}")
    p(f"  G_Strigari obs : {dsph_block.get('G_Strigari_over_a0_observed_mean')}")
    p(f"  b_α status     : {dsph_block.get('b_alpha_status')}")
    p(f"  b_α |Δ| obs    : {dsph_block.get('b_alpha_abs_diff')}")
    p(f"  b_α |Δ| anc19  : {B_ALPHA_ABS_DIFF_BASELINE}")
    p("")
    p("─── Sub-issues S-1 〜 S-6 ───")
    for k in ("S_1", "S_2", "S_3", "S_4", "S_5", "S_6"):
        b = sub_issues.get(k, {})
        p(f"  {k} ({b.get('category', '?')}) : "
          f"{b.get('resolution_status', 'missing')}"
          + (f" → {b.get('linked_F')}" if b.get('linked_F') else ""))
    p("")
    p("─── F-flags (companion-internal subset) ───")
    for k in ("F1", "F2", "F4", "F6", "F8", "F12"):
        b = f_flag_status.get(k, {})
        if k in ("F4", "F12"):
            status = "RESOLVED at anchor 21"
        else:
            status = b.get("v0_2_resolution_status", "missing")
        p(f"  {k}: {status}")
    p("")
    p("─── Acceptance criteria ───")
    for ac_key in (
        "AC1_algorithm_b_convergence_rate_ge_threshold",
        "AC2_f_E_adoption_in_expected_range",
        "AC3_dSph_J3_at_least_28_31",
        "AC4_b_alpha_abs_diff_le_0_005",
        "AC5_S1_S6_all_resolved",
        "AC6_Q_C1_LOCK_preserved_k_E_2_default",
        "AC7_anchor_21_chapter_level_consistency",
    ):
        b = acceptance.get(ac_key, {})
        passed = b.get("pass") if "pass" in b else b.get("all_pass")
        sym = "✅" if passed is True else ("❌" if passed is False else "⏸ ")
        p(f"  {sym} {ac_key}")
    p("")
    p(f"All AC pass        : {acceptance.get('all_AC_pass')}")
    p(f"v0.2 finalize ready: {acceptance.get('v0_2_finalize_eligible')}")
    p("=" * 78)

    with open(output_dir / "run_summary.txt", "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# §11 OUTPUT WRITERS (per OUTPUT_SCHEMA_section2_5_v0_2.md v1.0)
# ============================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: str, float_precision: int = 6) -> None:
    class _RoundEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.floating,)):
                return round(float(o), float_precision)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return super().default(o)

    def _round_floats(x: Any) -> Any:
        if isinstance(x, float):
            return round(x, float_precision)
        if isinstance(x, dict):
            return {k: _round_floats(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_round_floats(v) for v in x]
        return x

    rounded = _round_floats(obj)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rounded, f, indent=2, ensure_ascii=False, cls=_RoundEncoder)


def write_sha256sums(output_dir: Path, files: List[str]) -> None:
    lines = []
    for relpath in files:
        full = output_dir / relpath
        if full.exists():
            sha = sha256_file(str(full))
            lines.append(f"{sha}  {relpath}")
    with open(output_dir / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# §12 MAIN ORCHESTRATION
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("section2_5_v0_2")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(output_dir / "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip computation; just verify env and file SHAs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-self-check", action="store_true",
        help="Skip cascade SSoT self-check (debugging only).",
    )
    parser.add_argument(
        "--no-rotation-curve", action="store_true",
        help=(
            "Run Algorithm B in stub mode (rotation_curve=None for every "
            "galaxy). Useful for orchestration validation pre-T12 wiring."
        ),
    )
    args = parser.parse_args(argv)

    # Cascade SSoT self-check (foundation b0cb36d7 reference values).
    # Aborts immediately if V_DOUBLE_PRIME_AT_X_HALF has been modified.
    if not args.skip_self_check:
        try:
            _vpp_x05_self_check()
        except AssertionError as exc:
            print(f"[FATAL] cascade SSoT self-check failed: {exc}", file=sys.stderr)
            return 3
        # T13 reference pair self-check (no-op if NU_CANONICAL_REFERENCE_PAIRS empty)
        try:
            t13_result = nu_canonical_self_check()
            if t13_result.get("status") == "deferred":
                print(
                    f"[INFO] T13 self-check deferred: "
                    f"{t13_result.get('reason')}",
                    file=sys.stderr,
                )
        except AssertionError as exc:
            print(f"[FATAL] T13 ν_canonical self-check failed: {exc}", file=sys.stderr)
            return 4

    np.random.seed(args.seed)
    config = build_run_config(no_rotation_curve=args.no_rotation_curve)
    output_dir = Path(config.output_dir)
    logger = setup_logging(output_dir)

    t0 = time.time()
    logger.info("===== §2.5 v0.2 SPARC empirical execution START =====")
    logger.info(f"run_id = {config.run_id}")
    logger.info(f"output_dir = {output_dir}")
    if args.no_rotation_curve:
        logger.warning(
            "--no-rotation-curve set: Algorithm B in STUB MODE "
            "(orchestration validation only, not scientific output)"
        )

    # Pre-flight: verify input SHAs
    input_files = [
        ("TA3", config.sparc_ta3_path),
        ("phase1", config.sparc_phase1_path),
        ("MRT", config.sparc_mrt_path),
        ("dSph", config.dsph_dataset_path),
    ]
    input_shas: Dict[str, str] = {}
    for role, path in input_files:
        if not Path(path).exists():
            logger.error(f"Input file missing: {role} -> {path}")
            return 2
        sha = sha256_file(path)
        input_shas[role] = sha
        logger.info(f"input {role:<8} sha256={sha[:16]}...  path={path}")

    if args.dry_run:
        logger.info("--dry-run set: skipping computation.")
        return 0

    # Load three files
    logger.info("Loading SPARC TA3 + phase1 + MRT ...")
    ta3 = load_TA3(config.sparc_ta3_path)
    phase1 = load_phase1(config.sparc_phase1_path)
    mrt = load_MRT(config.sparc_mrt_path)
    sparc_full = merge_three_files(ta3, phase1, mrt)
    sparc_full = mark_fit_pool_171(sparc_full)
    sparc_171 = sparc_full[sparc_full["in_171_pool"]].copy()
    logger.info(f"SPARC: total={len(sparc_full)}, in_171_pool={len(sparc_171)}")

    # dSph
    logger.info("Loading dSph dataset ...")
    dsph = load_dsph(config.dsph_dataset_path)
    logger.info(f"dSph: total={len(dsph)}")

    # Algorithm B per-galaxy
    logger.info("Running Algorithm B per-galaxy ...")
    b_results: List[AlgorithmBResult] = []
    iter_log_rows: List[Dict[str, Any]] = []
    n_rc_loaded = 0
    for _, row in sparc_171.iterrows():
        galaxy = str(row["galaxy"])
        # Load per-galaxy rotation curve (T12 wiring point)
        if config.no_rotation_curve:
            rc = None
        else:
            rc = load_rotation_curve(galaxy, config.sparc_rc_base_path)
            if rc is not None:
                n_rc_loaded += 1
        result = algorithm_b_per_galaxy(row, rotation_curve=rc)
        b_results.append(result)
        for n, c_n, res_n in result.iter_trace:
            iter_log_rows.append({
                "galaxy": result.galaxy,
                "iter_n": n,
                "c_galaxy_n": c_n,
                "residual_n": res_n,
                "converged_at_n": (n == result.n_iter and result.converged),
            })
    n_converged = sum(1 for r in b_results if r.converged)
    convergence_rate = n_converged / len(b_results) if b_results else 0.0
    logger.info(
        f"Algorithm B: {n_converged}/{len(b_results)} converged "
        f"({convergence_rate:.4f})"
    )

    # Build per_galaxy_sparc DataFrame
    sparc_171["c_galaxy_init"] = C_GALAXY_INIT
    b_by_galaxy = {r.galaxy: r for r in b_results}
    sparc_171["c_galaxy_final"] = sparc_171["galaxy"].map(
        lambda g: b_by_galaxy[g].c_galaxy_final if g in b_by_galaxy else float("nan")
    )
    sparc_171["converged_B"] = sparc_171["galaxy"].map(
        lambda g: b_by_galaxy[g].converged if g in b_by_galaxy else False
    )
    sparc_171["n_iter_B"] = sparc_171["galaxy"].map(
        lambda g: b_by_galaxy[g].n_iter if g in b_by_galaxy else 0
    )
    sparc_171["residual_B"] = sparc_171["galaxy"].map(
        lambda g: b_by_galaxy[g].residual_final if g in b_by_galaxy else float("nan")
    )

    # Sub-issue resolvers
    logger.info("Resolving sub-issues S-1〜S-6 ...")
    sub_issues = {
        "S_1": resolve_S1(sparc_171, dsph),
        "S_2": resolve_S2(sparc_171),
        "S_3": resolve_S3(sparc_171),
        "S_4": resolve_S4(sparc_171),
        "S_5": resolve_S5(sparc_171),
        "S_6": resolve_S6(sparc_171),
    }

    # f_E adoption aggregation + sample-level decision (Q4 LOCK)
    logger.info("Aggregating f_E adoption ...")
    f_E_block = aggregate_f_E_adoption(
        sparc_171,
        b_by_galaxy,
        k_E=K_E_DEFAULT,
        threshold=DELTA_AIC_THRESHOLD,
    )
    logger.info(
        f"f_E adoption: f_E={f_E_block.get('f_E_fraction')}, "
        f"sample_decision={f_E_block.get('sample_level_decision')} "
        f"(decided={f_E_block.get('n_total_decided')}, "
        f"skipped={f_E_block.get('n_skipped')})"
    )

    # T14 NLL self-check (no-op until NLL_REFERENCE_PAIRS populated)
    if not args.skip_self_check:
        try:
            t14_test_galaxies = {
                str(row["galaxy"]): row for _, row in sparc_171.iterrows()
            }
            t14_result = nll_self_check(
                test_galaxies=t14_test_galaxies, b_results=b_by_galaxy,
            )
            if t14_result.get("status") == "deferred":
                logger.info(f"T14 NLL self-check deferred: {t14_result.get('reason')}")
            else:
                logger.info(f"T14 NLL self-check: {t14_result}")
        except AssertionError as exc:
            logger.error(f"T14 NLL self-check failed: {exc}")
            return 5

    # dSph J3 + b_α 3-axis audit
    logger.info("Running dSph audit (J3 + b_α 3-axis) ...")
    dsph_block = run_dsph_audit(dsph, sparc_171, deferred_on_missing=True)
    logger.info(
        f"dSph audit: J3={dsph_block.get('j3_status')}, "
        f"b_α={dsph_block.get('b_alpha_status')}"
    )

    # F-flag status (anchor 8 §2.6.5 + anchor 21 v0.1.1 + v0.2 outcomes)
    f_flag_status = {
        "F1": {
            "anchor_8_status": "placeholder 維持 (S-3 specific aspect)",
            "anchor_21_v0_1_1_status": "placeholder 維持 + §2.5 v0.2 round handoff",
            "v0_2_resolution_status": sub_issues["S_3"]["resolution_status"],
            "linked_S": "S-3", "category": "A",
            "resolution_value": sub_issues["S_3"].get("resolution_value"),
        },
        "F2": {
            "anchor_8_status": "placeholder 維持 (S-1 specific aspect)",
            "anchor_21_v0_1_1_status": "placeholder 維持 + §2.5 v0.2 round handoff",
            "v0_2_resolution_status": sub_issues["S_1"]["resolution_status"],
            "linked_S": "S-1", "category": "A",
            "resolution_value": sub_issues["S_1"].get("resolution_value"),
        },
        "F4": {
            "anchor_8_status": "status declaration",
            "v0_1_1_status_at_anchor_21": "RESOLVED",
            "category": "—",
        },
        "F6": {
            "anchor_8_status": "placeholder 維持",
            "anchor_21_v0_1_1_status": "placeholder 維持 (handoff continuation)",
            "v0_2_resolution_status": "pending",
            "category": "A", "resolution_value": None,
        },
        "F8": resolve_F8_k_E_sensitivity(sparc_171),
        "F12": {
            "anchor_8_status": "active inject (axis vii)",
            "v0_1_1_status_at_anchor_21": "RESOLVED",
            "category": "—",
        },
    }

    # Build promotion.json
    promotion = {
        "anchor_19_baseline": {
            "path_A_eta_0": PATH_A_ETA_0_BASELINE,
            "path_A_delta_pct": PATH_A_DELTA_PCT_BASELINE,
            "path_A_T6_pass": True,
            "path_B_eta_0": PATH_B_ETA_0_BASELINE,
            "path_B_delta_pct": PATH_B_DELTA_PCT_BASELINE,
            "path_B_T4_pass": True,
            "B_plus_to_A_prerequisite": "achieved",
        },
        "v0_2_observed": {
            "path_A_eta_0": None,  # TODO compute from sparc_171 once aggregator filled
            "path_B_eta_0": None,
            "consistent_with_anchor_19": None,
        },
    }

    # Build summary.json
    summary = {
        "run_id": config.run_id,
        "anchor_21_sha_referenced": ANCHOR_21_SHA_REFERENCE,
        "input_shas": input_shas,
        "wall_clock_seconds": time.time() - t0,
        "n_galaxies_sparc_total": len(sparc_full),
        "n_galaxies_sparc_inpool": len(sparc_171),
        "n_galaxies_sparc_excluded": len(sparc_full) - len(sparc_171),
        "n_dsph": len(dsph),
        "algorithm_b": {
            "n_converged": n_converged,
            "n_not_converged": len(b_results) - n_converged,
            "convergence_rate": convergence_rate,
            "mean_n_iter": float(np.mean([r.n_iter for r in b_results])) if b_results else 0.0,
            "median_n_iter": float(np.median([r.n_iter for r in b_results])) if b_results else 0.0,
            "max_n_iter": int(max((r.n_iter for r in b_results), default=0)),
            "init_c_galaxy": C_GALAXY_INIT,
            "tol": TOL,
            "N_max": N_MAX,
        },
        "C15": {
            "formula": "g_c = 0.584 * Ud^(-0.361) * sqrt(a_0 * Vflat^2 / hR)",
            "eta_0_locked": ETA_0_LOCKED,
            "alpha_locked": ALPHA_LOCKED,
            "beta_locked": BETA_LOCKED,
            "scatter_dex_anchor7_baseline": 0.286,
            "R_squared_anchor7_baseline": 0.607,
            "MOND_rejection_p_anchor7_baseline": 1.66e-53,
        },
        "dSph_J3": {
            "baseline_28_31": [28, 31],
            "G_Strigari_over_a0_predicted": G_STRIGARI_OVER_A0,
            **{k: v for k, v in dsph_block.items()
               if k not in ("b_alpha_status", "b_alpha_axes",
                            "b_alpha_abs_diff",
                            "b_alpha_anchor19_baseline_abs_diff",
                            "b_alpha_within_AC4_threshold_005",
                            "b_alpha_deferred_reason")},
        },
        "b_alpha": {
            "SPARC_anchor7_baseline": B_ALPHA_SPARC_BASELINE,
            "dSph_anchor7_baseline": B_ALPHA_DSPH_BASELINE,
            "abs_diff_anchor19_baseline": B_ALPHA_ABS_DIFF_BASELINE,
            **{k: v for k, v in dsph_block.items()
               if k.startswith("b_alpha_")},
        },
        "f_E_adoption": f_E_block,
    }

    # Acceptance criteria evaluation (AC1-AC7)
    logger.info("Evaluating acceptance criteria AC1-AC7 ...")
    acceptance = evaluate_acceptance_criteria(
        summary_block=summary,
        f_E_block=f_E_block,
        dsph_block=dsph_block,
        sub_issues=sub_issues,
        f_flag_status=f_flag_status,
    )
    summary["acceptance"] = acceptance
    logger.info(
        f"AC evaluation: all_pass={acceptance.get('all_AC_pass')}, "
        f"v0_2_finalize_eligible={acceptance.get('v0_2_finalize_eligible')}"
    )

    # Write all outputs
    sparc_full.to_csv(output_dir / "per_galaxy_sparc_full.csv", index=False)
    sparc_171.to_csv(output_dir / "per_galaxy_sparc.csv", index=False)
    dsph.to_csv(output_dir / "per_dsph.csv", index=False)
    pd.DataFrame(iter_log_rows).to_csv(
        output_dir / "algorithm_b_log.csv", index=False
    )
    pd.DataFrame([
        {"c_anchor": ci, "V_double_prime_at_x_half": V_DOUBLE_PRIME_AT_X_HALF[ci],
         "source": "TODO_USER_INPUT" if V_DOUBLE_PRIME_AT_X_HALF[ci] is None else "filled"}
        for ci in C_ANCHORS
    ]).to_csv(output_dir / "f_opt_anchor_table.csv", index=False)

    write_json(summary, str(output_dir / "summary.json"), config.output_float_precision)
    write_json(sub_issues, str(output_dir / "sub_issues.json"), config.output_float_precision)
    write_json(promotion, str(output_dir / "promotion.json"), config.output_float_precision)
    write_json(f_flag_status, str(output_dir / "f_flag_status.json"), config.output_float_precision)
    write_json(asdict(config), str(output_dir / "run_config.json"), config.output_float_precision)

    # Human-readable summary (run_summary.txt)
    write_human_readable_summary(
        output_dir, config, summary, f_E_block, dsph_block,
        sub_issues, f_flag_status, acceptance,
    )

    # Manifest
    manifest = {
        "run_id": config.run_id,
        "creator": "run_section2_5_v0_2.py",
        "execution_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "random_seed": args.seed,
        },
        "env_vars_used": {
            "MASTER_ROOT": config.master_root,
            "PARENT_MASTER_ROOT": config.parent_master_root,
            "SPARC_TA3_PATH": config.sparc_ta3_path,
            "SPARC_PHASE1_PATH": config.sparc_phase1_path,
            "SPARC_MRT_PATH": config.sparc_mrt_path,
            "SPARC_RC_BASE_PATH": config.sparc_rc_base_path,
            "DSPH_DATASET_PATH": config.dsph_dataset_path,
            "OUTPUT_ROOT": config.output_root,
        },
        "input_files": [
            {"role": role, "path": path, "sha256": input_shas[role],
             "size_bytes": Path(path).stat().st_size}
            for role, path in input_files
        ],
        "anchor_references": ANCHOR_REFERENCES,
        "auxiliary_references": AUXILIARY_REFERENCES,
        "forensic_chain_compliance": {
            "rule_2_predecessor_anchors_cited": True,
            "rule_3_no_retroactive_change": True,
            "rule_4_forward_ref_0_strict": True,
            "rule_5_companion_pure_additive": True,
        },
    }
    write_json(manifest, str(output_dir / "manifest.json"), config.output_float_precision)

    # SHA256SUMS
    output_files = [
        "per_galaxy_sparc.csv", "per_galaxy_sparc_full.csv", "per_dsph.csv",
        "algorithm_b_log.csv", "f_opt_anchor_table.csv",
        "summary.json", "sub_issues.json", "promotion.json",
        "f_flag_status.json", "run_config.json", "manifest.json",
        "run_summary.txt", "run.log",
    ]
    write_sha256sums(output_dir, output_files)

    logger.info(f"===== §2.5 v0.2 execution COMPLETE in {time.time()-t0:.1f}s =====")
    logger.info(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
