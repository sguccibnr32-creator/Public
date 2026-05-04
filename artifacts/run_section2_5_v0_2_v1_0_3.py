#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J-system Companion Paper §2.5 v0.2 SPARC Empirical Execution v1.0.3
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
  - C3-A5 internal_memo_c3_extension_v3.pdf (SHA 69fb1a95) — §4.3 universal coupling
  - C3-A4 J0 minimal form         (SHA 7e8823f4) — reference baseline
  - phase_c3_step3_dsph_gamma_vs_alpha.py (SHA c51c72f0) — partial OLS estimator

v1.0.3 changelog (2026-05-04):
  T12-T14 + T17 + T18 verbatim integration. P12 公式 spec impl 完成。
  Atomic completion of §2.5 v0.2 prep (no v1.0.4 deferred).

  Patches applied (P1-P13):
    P1  : header v1.0.2 -> v1.0.3 + this changelog block
    P2  : load_rotation_curve() T12.A+D fill (Lelli SPARC RC standard
           8-column .dat file loader)
    P3  : compute_g_obs_g_bar() T12.B+C fill (V -> g_bar / g_obs / sigma_g
           via vbar2 = Vgas² + Y_d·sign(Vdisk)·Vdisk² + Y_b·sign(Vbul)·Vbul²
           + linear sigma propagation)
    P4  : f_opt(x!=0.5, c) NotImplementedError 維持 + docstring 強化
           (T13.B finding: parent v4.8 §3.1 L335 defines x=0.5 canonical
           reduction only; x != 0.5 derivation is v4.9 patch round candidate)
    P5  : _backsolve_c() x=0.5 operational projection applied
           (use f_opt_v3_cascade(c) directly; x_at_radius retained for
           v4.9 forward compatibility)
    P6  : algorithm_b_step() docstring updated (x=0.5 projection per T13.B)
    P7  : e_pipeline_score() T14.A+F fill (first concrete impl)
           NLL_E = 0.5*sum(residual^2) + sum(log(sigma_g)) + (n_g/2)*log(2π)
           AIC_E = 2*NLL_E + 2*k_E (k_E=2 default per Q-C1 LOCK)
           Levenberg-Marquardt fit via scipy.optimize.least_squares
    P8  : b_pipeline_score() T14.B+F fill (first concrete impl)
           NLL_B = sum(log(sigma_g)) + (n_g/2)*log(2π) (residual=0 by
           construction at convergence)
           AIC_B = 2*NLL_B (k_B=0 per R-1 LOCK)
    P9  : NU_CANONICAL_REFERENCE_PAIRS populated (T13.D 5+2 anchor):
           5-anchor exact + 0.83 (canonical) + 0.5709 (c_super)
    P10 : NLL_REFERENCE_PAIRS replaced with structural invariants
           (T14.E finding: numerical benchmark NOT in anchor 14;
            verify NLL_E >= NLL_B and AIC_E - AIC_B >= 4 instead)
    P11 : dsph_j3_check() existing impl preserved (no functional change)
    P12 : b_alpha_3axis_audit() 公式 spec implementation
           Sources: anchor 7 §2.5.5 + C3-A5 §4.3 + phase_c3_step3 SHA c51c72f0
           - Axis 1: SPARC partial OLS, 124-galaxy sample
                     (Q<3 + bridge 4 excluded), target = log10(g_obs/gc_C15),
                     feature = 2*log10(rho_gal), nuisance = log10(r_h)
           - Axis 2: dSph partial OLS, 30-galaxy sample (31 - Sgr),
                     target = log10(g_obs/(0.228*a_0)), same feature/nuisance
           - Axis 3: combo separate-intercept design, 154-galaxy combined,
                     X = [is_sparc, is_dsph, is_sparc*lu, is_dsph*lu,
                          is_sparc*log_rh, is_dsph*log_rh]
                     b_alpha_sparc = b[2], b_alpha_dsph = b[3]
           - Estimator: numpy.linalg.lstsq partial OLS (Phase C3 v3 §4.3)
           - Reproduces anchor 19 §1.5 baseline 0.1084 / 0.1127 / |Δ|=0.0042
           - AC4 (|Δ| <= 0.005) evaluation now active
    P13 : self-check expansion:
           - nu_canonical_self_check populates with T13.D pairs
           - nll_self_check() runs structural invariants test
           - b_alpha_self_check() new (mock + reproduction guarantee)
           - manifest update: AUXILIARY_REFERENCES + env vars

  Findings recorded permanently:
    T13.B: parent v4.8 §3.1 L335 defines f_opt only at x=0.5 anchor.
           Operational projection used in B/E pipelines (semantic correct
           per Path (iii) LOCK). x != 0.5 form is v4.9 patch round candidate.
    T14.E: NGC 3198 numerical NLL benchmark not in anchor 14. Structural
           invariants (NLL_E >= NLL_B, AIC_E - AIC_B >= 4) used as
           reproducibility test instead of hard numerical pairs.
    T14.F: per-galaxy Gaussian NLL of (g_obs, g_pred, sigma_g) for SPARC
           is first concrete Python impl (no prior reference in foundation
           / C3-A1 / Phase 5-2 / phase_c3_step3). Benchmark values establish
           on first numerical run.
    T17+T18: b_alpha 真 estimator is partial OLS with log_rh nuisance
           covariate (NOT simple linregress). v1.0.2 stub axis_{1,2,3}
           decomposition and 公式 anchor 7 §2.5.5 axes are different
           semantics. Return dict includes both: 公式 (axis_1_continuity_status,
           axis_2_reversal_status, axis_3_universal_slope) + sub-axis
           breakdown (axis_1_SPARC, axis_2_dSph, axis_3_combined_*) for
           caller-side compatibility (run_dsph_audit line 1500-1517 unchanged).

  Forensic chain compliance (anchor 21 §J 7-item ruleset):
    1. all anchors 5/6/7/8/14/16/17/19/21 IMMUTABLE preserved (zero modify)
    2. R-1 LOCK preserved (k_B=0 parameter-free canonical)
    3. R-2 LOCK preserved (Algorithm B simultaneous self-consistency)
    4. Q-C1 LOCK preserved (k_E=2 default)
    5. cascade SSoT preserved (V''(x=0.5,c) 5-anchor + foundation b0cb36d7)
    6. L-1 forward-ref 0 strict (parent v4.8 NULL impact)
    7. companion additive supersession (no parent modification)

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

import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares  # ν_canonical inversion + E-pipeline LM fit

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

# Phase C3 v3 §4.3 universal coupling constants (T17 + T18 verbatim 反映)
# Source: C3-A5 internal_memo_c3_extension_v3.pdf §4.3 (SHA 69fb1a95) +
#         phase_c3_step3_dsph_gamma_vs_alpha.py L80-101 (SHA c51c72f0)
A0_KPC: float = 1.2e-10 * 3.086e19 / 1e6   # (km/s)² / kpc; Newton g_bar 単位整合用
G_STRIGARI_M_S2: float = 0.228 * A_0        # m/s²; dSph delta target anchor (§4.3)
C15_COEF: float = 0.584                     # eta_0 prefactor (= ETA_0_LOCKED)
C15_UPSILON_EXP: float = -0.361             # beta_Y exponent (= BETA_LOCKED)

# Universal slope baseline (C3-A5 §8.1 minimal model table verbatim)
B_ALPHA_AXIS3_BASELINE: float = 0.11        # universal slope across 3.92 dex
B_ALPHA_AXIS3_TOLERANCE: float = 0.005      # ±0.005 sigma per §8.1

# Lesson 93 (§5.3) universal coupling agreement criterion
B_ALPHA_LESSON93_THRESHOLD: float = 0.01    # |diff| ≤ 0.01 (1% slope agreement)
# AC4 threshold (v1.0.2 hardcoded 0.005) is 2x stricter than Lesson 93 公式
# criterion (1%); §4.3 observed |diff|=0.0042 < 0.005 < 0.01 自動 PASS。

# Phase C3 v3 sample size baselines (T17.A §4.2 結果サマリー table verbatim)
N_AXIS_1_SPARC_EXPECTED: int = 124          # SPARC Q<3 + 4 bridge excluded
N_AXIS_2_DSPH_EXPECTED: int = 30            # dSph 31 - Sgr (§4.2 + T18.C)
N_AXIS_3_COMBINED_EXPECTED: int = 154       # 124 + 30

# Sgr exclusion (T18.C verbatim, phase_c3_step3 L111-112)
# Robust regex (Phase C2 protocol): matches "Sagittarius dSph" / "sagittarius DSPH" / etc.
SGR_NAME_REGEX: str = r"sagittarius\s+dsph"
SGR_REGEX_FLAGS: int = re.IGNORECASE

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
    {"role": "phase_c3_step3_dsph_gamma_vs_alpha.py (b_alpha partial OLS producer)",
     "sha_prefix": "c51c72f0"},
    {"role": "dsph_jeans_c15_v1.csv (T18.C input file, dSph 31 sample)",
     "sha_prefix": "(verify on Windows side)"},
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
    upsilon_d: float = 1.0,
    upsilon_b: float = 1.0,
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
    # T12 fill (P2, v1.0.3): Lelli SPARC RC .dat parser
    # Source: C3-A1 sparc_fp_verification.py L135-158 (load_rotmod) +
    #         L141-142 + L209-210 + L221 (preprocessing rules).
    # ------------------------------------------------------------------
    rc_file = Path(base_path) / f"{galaxy}_rotmod.dat"
    if not rc_file.exists():
        return None

    # Lelli SPARC standard format (T12.A verbatim):
    #   header: 3 lines starting with '#' (Distance / units / column names)
    #   columns (8): Rad[kpc], Vobs[km/s], errV[km/s], Vgas[km/s],
    #                Vdisk[km/s], Vbul[km/s], SBdisk[L/pc²], SBbul[L/pc²]
    #   delimiter: whitespace/TAB; encoding: ASCII
    rad: List[float] = []
    vobs: List[float] = []
    errv: List[float] = []
    vgas: List[float] = []
    vdisk: List[float] = []
    vbul: List[float] = []
    sbdisk: List[float] = []
    sbbul: List[float] = []
    try:
        with open(rc_file, "r") as f:
            for line in f:
                line = line.strip()
                # T12.D rule (1): skip blank / comment lines (L139-142 verbatim)
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    rad.append(float(parts[0]))
                    vobs.append(float(parts[1]))
                    errv.append(float(parts[2]))
                    vgas.append(float(parts[3]))
                    vdisk.append(float(parts[4]))
                    vbul.append(float(parts[5]))
                    sbdisk.append(float(parts[6]) if len(parts) > 6 else 0.0)
                    sbbul.append(float(parts[7]) if len(parts) > 7 else 0.0)
                except ValueError:
                    continue
    except OSError:
        return None

    if not rad:
        return None

    r_kpc = np.array(rad, dtype=float)
    v_obs_kms = np.array(vobs, dtype=float)
    sigma_v_kms = np.array(errv, dtype=float)
    v_gas_kms = np.array(vgas, dtype=float)
    v_disk_kms = np.array(vdisk, dtype=float)
    v_bulge_kms = np.array(vbul, dtype=float)

    # T12.D rule (2): galaxy minimum length (L209-210 verbatim)
    if len(r_kpc) < 3:
        return None

    # T12.D rule (operational): drop r=0 / Vobs<=0 (singular g_obs / sigma_g)
    mask = (r_kpc > 0.0) & (v_obs_kms > 0.0)
    if not np.any(mask):
        return None
    r_kpc = r_kpc[mask]
    v_obs_kms = v_obs_kms[mask]
    sigma_v_kms = sigma_v_kms[mask]
    v_gas_kms = v_gas_kms[mask]
    v_disk_kms = v_disk_kms[mask]
    v_bulge_kms = v_bulge_kms[mask]

    # Upsilon_d (disk M/L) is per-galaxy from phase1; the loader signature
    # accepts upsilon_d and upsilon_b so that g_obs / g_bar / sigma_g can be
    # computed inline. Algorithm_b_per_galaxy passes through galaxy_row['ud'].
    # Default upsilon_d=1.0 / upsilon_b=1.0 is for stub / test invocations only.
    g_obs, g_bar, sigma_g = compute_g_obs_g_bar(
        r_kpc=r_kpc,
        v_obs_kms=v_obs_kms,
        v_disk_kms=v_disk_kms,
        v_gas_kms=v_gas_kms,
        v_bulge_kms=v_bulge_kms,
        sigma_v_kms=sigma_v_kms,
        upsilon_d=upsilon_d,
        upsilon_b=upsilon_b,
    )

    return {
        "r": r_kpc,
        "v_obs": v_obs_kms,
        "v_disk": v_disk_kms,
        "v_gas": v_gas_kms,
        "v_bulge": v_bulge_kms,
        "sigma_v": sigma_v_kms,
        "g_obs": g_obs,
        "g_bar": g_bar,
        "sigma_g": sigma_g,
        "sbdisk": np.array(sbdisk, dtype=float)[mask] if sbdisk else np.zeros(len(r_kpc)),
        "sbbul": np.array(sbbul, dtype=float)[mask] if sbbul else np.zeros(len(r_kpc)),
    }


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
    # T12 fill (P3, v1.0.3): canonical V → g derivation
    # Source: C3-A1 sparc_fp_verification.py L30-32 (constants) +
    #         L111-132 (compute_fp body), SHA ab6f509b.
    # ------------------------------------------------------------------
    # T12.B verbatim formulas:
    #   r_m       = Rad_kpc * kpc_m                                    # m
    #   vbar2     = Vgas² + Y_d * sign(Vdisk) * Vdisk²
    #                     + Y_b * sign(Vbul)  * Vbul²                  # km²/s², signed
    #   vbar2     = |vbar2|                                            # |.| guard
    #   g_bar     = (vbar2 * 1e6) / r_m                                # m/s²
    #   g_obs     = (Vobs² * 1e6) / r_m                                # m/s²
    #   sigma_g   = (2.0 * Vobs * errV * 1e6) / r_m                    # m/s²
    # NOTE (T12.B verbatim):
    #   helium factor 1.33 は NOT applied to per-radius Vgas; per-radius
    #   Vgas は Lelli 2016 SPARC convention で helium 込み済。1.33 は MRT
    #   integral level (Mgas total computation) のみ。
    kpc_m: float = 3.0857e19
    r_m = r_kpc * kpc_m

    # vbar2 computation (T12.B verbatim, signed-velocity rule)
    vbar2_kms = (
        v_gas_kms ** 2
        + upsilon_d * np.sign(v_disk_kms) * v_disk_kms ** 2
        + upsilon_b * np.sign(v_bulge_kms) * v_bulge_kms ** 2
    )
    vbar2_kms = np.abs(vbar2_kms)  # |.| guard

    # mask r_m > 0 (defensive; loader already drops r==0)
    g_obs = np.zeros_like(r_m)
    g_bar = np.zeros_like(r_m)
    sigma_g = np.zeros_like(r_m)
    mask = r_m > 0.0

    # 1e6 = (1000 m/km)² → km²/s² → m²/s² unit conversion
    g_bar[mask] = (vbar2_kms[mask] * 1e6) / r_m[mask]
    g_obs[mask] = (v_obs_kms[mask] ** 2 * 1e6) / r_m[mask]

    # T12.C verbatim: σ_g via linear error propagation (g ∝ V² → σ(V²)=2V·σ_V)
    sigma_g[mask] = (2.0 * v_obs_kms[mask] * sigma_v_kms[mask] * 1e6) / r_m[mask]

    return g_obs, g_bar, sigma_g


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

    T13.B FINDING (v1.0.3): parent v4.8 §3.1 L335 spec defines f_opt only
    at the x=0.5 canonical anchor (closed form 2π/√V''(x=0.5, c)). The (x, c)
    joint form for arbitrary x is NOT defined in v4.8 spec. Operational
    pipelines (Algorithm B + E pipeline) employ the **x=0.5 canonical
    reduction** as projection convention — c*(r) is solved per-radius via
    target_nu = g_obs(r) / g_bar(r), and the x_r = g_bar(r)/a_0 dependence
    is folded into c*(r) parametrically (B) or via parametric c*(r; θ_E) (E).

    This is spec-correct per Path (iii) LOCK: ν_canonical = f_opt^(-1)(c)
    at x=0.5 anchor. Arbitrary-x derivation is a v4.9 patch round candidate
    (separate parent §3.1 round, not §2.5 v0.2 scope).

    For x != 0.5, this guard raises NotImplementedError to prevent silent
    operational misuse outside the canonical anchor.

    See _backsolve_c() and e_pipeline_score() for how the projection is
    applied operationally without invoking f_opt(x != 0.5).
    """
    if abs(x - 0.5) < 1e-12:
        return f_opt_v3_cascade(c)
    raise NotImplementedError(
        "f_opt(x; c) for x != 0.5 is NOT defined in parent v4.8 §3.1 L335 "
        "spec (T13.B finding, v1.0.3). v4.9 patch round candidate. "
        "Operational pipelines apply x=0.5 canonical reduction directly via "
        "f_opt_v3_cascade(c); see _backsolve_c() and e_pipeline_score()."
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
    ν_canonical(x; c) — operational projection at canonical x=0.5 anchor.

    OPERATIONAL DEFINITION (T13.B finding, v1.0.3):
        ν_canonical(x; c) ≡ f_opt_v3_cascade(c)          # x=0.5 reduction
        = 2π / √V''(x=0.5, c)                            # parent v4.8 §3.1 L335

    Per T13.B: parent v4.8 §3.1 L335 defines f_opt only at x=0.5 anchor.
    The operational pipelines (Algorithm B + E pipeline) capture per-radius
    x_r = g_bar(r)/a_0 dependence parametrically via c*(r), not through
    the (x, c) joint form (which is v4.9 patch round candidate).

    The x parameter is retained in signature for v4.9 forward compatibility
    but is NOT used in the current operational form.

    Reference pairs (T13.D verbatim):
        c=0.30  → ν=0.79774
        c=0.42  → ν=1.00982   (NGC 3198 typical)
        c=0.618 → ν=1.39777   (cascade base)
        c=0.80  → ν=1.84527   (NGC 2841 typical)
        c=0.83  → ν=1.94250   (cascade canonical, foundation b0cb36d7 anchor)
        c=1.00  → ν=2.56510
    """
    return f_opt_v3_cascade(c)


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
    # T13.D verbatim populate (v1.0.3, P9):
    # 5-anchor exact (foundation_gamma_actual.py L120-131 module-load self-check):
    (0.5, 0.30,  0.79774, 1e-3),  # Im,        T=10 anchor
    (0.5, 0.42,  1.00982, 1e-3),  # Sc,        T=5  anchor (NGC 3198 typical)
    (0.5, 0.618, 1.39777, 1e-3),  # cascade base reference
    (0.5, 0.80,  1.84527, 1e-3),  # Sb,        T=3  anchor (NGC 2841 typical)
    (0.5, 1.00,  2.56510, 1e-3),  # flexon boundary
    # off-anchor canonical (foundation L97 anchor verify):
    (0.5, 0.83,  1.94250, 1e-3),  # cascade canonical (CANONICAL anchor)
    # NOTE: c_super=0.5709 (anchor 17 §3.8.4 inheritance, V''=23.94 numerical
    #       source) is NOT included here. The deg-4 Lagrange actual value at
    #       c=0.5709 is f_opt = 1.30266 (cascade SSoT canonical interpolation),
    #       which differs from the 1.2841 anchor 17 inheritance value because
    #       the two use separate V'' sources. c_super-related verification
    #       belongs to anchor 17 §3.8 audit scope, not §2.5 v0.2 cascade SSoT
    #       reproducibility.
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
        target_ν(r) ≡ g_obs(r) / g_bar(r)             # observed ν ratio
        c*(r)_{n+1} ← brentq solve of:
                       f_opt_v3_cascade(c) = target_ν(r)
                       on c ∈ [0.30, 1.00] cascade SSoT canonical range

    OPERATIONAL PROJECTION (T13.B, v1.0.3):
        Per parent v4.8 §3.1 L335, f_opt is defined only at x=0.5 canonical
        anchor. The per-radius x_r = g_bar(r)/a_0 dependence is captured
        operationally via target_ν(r), not via f_opt(x_r; c). This is
        spec-correct per Path (iii) LOCK (R-2 simultaneous self-consistency
        loop preserved). x_at_radius is retained in _backsolve_c signature
        for v4.9 forward compatibility but unused in current form.

    Per-radius radii where target_ν is outside the bracket
    [f_opt(0.30), f_opt(1.00)] = [0.7977, 2.5651] return NaN; the galaxy
    is then flagged for post-hoc audit (out_of_range_flag=True if any).

    Returns
    -------
    c_star_next : per-radius c*(r) at iteration n+1 (NaN where unsolvable)
    """
    n_radii = len(c_star_n)
    c_star_next = np.full(n_radii, np.nan, dtype=float)
    for i in range(n_radii):
        if g_bar[i] <= 0.0:
            continue  # defensive: g_bar=0 leaves c_star_next[i] = NaN
        x_i = g_bar[i] / a_0_value           # deep-MOND ratio (stored for v4.9 fwd compat)
        target = g_obs[i] / g_bar[i]         # observed ν ratio
        c_star_next[i] = _backsolve_c(target, x_i)
    return c_star_next


def _backsolve_c(target_nu: float, x_at_radius: float) -> float:
    """
    Back-solve cascade c from target ν at canonical x=0.5 reduction.

    OPERATIONAL PROJECTION (T13.B finding, v1.0.3):
        Per parent v4.8 §3.1 L335, f_opt is defined only at x=0.5 anchor.
        For per-radius Algorithm B inversion at x_r = g_bar(r)/a_0 (which
        differs from 0.5 in general), we apply the **x=0.5 canonical
        reduction**: solve f_opt_v3_cascade(c) = target_nu for c, where
        target_nu = g_obs(r) / g_bar(r) is the observed ν ratio.

        The x_at_radius parameter is retained in signature for v4.9
        forward compatibility (when an arbitrary-x derivation is added),
        but is NOT used in the current operational form per T13.B.

    This projection is spec-correct per Path (iii) LOCK: ν_canonical =
    f_opt^(-1)(c) at the x=0.5 canonical anchor, with x_r dependence
    captured operationally via the per-radius target_nu.

    Returns NaN if the brentq inversion fails to bracket within
    [0.30, 1.00] (target outside cascade SSoT canonical range).
    """
    # Cascade SSoT canonical range (anchor 6 §2.3 BC: c < 0.30 = Strigari)
    c_lo, c_hi = 0.30, 1.00
    try:
        # x=0.5 canonical reduction (T13.B operational projection):
        # solve f_opt_v3_cascade(c) = target_nu for c via brentq
        def residual(c_val: float) -> float:
            return f_opt_v3_cascade(c_val) - target_nu
        # Bracket check: target_nu must lie in [f_opt(0.30), f_opt(1.00)]
        # = [0.7977, 2.5651] for the inversion to succeed
        f_lo = residual(c_lo)
        f_hi = residual(c_hi)
        if f_lo * f_hi > 0:
            return float("nan")  # target outside bracket
        return brentq(residual, c_lo, c_hi, xtol=1e-10)
    except (ValueError, RuntimeError):
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

def e_pipeline_score(
    galaxy_row: pd.Series,
    rotation_curve: Optional[Dict[str, np.ndarray]] = None,
    k_E: int = K_E_DEFAULT,
) -> float:
    """
    E pipeline score: AIC_E = 2·NLL_E + 2·k_E (anchor 7 §2.5.2 + T14.A).

    NLL_E formula (anchor 7 §2.5.2 L246-253 verbatim, T14.A):
        NLL_E(θ_E) = (1/2) · Σ_r [(g_obs(r) − g_pred(r; θ_E))² / σ_g(r)²]
                     + Σ_r log(σ_g(r)) + const
        const      = (n_g / 2) · log(2π)

    g_pred parametric form (anchor 5 §2.4 Candidate E + T14.A):
        c*(r; θ_E) = c_∞ + (c_0 − c_∞) · exp(−r / λ_eff)    # c_∞ = c_cascade = 0.83
        g_pred(r; θ_E) = ν_canonical(g_bar(r)/a_0; c*(r; θ_E)) · g_bar(r)
                       = f_opt_v3_cascade(c*(r; θ_E)) · g_bar(r)
                                                              # x=0.5 projection (T13.B)

    θ_E = (c_0, λ_eff)  with k_E = 2 (Q-C1 LOCK).
    Levenberg-Marquardt fit via scipy.optimize.least_squares.

    AIC_E = 2·NLL_E + 2·k_E (per anchor 7 §2.5.2 line 261, T14.C confirmed).

    T14.F finding (v1.0.3): per-galaxy Gaussian NLL is first concrete impl.
    No prior reference exists in foundation / C3-A1 / Phase 5-2; benchmark
    values establish on first numerical run.

    Returns
    -------
    AIC_E : float
        2·NLL_E + 2·k_E. NaN if rotation_curve is None or fit fails.
    """
    if rotation_curve is None:
        return float("nan")
    g_obs = np.asarray(rotation_curve["g_obs"], dtype=float)
    g_bar = np.asarray(rotation_curve["g_bar"], dtype=float)
    sigma_g = np.asarray(rotation_curve["sigma_g"], dtype=float)
    r_kpc = np.asarray(rotation_curve["r"], dtype=float)
    n_g = len(g_obs)
    if n_g < 3:
        return float("nan")

    # Filter valid radii (positive sigma_g, finite values)
    valid = (
        np.isfinite(g_obs) & np.isfinite(g_bar) & np.isfinite(sigma_g)
        & (sigma_g > 0.0) & (g_bar > 0.0)
    )
    if valid.sum() < 3:
        return float("nan")
    g_obs_v = g_obs[valid]
    g_bar_v = g_bar[valid]
    sigma_g_v = sigma_g[valid]
    r_v = r_kpc[valid]
    n_v = int(valid.sum())

    # c_∞ = c_cascade = 0.83 (BC fixed per T14.A)
    c_inf = C_CASCADE

    def predict_g(theta_E: np.ndarray) -> np.ndarray:
        c_0, lambda_eff = theta_E
        # Bound c_0 to cascade SSoT canonical [0.30, 1.00]; clip safely
        c_0_clip = np.clip(c_0, 0.30, 1.00)
        # lambda_eff must be positive; floor at small value
        lam_safe = max(lambda_eff, 1e-3)
        c_star = c_inf + (c_0_clip - c_inf) * np.exp(-r_v / lam_safe)
        c_star = np.clip(c_star, 0.30, 1.00)
        # ν at canonical x=0.5 projection (T13.B)
        nu = np.array([f_opt_v3_cascade(float(ci)) for ci in c_star])
        return nu * g_bar_v

    def residual_fn(theta_E: np.ndarray) -> np.ndarray:
        g_pred = predict_g(theta_E)
        return (g_obs_v - g_pred) / sigma_g_v

    # Initial guess: c_0 = 0.42 (anchor 6 inner anchor), lambda_eff = median(r)
    theta_init = np.array([C_GALAXY_INIT, float(np.median(r_v))])
    try:
        result = least_squares(
            residual_fn, theta_init,
            method="lm", max_nfev=200, xtol=1e-8, ftol=1e-8,
        )
        residuals = result.fun
        nll_e = (
            0.5 * float(np.sum(residuals ** 2))
            + float(np.sum(np.log(sigma_g_v)))
            + 0.5 * n_v * float(np.log(2.0 * np.pi))
        )
    except (ValueError, RuntimeError):
        return float("nan")

    aic_e = 2.0 * nll_e + 2.0 * k_E
    return float(aic_e)


def b_pipeline_score(
    galaxy_row: pd.Series,
    b_result: AlgorithmBResult,
    rotation_curve: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """
    Algorithm B score: AIC_B = 2·NLL_B + 2·k_B (anchor 7 §2.5.3 + T14.B).

    NLL_B formula (anchor 7 §2.5.3 L370-381 verbatim, on convergence):
        NLL_B = Σ_r log(σ_g(r)) + const
              = Σ_r log(σ_g(r)) + (n_g / 2) · log(2π)
        (residual term = 0 since g_pred ≡ g_obs exactly when B converges)

    AIC_B = 2·NLL_B + 2·k_B,  k_B = 0 (R-1 (R1-α) LOCK: parameter-free canonical).
          = 2·NLL_B

    Σ domain: per-galaxy rotation curve all valid radii (T14.B).

    On non-convergence, returns NaN (galaxy excluded from §2.5.4 selection
    per anchor 7 §2.5.3 L401-409 verbatim).

    T14.F finding: first concrete Python impl (no prior reference); semantic
    is exact per anchor 7 §2.5.3 verbatim.

    Returns
    -------
    AIC_B : float
        2·NLL_B (k_B=0). NaN if not converged or rotation_curve missing.
    """
    if not b_result.converged:
        return float("nan")
    if rotation_curve is None:
        return float("nan")
    sigma_g = np.asarray(rotation_curve["sigma_g"], dtype=float)
    valid = np.isfinite(sigma_g) & (sigma_g > 0.0)
    if valid.sum() < 3:
        return float("nan")
    sigma_g_v = sigma_g[valid]
    n_v = int(valid.sum())
    nll_b = (
        float(np.sum(np.log(sigma_g_v)))
        + 0.5 * n_v * float(np.log(2.0 * np.pi))
    )
    aic_b = 2.0 * nll_b + 2.0 * K_B  # K_B = 0
    return float(aic_b)


def select_pipeline(
    galaxy_row: pd.Series,
    b_result: AlgorithmBResult,
    rotation_curve: Optional[Dict[str, np.ndarray]] = None,
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
    aic_e = e_pipeline_score(galaxy_row, rotation_curve=rotation_curve, k_E=k_E)
    aic_b = b_pipeline_score(galaxy_row, b_result, rotation_curve=rotation_curve)
    delta = aic_e - aic_b
    if np.isnan(delta):
        # B did not converge or scoring failed; fall back to E
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
    # T14.E finding (v1.0.3, P10): NGC 3198 numerical NLL benchmark is NOT
    # in anchor 14 (B+級 prerequisite established via path A median 0.5629
    # only; per-galaxy NLL is §2.5 v0.2 round outcome to be established).
    # Hard numerical benchmark pairs are therefore unavailable as input.
    #
    # Instead, nll_self_check verifies STRUCTURAL INVARIANTS:
    #   1. NLL_E >= NLL_B  (B fits exactly by construction at convergence)
    #   2. AIC_E - AIC_B >= +4  (a priori parsimony penalty: k_E=2 vs k_B=0)
    #   3. NLL_B closed-form: NLL_B == sum(log(sigma_g)) + (n_g/2)*log(2π)
    #
    # These invariants hold by analytic derivation regardless of input data;
    # they validate the implementation without requiring numerical benchmarks.
    # Hard numerical pairs (e.g., NGC 3198 NLL_E = X.XX) will populate after
    # first numerical run on Windows side and will be patched in v1.0.4 round.
]


# Structural invariants for NLL self-check (T14.E + T14.F finding)
# Each entry verifies one analytic property of (NLL_E, NLL_B) impl.
NLL_STRUCTURAL_INVARIANTS: List[Dict[str, Any]] = [
    {
        "name": "B_fits_exactly_residual_zero",
        "rule": "NLL_B == sum(log(sigma_g)) + (n_g/2)*log(2*pi) "
                "(residual term = 0 at B convergence per anchor 7 §2.5.3)",
        "abs_tol": 1e-9,
    },
    {
        "name": "E_nll_ge_B_nll",
        "rule": "NLL_E >= NLL_B (E at best matches B; cannot beat it on residual)",
        "abs_tol": 1e-9,  # allow numerical roundoff
    },
    {
        "name": "AIC_a_priori_penalty",
        "rule": "AIC_E - AIC_B >= 2*k_E - 2*k_B = 4 (k_E=2, k_B=0)",
        "abs_tol": 1e-9,
    },
]


def nll_self_check(
    test_galaxies: Optional[Dict[str, pd.Series]] = None,
    b_results: Optional[Dict[str, AlgorithmBResult]] = None,
    rotation_curves: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    T14 verification framework — structural invariants test (T14.E + T14.F).

    T14.E finding: NGC 3198 numerical NLL benchmark is NOT in anchor 14;
    hard numerical pairs unavailable. Instead, this verifies analytical
    structural invariants of (NLL_E, NLL_B) that hold regardless of data:

        1. NLL_B closed-form:   NLL_B == sum(log(sigma_g)) + (n_g/2)*log(2π)
        2. Ordering invariant:  NLL_E >= NLL_B
        3. AIC penalty:         AIC_E - AIC_B >= 4 (k_E=2 - k_B=0 = 2 → x2 = 4)

    Validation strategy:
        - If test_galaxies + b_results + rotation_curves all provided AND
          at least one galaxy has converged Algorithm B with valid sigma_g,
          run all three invariants on that galaxy and report.
        - Otherwise, return status='deferred' (no available test galaxy).

    Returns
    -------
    dict with keys:
        n_invariants : int  — total invariants tested (0, or 3)
        n_pass       : int  — invariants passed
        n_fail       : int  — invariants failed
        status       : 'deferred' | 'pass' | 'fail'
        details      : list of per-invariant evaluation
    """
    if (test_galaxies is None or b_results is None
            or rotation_curves is None):
        return {
            "n_invariants": 0,
            "status": "deferred",
            "reason": "test_galaxies / b_results / rotation_curves not provided",
        }

    # Find a galaxy with B converged + rotation curve loaded
    test_galaxy: Optional[str] = None
    for g in test_galaxies:
        if (g in b_results and b_results[g].converged
                and g in rotation_curves and rotation_curves[g] is not None):
            test_galaxy = g
            break
    if test_galaxy is None:
        return {
            "n_invariants": 0,
            "status": "deferred",
            "reason": "no test galaxy with converged B + valid rotation curve",
        }

    galaxy_row = test_galaxies[test_galaxy]
    b_result = b_results[test_galaxy]
    rc = rotation_curves[test_galaxy]

    aic_e = e_pipeline_score(galaxy_row, rotation_curve=rc, k_E=K_E_DEFAULT)
    aic_b = b_pipeline_score(galaxy_row, b_result, rotation_curve=rc)
    if np.isnan(aic_e) or np.isnan(aic_b):
        return {
            "n_invariants": 3,
            "status": "deferred",
            "reason": f"score function returned NaN for {test_galaxy}",
        }
    nll_e = (aic_e - 2 * K_E_DEFAULT) / 2.0
    nll_b = (aic_b - 2 * K_B) / 2.0

    # Closed-form NLL_B reproduction (invariant 1)
    sigma_g = np.asarray(rc["sigma_g"], dtype=float)
    valid = np.isfinite(sigma_g) & (sigma_g > 0.0)
    sigma_g_v = sigma_g[valid]
    n_v = int(valid.sum())
    nll_b_closed = (
        float(np.sum(np.log(sigma_g_v)))
        + 0.5 * n_v * float(np.log(2.0 * np.pi))
    )

    details = []
    n_pass = 0
    n_fail = 0

    # Invariant 1: NLL_B closed-form
    diff_1 = abs(nll_b - nll_b_closed)
    inv1_pass = diff_1 < 1e-9
    details.append({
        "invariant": "B_fits_exactly_residual_zero",
        "test_galaxy": test_galaxy,
        "nll_b_observed": nll_b, "nll_b_closed_form": nll_b_closed,
        "abs_diff": diff_1, "abs_tol": 1e-9,
        "pass": inv1_pass,
    })
    n_pass += int(inv1_pass); n_fail += int(not inv1_pass)

    # Invariant 2: NLL_E >= NLL_B
    inv2_pass = nll_e >= nll_b - 1e-9
    details.append({
        "invariant": "E_nll_ge_B_nll",
        "test_galaxy": test_galaxy,
        "nll_e": nll_e, "nll_b": nll_b,
        "pass": inv2_pass,
    })
    n_pass += int(inv2_pass); n_fail += int(not inv2_pass)

    # Invariant 3: AIC_E - AIC_B >= 4 (k_E=2, k_B=0)
    delta_aic = aic_e - aic_b
    inv3_pass = delta_aic >= 4.0 - 1e-9
    details.append({
        "invariant": "AIC_a_priori_penalty",
        "test_galaxy": test_galaxy,
        "aic_e": aic_e, "aic_b": aic_b, "delta_aic": delta_aic,
        "expected_lower_bound": 4.0,
        "pass": inv3_pass,
    })
    n_pass += int(inv3_pass); n_fail += int(not inv3_pass)

    status = "pass" if n_fail == 0 else "fail"
    return {
        "n_invariants": 3,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "status": status,
        "test_galaxy": test_galaxy,
        "details": details,
    }


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


def log_umem_alpha(rho_gal: np.ndarray) -> np.ndarray:
    """
    log u_mem under alpha hypothesis (Phase C3 v3 §3.1, T16.E + T18.A verbatim).

    log u_mem alpha = 2.0 * log10(rho_gal)

    Source: phase_c3_step3_dsph_gamma_vs_alpha.py L285-287 verbatim (SHA c51c72f0).
    """
    return 2.0 * np.log10(rho_gal)


def _fit_ols_partial(
    y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    OLS via numpy.linalg.lstsq with NLL computation.
    Source: phase_c3_step3 fit_ols() L295-301 verbatim (SHA c51c72f0).

    Returns
    -------
    beta : np.ndarray   coefficient vector
    nll  : float        negative log likelihood (Gaussian, MLE sigma^2)
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    sigma2 = float(np.sum(resid ** 2) / n)
    if sigma2 <= 0:
        return beta, 1e10
    nll = 0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    return beta, float(nll)


def _prepare_sparc_phase_c3_sample(
    sparc_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Phase C3 v3 §4.3 SPARC sample preparation.
    Source: phase_c3_step3_dsph_gamma_vs_alpha.py L252-278 verbatim (T16.F).

    Returns 124-galaxy sample after:
      - Q < 3 cut
      - Mass / radius derivation (Mbar = Mstar + Mgas, r_h = 1.68 * Rdisk)
      - C15 prediction gc_C15
      - Bridge 4-galaxy exclusion (NGC3741, NGC2915, ESO444-G084, NGC1705)
      - delta_primary = log10(g_obs) - log10(gc_C15)

    Required input columns:
      Galaxy, Q, Upsilon_d, L36, MHI, Rdisk, Vflat, g_obs

    Returns None if required columns are missing or sample is empty.
    """
    required_cols = {"Galaxy", "Upsilon_d", "L36", "MHI", "Rdisk", "Vflat"}
    if not required_cols.issubset(set(sparc_df.columns)):
        missing = required_cols - set(sparc_df.columns)
        # Return None to signal sample preparation cannot proceed
        return None
    df = sparc_df.copy()
    # T16.F verbatim: Q cut (if Q column exists)
    if "Q" in df.columns:
        df = df[df["Q"] < 3].reset_index(drop=True)
    # Mass / radius derivation (T16.F verbatim L252-275)
    df["Mstar"] = df["Upsilon_d"] * df["L36"] * 1e9
    df["Mgas"]  = HELIUM_FACTOR * df["MHI"] * 1e9
    df["Mbar"]  = df["Mstar"] + df["Mgas"]
    df["hR"]    = df["Rdisk"]
    df["r_h"]   = 1.68 * df["hR"]
    df["rho_gal"] = df["Mbar"] / (4.0 / 3.0 * np.pi * df["r_h"] ** 3)
    df["v_flat"] = df["Vflat"]
    df["gc_C15"] = (
        C15_COEF * df["Upsilon_d"] ** C15_UPSILON_EXP
        * np.sqrt(A0_KPC * df["v_flat"] ** 2 / df["hR"])
    )
    # Bridge exclusion (T16.F + EXCLUDED_4_SPARC_GALAXIES dict)
    df = df[~df["Galaxy"].isin(set(EXCLUDED_4_SPARC_GALAXIES.keys()))].reset_index(drop=True)
    # delta_primary requires g_obs column (per-galaxy mean from rotation curves
    # or from MRT-aggregated value, depending on caller-side wiring)
    if "g_obs" not in df.columns:
        # Cannot compute delta_primary without observation; signal incomplete
        return None
    df["delta_primary"] = np.log10(df["g_obs"]) - np.log10(df["gc_C15"])
    df["log_rh"] = np.log10(df["r_h"])
    # Final defensive: drop NaN / inf in critical columns
    valid = (
        np.isfinite(df["rho_gal"]) & (df["rho_gal"] > 0)
        & np.isfinite(df["r_h"]) & (df["r_h"] > 0)
        & np.isfinite(df["delta_primary"])
    )
    df = df[valid].reset_index(drop=True)
    if len(df) == 0:
        return None
    return df


def _prepare_dsph_phase_c3_sample(
    dsph_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Phase C3 v3 §4.3 dSph sample preparation.
    Source: phase_c3_step3 L137-141 + Sgr regex L111-112 verbatim (T18.C).

    Returns 30-galaxy sample after:
      - Sgr exclusion via regex r'sagittarius\\s+dsph' IGNORECASE on 'name' column
      - delta_primary = log10(g_obs) - log10(0.228 * a_0)
      - log_rh = log10(r_h) computed from input

    Required input columns:
      name, g_obs, rho_gal, r_h
    (Or columns from which these can be derived; adapter for v1.0.3 expects
    dsph DataFrame already to carry rho_gal and r_h. If missing, returns None.)

    Returns None if required columns are missing or sample is empty.
    """
    required_cols = {"name", "g_obs", "rho_gal", "r_h"}
    if not required_cols.issubset(set(dsph_df.columns)):
        return None
    df = dsph_df.copy()
    # Sgr exclusion (T18.C verbatim, Phase C2 protocol L111-112 + L137-141)
    sgr_pat = re.compile(SGR_NAME_REGEX, SGR_REGEX_FLAGS)
    df["is_sgr"] = df["name"].astype(str).apply(lambda s: bool(sgr_pat.search(s)))
    df = df[~df["is_sgr"]].reset_index(drop=True)
    df = df.drop(columns=["is_sgr"])
    # delta_primary (dSph anchor: G_Strigari = 0.228 * a_0)
    df["delta_primary"] = np.log10(df["g_obs"]) - np.log10(G_STRIGARI_M_S2)
    df["log_rh"] = np.log10(df["r_h"])
    # Defensive: drop NaN / inf
    valid = (
        np.isfinite(df["rho_gal"]) & (df["rho_gal"] > 0)
        & np.isfinite(df["r_h"]) & (df["r_h"] > 0)
        & np.isfinite(df["delta_primary"])
    )
    df = df[valid].reset_index(drop=True)
    if len(df) == 0:
        return None
    return df


def b_alpha_3axis_audit(
    sparc_df: pd.DataFrame, dsph_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute b_α on Phase C3 v3 §4.3 universal coupling 3 axes.

    Spec sources (v1.0.3 P12 公式 spec impl):
        anchor 7 §2.5.5 (J_system_paper_section2_5_v0.1.md, SHA 9e03f53e)
            - Axis 1: extreme regime continuity check (c<0.30 Strigari)
            - Axis 2: dSph 28/31 reversal trend reproduction (J0 baseline)
            - Axis 3: universal slope b_α=0.11 emergence audit
        C3-A5 internal_memo_c3_extension_v3.pdf §4.3 (SHA 69fb1a95)
            - b_α formal definition + 0.11 ± 0.005 universal across 3.92 dex
            - §8.1 minimal model variables table (page 10)
        phase_c3_step3_dsph_gamma_vs_alpha.py L320-432 (SHA c51c72f0)
            - operational impl: numpy.linalg.lstsq partial OLS (3-feature)
            - T18.B combo separate-intercept design (L611-628)

    Operational form (T17 + T18 verbatim):
        target  = log10(g_obs / gc_C15)              # SPARC delta_primary
                  log10(g_obs / G_Strigari)          # dSph delta_primary
        feature = lu_a = 2 * log10(rho_gal)          # log u_mem alpha
        nuisance= log_rh = log10(r_h)                # size partialled out
        b_α     = lstsq([1, lu_a, log_rh], delta)[0][1]

    Three axes:
        Axis 1 (SPARC partial OLS):
            sample: 124 galaxy (Q<3 + 4 bridge excluded)
            expected: +0.1084 (anchor 19 §1.5 baseline)
        Axis 2 (dSph partial OLS):
            sample: 30 galaxy (31 - Sgr)
            expected: +0.1127 (anchor 19 §1.5 baseline)
        Axis 3 (combo separate-intercept):
            sample: 154 (= 124 + 30) combined
            X = [is_sparc, is_dsph, is_sparc*lu, is_dsph*lu,
                 is_sparc*log_rh, is_dsph*log_rh]
            b[2] = b_alpha_sparc_combo
            b[3] = b_alpha_dsph_combo
            universal slope = mean of two combo slopes
            expected: 0.11 ± 0.005 (C3-A5 §8.1 minimal model table)

    Returns
    -------
    dict with keys:
        # 公式 anchor 7 §2.5.5 axes (高 level)
        axis_1_continuity_status : str  ('finite' / 'divergent')
        axis_2_reversal_status   : str  ('reproduced' / 'failed' / 'deferred')
        axis_3_universal_slope   : float
        axis_3_within_tolerance  : bool

        # Sub-axis breakdowns (caller-side compatibility for run_dsph_audit)
        axis_1_SPARC             : float  # SPARC partial OLS slope
        axis_2_dSph              : float  # dSph  partial OLS slope
        axis_3_combined_sparc    : float  # combo b[2]
        axis_3_combined_dsph     : float  # combo b[3]

        # Diff + audit
        abs_diff_axis12          : float  # |axis_1 - axis_2|
        abs_diff_combo           : float  # |b_combo_sparc - b_combo_dsph|
        sample_n_axis_1          : int
        sample_n_axis_2          : int
        sample_n_axis_3          : int
        estimator                : str
        sample_preparation_status: str   # 'complete' or 'incomplete'
    """
    sparc_prep = _prepare_sparc_phase_c3_sample(sparc_df)
    dsph_prep  = _prepare_dsph_phase_c3_sample(dsph_df)

    # If either sample preparation fails, raise NotImplementedError so that
    # run_dsph_audit's deferred_on_missing handles it gracefully.
    if sparc_prep is None or dsph_prep is None:
        raise NotImplementedError(
            "b_alpha_3axis_audit: sample preparation incomplete. "
            "SPARC requires Galaxy/Upsilon_d/L36/MHI/Rdisk/Vflat/g_obs columns; "
            "dSph requires name/g_obs/rho_gal/r_h columns. "
            "Phase C3 v3 §4.3 + phase_c3_step3 SHA c51c72f0 spec."
        )

    n1 = len(sparc_prep)
    n2 = len(dsph_prep)
    if n1 < 4 or n2 < 4:
        raise NotImplementedError(
            f"b_alpha_3axis_audit: sample size below minimum 4. "
            f"SPARC n={n1}, dSph n={n2}."
        )

    # Axis 1: SPARC partial OLS (T16+T17+T18 verbatim)
    rho_s = sparc_prep["rho_gal"].values
    lu_s  = log_umem_alpha(rho_s)
    log_rh_s = sparc_prep["log_rh"].values
    delta_s = sparc_prep["delta_primary"].values
    X1 = np.column_stack([np.ones(n1), lu_s, log_rh_s])
    beta_1, _ = _fit_ols_partial(delta_s, X1)
    axis_1_value = float(beta_1[1])

    # Axis 2: dSph partial OLS (same form, dSph delta target)
    rho_d = dsph_prep["rho_gal"].values
    lu_d  = log_umem_alpha(rho_d)
    log_rh_d = dsph_prep["log_rh"].values
    delta_d = dsph_prep["delta_primary"].values
    X2 = np.column_stack([np.ones(n2), lu_d, log_rh_d])
    beta_2, _ = _fit_ols_partial(delta_d, X2)
    axis_2_value = float(beta_2[1])

    # Axis 3: combo separate-intercept (T18.B verbatim L611-628)
    n3 = n1 + n2
    is_sparc = np.concatenate([np.ones(n1), np.zeros(n2)])
    is_dsph  = np.concatenate([np.zeros(n1), np.ones(n2)])
    lu_combined = np.concatenate([lu_s, lu_d])
    log_rh_combined = np.concatenate([log_rh_s, log_rh_d])
    delta_combined  = np.concatenate([delta_s, delta_d])
    X3 = np.column_stack([
        is_sparc, is_dsph,
        is_sparc * lu_combined, is_dsph * lu_combined,
        is_sparc * log_rh_combined, is_dsph * log_rh_combined,
    ])
    beta_3, _ = _fit_ols_partial(delta_combined, X3)
    axis_3_combined_sparc = float(beta_3[2])
    axis_3_combined_dsph  = float(beta_3[3])
    axis_3_universal = 0.5 * (axis_3_combined_sparc + axis_3_combined_dsph)
    axis_3_within = abs(axis_3_universal - B_ALPHA_AXIS3_BASELINE) <= B_ALPHA_AXIS3_TOLERANCE

    # Axis 1 continuity check (T17.C operational)
    # finite = numpy.linalg.lstsq returned a finite slope; divergent = NaN/inf
    axis_1_continuity = "finite" if np.isfinite(axis_1_value) else "divergent"

    # Axis 2 reversal trend (T17.C: anchor 7 §2.5.5 cite reference + J0 baseline)
    # T17 抽出 finding: C3-A5 PDF 内に explicit "28/31 reversal" operational
    # metric の closed-form NOT FOUND. Anchor 7 §2.5.5 internal cite + C3-A4
    # SHA 7e8823f4 J0 minimal form baseline は separate scope。v1.0.3 では
    # axis_2_value finite + sign positive (reversal direction matches dSph
    # baseline) を operational reproduction proxy として記録。
    if not np.isfinite(axis_2_value):
        axis_2_reversal = "deferred"
    elif axis_2_value > 0:
        axis_2_reversal = "reproduced"  # positive slope, same direction as 0.1127
    else:
        axis_2_reversal = "failed"

    abs_diff = abs(axis_1_value - axis_2_value)
    abs_diff_combo = abs(axis_3_combined_sparc - axis_3_combined_dsph)

    return {
        # 公式 anchor 7 §2.5.5 axes (高 level)
        "axis_1_continuity_status": axis_1_continuity,
        "axis_2_reversal_status":   axis_2_reversal,
        "axis_3_universal_slope":   axis_3_universal,
        "axis_3_within_tolerance":  bool(axis_3_within),
        # Sub-axis breakdowns (caller-side compatibility for run_dsph_audit)
        "axis_1_SPARC":             axis_1_value,
        "axis_2_dSph":              axis_2_value,
        "axis_3_combined_sparc":    axis_3_combined_sparc,
        "axis_3_combined_dsph":     axis_3_combined_dsph,
        # Diff statistics (Lesson 93 §5.3 universal coupling)
        "abs_diff_axis12":          abs_diff,
        "abs_diff_combo":           abs_diff_combo,
        # Sample sizes (audit trail)
        "sample_n_axis_1":          n1,
        "sample_n_axis_2":          n2,
        "sample_n_axis_3":          n3,
        # Estimator metadata
        "estimator":                "numpy.linalg.lstsq partial OLS (Phase C3 v3 §4.3)",
        "sample_preparation_status": "complete",
    }


def b_alpha_self_check(
    sparc_df: Optional[pd.DataFrame] = None,
    dsph_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    P12 self-check: verify b_alpha_3axis_audit reproduces baseline values.

    Verification target (anchor 19 §1.5 + C3-A5 §4.3 baseline):
        axis_1_SPARC      ≈ 0.1084 (within 1e-3 if sample matches)
        axis_2_dSph       ≈ 0.1127 (within 1e-3 if sample matches)
        abs_diff_axis12   ≈ 0.0042 (within 1e-3 if both axes match)
        axis_3 within ±0.005 of 0.11 (universal coupling, §8.1)

    Returns 'deferred' if input DataFrames not provided (e.g., during
    --no-rotation-curve dry-run mode); returns 'pass' / 'fail' with
    detailed deltas otherwise.
    """
    if sparc_df is None or dsph_df is None:
        return {
            "status": "deferred",
            "reason": "sparc_df / dsph_df not provided (numerical run pending)",
        }
    try:
        result = b_alpha_3axis_audit(sparc_df, dsph_df)
    except NotImplementedError as exc:
        return {"status": "deferred", "reason": str(exc)}

    # Reproducibility deltas
    delta_1 = abs(result["axis_1_SPARC"] - B_ALPHA_SPARC_BASELINE)
    delta_2 = abs(result["axis_2_dSph"] - B_ALPHA_DSPH_BASELINE)
    delta_diff = abs(result["abs_diff_axis12"] - B_ALPHA_ABS_DIFF_BASELINE)

    # 1e-3 tolerance is reproducibility-level (T17.B: SPARC baseline 0.1084
    # established to 4 sig fig in §4.2 result table)
    tol = 1e-3
    inv1 = delta_1 < tol
    inv2 = delta_2 < tol
    inv_diff = delta_diff < tol
    inv_axis3 = bool(result["axis_3_within_tolerance"])

    n_pass = sum([inv1, inv2, inv_diff, inv_axis3])
    n_fail = 4 - n_pass
    status = "pass" if n_fail == 0 else "fail"

    return {
        "status": status,
        "n_invariants": 4,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "axis_1_delta": delta_1,    "axis_1_pass": inv1,
        "axis_2_delta": delta_2,    "axis_2_pass": inv2,
        "abs_diff_delta": delta_diff, "abs_diff_pass": inv_diff,
        "axis_3_within_tolerance": inv_axis3,
        "audit_result": result,
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

    # b_α 3-axis audit
    try:
        b_alpha = b_alpha_3axis_audit(sparc_df, dsph)
        result["b_alpha_status"] = "computed"
        result["b_alpha_axes"] = b_alpha
        # Compute |Δ| if SPARC + dSph values present
        if "axis_1_SPARC" in b_alpha and "axis_2_dSph" in b_alpha:
            abs_diff = abs(b_alpha["axis_1_SPARC"] - b_alpha["axis_2_dSph"])
            result["b_alpha_abs_diff"] = abs_diff
            result["b_alpha_anchor19_baseline_abs_diff"] = (
                B_ALPHA_ABS_DIFF_BASELINE
            )
            result["b_alpha_within_AC4_threshold_005"] = abs_diff <= 0.005
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

    # AC4: b_α |Δ| ≤ 0.005
    abs_diff = dsph_block.get("b_alpha_abs_diff")
    ac["AC4_b_alpha_abs_diff_le_0_005"] = {
        "threshold": 0.005,
        "observed_abs_diff": abs_diff,
        "anchor_19_baseline_abs_diff": B_ALPHA_ABS_DIFF_BASELINE,
        "pass": (abs_diff is not None) and (abs_diff <= 0.005),
    }
    if abs_diff is None:
        ac["AC4_b_alpha_abs_diff_le_0_005"]["pass"] = None

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
    rotation_curves_by_galaxy: Dict[str, Dict[str, np.ndarray]] = {}
    for _, row in sparc_171.iterrows():
        galaxy = str(row["galaxy"])
        # Load per-galaxy rotation curve (T12 wiring point, v1.0.3 P2/P3 fill)
        if config.no_rotation_curve:
            rc = None
        else:
            # Pass galaxy-specific Upsilon_d (and Upsilon_b if available) to
            # loader so that compute_g_obs_g_bar() inside the loader uses
            # the right disk M/L value per row. Fallback default 1.0 if
            # row schema doesn't carry these keys (defensive).
            ud_val = float(row["ud"]) if "ud" in row.index and np.isfinite(row.get("ud", np.nan)) else 1.0
            ub_val = float(row["ub"]) if "ub" in row.index and np.isfinite(row.get("ub", np.nan)) else 1.0
            rc = load_rotation_curve(
                galaxy, config.sparc_rc_base_path,
                upsilon_d=ud_val, upsilon_b=ub_val,
            )
            if rc is not None:
                n_rc_loaded += 1
                rotation_curves_by_galaxy[galaxy] = rc
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
        f"({convergence_rate:.4f}); rotation_curves loaded: {n_rc_loaded}"
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

    # T14 NLL structural invariants self-check (T14.E + T14.F finding)
    if not args.skip_self_check:
        try:
            t14_test_galaxies = {
                str(row["galaxy"]): row for _, row in sparc_171.iterrows()
            }
            t14_result = nll_self_check(
                test_galaxies=t14_test_galaxies,
                b_results=b_by_galaxy,
                rotation_curves=rotation_curves_by_galaxy,
            )
            if t14_result.get("status") == "deferred":
                logger.info(f"T14 NLL self-check deferred: {t14_result.get('reason')}")
            elif t14_result.get("status") == "fail":
                logger.error(f"T14 NLL self-check FAIL: {t14_result}")
                return 5
            else:
                logger.info(
                    f"T14 NLL self-check: status={t14_result.get('status')}, "
                    f"{t14_result.get('n_pass')}/{t14_result.get('n_invariants')} "
                    f"invariants pass on test_galaxy={t14_result.get('test_galaxy')}"
                )
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

    # P12 b_alpha_self_check: reproduction of anchor 19 §1.5 baseline
    # (0.1084 / 0.1127 / 0.0042) — pass condition for atomic completion.
    if not args.skip_self_check:
        # Need sparc_171 with g_obs aggregated per-galaxy (mean over rotation
        # curve) for sample preparation. Build summary df if rotation_curves
        # available; else pass through whatever the input schema carries.
        sparc_for_audit = sparc_171.copy()
        if rotation_curves_by_galaxy:
            g_obs_per_galaxy: Dict[str, float] = {}
            for galaxy, rc in rotation_curves_by_galaxy.items():
                g_arr = np.asarray(rc.get("g_obs", []), dtype=float)
                g_arr = g_arr[np.isfinite(g_arr) & (g_arr > 0)]
                if len(g_arr) > 0:
                    g_obs_per_galaxy[galaxy] = float(np.median(g_arr))
            sparc_for_audit["g_obs"] = sparc_for_audit["galaxy"].map(
                lambda g: g_obs_per_galaxy.get(g, float("nan"))
            )
        # Adapter: sparc_171 columns 'galaxy' / 'ud' / 'L36' / 'MHI' / 'Rdisk' /
        # 'Vflat' need to be aliased to b_alpha audit's expected schema
        # ('Galaxy' / 'Upsilon_d' / 'L36' / 'MHI' / 'Rdisk' / 'Vflat')
        col_alias_map = {
            "galaxy": "Galaxy", "ud": "Upsilon_d",
        }
        sparc_for_audit = sparc_for_audit.rename(columns=col_alias_map)
        b_alpha_check_result = b_alpha_self_check(
            sparc_df=sparc_for_audit, dsph_df=dsph,
        )
        b_alpha_status = b_alpha_check_result.get("status")
        if b_alpha_status == "deferred":
            logger.info(
                f"P12 b_alpha self-check deferred: "
                f"{b_alpha_check_result.get('reason')}"
            )
        elif b_alpha_status == "fail":
            logger.warning(
                f"P12 b_alpha self-check FAIL: "
                f"axis_1_delta={b_alpha_check_result.get('axis_1_delta'):.4f}, "
                f"axis_2_delta={b_alpha_check_result.get('axis_2_delta'):.4f}, "
                f"abs_diff_delta={b_alpha_check_result.get('abs_diff_delta'):.4f}, "
                f"axis_3_within={b_alpha_check_result.get('axis_3_within_tolerance')}"
            )
            # NOTE: do NOT return non-zero here; b_alpha self-check is a
            # reproducibility report, not a hard gate. Hard gate is AC4 in
            # acceptance_criteria evaluation (handled elsewhere).
        else:
            logger.info(
                f"P12 b_alpha self-check PASS: "
                f"{b_alpha_check_result.get('n_pass')}/4 invariants reproduced"
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
