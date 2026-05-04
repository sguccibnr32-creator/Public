# §2.5 v0.2 SPARC Empirical Execution — Output Schema v1.0

**File**: `OUTPUT_SCHEMA_section2_5_v0_2.md`
**Companion to**: `EXECUTION_PLAN_section2_5_v0_2.md` v1.0 + `run_section2_5_v0_2.py` v1.0
**Date**: 2026-05-03
**Author**: 坂口 忍 / 坂口製麺所、宍粟市
**License**: CC-BY 4.0
**Encoding**: UTF-8 / LF / no BOM

**Upstream LOCK references**:
- anchor 7 §2.5.1-§2.5.6 (SHA 9e03f53e) — Algorithm B / E pipeline / ΔAIC selection / dSph J3 / [H-1] protocol
- anchor 8 §2.6.5 verbatim L553-564 — F1/F2/F6/F8 placeholder content
- anchor 21 v0.1.1 (SHA 44df9afb...) — chapter-level §2 LOCK ESTABLISHED, F4/F12 RESOLVED
- anchor 14 §4.13-§4.16 (SHA 295bc05c) — Layer B-α/B-β sub-divide, NGC 3198 reference (f_p=0.9930, c_mem=0.0070)
- anchor 17 §3.8-§3.11 (SHA 178dad11) — c_super=0.5709, Pattern B canonical Möbius, dual-display

---

## §0 Scope

This schema defines the bit-exact data contract between `run_section2_5_v0_2.py` (producer) and **anchor 22** = `J_system_paper_section2_5_v0.2.md` (consumer, future deliverable).

Three classes of outputs:

1. **Per-record CSVs** — one row per galaxy / dwarf
2. **Aggregate JSONs** — summary statistics + acceptance criteria evaluation
3. **Audit logs + manifest** — forensic chain compliance (per anchor 20 §J 7-item ruleset)

All outputs are **immutable on write** (no overwrites within a single run; reruns produce new timestamped directories).

---

## §1 Output directory layout

```
$env:OUTPUT_ROOT/section2_5_v0_2_<RUN_ID>/
├── per_galaxy_sparc.csv               # SPARC 171-galaxy fit pool, one row per galaxy
├── per_galaxy_sparc_full.csv          # SPARC 175 (171 + 4 excluded with reason)
├── per_dsph.csv                       # dSph 31 sample, one row per dwarf
├── algorithm_b_log.csv                # Algorithm B per-iteration trace (171 × ≤50 rows)
├── f_opt_anchor_table.csv             # 5-anchor V"(x=0.5, c) reference values (LOCKED reproduction)
├── summary.json                       # aggregate statistics
├── sub_issues.json                    # S-1〜S-6 numerical resolve results
├── promotion.json                     # A 級 prerequisite re-verification
├── f_flag_status.json                 # F1/F2/F6/F8/F8 sensitivity audit results
├── manifest.json                      # forensic manifest (input SHAs + output SHAs + config)
├── run_config.json                    # full env vars + parameter snapshot
├── SHA256SUMS.txt                     # all output files SHA256
└── run.log                            # human-readable execution log
```

`<RUN_ID>` = ISO-8601 timestamp + git commit short SHA, e.g., `2026-05-03T14-30-00_aeca1c3`.

---

## §2 CSV schemas

### §2.1 `per_galaxy_sparc.csv` (SPARC 171-galaxy fit pool)

| Column | Type | Unit | Source / formula | Notes |
|---|---|---|---|---|
| galaxy | str | — | MRT name field | primary key |
| T | int | Hubble type | MRT T column | for stratification |
| L36 | float | 10⁹ L_⊙ | MRT L36 | Mstar input |
| Ud | float | dimensionless | phase1 ud | Υ_d, mass-to-light |
| Mstar | float | 10⁹ M_⊙ | Ud × L36 | derived |
| MHI | float | 10⁹ M_⊙ | MRT MHI | gas mass input |
| Mgas | float | 10⁹ M_⊙ | 1.33 × MHI | derived (helium correction) |
| Mbar | float | 10⁹ M_⊙ | Mstar + Mgas | total baryonic mass |
| Rdisk | float | kpc | MRT Rdisk | disk scale length, h_R proxy |
| SBdisk0 | float | L_⊙/pc² | MRT SBdisk0 | central surface brightness |
| Vflat | float | km/s | MRT Vflat | flat rotation velocity |
| gc_over_a0 | float | dimensionless | TA3 (in a₀ units, fixed) | observed g_c |
| gc_obs | float | m/s² | gc_over_a0 × a₀ | observed g_c (SI) |
| gc_C15 | float | m/s² | C15 formula | predicted g_c by C15 |
| residual_C15_dex | float | dex | log10(gc_obs/gc_C15) | per-galaxy residual |
| c_galaxy_init | float | dimensionless | 0.42 (constant) | Algorithm B init |
| c_galaxy_final | float | dimensionless | Algorithm B output | post-convergence |
| converged_B | bool | — | iteration result | tol=1e-6, N_max=50 |
| n_iter_B | int | — | iteration count | min 0, max 50 |
| residual_B | float | — | final residual norm | <tol if converged |
| f_p | float | dimensionless | per Layer B-α | NGC 3198 reference 0.9930 |
| c_mem | float | dimensionless | per Layer B-α | NGC 3198 reference 0.0070 |
| chi_coh | float | dimensionless | Layer B-α coherence | per anchor 13/14 closed-form |
| nu_canonical | float | dimensionless | f_opt⁻¹(x; c) | path (iii) canonical |
| g_eff | float | m/s² | g_bar × ν × χ_coh | hybrid form |
| eta_0_pathA | float | dimensionless | per-galaxy contribution to median | |
| eta_0_pathB | float | dimensionless | per-galaxy contribution to v_flat²-weighted | |
| pipeline_E_score | float | — | E pipeline likelihood / AIC | k_E=2 default |
| pipeline_B_score | float | — | Algorithm B likelihood / AIC | k_B=0 |
| delta_AIC | float | — | AIC_E − AIC_B | sign convention: positive favors B |
| selected_pipeline | str | — | "E" or "B" | per ΔAIC threshold |
| in_171_pool | bool | — | fit-pool membership | excludes 4 galaxies |
| exclusion_reason | str | — | for the 4 excluded only | empty for 171 in-pool |

**Row count**: 171 (the in-pool members; full 175 in `_full.csv`).

### §2.2 `per_galaxy_sparc_full.csv` (SPARC 175)

Same schema as §2.1, but `in_171_pool` may be False for 4 rows; `exclusion_reason` populated.

### §2.3 `per_dsph.csv` (dSph 31 sample)

| Column | Type | Unit | Notes |
|---|---|---|---|
| dwarf | str | — | name (e.g., "Draco", "Sculptor") |
| J_factor | float | log10(GeV²/cm⁵) | annihilation J |
| sigma_los | float | km/s | line-of-sight velocity dispersion |
| Rhalf | float | pc | half-light radius |
| J3_metric | float | dimensionless | J3 metric per anchor 7 §2.5.5 |
| J3_pass | bool | — | included in 28/31 (or 30/31 per F6 audit) |
| J3_pass_reason | str | — | inclusion / exclusion rationale |
| s_0 | float | dimensionless | Bernoulli analytic value |
| G_Strigari_over_a0 | float | dimensionless | s_0(1-s_0) per-dwarf prediction |
| b_alpha_axis1 | float | dimensionless | b_α axis 1 |
| b_alpha_axis2 | float | dimensionless | b_α axis 2 |
| b_alpha_axis3 | float | dimensionless | b_α axis 3 |
| in_28_baseline | bool | — | anchor 7 §2.5.5 baseline membership |
| in_30_baseline | bool | — | F6 reference resolution outcome |

**Row count**: 31 (full dSph sample).

### §2.4 `algorithm_b_log.csv` (per-iteration trace)

| Column | Type | Notes |
|---|---|---|
| galaxy | str | foreign key to `per_galaxy_sparc.csv` |
| iter_n | int | iteration index (0 = init) |
| c_galaxy_n | float | c_galaxy at this iteration |
| residual_n | float | residual at this iteration |
| converged_at_n | bool | True for the row where convergence occurred |

Used for §2.2 convergence rate analysis (mean iterations per galaxy, distribution).

### §2.5 `f_opt_anchor_table.csv` (5-anchor reference values)

| Column | Type | Notes |
|---|---|---|
| c_anchor | float | one of {0.30, 0.42, 0.618, 0.80, 1.00} |
| V_double_prime_at_x_half | float | V"(x=0.5, c) reference value |
| source | str | "TODO_USER_INPUT" pending fill (see EXECUTION_PLAN §10 TODO list item 1) |

Static reference; identical across runs once filled. Required for f_opt deg-4 Lagrange interpolation.

---

## §3 JSON aggregate schemas

### §3.1 `summary.json`

```json
{
  "run_id": "2026-05-03T14-30-00_aeca1c3",
  "config_sha": "TBD",
  "anchor_21_sha_referenced": "44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f",
  "input_shas": {
    "TA3": "TBD",
    "phase1": "TBD",
    "MRT": "TBD",
    "dSph": "TBD"
  },
  "wall_clock_seconds": 0.0,
  "n_galaxies_sparc_total": 175,
  "n_galaxies_sparc_inpool": 171,
  "n_galaxies_sparc_excluded": 4,
  "n_dsph": 31,
  "algorithm_b": {
    "n_converged": 0,
    "n_not_converged": 0,
    "convergence_rate": 0.0,
    "mean_n_iter": 0.0,
    "median_n_iter": 0.0,
    "max_n_iter": 0,
    "init_c_galaxy": 0.42,
    "tol": 1e-6,
    "N_max": 50
  },
  "f_E_adoption": {
    "k_E_default": 2,
    "n_selected_E": 0,
    "n_selected_B": 0,
    "f_E_fraction": 0.0,
    "delta_AIC_threshold": "TODO_USER_INPUT",
    "delta_AIC_mean": 0.0,
    "delta_AIC_median": 0.0
  },
  "C15": {
    "formula": "g_c = 0.584 * Ud^(-0.361) * sqrt(a_0 * Vflat^2 / hR)",
    "eta_0_locked": 0.584,
    "alpha_locked": 0.5,
    "beta_locked": -0.361,
    "scatter_dex_observed": 0.0,
    "scatter_dex_anchor7_baseline": 0.286,
    "R_squared": 0.0,
    "R_squared_anchor7_baseline": 0.607,
    "MOND_rejection_p_observed": 0.0,
    "MOND_rejection_p_anchor7_baseline": 1.66e-53
  },
  "dSph_J3": {
    "n_pass_observed": 0,
    "n_total": 31,
    "baseline_28_31": [28, 31],
    "baseline_30_31_F6_audit": [0, 31],
    "G_Strigari_over_a0_predicted": 0.227948,
    "G_Strigari_over_a0_observed_mean": 0.0,
    "agreement_pct": 0.0
  },
  "b_alpha": {
    "SPARC_observed": 0.0,
    "SPARC_anchor7_baseline": 0.1084,
    "dSph_observed": 0.0,
    "dSph_anchor7_baseline": 0.1127,
    "abs_diff_observed": 0.0,
    "abs_diff_anchor19_baseline": 0.0042,
    "agreement_pct": 0.0
  }
}
```

### §3.2 `sub_issues.json`

```json
{
  "S_1": {
    "category": "A: reference resolution",
    "linked_F_flag": "F2",
    "anchor_8_verbatim": "Phase C3 v3 §X Lesson 91 chapter placeholder (bridge pre-cut 4 galaxy reference)",
    "resolution_method": "literal reference resolution",
    "resolution_input_required": "Phase C3 v3 §X chapter number (X = TODO_USER_INPUT)",
    "resolution_status": "pending | resolved",
    "resolution_value": null
  },
  "S_2": {
    "category": "B: operational protocol",
    "linked_F_flag": null,
    "resolution_method": "TODO_USER_INPUT (per anchor 7 §2.5.6 [H-1])",
    "resolution_status": "pending",
    "resolution_value": null
  },
  "S_3": {
    "category": "A: reference resolution",
    "linked_F_flag": "F1",
    "anchor_8_verbatim": "parent v4.8 §6 line XXX (σ_g(r) observational uncertainty reference)",
    "resolution_method": "literal reference resolution",
    "resolution_input_required": "parent v4.8 §6 line number (XXX = TODO_USER_INPUT)",
    "resolution_status": "pending | resolved",
    "resolution_value": null
  },
  "S_4": {
    "category": "B: operational protocol",
    "resolution_method": "TODO_USER_INPUT",
    "resolution_status": "pending",
    "resolution_value": null
  },
  "S_5": {
    "category": "B: operational protocol",
    "resolution_method": "TODO_USER_INPUT",
    "resolution_status": "pending",
    "resolution_value": null
  },
  "S_6": {
    "category": "C: numerical threshold",
    "resolution_method": "numerical threshold determination from per-galaxy data",
    "threshold_target": "TODO_USER_INPUT",
    "threshold_observed": null,
    "resolution_status": "pending | resolved"
  }
}
```

### §3.3 `promotion.json`

```json
{
  "anchor_19_baseline": {
    "path_A_eta_0": 0.5629,
    "path_A_delta_pct": -3.61,
    "path_A_T6_pass": true,
    "path_B_eta_0": 0.5649,
    "path_B_delta_pct": -3.27,
    "path_B_T4_pass": true,
    "B_plus_to_A_prerequisite": "achieved"
  },
  "v0_2_observed": {
    "path_A_eta_0": 0.0,
    "path_A_delta_pct": 0.0,
    "path_A_T6_pass": null,
    "path_B_eta_0": 0.0,
    "path_B_delta_pct": 0.0,
    "path_B_T4_pass": null,
    "consistent_with_anchor_19": null,
    "drift_path_A": 0.0,
    "drift_path_B": 0.0
  },
  "A_grade_4_conditions": {
    "condition_1_paths_within_5pct": null,
    "condition_2_active_F_flags_zero": null,
    "condition_3_severity_matrix_NULL": null,
    "condition_4_SHA_chain_immutable": null,
    "all_pass": null
  },
  "anchor_21_chapter_level_consistency": null
}
```

### §3.4 `f_flag_status.json`

```json
{
  "F1": {
    "anchor_8_status": "placeholder 維持 (S-3 specific aspect)",
    "anchor_21_v0_1_1_status": "placeholder 維持 + §2.5 v0.2 round handoff",
    "v0_2_resolution_status": "pending | resolved",
    "linked_S": "S-3",
    "category": "A",
    "resolution_value": null
  },
  "F2": {
    "anchor_8_status": "placeholder 維持 (S-1 specific aspect)",
    "anchor_21_v0_1_1_status": "placeholder 維持 + §2.5 v0.2 round handoff",
    "v0_2_resolution_status": "pending | resolved",
    "linked_S": "S-1",
    "category": "A",
    "resolution_value": null
  },
  "F6": {
    "anchor_8_status": "placeholder 維持",
    "anchor_21_v0_1_1_status": "placeholder 維持 (handoff continuation, F1/F2 同質)",
    "v0_2_resolution_status": "pending | resolved",
    "category": "A",
    "resolution_value": null
  },
  "F8": {
    "anchor_8_status": "k_E=2 default LOCK 完了 (Q-C1)、k_E=1 sensitivity audit pending",
    "anchor_21_v0_1_1_status": "placeholder 維持 (Q-C1 LOCK 維持 + audit pending handoff)",
    "v0_2_resolution_status": "pending | resolved",
    "category": "B",
    "k_E_2_default_observed": null,
    "k_E_1_sensitivity_observed": null,
    "delta_AIC_k_E_2_minus_k_E_1": null,
    "Q_C1_LOCK_preserved": null
  }
}
```

### §3.5 `manifest.json` (forensic chain compliance, per anchor 20 §J)

```json
{
  "run_id": "2026-05-03T14-30-00_aeca1c3",
  "creator": "run_section2_5_v0_2.py",
  "creator_sha": "TBD",
  "execution_environment": {
    "platform": "Windows + Claude Code",
    "python_version": "TBD",
    "numpy_version": "TBD",
    "scipy_version": "TBD",
    "pandas_version": "TBD",
    "random_seed": null
  },
  "env_vars_used": {
    "MASTER_ROOT": "TBD",
    "PARENT_MASTER_ROOT": "TBD",
    "SPARC_TA3_PATH": "TBD",
    "SPARC_PHASE1_PATH": "TBD",
    "SPARC_MRT_PATH": "TBD",
    "DSPH_DATASET_PATH": "TBD",
    "OUTPUT_ROOT": "TBD"
  },
  "input_files": [
    {"role": "TA3", "path": "TBD", "sha256": "TBD", "size_bytes": 0},
    {"role": "phase1", "path": "TBD", "sha256": "TBD", "size_bytes": 0},
    {"role": "MRT", "path": "TBD", "sha256": "TBD", "size_bytes": 0},
    {"role": "dSph", "path": "TBD", "sha256": "TBD", "size_bytes": 0}
  ],
  "anchor_references": [
    {"anchor": 5, "sha_prefix": "3270fb40", "role": "§2.4 v0.1"},
    {"anchor": 6, "sha_prefix": "6ac356c3", "role": "§2.1-§2.3 v0.1"},
    {"anchor": 7, "sha_prefix": "9e03f53e", "role": "§2.5 v0.1 (predecessor)"},
    {"anchor": 8, "sha_prefix": "f6a48b51", "role": "§2.6 v0.1 chapter milestone"},
    {"anchor": 14, "sha_prefix": "295bc05c", "role": "§4 v0.4 Layer B-α/B-β + NGC 3198"},
    {"anchor": 16, "sha_prefix": "69678018", "role": "§5 v0.2.1 disambig"},
    {"anchor": 17, "sha_prefix": "178dad11", "role": "§3 v0.2 c_super=0.5709"},
    {"anchor": 19, "sha_prefix": "0b269c10", "role": "§1 v0.4 A 級 prerequisite"},
    {"anchor": 20, "sha_prefix": "56afa4c2", "role": "milestone summary"},
    {"anchor": 21, "sha_prefix": "44df9afb", "role": "§2 closure v0.1.1 chapter-level §2 LOCK"}
  ],
  "output_files": [
    {"path": "per_galaxy_sparc.csv", "sha256": "TBD", "size_bytes": 0},
    {"path": "per_galaxy_sparc_full.csv", "sha256": "TBD", "size_bytes": 0},
    {"path": "per_dsph.csv", "sha256": "TBD", "size_bytes": 0},
    {"path": "algorithm_b_log.csv", "sha256": "TBD", "size_bytes": 0},
    {"path": "f_opt_anchor_table.csv", "sha256": "TBD", "size_bytes": 0},
    {"path": "summary.json", "sha256": "TBD", "size_bytes": 0},
    {"path": "sub_issues.json", "sha256": "TBD", "size_bytes": 0},
    {"path": "promotion.json", "sha256": "TBD", "size_bytes": 0},
    {"path": "f_flag_status.json", "sha256": "TBD", "size_bytes": 0},
    {"path": "run_config.json", "sha256": "TBD", "size_bytes": 0},
    {"path": "run.log", "sha256": "TBD", "size_bytes": 0}
  ],
  "forensic_chain_compliance": {
    "rule_1_SHA_chain_index_updated": null,
    "rule_2_predecessor_anchors_cited": true,
    "rule_3_no_retroactive_change": true,
    "rule_4_forward_ref_0_strict": true,
    "rule_5_companion_pure_additive": true,
    "rule_6_closure_declaration_explicit": null,
    "rule_7_cross_check_audit_reruns": null
  }
}
```

---

## §4 Acceptance criteria evaluation in `summary.json`

`summary.json` MUST include an `"acceptance"` block (added on completion):

```json
{
  "acceptance": {
    "AC1_algorithm_b_convergence_rate_ge_threshold": {
      "threshold": "TODO_USER_INPUT (suggest >= 0.95)",
      "observed": 0.0,
      "pass": null
    },
    "AC2_f_E_adoption_in_expected_range": {
      "expected_range": ["TODO_USER_INPUT", "TODO_USER_INPUT"],
      "observed": 0.0,
      "pass": null
    },
    "AC3_dSph_J3_at_least_28_31": {
      "threshold_min": 28,
      "threshold_total": 31,
      "observed_pass": 0,
      "observed_total": 31,
      "pass": null
    },
    "AC4_b_alpha_abs_diff_le_0_005": {
      "threshold": 0.005,
      "observed_abs_diff": 0.0,
      "anchor_19_baseline_abs_diff": 0.0042,
      "pass": null
    },
    "AC5_S1_S6_all_resolved": {
      "S_1": null, "S_2": null, "S_3": null,
      "S_4": null, "S_5": null, "S_6": null,
      "all_pass": null
    },
    "AC6_Q_C1_LOCK_preserved_k_E_2_default": {
      "k_E_2_default_used": true,
      "delta_AIC_k_E_2_vs_k_E_1": 0.0,
      "k_E_2_remains_preferred": null,
      "pass": null
    },
    "AC7_anchor_21_chapter_level_consistency": {
      "F4_remains_RESOLVED": null,
      "F12_remains_RESOLVED": null,
      "F1_F2_F6_F8_resolution_via_S_handoff": null,
      "10_axis_audit_no_regression": null,
      "pass": null
    },
    "all_AC_pass": null,
    "v0_2_finalize_eligible": null
  }
}
```

`v0_2_finalize_eligible = true` only if `all_AC_pass = true`.

---

## §5 Anchor 22 (J_system_paper_section2_5_v0.2.md) integration

### §5.1 Paraphrase points

The following numerical results MUST be paraphrased into anchor 22 (bit-exact preservation):

| Source field | Anchor 22 location |
|---|---|
| `summary.json::C15.eta_0_locked` (= 0.584) | §2.5.x intro paragraph (verify unchanged) |
| `summary.json::algorithm_b.convergence_rate` | §2.5.3 Algorithm B convergence subsection |
| `summary.json::f_E_adoption.f_E_fraction` | §2.5.4 ΔAIC selection subsection |
| `summary.json::dSph_J3.{n_pass_observed, n_total}` | §2.5.5 dSph J3 subsection |
| `summary.json::b_alpha.{SPARC_observed, dSph_observed, abs_diff_observed}` | §2.5.5 b_α 3-axis audit subsection |
| `promotion.json::v0_2_observed.{path_A_eta_0, path_B_eta_0}` | §2.5.4 / cross to anchor 21 §8.1 |
| `sub_issues.json::S_*.resolution_value` | §2.5.6 [H-1] protocol subsection (S-1〜S-6 resolve) |
| `f_flag_status.json::F8.{k_E_2_default_observed, k_E_1_sensitivity_observed}` | §2.5.2 E pipeline subsection (k_E sensitivity audit) |

### §5.2 SHA hand-off

```
anchor 22 (§2.5 v0.2)
  └── references manifest.json (run_id, creator_sha, all output SHAs)
  └── references anchor 21 v0.1.1 (44df9afb...) chapter-level §2 LOCK ESTABLISHED
  └── extends 21-anchor SHA chain → 22-anchor chain (anchor 02 still NOT_FOUND superseded)
  └── companion-internal F1/F2/F6/F8 status update from "placeholder + handoff" → "RESOLVED with v0.2 numerical resolution_value"
  └── anchor 7 (§2.5 v0.1, 9e03f53e) IMMUTABLE retained (R-1 LOCK preserved, anchor 22 純 additive supersession)
```

### §5.3 Bit-exact reproducibility

Anchor 22 paraphrase must use the EXACT decimal representation as in source JSON. Example:

- ✅ `convergence rate = 0.953216...` (from `summary.json`)
- ❌ `convergence rate ≈ 0.95` (rounded; introduces ambiguity)

Float precision policy: write JSON values with 6 significant digits (configurable in `run_config.json::output_float_precision`).

---

## §6 Verification protocol

After execution:

1. **Schema conformance check** — every output file matches the schema in this document
2. **SHA256 self-verification** — `SHA256SUMS.txt` reproduces by re-hashing all output files
3. **Cross-reference check** — every SHA in `manifest.json::input_files` matches the actual file on disk
4. **Acceptance criteria evaluation** — `summary.json::acceptance.all_AC_pass` is set
5. **Forensic chain compliance** — `manifest.json::forensic_chain_compliance` 7-item all PASS or NULL-with-rationale

If any of these fail, the run output directory is marked `FAILED_<reason>/` (not consumed by anchor 22).

---

## §7 Reproducibility notes

- `random_seed` set in `run_config.json` (default: 42)
- All package versions pinned in `manifest.json::execution_environment`
- Float precision deterministic (no `np.random` without seed; no parallel `numpy` ops with non-deterministic reduction)
- Same input SHAs + same config + same seed ⇒ same output SHAs (bit-exact reproducibility goal)

---

## Appendix A: Schema version history

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-05-03 | Initial schema (alongside anchor 21 v0.1.1 chapter-level §2 LOCK ESTABLISHED) |

---

**END OUTPUT_SCHEMA_section2_5_v0_2.md v1.0**
