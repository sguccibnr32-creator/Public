# Anchor References - sguccibnr32-creator/Public

SHA-pinned references for J-system companion paper section anchors.

---

## anchor 21 v0.1.2 (section 2.5 v0.2 step 4 reproducibility addendum)

```
source: forensic_anchors/section2_5_v0_3_step4_reproducibility/run_section2_5_v0_2.py
SHA256: 7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652
size:   137,856 B
parent: anchor 21 v0.1.1
        SHA 44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f
        (section 2 closure declaration, IMMUTABLE)
scope:  Step 4 - 5 inline patches reproducibility addendum
        (loader / derivation / aggregation fix only; no claim-kernel touch)
status: provisional - axis_1_SPARC alignment pending v1.0.4 (C) round
        (anchor 19 sec 1.5 baseline 0.1084 vs observed 0.11236, 3.6% rel)
rule 1: parent anchor 21 v0.1.1 (44df9afb) IMMUTABLE preserved
rule 7: companion-additive successor to v1.0.2 (dd762fd2, commit 8e8ed51)
        v1.0.2 retained IMMUTABLE at forensic_anchors/section2_5_v0_2_skeleton/
promotion path: v1.0.4 (C) axis_1 alignment 達成時に anchor 21 v0.2
                (full operational closure) として promote、本 v0.1.2 は
                historical retain
```

### Patch chain (v1.0.2 -> v1.0.3.1)

5 inline patches in 1 batch (Step 4 numerical reproducibility hardening):

| # | scope                                     | summary                                                |
|---|-------------------------------------------|--------------------------------------------------------|
| 1 | `load_phase1` L515-531 (T10)              | UTF-8-sig + BOM strip                                  |
| 2 | `load_MRT` L537-587 (T11)                 | whitespace-delim, skiprows=98, 18-col raw schema       |
| 3 | `_prepare_dsph_phase_c3_sample` L2029     | LV->M_bar / rh_pc->r_h / rho_gal derivation            |
| 4 | `col_alias_map` L3068-3070                | capital `Ud->Upsilon_d` (load_phase1 default)          |
| 5 | SPARC `g_obs` aggregation L3043-3061      | `mean(V^2/r)` for `r > 2*hR` (T16.F verbatim)          |

### Acceptance gates

| metric                  | observed   | criterion       | status                |
|-------------------------|------------|-----------------|-----------------------|
| `axis_2_dSph`           | 0.11266    | bit-exact       | PASS                  |
| `axis_3_universal`      | 0.11251    | within +/-1sigma| PASS                  |
| `abs_diff`              | 0.000306   | < 0.005 (AC4)   | PASS (16x margin)     |
| Lesson 93 slope agree   | 0.272% rel | < 1%            | PASS (3.7x stricter)  |
| `axis_1_SPARC`          | 0.11236    | (deferred)      | provisional           |

---

## anchor 21 v0.1.3 (section 2.5 v0.4a axis_1 mechanical alignment)

```
source: forensic_anchors/section2_5_v0_4a_axis_1_alignment/run_section2_5_v0_2.py
SHA256: cdca6afd634a3f730bc4b4002ab3082e187dc64b4ad9e87fd8147a6d81b04521
size:   137,861 B
parent: anchor 21 v0.1.2 (44df9afb..., commit 46ea829, IMMUTABLE preserved)
scope:  axis_1_SPARC mechanical alignment via 1-line Q-column patch
        (load_MRT L587 return statement, +5 bytes)
status: provisional - statistical validation pending Phase 1a-validation
        (anchor 21 v0.1.4)
rule 1: parent anchor 21 v0.1.2 IMMUTABLE preserved
rule 7: companion-additive successor to v1.0.3.1 (7cb540b1, commit 46ea829)
        v1.0.3.1 retained IMMUTABLE at section2_5_v0_3_step4_reproducibility/
        + paranoid retain at .bak.py inside this directory
promotion path: v1.0.4a-validation 完了 -> anchor 21 v0.1.4
                v1.0.4a-validation passes anchor 19 sec 1.5 baseline match
                + Lesson 93 + AC4/AC5 -> anchor 21 v0.2 (full operational
                closure) として promote、本 v0.1.3 は historical retain
```

### Patch detail (v1.0.3.1 -> v1.0.4a)

```diff
- return df[["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat"]].copy()
+ return df[["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat", "Q"]].copy()
```

1 line / 1 column / 5 bytes added. Activates Q<3 cut at
`_prepare_sparc_phase_c3_sample` L1978 which was silently skipped due to
missing Q column.

### Mechanical alignment gates (5/5 PASS)

| # | gate | result |
|---|---|---|
| 1 | Q column in `load_MRT` return | PASS |
| 2 | Q<3 cut active (per-galaxy Q in [1,2]) | PASS |
| 3 | `sample_n_axis_1` == 124 (binary) | PASS |
| 4 | galaxy identity vs phase_c3_step3 reference (sym diff = 0) | PASS |
| 5 | `sample_n_axis_3` == 154 (= 124 + 30) | PASS |

### Circularity-free design

Phase 1a success criterion does NOT reference b_alpha values. b_alpha
verification (axis_1_SPARC vs anchor 19 sec 1.5 baseline 0.1084,
Lesson 93, AC4/AC5, jackknife) is performed in **Phase 1a-validation**
(separate round, frozen Phase 1a output as input).

### Hypothesis correction (post-hoc)

Commit 46ea829 RELEASE_NOTES enumerated 5 candidate root causes for the
3.6% axis_1_SPARC deviation; forensic verify (this round) shows all 5
are non-causal. Actual root cause is a single-line column subset bug
(load_MRT return statement Q drop). See section2_5_v0_4a_axis_1_alignment/
RELEASE_NOTES.md for full forensic trace.

---

## anchor 21 v0.1.4 (Phase 1a-validation closure, 2026-05-05)

- canonical aggregate: forensic_anchors/section2_5_v0_4a_validation/validation_results.json
- SHA256: b6012483ce400f466d9e8a87c4025ff23270a4cb11fd465388da16e19d8f03df
- size: 13,053 B
- parent: anchor 21 v0.1.3 (commit 52ffc09, frozen canonical cdca6afd)
- tag: companion-v0.4a-validation-2026-05-05
- status: 12/12 PASS (G1-G6 + J1a/J1b/J1c + J2 + J3 + J4)
- key findings: J3 filter 1 = Q-patch causal proof (bit-exact reproduction of v1.0.3.1 state); J4 multi-route minimum (delta=0.000e+00); |sparc-dsph|/SE_combined=0.107 sigma
- promotion path: v0.1.4 -> v0.2 (full operational closure, deferred to separate chat)

---

## anchor 21 v0.2 (axis_1 full operational closure, 2026-05-05)

- canonical declaration: forensic_anchors/section2_axis_1_operational_closure/OPERATIONAL_CLOSURE.md
- canonical aggregate (JSON): forensic_anchors/section2_axis_1_operational_closure/axis_1_closure_summary.json
- SHA256 (declaration): 1c6d19ad77da2a3c7a489da62a541d0c5a538217dbb4e8f7f6fca4a99d1e65b6
- SHA256 (aggregate JSON): 55603a4d1db9c62203db68d2794ce01264715686890856df6a95f82e2b6eacce
- size (declaration): ~13,000 B (post stub-fill final)
- size (aggregate JSON): ~16,000 B (post stub-fill final)
- parent: anchor 21 v0.1.4 (commit 5783bef, tag companion-v0.4a-validation-2026-05-05)
- commit C1 (feat + tag): [C1-SHA-here]
- commit C2 (post-hoc xref): [C2-SHA-here]
- tag: companion-v4.9-axis-1-closure-2026-05-05 (annotated)
- status: CLOSED (operational layer, 6 file deliverable + 2 root modifications)
- inherited canonical values: axis_1_SPARC b_α = 0.108442979149252 (G1/J4 bit-exact); OLS SE = 0.0153 (J1b); universal coupling noise-floor 0.107σ
- key findings: Lesson 94 formal establishment (g_obs aggregation sensitivity, dominant effect aggregation choice -34.62% >> imputation choice ±20%); rule #26 multi-route minimum extended via 4-method cross-method comparison; v1 -> v2 redesign forensic record retained
- promotion path: arXiv v4.9 §7 paper edit (C3+, this round) + WordPress sync (separate round); next operational target: Phase 1b (anchor 22, f_opt(x≠0.5) operational + chi_coh integration)

### Lesson 94 establishment (g_obs aggregation sensitivity)

| property | value |
|---|---|
| shorthand | g_obs aggregation sensitivity |
| scope | SPARC rotation curve g_obs aggregation method choice |
| evidence (primary) | J3 filter 3, anchor 21 v0.1.4 (NaN imputation drift -9.53%) |
| evidence (cross-method) | per_filter_sensitivity_extension.csv (this round, 4-method) |
| principle | aggregation strategy substantially affects b_α point estimate; explicit NaN-handling protocol required; NaN g_obs not random, systematic bias direction |
| scope LIMIT | SPARC g_obs aggregation only (no extrapolation to dSph J3 / generic stat aggregation) |
| falsification path | alternative aggregation (median V²/r, k-NN, ML imputation) → drift indistinguishable within OLS SE |
| established | this round (anchor 21 v0.2) |

### Cross-method extension (Lesson 94 evidence base)

| scenario | n | n_imputed | b_α | SE | Δ vs baseline |
|---|---|---|---|---|---|
| A0_baseline | 124 | 0 | 0.108443 | 0.0158 | 0.0% |
| A1_median_V2r | 124 | 0 | 0.070903 | 0.0174 | -34.62% |
| A2_min_fill | 127 | 3 | 0.128639 | 0.0219 | +18.62% |
| A3_mean_fill | 127 | 3 | 0.094526 | 0.0167 | -12.83% |
| A4_knn_impute | 127 | 3 | 0.101052 | 0.0161 | -6.82% |

OLS structure: 3-feature partial OLS, target = log10(g_obs) - log10(g_C15),
features = [1, 2·log10(rho_gal), log10(r_h)], b_α = beta[1].
Precondition Q5 β PASS: A0 b_α delta = 4.16e-17 (machine epsilon), SE delta = 5.47e-4 (within tol 1e-3).

### Closure scope statement

**What is closed (operational layer):**
(a) point estimate bit-exact reproducible from frozen canonical (cdca6afd) via independent code paths (J4, rule #26)
(b) statistical reproducibility verified at 12/12 acceptance gates (v0.1.4)
(c) Q-patch (load_MRT L587 single-line) is unique sufficient causal mechanism for v1.0.3.1 → v1.0.4a +3.6% deviation (J3 filter 1 bit-exact)
(d) universal coupling vs axis_2_dSph holds at noise-floor scale (0.107σ)
(e) methodology principles formalized as Lesson 91-94

**What is NOT closed by this declaration:**
- axis_2 (dSph) operational hardening — separate round
- axis_3 (universal cross-axis) operational hardening — separate round
- Phase 1b (f_opt(x ≠ 0.5) operational + chi_coh integration) — anchor 22
- arXiv v4.9 §7 paper edit reflection — v0.2 round 内 C3+ commits
- WordPress site sync (sakaguchi-physics.com) — separate round

### Forensic chain rule compliance (9/9 PASS)

| rule | status | note |
|---|---|---|
| 1 (anchor IMMUTABLE) | ✅ | v0.1.4 untouched, C2 post-hoc additive |
| 2 (R-1 LOCK k_B=0) | ✅ | not modified |
| 3 (R-2 LOCK Algo B) | ✅ | not modified |
| 4 (Q-C1 LOCK k_E=2) | ✅ | not modified |
| 5 (cascade SSoT) | ✅ | vpp_x05(0.83)=10.462625, f_opt(0.83)=1.942493 preserved |
| 6 (L-1 forward-ref 0) | ✅ | parent v4.8 NULL impact maintained |
| 7 (additive supersession) | ✅ | v0.2 elevates v0.1.4 without revision |
| #26 (multi-route minimum) | ✅ | J4 inherited + Lesson 94 cross-method extension |
| 92 (parsimony first) | ✅ | Lesson 94 + §7.7 + cross-method in single round |

---
