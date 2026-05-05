# v0.1.4 - Phase 1a-validation closure (12/12 PASS)

**Tag**: `companion-v0.4a-validation-2026-05-05`
**Date**: 2026-05-05
**Canonical aggregate**: `forensic_anchors/section2_5_v0_4a_validation/validation_results.json`
**SHA256**: `b6012483ce400f466d9e8a87c4025ff23270a4cb11fd465388da16e19d8f03df`
**Size**: 13,053 B

## Scope

Phase 1a-validation completion of (C) round 3-phase split:

- **Phase 1a** (commit 52ffc09, anchor 21 v0.1.3): mechanical alignment via 1-line Q-patch
- **Phase 1a-validation** (this commit, anchor 21 v0.1.4): statistical + universal coupling + robustness
- **Phase 1b** (deferred, anchor 22): f_opt(x != 0.5) operational + chi_coh integration

Circularity-free design preserved: Phase 1a-validation operates on **frozen Phase 1a output** (canonical SHA `cdca6afd...`, 137,861 B) as input. b_alpha values verified against anchor 19 sec 1.5 IMMUTABLE baseline (0.1084 / 0.1127 / 0.0042) post-Phase-1a-mechanical-closure, preventing tautological self-validation.

## Gate breakdown (12/12 PASS)

### G layer (statistical reproducibility): 6/6 PASS

| gate | metric | observed | criterion | status |
|---|---|---|---|---|
| G1 | axis_1_SPARC b_alpha | 0.108443 | \|b - 0.1084\| < 1e-3 | PASS (Δ=4.30e-5, 23x margin) |
| G2 | axis_2_dSph b_alpha | 0.112662 | \|b - 0.1127\| < 1e-3 | PASS (Δ=3.76e-5, 27x margin) |
| G3 | axis_3 universal slope | 0.110553 | in [0.105, 0.115] | PASS (target band 0.110-0.111) |
| G4 | Lesson 93 (per-dex) | 0.974% | < 1% | PASS (driving: secondary_per_dex) |
| G5 | AC4 abs_diff | 0.004219 | < 0.005 | PASS (16x to baseline 0.0042) |
| G6 | AC5 universal | within_tol=True + R²=0.634 + AIC=-529.92 | metrics finite | PASS |

### J layer (robustness, post (γ) split): 6/6 PASS

| gate | metric | observed | criterion | status |
|---|---|---|---|---|
| J1a | per-LOO fluctuation | std_loo=0.001389 | < 0.005 | PASS (3.6x margin) |
| J1b | OLS SE relative precision | SE/\|b_alpha\|=0.141 | < 0.20 | PASS (1.41x margin, t-stat=7.07) |
| J1c | outlier influence | n_outliers=0 | == 0 | PASS |
| J2 | bootstrap CI | 95% CI [0.0790, 0.1388] | contains baseline 0.1084 | PASS (10,000 resamples, seed=42) |
| J3 | per-filter sensitivity | 4/4 executable scenarios | all PASS | PASS |
| J4 | phase_c3_step3 cross-check | Δ=0.000e+00 (bit-exact) | < 1e-12 | PASS |

## Key forensic findings

### J3 filter 1 = Q-patch causal proof (direct demonstration)

Dropping Q column from sparc_for_audit reproduces v1.0.3.1 state **bit-exact**:

| state | n | b_alpha | drift |
|---|---|---|---|
| baseline (Q applied) | 124 | 0.108443 | 0% |
| **filter 1 disable (Q dropped)** | **129** | **0.112356** | **+3.61%** |
| v1.0.3.1 documented | 129 | 0.11236 | +3.6% |

The 1-line patch (`load_MRT` L587, `, "Q"` added, +5 bytes) is the **unique causal mechanism**. No interaction term, no compound effect. Re-applying Q yields baseline state. This satisfies falsifiable causal proof of Phase 1a's diagnosis.

### J4 multi-route minimum (forensic chain rule #26)

Two **independent code paths** yield bit-exact identical result:

| code path | axis_1_SPARC b_alpha |
|---|---|
| canonical (cdca6afd, post-Q-patch) | 0.108442979149252 |
| phase_c3_step3 reference (standalone Q-intact loader, no canonical import) | 0.108442979149252 |
| Δ | **0.000e+00** |

Result independent of:
- loader implementation (canonical vs standalone)
- filter chain organization (function-internal vs explicit linear)
- module boundaries (no import dependency)

Forensic chain rule #26 (multi-route minimum: A + B-1/B-2/B-3/B-4 + C) satisfied with strict bit-exact agreement.

### J1 (γ) split universal coupling significance

| metric | value |
|---|---|
| sigma_jackknife (axis_1) | 0.015338 |
| bootstrap std (axis_1) | 0.015349 |
| ratio bootstrap/jackknife | 1.0007 (textbook OLS SE concordance) |
| sigma_jackknife (axis_2 dSph) | 0.036325 |
| SE_combined = sqrt(SE_1² + SE_2²) | 0.039428 |
| \|sparc - dsph\| | 0.004219 |
| **\|diff\| / SE_combined** | **0.107 σ** |
| t-statistic for axis_1 (b/SE) | 7.07 |

**Interpretation**: SPARC and dSph slopes are **statistically indistinguishable at noise-floor scale** (0.107σ). Universal coupling claim is consistent at both:
- absolute scale (G5: \|diff\| < 0.005 AC4 absolute bound)
- noise-floor scale (J1b auxiliary: \|diff\|/SE_combined ≈ 0.1σ)

The dual evidence strengthens the Lesson 93 universal coupling assertion beyond the single absolute-bound check.

## Side findings (documented for future audit)

### filter 4 corrective: v_flat>0 not moot

Original prediction: sparc_full has 0 rows with v_flat ≤ 0 (filter trivially satisfied).
**Observed**: sparc_full has **40 rows with v_flat ≤ 0**. Filter is functional in real data; canonical's defensive `delta_primary` finite check at L2005 catches them indirectly (gc_C15=0 → log10(0)=-inf → filtered). Reference impl's explicit `v_flat > 0` (L262) is therefore meaningful, not redundant.

### filter 3 sensitivity: g_obs NaN imputation drift

Filling NaN g_obs with median (vs default exclusion via finite filter) yields **−9.53% drift** in b_alpha. 11 galaxies (8.9% sample) have NaN g_obs (insufficient rotation curve data); their imputed presence shifts the OLS substantially. Lesson 94 candidate: **g_obs aggregation strategy is a sensitivity hotspot**, deserves explicit documentation in operational protocols.

## Forensic chain rule compliance

| rule | description | status |
|---|---|---|
| 1 | anchor IMMUTABLE preserved | ✅ anchor 19 sec 1.5 baseline (0b269c10) untouched; v0.1.3 canonical (cdca6afd) frozen |
| 2 | R-1 LOCK (k_B = 0) | preserved |
| 3 | R-2 LOCK (Algorithm B self-consistency) | preserved |
| 4 | Q-C1 LOCK (k_E = 2 default) | preserved |
| 5 | cascade SSoT | preserved (`vpp_x05(0.83)=10.462625 / f_opt(0.83)=1.942493`) |
| 6 | L-1 forward-ref 0 strict | parent v4.8 NULL impact |
| 7 | companion additive supersession | ESTABLISHED (parent commit 52ffc09 IMMUTABLE) |
| 26 | multi-route minimum | ✅ J4 bit-exact agreement (canonical + phase_c3_step3 reference) |

### Anchor IMMUTABLE verification

| target | SHA prefix | status |
|---|---|---|
| anchor 21 v0.1.3 (commit 52ffc09, frozen canonical) | `cdca6afd...` (canonical), `44df9afb...` (v0.1.1 root) | unchanged |
| anchor 19 sec 1.5 baseline | `0b269c10...` | unchanged |
| `forensic_anchors/section2_5_v0_2_skeleton/` (commit 8e8ed51, v1.0.2) | `dd762fd2...` | unchanged |
| `forensic_anchors/section2_5_v0_3_step4_reproducibility/` (commit 46ea829, v1.0.3.1) | `7cb540b1...` | unchanged |
| `forensic_anchors/section2_5_v0_4a_axis_1_alignment/run_section2_5_v0_2.py` (v1.0.4a) | `cdca6afd...` | unchanged (frozen Phase 1a output) |

### Supersession declaration

> Anchor 21 v0.1.3 (commit 52ffc09, canonical SHA `cdca6afd...`) is preserved IMMUTABLE at `forensic_anchors/section2_5_v0_4a_axis_1_alignment/`. Anchor 21 v0.1.4 (this commit) at `forensic_anchors/section2_5_v0_4a_validation/` is its companion-additive successor for **statistical + universal coupling + robustness closure**, without modifying the v0.1.3 file or any anchor.

## Files in this release

| file | size | SHA256 |
|---|---|---|
| `RELEASE_NOTES.md` | (this file) | (computed at commit-time, see SHA256SUMS) |
| `run_validation_bench.py` (G1-G3 driver) | 14,386 B | `8c7828a4944637cd8cc8206a33bfd598dc2ac69e5262f6847f4fdff635e5782f` |
| `run_validation_bench_g4_g6.py` (G4-G6 driver) | 17,825 B | `f4d2e52e83444cd37cb0a6dbe19ce6bda23288e0e82dcb61db0a7fbcf469a0ae` |
| `run_validation_bench_j_layer.py` (J1 initial + J2 + J3 stub + J4 driver) | 22,460 B | `e2e9115b0ac38007c733dee19cf9f502a168067c3186a37ce5b81e650c66ad8d` |
| `run_validation_bench_j_supplement.py` (J1 (γ) split + J3 5-filter) | 25,115 B | `8b9e250a7e8fb0a6df5edf3b6ffbe3950e4f5735d6caf3a9faf35694c6a5ac88` |
| `validation_results.json` (G1-J4 final 12/12 PASS aggregate) | 13,053 B | `b6012483ce400f466d9e8a87c4025ff23270a4cb11fd465388da16e19d8f03df` |
| `jackknife_axis_1_results.csv` (J1 124 LOO with metadata) | 19,000 B | `28886a65ca3de677442813eacdcdab560b640e499235c40c2bcb4308a16bb955` |
| `bootstrap_axis_1_results.csv` (J2 10,000 resamples seed=42) | 257,059 B | `5568321529c97f6806ca1a96870d5f8f2e641043a6619d5922eff4a4e4328627` |
| `per_filter_sensitivity.csv` (J3 baseline + 5 filter scenarios) | 2,022 B | `6cb3ee1524658782c76fd0a525fe0925cb67fe134922fcc15121b92f6bb9ddfe` |
| `phase_c3_step3_reference.json` (J4 independent reference) | 1,385 B | `6c8984d380ad8a5e86f532010945303bd63683b1ffc264d07e15b6d69337995e` |
| `cross_check_phase_c3_step3.json` (J4 comparison output) | 1,607 B | `a30466929ef183c83229630ee90e99bb3b28d0f5b0e4ec98cb62b2e2bcce12b8` |

### Repo root (modified)

- `.gitattributes` — `forensic_anchors/section2_5_v0_4a_validation/** -text` rule appended
- `ANCHOR_REFERENCES.md` — anchor 21 v0.1.4 entry appended
- `SHA256SUMS` — 11 new entries (this release block)

## Lesson compliance

- **Lesson 91** (bridge / extreme-regime pre-cut protocol): Phase 1a restored, Phase 1a-validation J3 filter 1 confirms causal mechanism
- **Lesson 92** (parsimony first): 1-line root cause definitively confirmed via direct demonstration; alternate hypotheses (5 candidates) all falsified
- **Lesson 93** (universal coupling slope agreement < 1%): G4 PASS at 0.974%, J1b auxiliary at 0.107σ noise-floor confirmation; **dual evidence strengthens claim**
- **forensic chain rule 1** (anchor IMMUTABLE): preserved, all prior anchors untouched
- **forensic chain rule 7** (companion additive supersession): explicit declaration above
- **forensic chain rule #26** (multi-route minimum): J4 bit-exact independent code path agreement

## Lessons added (P1aV-A through P1aV-E, candidate)

- **P1aV-A**: G layer / J layer separation enables circularity-free statistical validation (G = baseline-match, J = data-internal robustness)
- **P1aV-B**: σ definition disambiguation (std_loo vs jackknife SE) — both valid, threshold definition must specify which
- **P1aV-C**: filter sensitivity = causal mechanism — disabling a filter and observing bit-exact reproduction of prior bug-state is direct demonstration
- **P1aV-D**: multi-route minimum requires genuinely independent code paths (different loaders, different module boundaries), not mimicking
- **P1aV-E**: SE_combined-normalized universal coupling significance complements absolute-bound AC4

## Closure status

- Phase 1a (mechanical): **CLOSED** (anchor 21 v0.1.3, commit 52ffc09)
- Phase 1a-validation (statistical + robustness): **CLOSED** (this commit, anchor 21 v0.1.4)
- Phase 1b (`f_opt(x != 0.5)` operational): **DEFERRED** (separate round, anchor 22)
- Promotion to anchor 21 v0.2 (full operational closure): **deferred to separate chat** (operational closure / paper editing distinct thinking mode)

## Anchor record

```
anchor 21 v0.1.4 (Phase 1a-validation closure, statistical + universal coupling + robustness)
  canonical aggregate: forensic_anchors/section2_5_v0_4a_validation/validation_results.json
  SHA256: b6012483ce400f466d9e8a87c4025ff23270a4cb11fd465388da16e19d8f03df
  size:   13,053 B
  parent: anchor 21 v0.1.3
          commit 52ffc09, frozen canonical SHA cdca6afd...
          (Phase 1a mechanical alignment, IMMUTABLE preserved)
  scope:  Phase 1a-validation 12/12 PASS
          (G layer 6/6: G1-G6 statistical reproducibility;
           J layer 6/6: J1a/J1b/J1c/J2/J3/J4 robustness post (γ) split)
  status: closed for Phase 1a-validation; promotion to v0.2 deferred
  rule 1: parent anchor 21 v0.1.3 IMMUTABLE preserved (cdca6afd untouched)
  rule 7: companion-additive successor to v0.1.3 (commit 52ffc09)
  rule 26: multi-route minimum satisfied (J4 bit-exact agreement)
  promotion path: v0.1.4 -> v0.2 (full operational closure) deferred to new chat
                  (operational closure / paper editing distinct thinking mode)
```

## Roadmap

| phase | scope | timing | anchor |
|---|---|---|---|
| 1a (commit 52ffc09) | mechanical alignment (Q-patch + 5 gates) | done | 21 v0.1.3 |
| **1a-validation (this commit)** | **statistical + universal coupling + robustness** | **done** | **21 v0.1.4** |
| 1b | `f_opt(x != 0.5)` operational + chi_coh integration | deferred | 22 |
| 2 | full operational closure | post-1b, separate chat | 21 v0.2 |

---

## POST-HOC CROSS-REFERENCE 2 (anchor 21 v0.2 supersession annotation)

Added in commit C2 (anchor 21 v0.2 round, 2026-05-05) post-hoc to record
forward continuity to anchor 21 v0.2 axis_1 full operational closure.
forensic chain rule 1 (anchor IMMUTABLE): all content above this section
is unchanged; this annotation is additive only.

This is the second POST-HOC CROSS-REFERENCE in this anchor's RELEASE_NOTES.md
(the first was added in commit 3817b41 cross-referencing anchor 21 v0.1.3).
The two annotations form a chronological forward-continuity chain:
  v0.1.3 (cdca6afd, mechanical alignment)
    -> v0.1.4 (this anchor, statistical + universal coupling + robustness)
    -> v0.2 (axis_1 full operational closure, this annotation)

### Elevation to axis_1 full operational closure

The 12/12 gate PASS findings recorded in this anchor (v0.1.4) have been
elevated to **axis_1 full operational closure** status as anchor 21 v0.2:

  - declaration: `forensic_anchors/section2_axis_1_operational_closure/OPERATIONAL_CLOSURE.md`
  - round narrative: `forensic_anchors/section2_axis_1_operational_closure/RELEASE_NOTES.md`
  - machine-readable aggregate: `forensic_anchors/section2_axis_1_operational_closure/axis_1_closure_summary.json`
  - cross-method evidence: `forensic_anchors/section2_axis_1_operational_closure/per_filter_sensitivity_extension.{py, csv}`
  - v1 -> v2 redesign forensic record: `forensic_anchors/section2_axis_1_operational_closure/v1_precondition_fail_diagnostic.log`
  - tag: `companion-v4.9-axis-1-closure-2026-05-05` (annotated)
  - parent commit (C1, feat+tag): 8071c71
  - this commit (C2, post-hoc docs): [C2-SHA-here]

### Canonical references (post-v0.2)

The following canonical IMMUTABLE values from this anchor (v0.1.4) are
inherited by v0.2 and become the canonical reference set for arXiv v4.9
§7.6 / §7.7 / Appendix A:

  axis_1_SPARC b_α       = 0.108442979149252  (G1, J4 bit-exact)
  axis_2_dSph  b_α       = 0.11266236254145712 (G2)
  axis_3 universal slope = 0.11055267084535411 (G3)
  abs_diff (universal)   = 0.004219            (G5)
  R² / residual / AIC    = 0.634 / 0.172 / -529.92  (G6)
  OLS SE                 = 0.0153              (J1b)
  bootstrap 95% CI       = [0.0790, 0.1388]    (J2)
  J3 filter 1 reproduces v1.0.3.1 bit-exact (Q-patch causal proof)
  J4 multi-route Δ       = 0.000e+00           (rule #26)
  SE_combined            = 0.0394
  noise-floor sig        = 0.107σ              (J1b auxiliary)

### Lesson 94 formal establishment

J3 filter 3 finding (g_obs aggregation NaN imputation drift -9.53%) was
recorded in this anchor as "Lesson 94 candidate". v0.2 round で full
establishment 達成:

  - shorthand: "g_obs aggregation sensitivity"
  - 5-property structure (scope / evidence / principle / scope LIMIT /
    falsification path) recorded in OPERATIONAL_CLOSURE.md §4.4
  - cross-method confirmation: 4-method aggregation/imputation comparison
    via per_filter_sensitivity_extension.csv (anchor 21 v0.2)
      A1 median(V²/r) aggregation : b_α = 0.070903 (-34.62%)
      A2 min-fill imputation       : b_α = 0.128639 (+18.62%)
      A3 mean-fill imputation      : b_α = 0.094526 (-12.83%)
      A4 k=5 KDTree impute         : b_α = 0.101052 ( -6.82%)
    vs A0 baseline 0.108443
  - dominance ordering: aggregation choice (-34.62%) >> imputation choice
    (±20% range)
  - rule #26 multi-route minimum compliance: extended via cross-method
    (1-instance evidence base from this anchor's J3 filter 3 -> 5-method
     evidence base in v0.2)

### v1 -> v2 redesign forensic record (educational)

The seq-3 cross-method script (per_filter_sensitivity_extension.py) underwent
a v1 -> v2 redesign during v0.2 round development. Two structural mismatches
in v1 were detected by the precondition Q5 β two-axis check (b_α bit-exact
+ OLS SE):

  Mismatch 1 (PRIMARY): 2-feature simple OLS produced RAR-like slope ~1.2
                        instead of canonical 3-feature partial OLS rho_gal²
                        scaling exponent ~0.108
  Mismatch 2 (SECONDARY): post-filter g_obs aggregation exposed only 3 NaN
                          vs canonical sparc_171 ordering 11 NaN

v2 corrected to 3-feature partial OLS + sparc_171 ordering, achieving
precondition PASS at delta = 4.16e-17 (machine epsilon). The v1 diagnostic
log is retained at section2_axis_1_operational_closure/v1_precondition_fail_diagnostic.log
as educational forensic record (1-shot redesign, not 0-shot).

### forensic chain rule compliance (post-v0.2 annotation)

  rule 1 (anchor IMMUTABLE)        : ✅ above content untouched
                                       (only this annotation appended)
  rule 7 (additive supersession)   : ✅ v0.2 elevates without revision
  rule #26 (multi-route minimum)   : ✅ retained from v0.1.4 J4
                                       + extended via Lesson 94 cross-method
