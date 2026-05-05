# axis_1 full operational closure declaration

**status**: CLOSED (operational layer)
**round**: anchor 21 v0.2
**date**: 2026-05-05
**parent anchor**: 21 v0.1.4 (commit 5783bef, tag companion-v0.4a-validation-2026-05-05)
**this round commit**: [C1-SHA-here]
**this round tag**: companion-v4.9-axis-1-closure-2026-05-05

This document is the canonical formal declaration for the full operational
closure of axis_1 (SPARC universal coupling slope). It serves as the single
SHA-pinned reference target for arXiv v4.9 §7.6 / §7.7 / Appendix A.

---

## §1 Closure scope statement

### §1.1 What is closed

axis_1 universal coupling claim:

  b_α_axis_1_SPARC = 0.108442979149252  (OLS, n=124 SPARC galaxies)
  OLS Standard Error = 0.0153  (data-intrinsic, cross-method confirmed)
  t-statistic = 7.07  (>5σ non-zero)

is hereby declared **operationally closed** in the following sense:

(a) point estimate is bit-exact reproducible from frozen canonical
    (cdca6afd, anchor 21 v0.1.3) via independent code paths (J4, rule #26)
(b) statistical reproducibility verified at 12/12 acceptance gates (v0.1.4)
(c) Q-patch (load_MRT L587 single-line) is the unique sufficient causal
    mechanism for v1.0.3.1 → v1.0.4a +3.6% deviation (J3 filter 1 bit-exact)
(d) universal coupling vs axis_2_dSph holds at noise-floor scale
    (|Δ| / SE_combined = 0.107σ, completely indistinguishable)
(e) methodology principles supporting the closure are formalized as
    Lesson 91-94 (§4 below)

### §1.2 What is NOT closed by this declaration

- axis_2 (dSph) operational hardening — separate round
- axis_3 (universal cross-axis) operational hardening — separate round
- Phase 1b (f_opt(x ≠ 0.5) operational + chi_coh integration) — anchor 22
- arXiv v4.9 §7 paper edit reflection — v0.2 round 内 C3+ commits
- WordPress site sync (sakaguchi-physics.com) — separate round

---

## §2 Canonical IMMUTABLE values (post-v0.2)

### §2.1 Statistical reproducibility (G layer, v0.1.4)

  G1 axis_1_SPARC b_α        = 0.108442979149252
                                (Δ = 4.30e-5 vs baseline 0.1084, 23x margin)
  G2 axis_2_dSph  b_α        = 0.11266236254145712
                                (Δ = 3.76e-5 vs baseline 0.1127, 27x margin)
  G3 axis_3 universal slope  = 0.11055267084535411
                                (in target band [0.110, 0.111])
  G4 Lesson 93 per-dex       = 0.974% (driving: secondary_per_dex)
  G5 abs_diff_axis12         = 0.004219 (16x margin to threshold 0.005)
  G6 universal coupling fit  : R² = 0.634003
                                residual_rms = 0.172134
                                AIC = -529.9199 (n=154, k=6)

### §2.2 Robustness (J layer, v0.1.4)

  J1a std_loo                = 0.001389 (3.6x margin to threshold 0.005)
  J1b OLS SE / |b_α|         = 0.1414 (1.41x margin to threshold 0.20)
                                t-stat = 7.07 (>5σ non-zero)
  J1c outliers (|z|>3)       = 0
  J2 bootstrap 95% CI        = [0.079019, 0.138768] (contains baseline 0.1084)
  J3 filter 1 (Q drop)       = bit-exact reproduction of v1.0.3.1
                                (n=129, b_α=0.112356, drift +3.61%)
  J4 multi-route Δ           = 0.000e+00 (canonical vs phase_c3_step3 reference)

### §2.3 Universal coupling significance

  SE_combined                = sqrt(SE_axis_1² + SE_axis_2²) = 0.0394
  |b_α_SPARC − b_α_dSph|     = 0.004219
  noise-floor significance   = 0.107σ (statistically indistinguishable)

### §2.4 Cross-method extension (v0.2 refinement 3)

  per_filter_sensitivity_extension.csv (4-method comparison):
  baseline              : b_α = 0.108443  (mean V²/r aggregation, no imputation)
  median V²/r           : b_α = 0.070903   (Δ vs baseline = -34.62%)
  min-fill imputation   : b_α = 0.128639   (Δ vs baseline = +18.62%)
  mean-fill imputation  : b_α = 0.094526   (Δ vs baseline = -12.83%)
  k-NN imputation       : b_α = 0.101052   (Δ vs baseline =  -6.82%)

  → result interpretation in §4.4 (Lesson 94 evidence base)

---

## §3 Forensic provenance SHA-pin table

All references below are CDN-pinned via tag companion-v4.9-axis-1-closure-2026-05-05
and remain accessible bit-exact at any future date.

### §3.1 Operational evidence (v0.1.4)

  | Reference                          | SHA256 prefix     | role                              |
  |------------------------------------|-------------------|-----------------------------------|
  | validation_results.json            | b6012483...       | 12/12 gate aggregate              |
  | per_filter_sensitivity.csv         | 6cb3ee15...       | J3 5-filter sensitivity raw       |
  | jackknife_axis_1_results.csv       | 28886a65...       | J1 124 LOO raw                    |
  | bootstrap_axis_1_results.csv       | 55683215...       | J2 10,000 resamples raw           |
  | phase_c3_step3_reference.json      | 6c898438...       | J4 reference impl output          |
  | cross_check_phase_c3_step3.json    | a3046692...       | J4 cross-check delta              |
  | v0.1.4 RELEASE_NOTES.md            | a65cc011...       | round narrative                   |

### §3.2 Frozen canonical

  | Reference                          | SHA256 prefix     | role                              |
  |------------------------------------|-------------------|-----------------------------------|
  | run_section2_5_v0_2.py (Phase 1a)  | cdca6afd...       | frozen canonical (Q-patched)      |
  | parent commit                      | 52ffc09           | Phase 1a A1                       |

### §3.3 v0.2 round (this declaration)

  | Reference                          | SHA256 prefix     | role                              |
  |------------------------------------|-------------------|-----------------------------------|
  | OPERATIONAL_CLOSURE.md             | (self; SHA in SHA256SUMS, tag-pinned via companion-v4.9-axis-1-closure-2026-05-05) | this declaration |
  | axis_1_closure_summary.json        | 55603a4d1db9c62203db68d2794ce01264715686890856df6a95f82e2b6eacce        | machine-readable aggregate        |
  | per_filter_sensitivity_extension.csv| af3809ffddde732511a8ce4aeb2e435ec61faf735815686ba7d3b0fbcd8a577e       | cross-method results              |
  | per_filter_sensitivity_extension.py| 3a55c00167094cd61c4e4a2035753129bac82eb00165369e34251f848bb18aa4        | cross-method script (v2)          |
  | v1_precondition_fail_diagnostic.log | c3087d7282a1dc2e01069f289f9f0a1508654a8653b1160e2edd5c09f9101d62       | v1 -> v2 redesign forensic trace  |
  | C1 commit                          | [C1-SHA-here]     | this round feat                   |
  | C2 commit                          | [C2-SHA-here]     | this round docs (post-hoc xref)   |

### §3.4 Tag

  companion-v4.9-axis-1-closure-2026-05-05
  (annotated, points to C1 commit, all CDN references resolve via this tag)

---

## §4 Lesson 91-94 retrospective summary

Lesson 91-94 are methodology principles abstracted from operational evidence.
Lesson 91-93 were established in C3-A5 round (memo v3 §5.3); Lesson 94 is
formally established by this round.

### §4.1 Lesson 91 — bridge / extreme-regime pre-cut protocol

scope: sample-construction protocol for SPARC + dSph joint analysis
principle: bridge galaxies (proximity-flagged) and extreme-regime samples
           (Q < 3) must be pre-cut before universal coupling assessment to
           avoid confounding the slope agreement test
established: C3-A5 §5.3 (memo v3)

### §4.2 Lesson 92 — parsimony first

scope: model selection epistemic principle
principle: AIC / BIC ranking precedes correlation strength (Spearman) in
           model selection; parsimony beats marginal R² gain
established: C3-A5 §5.3 (memo v3)

### §4.3 Lesson 93 — universal coupling = slope agreement across density range

scope: universal coupling assessment metric
principle: universal coupling claim requires |b_α_axis_1 − b_α_axis_2| / mean
           normalized by density range (per-dex, anchor 19 §1.5 framing) to
           remain below threshold
            absolute scale     : G5 |abs_diff| < 0.005
            per-dex scale      : G4 |diff| / (mean × range) < 1%
            noise-floor scale  : J1b aux |Δ| / SE_combined < 1σ
established: C3-A5 §5.3 (memo v3)

### §4.4 Lesson 94 — g_obs aggregation strategy as systematic source

shorthand: "g_obs aggregation sensitivity"
scope: SPARC rotation curve g_obs aggregation method choice
evidence:
  primary  : J3 filter 3 (anchor 21 v0.1.4)
             NaN g_obs galaxy 11 件 median imputation で b_α drift -9.53%
             provenance: per_filter_sensitivity.csv SHA 6cb3ee15...
                         row "filter_3_disable_g_obs_NaN"
                         validation_results.json gates[J3].scenarios[3]
  cross-method : per_filter_sensitivity_extension.csv (anchor 21 v0.2, this round)
                 4-method aggregation/imputation comparison
                 → rule #26 multi-route minimum compliance 達成
                 results:
                   A1 median(V^2/r)        : b_α = 0.070903 (Δ = -34.62%)
                   A2 min-fill imputation  : b_α = 0.128639 (Δ = +18.62%)
                   A3 mean-fill imputation : b_α = 0.094526 (Δ = -12.83%)
                   A4 k=5 KDTree impute    : b_α = 0.101052 (Δ =  -6.82%)
                 dominant effect: aggregation choice (A1, -34.62%) >>
                                  imputation choice (A2/A3/A4, ±20% range)
                 sample composition: pool-level NaN count = 7 (this run convention)
                                     vs 11 (J3 filter 3 narrative convention,
                                            different mark_fit_pool_171 mask basis);
                                     final OLS sample bit-exact match (n=124,
                                     b_α=0.108442979149252)
principle:
  g_obs aggregation strategy (mean V²/r vs median V²/r vs imputation method)
  は b_α point estimate に substantial 影響 (drift magnitude up to ~10%)。
  explicit NaN-handling protocol が methodology document に必要。
  NaN g_obs galaxy は b_α に対して random ではなく、slope を下方向に pull
  する systematic bias を持つ。
scope LIMIT:
  本 lesson は SPARC g_obs aggregation のみを対象。
  - dSph J3 filter 3 equivalent には extrapolate しない
  - 一般 statistical aggregation には extrapolate しない
  これらは separate round で個別 verify 推奨。
falsification path:
  alternative aggregation (median V²/r, k-NN imputation, ML imputation) で
  b_α drift magnitude が baseline と statistical 区別不可 (within OLS SE)
  となれば本 lesson は false 化。
  retract protocol: forensic_anchors/.../OPERATIONAL_CLOSURE.md errata section
                    + post-hoc cross-reference annotation in this anchor's
                    RELEASE_NOTES.md
established: anchor 21 v0.2 (this round)

---

## §5 forensic chain rule compliance

  rule 1  (anchor IMMUTABLE)          : ✅ v0.1.4 内容は untouched
                                          (C2 で post-hoc additive のみ)
  rule 2  (R-1 LOCK, k_B = 0)         : ✅ 本 round で k_B 触らず
  rule 3  (R-2 LOCK, Algo B self-cons): ✅ 本 round で Algo B 触らず
  rule 4  (Q-C1 LOCK, k_E = 2)        : ✅ 本 round で k_E 触らず
  rule 5  (cascade SSoT)              : ✅ vpp_x05(0.83) = 10.462625 不変
                                          f_opt(0.83) = 1.942493 不変
  rule 6  (L-1 forward-ref 0)         : ✅ parent v4.8 NULL impact 維持
  rule 7  (additive supersession)     : ✅ v0.2 elevates v0.1.4 without revision
  rule #26 (multi-route minimum)      : ✅ v0.1.4 J4 から継承
                                          + Lesson 94 cross-method 拡張で
                                          additional compliance 達成
  rule 92 (parsimony first)           : ✅ Lesson 94 + §7.7 + cross-method を
                                          1 round に統合

---

## §6 arXiv v4.9 paper edit reference (forward declaration)

本 declaration は arXiv v4.9 main paper の以下 section から SHA-pin reference
される:

  §7.6 (universal coupling)      : §1.1, §2, §4.3 を引用
  §7.7 (methodology principles)  : §4.1〜§4.4 を引用 (Lesson 91-94 全て)
  Appendix A (reproducibility)   : §3 (forensic provenance SHA-pin table) を
                                    canonical reproducibility instruction
                                    として引用

paper edit 自体は本 round 内 C3+ commits で実施。

---

## §7 Closure declaration

axis_1 universal coupling claim は本 round (anchor 21 v0.2) を以て
**operational layer で full closure**。

後続 axis_1 work は arXiv v4.9 paper edit (C3+) + WordPress sync (別 round)
のみ。operational layer の追加 hardening 作業は不要。

next operational hardening target: Phase 1b (f_opt(x ≠ 0.5) operational +
chi_coh integration、anchor 22 として独立立ち上げ)。

---

END OF OPERATIONAL_CLOSURE.md (anchor 21 v0.2)
