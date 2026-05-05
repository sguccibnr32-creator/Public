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
