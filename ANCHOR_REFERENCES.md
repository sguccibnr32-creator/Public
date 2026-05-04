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
