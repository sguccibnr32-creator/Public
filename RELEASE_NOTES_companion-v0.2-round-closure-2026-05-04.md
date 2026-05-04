# section 2.5 v0.2 round closure — v1.0.3.1 reproducibility deliverable

**Tag**: `companion-v0.2-round-closure-2026-05-04`
**Date**: 2026-05-04
**Path**: `forensic_anchors/section2_5_v0_3_step4_reproducibility/run_section2_5_v0_2.py`
**SHA256**: `7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652`
**Size**: 137,856 B

## Summary

J-system companion paper section 2.5 v0.2 round の operational closure。
膜宇宙論 framework の Phase C3 §4.3 universal coupling 仮説 (b_alpha = 0.11
±0.005, 3.92 dex 密度範囲, separate anchor (G_Strigari vs gc_C15),
0.5% 以内 slope agreement) を Python script reproducibility level で立証。

## Patch chain (v1.0.2 → v1.0.3.1)

### v1.0.2 → v1.0.3 (claude.ai container atomic, SHA `2ec2e258...`, 135,333 B)

13 patches (P1-P13), +885 LOC, +7 helper functions, +41% file size。
詳細は handoff_memo_section2_5_v0_2_closure_2026-05-04.txt §8 参照。

### v1.0.3 → v1.0.3.1 (Windows host Step 4, SHA `7cb540b1...`, 137,856 B)

5 inline patches in 1 batch (no incremental SHA retained):

| # | scope                                     | summary                                                |
|---|-------------------------------------------|--------------------------------------------------------|
| 1 | `load_phase1` L515-531 (T10)              | `encoding="utf-8-sig"` + `lstrip("\ufeff")` BOM strip  |
| 2 | `load_MRT` L537-587 (T11)                 | `read_csv(sep=r"\s+", skiprows=98)`, 18-col raw schema |
| 3 | `_prepare_dsph_phase_c3_sample` L2029-2050 | `LV→M_bar` / `rh_pc→r_h` / `rho_gal` derivation        |
| 4 | `col_alias_map` L3068-3070                | capital `Ud→Upsilon_d` added (load_phase1 default)     |
| 5 | SPARC `g_obs` aggregation L3043-3061      | `mean(V^2/r)` for `r > 2*hR` (T16.F verbatim)          |

All 5 patches address loader / derivation / aggregation defects. None directly
modifies the universal coupling computation kernel.

## Acceptance gates

| metric                         | observed         | criterion           | status              |
|--------------------------------|------------------|---------------------|---------------------|
| `axis_2_dSph`                  | 0.11266          | bit-exact           | ✅ PASS             |
| `axis_3_universal_slope`       | 0.11251          | within ±1σ (§4.3)   | ✅ PASS             |
| `abs_diff` (combo & standalone)| 0.000306         | < 0.005 (AC4)       | ✅ PASS (16× margin)|
| Lesson 93 slope agreement      | 0.272% rel       | < 1%                | ✅ PASS (3.7× strict)|
| `axis_1_continuity`            | "finite"         | extreme regime      | ✅ Strigari OK      |
| `axis_2_reversal`              | "reproduced"     | dSph 28/31          | ✅ trend match      |
| `axis_1_SPARC`                 | 0.11236          | (deferred to v1.0.4)| ⚠ 3.6% rel deviation|

### `axis_1_SPARC` deviation: scope statement

3.6% relative deviation from anchor 19 §1.5 baseline (0.1084) は本 commit の
5 patches に起因しない。Root cause は v1.0.3 sample sub-cut logic divergence
(`sample_n_axis_1 = 129` vs reference 124, 5 galaxy excess)。候補:

- `Q < 3` cut strict ordering
- `v_flat > 0` filter timing
- NaN `g_obs` validity criterion
- 4-bridge exclusion timing (NGC3741 / NGC2915 / ESO444-G084 / NGC1705)
- `gc_C15` finite filter 不在

Hardening は **v1.0.4 in (C) arXiv v4.9 patch round** で実施予定。
universal coupling claim (axis_3 / Lesson 93) は影響を受けない。

## Forensic chain rule 7 compliance

| rule | description                              | status                         |
|------|------------------------------------------|--------------------------------|
| 1    | anchor IMMUTABLE preserved (zero modify) | ✅ FULL COMPLIANCE             |
| 2    | R-1 LOCK (k_B = 0)                       | ✅ preserved                   |
| 3    | R-2 LOCK (Algorithm B self-consistency)  | ✅ preserved                   |
| 4    | Q-C1 LOCK (k_E = 2 default)              | ✅ preserved                   |
| 5    | cascade SSoT preserved                   | ✅ vpp_x05(0.83)=10.462625 / f_opt(0.83)=1.942493 |
| 6    | L-1 forward-ref 0 strict                 | ✅ parent v4.8 NULL impact     |
| 7    | companion additive supersession          | ✅ **ESTABLISHED** by this commit |

### Anchor IMMUTABLE verification

| target                                       | SHA prefix  | status |
|----------------------------------------------|-------------|--------|
| anchor 21 v0.1.1 (J_system_paper_§2_closure) | `44df9afb...` | ✅ unchanged |
| anchor 5 / 6 / 7 / 8 / 14 / 16 / 17 / 19 / 20 | (各 SHA)    | ✅ unchanged |
| `EXECUTION_PLAN_section2_5_v0_2.md`          | `9f47bb7f...` | ✅ unchanged |
| `OUTPUT_SCHEMA_section2_5_v0_2.md`           | `fe007753...` | ✅ unchanged |
| `foundation_gamma_actual.py` (cascade SSoT)  | `b0cb36d7...` | ✅ unchanged |
| commit `8e8ed51` (v1.0.2 forensic anchor)    | `dd762fd2...` | ✅ IMMUTABLE retained |

### Supersession declaration

> v1.0.2 (`dd762fd25193748f2aae0f5958b4c2170f1c2a0a1fb9345208808b1cf8bf57e6`,
> commit `8e8ed51`) is preserved IMMUTABLE at
> `forensic_anchors/section2_5_v0_2_skeleton/`. v1.0.3.1
> (`7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652`) at
> `forensic_anchors/section2_5_v0_3_step4_reproducibility/` is its
> companion-additive successor and supersedes it for operational reproducibility,
> without modifying the v1.0.2 file or any anchor.

## Files in this release

### `forensic_anchors/section2_5_v0_3_step4_reproducibility/`

| file                                  | size      | SHA256                                                              |
|---------------------------------------|-----------|---------------------------------------------------------------------|
| `run_section2_5_v0_2.py` (v1.0.3.1)   | 137,856 B | `7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652` |
| `_v1_0_2_b_alpha_wip.bak.py`          | 104,306 B | `6c912e0adee46ca483a18c30dcd3860ce1038280c6b9b756e1a29ce4d4f4fc9e` |
| `_v1_0_1.bak.py`                      |  59,551 B | `34a749703f35e1b72b329b51fc86bf90decff34d138aae3bb922c03731ed04af` |

### `artifacts/`

| file                                            | size      | SHA256                                                              |
|-------------------------------------------------|-----------|---------------------------------------------------------------------|
| `run_section2_5_v0_2_v1_0_3.py` (atomic)        | 135,333 B | `2ec2e258c8f0f2b4eeda28d8e86434ef1d951e665ef41ab4257c101995423591` |
| `section2_5_v0_2_round_closure_2026-05-04.pdf`  | 202,340 B | `31153f5f928318e95e9f5135af3e507e70229935efeee3c24fcc9b822436492a` |
| `P12_FILL_PROCEDURE_v2_0.md`                    |  39,651 B | `fdd92346e380c4d1a9958020311b569f3ebce70b5bb4c8405ff83270c516d882` |
| `build_session_summary_pdf.py`                  |  47,429 B | `38fd13b146b1805241e11b9dd653f77055712a67856cf29792ac81e441fd7e6d` |

### Repo root (updated)

- `SHA256SUMS` — append entries per above + retain historical
- `ANCHOR_REFERENCES.md` — append anchor 21 v0.1.2 entry
- `RELEASE_NOTES_companion-v0.2-round-closure-2026-05-04.md` — this file (new)

## Anchor record

```
anchor 21 v0.1.2 (section 2.5 v0.2 step 4 reproducibility addendum)
  source: forensic_anchors/section2_5_v0_3_step4_reproducibility/run_section2_5_v0_2.py
  SHA256: 7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652
  size:   137,856 B
  parent: anchor 21 v0.1.1
          SHA 44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f
          (section 2 closure declaration, IMMUTABLE)
  scope:  Step 4 — 5 inline patches reproducibility addendum
          (loader / derivation / aggregation fix only; no claim-kernel touch)
  status: provisional — axis_1_SPARC alignment pending v1.0.4 (C) round
          (anchor 19 §1.5 baseline 0.1084 vs observed 0.11236, 3.6% rel)
  rule 1: parent anchor 21 v0.1.1 (44df9afb) IMMUTABLE preserved
  rule 7: companion-additive successor to v1.0.2 (dd762fd2, commit 8e8ed51)
          v1.0.2 retained IMMUTABLE at forensic_anchors/section2_5_v0_2_skeleton/
  promotion path: v1.0.4 (C) axis_1 alignment 達成時に anchor 21 v0.2
                  (full operational closure) として promote、本 v0.1.2 は
                  historical retain
```

## Roadmap

| phase | scope                                                | timing       |
|-------|------------------------------------------------------|--------------|
| (A)   | GitHub forensic anchor commit (本 release)           | 完了         |
| (C)   | arXiv v4.9 patch round prep — axis_1 sub-cut hardening + f_opt(x≠0.5) v4.9 candidate | 次次セッション |
| (B)   | section 2.6 anchor 8 chapter-level milestone bridge  | (C) 完了後   |
