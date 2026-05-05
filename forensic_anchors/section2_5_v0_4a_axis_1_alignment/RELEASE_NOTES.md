# v1.0.4a - axis_1_SPARC mechanical alignment

**Tag**: `companion-v0.4a-axis-1-alignment-2026-05-04`
**Date**: 2026-05-04
**Path**: `forensic_anchors/section2_5_v0_4a_axis_1_alignment/run_section2_5_v0_2.py`
**SHA256**: `cdca6afd634a3f730bc4b4002ab3082e187dc64b4ad9e87fd8147a6d81b04521`
**Size**: 137,861 B

## Scope

Phase 1a of (C) round split: **mechanical alignment only**.
Statistical / universal coupling validation (b_alpha vs anchor 19 sec 1.5
baseline 0.1084, Lesson 93, AC4/AC5, jackknife) is **deferred** to a separate
round (Phase 1a-validation, anchor 21 v0.1.4).

This split prevents tautological self-validation (fix design endpoint and
fix success criterion both being anchor 19 baseline 0.1084 in the same round).

## Root cause

Single-line bug in `load_MRT` at L587 of `run_section2_5_v0_2.py`:

```python
# v1.0.3.1 (incorrect):
return df[["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat"]].copy()
#                                                                         ^^^ Q missing
```

`load_MRT` reads 18 columns including Q at L573 (raw_names list), but the
return statement subset omitted Q. Effects:

1. `sparc_full` does not have Q column
2. `_prepare_sparc_phase_c3_sample` L1978: `if "Q" in df.columns` evaluates
   False -> Q<3 cut **silently skipped**
3. 12 galaxies with Q>=3 (mostly low-quality / bridge case) silently pass
   the Q gate
4. `defensive_filter` (rho_gal>0 & finite, r_h>0 & finite, delta_primary
   finite) catches 7 of these 12 indirectly (via gc_C15=0 -> log10(0)=-inf
   -> delta_primary=inf), leaving **5 silent excess** galaxies
5. `sample_n_axis_1` = 124 (expected) + 5 (silent pass) = 129 (observed
   in commit 46ea829)

## Patch

1 line, 5 bytes added:

```python
# v1.0.4a (corrected):
return df[["galaxy", "T", "Rdisk", "SBdisk0", "L36", "MHI", "Vflat", "Q"]].copy()
```

| metric | v1.0.3.1 | v1.0.4a |
|---|---|---|
| size | 137,856 B | 137,861 B (+5) |
| SHA256 | `7cb540b1...` | `cdca6afd...` |
| AST parse | OK | OK |

## Mechanical gates (5/5 PASS)

| # | criterion | result | evidence |
|---|---|---|---|
| 1 | Q column in `load_MRT` return statement | PASS | grep return line |
| 2 | Q<3 cut active (per-galaxy Q values) | PASS | per-galaxy Q in [1, 2], aggregate 124 rows |
| 3 | `sample_n_axis_1` == 124 (binary) | PASS | 124 == 124 |
| 4 | galaxy identity match vs phase_c3_step3 reference | PASS | symmetric difference = 0 |
| 5 | `sample_n_axis_3` == 154 (= 124 + 30) | PASS | 124 + 30 = 154 |

Gate 4 is the critical structural verification: `phase_c3_step3` reference
impl (independent code path with Q-intact loader) and `run_section2_5_v0_2.py`
v1.0.4a (Q-patched canonical) produce **identical galaxy sets** of size 124.
This excludes the possibility that `sample_n` accidentally matched 124 via
some other filter coincidence.

## Forbidden in this phase (circularity-free design)

```
X b_alpha 値の計算 / 比較 / assertion
X Lesson 93 universal coupling check
X AC4 / AC5 gate evaluation
X jackknife / bootstrap
```

These verifications belong in Phase 1a-validation (separate round, frozen
Phase 1a output as input).

## Hypothesis correction (post-hoc finding)

Commit 46ea829 RELEASE_NOTES sec axis_1_SPARC deviation listed 5 candidate
root causes:

```
- Q < 3 cut strict ordering
- v_flat > 0 filter timing
- NaN g_obs validity criterion
- 4-bridge exclusion timing (NGC3741, NGC2915, ESO444-G084, NGC1705)
- gc_C15 finite filter unavailability
```

Forensic verification (this round) shows all 5 candidates are **non-causal**.
Bridge 4 is correctly excluded at sparc_171 construction (L607-608); the
re-exclusion at L1993 is NO-OP. gc_C15 finite is caught indirectly via
delta_primary finite check.

The actual root cause is a **single-line column subset bug** in load_MRT
return statement (Q column drop). Lesson 92 (parsimony first) is
strengthened: 1 line / 1 column / 5 byte fix is more parsimonious than the
4 bridge + 1 finite-filter hypothesis.

The structural coincidence "5 = 4 bridges + 1 finite" was a numerical
accident: the 5 silent excess are Q>=3 non-bridge galaxies that have
v_flat>0 AND g_obs valid, not bridge galaxies.

## Forensic chain rule compliance

| rule | description | status |
|---|---|---|
| 1 | anchor IMMUTABLE preserved (zero modify) | PASS (v1.0.3.1 retained at `.bak.py`) |
| 2 | R-1 LOCK (k_B = 0) | preserved |
| 3 | R-2 LOCK (Algorithm B self-consistency) | preserved |
| 4 | Q-C1 LOCK (k_E = 2 default) | preserved |
| 5 | cascade SSoT preserved | `vpp_x05(0.83)=10.462625 / f_opt(0.83)=1.942493` |
| 6 | L-1 forward-ref 0 strict | parent v4.8 NULL impact |
| 7 | companion additive supersession | DECLARED (parent commit 46ea829 IMMUTABLE) |

### Anchor IMMUTABLE verification

| target | SHA prefix | status |
|---|---|---|
| anchor 21 v0.1.2 (commit 46ea829) | `44df9afb...` | unchanged |
| commit 46ea829 RELEASE_NOTES + ANCHOR_REFERENCES | (post-hoc CORRECTION append-only) | preserved |
| `forensic_anchors/section2_5_v0_2_skeleton/` (commit 8e8ed51, v1.0.2) | `dd762fd2...` | unchanged |
| `forensic_anchors/section2_5_v0_3_step4_reproducibility/` (commit 46ea829, v1.0.3.1) | `7cb540b1...` | unchanged |

### Supersession declaration

> v1.0.3.1 (`7cb540b1...`, commit 46ea829) is preserved IMMUTABLE at
> `forensic_anchors/section2_5_v0_3_step4_reproducibility/`. v1.0.4a
> (`cdca6afd...`) at `forensic_anchors/section2_5_v0_4a_axis_1_alignment/`
> is its companion-additive successor and supersedes it for mechanical
> alignment closure, without modifying the v1.0.3.1 file or any anchor.
> v1.0.3.1 file is also retained inside this directory as
> `run_section2_5_v0_2_v1_0_3_1.bak.py` for paranoid forensic chain rule 1
> compliance.

## Files in this release

| file | size | SHA256 |
|---|---|---|
| `run_section2_5_v0_2.py` (v1.0.4a) | 137,861 B | `cdca6afd634a3f730bc4b4002ab3082e187dc64b4ad9e87fd8147a6d81b04521` |
| `run_section2_5_v0_2_v1_0_3_1.bak.py` (paranoid retain) | 137,856 B | `7cb540b11650ece360c780061923196ba532638ac58574a652befb8a25037652` |
| `gate_phase_1a_mechanical.py` | 9,062 B | `4612150d03016c32517343d7efabeef10193b1783ca63c86212a895cbcfb1532` |
| `gate_phase_1a_results.json` | (tbd) | (tbd at commit-time post-finalize) |
| `RELEASE_NOTES.md` (this file) | (tbd) | (tbd) |

## Lesson compliance

- **Lesson 91** (bridge / extreme-regime pre-cut protocol): RESTORED
  - Q<3 cut now functional, pre-cut protocol active per anchor 7 sec 2.5.1
- **Lesson 92** (parsimony first): STRICT compliance
  - 1 line / 1 column / 5 byte fix; simpler than any prior multi-filter hypothesis
- **Lesson 93** (universal coupling slope agreement): DEFERRED to Phase 1a-validation
- **forensic chain rule 1** (anchor IMMUTABLE): PRESERVED via paranoid retain

## Closure status

- Phase 1a (mechanical): **CLOSED**
- Phase 1a-validation (statistical): **PENDING** (separate round)
- Phase 1b (`f_opt(x != 0.5)` operational): **DEFERRED** (separate round, anchor 22)

## Anchor record

```
anchor 21 v0.1.3 (section 2.5 v0.4a axis_1 mechanical alignment)
  source: forensic_anchors/section2_5_v0_4a_axis_1_alignment/run_section2_5_v0_2.py
  SHA256: cdca6afd634a3f730bc4b4002ab3082e187dc64b4ad9e87fd8147a6d81b04521
  size:   137,861 B
  parent: anchor 21 v0.1.2
          SHA 44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f
          (section 2 closure declaration, IMMUTABLE)
  scope:  axis_1_SPARC mechanical alignment via 1-line Q-column patch
          (load_MRT L587 return statement)
  status: provisional - statistical validation pending Phase 1a-validation
  rule 1: parent anchor 21 v0.1.2 (44df9afb) IMMUTABLE preserved
  rule 7: companion-additive successor to v1.0.3.1 (7cb540b1, commit 46ea829)
          v1.0.3.1 retained IMMUTABLE at forensic_anchors/section2_5_v0_3_step4_reproducibility/
          + paranoid retain at forensic_anchors/section2_5_v0_4a_axis_1_alignment/.bak.py
  promotion path: Phase 1a-validation 完了時に anchor 21 v0.1.4 として promote、
                  v0.1.4 完了時に anchor 21 v0.2 (full operational closure) として promote、
                  本 v0.1.3 は historical retain
```

## Roadmap

| phase | scope | timing | anchor |
|---|---|---|---|
| 1a (this commit) | mechanical alignment (Q-patch + 5 gates) | done | 21 v0.1.3 |
| 1a-validation | statistical + universal coupling on frozen 1a output | next | 21 v0.1.4 |
| 1b | `f_opt(x != 0.5)` operational + chi_coh integration | post-1a-validation | 22 |
| 2 | full operational closure | post-1b | 21 v0.2 |

---

## POST-HOC CROSS-REFERENCE (added 2026-05-05)

Phase 1a-validation has been completed at anchor 21 v0.1.4 (commit `5783bef`, tag `companion-v0.4a-validation-2026-05-05`) with 12/12 PASS. The forensic narrative initiated in this anchor (v0.1.3) is now closed:

- **Diagnosis** (this anchor): step trace at L1978 → "NO CUT" observed → load_MRT L587 return missing Q column identified
- **Patch** (this anchor): 1-line / 5-byte amendment to L587 return statement
- **Statistical reproducibility** (v0.1.4 G layer): G1-G6 6/6 PASS — baseline 0.1084 / 0.1127 / 0.0042 reproduced
- **Robustness** (v0.1.4 J layer): J1-J4 6/6 PASS (post (γ) split)
- **Causal direct demonstration** (v0.1.4 J3 filter 1): dropping Q column from sparc_for_audit reproduces v1.0.3.1 state bit-exact (n=129, b_α=0.112356, drift +3.61% matching documented 3.6%). Re-applying Q yields baseline (n=124, b_α=0.108443). The 1-line patch is therefore established as the **necessary and sufficient** causal mechanism.
- **Multi-route minimum** (v0.1.4 J4): canonical and phase_c3_step3 reference impl yield axis_1_SPARC = 0.108442979149252 bit-exact (Δ = 0.000e+00); reproducibility independent of pipeline detail.

The diagnosis recorded here is **retroactively validated** by Phase 1a-validation J3 filter 1 with falsifiable bit-exact reproduction. No alternative causal explanation survives.

See `forensic_anchors/section2_5_v0_4a_validation/RELEASE_NOTES.md` for full Phase 1a-validation closure documentation.
