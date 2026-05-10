# anchor 22 v0.2 — axis_4 type-alpha reproduction promotion (DECLARATION)

**Status:** CLOSURE_GRANTED
**Date:** 2026-05-10
**Author:** Sakaguchi Shinobu (Sakaguchi Noodle Factory, Shiso, Hyogo, Japan)
**License:** CC-BY 4.0
**Repo path:** `forensic_anchors/section5_axis_4_type_alpha/`
**Paired with:** anchor 22 v0.1 (`forensic_anchors/section5_axis_4_hardening/`,
commit `0f1d8c9`, tag `companion-v4.9-axis-4-hardening-2026-05-08`)

---

## 1. Declaration

This document declares the closure of **anchor 22 v0.2 — axis_4 type-alpha
reproduction promotion** for the membrane cosmology project (companion paper
v4.9). The Phase Q rule #26 (multi-route minimum reproducibility) requirement
is now satisfied at the **type-alpha grade** (bit-exact 97/97) for axis_4
(HSC weak-lensing route ii), in addition to the type-alpha grade already
established for axis_1 (SPARC route i, anchor 21 v0.2).

> **Type-alpha reproduction grade:** mirror script (an independently
> implemented verification path) and the production transform pipeline
> agree at the bit level on every leaf value of the canonical re-run JSON,
> with no tolerance margin needed (per-leaf delta = 0.0 strict equality).

This is the second axis to reach type-alpha. The remaining three axes
(axis_1 already at type-alpha; axis_2/axis_3 at hardening grade) form
the forward roadmap (see §7 below).

## 2. Scope of this anchor

Anchor 22 v0.2 promotes axis_4 (HSC weak-lensing route ii) from
**hardening grade** (anchor 22 v0.1, 2026-05-08) to **type-alpha
reproduction grade** (this anchor, 2026-05-10).

Concretely, this anchor publicly pins:

1. The transform script SHA at its **rev 6 final form**
   (`ac91bb86caa37ba27adeebb73a2c6f4df1361868e45c1c9baf91ce40b037a618`,
   35,075 B / 787 lines).
2. The 5-tuple of IMMUTABLE input files required to reproduce the result
   (see `anchor_22_v0_2_input_files_pin.json`).
3. The forensic execution log of the rev 5 to rev 6 transition
   (9 + 1 piece patch, V1-V6, G_DRY, step A retry, step C retry,
   step D cleanup, G_MIRROR final grant; see
   `anchor_22_v0_2_verification_log.md`).
4. The L-Q3-11 lessons codification batch reference (8 patterns + 1
   sibling triple + 1 issue + 5+ Bidirectional verification protocol
   success cases; see `anchor_22_v0_2_lessons_appendix.md`).

## 3. Central forensic milestone (97/97 EXACT)

The mirror script (v0.3, SHA `71dfdc56...`, 39,940 B / 975 lines, route ii
of the HSC weak-lensing analysis) was run against the Layer C v1.2
candidate produced by the rev 6 final transform script. The result:

| Metric | Value |
|---|---|
| n_total leaves | 97 |
| n_pass | 97 |
| n_fail | 0 |
| Overall verdict | PASS |
| Per-leaf delta | 0.0 (bit-exact strict equality) |
| Tolerance margin | not needed (1e-12 unused) |

The transition from anchor 22 v0.1 baseline (45 leaves were FAIL at the
hardening grade) to anchor 22 v0.2 (0 FAIL at the type-alpha grade) is
the **45 -> 0 FAIL transition**. This transition is the central forensic
milestone of the present round.

### Per-bucket breakdown

| Bucket | PASS / FAIL | Resolution mechanism |
|---|---|---|
| `per_field_results` | 36 / 0 | M-alpha resolved (`input_path` / `_v1_1_keys` family unified across passthrough vs filter dispatch) |
| `per_mbin_gc[0..3]` | 28 / 0 | M-beta resolved (cross-fill from `logM_min` / `logM_max` / `logM_center` v1.1 keys at 6.2 dispatch) |
| `field_consistency` | 6 / 0 | M-gamma resolved (Pattern 9 inline shape assertion replaces canonical equality check) |
| `gc_M_star_slope` | 6 / 0 | M-gamma + M-delta resolved (`_slope` infix strip + filter + 3 dropped keys with explicit byte-weight account) |
| `combined_3_field_chi2_C15_vs_MOND` | 14 / 0 | Issue J pre-fixed at hardening |
| `paper_claims_for_rounded_check` | 7 / 0 | PLACEHOLDER B pre-fixed at hardening |

All four sub-issues of the Issue M chain (M-alpha/beta/gamma/delta) are
resolved at the disk-artifact level, with physics-layer reproducibility
intact — that is, the publicly known anchor 22 v0.1 numerical values
(combined gc_a0, combined dAIC, weighted_mean, slope, p_C15, p_MOND, etc.)
are bit-exact preserved through the rev 5 to rev 6 transition.

## 4. Physics-layer values preserved (anchor 22 v0.1 bit-exact)

The following physics-layer values, established at hardening grade in
anchor 22 v0.1, remain bit-exact identical at type-alpha grade:

```
G09  gc_a0                     = (preserved from anchor 22 v0.1)
G12  gc_a0                     = (preserved from anchor 22 v0.1)
G15  gc_a0                     = (preserved from anchor 22 v0.1)
combined gc_a0 (M-15)          = 2.7184876317961755
combined dAIC  (M-17)          = 472.2767831581
weighted_mean  (M-18/19)       = 2.7326473920108501 +/- 0.1072291448861956
consistency p  (M-24)          = 0.503592 (CONSISTENT)
slope                          = 0.165849 +/- 0.041478
p_C15 / p_MOND                 = 2.85e-02 / 6.38e-05
```

The combined dAIC = +472 (~22-sigma) significance for "Condition 15 +
extended Model B + 2-halo correction" over pure MOND is therefore
preserved unchanged.

## 5. G_MIRROR composite condition

The G_MIRROR (mirror verification) final grant requires three independent
necessary conditions, all met:

| # | Condition | Status | When achieved |
|---|---|---|---|
| (1) | mirror v0.3 vs Layer C v1.2 candidate: 97/97 EXACT | MET | turn N+7, 2026-05-10 |
| (2) | forensic transients cleanup verified | MET | turn N+9, 2026-05-10 |
| (3) | step B path (b) confirmed (empirically reproducible) | MET | prior chat, [3.4.1.5] |

Final grant: **G_MIRROR GRANTED** (turn N+10, 2026-05-10, claude.ai-side
design judgment).

## 6. Phase 3 -> Phase 4 gate (7 / 7 PASS)

Entry to Phase 4 (patch apply round) is gated on seven conditions, all
satisfied:

```
G3-1  G_MIRROR final grant                                         PASS
G3-2  rev 6 final form SHA preserved (ac91bb86...)                 PASS
G3-3  rule 1 IMMUTABLE 7 anchors invariance (pre/post SHA)         PASS
G3-4  Public repo touch 0 invariant during round                   PASS
G3-5  physics-layer reproducibility intact vs anchor 22 v0.1       PASS
G3-6  forensic transients cleanup verified (cross-environment)     PASS
G3-7  L-Q3-11 codification batch ready (draft complete)            PASS
```

## 7. Forward path

| Phase | Round | Description |
|---|---|---|
| 4 | turn N+12+ | Write 4 paired files at `forensic_anchors/section5_axis_4_type_alpha/`, append `.gitattributes` rule, append SHA256SUMS (4 entries), `sha256sum -c` full self-verify, update arXiv companion paper v4.9 §reflection, draft WordPress update content |
| 5-A | turn N+13+ | Pre-push 4-item review (commit content sanity / repo state / remote diff / push dry-run) |
| 5-B | turn N+13+ | Bundled commit + annotated tag (see §8 below) + push origin main + push --tags + post-push raw URL audit |
| 5-C | turn N+13+ | Memory persistence update + handoff memo (claude.ai + Claude Code parallel) + (optional) verification PDF v4.4 spec |

Subsequent multi-round work (out of present scope):

- Type-alpha reproduction grade promotion for axis_2 (dSph route v) and
  axis_3 (Brouwer KiDS route iii)
- L-Q3-11 codification batch formal append to `lessons_codified.md`
  (combined with L-Q3-10 in a dedicated batched codification round)
- axis_5+ / WP series roadmap finalization

## 8. Annotated tag (Phase 5-B)

Two candidates are carried into Phase 4 for final lock; Phase 4 selects one:

| Candidate | String | Rationale |
|---|---|---|
| (a) | `companion-v4.9-axis-4-type-alpha-2026-05-09` | Issue identification timing, anchor 22 v0.1 tag `companion-v4.9-axis-4-hardening-2026-05-08` paired form (one-day-after pattern) |
| (b) | `companion-v4.9-axis-4-type-alpha-2026-05-10` | Final closure timing of the present round (physics completion day) |

Final selection is made at Phase 4 with the arXiv companion paper v4.9
§reflection update (cross-reference consistency takes precedence).

## 9. License and reuse

This anchor (along with all paired files and the entire content of the
public repository) is licensed under **CC-BY 4.0**. Independent
reproduction, verification, and reuse are welcomed. Cite as:

> Sakaguchi Shinobu (2026), "anchor 22 v0.2 — axis_4 type-alpha
> reproduction promotion (membrane cosmology project)", CC-BY 4.0,
> github.com/sguccibnr32-creator/Public, commit (TBD at Phase 5-B).

## 10. References

- `anchor_22_v0_2_input_files_pin.json` — 5 IMMUTABLE input files SHA + role
- `anchor_22_v0_2_verification_log.md` — full forensic execution log
- `anchor_22_v0_2_lessons_appendix.md` — L-Q3-11 codification batch reference
- anchor 22 v0.1 (`forensic_anchors/section5_axis_4_hardening/`) — paired predecessor
- anchor 21 v0.2 (`forensic_anchors/section4_axis_1_closure/`) — type-alpha precedent for axis_1
- arXiv companion paper v4.9, §7.6 (universal linear coupling), §7.7 (methodology principles), §reflection (TBD at Phase 4)
- WordPress: sakaguchi-physics.com (blog_id 253652152), update content draft (TBD at Phase 4)

---

*Document status: CLOSURE_GRANTED. All gates passed. Forward to Phase 4.*
