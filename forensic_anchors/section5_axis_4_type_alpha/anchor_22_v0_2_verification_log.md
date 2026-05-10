# anchor 22 v0.2 — Forensic Verification Log

**Anchor:** anchor 22 v0.2 (axis_4 type-alpha reproduction promotion)
**Round:** 2026-05-10, turn N+4 ~ N+10 (chat closure)
**Author:** Sakaguchi Shinobu
**License:** CC-BY 4.0
**Paired with:** `anchor_22_v0_2_declaration.md`,
`anchor_22_v0_2_input_files_pin.json`,
`anchor_22_v0_2_lessons_appendix.md`

---

## 1. Pre-flight integrity check (turn N+4 Step 1, 9 items)

| # | Object | Expected SHA-256 | Expected Size | Result |
|---|---|---|---|---|
| 1 | transform script (rev 5 docstring-synced baseline) | `a7931889...` | 27,491 B | MATCH |
| 2 | `.tmp_phase_3_v1_2_test.json` (step A rev 5 artifact) | `0ac7aa8b...` | 13,666 B | MATCH |
| 3 | mirror script v0.3 | `71dfdc56...` | 39,940 B | MATCH |
| 4 | input_files_pin (resolved) | `af986257...` | 6,182 B | MATCH |
| 5 | Layer C v1.1 IMMUTABLE | `5d9beb04...` | 11,096 B | MATCH |
| 6 | Working tree state | HEAD = origin/main = `86742b79...` | porcelain v1 = 0 lines | clean |
| 7 | Rendered section sign in transform script | 0 hits | — | OK |
| 8 | `verify_bit_exact_for_passthrough` references | 4 hits (1 def L331 + 3 callers L461/470/500, Piece 6 deletion target) | — | orphan reference absent |

**Verdict:** Pre-flight ALL GREEN. 9-piece patch apply ready.

## 2. rev 5 baseline preservation (turn N+4 Step 1)

```
Rename-Item E:\Q-3_route_ii_discovery_2026-05-07\.tmp_phase_3_v1_2_test.json
            -NewName .tmp_phase_3_v1_2_test_rev5.json
```

Pre/post SHA-256 bit-exact preserved (`0ac7aa8b...` / 13,666 B both).
Rename only, no content change. This preserves the rev 5 step A artifact
as a forensic chain reference.

## 3. 9-piece patch apply (turn N+4 Step 2)

The rev 5 -> rev 6 transition is performed as a sequence of 9 patch
pieces, all applied via the Edit tool (1-shot apply per piece) except
Piece 3 which used a PowerShell byte-precise replacement due to a
Pattern 12 region.

### 3.1 Piece 1 — header L3 title rev 5 -> rev 6 + L7 timestamp append

Edit tool 1-shot apply. SUCCESS.

### 3.2 Piece 2 — rev 6 changelog block append (L79 tail)

Edit tool 1-shot apply. SUCCESS at the str_replace level, **but** new_str
contained two `_paper_section_1.3` references whose 6-character ASCII
escape (intended literal) was silently auto-decoded by the JSON
parameter pipeline (RFC 8259) into the rendered section sign on disk.

#### Pattern 12 detection (post-Piece-2 grep audit)

```
Rendered section sign (UTF-8 0xC2 0xA7) in transform script:
  pre-Piece-2:  0 hits
  post-Piece-2: 2 hits at L106 + L127  -- POLICY VIOLATION
ASCII section escape (literal):
  pre-Piece-2:  10 hits (rev 5 baseline)
  post-Piece-2: 10 hits (Piece 2's 2 new escapes lost to auto-decoding)
```

#### Pattern 12 remediation

PowerShell byte-level fix-up: replace UTF-8 `0xC2 0xA7` byte sequence
(rendered section sign) with the 6-byte ASCII escape sequence `&#92;u00a7`,
2 / 2 occurrences corrected.

```
post-remediation:
  rendered section sign: 0 hits  (policy restored)
  ASCII section escape: 12 hits  (rev 5 baseline 10 + Piece 2 new 2)
```

Pattern 12 codification candidate accepted at turn N+4.5 by claude.ai-side
into the L-Q3-11 batch (see `anchor_22_v0_2_lessons_appendix.md` §2.7).

### 3.3 Piece 3 — docstring "Schema transformations" L94 + L97-99 update

5 -> 7 bullet structure. Same Pattern 12 region, so applied via
PowerShell byte-precise replacement (Edit tool would re-introduce the
auto-decode failure). Used `[char]92 + 'u00a7'` pattern to construct the
literal 6-character ASCII escape sequence.

Old_str uniqueness audit: 1 hit confirmed.
Post-replacement audit:
- rendered section sign: 0 hits (preserved)
- ASCII section escape: 12 hits (preserved)
- size: 30,532 B -> 31,147 B (+615 B)
- lines: 615 -> 623 (+8)

### 3.4 Pieces 4 / 5 / 5a / 5b / 5c — main body (Edit tool, all SUCCESS)

After PowerShell modification, transform script is Re-Read (Edit tool
state refresh).

| Piece | Region | Description | Apply |
|---|---|---|---|
| 4 | L209-260 | `transform_per_field` full function body replace: signature change + 3 guards retain + iterate `v1_1_keys` + `dropped_per_field` forensic record + 3-tuple return | SUCCESS (1-shot) |
| 5 | L444-455 | 6.1 dispatch site replace: `per_field_results` naming preserve + `v1_1_per_field_keys` 取得 + 3-tuple unpack + `dropped_per_field` record | SUCCESS (1-shot) |
| 5a | L457-464 | 6.2 `per_mbin_gc` dispatch: filter + v1.1 cross-fill from `logM_min`/`logM_max`/`logM_center` | SUCCESS (1-shot); caller count 6 -> 5 |
| 5b | L466-473 | 6.3 `field_consistency` dispatch: pass-through with v1.1 shape assertion (Pattern 9 inline application) | SUCCESS (1-shot); caller count 5 -> 4 |
| 5c | L496-503 | 6.5 `gc_M_star_slope` dispatch: rename "_slope" infix strip + filter + integrity assertion mirror `transform_paper_claims` pattern | SUCCESS (1-shot); caller count 4 -> 3 |

### 3.5 Piece 6 — `verify_bit_exact_for_passthrough` deletion (P6-α form)

(D-1) section comment + function body bulk deletion. Edit tool 1-shot
apply. SUCCESS. Definition + 3 caller block all deleted; caller count
3 -> 2 (changelog mentions only at L110 / L114).

**Order constraint compliance:** Pieces 5a / 5b / 5c removed all 3
callers first; Piece 6 deletion at the end leaves no orphan reference
(NameError risk = 0). This is the operational verification of the
Pattern 10 (refactor blind spot to existing invariants) codification.

## 4. V1-V6 verify (turn N+4 Step 3, rev 6.0 intermediate state)

| # | Metric | Envelope | Actual | Verdict |
|---|---|---|---|---|
| V1 | file size | ~28,500-29,000 B | 35,075 B (+21% over) | NOTE: verbose changelog |
| V2 | line count | ~680-700 | 787 (+12% over) / 0 CRLF / no BOM | NOTE: verbose changelog |
| V3 | AST parse | OK | OK | PASS |
| V4 | `--help` argparse description | "(rev 6)" | "(rev 5)" — **anomaly** | DEFER (Piece 7) |
| V5 | grep audit | rendered section sign 0; ASCII escape 12; `verify_bit_exact_for_passthrough` 2 (changelog only); `SLOPE_INFIX` 3 (def + 2 uses); `v1_1_*_keys` family many usages | All match | PASS |
| V6 | rev 6.0 intermediate SHA | (recorded forensic-only) | `df698d22343573a5008a1cdafec4979be6a5eda0d177e74fdbf170dd10619656` / 35,075 B / 787 lines | recorded |

V4 anomaly: argparse `description` string still reads "(rev 5)" because
Piece 1 only updated the header L3 title; the `main()` function's
`argparse.ArgumentParser(description=...)` was outside Pieces 1-7 scope.
This was caught here at the V re-run, not at apply time — Bidirectional
verification protocol operational success case.

V6 SHA `df698d22...` is forensic-only (rev 6.0 intermediate, not the
publicly pinned rev 6 final form).

## 5. Piece 7 (turn N+4.5 ~ N+4.7, V4 corrective bundle)

### 5.1 turn N+4.5 — capture 11-A (Claude Code -> claude.ai)

Capture 11-A delivers the argparse description region (L668-682,
verbatim, line-numbered cat -n format, boundary uniqueness audit).
- L668-669: single hit for `description=` `(rev 5)`
- Boundary uniqueness audit: `(rev 5)` literal in code = 1 hit (changelog
  excluded); `description=` attribute = 1 hit
- df698d22 reframed: rev 6.0 intermediate (forensic-only); not the
  publicly pinned anchor SHA. Two-stage SHA (intermediate -> final)
  accepted as forensic chain transparency enhancement.

### 5.2 turn N+4.6 — Piece 7 drafting (claude.ai-side)

3-line block boundary form, size delta = 0, single 1-character mutation
`(rev 5)` -> `(rev 6)`:

```
old_str:
    p = argparse.ArgumentParser(
        description="Layer C v1.1 -> v1.2 schema transform tool (rev 5)"
    )
new_str:
    p = argparse.ArgumentParser(
        description="Layer C v1.1 -> v1.2 schema transform tool (rev 6)"
    )
```

### 5.3 turn N+4.7 — Piece 7 apply

Edit tool 1-shot apply. SUCCESS. Pattern 12 not applicable (no section
sign in region) — remediation not needed.

### 5.4 V re-run (rev 6 final form)

| Metric | Value |
|---|---|
| **SHA-256** | `ac91bb86caa37ba27adeebb73a2c6f4df1361868e45c1c9baf91ce40b037a618` |
| Size | 35,075 B (size delta = 0 from rev 6.0 intermediate) |
| Lines | 787 (line count delta = 0) |
| Encoding | UTF-8 / LF / no BOM |
| Rendered section sign | 0 hits |
| ASCII section escape | 12 hits |
| AST parse | OK |
| `--help` description | `Layer C v1.1 -> v1.2 schema transform tool (rev 6)` |

Byte-level mutation precision audit: offset 30282 single-byte mutation
confirmed; SHA avalanche delta consistent with single-byte mutation
(clean form).

**rev 6 final form is the publicly pinned anchor 22 v0.2 SHA.**

## 6. G_DRY (turn N+5, 4 sequential dry-runs)

Five-metric set (M1-M5), 4 sequential dry-runs to establish a
deterministic structural signature.

| Metric | Description | Result |
|---|---|---|
| M1 | script SHA invariance across runs | INVARIANT |
| M2 | phase progression (forensic phase markers) | CONSISTENT |
| M3 | specific values (selected key numerics) | EXACT match across all 4 runs |
| M4 | top-level keys count | 23 (consistent) |
| M5 | Pattern 9 inline assertions | ALL PASS |

**G_DRY structural signature:**
`438d27f21d474b41e5d214314a72d8571f22b647e4fa31dd3cc1700e2131553d`
(stdout structural SHA, computed over lines 1-19 + 21-22; volatile L20
`frozen_utc` excluded by deterministic baseline policy).

## 7. step A retry (turn N+6, A1-A8 ALL PASS)

rev 6 commit mode, PowerShell -> WSL2 axis1_q2 conda env activation,
`source /home/sguccibnr32/miniconda3/etc/profile.d/conda.sh && conda activate axis1_q2`,
then transform script execution against canonical re-run JSON, writes
Layer C v1.2 candidate.

| # | Check | Result |
|---|---|---|
| A1 | invocation success | PASS |
| A2 | exit code 0 | PASS |
| A3 | output Layer C v1.2 candidate exists | PASS |
| A4 | output size sanity | PASS |
| A5 | output AST parse | PASS |
| A6 | **stdout structural SHA = G_DRY signature `438d27f2...` EXACT** | **CRITICAL PASS** |
| A7 | script SHA invariance (rev 6 final `ac91bb86...`) | PASS |
| A8 | rev 5 paired file invariance (`0ac7aa8b...`) | PASS |

**A6 critical PASS interpretation:** stdout structural SHA bit-exact
matches G_DRY signature confirms (a) `write_bytes` encoding integrity
(rev 2 Issue C fix preserved) + (b) commit / dry-run mode non-divergence
+ (c) Final SHA = In-memory serialization SHA (L-Q3-10 family preventive
behavior empirically confirmed).

step A artifact run-specific SHA: 11,989 B (forensic-only).

## 8. step C retry (turn N+7, C1-C8 ALL PASS, 97/97 EXACT — central milestone)

mirror v0.3 vs Layer C v1.2 candidate, WSL2 form.

| # | Check | Result |
|---|---|---|
| C1 | invocation success | PASS |
| C2 | leaves total = 97 / 97 | PASS |
| **bit-exact** | **n_pass=97 / n_fail=0 / overall: PASS** | **97/97 EXACT — 45 -> 0 FAIL transition** |
| C5 | per-bucket | (see §8.1 below) |
| C6 | per-leaf delta | 0.0 strict equality (1e-12 tolerance unused) |
| C7 | 3 input file SHA pre/post EXACT (Layer C v1.2 candidate / mirror script / input_files_pin) | PASS |
| C8 | (further envelope checks) | PASS |

### 8.1 Per-bucket Issue M sub-issue resolution detail

| Bucket | PASS / FAIL | Resolution |
|---|---|---|
| `per_field_results` | 36 / 0 | M-alpha resolved: `input_path` / `_v1_1_keys` family unification across passthrough vs filter dispatch |
| `per_mbin_gc[0..3]` | 28 / 0 | M-beta resolved: cross-fill from `logM_min`, `logM_max`, `logM_center` v1.1 keys at 6.2 dispatch |
| `field_consistency` | 6 / 0 | M-gamma resolved: Pattern 9 inline shape assertion (replaces canonical equality check, which is trivially true for pass-through) |
| `gc_M_star_slope` | 6 / 0 | M-gamma + M-delta resolved: `_slope` infix strip + filter + 3 dropped keys with explicit byte-weight account |
| `combined_3_field_chi2_C15_vs_MOND` | 14 / 0 | Issue J pre-fixed at hardening |
| `paper_claims_for_rounded_check` | 7 / 0 | PLACEHOLDER B pre-fixed at hardening |
| **Total** | **97 / 0** | **Issue M chain (M-α/β/γ/δ) ALL resolved at disk-artifact level** |

### 8.2 Physics-layer reproducibility intact

The following anchor 22 v0.1 publicly known values are bit-exact preserved:

```
G09 / G12 / G15 gc_a0:           preserved
combined gc_a0 (M-15 diag fit):  2.7184876317961755
combined dAIC  (M-17 full-cov):  472.2767831581
weighted_mean  (M-18/19):        2.7326473920108501 +/- 0.1072291448861956
consistency p  (M-24):           0.503592 (CONSISTENT)
slope:                           0.165849 +/- 0.041478
p_C15 / p_MOND:                  2.85e-02 / 6.38e-05
```

This is the operational verification COMPLETE state for the Issue M
chain comprehensive fix.

## 9. Minor scope addition (turn N+7 prologue, post-hoc accept turn N+8)

Claude Code-side defensive forensic preservation, parallel to step A
rev 5 `.tmp` rename pattern:

```
Rename-Item E:\Q-3_route_ii_mirror_2026-05-07\.tmp_phase_3_mirror_result.json
            -NewName .tmp_phase_3_mirror_result_rev5.json
```

Justification: handoff memo §3.3 "forensic anchor preservation" language
integrity; rule 1 + 7 compatible (SHA bit-exact preserved); cleanup
target enumeration transparency improved.

claude.ai-side post-hoc accept at turn N+8; codification candidate in
the L-Q3-11 batch as "Bidirectional verification protocol concrete
success case: defensive forensic preservation initiated by Claude Code
side autonomous prudent action".

## 10. step D cleanup (turn N+9, D1-D8 ALL PASS + cross-environment)

13 / 13 forensic transients removed (108,899 B total, all Public repo
**outside**, anchor structure untouched).

```
D:\...\C3 拡張版仮説関連2\               (9 files,  19,345 B): 0
  - g_dry_run.sh / g_dry_run_1..4.txt
  - step_a_retry.sh / step_a_retry_stdout.txt
  - step_c_retry.sh / step_c_retry_stdout.txt

E:\Q-3_route_ii_discovery_2026-05-07\    (.tmp rev 5 + rev 6 step A artifacts,  25,655 B): 0
E:\Q-3_route_ii_mirror_2026-05-07\       (.tmp rev 5 + rev 6 step C artifacts,  63,899 B): 0
```

| # | Check | Result |
|---|---|---|
| D1 | Public repo HEAD invariance (`86742b79...`) | PASS |
| D2 | porcelain v1 = 0 lines | PASS |
| D3 | rule 1 IMMUTABLE 7 anchors SHA pre/post invariance | PASS (final confirm) |
| D4 | rev 6 final form preserved (`ac91bb86...`) | PASS |
| D5 | rev 5 baseline forensic chain reference (`a7931889...`) preserved | PASS |
| D6 | 3 directory cleanup verified (count = 0) | PASS |
| D7 | (envelope check) | PASS |
| D8 | (envelope check) | PASS |

### 10.1 Cross-environment integrity

| View | Method | Result |
|---|---|---|
| Windows native | `Test-Path` against each `.tmp` path | False (all 13 absent) |
| WSL2 | `wsl -d Ubuntu ls /mnt/e/Q-3_route_ii_*/.tmp*` | no output (all absent) |
| IMMUTABLE anchor visibility (cross-env) | `wsl sha256sum` spot-check on mirror script | `71dfdc56...` bit-exact |

This achieves the most rigorous form of L-Q3-11 codification draft
"Six-layer countermeasure (3) cross-environment file system integrity"
empirically confirmed.

## 11. G_MIRROR final grant (turn N+10, claude.ai-side)

3 necessary conditions all met:

| # | Condition | Status |
|---|---|---|
| (1) | mirror v0.3 vs Layer C v1.2 candidate: 97/97 EXACT | MET (turn N+7) |
| (2) | forensic transients cleanup verified | MET (turn N+9) |
| (3) | step B path (b) confirmed (empirically reproducible) | MET (prior chat [3.4.1.5]) |

**G_MIRROR GRANTED.** Phase 3 [3.4.2-3.4.6] step C closure declared.
Phase 3 [3.4] step A / B / C trilogy all closed.

## 12. Phase 3 -> Phase 4 gate (7 / 7 PASS)

```
G3-1  G_MIRROR final grant                                  PASS
G3-2  rev 6 final form SHA preservation (ac91bb86...)       PASS
G3-3  rule 1 IMMUTABLE 7 anchors invariance                 PASS
G3-4  Public repo touch 0 invariant                         PASS
G3-5  physics-layer reproducibility intact                  PASS
G3-6  forensic transients cleanup verified                  PASS
G3-7  L-Q3-11 codification batch ready                      PASS
```

## 13. Public repo state at chat closure

```
HEAD          = origin/main = 86742b79246436fed58b9f2853cc6380c57304da
porcelain v1  = 0 lines
commit count  during round (turn N+4 ~ N+10): 0
tag count     during round: 0
push count    during round: 0
forensic transients in Public repo: none (all transients Public-repo-outside)
anchor structure modifications: none
rule 1+5+6+7+#26+92 compliance: all preserved
```

---

## Appendix A. Pre-flight ALL GREEN to G_MIRROR final grant — turn-by-turn timeline

| Turn | Activity | Key result |
|---|---|---|
| N+4 | 9-piece patch apply + Pattern 12 detection / remediation | rev 6.0 intermediate `df698d22...` |
| N+4 (V) | V1-V6 verify | V4 anomaly identified (argparse description) |
| N+4.5 | capture 11-A (Claude Code -> claude.ai) | argparse region L668-682 verbatim feed |
| N+4.6 | Piece 7 drafting (claude.ai-side) | 3-line block, 1-char mutation |
| N+4.7 | Piece 7 apply | rev 6 final form `ac91bb86...` |
| N+4.8 | V re-run (rev 6 final) | bit-precision audit: offset 30282 single-byte mutation |
| N+5 | G_DRY 4 sequential dry-runs | structural signature `438d27f2...` |
| N+6 | step A retry | A1-A8 ALL PASS, A6 critical PASS |
| N+7 | step C retry | **C1-C8 ALL PASS, 97/97 EXACT, 45 -> 0 FAIL transition** |
| N+8 | G_MIRROR preliminary judgment + step D GO | 3 conditions confirmed (1)(3) MET, (2) pending step D |
| N+9 | step D cleanup | D1-D8 ALL PASS + cross-environment integrity |
| N+10 | **G_MIRROR final grant + Phase 3 step C closure declaration** | Phase 3 -> Phase 4 gate 7 / 7 PASS |

---

*End of verification log. For lessons learned and codification batch
references, see `anchor_22_v0_2_lessons_appendix.md`.*
