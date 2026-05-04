# §2.5 v0.2 SPARC Empirical Execution — Execution Plan v1.0

**File**: `EXECUTION_PLAN_section2_5_v0_2.md`
**Date**: 2026-05-03
**Author**: 坂口 忍 / 坂口製麺所、宍粟市
**License**: CC-BY 4.0
**Encoding**: UTF-8 / LF / no BOM

**Companion deliverables (this prep package)**:
- `OUTPUT_SCHEMA_section2_5_v0_2.md` v1.0 (data contract spec)
- `run_section2_5_v0_2.py` v1.0 (executable script)

**Upstream LOCK references**:
- anchor 7  (§2.5 v0.1,         SHA 9e03f53e) — predecessor; R-1 / R-2 / Path (iii) / Q-LOCK preserved
- anchor 8  (§2.6 v0.1,         SHA f6a48b51) — chapter-level milestone (anchor 21 base)
- anchor 14 (§4 v0.4,           SHA 295bc05c) — Layer B-α/B-β + NGC 3198 (f_p=0.9930, c_mem=0.0070)
- anchor 17 (§3 v0.2,           SHA 178dad11) — c_super=0.5709
- anchor 19 (§1 v0.4,           SHA 0b269c10) — A 級 prerequisite (path A 0.5629 / path B 0.5649)
- anchor 21 (§2 closure v0.1.1, SHA 44df9afb) — chapter-level §2 LOCK ESTABLISHED, F4/F12 RESOLVED

**Execution environment**: Windows + Claude Code (claude.ai container cannot access master / SPARC / dSph data).

**Wall clock estimate**: ~80-105 min (full run, 171 SPARC galaxies + 31 dSph + S-1〜S-6 resolve + F8 k_E sensitivity audit)

---

## §0 Scope and goals

This plan executes §2.5 of the J-system companion paper at version v0.2, building on anchor 7 v0.1 numerical baselines. The four operational goals are:

1. **Algorithm B per-galaxy convergence measurement** — establish the convergence rate (per-galaxy) of the simultaneous self-consistency loop defined in anchor 7 §2.5.3 (R-2 LOCK), as a function of init c_galaxy=0.42 / tol=1e-6 / N_max=50.
2. **f_E adoption fraction determination** — measure the fraction of SPARC 171-galaxy fit pool where the E pipeline (anchor 7 §2.5.2, k_E=2 default Q-C1 LOCK) is preferred over Algorithm B by ΔAIC selection (anchor 7 §2.5.4).
3. **dSph J3 consistency check + b_α 3-axis audit** — re-verify the 28/31 baseline (anchor 7 §2.5.5) and the b_α SPARC=0.1084 / dSph=0.1127 / |Δ|=0.0042 Phase C3 cross-paper coherence (anchor 19 §1.5).
4. **Sub-issue S-1〜S-6 numerical resolve** — close the 6-item v0.2 deferral structure (anchor 7 §2.5.6 [H-1] protocol) including reference resolution for F1/F2/F6 (category A) and operational protocol resolution for F8 (category B, k_E sensitivity audit).

**Non-goal**: This plan does NOT modify anchor 7 v0.1 (R-1 LOCK preserved); it produces a NEW anchor 22 (`J_system_paper_section2_5_v0.2.md`) as a v0.2 successor that supersedes v0.1 in a 純 additive manner.

**Anchor 21 chapter-level §2 LOCK consistency**: All v0.2 outcomes must be consistent with anchor 21 v0.1.1 chapter-level §2 LOCK ESTABLISHED status. F4 and F12 chapter-level RESOLVED status MUST NOT regress; F1/F2/F6/F8 placeholder + handoff design is the SOURCE of this round's numerical resolve.

---

## §1 Environment and inputs

### §1.1 Environment variables (5 + 2 = 7 items, define before run)

| Variable | Required | Description | Example |
|---|---|---|---|
| `MASTER_ROOT` | yes | anchor master MD root | `D:\ドキュメント\エントロピー\膜宇宙論再考察AB効果有り\C3 拡張版仮説関連2` |
| `PARENT_MASTER_ROOT` | yes | parent v4.8 publication final root | `$env:MASTER_ROOT\arxiv\v48_release` |
| `SPARC_TA3_PATH` | yes | TA3 file (gc_over_a0 in a₀ units) | `<user supplies>` |
| `SPARC_PHASE1_PATH` | yes | phase1 file (ud = Υ_d) | `<user supplies>` |
| `SPARC_MRT_PATH` | yes | MRT fixed-width catalog | `<user supplies>` |
| `DSPH_DATASET_PATH` | yes | dSph 31 sample dataset | `<user supplies>` |
| `OUTPUT_ROOT` | yes | output directory parent | `C:\build\section2_5_v0_2_runs` |

The script `run_section2_5_v0_2.py` validates these at startup (raises `EnvironmentError` if missing) and records all values + input file SHA256 in `manifest.json`.

### §1.2 Three-file loader (anchor 7 §2.5.1 + project mandatory pattern)

| Role | Content | Required columns |
|---|---|---|
| TA3 | gc_over_a0 in a₀ units (always) | `galaxy`, `gc_over_a0` |
| phase1 | ud = Υ_d (mass-to-light ratio of disk) | `galaxy`, `Ud` |
| MRT | fixed-width catalog | `galaxy`, `T`, `MHI`, `L36`, `Rdisk`, `SBdisk0`, `Vflat` |

Derived (in script):

- `Mgas = 1.33 × MHI` (helium correction factor 1.33)
- `Mstar = Ud × L36`
- `Mbar = Mstar + Mgas`
- `gc_obs = gc_over_a0 × a₀` with a₀ = 1.2×10⁻¹⁰ m/s²

### §1.3 dSph dataset (anchor 7 §2.5.5)

Required per-dwarf columns (subject to F6 audit re-evaluation):

- `dwarf` (name)
- `J_factor` (annihilation J, log10(GeV²/cm⁵))
- `sigma_los` (line-of-sight velocity dispersion, km/s)
- `Rhalf` (half-light radius, pc)
- `J3_metric` (anchor 7 §2.5.5 definition — TODO_USER_VERIFY exact formula)

### §1.4 Anchor SHA references (forensic chain, recorded in manifest.json)

See `OUTPUT_SCHEMA_section2_5_v0_2.md` §3.5. Ten upstream anchors cited (5/6/7/8/14/16/17/19/20/21).

### §1.5 Numerical constants (LOCKED, do not modify)

| Symbol | Value | Source |
|---|---|---|
| a₀ | 1.2×10⁻¹⁰ m/s² | MOND universal |
| T_m | √6 | anchor 8 §2.6 / anchor 20 §C |
| C_anchors | {0.30, 0.42, 0.618, 0.80, 1.00} | parent v4.8 §3.1 L335 SHA 902f79c6 immutable |
| c_galaxy_init | 0.42 | anchor 7 §2.5.3 R-2 LOCK |
| c_cascade | 0.83 | anchor 6 boundary r→∞ |
| c_super | 0.5709 | anchor 17 §3.8 §3-B injection canonical |
| s_0 (Bernoulli at Q→0) | 0.3515 | anchor 16 / anchor 7 §2.5.5 |
| G_Strigari/a₀ | 0.227948 | s_0(1−s_0) |
| Helium factor | 1.33 | Mgas = 1.33×MHI |
| η_0 (locked prefactor) | 0.584 | anchor 7 / 14 / 19 |
| α (locked) | 0.5 | algebraic deep-MOND |
| β (locked) | −0.361 | anchor 19 baseline |
| tol | 1e-6 | Algorithm B |
| N_max | 50 | Algorithm B |
| k_E (default LOCK) | 2 | Q-C1 LOCK |
| k_E (sensitivity variant) | 1 | F8 audit |
| k_B | 0 | B canonical parameter-free |

---

## §2 Algorithm B per-galaxy convergence measurement (anchor 7 §2.5.3 R-2 LOCK)

### §2.1 Loop definition

```
init  : c_galaxy ← 0.42
loop  : for n in 0 .. N_max:
          c_galaxy_n+1 ← AlgorithmB_step(c_galaxy_n, galaxy_data)
          residual_n ← |c_galaxy_n+1 − c_galaxy_n|
          if residual_n < tol: break (converged)
        endfor
output: (converged: bool, n_iter: int, c_galaxy_final: float, residual_final: float)
```

### §2.2 Step function (TODO_USER_VERIFY — placeholder in script)

The script `run_section2_5_v0_2.py` `_placeholder_c_update()` is a heuristic damped fixed-point. **It must be replaced** with the actual anchor 7 §2.5.3 step formula before scientific use. The placeholder converges quickly to `0.5*(c_init + c_cascade)` regardless of galaxy data — this will produce meaningless results.

### §2.3 Convergence rate metric

For the 171-galaxy fit pool:

- **Galaxy-level**: per-galaxy `(converged, n_iter, residual_final)` recorded in `per_galaxy_sparc.csv` and `algorithm_b_log.csv` (full per-iteration trace).
- **Aggregate**: `convergence_rate = n_converged / 171`, `mean_n_iter`, `median_n_iter`, `max_n_iter` recorded in `summary.json::algorithm_b`.

### §2.4 Acceptance threshold

`AC1: convergence_rate ≥ 0.95` (suggested; final threshold = TODO_USER_INPUT). Galaxies failing convergence are flagged `converged_B=False` and contribute to anchor 22 §2.5.3 as known sub-population (not silently dropped).

### §2.5 Wall clock estimate

171 galaxies × ~50 iterations × O(1) operations per iteration ≈ ~30-45 min on a single thread (assuming the actual step function is moderately complex, e.g., involves brentq inversion of f_opt per call).

---

## §3 f_E adoption fraction (anchor 7 §2.5.2 + §2.5.4, Q-C1 LOCK)

### §3.1 E pipeline (k_E=2 default LOCK)

E pipeline parameters:

- k_E = 2 (default, **Q-C1 LOCK**)
- score: per anchor 7 §2.5.2 likelihood / AIC formula (TODO_USER_VERIFY)

### §3.2 Algorithm B (k_B=0 canonical)

B pipeline parameters:

- k_B = 0 (parameter-free, **R-1 LOCK** "non-parametric")
- score: per anchor 7 §2.5.3 likelihood / AIC formula

### §3.3 ΔAIC selection (anchor 7 §2.5.4)

```
ΔAIC = AIC_E − AIC_B    (sign convention: positive favors B)
selected = "B" if ΔAIC > threshold else "E"
```

`threshold` = TODO_USER_INPUT (typical AIC selection thresholds: 2, 4, 7, 10).

### §3.4 f_E adoption fraction

```
f_E = n_selected_E / 171
```

Recorded per-galaxy (`selected_pipeline`, `delta_AIC`) and aggregate (`f_E_adoption.f_E_fraction`).

### §3.5 F8 k_E sensitivity audit

In addition to the k_E=2 default run, the script re-runs E pipeline with k_E=1 (sensitivity variant). Three quantities:

| Quantity | Computation | Acceptance |
|---|---|---|
| `k_E_2_default_observed` | f_E with k_E=2 | baseline |
| `k_E_1_sensitivity_observed` | f_E with k_E=1 | comparison |
| `delta_AIC_k_E_2_minus_k_E_1` | difference in mean AIC | should remain consistent with k_E=2 preference |

**Q-C1 LOCK preservation**: k_E=2 must remain the preferred default after audit. If k_E=1 yields lower mean AIC AND statistically significant improvement, this triggers a Q-C1 LOCK review (severity event, would require parent paper / anchor 7 R-1 LOCK consultation — NOT silently overridden).

---

## §4 dSph J3 consistency check + b_α 3-axis audit (anchor 7 §2.5.5)

### §4.1 J3 baseline maintenance

Anchor 7 baseline: 28/31 dSph dwarfs pass J3 metric.

v0.2 outcome: re-evaluate against the same metric; record per-dwarf `J3_pass` and aggregate `n_pass_observed`.

**Acceptance**: `n_pass_observed ≥ 28` (no regression) is required. Improvements (e.g., 30/31 per F6 audit) are bonus.

### §4.2 F6 audit: "dSph 30 sample chapter 番号 placeholder"

Per anchor 8 §2.6.5 verbatim, F6 references "Phase C3 v3 §X dSph 30 sample chapter 番号" (X = TODO_USER_INPUT item 6). The 30 vs 28 distinction reflects an alternate inclusion criterion.

The script records BOTH `in_28_baseline` (anchor 7) and `in_30_baseline` (F6 alternate) per-dwarf in `per_dsph.csv`, allowing anchor 22 to discuss the variation transparently.

### §4.3 b_α 3-axis audit

Per anchor 7 §2.5.5, b_α is computed on three axes:

| Axis | Baseline value (anchor 7 / 19) |
|---|---|
| axis 1: SPARC density-weighted | b_α_SPARC = 0.1084 |
| axis 2: dSph density-weighted | b_α_dSph = 0.1127 |
| axis 3: combined (Phase C3 cross-paper coherence) | (anchor 7 §2.5.5 spec) |

v0.2 must reproduce these three values within tolerance, with `|diff| ≤ 0.005` (suggested AC4; anchor 19 baseline = 0.0042 i.e. ~0.5% agreement preserved).

**Critical**: this verifies Phase C3 cross-paper coherence; a deviation >0.005 would propagate to anchor 21 chapter-level §2 LOCK consistency check (potentially regressing the chapter-level closure).

### §4.4 Strigari G_Strigari/a₀ prediction verify

`G_Strigari/a_0 = s_0(1−s_0) = 0.227948 ≈ 0.228` (T_m=√6 first-principle derivation, anchor 7 §2.5.5 + anchor 16 §5).

v0.2 records observed mean across the 31 dSph and reports agreement %. Baseline target: dSph 5% / bridge 4% agreement (anchor 19 / 20).

---

## §5 Sub-issue S-1〜S-6 numerical resolve (anchor 7 §2.5.6 [H-1] + 3-tier deferral category)

3-tier category (anchor 7 R-2 LOCK):

- **A: reference resolution** — S-1, S-3 (each maps to a literal §-line reference that needs to be filled)
- **B: operational protocol** — S-2, S-4, S-5 (each maps to a runtime decision rule that needs to be defined)
- **C: numerical threshold** — S-6 (a single numerical threshold determined from per-galaxy data distribution)

### §5.1 S-1 (category A) — F2 link

- **anchor 8 §2.6.5 verbatim** (F2): "Phase C3 v3 §X Lesson 91 chapter placeholder (bridge pre-cut 4 galaxy reference)"
- **Resolution method**: literal reference resolution — substitute concrete §X chapter number
- **Input required (TODO_USER_INPUT item 5)**: Phase C3 v3 §X chapter number
- **Output**: `sub_issues.json::S_1.resolution_value` populated with `"§<n>"`
- **F-flag update**: F2 anchor 21 placeholder → RESOLVED in anchor 22

### §5.2 S-2 (category B) — operational protocol

- **Resolution method**: define a runtime decision rule per anchor 7 §2.5.6 [H-1]
- **Input required**: actual operational protocol spec (TODO_USER_INPUT)
- **Output**: `sub_issues.json::S_2.resolution_value`

### §5.3 S-3 (category A) — F1 link

- **anchor 8 §2.6.5 verbatim** (F1): "parent v4.8 §6 line XXX (σ_g(r) observational uncertainty reference)"
- **Resolution method**: literal reference resolution — substitute concrete line number
- **Input required (TODO_USER_INPUT item 4)**: parent v4.8 §6 line XXX → fill XXX
- **Output**: `sub_issues.json::S_3.resolution_value` populated with concrete line number
- **F-flag update**: F1 anchor 21 placeholder → RESOLVED in anchor 22

### §5.4 S-4, §5.5 S-5 (category B)

Both per anchor 7 §2.5.6 [H-1] operational protocol — TODO_USER_INPUT for spec.

### §5.6 S-6 (category C) — numerical threshold

- **Resolution method**: determine threshold from per-galaxy data distribution (e.g., percentile of residuals, ΔAIC quantile, etc.)
- **Threshold target**: TODO_USER_INPUT
- **Threshold observed**: computed from `per_galaxy_sparc.csv`

---

## §6 F1/F2/F6/F8 reference resolution mapping

This table summarizes the chain: anchor 8 §2.6.5 verbatim → anchor 21 v0.1.1 placeholder/handoff → §2.5 v0.2 round resolve.

| F-flag | anchor 8 verbatim | anchor 21 v0.1.1 status | v0.2 round resolution |
|---|---|---|---|
| F1 | parent v4.8 §6 line XXX (σ_g(r) obs uncertainty ref) | placeholder 維持 + §2.5 v0.2 handoff | S-3 (category A) literal reference resolution → resolved with line number filled |
| F2 | Phase C3 v3 §X Lesson 91 chapter placeholder (bridge pre-cut 4 galaxy ref) | placeholder 維持 + §2.5 v0.2 handoff | S-1 (category A) literal reference resolution → resolved with §X chapter filled |
| F6 | Phase C3 v3 §X dSph 30 sample chapter 番号 placeholder | placeholder 維持 (handoff continuation, F1/F2 同質) | F6 audit: re-evaluate dSph 30/31 inclusion alongside 28/31 baseline; chapter §X filled if needed |
| F8 | k_E sensitivity variant k_E=1 supplement | placeholder 維持 (Q-C1 LOCK 維持 + audit pending) | category B: dual-pass k_E=2 (default) + k_E=1 (sensitivity), record ΔAIC; Q-C1 LOCK preservation verified |

After v0.2 execution, anchor 22 will declare these 4 flags RESOLVED with concrete numerical / reference values, completing the §2 chapter-level closure beyond anchor 21 v0.1.1 PROPOSED status. This represents the explicit chapter-level §2 LOCK to fully formal effective state across all flags (not just F4/F12).

---

## §7 Output spec

See `OUTPUT_SCHEMA_section2_5_v0_2.md` v1.0 in full. Summary:

- 5 CSV files: `per_galaxy_sparc.csv` (171), `per_galaxy_sparc_full.csv` (175), `per_dsph.csv` (31), `algorithm_b_log.csv` (per-iteration), `f_opt_anchor_table.csv` (5-anchor LOCKED reference)
- 6 JSON files: `summary.json`, `sub_issues.json`, `promotion.json`, `f_flag_status.json`, `run_config.json`, `manifest.json`
- 2 ancillary: `SHA256SUMS.txt`, `run.log`

All outputs go to `$env:OUTPUT_ROOT/section2_5_v0_2_<RUN_ID>/`.

---

## §8 Acceptance criteria (v0.2 finalize go/no-go)

| AC | Criterion | Threshold | Source |
|---|---|---|---|
| AC1 | Algorithm B convergence rate | ≥ 0.95 (suggested) | §2.4 above |
| AC2 | f_E adoption fraction in expected range | TODO_USER_INPUT range | §3.4 above |
| AC3 | dSph J3 ≥ 28/31 (no regression) | n_pass ≥ 28 | §4.1 above |
| AC4 | b_α \|Δ\| (SPARC vs dSph) | ≤ 0.005 | §4.3 above (anchor 19 baseline 0.0042) |
| AC5 | All S-1〜S-6 RESOLVED | each has resolution_value | §5 above |
| AC6 | Q-C1 LOCK preserved (k_E=2 default) | k_E=2 remains preferred over k_E=1 by ΔAIC | §3.5 above |
| AC7 | anchor 21 chapter-level consistency | F4/F12 RESOLVED maintained, 10-axis audit no regression, F1/F2/F6/F8 RESOLVED in anchor 22 (forward) | §6 above |

`v0_2_finalize_eligible = true` only if all 7 AC pass.

If any AC fails, the run is marked `FAILED_<reason>/` and anchor 22 is NOT issued; remediation required (typically: investigate failing AC, possibly revise step function, re-run).

---

## §9 Sequence + wall clock estimate

| Step | Operation | Wall clock | Cumulative |
|---|---|---|---|
| 1 | Pre-flight env vars + input SHA verification | ~30 s | 0.5 min |
| 2 | Three-file load + merge (TA3 + phase1 + MRT, ~175 rows) | ~10 s | 1 min |
| 3 | dSph load (~31 rows) | ~5 s | 1 min |
| 4 | Algorithm B per-galaxy (171 × ≤50 iter, with f_opt brentq inversion) | **~30-45 min** | ~45 min |
| 5 | f_E adoption k_E=2 (171 galaxies, AIC scoring) | ~5-10 min | ~55 min |
| 6 | F8 k_E=1 sensitivity re-run | ~5-10 min | ~65 min |
| 7 | dSph J3 + b_α 3-axis audit | ~5-15 min | ~75 min |
| 8 | S-1〜S-6 resolve (mostly fast aggregation; S-1/S-3 reference resolution near-instant) | ~10-20 min | ~90 min |
| 9 | Output writing + SHA computation + manifest | ~1-2 min | ~95 min |
| **Total** | | **~80-105 min** | |

If the wall clock far exceeds 105 min: investigate the per-galaxy f_opt inversion (`scipy.optimize.brentq` may need bracket tuning) or parallelize the per-galaxy loop with `multiprocessing.Pool`.

---

## §10 Pre-flight checklist + TODO_USER_INPUT items

### §10.1 Pre-flight checklist (verify before running)

- [ ] All 7 environment variables set (`MASTER_ROOT`, `PARENT_MASTER_ROOT`, `SPARC_TA3_PATH`, `SPARC_PHASE1_PATH`, `SPARC_MRT_PATH`, `DSPH_DATASET_PATH`, `OUTPUT_ROOT`)
- [ ] All 4 input files exist and are readable
- [ ] Python 3.9+ with `numpy`, `scipy`, `pandas` installed
- [ ] `git` available (for `git rev-parse --short HEAD` in run_id)
- [ ] `OUTPUT_ROOT` directory exists or is writable
- [ ] Anchor 21 v0.1.1 SHA reference (`44df9afb...`) is recorded in script (constants section, line `ANCHOR_21_SHA_REFERENCE`)
- [ ] All TODO_USER_INPUT items below are filled (for scientific use; for `--dry-run` validation, only env vars + file paths needed)

### §10.2 TODO_USER_INPUT items (8 items, fill in `run_section2_5_v0_2.py`)

| # | Item | Location in script | Source |
|---|---|---|---|
| 1 | `V_DOUBLE_PRIME_AT_X_HALF` 5-anchor values | constants section | parent v4.8 §3.1 L335 SHA 902f79c6 (or anchor 7 §2.5.3 if recorded there) |
| 2 | `chi_coh()` closed-form formula | §6 of script | anchor 13/14 closed-form C3-A2 explicit cite |
| 3 | `_placeholder_c_update()` → real Algorithm B step | §7 of script | anchor 7 §2.5.3 step formula |
| 4 | F1: parent v4.8 §6 line **XXX** | `resolve_S3()` | parent v4.8 §6 (find line ref to σ_g(r) obs uncertainty) |
| 5 | F2: Phase C3 v3 **§X** chapter | `resolve_S1()` | Phase C3 v3 chapter map (Lesson 91 / bridge pre-cut 4 galaxy section) |
| 6 | F6: Phase C3 v3 **§X** dSph 30 chapter | dSph audit logic | Phase C3 v3 chapter map (dSph 30 sample section) |
| 7 | `DELTA_AIC_THRESHOLD` | constants section | anchor 7 §2.5.4 selection criterion (typical: 2, 4, 7, 10) |
| 8 | `EXCLUDED_4_SPARC_GALAXIES` (4 names + reasons) | constants section | anchor 7 §2.5.1 fit-pool definition (which 4 of 175 are excluded from 171 fit-pool) |

Plus 3 verification items (not strictly TODO but USER_VERIFY):

- T9: TA3 column delimiter / column names
- T10: phase1 column names
- T11: MRT fixed-width column specs (the placeholder colspecs in script may need adjustment)

### §10.3 Validation run (recommended before full run)

```powershell
python run_section2_5_v0_2.py --dry-run
```

Validates env vars + input file paths + SHA computation only. Should complete in <30 s.

---

## §11 Hand-off to anchor 22 drafting

After successful execution (all 7 AC pass, `v0_2_finalize_eligible = true`):

1. **Output snapshot SHA recording**: `manifest.json` and `SHA256SUMS.txt` capture all output file SHAs. These become the **input SHAs** for anchor 22.
2. **Anchor 22 drafting**: Use `OUTPUT_SCHEMA_section2_5_v0_2.md` §5.1 paraphrase points table to identify which numerical results enter the v0.2 anchor. Bit-exact preservation (6 sig figs).
3. **Anchor 22 SHA chain integration**:
   - Predecessor: anchor 7 (§2.5 v0.1, 9e03f53e) IMMUTABLE retained
   - Anchor 22 純 additive supersession (R-1 LOCK preserved)
   - Forward references: anchor 21 v0.1.1 chapter-level §2 LOCK (44df9afb...)
   - Output references: `manifest.json::output_files[].sha256` for each numerical claim
4. **Companion-internal F-flag closure**: anchor 22 declares F1, F2, F6, F8 RESOLVED (moving from anchor 21 v0.1.1 placeholder + handoff state to formal RESOLVED via §2.5 v0.2 numerical resolution). This completes the chapter-level §2 LOCK across all 14 companion-internal F-flags.
5. **TIER progression**: With anchor 22 + anchor 21 v0.1.1, §2 chapter-level closure becomes fully RESOLVED across all flag categories. TIER-3 milestone declaration unblock proceeds in concert with §3-§7 chapter-level closures (templates from anchor 21).

---

## Appendix A: Anchor SHA cross-reference table

| Anchor | File | SHA prefix |
|---|---|---|
| 5 | J_system_paper_section2_4_v0.1.md | 3270fb40 |
| 6 | J_system_paper_section2_1to3_v0.1.md | 6ac356c3 |
| 7 | J_system_paper_section2_5_v0.1.md | 9e03f53e |
| 8 | J_system_paper_section2_6_v0.1.md | f6a48b51 |
| 14 | J_system_paper_section4_v0.4.md | 295bc05c |
| 16 | J_system_paper_section5_v0.2.1.md | 69678018 |
| 17 | J_system_paper_section3_v0.2.md | 178dad11 |
| 19 | J_system_paper_section1_v0.4.md | 0b269c10 |
| 20 | J_system_v0.1_milestone_summary.md | 56afa4c2 |
| 21 | J_system_paper_section2_closure_v0.1.md | 44df9afbe9b57269405e88a6a824c447f720c946e81ec171ea965527c456322f |
| **22** | **J_system_paper_section2_5_v0.2.md** (post-execution future) | **TBD (after v0.2 finalize)** |

Parent v4.8 3-track SHA (referenced in `manifest.json`):

- latex_v48 (LuaLaTeX, ja): tex 902f79c6, pdf b8a88f04
- latex_v48_en (pdfLaTeX, en): tex 2dcf69e6
- latex_v478 (pdfLaTeX, en short): tex b7bf9629

---

## Appendix B: Numerical constants recap

```
a_0                    = 1.2e-10  m/s²
T_m                    = sqrt(6)
C_anchors              = (0.30, 0.42, 0.618, 0.80, 1.00)
c_galaxy_init          = 0.42
c_cascade              = 0.83
c_super                = 0.5709
s_0 (Bernoulli @ Q→0)  = 0.3515
G_Strigari/a_0         = 0.227948  (= s_0 * (1 - s_0))
helium_factor          = 1.33      (Mgas = 1.33 × MHI)

ETA_0_LOCKED           = 0.584
ALPHA_LOCKED           = 0.5       (algebraic deep-MOND)
BETA_LOCKED            = -0.361

PATH_A_ETA_0_BASELINE  = 0.5629    (Δ = -3.61% T6 PASS)
PATH_B_ETA_0_BASELINE  = 0.5649    (Δ = -3.27% T4 PASS)
T6_TOLERANCE_PCT       = 5.0
T4_TOLERANCE_PCT       = 5.0

B_ALPHA_SPARC_BASELINE = 0.1084
B_ALPHA_DSPH_BASELINE  = 0.1127
B_ALPHA_ABS_DIFF_BASELINE = 0.0042

tol                    = 1e-6
N_max                  = 50
k_E_default            = 2  (Q-C1 LOCK)
k_E_sensitivity        = 1  (F8 audit)
k_B                    = 0  (R-1 LOCK non-parametric)

scatter_dex_baseline   = 0.286
R_squared_baseline     = 0.607
MOND_rejection_p       = 1.66e-53  (anchor 7 §2.5 + v4.8 v4.7.8 result)
```

---

## Appendix C: Forensic chain compliance checklist (per anchor 20 §J 7-item)

| # | Rule | Compliance plan |
|---|---|---|
| 1 | SHA chain index updated | `manifest.json::input_files` + `SHA256SUMS.txt` capture all SHAs at run time |
| 2 | predecessor anchors cited bit-exactly | `manifest.json::anchor_references` lists 10 anchors with SHA prefix |
| 3 | no retroactive change to anchor 7 wording | anchor 7 (9e03f53e) NOT modified; anchor 22 純 additive supersession |
| 4 | forward-ref 0 strict | anchor 22 v0.2 will reference only anchors 1-21 (no forward refs to future anchors) |
| 5 | companion 純 additive (parent v4.8 NULL impact) | parent v4.8 SHAs recorded for verify; no parent file modification |
| 6 | closure declaration explicit | anchor 22 will declare §2.5 v0.2 closure + chapter-level §2 LOCK formal effective |
| 7 | cross-check audit re-runs | b_α 3-axis audit + 10-axis dependency audit (anchor 21) re-runs |

---

**END EXECUTION_PLAN_section2_5_v0_2.md v1.0**
