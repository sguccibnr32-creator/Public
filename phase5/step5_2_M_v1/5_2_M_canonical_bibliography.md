# Phase 5-2 chain canonical bibliography (5.2.M v1 commit)

**Generated**: 2026-04-26T16:32:13Z
**Strategy**: α (Option β single canonical commit)
**Overall verdict**: PASS

---

## §6 Main result — paper inject text

The Phase 5-2 chain comprises 39 TIER-1 posterior derivations across
4 chain steps (B-1 FIRAS μ-distortion: 6;
B-2 Plik CMB TT: 6;
B-3 21cm cosmic dawn: 9;
B-4 FIRAS × 21cm joint: 18).
Of these, 6 derivations in B-4 (3 priors × 2 single-likelihood
combinations) bit-exactly reproduce upstream references in B-1
(B14_FIRAS_only ↔ B-1 one_sided) and B-3 (B14_21cm_only ↔ B-3 S_SARAS3)
within 1×10⁻⁴ dex tolerance, with verified log_diff = 0.000000 dex in all
6 cases. The canonical count of unique posterior derivations is
therefore **33**.

Across all 8 kernel/coupling variants of the membrane modification ansatz,
the Plik TT likelihood difference between the Path A central value
(ε_scale = 1.0026) and pure ΛCDM (ε_scale = 1.0) is bounded by
**Δχ² ≤ 0.013**, with a median of **Δχ² = 0.003**.
This is more than two orders of magnitude below the canonical 1σ threshold
(Δχ² = 1) and confirms that Plik TT data, restricted to the linear membrane
modification ansatz examined here, does not distinguish Path A from
standard ΛCDM at any statistical significance.

---

## §5 Methodology — paper inject text

The companion analysis includes a kernel/coupling sensitivity sweep
(8 variants spanning α: 0.005–0.05, ℓ_peak: 1st–4th acoustic peaks,
ℓ_width: 100–200) treated as TIER-2 audit material. Across this sweep,
informed-prior posterior medians (log_normal, flat_band) remain stable at
±0.003 of the Path A central value, while the uninformative-prior posterior
exhibits ±0.34 dex spread, indicating that prior-independent data-only
conclusions are kernel-fragile. The location of the χ² minimum
(ε@χ²_min ∈ [0.749, 0.975] across α scan) is a mathematical consequence of
the linear (ε−1) ansatz scaling and is not reported as a physical signal.
Kernel functional form impacts data sensitivity by ≥ 100× (width_2x
Δχ²(ε=1) = 0.006 vs peak_3rd Δχ²(ε=1) = 0.72), with the third-acoustic-peak-
localized kernel approaching but not reaching the 1σ threshold, motivating
dedicated peak-resolved analyses in future work.

A separate methodological note: B-2 v1 results with 5 missing nuisance
defaults (CIB index and sub-pixel correction amplitude unset) produced a
spurious +1 dex shift in the uninformative posterior, fully eliminated in
v2 with all 20 nuisance parameters set to Planck 2018 best-fit values.
This demonstrates the necessity of complete nuisance parameter
introspection before claiming any data-driven posterior shift as a
physical signal.

---

## §6 Method 2 narrative — paper inject text

ε_scale is operationally defined via Method 1 inverse calibration (ε_scale_A = 1.0026, Path A inverse closure). Formal physical interpretation, including potential connection to foundation_scale (χ_F · V''(φ_0) · T_m / ℏ ≈ 2.283e+35) and resolution of the (a)/(b) 96 dex coexistence flagged in Phase 4b 1-R retract-impossible #22-25 alpha-3, deferred to v4.9 patch round future work.

---

## Gate verdict summary

| Gate | Status | Detail |
|------|--------|--------|
| G1 struct loadable      | PASS | 6/6 chain struct.json files |
| G2 Path A consistency   | PASS | central=1.0025822368421053, band reproducibility |
| G3 Path B variants      | PASS | 39 TIER-1 − 6 overlap = 33 canonical |
| G4 Path C bit-exact     | PASS | local: canonical=True, archive=True; upstream 5.2.C: PASS |
| G5 chain numerical      | PASS | max_log_diff=0.000000e+00 dex, drift_max=0.0000 dex |
| G6 Method 2 status      | PASS | Track A Case_3_passing, deferred v4.9_patch_round |

---

## Retract-impossible invariants compliance

- **#22(vi)**: cascade SSoT 3 active bit-exact + 1 archived ✓
- **#26**: multi-route min: A + B-1/B-2/B-3/B-4 + C verified ✓
- **#29**: foundation scale numerical OK, Method 2 deferred ✓
- **#30**: no single-value commit (33 canonical + 1 Path A + 8 v4 audit) ✓
- **#32**: v_flat layer separation c=0.42 strict galaxy-specific ✓

---

## Variant tier accounting

| TIER | Content | Count |
|------|---------|-------|
| **TIER-1 canonical** | unique posterior derivations | 33 |
| TIER-1 reproductions | B-4 ↔ B-1/B-3 bit-exact (log_diff = 0) | 6 |
| TIER-1 total        | all chain step derivations | 39 |
| TIER-2 audit        | B-2 v4 kernel sensitivity sweep | 8 |
| Path A              | central + band edges | 2 |
| Path C              | cascade SSoT verification (gate, not posteriors) | 0 |


---

*End of canonical bibliography.*
