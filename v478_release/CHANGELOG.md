# Changelog

All notable changes to the membrane cosmology v4.7.x series.

## [v4.7.8] - 2026-04-16

### Added — Pressure-Supported Dwarf Spheroidal Extension

#### Theory
- **c→0 limit of U(ε; c) analyzed**: self-consistent equation Q→0 yields universal `s_0 = 1/(1 + exp(3/(2·T_m))) = 0.3515` at `T_m = √6`.
- **Non-analytic √Q expansion derived**: `s(Q) ~ s_0 + α_s √Q + O(Q)` with `α_s = 0.110`, originating from the ln(1−ε) term in `U(ε; c)`. Numerically verified to <4% relative error for Q < 0.3.
- **Bernoulli prediction**: `G_Strigari = s_0(1 − s_0) · a_0 = 0.228 a_0 = 2.74 × 10⁻¹¹ m/s²` (proposed as theoretical origin of the Strigari 2008 universal acceleration).
- **dSph gc formula**: `gc_dSph = [s_0(1 − s_0)]² · a_0² / g_bar` (derived, B-grade for individual galaxies).

#### Observations
- **dSph catalogue (N=31)**: 15 MW + 11 M31 + 3 isolated + 2 unknown. σ_los calibration ratio 0.997 (Walker coefficient 2.5 applied consistently).
- **Unified RAR plot**: SPARC RAR cloud (3,389 points / 175 galaxies) + dSph (N=31). Population medians: SPARC 2.75, bridge outer 12.76, dSph 35.11 (in Υ_dyn).
- **J3 transition band defined**: 10 < Υ_dyn < 30. Four SPARC bridge galaxies identified (ESO444-G084, NGC2915, NGC1705, NGC3741).

#### Verification
- **J3 regime inversion confirmed**: M_J,mem ≪ M_J,std in 28/31 dSph. Root cause of +1.5 dex systematic C15 offset identified (A-grade).
- **Bridge galaxy continuous transition**: 4/4 bridge galaxies show statistically significant radial increase in Υ_dyn(R) with |z| = 2.4 to 21.6. Mann-Whitney tests confirm outer > inner in 4/4 (p < 0.03). All four A-grade criteria satisfied.
- **Two-population Bernoulli reproduction**:
  - dSph 31 galaxies: g_obs median = 0.240 a_0, agreement 5% with 0.228 a_0
  - Bridge outer 30 points: g_obs median = 0.219 a_0, agreement 4% with 0.228 a_0
  - Populations differ by 10× mass scale and dynamical support (pressure vs rotation)

#### Tautology Separation
- **g_obs M_bar-independence confirmed**: multivariate regression yields slope −0.008 ± 0.049 (0.2σ null). This is the genuine non-tautological membrane prediction, verified at A-grade.
- **Strigari universality partially retracted**: g_obs shows a statistically significant r_h dependence (slope = −0.47 ± 0.14, −3.4σ). This reflects the empirical dSph σ-size scaling σ ∝ r_h^0.27 (McConnachie 2012, Brasseur et al. 2011), not predicted by mean-field c→0 analysis.
- **H4 hypothesis reinterpreted**: the apparent "gc ∝ M_dyn^−1" decomposes as `gc ∝ σ^0 × r_h^(−1.5)` — the σ contribution is null, with r_h^(−3/2) being the real driver.

#### Documentation
- New Section 7 (`Pressure-Supported Dwarf Spheroidal Extension`) added to arXiv paper (12 pp total).
- WordPress main page updated with new Section ⑩ (`dSph extension`).
- Internal Japanese sections: §6.10, §9.1, §0, §0.ii.
- Release notes: `RELEASE_NOTES_v4.7.8.md`.

### Changed — Existing v4.7.7 Content
- **Abstract**: extended with dSph results (+~11 lines).
- **Section 1 roadmap**: Section 7 (new) described.
- **Section 8 Conclusions** (formerly Section 7): items 15-18 added for dSph extension.
- **Priority future work**: added item (6) on theoretical origin of dSph σ-size relation.
- **References**: 10 additions (McConnachie 2012, Wolf+2010, Walker+2009, Strigari+2008, Collins+2013, Tollerud+2012, Simon & Geha 2007, Simon 2019, Brasseur+2011, Lewis+2007, Kacharov+2017).

### Deprecated / Retracted in v4.7.8
- **H3 hysteresis-only hypothesis**: rejected by three independent tests (isolated g_c > satellite g_c, null distance correlation, AIC inferior to constant-model).
- **Strict Strigari universality** (g_obs = universal constant): partially retracted. g_obs = s_0(1 − s_0) a_0 holds only as the ensemble mean, with systematic r_h dependence at −3.4σ in individual galaxies.
- Earlier candidate formulas for dSph gc that matched observations primarily through tautological slope=−1 structure are deprecated in favor of the explicitly predictive Bernoulli form.

### Grade Upgrades
- J3 regime inversion: B+ → **A** (28/31 confirmation + theoretical explanation)
- Bernoulli prediction G = 0.228 a_0: B+ → **A** (2-population independent reproduction)
- C15 → Strigari continuous transition: unverified → **A** (bridge galaxy 4/4, all criteria satisfied)

---

## [v4.7.6] - 2026-04-15

### Added
- §6.8.1 Partial first-principles derivation of Υ_d exponent (T_m=√6 gives β=-0.33, 8% difference from observed -0.361, B+ grade)
- §6.9 HSC-SSP Y3 2-halo contribution measured empirically (Δslope = +0.034 ± 0.064, 0.5σ suggestion)
- HSC slope tension reduced from 2.2σ to ~1σ (C15 +0.075 + Model B +0.017 + 2-halo +0.034 = +0.126 vs observed +0.166 ± 0.041)
- GitHub repository published (MIT License, 220 scripts, Release v4.7.6 with PDF attached)

---

## [v4.7] - 2026-04-13

### Added
- Condition 15 final form finalized: `g_c = 0.584 × Υ_d^(−0.361) × √(a_0 × v_flat²/h_R)`, scatter 0.286 dex (intrinsic limit)
- HSC-SSP Y3 + GAMA DR4 independent lensing RAR analysis (157,338 isolated lenses, 503M pairs, 3 fields) yielding g_c = 2.73 ± 0.11 a_0 with ΔAIC = +472 for C15 over MOND
- LITTLE THINGS 26-galaxy external verification integrated (α = 0.576 ± 0.047, p(0.5) = 0.109, cannot reject; MOND rejection strengthened to p = 4×10⁻²⁵)

### Changed
- κ=0 confirmed (membrane rigidity ~ 10⁻⁴⁰ kpc²), simplifying theory structure to single-layer C15 with L ≈ U(ε; c)
- N-1 (η derivation) resolved as external parameter
- N-1' (α=0.5 origin) resolved as deep-MOND consequence

### Deprecated / Retracted in v4.7
- **Condition 14 dynamical formulation** `g_c,eff(r)`: retracted (pattern is C15 bias projection onto radial coordinate)
- Gradient term `κ(dε/dr)²` in Lagrangian: eliminated (κ ≈ 0)
- Euler-Lagrange derivation pathway: abandoned
- S_gal predictive significance: retracted (0.89 correlation is structural, not predictive)

---

## Versioning

This project uses a progressive refinement versioning:
- **v4.7.x** (x = 0, 6, 8): theoretical and observational refinements within the v4.7 framework
- Each micro-version adds verified results and may deprecate earlier provisional claims
- `v4.7.8` is fully backward-compatible with `v4.7.6` (data and scripts); Section 7 extension is purely additive to the established rotation-supported framework.
