# Data Directory

Analysis output files (CSV + JSON) from the v4.7.8 pipeline.

## Primary Data Files

### `rar_unified_galaxy_level.csv` (206 rows, 12 columns)
Galaxy-level scalars for the combined SPARC + dSph sample.

| Column | Type | Description |
|---|---|---|
| source | string | "SPARC" or "dSph" |
| galaxy | string | Galaxy name |
| log_gobs_a0 | float | log₁₀(g_obs / a₀) |
| log_gbar_a0 | float | log₁₀(g_bar / a₀) |
| upsilon_dyn | float | g_obs / g_bar (non-log) |
| gc_over_a0 | float | C15 gc measurement (a₀ units) |
| upsilon_d | float | Stellar M/L ratio (SPARC) or NaN (dSph) |
| v_flat_kms | float | Flat rotation velocity (SPARC) or NaN |
| rs_tanh_kpc | float | Scale radius from tanh fit (SPARC) or NaN |
| host | string | "SPARC", "MW", "M31", "Isolated", or "Unknown" |
| sigma_kms | float | σ_los (dSph) or NaN (SPARC) |
| rh_pc | float | Half-light radius (dSph) or NaN (SPARC) |

### `rar_sparc_cloud.csv` (3389 rows, 5 columns)
SPARC RAR cloud: all radius points from Rotmod_LTG files.

| Column | Type | Description |
|---|---|---|
| galaxy | string | Galaxy name (175 unique) |
| R_kpc | float | Radius in kpc |
| Vobs_kms | float | Observed rotation velocity |
| log_gobs_a0 | float | log₁₀(V_obs² / R / a₀) |
| log_gbar_a0 | float | log₁₀(baryonic acceleration / a₀) |

### `sparc_j3_candidates_ranked.csv` (175 rows)
SPARC galaxies ranked by outer Υ_dyn (descending).

| Column | Description |
|---|---|
| galaxy, upsilon_dyn_outer_med, v_flat_kms, rs_tanh_kpc, R_max_kpc, Vobs_max_kms, N_points, upsilon_d, gc_over_a0, log_gbar_outer_med, log_gobs_outer_med | ... |

Bridge galaxies (outer Υ_dyn > 10): top 4 rows = ESO444-G084, NGC2915, NGC1705, NGC3741.

### `sparc_outer_upsilon.csv` (175 rows)
Outer-half Upsilon_dyn statistics for all SPARC galaxies.

### `j3_threshold_summary.csv` (4 rows)
J3 threshold statistics table (thresholds = 3, 10, 30, 100).

| Column | Description |
|---|---|
| threshold | Υ_dyn threshold (3, 10, 30, 100) |
| SPARC_points | Fraction of 3,389 radius points exceeding threshold |
| SPARC_galaxies_any | SPARC galaxies with any radius above threshold (out of 175) |
| SPARC_galaxies_outer | SPARC galaxies with outer 25% median above threshold |
| dSph_galaxies | dSph galaxies above threshold (out of 31) |

## Theory and Candidate Analysis

### `dsph_gc_candidates_v1.csv` (31 rows)
Per-galaxy dSph gc predictions from candidate formulas.

| Column | Description |
|---|---|
| galaxy, host | Galaxy name and host classification |
| log_gbar_a0, log_gobs_a0 | Observed accelerations |
| log_gc_obs, gc_obs_a0 | Observed gc (deep-MOND extraction) |
| gc_C | Candidate C: (s₀·a₀)² / g_bar |
| gc_D | Candidate D: (G_fit)² / g_bar with G_fit = 0.240 a₀ |
| gc_E | Candidate E: SCE SPARC formula (failed) |

### `gc_candidates_residuals.csv` (5 rows)
Summary of all 5 candidate formulas.

| Column | Description |
|---|---|
| candidate | Formula label (A, B, C, D, E) |
| bias_dex | Mean residual (dex) |
| scatter_dex | Residual scatter (dex) |
| type | "const", "Strigari_fixed", "Strigari_fit", "SCE_SPARC" |

### `theory_key_values.csv` (9 rows)
Key theoretical and observational constants.

| Quantity | Value |
|---|---|
| T_m | √6 = 2.449 |
| s_0 | 1/(1 + exp(3/(2·T_m))) = 0.3515 |
| alpha_s | Small-Q correction coefficient = 0.110 |
| dU_at_c_0 | ΔU(c→0) = -3/2 |
| obs_gc_median_a0 | 5.69 |
| obs_gc_median_dex | 0.755 |
| G_Strigari_fit_a0 | 0.240 (fitted) |
| G_Strigari_fit_SI_ms2 | 2.876 × 10⁻¹¹ m/s² |
| gap_alpha_over_s0 | 16.19 (median obs / s_0) |

## Tautology Analysis

### `tautology_analysis_dsph.csv` (31 rows)
Per-dSph decomposition including σ, r_h, M_bar regressors.

| Column | Description |
|---|---|
| galaxy, host | Identifier |
| sigma_kms, rh_pc, MV, LV | Observational properties |
| log_sigma, log_rh, log_Mbar, log_Mdyn | Log-space regression inputs |
| g_obs_a0, g_bar_a0, gc_obs_a0, gc_exact_a0 | Derived accelerations |
| gc_tautological_a0 | Definitional-only gc prediction (σ⁴ / GM_bar) |
| physical_resid_dex | log₁₀(gc_obs / gc_tauto) — residual after tautology removal |
| gc_D_pred, D_resid | Candidate D prediction and residual |
| gc_Bern_pred, Bern_resid | Bernoulli prediction (G = 0.228 a₀) and residual |

### `tautology_results.json`
JSON summary of multivariate and univariate regression coefficients.

Schema:
```
{
  "N_dsph": 31,
  "gc_multivariate": {"intercept", "log_sigma", "log_rh", "log_Mbar", "R2", "scatter_dex"},
  "gobs_multivariate": {... R² = 1.0 by definition ...},
  "gobs_univariate": {
    "vs_sigma_slope", "vs_sigma_se", "vs_sigma_R2",
    "vs_rh_slope",    "vs_rh_se",    "vs_rh_R2",
    "vs_Mbar_slope",  "vs_Mbar_se",  "vs_Mbar_R2"
  },
  "strigari_null": {"log_gobs_mean", "log_gobs_std", "g_obs_const_a0", "g_obs_const_SI"},
  "candidate_D_vs_Bernoulli": {"G_fit_a0", "G_Bernoulli_a0", "D_resid_mean", ...}
}
```

Key verdicts derivable from this JSON:
- `g_obs` vs M_bar slope / stderr ≈ 0.2σ (null confirmed) ✓
- `g_obs` vs r_h slope / stderr ≈ 3.4σ (Strigari rejected)

## Bridge Galaxy Verification

### `bridge_profile_summary.csv` (4 rows)
Summary of bridge galaxy radial profiles.

| Column | Description |
|---|---|
| galaxy | ESO444-G084, NGC2915, NGC1705, NGC3741 |
| N_points, R_min_kpc, R_max_kpc | Sample size and range |
| inner_Ups_med, outer_Ups_med | Median Υ_dyn in inner 60%, outer 40% |
| ratio_outer_inner | Ratio (>1 for J3 transition) |
| slope_logU_logR, slope_se, slope_sig_z | Log-log slope of Υ vs R |
| mannwhitney_p_outer_greater | Mann-Whitney test p-value |
| outer_g_obs_med_a0 | Outer g_obs ensemble median (a₀ units) |
| outer_gc_dmond_med_a0 | Outer deep-MOND gc median |

### `bridge_all_points.csv` (72 rows)
All radius points from the four bridge galaxies.

### `bridge_outer_points.csv` (30 rows)
Outer 40% subset used for the two-population Bernoulli verification.

### `bridge_verdict.csv` (4 rows)
A-grade verdict records.

| Column | Description |
|---|---|
| criterion | Test description |
| value | Observed value |
| grade | "A", "B", or "C" |

All four criteria A-grade → continuous C15 → Strigari transition confirmed.

## Constants (for reference)

| Symbol | Value | Description |
|---|---|---|
| a₀ | 1.2 × 10⁻¹⁰ m/s² | MOND acceleration |
| T_m | √6 | Membrane Z2 SSB temperature |
| s₀ | 0.3515 | c→0 spontaneous elastic fraction |
| s₀(1−s₀) | 0.228 | Bernoulli variance prediction |
| G_Newton | 6.674 × 10⁻¹¹ m³/(kg·s²) | Gravitational constant |
| M_sun | 1.989 × 10³⁰ kg | Solar mass |
| kpc | 3.086 × 10¹⁹ m | Kiloparsec |
| pc | 3.086 × 10¹⁶ m | Parsec |
