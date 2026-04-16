# Figures Directory

Main figures from the v4.7.8 dSph extension analysis (PDF + PNG pairs).

## Figure List

### `fig_unified_RAR_v2.{pdf,png}`
**Figure 7.3 (arXiv §7.3): Unified RAR plot**

Two-panel figure showing:
- (A) log(g_bar/a₀) vs log(g_obs/a₀) for:
  - SPARC RAR cloud (3,389 points, blue small)
  - Bridge galaxy outer stars (4 galaxies, colored stars)
  - dSph by host (31 galaxies, red/blue/green markers)
  - Theoretical lines: Newton, MOND simple, membrane gc=1.2 a₀ (SPARC), gc=35 a₀ (dSph)
  - J3 transition band guides (Υ_dyn = 3, 10, 30)
- (B) Histogram of log(Υ_dyn) for SPARC cloud, bridge outer, dSph
  - Medians marked; J3 transition band shaded

Key takeaways:
- dSph (median Υ_dyn ~ 35) occupies a distinct regime from SPARC (median ~ 2.75)
- Bridge outer points (median ~ 12.76) form a bridge at the J3 band center
- Theoretical membrane interpolation matches both populations with different gc values

### `fig_theory_dSph_gc_v1.{pdf,png}`
**Figure 7.4-7.5 (arXiv §7.4-7.5): Theoretical foundation**

Four-panel figure:
- (a) U(ε;c) free energy landscape for c = 0, 0.25, 0.5, 0.75, 1.0
  - Equilibrium ε_eq(c) = √(1-c) marked
- (b) ΔU(c) (memo formula) vs U_eq(c), showing ΔU(0) = -3/2 limit
- (c) SCE solution s(Q) vs Q (log-x), numerical vs s₀ + α_s√Q asymptotic expansion
  - s₀ = 0.3515 horizontal line
- (d) gc_dSph candidate formulas vs observed dSph
  - Candidate D (Strigari-type, best fit)
  - Candidates A (s₀·a₀ const) and B ((1-s₀)/s₀ · a₀ const) shown for contrast

### `fig_tautology_analysis_v1.{pdf,png}`
**Figure 7.6 (arXiv §7.6): Tautology separation**

Six-panel regression plots on 31 dSph galaxies:
- Top row (a-c): log(gc_obs) vs log(σ), log(r_h), log(M_bar)
  - Predicted slopes from definitional tautology: +4, 0, -1
  - Observed fits shown
- Bottom row (d-f): log(g_obs) vs same three variables
  - Strigari null (g_obs = const) shown as red dashed line
  - Observed slope ± stderr in legend

Key results:
- (f) g_obs vs M_bar: slope = -0.008 ± 0.049 (0.2σ null confirmed)
- (e) g_obs vs r_h: slope = -0.47 ± 0.14 (3.4σ rejection of pure Strigari)
- (d) g_obs vs σ: slope = +0.75 ± 0.33 (2.3σ marginal, reflects σ-size relation)

### `fig_bridge_profiles_v1.{pdf,png}`
**Figure 7.7 (arXiv §7.7): Bridge galaxy radial profiles**

Four-row (one per bridge galaxy) × three-column plot:
- Column 1: g_obs(R) and g_bar(R) vs R, with Bernoulli line 0.228 a₀
- Column 2: Υ_dyn(R) with J3 transition band (10-30) shaded
  - Inner/outer split (60/40) marked by color
- Column 3: gc_exact(R) and gc_dMOND(R), with SPARC C15 band (0.3-3 a₀)
  and dSph Strigari band (3-30 a₀) shaded

Shows the radial transition from C15 regime (inner, blue) to J3 regime (outer, red)
in all four bridge galaxies.

### `fig_bridge_integration_v1.{pdf,png}`
**Figure 7.8 (arXiv §7.8): Bridge integration with SPARC and dSph**

Two-panel figure:
- (A) Unified RAR:
  - SPARC cloud (blue small background)
  - Bridge outer stars (colored, one color per galaxy)
  - dSph (red filled circles)
  - Bernoulli line g_obs = 0.228 a₀ (red dash-dot)
  - Newton and MOND reference lines
- (B) Υ_dyn histograms:
  - SPARC cloud (blue, 3389 points)
  - Bridge outer (orange, 30 points)
  - dSph (red, 31 galaxies)
  - J3 transition band (10 < Υ_dyn < 30) shaded

Shows that bridge outer points form an intermediate distribution between SPARC and dSph,
and cluster near the Bernoulli prediction line in the RAR plane.

## Figure Generation

Each figure is produced by the corresponding script in `../scripts/`. See
`../scripts/README.md` for the pipeline.

## Viewing Tips

- PNG versions are at 150-180 DPI, suitable for web display and presentations
- PDF versions are vector where possible (scatter plots rasterized for file size)
- All figures use IPA Gothic font for Japanese labels; if not installed,
  matplotlib may substitute a default font (layout preserved but glyphs may appear as boxes)
