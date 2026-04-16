# Membrane Cosmology — v4.7.8 Release

**Pressure-supported dwarf spheroidal extension with Bernoulli prediction and continuous regime transition**

Author: Shinobu Sakaguchi (Sakaguchi Seimensho, Independent Research)
Date: April 16, 2026
License: MIT
Previous release: [v4.7.6](https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.6)

---

## What's New in v4.7.8

Version 4.7.8 extends the membrane cosmology framework to **pressure-supported dwarf spheroidal (dSph) galaxies**, adding four A-grade results:

1. **J3 regime inversion** — In 28 of 31 dSph galaxies, the membrane Jeans mass `M_J,mem << M_J,std`, reversing the direction of membrane-baryon causal coupling. C15 cannot be applied directly; the offset is +1.5 dex.

2. **Bernoulli prediction G_Strigari = s_0(1 − s_0) a_0 = 0.228 a_0** — The `c → 0` limit of `U(ε; c)` yields a universal spontaneous elastic fraction `s_0 = 1/(1 + exp(3/(2·T_m))) = 0.3515` for `T_m = sqrt(6)`. The Bernoulli variance `s_0(1 − s_0)` identifies the Strigari (2008) universal acceleration from first principles.

3. **Two-population independent verification** — `G_Strigari = 0.228 a_0` is reproduced in two statistically independent populations to 4–5%:
   - dSph sample (N = 31, 0.240 a_0, 5% agreement)
   - SPARC bridge galaxy outer points (N = 30, 0.219 a_0, 4% agreement)

4. **Continuous C15 → Strigari transition** — Four SPARC bridge galaxies (ESO444-G084, NGC2915, NGC1705, NGC3741) exhibit statistically significant radial increase of `Υ_dyn(R)` (|z| = 2.4 to 21.6, 4/4 at p < 0.03), confirming a continuous regime transition within individual galaxies.

Secondary results:
- `g_obs` `M_bar`-independence verified at 0.2 sigma null (the genuine non-tautological prediction)
- Strict Strigari universality (`g_obs = const`) partially retracted: the `r_h` dependence (-3.4 sigma) reflects the empirical dSph σ-size scaling
- The H4 hypothesis `g_c ∝ M_dyn^(-1)` is reinterpreted as `g_c ∝ r_h^(-1.5)` with zero sigma contribution

See [CHANGELOG.md](CHANGELOG.md) for the complete change log from v4.7.6 to v4.7.8.

---

## Repository Structure

```
v478_release/
├── README.md                 This file
├── CHANGELOG.md              Changes from v4.7.6 to v4.7.8
├── RELEASE_NOTES_v4.7.8.md   Detailed release notes for v4.7.8
├── LICENSE                   MIT License
│
├── pdf/                      Main documents (6 PDFs)
│   ├── membrane_arxiv_v478.pdf            (12 pp, arXiv submission draft)
│   ├── section_6_10_dSph_extension_v2.pdf (§6.10 internal update, JP)
│   ├── section_9_1_dSph_gc_reformulation_v1.pdf (c→0 theory, JP)
│   ├── section_0_tautology_separation_v1.pdf    (tautology analysis, JP)
│   ├── section_0_ii_bridge_verification_v1.pdf  (bridge galaxy test, JP)
│   └── table_j3_transition_v1.pdf         (J3 transition tables)
│
├── scripts/                  Python analysis scripts (10 files)
│   ├── preprocess_unified_rar_v1.py       Step 1: data preprocessing
│   ├── analyze_unified_rar_v1.py          Step 2: unified RAR plot
│   ├── theory_dsph_gc_v1.py               Step 3: c→0 theory
│   ├── tautology_analysis_v1.py           Step 4: regression analysis
│   ├── bridge_galaxy_analysis_v1.py       Step 5: bridge verification
│   ├── generate_section_6_10_v2.py        §6.10 PDF generation
│   ├── generate_section_9_1.py            §9.1 PDF generation
│   ├── generate_section_0.py              §0 PDF generation
│   ├── generate_section_0_ii.py           §0.ii PDF generation
│   └── membrane_arxiv_v478.py             arXiv paper build script
│
├── data/                     Analysis output CSV/JSON (14 files)
│   ├── rar_unified_galaxy_level.csv       SPARC 175 + dSph 31 (primary)
│   ├── rar_sparc_cloud.csv                SPARC 3,389 radius points
│   ├── sparc_j3_candidates_ranked.csv     SPARC galaxies sorted by outer Upsilon_dyn
│   ├── sparc_outer_upsilon.csv            Outer Upsilon_dyn for all 175 SPARC
│   ├── j3_threshold_summary.csv           J3 threshold statistics table
│   ├── dsph_gc_candidates_v1.csv          Candidate formula residuals
│   ├── gc_candidates_residuals.csv        Summary of candidate evaluation
│   ├── theory_key_values.csv              Theoretical constants (s_0, alpha_s, ...)
│   ├── tautology_analysis_dsph.csv        Per-galaxy tautology analysis
│   ├── tautology_results.json             Regression coefficients and z-scores
│   ├── bridge_profile_summary.csv         Bridge galaxy radial profile summary
│   ├── bridge_all_points.csv              Bridge galaxies all radius points
│   ├── bridge_outer_points.csv            Bridge galaxies outer 40% points
│   └── bridge_verdict.csv                 4/4 A-grade verdict records
│
├── figures/                  Main figures (5 PDF + 5 PNG pairs)
│   ├── fig_unified_RAR_v2.{pdf,png}              Unified RAR (SPARC + dSph)
│   ├── fig_theory_dSph_gc_v1.{pdf,png}           U(ε;c), SCE, Bernoulli comparison
│   ├── fig_tautology_analysis_v1.{pdf,png}       6-panel regression analysis
│   ├── fig_bridge_profiles_v1.{pdf,png}          4-galaxy radial profiles
│   └── fig_bridge_integration_v1.{pdf,png}       RAR + Upsilon_dyn distributions
│
└── docs/                     Supplementary documents
    ├── handoff_memo_20260416_dsph_final.txt      Session handoff memo (JP, complete record)
    └── wordpress_v478_main_page.html             WordPress main page HTML (JP)
```

---

## Required Datasets (not in this repo)

The following observational datasets are required to reproduce the analysis but are not redistributed here (license restrictions):

| Dataset | Source | Usage |
|---|---|---|
| SPARC 175 galaxies | Lelli, McGaugh, Schombert 2016, [AJ 152, 157](https://doi.org/10.3847/0004-6256/152/6/157) | Section 7.3 SPARC cloud generation |
| SPARC Rotmod_LTG | [SPARC database](http://astroweb.cwru.edu/SPARC/) | Section 7.7 bridge galaxies |
| TA3_gc_independent.csv | v4.7.6 (this repository, previous release) | Section 7.3 galaxy-level scalars |
| McConnachie 2012 dSph | [AJ 144, 4](https://doi.org/10.1088/0004-6256/144/1/4) (VizieR `J/AJ/144/4`) | Section 7.1 dSph catalogue |
| sigma_los measurements | See `scripts/bridge_galaxy_analysis_v1.py` header for full citation list | dSph sigma calibration |

The file `TA3_gc_independent.csv` (SPARC-derived galaxy-level scalars from v4.7.6) is available from the previous release.

---

## Reproducing the v4.7.8 Analysis

### Prerequisites
```
python >= 3.10
numpy, pandas, scipy, matplotlib, astropy
reportlab (for PDF generation)
astroquery (for McConnachie VizieR fetch)
```

Install via:
```bash
uv pip install numpy pandas scipy matplotlib astropy reportlab astroquery
```

For IPA Gothic font support (PDF output):
```bash
# Ubuntu/Debian
sudo apt-get install fonts-ipafont-gothic
```

### Pipeline

Execute in order:

```bash
cd scripts/

# Step 1: Preprocess SPARC + dSph data (requires TA3_gc_independent.csv, dsph_jeans_c15_v1.csv, Rotmod_LTG/)
python preprocess_unified_rar_v1.py
#  -> rar_unified_galaxy_level.csv, rar_sparc_cloud.csv

# Step 2: Unified RAR plot and J3 threshold statistics
python analyze_unified_rar_v1.py
#  -> fig_unified_RAR_v2.{pdf,png}, j3_threshold_summary.csv, sparc_j3_candidates_ranked.csv

# Step 3: Theory analysis (c->0 limit, Bernoulli prediction)
python theory_dsph_gc_v1.py
#  -> fig_theory_dSph_gc_v1.{pdf,png}, dsph_gc_candidates_v1.csv, theory_key_values.csv

# Step 4: Tautology separation
python tautology_analysis_v1.py
#  -> fig_tautology_analysis_v1.{pdf,png}, tautology_results.json

# Step 5: Bridge galaxy verification
python bridge_galaxy_analysis_v1.py
#  -> fig_bridge_profiles_v1.{pdf,png}, fig_bridge_integration_v1.{pdf,png}
#  -> bridge_profile_summary.csv, bridge_verdict.csv

# Step 6: Generate section PDFs (JP)
python generate_section_6_10_v2.py     # Internal Section 6.10 update
python generate_section_9_1.py         # c->0 theory reformulation
python generate_section_0.py           # Tautology separation section
python generate_section_0_ii.py        # Bridge galaxy verification section

# Step 7: Build arXiv paper (EN, 12 pages)
python membrane_arxiv_v478.py
```

Total runtime: ~3 minutes on a modern laptop.

---

## Verdict Summary (A-grade criteria)

From `data/bridge_verdict.csv`:

| Criterion | Observed | Verdict |
|---|---|---|
| (a) Radial increase of Upsilon_dyn (slope > 0, significant) | 4/4 galaxies, z = 2.4 to 21.6 | **A** |
| (b) outer > inner (Mann-Whitney test) | 4/4 galaxies, p = 0.0001 to 0.029 | **A** |
| (c) Outer ensemble median Upsilon_dyn in transition band (5-20) | 12.76 (center of J3 band) | **A** |
| (d) Outer g_obs / Bernoulli prediction agreement | 0.96 (4% match) | **A** |

**All four criteria satisfied: continuous C15 -> Strigari transition confirmed at A-grade.**

---

## Key Predictions (testable)

### Non-tautological (A-grade):
1. `g_obs` is independent of M_bar in dSph (verified: 0.2 sigma null)
2. The Strigari universal acceleration equals `s_0(1 - s_0) · a_0 = 0.228 a_0`
3. Bridge galaxies (outer Upsilon_dyn > 10) exist and are rare (4 out of 175 SPARC)
4. `Υ_dyn(R)` increases monotonically in bridge galaxies

### Open issues (B-grade or lower):
1. Origin of the dSph σ ∝ r_h^0.27 relation (not predicted by mean-field)
2. Individual-galaxy gc formula `gc_dSph = [s_0(1-s_0)]² · a_0² / g_bar` has 0.7 dex residual scatter
3. Theoretical origin of `η_0 = 0.584` in C15
4. Ultra-faint dSph (Υ_dyn > 100) extrapolation

---

## Citing This Work

If you use this work, please cite:
```
Sakaguchi, S. (2026). "Galactic Rotation Without Dark Matter from Elastic
Membrane Cosmology: The Geometric Mean Law and Observational Tests", v4.7.8.
Available: https://github.com/sguccibnr32-creator/Public
          https://sakaguchi-physics.com
```

---

## Related Resources

- WordPress site: https://sakaguchi-physics.com
- Previous release: [v4.7.6](https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.6)
- arXiv paper (draft): `pdf/membrane_arxiv_v478.pdf`

---

## License

MIT License. See `LICENSE` file.

All analysis scripts, generated data, and figures are freely available for academic and commercial use. Observational datasets (SPARC, McConnachie 2012, etc.) retain their original licenses.
