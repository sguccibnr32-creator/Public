# Scripts Directory

This directory contains the v4.7.8 analysis pipeline, organized in execution order.

## Pipeline Overview

```
Step 1: preprocess_unified_rar_v1.py
  IN:  TA3_gc_independent.csv, dsph_jeans_c15_v1.csv, Rotmod_LTG/
  OUT: rar_unified_galaxy_level.csv (206 rows = SPARC 175 + dSph 31)
       rar_sparc_cloud.csv (3389 points from 175 galaxies)

Step 2: analyze_unified_rar_v1.py
  IN:  rar_unified_galaxy_level.csv, rar_sparc_cloud.csv
  OUT: fig_unified_RAR_v1.{pdf,png}
       j3_threshold_summary.csv
       sparc_j3_candidates_ranked.csv
       sparc_outer_upsilon.csv
       table_j3_transition_v1.pdf

Step 3: theory_dsph_gc_v1.py
  IN:  rar_unified_galaxy_level.csv
  OUT: fig_theory_dSph_gc_v1.{pdf,png}
       dsph_gc_candidates_v1.csv
       gc_candidates_residuals.csv
       theory_key_values.csv

Step 4: tautology_analysis_v1.py
  IN:  rar_unified_galaxy_level.csv
  OUT: fig_tautology_analysis_v1.{pdf,png}
       tautology_analysis_dsph.csv
       tautology_results.json

Step 5: bridge_galaxy_analysis_v1.py
  IN:  rar_sparc_cloud.csv, rar_unified_galaxy_level.csv
  OUT: fig_bridge_profiles_v1.{pdf,png}
       fig_bridge_integration_v1.{pdf,png}
       bridge_profile_summary.csv
       bridge_all_points.csv
       bridge_outer_points.csv
       bridge_verdict.csv
```

## PDF Generation Scripts

The following scripts produce standalone PDF documents; they consume the data
output by steps 1-5 above.

```
generate_section_6_10_v2.py
  → section_6_10_dSph_extension_v2.pdf (JP, ~500 kB, 4 pp)

generate_section_9_1.py
  → section_9_1_dSph_gc_reformulation_v1.pdf (JP, ~540 kB, 5 pp)

generate_section_0.py
  → section_0_tautology_separation_v1.pdf (JP, ~500 kB, 4 pp)

generate_section_0_ii.py
  → section_0_ii_bridge_verification_v1.pdf (JP, ~920 kB, 5 pp)

membrane_arxiv_v478.py
  → membrane_arxiv_v478.pdf (EN, ~52 kB, 12 pp — arXiv submission draft)
```

## Dependencies

```bash
uv pip install numpy pandas scipy matplotlib astropy reportlab astroquery
```

For PDF generation with Japanese text (sections 6.10, 9.1, 0, 0.ii):
```bash
# Ubuntu/Debian
sudo apt-get install fonts-ipafont-gothic

# Alternative: any TrueType fonts with JIS X 0208 coverage will work;
# edit the font_path variables at the top of each generate_section_*.py script.
```

## Notes on ASCII Compatibility

All Python source uses UTF-8 but is written to be ASCII-safe in variable names
and comments. Scientific notation follows v4.2 spec:
- `√x` → `sqrt(x)`
- `≪` → `<<`
- `≥` → `>=`
- `∝` → `proportional to`
- `→` → `->`
- Greek letters (σ, Υ, ε, ...) are preserved in strings where meaningful.

This avoids encoding issues in IPAGothic (which lacks many math glyphs) while
keeping scientific readability.

## Reproducibility

Running steps 1-5 in order produces the v4.7.8 analysis end-to-end from raw
SPARC + McConnachie data. All random seeds are fixed or not used (pure
statistical tests). Expected runtime on a modern laptop: ~3 minutes total.

The output data files under `../data/` and figures under `../figures/`
should match bitwise (within matplotlib rendering jitter) those included in
this release.
