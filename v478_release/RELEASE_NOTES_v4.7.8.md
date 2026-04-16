# Release v4.7.8 — Dwarf Spheroidal Extension

**Release date:** April 16, 2026
**Tag:** `v4.7.8`
**Previous release:** [v4.7.6](https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.6)
**License:** MIT

---

## TL;DR

v4.7.8 extends the membrane cosmology framework to **pressure-supported dwarf spheroidal (dSph) galaxies**. The Bernoulli variance of the c→0 limit of `U(ε; c)` predicts `G_Strigari = s_0(1 − s_0) · a_0 = 0.228 a_0` with no free parameters. This prediction is verified in **two independent observational populations**:

- 31 dSph galaxies: `g_obs` = 0.240 a_0 (5% agreement)
- 30 SPARC bridge galaxy outer points: `g_obs` = 0.219 a_0 (4% agreement)

Additionally, four SPARC bridge galaxies (ESO444-G084, NGC2915, NGC1705, NGC3741) display a **continuous C15 → Strigari regime transition within individual galaxies** (4/4 at A-grade).

---

## Highlights

### 1. New Physical Regime Identified: J3 Inversion

In 28 of 31 dSph galaxies, the membrane Jeans mass `M_J,mem << M_J,std`, inverting the causal direction of membrane-baryon coupling relative to SPARC spirals. In this regime, gravity is determined by the intrinsic thermal state of the membrane (parameters `T_m`, `a_0`) rather than by baryonic surface density. C15 cannot be applied directly; the offset is +1.5 dex.

### 2. c→0 Limit Predicts Strigari Relation from First Principles

The self-consistent equation `s = 1/(1 + exp(−ΔU(sQ)/T_m))` in the Q→0 limit yields a universal spontaneous elastic fraction:
- `s_0 = 1/(1 + exp(3/(2·T_m))) = 0.3515` at `T_m = √6`
- Bernoulli variance: `s_0(1 − s_0) = 0.228`
- **Prediction:** `G_Strigari = 0.228 × a_0 = 2.74 × 10⁻¹¹ m/s²`

Non-analytic √Q correction `s(Q) ~ s_0 + 0.110 √Q` arises from the `ln(1 − ε)` term in `U(ε; c)`.

### 3. Two-Population Independent Verification

| Population | N | g_obs [a_0] | Bernoulli ratio | Agreement |
|---|---|---|---|---|
| Bernoulli prediction | - | 0.228 | 1.000 | - |
| dSph sample | 31 | 0.240 | 1.053 | 5% |
| Bridge outer points | 30 | 0.219 | 0.961 | 4% |
| NGC2915 outer alone | 12 | 0.219 | 0.961 | 4% |

The two populations differ by 10× in mass scale and by dynamical support type (pressure vs rotation). Common agreement with a single theoretical constant (no fit parameters) constitutes a non-trivial confirmation.

### 4. Continuous Regime Transition in Bridge Galaxies

Four SPARC bridge galaxies with outer Υ_dyn > 10 show radius-resolved evidence of C15 → Strigari transition:

| Galaxy | N | slope(logΥ, logR) | z | MW test p | outer g_obs [a_0] |
|---|---|---|---|---|---|
| ESO444-G084 | 7 | +0.31 ± 0.13 | +2.4 | 0.029 | 0.317 |
| NGC2915 | 30 | +1.13 ± 0.09 | **+12.3** | 0.0003 | **0.219** |
| NGC1705 | 14 | +0.46 ± 0.07 | +6.8 | 0.0003 | 0.282 |
| NGC3741 | 21 | +0.55 ± 0.03 | **+21.6** | 0.0001 | 0.124 |

All four A-grade criteria satisfied:
- (a) Radial Υ_dyn increase significant in 4/4 galaxies
- (b) outer > inner Mann-Whitney test passes in 4/4
- (c) Bridge ensemble median Υ_dyn = 12.76 (center of J3 band 10-30)
- (d) Bridge g_obs agrees with Bernoulli prediction to 4%

### 5. Tautology Separation Completed

Multivariate regression of `log g_obs` on (σ, r_h, M_bar) in the 31-dSph sample:
- **g_obs vs M_bar**: slope = −0.008 ± 0.049 (0.2σ **null**) — the genuine non-tautological membrane prediction
- **g_obs vs r_h**: slope = −0.47 ± 0.14 (−3.4σ significant) — reflects the empirical dSph σ-size relation σ ∝ r_h^0.27 (McConnachie 2012), not predicted by mean-field theory

The earlier H4 hypothesis `gc ∝ M_dyn^−1` is shown to decompose as `gc ∝ σ^0 × r_h^(−1.5)`, with σ contribution null; the real driver is `r_h^(−3/2)`.

---

## What's Included

This release contains **42 files** organized into:

| Directory | Contents | Count |
|---|---|---|
| `scripts/` | Python analysis pipeline (reproducibility) | 10 |
| `data/` | Analysis output (CSV + JSON) | 14 |
| `figures/` | Main figures (PDF + PNG pairs) | 10 |
| `pdf/` | Main documents (arXiv draft, sections) | 6 |
| `docs/` | WordPress HTML, handoff memo | 2 |

Plus: README.md, CHANGELOG.md, LICENSE, RELEASE_NOTES_v4.7.8.md (this file).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sguccibnr32-creator/Public.git
cd Public

# Or download and extract the v4.7.8 release archive
wget https://github.com/sguccibnr32-creator/Public/releases/download/v4.7.8/v478_release.zip
unzip v478_release.zip
```

Dependencies:
```bash
uv pip install numpy pandas scipy matplotlib astropy reportlab astroquery
sudo apt-get install fonts-ipafont-gothic  # for PDF output
```

See `README.md` Section "Reproducing the v4.7.8 Analysis" for the full pipeline.

---

## Data Requirements

To reproduce from raw data, the following are required (not redistributed):
- `TA3_gc_independent.csv` — from v4.7.6 release
- `Rotmod_LTG/` — SPARC database
- McConnachie 2012 VizieR J/AJ/144/4 catalogue
- σ_los literature values (see script headers)

Processed intermediate files (`rar_unified_galaxy_level.csv`, `rar_sparc_cloud.csv`) are included in `data/`, allowing the v4.7.8-specific analysis (tautology, bridge verification, theory) to be reproduced directly.

---

## Grade Changes from v4.7.6

| Item | v4.7.6 | v4.7.8 | Basis |
|---|---|---|---|
| J3 regime inversion | — | **A** (new) | 28/31 dSph, theoretical explanation |
| Bernoulli G = 0.228 a_0 | — | **A** (new) | 2-population independent reproduction |
| C15 → Strigari continuous transition | — | **A** (new) | 4/4 bridge galaxies, 4/4 criteria |
| g_obs M_bar-independence | — | **A** (new) | 0.2σ null |
| Strict Strigari universality | implicit | **partially retracted** | -3.4σ r_h dependence |
| Individual dSph gc formula | — | B | 0.7 dex residual scatter |
| H4 `gc ∝ M_dyn^−1` | C (hypothesis) | reinterpreted as `gc ∝ r_h^(−1.5)` | σ-contribution null |

---

## Known Issues / Limitations

1. **dSph σ-size relation origin unclear**: The r_h dependence of g_obs reflects σ ∝ r_h^0.27 (empirical), not derived from the mean-field c→0 analysis. Either higher cumulants beyond the Bernoulli variance, or mean-field-breaking dynamics (tidal history, formation epoch), may be required.

2. **Individual-galaxy dSph gc formula has 0.7 dex residual scatter**: `gc_dSph = [s_0(1-s_0)]² × a_0² / g_bar` works as an ensemble mean but not for individual predictions.

3. **Ultra-faint dSph (Υ_dyn > 100) are at sample extrapolation limit**: 7/31 dSph exceed Υ_dyn = 100, a regime where Bernoulli prediction may be modified. Independent verification with a larger ultra-faint sample is recommended.

4. **η_0 = 0.584 theoretical origin remains open** (from v4.7.6, unchanged).

5. **BIG-SPARC external verification pending**: ~4,000 galaxies with baryon-separated rotation curves (in preparation) will be the definitive external test.

---

## Citing

```bibtex
@misc{sakaguchi2026v478,
  author = {Sakaguchi, Shinobu},
  title = {Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology:
           The Geometric Mean Law, Strigari Relation, and Continuous Regime Transition},
  year = {2026},
  version = {v4.7.8},
  url = {https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.8},
  note = {MIT License}
}
```

---

## Acknowledgments

Data sources: SPARC (Lelli, McGaugh, Schombert 2016), McConnachie (2012), Walker et al. (2009), Wolf et al. (2010), Collins et al. (2013), Tollerud et al. (2012), Simon & Geha (2007), Strigari et al. (2008), Brasseur et al. (2011), HSC-SSP, GAMA DR4.

---

## Contact

Shinobu Sakaguchi, Sakaguchi Seimensho (Independent Research)
Website: https://sakaguchi-physics.com
GitHub: https://github.com/sguccibnr32-creator/Public

---

## License

MIT License. See `LICENSE`.
