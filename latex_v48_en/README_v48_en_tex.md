# membrane_v48_en — v4.8 companion paper (English), arXiv submission package

## Overview

English translation of the v4.8 companion paper (original: 日本語, 14 pages) of
the Membrane Cosmology framework. This paper provides the theoretical foundation layer
to the v4.7.8 main body.

- **Target arXiv cat**: `astro-ph.CO` (primary) + `hep-th` (cross-list)
- **Pages**: 13 pages (matches original Japanese page count)
- **References**: 11 entries (DOI-annotated)
- **Language**: English
- **Source**: English translation of `membrane_v48_companion.pdf` (LuaLaTeX, 14 pages, 日本語)

## Files

| file | role |
|---|---|
| `membrane_v48_en.tex` | Main LaTeX source (UTF-8, pdfLaTeX compatible) |
| `refs_v48_en.bib` | BibTeX references (11 entries, DOI-annotated) |
| `membrane_v48_en.bbl` | BibTeX processed output (arXiv-ready) |
| `Makefile` | Compilation automation |
| `README_v48_en_tex.md` | This file |

## Build requirements

- **TeX Live 2023+**
- Required packages: `texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`, `texlive-pictures` (for tcolorbox)
- Compiler: **pdfLaTeX** (LuaLaTeX/XeLaTeX also work)

## Compilation

```bash
make              # pdflatex + bibtex + pdflatex x2
make arxiv        # generate arXiv submission tar.gz
make clean        # remove intermediate files
```

Manual:
```bash
pdflatex membrane_v48_en
bibtex   membrane_v48_en
pdflatex membrane_v48_en
pdflatex membrane_v48_en
```

## arXiv submission

1. `make arxiv` generates `membrane_v48_en_arxiv.tar.gz`
2. Go to https://arxiv.org/submit, start new submission
3. **Primary category**: `astro-ph.CO` (Cosmology and Nongalactic Astrophysics)
4. **Cross-list**: `hep-th` (High Energy Physics - Theory)
5. Upload the tarball (arXiv will re-run pdflatex + bibtex; the embedded `.bbl` guarantees bibliography resolution)
6. **Author**: Shinobu Sakaguchi
7. **Affiliation**: Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan
8. **Comments**: `v4.7.8 companion paper. 13 pages, 11 references. English version of the Japanese original. Companion to arXiv:XXXX.XXXXX (v4.7.8 main body).`

After arXiv ID is assigned for the v4.7.8 main body, update the `Sakaguchi2026a` BibTeX entry in `refs_v48_en.bib`:
```bibtex
@MISC{Sakaguchi2026a,
  ...
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO}
}
```

## Relationship to the Japanese original (v4.8 companion)

| | Japanese original | This English version |
|---|---|---|
| Layer | theoretical foundation | theoretical foundation |
| Pages | 14 (LuaLaTeX) | 13 (pdfLaTeX) |
| References | 15 | 11 (same content, consolidated) |
| BibTeX style | natbib/round/authoryear | natbib/round/authoryear |
| Figures | none | none |
| Target readers | 日本語圏 theoretical physics | international astro-ph + hep-th |
| arXiv classification | astro-ph.CO + hep-th cross | astro-ph.CO + hep-th cross |

Content (all 19 numerical anchors, 11+ closed-form equations, retract-unchangeable items #1–31, Lessons 91–93, M1–M5 claims, 11%/5.5% unification, $\Gamma_A$/$\Gamma_A'$ dual representation) is faithfully translated with no scientific deviation.

## Known issues

- Appendix B cross-reference table uses `Bo\"o II` (LaTeX escape for ö) to avoid UTF-8 issues in pdfLaTeX
- The `tcolorbox` package requires `texlive-pictures`; if compilation fails with `tcolorbox.sty not found`, install via `tlmgr install tcolorbox`

## Author contact

**Shinobu Sakaguchi** (坂口 忍)
Sakaguchi Seimensho, Shisō, Hyogo, Japan (兵庫県宍粟市)
Web: https://sakaguchi-physics.com

## License

MIT License (same as `v48_release`)
