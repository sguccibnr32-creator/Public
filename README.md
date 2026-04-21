# Membrane Cosmology — Public Research Archive

膜宇宙論 (Membrane Cosmology) framework の公開 research archive。
dark matter 現象を elastic membrane の折れ畳み構造から説明するアプローチで、
SPARC 175 銀河 + dSph 31 銀河 + HSC-SSP Y3 + GAMA DR4 weak lensing で検証済。

**Author**: 坂口 忍 (Shinobu Sakaguchi) / 坂口製麺所 (Sakaguchi Seimensho), 兵庫県宍粟市 (Shisō, Hyogo, Japan)
**Web**: https://sakaguchi-physics.com
**License**: [MIT](LICENSE)

---

## 🏆 主要成果

| 項目 | 統計 |
|---|---|
| **MOND 棄却** | p = 1.66×10⁻⁵³ (3 独立手法) |
| **C15 final form**: g_c = 0.584·Υ_d⁻⁰·³⁶¹·√(a₀·v_flat²/h_R) | R² = 0.607 (SPARC 175) |
| **HSC weak lensing**: g_c = 2.73±0.11 a₀ | ΔAIC = +472 (22σ vs MOND) |
| **Bernoulli relation**: G_Strigari = s₀(1−s₀)a₀ = 0.228 a₀ | dSph 5%, bridge 4% 一致 |
| **FIRAS μ 上限**: α_PT^upper = 2.96×10⁻⁵³ | V_ξ primary, NGC 3198 2.32% PASS |
| **universal b_α = 0.11** across 3.92 dex | SPARC vs dSph 0.5% 以内一致 |

---

## 📁 ディレクトリ構成

```
Public/
├── README.md                      ← 本ファイル
├── LICENSE                        ← MIT License
├── CHANGELOG.md                   ← version 履歴
│
├── papers/                        ← 論文 PDF 群 (6 files)
│   ├── membrane_arxiv_v478.pdf              — v4.7.8 本体 LaTeX 修正版 (18p, English) ⭐
│   ├── membrane_arxiv_v478_original.pdf     — v4.7.8 ReportLab original (13p, archive)
│   ├── membrane_v48_companion.pdf           — v4.8 companion LuaLaTeX (14p, 日本語) ⭐
│   ├── membrane_v48_body_reportlab.pdf      — v4.8 ReportLab オリジナル (13p, v3 patched)
│   ├── foundation_integrated.pdf            — foundation layer catalog (15p)
│   └── cross_reference_audit_v3.pdf         — QA record (5p)
│
├── latex_v478/                    ← v4.7.8 本体 LaTeX source (pdfLaTeX)
├── latex_v48/                     ← v4.8 companion LaTeX source (LuaLaTeX)
├── arxiv/                         ← arXiv upload-ready tarballs (2 本)
├── reportlab_source/              ← ReportLab PDF build scripts (3 files)
└── docs/                          ← WordPress HTML, layout spec, arXiv guide
```

---

## 📚 論文

### v4.7.8 本体 (main body)
**Title**: *Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology: The Geometric Mean Law and Observational Tests*
**Language**: English, 18 pages, 38 references
**arXiv**: `arXiv:XXXX.XXXXX [astro-ph.CO]` (forthcoming) <!-- 採番後に記入 -->
**PDF**: [`papers/membrane_arxiv_v478.pdf`](papers/membrane_arxiv_v478.pdf)

### v4.8 companion paper
**Title**: 膜宇宙論 foundation layer: FIRAS μ 歪み上限と universal density coupling の閉形式導出 (v4.7.8 companion paper, v4.8)
**Language**: 日本語, 14 pages, 15 references
**arXiv**: `arXiv:YYYY.YYYYY [astro-ph.CO + hep-th]` (forthcoming) <!-- 採番後に記入 -->
**PDF**: [`papers/membrane_v48_companion.pdf`](papers/membrane_v48_companion.pdf)

---

## 🚀 Release 一覧

| Release | Date | 内容 | Download |
|---|---|---|---|
| **v4.8** ⭐ latest | 2026-04-21 | v4.7.8 affiliation rev. + v4.8 companion paper | [v48_release.zip](https://github.com/sguccibnr32-creator/Public/releases/download/v4.8/v48_release.zip) |
| v4.7.8 | 2026-04 | observational establishment 完成版、19 結論 | [Releases page](https://github.com/sguccibnr32-creator/Public/releases) |
| v4.7.6 | 2026-04 | self-consistent equation η₀ origin, T_m confirmation | [Releases page](https://github.com/sguccibnr32-creator/Public/releases) |

全 release: [Releases tab](https://github.com/sguccibnr32-creator/Public/releases)

---

## 🔬 再現手順

### LaTeX ビルド

v4.7.8 本体 (pdfLaTeX):
```bash
cd latex_v478/
make
```

v4.8 companion (LuaLaTeX + luatexja):
```bash
cd latex_v48/
make
```

詳細: 各 `latex_*/README*.md` 参照。

### ReportLab PDF 再生成
```bash
cd reportlab_source/
uv run --with reportlab --with pypdf python build_membrane_v48_body.py
uv run --with reportlab --with pypdf python build_foundation_integrated.py
uv run --with reportlab --with pypdf python build_cross_reference_audit.py
```

必要: Python 3.10+、reportlab、pypdf、IPAGothic / IPAPGothic フォント (Linux: `fonts-ipafont`、Windows: `C:\Windows\Fonts\msgothic.ttc`)

---

## 📖 理論の骨子

### Lagrangian
```
U(ε; c) = -ε - ε²/2 - c·ln(1 - ε),   0 < ε < 1
```
equilibrium: ε₀ = √(1-c)、w(c)² = 2ε₀/(1-ε₀)

### Effective acceleration
```
g_obs = (g_N + √(g_N² + 4 g_c g_N)) / 2
```
MOND の simple interpolation を first-principles から導出。

### Geometric mean law
```
g_c = η·√(a₀·G·Σ₀)
```
SPARC 175 galaxies で α=0.5 を dAIC=-130 vs MOND で確立。

### C15 final form
```
g_c = 0.584 · Υ_d⁻⁰·³⁶¹ · √(a₀·v_flat²/h_R)
```
η₀=0.584 は ⟨1-f_p⟩_SPARC (自己整合方程式から解釈)。

### Bernoulli relation (dSph extension)
```
G_Strigari = s₀·(1-s₀)·a₀ = 0.228·a₀,
s₀ = 1/(1 + exp(3/(2T_m))),  T_m = √6
```
dSph 31 銀河 (0.240 a₀, 5%) と SPARC bridge 外側 (0.219 a₀, 4%) で独立検証。

---

## 🎓 引用方法

```bibtex
@misc{Sakaguchi2026a,
  author = {Sakaguchi, Shinobu},
  title  = {Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology:
            The Geometric Mean Law and Observational Tests},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  note   = {v4.7.8 (2026-04-21 affiliation rev.)}
}

@misc{Sakaguchi2026b,
  author = {Sakaguchi, Shinobu (坂口 忍)},
  title  = {膜宇宙論 foundation layer: FIRAS μ 歪み上限と universal density coupling の閉形式導出 --- v4.7.8 companion paper (v4.8)},
  year   = {2026},
  eprint = {YYYY.YYYYY},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  note   = {Companion to arXiv:XXXX.XXXXX}
}
```

---

## 🤝 Contribution

本 repo は坂口 忍による個人研究 archive です。issue / discussion で質問・コメントを歓迎します。
Pull request による本文修正は基本的に受け付けませんが、typo や BibTeX metadata の修正は歓迎します。

---

## 📞 連絡先

- **Web**: https://sakaguchi-physics.com
- **所在**: 坂口製麺所、兵庫県宍粟市 (Shisō City, Hyogo, Japan)
- **GitHub Issues**: [Issues tab](https://github.com/sguccibnr32-creator/Public/issues)

---

## ⚖ License

[MIT License](LICENSE) — 自由に複製・改変・再配布可。

Copyright (c) 2026 Shinobu Sakaguchi / Sakaguchi Seimensho
