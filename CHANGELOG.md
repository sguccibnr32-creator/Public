# Changelog

本 release (`v48_release`) は膜宇宙論 v4.7.8 本体(affiliation 修正版) + v4.8 companion paper の統合成果物パッケージ。

## v4.8 (2026-04-21) — Current release

**arXiv 投稿 READY (v4.7.8 本体 + v4.8 companion 日本語 + v4.8 companion 英語 の 3 論文併行投稿体制、affiliation 修正 + 英語版統合版)**

### Added — v4.8 companion 英語版 (本 release の追加更新)
- **`papers/membrane_v48_en_companion.pdf`** — v4.8 companion 英語版 PDF (pdfLaTeX、13 頁、international readers 向け)
- **`latex_v48_en/`** — v4.8 companion 英語版 LaTeX source (5 files、pdfLaTeX 互換)
  - `membrane_v48_en.tex` (本文 942 行)
  - `refs_v48_en.bib` (11 件 BibTeX、DOI 付)
  - `membrane_v48_en.bbl` (BibTeX 処理済)
  - `Makefile` (pdflatex x3 + bibtex)
  - `README_v48_en_tex.md` (ビルド手順)
- **`arxiv/membrane_v48_en_arxiv.tar.gz`** — v4.8 companion 英語版 arXiv ready-to-upload package (22 KB)
- `docs/arxiv_submission_guide.md` — 英語版 Step 2a / 日本語版 Step 2b の分離投稿手順を追加
- `docs/wordpress_page_v48.html` — 英語版 PDF ダウンロードボタンを追加

### 英語版の性質
- 日本語原本 (`membrane_v48_companion.pdf`, 14p LuaLaTeX) の完全英訳
- 全数値 anchor、全閉形式、全 retract 不可 #1–31、全 M1–M5 claims を保持
- 頁数: 13p(pdfLaTeX typography で日本語版より 1 頁少なく収束)
- overfull hbox 0 件、undefined reference 0 件(tabularx で roadmap / Γ route table の折り返し対応)
- 固有名詞: `Shinobu Sakaguchi`、`Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan`

### arXiv 投稿戦略の拡張
- **推奨 (新)**: v4.8 companion は**英語版を primary submission**、日本語版は sakaguchi-physics.com + GitHub で公開
- **並行投稿も可**: 英語版 (Step 2a) + 日本語版 (Step 2b) を独立 arXiv submission として投稿、相互に cross-reference

### Changed — v4.7.8 affiliation 修正 (本 release の主要更新)
- **Before** (2026-04 original ReportLab PDF, `papers/membrane_arxiv_v478_original.pdf` として archive):
  - Affiliation: `Sakaguchi Seimensho (Independent Research), Kobe, Japan`
  - Version tag: `April 2026 (v4.7.8, rev. LV independence)`
- **After** (2026-04-21 LaTeX 修正版, `papers/membrane_arxiv_v478.pdf`):
  - Affiliation: `Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan` (正: 兵庫県宍粟市)
  - Version tag: `April 2026 (v4.7.8, 2026-04-21 affiliation rev.)`
- 本文内容は完全一致(19 結論、全数値、全 table、全 38 references)
- LaTeX 再構築の副次効果: 18 頁(ReportLab 版 13 頁から +5、pdfLaTeX の article class デフォルト typography 差)

### Added (v4.7.8 LaTeX 修正版 + v4.8 companion paper)
- `papers/membrane_arxiv_v478.pdf` — v4.7.8 LaTeX 修正版 (18 頁、affiliation 修正済)
- `papers/membrane_arxiv_v478_original.pdf` — v4.7.8 ReportLab original (13 頁、Kobe 表記、archive 保存)
- `papers/membrane_v48_companion.pdf` — v4.8 companion paper (LuaLaTeX 版、14 頁、日本語)
- `papers/membrane_v48_body_reportlab.pdf` — v4.8 companion paper (ReportLab 版、13 頁、v3 patched)
- `papers/foundation_integrated.pdf` — foundation layer catalog (15 頁、reference)
- `papers/cross_reference_audit_v3.pdf` — 最終 QA record (5 頁)
- `latex_v478/` — v4.7.8 本体 LaTeX source (pdfLaTeX 互換、affiliation 修正版)
- `latex_v48/` — v4.8 companion LaTeX source (LuaLaTeX 互換)
- `arxiv/membrane_v478_arxiv.tar.gz` — v4.7.8 本体 arXiv ready-to-upload package (32 KB)
- `arxiv/membrane_v48_arxiv.tar.gz` — v4.8 companion arXiv ready-to-upload package (24 KB)
- `reportlab_source/` — ReportLab PDF build scripts (再現性保証)
- `docs/layout_spec_v4_3.txt` — PDF レイアウト仕様書
- `docs/wordpress_page_v48.html` — sakaguchi-physics.com 公開ページ HTML

### v4.7.8 の主な成果 (19 結論の抜粋)
- Condition 15 最終形: **g_c = 0.584 · Υ_d⁻⁰·³⁶¹ · √(a₀·v_flat²/h_R)**、SPARC 175 銀河で R²=0.607、ΔAIC=-14.2
- MOND を **p = 1.66×10⁻⁵³** で棄却 (C15 vs MOND 独立 3 手法)
- HSC-SSP Y3 + GAMA DR4 3 field 独立検証: **g_c = 2.73±0.11 a₀**、ΔAIC=+472 (22σ)
- Bernoulli 関係 **G_Strigari = s₀(1-s₀)a₀ = 0.228 a₀** を dSph 31 銀河 (0.240 a₀, 5% 一致) と SPARC bridge 外側 30 点 (0.219 a₀, 4% 一致) で独立検証
- **T_m = √6** (Z₂ SSB 臨界温度、補題 5) を scatter optimisation で独立検証 (T_m,opt=2.35 が √6 から 0.24σ)
- κ = 0 (膜剛性ゼロ)、Lagrangian L ≈ U(ε; c)、4 自由パラメータ削減

### v4.8 の主な理論的成果 (Abstract の骨子)
- **M1**: FIRAS μ 歪み + Chluba 2016 W_μ kernel + V_ξ primary から **α_PT^upper = 2.96×10⁻⁵³**
- **M2**: τ_m bound [1.9×10¹⁰, 1.4×10⁵³] s → Local Void (R=22 Mpc) で **5 桁幅に narrow 化** (38 桁 tighten)
- **M3**: N_mode SI canonical 定義 (Eq. 2B-II) が NGC 3198 を **2.32% 精度で再現** (A 級)
- **M4**: χ_E [s²] と χ_F [m³/²/s²] は独立 (universal conversion 不存在)
- **M5**: Pantheon+ の f=0.379 は μ kernel 非寄与 (explicit null)
- **universal b_α**: SPARC 124 銀河 +0.1084 vs dSph 30 銀河 +0.1127、**3.92 dex で 0.5% 以内一致**

### v4.8 = v3 patched 版 (v4.8 ReportLab 編集履歴)
- D1 patch: Finding 1 (A1 operational) / Finding 2 (V_ξ_void primary) を両 PDF に追加
- U3 patch: retract 不可 #17 (11%/5.5% 処理統一) を明示反映 — Γ_A = 8.091 (amplitude) / Γ_A' = 17.18 (power) を併記

## v4.7.8 (2026-04 original、deprecated, archived in this release)

**v4.7.8 本体 arXiv 原稿** 初版。本 release の `papers/membrane_arxiv_v478_original.pdf` に archive 保存。
`Kobe, Japan` 表記により arXiv 投稿には不適、2026-04-21 LaTeX 修正版 `papers/membrane_arxiv_v478.pdf` を使用。

## 以前の版 (継承のみ)

- v4.7.7 — non-SPARC LITTLE THINGS 外部検証、partial correlation test
- v4.7.6 — KiDS/HSC lensing RAR、self-consistent equation η₀ origin、T_m confirmation
- v4.7.4 — HSC-SSP Y3 + GAMA DR4 独立 lensing pipeline
- v4.7 — κ = 0、C14 retract、C15 final form
- v4.6 以前 — internal archive

詳細は `papers/membrane_arxiv_v478.pdf` Section 8 (Conclusions) の 19 項目および Retracted/revised セクションを参照。
