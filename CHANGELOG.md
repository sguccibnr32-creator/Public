# 5.2.M v2 CMB precision upgrade — B-2 v4 sensitivity + delta_chi2_query (2026-04-27)

## Added
- `phase5/step5_2_M_v2/` (18 files pulled from WSL2 ~/plik_b2_full/):
  - **canonical script**: `_step5_2_b2_cmb_tt_posterior.py` (59,244 B, SHA 03424bd0...)
  - **delta_chi2_query.json** (3,860 B, SHA 11ab6635...): cross-variant Δχ² aggregate
    across 9 variants (1 archived v3b grid-500 + 8 v4 sensitivity)、Path A central
    1.0025822368421053 vs LCDM ε=1 vs likelihood minimum
  - **8 v4 sensitivity struct JSON** (~9,727 B each, ~78 KB total):
    baseline / alpha_low / alpha_2x / alpha_5x / peak_2nd / peak_3rd / peak_4th / width_2x
  - **8 v4 sensitivity summary TXT** (~3,500 B each, ~28 KB total): matching variants
- `.gitattributes`: `phase5/step5_2_M_v2/** -text` rule

## Resolves
- Action 3 GitHub repo verify step (8) — previously ❌, now ✅:
  B-2 v4 sensitivity / delta_chi2_query JSON は repo に存在。

## Verified
- Δχ²(Path_A → LCDM) ≤ 0.013 across all 8 v4 kernel/coupling variants
  (median +0.0027, max +0.0133 at peak_3rd) — paper §6 核心数値 reproducibility 確保
- Plik TT-only native test χ² = -380.341 (v4_baseline 内 invariant)
- v3b_grid500_archived ↔ v4_baseline bit-exact equivalent (regression check)

---

# v4.8 README cosmetic update + arXiv tar.gz regenerate (2026-04-27)

## Patched
- `latex_v48/README_tex.md` (SHA256: 7f7680f4... → cc5278f1...):
  - 14 pages → 21 pages (publication final, post-Phase 5-2 chain inject)
  - Known issues: page 8 overfull \hbox 1 件 → "overfull / underfull / errors all 0"
  - 変更履歴 entry 追加: "v4.8 publication final 2026-04-27"
  - 概要・対比表 (v4.7.8 vs v4.8 companion) 反映
- `arxiv/membrane_v48_arxiv.tar.gz` regenerated (33,819 → 34,015 bytes):
  - tex / bbl / bib bit-exact unchanged (publication final immutable)
  - README.md updated (= latex_v48/README_tex.md SHA cc5278f1...)
  - tar.gz SHA256: e40c87b57b4b59dcb296e5c21dddafbe2e96e9cdc7de72fea41b91b9d2e15cf4

## Rationale
- Bundled README claimed "14 pages" but publication final is 21 pages
  (post-Phase 5-2 chain inject). arXiv reviewer / Zenodo archivist が
  tar.gz 解凍時に paper 本体と矛盾を見ることを防止。
- Forensic chain (paper §A1 anchor block) は README SHA256 を参照しないため、
  immutable invariants には影響なし。

---

# v4.8 publication final pass FULL CLOSURE (2026-04-27) — Phase 5-2 chain inject

## Added
- **Phase 5-2 chain inject** into `latex_v48/membrane_v48.tex` (8 deliverables,
  +362 lines: 1019 → 1382 lines):
  - §3.2 footnote, §5.5 (main + 3 subsubsections: nuisance / kernel sweep / TIER),
    §6.6 canonical bibliography, §6.7 Plik TT consistency, §6.8 ε_scale operational
    + Method 2 deferral, §7.5 patch, §A1 Phase 5-2 chain anchor + forensic anchor
    block, §A2 patch (#1--32 + 3 row update + #32 row 新設)
- **5.2.M v1 canonical commit deliverables** at `phase5/step5_2_M_v1/`:
  - `_step5_2_M_multi_route_min.py` (SHA256 09d91d30...0f381f)
  - `5_2_M_canonical_bibliography.md` (SHA256 f63696d7...036056)
  - `5_2_M_canonical_bibliography.txt` (SHA256 248548a6...082a48)
  - `phase5_step5_2_M_v1_canonical_commit_struct.json` (SHA256 c2837e2c...55e2e)
  - `phase5_step5_2_M_v1_canonical_commit_summary.txt` (= bibliography.txt と二重)
- **Forensic baseline backup**: `latex_v48/membrane_v48.tex.bak.pre_phase5_2_inject`
  (SHA256 368a1b3d...4af06, 1019 lines, base v4.8 pre-inject 状態保存)
- **`.gitattributes`**: `phase5/step5_2_M_v1/**` および `latex_v48/membrane_v48.tex*`
  に対し `-text` 指定 (autocrlf による cross-platform clone での line ending
  conversion drift 防止 → paper §A1 forensic anchor SHA256 claim integrity 保護)
- **`.gitignore`**: `!latex_v48/membrane_v48.pdf` exception 追加 (publication
  final PDF を tracked 化)

## Patched
- 5 件 corrections / mitigations to `latex_v48/membrane_v48.tex`:
  1. §6.8 P3 (ii) attribution fix (finding_1_correction.md ベース、(a)/(b) を
     §7.1 (a) σ rest energy / (b) membrane coherence と alignment)
  2. §6.8 内 `\paragraph{}` × 2 → `\medskip\noindent\textbf{}` (titlesec error mitigation)
  3. §7.5 v4.9 roadmap (tabular → tabularx + X column wrap)
  4. §1.4 M1-M5 claims table (col 2 `l` → `p{75mm}` raggedright + col 4 → `p{45mm}`)
  5. §A1 longtable (`{@{}lll@{}}` → `{@{}p{43mm}p{53mm}p{60mm}@{}}` raggedright)
- arXiv submission package regenerated: `arxiv/membrane_v48_arxiv.tar.gz`
  (33,819 bytes; tex + bbl + bib + README, PDF excluded per arXiv recompile policy)

## Verified
- compile: lualatex × 3 + bibtex (TeX Live 2026, LuaHBTeX 1.24.0)
- compile errors: 0, overfull \hbox: 0, underfull boxes: 0, harmful warnings: 0
- final PDF: `latex_v48/membrane_v48.pdf` (21 pages, 1,013,247 bytes,
  SHA256 5d9467ff...41e88)
- final tex: `latex_v48/membrane_v48.tex` (1382 lines, 77,531 bytes,
  SHA256 394f2571...c7e91053)

## Phase 5-2 chain immutable claims (publication 確定数値)
- canonical posterior derivations: 33 (TIER-1 39 − reproductions 6)
- B-4 reproductions log_diff = 0.000000 dex (bit-exact, 6/6 within 1e-4 dex)
- Δχ²(Path_A → LCDM) ≤ 0.013 across all 8 kernel/coupling variants
  (median +0.0027, max +0.0133 peak_3rd)
- chain extension drift_max = 0.0 dex
- Plik TT-only native test χ² = −380.341 (FULL mode anchor)
- Plik TTTEEE native test χ² = −1172.47 (PROXY mode anchor)
- Path A central 1.0025822368421053 (unchanged from v4.7.8)

## Pending (carry-over)
- **MEDIUM-OPTIONAL**: B-2 v4 sensitivity + delta_chi2_query JSON pull back from
  WSL2 → 5.2.M v2 canonical commit (precision upgrade、forensic chain 後付け強化、
  数値は同値)
- **LOW**: kernel choice motivation 別検討 (theory work, v4.9 patch round)
- **LOW**: cleanup (~/plik_b2_full/sensitivity_v4/ archive 等)

## Retract 不可 invariants (全 step 遵守済)
- #22(vi) cascade SSoT 3 active bit-exact + 1 archived
- #26 multi-route min: A + B-1/B-2/B-3/B-4 + C
- #29 foundation_scale primary (Method 2 formal closure deferred to v4.9)
- #30 no single-value commit (33 canonical + 1 Path A + 8 v4 audit)
- #32 v_flat layer separation (c=0.42 galaxy-specific NGC_3198 / c=0.83 cascade canonical 完全分離)

---

# v4.8 erratum v3 (2026-04-23) — deg-4 f_opt cascade + universal Λ_UV retract

## Changed
- **f_opt(0.83)**: 1.9163 (v4.8 v1 2-pt f_opt-linear artifact) → **1.9425** (deg-4 V''-fit)
  - Method: Lagrange interpolation on Table 18-2 V''(x=0.5, c) at 5 points
    (c = 0.30, 0.42, 0.618, 0.80, 1.00); V''(x=0.5, 0.83) = 10.4626
  - F1 canonical form: f_opt = 2π/√V''
- **Cascade** (c₀ = 0.83, V_FLAT_TYP = 2.00×10⁵ m/s, a₀ = 1.2×10⁻¹⁰ m/s²):
  - c_mem: 3.833×10⁵ → **3.885×10⁵** m/s (+1.37%)
  - χ_F:   4.198 → **4.256** m^(3/2)/s² (+1.37%)
  - V_ξ:   7.683×10⁶³ → **8.335×10⁶³** m³ (+8.49%, c_mem⁶ scaling)

## Retracted
- **Universal Λ_UV(c)**: 9.549×10⁻⁴⁹ J [†] → **COMPLETELY RETRACTED**
  - Rationale (F2 structural): m_σ(c, galaxy) = √V''(x=0.5, c) / τ_dyn(galaxy)
    is galaxy-specific through τ_dyn, so universal m_σ(c) and hence universal
    Λ_UV(c) do not exist structurally in v3.7 sources.
  - Per-galaxy Λ_UV remain as sole anchors:
    IC 2574=1.955×10⁻⁴⁹ J, NGC 3198=2.307×10⁻⁴⁹ J (M1), NGC 2841=9.019×10⁻⁴⁹ J

## Unchanged (v2 per-galaxy primary, carried over)
- α_PT_upper(NGC 3198, V_ξ) = 1.76×10⁻⁵¹ (M1 anchor)
- 3-galaxy range: [1.6×10⁻⁵³, 1.8×10⁻⁵⁰] (~3 orders of magnitude)
- M3 symbolic identity: ratio = 1.0000 exact (MRT-invariant)
- B_FIRAS_combo, I_μ^S1, S2/S1, τ_m_lower (c_mem-independent)
- Factor 59 weakening narrative (v2 vs v4.8 v1)

## Patched (5 layers, synchronized)
- arXiv LaTeX JA: `latex_v48/membrane_v48.tex` (5 locations: §3.1, §3.3, §4.2, §7.3, §A)
- arXiv LaTeX EN: `latex_v48_en/membrane_v48_en.tex` (5 locations)
- `foundation_gamma_actual.py` × 4 copies (sha256 synchronized, outside public repo)
- `step_iv_d_firas_mu_bound.py` (np.polyfit runtime, universal Λ_UV block removed; outside public repo)
- reportlab build scripts × 3: `build_foundation_integrated.py`,
  `build_membrane_v48_body.py`, `build_cross_reference_audit.py`

## Verified
- Python smoke tests: Route A=8.091 invariant, Route B Hubble=4.180e-18 (+1.38% cascade),
  Route C=5.827e-18 (+1.37%)
- Symbolic identity: k_mem · N_mode = (2/9π) c_mem² Λ²/(a₀^(5/2) ℏ³), ratio = 1.0000 exact
- All assertions PASS including v37 2.32% legacy (retained with [not physical,
  internal rounding residual] annotation)

## Pending (follow-up commit)
- `papers/membrane_v48_companion.pdf` and `papers/membrane_v48_en_companion.pdf`
  require LaTeX recompile (local LaTeX env not installed); will be committed in
  a follow-up commit after user's local LaTeX setup.
- `arxiv_v48_erratum_v3.md` / `.pdf` (Japanese) and EN counterpart — claude.ai side
  generation pending (Option 3).

## New SSoT (single source of truth, outside public repo)
- `v37_chap18_table18_2_vpp_full.csv` (30 points; 5 used for deg-4 fit input)
- `v2_per_galaxy_firas_table.csv` (per-galaxy primary, unchanged from v2)

---

# v48 release bilingual v3 (2026-04-23) — v1.1 anchor fix + PDF rebuild [commit b86cf24]

## papers/ PDF rebuilt to v1.1 erratum state
- papers/membrane_v48_companion.pdf: rebuilt via lualatex + bibtex + lualatex x2 (15 pages, 871 KB)
- papers/membrane_v48_en_companion.pdf: rebuilt via pdflatex + bibtex + pdflatex x2 (15 pages, 483 KB)
- Both PDFs now reflect v1.1 erratum anchors: 1.76e-51 primary (NGC 3198 MRT), factor 59 weakening, per-galaxy 3-row table, M3 closed-form reframe

## Source fix
- latex_v48_en/membrane_v48_en.tex L146: Unicode U+2248 replaced with $\approx$ (pdflatex T1 compatibility)

---

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
