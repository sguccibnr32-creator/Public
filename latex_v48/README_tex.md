# membrane_v48 — v4.7.8 companion paper (v4.8) arXiv submission

## 概要

膜宇宙論 v4.7.8 本体の companion paper (v4.8)。FIRAS μ 歪み上限と universal density
coupling の閉形式導出を 21 頁 (LuaLaTeX 版、A4, 20mm margin) で提示する
(publication final, post-Phase 5-2 chain inject)。

- Target arXiv cat: **astro-ph.CO + hep-th cross**
- Source: ReportLab PDF (`membrane_v48_body.pdf`, 13 頁, v3 patched) からの LaTeX 移植
- v3 までの patch (D1 = A1 operational / V_ξ_void, U3 = #17 11%/5.5% 処理統一) を全て反映済
- v4.8 publication final pass (2026-04-27) で Phase 5-2 chain inject (5.2.A/B-1/B-3/B-4/C/M
  全閉合) + 5.2.M v1 canonical commit + 5 件 corrections/mitigations を適用済

## ファイル一覧

| file | 役割 |
|---|---|
| `membrane_v48.tex` | 本文 LaTeX ソース (UTF-8, LuaLaTeX) |
| `references.bib` | BibTeX 参考文献 (15 件、DOI/eprint 付) |
| `membrane_v48.bbl` | BibTeX 処理済 (arXiv 投稿用) |
| `Makefile` | コンパイル自動化 |
| `README.md` | 本ファイル |

## ビルド要件

- **TeX Live 2023+** (arXiv 互換)
- 必須パッケージ: `texlive-luatex`, `texlive-lang-japanese`, `texlive-latex-extra`, `texlive-fonts-recommended`
- 必要フォント: **IPAGothic + IPAPGothic** (`texlive-lang-japanese` 同梱)
- コンパイラ: **LuaLaTeX** (XeLaTeX/pdfLaTeX は使わない)

## コンパイル手順

### Make 経由 (推奨)

```bash
make              # lualatex x 1 + bibtex + lualatex x 2 で完全生成
make quick        # bibtex を省略した高速再ビルド
make clean        # 中間ファイル削除
make arxiv        # arXiv 投稿用 tar.gz 生成
```

### 手動コンパイル

```bash
lualatex membrane_v48
bibtex   membrane_v48
lualatex membrane_v48
lualatex membrane_v48   # 3 回目は cross-reference 解決用
```

## arXiv 投稿手順

1. `make arxiv` で `membrane_v48_arxiv.tar.gz` を生成
2. arXiv の submission interface (https://arxiv.org/submit) で新規投稿
3. Primary category: **astro-ph.CO**、Cross-list: **hep-th**
4. tar.gz をアップロード
   - arXiv は自動的に `lualatex` + `bibtex` をリランするが、同梱 `.bbl` により参考文献は確実に解決
5. author: 坂口 忍 (Sakaguchi Shinobu)、affiliation: 坂口製麺所 (兵庫県宍粟市)
6. comment 欄に `v4.7.8 companion paper; 21 pages, 15 references; LuaLaTeX (Japanese)` を記載

## v4.7.8 本体との関係

本 companion paper (v4.8) は v4.7.8 本体 (`Sakaguchi 2026a`) の theoretical foundation
layer を提供する分離戦略の一環。両論文は非干渉で、arXiv に独立投稿する。

| | v4.7.8 本体 | v4.8 companion |
|---|---|---|
| 層 | observational establishment | theoretical foundation |
| 証拠 | SPARC 175 + dSph 31 + HSC 503M | 閉形式 11+ 個 + NGC 3198 2.32% |
| 主結論 | C15, MOND 棄却 p=1.66×10⁻⁵³ | α_PT^upper, τ_m bound |
| 頁数 | 12 | 21 (TeX, publication final) / 13 (ReportLab PDF) |
| arXiv cat | astro-ph.CO | astro-ph.CO + hep-th cross |

## 移植上の変更点 (ReportLab PDF → LaTeX)

1. **指数表記統一**: `2.96e-53` → `$2.96 \times 10^{-53}$` (全 40+ 箇所)
2. **BibTeX 化**: inline reference table → `natbib` 形式の 15 件
3. **マクロ定義**: 頻出物理量 (`\kmem`, `\cmem`, `\Vxi`, `\Tm`, `\aPT`, `\taum`, `\bal` 等 40+) を preamble に集約
4. **ボックス環境**: ReportLab の STYLE_KEY/NOTE/WARN → `tcolorbox` 環境 (`keybox`, `notebox`, `warnbox`)
5. **表**: ReportLab Table → LaTeX `booktabs` / `tabularx` / `longtable`
6. **日本語フォント**: IPAPGothic (main) + IPAGothic (sans/bold) を `luatexja-fontspec` 経由で統一

## 内容上の変更点 (v3 patched PDF からの版間差異)

**なし** (内容は完全一致)。v1→v2 の D1 patch (A1 operational / V_ξ_void)、v2→v3 の U3 patch
(#17 11%/5.5% 処理統一) は全て反映済。

## Known issues

- (publication final で 0 解消) overfull \hbox / underfull boxes: 0
- (publication final で 0 解消) compile errors: 0 / harmful warnings: 0 (141 全 harmless)
- `hyperref` による PDF bookmark で一部数式記号が warning を出すが、PDF 本文には影響なし

## 再現性

- 付属 Windows Claude Code 実装 (`foundation_gamma_actual.py` ほか 13 scripts) は
  `D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\` で
  `uv run --with numpy --with scipy --with matplotlib --with pandas python {script}` により実行可能
- GitHub 公開先: https://github.com/sguccibnr32-creator/Public (MIT License、v4.8 release 準備中)

## 変更履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v3 (PDF) | 2026-04-21 | D1 + U3 patch 適用、arXiv 投稿 READY |
| TeX v1 | 2026-04-21 | ReportLab PDF → LuaLaTeX 移植 (Session +19) |
| v4.8 publication final | 2026-04-27 | Phase 5-2 chain inject (5.2.A/B-1/B-3/B-4/C/M 全閉合) + 5.2.M v1 canonical commit + 5 件 corrections/mitigations 適用、21 pages、compile errors 0 / overfull 0、forensic anchor 3 段 (5.2.M v1+v2+v3) push 完了 (commit 03e3aab → 18e1630 → 45d5461 → README 更新 commit) |

## 連絡先

坂口 忍 (Sakaguchi Shinobu)
坂口製麺所、兵庫県宍粟市
Web: https://sakaguchi-physics.com
