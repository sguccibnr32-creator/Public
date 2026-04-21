# membrane_v478 — v4.7.8 main body, arXiv submission package

## 概要

膜宇宙論 v4.7.8 本体の LaTeX source と arXiv 投稿用パッケージ。
**2026-04-21 affiliation 修正版** (Kobe → Shisō, Hyogo)。

- Target arXiv cat: **astro-ph.CO**
- 18 pages, 38 references (full bibliography via `\nocite{*}`)
- Language: English
- Source: reconstructed from original ReportLab PDF (57 KB, 13 pages, 2026-04 original)

## 修正内容 (2026-04-21)

- **Affiliation**: `Kobe, Japan` → `Shisō, Hyogo, Japan` (正: 兵庫県宍粟市)
- **Version tag**: `v4.7.8, rev. LV independence` → `v4.7.8, 2026-04-21 affiliation rev.`
- 本文内容は完全一致(19 結論、全数値、全 table、全 reference)

## ファイル一覧

| file | 役割 |
|---|---|
| `membrane_v478.tex` | 本文 LaTeX ソース (UTF-8, pdfLaTeX 互換) |
| `refs_v478.bib` | BibTeX 参考文献 (38 件、DOI 付) |
| `membrane_v478.bbl` | BibTeX 処理済 (arXiv 投稿用) |
| `Makefile` | コンパイル自動化 |
| `README.md` | 本ファイル |

## ビルド要件

- **TeX Live 2023+**
- 必須パッケージ: `texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`
- コンパイラ: **pdfLaTeX** (LuaLaTeX/XeLaTeX でも可)

## コンパイル手順

```bash
make              # pdflatex + bibtex + pdflatex x2
make arxiv        # arXiv 投稿用 tar.gz 生成
make clean        # 中間ファイル削除
```

手動の場合:
```bash
pdflatex membrane_v478
bibtex   membrane_v478
pdflatex membrane_v478
pdflatex membrane_v478
```

## arXiv 投稿手順

1. `make arxiv` で `membrane_v478_arxiv.tar.gz` 生成
2. https://arxiv.org/submit で新規投稿
3. Primary category: **astro-ph.CO**
4. tar.gz をアップロード (arXiv は自動で pdflatex + bibtex を走らせるが、同梱 `.bbl` により参考文献解決が保証される)
5. author: Shinobu Sakaguchi
6. affiliation: Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan
7. comments 欄: `v4.7.8, 18 pages, 38 references. Affiliation corrected from Kobe to Shisō, Hyogo (2026-04-21).`

## v4.8 companion paper との関係

本論文 (v4.7.8) と companion paper (v4.8) は分離戦略により並行投稿する:

| | v4.7.8 本体 (本パッケージ) | v4.8 companion |
|---|---|---|
| 層 | observational establishment | theoretical foundation |
| 言語 | 英語 | 日本語 |
| 証拠 | SPARC 175 + dSph 31 + HSC 503M pairs | 閉形式 11+ 個 + NGC 3198 2.32% |
| 主結論 | C15, MOND 棄却 p=1.66×10⁻⁵³ | α_PT^upper = 2.96×10⁻⁵³, τ_m bound |
| 頁数 | 18 (LaTeX) | 14 (LaTeX) |
| arXiv primary | astro-ph.CO | astro-ph.CO + hep-th cross |

v4.8 companion も同じ v48_release package に同梱。

## Known issues

- Page count: 18 pages (LaTeX 版) vs 13 pages (original ReportLab)
  - 英語 LaTeX article class のデフォルト行間と余白が ReportLab 本体より緩く、同一内容で自然に 5 頁増
  - 内容は完全一致、arXiv 投稿上は問題なし
- Bibliography: 38 entries displayed via `\nocite{*}`
  - 原論文の References list 形式と一致
- 指数表記: 全て LaTeX math form (`\e{-53}` マクロ → `\times 10^{-53}`)

## 連絡先

**Shinobu Sakaguchi** (坂口 忍)
Sakaguchi Seimensho, Shisō, Hyogo, Japan (兵庫県宍粟市)
Web: https://sakaguchi-physics.com

## License

MIT License (同 v48_release)
