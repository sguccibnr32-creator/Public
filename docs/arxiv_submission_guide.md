# arXiv 投稿ガイド (v4.7.8 本体 + v4.8 companion 英語 + v4.8 companion 日本語 の併行投稿)

本ガイドは 2026-04-21 affiliation 修正 + 英語版統合版の arXiv 投稿手順を step-by-step で説明します。

**新方針 (推奨)**: v4.8 companion は英語版を arXiv primary submission、日本語版は GitHub + WordPress のみ。
**代替**: 英語版と日本語版を独立 arXiv submission で並行投稿。

## 投稿前チェックリスト

- [ ] arXiv アカウント作成済 / endorsement 確認済
  - Independent researcher の場合、astro-ph.CO への最初の投稿には **endorsement** が必要
  - endorsement は既 arXiv 投稿者(推奨: 博士号以上の astro-ph 投稿者)から取得
  - 参考: https://arxiv.org/help/endorsement
- [ ] ORCID ID 取得済(推奨、publication history 追跡用)
- [ ] `arxiv/membrane_v478_arxiv.tar.gz` (32 KB) のダウンロード確認
- [ ] `arxiv/membrane_v48_en_arxiv.tar.gz` (22 KB、**英語版、primary**) のダウンロード確認
- [ ] `arxiv/membrane_v48_arxiv.tar.gz` (24 KB、日本語版、optional) のダウンロード確認
- [ ] 全 tarball の standalone compile test 済 (本リリースで済)

---

## Step 1: v4.7.8 本体の arXiv 投稿

### 1-1. 投稿画面へアクセス
https://arxiv.org/submit でログイン → "Start New Submission"

### 1-2. メタデータ入力

| フィールド | 入力値 |
|---|---|
| **Title** | `Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology: The Geometric Mean Law and Observational Tests` |
| **Author(s)** | `Shinobu Sakaguchi` |
| **Affiliation** | `Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan` |
| **Abstract** | 論文 Abstract 全文 (コピー&ペースト)、または arXiv が tex から自動抽出 |
| **Comments** | `v4.7.8, 18 pages, 38 references. Affiliation corrected from Kobe to Shisō, Hyogo (2026-04-21 rev.)` |
| **Primary category** | `astro-ph.CO` (Cosmology and Nongalactic Astrophysics) |
| **Cross-list** | なし(本体は単独 astro-ph.CO) |
| **MSC-class** | 85A40 (Cosmology) を追加推奨 |

### 1-3. ソースファイルアップロード
- upload: **`arxiv/membrane_v478_arxiv.tar.gz`**
- 含まれるもの: `membrane_v478.tex`, `membrane_v478.bbl`, `refs_v478.bib`, `README.md`
- arXiv の auto-compilation で PDF 生成 → 目視確認で 18 頁、affiliation "Shisō, Hyogo, Japan" になっていることを確認
- もし auto-compile に失敗する場合、`.bbl` が含まれていれば bibtex step は不要

### 1-4. 確認と submit
- Preview PDF で表紙の affiliation が **"Shisō, Hyogo, Japan"** になっていることを必ず確認
- License: `arXiv.org perpetual, non-exclusive license 1.0` 推奨(デフォルト)
- Submit → arXiv ID 採番(例: `arXiv:2604.XXXXX`)

### 1-5. arXiv ID の記録
採番された ID を控える:
```
arXiv ID (v4.7.8): arXiv:____________  (←ここに記録)
Announcement date: ____________
```

---

## Step 2a: v4.8 companion paper 英語版 の arXiv 投稿 ⭐ **推奨 primary**

**重要**: Step 1 で採番された v4.7.8 の arXiv ID を v4.8 英語版の reference に反映する。

### 2a-1. 事前準備: refs_v48_en.bib の更新
`latex_v48_en/refs_v48_en.bib` の `Sakaguchi2026a` エントリを編集:
```bibtex
@MISC{Sakaguchi2026a,
  author       = {{Sakaguchi}, S.},
  title        = {{Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology:
                   The Geometric Mean Law and Observational Tests (v4.7.8)}},
  howpublished = {arXiv preprint},
  year         = {2026},
  note         = {arXiv:XXXX.XXXXX [astro-ph.CO]},    ← Step 1-5 の ID を記入
  eprint       = {XXXX.XXXXX},                         ← Step 1-5 の ID を記入
  archivePrefix = {arXiv}
}
```

編集後、再コンパイルして tarball を再作成:
```bash
cd latex_v48_en/
make clean
make
make arxiv
```

### 2a-2. 投稿画面へアクセス
https://arxiv.org/submit で "Start New Submission"

### 2a-3. メタデータ入力

| フィールド | 入力値 |
|---|---|
| **Title** | `Membrane Cosmology Foundation Layer: Closed-form Derivation of FIRAS μ-distortion Upper Bound and Universal Density Coupling (v4.7.8 companion paper, v4.8)` |
| **Author(s)** | `Shinobu Sakaguchi` |
| **Affiliation** | `Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan` |
| **Abstract** | 論文 Abstract 全文 (コピー&ペースト)、または arXiv が tex から自動抽出 |
| **Comments** | `v4.7.8 companion paper (English version). 13 pages, 11 references. Companion to arXiv:XXXX.XXXXX (v4.7.8 main body).` |
| **Primary category** | `astro-ph.CO` |
| **Cross-list** | **`hep-th`** を追加 |

### 2a-4. ソースファイルアップロード
- upload: **`arxiv/membrane_v48_en_arxiv.tar.gz`** (2a-1 で更新後の場合は make arxiv で再生成済のもの)
- arXiv は pdfLaTeX を自動で走らせる (英語のみのため問題なし)
- auto-compilation で 13 頁 PDF 生成確認

### 2a-5. arXiv ID 記録
```
arXiv ID (v4.8 English): arXiv:____________
Announcement date: ____________
```

---

## Step 2b: v4.8 companion paper 日本語版 の arXiv 投稿 (optional、国内読者向け)

**判断**: Step 2a の英語版のみで十分と判断する場合は Step 2b をスキップ可。
日本語圏読者への配慮として独立 submission を行う場合のみ実行。

### 2b-1. 事前準備: references.bib の更新
`latex_v48/references.bib` の `Sakaguchi2026a` エントリを Step 1 の arXiv ID に更新:
```bibtex
@MISC{Sakaguchi2026a,
  author       = {{Sakaguchi}, S. (坂口 忍)},
  title        = {{Galactic Rotation Without Dark Matter from Elastic Membrane Cosmology:
                   The Geometric Mean Law and Observational Tests (v4.7.8)}},
  howpublished = {arXiv preprint},
  year         = {2026},
  note         = {arXiv:XXXX.XXXXX [astro-ph.CO]},
  eprint       = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

編集後、再コンパイル:
```bash
cd latex_v48/
make clean
make
make arxiv
```

### 2b-2. 投稿画面へアクセス
https://arxiv.org/submit で "Start New Submission"

### 2b-3. メタデータ入力

| フィールド | 入力値 |
|---|---|
| **Title** | `膜宇宙論 foundation layer: FIRAS μ 歪み上限と universal density coupling の閉形式導出 --- v4.7.8 companion paper (v4.8)` |
| **English title** | `Membrane Cosmology Foundation Layer: Closed-form Derivation of FIRAS μ-distortion Upper Bound and Universal Density Coupling (v4.7.8 companion paper, v4.8; Japanese version)` |
| **Author(s)** | `Shinobu Sakaguchi (坂口 忍)` |
| **Affiliation** | `Sakaguchi Seimensho (Independent Research), Shisō, Hyogo, Japan` |
| **Abstract** | Abstract (日本語) + English summary 2-3 sentences 併記を推奨 |
| **Comments** | `v4.7.8 companion paper (Japanese version). 14 pages, 15 references. English version: arXiv:YYYY.YYYYY. Companion to arXiv:XXXX.XXXXX.` |
| **Primary category** | `astro-ph.CO` |
| **Cross-list** | `hep-th` |

### 2b-4. ソースファイルアップロード
- upload: **`arxiv/membrane_v48_arxiv.tar.gz`**
- arXiv は LuaLaTeX を自動検出(`ltjsarticle` class から判定)
- auto-compile が日本語で失敗する場合のため、`latex_v48/README_tex.md` の compile instruction を comment 欄に含める

### 2b-5. 日本語論文特有の注意
- arXiv は日本語論文を受け入れ可能だが、**English abstract** を必須とするため Abstract 欄には英語 summary も記載
- 英語版 (Step 2a) が primary なので日本語版の審査遅延リスクは低い
- Comments 欄で英語版への cross-reference を明示

### 2b-6. arXiv ID 記録
```
arXiv ID (v4.8 Japanese, optional): arXiv:____________
Announcement date: ____________
```

---

## Step 3: GitHub release 公開

### 3-1. GitHub にログイン
https://github.com/sguccibnr32-creator/Public

### 3-2. 新規 release 作成
- "Releases" タブ → "Draft a new release"
- Tag: `v4.8` (create new tag on publish)
- Title: `v4.8 (v4.7.8 affiliation rev. + v4.8 companion)`

### 3-3. Release notes (body)
以下を body に貼り付け:

```markdown
## v4.8 Release (2026-04-21)

**arXiv 投稿 READY (v4.7.8 本体 + v4.8 companion 英語 primary + 日本語 optional の 3 論文併行投稿体制)**

### 主要な変更 (v4.7.8 affiliation 修正)
- v4.7.8 本体の affiliation を **Kobe → Shisō, Hyogo** に修正
- LaTeX 版 (18 頁) を arXiv 投稿版として採用、original ReportLab 版 (13 頁) も archive 保存

### arXiv 投稿
- v4.7.8 本体: arXiv:XXXX.XXXXX (astro-ph.CO)  ← Step 1-5 で採番された ID
- v4.8 companion 英語 (primary): arXiv:YYYY.YYYYY (astro-ph.CO + hep-th cross)  ← Step 2a-5 で採番された ID
- v4.8 companion 日本語 (optional): arXiv:ZZZZ.ZZZZZ (astro-ph.CO + hep-th cross)  ← Step 2b-6 で採番された ID (Step 2b 実施時のみ)

### 含まれるファイル
- v4.7.8 本体: LaTeX source + 18頁 PDF + 元 13頁 PDF archive
- v4.8 companion: LaTeX source + 14頁 PDF + ReportLab 13頁 PDF
- foundation_integrated.pdf (15頁)、cross_reference_audit_v3.pdf (5頁)
- ReportLab build scripts (再現性保証)
- WordPress 公開ページ HTML

詳細: 同梱 `README.md` および `CHANGELOG.md` を参照。
```

### 3-4. Asset アップロード
- **attach**: `v48_release.zip` (2.5 MB、42 files)

### 3-5. Publish release
- "Publish release" クリック
- Download URL を控える(WordPress で使用)

---

## Step 4: WordPress 公開ページ投稿

### 4-1. WordPress.com admin にログイン
https://sakaguchi-physics.com/wp-admin/

### 4-2. 新規固定ページ作成
- "固定ページ" → "新規追加"
- Title: `膜宇宙論 v4.8 companion paper 公開`
- Slug: `v48-companion` 推奨

### 4-3. HTML モードでペースト
- エディタを **クラシック(HTML)モード** に切替
  - Gutenberg の場合: 右上 "..." メニュー → "Code editor"
- `docs/wordpress_page_v48.html` **全体**をコピー & ペースト
- **重要**: `pages.update` は MCP 経由で truncation リスクがあるため、**必ず admin UI から手動 paste**

### 4-4. arXiv ID 反映
WordPress HTML および 関連 text 内の以下の placeholder を、採番された実 ID に置換:
- `XXXX.XXXXX` (v4.7.8) → Step 1-5 で採番された ID
- `YYYY.YYYYY` (v4.8 英語 primary) → Step 2a-5 で採番された ID
- `ZZZZ.ZZZZZ` (v4.8 日本語 optional) → Step 2b-6 で採番された ID (Step 2b 実施時のみ)

### 4-5. GitHub release ダウンロード URL 反映
HTML 内のダウンロードリンクを Step 3-5 の URL に更新:
- `https://github.com/sguccibnr32-creator/Public/releases/download/v4.8/membrane_v48_companion.pdf`
- `https://github.com/sguccibnr32-creator/Public/releases/download/v4.8/membrane_arxiv_v478.pdf`
- `https://github.com/sguccibnr32-creator/Public/releases/download/v4.8/v48_release.zip`

### 4-6. プレビュー → 公開
- プレビューで全セクションの表示を確認
- 特に arXiv ID、ダウンロードボタンの動作を確認
- "公開" クリック

---

## Step 5: 投稿後の確認

### 5-1. arXiv 投稿確認(3-5 営業日後)
- Moderation を通過したら公開される
- announcement date を上記表に記録
- Google Scholar / NASA ADS への自動収録を数日後に確認

### 5-2. 相互リンク検証
- arXiv v4.7.8 の abstract で v4.8 companion が mentioned されているか
- arXiv v4.8 の abstract で v4.7.8 main body reference が正しいか
- WordPress / GitHub / arXiv 間のリンクがすべて機能するか

### 5-3. 引用情報の整理
Anyone who wants to cite:

**v4.7.8 main body:**
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
```

**v4.8 companion (English, primary):**
```bibtex
@misc{Sakaguchi2026b_en,
  author = {Sakaguchi, Shinobu},
  title  = {Membrane Cosmology Foundation Layer: Closed-form Derivation of FIRAS
            μ-distortion Upper Bound and Universal Density Coupling
            (v4.7.8 companion paper, v4.8)},
  year   = {2026},
  eprint = {YYYY.YYYYY},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  note   = {English version, companion to arXiv:XXXX.XXXXX}
}
```

**v4.8 companion (Japanese, optional):**
```bibtex
@misc{Sakaguchi2026b_ja,
  author = {Sakaguchi, Shinobu (坂口 忍)},
  title  = {膜宇宙論 foundation layer: FIRAS μ 歪み上限と universal density coupling の閉形式導出 --- v4.7.8 companion paper (v4.8)},
  year   = {2026},
  eprint = {ZZZZ.ZZZZZ},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  note   = {Japanese version, companion to arXiv:XXXX.XXXXX; English version: arXiv:YYYY.YYYYY}
}
```

**推奨引用**: 国際誌には `Sakaguchi2026a` + `Sakaguchi2026b_en`、日本語文献には `Sakaguchi2026a` + `Sakaguchi2026b_ja` を使用。

---

## トラブルシューティング

### v4.7.8 tarball が arXiv で compile 失敗
- symptom: "LaTeX Warning: Citation 'XXX' undefined" 等
- 原因: `.bbl` は同梱しているが、arXiv は強制的に bibtex を再実行することがある
- 対処: refs_v478.bib が tarball に含まれていることを確認(含まれている)

### v4.8 tarball が arXiv で LuaLaTeX を検出しない
- symptom: pdflatex で compile を試みて Japanese font エラー
- 原因: arXiv の default は pdflatex
- 対処: submission form の "format" で "LaTeX using ptex (Japanese)" を選択、または comments 欄に "Compile with lualatex (luatexja/ltjsarticle required)" を記載

### endorsement が取れない
- Independent researcher の場合、最初の astro-ph 投稿には endorsement が必須
- 対処: (i) 知り合いの研究者 (arXiv 投稿経験 >5 回) に依頼, (ii) sakaguchi-physics.com の contact form で endorsement request を受け付ける可能性のある研究者にメール, (iii) 学会 (日本物理学会、天文学会) で知り合った研究者に依頼
- 参考: https://arxiv.org/help/endorsement

### 日本語論文のため referee 不足
- astro-ph の査読者が日本語を読めないため査読遅延リスク
- 対処: v4.8 companion paper の Abstract を英語併記、主要 conclusion は英語、Section headings を英語併記

---

## 優先順位と recommendation

**投稿順序**: **Step 1 (v4.7.8) → Step 2a (v4.8 英語) → [optional Step 2b (v4.8 日本語)] → Step 3 (GitHub) → Step 4 (WordPress)**

### なぜこの順序か
1. v4.7.8 を先行投稿することで **priority claim**(優先権主張)を確立
2. v4.8 companion (英語版) は v4.7.8 arXiv ID を reference に含める必要があるため順序依存
3. 日本語版 (Step 2b) は optional、英語版で international reach を確保してから判断
4. GitHub release は v4.7.8 + v4.8 英語の arXiv ID を含める形で公開
5. WordPress は最終的に全 ID を統合して公開

### 推定所要時間
- Step 1: 30 分 (初回 arXiv 投稿、endorsement 取得済の場合)
- Step 2a: 30 分 (v4.7.8 ID 反映 + 再 compile + 投稿、英語版)
- Step 2b: 30 分 (optional、日本語版)
- Step 3: 15 分
- Step 4: 15 分 (HTML コピペ + ID 反映)
- **合計: 1.5 時間** (Step 2a まで、推奨) / **2 時間** (Step 2b 含む)

### endorsement 未取得の場合
- endorsement 取得に 1-2 週間要する場合あり
- 先に Step 3, 4 (GitHub + WordPress) を arXiv ID なしで公開し、ID 採番後に追記する戦略も可能

### 英語版のみ投稿する戦略的理由
1. **International reach**: astro-ph.CO の reader base はほぼ全員英語読解可能
2. **Referee 確保**: 日本語版は astro-ph の査読者プール制約によ り moderation 遅延リスク
3. **Citation tracking**: NASA ADS / Google Scholar / Semantic Scholar での発見性は英語版が有利
4. **日本語読者**: sakaguchi-physics.com + GitHub で日本語版 PDF を公開すれば日本語圏 reach も確保可能

したがって、**英語版 (Step 2a) のみ arXiv 投稿、日本語版は GitHub + WordPress でのみ公開**が最も効率的。
