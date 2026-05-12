# anchor 25 v0.1 lessons appendix — Q4 codify round v0.1

## metadata

- anchor id: anchor 25 v0.1
- round name: Q4 codify round v0.1
- creation date (JST): 2026-05-12
- baseline commit: cbc270041c7627b95e90399dc8a9eaee4f3cc8e1
- format: UTF-8 LF-only, no BOM (per Pattern 30 refinement / L-Q3-32 scope: data file は no-BOM)
- scope: 10 entries (substantive new codify 9 + disposition 1) — step 3a で Pattern 39 (amendment 03)、step 3b で Pattern 40 (amendment 04)、step 4 で Pattern 41 発見 (amendment 05)
- 本 round の特徴: forensic chain self-discovery round (★ 6 新 pattern を round 進行中に self-discover + in-round codify、+ Pattern 37/38 anchor 26 deferred candidate を含めると 8 件)

## entries (numeric order)

- L-Q3-10: disposition note (documentation drift 1st + 2nd instance canonical record)
- L-Q3-30: Pattern 24c — PS string interpolation `${(expr)}` literal name quirk
- L-Q3-31: Pattern 34 — `[UTF8Encoding]::new($true).GetBytes()` no BOM
- L-Q3-32: Pattern 30 refinement — BOM scope = `.ps1` only, data files exempt
- L-Q3-33: Pattern 35 — PS 5.1 `Get-Date -Format` culture-sensitive parsing
- L-Q3-34: Pattern 36 — forensic record 内 path reference 実存性 verify 必須
- L-Q3-35: Pattern 29 refinement — CWD divergence detection の 2 階層化
- L-Q3-36: Pattern 39 — PS cmdlet vs .NET BCL CWD source divergence
- L-Q3-37: Pattern 40 — PS native cmd output CRLF + regex `$` anchor incompatibility
- L-Q3-38: Pattern 41 — structure validator regex must account for legitimate non-data lines

---

## L-Q3-10 disposition (anchor 25 v0.1 closure)

### 状態
**fully resolved**、新 substantive codify body 不要、defer queue から除外。
本 entry は disposition note のみで構成 (meta-resolution、cross-reference 形式)。

### Interpretation A: autocrlf-induced SHA shift on text-mode files

- **origin**: `section5_axis_4_type_alpha/anchor_22_v0_2_lessons_appendix.md` L17-18, L183, L199
- **root cause**: `core.autocrlf=true` (system scope) + `.gitattributes` self-coverage gap、commit-time CRLF conversion で wt SHA / blob SHA divergence 発生
- **resolution**: **L-Q3-29 / Pattern 33 で structural fix 完了** (anchor 24 v0.1、commit `cbc2700`)
  - `.gitattributes` self-cover (`".gitattributes -text"`、21 B 追加)
  - section7 directive (`"forensic_anchors/section7_lessons_codified_q3_v0_2/** -text"`、60 B 追加)
  - 6/6 wt_eq_blob 三重一致 verified (Protocol 1 git wire + Protocol 2 HTTPS raw URL)
- **disposition**: L-Q3-10 (A) は **SUBSUMED by L-Q3-29**、independent codify 不要

### Interpretation B: scipy cross-version reproducibility floor

- **記述位置**: `section6_lessons_codified_q3_v0_1/anchor_23_v0_1_lessons_appendix.md` L557
- **記述内容**: "scipy version-dependent reproducibility floor (same-scipy 1.17.1 では 12/12 bit-exact、cross-version での floor は別途要 evidence)"
- **主張**: "既に anchor 22 v0.2 verification_log 内で記述済"
- **検証結果 (anchor 25 v0.1 evidence harvest H.2)**: ★ **documentation drift 1st instance 検出**
  - `section5_axis_4_type_alpha/anchor_22_v0_2_verification_log.md` 全体 search、`scipy` / `1\.17` / `reproducibility` / `cross.version` 関連 occurrence **0 件**
  - 該当 L230-234 は A6 critical PASS context (encoding integrity、autocrlf prevention)、scipy 言及なし
  - → anchor 23 v0.1 L557 の "v0.2 verification_log 内で記述済" claim は empirically **reject**

- **真の canonical entry**: L-Q3-8 の substantive content は public repo の以下 3 file に完全保存 (`lessons_codified.md` は phantom path、本 round で public repo 内 absent 確認):

  | role | file (public repo verified) | size | sha256_12 |
  |------|------------------------------|------|-----------|
  | primary empirical evidence | `forensic_anchors/section5_axis_4_hardening/empirical_precheck_canonical_rerun.json` | 7,477 B | `52a2bcec4f78` |
  | closure record (cross-ref) | `forensic_anchors/section5_axis_4_hardening/axis_4_closure_summary.json` | 22,019 B | `16ed724c16e6` |
  | operational declaration | `forensic_anchors/section5_axis_4_hardening/OPERATIONAL_CLOSURE.md` | 25,051 B | `d9e3e12c3ac6` |

  empirical content (`empirical_precheck_canonical_rerun.json`):
  - L3: "Empirical pre-check result for L-Q3-8 cross-scipy-version reproducibility floor hypothesis"
  - L22: `"scipy_version": "1.17.1"`
  - L65: variant A CONFIRMED — L-Q3-8 same-scipy-version achievement at delta=0 (12 orders below 1e-12 atol)
  - L81: validation_strength: strong — 12/12 delta=0 (not merely <=1e-12 PASS but actual bit-exact float)
  - L82: cross-version floor is therefore confirmed scipy-version-induced not algorithm-induced

  ★ phantom path 注記 (documentation drift 2nd instance): `axis_4_closure_summary.json` L175 内の "lessons_codified.md L146" reference は public repo に実体存在せず (本 round V.3 全 repo search で 0 match)、internal working document (publish されず metadata のみ closure summary に記録) の可能性。本 disposition note では actual public repo files を canonical reference として採用。
  → 関連 lesson **L-Q3-34 (Pattern 36)** として本 round で codify (forensic record 内 path reference は public repo 実存性 verify 必須)。

- **disposition**: L-Q3-10 (B) は **上記 3 file が canonical reference**、independent codify 不要

### forensic chain self-consistency 回復

`section7_lessons_codified_q3_v0_2/anchor_24_v0_1_lessons_appendix.md` L279 の "still deferred、後続 round 候補" 文言は、anchor 23 v0.1 L557 の documentation drift 1st instance を carry forward した結果。anchor 25 v0.1 closure で本 disposition を確定することで、forensic chain の self-consistency が回復。

rule 1 IMMUTABLE 制約により section6 / section7 の retro-edit は不可、よって本 disposition note 自体が drift 1st + 2nd instance の canonical record となる。

### cross-reference summary

| identifier | role | location |
|------------|------|----------|
| L-Q3-8 | Interpretation B canonical (~1e-11 cross-scipy floor) | section5_axis_4_hardening/ 配下 3 file |
| L-Q3-29 | Interpretation A structural fix (Pattern 33) | section7_lessons_codified_q3_v0_2/anchor_24_v0_1_lessons_appendix.md |
| L-Q3-34 | Pattern 36 (phantom path detection、本 round 発見) | 本 lessons_appendix |
| L-Q3-10 (本 entry) | disposition note (drift 1st + 2nd instance canonical record) | 本 lessons_appendix |

---

## L-Q3-30: Pattern 24c — PS string interpolation `${(expr)}` literal name quirk

### 発見経緯

anchor 24 v0.1 step 3 (P15.1.1 redo) packet 起草中、PS 5.1 で `"...${(Get-FileHash $p).Hash}..."` 形式の interpolation が empty string を返す現象を発見。原因調査で `${...}` syntax は変数名 escape のための syntax (e.g., `${weird name with spaces}`) であって、内部の `(...)` は literal な名前として扱われ、subexpression evaluation は行われないことが判明。

### root cause

PS 5.1 string interpolation の 2 つの syntax:
- `$VAR`, `${VAR}`: 変数 reference (`${VAR}` は変数名 escape 用、内部は variable name として literal 解釈)
- `$(expr)`: subexpression、任意の式を evaluate して string 化

`${(expr)}` という記法は文法的には valid だが、`{` と `}` の間が "literal variable name" として解釈される。PS は `(expr)` という名前の変数を探そうとし、見つからないので empty string が返る。

### impact

- packet 起草時に `${...}` を subexpression 用に誤用すると、verification 出力が空白になり、SHA print や conditional check が silently fail
- detect 困難 (syntax error にならない、ただ empty 返す)
- 既に確立済の Pattern 24b (`${VAR}:` PSDrive collision avoidance) と sibling、両者の混同 risk あり

### mitigation

string interpolation 内で式を evaluate したい場合は **必ず `$(expr)`** (subexpression、`$()` 形式) を使用。`${expr}` は変数名 escape 専用と認識:

```powershell
# CORRECT
$sha = "$(Get-FileHash $p | Select -Expand Hash)"
"size=$($file.Length) bytes"

# INCORRECT (silently empty)
$sha = "${(Get-FileHash $p).Hash}"
"size=${(Get-Item $p).Length} bytes"
```

verify gate: packet 起草後 `grep '\${('` で 0 件 confirm (subexpression syntax 用に `${(` が登場することはない)。

### related patterns

- Pattern 24 (line-comment narrative parser): packet syntax integrity
- Pattern 24a (continuation `#` prefix): narrative format integrity
- Pattern 24b (`${VAR}:` PSDrive collision): 変数名 escape の必要性
- Pattern 24c (本 entry): subexpression 用 syntax と変数名 escape syntax の区別
- 4 entries collectively cover PS string syntax pitfalls in packet drafting

### empirical evidence

- 発見 source: anchor 24 v0.1 step 3 P15.1.1 redo packet、SHA print が empty で blob predicate verify が degenerate に PASS する silent failure form
- root cause isolation: PS 5.1 documentation review + 実機 echo test (`"${(1+1)}"` → empty string)
- mitigation 確立後: anchor 24 v0.1 step 3 以降の全 packet で `$(...)` 使用統一、grep `'\${('` で 0 件 confirm

---

## L-Q3-31: Pattern 34 — `[UTF8Encoding]::new($true).GetBytes()` no BOM

### 発見経緯

anchor 24 v0.1 step 5a 起草中、commit message の UTF-8 BOM 付与で `[System.Text.UTF8Encoding]::new($true)` の `.GetBytes(string)` の出力 bytes に BOM (EF BB BF) が含まれないことを実機検証で発見。constructor の `$true` 引数は "encoderShouldEmitUTF8Identifier" を意味し、その encoding が BOM-emitting "stamp" 付きであることを示すが、`.GetBytes(string)` 自体は **content の byte 列のみ** を返し、BOM byte は別途 `.GetPreamble()` で取得する必要がある。

### root cause

.NET の `Encoding.GetBytes(string)` method の API contract:
- pure content bytes のみ返す
- BOM (preamble) は含まない
- BOM byte は `.GetPreamble()` で取得 (`[byte[]]`、UTF-8 では `0xEF, 0xBB, 0xBF`)
- BOM-aware writers (`StreamWriter`, `File.WriteAllText(path, str, Encoding)`) は内部で `GetPreamble()` + content を concat して書き出す

つまり `Encoding` instance の constructor 引数は writer-related metadata であり、`GetBytes` 自体の動作には影響しない。

### impact

- BOM 必要な context (e.g., PS 5.1 `.ps1` script で UTF-8 explicit 認識させたい場合) で `.GetBytes` 直叩きすると BOM 抜け
- 受信側が encoding 推定で CP932 etc に misinterpret する risk
- `[UTF8Encoding]::new($true)` の `$true` を BOM 付き output と誤読する documentation drift も派生

### mitigation

BOM 必要時の 2 つの adequate path:

1. **WriteAllText 経由** (推奨、最 idiomatic):
   ```powershell
   $utf8_bom = New-Object System.Text.UTF8Encoding $true
   [System.IO.File]::WriteAllText($path, $str, $utf8_bom)
   ```

2. **Preamble + content 明示 concat** (low-level):
   ```powershell
   $enc  = [System.Text.UTF8Encoding]::new($true)
   $bom  = $enc.GetPreamble()
   $body = $enc.GetBytes($str)
   $all  = New-Object byte[] ($bom.Length + $body.Length)
   [Array]::Copy($bom,  0, $all, 0,           $bom.Length)
   [Array]::Copy($body, 0, $all, $bom.Length, $body.Length)
   [System.IO.File]::WriteAllBytes($path, $all)
   ```

本 project default は **Option C: UTF-8 no-BOM** で全 data file 統一 (commit msg / JSON / MD)、BOM が必要なのは `.ps1` script のみ (Pattern 30 refinement / L-Q3-32 scope)。

### related patterns

- Pattern 22 (harness sandbox FP): base64 encoding 経由で binary integrity 担保
- Pattern 30 (UTF-8 BOM for .ps1): BOM 必要 scope
- Pattern 30 refinement / L-Q3-32 (本 round): BOM scope 厳密化 (.ps1 only)
- Pattern 34 (本 entry): BOM 付与 API の正確な使い方

### empirical evidence

- 発見 source: anchor 24 v0.1 step 5a、commit message BOM 確認で `.GetBytes` 直叩き output に BOM byte 0 件
- 実機 verify: `[UTF8Encoding]::new($true).GetPreamble()` → `[0xEF, 0xBB, 0xBF]` (3 bytes return)、`.GetBytes("test")` → `[0x74, 0x65, 0x73, 0x74]` (4 bytes、BOM なし)
- mitigation 確立後: BOM 不要 (Option C 継続) を default、必要時のみ `WriteAllText` 経由

---

## L-Q3-32: Pattern 30 refinement — BOM scope = `.ps1` only, data files exempt

### 発見経緯

anchor 24 v0.1 step 4-5 起草中、Pattern 30 original (UTF-8 BOM prefix for PS 5.1) の適用範囲が曖昧であることが判明:
- `.ps1` script: PS 5.1 default reading で BOM 必要 (CP932 解釈防止)
- commit message / JSON / Markdown data file: BOM 含むと git 経由で `.gitattributes` の treatment や cross-tool integrity に影響可能性、本 project では BOM 不要が望ましい

両者を同じ "UTF-8 BOM" pattern として扱うと、data file にも誤って BOM が付与され git diff noise / cross-tool inconsistency を生じる。

### root cause

Pattern 30 original の wording が scope 不明瞭:

> "Pattern 30: PS 5.1 .ps1 default cp932 misread → UTF-8 BOM prefix"

"PS 5.1 default" は `.ps1` 読み込み時の動作を指していたが、後続 round の packet 起草時に "PS 5.1 で扱う全 file" と過剰一般化される drift risk。anchor 24 v0.1 step 4-5 で実際にこの drift が一時的に発生 (data file にも BOM 付与しかけた)、確認で阻止。

### impact

- data file に不要な BOM が混入すると:
  - git diff で先頭 3 bytes 変化として noise
  - JSON parser が BOM を unexpected character として reject する場合あり (一部 strict parser)
  - cross-platform / cross-tool 整合性で問題
- 一方 `.ps1` に BOM が無いと CP932 misread (Japanese path / 文字列で mojibake)、execute 不能 or 文字列 corruption

### mitigation

Pattern 30 を refine、scope を明示分離:

| file type | BOM 必要性 | 理由 |
|-----------|------------|------|
| `.ps1` (PowerShell script) | **必要** | PS 5.1 default reading が BOM-driven encoding detection |
| `.md` (Markdown) | 不要 | 本 project default = UTF-8 no-BOM、git/cross-tool integrity |
| `.json` (JSON) | 不要 | strict parser BOM-aversion 回避 |
| commit message / tag annotation | 不要 | git internal handling、Option C 継続 |
| `.tex` (LaTeX source) | 不要 | latex engine が BOM-aware だが本 project default no-BOM |
| その他 data file | 不要 | UTF-8 no-BOM default |

verification gate として packet 起草時に:
- `.ps1` packet 生成時: BOM 明示付与 (`printf '\xEF\xBB\xBF'` 等)
- data file 生成時: BOM-free 厳守 + post-write BOM byte verify

### related patterns

- Pattern 22 (base64 encoding): binary integrity (BOM 含む) 担保
- Pattern 30 original: BOM 必要性の identification
- Pattern 30 refinement / L-Q3-32 (本 entry): scope 厳密化
- Pattern 31 (autocrlf + -text directive): file content integrity の別側面
- Pattern 33 / L-Q3-29 (.gitattributes self-cover): structural fix で完全解消
- Pattern 34 / L-Q3-31 (BOM via GetBytes): BOM 付与 API の正しい使用

### empirical evidence

- 発見 source: anchor 24 v0.1 step 4-5、data file への BOM 付与誤適用の risk 認識
- mitigation 確立後: anchor 24 v0.1 完了 + 本 round で self-apply (step 2a G.6 で data file no-BOM verify、PASS)
- 本 round step 2a での実機 verify: declaration.md / input_files_pin.json / verification_log.md 3 file 全 BOM=False confirm
- ★ 本 round step 2a の partial rigor violation: claude.ai 側 distribute した packet (`anchor_25_v0_1_step_2a_packet.ps1`) が no-BOM、`.ps1` scope の本 entry が要求する BOM 付与を満たさず、Claude Code 側で `[scriptblock]::Create` workaround を要した。本 violation は本 entry codify の触媒となり、step 2b 以降の packet では BOM 必須として rigor 回復 (Pattern 38 candidate と関連、anchor 26 v0.x で codify 候補)

---

## L-Q3-33: Pattern 35 — PS 5.1 `Get-Date -Format` culture-sensitive parsing

### 発見経緯

anchor 25 v0.1 turn N+18.0 sync verify packet paste-back で、packet 内 `Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'` の出力 timestamp が `08-05-12 06:20:39 +09:00` (year `08-` 縮退) として報告された。実 system date は 2026-05-12、format string は正しく `yyyy-MM-dd` 指定にもかかわらず、年表記が 2-digit (8 = 2026 mod 100) に縮退、または culture-era 解釈で異常 transformation を経た形跡。

### root cause

PS 5.1 の `Get-Date -Format` 内部実装は `[CultureInfo]::CurrentCulture` (process-default、user culture で決まる) を使う:
- Japanese culture (`ja-JP`) 環境では `yyyy` パターンが culture-specific 解釈を受ける可能性
- 特定 wrapper / process-launch context で culture が異常 state (uninitialized / fallback) に遷移すると、format parsing が degenerate
- 結果として year `2026` が `8` (or `08`) に縮退

詳細 root cause は environment-specific (PS 5.1 内部 culture handling)、ただし mitigation で安全 path は確立可能。

### impact

- forensic record の timestamp readability に影響、audit trail での date confusion risk
- ただし gate 判定は format-independent (SHA / size 比較等は culture と無関係) なので、operational integrity は intact
- 主に "documentation / readability layer" の issue

### mitigation

`Get-Date -Format` を culture-explicit な `.ToString` 呼び出しに置き換え:

```powershell
# BEFORE (culture-sensitive、drift risk)
$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'

# AFTER (Pattern 35 mitigation、culture explicit)
$inv = [Globalization.CultureInfo]::InvariantCulture
$ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz', $inv)
```

`InvariantCulture` は culture-neutral、wrapper context / process-launch state に依存せず deterministic な出力を保証。本 project では全 packet template の timestamp record で `$inv` 経由を統一。

### related patterns

- Pattern 29 (PS vs .NET CWD divergence): wrapper-context 依存性の別側面
- Pattern 29 refinement / L-Q3-35: divergence detection の 2 階層化
- Pattern 35 (本 entry): culture-sensitive parsing の explicit override
- 3 patterns collectively cover "wrapper / process-launch context fragility" の異なる aspect

### empirical evidence

- 発見 source: anchor 25 v0.1 turn N+18.0 sync verify packet paste-back (Claude Code 側 flag)
  - 観察: `executed: 08-05-12 06:20:39 +09:00`、true system date = 2026-05-12
- 想定 root cause: PS 5.1 `Get-Date -Format` が culture-sensitive、wrapper context で `ja-JP` culture handling が degenerate state
- mitigation 確立後: anchor 25 v0.1 turn N+18.x 以降の全 packet (sync verify 第 2 弾、L-Q3-8 path verify、step 1、step 2a) で `InvariantCulture` explicit、`Get-Date` culture drift 0 件
- 実機 verify: step 2a output `executed: 2026-05-12 07:55:04 +09:00` clean (year `2026` 正常表示)

---

## L-Q3-34: Pattern 36 — forensic record 内 path reference の public repo 実存性 verify 必須

### 発見経緯

anchor 25 v0.1 round の L-Q3-8 path verify packet (gates V.1-V.7) 実行中、`axis_4_closure_summary.json` L175 が "lessons_codified.md L146, size 11,986 B, +4,518 B delta from 7,468, occurrences=4" を L-Q3-8 codify completion evidence として記述しているにもかかわらず、V.3 全 repo search で `lessons_codified.md` が 0 match (public repo 内に存在せず) と判明。

### root cause

forensic record (closure summary、declaration、appendix 等) 起草時、内部 working document の metadata を canonical evidence として記述したが、当該 file は public repo に publish されず metadata のみ残った。後続 round では当該 reference を canonical entry として参照しようとして phantom (実体なき) 状態に直面する。

### impact

- forensic chain audit trail に "実体ない参照" が混入、後続 auditor が確認不能
- canonical entry の duplicate codify or 不要な re-codify を誘発する risk
- 本 round の L-Q3-10 Interpretation B が この pattern に直接該当 (L-Q3-8 と内容重複 documentation drift 1st instance の誘因の 1 つ)

### mitigation

forensic record 起草時の必須 gate として:

1. **path 実存性 verify**: 起草中 record 内で参照する全 file path を public repo (or 当該 chain が public reference する target) 上で **同一 closure round 内** に file existence + sha256 verify
2. **phantom detection**: verify で absent detect 時、当該 reference は (a) public repo に publish、(b) 別 canonical reference に置換、(c) "internal working document、public verify 不可" と明示注記、のいずれかを採択
3. **closure gate 組み込み**: closure 直前 step (verification_log の最終 gate) で input_files_pin 内全 referenced path の public 実存 confirm
4. **本 round の self-apply**: anchor 25 v0.1 step 1 で `anchor_25_v0_1_input_files_pin.json` 内 referenced path を全 verify (★ Pattern 36 mitigation の本 round 実演)

### related patterns

- Pattern 33 (.gitattributes self-coverage gap、anchor 24 v0.1): forensic chain self-consistency の structural 側面
- Pattern 36 (本 entry): forensic chain self-consistency の reference integrity 側面
- 両 pattern とも "forensic chain は self-consistent でなければ audit 不能" という共通命題の派生

### empirical evidence

- 発見 source: anchor 25 v0.1 turn N+18.x、L-Q3-8 path verify packet V.3 全 repo search 0 match
- 影響範囲: `axis_4_closure_summary.json` L175 (10 L-Q3-8 occurrences の 1 つ、他 9 は別 context)
- impact preserved: L-Q3-8 substantive content は `empirical_precheck_canonical_rerun.json` に完全保存、catastrophic loss なし (forensic chain robustness 担保)
- 本 round self-apply 実機 verify: step 1 G.10 (L-Q3-8 canonical 3 files 実存 + SHA + size match) + G.11 (phantom path `lessons_codified.md` 不存在 confirm) で Pattern 36 mitigation 実演、両 gate PASS

---

## L-Q3-35: Pattern 29 refinement — CWD divergence detection の 2 階層化

### 発見経緯

anchor 25 v0.1 step 1 G.0 で `(Get-Location).Path -eq [System.IO.Directory]::GetCurrentDirectory()` strict parity check が FAIL、ただし G.1-G.11 全 operational gate は PASS (turn N+18.x、2026-05-12 JST)。

Claude Code wrapper は packet を `Push-Location 'E:\...\Public'; & { script-block }; Pop-Location` の form で起動。PS 5.1 design behavior により Push-Location は PSDrive provider location のみ更新、`[Environment]::CurrentDirectory` は非同期で remain at `C:\Users\sgucc` (env primary cwd)。

### root cause

Pattern 29 original の 5 rules (Push-Location 全廃、絶対 path 固定変数、.NET BCL 絶対 path 渡し、git -C $REPO_ROOT 経由、post-state predicate verify) は **packet 内部での divergence prevention** を扱うが、**packet が wrapper-context 内で起動される場合の inherited divergence** は scope 外。

native executable (git.exe 等) launch 時は PowerShell が bridging で .NET CWD を provider-relative に解決、file API も provider 経由なので、wrapper-induced divergence は packet 内部操作に影響しない。empirical 証拠: G.1-G.7 (git rev-parse, Get-FileHash) + G.8-G.11 (Test-Path, Get-ChildItem) 全 PASS、operational integrity 100% intact。

### impact

- Pattern 29 strict gate を full FAIL とすると、wrapper-induced benign divergence で round が誤停止
- 一方、divergence detection 自体は valuable (実 environment では真の不具合 indicator になる場合あり、例: PS script が `[IO.File]::WriteAllText` を relative path で叩くケース等)
- 両立必要

### mitigation: 2 階層 gate refinement

Pattern 29 strict gate を refine、下記 2 階層構造に分割:

**Tier 1 — primary gate (hard, must PASS):** PS CWD == git toplevel
- `(Get-Location).Path` と `(git rev-parse --show-toplevel 2>&1 | Out-String).Trim()` を path normalization 後比較 (Windows backslash → forward slash + 末尾 `/` 削除)
- これが PASS なら native command + file API + git operation は全て正常 path で実行
- operational integrity の真の indicator

**Tier 2 — secondary diagnostic (informational, soft FAIL OK):** PS CWD vs .NET CWD divergence
- 検出結果は report する (debugging 価値、wrapper context 把握用)
- ただし Tier 1 PASS なら round continue、Tier 2 単独 FAIL では round 停止しない
- wrapper-context divergence で誤停止しない

### related patterns

- Pattern 29 original: divergence prevention scope (packet 内部) — preserve
- 本 entry: divergence **detection** scope の厳密化、Tier 1/2 分割
- 両 pattern は並存、conflicting なし

### 本 round application + first empirical demonstration

anchor 25 v0.1 step 1 G.0 を retroactively re-evaluate:
- Tier 1 (PS CWD == git toplevel) は本 round step 1 packet には未実装、ただし G.1-G.11 全 PASS で間接確認 (git rev-parse 等が正常実行されたため)
- 本 round step 2 以降の packet では G.0 を **refined version** で実装

**first empirical demonstration**: anchor 25 v0.1 step 2a G.0 で refined logic が設計通り動作確認 (turn N+18.x、2026-05-12 07:55:04 JST):
- Tier 1: PS CWD norm `E:/GitHub repo/github_workspace/Public` == git toplevel norm `E:/GitHub repo/github_workspace/Public` → **EXACT match、PASS**
- Tier 2: PS CWD raw `E:\GitHub repo\github_workspace\Public` vs .NET CWD raw `C:\Users\sgucc` → divergent、informational only、round non-blocking
- Result: G.0 PASS based on Tier 1 only、step 1 で観察された wrapper-context-induced divergence は本 round 内で structurally resolved
- これは L-Q3-35 codify body の self-validation forensic record、Pattern 29 refinement の round-internal empirical demonstration

### empirical evidence

- 発見 source: anchor 25 v0.1 turn N+18.x、step 1 packet G.0 FAIL + G.1-G.11 PASS
- divergence の root cause: PS 5.1 Push-Location design (documented MS behavior)
- impact preserved: G.0 disposition (α+δ) で本 round continue、L-Q3-35 codify で structural refinement、future round で同 issue 再発防止
- 実機 verify: step 2a 以降の全 packet で refined G.0 PASS (Tier 1 hard gate satisfied)

★ refinement note (本 round step 3a で発見): 上記の「operational integrity 100% intact」claim は本 round step 1 packet の implementation 偶然 (PS cmdlet 専用、.NET BCL relative path 未使用) によるもので、universal claim ではない。.NET BCL relative path methods では Tier 2 divergence が catastrophic となる条件が存在する。詳細は **L-Q3-36 (Pattern 39)** を参照、本 entry の Tier 2 wording 完成版として併読のこと。本 cross-ref note は L-Q3-35 core wording を preserve しつつ追記された (option iii 採用、rule 1 IMMUTABLE spirit 尊重)。

---

## L-Q3-36: Pattern 39 — PS cmdlet vs .NET BCL CWD source divergence

### 発見経緯

anchor 25 v0.1 step 3a packet (turn N+18.x、2026-05-12 JST) 1st attempt で G.4 が FAIL: `[System.IO.File]::ReadAllBytes('.gitattributes')` が `C:\Users\sgucc\.gitattributes` を lookup → `FileNotFoundException`。2nd attempt で `[System.IO.Directory]::SetCurrentDirectory((Get-Location).Path)` を packet entry で実行後、retry で全 13 gate PASS。

これは L-Q3-35 (Pattern 29 refinement、本 round step 1 G.0 で発見) の **completion observation**: L-Q3-35 は「PS CWD == git toplevel が Tier 1、PS CWD vs .NET CWD は Tier 2 informational」と codify したが、**Tier 2 が "informational only"** という claim は **不完全**。.NET BCL relative path methods (例: `[System.IO.File]::ReadAllBytes('relative_path')`) は **process-level `[Environment]::CurrentDirectory`** を CWD source として参照、これは PS Push-Location で sync されない。

### root cause

PS 5.1 + .NET の CWD source 二重性:

| API family | CWD source | wrapper Push-Location 後の挙動 |
|------------|------------|-------------------------------|
| PS provider-aware cmdlets (`Get-Item`, `Test-Path`, `Get-FileHash`, `Set-Location`) | PSDrive location (`(Get-Location).Path`) | 正常 (PSDrive 経由で resolve) |
| .NET BCL static methods (`[System.IO.File]::*`, `[System.IO.Directory]::*` 等) | `[Environment]::CurrentDirectory` (process-level) | **divergent** (Push-Location で sync されず) |
| native executables (`git.exe`, etc.) | inherit PowerShell's resolved CWD via bridging | 通常正常 (PS が launch 時 sync) |

つまり同 PS session 内で 2 つの CWD 値が並存可能、API 選択で挙動が変わる。

### impact

L-Q3-35 codify body の core claim「operational integrity 100% intact」は本 round step 1 packet の implementation 偶然 (PS cmdlet のみ使用、.NET BCL relative path 未使用) によるもので、universal claim ではない:

- step 1 packet: PS cmdlet (Get-FileHash, Test-Path, Get-Item) のみ → divergence の影響受けず G.1-G.11 全 PASS (偶然)
- step 2a/2b packet: `[System.IO.File]::WriteAllBytes` 使用、ただし `$target` は always absolute path (`Join-Path $STAGING_ROOT $name`) → absolute path で逃げた
- step 3a packet: `[System.IO.File]::ReadAllBytes('.gitattributes')` (relative path) → divergence が即 hit、G.4 FAIL

### mitigation

3 options、本 project の standard は **(A) packet entry explicit sync** を採用 (本 round step 3a retry で empirical VERIFIED):

**(A) packet entry explicit sync** ★ 本 project standard:

```powershell
# packet 最初に追加 (G.0 前 or G.0 内に組み込み)
[System.IO.Directory]::SetCurrentDirectory((Get-Location).Path)
```

これで .NET CWD が PS CWD と同期、以降の `.NET BCL relative path` も正常動作。Tier 2 divergence が "informational" から "auto-fixed at entry" に降格。

**(B) .NET BCL は absolute path のみ使用**:

```powershell
# AVOID: [System.IO.File]::ReadAllBytes('.gitattributes')
# USE:   [System.IO.File]::ReadAllBytes((Join-Path (Get-Location).Path '.gitattributes'))
```

完全 mitigation だが、packet 起草時に regress 困難 (全 .NET BCL call で path 構築必要)。

**(C) G.0 logic 拡張で auto-sync**:

G.0 内で Tier 1 PASS 後に `SetCurrentDirectory` を silent 実行。Mitigation effective だが silent action は forensic-rigor 観点で transparency 落ちる、本 project では non-preferred。

### related patterns

- Pattern 29 original (anchor 22 v0.2): divergence prevention scope (packet 内部) — preserve
- Pattern 29 refinement / L-Q3-35 (本 round step 1 G.0 で発見): Tier 1/2 detection 分離
- Pattern 39 / L-Q3-36 (本 entry、本 round step 3a G.4 で発見): Tier 2 が .NET BCL relative path 経路で catastrophic になる条件の codify、L-Q3-35 の completion

3 entries together form complete coverage of "PS vs .NET CWD divergence" topology:

- 29 = prevention rules
- 35 = detection 2-tier
- 39 = .NET BCL relative path danger + entry sync mitigation

### empirical evidence

- 発見 source: anchor 25 v0.1 step 3a packet 1st attempt、G.4 で `[System.IO.File]::ReadAllBytes('.gitattributes')` が `C:\Users\sgucc\.gitattributes` lookup → FileNotFoundException
- mitigation verify: 2nd attempt で `[System.IO.Directory]::SetCurrentDirectory((Get-Location).Path)` 適用後、retry で全 13 gate PASS
- L-Q3-35 cross-reference update: 本 entry が Tier 2 wording の completion を提供、L-Q3-35 codify body 末尾に cross-ref note 追記 (本 round 内 update、core wording は preserve)
- step 3b 以降の全 packet で Pattern 39 mitigation (A) を必須適用 (`SetCurrentDirectory` を packet entry に組み込み)
- step 2b retry packet (本 disposition 採用後 regenerate) は本 entry を含む 8 entries 化 lessons_appendix と共に最初に Pattern 39 mitigation を実装する packet となる

---

## L-Q3-37: Pattern 40 — PS native cmd output CRLF + regex `$` anchor incompatibility

### 発見経緯

anchor 25 v0.1 step 3b packet (turn N+18.x、2026-05-12 JST) G.14 で git status output parse が **false FAIL**。Claude Code 側 independent verify で実 git state は完全 correct (5 entries: `.gitattributes` M + 4 section8 A)、ただし G.14 内 regex match が False。root cause: PS native cmd output (git status) via `Out-String` が Windows CRLF を保持、`-split "\`n"` が `\n` のみ split するため各 line 末尾に `\r` 残存、regex `$` anchor が end-of-string で match できず誤 FAIL。

operational state は intact、Pattern 31 mitigation 効果 (G.12) も successfully VERIFIED、ただし forensic documentation rigor の観点で false-negative report は重大 issue。

### root cause

PS 5.1 + native cmd output の CRLF chain:

1. `git status` output: Windows convention で CRLF line endings
2. `2>&1 | Out-String`: stdout/stderr 統合、line endings 保持 (CRLF)
3. `-split "\`n"`: `\n` (LF) のみ separator、split 前の line 末尾 `\r` は consumed されず line 内容に残存
4. 結果: 各 line = `<content>\r` (trailing CR、byte 0x0D)
5. regex `\.gitattributes$`: PS regex `$` は default で end-of-string、`\r` が line 末尾 char として存在、`$` の前が CR のため match False

empirical verify (Claude Code 側 diagnose):

- 5/5 line で line 末 byte = 0x0D (CR) 確認
- `-replace "\`r", ""` 後の regex match = True (CR strip で正常)
- counter regex `^M[ M]\s+\.gitattributes\s*$` (CR tolerate via `\s*$`) = True

### impact

- git status / git diff / git log 等の native cmd output を line-wise regex match で parse する全 packet が affected
- silent false-negative: gate が正常 state を FAIL report (本 round step 3b G.14 と同型)
- documentation forensic record の正確性に影響、operational state は intact
- 早期 detect されないと multi-step packet で累積 false-FAIL、root cause isolation 困難

### mitigation

4 options、本 project standard は **(A) regex split** を default、必要時 **(B) post-strip** defensive 適用:

**(A) regex split (`-split "\r?\`n"`)** ★ 本 project standard:

```powershell
# AVOID:  -split "\`n"      (LF only、trailing CR 残存)
# USE:    -split "\r?\`n"  (CR optional + LF、trailing CR consumed)
$lines = @($st -split "\r?\`n" | Where-Object { $_ -match '\S' })
```

**(B) post-split strip**:

```powershell
$lines = @($st -split "\`n" | ForEach-Object { $_ -replace "\`r", "" } | Where-Object { $_ -match '\S' })
```

**(C) regex `\s*$` tolerate**: 部分対応のみ

```powershell
# 個別 regex で CR tolerate、ただし split 段階の trailing CR は未解決
$line -match '^M[ M]\s+\.gitattributes\s*$'
```

**(D) `[regex]::Split` explicit**:

```powershell
$lines = [regex]::Split($st, "\r?\`n") | Where-Object { $_ -match '\S' }
```

本 project では (A) を default、(B) を defensive。(C) は regex 内のみ mitigation で incomplete (split 段階の CR 未解決)、(D) は equivalent だが verbose で readability 落ちる。

### related patterns

- Pattern 32 (PS git stderr NativeCommandError wrap、anchor 23 v0.1): native cmd output 処理の別側面
- Pattern 40 (本 entry、anchor 25 v0.1 step 3b 発見): native cmd output line endings + regex anchor の incompatibility
- 両 pattern collectively cover "native cmd output 処理" の重要 mitigation topology

### empirical evidence

- 発見 source: anchor 25 v0.1 step 3b G.14 false FAIL (turn N+18.x、2026-05-12)、Claude Code 側 diagnose で root cause isolation
- diagnose 内容: line 末 byte 確認 (0x0D)、`-replace "\`r"` 後 regex match = True、counter regex `\s*$` で CR tolerate verify
- operational impact: 0 (実 git state は正常、Pattern 31 mitigation 効果 G.12 も VERIFIED)、forensic documentation rigor の観点で false-negative report が issue
- mitigation 確立後: step 3c 以降の全 packet で `-split "\r?\`n"` を default 採用、git status/diff/log 系 parse に適用
- step 3c packet (本 disposition 採用後 regenerate) は本 entry を含む 9 entries 化 lessons_appendix と共に最初に Pattern 40 mitigation を実装する packet となる

---

## L-Q3-38: Pattern 41 — structure validator regex must account for legitimate non-data lines

### 発見経緯

anchor 25 v0.1 step 4 packet (turn N+18.x、2026-05-12 JST) G.5 で SHA256SUMS structure validation が **false-positive** を発生。validator regex `^[0-9a-f]{64}\s+\S+` が 19 lines を "malformed" として flag、Claude Code 側 independent verify で全て legitimate comment / header / section divider lines と判明。

具体的 examples:

- L1: `# SHA256 SUMS - sguccibnr32-creator/Public` (repo header comment)
- L2: `#` (single-`#` separator)
- L3: `# === companion-v0.2-round-closure-2026-05-04 (this release) ===` (section divider)
- L10: `# === backup chain in this release (v1.0.x dev history, IMMUTABLE) ===` (backup section header)

SHA256SUMS structure は anchor 22-24 closure 全 round で valid と確認済 (HTTPS raw URL audit 2-protocol 通過)。G.5 regex の comment line 非考慮が false-positive を発生。

### root cause

comment-allowing format files の structure には以下が混在:

1. **Data lines**: 主要 content (SHA256SUMS の場合 `<64 hex>  <path>`)
2. **Comment lines**: `#` で始まる行 (header / section divider / note)
3. **Blank lines**: 空行 or whitespace-only
4. **Section dividers**: `# === ... ===` style
5. **Truly malformed lines**: format violation (本 entry が target とする検出対象)

strict regex validator that only accepts (1) erroneously flags (2)/(3)/(4) as malformed。本 case では (5) の真の malformed = 0 件、19 false-positive 全て (2)(4) categories。

### impact

- structure validation gates が false-positive 発生
- operational state は intact、forensic documentation rigor 観点での issue
- L-Q3-37 (Pattern 40) と同じ "false-FAIL while actual state correct" 系の defect、ただし root cause が異なる:
  - **Pattern 40**: line ending parser (CRLF + regex `$` anchor) — *line ending classification* failure
  - **Pattern 41**: content classifier (comment vs data line) — *line type classification* failure
- 影響範囲: SHA256SUMS / .gitignore / .gitattributes / requirements.txt 等 comment-allowing format files の structure validator

### mitigation

3 options、本 project 標準は **(A) pre-filter** を採用:

**(A) pre-filter approach** ★ project standard:

```powershell
# data lines のみ抽出してから format check
$data_lines = @($ss_lines | Where-Object { $_ -notmatch '^\s*#' })
foreach ($line in $data_lines) {
    if ($line -notmatch '^[0-9a-f]{64}\s+\S+') {
        $malformed++
    }
}
```

readable、各 line type を明示的 classify、debug 時 trace 容易。

**(B) compound regex**:

```powershell
foreach ($line in $ss_lines) {
    if ($line -notmatch '^(#.*|[0-9a-f]{64}\s+\S+)$') {
        $malformed++
    }
}
```

単一 regex で compact、ただし複雑化時 readability 低下。

**(C) line classification** (defensive):

```powershell
$classification = @{ data=0; comment=0; blank=0; malformed=0 }
foreach ($line in $ss_lines) {
    if ($line -match '^\s*$')              { $classification.blank++ }
    elseif ($line -match '^\s*#')          { $classification.comment++ }
    elseif ($line -match '^[0-9a-f]{64}\s+\S+') { $classification.data++ }
    else                                    { $classification.malformed++ }
}
```

最 thorough、各 line type を count、forensic report quality 高。verbose だが diagnose 時に value。

本 project では (A) default、必要時 (C) defensive 適用。

### related patterns

- Pattern 40 (L-Q3-37、anchor 25 v0.1 step 3b 発見): native cmd output line endings + regex anchor の incompatibility — *line ending* 系 false-FAIL
- Pattern 41 (本 entry、anchor 25 v0.1 step 4 G.5 発見): structure validator が comment lines を考慮しない — *line type* 系 false-FAIL
- 両 pattern collectively cover "regex-based validator の false-positive" topology、operational impact 0 だが forensic rigor 維持に必要

### empirical evidence

- 発見 source: anchor 25 v0.1 step 4 packet G.5 false-FAIL (turn N+18.x、2026-05-12)
- Claude Code 側 diagnose: 19 "malformed" lines を line-by-line verify、全て comment/header/section divider と classify、true malformed count = 0
- operational impact: 0 (actual SHA256SUMS structure は anchor 22-24 closure 全 round で valid、HTTPS raw URL audit 2-protocol 通過)、step 4 substantive operations (G.10/G.11/G.12 等) は全 PASS
- mitigation 確立後: step 4-retry combined packet の structure validation gates で pre-filter approach (A) 実装、本 round 6th new pattern (L-Q3-38) self-apply forensic-grade reference implementation
- step 5 以降の全 packet で structure validator 使用箇所に Pattern 41 mitigation default 採用

---

## cross-reference summary

| identifier | pattern | discovery context | resolution status |
|------------|---------|-------------------|--------------------|
| L-Q3-10 | (disposition) | drift 1st instance (anchor 23 v0.1 L557) + drift 2nd instance (axis_4_closure_summary.json L175) | fully resolved (cross-ref to L-Q3-8 + L-Q3-29) |
| L-Q3-30 | Pattern 24c | anchor 24 v0.1 step 3 | codified, mitigation = `$(...)` subexpression syntax |
| L-Q3-31 | Pattern 34 | anchor 24 v0.1 step 5a | codified, mitigation = `WriteAllText` for BOM |
| L-Q3-32 | Pattern 30 refinement | anchor 24 v0.1 step 4-5 | codified, scope = `.ps1` only |
| L-Q3-33 | Pattern 35 | anchor 25 v0.1 turn N+18.0 sync verify | codified, mitigation = `InvariantCulture` explicit |
| L-Q3-34 | Pattern 36 | anchor 25 v0.1 turn N+18.x L-Q3-8 path verify | codified, mitigation = phantom path detection gate |
| L-Q3-35 | Pattern 29 refinement | anchor 25 v0.1 turn N+18.x step 1 G.0 | codified, mitigation = Tier 1/2 gate refinement (L-Q3-36 で completion) |
| L-Q3-36 | Pattern 39 | anchor 25 v0.1 turn N+18.x step 3a G.4 | codified, mitigation = SetCurrentDirectory at packet entry (本 project standard) |
| L-Q3-37 | Pattern 40 | anchor 25 v0.1 turn N+18.x step 3b G.14 false FAIL | codified, mitigation = `-split "\r?\`n"` (本 project standard) + post-strip defensive |
| L-Q3-38 | Pattern 41 | anchor 25 v0.1 turn N+18.x step 4 G.5 false FAIL | codified, mitigation = pre-filter `^\s*#` then format check (本 project standard) |

## forward defer queue (anchor 26 v0.x candidates)

- **Pattern 37** (turn N+18.x MEMORY.md verify で発見): PS 5.1 `[Console]::OutputEncoding` default (OEM code page、Windows-JP では CP932) vs UTF-8 file content display mismatch (mojibake)
  - mitigation: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` 明示 + `Get-Content -Encoding UTF8` 明示
- **Pattern 38 candidate** (turn N+18.x step 2a packet distribution で発見): PS 5.1 `.ps1` file 起動が execution policy block + no-BOM 2 重 block 状態の場合、`[scriptblock]::Create((Get-Content -Raw -Encoding UTF8))` + `& $sb` で両者同時 bypass 可能
  - 関連: 本 round L-Q3-32 self-apply rigor の一時的 violation (claude.ai 側 packet が no-BOM .ps1 distribution、Pattern 38 workaround が必要だった、本 turn 以降は BOM 付与で rigor 回復)

## round-internal self-discovery empirical record

本 round は「forensic chain self-discovery round」の名にふさわしく、round 進行中に 4 新 pattern を self-discover (+ Pattern 38 candidate を含めると 5 件) + 2 件の documentation drift を canonical recording。下記は本 round 内での pattern empirical demonstration record:

- **L-Q3-35 first round-internal demo**: anchor 25 v0.1 step 2a G.0 で refined Tier 1/2 logic が設計通り動作確認 (Tier 1 PASS、Tier 2 informational divergence non-blocking)
- **L-Q3-32 self-apply verify**: step 2a G.6 で data file 3 件全 BOM=False confirm
- **L-Q3-34 self-apply verify**: step 1 G.10 (L-Q3-8 canonical 3 files 実存) + G.11 (phantom path 不存在) で Pattern 36 mitigation 実演
- **L-Q3-33 self-apply verify**: 全 packet で `InvariantCulture` explicit、`Get-Date` culture drift 0 件 (step 2a output `2026-05-12` clean)
- **Pattern 38 candidate first encountered**: step 2a packet (claude.ai 側 .ps1 distribution、no-BOM) で execution policy + no-BOM 2 重 block 観察、`[scriptblock]::Create` workaround 確立、本 turn 以降は BOM 付与 .ps1 で rigor 回復
- **Pattern 39 / L-Q3-36 discovery + L-Q3-35 completion**: step 3a packet 1st attempt G.4 FAIL で .NET BCL relative path 経路の divergence catastrophic 性を発見、retry で `SetCurrentDirectory` mitigation empirical VERIFIED、本 round 内 codify (option iii、L-Q3-35 core wording preserve + cross-ref note 追記)
- **Pattern 40 / L-Q3-37 discovery**: step 3b packet G.14 で git status parse 誤 FAIL を発見、Claude Code 側 diagnose で root cause = PS `Out-String` の CRLF preserve + `-split "\`n"` の trailing CR 残存 + regex `$` anchor 不一致 と isolation、step 3c で v3 lessons_appendix.md regen + mitigation default 採用 (`-split "\r?\`n"`)、本 round 内 codify (option ii、self-discovery streak 5 件目維持)
- **Pattern 41 / L-Q3-38 discovery**: step 4 packet G.5 で SHA256SUMS structure validator が 19 comment lines を false-malformed として flag、Claude Code 側 diagnose で line classification (data/comment/blank/section divider) の混在問題と isolation、step 4-retry combined packet で v4 lessons_appendix regen + Pattern 41 mitigation (pre-filter `^\s*#`) default 採用、本 round 内 codify (option ii、self-discovery streak 6 件目維持)
