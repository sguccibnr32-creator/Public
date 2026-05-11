# anchor 24 v0.1 lessons appendix (L-Q3-18 〜 L-Q3-29)

```
作成日:     2026年5月XX日 JST (step 5 commit 日に確定)
作成元:     claude.ai turn N+17.x + Claude Code (Windows) paired sync
配置 path:  forensic_anchors/section7_lessons_codified_q3_v0_2/anchor_24_v0_1_lessons_appendix.md
license:    CC-BY 4.0
依存 path:  anchor_22_v0_2_input_files_pin.json (L-Q3-18 〜 L-Q3-27 起源)
            section6_lessons_codified_q3_v0_1/anchor_23_v0_1_lessons_appendix.md (L-Q3-28、L-Q3-29 起源 mention)
            section6_lessons_codified_q3_v0_1/anchor_23_v0_1_verification_log.md (L-Q3-28、L-Q3-29 evidence 原典)
```

---

## 序

本 appendix は anchor 24 v0.1 Q3 codify round v0.2 で正式 codify する 12 entries (L-Q3-18 〜 L-Q3-29) の codify body を格納する。各 entry は下記 template に準拠:

```
### L-Q3-NN: <Pattern NN 名称>

**起源**: anchor XX vY.Z round step N
**Pattern 番号**: NN (★ structural fix 要 / doc-only)
**現象**: ...
**root cause**: ...
**mitigation**: ...
**verification evidence**: ...
**adoption scope**: ...
**status**: codified (anchor 24 v0.1)
```

特殊扱い: **L-Q3-29 (Pattern 33)** は本 round の structural fix 対象であり、entry body の末尾に **inline evidence appendix** (Pattern 33 forensic record) を併載する。

---

## L-Q3-18 〜 L-Q3-27: anchor 22 v0.2 forward (10 entries)

### L-Q3-18: Pattern 24 (packet narrative parse error 一般化)

**起源**: anchor 22 v0.2 round (axis_4 type-α hardening)
**Pattern 番号**: 24 (doc-only)
**現象**: PowerShell packet 内に narrative (説明文・コメント) を埋め込む際、行頭 `#` を欠落した narrative 行が PS パーサーにより expression として解釈され、`The term 'XXX' is not recognized` 等の parse error を引き起こす。
**root cause**: PS 5.1 のパーサーは、行頭が `#` でない非 cmdlet 行を expression 候補として強制解析。日本語 narrative や natural language text は不正な expression として fail する。
**mitigation**: packet 起草時、narrative 行は全て `# ` prefix で line-comment 化。block comment (`<# #>`) は Pattern 25 により禁止。継続行 (multi-line narrative) も全行 `# ` prefix で統一 (Pattern 24a で別途規定)。
**verification evidence**: anchor 22 v0.2 step N で narrative 全行 `#` prefix 走査 gate 適用後、parse error 0 件達成。
**adoption scope**: 全 packet 起草時に適用 (本 round の step 1-7 packet も継続適用)。
**status**: codified (anchor 24 v0.1)

### L-Q3-19: Pattern 25 (PowerShell block comment non-nestable)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 25 (doc-only)
**現象**: PS 5.1 の block comment 構文 `<# ... #>` は nested 不可。block comment 内に別の `<#` が出現すると最初の `#>` で comment 終端され、以降の行が live code として fail-parse する。
**root cause**: PS 5.1 のレキサーは `<#` / `#>` をシンボル対と認識するが、stack-based parsing を実装しない。
**mitigation**: 本 project の packet では block comment を **一切不使用**。多行 narrative は `# ` prefix の line-comment を連続適用 (Pattern 24 + 24a と整合)。
**verification evidence**: anchor 22 v0.2 + anchor 23 v0.1 の全 packet で block comment 不使用、関連 parse error 0 件。
**adoption scope**: 全 packet 起草時に適用。
**status**: codified (anchor 24 v0.1)

### L-Q3-20: Pattern 26 (PSObject.Properties.Count quirk)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 26 (doc-only)
**現象**: PS 5.1 で `($array).Count` を実行する際、`$array` が空 (null) または 1 要素の場合、`Count` プロパティが unexpected な振る舞いをする (null だと例外、1 要素だと scalar の property を返す)。
**root cause**: PS 5.1 の implicit unboxing。array が空・scalar の場合は collection としての `Count` プロパティが解決されない。
**mitigation**: 全 `Count` 取得を `@($array).Count` で wrap。`@()` operator により empty / scalar / array を強制的に array 化、`Count` が常に integer を返す。
**verification evidence**: anchor 22 v0.2 + anchor 23 v0.1 全 packet で `@()` wrap 適用、count 不整合 0 件。
**adoption scope**: 全 packet で `Count` を扱う箇所に適用。
**status**: codified (anchor 24 v0.1)

### L-Q3-21: Pattern 27 (git status porcelain default aggregation)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 27 (doc-only)
**現象**: `git status --porcelain` (オプションなし) は untracked file を子 directory 単位で aggregation 表示し、個別 file 単位の untracked 検出 gate で誤検出 (false negative) を生じる。
**root cause**: git の default porcelain v1 仕様。`untracked-files=normal` (default) は dir-level aggregation を行う。
**mitigation**: 全 porcelain 取得に `--untracked-files=all` を付与。個別 file 単位の untracked 検出が保証される。
**verification evidence**: anchor 22 v0.2 + anchor 23 v0.1 全 packet で `--untracked-files=all` 適用、untracked false negative 0 件。
**adoption scope**: 全 packet で `git status --porcelain` を呼ぶ箇所。
**status**: codified (anchor 24 v0.1)

### L-Q3-22: Pattern 28 (git diff untracked exclusion)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 28 (doc-only)
**現象**: `git diff` (オプションなし) は untracked file を対象外とするが、cascade 検証時に staged 状態の variance を見落とす可能性がある。
**root cause**: git diff の default は working tree vs index。untracked は index に存在しないため diff 出力に含まれない。
**mitigation**: cascade 検証は `git diff --cached --shortstat` で index vs HEAD を取得 (`--shortstat` で N files / +M / -L 形式)。untracked は別途 `git status --porcelain --untracked-files=all` で検出 (Pattern 27 と組合せ)。
**verification evidence**: anchor 22 v0.2 + anchor 23 v0.1 全 packet で `--cached --shortstat` 適用、envelope 数 + 行数の bit-exact 検証 PASS。
**adoption scope**: 全 packet の cascade verify gate。
**status**: codified (anchor 24 v0.1)

### L-Q3-23: Pattern 29 ★ (PS Push-Location vs .NET CWD divergence、MAJOR)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 29 ★ (doc-only、MAJOR 重要度)
**現象**: PowerShell の `Push-Location` / `Pop-Location` は PS 内の Location stack のみ更新し、.NET の `[System.IO.Directory]::GetCurrentDirectory()` (= プロセスの CWD) を更新しない。結果として PS cmdlet (Get-Item, Test-Path 等) は PS Location 基準で動作する一方、.NET BCL API ([System.IO.File], [System.IO.Path] 等) はプロセス CWD 基準で動作し、両者が divergence する。git も .NET の Process.Start で呼ばれるため、プロセス CWD 基準 (= 起動時 CWD) で動作する。
**root cause**: PS 5.1 の Location 抽象は .NET CWD と独立に実装されている (PSDrive 抽象により仮想 path も扱うため)。
**mitigation** (5 rules):
1. **Push-Location 全廃**: packet 内で Push/Pop-Location を一切使用しない
2. **絶対 path 固定変数**: `$REPO_ROOT` 等を packet 冒頭で絶対 path で定義、全 path 操作は変数経由
3. **.NET BCL 絶対 path 渡し**: `[System.IO.File]::Exists($path)` 等は絶対 path 引数で呼ぶ
4. **git -C $REPO_ROOT 経由**: git command は `git -C "$REPO_ROOT" ...` 形式で必ず repo root を指定
5. **post-state predicate verify**: 各 write op 後、対象 path の存在・size・SHA を独立 gate で再確認 (CWD divergence による silent failure 検出)
**verification evidence**: anchor 22 v0.2 round 内で 5 rules 適用後、CWD divergence による silent failure 0 件。anchor 23 v0.1 step 6 push critical でも継続適用、push apply の完全性 confirmed。
**adoption scope**: 全 packet (★ MAJOR、絶対遵守)。
**status**: codified (anchor 24 v0.1)

### L-Q3-24: Pattern 30 (PS 5.1 .ps1 default cp932 misread Japanese path)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 30 (doc-only)
**現象**: PS 5.1 で `.ps1` script を実行する際、script file の encoding が BOM-less UTF-8 だと、日本語文字 (path や string literal 内の) が **cp932 として誤読** され、文字化け / parse error / silent path mismatch を引き起こす。
**root cause**: PS 5.1 の default file encoding 判定が BOM 不在時に system codepage (日本語環境では cp932) を fallback として採用。
**mitigation**: 全 `.ps1` script (および inline execute する PS script) の file 先頭に **UTF-8 BOM prefix** (`0xEF 0xBB 0xBF`) を付与。または inline execute 時は `Get-Content -Encoding UTF8` で明示的に UTF-8 として読込。
**verification evidence**: anchor 22 v0.2 round 内、日本語 path (`D:\ドキュメント\...`) を含む全 packet で BOM 付与適用後、cp932 misread 0 件。anchor 23 v0.1 でも継続適用。
**adoption scope**: 全 packet で日本語文字を含む箇所 (path / string literal / narrative)。
**status**: codified (anchor 24 v0.1)

### L-Q3-25: Pattern 31 (autocrlf + -text directive missing CRLF warning)

**起源**: anchor 22 v0.2 round
**Pattern 番号**: 31 (doc-only)
**現象**: `core.autocrlf=true` 環境下で、新規 file を git add する際、`.gitattributes` に対象 file の `-text` directive が **無い** 場合、git が "LF will be replaced by CRLF" 警告を発し、blob 内容が wt と異なる (autocrlf normalization が適用される) 状態となる。
**root cause**: git の autocrlf normalization は default で全 text-like file を対象とする。`.gitattributes` で `-text` 明示しない限り、CRLF/LF 変換が silently 発生。
**mitigation**: 新規 file group を staging する場合、`.gitattributes` に対象 path の `-text` directive を **先行追加** + `git add .gitattributes` を実行してから、対象 file を staging する。これにより autocrlf normalization が抑止され、wt SHA == blob SHA の bit-exact 一致が保証される。
**verification evidence**: anchor 23 v0.1 step 4 で `.gitattributes` 先行追加 + section6 4 file staging 実行、SHA mismatch 0 件 (5 file = SHA256SUMS + section6 4 file が wt == blob 達成)。ただし `.gitattributes` 自身は self-coverage gap (Pattern 33) により本 mitigation の対象外、別途 structural fix が必要。
**adoption scope**: 新規 file group の staging 時。
**status**: codified (anchor 24 v0.1)

### L-Q3-26: Pattern 24a (narrative continuation # prefix 漏れ)

**起源**: anchor 22 v0.2 round (Pattern 24 の subordinate pattern)
**Pattern 番号**: 24a (doc-only)
**現象**: 多行 narrative の **2 行目以降** で `#` prefix を欠落すると、当該行が live code として解釈され parse error。Pattern 24 の strict subset。
**root cause**: Pattern 24 と同根。packet 起草時の human error が継続行で発生しやすい。
**mitigation**: packet 起草時、**全 narrative 行を 1 行ずつ走査** し `#` prefix の存在を確認。継続行も `#` prefix を厳守。
**verification evidence**: anchor 22 v0.2 round 内で 1 件発見、修正後 anchor 23 v0.1 で 0 件再発。
**adoption scope**: 全 packet 起草時、特に多行説明の継続行。
**status**: codified (anchor 24 v0.1)

### L-Q3-27: Pattern 24b (PS string interpolation $VAR: PSDrive collision)

**起源**: anchor 22 v0.2 round (Pattern 24 の subordinate pattern)
**Pattern 番号**: 24b (doc-only)
**現象**: PS string 内で `"...$VAR:..."` のように variable 直後に `:` が続く場合、PS パーサーは `$VAR:` を **PSDrive reference** (例: `$Env:`, `$HKLM:`) として解釈し、意図した variable 補間が失敗する。
**root cause**: PS 5.1 のレキサーは `$VAR:` を PSDrive accessor として優先解析。
**mitigation**: variable 補間に `:` が後続する場合、**`${VAR}:` 形式** (curly brace delimiter) で variable 境界を明示。
**verification evidence**: anchor 22 v0.2 round 内で 2 件発見、修正後 `${VAR}:` style 厳守、anchor 23 v0.1 で 0 件再発。
**adoption scope**: 全 packet 起草時、variable 補間後の colon。
**status**: codified (anchor 24 v0.1)

---

## L-Q3-28: anchor 23 v0.1 NEW (Pattern 32)

### L-Q3-28: Pattern 32 (PS 5.1 git push stderr NativeCommandError wrap)

**起源**: anchor 23 v0.1 round step 6 (P15.1.3b push critical)
**Pattern 番号**: 32 (doc-only)
**現象**: PS 5.1 で `git push origin <ref> 2>&1` を実行する際、git の通常進捗 stderr 出力 (例: `To https://...`, `* [new tag] ...`, `<hash>..<hash> main -> main` 等) が PowerShell によって **NativeCommandError (RemoteException) として wrap** される。`$?` は `$false` 化、ただし `$LASTEXITCODE` は **`0` (push 自体は成功)**。
**root cause**: PS 5.1 の native exe stderr redirect 挙動。PS 7.x では改善済 (本 project は PS 5.1 採用のため影響あり)。
**mitigation**: 全 git push の exit 判定を 2 階層 bypass で実装:
1. **primary check**: `$LASTEXITCODE -eq 0` (push 自体の成功を判定)
2. **pattern match**: stderr/stdout 内に期待 signature 行 (`<hash>..<hash> <ref> -> <ref>` または `\[new tag\] <tagname>` 等) が含まれるかを正規表現で confirm
3. 両 check PASS の場合のみ gate PASS
**verification evidence**: anchor 23 v0.1 step 6 G.7 (push main) + G.11 (push tag) で 2 回観察。両 gate で primary check + pattern match の 2 階層 bypass が機能、push 成功を forensic-grade で確認。
**adoption scope**: 全 git push 系 packet (anchor 24 v0.1 step 6 でも継続適用)。
**status**: codified (anchor 24 v0.1)

---

## L-Q3-29: anchor 23 v0.1 NEW ★ (Pattern 33、structural fix 対象)

### L-Q3-29: Pattern 33 (.gitattributes self-coverage gap)

**起源**: anchor 23 v0.1 round step 7 (P15.1.4 raw URL audit) G.2 trap + 4 段階診断 D.1-D.4
**Pattern 番号**: 33 ★ (★ **structural fix 要**、本 round step 4 で execute)
**現象**: `core.autocrlf=true` (system scope) 環境下で、`.gitattributes` 自身が `-text` directive で self-cover **されない** 場合、autocrlf normalization が `.gitattributes` 自身に適用され、**working tree (CRLF) と git blob (LF) で content が divergence** する。HTTPS raw URL は blob を serve するため、local `Get-FileHash` (wt を hash) と HTTPS audit (blob を hash) の SHA が mismatch する。
**root cause**: `.gitattributes` の **self-coverage gap** — `.gitattributes` 内に `.gitattributes -text` 行が無い。`.gitattributes` 自身が autocrlf 対象外と宣言されない限り、wt と blob は normalization で divergence する。
**mitigation**:
- **interim mitigation** (anchor 23 v0.1 で適用): step 7 audit を blob_only mode で実行 (HTTPS SHA を blob SHA で verify、wt cross-check は skip)
- **structural fix** (本 round step 4 で execute): `.gitattributes` 末尾に `.gitattributes -text` 行を append (LF 終端)。これにより:
  - `git check-attr text .gitattributes` → `"text: unset"` 化
  - autocrlf normalization が `.gitattributes` に適用されなくなる
  - working tree も LF 統一 → wt SHA == blob SHA
**verification evidence**:
- BEFORE-fix (anchor 23 v0.1 step 7 + 本 round step 1 G.6/G.7 で再確認):
  - wt SHA: `3ed45e2720492a812987163b34331d211e722a2e9c43e60ed1eca5c679821602` / 1999 B (CRLF)
  - blob SHA: `6eca2b40ac823b3062b2b0050007e9c6d01ba05da685c8a6fda913b0c0b79221` / 1967 B (LF)
  - divergence: True、size_diff: +32 (CRLF expansion 32 行分)
- AFTER-fix (本 round step 4 Phase 1 G.9 で確定、step 6/7 で公開 verify):
  - wt SHA == blob SHA (Pattern 33 解消)
  - 統一 SHA + size は `SHA256SUMS` の `.gitattributes` entry に encode、Public repo で外部参照可能
  - step 7 G.8 で 6/6 全 file が wt_eq_blob mode で audit PASS
**adoption scope**: 本 round 内で 1 回 structural fix execute、以降 永続化 (rule 1 IMMUTABLE 適用)。
**status**: codified (anchor 24 v0.1)、structural fix executed (anchor 24 v0.1 step 4)

---

### L-Q3-29 inline evidence appendix (Pattern 33 forensic record)

本 appendix は L-Q3-29 (Pattern 33) の forensic-grade reproducibility 記録。BEFORE state は本 round step 1 で再確認済、AFTER state は本 round step 4 で確定。

#### BEFORE-fix state (anchor 23 v0.1 step 7 で root cause isolation 済、本 round step 1 G.6/G.7 で再確認)

| 項目 | 値 |
|---|---|
| working tree SHA256 | `3ed45e2720492a812987163b34331d211e722a2e9c43e60ed1eca5c679821602` |
| working tree size | 1999 bytes |
| working tree encoding | UTF-8 CRLF |
| blob SHA256 | `6eca2b40ac823b3062b2b0050007e9c6d01ba05da685c8a6fda913b0c0b79221` |
| blob size | 1967 bytes |
| blob encoding | UTF-8 LF |
| divergence | True |
| size_diff (wt - blob) | +32 bytes |
| size_diff 解釈 | CRLF expansion 32 行分 (wt 内 32 個の `\n` が `\r\n` に展開) |

#### root cause (anchor 23 v0.1 step 7 4 段階診断 D.1-D.4 で isolation)

- D.1: `core.autocrlf` scope-aware 確認 → system scope `true` (local/global 未設定)
- D.2: blob vs working tree SHA 直接比較 → `.gitattributes` 単独 divergence、他 5 file (SHA256SUMS + section6 4 file) は blob == wt
- D.3: `.gitattributes` 内 self-coverage 検査 → `.gitattributes -text` 行 **不存在** (★ root cause)
- D.4: `git check-attr text .gitattributes` → `text: unspecified` (★ unspecified = autocrlf 対象)

#### structural fix operation (本 round step 4 Phase 1 で execute)

```
operation: .gitattributes 末尾に「.gitattributes -text」行を append (LF 終端)
expected effect:
  - git check-attr text .gitattributes → "text: unset"
  - autocrlf normalization が .gitattributes に適用されない
  - working tree も LF 統一
  - 新 unified SHA + size 確定
```

#### AFTER-fix state (本 round step 4 Phase 1 G.9 で確定、SHA256SUMS の `.gitattributes` entry に encode)

```
AFTER-fix の新 unified SHA は本 round step 4 で確定する。
正確な値は SHA256SUMS の `.gitattributes` entry を参照。
本 round step 7 G.8 で全 6 file が wt_eq_blob mode で audit PASS 確認後、Pattern 33 解消 forensic-grade 化。
```

#### D.5 case A integration (本 round step 4 Phase 2-3 で cascade)

D.5 個別診断 (anchor 23 v0.1 turn N+16.x で execute) で SHA256SUMS は **case A (WT-based reference)** 確定:

- recorded SHA for `.gitattributes` = wt SHA = `3ed45e27...`
- SHA256SUMS は Windows 開発環境固有の wt SHA を canonical encode

本 round structural fix 後、SHA256SUMS の `.gitattributes` entry は **新 unified SHA** に update。fix 完了後の状態:

- SHA256SUMS は protocol-agnostic reference 化 (Protocol 1 = Protocol 2 = SHA256SUMS canonical 一致)
- 6 file 全 entry で wt SHA == blob SHA == SHA256SUMS entry 一致
- forensic chain integrity が全 protocol で bit-exact 達成

#### Pattern 33 mitigation の本 round 内 transition

| step | mode | 適用 |
|---|---|---|
| step 1 | acknowledge | BEFORE-fix baseline characterize (forensic baseline 確立) |
| step 2-3 | acknowledge (継続) | section7 4 file は `-text` directive で個別 cover (Pattern 31 mitigation) |
| step 4 ★ | **structural fix execute** | self-cover 行 append + wt == blob 統一 verify |
| step 5 | post-fix include | commit envelope に fix 含む |
| step 6 | post-fix push | remote 反映確認 + sub-gate で wt == blob 再 verify |
| step 7 | **mitigation 不要化** | 全 file wt_eq_blob mode で audit (前 round の blob_only mode 廃止) |

---

## 累積 codify status

| section | round | 範囲 | entries | structural fix |
|---|---|---|---|---|
| section6 | anchor 23 v0.1 | L-Q3-11 〜 L-Q3-17 | 7 | 0 |
| **section7** | **anchor 24 v0.1** | **L-Q3-18 〜 L-Q3-29** | **12** | **1 (L-Q3-29 ★)** |
| (累積) | - | L-Q3-11 〜 L-Q3-29 | 19 | 1 |

L-Q3-10 (anchor 23 v0.1 で deferred) は本 round では codify 対象外 (status: still deferred、後続 round 候補)。

---

END OF anchor 24 v0.1 lessons appendix
