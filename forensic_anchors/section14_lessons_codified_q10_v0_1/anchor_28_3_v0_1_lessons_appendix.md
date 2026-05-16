# anchor 28.3 v0.1 lessons appendix

Q10 codify round (Option 2-prime architecture per、8 items full codify + character integrity 4-axis attest、Tier 1: 7 + Tier 2 meta-recursive: 1)

## §1. metadata header

date           : 2026-05-17
author         : Sakaguchi Shinobu (sole author / saka_seimensho / shisou-shi)
license        : CC-BY 4.0 (repository inherited)
round          : anchor 28.3 v0.1 (Q10 codify round)
parent anchor  : anchor 28.2 v0.1
parent HEAD    : 4ab9d0d515a29cc2451bd7014c1d6551206db2aa
parent tag     : companion-v4.9-q9-codify-round-2026-05-16 (tag obj a9b8200bdb3337655a02af0ef9deed482b240d41)
scope          : 8 items parallel codify (Tier 1: 7 + Tier 2 meta-recursive: 1)
inscription form: ## §N. {ID}: {short title} per lesson、§10 で character integrity 4-axis attest table

cluster mapping:
  cluster A (F-28.4 recovery)   : §2 F-28.4 + §3 F-28.4-A + §4 F-28.4-B + §5 F-28.4-C
  cluster B (sync-protocol)     : §6 F-28.5 + §7 F-28.6
  cluster C (PS syntax)         : §8 Pattern 24d
  Tier 2 (meta-recursive)       : §9 L-Q3-57 (17-instance full enumeration + Path B distribution)
  attest                        : §10 character integrity 4-axis table

precedent inheritance:
  anchor 28.1 v0.1 lessons_appendix.md (section12) + anchor 28.2 v0.1 lessons_appendix.md
  REVISED (section13) の inscription form を mirroring。proposal B retroactive amendment
  prohibition per、section11/12/13 read-only、本 section14 new inscription only。

## §2. F-28.4: Layer C v1.1 baseline recovery inscription

### context (1st-encounter)
1st-encounter anchor: 28.1 v0.1
encounter circumstance: anchor 28.1 v0.1 round で Layer C v1.1 baseline (SHA prefix
5d9beb04..) の locus を Public repo 内で recovery search、NOT LOCATED state
detected。search scope を Public repo 内に限定したことが root cause。

### observation
anchor 28.2 v0.1 phase 1 search で 2 path 並列実行:
- path B (claude.ai conversation_search): "Q-3 immutable hsc values draft" 等の
  keyword set で過去 chat 履歴を search、Layer C v1.1 baseline SHA prefix 5d9beb04
  at E:\Q-3_route_ii_discovery_2026-05-07\ を strong evidence として predict。
- path A (Claude Code D:\ + E:\ extended scan): 5 dirs (D:\, E:\, E:\Q-3_route_ii_
  discovery_2026-05-07\, 他 2) * 11 ext (.json, .draft.json, .csv 他) で実行、
  2,032 unique file scan、1 hit at exact candidate path、SHA 5d9beb04.. / 11,096 B
  / 300 LF / 300 CR / no LF-term / no BOM 全 axis MATCH。
- cross-attest result: path B prediction == path A measurement、bit-exact
  corroborated。F-28.4-A locus topology lesson + F-28.4-B pre-discipline encoding
  lesson が同 phase で副次的に observed。

### lesson statement
anchor-time locus topology が Public repo 外に存在する pre-discipline IMMUTABLE
baseline は、(a) recovery inscription を以て forensic chain へ再接続可能、(b)
artifact 自身は as-is preserve (後付け conversion 禁止、rule 1 IMMUTABLE 整合性
維持)、(c) out-of-repo IMMUTABLE pin sub-class taxonomy (F-28.4-C) で input_files_
pin.json 内 SHA pin track、envelope tracking NOT。

### mitigation / inscription discipline
- recovery target の 8-axis attest を input_files_pin.json 内
  f_28_4_recovery_layer_c_v1_1 block で formal inscribe (axis 1 SHA + axis 2 size
  + axis 3 LF + axis 4 CR + axis 5 LF-term + axis 6 BOM + axis 7 Pattern 46
  compliance status + axis 8 envelope tracking status)。
- recovery 時点で pre-discipline encoding を identify、artifact class を declaration
  内で明示 (F-28.4-B per、Windows CRLF, no LF-term encoding class 等)。
- F-28.4 自体は anchor 28.1 v0.1 1st-encounter → anchor 28.3 v0.1 codify、delay
  metric 2 round (hard cap boundary、L-Q3-57 distribution per #10 entry)。

### evidence
target file        : Q-3_immutable_hsc_values.draft.json
canonical location : E:\Q-3_route_ii_discovery_2026-05-07\Q-3_immutable_hsc_values.draft.json
SHA-256            : 5d9beb04361e0b21f1b703e68ba90f8cbf28efbfa0ff56c84c9d4abffb048ef3
size               : 11,096 B
LF count           : 300
CR count           : 300 (CRLF-encoded)
LF-term            : False (no trailing newline)
BOM                : False
Pattern 46         : NON-compliant (intentional preserve、F-28.4-B per)
envelope tracking  : NOT (F-28.4-C sub-class per)
role               : Phase Q-3 route ii (HSC weak lensing) axis_4 type-alpha
                     verification IMMUTABLE baseline、70 numeric + 2 categorical
                     values
inscribed date     : 2026-05-07 (Q-3 fourth-step deliverable directory、anchor 22
                     v0.2 pre-cursor)

### cross-reference
F-28.4-A (locus topology drift)、F-28.4-B (pre-discipline encoding class)、F-28.4-C
(out-of-repo IMMUTABLE pin sub-class)、L-Q3-57 instance #10 (delay 2 round)、
rule 1 IMMUTABLE (X1 / X2 precedent)、Pattern 46 (a)(b)(c) NON-compliant
intentional class。

## §3. F-28.4-A: locus topology anchor-time drift

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 phase 1 search で path B (claude.ai
conversation_search) predict と path A (Claude Code extended scan) measurement
が bit-exact corroborate した時点、anchor 28.1 v0.1 NOT LOCATED の root cause
を「search scope を Public repo 内に限定」として identify。

### observation
anchor inscription 時点の locus topology (Phase Q era top-level dirs を含む
当時の dir 配置) と、後 anchor から recovery search する時点の locus topology
は drift する。drift の発生源:
- repo 整理 (Public/ への consolidation) で外部 dirs が search default scope
  から外れる。
- anchor inscription 当時の locus topology は anchor の input_files_pin.json
  内 path 記述 + claude.ai 側 conversation history のみが保持。
- post-anchor 時点で当時の locus topology を reconstruct するには path B
  (claude.ai conversation_search) による pre-search が prerequisite。

### lesson statement
recovery-class search では path B (claude.ai conversation_search) で anchor-time
locus topology を pre-search → search scope を当時の locus topology に基づいて
広げる、を必須 step 化。Public repo 内限定 search は false negative (NOT LOCATED)
を induce。

### mitigation / inscription discipline
- recovery search protocol に "step 0: path B locus pre-search" を新設。
- path B output (e.g., "candidate path: E:\Q-3_route_ii_discovery_2026-05-07\")
  を path A scan scope に inject。
- search scope expand は anchor-time locus topology に proportional、無制限
  expansion 禁止 (false positive 防止)。

### evidence
- anchor 28.1 v0.1 recovery attempt: Public repo 内 scope、NOT LOCATED
- anchor 28.2 v0.1 phase 1 search: path B pre-search + path A scope-expanded
  scan、1/2,032 hit bit-exact corroborate
- 本 anchor 28.3 v0.1 preliminary paired sync verify S.5: F-28.4 candidate
  location stability re-verify、byte-exact preserve attest

### cross-reference
F-28.4 (recovery inscription)、F-28.4-B (pre-discipline encoding class、同 phase
副次 observation)、L-Q3-57 instance #11 (delay 1 round、strict fit)。

## §4. F-28.4-B: pre-discipline encoding class

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 phase 1 search で Layer C v1.1 baseline
の byte state (300 LF + 300 CR + no LF-term + no BOM) を measure、Pattern 46
(a)(b)(c) discipline 確立 (anchor 28.2 v0.1 round 内 codify) より前 (2026-05-07
inscription) の artifact であることを認識。

### observation
- Pattern 46 (a)(b)(c) discipline 確立より前の artifact は当時の OS / editor
  default encoding (Windows CRLF, no LF-term 等) で生成された byte state を
  保持。
- 後付け conversion (CRLF → LF + LF-term 付与) を施すと SHA pin が変化、original
  pin (e.g., 5d9beb04..) と mismatch。
- rule 1 IMMUTABLE は SHA pin が encode する original byte state の preservation
  を要求、conversion は rule 1 違反。

### lesson statement
pre-discipline encoding class artifact は as-is preserve、conversion 禁止。
encoding class を declaration 内で明示 (e.g., "Windows CRLF, no LF-term") し、
forensic chain 上で current discipline (Pattern 46) との差分を traceable に保持。

### mitigation / inscription discipline
- input_files_pin.json 内 axis 7 で "Pattern 46 compliance status: NON-compliant
  (intentional preserve、F-28.4-B per、pre-discipline artifact)" を明示 inscribe。
- artifact class 表記: "Windows CRLF, no LF-term" 等の encoding class string を
  declaration / pin entry に annotation として付与。
- 8-axis attest の axis 5 (LF-term) + axis 4 (CR count) が pre-discipline class
  の indicator として function。

### evidence
- Layer C v1.1 baseline: 300 LF + 300 CR + no LF-term + no BOM (Windows CRLF
  class)
- Pattern 46 (a)(b)(c) compliant artifact (e.g., 本 lessons_appendix.md): 0 CR
  + LF-term True + no BOM
- 両 class の byte state distinguishable、forensic chain 上で artifact class
  分類可能

### cross-reference
F-28.4 (recovery inscription、本 lesson 適用)、F-28.4-A (locus topology drift、
同 phase 副次 observation)、Pattern 46 (a)(b)(c) (positive discipline、本 lesson
は negative space)、rule 1 IMMUTABLE (preservation requirement)、L-Q3-57
instance #12 (delay 1 round、strict fit)。

## §5. F-28.4-C: out-of-repo IMMUTABLE pin sub-class

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 phase 1 search で Layer C v1.1 baseline
が pre-discipline encoding class (F-28.4-B) + anchor-time locus が Public 外
(E:\Q-3_route_ii_discovery_2026-05-07\) であることが確定、本 artifact を repo
内へ移送せず out-of-repo 状態で IMMUTABLE pin する sub-class taxonomy の必要性
を identify。

### observation
従来 IMMUTABLE pin (X1, X2 等) は repo 内 path に配置、envelope (.gitattributes
/ SHA256SUMS) で track。pre-discipline artifact を repo 内へ移送するには:
- option (i): as-is move、ただし path A workflow で Pattern 46 (a)(b)(c)
  compliance を assume する全 tooling と conflict。
- option (ii): conversion 後 move、rule 1 IMMUTABLE 違反 (F-28.4-B per)。
- option (iii): out-of-repo 配置、input_files_pin.json 内のみ SHA pin track、
  envelope tracking NOT。
option (iii) のみが rule 1 + Pattern 46 (a)(b)(c) + proposal B retroactive
amendment prohibition 全 conflict 回避可能。

### lesson statement
out-of-repo IMMUTABLE pin sub-class を新設。input_files_pin.json 内で SHA pin
+ 8-axis attest を保持、envelope (.gitattributes / SHA256SUMS) で track NOT、
artifact 本体は anchor-time locus に as-is preserve。

### mitigation / inscription discipline
- input_files_pin.json 内に sub-class taxonomy block を新設 (本 anchor 28.3 v0.1
  phase γ.1 で f_28_4_recovery_layer_c_v1_1 として初 instance inscription)。
- 8-axis attest format:
    axis 1: SHA-256
    axis 2: size (bytes)
    axis 3: LF count
    axis 4: CR count
    axis 5: LF-term (boolean)
    axis 6: BOM (boolean)
    axis 7: Pattern 46 compliance status (NON-compliant / compliant + class)
    axis 8: envelope tracking status (NOT / .gitattributes-only / .gitattributes
            + SHA256SUMS)
- envelope tracking NOT は SHA256SUMS への entry 不在を意味、L-Q3-49 closure form
  (19 ^# + N entries = N + 19 LF) に entry 数 contribution なし。

### evidence
本 F-28.4 recovery が新 sub-class 初 instance instantiate (本 anchor 28.3 v0.1
phase γ.1 で f_28_4_recovery_layer_c_v1_1 block として formal inscription)。
post-本 round では sub-class taxonomy が established となり、後 round で同 class
新規 instance 発生時の inscription pattern として function。

### cross-reference
F-28.4 (recovery inscription、本 sub-class instantiate)、F-28.4-A (locus topology
drift、out-of-repo の locus 安定性根拠)、F-28.4-B (pre-discipline encoding class、
out-of-repo 配置の必要性根拠)、rule 1 IMMUTABLE、L-Q3-49 (SHA256SUMS closure form、
本 sub-class は entry 数 contribution なし)、L-Q3-57 instance #13 (delay 1 round、
strict fit)。

## §6. F-28.5: phase-aware verify criteria refinement

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 mid-round resumption chat 内 phase γ.2
で preliminary paired sync verify を re-run、wt_clean=False (untracked file = mid-
round inscribed artifacts のみ) state を detect。従来 protocol では wt_clean=True
を strict 要求するため、本 state は FAIL と evaluate されるが、実質的には mid-
round の expected state。

### observation
preliminary paired sync verify protocol の wt_clean criteria は protocol 設計時
に round opening (clean break point) を assumed、mid-round phase γ.* での re-run
を考慮していなかった。mid-round では:
- inscribed artifacts は untracked state (まだ commit していない)
- working tree state は untracked subset == inscribed artifacts と一致
- substantive integrity は preserved (unexpected untracked entry なし)

### lesson statement
wt_clean criteria は phase-aware 化:
- mid-round phase α / β / γ.1 / γ.2 / δ: untracked subset が inscribed artifacts
  のみの場合 PASS (substantive integrity per)。unexpected entry 検出時 FAIL。
- phase ε (pre-commit): strict wt_clean=True 要求 (commit 直前は全 inscribed
  artifacts が staging に投入された clean state expected)。
- round opening / round closure: strict wt_clean=True 要求。

### mitigation / inscription discipline
- paired sync verify script S.3 step に phase-aware logic 導入:
    if (phase ∈ {mid-round α-δ}):
        wt_clean = (untracked ⊆ inscribed_artifacts) and (unexpected_count == 0)
    else:
        wt_clean = (untracked.count == 0)
- inscribed artifacts list は dispatch directive 内で explicit 指定。
- F-28.5 適用 phase の paste-back format で "F-28.5 phase-aware: untracked
  subset == inscribed only, substantive PASS" の annotation 付与。

### evidence
- anchor 28.2 v0.1 phase γ.2 re-run: wt_clean=False (untracked = lessons_
  appendix.md mid-transfer state)、F-28.5 phase-aware PASS evaluation
- 本 anchor 28.3 v0.1 phase α destination canonical verify: untracked =
  {declaration.md}、F-28.5 phase-aware substantive PASS

### cross-reference
proposal C-as-6 (parallel axis、operational protocol skeleton、本 lesson は
refinement)、L-Q3-48 (working tree state attest --untracked-files=all)、L-Q3-57
instance #14 (delay 1 round、strict fit)。

## §7. F-28.6: browser-side filename normalization protocol

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 phase γ.2 で claude.ai container
present_files dispatch → browser download → D:\ intermediate 経路で filename が
"(1)" suffix 付与 / leading-dot strip の 2 axis normalization を受ける case
detect。dispatch script が source filename を canonical name で hardcode して
いた場合 Copy-Item failure。

### observation
browser-side normalization の 2 axis:
- axis 1: leading-dot strip (.filename.md → filename.md)、browser UI で hidden
  file 扱い回避目的
- axis 2: dedup-suffix 付与 (filename.md → filename (1).md)、同名 file が D:\
  intermediate に既存の場合の dedup
両 axis は browser implementation 依存、user explicit control 不可。

### lesson statement
dispatch script は source filename adjustment + destination canonical name
restoration の 2-step 設計を採用:
- step (a): source filename candidates list で normalization variant を全 cover
- step (b): destination canonical name explicit specification で E:\ canonical
  path 配置時に正規 filename へ rename

### mitigation / inscription discipline
- dispatch directive template に "candidates tested" list 明記:
    ["filename.md", "filename (1).md", "filename (2).md", "filename (3).md"]
- script logic: Test-Path で各 candidate 検査 → 最 recent (LastWriteTime per)
  を select → destination canonical name で Copy-Item -Force
- paste-back format に "source resolution (F-28.6 normalization check)" block
  追加、resolved filename + normalization triggered/NOT triggered を attest

### evidence
- anchor 28.2 v0.1 phase γ.2 dispatch: normalization triggered case、(1) suffix
  resolved
- 本 anchor 28.3 v0.1 phase α dispatch: clean download (NOT triggered) case、
  candidates list 検査で canonical filename resolve 確認 (preventive 適用)

### cross-reference
proposal C-as-6 (parallel axis、operational protocol)、F-28.5 (phase-aware verify、
F-28.6 適用 phase の paste-back annotation との関連)、L-Q3-57 instance #15
(delay 1 round、strict fit)。

## §8. Pattern 24d: preventive ${var} delimit syntax (literal colon scope-qualifier delimit)

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 phase γ.2 で PowerShell dispatch script
内 "$dst_dir:literal" 形式の double-quoted string interpolation 箇所が parse
error 誘発。PowerShell が ":" を scope qualifier ($env:, $global:, $script:,
$private:, $local: 等) として解釈し、$dst_dir + ":literal" の連結ではなく
$dst_dir:literal という存在しない scope-qualified variable を参照しようとする
ため。

### observation
PowerShell variable interpolation の delimit rule:
- "$var" : variable boundary は次の非 identifier char で auto-terminate
- "$var:" : ":" は scope qualifier delimiter として解釈 ($scope:varname form)
- "${var}" : 明示的 boundary delimit、続く char に関わらず variable は ${...}
  内で完結
Pattern 24c (interpolation-context sibling) は "$var の auto-terminate を assume
した interpolation で literal char 直前の boundary 誤解釈" を捕捉、Pattern 24d
は specifically "$var:literal" colon scope-qualifier 誤解釈 を捕捉。

### lesson statement
double-quoted string 内で variable 直後に literal colon を配置する場合、必ず
"${var}:literal" 形式で variable boundary を explicit delimit。全 PowerShell
script で preventive 適用 (該当箇所が存在しなくても future modification 時の
regression 防止)。

### mitigation / inscription discipline
- dispatch script template + paired sync verify script の variable interpolation
  全箇所で ${var} 形式採用 (boundary 不要箇所も preventive)。
- 代表 example:
    NG : "$dst_dir:literal_path"
    OK : "${dst_dir}:literal_path"
    NG : "git log --format=$fmt:%H"
    OK : "git log --format=${fmt}:%H"
- script header comment に "Pattern 24d preventive ${var} delimit syntax adopted"
  notation 付与。

### evidence
- anchor 28.2 v0.1 phase γ.2: "$dst_dir:literal" form で parse error 観測
- anchor 28.2 v0.1 phase γ.2 mid-fix: "${dst_dir}:literal" form で resolve、
  以降の全 script で preventive 適用
- anchor 28.2 v0.1 phase ε: 9-sub-step 全 script で preventive 採用、parse
  error 0 attest
- 本 anchor 28.3 v0.1 preliminary paired sync verify: Pattern 24d notation を
  S.1 で attest ("Pattern 24d: preventive ${var} delimit syntax adopted" PASS)

### cross-reference
Pattern 24c (interpolation-context sibling、auto-terminate boundary 誤解釈)、
Pattern 24a/24b (PowerShell quoting discipline、original Pattern 24 family)、
L-Q3-52 (PowerShell ${var} delimit discipline、L-Q3 系列の precedent)、L-Q3-57
instance #16 (delay 1 round、strict fit)。

## §9. L-Q3-57: 1-round-delay pattern self-meta-codification (17-instance full enumeration、Path B distribution)

### context (1st-encounter)
1st-encounter anchor: 28.2 v0.1
encounter circumstance: anchor 28.2 v0.1 lessons_appendix.md (REVISED、section13)
内 1-round-delay observation table (line 267-281) が累積 9 instances (L-Q3-48..56)
を inscribe、anchor 28.2 v0.1 closure 時点で「1-round-delay pattern 自体を
meta-pattern として codify する」必要性が effective reveal (option b adoption
per、section13 lessons_appendix.md line 248-253)。effective reveal anchor を
post-option-b の 28.2 v0.1 closure に redefine。

### observation
1-round-delay pattern の累積 observation:
- 1st-encounter anchor (lesson の context が initially observed する anchor) と
  codify anchor (lesson が formal codify される anchor) は通常 N → N+1 の
  1-round delay。
- F-28.4 (anchor 28.1 → anchor 28.3) は 2-round delay (cap boundary)、F-28.4-A/B/C
  + F-28.5/6 + Pattern 24d (anchor 28.2 → anchor 28.3) は 1-round delay (strict
  fit)。
- 本 lesson L-Q3-57 自身も anchor 28.2 v0.1 effective reveal → anchor 28.3 v0.1
  codify の 1-round delay (self-recursive)、17th instance position は本 commit
  時点 fully determined static observation。

### lesson statement
1-round-delay pattern を Tier 2 meta-pattern として formal codify。hard cap
≤ 2 rounds (= 2-round が cap boundary、3-round 以上は protocol violation)。
本 L-Q3-57 自体が pattern の 17th instance (self-recursive)。

### 17-instance full enumeration (Path B distribution、post-option-b adoption)

```
| #  | ID                | 1st-encounter | codify   | delay metric          | scope summary                                                                          |
|----|-------------------|---------------|----------|-----------------------|----------------------------------------------------------------------------------------|
|  1 | L-Q3-48           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | working tree state attest with --untracked-files=all                                   |
|  2 | L-Q3-49           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | SHA256SUMS line-type accounting with ^# (any) pattern                                  |
|  3 | L-Q3-50           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | dispatch script Mandatory parameter coverage discipline                                |
|  4 | L-Q3-51           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | dispatch script design-intent vs invocation-phase context fit                          |
|  5 | L-Q3-52           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | PowerShell ${var} delimit discipline                                                   |
|  6 | L-Q3-53           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | git ls-remote wildcard refspec for tag pair attest                                     |
|  7 | L-Q3-54           | 28 v0.1       | 28.1 v0.1| 1 round (strict fit)  | CultureInfo equality via Equals method [DEFAULT MITIGATION]                            |
|  8 | L-Q3-55           | 28.1 v0.1     | 28.2 v0.1| 1 round (strict fit)  | cross-locus reconstruction class authoritative source single specification             |
|  9 | L-Q3-56           | 28.1 v0.1     | 28.2 v0.1| 1 round (strict fit)  | claude.ai-side projection / framing / measurement precision discipline                 |
| 10 | F-28.4            | 28.1 v0.1     | 28.3 v0.1| 2 rounds (cap boundary)| Layer C v1.1 baseline recovery、out-of-repo IMMUTABLE pin、P46 NON-compliant preserve |
| 11 | F-28.4-A          | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | locus topology anchor-time drift lesson                                                |
| 12 | F-28.4-B          | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | pre-discipline encoding class lesson                                                   |
| 13 | F-28.4-C          | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | out-of-repo IMMUTABLE pin sub-class                                                    |
| 14 | F-28.5            | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | phase-aware verify criteria refinement                                                 |
| 15 | F-28.6            | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | browser-side filename normalization protocol                                           |
| 16 | Pattern 24d       | 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | preventive ${var} delimit syntax (literal colon scope-qualifier delimit)               |
| 17 | L-Q3-57 (self-rec)| 28.2 v0.1     | 28.3 v0.1| 1 round (strict fit)  | self-recursive position、dual role (data point + meta-codifier)                        |
```

### count cross-attest (post-Path-B resolution)

prior verified (post-anchor-28.2-closure)        : 9 instances (#1-9, L-Q3-48..56)
in-round codified (anchor 28.3 v0.1 carry queue) : 7 instances (#10-16)
self-recursive (L-Q3-57)                         : 1 instance (#17, dual role per)
total instance set                               : 17

### delay metric distribution (post option b adoption、Path B per)

1-round (strict fit)         : 16 instances (#1-9, #11-16, #17)
2-round (cap boundary)       : 1 instance  (#10 F-28.4)
verified codify-delay total  : 17 instances (#1-17 全 codify-delay axis instances)
self-recursive meta-property : #17 (data point role: 17th-position recursive
                                    self-application instance; meta-codifier
                                    role: 1-round-delay pattern 自体の formal
                                    codification)
hard cap (≤ 2 rounds)        : 17/17 verified compliance、0 boundary violation

### self-recursive meta-property analysis (dual role)

#17 (L-Q3-57 自身) は 2 role を simultaneously 担う:

(role 1) data point role: codify-delay axis 上の 17th instance。1st-encounter
anchor 28.2 v0.1 (option b adoption per effective reveal) → codify anchor 28.3
v0.1 (本 round)、delay metric 1 round (strict fit)、他 16 instances と同 axis
で plot 可能。

(role 2) meta-codifier role: 1-round-delay pattern 自体を formal codify する
lesson。instance 1-16 を data points として subject 化、本 §9 が全 instance set
の codification act を実行。

self-recursive 17th instance position は本 commit 時点 fully determined static
observation:
- commit 時点で #1-#16 全 instance の delay metric が known (#10 = 2 round、他
  15 instances = 1 round)
- 本 L-Q3-57 codify が #17 として self-add される時点で、#17 自身の delay metric
  も 1 round (anchor 28.2 v0.1 effective reveal → anchor 28.3 v0.1 codify) と
  determined
- 結果として instance set cardinality 17、distribution {1-round: 16, 2-round: 1}
  が commit 時点で fully determined、後続 round で本 set は modify されない
  (proposal B retroactive amendment prohibition per)

### Pattern 31 + Pattern 45 precedent compliance argument

Pattern 31 (self-cover discipline): codification act 自体が codification 対象に
含まれる場合、self-reference の loop を 1 round 内で closure させる discipline。
本 L-Q3-57 は self-cover による 1-round closure (本 §9 で 17-instance set
inscription、self-instance position 含めて静的決定)、Pattern 31 compliant。

Pattern 45 (state-class CRV、anchor 27 v0.1 codified): commit 時点で fully
determined static observation を state-class CRV family として treat する
discipline。本 L-Q3-57 17th instance position は state-class CRV instance、
Pattern 45 precedent direct application。

### option b inheritance (section13 lessons_appendix.md:248-253)

anchor 28.2 v0.1 lessons_appendix.md (REVISED、section13) line 248-253 で option
b (effective reveal redefine) が adopted。本 lesson 内容:
- pre-option-b definition: L-Q3-57 effective reveal = 1-round-delay pattern が
  first observed として認識される最初の anchor
- post-option-b definition: L-Q3-57 effective reveal = anchor 28.2 v0.1 closure
  (本 pattern の codify 必要性が confirmed となる closure)

Path B 採用 (本 round) は section13 既 adopted option b の direct inheritance、
新 classification rule の追加不要。

### forensic trace notation

(a) memo §7 line 643 L-Q3-57 entry "delay metric: 2 rounds (cap boundary)" は
draft-state pre-option-b inheritance と認定 (pre-revision reveal = anchor 28.1
を base にすると 28.1 → 28.3 = 2 rounds、option b adoption 後の effective
reveal = anchor 28.2 への update が memo generation 時点で未反映)。post-option-b
operative metric は 1 round (strict fit)、section13 lessons_appendix.md:248-253
option b adopted consistency。

(b) memo §5 line 562 "annotated tag companion-v4.10-q10-codify-round-2026-05-1?"
は同 generation-time slip class。Path X (v4.9 typo authentication、date
"2026-05-17" pin) で round opening 時点に resolved、本 anchor 28.3 v0.1 round
の operative tag は companion-v4.9-q10-codify-round-2026-05-17。

(c) (a) + (b) 両 notation は draft state ↔ operative state の gap を forensic
trail に preserve、後 round 参照時の context restoration 可能性確保。

### cross-reference
Pattern 31 (self-cover discipline、anchor 27 v0.1 codified)、Pattern 45 (state-
class CRV、anchor 27 v0.1 codified)、section13 anchor 28.2 v0.1 lessons_
appendix.md:248-253 (option b adoption)、section13 anchor 28.2 v0.1 lessons_
appendix.md:267-281 (1-round-delay observation table、9 prior verified)、
L-Q3-48..56 (instances #1-#9)、F-28.4 + F-28.4-A/B/C + F-28.5 + F-28.6 +
Pattern 24d (instances #10-#16、§2-§8)。

## §10. character integrity 4-axis attest

本 §10 は §2-§9 で codify した 8 items の inscription block に対する character
integrity 4-axis attest table。本 lessons_appendix.md 自体の character integrity
は verification_log.md §6.2 + post-closure attestation slot で attest される。

attest axis:
  axis 1: section reference (§N)
  axis 2: item ID
  axis 3: inscription block size (approx LF count)
  axis 4: Pattern 46 compliance status (本 lessons_appendix.md 全体 inherit)

```
| §  | item ID     | inscription block LF (approx) | Pattern 46 status                          |
|----|-------------|-------------------------------|--------------------------------------------|
| §2 | F-28.4      | 50                            | inherited (本 file 全体 P46 (a)(b)(c) per) |
| §3 | F-28.4-A    | 40                            | inherited                                  |
| §4 | F-28.4-B    | 40                            | inherited                                  |
| §5 | F-28.4-C    | 50                            | inherited                                  |
| §6 | F-28.5      | 50                            | inherited                                  |
| §7 | F-28.6      | 45                            | inherited                                  |
| §8 | Pattern 24d | 55                            | inherited                                  |
| §9 | L-Q3-57     | 130 (含 17-instance table)    | inherited                                  |
```

本 lessons_appendix.md 全体 character integrity:
- Pattern 46 (a) no BOM: phase β dispatch attest で attest
- Pattern 46 (b) no CR: phase β dispatch attest で attest
- Pattern 46 (c) LF-term True: phase β dispatch attest で attest
- v4.4 spec §15 forbidden Unicode char scan: pre-dispatch で全 axis PASS attest
- character inventory: ASCII + 標準 Japanese (hiragana/katakana/kanji) + § (U+00A7)
  のみ、IPAGothic/IPAPGothic renderable

post-closure attestation:
本 lessons_appendix.md の SHA + size + LF count は phase β dispatch 時に
claude.ai-side で先行 attest、Claude Code Windows side destination canonical
verify で MATCH attest、phase ε commit 後に verification_log.md §6.2 で final
attest。

end of anchor 28.3 v0.1 lessons appendix
