# anchor 28.6 v0.1 lessons appendix

## §1. D-T: 3-class bug taxonomy + dispatch protocol design discipline

### §1.1 Taxonomy formal definition

Three orthogonal failure axes are formally codified for PowerShell dispatch script tooling, derived from anchor 28.5 round forensic ground truth (verification_log.md §6.7 reference, SHA cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9).

**axis A — parse-time failure** (PowerShell tokenizer / AST parser layer)

- failure mode: script source cannot complete tokenization or AST construction; emits ParseError before any execution begins
- representative trigger: unbalanced parenthesis, unbalanced brace, malformed string literal, syntactically invalid command invocation
- detection mechanism: pre-validate via AST parse ([System.Management.Automation.Language.Parser]::ParseInput) or scriptblock construction ([scriptblock]::Create($source))
- mitigation: paren/brace balance audit at draft time, script load pre-validation before execution dispatch, line-count sanity check at paste boundary

**axis B — parse-context failure** (PowerShell parser command-mode vs expression-mode disambiguation)

- failure mode: parser interprets cmdlet argument tokens under command-mode semantics when expression-mode was intended; operators are treated as literal arguments (or vice versa)
- representative trigger: Write-Output "text " + $var (parser sees three command-mode arguments, NOT a string concatenation expression)
- detection mechanism: output text mismatch (concatenation operator preserved literally), or $args count mismatch in helper functions
- mitigation: paren-wrap concatenation expressions to force expression-mode evaluation, example: Write-Output ("text " + $var + " more")

**axis C — runtime-semantic failure** (PowerShell function output stream collection)

- failure mode: helper function emits multiple values to pipeline (via Write-Output, implicit return, or unsuppressed expression evaluation); values accumulate into a collection; caller assignment receives an array, not the intended scalar
- representative trigger: helper function uses Write-Output for diagnostics AND return $value for actual return; caller pattern $r = func() receives 2-element Object array
- detection mechanism: variable type mismatch ($r is Object[] instead of expected scalar type), or downstream cmdlet receiving unexpected pipeline input count
- mitigation: helper function uses Write-Host for information stream (not pipeline); only return $value for pipeline emission; caller pattern $r = function_call expects single value capture

### §1.2 Bug 1/2/3 case studies (28.5 §6.7 forensic ground truth)

The 3-class taxonomy emerged inductively from 3 distinct bug occurrences during anchor 28.5 round dispatch script development:

| bug   | axis                | mitigation applied                                       |
|-------|---------------------|----------------------------------------------------------|
| bug 1 | A (parse-time)      | paren balance correction + script load pre-validate      |
| bug 2 | B (parse-context)   | Write-Output paren-wrap (expression-mode force)          |
| bug 3 | C (runtime-semantic)| Write-Host substitution + return $value isolation        |

Forensic ground truth detail and full trace: anchor 28.5 v0.1 verification_log.md §6.7 (committed SHA cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9, 44,733 B / 534 LF).

### §1.3 OL-13 status transition (candidate → nominal)

OL-13 (candidate from anchor 28.5 round, sourced from D-T augmentation) is **nominally promoted** to canonical OL entry via this inscribe.

post-inscribe OL accumulator state (OL_nominal axis):
- pre-D-T   : n=12 (OL-1 through OL-12) + OL-13 candidate
- post-D-T  : n=13 (OL-1 through OL-13 all nominal)
- D-Z α1.2  : OL-13 framework-external classification (axis taxonomy definition itself, NOT classified within L-Q3-59 sub-class cross-tab)

OL-13 nominal definition: dispatch script tooling defect 3-class axis taxonomy (parse-time / parse-context / runtime-semantic) as canonical framework for L-Q3-59 sub-class cross-tabulation, with bug 1/2/3 as representative case studies and axis A/B/C mitigation patterns formally codified per §1.1.

### §1.4 Pattern 35/39 amendment textual codification (D-X absorbed)

The following amendments to Pattern 35 (InvariantCulture explicit timestamps) and Pattern 39 (Set-Location + .NET CWD sync) are formally codified here as D-X absorption.

**D-X queue closure marking**: "D-X absorbed-in-D-T §1.4" (cross-reference, NOT nominal close; queue traceability preserved for anchor 28.7+ reference).

**Pattern 35 amendment — $ci binding scope clarification**

Apply $ci binding to ALL .ToString() calls emitting datetime, not only at obvious timestamp emit points.

- script-global declaration mandatory at start: $ci = [System.Globalization.CultureInfo]::InvariantCulture
- per-call binding mandatory at every (Get-Date).ToString(..., $ci) invocation
- per-call binding mandatory at any custom datetime format emit (e.g., [datetime]::Now.ToString("o", $ci))
- rationale: locale-dependent culture (ja-JP, en-US, etc.) emits different separators / digit groupings; cross-environment forensic reproducibility requires culture-invariant string form

**Pattern 39 amendment — cwd_sync self-check + .NET BCL coupling**

After every Set-Location, MUST immediately call .NET BCL SetCurrentDirectory and attest cwd_sync.

- step 1: Set-Location -LiteralPath $target
- step 2: [System.IO.Directory]::SetCurrentDirectory($PWD.Path)
- step 3: $cwd_sync = ($PWD.Path -eq [System.IO.Directory]::GetCurrentDirectory())
- step 4: HALT if $cwd_sync is False
- rationale: PowerShell session and .NET BCL hold separate CWD state; git CLI uses .NET BCL CWD; misalignment causes silent path resolution failure with no visible error in PowerShell session

### §1.5 3-class axis × L-Q3-59 sub-class cross-tabulation

#### §1.5.1 Framework-external notes (pre-cross-tab)

The following OL entries are NOT classified within the L-Q3-59 sub-class cross-tabulation:

- **OL-5**: L-Q3-60-adjacent (NOT L-Q3-59 sub-class instance); cross-classification is category mismatch
- **OL-13**: axis taxonomy definition itself (framework / meta-class); self-classification is circular self-reference (category error)

cross-tab population scope: OL-1 through OL-12 minus OL-5 = 11 L-Q3-59 sub-class instances.

#### §1.5.2 Canonical sub-class semantics (28.5 v0.1 inscribed, IMMUTABLE)

source: anchor 28.5 v0.1 verification_log.md §5.2.2 (SHA cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9, verbatim cite)

| sub-class | label                       | semantic                                                                       |
|-----------|------------------------------|--------------------------------------------------------------------------------|
| (a)       | path mental-model            | actual filesystem path を verify せず mental-model で fill                       |
| (b)       | architectural inheritance    | 既存 design convention を verify せず想定型で fill                              |
| (c)       | semantic resolution          | term semantic scope を verify せず over-narrow / over-broad                     |
| (d)       | orchestration framing        | workflow framing element を verify せず convention 想定 fill                     |
| (e)       | verification design          | verify check 自体の design parameter を verify せず想定 fill                     |
| (f)       | quantitative pre-comp        | quantitative value を pre-compute せず想定値で fill                              |

#### §1.5.3 Cross-tab matrix (per-cell aggregate count, 28.5 vlog §5.2.2 cumulative view preserve)

cell content: per-sub-class aggregate count from 28.5 vlog §5.2.2 cumulative-as-of-28.5-close (delta_28_6_round = 0 on L-Q3-59 axis, preserve identical).

axis distribution per cell is anchor 28.7+ formal fact-finding scope (deferred per accuracy-guarantee posture: specific OL → axis mapping は post-reclassification inference を要し本 round で byte-exact verifiable ではない、28.5 cross-artifact inscribed inconsistency 影響範囲のため).

per-sub-class aggregate (axis distribution TBD-28.7):

| sub-class | total n | classification | label                       |
|-----------|---------|----------------|------------------------------|
| (a)       | 2       | formal         | path mental-model            |
| (b)       | 3       | formal         | architectural inheritance    |
| (c)       | 1       | formal         | semantic resolution          |
| (d)       | 1       | formal         | orchestration framing        |
| (e)       | 1       | provisional    | verification design          |
| (f)       | 2       | provisional    | quantitative pre-comp        |
| **total** | **10**  | (4 formal / 2 provisional) | cumulative-as-of-28.5-close cite |

#### §1.5.4 Axis-level scaffold (rows reserved, TBD-28.7 population)

| axis                 | aggregate count | cells specific population |
|----------------------|------------------|---------------------------|
| A (parse-time)       | TBD-28.7         | TBD-28.7                  |
| B (parse-context)    | TBD-28.7         | TBD-28.7                  |
| C (runtime-semantic) | TBD-28.7         | TBD-28.7                  |
| **axis-aggregate total** | **11** (OL-1..OL-12 minus OL-5) | per axis × sub-class cell-level aggregate TBD-28.7 |

note: aggregate-level count (sub-class total = 10 + framework-external = 2 (OL-5, OL-13)) ≠ cross-tab population count (11 = OL-1..OL-12 minus OL-5)。差分 = -1 (cumulative cross-round vs this-round-12 distinction)。28.7 round fact-finding で full OL → sub-class assignment 確定後 reconcile。

#### §1.5.5 Forensic deferral rationale

per-cell specific OL identity 列挙が本 round で実施されない rationale:

- 28.5 inscribed canonical (declaration §4-2 + lessons §4-2 + verification_log §5.2.2) は cross-artifact inscribed inconsistency 含有 (本 round §6.7 inscribe 詳細参照)
- specific OL → sub-class mapping は post-reclassification view を inference で再構成する必要、accuracy-guarantee posture (inference layer 不在) に反する
- 28.7 round 内 formal fact-finding scope に defer、本 round inscribe scope LOCK preserve

## §2. D-Z: OL accumulator merge reconciliation + L-Q3-59 cross-tab population

### §2.1 HARD-GATE pre-merge protocol (R-α2 mitigation, OL-6 precedent)

pre-merge gate execution result (α1.2 dispatch script):

- **G1** : anchor 28.5 v0.1 verification_log.md SHA verify
  - expected: cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9
  - actual  : (see verification_log §4.1)
  - verdict : PASS condition for merge proceed
- **G2** : section17 α1.1 inscribed 4 artifacts SHA verify (post-R1 strict P46)
  - declaration.md       : 9a43738d027005755ab2eba1c06f19d7c2033a22d5c3bcccc5758cf92a50e139
  - lessons_appendix.md  : 48bf894691d28f7f69c91e495ac9170317a216e44301014c3fa1abad248516c1
  - input_files_pin.json : 39830719ed8bf39fbf208403a80486de6bf2e6fdebd2c00029eec7be01bd7c20
  - verification_log.md  : d27dc5a23a34e877f62c5faa30aa9c4dc0a1e40d2496352b686b787de87637b1
  - verdict : PASS condition for merge proceed
- **G3** : L-Q3-59 axis enumeration arithmetic
  - expected_cumul_28_5_close.total = 10
  - delta_28_6_round (L-Q3-59 axis) = 0
  - expected_cumul_28_6_close.total = 10 (preserve identical)
  - OL_nominal axis delta = +1 (OL-13 D-T α1.1 promotion, OL_nominal-axis separate accounting)
  - verdict : PASS condition for merge proceed

ALL 3 gates PASS → dedupe-aware merge proceed. ANY FAIL → MERGE ABORT + investigation branch.

### §2.2 Dedupe-aware merge output

post-merge canonical state (L-Q3-59 axis):

- source A : 28.5 vlog §5.2.2 cumulative-as-of-28.5-close (n=10, per-sub-class IMMUTABLE)
- source B : 28.6 round L-Q3-59 increments (n=0, this round no L-Q3-59 instance added)
- merge operation : union with semantic equivalence
- output : preserve identical to source A (since delta = 0)

post-merge canonical state (OL_nominal axis):

- source A : 28.5 cumulative OL_nominal (OL-1..OL-12, n=12)
- source B : 28.6 round OL_nominal increment (OL-13 D-T α1.1 promotion, +1)
- output : OL-1..OL-13, n=13 (framework-external: OL-5 L-Q3-60-adjacent, OL-13 axis taxonomy framework)

dedupe verification: no semantic duplicate detected (D-T axis taxonomy framework は L-Q3-59 sub-class とは distinct semantic axis、merge collision なし).

### §2.3 Sub-class refinement scope clarification (accuracy-guarantee posture)

本 round D-Z scope は cross-tab **per-cell aggregate population** only:
- sub-class aggregate count : 28.5 vlog §5.2.2 verbatim preserve (§1.5.3 inscribed)
- per-OL → sub-class specific mapping : 28.7+ round 内 formal fact-finding scope deferred

rationale (accuracy-guarantee criteria C1-C4):
- C1 byte-exact verifiability : per-OL mapping inference は byte-exact ではなく structural plausibility argument、accuracy 担保不可
- C2 canonical baseline alignment : 28.5 cross-artifact inscribed inconsistency (§6.7) に追加 inference layer 重畳 risk 回避
- C3 scope LOCK preservation : 28.5 process state reverse-engineering は本 round scope outside
- C4 forensic chain integrity : 28.7+ で reference する際の derived fact ambiguity 回避

α1.2 D-Z scope final inscribe :
- L-Q3-59 sub-class enumeration formal codify sub-task : REMOVED (28.5 vlog §5.2.2 既 formal codified、re-codification は IMMUTABLE violation risk)
- cross-tab cell content : per-sub-class aggregate count (§1.5.3) + framework-external note (§1.5.1) + axis-level scaffold (§1.5.4) + forensic deferral rationale (§1.5.5)

### §2.4 Reference to 28.5 v0.1 closure cross-artifact inscribed inconsistency

D-Z fact-finding (Code-side scope, α1.2 dispatch preparation) で anchor 28.5 v0.1 closure に **cross-artifact inscribed inconsistency** を発見:

- declaration.md §4-2 + lessons_appendix.md §4-2 : mutually consistent (28.4-origin attribution view)
- verification_log.md §5.2.2 : cumulative-as-of-28.5-close view (OL-3 → (e) provisional, OL-4 → (f) provisional annotations reflect 28.5-round-new instances)
- per-sub-class divergence : (b)+1 vlog / (d)-1 vlog / (e)-1 vlog / (f)+1 vlog (net 0, total preserved 10) + (c)/(e) classification (formal ↔ provisional) swap

詳細 inscribe : verification_log §6.7 (OL-15 candidate flag for anchor 28.7 deferred queue, cross-artifact inscribed inconsistency class)。

本 round D-Z は (A) vlog §5.2.2 を canonical source として採用 (semantic view alignment with G3 cumulative arithmetic + user-explicit source naming + 28.5-round-new annotations reflect)、inconsistency awareness 含む forensic-grade decision。

## §3. references

- parent anchor: 28.5 v0.1 (HEAD 203ac68016bc4979ed11103918c153fa298f3166)
- parent verification_log §5.2.2: L-Q3-59 sub-class taxonomy refinement codified entry (SHA cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9, cumulative as-of-28.5-close, IMMUTABLE)
- parent verification_log §6.7: bug 1/2/3 forensic ground truth (same artifact, SHA cdd8cce6c2959ce74e49f022c4de91454c27ec307ae3786844797956165938d9)
- OL-6 precedent: text claim enumeration count pre-merge cross-verify (28.5 inscribed)
- L-Q3-60 axis (ii) tool independence : Code-side / claude.ai-side cross-attest convention (28.5 inscribed)
- 4-layer audit structure (本 round emergence): L-Q3-60 axes (i)/(ii)/(iii)/(iv) + user-layer audit (4th independent layer, OL-14 candidate scope inclusion candidate)
