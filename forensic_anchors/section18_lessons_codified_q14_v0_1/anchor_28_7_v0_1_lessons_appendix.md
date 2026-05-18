# anchor 28.7 v0.1 lessons appendix

generation TS  : 2026-05-18T<late-evening>+09:00
author         : Sakaguchi Shinobu (坂口忍) / 坂口製麺所 / 宍粟市 (Shisō City, Hyogo)
license        : CC-BY 4.0
parent         : anchor 28.6 v0.1 (HEAD 2ca2c6d4eaf6d7ec3ad2c4b772e0dd9105dee6ed)
artifact_class : lessons_appendix
phase          : Stage 3 v0.1 draft (claude.ai-side emit, pre-attest)
encoding       : UTF-8 no BOM, LF only, trailing LF mandatory (Pattern 31)

primary codify scope:
  - OL-14 nominal codify (agent-agnostic baseline reference miss class)
  - Pattern 47 inline codify (SHA / cryptographic hex equality ordinal discipline)
  - 4-layer audit structure formal codify (D-1/D-2/D-3/D-4 with P3 hybrid)
  - Phase 3 mitigation patterns M1-M5 codify

axis arithmetic (Stage 5 G3 HARD-GATE expected):
  OL_nominal     : 13 -> 14 (+1, OL-14)
  Pattern axis   : 46 -> 47 (+1, Pattern 47)
  L-Q3-59        : 10 -> 10 (delta = 0, preserve identical)
  audit layer    : 4 (D-1/D-2/D-3/D-4 NEW)
  M-axis         : 5 (M1-M5 NEW)
  forensic chain : 13 -> 14 (Stage 5 atomic commit 時)

================================================================================
§1. OL-14 canonical definition v0.2
================================================================================

§1.1 core statement

OL-14 (Operational Lapse #14) — agent (claude.ai-side / Code-side いずれか) が
action を initiate / claim / commit する際、design-time pre-action reconciliation
を欠いた結果、agent の action と既存 canonical baseline 間に detection-layer-
distributed observability を持つ divergence が発生する operational lapse の
class。

§1.2 4 baseline-types (cause-end coverage)

  (i)   inscribed nominal OL
  (ii)  previously codified definition (Pattern / Rule / 概念定義)
  (iii) forensic-anchor-attested enumeration
  (iv)  established pipeline state

§1.3 external 6 evidence base (Phase 1 definition derivation canonical)

  §6.1 atomicity claim ambiguity     (claude.ai, B-(iv), C-1, D-2, lineage: OL-11)
  §6.2 OL-6 precedent reference miss (claude.ai, B-(i),  C-2, D-2)
  §6.4 P46 definitional drift        (claude.ai, B-(ii), C-3, D-2)
  §6.5 build pipeline simulation gap (claude.ai, B-(iv), C-4, D-2)
  §6.6 L-Q3-59 enumeration shortfall (Code,      B-(iii), C-5, D-4)
  anchor 28.1 §4 Layer C re-attest deferred
                                     (Code,      B-(i)/(iii), C-2/C-5, D-1b, pre-28.5)

axis D distribution: D-1b x1 / D-2 x4 / D-4 x1
  (D-2 dominance 4/6, L-Q3-60 dual-channel discipline operational central value 実証)

§1.4 structural generality 4 axes

  axis A (agent layer)             : A-1 (claude.ai) / A-2 (Code)
  axis B (missed baseline type)    : B-(i)/(ii)/(iii)/(iv)
  axis C (manifestation trigger)   : C-1/C-2/C-3/C-4/C-5
  axis D (detection layer)         : D-1/D-2/D-3/D-4

詳細 axis content は §2 で full inscribe。

§1.5 relation summary

  L-Q3-59 subset OL-14   (R1: proper subset, manifestation outcome-bearing; §6 で詳述)
  OL-6    subset OL-14   (baseline-type (i))
  OL-11   adjacent OL-14 (lineage adjacency, NOT missed baseline type, separation D2)
  OL-13   orthogonal OL-14
  L-Q3-60 structural-dual OL-14 (positive form / negative form pair)
  Pattern 47 orthogonal OL-14    (cause-end class orthogonal verification-end discipline)

================================================================================
§2. Phase 2 manifestation axis taxonomy (locked)
================================================================================

§2.1 axis A (agent layer)

  A-1: claude.ai-side
  A-2: Code-side

§2.2 axis B (missed baseline type, cause-end)

  B-(i)  : inscribed nominal OL
  B-(ii) : previously codified definition (Pattern / Rule / 概念定義)
  B-(iii): forensic-anchor-attested enumeration
  B-(iv) : established pipeline state

§2.3 axis C (manifestation trigger pattern)

  C-1: unilateral claim       - pre-reconciliation 欠落の claim emission
  C-2: precedent omission     - 既 codified precedent への reference miss
  C-3: definitional drift     - 定義 inscribed scope からの implicit deviation
  C-4: simulation gap         - pipeline state 観測無しでの action emission
  C-5: enumeration shortfall  - forensic attestation 列挙の incomplete cite

§2.4 axis D (detection layer)

  D-1: self-correction (agent-internal detection)
    D-1a: same-turn
    D-1b: cross-round same-agent retro-detection
    channel: in-context reasoning audit

  D-2: cross-attest (cross-tool / cross-execution-path)
    channel: independent compute path verify
    mapping: L-Q3-60 axis (i) + (ii)
    NOTE   : 本 layer は L-Q3-60 dual-channel discipline operational central
             value、external 6 instances で 4/6 dominance、round-internal 7
             instances で 5/7 dominance

  D-3: paired sync (cross-agent / cross-round)
    channel: handoff package SHA pin + post-closure paired verify
    mapping: L-Q3-60 axis (iii) + (iv)
    discipline: Pattern 47 ordinal SHA equality MANDATORY (§5 参照)

  D-4: user clarification (last-resort meta-detection layer, P3 hybrid)
    channel    : user-initiated only
    structural : framework 内 position だが framework 不可 invoke
    functional : D-1/D-2/D-3 全 escape を catch、agent 4-axis matrix と
                 independent な 4th 独立 attest layer
    evidence   : 28.6 vlog §6.6 + §6.7 + 28.7 §B.1.1 (instance #3 U+00AD injection)
    mitigation : M5 (D-4 invocation context preserve、§3.5 参照)

================================================================================
§3. Phase 3 mitigation patterns M1-M5 (NEW codify)
================================================================================

§3.1 M1 - baseline reference protocol

  scope     : phase opening 時に required baselines を explicit cite
  per-type  : B-(i)/(ii)/(iii)/(iv) coverage で baseline cite list emit
  applicable: axis B 全 type

§3.2 M2 - pre-action verify discipline

  scope     : commit / inscribe / claim 前 baseline-citing check step 義務化
  timing    : design-time 履行 (D3 narrowing 反映)
  applicable: axis C 全 trigger pattern

§3.3 M3 - continuous re-anchor

  scope     : long context (>N turns or >M phases) で periodic baseline
              re-citation 義務化
  applicable: axis A/C cross-cutting
  refinement candidate (anchor 28.8+ deferred queue):
    short-cycle (<1-turn) 3-tier discipline
      tier 1: long-context periodic re-anchor
      tier 2: just-codified immediate re-anchor (last 1-2 turns scope)
      tier 3: same-action iterative verify (intra-turn discipline)
    evidence : instance #5 + #7 triangulation

§3.4 M4 - cross-agent baseline-share

  scope      : claude.ai-side / Code-side 間で baseline reference state を
               sync gate に embed
  applicable : axis A (cross-agent) + axis D-3 (paired sync)
  embed point: handoff package (claude_ai_handoff_memo + claude_code_sync_memo)
               SHA pins + design canonical state 両 memo 100% consistent

§3.5 M5 - D-4 invocation context preserve

  scope     : user clarification request 時 context lock-in protocol
  applicable: axis D-4 specific
  sub-rules :
    P-pre1: user clarification を「routine」ではなく「potential D-4 invocation」
            として handle identification protocol
    P-pre2: D-4 invocation 時点での context state を user request response に
            explicit cite (lock-in)
    P-pre3: D-4 invocation 後の resolution は agent layer (D-1/D-2/D-3) で
            independent verify、user judgment dependency 不可
    P-pre4: D-4 invocation で surface した divergence は verification_log §6
            inscribe 義務化

================================================================================
§4. Phase 4 4-layer audit structure formal codify (NEW inscribe)
================================================================================

§4.1 4-piece integrated framework

  L-Q3-60                : positive form / cross-source detection topology
                           (28.5 vlog §5.2.2 既 codified, preserve identical)
  L-Q3-59                : negative form / mental-model fill manifestation
                           (28.5 vlog §5.2.2 既 codified, preserve identical;
                            OL-14 manifestation subset R1)
  OL-14                  : cross-cutting cause-end class
                           (28.7 NEW nominal codify, 本 §1)
  4-layer audit structure: detection topology D-1/D-2/D-3/D-4 with P3 hybrid
                           (28.7 NEW inscribe via OL-14 axis D + M5)

§4.2 P3 hybrid rationale (D-4 = framework-internal but framework-cannot-invoke)

  evidence E1-E4 (forensic ground truth):

    E1: 28.6 vlog §6.6 lesson
        "cross-agent attest は L-Q3-60 axis (ii) tool independence +
         (i)/(iii)/(iv) と独立に user-layer audit が 4th 独立 attest layer
         として機能"
    E2: 28.6 lessons_appendix §X
        "4-layer audit structure: L-Q3-60 axes (i)/(ii)/(iii)/(iv) +
         user-layer audit (4th independent layer, OL-14 candidate scope
         inclusion candidate)"
    E3: 28.6 vlog §6.6
        "cross-detection: user clarification request が catch"
    E4: 28.6 vlog §6.7
        "OL-15 candidate (cross-artifact inscribed inconsistency) も user
         clarification dependency context で surface"

§4.3 detection layer cumulative cross-cutting view

  D-1 (agent-internal)    : reactive / self-bound, low cost, recursive risk あり
  D-2 (cross-attest)      : proactive / cross-path, operational central, L-Q3-60 主軸
  D-3 (paired sync)       : checkpoint-bound, cross-agent integrity, Pattern 47 適用
  D-4 (user clarification): unscheduled / last-resort, agent-independent, P3 hybrid

  layer ordering principle: D-1 -> D-2 -> D-3 -> D-4 escalation
                            (low-cost first, framework-internal last-resort fall-through)

================================================================================
§5. Pattern 47 inline codify NEW (cross-cutting tooling discipline)
================================================================================

§5.1 nominal label + scope

  Pattern 47 - SHA / cryptographic hex string equality ordinal discipline
  scope    : PowerShell dispatch / verify script の SHA / fingerprint / hex pin
             equality verification 全般

§5.2 rule (MANDATORY)

  [String]::Equals($a, $b, [System.StringComparison]::Ordinal)
    is MANDATORY for any hex / SHA / fingerprint equality check。

§5.3 forbid (PROHIBITED)

  PowerShell `-eq` (string operator) is PROHIBITED for hex/SHA equality。

§5.4 rationale

  .NET CompareInfo.Compare culture-aware semantics treats Unicode
  Default_Ignorable_Code_Point (U+00AD soft hyphen, U+200B ZWSP, 等) as
  equivalent to absence -> silent false-positive on length-different strings
  differing only in ignorable codepoints。

§5.5 evidence + emergence cascade

  evidence : anchor 28.7 round Stage 1 §B H2-refined verdict (Code-side inline
             check)
  root cause cascade documented:
    anchor 28.5 MEMORY.md -> 28.6 handoff -> sync memo §2.5 display ->
    sync memo §3.1 script $u9_expected -> U.9 silent PASS
  emergence trigger:
    instance #3 (U+00AD injection D-4 detection) + instance #4 (-eq semantic
    D-1b detection) dual-trigger

§5.6 axis classification

  Pattern axis (cross-cutting tooling discipline)
  Pattern 31/35/39/46 同 abstraction layer
  cumul: Pattern axis 46 (anchor 28.5 latest Pattern 46) -> 47 (28.7 NEW)

§5.7 relation: OL-14 orthogonal Pattern 47

  cause-end class (OL-14) orthogonal verification-end discipline (Pattern 47)
  synergistic effect: Pattern 47 が D-2 cross-attest channel integrity を保護
                      -> D-2 instance density 4/6 の operational trust 強化

§5.8 paired sync script applicability

  anchor 28.7 round 以降の dispatch / verify script は全 SHA equality verify
  point で Pattern 47 適用 MANDATORY。post-applicable retro-fix scope は
  anchor 28.8 round deferred queue HIGH (paired sync U.9 broader transcription
  audit) で処理。

================================================================================
§6. OL-14 superset L-Q3-59 relation R1 articulation
================================================================================

§6.1 causal chain

  [OL-14 cause]                            [L-Q3-59 outcome]
  baseline reference miss            ->    mental-model fill manifestation
  (any of baseline-types                   (one of 6 sub-classes (a)-(f),
   (i)/(ii)/(iii)/(iv))                     28.5 vlog §5.2.2)

§6.2 L-Q3-59 subset OL-14 manifestation set (proper subset)

  L-Q3-59 は OL-14 manifestation の fill-outcome-bearing subset、但し OL-14
  manifestation 全体は L-Q3-59 では cover されない。

  counter-example:
    §6.1 (28.6) atomicity claim ambiguity:
      reference miss = OL-14 instance
      outcome        = ambiguity claim emission (NOT mental-model fill)
      -> L-Q3-59 6 sub-class いずれにも該当しない、OL-14 instance だが
         L-Q3-59 instance ではない

§6.3 rejected alternatives

  R2 (orthogonal) : reject - abstraction layer 不一致、L-Q3-59 は OL-14
                    manifestation 経路で実体化される、causal chain 不可 ignore
  R3 (equivalent) : reject - manifestation coverage 不一致、§6.2 反例 1 件で
                    equivalence 不成立

§6.4 L-Q3-59 preserve identical statement

  cumul     : 10 -> 10 (delta = 0)
  scope     : 28.5 vlog §5.2.2 codified content 不変
  rationale : 28.7 round で L-Q3-59 sub-class 拡張または再分類 不実施、
              OL-14 codify を通じた structural recontextualize のみ実施

================================================================================
§7. manifestation lineage vs missed baseline type 分離規律 (D2 adopt)
================================================================================

§7.1 規律 statement

  各 OL-14 instance 記述では 2 dimension を必ず separate inscribe:

    missed baseline (cause)   : baseline-type (i)/(ii)/(iii)/(iv) + specific
                                reference (どの inscribed entity を miss したか)
    manifestation lineage     : OL-11 / OL-6 / Mod-N / Finding-N 等
    (outcome)                   (どの既存 entity に lineage 接続するか)

§7.2 conflation 防止 (anti-pattern)

  禁止 anti-pattern:
    "OL-11 instance (= reference miss)" のような lineage-cause conflation 記述
    -> missed baseline と manifestation lineage を同一視、OL-14 cause-end
      class 性質を obscure する

§7.3 inscribe template

  推奨記述形式:
    instance-N: <short label>
      missed baseline      : B-(x) - <specific inscribed entity reference>
      manifestation lineage: <OL/Mod/Finding entity, if any>
      detection layer      : D-x
      axis                 : A-x / B-(x) / C-x / D-x

================================================================================
§8. round-internal OL-14 instances (7 total, locked at v0.1 closure)
================================================================================

§8.1 enumeration table

  #  | instance                          | A   | B            | C       | D    | detect channel
  ---|-----------------------------------|-----|--------------|---------|------|--------------------
  1  | D-axis regression                 | A-1 | B-(iii)      | C-2     | D-2  | Stage 1 review
  2  | 6 vs 7 typo                       | A-1 | B-(ii)       | C-3     | D-2  | Stage 1 review
  3  | U+00AD injection                  | A-1 | B-(iii)      | C-3     | D-4  | Stage 1 review
  4  | paired sync -eq semantic gap      | A-2 | B-(ii)       | C-4     | D-1b | prior chat inline
  5  | instance #4 D-axis misattribution | A-1 | B-(ii)       | C-3     | D-2  | prior chat (§C)
  6  | X1 path memory inheritance        | A-1 | B-(i)        | C-2/C-5 | D-2  | Stage 1 review (D3 FAIL)
  7  | length estimation gap             | A-1 | B-(i)/B-(ii) | C-4     | D-2  | tool validation

§8.2 axis distribution summary (v0.1 snapshot, 7 instances)

  axis A : A-1 x6 + A-2 x1 = 7
  axis B : B-(i) x1 + B-(ii) x4 + B-(iii) x2 = 7+ (dual count for #7: B-(i)/B-(ii))
  axis D : D-1b x1 + D-2 x5 + D-4 x1 = 7
  D-2 dominance: 5/7 (71.4%) - L-Q3-60 dual-channel discipline operational
                              central value 実証

§8.3 recursion bounding (C1/C2/C3 preserved)

  recursion chain: #1 -> ... -> #5 (bounded at #5)
  bounding logic:
    C1: #5 baseline reference miss resolved (axis D rectification confirmed)
    C2: rectification = M3 execution, framework-internal, no new miss
    C3: subsequent action references codified baseline cleanly
  #6 + #7 : independent observations outside recursion chain (no chain extension)

§8.4 individual instance inscribe (per §7 template, abbreviated)

  instance #1: D-axis regression
    missed baseline      : B-(iii) - 28.6 vlog forensic-anchor-attested axis D
                                     classification (initial D-1 x3 / D-2 x2 / D-4 x1)
    manifestation lineage: D-axis regression (Stage 1 review §A)
    detection layer      : D-2 (cross-attest with section17 vlog L131-142)
    axis                 : A-1 / B-(iii) / C-2 / D-2

  instance #2: 6 vs 7 typo
    missed baseline      : B-(ii) - external instances count = 6 (codified in §1.3)
    manifestation lineage: nominal typo (7 instead of 6)
    detection layer      : D-2 (cross-tool count verify)
    axis                 : A-1 / B-(ii) / C-3 / D-2

  instance #3: U+00AD injection
    missed baseline      : B-(iii) - inscribed SHA value pure ASCII 64-hex norm
                                     (forensic-anchor-attested encoding pattern)
    manifestation lineage: F-α non-ASCII injection class adjacent (canonical
                           hex pattern violation)
    detection layer      : D-4 (user clarification request surfaced)
    axis                 : A-1 / B-(iii) / C-3 / D-4

  instance #4: paired sync -eq semantic gap
    missed baseline      : B-(ii) - .NET CompareInfo culture-aware semantics
                                    pre-codified knowledge (Pattern 47 emergence
                                    source)
    manifestation lineage: U.9 silent PASS (verification escape)
    detection layer      : D-1b (Code-side cross-round retro-detection)
    axis                 : A-2 / B-(ii) / C-4 / D-1b

  instance #5: instance #4 D-axis misattribution
    missed baseline      : B-(ii) - D-1a (same-turn) vs D-1b (cross-round)
                                    sub-property distinction (Phase 2 axis D
                                    definition)
    manifestation lineage: instance #4 axis D inscribed as D-1 instead of D-1b
    detection layer      : D-2 (this turn §C proposal, cross-attest with
                                detection timing verify)
    axis                 : A-1 / B-(ii) / C-3 / D-2

  instance #6: X1 path memory inheritance error
    missed baseline      : B-(i) + B-(iii) dual-source:
                           B-(i) inscribed nominal X1 path (section5_axis_4_type_alpha)
                           B-(iii) F-27.4 linear mapping forensic-anchor-attested
                           (section_N = anchor_N - 17, anchor 23+; pre-linear for 22 v0.2)
    manifestation lineage: section3_lessons_codified_q1_v0_2 naming-inference
                           fallback error class
    detection layer      : D-2 (paired sync D3 FAIL detect via X1 byte-exact mismatch)
    axis                 : A-1 / B-(i)/B-(iii) / C-2/C-5 / D-2

  instance #7: length estimation gap (memory replace)
    missed baseline      : B-(i) + B-(ii) dual-source:
                           B-(i) tool input length constraint inscribed
                           B-(ii) F-α LF counting / size projection precedent
                           (claude.ai-side measurement discipline)
    manifestation lineage: memory replace operation length overflow (tool
                           validation feedback)
    detection layer      : D-2 (tool validation channel = independent compute path)
    axis                 : A-1 / B-(i)/B-(ii) / C-4 / D-2

================================================================================
§9. axis D classification rectification record
================================================================================

§9.1 external 6 instances initial vs corrected

  initial v0.1 classification (Stage 1 prep):
    D-1 x3 / D-2 x2 / D-4 x1

  corrected (Stage 1 review §A, instance #1):
    D-1b x1 / D-2 x4 / D-4 x1

§9.2 rectification rationale

  section17 vlog L131-142 verbatim cite scope:
    L-Q3-60 axis (i) tool independence と (ii) cross-execution-path は
    operational central detection channel、agent-internal self-correction
    (D-1) ではなく cross-attest (D-2) として inscribe すべき。
    D-1 適用は self-correction の "agent-internal" 性質を要求、cross-tool
    verify は D-2 scope。

§9.3 round-internal 7 instances axis D rectification (instance #5)

  initial classification (prior turn 前):
    instance #4 axis D inscribed as D-1
  rectified (prior chat §C proposal):
    instance #4 axis D = D-1b (cross-round same-agent retro-detection)
    instance #5 = #4 D-axis misattribution itself、axis D = D-2

  rationale:
    D-1a (same-turn) は instance #4 detection timing (cross-round) と合致せず
    D-1b (cross-round same-agent retro-detection) が canonical fit

================================================================================
§10. forward principle / deferred queue carry
================================================================================

§10.1 Option C boundary discipline (forward principle, anchor 28.7 round で codify)

  scope :
    declaration                            = Stage 1 closure 時 freeze
    vlog / input_files_pin / lessons_appendix = Stage 4 v0.1 emit 時 freeze

  exception trigger (3 condition, OR):
    (a) round-internal instance count > 8 -> recursion bounding re-evaluation
    (b) Pattern 47 級 new operational discipline emergence
    (c) Phase 1-4 design state lock 解除相当 change

  exception 発動時の handling:
    Stage 1 reopen 不要、path 1 recovery (re-emit + new canonical) permit、
    cascade update scope は declaration SHA reference + provenance +
    abandonment record の 3 dimension に限定

  本 round v0.1 closure 時点 status: exception trigger 不発動 (count <= 7,
    Pattern 47 emergence は 本 round codify scope 内、Phase 1-4 lock preserve)

§10.2 deferred queue (priority sorted, v0.1 snapshot)

  HIGH:
    - paired sync U.9 broader transcription audit
      scope: past-round MEMORY.md / handoff memo SHA pin transcription audit
      source: Pattern 47 emergence cascade (anchor 28.5 MEMORY.md ->
              28.6 handoff -> sync memo §2.5 display -> sync memo §3.1
              script $u9_expected -> U.9 silent PASS)

  MEDIUM_HIGH:
    - M3 short-cycle (<1-turn) refinement
      scope: 3-tier discipline design (tier 1 long-context periodic +
             tier 2 just-codified immediate + tier 3 same-action iterative)
      evidence: instance #5 + #7 triangulation

    - OL-16 candidate cluster formal codify
      scope: claude.ai-side measurement/estimation discipline class
      members: F-α LF counting / size projection gap / length estimation gap /
               non-ASCII char injection

  MEDIUM:
    - OL-15 candidate formal codify (anchor 28.6 §6.7 single-instance)
    - D-W / D-V / D-U (anchor 28.5 carry)
    - Non-ASCII char injection canonical hex pattern (OL-16 / F-α adjacent)

  LOW:
    - D-Y (vlog dual-clause single-clause refactoring, 28.5 carry)
    - per-OL -> L-Q3-59 sub-class fact-finding (28.6 §2.3 deferral)
    - 2-instance paste-back-without-instruction pattern (D-4 adjacent)
    - Pattern axis canonical accumulator file
      (forensic_anchors/pattern_axis_state.json)
    - broader MEMORY.md / handoff X1 path audit (28.8 round opening 時履行)

§10.3 resolved in 28.7 round (carry list 不要)

  - 28.6 §B Option C boundary discipline candidate -> 本 §10.1 forward principle
    化で resolved
  - L-Q3-59 sub-class fact-finding は per-OL mapping に subsume、本 round
    independent 履行 不実施

================================================================================
§11. Stage 5 G3 HARD-GATE axis arithmetic recap
================================================================================

§11.1 expected axis arithmetic (Stage 5 atomic commit 直前 verify scope)

  OL_nominal     : 13 -> 14 (+1, OL-14 nominal codify)
  Pattern axis   : 46 -> 47 (+1, Pattern 47 inline codify)
  L-Q3-59        : 10 -> 10 (delta = 0, preserve identical)
  audit layer    : 4 (D-1/D-2/D-3/D-4 NEW formal codify)
  M-axis         : 5 (M1-M5 NEW codify)
  forensic chain : 13 -> 14 (Stage 5 atomic commit 時)

§11.2 G3 5 sub-gate hardcoded constants (dispatch script v0.1 reference)

  $g3_ol_target       = 14
  $g3_pattern_target  = 47
  $g3_lq359_delta     = 0
  $g3_audit_target    = 4
  $g3_m_target        = 5

§11.3 G2 4-tuple distinct verdict expectation

  Stage 5 で 4 paired artifacts (declaration / lessons_appendix /
  input_files_pin / verification_log) の SHA-256 全 distinct を verify、
  $sha_set.Count == 4 確認後 G3 5 sub-gate に proceed。

================================================================================
§12. END
================================================================================

本 lessons_appendix.md v0.1 は anchor 28.7 round codified knowledge の
canonical inscribe artifact。Stage 5 atomic commit で section18 directory に
file-ize、Q14 annotated tag (companion-v4.9-q14-codify-round-<DATE>) 付与、
rule 92 strict push で origin/main + Q14 tag 同期予定。

post-v0.1 update scope (out of preserve identical):
  本 v0.1 freeze 後 detect された instance #8/#9/#10 + Pattern 48 candidate
  emergence + memo (6).txt §1.1 revocation は verification_log.md v0.2 §10.4 +
  §10.5 NEW inscribe で記録、本 lessons_appendix は v0.1 preserve identical
  (28.8 round opening package で Pattern 48 candidate を proper codify として
   28.8 lessons_appendix に inscribe 予定、scope-separation discipline 整合)。
