# anchor 28.5 v0.x declaration

## §1. round identity

| axis | value |
|---|---|
| document | declaration for anchor 28.5 v0.x codify round (Q12) |
| round id | anchor 28.5 v0.x codify round |
| Q tag | companion-v4.9-q12-codify-round-2026-05-XX (XX = 新 chat closure 実行日、TBD) |
| parent anchor | anchor 28.4 v0.1 (HEAD 22c556b8b38cca731b0c197db5683c80f3c06f28) |
| parent Q tag | companion-v4.9-q11-codify-round-2026-05-17 (obj 2e686db25e5e3b2b3a0842307f888558e13a66ae) |
| codify date | 2026-05-XX (新 chat closure 実行日 反映、TBD) |
| author | Sakaguchi Shinobu (宍粟市) / 坂口製麺所 |
| license | CC-BY 4.0 |

## §2. scope LOCK

| priority | item | scope decision |
|---|---|---|
| P0-A | L-Q3-60 codify: dual-channel verification discipline | adopt (本 round primary innovation) |
| P0-B | L-Q3-59 sub-class taxonomy refinement (n=10 → 6 sub-class) | adopt (paired codify、L-Q3-60 violation negative form) |
| P1 | packet 10/11 inference error observation note | defer to anchor 28.6 (candidate space class、n=1、threshold 未達) |
| P2 | rule 1 IMMUTABLE pin path 明記 convention | lessons_appendix §6 inline note (軽量、formal L 番号無付与) |

## §3. L-Q3-60 canonical pin

### 3-1. canonical identity

| axis | value |
|---|---|
| ID | L-Q3-60 |
| class | discipline (Pattern → Lesson 昇格、Pattern 31/41/44 系統拡張) |
| canonical name | dual-channel verification discipline |
| codify round | anchor 28.5 v0.x (本 round) |

### 3-2. core principle (LOCKED text)

> 単一 verification channel (script output のみ / memo paste-back のみ / artifact 内 self-attestation のみ) は verification subject 自体への mental-model fill (期待値投影) を構造的に許容する。verification integrity は independent dual channel (channel A + channel B、source 系統独立) の cross-confirm により担保される。

### 3-3. 4 independence axis (orthogonal)

1. **execution path independent**: 例 — git wire protocol SHA + filesystem cat-file SHA
2. **data lineage independent**: 例 — script direct stdout + HTTPS raw URL retrieval
3. **temporal independent**: 例 — round closure 時 attest + post-round paired sync verify
4. **agent independent**: 例 — claude.ai-side draft SHA + Claude Code-side actual file SHA

### 3-4. 3 reference instances (LOCKED)

| # | instance | channel A | channel B | axis | round |
|---|---|---|---|---|---|
| ref-1 | triple-channel reproducibility | Protocol 1 git wire | Protocol 1' cat-file + Protocol 2 HTTPS raw URL | (i)+(ii) | 27 v0.1 |
| ref-2 | post-closure paired sync verify | claude.ai-side handoff memo SHA pins | Claude Code-side S.1-S.7 actual verify | (iii)+(iv) | 28.4 v0.1 |
| ref-3 | F-28.4-C out-of-repo IMMUTABLE pin | input_files_pin.json inscribe | filesystem actual SHA + size + LWT | (i)+(ii) | 28.4 v0.1 |

### 3-5. existing discipline relationship

L-Q3-60 は Pattern 31 / Pattern 41 / Pattern 44 系統の Lesson 昇格 + 拡張。先行 Pattern が individual verification act の discipline であるのに対し、L-Q3-60 は verification の dual-channel structural requirement を formal principle として固定。本 round mid-chat handoff 採用 (structural signature 5-2) は L-Q3-60 自己適用の初 instance。

## §4. L-Q3-59 sub-class taxonomy pin

### 4-1. refinement context

anchor 28.4 v0.1 closure 時 L-Q3-59 (verification subject mental-model fill) は n=8 で formal codified、single-class aggregation。本 round では taxonomy refinement で 6 sub-class に分割、additional instance discovery 含め total n=10。

### 4-2. 6 sub-class (LOCKED)

| ID | name | root mechanism | n | flag |
|---|---|---|---|---|
| L-Q3-59(a) | path mental-model | actual filesystem path を verify せず mental-model で fill | 2 | formal |
| L-Q3-59(b) | architectural inheritance | 既存 design convention を verify せず想定型で fill | 2 | formal |
| L-Q3-59(c) | semantic resolution | term semantic scope を verify せず over-narrow / over-broad | 1 | provisional |
| L-Q3-59(d) | orchestration framing | workflow framing element を verify せず convention 想定 fill | 2 | formal |
| L-Q3-59(e) | verification design | verify check 自体の design parameter を verify せず想定 fill | 2 | formal |
| L-Q3-59(f) | quantitative pre-comp | quantitative value を pre-compute せず想定値で fill | 1 | provisional |

### 4-3. threshold convention (本 round で formal establish)

- 原則 threshold: n ≥ 3
- 本 round application: (a)(b)(d)(e) を n=2 ながら parallel completeness exception で formal 採用
  (4 sub-class が same-round 同時 emerge の structural symmetry を根拠)
- (c)(f) は n=1 で provisional flag 付与
- provisional flag 解除条件: 次回以降 round で同 sub-class instance 追加 confirm 時

### 4-4. total instance count

**total instance count**: 10 (formal classified 8 = (a)(b)(d)(e) × 2 each + provisional 2 = (c)(f) × 1 each)
(anchor 28.4 v0.1 closure memory "n=8 threshold breach" = formal classified subset、provisional 2 は taxonomy refinement 時 additional discovery)

## §5. structural signature (本 round 3 components)

1. **paired codify innovation**: L-Q3-60 (discipline positive form) + L-Q3-59 sub-class taxonomy (violation negative form) の同 round 同時 codify、初 instance
2. **meta-self-reference avoidance**: L-Q3-60 codify round 自体が L-Q3-60 violation で finalize する self-undermining risk → mid-chat handoff 採用で構造的回避 (本 round 内 self-instantiating discipline application、初 instance)
3. **threshold convention formal establishment**: sub-class formal 命名 n≥3 原則 + n=2 parallel completeness exception + n=1 provisional flag convention の formal inscribe

## §6. forensic continuity invariant

### 6-1. parent chain (本 round closure 時 12-deep)

| depth | SHA | round |
|---|---|---|
| 1 | TBD (新 chat closure HEAD、phase η commit 後 fix) | anchor 28.5 v0.1 (本 round) |
| 2 | 22c556b8b38cca731b0c197db5683c80f3c06f28 | anchor 28.4 v0.1 |
| 3 | 2de39308978befeb223c7356a691ab37fc83d559 | anchor 28.3 v0.1 |
| 4 | 4ab9d0d515a29cc2451bd7014c1d6551206db2aa | anchor 28.2 v0.1 (Q9) |
| 5 | cc35c0983e8b8baaafa5d83689d69a31f880e38f | anchor 28.1 v0.1 (Q8) |
| 6 | cf834ea49ea5cc5657ea8601c05f44f4464ba740 | anchor 28 v0.1 (Q7) |
| 7 | 0fe208e0937764617932727e88967b7ac083e1da | anchor 27 v0.1 (Q6) |
| 8 | d0e5d2e1940fbd516fdcb0a1ffb06be736c66d29 | anchor 26 v0.1 (Q5) |
| 9 | d3920ca4458ed788af90f542aabaf248077ce707 | anchor 25 v0.1 (Q4) |
| 10 | cbc270041c7627b95e90399dc8a9eaee4f3cc8e1 | anchor 24 v0.1 (Q3 v0.2) |
| 11 | 3aef5142167f993f2ba8a6f67d9b925c1252cc4b | anchor 23 v0.1 (Q3 v0.1) |
| 12 | 491ff34cce22040e052f226e64adddc1669ea1b4 | anchor 22 v0.2 |

### 6-2. rule 1 IMMUTABLE preservation pins (cross-round invariant)

```
X1 SHA : 435bf4b68a48e251e7a591564ee870b94072f9573292579ddac0ead6f7eff2be
X1 path: forensic_anchors\section5_axis_4_type_alpha\anchor_22_v0_2_input_files_pin.json
X2 SHA : d43985b896b63e625718bd6d5d644abbfe6b2cee99721290d0165a39c212e5dd
X2 path: latex_v48\membrane_v48.tex
```

X1 path 明記 convention: 本 round より、X1 SHA に対し IMMUTABLE preservation source path を declaration §6-2 内で明示 inscribe (lessons_appendix §6 inline note per scope P2)。anchor 28.4 v0.1 以前は SHA のみ inscribe、本 round 以降 path 併記 convention 確立。

### 6-3. F-rule cumulative

F-rule cumulative: **11** (anchor 28.4 v0.1 closure 時点と同値、本 round で F-rule 新規追加なし)

### 6-4. prior tag preservation pins (rule 92 strict、本 round closure 時 6 tag)

```
Q12 (companion-v4.9-q12-codify-round-2026-05-XX): TBD (本 round 新規、phase η tag obj 後 fix)
Q11 (companion-v4.9-q11-codify-round-2026-05-17): 2e686db25e5e3b2b3a0842307f888558e13a66ae
Q10 (companion-v4.9-q10-codify-round-2026-05-17): dd91c8861c14ef5c2436cf3e0c73a36c1ac666a1
Q9  (companion-v4.9-q9-codify-round-2026-05-16) : a9b8200bdb3337655a02af0ef9deed482b240d41
Q8  (companion-v4.9-q8-codify-round-2026-05-15) : a873e8785c55d17fafa56d06320a4daea27ffb28
Q7  (companion-v4.9-q7-codify-round-2026-05-15) : 0fc3df9eb2d42c81e04e84a79d1b3e0f79773986
```

rule 92 strict push enforced (no --force/--all/--tags/--mirror) throughout 本 round phase η。

## §7. mid-chat handoff metadata (anchor 28.5 unique section)

本 round は L-Q3-60 codify 工程自体が L-Q3-60 violation で finalize する self-undermining risk の構造的回避のため、phase β 完遂後 mid-chat handoff を採用 (structural signature 5-2 application)。anchor 28.3 v0.1 / 28.4 v0.1 declaration には本 section 不在、anchor 28.5 v0.x unique。

### 7-1. mid-chat split structure

| phase | locus | content |
|---|---|---|
| §α scope LOCK | 前 chat | round scope + L-Q3-60 / L-Q3-59 sub-class 構造設計 LOCK |
| §β-1 declaration.md draft v2 | 前 chat | core draft v1 + 修正 #1-#3 inline 反映済 |
| §β-2 lessons_appendix.md draft v1 | 前 chat | core draft 完成 |
| §μ-1/2/3 mid-chat handoff package | 前 chat | claude_ai_mid_handoff_memo.txt + claude_code_mid_handoff_memo.txt + anchor_28_5_v0_x_transition_state.pdf |
| §β-cont (M4 cycle 2-N) | 本 chat | draft revisions |
| §γ-ι | 本 chat | input_files_pin / verification_log / envelope / M1 audit / phase Z / F1 / FULL CLOSURE |

### 7-2. 3-file redundancy handoff package SHA

```
claude_ai_mid_handoff_memo.txt   : 13fe050d8dfe3c8240187c12d9637e2f64218cdd037930be4593aabe5aa55d6f (Pattern 46 PASS)
claude_code_mid_handoff_memo.txt : 0458067c7c807dea3eb350b6415ab15fef6f953989f46c574c83d25e4e953f48 (Pattern 46 PASS)
anchor_28_5_v0_x_transition_state.pdf: (build後 self-pin、PDF 内 未記載が正常)
```

3-file cross-channel SHA bit-exact 一致を本 chat 開始時 claude.ai-side independent verify で確認済 (L-Q3-60 axis (iv) application、handoff integrity 担保)。

### 7-3. bilateral verify protocol record (本 round 内 L-Q3-60 self-instantiating application)

| step | content | result |
|---|---|---|
| step 1 | channel B persist (Claude Code-side filesystem) | snapshot SHA 096f9a0b0a867a322f64b3af28ee7cf76493ae4cd9aaf76bd920958acdb9aa5a / 3068 B / Pattern 46 PASS |
| step 2 | channel A reproduction (claude.ai memory retrieval、partial → user paste 補完) | full text 確定 |
| step 3 | bilateral cross-verify (claude.ai-side compute vs Claude Code-side persist) | bit-exact match → L-Q3-60 axis 4 種全 strict SATISFIED |

L-Q3-60 4-axis 全 satisfaction を達成した self-instantiating application(本 round 内 instantiate、cross-round 通算 4 件目 instance、anchor 27 triple-channel + anchor 28.4 paired sync + F-28.4-C out-of-repo pin に続く)。

## §8. round closure deferred queue (新 chat 内 handle、11 entries)

| priority | item | source |
|---|---|---|
| P0-cycle | declaration.md M4 cycle 2-N (本 chat draft revisions) | §β-1 draft |
| P0-cycle | lessons_appendix.md M4 cycle 2-N (本 chat draft revisions) | §β-2 draft |
| P0-new | input_files_pin.json draft (self_sha_omitted pattern inherit) | 新 chat |
| P0-new | verification_log.md draft (M1 audit results inscribe) | 新 chat |
| P0-new | envelope updates (.gitattributes + SHA256SUMS) | 新 chat |
| P0-new | M1 cross-artifact invariant audit (5 axis A-E) | 新 chat |
| P0-new | phase Z (commit + tag + push、rule 92 strict) | 新 chat |
| P0-new | F1 post-closure paired sync verify dispatch | 新 chat |
| P0-new | FULL CLOSURE handoff memo 3-file package + verification_report PDF | 新 chat |
| P0-new | M3 cross-artifact lineage audit (anchor 28.4 cluster E declared instruction inherit) | 新 chat |
| P0-new | canonical_index update (+L-Q3-60 + L-Q3-59 refinement entries inscribe) | 新 chat |

## §9. closure metadata

| axis | value |
|---|---|
| status | DRAFT v2 (M4 cycle 2 input ready、phase η commit 後 FINAL transition) |
| canonical SHA | TBD (M4-2.4 pre-compute reference + phase η commit 後 fix) |
| Pattern 46 compliance | (a) no BOM + (b) no CR + (c) LF-term (本 round byte-write 時 enforce) |
| cumulative state delta from anchor 28.4 v0.1 | L-Q3 +1 (59→60) / pattern +0 (33) / nonet +0 (9) / F-rule +0 (11) / canonical_index +2 (L-Q3-60 + L-Q3-59 refinement) |
| forensic chain depth at closure | 12 (parent 11 + 本 round 1) |
| author | Sakaguchi Shinobu (宍粟市) / 坂口製麺所 |
| license | CC-BY 4.0 |

end of anchor_28_5_v0_x_declaration.md (DRAFT v2)
