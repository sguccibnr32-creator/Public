# anchor 28.5 v0.x lessons_appendix

## §1. document identity

| axis | value |
|---|---|
| document | lessons_appendix for anchor 28.5 v0.x codify round (Q12) |
| parent declaration | anchor_28_5_v0_x_declaration.md (本 round companion) |
| scope | L-Q3-60 codify (primary) + L-Q3-59 sub-class taxonomy refinement (paired) + P2 inline note |
| author | Sakaguchi Shinobu (宍粟市) / 坂口製麺所 |
| license | CC-BY 4.0 |

## §2. cumulative state delta from anchor 28.4 v0.1

| axis | anchor 28.4 v0.1 | anchor 28.5 v0.x (本 round 完遂時 expected) | delta |
|---|---|---|---|
| forensic chain depth | 11 | 12 | +1 |
| L-Q3 cumulative | 59 | 60 | +1 (L-Q3-60) |
| L-Q3-59 sub-class | unsorted n=8 | 6 sub-class taxonomy formal (n=10) | refinement + 2 |
| canonical_index entries | (28.4 +3) | +2 (L-Q3-60 + L-Q3-59 refinement) | +2 |
| pattern cumulative | 33 | 33 | 0 |
| nonet cumulative | 9 | 9 | 0 |
| F-rule cumulative | 11 | 11 | 0 |

## §3. L-Q3-60 formal codify (primary innovation)

### 3-1. canonical identity

| axis | value |
|---|---|
| ID | L-Q3-60 |
| class | discipline (Pattern → Lesson 昇格、generality 拡張) |
| canonical name | dual-channel verification discipline |
| codify round | anchor 28.5 v0.x (本 round、初 codify) |
| relationship | Pattern 31 (paired sync verify) の principle 化 + Pattern 41/44 (3-layer compound) と orthogonal 拡張 + L-Q3-1 (tolerance schema alignment) の 上位 channel-selection discipline |

### 3-2. core principle

単一 verification channel (script output のみ / memo paste-back のみ / artifact 内 self-attestation のみ) は verification subject 自体への mental-model fill (期待値投影) を構造的に許容する。verification integrity は **independent dual channel** (channel A + channel B、source 系統独立) の cross-confirm により担保される。

### 3-3. independence axis (4 種、orthogonal)

| axis | description | example |
|---|---|---|
| (i) execution path independent | tool/protocol が異なる execution path で同 quantity を independent compute | git wire protocol SHA + filesystem cat-file SHA |
| (ii) data lineage independent | data source 系統が異なる retrieval path で同 invariant を independent confirm | script direct stdout + HTTPS raw URL retrieval |
| (iii) temporal independent | 同 invariant を異なる時刻 (round closure 時 + post-round) で independent attest | round closure 時 attest + post-round paired sync verify |
| (iv) agent independent | 同 invariant を異なる agent (claude.ai / Claude Code) で independent compute | claude.ai-side draft SHA + Claude Code-side actual file SHA |

### 3-4. reference instances

| # | instance | channel A | channel B | independence axis | round |
|---|---|---|---|---|---|
| ref-1 | triple-channel reproducibility | Protocol 1 git wire | Protocol 1' cat-file + Protocol 2 HTTPS raw URL | (i) + (ii) | anchor 27 v0.1 |
| ref-2 | post-closure paired sync verify | claude.ai-side handoff memo SHA pins | Claude Code-side S.1-S.7 actual verify | (iii) + (iv) | anchor 28.4 v0.1 |
| ref-3 | F-28.4-C out-of-repo IMMUTABLE pin | input_files_pin.json inscribe | filesystem actual SHA + size + LWT | (i) + (ii) | anchor 28.4 v0.1 |

### 3-5. violation reference (L-Q3-59 sub-class全体への参照)

L-Q3-60 violation cases = L-Q3-59 sub-class taxonomy (§4)。L-Q3-60 codify と L-Q3-59 sub-class refinement は positive-negative paired codify (本 round structural signature #1)。

### 3-6. relationship to existing discipline

| existing | L-Q3-60 との関係 |
|---|---|
| Pattern 31 (paired sync verify) | L-Q3-60 は Pattern 31 の generality 拡張 + principle 化 (Pattern → Lesson 昇格) |
| Pattern 41/44 (3-layer compound discipline) | orthogonal (Pattern 41/44 = same-channel layer 化、L-Q3-60 = cross-channel independence) |
| L-Q3-1 (tolerance schema alignment) | L-Q3-1 = single-channel 内 schema、L-Q3-60 = channel selection 上位 discipline |
| L-Q3-8 (cross-scipy-version reproducibility) | L-Q3-8 = (i) execution path independence の特殊例 (scipy version axis) |

### 3-7. application guideline

新規 verification design 時 check list:
1. 当該 verification は single-channel か?
2. single-channel の場合、mental-model fill risk が結果に影響するか?
3. 影響あり判定時、independence axis (i)-(iv) のいずれかで dual-channel 化 可能か?
4. dual-channel 化困難な場合、L-Q3-60 violation accepted と明示 + risk note inscribe

## §4. L-Q3-59 sub-class taxonomy refinement (paired codify)

### 4-1. refinement identity

| axis | value |
|---|---|
| parent | L-Q3-59 (anchor 28.4 v0.1 codify、descriptive class、mechanical replication residue) |
| refinement type | sub-class taxonomy formal establishment |
| codify round | anchor 28.5 v0.x (本 round) |
| relationship to L-Q3-60 | L-Q3-60 の structural inverse (violation negative form) |

### 4-2. 6 sub-class taxonomy

| sub-class ID | name | root mechanism | n (28.4) | flag |
|---|---|---|---|---|
| L-Q3-59(a) | path mental-model | actual filesystem path を verify せず mental-model で fill | 2 | formal |
| L-Q3-59(b) | architectural inheritance | 既存 design convention を verify せず想定型で fill | 2 | formal |
| L-Q3-59(c) | semantic resolution | term semantic scope を verify せず over-narrow / over-broad | 1 | provisional |
| L-Q3-59(d) | orchestration framing | workflow framing element を verify せず convention 想定 fill | 2 | formal |
| L-Q3-59(e) | verification design | verify check 自体の design parameter を verify せず想定 fill | 2 | formal |
| L-Q3-59(f) | quantitative pre-comp | quantitative value を pre-compute せず想定値で fill | 1 | provisional |

### 4-3. instance assignment (anchor 28.4 v0.1 origin)

| sub-class | instance | source |
|---|---|---|
| (a) | section14 directory path drift | anchor 28.4 v0.1 round packet 10 |
| (a) | X1 path 未指定 (memo §3-4 gap) | anchor 28.4 v0.1 post-closure sync verify (本 round discovery) |
| (b) | self_sha_slot new design | anchor 28.4 v0.1 round packet 23-24 |
| (b) | v3 inherited pattern (input_files_pin) | anchor 28.4 v0.1 round packet 26 |
| (c) | "memo" semantic over-narrowing | anchor 28.4 v0.1 round packet 11 |
| (d) | paste source location convention | anchor 28.4 v0.1 round packet 21 |
| (d) | cross-window observation framing | anchor 28.4 v0.1 round carry-over |
| (e) | spot-check key full SHA convention | anchor 28.4 v0.1 round packet 29 |
| (e) | M1 [B] symmetric SHA expectation | anchor 28.4 v0.1 round packet 31 |
| (f) | envelope post-size calculation | anchor 28.4 v0.1 round packet 45 |

### 4-4. threshold convention (本 round formal establish)

- 原則 threshold: n ≥ 3
- 本 round application: (a)(b)(d)(e) を n=2 ながら parallel completeness exception で formal 採用
  (4 sub-class が same-round 同時 emerge の structural symmetry を根拠)
- (c)(f) は n=1 で provisional flag 付与
- provisional flag 解除条件: 次回以降 round で同 sub-class instance 追加 confirm 時

### 4-5. 共通 root mechanism

全 sub-class 共通: verifiable な actual value / convention / design parameter を、verify 工程を省略して mental-model から fill する構造。L-Q3-60 (dual-channel verification discipline) の structural inverse。

### 4-6. total instance count

10 (formal classified 8 = (a)(b)(d)(e) × 2 each + provisional 2 = (c)(f) × 1 each)。anchor 28.4 v0.1 closure memory "n=8 threshold breach" = formal classified subset、provisional 2 は taxonomy refinement 時 additional discovery。

## §5. structural signature 3 components (本 round)

### 5-1. paired codify innovation

L-Q3-60 (discipline positive form) + L-Q3-59 sub-class taxonomy (violation negative form) の同 round 同時 codify、初 instance。positive + negative dual inscribe pattern を本 round で establish。今後の codify round で violation pattern accumulation 時、同型 paired codify 適用候補。

### 5-2. meta-self-reference avoidance

L-Q3-60 codify round 自体が L-Q3-60 violation (single-channel mental-model fill) で finalize する self-undermining risk → mid-chat handoff 採用で構造的回避。本 round 内 self-instantiating discipline application、初 instance。

具体構造:
- risk: 単一 chat session 内 M4 cycle compression + M1 audit compression → single-channel verification fill
- mitigation: chat session A (本 chat) + chat session B (新 chat) の agent-independent (independence axis (iv)) + temporal-independent (independence axis (iii)) cross-confirm
- result: L-Q3-60 codify 工程自体が L-Q3-60 application 例 (recursive self-reference 構造的回避)

### 5-3. threshold convention formal establishment

sub-class formal 命名 n≥3 原則 + n=2 parallel completeness exception + n=1 provisional flag convention の formal inscribe。anchor 28.5 round 由来 new convention、今後の taxonomy refinement に適用。

## §6. P2 inline note: rule 1 IMMUTABLE pin path 明記 convention

### 6-1. observation

anchor 28.4 v0.1 closure memo §3-4 で rule 1 IMMUTABLE preservation pin (X1, X2) は SHA 単独 inscribe、path 未指定。post-closure paired sync verify S.4 で actual path を script-side で independent discover 必要。SHA bit-exact 一致のため rule 1 IMMUTABLE preservation 自体は成立、技術的問題なし。

### 6-2. discovered actual path

- X1 actual path: `forensic_anchors\section5_axis_4_type_alpha\anchor_22_v0_2_input_files_pin.json`
- X2 actual path: `latex_v48\membrane_v48.tex` (28.4 memo §3-4 で明記済)

### 6-3. convention establish (軽量、formal L 番号無付与)

今後の rule 1 IMMUTABLE pin inscribe 時、SHA + path 両方 inscribe を convention 化。authoritative invariant は SHA、path は orchestration-side implementation detail だが、補助情報として明記により dual-channel verify (script execute 時の path resolution + SHA confirm) を強化。

### 6-4. classification

本 P2 inline note は L-Q3-59(a) path mental-model sub-class の memo-completeness gap 側面 (replication residue とは別 axis)。formal L 番号無付与、convention level codify。

## §7. closure metadata

| axis | value |
|---|---|
| document generation | claude.ai-side mid-chat packet (本 chat §β-2) |
| draft version | v2 (M4 cycle 2) |
| status | DRAFT v2 (M4 cycle 2 input ready、phase η commit 後 FINAL transition) |
| canonical SHA | TBD (新 chat byte-write 後 inscribe) |
| Pattern 46 compliance | (a) no BOM + (b) no CR + (c) LF-term (新 chat byte-write 時 enforce) |

end of lessons_appendix.md draft v1
