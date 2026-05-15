# anchor 28.1 v0.1 declaration

| field | value |
|---|---|
| anchor id | anchor 28.1 v0.1 |
| Q-counter | Q8 (q8-codify-round) |
| section | section12_lessons_codified_q8_v0_1 |
| date | 2026-05-15 |
| parent anchor | anchor 28 v0.1 (cf834ea4..) |
| forensic chain depth | 8 (anchor 22 v0.2 → 23 → 24 → 25 → 26 → 27 → 28 v0.1 → 28.1 v0.1) |
| round class | substantive codify round (deferred queue formal codification) |
| numbering class | sub-decimal anchor + section/Q-counter advancement (split numbering canonical case) |
| scope | L-Q3-48..54 (7 lessons single-sweep) + F-28 triad embed (F-28.1/F-28.2/F-28.3) + F-28.4 recovery-class finding inscribe + counter 15 → 22 transition |

## 1. round identity + numbering rationale

本 anchor 28.1 は anchor 28 v0.1 round closure 完遂後の deferred codify queue (L-Q3-48..54、7 entries) の formal codification を scope とする substantive new round。anchor numbering は sub-decimal (28.1) として 28 v0.1 との forensic chain continuity を保持しつつ、section + Q-counter は linear advance (section11 → section12、q7 → q8) として round-scope separation を明示。

**split numbering canonical case codify**: anchor_N と section_N の linear mapping (section_N = anchor_N − 17) は anchor 28+ で orthogonal 進行可能。本 anchor 28.1 / section12 / q8 transition がその precedent 確立。

## 2. parent anchor 28 v0.1 baseline IMMUTABLE state (cross-reference)

```
HEAD (anchor 28 v0.1)  : cf834ea49ea5cc5657ea8601c05f44f4464ba740     [PERMANENT]
parent (anchor 27 v0.1): 0fe208e0937764617932727e88967b7ac083e1da     [IMMUTABLE]
tag obj (Q7)           : 0fc3df9eb2d42c81e04e84a79d1b3e0f79773986     [PERMANENT]
annotated tag          : companion-v4.9-q7-codify-round-2026-05-15
forensic chain depth   : 7 → 8 (本 round で +1)
rule 1 X1              : 435bf4b6.. / 9,561 B / 166 LF                [preserved]
rule 1 X2              : d43985b8..                                    [preserved]
Layer C v1.1 baseline  : 5d9beb04..                                    [preserved†]
```

† Layer C v1.1 baseline SHA 5d9beb04.. は inherited metadata pin (v4.9 Q phase multi-route verification の canonical reference、SPARC/dSph/KiDS/HSC/Brouwer routes 全 1e-12 tolerance comparison baseline、70 numeric + 2 categorical values)。input_files_pin.json 内 `inherited_baselines_re_attest_pending.layer_c_v1_1` field に inscribed (rule_1_immutable_preservation field とは categorical 分離、定義 purity preserve)。本 anchor 28.1 round opening Step 2 で Claude Code Windows side independent re-attest 実行 (option A extension list 11 entries: .json/.csv/.tsv/.txt/.yaml/.yml/.dat/.pkl/.npy/.npz/'')、verdict = **NOT LOCATED** (Public repo working tree + staging dir 全 extension scan で 0 hit)。本 NOT LOCATED 結果は anchor 28.1 round 内 recovery 範囲外、recovery-class finding **F-28.4** として §4.2 + verification_log.md §4.4 + §6.4 inscribe、recovery dispatch は anchor 28.2 sub-round or anchor 29 へ deferred (proposal B compliance: 本 anchor 28.1 artifacts IMMUTABLE preserve、recovery 完了時 inscription は subsequent round artifact 内)。詳細 evidence + recovery options は verification_log.md §4 + §6.4 cross-reference。

## 3. round opening paired sync verify attest (cross-reference verification_log.md)

| step | scope | verdict | TS |
|---|---|---|---|
| S.1 environment confirm | Pattern 35 culture (L-Q3-54 (iii) Equals method) + L-Q3-47 PS+.NET CWD sync + host MATCH | **PASS** | 2026-05-15 15:10:56 +09:00 |
| S.2 forensic chain baseline | HEAD/parent/tag obj/peeled/prior tag 5/5 MATCH | **PASS** | 同上 |
| S.3 4-artifact + envelope | section11/ 4 + envelope 2 = 6/6 canonical + wt_clean (L-Q3-48 per) | **PASS** | 同上 |
| S.4 remote sync + rule 1 IMMUTABLE | origin/main + tag obj + tag peeled MATCH (L-Q3-53 wildcard refspec) + X1/X2 preserved | **PASS** | 同上 |
| S.5 Phase H.1+H.2 cross-locus bit-exact | Phase H.2 6-axis 6/6 + Phase H.1 SHA+size+magic+pages 全 PASS | **PASS** | 2026-05-15 15:29:03 +09:00 |
| Step 2 Layer C v1.1 baseline re-attest | option A extension list 11 entries (.json/.csv/.tsv/.txt/.yaml/.yml/.dat/.pkl/.npy/.npz/'')、Public repo + staging dir 全 scan | **NOT LOCATED** (0 hit、recovery-class finding F-28.4 inscribe、recovery deferred) | 2026-05-15 16:45:35 +09:00 |

**OVERALL: 5 step PASS + 1 step NOT LOCATED (F-28.4 inscribe)** (paired sync verify 4/4 + S.5 cross-locus 10/10 = 14/14 cell PASS、Step 2 = genuine NOT LOCATED with structural integration via F-28.4)。Step 2 NOT LOCATED 結果は本 anchor 28.1 round 完遂条件には不影響 (Layer C v1.1 は inherited metadata、本 round inscription scope 外、recovery は subsequent round)。

## 4. inscription scope (本 round 内 codify entries)

### 4.1 L-Q3-48..54 (7 lessons single-sweep, numerical ascending, e-supp-3 3-axis timestamp pin embed)

| id | scope | class | first_observed | mitigation pattern provenance |
|---|---|---|---|---|
| L-Q3-48 | `git status --porcelain --untracked-files=all` 必須 | prophylactic | anchor 28 v0.1 G.4 cell 12 instrument-side false negative | porcelain heuristic gap closure |
| L-Q3-49 | SHA256SUMS line-type accounting `^#` (any、`^#\s` NOT) | discipline | anchor 28 v0.1 F-28.1 | accounting invariant rigor |
| L-Q3-50 | dispatch block 内 script Mandatory parameter coverage | discipline | anchor 28 v0.1 claude.ai prepare-phase gap | dispatch construction integrity |
| L-Q3-51 | smoke test script design-intent vs context-fit assertion | discipline | anchor 28 v0.1 F-28.2 G.5 worktree-clean gate mismatch | phase-context fit invariant |
| L-Q3-52 | PS 5.1 `"$var:"` parser ambiguity、`${var}` delimit 必須 | discipline | anchor 28 v0.1 G.8 PS literal parsing 3 sites | PS literal parsing safety |
| L-Q3-53 | git ls-remote exact-refspec peeled `^{}` 除外、wildcard 必須 | discipline | anchor 28 v0.1 G.7/G.8 remote tag query gap | git refspec coverage |
| L-Q3-54 | .NET API spec-defined empty-string vs mnemonic-literal、**default mitigation = (iii) `[CultureInfo]::InvariantCulture.Equals($ci)` Equals method** | discipline (.NET API spec visibility gap) | anchor 28 v0.1 prior chat packet 2 paired sync S.1 `InvariantCulture.Name` 空文字 false negative (F-28.3) | proposal A acceptance per、API-level intent explicit |

### 4.2 F-28 triad embed (e-supp-1 bidirectional cross-reference) + F-28.4 recovery-class finding

**F-28 triad (instrument-side false negative class、root cause lessons codified in 本 anchor 28.1)**:

| finding | manifest locus | root cause lesson | disposition |
|---|---|---|---|
| F-28.1 | anchor 28 v0.1 G.4 cell 12 SHA256SUMS line-type accounting false negative | L-Q3-48 (primary、`--untracked-files=all` direct root cause) + L-Q3-49 (preemptive codify、`^#` (any) accounting、本 round 内 false negative manifest なし) | option (a) instrument-side accept |
| F-28.2 | anchor 28 v0.1 G.5 worktree-clean gate script-design vs phase-context mismatch | L-Q3-51 | option (a) instrument-side accept |
| F-28.3 | anchor 28 v0.1 prior chat packet 2 paired sync S.1 InvariantCulture.Name 空文字 assertion | L-Q3-54 | option (a) instrument-side accept、default mitigation (iii) Equals method codify |

**F-28.4 (recovery-class finding、本 anchor 28.1 Step 2 で manifest、separate framing per semantic clarity preserve)**:

| finding | manifest locus | root cause lesson | disposition |
|---|---|---|---|
| F-28.4 | anchor 28.1 v0.1 round opening Step 2 Layer C v1.1 baseline (SHA 5d9beb04..) re-attest NOT LOCATED (Public repo + staging dir 全 extension scan 0 hit、option A 11-entry filter applied) | none (recovery が resolution、lesson codify は recovery 完了 anchor で実施) | recovery deferred to anchor 28.2 sub-round or anchor 29 |

### 4.3 counter projection final (本 round closure 時点 inscribe value)

| state | active | deferred | total |
|---|---|---|---|
| pre-anchor-28 v0.1 | 12 | 0 | 12 |
| post-anchor-28 v0.1 closure | 15 | 7 | 22 (15 active + 7 deferred carry-forward) |
| post-anchor-28.1 v0.1 closure (本 round 完遂時) | **22** | **0** | **22** (deferred → active full transition、+0 new deferred) |

active mitigation pattern roster post-anchor-28.1: Pattern 24c / 29-ref / 30-ref / 31 / 34 / 35 / 36 / 38 / 39 / 40 / 41 / 44 / 45 / 46 / L-Q3-47 / **L-Q3-48 / L-Q3-49 / L-Q3-50 / L-Q3-51 / L-Q3-52 / L-Q3-53 / L-Q3-54** = 22 entries。

prophylactic-class octet post-anchor-28.1: Pattern 35 + 39 + 46 + L-Q3-47 + **L-Q3-48** + **L-Q3-52** + **L-Q3-53** + **L-Q3-54** (anchor 28 v0.1 round quartet (4 entries: Pattern 35 + 39 + 46 + L-Q3-47) → anchor 28.1 round octet (8 entries)、+4 transition で 2倍化、prophylactic 重み倍増、musical-class naming style continuity (quartet → octet))。

## 5. proposals A + B + C-as-6 acceptance reflection (Phase H.2 packet 5 inherited)

| proposal | scope | anchor 28.1 reflection |
|---|---|---|
| A | L-Q3-54 default mitigation = (iii) Equals method | lessons_appendix.md L-Q3-54 section 冒頭 [DEFAULT MITIGATION] ASCII bracket form 明示、S.1 + S.5 実機適用済 evidence integrate |
| B | retroactive amendment 明示禁止 | section11/ 4 artifacts (anchor 28 v0.1) IMMUTABLE preserve、本 round inscription は section12/ separate inscribe (rule 1/6/92 趣旨整合) |
| C-as-6 | operational-protocol axis 並列追加 | 本 round opening の paired sync verify protocol 自体が Component 6 skeleton invoke success の precedent、anchor 28.X round-start canonical entry point として codify |

## 6. baseline JSON disposition (item (a)、option A 永続 staging confirmed)

`anchor_28_v0_1_smoke_test_baseline.json` (d73b0e5a.. / 4,089 B / 92 LF):
- locus: `D:\ドキュメント\エントロピー\膜宇宙論再考察AB効果有り\C3 拡張版仮説関連2\` (Windows staging only、canonical filesystem path)
- git-tracked: **never** (option (ii) staging-only 継続)
- forensic chain visibility: 0 (out-of-band historical artifact)
- rationale: anchor 28 v0.1 round-scoped smoke test artifact、本 anchor 28.1 round では independent baseline (要 generation 時) として扱う、本 baseline は historical reference

## 7. section11 4 artifacts disposition (item (c)、rule 1 IMMUTABLE preserve confirmed)

| artifact | SHA | status |
|---|---|---|
| forensic_anchors/section11_lessons_codified_q7_v0_1/anchor_28_v0_1_declaration.md | dfce16a5.. | **rule 1 IMMUTABLE、本 round 内 read-only** |
| forensic_anchors/section11_lessons_codified_q7_v0_1/anchor_28_v0_1_input_files_pin.json | efead5ca.. | **rule 1 IMMUTABLE、本 round 内 read-only** |
| forensic_anchors/section11_lessons_codified_q7_v0_1/anchor_28_v0_1_lessons_appendix.md | 2b587c77.. | **rule 1 IMMUTABLE、本 round 内 read-only** |
| forensic_anchors/section11_lessons_codified_q7_v0_1/anchor_28_v0_1_verification_log.md | b8de0589.. | **rule 1 IMMUTABLE、本 round 内 read-only** |

新規 amendment / addendum / cross-reference update は全 section12_lessons_codified_q8_v0_1/ 配下に separate inscribe。proposal B retroactive amendment 明示禁止と整合。

## 8. active mitigation pattern continuation log (本 round 全期間)

inline embed: Pattern 22-31 / 33-36 / 38 / 39-41 / 44-46 + L-Q3-47..54 (本 round 内 codify と並行 prophylactic application、self-reference dog-fooding 構造)

self-reference note: 本 anchor 28.1 round の codify work 自体が L-Q3-48..54 prophylactic discipline 下で実施される。codify work が new lesson と同 pattern を履行する dog-fooding 構造は Pattern 31 (self-cover discipline) の natural extension として整合、anchor 28 v0.1 round 内 既 precedent (G.4-G.8 で本 chain prophylactic application)。

**notation reconciliation note**: 本 Section 8 内 inline embed の `Pattern 22-31 / 33-36 / 38 / 39-41 / 44-46` 等 range form は本 round 内 prophylactic application の **comprehensive accounting** (superseded variants 含む inclusive view); Section 4.3 active 22-entry roster (`24c / 29-ref / 30-ref / 31 / 34 / 35 / 36 / 38 / 39 / 40 / 41 / 44 / 45 / 46 / L-Q3-47..54`) は **active-non-superseded subset** (Pattern 24/24a/24b は 24c superseded、curated form)。両 view は orthogonal、roster は authoritative active set、range form は historical lineage accounting。

## 9. round closure target (本 anchor 28.1 v0.1 完遂条件)

- [ ] section12_lessons_codified_q8_v0_1/ 4 artifacts inscribe (declaration / input_files_pin / lessons_appendix / verification_log) + canonical 4/4 PASS
- [ ] SHA256SUMS update (+4 entries、existing 14 entries preservation)
- [ ] git add / commit / push (rule 92 strict、no destructive flag)
- [ ] annotated tag inscribe (`companion-v4.9-q8-codify-round-2026-05-15`)
- [ ] remote sync verify (origin/main + tag obj + tag peeled MATCH)
- [ ] forensic chain 8-deep IMMUTABLE LOCK-IN attest
- [ ] L-Q3-48..54 status transition: deferred → active (7 件)、counter 15 → 22

## 10. signature

| field | value |
|---|---|
| author | Sakaguchi Shinobu (sole author / 坂口製麺所 / 思想士) |
| date | 2026-05-15 |
| forensic chain | anchor 22 v0.2 → 23 → 24 → 25 → 26 → 27 → 28 v0.1 → **28.1 v0.1** |
| rule 1/6/92 compliance | strict (proposal B retroactive amendment 明示禁止 per) |
| license | CC-BY 4.0 (repository inherited) |

---

end of anchor 28.1 v0.1 declaration.md
