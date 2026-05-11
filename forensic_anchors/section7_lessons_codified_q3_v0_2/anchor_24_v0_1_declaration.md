# anchor 24 v0.1 Q3 codify round v0.2 declaration

```
作成日:     2026年5月XX日 JST (step 5 commit 日に確定)
作成元:     claude.ai turn N+17.x + Claude Code (Windows) paired sync
配置 path:  forensic_anchors/section7_lessons_codified_q3_v0_2/anchor_24_v0_1_declaration.md
license:    CC-BY 4.0
依存 path:  forensic_anchors/section5_axis_4_type_alpha/anchor_22_v0_2_input_files_pin.json (X1 IMMUTABLE, rule 1)
            forensic_anchors/section6_lessons_codified_q3_v0_1/* (anchor 23 v0.1 closure deliverable)
            latex_v48/membrane_v48.tex (X2 IMMUTABLE, rule 6)
```

---

## 1. round 識別

| 項目 | 値 |
|---|---|
| anchor id | anchor 24 v0.1 |
| round name | Q3 codify round v0.2 |
| Phase | 5-D (documentation step + structural fix、mixed 性質) |
| 性質 | 11 entries documentation-only + 1 entry ★ structural fix |
| baseline commit | `3aef5142167f993f2ba8a6f67d9b925c1252cc4b` (anchor 23 v0.1 closure) |
| baseline tag | `companion-v4.9-q3-codify-round-2026-05-11` |
| 想定 envelope | 6 files (single envelope commit、D4 option A 採用) |
| forensic chain index | C8 (anchor 23 v0.1 [C7] からの連番) |

---

## 2. round 目的

anchor 22 v0.2 round (axis_4 type-α hardening) から forward された Pattern 24-31 + 24a/24b の 10 entries、および anchor 23 v0.1 round (Q3 codify round) で新規認識された Pattern 32 + 33 の 2 entries、合計 **12 entries** を正式 codify し、`forensic_anchors/section7_lessons_codified_q3_v0_2/` に永続化する。

加えて、L-Q3-29 (Pattern 33: `.gitattributes` self-coverage gap) は anchor 23 v0.1 で root cause isolation 済の structural defect であり、本 round の step 4 で **structural fix** を execute する:

```
.gitattributes 末尾に「.gitattributes -text」行を append
→ git check-attr text .gitattributes → "text: unset"
→ core.autocrlf=true 下でも .gitattributes 自身は normalization 対象外
→ working tree (CRLF) と blob (LF) の二重性 解消、wt SHA == blob SHA 統一
```

structural fix の完了により、`.gitattributes` は HTTPS raw URL audit (Protocol 2) と local Get-FileHash (Protocol 1) の両方で bit-exact 一致を達成、本 round step 7 で 6/6 wt_eq_blob mode で audit 完走することで Pattern 33 の forensic-grade 解消が verify される。

---

## 3. scope (12 entries)

| 番号 | Pattern | 名称 | 起源 round | 種別 |
|---|---|---|---|---|
| L-Q3-18 | 24 | packet narrative parse error 一般化 | anchor 22 v0.2 | doc-only |
| L-Q3-19 | 25 | PowerShell block comment non-nestable | anchor 22 v0.2 | doc-only |
| L-Q3-20 | 26 | PSObject.Properties.Count quirk | anchor 22 v0.2 | doc-only |
| L-Q3-21 | 27 | git status porcelain default aggregation | anchor 22 v0.2 | doc-only |
| L-Q3-22 | 28 | git diff untracked exclusion | anchor 22 v0.2 | doc-only |
| L-Q3-23 | 29 ★ | PS Push-Location vs .NET CWD divergence (MAJOR) | anchor 22 v0.2 | doc-only |
| L-Q3-24 | 30 | PS 5.1 .ps1 default cp932 misread Japanese path | anchor 22 v0.2 | doc-only |
| L-Q3-25 | 31 | autocrlf + -text directive missing CRLF warning | anchor 22 v0.2 | doc-only |
| L-Q3-26 | 24a | narrative continuation # prefix 漏れ | anchor 22 v0.2 | doc-only |
| L-Q3-27 | 24b | PS string interpolation $VAR: PSDrive collision | anchor 22 v0.2 | doc-only |
| L-Q3-28 | 32 | PS 5.1 git push stderr NativeCommandError wrap | anchor 23 v0.1 | doc-only |
| L-Q3-29 | 33 ★ | .gitattributes self-coverage gap | anchor 23 v0.1 | ★ structural fix |

詳細 codify body は `anchor_24_v0_1_lessons_appendix.md` を参照。

---

## 4. baseline integrity (anchor 23 v0.1 closure preserved)

本 round の baseline は anchor 23 v0.1 closure commit `3aef5142` である。下記 forensic chain element は本 round で **rule 1 IMMUTABLE preserve**:

- `3aef5142` (anchor 23 v0.1 closure commit、HEAD~1 として preserve)
- `companion-v4.9-q3-codify-round-2026-05-11` annotated tag (object SHA `e6444a38...`)
- section6 配下 4 file 全 SHA + size
- `491ff34c` (anchor 22 v0.2 closure commit、HEAD~2 として preserve)
- `companion-v4.9-axis-4-type-alpha-2026-05-10` annotated tag
- X1 `435bf4b6...` (rule 1 IMMUTABLE、`forensic_anchors/section5_axis_4_type_alpha/anchor_22_v0_2_input_files_pin.json`)
- X2 `d43985b8...` (rule 6 IMMUTABLE、`latex_v48/membrane_v48.tex`)

詳細 pin 値は `anchor_24_v0_1_input_files_pin.json` を参照。

---

## 5. execution overview (step 1-7、~107 gates)

| step | gates | write target | 主要内容 |
|---|---|---|---|
| step 1 | 10 | none | pre-state verify、Pattern 33 BEFORE-fix baseline characterize |
| step 2 | 8 | staging dir | section7 4 file 起草 + base64 staging |
| step 3 | 18 | section7 4 file + git index | base64 decode + 本配置 cp + git add |
| step 4 ★ | 22 | .gitattributes + SHA256SUMS + git index | ★ Pattern 33 structural fix + cascade |
| step 5 | 16 | local commit + tag | single envelope commit (6 files 一括) + annotated tag |
| step 6 | 19 | remote refs (★ irreversible) | push main + push tag |
| step 7 | 14 | none | raw URL audit (Protocol 2)、6/6 wt_eq_blob mode |

forensic-grade closure 判定基準: 全 step ALL PASS + 2-protocol external reproducibility 確立 + Pattern 33 解消 verify。

---

## 6. roles (paired sync 設計)

- **claude.ai (本 chat side)**: planning_prep / packet drafting / PDF generation / closure record drafting / WordPress (本 round では未予定)
- **Claude Code (Windows side)**: packet inline execute / SHA verify / git op / remote ref update
- **paired sync**: claude.ai 側で packet 起草 → user paste → Claude Code 側 inline execute → output を claude.ai 側に paste back → 次 packet 起草 の loop

paired sync は両 chat の state を bit-exact 整合性で運用する forensic discipline。本 round の各 step 完了時に必要に応じ verify packet を挿入し drift detection を実施。

---

## 7. 取扱 rules (forensic discipline)

- rule 1 IMMUTABLE: 確定済 anchor 不可侵 (anchor 21 v0.2、22 v0.1、22 v0.2、23 v0.1 全 element)
- rule 6 IMMUTABLE: `membrane_v48.tex/pdf` の SHA 不変
- rule 26: 4-route 最小一致 (multi-route minimum、Phase Q PASS_EXCEEDED の根拠)
- rule 92 strict: 各 packet の write target 明示、scope 逸脱禁止。push 系では `--all` / `--tags` / `--force` / `--mirror` 一切不使用、specific ref push only
- fail-fast sequential: 各 gate PASS 確定後にのみ次 gate、途中 FAIL は即時 stop + root cause isolation (4 段階診断 D.1-D.4 pattern 適用可能)

---

## 8. Pattern mitigation 全 packet 適用 (12 件)

本 round の全 packet (step 1-7) には下記 Pattern mitigation を一律適用:

- Pattern 22-31 + 24a/24b (10 件、anchor 22 v0.2 で確立)
- Pattern 32 (anchor 23 v0.1 で確立): `$LASTEXITCODE` primary check + pattern match の 2 階層 bypass
- Pattern 33 (anchor 23 v0.1 で確立): step 1-3 で acknowledge、step 4 で structural fix execute、step 7 で fix 解消 verify

詳細は `anchor_24_v0_1_lessons_appendix.md` を参照。

---

## 9. closure criteria (forensic-grade)

下記全条件を満たした時点で本 round を formal CLOSED と判定する:

1. step 1-7 全 packet で全 gate PASS (累積 ~107 gates)
2. envelope 6 files の single commit + annotated tag が remote (origin/main) に publish 済
3. 2-protocol external reproducibility 確立 (Protocol 1 = git wire、Protocol 2 = HTTPS raw URL)
4. Pattern 33 解消 verify (step 7 G.8 で 6/6 全 file が wt_eq_blob mode で PASS)
5. closure record (claude.ai 側 turn N+17.x 末で生成) で本 round 全体を traceable に記録
6. baseline anchor 23 v0.1 closure commit `3aef5142` が本 round 全 step で preserve (rule 1 IMMUTABLE 違反なし)

---

## 10. 関連 deliverable

- `anchor_24_v0_1_input_files_pin.json`: 本 round で参照する全 pin set (D7 scope)
- `anchor_24_v0_1_lessons_appendix.md`: 12 entries codify body + L-Q3-29 inline evidence
- `anchor_24_v0_1_verification_log.md`: step 1-N verification record (step 5 commit 時点で stable)
- (claude.ai side 同時生成、Public repo 配置 外):
  - `anchor_24_v0_1_planning_prep.md` (本 round 設計 contract、turn N+17.0 で生成)
  - `anchor_24_v0_1_q3_codify_round_v0_2_closure_record.txt` (本 round closure 後生成予定)
  - `anchor_24_v0_1_verification_report.pdf` (本 round 全体 verification 記録、closure 後生成予定)

---

END OF anchor 24 v0.1 Q3 codify round v0.2 declaration
