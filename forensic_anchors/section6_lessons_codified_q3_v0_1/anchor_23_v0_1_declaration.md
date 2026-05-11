# anchor 23 v0.1 — declaration (Q3 codify round sub-round 宣言)

> **anchor 23 v0.1** — companion v4.9 系列 axis_4 type-alpha 公開 round
> (anchor 22 v0.2 / commit `491ff34c` / tag
> `companion-v4.9-axis-4-type-alpha-2026-05-10`) で観察された forensic / shell
> / git / harness 関連 lesson 7 件を Q3 series (L-Q3-11 〜 L-Q3-17) として
> 正式 codify する documentation 専用 sub-round。
>
> Phase 5-C (documentation + memory persistence + handoff) 内に位置付ける。
> rule 1 IMMUTABLE 遵守、anchor 22 v0.2 element 不可侵。

---

## 1. round 概要

### 1.1 sub-round 識別 metadata

| key | value |
|---|---|
| anchor id | anchor 23 v0.1 |
| round name | Q3 codify round |
| Phase | 5-C (documentation step) |
| 起草 date (JST) | 2026-05-11 |
| 起草 chat | claude.ai turn N+15.0 (closure 直前) + N+15.1 (本 round 実 execute) |
| 実 execute (planned) | Claude Code (Windows MSI-Z790ACEMAX、PowerShell 5.1) |
| baseline commit | `491ff34cce22040e052f226e64adddc1669ea1b4` (anchor 22 v0.2) |
| baseline tag | `companion-v4.9-axis-4-type-alpha-2026-05-10` (annotated) |
| baseline tag object | `bb45be3f01eede0f47f3e18c4e3ab34784a521c3` |
| 本 round tag candidate | `companion-v4.9-q3-codify-round-2026-05-11` |
| 配置 section | `forensic_anchors/section6_lessons_codified_q3_v0_1/` |

### 1.2 本 round の性質

本 round は **documentation 専用 sub-round** である。新規 scientific content
(企画コア理論 / 観測 / data 解析 / 数値 reproducibility) の追加・変更を一切
含まない。codify 対象は anchor 22 v0.2 round 内および本 round 起頭時点で
発見された **forensic / shell / git / harness 系の挙動 pattern lesson** 7 件
のみである。

scope を意図的に狭く保つ理由:
- anchor 22 v0.2 (axis_4 type-alpha hardening) の scientific claim と本 round
  の workflow lesson を混合せず分離管理することで forensic narrative の
  cleanness を保つ
- rule 1 IMMUTABLE 上、既 publish の anchor 22 v0.2 element は不可侵。本 round
  は anchor 22 v0.2 の "extending / superseding entries" のみを記録する
  位置付け

---

## 2. scope 定義

### 2.1 codify 対象 (in-scope)

7 件の Q3 lesson を本 round で正式 codify する。詳細は同 section 内
`anchor_23_v0_1_lessons_appendix.md` を参照:

| Lesson ID | Pattern ID | family | 関係 |
|---|---|---|---|
| L-Q3-11 | 17 | shell calc / encoding | extending |
| L-Q3-12 | 18 | PowerShell 5.1 cmdlet quirk | extending |
| L-Q3-13 | 19 | workflow / metadata | extending |
| L-Q3-14 | 20 | git semantics | extending |
| L-Q3-15 | 21 | git semantics | **superseding** anchor 22 v0.2 |
| L-Q3-16 | 22 | harness / sandbox | extending (新規) |
| L-Q3-17 | 23 | PowerShell 5.1 cmdlet quirk | extending (新規) |

### 2.2 4-file structure (anchor 22 v0.2 流用)

```
forensic_anchors/section6_lessons_codified_q3_v0_1/
  anchor_23_v0_1_declaration.md       (本 file、sub-round 宣言)
  anchor_23_v0_1_input_files_pin.json (Phase X lock 値 + 起頭時点 host state)
  anchor_23_v0_1_lessons_appendix.md  (Pattern 17-23 codify 本体、7 entries)
  anchor_23_v0_1_verification_log.md  (verify gate 記録、P15.1.x post-fill)
```

### 2.3 cascade scope (本 round で更新する forensic chain element)

**必須**:
- `SHA256SUMS`: 新規 4 anchor files (+ `.gitattributes` 更新分) の SHA-256 entry
  を append

**option 含む** (本 round 採択):
- `.gitattributes`: 既存 forensic preservation 慣習に整合する directive 追加
  (Pattern 21 訂正版の副次 mitigation、autocrlf 影響を section6 配下から
  完全排除、binary mode で git touch 全面禁止):
  ```
  forensic_anchors/section6_lessons_codified_q3_v0_1/** -text
  ```
  既存 anchor (section2_*, section5_*) が同 pattern (`<section>/** -text`)
  を採用しており、design 一貫性を維持。`text eol=lf` 等の text mode より
  forensic 上強力 (autocrlf 影響を受ける余地が原理的に存在しない)。

**out-of-scope** (本 round では touch しない):
- `latex_v48/membrane_v48.tex`: 本 round は documentation 専用、paper-side
  inventory への anchor 23 v0.1 row 追加は scientific narrative の希釈を避ける
  ため見送り。次 scientific round (v4.10 系列等) 時に inventory 拡張判断
- anchor 22 v0.2 配下 element 一切: rule 1 IMMUTABLE 遵守
- 他 forensic_anchors/ 既存 section 配下: 不可侵

### 2.4 想定 envelope (commit envelope estimate)

```
files changed: 5-6 (anchor 23 v0.1 4 files + SHA256SUMS + .gitattributes)
insertions:    +550 〜 +700
deletions:     -0 〜 -3 (.gitattributes 既存内容次第)
```

実 envelope は P15.1.3a pre-push diff で確定する。

---

## 3. round-level forensic 規律

### 3.1 rule 1 IMMUTABLE (確定済 anchor 不可侵)

本 round 起草時点で IMMUTABLE な anchor (本 round で touch 禁止):

- anchor 21 v0.2 配下全 entries (turn N+10 baseline 内、6 entries)
- anchor 22 v0.1 配下全 entries (4 entries)
- anchor 22 v0.2 配下全 entries (4 entries、turn N+14.4 publish: commit
  `491ff34c`、tag `companion-v4.9-axis-4-type-alpha-2026-05-10`)

= 計 10 anchors / 14 entries (累積)

本 round の anchor 23 v0.1 は **新規追加**、既存 IMMUTABLE element 一切に
write しない。superseding が必要な entry (L-Q3-15 / Pattern 21) は本 round
の lessons_appendix §2.5 に **新規 entry として記録** し、cross-reference 経由
で先行 entry (anchor 22 v0.2 内) と意味的に連関させる。

### 3.2 rule 6 IMMUTABLE (`membrane_v48.pdf` SHA 不変性)

`membrane_v48.pdf` の SHA-256 は turn N+10 baseline 以降不変
(IMMUTABLE preserve)。本 round は `.tex` source / PDF いずれにも touch せず、
本 immutability 維持。

### 3.3 rule 92 strict (scope 守備)

本 round の disk write target:

- `forensic_anchors/section6_lessons_codified_q3_v0_1/` 配下 4 files (新規作成)
- `SHA256SUMS` (existing file、append のみ)
- `.gitattributes` (existing file、append/modify)

それ以外 (anchor 22 v0.2 element、他 section、`latex_v48/`、`docs/`、`.git/`
等) への write は **完全禁止**。

### 3.4 commit / tag 規律

- commit message:
  ```
  feat: anchor 23 v0.1 Q3 codify round + Pattern 21 superseding + .gitattributes
  hardening [C7]

  Documentation-only sub-round codifying 7 Q3-series lessons (L-Q3-11〜L-Q3-17,
  Pattern 17-23) discovered during anchor 22 v0.2 round and turn N+15.0 paired
  sync. Includes Pattern 21 (autocrlf) corrected interpretation as superseding
  entry. .gitattributes hardening removes system-scope autocrlf=true dependency.

  Envelope: 5-6 files / +550 to +700 / -0 (approx, finalized at pre-push diff)
  ```
- annotated tag: `companion-v4.9-q3-codify-round-2026-05-11`
  - lock date: 2026-05-11 JST (本日)
  - Pattern 19 mitigation: 同日内に commit + push 完了想定で cross-day deviation
    回避狙い
- push: rule 92 strict 遵守、specific ref push のみ
  (`git push origin main` + `git push origin <tagname>`、`--all` / `--tags` /
  `--force` / `--mirror` 一切なし)

---

## 4. round execute plan (P15.1.0 〜 P15.1.4)

anchor 22 v0.2 round (turn N+14.2 系列 5 packet) の fail-fast sequential design
を流用する:

| packet | 内容 | gate 概算 | 性質 |
|---|---|---|---|
| P15.1.0 | sanity check + section6_*/ scaffold confirm (read-only 大半) | ~12 | abort-able |
| (pre-1.1) | 4-file 一括 base64 staging (Pattern 22 mitigation) | ~8 | local OUTSIDE write |
| P15.1.1 | staging → Public repo target copy + SHA verify | ~18 | in-repo write、reversible |
| P15.1.2 | cascade: SHA256SUMS append + `.gitattributes` 更新、staging sync、blob hash triangle | ~20 | local write、reversible |
| P15.1.3a | pre-push: commit + annotated tag + dry-run | ~16 | local write、reversible |
| P15.1.3b | ★ CRITICAL irreversible: push main + push tag、post-fetch verify | ~19 | irreversible |
| P15.1.4 | raw URL audit (新 4 file + 更新 SHA256SUMS の HTTPS round-trip) | ~14 | read-only |

累積 gate 推定: ~107 (anchor 22 v0.2 round の 86 を上回るのは pre-1.1 staging
gate 追加分 + cascade option work 分)。

---

## 5. 起頭時点 host state (lock 参照値、詳細は pin.json)

| key | value |
|---|---|
| Public repo HEAD | `491ff34cce22040e052f226e64adddc1669ea1b4` |
| origin/main | (same) |
| porcelain | clean (0 lines) |
| X1 (PIN_JSON) SHA | `435bf4b68a48e251e7a591564ee870b94072f9573292579ddac0ead6f7eff2be` |
| X2 (TEX) SHA | `d43985b896b63e625718bd6d5d644abbfe6b2cee99721290d0165a39c212e5dd` |
| X3 (SUMS) SHA | `d7ac7050c0cf7c42b302db0c18e73eb3d3834753b7f444ec3591f5c3fe3bcd82` |
| core.autocrlf (scope-aware) | `system	true` (verbatim) |
| paired memo v1.1 SHA | `00D87DC07EBCC2DE656CABAAEC917BD499314925901BF1DEA10051B51E40A09B` |
| paired memo v2.0 SHA | `64445040B9FC9E08256CC35D8E3477D4849B5DAA1C4E6604F23A247B2BF603D3` |

完全な lock metadata は `anchor_23_v0_1_input_files_pin.json` 参照。

---

## 6. 期待 outcome

本 round 完了時点で:

1. `forensic_anchors/section6_lessons_codified_q3_v0_1/` 配下に 4 anchor files
   永続記録
2. SHA256SUMS に新規 4-5 entries append (CAP STONE 拡張)
3. `.gitattributes` に project-side normalization directive 明示追加
4. 新 annotated tag `companion-v4.9-q3-codify-round-2026-05-11` GitHub remote
   published
5. forensic chain external reproducibility 2 protocol (git wire + HTTPS raw URL)
   で新 4 file の独立確認確定
6. Q3 series 全 lesson (L-Q3-1 〜 L-Q3-17) のうち、本 round で L-Q3-11 〜
   L-Q3-17 の 7 entries が正式 codify 完了
7. Phase 5-C documentation step 概ね closure (Pattern 21 訂正の superseding
   構造を含む、forensic honesty 上の完全性確保)

---

## 7. 履歴

| revision | date | author | summary |
|---|---|---|---|
| v0.1-draft | 2026-05-11 | claude.ai turn N+15.0 起草 | initial draft、Pattern 17-23 codify scope 定義、4-file structure 宣言、cascade scope 確定、P15.1.0〜P15.1.4 execute plan 提示 |
| v0.2-draft | 2026-05-11 | claude.ai turn N+15.1 (P15.1.2 silent failure 後 correction) | §2.3 cascade scope option の `.gitattributes` directive 例を既存 forensic preservation 慣習 (`<section>/** -text` pattern) に修正、anchor 21/22 round と design 一貫性確保 |
| (post-review revision) | (TBD) | (post user review) | (revisions applied as needed before P15.1.1 write) |

---

*End of anchor 23 v0.1 declaration draft v0.1*
