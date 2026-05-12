# anchor 25 v0.1 declaration — Q4 codify round v0.1

## round identification

- anchor id: anchor 25 v0.1
- round name: Q4 codify round v0.1
- baseline commit: cbc270041c7627b95e90399dc8a9eaee4f3cc8e1 (anchor 24 v0.1 closure)
- baseline tag: companion-v4.9-q3-codify-round-v0-2-2026-05-11
- baseline tag obj: 4a11c71e7535d11f5d8a2d09e55362d3783ec15f
- section path: forensic_anchors/section8_lessons_codified_q4_v0_1/
- 性質: documentation-only round (structural fix なし)
- 起草日: 2026-05-12 JST

## scope: 7 entries

substantive new codify 6 + disposition 1:

- L-Q3-10 (disposition note、documentation drift 1st + 2nd instance canonical record)
- L-Q3-30 (Pattern 24c: PS `${(expr)}` literal name quirk)
- L-Q3-31 (Pattern 34: `[UTF8Encoding]::new($true).GetBytes()` no BOM)
- L-Q3-32 (Pattern 30 refinement: scope = `.ps1` only)
- L-Q3-33 (Pattern 35: PS 5.1 `Get-Date -Format` culture-sensitive parsing)
- L-Q3-34 (Pattern 36: forensic record 内 path reference 実存性 verify 必須)
- L-Q3-35 (Pattern 29 refinement: CWD divergence detection の 2 階層化)

## envelope (6 file)

forensic_anchors/section8_lessons_codified_q4_v0_1/:
- anchor_25_v0_1_declaration.md (本 file)
- anchor_25_v0_1_input_files_pin.json
- anchor_25_v0_1_lessons_appendix.md (7 entries body)
- anchor_25_v0_1_verification_log.md

(repo root level updates):
- SHA256SUMS (4 section8 entries 追加 + cascade)
- .gitattributes (section8 directive 追加)

## design contract references

- planning_prep.md: sha256 `54da57ccd8aa082fbc664cb53b453c824acee0502eaa10a9e5c072f09f7b8f43` (20,632 B、本 round 不変 preserve)
- planning_prep amendment 01: sha256 `dc19a9df9f4e59d54001d569e5633c3e0f9f91daf50dc5d6250446819ada0bab` (13,499 B、original を overlay supersede)

両 file は claude.ai 側 deliverable、public repo 外 (Pattern 36 awareness: internal working document、phantom path 化を防止するため input_files_pin で明示マーキング)。

## forensic chain rules (本 round 全 step で絶対遵守)

- rule 1 IMMUTABLE: 確定 anchor 不可侵 (anchor 21-24 全 element preserve)
- rule 6 IMMUTABLE: membrane_v48.tex/pdf SHA 不変 (X2 preserve)
- rule 26: 4-route 最小一致 (Phase Q PASS_EXCEEDED preserve)
- rule 92 strict: push 系は specific ref only、`--all`/`--tags`/`--force`/`--mirror` 不使用
- fail-fast sequential: 各 gate PASS 確定後にのみ次 gate、FAIL は即時 stop + 4 段階診断 (D.1-D.4)

## Pattern mitigation 累積

確立済 (Pattern 22-33 + 24a/24b): 全 packet 適用

本 round 新規 codify:
- Pattern 24c (L-Q3-30): subexpression `$(...)` 使用統一
- Pattern 34 (L-Q3-31): BOM 必要時 `WriteAllText` 経由
- Pattern 30 refinement (L-Q3-32): scope = `.ps1` only
- Pattern 35 (L-Q3-33): `InvariantCulture` explicit
- Pattern 36 (L-Q3-34): forensic path 実存性 verify
- Pattern 29 refinement (L-Q3-35): G.0 Tier 1 (hard) + Tier 2 (informational)

## closure criteria

(a) 全 step (1-7) で全 gate PASS (step 1 effective PASS = G.0 wrapper-context benign + G.1-G.11 strict PASS)
(b) envelope 6 files single commit + annotated tag が remote publish 済
(c) 2-protocol external reproducibility 確立 (Protocol 1 = git wire、Protocol 2 = HTTPS raw URL)
(d) baseline anchor 24 v0.1 closure commit cbc2700 が本 round 全 step で preserve
(e) closure record TXT + verification report PDF 生成 (post-closure deliverable)
(f) Pattern 36 self-apply gate (step 1) 全 PASS: input_files_pin 内 referenced path 全 public 実存性 verified
(g) Pattern 29 refinement self-apply gate (step 2-7 G.0) で Tier 1 全 PASS: PS CWD == git toplevel parity が hard gate として step 2-7 全てで satisfied

## 本 round の特徴

「forensic chain self-discovery round」: 4 新 pattern を round 進行中に self-discover:

- L-Q3-33 (Pattern 35、turn N+18.0 sync verify paste-back で発見)
- L-Q3-34 (Pattern 36、turn N+18.x L-Q3-8 path verify で発見)
- L-Q3-35 (Pattern 29 refinement、turn N+18.x step 1 G.0 で発見)
- Pattern 37 candidate (turn N+18.x MEMORY.md verify で発見、anchor 26 v0.x defer)

加えて documentation drift 2 instance canonical recording (L-Q3-10 disposition note 内):
- 1st: anchor 23 v0.1 L557 misattribution (L-Q3-8 content を L-Q3-10 と誤記)
- 2nd: `axis_4_closure_summary.json` L175 phantom path reference (`lessons_codified.md` 不存在)

anchor 24 v0.1 (Pattern 33 single structural fix) より rich round。

## forward defer queue (anchor 26 v0.x candidate)

- Pattern 37: PS 5.1 `[Console]::OutputEncoding` default (OEM code page、Windows-JP では CP932) vs UTF-8 file content display mismatch (mojibake)
  - mitigation: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` 明示 + `Get-Content -Encoding UTF8` 明示
  - empirical: anchor 25 v0.1 turn N+18.x、MEMORY.md verify 初回出力で CP932 mojibake、UTF-8 設定後 clean Japanese 表示確認

## forensic chain extension plan

post-closure 想定 chain (4-deep):
- HEAD = anchor 25 v0.1 closure (新 commit、tag companion-v4.9-q4-codify-round-2026-05-1X)
- HEAD~1 = cbc2700 (anchor 24 v0.1 closure、本 round baseline)
- HEAD~2 = 3aef5142 (anchor 23 v0.1 closure)
- HEAD~3 = 491ff34c (anchor 22 v0.2 closure)
- origin/main = anchor 25 v0.1 closure
