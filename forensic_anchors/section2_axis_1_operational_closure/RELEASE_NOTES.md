# anchor 21 v0.2 — axis_1 full operational closure

**round**: v0.2 promotion (full operational closure of axis_1)
**date**: 2026-05-05
**parent**: anchor 21 v0.1.4 (commit 5783bef, tag companion-v0.4a-validation-2026-05-05)
**tag**: companion-v4.9-axis-1-closure-2026-05-05
**status**: CLOSED

---

## 1. Round purpose

J-system companion paper (C) round の axis_1 hardening を operational layer
で完結させる。anchor 21 v0.1.1 → v0.1.2 → v0.1.3 (Phase 1a, mechanical
alignment) → v0.1.4 (Phase 1a-validation, 12/12 PASS) で積み上げた axis_1
canonical state を、上位 declaration として **full operational closure** に
elevate する round。

新規 computation は per_filter_sensitivity_extension (refinement 3 cross-method)
のみ。本体は declaration + provenance consolidation + Lesson 94 establishment
で構成。

---

## 2. Frozen input (IMMUTABLE)

anchor 21 v0.1.4 (commit 5783bef、tag companion-v0.4a-validation-2026-05-05) を
frozen input。12/12 gate PASS の全結果を canonical reference として継承:

  axis_1_SPARC b_α       = 0.108442979149252  (G1)
  axis_2_dSph  b_α       = 0.11266236254145712 (G2)
  axis_3 universal slope = 0.11055267084535411 (G3)
  abs_diff (universal)   = 0.004219            (G5)
  R² / residual / AIC    = 0.634 / 0.172 / -529.92  (G6)
  OLS SE                 = 0.0153              (J1b)
  bootstrap 95% CI       = [0.0790, 0.1388]    (J2)
  J3 filter 1 reproduces v1.0.3.1 bit-exact (Q-patch causal proof)
  J4 multi-route Δ       = 0.000e+00           (rule #26)
  SE_combined            = 0.0394
  noise-floor sig        = 0.107σ              (J1b auxiliary)

詳細値 + provenance SHA は OPERATIONAL_CLOSURE.md (SHA 2bd9ddd5...) および
axis_1_closure_summary.json 参照。

---

## 3. v0.2 round で達成した事項

(a) **axis_1 full operational closure declaration** (OPERATIONAL_CLOSURE.md)
    arXiv v4.9 §7 から SHA-pin で reference 可能な formal 宣言文書を establish。
    forensic provenance SHA-pin table (4 sub-table) を集約。

(b) **Lesson 94 formal establishment** (g_obs aggregation sensitivity)
    v0.1.4 で "candidate" 表記だった J3 filter 3 finding を、Lesson 91-93 群と
    同型 framing (scope / evidence / principle / scope LIMIT / falsification path)
    で full establishment。

(c) **cross-method confirmation extension** (refinement 3)
    per_filter_sensitivity_extension.{py, csv} で 4-method aggregation/imputation
    比較を実行。Lesson 94 evidence base が rule #26 multi-route minimum
    compliance に到達。dominant effect: aggregation choice (A1 median, -34.62%)
    >> imputation choice (A2/A3/A4, ±20% range)。

(d) **v1 → v2 redesign forensic record** (v1_precondition_fail_diagnostic.log)
    seq-3 v1 で 2 件 structural mismatch (2-feature simple OLS、post-filter
    aggregation ordering) を precondition Q5 beta が abort で検出、v2 で
    3-feature partial OLS + sparc_171 ordering に修正して PASS。1-shot 正解せず
    diagnostic feedback による redesign を経た educational trace として public
    record に保持。

(e) **machine-readable aggregate** (axis_1_closure_summary.json)
    v0.1.1 → v0.2 lineage の全 IMMUTABLE values + forensic SHA-pin を 1 file に
    集約。後続 round (Phase 1b 等) からの consume を簡潔化。

---

## 4. Forward declaration (v0.2 round 直後の後続作業)

### 4.1 arXiv v4.9 main paper §7 axis_1 hardening 反映

本 round (v0.2) の declaration を反映する paper edit を **v0.2 round 内 C3+
commits** で実施。具体 scope:

  §7.6 (universal coupling)      : 12/12 PASS、Q-patch causal proof、
                                    multi-route minimum、0.107σ
  §7.7 (methodology principles)  : (γ) split、filter sensitivity、
                                    multi-route minimum、SE_combined、Lesson 94
  Appendix A (reproducibility)   : forensic anchor SHA reference 更新

figure 群 (jackknife / bootstrap / 5-filter sensitivity plot) は v4.9 figure
sub-round で別途処理。

### 4.2 WordPress sync (sakaguchi-physics.com, blog_id 253652152)

v0.2 + arXiv v4.9 確定後の別 round で実施。対応候補 page (forward-declare):

  - axis_1 universal coupling page (要 page id 確定)
  - methodology principles / lessons page
  - SPARC analysis pipeline overview page

WordPress MCP workflow: pages.get with `{"id": integer}`、pages.update は manual
paste 推奨 (large HTML truncation risk)。

---

## 5. forensic chain rule compliance

  rule 1  (anchor IMMUTABLE)          : ✅ v0.1.4 content untouched
                                          (C2 で post-hoc additive only)
  rule 5  (cascade SSoT)              : ✅ vpp_x05(0.83) / f_opt(0.83) 不変
  rule 7  (additive supersession)     : ✅ v0.2 elevates v0.1.4 without revision
  rule #26 (multi-route minimum)      : ✅ v0.1.4 J4 から継承 + Lesson 94 で
                                          cross-method 拡張により additional
                                          compliance 達成
  rule 92 (parsimony first)           : ✅ Lesson 94 + §7.7 + cross-method を
                                          1 round に統合 (別 round 切り出しは
                                          ~50% admin overhead asymmetry)

---

## 6. File inventory (anchor 21 v0.2)

  RELEASE_NOTES.md                       (本 file)
  OPERATIONAL_CLOSURE.md                 (formal declaration + Lesson 94 + SHA-pin)
  axis_1_closure_summary.json            (machine-readable aggregate)
  per_filter_sensitivity_extension.py    (cross-method script v2、claude.ai draft)
  per_filter_sensitivity_extension.csv   (cross-method results、Windows run output)
  v1_precondition_fail_diagnostic.log    (v1 -> v2 redesign forensic record)

Root modifications:
  ANCHOR_REFERENCES.md                   (anchor 21 v0.2 entry 追加)
  SHA256SUMS                             (6 file entries 追加)

---

## 7. Commit chain

  C1 (feat + tag): [C1-SHA-here]
      anchor 21 v0.2 axis_1 full operational closure
      files: 6 new + 2 modified
      tag  : companion-v4.9-axis-1-closure-2026-05-05 (annotated)
      parent: 3817b41 (Phase 1a-V B2)

  C2 (docs)      : [C2-SHA-here]
      anchor 21 v0.1.4 RELEASE_NOTES post-hoc cross-reference
      files: 1 modified
      parent: [C1-SHA-here]

  C3+ (paper edit): arXiv v4.9 §7.6 / §7.7 / Appendix A 反映
      暫定見立て: 単一 commit (C3) で完結 parsimonious
                   §7.6 が大幅 rewrite の場合は §7.6 (C3) / §7.7+Appendix A (C4)
                   の 2-split も合理的 (paper edit 着手時に LaTeX 編集量を見て判断)
      files: arXiv source tree 内 (詳細は C3+ 設計時に確定)
      parent chain: C2 → C3 → (C4 if split)

CDN bit-exact audit: C1 6 file + C2 1 file = 7/7 expected PASS
                     (C3+ は paper edit のため CDN audit 対象外)

---

## 8. Closure declaration

axis_1 universal coupling claim は本 round (v0.2) を以て operational layer で
**full closure**。後続 axis_1 work は arXiv v4.9 paper edit (C3+) + WordPress
sync (別 round) のみ (operational layer の追加 hardening は不要)。

次の operational hardening target は Phase 1b (f_opt(x ≠ 0.5) operational +
chi_coh integration、anchor 22 として立ち上げ)。

---

END OF RELEASE_NOTES (anchor 21 v0.2)
