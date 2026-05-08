# anchor 22 v0.1 RELEASE NOTES

axis_4 hardening declaration (cross-version floor catalogued)

## Round meta

- **Anchor**: 22
- **Version**: v0.1
- **Round label**: axis_4 hardening declaration (cross-version floor catalogued)
- **Round date**: 2026-05-08
- **Tag**: companion-v4.9-axis-4-hardening-2026-05-08
- **Parent anchor**: 21 v0.2 (commit 3b9c714, tag companion-v4.9-axis-1-closure-2026-05-05)
- **Status**: CLOSED at axis_4 hardening level
- **Next milestone**: anchor 22 v0.2 axis_4 type-α reproduction promotion (deferred to separate round)

## TL;DR

axis_4 (route ii HSC weak lensing) operational hardening が完了。Phase Q
baseline が rule #26 multi-route minimum (3-axis required) を超えて 4-axis
(axis_1 SPARC + axis_2 dSph + axis_3 Brouwer KiDS + axis_4 HSC) で完成した。

Layer C v1.1 (SHA 5d9beb04) 内 99 fields のうち 93/99 が strict 1e-12 atol
で再現 (91 numerical + 2 categorical)。残 6 noise-floor paths (3.0e-12 から
1.3e-11) は L-Q3-8 cross-scipy-version reproducibility floor として codify
(lessons_codified.md L146, +4,518 B, occurrences=4)。

5 paper-claim aggregates は全件 1e-12 atol PASS (3 件 delta=0 true
bit-exact、2 件 sub-tol delta):

  combined.gc_a0       = 2.7184876317961755  a_0    [delta=0]
  combined.delta_AIC   = 472.2767831580862           [delta=0]
  combined.dAIC_sigma  = 21.777896665153094          [delta=0]
  field_consistency.weighted_mean_gc_a0 = 2.732647...  [delta=8.17e-13]
  gc_M_star_slope      = 0.166 ± 0.041              [delta=3.91e-13 / 6.38e-15]

5/5 closure conditions SATISFIED、9/9 rule compliance、empirical
pre-check (variant A confirmed 2026-05-08、12/12 delta=0 EXACT in
same-scipy 1.17.1) により L-Q3-8 が empirically validated、anchor 22
v0.2 axis_4 type-α reproduction promotion path は forensic formality
に縮退。

## What's new in this round (v0.1)

### 4-axis Phase Q baseline complete

axis_4 PASS により、Phase Q baseline は rule #26 minimum (3-axis required)
を超えて 4-axis 完成状態となった。これは companion paper §5.7 / §6 で
"4-route Phase Q baseline" として正式記録される foundation を提供する。

  axis_1 (route i SPARC):       PASS [delta 1.39e-17]   anchor 21 v0.2
  axis_2 (route v dSph):        PASS [delta 1.04e-15]   (Layer A inherited)
  axis_3 (route iii Brouwer):   PASS [58/58 printed prec] (Layer B inherited)
  axis_4 (route ii HSC):        PASS [93/99 strict 1e-12 + 6 noise-floor]   ←★ this round

### L-Q3-8 cross-scipy-version reproducibility floor codified AND empirically validated

本 round で初めて empirical 確立された floor lesson。1e-12 atol の
strict bit-exact reproducibility は same-scipy-version pinning 下でのみ
achievable で、cross-version では ~1e-11 が effective floor となる。

**empirical validation (2026-05-08, variant A confirmed)**: same-scipy
(1.17.1) canonical re-run vs mirror v0.3 で 12/12 delta=0 EXACT を達成。
これは L-Q3-8 hypothesis の strong empirical backing で、reproducibility
floor under scipy version pinning は実質 delta=0 (true bit-exact float、
1e-12 atol gate ではない) であることを実証。さらに、cross-scipy で
sub-tol delta>0 だった 5 fields (combined.gc_a0_err 1.74e-14、
weighted_mean_gc_a0 8.17e-13 等) も same-scipy では delta=0 に収束。

provisional root-cause hypothesis として scipy.optimize curvature
evaluation の version 差が candidate (PASS_DECLARATION_axis_4_v0.3
§2.2.1 で gc_err/gc 0.0812 vs EXACT 0.0742、ratio 1.094× の strong
evidence at aggregate level) であり、bit-trace level identification は
依然 pending だが、もはや anchor 22 v0.2 promotion の prerequisite では
ない (v0.2 は forensic formality に縮退)。

### M-15 + M-17 root cause for combined block 0.014 a_0 delta identified

v0.2 → v0.3 redraft で combined.gc が 0.014 a_0 ずれていた真因は、
canonical wrapper の 2-step pattern (diagonal-sigma fit_gc で gc 求解
→ full-cov chi2 評価) を v0.2 が single full-cov fit に collapse して
いたこと。canonical wrapper L621 binding 確認により M-15 + M-17 が
unique sufficient causal mechanism として特定 (rule #26 J4-equivalent
uniqueness for axis_4)。

### Anchor 22 v0.1 paired format (4-file 体制)

anchor 22 v0.1 は OPERATIONAL_CLOSURE.md + axis_4_closure_summary.json +
RELEASE_NOTES.md (this file) + empirical_precheck_canonical_rerun.json
の 4-file paired format で構成、anchor 21 v0.2 の 3-file paired format
を rhyme + 4 件目として same-scipy validation evidence を追加した拡張形。
machine-readable aggregate (JSON) と human-readable canonical declaration
(MD) を分離保持 + L-Q3-8 empirical evidence (JSON) を独立 catalog 化。

## Round narrative (forensic chain)

### Phase Q-3 fifth-step opening

本 round は Phase Q-3 fifth-step として、route ii HSC weak lensing の
axis_4 を 4-route 完成させる目的で開始された。先行する axis_1/2/3 は
既に PASS、axis_4 のみが Layer C v1.1 (SHA 5d9beb04) IMMUTABLE
preserved の状態で independent mirror 検証待ちだった。

検証 scope は canonical wrapper の Stage 5-6 (fit_gc + chi2_comparison
+ gc_M_star_slope + paper claims) のみで、Stage 1-4 (HSC 9.6GB raw
処理) は blackbox 扱い、中間 RAR file から開始。build deps は numpy +
scipy.optimize + scipy.stats のみ (healpy / treecorr / fitsio / pyccl
不要)、これにより独立性が確保された。

### v0.1 起草 (12 schema mismatch M-1..M-12)

最初の v0.1 起草は handoff memo §3 例示 dotted-paths から schema 推測
する形で行われ、12 件の structural mismatch (M-1..M-12) を生んだ。
これは Layer C v1.1 actual JSON keys との binding を実測ではなく推測に
依存した failure mode で、後に provisional L-Q3-4 (mirror起草前 Layer
C JSON keys 確認、jq dump 実施) として codify 候補に挙がる。

### v0.2 redraft (12 algorithm mismatch M-13..M-24)

M-1..M-12 の schema-level fix を行った v0.2 では、Claude Code 側
execution review で 12 件の algorithm-level mismatch (M-13..M-24) が
検出された。代表的な corrections:

  M-13 [high]:    tolerance_per_field は source.* に nested、top-level
                   ではない (default fallback で 18 int / 5 cat /
                   5 pval が load されない問題)
  M-14 [high]:    compare path は 6-prefix whitelist 化必要 (per_field_
                   results. / field_consistency. /
                   combined_3_field_chi2_C15_vs_MOND. / per_mbin_gc[ /
                   gc_M_star_slope. / paper_claims_for_rounded_check.)
  M-15 [fatal]:   combined block の gc は diagonal sigma fit_gc(g_bar,
                   g_obs, g_obs_err) で求める (jk_cov を fit_gc に
                   渡さない); v0.2 の full-cov collapse が 0.014 a_0
                   delta の真因
  M-16 [substan-: minimize_scalar() に xatol option なし (scipy default
   tive]          1e-5)、brentq bracket は [res.x, res.x±2]、fallback
                   ±0.5、gc_err = (10**lhi - 10**llo)/2 (linear gc 空間)
  M-17:           combined chi2_c15 / chi2_mond は full-cov で eval、
                   ただし gc は M-15 の diag fit 結果 (canonical 2-step
                   pattern)
  M-22:           per_mbin_gc は 7-field のみ、MSTAR_EDGES_CUT =
                   [8.5, 10.3, 10.6, 10.8, 11.0]
  M-23:           n_lens は RAR header "# G09: 49272 lenses, 175735166
                   pairs" parse

### v0.3 redraft + PASS Declaration

canonical wrapper phase_b_step3_jsondump_v1_fast.py SHA 52a6a67a の
L583-644 (fit_gc / chi2_comparison / gc_slope full-source) を 1:1
mirror 化することで、12 件の algorithm-level corrections を反映。
結果:

  - 91/97 numerical fields PASS at 1e-12 atol
  - 2/2 categorical fields PASS
  - 6 noise-floor paths (3.0e-12 to 1.3e-11) catalogued
  - 0 / 99 failure (above floor)

これは PASS_DECLARATION_axis_4_v0.3.md (SHA-16 c55c836d, 13,585 B) と
して正式 declaration。§2.2.1 で L-Q3-8 curvature hypothesis verification
(noise-floor cases gc_err/gc=0.0812 vs EXACT 0.0742、ratio 1.094×、
strong evidence at aggregate level、per-mbin[0] caveat 明記) を含む。

### PDF verification report (claude.ai 側 deliverable)

26-page 日本語 PDF (phase_q3_verification_report.pdf, SHA-16 17593f2b,
215,632 B) を v4.4 PDF レイアウト仕様準拠で生成 (IPAGothic / IPAPGothic
font、v4.3 標準余白、STYLE_FLOW + STYLE_CODE_JP 補助 styles)。10 章
本文 + 4 付録で M-1..M-24 forensic record + 5/5 closure gate evidence +
L-Q3-8 codify 過程を視覚化。再生成は build_verification_pdf.py
(SHA-16 46d4cb6b, 51,938 B) で可能。

### Memory persistence

C:\Users\sgucc\.claude\projects\C--Users-sgucc\memory\ 配下に永続化:

  - MEMORY.md: 1-line index 追加 (Phase Q completion)
  - project_phase_q_baseline_4route_complete.md: 新規 66 行
  - feedback_tolerance_schema_layer_alignment.md: +L-Q3-8 cross-scipy-
    version floor、+2 行

### anchor 22 v0.1 起草 (本 file 含む 4-file paired format)

claude.ai 側起草、Claude Code 側 fact-check + WSL2/git 検証 + empirical
pre-check execution の役割分担。起草に先立ち §12 開始確認 4 項目
(Layer C v1.1 SHA / repo HEAD / mirror dir freeze / lessons_codified.md
L-Q3-8 append) が all PASS で confirmed。

起草過程:
  1. 初稿 3-file paired format (OPERATIONAL_CLOSURE.md + axis_4_closure_
     summary.json + RELEASE_NOTES.md) 起草
  2. Claude Code 側 fact-check 4 項目 (rule list registry / paper-claim
     値 1e-12 verify / noise-floor 6 paths actual / dir naming pattern)
     → 4 refinements 反映 (rule registry actual 9 rules、paper-claim
     status nuance、n_pairs Layer C IMMUTABLE actual 502867432、
     noise-floor verified actual paths)
  3. Empirical pre-check execution (Claude Code WSL2 axis1_q2、scipy
     1.17.1 / numpy 2.4.3 / python 3.13.13、~5-10 min) → variant A
     CONFIRMED (12/12 delta=0 EXACT)
  4. 4-file 体制 finalize (empirical_precheck_canonical_rerun.json
     SHA-16 52a2bcec を §3.3 / §4.1 / forward declaration / cross-
     references の 6 箇所に concurrent 反映)

design decisions:

  - file 名: rhyme template 候補 AXIS4_HARDENING_CLOSURE.md ではなく
    OPERATIONAL_CLOSURE.md を採用 (anchor 21 v0.2 actual と true
    parallelism、dir 名 section5_axis_4_hardening/ で scope は
    既に disambiguate 済)
  - dir naming: section5_axis_4_hardening/ (既存 forensic_anchors/
    section{N}_*/ pattern と整合、anchor 21 v0.2 closure dir
    section2_axis_1_operational_closure と最 parallel な closure-level
    rhyme; companion paper §5.7 reflection は §3.4 / §6.1 で別途 record)
  - closure gate (d): physics + forensic dual structure に分割
    (Refinement B 反映)
  - empirical pre-check: §3.3.1 で variant A/B 影響範囲を pre-declared
    し、Claude Code 側 WSL2 run で variant A confirmed → §6.2 v0.2 plan
    が forensic formality に縮退、L-Q3-8 が empirically validated に昇格
  - L-Q3-4..L-Q3-7 (provisional): existence のみ pin、numbering 整理 +
    Discovery lessons_codified.md への正式 append は v0.2 round で実施

## Key forensic findings

### F-1..F-5 (route iii との systematic 差分)

mirror が canonical 整合のため必須:

  F-1: G15 RA range = 211.5 -- 223.5 (paper rounded ではない)
  F-2: nu = McGaugh exponential、g_obs = g_bar / (1 - exp(-sqrt(g_bar/gc)))
  F-3: chi2 metric = linear-space residual (NOT log-space)
  F-4: chi2 space = g_obs-space (NOT ESD-space)
  F-5: MOND baseline = same McGaugh nu pinned at gc = a_0_SI

### Noise-floor 6 paths (L-Q3-8 catalog)

  primary (3 paths):
    per_field_results.G15.gc_a0                            delta=3.78e-12
    per_mbin_gc[1].gc_a0                                   delta=3.75e-12
    per_mbin_gc[3].gc_a0                                   delta=3.04e-12
  propagated (3 paths):
    field_consistency.chi2                                 delta=1.31e-11  (from G15.gc_a0)
    gc_M_star_slope.vs_C15_predicted_0p075.t_statistic     delta=9.76e-12  (from per_mbin[1/3].gc_a0)
    gc_M_star_slope.vs_MOND_predicted_0p000.t_statistic    delta=1.00e-11  (from per_mbin[1/3].gc_a0)

これらは L-Q3-8 cross-scipy-version reproducibility floor 範囲内 (~1e-11
upper bound)、failure ではなく empirical bound 内 expected behavior。
propagation chain は primary 3 paths (G15.gc_a0 + per_mbin[1/3].gc_a0)
からの downstream 影響 (chi2 / t_statistic 計算で primary delta が
amplify される)。AIC 関連 fields は propagation 範囲外 (true bit-exact
delta=0 維持)。

## Verification summary

### 5/5 closure conditions SATISFIED (§1.1 of OPERATIONAL_CLOSURE.md)

  (a) bit-exact reproducible from frozen canonical (Layer C v1.1
      5d9beb04) via independent code paths (mirror v0.3 71dfdc56,
      canonical L583-644 1:1 binding)
  (b) statistical reproducibility: 91/97 num + 2/2 cat = 93/99 strict
      1e-12 PASS, 6 noise-floor catalog; rule #26 4-axis baseline 全 PASS
  (c) unique sufficient causal mechanism: M-15 + M-17 (combined block
      diag-fit + full-cov chi2 2-step pattern, canonical L621 binding)
  (d) dual closure (physics + forensic):
      (d-physics) 5/5 paper-claim aggregates 1e-12 EXACT
      (d-forensic) 6 noise-floor paths under L-Q3-8 floor catalog
  (e) L-Q3-8 codify complete (lessons_codified.md L146, +4,518 B,
      occurrences=4)

### 9/9 rule compliance (repo registry, anchor 21 v0.2 §5 と parallel)

  rule 1   (anchor IMMUTABLE):                          ✅ Layer C v1.1 SHA 5d9beb04 unchanged
  rule 2   (R-1 LOCK, k_B = 0):                         ✅ 本 round で k_B 触らず
  rule 3   (R-2 LOCK, Algo B self-cons):                ✅ 本 round で Algo B 触らず
  rule 4   (Q-C1 LOCK, k_E = 2):                        ✅ 本 round で k_E 触らず
  rule 5   (cascade SSoT):                              ✅ vpp_x05(0.83)=10.462625 / f_opt(0.83)=1.942493 不変
  rule 6   (L-1 forward-ref 0):                         ✅ parent v4.8 NULL impact 維持
  rule 7   (additive supersession):                     ✅ Phase Q-3 5th-step deliverables を revision なしで elevate
  rule #26 (multi-route minimum):                       ✅ EXCEEDED (4-axis, 3-axis minimum 超過)
  rule 92  (parsimony first):                           ✅ Option 2 staged promotion (single-round 統合)

  Note: 起草段階で独自 rules (tolerance schema declaration / layer SHA
  pin / independent code path / noise-floor catalog) を rule 番号化して
  いたが、これらは repo 既定 rule registry 未定義。forensic verification
  は §1.1 (a)-(e) / §3 / §4 / §2.6 で既に実施済。

### Empirical pre-check (forensic strengthening) — VARIANT A CONFIRMED

L-Q3-8 prediction empirically validated 2026-05-08 (Claude Code side、
WSL2 axis1_q2、scipy 1.17.1 / numpy 2.4.3 / python 3.13.13)。

  Pre-check execution: 2026-05-08T06:43:34Z
  Canonical re-run output SHA-256: 062f73aac7133fb8de44f41b83616f20
                                   f700177b1bd53b4308af1748000b26b5
  Pre-check artifact: empirical_precheck_canonical_rerun.json
                      (SHA-16: 52a2bcec, 7,477 B)

  Compare result (mirror v0.3 ↔ canonical re-run, both axis1_q2 1.17.1):

    6 noise-floor paths (vs Layer C v1.1 で delta>0 だった 6 paths):
      → 6/6 delta=0 EXACT (same-scipy 環境では完全消失)

    6 paper-claim values:
      → 6/6 delta=0 EXACT (cross-scipy で sub-tol delta>0 だった
                           combined.gc_a0_err / weighted_mean_gc_a0 等
                           5 fields も same-scipy では delta=0 に収束)

    Total: 12/12 delta=0 (true bit-exact float、12 orders below
                          1e-12 atol gate)

  L-Q3-8 status: empirically validated。reproducibility floor under
                 scipy version pinning は delta=0 (true bit-exact) で
                 1e-12 atol gate ではない。cross-version floor (~1e-11
                 effective) は scipy-version-induced であることが
                 forensic chain 内で empirically demonstrated。

  Anchor 22 v0.2 path impact: forensic formality に縮退 (compute-heavy
                              investigation 不要、§6.2 参照)。

pre-check 完了は本 v0.1 closure gate には影響しない (additive)、ただし
v0.1 forensic strength は実質的に強化された。

## Forward path

### anchor 22 v0.2 (axis_4 type-α reproduction promotion) — forensic formality に縮退

post-v0.1 別 round。pre-check variant A confirmed (2026-05-08, 12/12
delta=0 EXACT) により、v0.2 は当初想定の compute re-run + 99 field full
diff + scipy bit-trace investigation 拡大 scope から、以下の 4 mechanical
steps に縮退:

  (i)   canonical re-run JSON (062f73aa...) を Layer C v1.2 として
        Q-3_immutable_hsc_values.draft.v1.2.json に rename + frozen copy
  (ii)  tolerance_per_field block を v1.1 から inherit (rule 30 schema
        unchanged)
  (iii) v1.1 IMMUTABLE preserved in parallel (rule 1 + rule 7 additive)
  (iv)  mirror v0.3 vs Layer C v1.2 full 99/99 EXACT verify
        (12/99 sample が already empirically demonstrated、残 87 fields
         は mechanical extension)

本 v0.1 は v0.2 で superseded ではなく hardening series 内 staged member
として preserved (v0.1 = "cross-scipy floor empirically catalogued"、
v0.2 = "same-scipy true bit-exact promoted")。

### arXiv v4.9 §5.7 / §6 paper edit reflection

本 v0.1 round 内 C3+ commit OR forward path (c) separate chat。
4-axis Phase Q baseline 正式記録 + L-Q3-8 cross-scipy floor catalog +
5/5 closure gate verdict + mirror v0.3 と canonical wrapper の relation
明記。citing reference は anchor 22 v0.1 (tag-pinned)。figures は PDF
verification report (17593f2b...) からの rhyme 移植可。

### Other deferred items

  - WordPress sync (sakaguchi-physics.com): post v0.2 + arXiv 確定後、
    Phase Q baseline 4-axis page 新設 or 既存 axis_1/2/3 cross-link
  - Phase 1b (anchor 23+, 新規立ち上げ): f_opt(x ≠ 0.5) operational +
    chi_coh integration、anchor 22 series は axis_4 hardening v0.1/v0.2
    chain 専用
  - route (vi)/(vii) GAMA cross-z / M_* function: future work (axis_5+
    候補)、GAMA 5 DMU acquisition pending

## Cross-references

### This round artifacts (forensic_anchors/section5_axis_4_hardening/)

  OPERATIONAL_CLOSURE.md                  — formal declaration (this round)
  axis_4_closure_summary.json             — machine-readable aggregate (this round)
  RELEASE_NOTES.md                        — round narrative (this file)
  empirical_precheck_canonical_rerun.json 52a2bcec... (7,477 B) — L-Q3-8
                                             same-scipy validation (variant A
                                             confirmed 2026-05-08)

### Mirror v0.3 reproduction evidence (E:\Q-3_route_ii_mirror_2026-05-07\)

  route_ii_mirror_result.json        97c73bda... (29,099 B)
  PASS_DECLARATION_axis_4_v0.3.md    c55c836d... (13,585 B)
  phase_b_three_field_mirror.py      71dfdc56... (39,940 B)
  run_route_ii_mirror.sh             8b223a85... (9,625 B)
  structural_delta_v0_1_to_v0_2.md   3e956c6a... (9,412 B)
  structural_delta_v0_2_to_v0_3.md   29f455eb... (17,031 B)
  input_files_pin_resolved.json      af986257... (6,182 B)
  run_route_ii_mirror.log            a4daa1c6... (1,902 B)
  phase_q3_verification_report.pdf   17593f2b... (215,632 B, 26 ページ)
  build_verification_pdf.py          46d4cb6b... (51,938 B)

### Frozen canonical (IMMUTABLE preserved, rule 1)

  Q-3_immutable_hsc_values.draft.json     5d9beb04... (Layer C v1.1)
  phase_b_step3_jsondump_v1_fast.py       52a6a67a... (canonical wrapper)
  phase_b_step3_three_fields.py           979d6f79... (canonical core)
  Q-2_immutable_values.json               3a3e08dc... (Layer A)
  axis_1_closure_summary.json (anchor 21) 55603a4d... (parent anchor)

### Tag

  companion-v4.9-axis-4-hardening-2026-05-08
  (annotated, points to C1 commit)

## Process notes

本 round は claude.ai 側起草 + Claude Code 側 fact-check + WSL2/git 検証
の natural な役割分担で進行。役割分担の rationale:

  - Layer C v1.1 binding fact-check / mirror dir SHA 検証 / lessons_
    codified.md state confirmation: WSL2 + Windows mount + local git
    access が必要 (Claude Code 側のみ可能)
  - anchor 22 v0.1 schema 起草 + rhyme template extraction + L-Q3-x
    lesson narrative + closure gate refinement: anchor 21 v0.2 sample
    analysis + structural reasoning が中心 (claude.ai 側で効率的)

context preservation:

  Phase Q-3 fifth-step (v0.1 → v0.2 → v0.3 → PASS Declaration → PDF
  → memory persist) は前 chat (handoff_phase_q3_completion_2026-05-08.txt)
  で実施、本 round (anchor 22 v0.1 起草) は新 chat で handoff memo を
  起点に開始。これにより前 chat の context 肥大を持ち越さず、forward
  path に集中できた。

## Provenance and immutability

このリリースおよび関連 artifacts は全て tag
**companion-v4.9-axis-4-hardening-2026-05-08** で CDN-pin される。
Layer C v1.1 (SHA 5d9beb04) は anchor 21 v0.2 確立時点から本 round
完了時点まで write 0 件 (rule 1 准守)、anchor 22 v0.2 round でも
preserve される。

本 RELEASE_NOTES.md および同梱の OPERATIONAL_CLOSURE.md /
axis_4_closure_summary.json は v0.1 SHA pin 下で IMMUTABLE。subsequent
rounds (anchor 22 v0.2, anchor 23+, etc.) は extend するが supersede
しない。

—END OF RELEASE NOTES—
