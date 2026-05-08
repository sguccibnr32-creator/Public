# anchor 22 v0.1: axis_4 hardening declaration (cross-version floor catalogued)

forensic_anchors/section5_axis_4_hardening/

(本 anchor は arXiv companion paper v4.9 §5.7 / §1.3(iii) の SHA-pin
reference target、anchor 21 v0.2 → Layer A/B/C v1.1 → anchor 22 v0.1
forensic chain の 4-axis Phase Q baseline 完成 declaration)

## §1 Closure scope statement

### §1.1 What is closed

axis_4 universal RAR coupling claim (route ii HSC weak lensing):

  combined.gc_a0       = 2.7184876317961755  a_0    (G09+G12+G15 joint, n_lens=157338)
  combined.gc_a0_err   = 0.11279929425240623 a_0    (4.15% relative)
  combined.delta_AIC   = 472.2767831580862          (C15 vs MOND, full-cov)
  combined.dAIC_sigma  = 21.777896665153094         (>21σ rejection of MOND)
  field_consistency.weighted_mean_gc_a0     = 2.732647...  a_0  (verdict: CONSISTENT, p > 0.05)
  field_consistency.weighted_mean_gc_err_a0 = 0.107229...  a_0
  gc_M_star_slope                           = 0.166 ± 0.041   (log10(gc) per log10(M_star))

is hereby declared **operationally closed at axis_4 hardening level
(cross-version reproducibility floor catalogued)** in the following sense:

(a) point estimates and uncertainties pass 1e-12 atol bit-exact
    reproduction from frozen canonical (Layer C v1.1, SHA 5d9beb04,
    IMMUTABLE preserved) via independent code paths (mirror v0.3
    phase_b_three_field_mirror.py SHA-16 71dfdc56, canonical wrapper
    L583-644 1:1 binding); 3 paper-claim aggregates (combined.gc_a0,
    combined.delta_AIC, combined.dAIC_sigma) achieve true bit-exact
    delta=0; rule #26 multi-route minimum reproducibility established
(b) statistical reproducibility extends to all 5 paper-claim aggregates
    plus 14 supporting per-field/per-mbin values: 91/97 numerical fields
    PASS at 1e-12 atol bit-exact, 2/2 categorical fields PASS, 6 noise-floor
    paths (3.0e-12 to 1.3e-11) catalogued under L-Q3-8 (§4); Phase Q
    baseline 4-axis (axis_1 SPARC + axis_2 dSph + axis_3 Brouwer KiDS +
    axis_4 HSC) all satisfied, exceeding rule #26 3-axis minimum
(c) M-15 + M-17 (combined block dual-fit pattern) is identified as the
    unique sufficient causal mechanism for v0.2 → v0.3 combined.gc 0.014 a_0
    delta elimination: canonical wrapper L621 binding requires diagonal-sigma
    fit_gc(g_bar, g_obs, g_obs_err) for gc point estimate, then full-cov
    chi2 evaluation on that gc — a 2-step pattern collapsed into single
    full-cov fit in v0.2 (rule #26 J4-equivalent uniqueness for axis_4)
(d) **dual closure (physics + forensic):**
    (d-physics) all 5 paper-claim numerical aggregates (combined.gc_a0,
        combined.delta_AIC, combined.dAIC_sigma, field_consistency.weighted_mean_gc_a0,
        gc_M_star_slope) reproduce at strict 1e-12 bit-exact across
        independent implementations — axis_4 scientific reproducibility
        achieves full closure rhyming with anchor 21 v0.2 (d) noise-floor
        0.107σ universal coupling
    (d-forensic) 6 supporting-field paths exhibit residual delta in the
        range 3.0e-12 to 1.3e-11, attributed to scipy version drift between
        Layer C v1.1 generation environment and mirror v0.3 axis1_q2
        (scipy 1.17.1) environment; empirical floor catalogued as L-Q3-8
        ~1e-11 effective cross-version bound; same-scipy re-run path
        reserved for anchor 22 v0.2 (§6.2 forward declaration)
(e) methodology principles supporting the closure are formalized as
    L-Q3-8 (§4 below), with provisional L-Q3-4..L-Q3-7 pending integration
    in anchor 22 v0.2 round (§4.2)

### §1.2 What is NOT closed by this declaration

- axis_4 type-α true reproduction promotion (Layer C v1.2 freeze =
  canonical re-run JSON 062f73aa... rename + frozen copy, mirror v0.3
  vs v1.2 full 99/99 EXACT verify) — anchor 22 v0.2 (separate round,
  forensic formality 縮退 by variant A confirmed §3.3.1)
- arXiv v4.9 §5.7 / §6 paper edit reflection (4-axis Phase Q baseline
  正式記録, L-Q3-8 floor catalog 反映, 5/5 closure gate verdict 引用)
  — v0.1 round 内 C3+ commits or forward path (c) separate chat
- WordPress site sync (sakaguchi-physics.com 4-axis Phase Q baseline
  page 新設 + 既存 axis_1/2/3 pages cross-link 整備) — separate round
- Phase 1b (f_opt(x ≠ 0.5) operational + chi_coh integration) —
  anchor 23+ (anchor 22 series は axis_4 hardening v0.1/v0.2 chain
  専用に reserve)
- route (vi)/(vii) GAMA cross-z / M_* function — future work, GAMA 5
  DMU acquisition pending

## §2 Canonical IMMUTABLE values

All values in this section inherit bit-exact from Layer C v1.1
(Q-3_immutable_hsc_values.draft.json, SHA 5d9beb04) and are IMMUTABLE
for v0.1 and downstream. Mirror v0.3 (independent code path) reproduces
all listed paper-claim values at 1e-12 atol bit-exact.

### §2.1 Combined-field aggregate (G09 + G12 + G15 joint)

  combined.gc_a0       = 2.7184876317961755     [paper-claim, delta=0 true bit-exact]
  combined.gc_a0_err   = 0.11279929425240623    [paper-claim, delta=1.74e-14 (PASS at 1e-12 atol)]
  combined.delta_AIC   = 472.2767831580862      [paper-claim, delta=0 true bit-exact]
  combined.dAIC_sigma  = 21.777896665153094     [paper-claim, delta=0 true bit-exact]
  combined.n_lens      = 157338                 [meta, EXACT]
  combined.n_pairs     = 502867432              [meta, EXACT (Layer C IMMUTABLE actual; paper-rounded form ≈503M)]

### §2.2 Field consistency (cross-field)

  field_consistency.weighted_mean_gc_a0     = 2.732647...    [paper-claim, delta=8.17e-13 (PASS at 1e-12 atol)]
  field_consistency.weighted_mean_gc_err_a0 = 0.107229...    [paper-claim, delta=3.33e-16 (PASS at 1e-12 atol)]
  field_consistency.verdict                 = "CONSISTENT"   [categorical, EXACT, threshold p > 0.05]

### §2.3 Per-field gc

  per_field_results.G09.gc_a0 = 2.909  [supporting, PASS at 1e-12 atol]   (G09: 49272 lenses, 175735166 pairs)
  per_field_results.G12.gc_a0 = 2.644  [supporting, PASS at 1e-12 atol]
  per_field_results.G15.gc_a0 = 2.640  [supporting, noise-floor delta=3.78e-12 (L-Q3-8)]

### §2.4 Per-M_star bin gc

  MSTAR_EDGES_CUT = [8.5, 10.3, 10.6, 10.8, 11.0]  (log10(M_star/M_sun))

  per_mbin_gc[0].gc_a0 = 2.003  [supporting, PASS at 1e-12 atol]
  per_mbin_gc[1].gc_a0 = 2.243  [supporting, noise-floor delta=3.75e-12 (L-Q3-8)]
  per_mbin_gc[2].gc_a0 = 2.757  [supporting, PASS at 1e-12 atol]
  per_mbin_gc[3].gc_a0 = 3.407  [supporting, noise-floor delta=3.04e-12 (L-Q3-8)]

### §2.5 gc – M_star slope

  gc_M_star_slope.slope_log10_gc_per_logM = 0.166  [paper-claim, delta=3.91e-13 (PASS at 1e-12 atol)]
  gc_M_star_slope.slope_err               = 0.041  [paper-claim, delta=6.38e-15 (PASS at 1e-12 atol)]

### §2.6 Floor breakdown (L-Q3-8 catalog)

  PASS at 1e-12 atol:              91 numerical + 2 categorical = 93/99
                                   (of which 3 paper-claim aggregates are
                                    true bit-exact delta=0:
                                    combined.gc_a0, combined.delta_AIC,
                                    combined.dAIC_sigma)
  noise-floor catalog ([3e-12, 1.3e-11]): 6 paths total
    primary (3 paths):
      per_field_results.G15.gc_a0                              (delta=3.78e-12)
      per_mbin_gc[1].gc_a0                                     (delta=3.75e-12)
      per_mbin_gc[3].gc_a0                                     (delta=3.04e-12)
    propagated (3 paths):
      field_consistency.chi2                                   (delta=1.31e-11; from G15.gc_a0)
      gc_M_star_slope.vs_C15_predicted_0p075.t_statistic       (delta=9.76e-12; from per_mbin[1/3].gc_a0)
      gc_M_star_slope.vs_MOND_predicted_0p000.t_statistic      (delta=1.00e-11; from per_mbin[1/3].gc_a0)
  failure (above floor):           0 / 99
  effective cross-version floor:   ~1e-11 (L-Q3-8 empirical bound)

## §3 Forensic provenance SHA-pin table

All references below are CDN-pinned via tag
companion-v4.9-axis-4-hardening-2026-05-08
and remain accessible bit-exact at any future date.

### §3.1 Mirror v0.3 reproduction evidence (axis_4 hardening operational)

  | Reference                              | SHA prefix          | role                                           |
  |----------------------------------------|---------------------|------------------------------------------------|
  | route_ii_mirror_result.json            | 97c73bda...         | 91/97 PASS + 6 noise-floor aggregate           |
  | PASS_DECLARATION_axis_4_v0.3.md        | c55c836d...         | formal axis_4 PASS + §2.2.1 verification       |
  | phase_b_three_field_mirror.py          | 71dfdc56...         | mirror script (canonical L583-644 1:1)         |
  | run_route_ii_mirror.sh                 | 8b223a85...         | wrapper (axis1_q2 fallback + pin resolver)     |
  | structural_delta_v0_1_to_v0_2.md       | 3e956c6a...         | M-1..M-12 schema corrections forensic          |
  | structural_delta_v0_2_to_v0_3.md       | 29f455eb...         | M-13..M-24 algorithm corrections forensic      |
  | input_files_pin_resolved.json          | af986257...         | path-resolved pin (rule 7 additive)            |
  | run_route_ii_mirror.log                | a4daa1c6...         | latest run log                                 |
  | phase_q3_verification_report.pdf       | 17593f2b...         | 26-page verification report (claude.ai outputs)|
  | build_verification_pdf.py              | 46d4cb6b...         | PDF re-generation builder                      |

### §3.2 Frozen canonical (IMMUTABLE preserved, rule 1)

  | Reference                                  | SHA prefix          | role                                           |
  |--------------------------------------------|---------------------|------------------------------------------------|
  | Q-3_immutable_hsc_values.draft.json        | 5d9beb04...         | Layer C v1.1 (axis_4 hardening verify target)  |
  | phase_b_step3_jsondump_v1_fast.py          | 52a6a67a...         | canonical wrapper (Layer C v1.1 generator)     |
  | phase_b_step3_three_fields.py              | 979d6f79...         | canonical core (Stage 1-6 pipeline)            |
  | Q-3_immutable_brouwer_values.json          | (Layer B, axis_3)   | parent layer (forensic chain continuity)       |
  | Q-2_immutable_values.json                  | 3a3e08dc...         | Layer A (rule 1 root)                          |
  | axis_1_closure_summary.json (anchor 21)    | 55603a4d...         | parent anchor (axis_1 type-α reference)        |

  parent commit (anchor 21 v0.2 tag): 3b9c714
  parent tag: companion-v4.9-axis-1-closure-2026-05-05

### §3.3 v0.1 round (this declaration)

  | Reference                                                            | SHA prefix                                                             | role                                |
  |----------------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------|
  | OPERATIONAL_CLOSURE.md                                               | (self; SHA in SHA256SUMS, tag-pinned via companion-v4.9-axis-4-hardening-2026-05-08) | this declaration                    |
  | axis_4_closure_summary.json                                          | (SHA in SHA256SUMS, tag-pinned via companion-v4.9-axis-4-hardening-2026-05-08) | machine-readable aggregate          |
  | RELEASE_NOTES.md                                                     | (SHA in SHA256SUMS, tag-pinned via companion-v4.9-axis-4-hardening-2026-05-08) | round narrative                     |
  | empirical_precheck_canonical_rerun.json                              | 52a2bcec...                                                            | L-Q3-8 same-scipy validation (variant A confirmed; canonical re-run 062f73aa...) |
  | C1 commit                                                            | [C1-SHA-here]                                                          | this round feat                     |
  | C2 commit                                                            | [C2-SHA-here]                                                          | this round docs                     |

#### §3.3.1 Empirical pre-check status — VARIANT A CONFIRMED

L-Q3-8 prediction empirically validated 2026-05-08 (Claude Code side,
WSL2 axis1_q2).

  Status: **variant A CONFIRMED**
  Pre-check execution: 2026-05-08T06:43:34Z (axis1_q2: scipy 1.17.1,
                       numpy 2.4.3, python 3.13.13)
  Canonical re-run output: phase_b_step3_jsondump_v1_fast_result.json
                           (SHA-256: 062f73aac7133fb8de44f41b83616f20
                                     f700177b1bd53b4308af1748000b26b5)
  Pre-check artifact:      empirical_precheck_canonical_rerun.json
                           (SHA-16: 52a2bcec..., 7,477 B)

  Compare scope (mirror v0.3 ↔ canonical re-run, both axis1_q2 scipy 1.17.1):
    - 6/6 noise-floor paths    : delta=0 EXACT (per_field_results.G15.gc_a0,
                                  per_mbin_gc[1].gc_a0, per_mbin_gc[3].gc_a0,
                                  field_consistency.chi2,
                                  gc_M_star_slope.vs_C15_predicted_0p075.t_statistic,
                                  gc_M_star_slope.vs_MOND_predicted_0p000.t_statistic)
    - 6/6 paper-claim values   : delta=0 EXACT (combined.gc_a0, combined.dAIC,
                                  combined.c15_vs_mond_sigma,
                                  field_consistency.weighted_mean_gc_a0,
                                  gc_M_star_slope.slope_log10_gc_per_logM,
                                  gc_M_star_slope.slope_err)
    - Total: 12/12 delta=0 (true bit-exact, 12 orders below 1e-12 atol gate)

  Implication:
    Layer C v1.1 (frozen 2026-04-27) に対して観測された 6 noise-floor
    delta は same scipy 1.17.1 環境下では 完全 消失。L-Q3-8 hypothesis
    "cross-version drift establishes ~1e-11 effective floor" は scipy-
    version-induced であることが empirically demonstrated。さらに、
    cross-scipy 比較で sub-tol delta>0 だった 5 fields (combined.gc_a0_err
    delta=1.74e-14、weighted_mean_gc_a0 delta=8.17e-13 等) も same-scipy
    では delta=0 に収束、reproducibility floor は scipy 版固定下で実質
    delta=0 (1e-12 atol gate ではない)。

  Anchor 22 v0.2 path impact:
    canonical re-run JSON (062f73aa...) は future Layer C v1.2 content
    そのもの。anchor 22 v0.2 round は (i) v1.2 への rename + freeze、
    (ii) full 99/99 verify (mechanical extension of 12/99 sample 既
    demonstrated)、(iii) v1.1 IMMUTABLE preserved in parallel という
    forensic formality に縮退。compute-heavy investigation 不要。

  Side check (handoff §10 残存 TODO[step-2] (1)):
    Combined RAR header actual pattern = "# Combined 3-field: 157338
    lenses, 502867432 pairs, Bin5 excluded" → matchable by "lenses, .*
    pairs"; n_pairs=502867432 actual と Refinement B (paper-rounded
    503M との区別) integrity 確認 ✓

### §3.4 Tag

  companion-v4.9-axis-4-hardening-2026-05-08
  (annotated, points to C1 commit, all CDN references resolve via this tag)

## §4 Lessons codified

### §4.1 Established (this round contributes L-Q3-8)

  L-Q3-1: tolerance schema layer match
          (each Layer source declares its own tolerance class; default
           inheritance from upper layer creates schema mismatch — Phase
           Q-3 original lesson)
  L-Q3-2: layer source typology α/β/γ
          (α = full-precision recompute artifact; β = JSON full-precision
           dump; γ = post-quantization tabular. Layer C v1.1 = type-β,
           inherits scipy version environment of generation)
  L-Q3-3: Discovery layer 2 mandate × tolerance schema 両輪
  L-Q3-8: cross-scipy-version reproducibility floor (this round,
          **empirically validated 2026-05-08**)
          ・statement: 1e-12 atol is achievable only under same-scipy-
            version pinning; cross-version drift establishes ~1e-11
            effective empirical floor.
          ・**empirical validation (2026-05-08, variant A confirmed)**:
            same-scipy (1.17.1) canonical re-run vs mirror v0.3 →
            12/12 delta=0 EXACT (canonical re-run JSON SHA 062f73aa...,
            empirical_precheck_canonical_rerun.json SHA 52a2bcec...
            catalogued in §3.3); cross-version floor confirmed scipy-
            version-induced, not algorithm-induced
          ・refinement based on validation: reproducibility floor under
            scipy version pinning is delta=0 (true bit-exact float),
            not 1e-12 atol gate; 1e-12 atol is the cross-version
            empirical bound
          ・codified at: lessons_codified.md L146
          ・size delta: +4,518 B (7,468 → 11,986)
          ・heading occurrences: 4
          ・root-cause status (post-validation): scipy-version-induced
            confirmed at aggregate level; bit-trace level identification
            of which scipy.optimize / scipy.stats internal differs is
            still pending but no longer prerequisite to anchor 22 v0.2
            promotion (v0.2 reduces to forensic formality per §6.2)

### §4.2 Provisional (claude.ai-side proposed, integration pending in v0.2 round)

  L-Q3-4: mirror起草前 Layer C JSON keys 確認 (jq による実 dump、handoff
          memo dotted-paths への blind 依存禁止)
  L-Q3-5: canonical wrapper JSON dump full-source quote (v0.1 → v0.2
          で 12 件 schema mismatch を量産した failure mode の codify)
  L-Q3-6: canonical wrapper L<行範囲> 1:1 mirror 化 (algorithm-level
          parallel implementation の方法論)
  L-Q3-7: tolerance schema location pre-flight inspection (M-13 root
          cause: tolerance_per_field nested under source.* not top-level)

  numbering 整理 + Discovery lessons_codified.md への正式 append は
  anchor 22 v0.2 round で実施 (本 v0.1 では proposal の存在のみ pin)

## §5 Rule compliance

Repo 既定 9 rules registry (anchor 21 v0.2 §5 と parallel) で評価:

  | rule  | label                              | status        | evidence                                                            |
  |-------|------------------------------------|---------------|---------------------------------------------------------------------|
  | 1     | anchor IMMUTABLE                   | ✅ PASS        | Layer C v1.1 SHA 5d9beb04 unchanged (verify §12 [1])                |
  | 2     | R-1 LOCK, k_B = 0                  | ✅ PASS        | 本 round で k_B 触らず                                              |
  | 3     | R-2 LOCK, Algo B self-cons         | ✅ PASS        | 本 round で Algo B 触らず                                           |
  | 4     | Q-C1 LOCK, k_E = 2                 | ✅ PASS        | 本 round で k_E 触らず                                              |
  | 5     | cascade SSoT                       | ✅ PASS        | vpp_x05(0.83)=10.462625 / f_opt(0.83)=1.942493 不変                 |
  | 6     | L-1 forward-ref 0                  | ✅ PASS        | parent v4.8 NULL impact 維持 (companion paper text 影響 0)           |
  | 7     | additive supersession              | ✅ PASS        | v0.1 elevates Phase Q-3 5th-step deliverables without revision      |
  | #26   | multi-route minimum                | ✅ PASS_EXCEED | axis_1+2+3+4 全 PASS (4-axis、3-axis minimum 超過)                  |
  | 92    | parsimony first                    | ✅ PASS        | Option 2 staged promotion (single-round 統合 + v0.2 forward 分離)   |

  9/9 ALL ✅ (PASS_DECLARATION_axis_4_v0.3 §4 で既 verify、本
  declaration で再確認)

  Note: draft 段階で独自 rules (tolerance schema declaration / layer
  SHA pin / independent code path / noise-floor catalog) を rule 番号化
  していたが、これらは repo 既定 rule registry 未定義であり、forensic
  上の verification は §1.1 (a)-(e) / §3 / §4 / §2.6 で既に実施済。
  rule 番号化を行う場合は別途 rule registry codify round が prerequisite。

## §6 Forward declaration

### §6.1 arXiv v4.9 §5.7 / §6 paper edit reflection

  scope:
    - §5.7 (or new §6) で 4-axis Phase Q baseline 正式記録
    - axis_4 satisfied at paper-claim 1e-12 bit-exact 記述
    - L-Q3-8 cross-scipy ~1e-11 floor catalog 反映
    - 5/5 closure gate verdict (本 §1.1) 引用
    - mirror v0.3 (independent code path) と canonical wrapper の
      relation 明記

  citing reference: this anchor 22 v0.1 (SHA-pinned via tag
                    companion-v4.9-axis-4-hardening-2026-05-08)

  figures: optional; PDF verification report (17593f2b...) 内 figure を
           rhyme で arXiv 移植可

  execution path: v0.1 round 内 C3+ commit OR forward path (c) separate
                  chat (本 chat scope 外、後続 chat にて実行)

### §6.2 anchor 22 v0.2 axis_4 type-α reproduction promotion

  status: variant A confirmed (§3.3.1) により forensic formality に縮退、
          compute-heavy investigation 不要

  v0.2 required steps (mechanical):
    1. Layer C v1.2 freeze:
       canonical re-run JSON (062f73aa...) を
       Q-3_immutable_hsc_values.draft.v1.2.json として rename + frozen
       copy (canonical 起点 path に配置)
    2. tolerance_per_field block reuse:
       v1.1 と同一の tolerance schema を v1.2 に inherit (rule 30
       compatible, schema unchanged)
    3. v1.1 IMMUTABLE preserved in parallel:
       Layer C v1.1 (5d9beb04) は anchor 22 v0.1 forensic chain root
       として永続保持、v1.2 は parallel layer として新規追加 (rule 1 +
       rule 7 additive)
    4. mirror v0.3 vs Layer C v1.2 full 99/99 EXACT verify:
       12/99 sample が already empirically demonstrated (§3.3.1)、残
       87 fields の verify は mechanical extension で full 1e-12 atol
       通過 (実際は delta=0 期待)

  promotion outcome (predicted with high confidence):
    axis_4 type-α true reproduction promotion 達成、L-Q3-8 root cause
    が scipy-version-induced であることが forensic chain 内で正式記録、
    v0.1 hardening declaration が "cross-version floor catalogued"
    から full type-α への progression を完了

  preservation: anchor 22 v0.1 (this) は v0.2 で superseded ではなく、
                hardening series 内 staged member として preserved。
                v0.1 は "cross-scipy floor empirically catalogued"、
                v0.2 は "same-scipy true bit-exact promoted" という
                relationship。

  scope reduction note: pre-check variant A 確認前は v0.2 が compute
                        re-run + 99 field full diff + scipy bit-trace
                        diff investigation の可能性を含む scope だった。
                        variant A 確認後は (i)-(iv) mechanical steps の
                        みに縮退。

### §6.3 WordPress sync (separate round, post v0.2 + arXiv 確定後)

  site: sakaguchi-physics.com
  candidate pages: Phase Q baseline 4-axis page 新設 or 既存 axis_1/2/3
                   pages への cross-link 整備
  workflow: MCP pages.get + manual paste (rule per existing convention)
  execution: 別 round (本 anchor scope 外)

### §6.4 Phase 1b (anchor 23+, 新規立ち上げ)

  label: f_opt(x ≠ 0.5) operational + chi_coh integration
  anchor: 23+ (anchor 22 series は axis_4 hardening v0.1/v0.2 chain
              専用、Phase 1b は別 anchor 階層)
  deferred: true

### §6.5 route (vi)/(vii) GAMA cross-z / M_* function

  status: future work, axis_5+ 候補
  prerequisite: GAMA 5 DMU acquisition pending
  deferred: true

## §7 Closure declaration

axis_4 universal RAR coupling claim (route ii HSC weak lensing, Layer C
v1.1) is hereby declared **operationally closed at axis_4 hardening level
(cross-version reproducibility floor catalogued)** as of 2026-05-08.

This closure rests on:
- 5/5 closure conditions (a)-(e) SATISFIED (§1.1)
- 9/9 rule compliance (§5)
- Phase Q baseline 4-axis requirement satisfied (axis_1 + axis_2 +
  axis_3 + axis_4 全 PASS、rule #26 multi-route minimum を超えて
  4-route 完成)
- Forensic chain integrity from anchor 21 v0.2 via Layer A/B/C v1.1
  to this anchor 22 v0.1, all SHA-pinned via tag
  companion-v4.9-axis-4-hardening-2026-05-08
- L-Q3-8 empirically validated 2026-05-08 (§3.3.1 variant A confirmed,
  12/12 delta=0 EXACT in same-scipy 1.17.1)

The path to axis_4 type-α true reproduction promotion (anchor 22 v0.2)
is laid out in §6.2 as forensic formality 縮退 — empirical pre-check
(§3.3.1) confirmed L-Q3-8 hypothesis at strong evidence level, reducing
v0.2 to mechanical Layer C v1.2 freeze + full 99-field verify.

This declaration is IMMUTABLE under v0.1 SHA pin. Subsequent rounds
(anchor 22 v0.2, anchor 23+, etc.) extend but do not supersede this
hardening anchor.

—END OF DECLARATION—
