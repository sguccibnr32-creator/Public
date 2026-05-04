# J-system Companion Paper §2 Chapter-Level Closure Declaration v0.1.1

**File**: `J_system_paper_section2_closure_v0.1.md`
**Anchor candidate**: 21
**Version**: v0.1.1 (finalize — F1/F2/F6/F8 classification corrected per anchor 8 §2.6.5 verbatim L553-564; chapter-level §2 LOCK ESTABLISHED)
**Date**: 2026-05-03
**Author**: 坂口 忍 (Shinobu Sakaguchi) / 坂口製麺所、兵庫県宍粟市
**License**: CC-BY 4.0
**Encoding**: UTF-8 / LF / no BOM
**SHA**: TBD (set on file finalize after this v0.1.1 patch)

**Forward-ref level**: L-1 LOCK forward-ref 0 strict
**LOCK tier introduced**: chapter-level §2 LOCK (ESTABLISHED, formal effective)
**Companion-internal designation**: anchor 21
**Supersedes**: nothing (additive layer; anchor 8 v0.1 IMMUTABLE retained)
**Parent v4.8 impact**: NULL 🟢 (companion 純 additive principle preserved)

---

## §0 Preamble

This anchor (anchor 21) is the **§2 chapter-level closure declaration** for the J-system companion paper. It does not re-state the content of anchors 5, 6, 7, or 8 — those remain immutable per their respective LOCK tiers (R-1 / R-2 / Path A / Path (iii) / I-1 / boundary precision 3-regime / cascade SSoT P1-P3 / forward-ref 0).

Anchor 21 performs four functions:

1. **§2 dependency closure audit consolidation** — re-runs the 10-axis framework established in anchor 8 §2.6.4 against current downstream state (anchor 14/16/17), confirming all 10 axes pass at chapter level.
2. **F-flag chapter-level hardening** — moves F4 and F12 (companion-internal) from `status declaration` / `active inject (axis vii)` states (per anchor 8 §2.6.5) to `RESOLVED` at chapter level via 10-axis closure basis. F1/F2/F6/F8 remain `placeholder 維持` by design (LOCK §2.6-B chapter-level closure scope clean preservation; explicit handoff to §2.5 v0.2 round per Sub-issue S-1/S-3 category A semantic coherence and Q-C1 LOCK preservation).
3. **§2 → downstream output spec final lock** — re-affirms the §3-§7+AppB forward-spec from anchor 8 §2.6.2 against anchor 17 (§3 v0.2, SHA 178dad11) / anchor 14 (§4 v0.4, SHA 295bc05c) / anchor 16 (§5 v0.2.1, SHA 69678018) latest states.
4. **TIER progression vehicle** — provides the closure declaration that unblocks TIER-2 → TIER-3 milestone progression (per anchor 19 §1.5 declared next phase).

Anchor 21 is the FIRST chapter-level closure declaration in the J-system companion paper anchor chain. §3-§7 chapter-level closures (if pursued) will follow this template.

---

## §1 §2 Section-by-Section Recap

### §1.1 anchor 6 (§2.1-§2.3 v0.1, SHA 6ac356c3..., 332 lines)

- **§2.1 Motivation**: cascade c → radial profile justification
- **§2.2 Definition**: `c*(r) ≡ c_local(r)`
- **§2.3 Boundary conditions** (3-regime):
  - r → ∞: → 0.83 (cascade c)
  - r → 0: → 0.42 (galactic c)
  - extreme c < 0.30: → Strigari Bernoulli regime
- **LOCKs established**: Path A LOCK / I-1 LOCK / boundary precision 3-regime / cascade SSoT auto-extension P1-P3 / forward-ref 0

### §1.2 anchor 5 (§2.4 v0.1, SHA 3270fb40..., 211 lines)

- **§2.4 Candidate functional forms**:
  - Candidate E (exp decay parametric)
  - Candidate B (cascade SSoT inversion)
  - Candidate L (linear) — **rejected per axis (vii)**
- **LOCKs established**: Path A LOCK / path δ LOCK (L reject + E/B parallel) / Issue 1-3 fix
- **Axis (vii) Locus**: A1 (manifest L47-49) + A2 (本文 L116-119), b_α=0.11 reference

### §1.3 anchor 7 (§2.5 v0.1, SHA 9e03f53e..., 914 lines)

- **§2.5.1** SPARC 171-galaxy fit pool
- **§2.5.2** E pipeline (k_E=2 default)
- **§2.5.3** Algorithm B (k_B=0, simultaneous self-consistency loop, init c_galaxy=0.42 / iterate / tol=1e-6 / N_max=50)
- **§2.5.4** ΔAIC selection criterion
- **§2.5.5** dSph J3 28/31 + b_α 3-axis audit
- **§2.5.6** [H-1] protocol-level resolve
- **3 階層 LOCK + Q-LOCK established**: Path (iii) LOCK / R-1 (R1-α) / R-2 (scenario B) / Q-C1〜C3 / Q3〜Q5 / Sub-issue F-2 / Issue 1-3 / Issue δ
- **Sub-issue 3-tier deferral category** (S-1〜S-6 v0.2):
  - A: reference resolution (S-1 / S-3)
  - B: operational protocol (S-2 / S-4 / S-5)
  - C: numerical threshold (S-6)

### §1.4 anchor 8 (§2.6 v0.1, SHA f6a48b51..., 740 lines)

- **§2.6.1** chapter summary
- **§2.6.2** §3-§7+AppB forward-spec
- **§2.6.3** forward-ref 0 chapter audit
- **§2.6.4** dependency closure (10-axis framework — anchor 5 ↔ 8 / anchor 6 ↔ 8 / anchor 7 ↔ 8)
- **§2.6.5** F1-F14 closure (companion-internal)
- **§2.6.6** closure declaration (E6-α formal final reaffirm)
- **Dependency SHAs**: anchor 5 (3270fb40) + anchor 6 (6ac356c3) + anchor 7 (9e03f53e)

---

## §2 4-Identification Chain Closure Audit

The 4-identification chain `c*(r) → c_s → χ_coh → g_eff` is L-1 LOCK forward-ref 0 strict across §2 entirety. Verified by anchor 21 chapter-level sweep:

| Step | Source anchor | Forward-ref count | Status |
|---|---|---|---|
| c*(r) | anchor 6 §2.1-§2.3 | 0 | ✅ closed within §2 |
| c_s | anchor 5 / anchor 7 | 0 | ✅ closed within §2 |
| χ_coh | anchor 7 | 0 | ✅ closed within §2 |
| g_eff | anchor 8 §2.6.1 | 0 | ✅ closed within §2 |

**Topology** (per anchor 20 §C, 7/7 ✅ MATCH): 1 base (`c*(r)`) + 3 derived (`c_s`, `χ_coh`, `g_eff`). Graph topology consistent across anchor 5/6/7/8 ↔ anchor 14/16/17.

**Hybrid g_eff form** (parent v4.8 §3.1 line 335 immutable, SHA 902f79c6):

```
g_eff = g_bar × ν_canonical(x; c) × χ_coh
```

where `ν_canonical(x; c) = f_opt⁻¹(x; c)` with `f_opt` = deg-4 Lagrange interp of V"(x=0.5, c) at 5 anchors `c ∈ {0.30, 0.42, 0.618, 0.80, 1.00}`.

**L-1 LOCK forward-ref 0 chapter-level audit result**: ✅ PASS

---

## §3 §3-§5+ Output Spec Recap

Anchor 21 re-affirms the §2 → downstream forward-spec defined in anchor 8 §2.6.2, against the latest state of each downstream anchor.

### §3.1 §2 → §3 hand-off (c_s, supersonic regime)

- **Receiver**: anchor 17 (§3 v0.2, SHA 178dad11, 332 lines)
- **Hand-off content**:
  - c_s definition from §2.4 (anchor 5)
  - supersonic boundary `c = 0.5709` (anchor 17 §3.8 — §3-B injection independent CANONICAL; parent v4.x candidate + Phase C3 candidate both rejected)
  - Pattern B canonical: Möbius `g_J = 2u/(1+u)` (anchor 17 §3.9, MAINTAINED)
  - Pattern A: alternative additive ADOPTED
  - dual-display retain (frequency ↔ group velocity, `T_m=√6` identical numerical identifier; anchor 17 §3.10)
- **Numerical**: `u_super=0.6551, w²=3.7981, w=1.9489, M_super=1.6452, c_mem(0.5709)=1.949`
- **F-flags resolved at this hand-off**: F-15 / F-16 / F-17 (already RESOLVED per anchor 17)
- **Status**: ✅ closed forward-spec verified

### §3.2 §2 → §4 hand-off (g_eff hybrid form for SPARC empirical)

- **Receiver**: anchor 14 (§4 v0.4, SHA 295bc05c, 269 lines)
- **Hand-off content**:
  - hybrid g_eff form (Layer A `ν_canonical` × Layer B-α `χ_coh`)
  - cascade c=0.83 (Layer A reference)
  - Layer B-α operational realization (`f_p` ↔ Bernoulli s, anchor 13/14)
- **§4.13 CRITICAL FINDING**: C3-A1 has NO iterative solver → Layer B-α/B-β sub-divide formal
- **§4.14 NGC 3198 formal evaluate**: `f_p=0.9930, c_mem=0.0070, CSV bit-exact`
- **§4.15 path A/B canonical**:
  - path A median estimator `η_0=0.5629` (Δ=−3.61%, T6 PASS) ✅
  - path B v_flat² weighted `η_0=0.5649` (Δ=−3.27%, T4 PASS) ✅
- **§4.16**: 5/6 acceptance criteria PASS → B+ → A 級昇格 prerequisite 完成 ✅
- **F-flags resolved at this hand-off**: F-§4-1 / F-§4-2 / F-§4-3-A / F-§4-3-B / F-§4-4 / F-§4-5
- **Status**: ✅ closed forward-spec verified, A 級 prerequisite 形式的満足

### §3.3 §2 → §5 hand-off (Layer B-β analytic Bernoulli closure)

- **Receiver**: anchor 16 (§5 v0.2.1, SHA 69678018, 136 lines) + anchor 15 (§5 v0.2 dependency)
- **Hand-off content**:
  - Layer B-β analytic closure (Bernoulli `s_0=0.3515` at Q→0)
  - Strigari universal acceleration `G_Strigari/a_0 = s_0(1−s_0) = 0.227948 ≈ 0.228`
  - Newtonian limit per-galaxy Q-monotonic (high-Q → g_bar)
  - lensing per-layer (path A canonical + path B alt + Layer A ref)
  - g_eff hybrid form (Layer A ν + Layer B-α χ_coh)
- **§5.7 Layer B-α / B-β disambig** (anchor 16 errata):
  - Layer B-α: CSV operational, SPARC LSB ~0.0070 boundary clipped
  - Layer B-β: Bernoulli analytic, dSph extreme `s_0=0.3515`
  - 両 layer は disjoint domain で legitimate coexist
- **F-flags resolved at this hand-off**: F-§5-1 / F-§5-2 / F-§5-3 / B-α/β disambig
- **Status**: ✅ closed forward-spec verified

### §3.4 §2 → §6 / §7 / Appendix B hand-off

Per anchor 8 §2.6.2 forward-spec. anchor 21 chapter-level reference inheritance only; no content change.

- §6 receives: chapter-level closure protocol pattern (anchor 21 establishes the template)
- §7 receives: synthesis hooks for Phase C3 cross-paper coherence
- AppB receives: SHA chain index + LOCK ledger

---

## §4 Dependency Closure Audit (10-axis framework)

Anchor 21 re-runs the 10-axis framework established in anchor 8 §2.6.4 (LOCK §2.6-D) against current state.

### §4.1 anchor 5 ↔ 8 (3-axis)

| Axis | Content | anchor 21 result |
|---|---|---|
| (i) | R-1 anchor 5 wording immutability | ✅ MAINTAINED (no retroactive change) |
| (ii) | F12 axis (vii) inject content vs anchor 5 Locus A1+A2 PRIMARY MATCH + secondary B2/B3/B4 | ✅ PRIMARY MATCH preserved |
| (iii) | §2.5.4 selection criterion forward-spec の §2.4 L rejection eligibility filter consistency | ✅ consistent |

### §4.2 anchor 6 ↔ 8 (3-axis)

| Axis | Content | anchor 21 result |
|---|---|---|
| (iv) | cascade SSoT P1-P3 closure (Method A canonical / deg-4 Lagrange 5-anchor immutable / #32 retract LOCK c_galaxy=0.42 + c_cascade=0.83 bit-exact maintenance) | ✅ bit-exact preserved |
| (v) | c*(r) functional form forward-spec (§3 c_s / §4 χ_coh / §5 g_eff export) consistency | ✅ consistent across anchor 14/16/17 |
| (vi) | #32 retract LOCK chapter-level maintained | ✅ maintained |

### §4.3 anchor 7 ↔ 8 (4-axis)

| Axis | Content | anchor 21 result |
|---|---|---|
| (vii) | §2.5 INDEPENDENT BASE 性 forward-ref 0 chapter-level | ✅ closed (provides F12 resolution basis, see §5.2) |
| (viii) | 3 階層 LOCK (Path (iii) + R-1 + R-2) + Q-LOCK 6 件 + [H-1] protocol-level resolve consistency | ✅ all 3-tier LOCKs preserved |
| (ix) | S-1〜S-6 6-item v0.2 一括 deferral 構造 (3-tier category A/B/C) chapter-level 維持 | ✅ category structure maintained |
| (x) | k_E=2 default LOCK (Q-C1) chapter-level reaffirm | ✅ reaffirmed |

**10-axis chapter-level audit result**: **✅ ALL 10/10 PASS**

---

## §5 F-flag Bulk Resolve Table

### §5.1 §-numbered F-flags (already RESOLVED per anchors 17/14/15/16, recap)

| # | Flag | Resolving anchor | § | Resolution summary |
|---|---|---|---|---|
| 1 | F-15 | anchor 17 | §3 | c_super=0.5709 §3-B independent specification CANONICAL |
| 2 | F-16 | anchor 17 | §3 | Pattern B canonical Möbius g_J=2u/(1+u) + Pattern A alternative additive |
| 3 | F-17 | anchor 17 | §3 | dual-display retain (frequency ↔ group velocity, T_m=√6 identical numerical identifier) |
| 4 | F-§4-1 | anchor 13/14 | §4 | χ_coh definition refinement closed-form C3-A2 explicit cite |
| 5 | F-§4-2 | anchor 13/14 | §4 | Layer B-α operational realization (f_p ↔ Bernoulli s) |
| 6 | F-§4-3-A | anchor 14 | §4 | NGC 3198 c_mem = 0.0072 LSB physical (Layer B-α) |
| 7 | F-§4-3-B | anchor 14 | §4 | cascade c=0.83 path A median η_0=0.5629 (Δ=−3.61% T6 ✅) |
| 8 | F-§4-4 | anchor 13/14 | §4 | path A canonical η_0 establish (sign-level monotonicity proof) |
| 9 | F-§4-5 | anchor 13 | §4 | path B alternative additive establish (5-layer cascade chain) |
| 10 | F-§5-1 | anchor 15 | §5 | Newtonian limit per-galaxy Q-monotonic, high-Q→g_bar |
| 11 | F-§5-2 | anchor 15 | §5 | lensing per-layer (path A canonical + path B alt + Layer A ref) |
| 12 | F-§5-3 | anchor 15 | §5 | g_eff hybrid form Layer A ν + Layer B-α χ_coh |
| — | B-α/β disambig | anchor 16 | §5 | Layer B-α (operational) / B-β (analytic) regime separation explicit |

**§-numbered tally (anchor 21 chapter-level)**: 12 RESOLVED + 1 disambig RESOLVED. Active count = 0.
Per anchor 19 §1.5: §1=0 / §2=0 / §3=0 / §4=0 / §5=0 / §6=0 / §7=0 (TIER-1 完全消化 maintained).

### §5.2 Companion-internal F1-F14 closure (anchor 8 §2.6.5 → anchor 21 chapter-level final)

Anchor 8 §2.6.5 placed companion-internal F1-F14 in mixed states. Anchor 21 hardens F4 and F12 at chapter level via 10-axis closure basis. F1/F2/F6/F8 are intentionally maintained as `placeholder` per LOCK §2.6-B (chapter-level closure scope clean preservation), with explicit handoff to §2.5 v0.2 round per Sub-issue S-1/S-3 category A semantic coherence and Q-C1 LOCK preservation.

Verbatim source: `J_system_paper_section2_6_v0.1.md` L553-564 (E5-γ block).

| Flag | Content (anchor 8 §2.6.5 verbatim) | 3-tier category | Status @ anchor 8 | Status @ anchor 21 | Resolution path |
|---|---|---|---|---|---|
| F1 | parent v4.8 §6 line XXX (σ_g(r) observational uncertainty reference) | A: reference resolution | placeholder 維持 (S-3 specific aspect) | placeholder 維持 (chapter-level closure scope clean preservation) | §2.5 v0.2 round literal reference resolution |
| F2 | Phase C3 v3 §X Lesson 91 chapter placeholder (bridge pre-cut 4 galaxy reference) | A: reference resolution | placeholder 維持 (S-1 specific aspect) | placeholder 維持 | §2.5 v0.2 round literal reference resolution |
| F3 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F4 | (status declaration) | — | status declaration | **RESOLVED** ✅ | basis: 10-axis (i)+(ii)+(iii) anchor 5↔8 ALL PASS, R-1 LOCK preserved (§4.1 above); status declaration → chapter-level RESOLVED upgrade |
| F5 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F6 | Phase C3 v3 §X dSph 30 sample chapter 番号 placeholder | A: reference resolution | placeholder 維持 | placeholder 維持 (anchor 8 §2.6 closure では active resolve せず) | handoff 継承（F1/F2 と同 reference resolution 性質） |
| F7 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F8 | k_E sensitivity variant k_E=1 supplement | B: operational protocol | k_E=2 default LOCK 完了 (Q-C1)、k_E=1 sensitivity audit pending | placeholder 維持 (Q-C1 LOCK 維持 + k_E=1 audit pending handoff) | §2.5 v0.2 round (SPARC pipeline empirical execution 内 natural execution) |
| F9 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F10 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F11 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |
| F12 | (active inject axis vii) | — | active inject (axis vii) | **RESOLVED** ✅ | basis: 10-axis (vii) §2.5 INDEPENDENT BASE 性 forward-ref 0 chapter-level closure (§4.3 above); active inject content cleared by chapter-level forward-ref 0 audit ✅ PASS |
| F13 | (already resolved record) | — | already resolved record | RESOLVED (record retained) | per anchor 8 §2.6.5 |
| F14 | (handoff continuation flag) | — | handoff continuation | handoff continuation (unchanged) | per anchor 8 §2.6.5 (by design) |

**Semantic coherence audit** (per anchor 8 §2.6.5 design rationale):

- F1, F2 align with Sub-issue **S-3 / S-1** respectively (3-tier deferral category A: reference resolution; established at anchor 7 §2.5 R-2 LOCK)
- F6 shares F1/F2 reference resolution character (same category A); no S-mapping but coherent grouping
- F8 falls under **Q-C1 LOCK** (k_E=2 default) already established at anchor 7; k_E=1 sensitivity audit is the remaining operational protocol (category B)
- All four placeholders preserve **LOCK §2.6-B chapter-level closure scope clean** invariant — actively resolving them at chapter level would import §2.5 round-level operational detail into closure scope, violating the scope-clean principle

**Companion-internal tally (anchor 21 v0.1.1)**:

- RESOLVED at anchor 21 chapter-level: **F4, F12** (newly), F13 (record retained) → 3 flags
- placeholder 維持 (by design, LOCK §2.6-B): F1, F2, F6, F8 → 4 flags, all handed off to §2.5 v0.2 round
- handoff continuation (by design): F3, F5, F7, F9, F10, F11, F14 → 7 flags

**No anchor 21-internal pending TODO remains.** All flag states are either RESOLVED at chapter level, or by-design placeholder/handoff with explicit forward path declared.

### §5.3 Active count summary

| Layer | RESOLVED | active | placeholder 維持 (by design + handoff) | handoff continuation (by design) |
|---|---|---|---|---|
| §-numbered F-flags | 13 (12 + B-α/β disambig) | **0** | 0 | 0 |
| Companion-internal F1-F14 (anchor 21 v0.1.1) | 3 (F4, F12, F13) | **0** | 4 (F1, F2, F6, F8 → §2.5 v0.2 handoff) | 7 (F3, F5, F7, F9, F10, F11, F14) |
| **Combined chapter-level active count** | — | **0** | — | — |

**Active F-flag count at anchor 21 chapter-level closure**: **0** (TIER-1 完全消化 maintained; A 級昇格 prerequisite condition 2 preserved).

---

## §6 Chapter-Level LOCK Declaration

### §6.1 §2 closure formal statement

> **§2 (Membrane kinematic + acoustic + coherence + g_eff) is hereby declared CLOSED at chapter-level by anchor 21 v0.1.1.**

The 4-identification chain `c*(r) → c_s → χ_coh → g_eff` is fully derived within §2 with L-1 LOCK forward-ref 0 strict and 10-axis dependency closure verified ALL PASS. F4 and F12 are RESOLVED at chapter level; F1/F2/F6/F8 are intentionally maintained as placeholder per LOCK §2.6-B with explicit §2.5 v0.2 round handoff. All anchor-21-internal pending TODOs are cleared.

### §6.2 LOCK tier ledger (post anchor 21)

| Tier | LOCK content | Establishing anchor | anchor 21 status |
|---|---|---|---|
| Path (iii) LOCK | B canonical = ν_canonical = f_opt⁻¹(x; c) deg-4 Lagrange 5-anchor inversion (parent §3.1 L335 immutable). §5 [H-4] resolve 時 alternative ν form は additive のみ + invalidation 不可 | anchor 7 | ✅ chapter-level reaffirmed |
| R-1 (R1-α) LOCK | anchor 5 wording "non-parametric" immutable + §2.5.3 内 "parameter-free canonical (k_B=0、parameter-free via deg-4 Lagrange 5-anchor inversion)" precision refinement、anchor 5 retroactive change 0 | anchor 7 | ✅ chapter-level reaffirmed |
| R-2 (scenario B) LOCK | §2.5.3 Algorithm B simultaneous self-consistency loop (initialize c_galaxy=0.42 / iterate / tol=1e-6 / N_max=50)、Candidate B standalone selection candidate role 維持、anchor 5 意味論的 retroactive change 0 | anchor 7 | ✅ chapter-level reaffirmed |
| Path A LOCK | anchors 5/6 共通 | anchor 5 / 6 | ✅ maintained |
| I-1 LOCK | boundary precision 3-regime | anchor 6 | ✅ maintained |
| boundary precision 3-regime LOCK | r→∞ 0.83 / r→0 0.42 / extreme c<0.30 Strigari | anchor 6 | ✅ maintained |
| cascade SSoT auto-extension P1-P3 LOCK | Method A canonical / deg-4 Lagrange 5-anchor / #32 retract | anchor 6 | ✅ maintained |
| path δ LOCK | L reject + E/B parallel | anchor 5 | ✅ maintained |
| Q-LOCK | Q-C1〜C3 + Q3〜Q5 + Sub-issue F-2 + Issue 1-3 + Issue δ | anchor 7 | ✅ maintained |
| **chapter-level §2 LOCK** | §2 整体 closure (anchors 5/6/7/8 統合 + 10-axis dependency closure ALL PASS + F4/F12 chapter-level RESOLVED + F1/F2/F6/F8 placeholder 維持 §2.5 v0.2 handoff + §3-§5+ forward-spec consistent) | **anchor 21 v0.1.1 (this document)** | **ESTABLISHED** ✅ |

### §6.3 SHA chain hand-off

- anchor 19 (§1 v0.4 SHA 0b269c10) declared 18-anchor SHA chain immutable (anchor 1-17 + anchor 18 breadth v0.3 SHA 7dd9fab0...)
- anchor 20 (milestone summary SHA 56afa4c2) extends to 19-anchor chain
- **anchor 21 will extend to 21-anchor chain** (anchor 1-17 + 18 + 19 + 20 + 21; cascade churn 0 strict maintained; anchor 02 historical superseded retained NOT_FOUND record)
- anchor 21 SHA: TBD (set on v0.1.1 finalize after this patch; recorded in updated SHA256SUMS.txt for companion-v0.1-2026-05-03 release if included)

---

## §7 Cross-Paper Coherence Verification

### §7.1 Phase C3 b_α 0.5% agreement

| Source | b_α value | Reference |
|---|---|---|
| SPARC | +0.1084 | anchor 14 §4 + anchor 7 §2.5.5 |
| dSph | +0.1127 | anchor 7 §2.5.5 + anchor 19 §1.5 |
| \|diff\| | 0.0042 | anchor 19 §1.5 verbatim |
| Density range | 3.92 桁 | per anchor 20 §I |
| Agreement | 0.5% | within Phase C3 tolerance |
| ΔAIC | −2.00 | per anchor 20 §I |

**Note on numerical consistency**: handoff memo (2026-05-03) line 71 records `|Δ|=0.0043`. Anchor 19 §1.5 verbatim reads `|diff|=0.0042`. Anchor 21 follows anchor 19 (most recent A-grade declaration) for the canonical chapter-level value. This rounding-step discrepancy does not affect the 0.5% agreement claim.

### §7.2 Parent v4.8 NULL impact verification

- Severity matrix: all NULL ✅ (anchor 19 §1.5 condition 3)
- companion 純 additive principle preserved
- Parent v4.8 3-track structure NOT modified by anchor 21:
  - latex_v48 (LuaLaTeX, ja): tex SHA 902f79c6, pdf SHA b8a88f04
  - latex_v48_en (pdfLaTeX, en): tex SHA 2dcf69e6
  - latex_v478 (pdfLaTeX, en short): tex SHA b7bf9629
- Parent v4.8 §3.1 line 335 (ν_canonical canonical anchor) MAINTAINED bit-exact

---

## §8 Promotion Status Update

### §8.1 A 級昇格 prerequisite achievement reaffirmation

Per anchor 19 §1.5 + anchor 14 §4.16:

| Path | η_0 estimate | Δ from cascade c=0.83 | Tolerance | Status |
|---|---|---|---|---|
| path A median | 0.5629 | −3.61% | T6 (5%) | ✅ PASS |
| path B v_flat² weighted | 0.5649 | −3.27% | T4 (5%) | ✅ PASS |

**B+ → A 級 prerequisite 形式的満足 ✅** — 4 conditions all met (per anchor 19 §1.5):

1. ✅ path A 0.5629 (Δ−3.61%) + path B 0.5649 (Δ−3.27%) 両 5% 内
2. ✅ All active F-flag RESOLVED (12 F-flag, chapter-level tally §1=0 / §2=0 / §3=0 / §4=0 / §5=0 / §6=0 / §7=0)
3. ✅ severity matrix all NULL (parent v4.8 NO impact, companion 純 additive)
4. ✅ 18-anchor (now 21-anchor on anchor 21 finalize) SHA chain immutable (cascade churn 0 strict)

Anchor 21 chapter-level reaffirmation: **A 級 prerequisite preserved** ✅

### §8.2 TIER progression

Per anchor 19 §1.5 + anchor 20 §K:

| TIER | State pre-anchor 21 | State post-anchor 21 v0.1.1 |
|---|---|---|
| TIER-1 (active F-flag = 0) | 完全消化 | 完全消化 (maintained) |
| TIER-2 (chapter-level closure) | ready (no chapter-level closure issued yet) | **§2 first chapter-level closure ESTABLISHED** ✅ (anchor 21 v0.1.1) |
| TIER-3 (milestone declaration) | next phase declared | **partial unblock** — §2 chapter-level closure 完了; full unblock awaits §3-§7 chapter-level closures (template now established by anchor 21) |
| TIER-4 (publication advancement) | blocked | blocked (anchor 21 alone insufficient; chapters §3-§7 still pending chapter-level closure following anchor 21 template) |

**Implication**: anchor 21 is the **first chapter-level closure** in the J-system companion paper, establishing the closure declaration template for §3 (anchor 17 base) / §4 (anchor 14 base) / §5 (anchor 16 base) / §6 / §7 / AppB future chapter-level closures.

---

## §9 Forensic Chain Protocol Compliance

Per anchor 20 §J 7-item ruleset, anchor 21 satisfies:

| # | Rule | Status |
|---|---|---|
| 1 | SHA chain index updated | ✅ planned: companion-v0.1-2026-05-03 SHA256SUMS.txt entry on finalize |
| 2 | predecessor anchors 5/6/7/8 cited bit-exactly via SHA prefix | ✅ §1 + Appendix A |
| 3 | no retroactive change to anchor 5/6/7/8 wording (R-1 LOCK preserved) | ✅ §4.1 axis (i) + §6.2 R-1 row |
| 4 | forward-ref 0 strict maintained | ✅ §2 audit table |
| 5 | companion 純 additive (parent v4.8 NULL impact) | ✅ §7.2 |
| 6 | closure declaration explicit, LOCK tier assigned | ✅ §6.1 + §6.2 chapter-level §2 LOCK row |
| 7 | cross-check audit re-runs | ✅ §4 (10-axis ALL PASS) |

**Item count**: **7/7 ✅ MATCH** (v0.1.1 finalize: chapter-level §2 LOCK ESTABLISHED, all anchor-21-internal TODOs cleared)

---

## §10 Anchor 21 deliverable metadata

| Field | Value |
|---|---|
| Anchor number | 21 |
| File name | `J_system_paper_section2_closure_v0.1.md` |
| Version | v0.1.1 (finalize — F1/F2/F6/F8 classification corrected per anchor 8 §2.6.5 verbatim L553-564; chapter-level §2 LOCK ESTABLISHED) |
| Predecessor anchors cited | 5 / 6 / 7 / 8 / 14 / 16 / 17 / 19 / 20 (9 anchors) |
| Predecessor SHA references | 6ac356c3 (anchor 6) / 3270fb40 (anchor 5) / 9e03f53e (anchor 7) / f6a48b51 (anchor 8) / 0b269c10 (anchor 19) / 56afa4c2 (anchor 20) / 178dad11 (anchor 17) / 295bc05c (anchor 14) / 69678018 (anchor 16) |
| Parent v4.8 SHA cited | 902f79c6 (latex_v48 tex) / b8a88f04 (latex_v48 pdf) / 2dcf69e6 (latex_v48_en tex) / b7bf9629 (latex_v478 tex) |
| LOCKs introduced | chapter-level §2 LOCK (ESTABLISHED ✅) |
| F-flags newly RESOLVED at chapter level | F4, F12 (companion-internal) |
| F-flags placeholder 維持 (by-design + §2.5 v0.2 handoff) | F1, F2, F6, F8 (companion-internal) |
| Forward-ref count | 0 (L-1 LOCK strict) |
| Encoding | UTF-8 / LF / no BOM (per J-system convention) |
| License | CC-BY 4.0 |
| Release inclusion (decision pending) | companion-v0.1-2026-05-03 (would extend release zip from 19 → 20 anchors + 1 PDF) OR held for next release |

---

## Appendix A: SHA reference table

| Anchor | File (per extract_v0.1) | SHA prefix |
|---|---|---|
| 1 | J_system_paper_section1_draft_v0.2.md | 523fbc2a |
| 2 | (J_system_paper_sections2to7_breadth_v0.1.md — manifest NOT_FOUND, historical superseded) | — |
| 3 | J_system_paper_sections2to7_breadth_v0.2.md | 2d987c0f |
| 4 | J_system_paper_section1_draft_v0.3.md | 22167ee1 |
| 5 | J_system_paper_section2_4_v0.1.md | 3270fb40 |
| 6 | J_system_paper_section2_1to3_v0.1.md | 6ac356c3 |
| 7 | J_system_paper_section2_5_v0.1.md | 9e03f53e |
| 8 | J_system_paper_section2_6_v0.1.md | f6a48b51 |
| 9 | J_system_paper_section3_v0.1.md | 5277c8f6 |
| 10 | J_system_paper_section4_v0.1.md | a98d4640 |
| 11 | J_system_paper_section4_v0.2.md | 024baf67 |
| 12 | J_system_paper_section5_v0.1.md | bdf2a470 |
| 13 | J_system_paper_section4_v0.3.md | 49b1554a |
| 14 | J_system_paper_section4_v0.4.md | 295bc05c |
| 15 | J_system_paper_section5_v0.2.md | f18edd9a |
| 16 | J_system_paper_section5_v0.2.1.md | 69678018 |
| 17 | J_system_paper_section3_v0.2.md | 178dad11 |
| 18 | J_system_paper_sections2to7_breadth_v0.3.md | 7dd9fab0 |
| 19 | J_system_paper_section1_v0.4.md | 0b269c10 |
| 20 | J_system_v0.1_milestone_summary.md | 56afa4c2 |
| **21** | **J_system_paper_section2_closure_v0.1.md** (this, v0.1.1) | **TBD (computed on this finalize)** |

(anchor 02 = J_system_paper_sections2to7_breadth_v0.1.md is `manifest NOT_FOUND` — historical superseded by anchor 03 v0.2 then anchor 18 v0.3 — excluded from chain.)

---

## Appendix B: Post-finalize action items

(Anchor 21 v0.1.1 internal TODOs cleared. Items below are **external dependencies** and **forward roadmap** items, not blockers for v0.1.1 chapter-level §2 LOCK ESTABLISHED status.)

1. **Anchor 21 SHA finalize (post-this-patch)** — compute final anchor 21 v0.1.1 SHA256, update Appendix A row 21, generate companion-v0.1-2026-05-03 release SHA256SUMS.txt entry (if release inclusion confirmed).
2. **F1/F2/F6/F8 §2.5 v0.2 round handoff resolution** — these placeholders are explicitly handed off to the §2.5 v0.2 SPARC empirical execution round (Sub-issue S-1/S-3 reference resolution + k_E=1 sensitivity audit). Resolution occurs there, not as anchor 21 v0.1.x patch.
3. **Release inclusion decision** — confirm whether anchor 21 v0.1.1 enters companion-v0.1-2026-05-03 release zip:
   - Option A: include → release zip extends from 19 → 20 anchors + 1 PDF; chapter-level closure visible at release time
   - Option B: hold for next release cycle → cleaner separation between A 級昇格 declaration release and chapter-level closure release
4. **Subsequent chapter-level closure templates** — anchor 21 v0.1.1 establishes the format. §3 (anchor 17 base) / §4 (anchor 14 base) / §5 (anchor 16 base) chapter-level closures can follow this pattern. Decide ordering / priority for these (TIER-3 milestone declaration full unblock requires §3-§7 chapter-level closures).
5. **WordPress A-1 / A-5 release link refresh** — once companion-v0.1-2026-05-03 release is published, refresh the GitHub release links in A-1 homepage snippet and A-5 companion paper page (per handoff memo §6 release 完了後 2 次タスク).

---

**END J_system_paper_section2_closure_v0.1.md (anchor 21 v0.1.1 finalize; chapter-level §2 LOCK ESTABLISHED; F4/F12 RESOLVED; F1/F2/F6/F8 placeholder 維持 + §2.5 v0.2 handoff per anchor 8 §2.6.5 verbatim L553-564)**
