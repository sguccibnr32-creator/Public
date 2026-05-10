# anchor 22 v0.2 — Lessons Appendix (L-Q3-11 codification batch reference)

**Anchor:** anchor 22 v0.2 (axis_4 type-alpha reproduction promotion)
**Round:** 2026-05-10, turn N+4 ~ N+10
**Author:** Sakaguchi Shinobu
**License:** CC-BY 4.0
**Paired with:** `anchor_22_v0_2_declaration.md`,
`anchor_22_v0_2_input_files_pin.json`,
`anchor_22_v0_2_verification_log.md`

---

## 1. Status of formal codification

The L-Q3-11 codification batch is in **draft-ready** state at the time of
anchor 22 v0.2 closure. Formal append to `lessons_codified.md` (with SHA
pin) is **deferred** to a dedicated batched codification round, in
combination with L-Q3-10 (autocrlf-induced SHA shift on text-mode
files), as the natural progression for a clean batch. The present
appendix preserves a draft summary plus pointers; it is **not** the
formal codified record.

Quoting the round's recommendation policy:

> Codification recommendation: combined with L-Q3-10 in a batched
> codification round; comprehensive draft completed during the present
> round; formal append in a dedicated round. The present
> `anchor_22_v0_2_lessons_appendix.md` carries pointer + draft summary,
> not the formal record.

## 2. Pattern catalog (8 patterns, draft summary)

### 2.1 Pattern 7 sub-clause refinement (prior chat, established)

**WSL invocation transport-layer compatibility.** Git Bash MSYS2 path
translation can silently auto-convert `/mnt/*` paths to Windows paths,
breaking WSL2 invocations. Codified policy: **PowerShell-only invocation
for WSL transport**, with a bash wrapper script consolidating all WSL
commands. The pattern in this round (consistently applied):
`PowerShell -> wrapper.sh -> wsl -d Ubuntu bash <wrapper.sh>`.

### 2.2 Pattern 8 (prior chat, established)

**Whitelist scope path-level expansion audit.** A prefix-level whitelist
and a path-level scope can silently disagree. Empirical example: 99 ->
126 -> 97 envelope refinement, with 99 / 126 systematic divergence cases
caught only by path-level audit. Lesson: always audit scope at the
**path level**, not just the prefix.

### 2.3 Pattern 5 sub-clause refinement (prior chat, established)

**PLACEHOLDER deferred-judgment criteria.** When a value is to be
deferred (PLACEHOLDER), explicitly record the deferral conditions and
the lift criterion. Without this, deferred items can leak into "PASS"
state via implicit fall-through.

### 2.4 Pattern 9 (prior chat, established)

**Pass-through verification self-defeating tautology.** Canonical
equality check on a pass-through transform is trivially true; it
verifies nothing. Replace with **shape-difference check against the
schema reference**. Operational verification this round: M-gamma
resolved at `field_consistency` bucket via Pattern 9 inline shape
assertion (Piece 5b).

### 2.5 Pattern 10 (prior chat, established)

**Refactor blind spot to existing invariants.** When deleting a
function, the order of caller removal vs definition removal matters.
Operational verification this round: Pieces 5a/5b/5c remove all 3
callers first; Piece 6 deletes the definition last; orphan reference
NameError risk = 0.

### 2.6 Pattern 11 (prior chat, established)

**Snippet-with-annotation context misuse risk.** Annotations attached
to snippets can become detached from the snippet during downstream
serialization. Codified protocol: capture 10-A through 10-F verbatim
feed protocol with line-numbered cat -n format and boundary uniqueness
audit. Operational verification this round: capture 11-A delivery for
the V4 anomaly identification used the same protocol successfully.

### 2.7 Pattern 12 (NEW THIS ROUND)

**JSON transport escape sequence auto-decoding.** When a 6-character
ASCII escape sequence (e.g., `&#92;u00a7` representing the section sign)
is sent through the Edit tool's JSON parameter pipeline (RFC 8259
string parsing), the JSON layer silently decodes it into a single
codepoint on disk. The intended literal form is lost without warning.

**Avoidance forms (preference order):**
1. Double-escape at draft time (`&#92;&#92;u00a7` -> JSON decodes one
   level -> `&#92;u00a7` literal lands on disk).
2. Alternative notation that does not contain the auto-decoded
   sequence.
3. PowerShell byte-level replacement after Edit tool apply (the
   remediation form used this round; works but is reactive, not
   preventive).

**Pre-condition addition:** for any patch piece touching a Pattern 12
region, run a grep audit on the rendered codepoint immediately after
apply; mismatch vs the policy baseline triggers byte-level remediation.

**Empirical confirmation:** Detected at Piece 2 apply, remediated
in-place; same region in Piece 3 was applied via PowerShell byte-precise
replacement preventively (not Edit tool); Piece 7 region had no Pattern
12 byte content so Edit tool 1-shot apply succeeded directly.

### 2.8 Pattern 13 family (NEW THIS ROUND, 1 entry + 3 sub-clauses)

A sibling triple of envelope-vs-actual gap patterns:

#### 13a. Intra-patch literal under-counting

V5b discrepancy origin. Focal mutation site is salient; non-focal
sites within the same patch (multi-piece patches especially) are
attention-blind. **Pre-condition:** scan `new_str` of every piece for
literal occurrences before envelope is finalized.

#### 13b. Per-entry vs per-collection terminology disambiguation

Phase 6.2 envelope wording origin. Words like "entries", "items",
"records" can mean either per-instance or per-collection without
explicit prefix. **Pre-condition:** prefix-align (`n_entries`,
`input_count_per_entry`, `total_count_per_collection`, etc.) to remove
ambiguity at the wording stage.

#### 13c. Filter-vs-passthrough byte-weight under-account

step A3 envelope. When migrating a dispatch from pass-through to
filter, the dropped keys carry byte-weight that is silently lost from
the output size. **Pre-condition:** envelope design must explicitly
account for the byte-weight of dropped keys.

## 3. Issue catalog (1 entry)

### 3.1 Issue I-2 — canonical -> v1.1 naming alignment audit scope expansion

Naming alignment between the canonical re-run JSON keys and the v1.1
schema keys requires a 6-prefix audit (`gc_*`, `slope_*`, `chi2_*`,
`p_*`, `weighted_*`, `consistency_*`). Out of scope for the present
rev 6 round; deferred to **rev 7+**. Carrier task in the codification
batch.

## 4. Bidirectional verification protocol — operational success cases (this round)

Five (or more) empirical confirmations of the protocol in action:

1. **Pattern 12 codification empirical.** claude.ai-side patch text
   transport awareness gap (the section-sign auto-decode was not
   surfaced at draft time) -> Claude Code-side grep audit detection +
   PowerShell byte-level remediation. claude.ai-side codification
   accepted at turn N+4.5.

2. **V4 anomaly identification + Piece 7 corrective bundle.** Claude
   Code-side V re-run detects the argparse description "(rev 5)"
   residual; claude.ai-side design judgment turn N+4.6 drafts the
   Piece 7 corrective bundle (3-line block, 1-character mutation, size
   delta = 0). Apply succeeds 1-shot.

3. **Minor scope addition autonomous (rev 5 mirror result rename).**
   Claude Code-side initiates defensive forensic preservation parallel
   to the established step A `.tmp` rename pattern; claude.ai-side
   post-hoc accept at turn N+8 + L-Q3-11 batch codification candidate
   recognition for "autonomous prudent action".

4. **A6 critical PASS.** stdout structural SHA = G_DRY signature
   `438d27f2...` EXACT confirms (a) `write_bytes` encoding integrity +
   (b) commit / dry-run mode non-divergence + (c) Final SHA =
   In-memory serialization SHA. L-Q3-10 family preventive behavior
   empirically confirmed.

5. **Cross-environment integrity verify.** Windows native + WSL2 dual
   view consistency + IMMUTABLE anchor visibility from both views +
   WSL2 `sha256sum` spot-check bit-exact. L-Q3-11 codification draft
   "Six-layer countermeasure (3) cross-environment file system
   integrity" achieved at most rigorous form.

## 5. Forward path

| Item | Round | Description |
|---|---|---|
| L-Q3-11 + L-Q3-10 batched codification | dedicated future round | Formal append to `lessons_codified.md` with SHA pin; combine 8 patterns + 1 sibling triple + 1 issue + 5+ success cases of the present batch with L-Q3-10 (autocrlf-induced SHA shift) |
| Issue I-2 | rev 7+ | 6-prefix naming alignment audit |
| Pattern 12 + Pattern 13 family integration | as needed | Promote into pre-flight checklist for any patch piece touching JSON transport-sensitive regions |

## 6. References

- Predecessor / source: handoff memo §8.4 (prior chat) + §8.4 (this
  round's Claude Code-side handoff memo)
- Pattern 12 detection / remediation detail:
  `anchor_22_v0_2_verification_log.md` §3.2 - §3.3
- Pattern 9 inline shape assertion operational verification:
  `anchor_22_v0_2_verification_log.md` §3.4 (Piece 5b)
- Pattern 10 ordering compliance:
  `anchor_22_v0_2_verification_log.md` §3.5 (Piece 6)
- Capture 11-A (Pattern 11 protocol):
  `anchor_22_v0_2_verification_log.md` §5.1
- A6 critical PASS (L-Q3-10 family preventive empirical):
  `anchor_22_v0_2_verification_log.md` §7
- Cross-environment integrity (Six-layer countermeasure (3)):
  `anchor_22_v0_2_verification_log.md` §10.1

---

*End of lessons appendix. Formal codification deferred. Draft summary
preserved as-of 2026-05-10 closure.*
