# anchor 28.1 v0.1 lessons appendix (Q8 codify round)

| field | value |
|---|---|
| author | Sakaguchi Shinobu (sole author / saka_seimensho / shisou-shi) |
| date | 2026-05-15 |
| round | anchor 28.1 v0.1 (Q8 codify round) |
| parent round | anchor 28 v0.1 (Q7 codify round) |
| scope | L-Q3-48 → L-Q3-54 codify (7 new lessons) + F-28 triad documentation (F-28.1 / F-28.2 / F-28.3) |
| companion artifacts | section12/anchor_28_1_v0_1_declaration.md / input_files_pin.json / verification_log.md |
| counter transition | 12 active + 7 deferred → 22 active + 0 deferred (delta +10 active、−7 deferred、+0 new deferred for L-Q3-48..54 alone) |
| inscribed prophylactic octet | Pattern 35 + Pattern 39 + Pattern 46 + L-Q3-47 + L-Q3-48 + L-Q3-52 + L-Q3-53 + L-Q3-54 |
| inherited deferred queue source | anchor 28 v0.1 closure 7 deferred entries (L-Q3-48 → L-Q3-54) |
| forensic chain depth post-closure | 8-deep IMMUTABLE (anchor 22 v0.2 → 28.1 v0.1) projected post-packet 4d |

本 lessons appendix は anchor 28 v0.1 (Q7 round) closure 時点で deferred queue に carried-over された 7 entries (L-Q3-48..54) を anchor 28.1 v0.1 round で formal codify した artifact。各 lesson は anchor 28 v0.1 round 内 manifest locus を持ち、本 round で codify discipline (subject / root cause / mitigation / evidence / F-28 triad reference / cross-reference) を full inscribe。L-Q3-55 candidate (本 round 内 manifest、cross-locus reconstruction class error) は option B per anchor 28.2 sub-round で formal codify、本 appendix では言及せず (verification_log.md §6.6 + input_files_pin.json field で manifest record entry のみ inscribed)。

## L-Q3-48: working tree state attest with --untracked-files=all

### subject

git status porcelain output is subject to a directory-level collapse heuristic when many untracked files exist in a single directory. The default `--untracked-files=normal` invocation can mask per-file untrackness, presenting only the directory name as a single entry. Working tree state attest discipline requires explicit per-file enumeration.

### manifest locus

anchor 28 v0.1 round packet 4d preparation phase: working tree state attest via `git status --porcelain` (default mode) produced ambiguous output where multiple untracked files in section11_lessons_codified_q7_v0_1/ directory were collapsed to a single directory-level entry, masking individual file presence.

### root cause

default porcelain mode includes a directory-collapse heuristic for "many untracked files in one directory" condition. Threshold and behavior are git-version dependent and not specified in user-facing documentation as a stable contract. F-28.1 false-negative class instrument-side gap.

### default mitigation

all working tree state attests in dispatch scripts MUST invoke:

```
& git status --porcelain --untracked-files=all
```

`--untracked-files=all` flag disables directory-level collapse, forcing per-file enumeration regardless of count. Result is stable across git versions and predictable for downstream attest parsing.

### evidence

- anchor 28 v0.1 round packet 4d initial attest: directory-level collapse manifested (1 entry returned for section11/ 4-file directory)
- anchor 28 v0.1 round packet 4d corrective dispatch: `--untracked-files=all` per-file enumeration confirmed (4 entries returned)
- anchor 28.1 v0.1 round paired sync verify S.3: 3 entries enumerated (3 inscribed artifacts、no collapse)
- F-28.1 triad cell: instrument-side false-negative root cause locus

### F-28 triad reference

F-28.1 (primary mitigation = L-Q3-48 + preemptive mitigation = L-Q3-49 for SHA256SUMS accounting): both lessons codified as paired discipline against instrument-side false-negative class

### cross-reference

- declaration.md §2 (working tree state attest discipline section)
- verification_log.md §4.2 (paired sync verify S.3 evidence section)
- input_files_pin.json: active_patterns["L-Q3-48"] field
- prophylactic octet member (4th entry of post-anchor-28.1 octet)

## L-Q3-49: SHA256SUMS line-type accounting with ^# (any) pattern

### subject

SHA256SUMS file accounting (comment line count vs entry line count) requires line-type classification regex resilient to comment convention changes. Brittle patterns matching specific comment prefixes (e.g., `^#\s` requiring space after `#`) fail when comment convention shifts (e.g., `#timestamp` no-space prefix). Robust pattern: `^#` (any character or end-of-line after `#`).

### manifest locus

anchor 28 v0.1 round packet 4d SHA256SUMS pre-commit accounting attest: comment line count miscount manifested when SHA256SUMS contained header comments using `#` immediately followed by content (no whitespace). Brittle regex `^#\s` returned undercount; correct regex `^#` returned accurate count.

### root cause

SHA256SUMS comment convention is project-level and may evolve; locking line-type accounting to a specific whitespace-presence assertion creates brittle coupling to a convention that is not part of the file format specification. F-28.1 preemptive class (no actual manifest in canonical attest but anticipated drift class).

### default mitigation

SHA256SUMS line-type accounting in dispatch scripts MUST use:

```
$comment_count = ($lines | Where-Object { $_ -match '^#' }).Count
$entry_count   = ($lines | Where-Object { $_ -match '^[0-9a-f]{64}\s' }).Count
```

NOT `$_ -match '^#\s'` or any pattern assuming specific whitespace after `#`.

### evidence

- anchor 28 v0.1 round packet 4d initial accounting attest: brittle `^#\s` pattern produced undercount when no-space comment present
- anchor 28 v0.1 round packet 4d corrective dispatch: `^#` (any) pattern accounting accurate
- preemptive class (no actual SHA256SUMS convention shift detected in anchor 28 v0.1, but lesson codified to prevent future drift)

### F-28 triad reference

F-28.1 preemptive cell: paired with L-Q3-48 primary mitigation against instrument-side false-negative class

### cross-reference

- declaration.md §2 (SHA256SUMS accounting discipline subsection)
- verification_log.md §4.3 (packet 4d projection section、accounting attest discipline)
- input_files_pin.json: active_patterns["L-Q3-49"] field

## L-Q3-50: dispatch script Mandatory parameter coverage discipline

### subject

PowerShell dispatch scripts with `param()` block Mandatory attributes require coverage attest: prepare-phase (claude.ai supply) verifies all Mandatory params are explicit in script signature, exec-phase (Claude Code pre-exec) attests all Mandatory params receive non-null values before invocation. Gap in either phase leads to interactive prompt or null-param exception, breaking dispatch atomicity.

### manifest locus

anchor 28 v0.1 round packet 4d-prepare phase: dispatch script supply with `[Parameter(Mandatory)]` param without corresponding invocation argument; Claude Code-side received supplied text without pre-exec attest, leading to PowerShell interactive prompt waiting for user input during expected non-interactive dispatch.

### root cause

dispatch script lifecycle (prepare-phase = supply-time + exec-phase = invocation-time) has two attest opportunities. Default workflow assumed prepare-phase param specification implicitly covered exec-phase invocation; this assumption breaks when supply text contains Mandatory attribute without matching invocation signature. F-28.2 cell.

### default mitigation

dispatch script supply discipline (claude.ai prepare-phase):
1. param() block all Mandatory attributes enumerated
2. corresponding `& script.ps1 -ParamName value` invocation example inline embedded
3. invocation argument completeness attest in supply message

dispatch script exec discipline (Claude Code exec-phase):
1. pre-exec param() block enumeration
2. invocation argument map cross-reference (all Mandatory params have non-null value)
3. attest PASS before invocation; FAIL → halt + paste-back

### evidence

- anchor 28 v0.1 round packet 4d-prepare phase initial dispatch: Mandatory param `-Phase` supplied without invocation argument; interactive prompt manifest
- anchor 28 v0.1 round packet 4d-prepare corrective dispatch: full param coverage attest in supply + exec pre-attest = PASS、non-interactive dispatch
- F-28.2 cell: dispatch script lifecycle coverage gap root cause locus

### F-28 triad reference

F-28.2 (primary mitigation = L-Q3-50): paired discipline of prepare-phase + exec-phase attest

### cross-reference

- declaration.md §3 (dispatch script discipline section)
- verification_log.md §4.4 (F-28.2 triad documentation subsection)
- input_files_pin.json: active_patterns["L-Q3-50"] field
- companion: L-Q3-51 (script design-intent vs invocation-phase context fit、Part 2 inscribed)

---

## L-Q3-51: dispatch script design-intent vs invocation-phase context fit

### subject

dispatch scripts often serve multiple lifecycle phases (pre-add / post-add / pre-commit / post-commit / pre-push / post-push), each with distinct assumptions about working tree state, index state, and remote sync state. A script designed for one phase invoked at another phase produces silent failure or misleading PASS verdicts. Phase-context fit attest discipline requires the design-intent phase to be embedded in the script signature and asserted at invocation.

### manifest locus

anchor 28 v0.1 round packet 4d preparation: a dispatch script designed for the post-add phase (assumes target artifacts present in git index) was invoked at the pre-add phase (artifacts only in working tree, not yet staged). The script's internal assertions referenced index entries that did not yet exist, producing FAIL verdict that was correctly interpreted as "script not yet runnable" rather than "artifacts missing" only after manual investigation.

### root cause

dispatch script design-intent (target lifecycle phase) is implicit in the assertion logic but not explicit in the script signature or invocation contract. Without explicit Phase parameter and pre-exec phase-context assertion, the invocation phase is left to operator memory, creating context-drift opportunity. F-28.3 cell.

### default mitigation

dispatch script signature MUST include explicit Phase parameter with ValidateSet:

```
param(
    [Parameter(Mandatory)]
    [ValidateSet('pre-add', 'post-add', 'pre-commit', 'post-commit', 'pre-push', 'post-push')]
    [string]$Phase
)
```

Pre-exec phase-context assertion (embedded in script body):

```
switch ($Phase) {
    'pre-add'     { assert: wt entries > 0 AND index entries = 0 for target artifacts }
    'post-add'    { assert: index entries = expected count }
    'pre-commit'  { assert: index entries staged AND HEAD unchanged from baseline }
    'post-commit' { assert: HEAD parent matches prior HEAD baseline }
    'pre-push'    { assert: local HEAD ahead of origin/main }
    'post-push'   { assert: origin/main matches local HEAD }
}
```

Invocation contract: claude.ai-side prepare-phase supplies explicit `-Phase` argument; Claude Code-side exec-phase pre-attests phase argument vs script Phase parameter ValidateSet membership before invocation.

### evidence

- anchor 28 v0.1 round packet 4d-prepare phase initial invocation: post-add script invoked at pre-add phase, internal assertions FAIL due to missing index entries
- anchor 28 v0.1 round packet 4d-prepare corrective dispatch: explicit `-Phase 'post-add'` argument + pre-exec phase attest = PASS gate
- F-28.3 cell: PS dispatch context-fit class root cause locus

### F-28 triad reference

F-28.3 primary cell: paired with L-Q3-54 (Equals method culture assertion) as the complete F-28.3 primary mitigation pair (design-intent context fit + dispatch script runtime culture assertion)

### cross-reference

- declaration.md §3 (dispatch script discipline section、Phase parameter discipline subsection)
- verification_log.md §4.5 (F-28.3 triad documentation subsection)
- input_files_pin.json: active_patterns["L-Q3-51"] field
- companion: L-Q3-50 (Mandatory parameter coverage、lifecycle prepare-phase + exec-phase distinction explicit)

## L-Q3-52: PowerShell ${var} delimit discipline

### subject

PowerShell variable interpolation in string contexts is subject to greedy parsing when the variable is immediately followed by certain punctuation, especially colon (`:`), which the PS parser interprets as a scope or drive notation separator. The form `$var:something` is parsed as `${var:something}` (scoped variable reference), not `$var` followed by literal `:something`. Bare `$var` interpolation is unsafe in any context where the variable may be followed by parser-significant punctuation.

### manifest locus

anchor 28 v0.1 round packet 4d preparation: a string template `"refs/tags/$tag_name*"` worked correctly because `*` is not parser-significant; however, related construct patterns like `"prefix$var:suffix"` (colon attached directly to variable name) produce parser ambiguity. Anticipated drift class (preemptive codification), with proximate evidence from refactor-time substitution attempts where colon-suffix expansion broke string templates.

### root cause

PowerShell string interpolation greedy parsing assumes maximum-length variable name; `:` is part of variable scope syntax (`${scope:name}` form). Without explicit delimiter braces `${var}`, the parser interprets ambiguously when colon or other parser-significant punctuation follows. F-28.3 complementary class (not a primary triad cell but discipline-paired with L-Q3-51).

### default mitigation

ALL variable interpolation in PS dispatch scripts MUST use `${var}` delimit form:

```
$tag_name = 'companion-v4.9-q8-codify-round-2026-05-15'
$refspec  = "refs/tags/${tag_name}*"
$msg      = "HEAD: ${local_head}, parent: ${local_parent}"
```

NOT: `"refs/tags/$tag_name*"` or `"HEAD: $local_head, parent: $local_parent"` (works by accident only; not safe under refactor).

### evidence

- anchor 28 v0.1 round dispatch scripts: post-codify audit refactored all `$var` bare interpolation to `${var}` delimit form
- anchor 28.1 v0.1 round paired sync verify script (sync memo §5): `${tag_name}` delimit form throughout, especially in `git ls-remote` refspec construction
- preemptive codification (no actual parser failure in canonical anchor 28 v0.1 dispatch, but discipline locked-in to prevent future drift)

### F-28 triad reference

F-28.3 complementary cell: PowerShell discipline pattern, paired with L-Q3-51 (Phase parameter) and L-Q3-53 (wildcard refspec) as the complete PS dispatch context-fit triad of complementary patterns

### cross-reference

- declaration.md §3 (dispatch script discipline section、interpolation safety subsection)
- input_files_pin.json: active_patterns["L-Q3-52"] field
- prophylactic octet member (6th entry of post-anchor-28.1 octet)
- companion: L-Q3-53 (wildcard refspec、PS string template safety)

## L-Q3-53: git ls-remote wildcard refspec for tag pair attest

### subject

`git ls-remote --tags origin <refspec>` returns annotated tag information across two distinct ref lines: the tag object SHA (`refs/tags/<name>`) and the peeled commit SHA (`refs/tags/<name>^{}`). A bare refspec without wildcard suffix returns only the tag object line; the peeled line requires wildcard match. Remote tag attest discipline must enumerate both lines to confirm tag-to-commit binding integrity end-to-end.

### manifest locus

anchor 28 v0.1 round packet 4d-post phase remote sync attest: `git ls-remote --tags origin refs/tags/companion-v4.9-q7-codify-round-2026-05-15` returned single line (tag object SHA only). Peeled commit attest required separate query or wildcard match. F-28.3 complementary class.

### root cause

git ls-remote refspec exact-match returns only the explicitly named ref; the peeled commit pseudo-ref `<name>^{}` is technically a separate ref name that does not match the exact pattern. Wildcard suffix `*` covers both forms in a single query, returning both lines for downstream parsing. Without wildcard, peeled attest is gap.

### default mitigation

remote tag attest in dispatch scripts MUST use wildcard refspec:

```
$tag_name = 'companion-v4.9-q8-codify-round-2026-05-15'
$lines = & git ls-remote --tags origin "refs/tags/${tag_name}*"
$tag_obj = $null
$peeled  = $null
foreach ($line in $lines) {
    if     ($line -match '^([0-9a-f]{40})\s+refs/tags/.+\^\{\}$') { $peeled  = $matches[1] }
    elseif ($line -match '^([0-9a-f]{40})\s+refs/tags/.+$')        { $tag_obj = $matches[1] }
}
```

Both `$tag_obj` and `$peeled` populated; downstream attest compares both against expected baseline values.

### evidence

- anchor 28 v0.1 round packet 4d-post phase initial attest: bare refspec returned 1 line, peeled gap
- anchor 28 v0.1 round packet 4d-post corrective dispatch: wildcard refspec returned 2 lines, full tag-to-commit binding attest PASS
- anchor 28.1 v0.1 round paired sync verify S.4: wildcard refspec applied, remote tag obj + peeled both MATCH per OVERALL PASS

### F-28 triad reference

F-28.3 complementary cell: PowerShell + git discipline pattern, paired with L-Q3-51 and L-Q3-52 as PS dispatch context-fit complementary triad

### cross-reference

- declaration.md §3 (remote sync attest discipline subsection)
- verification_log.md §4.6 (S.4 remote sync attest cross-reference)
- input_files_pin.json: active_patterns["L-Q3-53"] field
- prophylactic octet member (7th entry of post-anchor-28.1 octet)

## L-Q3-54: CultureInfo equality via Equals method [DEFAULT MITIGATION]

### subject

PowerShell CultureInfo comparison via `Name` property attribute is brittle because `[System.Globalization.CultureInfo]::InvariantCulture.Name` returns empty string (`""`), not `'Invariant'` or `'InvariantCulture'`. Attest patterns of the form `$culture.Name -eq 'Invariant'` return False against an actual invariant culture object, producing silent false-negative in dispatch script culture assertion. Correct discipline uses the `.Equals()` method against the canonical singleton.

### first inscribed

anchor 28 v0.1 round 内 manifest (packet 4d-prepare phase dispatch script culture assertion gap; root cause locus of F-28.3 cell)

### manifest locus

anchor 28 v0.1 round packet 4d-prepare phase: dispatch script attest line `$c_culture = ($culture.Name -eq 'Invariant')` returned False against actual InvariantCulture object, producing FAIL verdict despite culture context being correct. Initial diagnosis pursued culture context drift hypothesis; root cause traced to Name property string-compare assumption.

### root cause

`[System.Globalization.CultureInfo]::InvariantCulture` is a singleton with `Name = ""` by design (invariant = no language tag, no country tag, no collation tag). Equality via Name string compare against literal `'Invariant'` is invalid for invariant culture. Equality via the Equals method against the canonical singleton is valid for any culture including invariant. F-28.3 primary cell.

### [DEFAULT MITIGATION]

three mitigation options exist; option (iii) is the default:

option (i) — Name string compare with empty string assertion:
```
$c_culture = ($culture.Name -eq '')
```
NOT adopted: Name property contract not stable across .NET versions; ambiguous against any future culture with empty Name; semantically opaque (intent not self-evident from code).

option (ii) — DisplayName or EnglishName compare:
```
$c_culture = ($culture.DisplayName -eq 'Invariant Language (Invariant Country)')
```
NOT adopted: locale-dependent string representation; breaks under non-English PS host configurations; brittle to .NET localization updates.

option (iii) — Equals method against canonical singleton [DEFAULT MITIGATION]:
```
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$c_culture = [System.Globalization.CultureInfo]::InvariantCulture.Equals($ci)
```
ADOPTED: type-safe reference equality semantics via overridden Equals method on the singleton; stable across PS versions, locales, and host configurations; intent self-evident from code (assertion target is the canonical InvariantCulture object itself).

### evidence

- anchor 28 v0.1 round packet 4d-prepare phase initial dispatch: option (i)-equivalent pattern returned False, FAIL verdict
- anchor 28 v0.1 round packet 4d-prepare corrective dispatch: option (iii) Equals method PASS
- anchor 28.1 v0.1 round paired sync verify S.1: option (iii) inline embedded, `L-Q3-54 culture : True` per OVERALL PASS
- proposal A acceptance (Sakaguchi-san decision): option (iii) ratified as DEFAULT MITIGATION for F-28.3 culture assertion class

### F-28 triad reference

F-28.3 primary cell (paired with L-Q3-51): together the primary mitigation pair for F-28.3 PS dispatch context-fit class

### cross-reference

- declaration.md §3 (dispatch script discipline section、culture assertion subsection)
- verification_log.md §4.7 (F-28.3 triad documentation subsection、option (iii) decision rationale)
- input_files_pin.json: active_patterns["L-Q3-54"] field、`first_inscribed` field = "anchor 28 v0.1 round 内 manifest"
- prophylactic octet member (8th entry of post-anchor-28.1 octet)
- companion: L-Q3-51 (Phase parameter design-intent context fit、F-28.3 primary pair)

## F-28 triad documentation

F-28 is the anchor 28 v0.1 round's primary findings cluster, manifesting as three distinct false-negative cells in dispatch script attest discipline. Each cell received option (a) instrument-side acceptance (root cause traced to dispatch script discipline gap, not target system pathology), with codified root cause lessons inscribed in the anchor 28.1 v0.1 round (this round). All three cells are now closed via primary mitigation lesson codification.

### F-28.1 — instrument-side false-negative class (working tree attest)

| field | value |
|---|---|
| cell symptom | git status porcelain directory-level collapse heuristic masking per-file untrackness |
| option (a) acceptance | yes |
| primary mitigation | L-Q3-48 (`--untracked-files=all` per-file enumeration) |
| preemptive mitigation | L-Q3-49 (`^#` any pattern for SHA256SUMS line-type accounting) |
| companion artifact discipline | SHA256SUMS canonical attest discipline (line-type accounting robustness) |
| status | closed (primary + preemptive both codified、distinction explicit) |

### F-28.2 — dispatch script lifecycle gap class

| field | value |
|---|---|
| cell symptom | Mandatory param attribute supplied without invocation argument, producing interactive prompt during expected non-interactive dispatch |
| option (a) acceptance | yes |
| primary mitigation | L-Q3-50 (Mandatory parameter coverage discipline、prepare-phase + exec-phase) |
| lifecycle distinction | prepare-phase (claude.ai supply-time attest) + exec-phase (Claude Code pre-exec attest) explicit |
| status | closed (primary codified、lifecycle distinction explicit) |

### F-28.3 — PS dispatch context-fit class

| field | value |
|---|---|
| cell symptom | culture assertion via Name property returning False against invariant culture (silent false-negative); related dispatch context-fit discipline gaps in Phase parameter, variable interpolation, and refspec construction |
| option (a) acceptance | yes |
| primary mitigation pair | L-Q3-51 (Phase parameter design-intent context fit) + L-Q3-54 (Equals method culture assertion [DEFAULT MITIGATION] = option (iii)) |
| complementary pair | L-Q3-52 (`${var}` delimit) + L-Q3-53 (wildcard refspec) |
| status | closed (primary pair + complementary pair both codified) |

### F-28.4 (recovery-class、separate from triad)

Distinct from the F-28 triad (instrument-side false-negative class). F-28.4 manifested at the anchor 28.1 v0.1 round opening Step 2 Layer C v1.1 baseline re-attest as NOT LOCATED (Public repo working tree + Windows staging dir 11-extension scan 0 hit). Recovery deferred to anchor 28.2 sub-round or anchor 29. Not a lesson-codify class; not inscribed in this appendix beyond reference. Manifest record inscribed in declaration.md §2 footnote + §4.2 + verification_log.md §4.4 + §6.5 + input_files_pin.json `f_28_4_recovery_class_finding` field.

## metadata block

### codification scope tally

| metric | pre-anchor-28.1 | post-anchor-28.1 (this round closure) | delta |
|---|---|---|---|
| active mitigation patterns | 15 (12 inherited + 3 anchor 28 v0.1) | 22 (15 + 7 anchor 28.1 v0.1) | +7 |
| deferred queue entries | 7 (L-Q3-48..L-Q3-54) | 0 (full transition to active) | -7 |
| F-28 triad open cells | 3 (F-28.1 / F-28.2 / F-28.3 all option (a) pending codify) | 0 (all closed via lesson codification) | -3 |
| recovery-class findings | 0 | 1 (F-28.4 inscribed, recovery deferred) | +1 |
| prophylactic octet count | 4 (quartet: Pattern 35 + 39 + 46 + L-Q3-47) | 8 (octet) | +4 |
| forensic chain depth | 7 (anchor 22 v0.2 → 28 v0.1) | 8 (anchor 22 v0.2 → 28.1 v0.1) projected | +1 |

### lesson-to-pattern naming continuity

L-Q3-48 through L-Q3-54 inherits the L-Q3-N numbering scheme from Phase Q-3 verification work. Pattern N (24c / 29-ref / 30-ref / 31 / 34 / 35 / 36 / 38 / 39 / 40 / 41 / 44 / 45 / 46) and L-Q3-N (47 / 48 / 49 / 50 / 51 / 52 / 53 / 54) coexist in the active mitigation patterns roster post-closure; both forms are equally authoritative. The transition from Pattern N to L-Q3-N naming reflects the shift from cross-round structural patterns (Pattern N, infrastructure-class) to Phase Q-3 codify-round-specific lessons (L-Q3-N, discipline-class), with anchor 28 v0.1 round serving as the transitional round (Pattern 38 + Pattern 46 + L-Q3-47 all first inscribed at anchor 28 v0.1).

### prophylactic octet enumeration

| position | entry | class | first inscribed round |
|---|---|---|---|
| 1 | Pattern 35 | InvariantCulture explicit | anchor 22 v0.2 era |
| 2 | Pattern 39 | PS+.NET CWD sync base | anchor 22-26 era |
| 3 | Pattern 46 | byte-level 6-axis canonical metric | anchor 28 v0.1 |
| 4 | L-Q3-47 | Pattern 39 canonical invocation form | anchor 28 v0.1 |
| 5 | L-Q3-48 | --untracked-files=all | anchor 28.1 v0.1 (this round) |
| 6 | L-Q3-52 | ${var} delimit | anchor 28.1 v0.1 (this round) |
| 7 | L-Q3-53 | wildcard refspec | anchor 28.1 v0.1 (this round) |
| 8 | L-Q3-54 | Equals method assertion (option (iii) [DEFAULT MITIGATION]) | anchor 28.1 v0.1 (this round) |

quartet → octet transition: +4 entries, doubling. anchor 28.1 v0.1 round contribution = 4 of the 8 entries (50% of post-closure octet); naming continuity musical-class (quartet → octet) preserved.

### L-Q3-55 manifest record reference (option B per、anchor 28.2 sub-round formal codify scheduled)

L-Q3-55 candidate (cross-locus reconstruction class error) manifested during anchor 28.1 v0.1 round 1.1 initial inscribe stage. Decision per Sakaguchi-san 2026-05-15 authorization: option B (formal codification deferred to anchor 28.2 sub-round) + option II (manifest record entry inscribed in anchor 28.1 v0.1 round artifacts). Formal codify is NOT included in this lessons_appendix.md (proposal B retroactive amendment prohibition compliance — section11/ read-only + section12/ codify scope = L-Q3-48..54 only). Manifest record entry inscribed in:

- verification_log.md §6.6 (L-Q3-55 candidate manifest evidence)
- input_files_pin.json `l_q3_55_candidate_manifest_record` field

Reference inscribed in this appendix solely for completeness of round 28.1 v0.1 codify scope clarification.

### 1-round-delay pattern observation (meta-codify candidate)

three instances accumulated in anchor 28 → 28.1 sequence:

| instance | reveal round | codify or resolve round | delay |
|---|---|---|---|
| L-Q3-48..54 codify | anchor 28 v0.1 | anchor 28.1 v0.1 (this round) | 1 round |
| L-Q3-55 codify | anchor 28.1 v0.1 | anchor 28.2 v0.1 (option B) | 1 round projected |
| F-28.4 recovery | anchor 28.1 v0.1 | anchor 28.2 sub-round or anchor 29 | 1+ round projected |

meta-codify candidate: 1-round-delay pattern itself as a Pattern 31 self-cover discipline meta-layer extension. NOT adopted in anchor 28.1 internal codify (cumulative evidence threshold consideration: 3 instances current, 4-instance threshold preferred for meta-codify); anchor 28.2 re-evaluation pending with potential 4th instance accumulation.

### round closure status (projected post-packet 4d)

- forensic chain 8-deep IMMUTABLE LOCK-IN: pending packet 4d annotated tag inscribe + push
- target tag candidate: `companion-v4.9-q8-codify-round-2026-05-15`
- target commit message scope: `anchor 28.1 v0.1 Q8 codify round: L-Q3-48..54 + F-28 triad + F-28.4 + L-Q3-55 manifest`
- rule 92 strict compliance: no --force / --all / --tags / --mirror throughout
- proposal B compliance: section11/ read-only preserved; all new inscription in section12/
- proposal A compliance: L-Q3-54 [DEFAULT MITIGATION] = option (iii) ratified, dispatch evidence inscribed
- proposal C-as-6 parallel axis: Component 6 operational-protocol skeleton inherited (paired sync verify protocol)

end of anchor 28.1 v0.1 lessons appendix (Q8 codify round)
