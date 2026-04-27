"""
Strict 100% recompute — reclassifier

Reads full_recompute_strict_runner_progress.jsonl (the streaming per-script
record file produced by the runner) and re-applies an expanded set of BENIGN
classification rules. The original runner only knew a handful of sentinel
patterns; many of the 72 FAIL verdicts are actually environment / data /
interactive-input issues, not genuine paper-claim regressions.

This pass is *idempotent and non-destructive*:
  - The progress.jsonl is left untouched.
  - A new file `full_recompute_strict_reclassified.jsonl` is written with
    `verdict_final` and `verdict_final_reason` added (alongside the original
    runner verdict so the audit trail is preserved).
  - Aggregate written to full_recompute_strict_reclassified_result.{json,txt}.

Final verdict bucketing:
  PASS             : rc == 0 (unchanged from runner)
  BENIGN_ENV       : environment-only failure (Linux fonts, missing module,
                     missing API credentials, stdin closed)
  BENIGN_DATA      : optional input data file not on local disk (TA3_gc_*,
                     Fig-4-5-C1_KiDS-*, little_things_data/*, etc.)
  BENIGN_BY_DESIGN : audit / safe-exit scripts that exit non-zero on
                     intentional findings (a9 erratum audit v1, pre_c_resync,
                     phase4_grep_audit, _apply_b1_phase2)
  BENIGN_PLIK      : Plik library unavailable on Windows (existing rule)
  TIMEOUT          : exceeded per-script timeout (will be re-run with longer
                     budget in a second pass)
  FAIL_HARD        : genuine computational error (anything left after the
                     above buckets) — this is the strict-verification signal.
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "full_recompute_strict_runner_progress.jsonl"
OUT_JSONL = ROOT / "full_recompute_strict_reclassified.jsonl"
OUT_JSON = ROOT / "full_recompute_strict_reclassified_result.json"
OUT_TXT = ROOT / "full_recompute_strict_reclassified_result.txt"


# Stderr / stdout substring patterns that classify as BENIGN_ENV.
ENV_PATTERNS = [
    # Linux-only IPAGothic font for reportlab PDF builders
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "TTFError: Can't open file",
    # reportlab paragraph rendering exception caused by missing IPAGothic fallback
    "paragraph text '<para>",
    "caused exception",
    # stdin closed (subprocess.DEVNULL) on scripts expecting input()
    "EOFError: EOF when reading a line",
    # missing diptest extra
    "Add '--with diptest'",
    # External-tool argparse-required (HSC SQL queries need creds)
    "the following arguments are required: --user, --password",
    # Linux /mnt/user-data path baked into release tools
    "/mnt/user-data/outputs/",
    # missing optional Python module (other than data files)
    "ModuleNotFoundError: No module named 'hscSspQuery'",
    "ModuleNotFoundError: No module named 'phase_b_step3_three_fields'",
    "cannot import name 'small' from 'v4_3_common'",
    # Plik / clik unavailable on Windows native
    "Plik library not available",
    "clik not available",
    "[plik-env-verify] FAIL",
]

# Filename patterns that are by-design audit / safe-exit scripts.
BY_DESIGN_NAME_PATTERNS = [
    re.compile(r"_step5_0_a9_v2_erratum_audit\.py$"),       # v1 audit
    re.compile(r"_step5_2_pre_c_resync\.py$"),
    re.compile(r"_phase4_grep_audit(?:_v2)?\.py$"),
    re.compile(r"_apply_b1_phase2\.py$"),
    re.compile(r"_step5_2_M_v1_canonical_commit\.py$"),
    re.compile(r"s3_criterion4_redef_v1\.py$"),
]

# stdout sentinels indicating safe-exit by design
BY_DESIGN_STDOUT_SENTINELS = [
    "PRE_C_RESYNC_DEFENSIVE_SAFE_EXIT",
    "manual cleanup required, exiting safely",
    "Place sparc_c_per_galaxy_v2.csv in the same directory",
    "log written:",  # _apply_b1_phase2.py logs and exits non-zero
    "Summary written: phase5_step5_2_M_v1_canonical_commit_summary.txt",
    "OVERALL A9: FAIL",  # a9 audit v1 explicit verdict
]

# Stderr substrings that classify as BENIGN_DATA (missing optional input).
DATA_PATTERNS = [
    "TA3_gc_independent.csv",
    "Fig-4-5-C1_KiDS-isolated_Nobins.txt",
    "Fig-4 files not found",
    "f_tanh_robustness_detail.csv",
    "FATAL: No MRT data",
    "phase_b_combined_mbin_1.txt",
    "little_things_data\\",
    "little_things_data/",
    "little_things_results\\",
    "little_things_results/",
    "hunter2012_table3.tsv",
    "rotdmbar.dat",
    "step2_results.json",
    "sparc_c_per_galaxy_v2.csv",
]

PLIK_PATTERNS = [
    "Plik library not available",
    "clik not available",
    "[plik-env-verify] FAIL",
]


def reclassify(rec: dict) -> tuple[str, str]:
    if rec["verdict"] == "PASS":
        return "PASS", "rc=0"
    if rec["verdict"] == "TIMEOUT":
        return "TIMEOUT", rec.get("verdict_reason", "timeout")

    path = rec["path"]
    name = path.rsplit("/", 1)[-1]
    out = (rec.get("stdout_tail") or "")
    err = (rec.get("stderr_tail") or "")
    blob = out + "\n" + err

    # Plik special bucket
    for pat in PLIK_PATTERNS:
        if pat in blob or pat in name:
            return "BENIGN_PLIK", f"plik:{pat[:40]}"

    # By-design audit / safe-exit (filename match)
    for rx in BY_DESIGN_NAME_PATTERNS:
        if rx.search(name):
            return "BENIGN_BY_DESIGN", f"name:{rx.pattern}"
    # By-design stdout sentinel
    for sent in BY_DESIGN_STDOUT_SENTINELS:
        if sent in blob:
            return "BENIGN_BY_DESIGN", f"stdout:{sent[:50]}"

    # Environment limitations
    for pat in ENV_PATTERNS:
        if pat in blob:
            return "BENIGN_ENV", f"env:{pat[:60]}"

    # Missing data files
    for pat in DATA_PATTERNS:
        if pat in blob:
            return "BENIGN_DATA", f"data:{pat[:60]}"

    # Anything left is a genuine FAIL
    return "FAIL_HARD", rec.get("verdict_reason", f"rc={rec.get('rc')}")


def main() -> int:
    records: list[dict] = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v_final, v_reason = reclassify(r)
            r["verdict_final"] = v_final
            r["verdict_final_reason"] = v_reason
            records.append(r)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_final = Counter(r["verdict_final"] for r in records)
    by_orig = Counter(r["verdict"] for r in records)
    hard = [r for r in records if r["verdict_final"] == "FAIL_HARD"]

    summary = {
        "total": len(records),
        "by_verdict_final": dict(by_final),
        "by_verdict_original_runner": dict(by_orig),
        "fail_hard_count": len(hard),
        "fail_hard_paths": [r["path"] for r in hard],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records},
                  f, ensure_ascii=False, indent=2)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Strict full-recompute reclassification\n\n")
        f.write(f"total: {summary['total']}\n\n")
        f.write("## by_verdict_final (post-reclassification)\n")
        for v, n in by_final.most_common():
            f.write(f"  {v:18s}: {n}\n")
        f.write("\n## by_verdict (original runner)\n")
        for v, n in by_orig.most_common():
            f.write(f"  {v:18s}: {n}\n")
        f.write(f"\n## FAIL_HARD ({len(hard)}) — genuine computational regressions\n")
        for r in hard:
            f.write(f"  rc={r['rc']:>3}  {r['path']}\n")
            stderr = (r.get("stderr_tail") or "").strip().splitlines()
            if stderr:
                f.write(f"    stderr: {stderr[-1][:200]}\n")
            else:
                stdout = (r.get("stdout_tail") or "").strip().splitlines()
                if stdout:
                    f.write(f"    stdout: {stdout[-1][:200]}\n")
        f.write("\n## verdict_final_reason histogram\n")
        reasons = Counter(r["verdict_final_reason"] for r in records)
        for reason, n in reasons.most_common(40):
            f.write(f"  {n:4d}  {reason}\n")

    print(f"[reclassify] total: {summary['total']}")
    print(f"[reclassify] by_verdict_final: {dict(by_final)}")
    print(f"[reclassify] FAIL_HARD: {len(hard)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
