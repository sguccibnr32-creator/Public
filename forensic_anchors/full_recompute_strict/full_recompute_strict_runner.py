"""
Strict 100% recompute — runner

Reads full_recompute_strict_inventory.json, executes every "executable"
script (those with a top-level `if __name__ == "__main__":` block) under
controlled timeout + working-directory + UTF-8 IO, and writes per-script
result records incrementally to full_recompute_strict_runner_progress.jsonl
(streaming JSON Lines, so partial results survive interrupts).

After all scripts complete, aggregates into:
  - full_recompute_strict_runner_result.json
  - full_recompute_strict_runner_result.txt

Verdict mapping:
  - rc == 0                                        -> PASS
  - rc != 0 + classified BENIGN by skip rule       -> BENIGN (Plik unavail / safe-exit)
  - timeout reached                                -> TIMEOUT
  - rc != 0 + not benign                           -> FAIL

Per-script execution context:
  - cwd = script's parent directory (so relative file loads work)
  - env: PYTHONIOENCODING=utf-8 (avoid cp932 codec failures with em-dash etc.)
  - stdin: closed (avoid hangs on accidental input())

Timeouts (heuristic, tuned to keep total runtime tractable):
  - HEAVY_KEYWORDS scripts: 600s (cluster_stack, manga, phase_b_step3, lambda_v3,
                                  fetch, hsc, kids, gama)
  - DEFAULT                : 120s

Skip rules (rc != 0 categorized BENIGN — known design-choice or env-limitation):
  - filename matches "plik_env" / "_step5_2_b2_plik_env_verify"
    (Plik clik library not installed on Windows — full TT runs are WSL2-only)
  - any script whose stderr/stdout contains the persistent
    "PRE_C_RESYNC_DEFENSIVE_SAFE_EXIT" sentinel
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
INVENTORY = ROOT / "full_recompute_strict_inventory.json"
PROGRESS = ROOT / "full_recompute_strict_runner_progress.jsonl"
RESULT_JSON = ROOT / "full_recompute_strict_runner_result.json"
RESULT_TXT = ROOT / "full_recompute_strict_runner_result.txt"

DEFAULT_TIMEOUT_S = 120
HEAVY_TIMEOUT_S = 600

HEAVY_KEYWORDS = (
    "cluster_stack",
    "manga",
    "phase_b_step3",
    "phase_b_step2",
    "lambda_v3",
    "lambda_v2",
    "fetch",
    "hsc_",
    "kids_",
    "gama_",
    "little_things_step",
    "little_things_fetch",
    "_full_",
    "_e_full",
)

# Skip-RUN list: scripts that are known to be unrunnable in this environment
# *and* whose absence does not break paper claims. These are recorded as
# BENIGN with category="skipped".
SKIP_KEYWORDS = (
    # E-drive raw HSC pipeline (10+ GB inputs, multi-hour) — covered by
    # phase_b forensic anchor (5.2.M v2) which uses persisted RAR+JK output.
    # We keep them in scope but mark long-runtime; do NOT auto-skip.
)

# rc!=0 verdicts that should be reclassified as BENIGN
BENIGN_NAME_PATTERNS = (
    "plik_env",
    "_step5_2_b2_plik_env_verify",
)
BENIGN_OUTPUT_SENTINELS = (
    "PRE_C_RESYNC_DEFENSIVE_SAFE_EXIT",
    "Plik library not available",
    "clik not available",
    "[plik-env-verify] FAIL",  # plik_env_verify expected failure on Windows
)


def pick_timeout(path: Path) -> int:
    name = path.name.lower()
    full = str(path).lower()
    if any(k in name or k in full for k in HEAVY_KEYWORDS):
        return HEAVY_TIMEOUT_S
    return DEFAULT_TIMEOUT_S


def is_benign_failure(path: Path, stdout: str, stderr: str) -> tuple[bool, str]:
    name = path.name.lower()
    for pat in BENIGN_NAME_PATTERNS:
        if pat in name:
            return True, f"benign:name_pattern:{pat}"
    blob = (stdout or "") + (stderr or "")
    for sent in BENIGN_OUTPUT_SENTINELS:
        if sent in blob:
            return True, f"benign:output_sentinel:{sent[:30]}"
    return False, ""


def already_done(progress_path: Path) -> set[str]:
    """Return set of paths already processed (resume support)."""
    done: set[str] = set()
    if not progress_path.exists():
        return done
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["path"])
            except json.JSONDecodeError:
                continue
    return done


def run_one(path_str: str) -> dict:
    path = Path(path_str)
    timeout_s = pick_timeout(path)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    rec: dict = {
        "path": path_str,
        "timeout_s": timeout_s,
        "rc": None,
        "elapsed_s": None,
        "verdict": None,
        "verdict_reason": None,
        "stdout_tail": None,
        "stderr_tail": None,
    }
    start = time.monotonic()
    try:
        r = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            timeout=timeout_s,
            cwd=str(path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
        )
        rec["rc"] = r.returncode
        rec["elapsed_s"] = round(time.monotonic() - start, 3)
        # decode safely
        out = r.stdout.decode("utf-8", errors="replace")
        err = r.stderr.decode("utf-8", errors="replace")
        rec["stdout_tail"] = out[-1500:]
        rec["stderr_tail"] = err[-1500:]
        if r.returncode == 0:
            rec["verdict"] = "PASS"
            rec["verdict_reason"] = "rc=0"
        else:
            benign, reason = is_benign_failure(path, out, err)
            if benign:
                rec["verdict"] = "BENIGN"
                rec["verdict_reason"] = reason
            else:
                rec["verdict"] = "FAIL"
                rec["verdict_reason"] = f"rc={r.returncode}"
    except subprocess.TimeoutExpired as e:
        rec["rc"] = -1
        rec["elapsed_s"] = round(time.monotonic() - start, 3)
        out = (e.stdout or b"").decode("utf-8", errors="replace")
        err = (e.stderr or b"").decode("utf-8", errors="replace")
        rec["stdout_tail"] = out[-1500:]
        rec["stderr_tail"] = err[-1500:] + f"\n[TIMEOUT after {timeout_s}s]"
        rec["verdict"] = "TIMEOUT"
        rec["verdict_reason"] = f"timeout_s={timeout_s}"
    except Exception as ex:
        rec["rc"] = -2
        rec["elapsed_s"] = round(time.monotonic() - start, 3)
        rec["stdout_tail"] = ""
        rec["stderr_tail"] = f"{type(ex).__name__}: {ex}"
        rec["verdict"] = "FAIL"
        rec["verdict_reason"] = f"runner_exception:{type(ex).__name__}"
    return rec


def aggregate() -> None:
    records: list[dict] = []
    with open(PROGRESS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    by_verdict: dict[str, int] = {}
    by_verdict_paths: dict[str, list[str]] = {}
    total_elapsed = 0.0
    for r in records:
        v = r["verdict"]
        by_verdict[v] = by_verdict.get(v, 0) + 1
        by_verdict_paths.setdefault(v, []).append(r["path"])
        if r["elapsed_s"]:
            total_elapsed += r["elapsed_s"]

    summary = {
        "total_executable_scripts": len(records),
        "by_verdict": by_verdict,
        "total_elapsed_s": round(total_elapsed, 1),
        "default_timeout_s": DEFAULT_TIMEOUT_S,
        "heavy_timeout_s": HEAVY_TIMEOUT_S,
        "heavy_keywords": list(HEAVY_KEYWORDS),
        "benign_name_patterns": list(BENIGN_NAME_PATTERNS),
        "benign_output_sentinels": list(BENIGN_OUTPUT_SENTINELS),
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "records": records},
            f, ensure_ascii=False, indent=2,
        )

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("# Strict full-recompute runner — aggregate result\n\n")
        f.write(f"total_executable_scripts: {summary['total_executable_scripts']}\n")
        f.write(f"total_elapsed_s        : {summary['total_elapsed_s']}\n")
        f.write(f"default_timeout_s      : {summary['default_timeout_s']}\n")
        f.write(f"heavy_timeout_s        : {summary['heavy_timeout_s']}\n\n")
        f.write("## by_verdict\n")
        for v, n in by_verdict.items():
            f.write(f"  {v:8s}: {n}\n")
        f.write("\n## verdict per file\n")
        for r in sorted(records, key=lambda x: (x["verdict"], x["path"])):
            f.write(f"  {r['verdict']:8s}  rc={r['rc']!s:>4s}  t={r['elapsed_s']!s:>7s}s  {r['path']}\n")
            if r["verdict"] in ("FAIL", "TIMEOUT"):
                stderr_tail = (r["stderr_tail"] or "").strip().splitlines()
                if stderr_tail:
                    f.write(f"    last stderr: {stderr_tail[-1][:200]}\n")
        f.write("\n## verdict_reason histogram\n")
        from collections import Counter
        reasons = Counter(r["verdict_reason"] for r in records)
        for reason, n in reasons.most_common(30):
            f.write(f"  {n:4d}  {reason}\n")
    print(f"[runner] wrote {RESULT_JSON.name} + {RESULT_TXT.name}", flush=True)
    print(f"[runner] verdicts: {by_verdict}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run first N executable scripts (0 = all)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip running, just aggregate from progress.jsonl")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return 0

    with open(INVENTORY, encoding="utf-8") as f:
        inv = json.load(f)

    targets = [r for r in inv["records"] if r["category"] == "executable"]
    targets.sort(key=lambda r: r["path"])
    if args.limit > 0:
        targets = targets[: args.limit]

    done = already_done(PROGRESS)
    if done:
        print(f"[runner] resuming: {len(done)} already complete", flush=True)
    todo = [r for r in targets if r["path"] not in done]
    print(f"[runner] total targets: {len(targets)}, todo: {len(todo)}", flush=True)

    t0 = time.monotonic()
    with open(PROGRESS, "a", encoding="utf-8") as prog:
        for i, rec in enumerate(todo, 1):
            path_str = rec["path"]
            print(f"[{i}/{len(todo)}] {path_str}", flush=True)
            result = run_one(path_str)
            prog.write(json.dumps(result, ensure_ascii=False) + "\n")
            prog.flush()
            elapsed = time.monotonic() - t0
            print(f"  -> {result['verdict']:8s}  t={result['elapsed_s']}s  "
                  f"(cumulative {elapsed:.0f}s)", flush=True)

    aggregate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
