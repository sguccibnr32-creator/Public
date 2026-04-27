"""
Strict 100% recompute — pass 2

Re-runs the 17 TIMEOUT scripts from pass 1 with a 1800s timeout (was 120s).
These are heavy E:/ HSC pipeline / probe scripts that need >2 minutes.

Output: full_recompute_strict_pass2_progress.jsonl + result.json/txt.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PASS1_RESULT = ROOT / "full_recompute_strict_runner_result.json"
PROG = ROOT / "full_recompute_strict_pass2_progress.jsonl"
RES_JSON = ROOT / "full_recompute_strict_pass2_result.json"
RES_TXT = ROOT / "full_recompute_strict_pass2_result.txt"

PASS2_TIMEOUT_S = 1800  # 30 min per script


def run_one(path_str: str) -> dict:
    path = Path(path_str)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    rec: dict = {
        "path": path_str,
        "timeout_s": PASS2_TIMEOUT_S,
        "rc": None,
        "elapsed_s": None,
        "verdict": None,
        "stdout_tail": None,
        "stderr_tail": None,
    }
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            timeout=PASS2_TIMEOUT_S,
            cwd=str(path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
        )
        rec["rc"] = r.returncode
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        rec["stdout_tail"] = r.stdout.decode("utf-8", errors="replace")[-2000:]
        rec["stderr_tail"] = r.stderr.decode("utf-8", errors="replace")[-2000:]
        rec["verdict"] = "PASS" if r.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired as e:
        rec["rc"] = -1
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        rec["stdout_tail"] = (e.stdout or b"").decode("utf-8", errors="replace")[-2000:]
        rec["stderr_tail"] = ((e.stderr or b"").decode("utf-8", errors="replace")
                              + f"\n[TIMEOUT after {PASS2_TIMEOUT_S}s]")[-2000:]
        rec["verdict"] = "TIMEOUT"
    except Exception as ex:
        rec["rc"] = -2
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        rec["stdout_tail"] = ""
        rec["stderr_tail"] = f"{type(ex).__name__}: {ex}"
        rec["verdict"] = "FAIL"
    return rec


def already_done() -> set[str]:
    done = set()
    if not PROG.exists():
        return done
    with open(PROG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["path"])
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    with open(PASS1_RESULT, encoding="utf-8") as f:
        pass1 = json.load(f)
    targets = [r["path"] for r in pass1["records"] if r["verdict"] == "TIMEOUT"]
    targets.sort()
    done = already_done()
    todo = [p for p in targets if p not in done]
    print(f"[pass2] targets={len(targets)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.monotonic()
    with open(PROG, "a", encoding="utf-8") as prog:
        for i, p in enumerate(todo, 1):
            print(f"[pass2 {i}/{len(todo)}] {p}", flush=True)
            r = run_one(p)
            prog.write(json.dumps(r, ensure_ascii=False) + "\n")
            prog.flush()
            print(f"  -> {r['verdict']:8s}  t={r['elapsed_s']}s "
                  f"(cumulative {time.monotonic()-t0:.0f}s)", flush=True)

    records = []
    with open(PROG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    from collections import Counter
    by_verdict = Counter(r["verdict"] for r in records)
    summary = {
        "total": len(records),
        "by_verdict": dict(by_verdict),
        "timeout_s": PASS2_TIMEOUT_S,
        "total_elapsed_s": round(sum(r["elapsed_s"] or 0 for r in records), 1),
    }
    with open(RES_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records},
                  f, ensure_ascii=False, indent=2)
    with open(RES_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Pass 2 (TIMEOUT re-run @ {PASS2_TIMEOUT_S}s timeout)\n\n")
        f.write(f"total: {summary['total']}\n")
        f.write(f"wall : {summary['total_elapsed_s']}s\n")
        f.write(f"by_verdict: {summary['by_verdict']}\n\n")
        for r in sorted(records, key=lambda x: (x["verdict"], x["path"])):
            f.write(f"  {r['verdict']:8s}  rc={r['rc']!s:>4s}  t={r['elapsed_s']!s:>8s}s  {r['path']}\n")
            stderr = (r.get("stderr_tail") or "").strip().splitlines()
            if stderr and r["verdict"] != "PASS":
                f.write(f"    stderr: {stderr[-1][:200]}\n")
    print(f"[pass2] verdicts: {dict(by_verdict)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
