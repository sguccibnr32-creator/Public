#!/usr/bin/env python3
"""
full_recompute_tier3_runner.py
------------------------------
Tier-3: variant scripts + cascade SSoT mirror sites + lightweight env verifies.
"""

import subprocess
import hashlib
import json
import os
import sys
import time
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
AB_BASE = r"D:\ドキュメント\エントロピー\膜宇宙論再考察AB効果有り"

# Tier 3: variants + SSoT mirror execution check
TIER3 = [
    # variant scripts
    ("cluster_stack_v3_fix.py", 600, ["fix", "cluster"], BASE, "Cluster stack v3 fix variant"),
    ("cluster_stack_v3_fix2.py", 600, ["fix2", "cluster"], BASE, "Cluster stack v3 fix2 variant"),
    ("little_things_external_v1.py", 600, ["LITTLE", "external", "v1"], BASE, "Little Things external v1"),
    ("little_things_external_v2.py", 600, ["LITTLE", "external", "v2"], BASE, "Little Things external v2"),
    ("little_things_fetch.py", 300, ["fetch", "Hunter", "LITTLE"], BASE, "Little Things data fetch"),
    ("phase_c3_step2_sparc_lambda.py", 600, ["lambda", "C3", "step2"], BASE, "Phase C3 step 2 SPARC lambda"),
    ("phase_c3_step2_sparc_lambda_v2.py", 600, ["lambda", "v2", "C3"], BASE, "Phase C3 step 2 SPARC lambda v2"),
    ("phase_c3_step2_sparc_lambda_v3.py", 600, ["lambda", "v3", "C3"], BASE, "Phase C3 step 2 SPARC lambda v3"),
    ("_step5_2_b2_plik_env_verify.py", 300, ["Plik", "env", "verify", "OK", "PASS"], BASE, "Plik env verify (5.2.B-2 prerequisite)"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_numerics(stdout, patterns):
    matches = {}
    lines = stdout.splitlines()
    for p in patterns:
        hits = [ln.strip() for ln in lines if p.lower() in ln.lower()]
        if hits:
            matches[p] = hits[:6]
    return matches


def run_script(script_relpath, timeout_sec, patterns, cwd):
    script_path = os.path.join(cwd, script_relpath)
    if not os.path.isfile(script_path):
        return {"status": "SKIP", "reason": "script_not_found", "script_path": script_relpath}
    sha = sha256_file(script_path)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run([PYTHON, script_path], cwd=cwd, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_sec, env=env)
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode
        stdout_tail = "\n".join(stdout.splitlines()[-200:])
        stderr_tail = "\n".join(stderr.splitlines()[-50:])
        matches = extract_numerics(stdout, patterns)
        status = "PASS" if rc == 0 else "FAIL"
        return {"status": status, "script_path": script_relpath, "script_sha256": sha,
                "cwd": cwd, "return_code": rc, "elapsed_sec": round(elapsed, 2),
                "stdout_total_lines": len(stdout.splitlines()),
                "stderr_total_lines": len(stderr.splitlines()),
                "stdout_tail_200": stdout_tail, "stderr_tail_50": stderr_tail,
                "extracted_patterns": matches}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "script_path": script_relpath, "script_sha256": sha,
                "cwd": cwd, "timeout_sec": timeout_sec, "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"status": "ERROR", "script_path": script_relpath, "script_sha256": sha,
                "cwd": cwd, "error": str(e), "elapsed_sec": time.time() - t0}


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = []
    summary = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0, "SKIP": 0}

    for script_relpath, timeout_sec, patterns, cwd, anchor in TIER3:
        print(f"[Tier3] Running: {script_relpath} (cwd: {cwd}, timeout {timeout_sec}s)")
        sys.stdout.flush()
        r = run_script(script_relpath, timeout_sec, patterns, cwd)
        r["paper_anchor_hint"] = anchor
        results.append(r)
        summary[r["status"]] = summary.get(r["status"], 0) + 1
        flag = {"PASS": "OK", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT",
                "ERROR": "ERROR", "SKIP": "SKIP"}[r["status"]]
        elapsed = r.get("elapsed_sec", 0)
        print(f"  -> {flag} (elapsed {elapsed:.1f}s)")

    manifest = {
        "schema_version": "full_recompute_tier3_runner_v1.0",
        "purpose": "Tier-3 variant + mirror site re-run forensic anchor",
        "execution_timestamp_utc": t0,
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "n_scripts_tier3": len(TIER3),
        "summary": summary,
        "results": results,
    }

    out_json = os.path.join(BASE, "full_recompute_tier3_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("Tier 3 re-run COMPLETE")
    print("=" * 70)
    print(f"  Summary: {summary}")
    print(f"  Output JSON: {out_json}")
    print(f"  JSON SHA256: {sha256_file(out_json)}")


if __name__ == "__main__":
    main()
