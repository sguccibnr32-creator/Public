#!/usr/bin/env python3
"""
full_recompute_tier1_runner.py
------------------------------
Re-execute canonical Tier-1 numerical-claim-producing scripts under
D:/ドキュメント/エントロピー/ and aggregate stdout + extracted numerical
outputs into a single forensic JSON manifest.

Tier 1 = lightweight (seconds-minutes), no E:/ raw HSC dependency.
Tier 2 = heavy (hours), separate scheduling.

Each script is run with subprocess.run(timeout=...) and its stdout +
return code captured. A best-effort numerical extractor pulls key
numbers (gc, AIC, sigma, chi2, R^2, etc.) from stdout for cross-check.
"""

import subprocess
import hashlib
import json
import os
import sys
import re
import time
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
PYTHON = sys.executable

# Tier 1 canonical script list — lightweight, runnable in seconds-minutes
# Each entry: (script_relpath, timeout_sec, claim_extract_pattern_list, expected_paper_anchor)
TIER1 = [
    # 5.2.M v1 canonical chain
    ("_step5_2_M_multi_route_min.py", 300, ["OVERALL", "PASS", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "canonical", "central"], "Phase 5-2 5.2.M v1: G1-G7 PASS, canonical 33"),
    # Phase 5-2 sub-steps (A: Path A uncertainty propagation)
    ("_step5_2_a_path_a_uncertainty_propagation.py", 300, ["central", "band", "sigma", "1.002582"], "Path A 1.0025822368421053"),
    # 5.2.B-1 FIRAS posterior
    ("_step5_2_b1_firas_posterior.py", 300, ["FIRAS", "OVERALL", "median", "alpha_PT"], "FIRAS B-1 prior-dominated"),
    # 5.2.B-3 21cm posterior
    ("_step5_2_b3_21cm_posterior.py", 300, ["21cm", "SARAS", "EDGES", "OVERALL", "median"], "21cm SARAS3 dominant"),
    # 5.2.B-4 joint posterior
    ("_step5_2_b4_joint_posterior.py", 300, ["joint", "OVERALL", "PASS", "log_diff", "median"], "B-4 joint, 6 reproductions log_diff = 0"),
    # 5.2.C cascade SSoT consistency
    ("_step5_2_c_cascade_ssot_consistency.py", 300, ["cascade", "SSoT", "OVERALL", "PASS", "log_diff"], "C cascade SSoT 8 gates PASS"),
    # 5.2.M G3 TIER variant accounting
    ("_step5_2_m_g3_variant_accounting.py", 300, ["TIER", "canonical", "OVERALL", "PASS", "33"], "G3 PASS canonical 33"),
    # 5.2.M Method 2 A1 epsilon scale grep
    ("_step5_2_method2_a1_epsilon_scale_grep.py", 300, ["epsilon", "1.002", "Track A", "Method 2"], "Method 2 Track A 1.0026"),
    # Phase 5-2 pre-C resync
    ("_step5_2_pre_c_resync.py", 300, ["resync", "OVERALL", "PASS", "SSoT"], "pre-C resync"),
    # Phase 5-3 step 2 NGC 3198 C15 prediction (already in JSON, re-run for consistency)
    ("phase5_3_step2_c15_predict.py", 300, ["NGC", "3198", "gc_predicted", "method_ratio", "MARGINAL"], "Phase 5-3 step 2 NGC 3198 C15 1.07 sigma MARGINAL"),
    # foundation_gamma_actual cascade SSoT canonical
    ("foundation_gamma_actual.py", 600, ["c_mem", "Lambda_UV", "alpha_PT_upper", "NGC", "3198", "0.42", "0.83"], "Cascade SSoT canonical b0cb36d7..."),
    # step_iv_d FIRAS mu bound
    ("step_iv_d_firas_mu_bound.py", 600, ["alpha_PT_upper", "1.76e-51", "FIRAS", "mu", "Chluba"], "FIRAS mu bound, alpha_PT 1.76e-51"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_numerics(stdout, patterns):
    """Best-effort numerical extraction. Returns matched lines per pattern."""
    matches = {}
    lines = stdout.splitlines()
    for p in patterns:
        hits = [ln.strip() for ln in lines if p in ln]
        if hits:
            matches[p] = hits[:6]  # cap at 6 per pattern to keep manifest sane
    return matches


def run_script(script_relpath, timeout_sec, patterns):
    script_path = os.path.join(BASE, script_relpath)
    if not os.path.isfile(script_path):
        return {
            "status": "SKIP",
            "reason": "script_not_found",
            "script_path": script_relpath,
        }
    sha = sha256_file(script_path)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [PYTHON, script_path],
            cwd=BASE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return_code = proc.returncode
        # tail to keep size sane
        stdout_tail = "\n".join(stdout.splitlines()[-200:])
        stderr_tail = "\n".join(stderr.splitlines()[-50:])
        matches = extract_numerics(stdout, patterns)
        status = "PASS" if return_code == 0 else "FAIL"
        return {
            "status": status,
            "script_path": script_relpath,
            "script_sha256": sha,
            "return_code": return_code,
            "elapsed_sec": round(elapsed, 2),
            "stdout_total_lines": len(stdout.splitlines()),
            "stderr_total_lines": len(stderr.splitlines()),
            "stdout_tail_200": stdout_tail,
            "stderr_tail_50": stderr_tail,
            "extracted_patterns": matches,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "script_path": script_relpath,
            "script_sha256": sha,
            "timeout_sec": timeout_sec,
            "elapsed_sec": time.time() - t0,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "script_path": script_relpath,
            "script_sha256": sha,
            "error": str(e),
            "elapsed_sec": time.time() - t0,
        }


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = []
    summary = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0, "SKIP": 0}

    for script_relpath, timeout_sec, patterns, anchor in TIER1:
        print(f"[Tier1] Running: {script_relpath} (timeout {timeout_sec}s)")
        sys.stdout.flush()
        r = run_script(script_relpath, timeout_sec, patterns)
        r["paper_anchor_hint"] = anchor
        results.append(r)
        summary[r["status"]] = summary.get(r["status"], 0) + 1
        flag = {"PASS": "OK", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT", "ERROR": "ERROR", "SKIP": "SKIP"}[r["status"]]
        elapsed = r.get("elapsed_sec", 0)
        print(f"  -> {flag} (elapsed {elapsed:.1f}s)")

    manifest = {
        "schema_version": "full_recompute_tier1_runner_v1.0",
        "purpose": "Tier-1 canonical claim-producing script re-run forensic anchor",
        "execution_timestamp_utc": t0,
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "python_executable": PYTHON,
        "python_version": sys.version,
        "base_dir": BASE,
        "n_scripts_tier1": len(TIER1),
        "summary": summary,
        "results": results,
    }

    out_json = os.path.join(OUT_DIR, "full_recompute_tier1_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(OUT_DIR, "full_recompute_tier1_result.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Full Recompute — Tier 1 Canonical Script Re-run Forensic Anchor\n")
        f.write("=" * 70 + "\n")
        f.write(f"Execution UTC : {t0}\n")
        f.write(f"Python        : {sys.version.splitlines()[0]}\n")
        f.write(f"Total Tier 1  : {len(TIER1)}\n")
        f.write(f"Summary       : {summary}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Per-script verdict\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"\n[{r['status']}] {r['script_path']}\n")
            f.write(f"  SHA256: {r.get('script_sha256', 'N/A')}\n")
            f.write(f"  Anchor: {r.get('paper_anchor_hint', '')}\n")
            f.write(f"  Elapsed: {r.get('elapsed_sec', 'N/A')}s\n")
            if r.get("return_code") is not None:
                f.write(f"  RC: {r['return_code']}\n")
            if r.get("extracted_patterns"):
                f.write("  Key extracts:\n")
                for p, hits in r["extracted_patterns"].items():
                    f.write(f"    [{p}]\n")
                    for h in hits:
                        f.write(f"      {h}\n")

    print()
    print("=" * 70)
    print("Tier 1 re-run COMPLETE")
    print("=" * 70)
    print(f"  Summary: {summary}")
    print(f"  Output JSON: {out_json}")
    print(f"  Output TXT : {out_txt}")
    print(f"  JSON SHA256: {sha256_file(out_json)}")
    print(f"  TXT  SHA256: {sha256_file(out_txt)}")


if __name__ == "__main__":
    main()
