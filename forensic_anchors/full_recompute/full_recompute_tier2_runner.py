#!/usr/bin/env python3
"""
full_recompute_tier2_runner.py
------------------------------
Re-execute Tier-2 (auxiliary numerical claim sources, lightweight) scripts.
"""

import subprocess
import hashlib
import json
import os
import sys
import time
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
PYTHON = sys.executable

# Tier 2: auxiliary scripts (SPARC fits, dSph, cluster, joint, manga, little_things)
# Each: (script_relpath, timeout_sec, claim_extract_patterns, expected_paper_anchor)
TIER2 = [
    ("sparc_cond15_alpha_transition.py", 600, ["alpha", "transition", "C15", "AIC", "R^2", "175", "PASS"],
     "SPARC C15 alpha transition (175 galaxies, R^2=0.607)"),
    ("sparc_cond15_residual_structure.py", 600, ["residual", "AIC", "R", "175", "C15"],
     "SPARC C15 residual structure"),
    ("sparc_cond15_sgal_model.py", 600, ["sgal", "C15", "TA3", "0.584", "0.36"],
     "SPARC C15 sgal model (gc = 0.584 * Y_d^-0.36 * ...)"),
    ("sparc_cond15_bias_cond14_correction.py", 600, ["C14", "C15", "bias", "correction", "AIC"],
     "SPARC C14/C15 bias correction"),
    ("dsph_jeans_c15_v1.py", 600, ["dSph", "Jeans", "C15", "0.240", "0.228", "G_Strigari", "31"],
     "dSph 31 galaxies G_Strigari = 0.240 a_0 (paper M2)"),
    ("cluster_stack_v3.py", 900, ["cluster", "stack", "shear", "AIC", "Z=", "z="],
     "Cluster stack v3 (16 clusters)"),
    ("joint_analysis.py", 900, ["joint", "AIC", "dAIC", "SPARC", "kids", "GAMA"],
     "Joint SPARC + lensing analysis"),
    ("manga_v2.py", 600, ["MaNGA", "manga", "C15", "AIC", "v2"],
     "MaNGA v2 IFU rotation curves"),
    ("little_things_step2.py", 600, ["LITTLE", "Hunter", "step2", "RAR", "irregular"],
     "Little Things irregular dwarf RAR"),
    ("phase_c3_step1_verification.py", 600, ["C3", "step1", "verification", "OK", "PASS"],
     "Phase C3 step 1 verification"),
    ("phase_c3_step2_sparc_gamma_vs_alpha.py", 600, ["gamma", "alpha", "SPARC", "OK", "PASS"],
     "Phase C3 step 2 SPARC gamma vs alpha"),
    ("phase_c3_step3_dsph_gamma_vs_alpha.py", 600, ["dSph", "gamma", "alpha", "OK", "PASS"],
     "Phase C3 step 3 dSph gamma vs alpha"),
    ("phase5_3_step1_ngc3198_extract.py", 300, ["NGC", "3198", "v_flat", "h_R", "extract"],
     "Phase 5-3 step 1 NGC 3198 SPARC extract"),
    ("phase5_3_step3_nu_candidate_select.py", 300, ["nu", "candidate", "select", "OK", "PASS"],
     "Phase 5-3 step 3 nu candidate selection"),
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


def run_script(script_relpath, timeout_sec, patterns):
    script_path = os.path.join(BASE, script_relpath)
    if not os.path.isfile(script_path):
        return {"status": "SKIP", "reason": "script_not_found", "script_path": script_relpath}
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
        return {"status": "TIMEOUT", "script_path": script_relpath, "script_sha256": sha,
                "timeout_sec": timeout_sec, "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"status": "ERROR", "script_path": script_relpath, "script_sha256": sha,
                "error": str(e), "elapsed_sec": time.time() - t0}


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = []
    summary = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0, "SKIP": 0}

    for script_relpath, timeout_sec, patterns, anchor in TIER2:
        print(f"[Tier2] Running: {script_relpath} (timeout {timeout_sec}s)")
        sys.stdout.flush()
        r = run_script(script_relpath, timeout_sec, patterns)
        r["paper_anchor_hint"] = anchor
        results.append(r)
        summary[r["status"]] = summary.get(r["status"], 0) + 1
        flag = {"PASS": "OK", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT",
                "ERROR": "ERROR", "SKIP": "SKIP"}[r["status"]]
        elapsed = r.get("elapsed_sec", 0)
        print(f"  -> {flag} (elapsed {elapsed:.1f}s)")

    manifest = {
        "schema_version": "full_recompute_tier2_runner_v1.0",
        "purpose": "Tier-2 auxiliary claim-producing script re-run forensic anchor",
        "execution_timestamp_utc": t0,
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "python_executable": PYTHON,
        "python_version": sys.version,
        "base_dir": BASE,
        "n_scripts_tier2": len(TIER2),
        "summary": summary,
        "results": results,
    }

    out_json = os.path.join(OUT_DIR, "full_recompute_tier2_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(OUT_DIR, "full_recompute_tier2_result.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Full Recompute - Tier 2 Auxiliary Script Re-run Forensic Anchor\n")
        f.write("=" * 70 + "\n")
        f.write(f"Execution UTC : {t0}\n")
        f.write(f"Total Tier 2  : {len(TIER2)}\n")
        f.write(f"Summary       : {summary}\n\n")
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
            if r["status"] in ("FAIL", "TIMEOUT", "ERROR"):
                f.write(f"  stderr_tail: {r.get('stderr_tail_50', '')[:500]}\n")
                f.write(f"  stdout_tail: {r.get('stdout_tail_200', '')[:500]}\n")

    print()
    print("=" * 70)
    print("Tier 2 re-run COMPLETE")
    print("=" * 70)
    print(f"  Summary: {summary}")
    print(f"  Output JSON: {out_json}")
    print(f"  Output TXT : {out_txt}")
    print(f"  JSON SHA256: {sha256_file(out_json)}")
    print(f"  TXT  SHA256: {sha256_file(out_txt)}")


if __name__ == "__main__":
    main()
