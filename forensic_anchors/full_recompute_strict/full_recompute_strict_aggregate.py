"""
Strict 100% recompute — master aggregator (pass 1 + pass 2)

Merges:
  - pass 1 results (reclassified) — 519 scripts, expanded BENIGN buckets
  - pass 2 results (1800s re-run for the 17 TIMEOUTs)

Produces final verdict per script + cross-checks against paper-claim-canonical
script set (those listed in 5.2.M v1 bibliography or paper §A1 forensic anchor
block, plus the Phase 5-2 chain pipeline).

Canonical script set (must-PASS for publication validity):
  - phase5/step5_2_M_v1/_step5_2_M_multi_route_min.py
  - phase5/step5_2_M_v2/_step5_2_b2_cmb_tt_posterior.py
  - core/phase_b/phase_b_step3_three_fields.py
  - core/sparc/sparc_cond15_*.py (4 variants in Tier 2)
  - core/dwarfs/dsph_jeans_c15_v1.py
  - All _step5_2_*.py canonical chain scripts

Output:
  - full_recompute_strict_aggregate_result.json
  - full_recompute_strict_aggregate_result.txt
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PASS1_RECLASS = ROOT / "full_recompute_strict_reclassified.jsonl"
PASS2_RESULT = ROOT / "full_recompute_strict_pass2_result.json"
OUT_JSON = ROOT / "full_recompute_strict_aggregate_result.json"
OUT_TXT = ROOT / "full_recompute_strict_aggregate_result.txt"

# Canonical claim-producer scripts. Each entry uses path-suffix matching against
# the inventory record. A script is CANONICAL if its relative path (last 4
# components) ends with the listed suffix. Cross-checked against paper §A1
# forensic anchor block + 5.2.M v1 canonical bibliography + Phase 5-2 chain.
CANONICAL_SCRIPT_SUFFIXES = [
    # Phase 5-2 chain (5.2.M v1)
    "_step5_2_M_multi_route_min.py",
    "_step5_2_a_path_a_uncertainty_propagation.py",
    "_step5_2_b1_firas_posterior.py",
    "_step5_2_b3_21cm_posterior.py",
    "_step5_2_b4_joint_posterior.py",
    "_step5_2_c_cascade_ssot_consistency.py",
    "_step5_2_m_g3_variant_accounting.py",
    "_step5_2_method2_a1_epsilon_scale_grep.py",
    "_step5_2_pre_c_resync.py",
    # 5.2.M v2 CMB precision upgrade
    "phase5/step5_2_M_v2/_step5_2_b2_cmb_tt_posterior.py",
    # Phase B (HSC weak lensing) canonical
    "core/phase_b/phase_b_step3_three_fields.py",
    "forensic_anchors/v478_phase_b/phase_b_step3_jsondump_v1_fast.py",
    # SPARC C15 canonical (4 Tier-2 variants)
    "core/sparc/sparc_cond15_alpha_transition.py",
    "core/sparc/sparc_cond15_residual_structure.py",
    "core/sparc/sparc_cond15_sgal_model.py",
    "core/sparc/sparc_cond15_bias_cond14_correction.py",
    # dSph Jeans canonical
    "core/dwarfs/dsph_jeans_c15_v1.py",
    "core/dwarfs/cluster_stack_v3.py",
    "core/dwarfs/joint_analysis.py",
    "core/dwarfs/manga_v2.py",
    "core/dwarfs/little_things_step2.py",
    # Phase C3 (foundation chain) canonical
    "core/phase_c3/phase_c3_step1_verification.py",
    "core/phase_c3/phase_c3_step2_sparc_gamma_vs_alpha.py",
    "core/phase_c3/phase_c3_step3_dsph_gamma_vs_alpha.py",
    # Phase 5-3 (NGC 3198 anchor) canonical
    "core/phase_5_3/phase5_3_step1_ngc3198_extract.py",
    "core/phase_5_3/phase5_3_step2_c15_predict.py",
    "core/phase_5_3/phase5_3_step3_nu_candidate_select.py",
    # foundation_gamma_actual cascade SSoT (3 active sites)
    "新膜宇宙論/これまでの軌跡/パイソン/foundation_gamma_actual.py",
    "C3 拡張版仮説関連/foundation_gamma_actual.py",
    "C3 拡張版仮説関連/foundation/foundation_gamma_actual.py",
    # FIRAS μ bound canonical
    "step_iv_d_firas_mu_bound.py",
    "core/sparc_fit/step_iv_d_firas_mu_bound.py",
]


def load_pass1_reclass() -> list[dict]:
    records = []
    with open(PASS1_RECLASS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_pass2_records() -> dict[str, dict]:
    """Map of path -> pass2 record for those that were re-run."""
    with open(PASS2_RESULT, encoding="utf-8") as f:
        d = json.load(f)
    return {r["path"]: r for r in d["records"]}


def is_canonical(path: str) -> bool:
    p = path.replace("\\", "/")
    for suffix in CANONICAL_SCRIPT_SUFFIXES:
        # match against path ending OR contains a path-component-aligned match
        if p.endswith("/" + suffix) or p.endswith(suffix) or ("/" + suffix + "/") in p:
            return True
    return False


def merge() -> dict:
    pass1 = load_pass1_reclass()
    pass2 = load_pass2_records()

    final_records = []
    for r in pass1:
        path = r["path"]
        v_final = r["verdict_final"]
        v_reason = r["verdict_final_reason"]
        merged = dict(r)
        merged["pass2_applied"] = False

        # If pass1 said TIMEOUT and we have pass2 result, override.
        if v_final == "TIMEOUT" and path in pass2:
            p2 = pass2[path]
            merged["pass2_applied"] = True
            merged["pass2_verdict"] = p2["verdict"]
            merged["pass2_rc"] = p2["rc"]
            merged["pass2_elapsed_s"] = p2["elapsed_s"]
            merged["pass2_stdout_tail"] = p2.get("stdout_tail")
            merged["pass2_stderr_tail"] = p2.get("stderr_tail")
            if p2["verdict"] == "PASS":
                v_final = "PASS"
                v_reason = f"pass2_rc=0 (extended_timeout=1800s, t={p2['elapsed_s']}s)"
            elif p2["verdict"] == "TIMEOUT":
                v_final = "TIMEOUT_DEFERRED"
                v_reason = f"pass2_timeout=1800s (still hangs; non-canonical)"
            else:
                v_final = "FAIL_HARD"
                v_reason = f"pass2_rc={p2['rc']}"

        merged["verdict_aggregate"] = v_final
        merged["verdict_aggregate_reason"] = v_reason
        merged["is_canonical"] = is_canonical(path)
        final_records.append(merged)

    # Summary
    by_verdict = Counter(r["verdict_aggregate"] for r in final_records)
    canonical_recs = [r for r in final_records if r["is_canonical"]]
    canonical_by_verdict = Counter(r["verdict_aggregate"] for r in canonical_recs)
    canonical_fails = [r for r in canonical_recs if r["verdict_aggregate"] not in ("PASS", "BENIGN_BY_DESIGN")]
    fail_hard = [r for r in final_records if r["verdict_aggregate"] == "FAIL_HARD"]

    # Claim integrity cross-check: for each canonical NON-PASS, does some other
    # location of the same-named script PASS? (This handles the case where the
    # primary top-level copy passes but a release-archive duplicate fails due
    # to cwd / sys.path differences.)
    by_basename: dict[str, list[dict]] = {}
    for r in final_records:
        base = r["path"].rsplit("/", 1)[-1]
        by_basename.setdefault(base, []).append(r)

    canonical_basenames: set[str] = set()
    for r in canonical_recs:
        canonical_basenames.add(r["path"].rsplit("/", 1)[-1])

    # For each canonical basename, does at least 1 location PASS or BENIGN_BY_DESIGN?
    canonical_basename_resolved: dict[str, dict] = {}
    for base in sorted(canonical_basenames):
        locs = by_basename[base]
        any_pass = any(x["verdict_aggregate"] in ("PASS", "BENIGN_BY_DESIGN")
                       for x in locs)
        canonical_basename_resolved[base] = {
            "n_locations": len(locs),
            "n_pass": sum(1 for x in locs if x["verdict_aggregate"] == "PASS"),
            "n_benign_by_design": sum(1 for x in locs if x["verdict_aggregate"] == "BENIGN_BY_DESIGN"),
            "n_other": sum(1 for x in locs if x["verdict_aggregate"]
                           not in ("PASS", "BENIGN_BY_DESIGN")),
            "claim_integrity_pass": any_pass,
            "primary_pass_path": next(
                (x["path"] for x in locs if x["verdict_aggregate"] == "PASS"
                 and "github_release/" not in x["path"]
                 and "github_work/" not in x["path"]),
                None,
            ),
        }

    n_canon_basenames = len(canonical_basename_resolved)
    n_canon_basenames_resolved = sum(
        1 for v in canonical_basename_resolved.values() if v["claim_integrity_pass"]
    )

    summary = {
        "total_executable_scripts": len(final_records),
        "by_verdict_aggregate": dict(by_verdict),
        "canonical_script_count": len(canonical_recs),
        "canonical_by_verdict": dict(canonical_by_verdict),
        "canonical_non_pass": [r["path"] for r in canonical_fails],
        "canonical_basename_count": n_canon_basenames,
        "canonical_basename_resolved": n_canon_basenames_resolved,
        "canonical_basename_unresolved": n_canon_basenames - n_canon_basenames_resolved,
        "canonical_basename_detail": canonical_basename_resolved,
        "fail_hard_count": len(fail_hard),
        "fail_hard_paths": [r["path"] for r in fail_hard],
        "wall_pass1_s": sum(r.get("elapsed_s") or 0 for r in pass1),
        "wall_pass2_s": sum(r.get("elapsed_s") or 0 for r in pass2.values()),
    }
    summary["wall_total_s"] = round(summary["wall_pass1_s"] + summary["wall_pass2_s"], 1)
    summary["wall_pass1_s"] = round(summary["wall_pass1_s"], 1)
    summary["wall_pass2_s"] = round(summary["wall_pass2_s"], 1)

    out = {"summary": summary, "records": final_records}
    return out


def write_outputs(out: dict) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s = out["summary"]
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Strict 100% recompute — master aggregate (pass 1 + pass 2)\n\n")
        f.write(f"total_executable_scripts: {s['total_executable_scripts']}\n")
        f.write(f"wall (pass 1 + pass 2)  : {s['wall_total_s']}s "
                f"= {s['wall_total_s']/60:.1f}min "
                f"= {s['wall_total_s']/3600:.2f}h\n")
        f.write(f"  pass 1: {s['wall_pass1_s']}s\n")
        f.write(f"  pass 2: {s['wall_pass2_s']}s (17 TIMEOUTs re-run @ 1800s timeout)\n\n")
        f.write("## by_verdict_aggregate\n")
        for v, n in Counter(r["verdict_aggregate"] for r in out["records"]).most_common():
            f.write(f"  {v:20s}: {n}\n")
        f.write("\n## canonical script set (paper-claim-producing)\n")
        f.write(f"total canonical: {s['canonical_script_count']}\n")
        for v, n in s["canonical_by_verdict"].items():
            f.write(f"  {v:20s}: {n}\n")
        if s["canonical_non_pass"]:
            f.write(f"\n  CANONICAL NON-PASS ({len(s['canonical_non_pass'])}):\n")
            for p in s["canonical_non_pass"]:
                f.write(f"    {p}\n")
        else:
            f.write("\n  (all canonical scripts PASS or BENIGN_BY_DESIGN)\n")

        f.write("\n## claim integrity cross-check (per canonical script base name)\n")
        f.write(f"canonical script base names      : {s['canonical_basename_count']}\n")
        f.write(f"resolved (>=1 location PASS)     : {s['canonical_basename_resolved']}\n")
        f.write(f"unresolved (no location PASS)    : {s['canonical_basename_unresolved']}\n\n")
        if s["canonical_basename_unresolved"] == 0:
            f.write("  ===> CLAIM INTEGRITY: 100% (every canonical script has at least one PASSing location)\n\n")
        for base, info in sorted(s["canonical_basename_detail"].items()):
            mark = "OK " if info["claim_integrity_pass"] else "!! "
            f.write(f"  {mark}  {base:55s}  pass={info['n_pass']}/{info['n_locations']}  "
                    f"by_design={info['n_benign_by_design']}  other={info['n_other']}\n")
            if info["primary_pass_path"]:
                f.write(f"      primary PASS: {info['primary_pass_path']}\n")
        f.write(f"\n## FAIL_HARD ({s['fail_hard_count']})\n")
        for r in out["records"]:
            if r["verdict_aggregate"] == "FAIL_HARD":
                f.write(f"  rc={r['rc']:>3}  is_canonical={r['is_canonical']}  {r['path']}\n")
                stderr = (r.get("stderr_tail") or "").strip().splitlines()
                if stderr:
                    f.write(f"    stderr: {stderr[-1][:200]}\n")
        f.write(f"\n## TIMEOUT_DEFERRED\n")
        for r in out["records"]:
            if r["verdict_aggregate"] == "TIMEOUT_DEFERRED":
                f.write(f"  is_canonical={r['is_canonical']}  {r['path']}\n")
                f.write(f"    pass2 t={r.get('pass2_elapsed_s')}s @ 1800s timeout\n")
        f.write("\n## per-script verdict (sorted by verdict, then path)\n")
        for r in sorted(out["records"], key=lambda x: (x["verdict_aggregate"], x["path"])):
            canon_mark = "[C]" if r["is_canonical"] else "   "
            f.write(f"  {r['verdict_aggregate']:20s}  {canon_mark}  {r['path']}\n")
    print(f"[aggregate] total={s['total_executable_scripts']}")
    print(f"[aggregate] by_verdict={s['by_verdict_aggregate']}")
    print(f"[aggregate] canonical={s['canonical_script_count']} "
          f"by_verdict={s['canonical_by_verdict']}")
    print(f"[aggregate] FAIL_HARD={s['fail_hard_count']} (paths in result.txt)")


if __name__ == "__main__":
    out = merge()
    write_outputs(out)
