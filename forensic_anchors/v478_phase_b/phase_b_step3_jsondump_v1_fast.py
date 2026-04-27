#!/usr/bin/env python3
"""
phase_b_step3_jsondump_v1_fast.py
---------------------------------
Phase 5-2 5.2.M v2-class forensic anchor for v4.7.8 phase B 3-field result.

Reads the persisted RAR + JK covariance + mbin outputs in phase_b_output/
(produced by canonical phase_b_step3_three_fields.py against E:/スバル望遠鏡データ),
re-invokes the canonical chi2_comparison() and gc_slope() functions on those
binned arrays, and dumps a persistent JSON anchor with SHA256 manifest.

This is "fast" mode (seconds) — uses persisted intermediate results to skip
the 503M-pair re-pair calculation. Produces the same numerical claim values
(AIC +472 / 22σ / gc = 2.73 ± 0.11 a₀) as the original stdout-only run.

A separate "full" mode (phase_b_step3_jsondump_v2_full.py) re-runs from
E:/ raw HSC + GAMA_DR4 catalogs for strict end-to-end reproducibility.

Forensic chain:
  - Canonical script: core/phase_b/phase_b_step3_three_fields.py (functions reused)
  - Input ground truth: phase_b_output/{phase_b_combined_rar,phase_b_jk_covariance,
                        phase_b_combined_mbin_*,phase_b_G09_rar,phase_b_G12_rar,
                        phase_b_G15_rar}.txt
  - Output anchor: phase_b_step3_jsondump_v1_fast_result.json
"""

import json
import hashlib
import os
import sys
import datetime
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PHASE_B_OUTPUT = os.path.join(BASE, "phase_b_output")
CANONICAL_SCRIPT = os.path.join(BASE, "core", "phase_b", "phase_b_step3_three_fields.py")

sys.path.insert(0, os.path.join(BASE, "core", "phase_b"))
import phase_b_step3_three_fields as m1  # noqa: E402

a0_si = m1.a0_si
N_GBAR_BINS = m1.N_GBAR_BINS


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_field_rar(field_name):
    """Load per-field RAR (15 bins, columns: g_bar g_obs g_obs_err N_pairs)."""
    fp = os.path.join(PHASE_B_OUTPUT, f"phase_b_{field_name}_rar.txt")
    arr = np.loadtxt(fp, comments="#")
    n_lens = None
    n_pairs_total = None
    with open(fp, "r") as f:
        first = f.readline().strip()
    parts = first.replace("#", "").strip().split()
    for i, p in enumerate(parts):
        if p == "lenses,":
            n_lens = int(parts[i - 1])
        if p == "pairs":
            n_pairs_total = int(parts[i - 1])
    return {
        "field": field_name,
        "gbar": arr[:, 0].copy(),
        "gobs": arr[:, 1].copy(),
        "gobs_err": arr[:, 2].copy(),
        "npair": arr[:, 3].astype(np.int64).copy(),
        "n_lens": n_lens,
        "n_pairs": n_pairs_total,
        "_path": fp,
        "_sha256": sha256_file(fp),
    }


def load_combined_rar():
    fp = os.path.join(PHASE_B_OUTPUT, "phase_b_combined_rar.txt")
    arr = np.loadtxt(fp, comments="#")
    n_lens = None
    n_pairs_total = None
    with open(fp, "r") as f:
        first = f.readline().strip()
    parts = first.replace("#", "").strip().split()
    for i, p in enumerate(parts):
        if p == "lenses,":
            n_lens = int(parts[i - 1])
        if p == "pairs," or p == "pairs":
            n_pairs_total = int(parts[i - 1])
    return {
        "gbar": arr[:, 0].copy(),
        "gobs": arr[:, 1].copy(),
        "gobs_err": arr[:, 2].copy(),
        "npair": arr[:, 3].astype(np.int64).copy(),
        "n_lens": n_lens,
        "n_pairs": n_pairs_total,
        "_path": fp,
        "_sha256": sha256_file(fp),
        "_header": first,
    }


def load_jk_cov():
    """Load canonical g_obs-scale JK covariance (saved by canonical
    phase_b_step3_three_fields.py:818-820 as 'phase_b_combined_jk_cov.txt';
    contains jk_cov * (4*G)^2 already applied for g_obs units)."""
    fp = os.path.join(PHASE_B_OUTPUT, "phase_b_combined_jk_cov.txt")
    cov = np.loadtxt(fp, comments="#")
    return cov, fp, sha256_file(fp)


def load_mbins():
    """Load per mass-bin combined RAR (Bin5 excluded -> 4 bins)."""
    mbins = []
    for k in (1, 2, 3, 4):
        fp = os.path.join(PHASE_B_OUTPUT, f"phase_b_combined_mbin_{k}.txt")
        if not os.path.isfile(fp):
            continue
        arr = np.loadtxt(fp, comments="#")
        with open(fp, "r") as f:
            first = f.readline().strip()
        # header may contain mass range
        mlo = mhi = None
        for tok in first.split():
            if tok.startswith("logM"):
                pass
        mbins.append({
            "k": k,
            "gobs": arr[:, 1].copy() if arr.shape[1] >= 2 else None,
            "gobs_err": arr[:, 2].copy() if arr.shape[1] >= 3 else None,
            "npair": arr[:, 3].astype(np.int64).copy() if arr.shape[1] >= 4 else None,
            "_path": fp,
            "_sha256": sha256_file(fp),
            "_header": first,
            "_arr_shape": list(arr.shape),
        })
    return mbins


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1) load all persisted phase_b_output/ files
    fields = {fn: load_field_rar(fn) for fn in ("G09", "G12", "G15")}
    combined = load_combined_rar()
    jk_cov, jk_path, jk_sha = load_jk_cov()
    mbins = load_mbins()

    gbar = combined["gbar"]
    n = N_GBAR_BINS
    if gbar.shape[0] != n:
        print(f"WARN: gbar bins {gbar.shape[0]} != expected {n}")

    # 2) per-field chi2 + gc fit (diagonal cov, since per-field jk_cov 未保管)
    per_field = []
    for fn, fr in fields.items():
        chi2r = m1.chi2_comparison(gbar, fr["gobs"], fr["gobs_err"], jk_cov=None)
        per_field.append({
            "field": fn,
            "n_lens": fr["n_lens"],
            "n_pairs": fr["n_pairs"],
            "n_pairs_M": (fr["n_pairs"] / 1e6) if fr["n_pairs"] else None,
            "chi2_mond": chi2r["chi2_mond"] if chi2r else None,
            "chi2_c15": chi2r["chi2_c15"] if chi2r else None,
            "gc_si": chi2r["gc"] if chi2r else None,
            "gc_a0": (chi2r["gc"] / a0_si) if (chi2r and np.isfinite(chi2r["gc"])) else None,
            "gc_err_si": chi2r["gc_err"] if chi2r else None,
            "gc_err_a0": (chi2r["gc_err"] / a0_si) if (chi2r and np.isfinite(chi2r["gc_err"])) else None,
            "dchi2": chi2r["dchi2"] if chi2r else None,
            "dAIC": chi2r["dAIC"] if chi2r else None,
            "n_bins_used": chi2r["n"] if chi2r else None,
            "cov_mode": "diagonal",
            "input_sha256": fr["_sha256"],
            "input_path": os.path.relpath(fr["_path"], BASE),
        })

    # 3) field consistency (weighted mean across G09/G12/G15)
    valid = [(p["field"], p["gc_a0"], p["gc_err_a0"]) for p in per_field
             if p["gc_a0"] is not None and p["gc_err_a0"] is not None and p["gc_err_a0"] > 0]
    field_consistency = None
    if len(valid) >= 2:
        vals = np.array([v[1] for v in valid])
        errs = np.array([v[2] for v in valid])
        w = 1.0 / errs**2
        gc_wm = float(np.sum(vals * w) / np.sum(w))
        gc_wm_err = float(1.0 / np.sqrt(np.sum(w)))
        chi2_cons = float(np.sum(((vals - gc_wm) / errs) ** 2))
        ndof_cons = len(vals) - 1
        from scipy.stats import chi2 as chi2_dist
        p_cons = float(chi2_dist.sf(chi2_cons, ndof_cons))
        field_consistency = {
            "weighted_mean_gc_a0": gc_wm,
            "weighted_mean_gc_err_a0": gc_wm_err,
            "chi2": chi2_cons,
            "ndof": ndof_cons,
            "p_value": p_cons,
            "verdict": "CONSISTENT" if p_cons > 0.05 else "INCONSISTENT",
        }

    # 4) combined chi2 + gc with full JK covariance (THIS IS THE +472 ANCHOR)
    chi2r_combined = m1.chi2_comparison(gbar, combined["gobs"], combined["gobs_err"], jk_cov=jk_cov)
    combined_result = {
        "n_lens": combined["n_lens"],
        "n_pairs": combined["n_pairs"],
        "n_pairs_M": (combined["n_pairs"] / 1e6) if combined["n_pairs"] else None,
        "header": combined["_header"],
        "chi2_mond": chi2r_combined["chi2_mond"] if chi2r_combined else None,
        "chi2_c15": chi2r_combined["chi2_c15"] if chi2r_combined else None,
        "gc_si": chi2r_combined["gc"] if chi2r_combined else None,
        "gc_a0": (chi2r_combined["gc"] / a0_si) if (chi2r_combined and np.isfinite(chi2r_combined["gc"])) else None,
        "gc_err_si": chi2r_combined["gc_err"] if chi2r_combined else None,
        "gc_err_a0": (chi2r_combined["gc_err"] / a0_si) if (chi2r_combined and np.isfinite(chi2r_combined["gc_err"])) else None,
        "dchi2": chi2r_combined["dchi2"] if chi2r_combined else None,
        "dAIC": chi2r_combined["dAIC"] if chi2r_combined else None,
        "n_bins_used": chi2r_combined["n"] if chi2r_combined else None,
        "cov_mode": "full_JK_15x15",
        "jk_cov_path": os.path.relpath(jk_path, BASE),
        "jk_cov_sha256": jk_sha,
        "combined_rar_path": os.path.relpath(combined["_path"], BASE),
        "combined_rar_sha256": combined["_sha256"],
    }

    # 5) significance: AIC +472 ↔ Δχ²=474 ↔ √474 σ ≈ 21.77 → ≈ 22σ
    sig_sigma = None
    p_value = None
    if combined_result["dchi2"] is not None and combined_result["dchi2"] > 0:
        sig_sigma = float(np.sqrt(combined_result["dchi2"]))
        from scipy.stats import norm
        p_value = float(2 * norm.sf(abs(sig_sigma)))
    combined_result["c15_vs_mond_sigma"] = sig_sigma
    combined_result["c15_vs_mond_p_value"] = p_value

    # 6) per-mbin gc estimates and slope
    mbin_results = []
    mbin_struct_for_slope = []  # for gc_slope() format
    for mb in mbins:
        if mb["gobs"] is None or mb["gobs_err"] is None or mb["npair"] is None:
            continue
        mask = np.isfinite(mb["gobs"]) & (mb["gobs_err"] > 0) & (mb["npair"] > 30)
        if np.sum(mask) < 3:
            mbin_results.append({"k": mb["k"], "skip_reason": "insufficient bins"})
            continue
        gc_si, gc_err_si = m1.fit_gc(gbar[mask], mb["gobs"][mask], mb["gobs_err"][mask])
        mbin_results.append({
            "k": mb["k"],
            "header": mb["_header"],
            "gc_si": float(gc_si) if np.isfinite(gc_si) else None,
            "gc_a0": float(gc_si / a0_si) if np.isfinite(gc_si) else None,
            "gc_err_si": float(gc_err_si) if np.isfinite(gc_err_si) else None,
            "gc_err_a0": float(gc_err_si / a0_si) if np.isfinite(gc_err_si) else None,
            "n_bins_used": int(np.sum(mask)),
            "input_path": os.path.relpath(mb["_path"], BASE),
            "input_sha256": mb["_sha256"],
        })
        if np.isfinite(gc_si) and gc_si > 0 and np.isfinite(gc_err_si) and gc_err_si > 0:
            # Reconstruct mlo/mhi from header — use canonical MSTAR_EDGES_CUT
            # MSTAR_EDGES_CUT = [8.5, 10.3, 10.6, 10.8, 11.0]
            edges = m1.MSTAR_EDGES_CUT
            if mb["k"] - 1 < len(edges) - 1:
                mlo = edges[mb["k"] - 1]
                mhi = edges[mb["k"]]
                mbin_struct_for_slope.append({
                    "mlo": mlo, "mhi": mhi,
                    "gobs": mb["gobs"], "gobs_err": mb["gobs_err"], "npair": mb["npair"],
                })

    # 7) gc-M* slope
    slope_result = None
    if len(mbin_struct_for_slope) >= 2:
        sr = m1.gc_slope(gbar, mbin_struct_for_slope)
        if sr is not None:
            sl, sle, x, gcv, gce = sr
            from scipy.stats import norm
            t_c15 = (sl - 0.075) / sle
            t_mond = sl / sle
            slope_result = {
                "slope_log10_gc_per_logM": float(sl),
                "slope_err": float(sle),
                "logM_centers": [float(v) for v in x],
                "gc_a0_values": [float(v) for v in gcv],
                "gc_a0_errors": [float(v) for v in gce],
                "vs_C15_predicted_slope_0p075": {
                    "t_statistic": float(t_c15),
                    "p_value_two_sided": float(2 * norm.sf(abs(t_c15))),
                },
                "vs_MOND_predicted_slope_0p000": {
                    "t_statistic": float(t_mond),
                    "p_value_two_sided": float(2 * norm.sf(abs(t_mond))),
                },
            }

    # 8) paper claim cross-validation
    paper_claims = {
        "n_lenses_paper_§1.3": 157338,
        "n_pairs_M_paper_§1.3": 503,
        "gc_a0_paper_§1.3": 2.73,
        "gc_err_a0_paper_§1.3": 0.11,
        "dAIC_paper_§1.3_vs_MOND": 472,
        "sigma_paper_§1.3": 22,
        "p_value_paper_§1.3_MOND_rejection": 1.66e-53,
    }
    cross_check = {
        "n_lenses_match": combined_result["n_lens"] == paper_claims["n_lenses_paper_§1.3"],
        "n_pairs_M_paper_round": (
            combined_result["n_pairs_M"] is not None
            and round(combined_result["n_pairs_M"]) == paper_claims["n_pairs_M_paper_§1.3"]
        ),
    }
    if combined_result["gc_a0"] is not None:
        cross_check["gc_a0_within_paper_err"] = (
            abs(combined_result["gc_a0"] - paper_claims["gc_a0_paper_§1.3"])
            < 5 * paper_claims["gc_err_a0_paper_§1.3"]
        )
        cross_check["gc_a0_diff_from_paper"] = combined_result["gc_a0"] - paper_claims["gc_a0_paper_§1.3"]
    if combined_result["dAIC"] is not None:
        cross_check["dAIC_within_5pct_of_paper"] = (
            abs(combined_result["dAIC"] - paper_claims["dAIC_paper_§1.3_vs_MOND"])
            < 0.05 * paper_claims["dAIC_paper_§1.3_vs_MOND"]
        )
        cross_check["dAIC_diff_from_paper"] = combined_result["dAIC"] - paper_claims["dAIC_paper_§1.3_vs_MOND"]
    if sig_sigma is not None:
        cross_check["sigma_within_1_of_paper"] = abs(sig_sigma - paper_claims["sigma_paper_§1.3"]) < 1.0
        cross_check["sigma_diff_from_paper"] = sig_sigma - paper_claims["sigma_paper_§1.3"]

    # 9) build manifest
    manifest = {
        "schema_version": "phase_b_step3_jsondump_v1_fast_v1.0",
        "purpose": "v4.7.8 phase B 3-field forensic JSON anchor (5.2.M v2-class)",
        "mode": "fast (re-fit on persisted phase_b_output/ binned arrays; no E:/ raw re-pair)",
        "execution_timestamp_utc": t0,
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "canonical_script_path": os.path.relpath(CANONICAL_SCRIPT, BASE),
        "canonical_script_sha256": sha256_file(CANONICAL_SCRIPT),
        "wrapper_script_path": os.path.relpath(__file__, BASE),
        "wrapper_script_sha256": sha256_file(__file__),
        "input_files": {
            "combined_rar": {"path": combined_result["combined_rar_path"], "sha256": combined_result["combined_rar_sha256"]},
            "jk_covariance": {"path": combined_result["jk_cov_path"], "sha256": combined_result["jk_cov_sha256"]},
            "G09_rar": {"path": os.path.relpath(fields["G09"]["_path"], BASE), "sha256": fields["G09"]["_sha256"]},
            "G12_rar": {"path": os.path.relpath(fields["G12"]["_path"], BASE), "sha256": fields["G12"]["_sha256"]},
            "G15_rar": {"path": os.path.relpath(fields["G15"]["_path"], BASE), "sha256": fields["G15"]["_sha256"]},
            "mbins": [
                {"k": mb["k"], "path": os.path.relpath(mb["_path"], BASE), "sha256": mb["_sha256"]}
                for mb in mbins
            ],
        },
        "constants": {
            "G_SI": m1.G_SI,
            "a0_si": m1.a0_si,
            "MSTAR_EDGES_CUT": list(m1.MSTAR_EDGES_CUT),
            "N_GBAR_BINS": m1.N_GBAR_BINS,
            "GBAR_MIN": m1.GBAR_MIN,
            "GBAR_MAX": m1.GBAR_MAX,
        },
        "per_field": per_field,
        "field_consistency": field_consistency,
        "combined_3_field_chi2_C15_vs_MOND": combined_result,
        "per_mbin_gc": mbin_results,
        "gc_M_star_slope": slope_result,
        "paper_claims": paper_claims,
        "paper_claim_cross_check": cross_check,
        "verdicts": {
            "n_lens_match": cross_check.get("n_lenses_match"),
            "n_pairs_match": cross_check.get("n_pairs_M_paper_round"),
            "gc_match_within_5sigma": cross_check.get("gc_a0_within_paper_err"),
            "dAIC_match_within_5pct": cross_check.get("dAIC_within_5pct_of_paper"),
            "sigma_match_within_1": cross_check.get("sigma_within_1_of_paper"),
        },
    }

    # 10) write JSON + summary txt
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return [_sanitize(v) for v in o.tolist()]
        return o

    out_json = os.path.join(BASE, "phase_b_step3_jsondump_v1_fast_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_sanitize(manifest), f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(BASE, "phase_b_step3_jsondump_v1_fast_result.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Phase B Step 3 — Three-Field Forensic JSON Anchor (v1 fast)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Execution UTC : {t0}\n")
        f.write(f"Wrapper SHA256: {manifest['wrapper_script_sha256']}\n")
        f.write(f"Canonical SHA : {manifest['canonical_script_sha256']}\n")
        f.write("\n")
        f.write("=== Combined 3-field (Bin5 excluded) ===\n")
        f.write(f"  N lenses        : {combined_result['n_lens']}\n")
        f.write(f"  N pairs         : {combined_result['n_pairs']:,} = {combined_result['n_pairs_M']:.2f}M\n")
        f.write(f"  chi2_MOND       : {combined_result['chi2_mond']:.2f} (dof {combined_result['n_bins_used']})\n")
        f.write(f"  chi2_C15        : {combined_result['chi2_c15']:.2f} (dof {combined_result['n_bins_used'] - 1})\n")
        f.write(f"  Δχ² (MOND-C15)  : {combined_result['dchi2']:+.2f}\n")
        f.write(f"  ΔAIC (MOND-C15) : {combined_result['dAIC']:+.2f}\n")
        f.write(f"  C15 gc          : {combined_result['gc_a0']:.3f} ± {combined_result['gc_err_a0']:.3f} a₀\n")
        f.write(f"  C15 vs MOND σ   : {sig_sigma:.2f}\n" if sig_sigma else "  C15 vs MOND σ   : N/A\n")
        f.write(f"  Cov mode        : {combined_result['cov_mode']}\n")
        f.write("\n")
        f.write("=== Paper claim cross-check ===\n")
        for k, v in cross_check.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("=== Forensic invariants ===\n")
        f.write(f"  Canonical script (phase_b_step3_three_fields.py): {manifest['canonical_script_sha256']}\n")
        f.write(f"  Combined RAR : {combined_result['combined_rar_sha256']}\n")
        f.write(f"  JK covariance: {combined_result['jk_cov_sha256']}\n")
        for fn in ("G09", "G12", "G15"):
            f.write(f"  {fn} RAR      : {fields[fn]['_sha256']}\n")

    json_sha = sha256_file(out_json)
    txt_sha = sha256_file(out_txt)

    print("=" * 70)
    print("Phase B Step 3 — Forensic JSON anchor (v1 fast) DONE")
    print("=" * 70)
    print(f"Output JSON: {out_json}")
    print(f"  SHA256: {json_sha}")
    print(f"Output TXT : {out_txt}")
    print(f"  SHA256: {txt_sha}")
    print()
    print("=== Combined 3-field chi^2: C15 vs MOND (paper §1.3 (iii) anchor) ===")
    print(f"  N lenses : {combined_result['n_lens']}")
    print(f"  N pairs  : {combined_result['n_pairs']:,} ({combined_result['n_pairs_M']:.2f}M)")
    if combined_result["dAIC"] is not None:
        print(f"  ΔAIC     : {combined_result['dAIC']:+.2f}  (paper claim: +472)")
    if combined_result["gc_a0"] is not None:
        print(f"  gc       : {combined_result['gc_a0']:.3f} ± {combined_result['gc_err_a0']:.3f} a₀  (paper claim: 2.73 ± 0.11)")
    if sig_sigma is not None:
        print(f"  σ vs MOND: {sig_sigma:.2f}  (paper claim: 22)")
    print()
    print("=== Verdicts ===")
    for k, v in manifest["verdicts"].items():
        flag = "✅" if v is True else ("❌" if v is False else "?")
        print(f"  {flag} {k}: {v}")

    return manifest


if __name__ == "__main__":
    main()
