"""
J-system companion paper (C) Phase 1a-validation
J layer (J1 / J2 / J3 / J4) - robustness gates

Anchor: 21 v0.1.4 candidate
Parent: anchor 21 v0.1.3 (commit 52ffc09, frozen canonical cdca6afd)
Predecessor: G1-G6 PASS (validation_results.json existing entries preserved)

Gates:
  J1: jackknife 124-LOO (axis_1_SPARC)
      sigma_LOO < 0.005, no 3-sigma outlier
      Outliers are flagged with Q / delta_primary / log_rh metadata
      (jackknife identifies influential samples; does NOT exclude from re-fit)
  J2: bootstrap CI (10,000 resample, deterministic seed)
      95% CI contains baseline 0.1084
  J3: per-filter sensitivity (5 filters)  [SKELETON / PENDING]
      Q<3 / v_flat>0 / g_obs valid / bridge / delta_primary finite
      Filter-disable variants require canonical _prepare_sparc_phase_c3_sample
      source replication. Default: PENDING (J1/J2/J4 deliver first).
  J4: phase_c3_step3 cross-check (file-based)
      Reads phase_c3_step3_reference.json (independent code path with Q intact
      loader). Bit-exact match to G1's axis_1_SPARC = 0.108442979149252.
      Default: PENDING if file absent.

OLS structure for axis_1_SPARC (replicated for jackknife / bootstrap):
  target:    delta_primary
  features:  [intercept, lu = 2*log10(rho_gal), log_rh]
  b_alpha = beta[1]  (lu coefficient)

Bit-exact verification step runs before J1/J2 to confirm OLS structure matches
canonical (compares replicated b_alpha against audit axis_1_SPARC value).
If mismatch: halt -- OLS structure differs from canonical, results invalid.

Output (forensic_anchors/section2_5_v0_4a_validation/):
  - validation_results.json       (J entries appended)
  - jackknife_axis_1_results.csv  (J1: 124 LOO entries + metadata)
  - bootstrap_axis_1_results.csv  (J2: 10,000 resample b_alpha values)
  - per_filter_sensitivity.csv    (J3: deferred until canonical source)
  - cross_check_phase_c3_step3.json (J4: comparison summary)

Run (Windows, Claude Code):
  python run_validation_bench_j_layer.py
"""

from __future__ import annotations

import sys
import json
import hashlib
import importlib.util
import platform as _platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================================================================
# IMMUTABLE constants
# =============================================================================

FROZEN_DIR = (
    Path(__file__).resolve().parent.parent
    / "section2_5_v0_4a_axis_1_alignment"
)
FROZEN_CANONICAL = FROZEN_DIR / "run_section2_5_v0_2.py"
FROZEN_SHA256 = (
    "cdca6afd634a3f730bc4b4002ab3082e187dc64b4ad9e87fd8147a6d81b04521"
)
FROZEN_SIZE_B = 137_861
FROZEN_PATH_REPO_REL = (
    "forensic_anchors/section2_5_v0_4a_axis_1_alignment/run_section2_5_v0_2.py"
)

BASELINE_AXIS_1 = 0.1084

J1_SIGMA_THRESHOLD = 0.005
J1_OUTLIER_Z = 3.0

J2_N_RESAMPLE = 10_000
J2_RNG_SEED = 42
J2_CI_LOW_PCT = 2.5
J2_CI_HIGH_PCT = 97.5

PHASE_C3_STEP3_REF_FILE = (
    Path(__file__).resolve().parent / "phase_c3_step3_reference.json"
)
J4_BIT_EXACT_TOL = 1e-12

DATA_ROOT = Path(
    r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
)
TA3_PATH = DATA_ROOT / "TA3_gc_independent.csv"
PHASE1_PATH = DATA_ROOT / "phase1" / "sparc_results.csv"
MRT_PATH = DATA_ROOT / "SPARC_Lelli2016c.mrt"
DSPH_PATH = DATA_ROOT / "dsph_jeans_c15_v1.csv"
RC_DIR = DATA_ROOT / "Rotmod_LTG"

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "validation_results.json"
OUT_CSV_J1 = OUT_DIR / "jackknife_axis_1_results.csv"
OUT_CSV_J2 = OUT_DIR / "bootstrap_axis_1_results.csv"
OUT_CSV_J3 = OUT_DIR / "per_filter_sensitivity.csv"
OUT_JSON_J4 = OUT_DIR / "cross_check_phase_c3_step3.json"


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def precondition_verify_frozen() -> dict:
    if not FROZEN_CANONICAL.exists():
        return {"status": "FAIL", "reason": "frozen canonical not found"}
    size = FROZEN_CANONICAL.stat().st_size
    if size != FROZEN_SIZE_B:
        return {"status": "FAIL", "reason": f"size mismatch: {size} vs {FROZEN_SIZE_B}"}
    actual_sha = sha256_of_file(FROZEN_CANONICAL)
    if actual_sha != FROZEN_SHA256:
        return {"status": "FAIL", "reason": "SHA mismatch"}
    return {"status": "PASS", "size_b": size, "sha256": actual_sha}


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("phase_1a_frozen", FROZEN_CANONICAL)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_1a_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_sparc_for_audit(mod) -> pd.DataFrame:
    ta3 = mod.load_TA3(str(TA3_PATH))
    phase1 = mod.load_phase1(str(PHASE1_PATH))
    mrt = mod.load_MRT(str(MRT_PATH))
    sparc_full = mod.merge_three_files(ta3, phase1, mrt)
    sparc_full = mod.mark_fit_pool_171(sparc_full)
    sparc_171 = sparc_full[sparc_full["in_171_pool"]].copy()

    rotation_curves_by_galaxy = {}
    for galaxy in sparc_171["galaxy"]:
        p = RC_DIR / f"{galaxy}_rotmod.dat"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(
                p, sep=r"\s+", comment="#", header=None,
                names=["Rad", "Vobs", "errV", "Vgas", "Vdisk",
                       "Vbul", "SBdisk", "SBbul"],
                engine="python",
            )
            rotation_curves_by_galaxy[galaxy] = {
                "r": df["Rad"].values,
                "v_obs": df["Vobs"].values,
                "errV": df["errV"].values,
            }
        except Exception:
            pass

    sparc_for_audit = sparc_171.copy()
    hR_by_galaxy = dict(zip(sparc_for_audit["galaxy"], sparc_for_audit["Rdisk"]))
    g_obs_per_galaxy = {}
    for galaxy, rc in rotation_curves_by_galaxy.items():
        r_kpc = np.asarray(rc.get("r", []), dtype=float)
        v_kms = np.asarray(rc.get("v_obs", []), dtype=float)
        hR_g = float(hR_by_galaxy.get(galaxy, np.nan))
        if not np.isfinite(hR_g) or hR_g <= 0:
            continue
        mask = (r_kpc > 2.0 * hR_g) & (v_kms > 0) & np.isfinite(r_kpc) & np.isfinite(v_kms)
        if int(mask.sum()) < 2:
            continue
        g_obs_per_galaxy[galaxy] = float(np.mean(v_kms[mask] ** 2 / r_kpc[mask]))
    sparc_for_audit["g_obs"] = sparc_for_audit["galaxy"].map(
        lambda g: g_obs_per_galaxy.get(g, float("nan"))
    )

    col_alias_map = {"galaxy": "Galaxy", "Ud": "Upsilon_d", "ud": "Upsilon_d"}
    sparc_for_audit = sparc_for_audit.rename(columns=col_alias_map)
    return sparc_for_audit


def _design_matrix_axis_1(prep: pd.DataFrame):
    rho = prep["rho_gal"].values
    lu = 2.0 * np.log10(rho)
    log_rh = prep["log_rh"].values
    delta = prep["delta_primary"].values
    n = len(prep)
    X = np.column_stack([np.ones(n), lu, log_rh])
    return X, delta


def fit_b_alpha_axis_1(prep: pd.DataFrame) -> float:
    X, y = _design_matrix_axis_1(prep)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def precondition_verify_ols_replication(prep: pd.DataFrame, audit_axis_1: float) -> dict:
    b_alpha_repl = fit_b_alpha_axis_1(prep)
    delta = abs(b_alpha_repl - audit_axis_1)
    bit_exact = delta < 1e-10
    return {
        "status": "PASS" if bit_exact else "FAIL",
        "b_alpha_replicated": b_alpha_repl,
        "b_alpha_canonical": float(audit_axis_1),
        "delta_abs": float(delta),
        "tolerance": 1e-10,
    }


# =============================================================================
# J1: jackknife
# =============================================================================

def run_jackknife(prep: pd.DataFrame) -> list:
    n = len(prep)
    galaxy_col = "Galaxy" if "Galaxy" in prep.columns else "galaxy"
    galaxies = prep[galaxy_col].values
    rho = prep["rho_gal"].values
    log_rh = prep["log_rh"].values
    delta = prep["delta_primary"].values
    lu = 2.0 * np.log10(rho)

    Q_col = prep["Q"].values if "Q" in prep.columns else [None] * n
    gc_C15_col = prep["gc_C15"].values if "gc_C15" in prep.columns else [None] * n

    indices = np.arange(n)
    results = []
    for i in range(n):
        mask = indices != i
        X = np.column_stack([np.ones(n - 1), lu[mask], log_rh[mask]])
        beta, _, _, _ = np.linalg.lstsq(X, delta[mask], rcond=None)
        b_alpha_loo = float(beta[1])

        Q_val = Q_col[i]
        gc_val = gc_C15_col[i]
        results.append({
            "dropped_galaxy": galaxies[i],
            "b_alpha_loo": b_alpha_loo,
            "Q": (int(Q_val) if Q_val is not None and pd.notna(Q_val) else None),
            "gc_C15": (float(gc_val) if gc_val is not None and pd.notna(gc_val) else None),
            "delta_primary": float(delta[i]),
            "log_rh": float(log_rh[i]),
            "lu": float(lu[i]),
            "rho_gal": float(rho[i]),
        })
    return results


def gate_j1_jackknife(loo_results: list) -> dict:
    b_alpha_loo = np.array([r["b_alpha_loo"] for r in loo_results])
    n = len(b_alpha_loo)
    mean_loo = float(np.mean(b_alpha_loo))
    std_loo = float(np.std(b_alpha_loo, ddof=1))
    sigma_jk = float(np.sqrt((n - 1) / n * np.sum((b_alpha_loo - mean_loo) ** 2)))
    z_scores = (np.abs(b_alpha_loo - mean_loo) / sigma_jk
                if sigma_jk > 0 else np.zeros(n))
    outlier_idx = np.where(z_scores > J1_OUTLIER_Z)[0]

    for i, r in enumerate(loo_results):
        r["z_score"] = float(z_scores[i])
        r["is_3sigma_outlier"] = bool(z_scores[i] > J1_OUTLIER_Z)

    pass_sigma = sigma_jk < J1_SIGMA_THRESHOLD
    pass_outliers = len(outlier_idx) == 0
    status = "PASS" if (pass_sigma and pass_outliers) else "FAIL"

    return {
        "gate": "J1",
        "status": status,
        "n": n,
        "mean_loo": mean_loo,
        "std_loo": std_loo,
        "sigma_jackknife": sigma_jk,
        "sigma_threshold": J1_SIGMA_THRESHOLD,
        "n_outliers_3sigma": int(len(outlier_idx)),
        "outlier_galaxies": [
            {
                "dropped_galaxy": loo_results[i]["dropped_galaxy"],
                "b_alpha_loo": loo_results[i]["b_alpha_loo"],
                "z_score": float(z_scores[i]),
                "Q": loo_results[i].get("Q"),
                "gc_C15": loo_results[i].get("gc_C15"),
                "delta_primary": loo_results[i].get("delta_primary"),
            }
            for i in outlier_idx
        ],
        "criteria": {
            "sigma_LOO_lt_0_005": pass_sigma,
            "no_3_sigma_outlier": pass_outliers,
        },
        "note": (
            "Outliers are flagged via z_score > 3 but NOT excluded from any "
            "re-fit. Jackknife identifies influential samples; baseline "
            "0.1084 derived from full 124 sample remains the IMMUTABLE "
            "reference."
        ),
    }


def write_jackknife_csv(loo_results: list) -> None:
    df = pd.DataFrame(loo_results)
    column_order = [
        "dropped_galaxy", "b_alpha_loo", "z_score", "is_3sigma_outlier",
        "Q", "gc_C15", "delta_primary", "log_rh", "lu", "rho_gal",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    df.to_csv(OUT_CSV_J1, index=False, encoding="utf-8")


# =============================================================================
# J2: bootstrap
# =============================================================================

def run_bootstrap(prep: pd.DataFrame, n_resample: int, seed: int) -> np.ndarray:
    rho = prep["rho_gal"].values
    log_rh = prep["log_rh"].values
    delta = prep["delta_primary"].values
    lu = 2.0 * np.log10(rho)
    n = len(prep)

    rng = np.random.default_rng(seed)
    b_alphas = np.empty(n_resample, dtype=float)
    for k in range(n_resample):
        idx = rng.integers(0, n, size=n)
        X = np.column_stack([np.ones(n), lu[idx], log_rh[idx]])
        beta, _, _, _ = np.linalg.lstsq(X, delta[idx], rcond=None)
        b_alphas[k] = beta[1]
    return b_alphas


def gate_j2_bootstrap(b_alphas: np.ndarray) -> dict:
    n = int(len(b_alphas))
    ci_low = float(np.percentile(b_alphas, J2_CI_LOW_PCT))
    ci_high = float(np.percentile(b_alphas, J2_CI_HIGH_PCT))
    contains = bool(ci_low <= BASELINE_AXIS_1 <= ci_high)

    return {
        "gate": "J2",
        "status": "PASS" if contains else "FAIL",
        "n_resample": n,
        "rng_seed": J2_RNG_SEED,
        "mean": float(np.mean(b_alphas)),
        "std": float(np.std(b_alphas, ddof=1)),
        "median": float(np.median(b_alphas)),
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "ci_width": ci_high - ci_low,
        "baseline": BASELINE_AXIS_1,
        "contains_baseline": contains,
    }


def write_bootstrap_csv(b_alphas: np.ndarray) -> None:
    df = pd.DataFrame({
        "resample_idx": np.arange(len(b_alphas)),
        "b_alpha": b_alphas,
    })
    df.to_csv(OUT_CSV_J2, index=False, encoding="utf-8")


# =============================================================================
# J3: per-filter sensitivity (PENDING)
# =============================================================================

def gate_j3_per_filter_sensitivity_pending() -> dict:
    return {
        "gate": "J3",
        "status": "PENDING",
        "reason": (
            "5 filter-disable variants (Q<3 / v_flat>0 / g_obs valid / bridge "
            "/ delta_primary finite) operate at different pipeline stages "
            "(prepare-internal, sparc_171 construction, build_sparc_for_audit "
            "masking). Replication requires _prepare_sparc_phase_c3_sample "
            "source for clean toggle implementation."
        ),
        "next_action": (
            "Share canonical _prepare_sparc_phase_c3_sample (and any upstream "
            "sparc_171 / g_obs masking logic) source, or provide 5 filter-"
            "disable variant DataFrames for parallel b_alpha fit."
        ),
        "expected_output": "per_filter_sensitivity.csv (5 rows)",
    }


# =============================================================================
# J4: phase_c3_step3 cross-check
# =============================================================================

def gate_j4_phase_c3_step3_cross_check(audit_axis_1: float) -> dict:
    if not PHASE_C3_STEP3_REF_FILE.exists():
        return {
            "gate": "J4",
            "status": "PENDING",
            "reason": (
                f"reference file not found: {PHASE_C3_STEP3_REF_FILE.name}"
            ),
            "next_action": (
                f"Run phase_c3_step3 reference impl independently and write "
                f"its axis_1_SPARC b_alpha to {PHASE_C3_STEP3_REF_FILE.name} "
                f'in JSON: {{"axis_1_SPARC_b_alpha": <float>, '
                f'"source": "<path or commit ref>", "computed_at": "<iso>"}}'
            ),
            "expected_path": str(PHASE_C3_STEP3_REF_FILE),
        }

    try:
        ref = json.loads(PHASE_C3_STEP3_REF_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "gate": "J4",
            "status": "ERROR",
            "reason": f"reference file JSON parse error: {exc}",
        }

    if "axis_1_SPARC_b_alpha" not in ref:
        return {
            "gate": "J4",
            "status": "ERROR",
            "reason": (
                f"reference file missing key `axis_1_SPARC_b_alpha`. "
                f"Available keys: {list(ref.keys())}"
            ),
        }

    b_alpha_ref = float(ref["axis_1_SPARC_b_alpha"])
    delta = abs(b_alpha_ref - audit_axis_1)
    bit_exact = delta < J4_BIT_EXACT_TOL

    cross_check_payload = {
        "b_alpha_phase_c3_step3": b_alpha_ref,
        "b_alpha_audit_axis_1": float(audit_axis_1),
        "delta_abs": float(delta),
        "bit_exact_threshold": J4_BIT_EXACT_TOL,
        "bit_exact": bit_exact,
        "reference_metadata": {
            k: v for k, v in ref.items() if k != "axis_1_SPARC_b_alpha"
        },
    }
    OUT_JSON_J4.write_text(
        json.dumps(cross_check_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "gate": "J4",
        "status": "PASS" if bit_exact else "FAIL",
        "b_alpha_phase_c3_step3": b_alpha_ref,
        "b_alpha_audit_axis_1": float(audit_axis_1),
        "delta_abs": float(delta),
        "bit_exact_threshold": J4_BIT_EXACT_TOL,
        "bit_exact": bool(bit_exact),
        "output_file": OUT_JSON_J4.name,
    }


def load_existing_results() -> dict:
    if not OUT_JSON.exists():
        return {}
    try:
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[start] Phase 1a-validation J layer  {timestamp}")

    print("[precondition] frozen canonical SHA / size verify ...")
    pre = precondition_verify_frozen()
    print(f"  status: {pre['status']}")
    if pre["status"] != "PASS":
        print(f"  reason: {pre.get('reason')}")
        return 1
    print(f"  SHA256: {pre['sha256']}")

    print("[load] importing frozen canonical ...")
    mod = load_frozen_module()
    print("  OK")

    print("[input] building sparc_for_audit + dsph + sparc_prep ...")
    sparc_for_audit = build_sparc_for_audit(mod)
    dsph = pd.read_csv(DSPH_PATH)
    sparc_prep = mod._prepare_sparc_phase_c3_sample(sparc_for_audit)
    print(f"  sparc_for_audit: {len(sparc_for_audit)} rows")
    print(f"  dsph:            {len(dsph)} rows")
    print(f"  sparc_prep:      {len(sparc_prep)} rows  (expected 124)")
    print(f"  prep columns:    {list(sparc_prep.columns)}")

    print("[audit] axis_1_SPARC reference value ...")
    audit = mod.b_alpha_3axis_audit(sparc_for_audit, dsph)
    audit_axis_1 = float(audit["axis_1_SPARC"])
    print(f"  axis_1_SPARC = {audit_axis_1}")

    print("[precondition] OLS replication bit-exact verify ...")
    ols_pre = precondition_verify_ols_replication(sparc_prep, audit_axis_1)
    print(f"  status: {ols_pre['status']}")
    print(f"  replicated: {ols_pre['b_alpha_replicated']}")
    print(f"  canonical:  {ols_pre['b_alpha_canonical']}")
    print(f"  delta:      {ols_pre['delta_abs']:.3e}  (tol {ols_pre['tolerance']:.0e})")
    if ols_pre["status"] != "PASS":
        print("  HALT: OLS replication mismatch -- J1/J2 results would be invalid")
        return 1

    print("[J1] running 124-LOO jackknife ...")
    loo_results = run_jackknife(sparc_prep)
    j1 = gate_j1_jackknife(loo_results)
    write_jackknife_csv(loo_results)
    print(f"  -> {j1['status']}: sigma_jk = {j1['sigma_jackknife']:.6f}, "
          f"n_outliers = {j1['n_outliers_3sigma']}")
    print(f"  csv: {OUT_CSV_J1.name}")

    print(f"[J2] running bootstrap ({J2_N_RESAMPLE} resamples, seed={J2_RNG_SEED}) ...")
    b_alphas = run_bootstrap(sparc_prep, n_resample=J2_N_RESAMPLE, seed=J2_RNG_SEED)
    j2 = gate_j2_bootstrap(b_alphas)
    write_bootstrap_csv(b_alphas)
    print(f"  -> {j2['status']}: 95% CI = [{j2['ci_2_5']:.6f}, {j2['ci_97_5']:.6f}], "
          f"contains baseline = {j2['contains_baseline']}")
    print(f"  csv: {OUT_CSV_J2.name}")

    print("[J3] per-filter sensitivity ...")
    j3 = gate_j3_per_filter_sensitivity_pending()
    print(f"  -> {j3['status']}: {j3['reason'][:80]}...")

    print("[J4] phase_c3_step3 cross-check ...")
    j4 = gate_j4_phase_c3_step3_cross_check(audit_axis_1)
    print(f"  -> {j4['status']}: ", end="")
    if j4["status"] == "PASS":
        print(f"bit-exact match (delta = {j4['delta_abs']:.3e})")
    elif j4["status"] == "FAIL":
        print(f"delta = {j4['delta_abs']:.3e} > {j4['bit_exact_threshold']:.0e}")
    else:
        print(j4.get("reason", "")[:80])

    new_gates = [j1, j2, j3, j4]
    existing = load_existing_results()
    existing_gates = existing.get("gates", [])
    keep_gates = [g for g in existing_gates if g.get("gate") not in ("J1", "J2", "J3", "J4")]
    all_gates = keep_gates + new_gates

    n_pass = sum(1 for g in all_gates if g["status"] == "PASS")
    n_fail = sum(1 for g in all_gates if g["status"] == "FAIL")
    n_pending = sum(1 for g in all_gates if g["status"] == "PENDING")
    n_error = sum(1 for g in all_gates if g["status"] == "ERROR")
    n_total = len(all_gates)
    all_pass = (n_pass == n_total)

    result = {
        "timestamp": timestamp,
        "phase": "Phase 1a-validation full (G1-G6 + J1-J4)",
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": _platform.platform(),
        },
        "frozen_canonical": existing.get(
            "frozen_canonical",
            {
                "path_repo_relative": FROZEN_PATH_REPO_REL,
                "sha256": FROZEN_SHA256,
                "size_b": FROZEN_SIZE_B,
            },
        ),
        "baseline_immutable": existing.get("baseline_immutable", {}),
        "audit_result_full": existing.get("audit_result_full", {}),
        "ols_metrics_g6": existing.get("ols_metrics_g6", {}),
        "ols_replication_check": ols_pre,
        "j2_config": {
            "n_resample": J2_N_RESAMPLE,
            "rng_seed": J2_RNG_SEED,
            "ci_pcts": [J2_CI_LOW_PCT, J2_CI_HIGH_PCT],
        },
        "precondition": pre,
        "g4_config": existing.get("g4_config", {}),
        "gates": all_gates,
        "summary": {
            "total": n_total,
            "passed": n_pass,
            "failed": n_fail,
            "pending": n_pending,
            "error": n_error,
            "all_pass": all_pass,
            "next_phase": (
                "anchor 21 v0.1.4 forensic commit + v0.2 promotion path"
                if (all_pass and n_pending == 0)
                else f"address {n_fail} FAIL + {n_pending} PENDING + {n_error} ERROR"
            ),
        },
    }

    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\n[done] G1-J4 ({n_total} gates): "
        f"{n_pass} PASS / {n_fail} FAIL / {n_pending} PENDING / {n_error} ERROR"
    )
    print(f"[out]  {OUT_JSON}")

    return 0 if (n_fail == 0 and n_error == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
