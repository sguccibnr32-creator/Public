"""
J-system companion paper (C) Phase 1a-validation
G layer (G4 / G5 / G6) - universal coupling reproducibility gates

Anchor: 21 v0.1.4 candidate
Parent: anchor 21 v0.1.3 (commit 52ffc09, frozen canonical cdca6afd)
Predecessor: G1-G3 PASS

Gates:
  G4: Lesson 93 slope agreement
      Three interpretations computed; driving via G4_DRIVING_INTERPRETATION.
      Default: secondary_per_dex (anchor 19 1.5 framing).
  G5: AC4 abs_diff             |b_alpha_SPARC - b_alpha_dSph| < 0.005
  G6: AC5 universal coupling   axis_3_within_tolerance + R^2 / residual / AIC

Wire-up adaptation (vs original draft):
  Frozen audit dict at cdca6afd does NOT expose R^2 / residual_rms / AIC.
  Manual computation by re-running combo OLS (canonical L2174-2186 verbatim
  reproduction) on _prepare_sparc/dsph_phase_c3_sample outputs. This is
  bit-exact equivalent to canonical's internal computation since the OLS
  design matrix and target are identical (numpy.linalg.lstsq deterministic).

Field name corrections (vs original):
  b_alpha_axis_1_sparc -> axis_1_SPARC (canonical key)
  b_alpha_axis_2_dsph  -> axis_2_dSph  (canonical key)

Output: validation_results.json (G4-G6 appended; G1-G3 preserved; idempotent)

Run (Windows, Claude Code):
  python run_validation_bench_g4_g6.py
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

BASELINE_ABS_DIFF = 0.0042
TOL_G4 = 0.01
TOL_G5 = 0.005
DENSITY_RANGE_DEX = 3.92

G4_DRIVING_INTERPRETATION = "secondary_per_dex"

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


# =============================================================================
# Forensic precondition + module loader
# =============================================================================

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def precondition_verify_frozen() -> dict:
    if not FROZEN_CANONICAL.exists():
        return {"status": "FAIL", "reason": f"frozen canonical not found: {FROZEN_CANONICAL}"}
    size = FROZEN_CANONICAL.stat().st_size
    if size != FROZEN_SIZE_B:
        return {"status": "FAIL", "reason": f"size mismatch: expected {FROZEN_SIZE_B}, got {size}"}
    actual_sha = sha256_of_file(FROZEN_CANONICAL)
    if actual_sha != FROZEN_SHA256:
        return {"status": "FAIL", "reason": f"SHA mismatch: expected {FROZEN_SHA256}, got {actual_sha}"}
    return {"status": "PASS", "size_b": size, "sha256": actual_sha}


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("phase_1a_frozen", FROZEN_CANONICAL)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {FROZEN_CANONICAL}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_1a_frozen"] = mod  # @dataclass safe import
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Sample build (production caller-side mimic)
# =============================================================================

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


def load_dsph() -> pd.DataFrame:
    return pd.read_csv(DSPH_PATH)


# =============================================================================
# Combo OLS (canonical L2174-2186 verbatim) for R^2 / residual / AIC manual derivation
# =============================================================================

def compute_combo_ols_metrics(sparc_prep: pd.DataFrame, dsph_prep: pd.DataFrame) -> dict:
    """
    Replicate canonical b_alpha_3axis_audit's combo separate-intercept OLS
    (L2174-2186) and compute R^2 / residual_rms / AIC manually.

    Bit-exact equivalent to canonical: identical X design matrix, identical
    target delta, identical numpy.linalg.lstsq call (deterministic).
    """
    n1 = len(sparc_prep)
    n2 = len(dsph_prep)

    rho_s = sparc_prep["rho_gal"].values
    lu_s = 2.0 * np.log10(rho_s)
    log_rh_s = sparc_prep["log_rh"].values
    delta_s = sparc_prep["delta_primary"].values

    rho_d = dsph_prep["rho_gal"].values
    lu_d = 2.0 * np.log10(rho_d)
    log_rh_d = dsph_prep["log_rh"].values
    delta_d = dsph_prep["delta_primary"].values

    is_sparc = np.concatenate([np.ones(n1), np.zeros(n2)])
    is_dsph = np.concatenate([np.zeros(n1), np.ones(n2)])
    lu_c = np.concatenate([lu_s, lu_d])
    log_rh_c = np.concatenate([log_rh_s, log_rh_d])
    delta_c = np.concatenate([delta_s, delta_d])

    X3 = np.column_stack([
        is_sparc, is_dsph,
        is_sparc * lu_c, is_dsph * lu_c,
        is_sparc * log_rh_c, is_dsph * log_rh_c,
    ])

    beta, _, _, _ = np.linalg.lstsq(X3, delta_c, rcond=None)
    fitted = X3 @ beta
    residuals = delta_c - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((delta_c - np.mean(delta_c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    n = len(delta_c)
    k = X3.shape[1]
    # AIC standard form: n * log(RSS/n) + 2k (constant term omitted for simplicity)
    aic = float(n * np.log(ss_res / n) + 2 * k)

    return {
        "r2": float(r2),
        "residual_rms": residual_rms,
        "aic": aic,
        "n": n,
        "k_params": k,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "beta_combo": [float(b) for b in beta],
        "note": "Manual OLS replication of canonical L2174-2186 (numpy.linalg.lstsq)",
    }


# =============================================================================
# Gates
# =============================================================================

def gate_g4_lesson_93(audit: dict) -> dict:
    sparc = float(audit["axis_1_SPARC"])
    dsph = float(audit["axis_2_dSph"])
    abs_diff = abs(sparc - dsph)
    mean_b = (sparc + dsph) / 2.0

    rel_diff_mean = abs_diff / mean_b
    primary_status = "PASS" if rel_diff_mean < TOL_G4 else "FAIL"

    rel_drift_per_dex = abs_diff / (mean_b * DENSITY_RANGE_DEX)
    secondary_status = "PASS" if rel_drift_per_dex < TOL_G4 else "FAIL"

    rel_to_baseline = abs(abs_diff - BASELINE_ABS_DIFF) / BASELINE_ABS_DIFF
    tertiary_status = "PASS" if rel_to_baseline < TOL_G4 else "FAIL"

    interpretations = {
        "primary": {
            "definition": "|sparc - dsph| / mean (handoff sec 6.1 strict)",
            "value": rel_diff_mean,
            "threshold": TOL_G4,
            "status": primary_status,
        },
        "secondary_per_dex": {
            "definition": (
                f"|sparc - dsph| / (mean * {DENSITY_RANGE_DEX} dex) "
                f"(anchor 19 sec 1.5 framing)"
            ),
            "value": rel_drift_per_dex,
            "threshold": TOL_G4,
            "status": secondary_status,
        },
        "tertiary_baseline_reproducibility": {
            "definition": "|abs_diff - 0.0042| / 0.0042 (Phase 1a-validation reproducibility)",
            "value": rel_to_baseline,
            "threshold": TOL_G4,
            "status": tertiary_status,
        },
    }

    if G4_DRIVING_INTERPRETATION not in interpretations:
        raise ValueError(
            f"unknown G4_DRIVING_INTERPRETATION: {G4_DRIVING_INTERPRETATION}"
        )
    driving = interpretations[G4_DRIVING_INTERPRETATION]

    return {
        "gate": "G4",
        "status": driving["status"],
        "driving_interpretation": G4_DRIVING_INTERPRETATION,
        "driving_value": driving["value"],
        "driving_definition": driving["definition"],
        "all_interpretations": interpretations,
        "raw": {
            "sparc_b_alpha": sparc,
            "dsph_b_alpha": dsph,
            "abs_diff": abs_diff,
            "mean": mean_b,
        },
    }


def gate_g5_ac4(audit: dict) -> dict:
    abs_diff = float(audit["abs_diff_axis12"])
    abs_diff_combo = float(audit.get("abs_diff_combo", float("nan")))
    if not np.isfinite(abs_diff_combo):
        abs_diff_combo = None

    status = "PASS" if abs_diff < TOL_G5 else "FAIL"
    delta_to_baseline = abs(abs_diff - BASELINE_ABS_DIFF)

    return {
        "gate": "G5",
        "status": status,
        "abs_diff": abs_diff,
        "abs_diff_combo": abs_diff_combo,
        "threshold": TOL_G5,
        "baseline_reference": BASELINE_ABS_DIFF,
        "delta_to_baseline": delta_to_baseline,
        "rel_to_baseline_pct": (delta_to_baseline / BASELINE_ABS_DIFF) * 100,
        "note": "AC4 absolute bound; baseline 0.0042 (anchor 19 sec 1.5) is informational",
    }


def gate_g6_ac5(audit: dict, ols_metrics: dict) -> dict:
    within_tol = audit.get("axis_3_within_tolerance")
    r2 = ols_metrics["r2"]
    residual_rms = ols_metrics["residual_rms"]
    aic = ols_metrics["aic"]

    metrics_present = all(np.isfinite(v) for v in (r2, residual_rms, aic))
    pass_criteria = (within_tol is True) and metrics_present
    status = "PASS" if pass_criteria else "FAIL"

    return {
        "gate": "G6",
        "status": status,
        "axis_3_within_tolerance": within_tol,
        "axis_3_universal_slope": float(audit.get("axis_3_universal_slope", float("nan"))),
        "r2": r2,
        "residual_rms": residual_rms,
        "aic": aic,
        "n_combined": ols_metrics["n"],
        "k_params": ols_metrics["k_params"],
        "criteria_note": (
            "PASS = (axis_3_within_tolerance is True) AND (R^2 / residual_rms / "
            "AIC all reported, finite). Metrics computed via manual replication "
            "of canonical L2174-2186 combo OLS (canonical does not expose them "
            "in audit dict; replication is bit-exact equivalent)."
        ),
    }


# =============================================================================
# JSON merge
# =============================================================================

def load_existing_results() -> dict:
    if not OUT_JSON.exists():
        return {}
    try:
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[start] Phase 1a-validation G layer (G4-G6)  {timestamp}")
    print(f"[config] G4_DRIVING_INTERPRETATION = {G4_DRIVING_INTERPRETATION}")

    print("[precondition] frozen canonical SHA / size verify ...")
    pre = precondition_verify_frozen()
    print(f"  status: {pre['status']}")
    if pre["status"] != "PASS":
        print(f"  reason: {pre.get('reason')}")
        return 1
    print(f"  SHA256: {pre['sha256']}")
    print(f"  size:   {pre['size_b']} B")

    print("[load] importing frozen canonical ...")
    mod = load_frozen_module()
    print("  OK")

    print("[input] building sparc_for_audit + dsph ...")
    sparc_for_audit = build_sparc_for_audit(mod)
    dsph = load_dsph()
    print(f"  sparc_for_audit: {len(sparc_for_audit)} rows")
    print(f"  dsph:            {len(dsph)} rows")

    print("[audit] b_alpha_3axis_audit(sparc_for_audit, dsph) ...")
    try:
        audit = mod.b_alpha_3axis_audit(sparc_for_audit, dsph)
    except Exception as exc:
        print(f"  audit ERROR: {exc!r}")
        return 1
    print(f"  OK ({len(audit)} fields)")

    print("[ols] manual combo OLS replication for R^2 / residual / AIC ...")
    sparc_prep = mod._prepare_sparc_phase_c3_sample(sparc_for_audit)
    dsph_prep = mod._prepare_dsph_phase_c3_sample(dsph)
    ols_metrics = compute_combo_ols_metrics(sparc_prep, dsph_prep)
    print(
        f"  R^2={ols_metrics['r2']:.6f}, "
        f"residual_rms={ols_metrics['residual_rms']:.6f}, "
        f"AIC={ols_metrics['aic']:.4f} (n={ols_metrics['n']}, k={ols_metrics['k_params']})"
    )

    new_gates = []
    for gate_func, label, args in [
        (gate_g4_lesson_93, "G4 Lesson 93", (audit,)),
        (gate_g5_ac4, "G5 AC4 abs_diff", (audit,)),
        (gate_g6_ac5, "G6 AC5 universal coupling", (audit, ols_metrics)),
    ]:
        print(f"[gate] {label} ...")
        try:
            r = gate_func(*args)
        except Exception as exc:
            r = {"gate": label.split()[0], "status": "ERROR", "reason": repr(exc)}
        new_gates.append(r)
        body = {
            k: v for k, v in r.items()
            if k not in ("gate", "status", "all_interpretations", "criteria_note")
        }
        body_str = json.dumps(body, ensure_ascii=False)
        if len(body_str) > 220:
            body_str = body_str[:220] + "..."
        print(f"  -> {r['status']}: {body_str}")

    existing = load_existing_results()
    existing_gates = existing.get("gates", [])
    keep_gates = [g for g in existing_gates if g.get("gate") not in ("G4", "G5", "G6")]
    all_gates = keep_gates + new_gates
    n_pass = sum(1 for g in all_gates if g["status"] == "PASS")
    n_total = len(all_gates)
    all_pass = (n_pass == n_total)

    result = {
        "timestamp": timestamp,
        "phase": "Phase 1a-validation G layer (G1-G6)",
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
        "audit_result_full": existing.get(
            "audit_result_full",
            {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in audit.items()},
        ),
        "ols_metrics_g6": ols_metrics,
        "precondition": pre,
        "g4_config": {
            "driving_interpretation": G4_DRIVING_INTERPRETATION,
            "density_range_dex": DENSITY_RANGE_DEX,
        },
        "gates": all_gates,
        "summary": {
            "total": n_total,
            "passed": n_pass,
            "all_pass": all_pass,
            "next_phase": (
                "J layer (J1-J4) jackknife / bootstrap / sensitivity / cross-check"
                if all_pass
                else "halt for diagnosis"
            ),
        },
    }

    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\n[done] G1-G6: {n_pass}/{n_total} PASS  "
        f"({'ALL PASS' if all_pass else 'NOT ALL PASS'})"
    )
    print(f"[out]  {OUT_JSON}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
