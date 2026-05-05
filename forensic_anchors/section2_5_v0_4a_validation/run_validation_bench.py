"""
J-system companion paper (C) Phase 1a-validation
G layer (G1 / G2 / G3) - pure reproducibility gates

Anchor: 21 v0.1.4 candidate
Parent: anchor 21 v0.1.3 (commit 52ffc09, frozen canonical cdca6afd)

Gates:
  G1: axis_1_SPARC b_alpha   |b - 0.1084| < 1e-3
  G2: axis_2_dSph  b_alpha   |b - 0.1127| < 1e-3
  G3: axis_3 universal slope in [0.105, 0.115]   (target band 0.110-0.111)

Forensic precondition:
  frozen canonical SHA256 must equal cdca6afd... (137,861 B exact)

Layer ordering (agreed with handoff):
  G1-G3 (this file) -> G4-G6 (Lesson 93 / AC4 / AC5) -> J1-J4 (robustness)

Output: validation_results.json (G1-G3 section)

Wire-up adaptation note:
  Original draft assumed three separate per-axis compute functions
  (compute_b_alpha_axis_1_sparc, ..._axis_2_dsph, ..._axis_3_universal),
  which the frozen canonical does NOT expose. Actual API at cdca6afd:
      b_alpha_3axis_audit(sparc_df, dsph_df) -> dict
  returns all three axes in a single unified result. This script is
  refactored to call the unified API once and extract per-axis values
  from the result dict, which is structurally cleaner and avoids
  duplicate sample preparation across gates. Sample-build logic
  mimics production caller side at run_section2_5_v0_2.py L3043-3074
  (sparc_171 + g_obs aggregation + col_alias_map).

Run (Windows, Claude Code):
  python run_validation_bench.py
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import importlib.util
import platform as _platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Mandatory per environment convention (Windows console UTF-8)
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

BASELINE_AXIS_1_SPARC = 0.1084
BASELINE_AXIS_2_DSPH = 0.1127
TOL_G1 = 1e-3
TOL_G2 = 1e-3
RANGE_G3_LO = 0.105
RANGE_G3_HI = 0.115
TARGET_G3_BAND = "0.110-0.111"

EXPECTED_N_AXIS_1 = 124
EXPECTED_N_AXIS_2 = 30
EXPECTED_N_AXIS_3 = 154

# Data paths (mirrors production env var convention)
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
# Forensic precondition
# =============================================================================

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def precondition_verify_frozen() -> dict:
    if not FROZEN_CANONICAL.exists():
        return {
            "status": "FAIL",
            "reason": f"frozen canonical not found: {FROZEN_CANONICAL}",
        }
    size = FROZEN_CANONICAL.stat().st_size
    if size != FROZEN_SIZE_B:
        return {
            "status": "FAIL",
            "reason": f"size mismatch: expected {FROZEN_SIZE_B}, got {size}",
        }
    actual_sha = sha256_of_file(FROZEN_CANONICAL)
    if actual_sha != FROZEN_SHA256:
        return {
            "status": "FAIL",
            "reason": f"SHA mismatch: expected {FROZEN_SHA256}, got {actual_sha}",
        }
    return {
        "status": "PASS",
        "size_b": size,
        "sha256": actual_sha,
    }


# =============================================================================
# Frozen module loader (importlib isolated, __main__ guard at L3296)
# =============================================================================

def load_frozen_module():
    spec = importlib.util.spec_from_file_location(
        "phase_1a_frozen", FROZEN_CANONICAL
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {FROZEN_CANONICAL}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so that @dataclass and other
    # decorators that introspect via sys.modules.get(cls.__module__) work.
    sys.modules["phase_1a_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Sample preparation (production caller-side mimic, L3043-3074)
# =============================================================================

def build_sparc_for_audit(mod) -> pd.DataFrame:
    """Mirror run_section2_5_v0_2.py main flow up through col_alias_map."""
    ta3 = mod.load_TA3(str(TA3_PATH))
    phase1 = mod.load_phase1(str(PHASE1_PATH))
    mrt = mod.load_MRT(str(MRT_PATH))

    sparc_full = mod.merge_three_files(ta3, phase1, mrt)
    sparc_full = mod.mark_fit_pool_171(sparc_full)
    sparc_171 = sparc_full[sparc_full["in_171_pool"]].copy()

    # Build rotation_curves_by_galaxy
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

    # g_obs aggregation (T16.F: mean(V^2/r) for r > 2*hR, km^2/s^2/kpc)
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

    # col_alias_map
    col_alias_map = {"galaxy": "Galaxy", "Ud": "Upsilon_d", "ud": "Upsilon_d"}
    sparc_for_audit = sparc_for_audit.rename(columns=col_alias_map)
    return sparc_for_audit


def load_dsph() -> pd.DataFrame:
    return pd.read_csv(DSPH_PATH)


# =============================================================================
# Gates (extract from unified b_alpha_3axis_audit dict)
# =============================================================================

def gate_g1_axis_1_sparc(audit_result) -> dict:
    sample_n = int(audit_result.get("sample_n_axis_1", -1))
    if sample_n != EXPECTED_N_AXIS_1:
        return {
            "gate": "G1",
            "status": "FAIL",
            "reason": f"sample_n != {EXPECTED_N_AXIS_1} (got {sample_n})",
            "sample_n": sample_n,
        }
    b_alpha = float(audit_result["axis_1_SPARC"])
    delta = abs(b_alpha - BASELINE_AXIS_1_SPARC)
    status = "PASS" if delta < TOL_G1 else "FAIL"
    return {
        "gate": "G1",
        "status": status,
        "sample_n": sample_n,
        "b_alpha": b_alpha,
        "baseline": BASELINE_AXIS_1_SPARC,
        "delta_abs": delta,
        "tolerance": TOL_G1,
    }


def gate_g2_axis_2_dsph(audit_result) -> dict:
    sample_n = int(audit_result.get("sample_n_axis_2", -1))
    b_alpha = float(audit_result["axis_2_dSph"])
    delta = abs(b_alpha - BASELINE_AXIS_2_DSPH)
    status = "PASS" if delta < TOL_G2 else "FAIL"
    return {
        "gate": "G2",
        "status": status,
        "sample_n": sample_n,
        "b_alpha": b_alpha,
        "baseline": BASELINE_AXIS_2_DSPH,
        "delta_abs": delta,
        "tolerance": TOL_G2,
        "note": "re-confirmation under frozen Phase 1a; expected unchanged",
    }


def gate_g3_axis_3_universal(audit_result) -> dict:
    sample_n = int(audit_result.get("sample_n_axis_3", -1))
    slope = float(audit_result["axis_3_universal_slope"])
    status = "PASS" if (RANGE_G3_LO <= slope <= RANGE_G3_HI) else "FAIL"
    return {
        "gate": "G3",
        "status": status,
        "sample_n": sample_n,
        "slope": slope,
        "range_low": RANGE_G3_LO,
        "range_high": RANGE_G3_HI,
        "target_band": TARGET_G3_BAND,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[start] Phase 1a-validation G layer (G1-G3)  {timestamp}")

    # Precondition
    print("[precondition] frozen canonical SHA / size verify ...")
    pre = precondition_verify_frozen()
    print(f"  status: {pre['status']}")
    if pre["status"] != "PASS":
        print(f"  reason: {pre.get('reason', '?')}")
        result = {
            "timestamp": timestamp,
            "phase": "Phase 1a-validation G layer (G1-G3)",
            "frozen_canonical": {
                "path_repo_relative": FROZEN_PATH_REPO_REL,
                "sha256_expected": FROZEN_SHA256,
                "size_b_expected": FROZEN_SIZE_B,
            },
            "precondition": pre,
            "gates": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "all_pass": False,
                "halted": True,
                "halt_reason": "precondition fail",
            },
        }
        OUT_JSON.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[halt] precondition fail; results -> {OUT_JSON}")
        return 1
    print(f"  SHA256: {pre['sha256']}")
    print(f"  size:   {pre['size_b']} B")

    # Load frozen module
    print("[load] importing frozen canonical ...")
    mod = load_frozen_module()
    print("  OK (importlib isolated, __main__ guard preserves module-level safety)")

    # Build inputs
    print("[input] building sparc_for_audit (171 + g_obs + col_alias_map) ...")
    sparc_for_audit = build_sparc_for_audit(mod)
    print(f"  sparc_for_audit: {len(sparc_for_audit)} rows")
    dsph = load_dsph()
    print(f"  dsph:            {len(dsph)} rows")

    # Unified audit call
    print("[audit] b_alpha_3axis_audit(sparc_for_audit, dsph) ...")
    try:
        audit_result = mod.b_alpha_3axis_audit(sparc_for_audit, dsph)
        print("  audit OK")
    except Exception as exc:
        print(f"  audit ERROR: {exc!r}")
        result = {
            "timestamp": timestamp,
            "phase": "Phase 1a-validation G layer (G1-G3)",
            "precondition": pre,
            "audit_error": repr(exc),
            "summary": {"total": 0, "passed": 0, "all_pass": False, "halted": True},
        }
        OUT_JSON.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 1

    # Gates
    gates = []
    for gate_func, label in [
        (gate_g1_axis_1_sparc, "G1 axis_1_SPARC"),
        (gate_g2_axis_2_dsph, "G2 axis_2_dSph"),
        (gate_g3_axis_3_universal, "G3 axis_3 universal"),
    ]:
        print(f"[gate] {label} ...")
        try:
            r = gate_func(audit_result)
        except Exception as exc:
            r = {
                "gate": label.split()[0],
                "status": "ERROR",
                "reason": repr(exc),
            }
        gates.append(r)
        body = {k: v for k, v in r.items() if k not in ("gate", "status")}
        print(f"  -> {r['status']}: {json.dumps(body, ensure_ascii=False)}")

    n_pass = sum(1 for g in gates if g["status"] == "PASS")
    n_total = len(gates)
    all_pass = (n_pass == n_total)

    result = {
        "timestamp": timestamp,
        "phase": "Phase 1a-validation G layer (G1-G3)",
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": _platform.platform(),
        },
        "frozen_canonical": {
            "path_repo_relative": FROZEN_PATH_REPO_REL,
            "sha256": FROZEN_SHA256,
            "size_b": FROZEN_SIZE_B,
        },
        "baseline_immutable": {
            "axis_1_sparc": BASELINE_AXIS_1_SPARC,
            "axis_2_dsph": BASELINE_AXIS_2_DSPH,
            "axis_3_range": [RANGE_G3_LO, RANGE_G3_HI],
            "axis_3_target_band": TARGET_G3_BAND,
            "source": "anchor 19 1.5 (SHA prefix 0b269c10)",
        },
        "precondition": pre,
        "audit_result_full": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in audit_result.items()
        },
        "gates": gates,
        "summary": {
            "total": n_total,
            "passed": n_pass,
            "all_pass": all_pass,
            "next_phase": (
                "G layer (G4-G6) Lesson 93 / AC4 / AC5"
                if all_pass else "halt for diagnosis"
            ),
        },
    }

    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\n[done] G1-G3: {n_pass}/{n_total} PASS  "
        f"({'ALL PASS' if all_pass else 'NOT ALL PASS'})"
    )
    print(f"[out]  {OUT_JSON}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
