"""
Phase 1a mechanical alignment gates (b_α 非言及、circularity-free).

Gates:
  1. Q column present in load_MRT return statement
  2. Q<3 cut active in _prepare_sparc_phase_c3_sample
  3. sample_n_axis_1 == 124 (binary)
  4. galaxy identity match against phase_c3_step3 reference
  5. sample_n_axis_3 == 154 (= 124 + 30)

Forbidden in this phase:
  - b_alpha 値の計算 / 比較 / assertion
  - Lesson 93 universal coupling check
  - AC4 / AC5 gate
  - jackknife / bootstrap
"""
import os
import sys
import inspect
import importlib.util
import numpy as np
import pandas as pd

ROOT = r"D:\ドキュメント\エントロピー\膜宇宙論再考察AB効果有り\C3 拡張版仮説関連2"
DATA = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"

# Import patched canonical
spec = importlib.util.spec_from_file_location(
    "run_canonical", os.path.join(ROOT, "run_section2_5_v0_2.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_canonical"] = mod
spec.loader.exec_module(mod)

results = {}

# ===== Gate 1: Q column present in load_MRT return =====
print("=" * 70)
print("Gate 1: Q column present in load_MRT return statement")
print("=" * 70)
src = inspect.getsource(mod.load_MRT)
return_lines = [l for l in src.split("\n") if l.strip().startswith("return")]
return_line = return_lines[-1] if return_lines else ""
gate1 = '"Q"' in return_line
print(f"  return line: {return_line.strip()}")
print(f"  Gate 1: {'PASS' if gate1 else 'FAIL'} (Q column in return)")
results["gate_1_Q_in_load_MRT"] = gate1

# ===== Gate 2: Q<3 cut active =====
print()
print("=" * 70)
print("Gate 2: Q<3 cut active in _prepare_sparc_phase_c3_sample")
print("=" * 70)
ta3 = mod.load_TA3(os.path.join(DATA, "TA3_gc_independent.csv"))
phase1 = mod.load_phase1(os.path.join(DATA, "phase1", "sparc_results.csv"))
mrt = mod.load_MRT(os.path.join(DATA, "SPARC_Lelli2016c.mrt"))
print(f"  MRT columns: {list(mrt.columns)}")
gate2_mrt_has_Q = "Q" in mrt.columns
print(f"  MRT Q column present: {gate2_mrt_has_Q}")

sparc_full = mod.merge_three_files(ta3, phase1, mrt)
sparc_full = mod.mark_fit_pool_171(sparc_full)
sparc_171 = sparc_full[sparc_full["in_171_pool"]].copy()
print(f"  sparc_171 columns: {list(sparc_171.columns)}")
gate2_sparc_has_Q = "Q" in sparc_171.columns
print(f"  sparc_171 Q column present: {gate2_sparc_has_Q}")

# Build sparc_for_audit (mimic production caller side)
RC_DIR = os.path.join(DATA, "Rotmod_LTG")
rotation_curves_by_galaxy = {}
for g in sparc_171["galaxy"]:
    p = os.path.join(RC_DIR, f"{g}_rotmod.dat")
    if not os.path.exists(p):
        continue
    try:
        df = pd.read_csv(p, sep=r"\s+", comment="#", header=None,
                         names=["Rad", "Vobs", "errV", "Vgas", "Vdisk",
                                "Vbul", "SBdisk", "SBbul"], engine="python")
        rotation_curves_by_galaxy[g] = {
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

sparc_prep = mod._prepare_sparc_phase_c3_sample(sparc_for_audit)
gate2_q_active = sparc_prep is not None and "Q" in sparc_prep.columns and (sparc_prep["Q"] < 3).all()
if sparc_prep is not None and "Q" in sparc_prep.columns:
    print(f"  per-galaxy Q < 3: {(sparc_prep['Q'] < 3).all()}")
    print(f"  Q values: min={sparc_prep['Q'].min()}, max={sparc_prep['Q'].max()}")
print(f"  Gate 2: {'PASS' if (gate2_mrt_has_Q and gate2_sparc_has_Q and gate2_q_active) else 'FAIL'}")
results["gate_2_Q_lt_3_active"] = gate2_mrt_has_Q and gate2_sparc_has_Q and gate2_q_active

# ===== Gate 3: sample_n_axis_1 == 124 (binary) =====
print()
print("=" * 70)
print("Gate 3: sample_n_axis_1 == 124 (binary count match)")
print("=" * 70)
n1 = len(sparc_prep) if sparc_prep is not None else 0
print(f"  sample_n_axis_1 = {n1}")
print(f"  expected = 124")
gate3 = n1 == 124
print(f"  Gate 3: {'PASS' if gate3 else 'FAIL'}")
results["gate_3_sample_n_axis_1"] = gate3

# ===== Gate 4: galaxy identity match against reference =====
print()
print("=" * 70)
print("Gate 4: galaxy identity match against phase_c3_step3 reference")
print("=" * 70)
# Build reference set by running phase_c3_step3 filter chain explicitly
ref_df = sparc_full.copy().rename(columns={"Ud": "Upsilon_d"} if "Ud" in sparc_full.columns else {})
if "Q" in ref_df.columns:
    ref_df = ref_df[ref_df["Q"] < 3].reset_index(drop=True)
ref_df["Mstar"] = ref_df["Upsilon_d"] * ref_df["L36"] * 1e9
ref_df["Mgas"] = mod.HELIUM_FACTOR * ref_df["MHI"] * 1e9
ref_df["Mbar"] = ref_df["Mstar"] + ref_df["Mgas"]
ref_df["hR"] = ref_df["Rdisk"]
ref_df["r_h"] = 1.68 * ref_df["hR"]
ref_df["rho_gal"] = ref_df["Mbar"] / (4.0 / 3.0 * np.pi * ref_df["r_h"] ** 3)
ref_df["v_flat"] = ref_df["Vflat"]
ref_df = ref_df[ref_df["v_flat"] > 0].reset_index(drop=True)
ref_df["gc_C15"] = (mod.C15_COEF * ref_df["Upsilon_d"]**(mod.C15_UPSILON_EXP)
                    * np.sqrt(mod.A0_KPC * ref_df["v_flat"]**2 / ref_df["hR"]))
ref_df = ref_df[(ref_df["gc_C15"] > 0) & np.isfinite(ref_df["gc_C15"])].reset_index(drop=True)
ref_df = ref_df.dropna(subset=["rho_gal", "r_h", "Upsilon_d", "hR", "Mbar"]).reset_index(drop=True)
ref_df = ref_df[(ref_df["rho_gal"] > 0) & (ref_df["r_h"] > 0) & (ref_df["Mbar"] > 0)].reset_index(drop=True)
BRIDGE = {"NGC3741", "NGC2915", "ESO444-G084", "NGC1705"}
ref_df = ref_df[~ref_df["galaxy"].isin(BRIDGE)].reset_index(drop=True)

def gC2(g, hR):
    p = os.path.join(RC_DIR, f"{g}_rotmod.dat")
    if not os.path.exists(p): return np.nan
    try:
        d = pd.read_csv(p, sep=r"\s+", comment="#", header=None,
                        names=["Rad", "Vobs", "errV", "Vgas", "Vdisk",
                               "Vbul", "SBdisk", "SBbul"], engine="python")
        R = d["Rad"].values; V = d["Vobs"].values
        if len(R) < 3: return np.nan
        m = (R > 2.0 * hR) & (V > 0)
        if int(m.sum()) < 2: return np.nan
        return float(np.mean(V[m]**2 / R[m]))
    except Exception:
        return np.nan
ref_df["g_obs"] = [gC2(r["galaxy"], r["hR"]) for _, r in ref_df.iterrows()]
with np.errstate(divide="ignore"):
    ref_df["delta_primary"] = np.log10(ref_df["g_obs"]) - np.log10(ref_df["gc_C15"])
ref_df = ref_df[np.isfinite(ref_df["delta_primary"])].reset_index(drop=True)

ref_set = set(ref_df["galaxy"])
can_set = set(sparc_prep["Galaxy"]) if sparc_prep is not None else set()
sym_diff = (can_set ^ ref_set)
gate4 = (can_set == ref_set) and (len(can_set) == 124)
print(f"  reference (phase_c3_step3) sample size: {len(ref_set)}")
print(f"  canonical (post-patch) sample size:     {len(can_set)}")
print(f"  symmetric difference:                   {len(sym_diff)}")
if sym_diff:
    print(f"    excess (CAN-REF): {sorted(can_set - ref_set)}")
    print(f"    missing (REF-CAN): {sorted(ref_set - can_set)}")
print(f"  Gate 4: {'PASS' if gate4 else 'FAIL'}")
results["gate_4_galaxy_identity"] = gate4

# ===== Gate 5: sample_n_axis_3 == 154 =====
print()
print("=" * 70)
print("Gate 5: sample_n_axis_3 == 154 (= 124 + 30)")
print("=" * 70)
dsph = pd.read_csv(os.path.join(DATA, "dsph_jeans_c15_v1.csv"))
dsph_prep = mod._prepare_dsph_phase_c3_sample(dsph)
n2 = len(dsph_prep) if dsph_prep is not None else 0
n3 = n1 + n2
print(f"  sample_n_axis_2 (dSph) = {n2}")
print(f"  sample_n_axis_3 (combined) = {n3} (= n1 + n2)")
print(f"  expected = 154")
gate5 = n3 == 154
print(f"  Gate 5: {'PASS' if gate5 else 'FAIL'}")
results["gate_5_sample_n_axis_3"] = gate5

# ===== Summary =====
print()
print("=" * 70)
print("Phase 1a mechanical gates summary")
print("=" * 70)
all_pass = True
for k, v in results.items():
    status = "PASS" if v else "FAIL"
    print(f"  {k}: {status}")
    if not v:
        all_pass = False
print()
print(f"Phase 1a alignment: {'PASS' if all_pass else 'FAIL'} ({sum(results.values())}/{len(results)})")
print()
print("FORBIDDEN in Phase 1a (per circularity-free design):")
print("  X b_alpha 値の計算 / 比較 / assertion")
print("  X Lesson 93 universal coupling check")
print("  X AC4 / AC5 gate")
print("  X jackknife / bootstrap")
print("These verifications belong in Phase 1a-validation (separate round, frozen Phase 1a output).")
