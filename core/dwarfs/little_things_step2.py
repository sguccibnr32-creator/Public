"""
LITTLE THINGS 独立検証 Step 2: g_c測定 → α=0.5 検定
======================================================
uv run --with scipy --with matplotlib --with numpy python little_things_step2.py

Oh et al. 2015 のスケール済み回転曲線データから:
  rotdmbar.dat (DM+baryon = total) と rotdm.dat (DM only)
  → V_bar = sqrt(V_obs² - V_DM²) でバリオン成分を抽出
  → RAR g_obs vs g_bar から g_c をフィット
  → α=0.5 (幾何平均法則) を独立検証
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, t as t_dist
from pathlib import Path
import json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 0. 定数・設定
# ============================================================
DATADIR = Path("little_things_data")
OUTDIR  = Path("little_things_results"); OUTDIR.mkdir(exist_ok=True)
a0 = 1.2e-10          # m/s²  MOND加速度定数
G  = 6.674e-11         # m³ kg⁻¹ s⁻²
Msun = 1.989e30        # kg
kpc_m = 3.086e19       # m/kpc

# ============================================================
# 1. rotdmbar.dat / rotdm.dat パーサー
# ============================================================
def parse_rot_file(filepath):
    """
    Oh+2015 スケール済み回転曲線ファイルを読み込む。
    形式: Name(A8) Type(A5) R0.3(F8.6,kpc) V0.3(F10.6,km/s) R/R0.3 V/V0.3 e_V/V0.3
    戻り値: {galaxy_name: {"Data": {"r_kpc":[], "v_kms":[], "ev_kms":[]},
                            "Model": {...}}}
    """
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            name = parts[0]
            dtype = parts[1]   # "Data" or "Model"
            try:
                R03  = float(parts[2])   # kpc
                V03  = float(parts[3])   # km/s
                r_sc = float(parts[4])   # R/R0.3 (dimensionless)
                v_sc = float(parts[5])   # V/V0.3
                ev_sc= float(parts[6])   # e_V/V0.3
            except ValueError:
                continue

            r_kpc = r_sc * R03           # physical radius [kpc]
            v_kms = v_sc * V03           # physical velocity [km/s]
            ev_kms= ev_sc * V03          # uncertainty [km/s]

            if name not in galaxies:
                galaxies[name] = {}
            if dtype not in galaxies[name]:
                galaxies[name][dtype] = {"r_kpc": [], "v_kms": [], "ev_kms": [],
                                          "R03": R03, "V03": V03}
            galaxies[name][dtype]["r_kpc"].append(r_kpc)
            galaxies[name][dtype]["v_kms"].append(v_kms)
            galaxies[name][dtype]["ev_kms"].append(ev_kms)

    # リスト → numpy配列
    for gname in galaxies:
        for dtype in galaxies[gname]:
            for key in ("r_kpc", "v_kms", "ev_kms"):
                galaxies[gname][dtype][key] = np.array(galaxies[gname][dtype][key])
    return galaxies

# ============================================================
# 2. table2.dat パーサー (Mgas, Mstar, Rc 等)
# ============================================================
def parse_table2(filepath):
    """
    table2.dat: pipe-delimited, fixed-width
    Byte-by-byte format from ReadMe.
    """
    props = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            # 固定幅パース (ReadMe参照)
            name = line[0:8].strip()
            try:
                Rmax   = float(line[9:13].strip())    if line[9:13].strip() else np.nan
                R03    = float(line[14:18].strip())   if line[14:18].strip() else np.nan
                VRmax  = float(line[19:24].strip())   if line[19:24].strip() else np.nan
                Rc     = float(line[85:89].strip())   if line[85:89].strip() else np.nan
                eRc    = float(line[90:95].strip())   if line[90:95].strip() else np.nan
                # Mgas (bytes 140-145), MstarK (147-151), MstarSED (153-157)
                Mgas_str    = line[139:145].strip()
                MstarK_str  = line[146:151].strip()
                MstarSED_str= line[152:157].strip()
                Mgas    = float(Mgas_str)    if Mgas_str else np.nan
                MstarK  = float(MstarK_str)  if MstarK_str else np.nan
                MstarSED= float(MstarSED_str)if MstarSED_str else np.nan
            except (ValueError, IndexError):
                continue

            # 最良の恒星質量: SED優先、なければ運動学的
            Mstar = MstarSED if not np.isnan(MstarSED) else MstarK

            props[name] = {
                "Rmax_kpc": Rmax,
                "R03_kpc": R03,
                "VRmax_kms": VRmax,
                "Rc_kpc": Rc,
                "Mgas_1e7Msun": Mgas,
                "Mstar_1e7Msun": Mstar,
                "MstarK_1e7Msun": MstarK,
                "MstarSED_1e7Msun": MstarSED,
            }
    return props

# ============================================================
# 3. table1.dat パーサー (距離等)
# ============================================================
def parse_table1(filepath):
    """table1.dat: pipe-delimited, fixed-width"""
    props = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            name = line[0:8].strip()
            try:
                dist = float(line[32:36].strip())  # bytes 33-36: Distance [Mpc]
                VMag = float(line[69:74].strip())   # bytes 70-74: abs V mag
            except (ValueError, IndexError):
                continue
            props[name] = {"dist_Mpc": dist, "VMag": VMag}
    return props

# ============================================================
# 4. V_bar 抽出: V_bar² = V_obs² - V_DM²
# ============================================================
def extract_v_bar(total_data, dm_data):
    """
    rotdmbar (total) と rotdm (DM only) の "Data" 行を結合。
    物理半径でマッチし、V_bar = sqrt(V_obs² - V_DM²) を計算。
    """
    r_tot = total_data["r_kpc"]
    v_tot = total_data["v_kms"]
    ev_tot= total_data["ev_kms"]

    r_dm  = dm_data["r_kpc"]
    v_dm  = dm_data["v_kms"]

    # DM速度をtotalの半径に線形補間
    v_dm_interp = np.interp(r_tot, r_dm, v_dm, left=np.nan, right=np.nan)

    # V_bar² = V_tot² - V_DM²
    v_bar_sq = v_tot**2 - v_dm_interp**2

    # V_bar² < 0 の場合は V_bar = 0 として扱う（DM-dominant regime）
    v_bar = np.where(v_bar_sq > 0, np.sqrt(v_bar_sq), 0.0)

    valid = ~np.isnan(v_dm_interp) & (r_tot > 0)

    return {
        "r_kpc": r_tot[valid],
        "v_obs": v_tot[valid],
        "ev_obs": ev_tot[valid],
        "v_bar": v_bar[valid],
        "v_dm": v_dm_interp[valid],
    }

# ============================================================
# 5. g_c フィット (MOND/膜補間関数)
# ============================================================
def mond_interp(g_N, g_c):
    """Simple interpolating function: g_obs = g_N / (1 - exp(-sqrt(g_N/g_c)))"""
    x = np.sqrt(g_N / g_c)
    # 数値安定性
    return np.where(x > 30, g_N, g_N / (1.0 - np.exp(-x)))


def mond_simple(g_N, g_c):
    """Standard MOND: g_obs = (g_N + sqrt(g_N² + 4*g_c*g_N)) / 2"""
    return (g_N + np.sqrt(g_N**2 + 4 * g_c * g_N)) / 2


def fit_gc(r_kpc, v_obs, ev_obs, v_bar):
    """
    g_obs vs g_bar の RAR から g_c をフィット。
    戻り値: dict(g_c, log_gc_a0, chi2_dof, n_points, success)
    """
    r_m  = r_kpc * kpc_m
    v_o  = v_obs * 1e3       # m/s
    v_b  = v_bar * 1e3
    ev   = np.maximum(ev_obs, 1.0) * 1e3   # 下限 1 km/s

    # g_obs と g_bar
    mask = (r_m > 0) & (v_b > 0)
    if np.sum(mask) < 4:
        # v_bar=0 の点も含めてみる（g_bar=0 に近い極端なDM-dominatedケース）
        mask = r_m > 0
        if np.sum(mask) < 4:
            return {"success": False, "reason": "valid < 4"}

    g_obs = v_o[mask]**2 / r_m[mask]
    g_bar = v_b[mask]**2 / r_m[mask]
    # 誤差伝播: δg_obs = 2*v*δv / r
    e_g = 2 * v_o[mask] * ev[mask] / r_m[mask]
    e_g = np.maximum(e_g, 1e-13)

    # g_bar > 0 の点のみ使用（log空間の回帰に必要）
    pos = (g_bar > 0) & (g_obs > 0)
    if np.sum(pos) < 4:
        return {"success": False, "reason": "g_bar>0 points < 4"}

    g_obs_v = g_obs[pos]
    g_bar_v = g_bar[pos]
    e_g_v   = e_g[pos]

    # chi² minimization over log(g_c)
    def chi2_func(log_gc):
        gc = 10**log_gc
        g_pred = mond_simple(g_bar_v, gc)
        return np.sum(((g_obs_v - g_pred) / e_g_v)**2)

    # グリッドサーチ → Brent精密化
    best = min(((chi2_func(lgc), lgc) for lgc in np.linspace(-12, -8, 200)),
               key=lambda x: x[0])

    try:
        res = minimize_scalar(chi2_func,
                              bounds=(best[1] - 1.5, best[1] + 1.5),
                              method='bounded')
        gc_best = 10**res.x
        chi2_best = res.fun
    except Exception:
        gc_best = 10**best[1]
        chi2_best = best[0]

    dof = np.sum(pos) - 1
    chi2_dof = chi2_best / max(dof, 1)

    return {
        "success": True,
        "g_c": gc_best,
        "log_gc_a0": np.log10(gc_best / a0),
        "chi2_dof": chi2_dof,
        "n_points": int(np.sum(pos)),
        "g_bar_range": (float(g_bar_v.min()), float(g_bar_v.max())),
    }

# ============================================================
# 6. G·Σ₀ の推定
# ============================================================
def estimate_G_Sigma0(gal_props, v_flat_kms, r_kpc_arr):
    """
    方法1: V_flat² / h_R  (SPARCと同じ)
    方法2: G*(Mgas+Mstar) / (2π R_d²)  (table2の質量使用)
    """
    results = {}

    # === 方法1: V_flat² / h_R ===
    # h_R を回転曲線のピーク半径から推定
    # 指数ディスク: V_bar ピーク ≈ 2.2 h_R
    # ここではR0.3 (dlog V/dlog R = 0.3 の半径) を代理として使う
    # R0.3 ≈ 2 h_R (Oh+2015 の定義から概算)
    R03_kpc = gal_props.get("R03_kpc", np.nan)
    if not np.isnan(R03_kpc) and R03_kpc > 0:
        h_R_est = R03_kpc / 2.0   # R0.3 ≈ 2 h_R (近似)
    else:
        # フォールバック: 回転曲線の1/3半径
        h_R_est = np.median(r_kpc_arr) / 2.0

    h_R_m = h_R_est * kpc_m
    v_flat_ms = v_flat_kms * 1e3
    GS0_v = v_flat_ms**2 / h_R_m
    results["method1_vflat"] = GS0_v
    results["h_R_kpc"] = h_R_est

    # === 方法2: 質量ベース ===
    Mgas  = gal_props.get("Mgas_1e7Msun", np.nan)
    Mstar = gal_props.get("Mstar_1e7Msun", np.nan)
    Rc    = gal_props.get("Rc_kpc", np.nan)

    if not np.isnan(Mgas) and not np.isnan(Rc) and Rc > 0:
        Mbar_kg = (Mgas + (Mstar if not np.isnan(Mstar) else 0)) * 1e7 * Msun
        Rd_m = Rc * kpc_m  # pseudo-isothermal core radius ≈ disk scale
        Sigma0 = Mbar_kg / (2 * np.pi * Rd_m**2)
        GS0_mass = G * Sigma0
        results["method2_mass"] = GS0_mass
    else:
        results["method2_mass"] = np.nan

    # 採用: 方法1を主、方法2をクロスチェック
    results["adopted"] = GS0_v
    return results

# ============================================================
# 7. メイン
# ============================================================
def main():
    print("=" * 70)
    print("LITTLE THINGS Step 2: g_c測定 → α=0.5 検定")
    print("=" * 70)

    # --- データ読み込み ---
    print("\nデータ読み込み...")
    total = parse_rot_file(DATADIR / "rotdmbar.dat")
    dm    = parse_rot_file(DATADIR / "rotdm.dat")
    t2    = parse_table2(DATADIR / "table2.dat")
    t1    = parse_table1(DATADIR / "table1.dat")

    print(f"  rotdmbar.dat: {len(total)} 銀河")
    print(f"  rotdm.dat:    {len(dm)} 銀河")
    print(f"  table2.dat:   {len(t2)} 銀河")

    # 共通銀河 (Data行がある)
    common = sorted(set(total.keys()) & set(dm.keys()))
    common = [g for g in common
              if "Data" in total[g] and "Data" in dm[g]]
    print(f"  共通銀河 (Data有): {len(common)}")

    # --- 各銀河を処理 ---
    results = []
    failures = []

    hdr = (f"{'銀河':<10} {'N':>3} {'log(gc/a0)':>10} {'V_flat':>6} "
           f"{'h_R':>5} {'log(GΣ₀)':>9} {'χ²/dof':>7}")
    print(f"\n{hdr}")
    print("-" * 60)

    for gname in common:
        # V_bar 抽出
        vbar_data = extract_v_bar(total[gname]["Data"], dm[gname]["Data"])
        if len(vbar_data["r_kpc"]) < 4:
            failures.append((gname, "too few matched points"))
            continue

        # V_barが全てゼロ（完全DM-dominated）→ スキップ
        if np.all(vbar_data["v_bar"] < 0.1):
            failures.append((gname, "v_bar ≈ 0 everywhere"))
            continue

        # g_c フィット
        fit = fit_gc(vbar_data["r_kpc"], vbar_data["v_obs"],
                     vbar_data["ev_obs"], vbar_data["v_bar"])

        if not fit["success"]:
            failures.append((gname, fit.get("reason", "fit failed")))
            continue

        # V_flat: 外側1/4の中央値
        n_outer = max(3, len(vbar_data["v_obs"]) // 4)
        v_flat = np.median(vbar_data["v_obs"][-n_outer:])

        # G·Σ₀
        gprops = t2.get(gname, {})
        gs0 = estimate_G_Sigma0(gprops, v_flat, vbar_data["r_kpc"])

        log_GS0 = np.log10(gs0["adopted"]) if gs0["adopted"] > 0 else np.nan

        entry = {
            "name": gname,
            "g_c": fit["g_c"],
            "log_gc_a0": fit["log_gc_a0"],
            "log_gc_abs": np.log10(fit["g_c"]),
            "chi2_dof": fit["chi2_dof"],
            "n_points": fit["n_points"],
            "v_flat": v_flat,
            "h_R_kpc": gs0["h_R_kpc"],
            "G_Sigma0": gs0["adopted"],
            "log_G_Sigma0": log_GS0,
            "gc_pred_alpha05": np.sqrt(a0 * gs0["adopted"]),
            # 回転曲線データ（プロット用）
            "_r": vbar_data["r_kpc"].tolist(),
            "_vobs": vbar_data["v_obs"].tolist(),
            "_vbar": vbar_data["v_bar"].tolist(),
        }
        results.append(entry)

        print(f"{gname:<10} {fit['n_points']:>3} {fit['log_gc_a0']:>10.3f} "
              f"{v_flat:>6.1f} {gs0['h_R_kpc']:>5.2f} "
              f"{log_GS0:>9.3f} {fit['chi2_dof']:>7.2f}")

    if failures:
        print(f"\n--- 失敗 ({len(failures)}銀河) ---")
        for gn, reason in failures:
            print(f"  {gn}: {reason}")

    N = len(results)
    if N < 3:
        print(f"\n有効銀河数 {N} < 3。中断。")
        sys.exit(1)

    # ============================================================
    # 8. α 検定
    # ============================================================
    print(f"\n{'='*70}")
    print(f"α 検定  (N = {N} 銀河)")
    print(f"{'='*70}")

    log_gc  = np.array([r["log_gc_abs"] for r in results])
    log_GS0 = np.array([r["log_G_Sigma0"] for r in results])

    # 欠損除去
    valid = np.isfinite(log_gc) & np.isfinite(log_GS0)
    log_gc  = log_gc[valid]
    log_GS0 = log_GS0[valid]
    N = len(log_gc)

    # OLS: log(gc) = a + b * log(G·Σ₀)   →  b = α
    A = np.vstack([np.ones(N), log_GS0]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_gc, rcond=None)
    intercept, alpha_fit = coeffs

    y_pred = A @ coeffs
    resid  = log_gc - y_pred
    s2     = np.sum(resid**2) / (N - 2)
    cov    = s2 * np.linalg.inv(A.T @ A)
    se_alpha    = np.sqrt(cov[1, 1])
    se_intercept= np.sqrt(cov[0, 0])
    resid_std   = np.std(resid)

    print(f"\n  log(g_c) = {intercept:.3f} + {alpha_fit:.3f} × log(G·Σ₀)")
    print(f"  α = {alpha_fit:.3f} ± {se_alpha:.3f}")
    print(f"  切片 = {intercept:.3f} ± {se_intercept:.3f}")
    print(f"  残差σ = {resid_std:.3f} dex")

    # 仮説検定
    t_05 = (alpha_fit - 0.5) / se_alpha
    p_05 = 2 * t_dist.sf(abs(t_05), N - 2)

    t_0  = alpha_fit / se_alpha
    p_0  = 2 * t_dist.sf(abs(t_0), N - 2)

    t_1  = (alpha_fit - 1.0) / se_alpha
    p_1  = 2 * t_dist.sf(abs(t_1), N - 2)

    t_crit = t_dist.ppf(0.975, N - 2)
    ci = (alpha_fit - t_crit * se_alpha, alpha_fit + t_crit * se_alpha)

    print(f"\n--- 仮説検定 ---")
    print(f"  α=0.5 (幾何平均): t={t_05:+.3f}, p={p_05:.4f}  "
          f"{'棄却不可 ✓' if p_05 > 0.05 else '棄却 ✗'}")
    print(f"  α=0   (MOND):     t={t_0:+.3f}, p={p_0:.2e}  "
          f"{'棄却不可' if p_0 > 0.05 else '棄却 ✗'}")
    print(f"  α=1   (純Σ₀):    t={t_1:+.3f}, p={p_1:.2e}  "
          f"{'棄却不可' if p_1 > 0.05 else '棄却 ✗'}")
    print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  α=0.5 in CI: {'YES ✓' if ci[0] <= 0.5 <= ci[1] else 'NO ✗'}")

    rho, p_sp = spearmanr(log_GS0, log_gc)
    print(f"  Spearman ρ = {rho:.3f} (p = {p_sp:.2e})")

    # ============================================================
    # 9. SPARC比較
    # ============================================================
    print(f"\n{'='*70}")
    print("SPARC との比較")
    print(f"{'='*70}")

    sparc_alpha, sparc_se = 0.545, 0.041

    print(f"  {'':25} {'SPARC(175)':>12} {'LT('+str(N)+')':>12}")
    print(f"  {'-'*49}")
    print(f"  {'α':25} {sparc_alpha:>12.3f} {alpha_fit:>12.3f}")
    print(f"  {'σ_α':25} {sparc_se:>12.3f} {se_alpha:>12.3f}")
    print(f"  {'残差σ [dex]':25} {'0.313':>12} {resid_std:>12.3f}")
    print(f"  {'p(α=0.5)':25} {'0.27':>12} {p_05:>12.4f}")

    diff = abs(alpha_fit - sparc_alpha)
    comb_se = np.sqrt(se_alpha**2 + sparc_se**2)
    z_diff = diff / comb_se
    print(f"\n  α差: |{alpha_fit:.3f} - {sparc_alpha:.3f}| = {diff:.3f}")
    print(f"  合成σ = {comb_se:.3f},  z = {z_diff:.2f}  "
          f"({'整合 ✓' if z_diff < 2 else '不整合 ✗'})")

    # ============================================================
    # 10. AIC比較
    # ============================================================
    print(f"\n{'='*70}")
    print("モデル比較 (AIC)")
    print(f"{'='*70}")

    # MOND: gc = a0 (0パラメータ)
    gc_mond = np.full(N, np.log10(a0))
    rss_mond = np.sum((log_gc - gc_mond)**2)
    aic_mond = N * np.log(rss_mond / N)  # + 0

    # 幾何平均 α=0.5 固定 (1パラメータ: η)
    gc_geom = 0.5 * (np.log10(a0) + log_GS0)
    eta_shift = np.mean(log_gc - gc_geom)
    gc_geom_fit = gc_geom + eta_shift
    rss_geom = np.sum((log_gc - gc_geom_fit)**2)
    aic_geom = N * np.log(rss_geom / N) + 2 * 1

    # α自由 (2パラメータ)
    rss_free = np.sum(resid**2)
    aic_free = N * np.log(rss_free / N) + 2 * 2

    d_geom = aic_geom - aic_mond
    d_free = aic_free - aic_mond

    print(f"  {'モデル':30} {'k':>2} {'σ[dex]':>8} {'ΔAIC':>8}")
    print(f"  {'-'*48}")
    print(f"  {'MOND (gc=a0)':30} {'0':>2} {np.sqrt(rss_mond/N):>8.3f} {'0':>8}")
    print(f"  {'幾何平均(α=0.5)':30} {'1':>2} {np.sqrt(rss_geom/N):>8.3f} {d_geom:>8.1f}")
    print(f"  {'幾何平均(α自由)':30} {'2':>2} {np.sqrt(rss_free/N):>8.3f} {d_free:>8.1f}")
    print(f"  η(α=0.5) = 10^{eta_shift:.3f} = {10**eta_shift:.3f}")

    # ============================================================
    # 11. ブートストラップ（α安定性）
    # ============================================================
    print(f"\n{'='*70}")
    print("ブートストラップ (10000回)")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)
    alpha_boot = []
    for _ in range(10000):
        idx = rng.choice(N, N, replace=True)
        Ab = np.vstack([np.ones(N), log_GS0[idx]]).T
        try:
            cb, _, _, _ = np.linalg.lstsq(Ab, log_gc[idx], rcond=None)
            alpha_boot.append(cb[1])
        except Exception:
            pass
    alpha_boot = np.array(alpha_boot)
    boot_ci = np.percentile(alpha_boot, [2.5, 97.5])
    frac_05 = np.mean((alpha_boot > 0.4) & (alpha_boot < 0.6))

    print(f"  α(boot) = {np.mean(alpha_boot):.3f} ± {np.std(alpha_boot):.3f}")
    print(f"  95% CI: [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]")
    print(f"  α ∈ [0.4, 0.6] の割合: {frac_05*100:.1f}%")

    # ============================================================
    # 12. 結果保存
    # ============================================================
    # プロット用の内部データを除去
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if not k.startswith("_")}
        results_clean.append(rc)

    summary = {
        "dataset": "LITTLE THINGS (Oh et al. 2015)",
        "n_galaxies": N,
        "n_failed": len(failures),
        "alpha_fit": float(alpha_fit),
        "alpha_se": float(se_alpha),
        "alpha_95ci": [float(ci[0]), float(ci[1])],
        "alpha_boot_95ci": [float(boot_ci[0]), float(boot_ci[1])],
        "p_alpha_05": float(p_05),
        "p_alpha_0": float(p_0),
        "p_alpha_1": float(p_1),
        "residual_std_dex": float(resid_std),
        "spearman_rho": float(rho),
        "spearman_p": float(p_sp),
        "dAIC_geom_vs_mond": float(d_geom),
        "dAIC_free_vs_mond": float(d_free),
        "eta_alpha05": float(10**eta_shift),
        "sparc_comparison": {
            "alpha_diff": float(diff),
            "z_score": float(z_diff),
            "consistent": bool(z_diff < 2),
        },
        "galaxies": results_clean,
        "failures": [{"name": n, "reason": r} for n, r in failures],
    }

    outpath = OUTDIR / "step2_results.json"
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n結果保存: {outpath}")

    # ============================================================
    # 13. 診断プロット
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("LITTLE THINGS: Geometric-Mean Law  "
                     r"$\alpha=0.5$ Test (Independent of SPARC)",
                     fontsize=14, fontweight='bold')

        # (a) log(gc) vs log(G·Σ₀)
        ax = axes[0, 0]
        ax.scatter(log_GS0, log_gc, c='steelblue', s=60, zorder=3,
                   edgecolors='navy', linewidth=0.5, label='LITTLE THINGS')
        x_r = np.linspace(log_GS0.min() - 0.3, log_GS0.max() + 0.3, 100)
        ax.plot(x_r, intercept + alpha_fit * x_r, 'r-', lw=2,
                label=rf'$\alpha={alpha_fit:.3f}\pm{se_alpha:.3f}$')
        ax.plot(x_r, 0.5 * np.log10(a0) + 0.5 * x_r + eta_shift, 'g--', lw=1.5,
                label=r'$\alpha=0.5$')
        ax.axhline(np.log10(a0), color='gray', ls=':', lw=1, label='MOND ($g_c=a_0$)')
        for r in results:
            ax.annotate(r["name"], (r["log_G_Sigma0"], r["log_gc_abs"]),
                       fontsize=6, alpha=0.7, xytext=(3, 3),
                       textcoords='offset points')
        ax.set_xlabel(r"$\log(G\cdot\Sigma_0)$ [m/s$^2$]")
        ax.set_ylabel(r"$\log(g_c)$ [m/s$^2$]")
        ax.set_title(rf"$\alpha={alpha_fit:.3f}\pm{se_alpha:.3f}$, "
                     rf"$p(\alpha=0.5)={p_05:.3f}$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (b) 残差
        ax = axes[0, 1]
        ax.hist(resid, bins=max(5, N // 3), color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', ls='--')
        ax.set_xlabel("Residual [dex]")
        ax.set_ylabel("Count")
        ax.set_title(rf"Residuals ($\sigma={resid_std:.3f}$ dex)")
        ax.grid(True, alpha=0.3)

        # (c) SPARC比較: αのCI
        ax = axes[1, 0]
        yp = [0, 1]
        labels = ['SPARC\n(N=175)', f'LITTLE THINGS\n(N={N})']
        als = [sparc_alpha, alpha_fit]
        ers = [sparc_se * 1.96, se_alpha * t_crit]
        colors_ci = ['navy', 'steelblue']
        for i in range(2):
            ax.errorbar(als[i], yp[i], xerr=ers[i], fmt='o', ms=10,
                       capsize=8, color=colors_ci[i], elinewidth=2)
        ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
        ax.axvline(0, color='gray', ls=':', label=r'$\alpha=0$ (MOND)')
        ax.axvline(1, color='gray', ls=':')
        ax.set_xlabel(r'$\alpha$')
        ax.set_yticks(yp)
        ax.set_yticklabels(labels)
        ax.set_title("95% CI comparison")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.3, 1.5)
        ax.grid(True, alpha=0.3)

        # (d) ブートストラップ分布
        ax = axes[1, 1]
        ax.hist(alpha_boot, bins=50, color='steelblue', edgecolor='white',
                alpha=0.7, density=True)
        ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
        ax.axvline(alpha_fit, color='red', lw=2, label=f'OLS $\\alpha$={alpha_fit:.3f}')
        ax.axvline(sparc_alpha, color='navy', ls=':', lw=1.5,
                   label=f'SPARC $\\alpha$={sparc_alpha}')
        ax.set_xlabel(r'$\alpha$ (bootstrap)')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap (N=10000): '
                     rf'$\alpha={np.mean(alpha_boot):.3f}\pm{np.std(alpha_boot):.3f}$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figpath = OUTDIR / "step2_alpha_test.png"
        plt.savefig(figpath, dpi=150, bbox_inches='tight')
        print(f"プロット保存: {figpath}")

        # === 追加プロット: RAR (g_obs vs g_bar) 全銀河重ね描き ===
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        colors_gal = plt.cm.tab20(np.linspace(0, 1, min(N, 20)))
        for i, r in enumerate(results):
            rr = np.array(r["_r"])
            vo = np.array(r["_vobs"])
            vb = np.array(r["_vbar"])
            rm = rr * kpc_m
            mask2 = (rm > 0) & (vb > 0)
            if np.sum(mask2) < 2:
                continue
            go = (vo[mask2]*1e3)**2 / rm[mask2]
            gb = (vb[mask2]*1e3)**2 / rm[mask2]
            ax2.scatter(np.log10(gb), np.log10(go), s=15, alpha=0.6,
                       color=colors_gal[i % 20], label=r["name"])
        # 理論線
        gb_range = np.logspace(-13, -8, 200)
        ax2.plot(np.log10(gb_range), np.log10(mond_simple(gb_range, a0)),
                'k-', lw=2, label='MOND ($g_c=a_0$)')
        gc_med = np.median([r["g_c"] for r in results])
        ax2.plot(np.log10(gb_range), np.log10(mond_simple(gb_range, gc_med)),
                'r--', lw=2, label=f'Median $g_c$={gc_med/a0:.2f}$a_0$')
        ax2.plot(np.log10(gb_range), np.log10(gb_range),
                'gray', ls=':', lw=1, label='1:1')
        ax2.set_xlabel(r'$\log(g_{\rm bar})$ [m/s$^2$]')
        ax2.set_ylabel(r'$\log(g_{\rm obs})$ [m/s$^2$]')
        ax2.set_title('Radial Acceleration Relation — LITTLE THINGS')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        figpath2 = OUTDIR / "step2_RAR.png"
        plt.savefig(figpath2, dpi=150, bbox_inches='tight')
        print(f"RARプロット保存: {figpath2}")

    except ImportError:
        print("matplotlib なし。プロットスキップ。")

    # ============================================================
    # 14. 最終判定
    # ============================================================
    print(f"\n{'='*70}")
    print("最終判定")
    print(f"{'='*70}")

    verdicts = []
    if p_05 > 0.05:
        verdicts.append(f"✓ α=0.5 棄却不可 (p={p_05:.3f})")
    else:
        verdicts.append(f"✗ α=0.5 棄却 (p={p_05:.3f})")

    if z_diff < 2:
        verdicts.append(f"✓ SPARC α={sparc_alpha} と整合 (z={z_diff:.2f})")
    else:
        verdicts.append(f"✗ SPARC α={sparc_alpha} と不整合 (z={z_diff:.2f})")

    if d_geom < -2:
        verdicts.append(f"✓ 幾何平均 > MOND (ΔAIC={d_geom:.1f})")
    elif d_geom < 0:
        verdicts.append(f"△ 幾何平均 ≳ MOND (ΔAIC={d_geom:.1f})")
    else:
        verdicts.append(f"✗ 幾何平均 ≤ MOND (ΔAIC={d_geom:.1f})")

    if p_sp < 0.01:
        verdicts.append(f"✓ gc-GΣ₀ 相関有意 (ρ={rho:.3f}, p={p_sp:.2e})")
    elif p_sp < 0.05:
        verdicts.append(f"△ gc-GΣ₀ 相関有意 (ρ={rho:.3f}, p={p_sp:.3f})")
    else:
        verdicts.append(f"✗ gc-GΣ₀ 相関非有意 (ρ={rho:.3f}, p={p_sp:.3f})")

    for v in verdicts:
        print(f"  {v}")

    n_pass = sum(1 for v in verdicts if v.startswith("✓"))
    print(f"\n  総合: {n_pass}/{len(verdicts)} 通過")
    if n_pass == len(verdicts):
        print("  → 幾何平均法則は独立データセットで完全に再現")
    elif n_pass >= len(verdicts) - 1:
        print("  → 幾何平均法則は独立データセットで概ね支持")
    elif n_pass >= 2:
        print("  → 部分的支持、追加検証が必要")
    else:
        print("  → 独立再現に失敗")

    print(f"\n{'='*70}")
    print("Step 2 完了")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
