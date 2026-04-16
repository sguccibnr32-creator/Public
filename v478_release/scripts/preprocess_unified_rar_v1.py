# -*- coding: utf-8 -*-
"""
preprocess_unified_rar_v1.py
============================
膜宇宙論 v4.7.7 §9.2 統一RARプロット用前処理

実行環境: Claude Code on Windows
実行コマンド:
    uv run --with pandas --with numpy python preprocess_unified_rar_v1.py

入力ファイル (同じディレクトリに配置):
    TA3_gc_independent.csv       SPARC 175銀河、銀河スカラー (必須)
    dsph_jeans_c15_v1.csv        dSph 31銀河、前セッション出力 (必須)
    Rotmod_LTG/                  SPARC Rotmod_LTG フォルダ (推奨、RAR雲生成用)

出力 (claude.aiへアップロードするもの):
    rar_unified_galaxy_level.csv  SPARC 175 + dSph 31 = 206行、銀河スカラー
    rar_sparc_cloud.csv           SPARC RAR雲、~2700点 (Rotmodがある場合のみ)

作業ルール遵守:
    - TA3_gc_independent.csv 使用
    - カラム名: galaxy (小文字), gc_over_a0, upsilon_d, rs_tanh, v_flat
    - r_s は rs_tanh 使用 (MRTのRdiskは使わない)
    - ASCII互換 (ギリシャ文字禁止)
"""

import os
import glob
import numpy as np
import pandas as pd

# =======================================================
# 物理定数
# =======================================================
A0 = 1.2e-10       # MOND a0 [m/s^2]
KPC_TO_M = 3.0857e19
KM_TO_M = 1000.0
G_NEWTON = 6.674e-11

# SPARC Rotmod 用標準 Upsilon (銀河スカラー推奨値、C15解析と整合)
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7

# =======================================================
# Step 1: SPARC 銀河スカラー (TA3)
# =======================================================
print("=" * 60)
print("Step 1: SPARC galaxy-level scalars from TA3")
print("=" * 60)

ta3 = pd.read_csv("TA3_gc_independent.csv")
print(f"TA3 columns: {list(ta3.columns)}")
print(f"TA3 rows: {len(ta3)}")

required = ["galaxy", "gc_over_a0", "upsilon_d", "rs_tanh", "v_flat"]
missing = [c for c in required if c not in ta3.columns]
if missing:
    print(f"WARNING: missing columns {missing}")
    print("Please check TA3 structure.")

# 銀河特性加速度 g_obs_char = v_flat^2 / rs_tanh  [m/s^2]
v_flat_ms = ta3["v_flat"].to_numpy() * KM_TO_M
rs_tanh_m = ta3["rs_tanh"].to_numpy() * KPC_TO_M
g_obs_char = v_flat_ms**2 / rs_tanh_m

# g_bar_char: gc = g_obs^2 / g_bar (deep MOND 抽出) から逆算
#   g_bar_char = g_obs_char^2 / (gc_over_a0 * a0)
#   注: これは TA3 のgc_over_a0が独立抽出された場合のみ有意。
#       C15テスト用の gc 抽出方法に従う。
gc_si = ta3["gc_over_a0"].to_numpy() * A0
g_bar_char = g_obs_char**2 / gc_si

sparc_scalar = pd.DataFrame({
    "source": "SPARC",
    "galaxy": ta3["galaxy"].values,
    "log_gobs_a0": np.log10(g_obs_char / A0),
    "log_gbar_a0": np.log10(g_bar_char / A0),
    "upsilon_dyn": g_obs_char / g_bar_char,
    "gc_over_a0": ta3["gc_over_a0"].values,
    "upsilon_d": ta3["upsilon_d"].values,
    "v_flat_kms": ta3["v_flat"].values,
    "rs_tanh_kpc": ta3["rs_tanh"].values,
    "host": "SPARC",
    "sigma_kms": np.nan,
    "rh_pc": np.nan,
})

print(f"SPARC scalar rows: {len(sparc_scalar)}")
print(f"  log_gobs_a0: median={sparc_scalar['log_gobs_a0'].median():.3f}, "
      f"range=[{sparc_scalar['log_gobs_a0'].min():.2f}, {sparc_scalar['log_gobs_a0'].max():.2f}]")
print(f"  log_gbar_a0: median={sparc_scalar['log_gbar_a0'].median():.3f}, "
      f"range=[{sparc_scalar['log_gbar_a0'].min():.2f}, {sparc_scalar['log_gbar_a0'].max():.2f}]")
print(f"  upsilon_dyn: median={sparc_scalar['upsilon_dyn'].median():.2f}, "
      f"range=[{sparc_scalar['upsilon_dyn'].min():.2f}, {sparc_scalar['upsilon_dyn'].max():.2f}]")

# =======================================================
# Step 2: dSph 銀河スカラー (前セッション出力)
# =======================================================
print()
print("=" * 60)
print("Step 2: dSph galaxy-level scalars from previous session")
print("=" * 60)

dsph = pd.read_csv("dsph_jeans_c15_v1.csv")
print(f"dSph columns: {list(dsph.columns)}")
print(f"dSph rows: {len(dsph)}")

for col in ["log_gobs_a0", "log_gbar_a0"]:
    if col not in dsph.columns:
        raise SystemExit(f"ERROR: dsph_jeans_c15_v1.csv missing {col}")

dsph_scalar = pd.DataFrame({
    "source": "dSph",
    "galaxy": dsph["name"].values if "name" in dsph.columns else dsph.iloc[:, 0].values,
    "log_gobs_a0": dsph["log_gobs_a0"].values,
    "log_gbar_a0": dsph["log_gbar_a0"].values,
    "upsilon_dyn": 10**(dsph["log_gobs_a0"].values - dsph["log_gbar_a0"].values),
    "gc_over_a0": dsph["gc_exact_a0"].values if "gc_exact_a0" in dsph.columns else np.nan,
    "upsilon_d": np.nan,
    "v_flat_kms": np.nan,
    "rs_tanh_kpc": np.nan,
    "host": dsph["host"].values if "host" in dsph.columns else "Unknown",
    "sigma_kms": dsph["sigma"].values if "sigma" in dsph.columns else np.nan,
    "rh_pc": dsph["rh_pc"].values if "rh_pc" in dsph.columns else np.nan,
})

print(f"dSph scalar rows: {len(dsph_scalar)}")
print(f"  log_gobs_a0: median={dsph_scalar['log_gobs_a0'].median():.3f}, "
      f"range=[{dsph_scalar['log_gobs_a0'].min():.2f}, {dsph_scalar['log_gobs_a0'].max():.2f}]")
print(f"  log_gbar_a0: median={dsph_scalar['log_gbar_a0'].median():.3f}, "
      f"range=[{dsph_scalar['log_gbar_a0'].min():.2f}, {dsph_scalar['log_gbar_a0'].max():.2f}]")
print(f"  upsilon_dyn: median={dsph_scalar['upsilon_dyn'].median():.2f}, "
      f"range=[{dsph_scalar['upsilon_dyn'].min():.2f}, {dsph_scalar['upsilon_dyn'].max():.2f}]")

# =======================================================
# Step 3: マージ & 銀河レベル出力
# =======================================================
print()
print("=" * 60)
print("Step 3: Merge and output galaxy-level CSV")
print("=" * 60)

unified = pd.concat([sparc_scalar, dsph_scalar], ignore_index=True)
unified.to_csv("rar_unified_galaxy_level.csv", index=False)
print(f"OUTPUT: rar_unified_galaxy_level.csv  ({len(unified)} rows)")

# J3体制遷移候補の予備統計
print()
print("--- J3 transition threshold statistics ---")
for th in [3, 10, 30, 100]:
    n_sparc = (sparc_scalar["upsilon_dyn"] > th).sum()
    n_dsph = (dsph_scalar["upsilon_dyn"] > th).sum()
    print(f"  upsilon_dyn > {th:3d}:  SPARC={n_sparc:3d} / 175,  dSph={n_dsph:3d} / 31")

# =======================================================
# Step 4: SPARC RAR 雲生成 (Rotmod_LTG があれば)
# =======================================================
print()
print("=" * 60)
print("Step 4: SPARC RAR cloud from Rotmod_LTG (optional)")
print("=" * 60)

rotmod_dir = "Rotmod_LTG"
if not os.path.isdir(rotmod_dir):
    # 代替パスを試す
    for alt in ["./Rotmod_LTG", "../Rotmod_LTG", "SPARC_Rotmod_LTG"]:
        if os.path.isdir(alt):
            rotmod_dir = alt
            break

if os.path.isdir(rotmod_dir):
    print(f"Found: {rotmod_dir}")
    cloud_rows = []
    files = sorted(glob.glob(os.path.join(rotmod_dir, "*.dat")))
    n_ok = 0
    n_skip = 0
    for fpath in files:
        basename = os.path.basename(fpath)
        # 銀河名抽出: NGC2403_rotmod.dat -> NGC2403
        galaxy = basename
        for suffix in ["_rotmod.dat", ".rotmod.dat", ".dat"]:
            if galaxy.endswith(suffix):
                galaxy = galaxy[: -len(suffix)]
                break
        try:
            data = np.loadtxt(fpath, comments="#")
            if data.ndim == 1:
                data = data[np.newaxis, :]
            # SPARC Rotmod format:
            # R[kpc] Vobs[km/s] e_Vobs Vgas Vdisk Vbul SBdisk SBbul
            if data.shape[1] < 5:
                n_skip += 1
                continue
            R = data[:, 0]
            Vobs = data[:, 1]
            Vgas = data[:, 3]
            Vdisk = data[:, 4]
            Vbulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)

            # V_bar^2 = V_gas*|V_gas| + Upsd*V_disk^2 + Upsb*V_bul^2
            Vbar2 = Vgas * np.abs(Vgas) + UPSILON_DISK * Vdisk**2 + UPSILON_BULGE * Vbulge**2

            R_m = R * KPC_TO_M
            # R=0 回避
            safe = R_m > 0
            g_obs = np.where(safe, (Vobs * KM_TO_M) ** 2 / np.where(safe, R_m, 1.0), 0.0)
            g_bar = np.where(safe, Vbar2 * (KM_TO_M ** 2) / np.where(safe, R_m, 1.0), 0.0)

            mask = safe & (g_obs > 0) & (g_bar > 0)
            for i in np.where(mask)[0]:
                cloud_rows.append({
                    "galaxy": galaxy,
                    "R_kpc": R[i],
                    "Vobs_kms": Vobs[i],
                    "log_gobs_a0": float(np.log10(g_obs[i] / A0)),
                    "log_gbar_a0": float(np.log10(g_bar[i] / A0)),
                })
            n_ok += 1
        except Exception as e:
            n_skip += 1
            print(f"  Skip {galaxy}: {e}")

    cloud = pd.DataFrame(cloud_rows)
    cloud.to_csv("rar_sparc_cloud.csv", index=False)
    print(f"OUTPUT: rar_sparc_cloud.csv  ({len(cloud)} points from {n_ok} galaxies, {n_skip} skipped)")
    if len(cloud) > 0:
        print(f"  log_gobs_a0: range=[{cloud['log_gobs_a0'].min():.2f}, {cloud['log_gobs_a0'].max():.2f}]")
        print(f"  log_gbar_a0: range=[{cloud['log_gbar_a0'].min():.2f}, {cloud['log_gbar_a0'].max():.2f}]")
else:
    print(f"Rotmod_LTG directory not found.")
    print("If RAR cloud is needed, place SPARC Rotmod_LTG folder in current directory,")
    print("or edit rotmod_dir variable in the script.")

# =======================================================
# 完了
# =======================================================
print()
print("=" * 60)
print("Preprocessing complete")
print("=" * 60)
print("Upload to claude.ai:")
print("  1. rar_unified_galaxy_level.csv  (REQUIRED)")
cloud_path = "rar_sparc_cloud.csv"
if os.path.exists(cloud_path):
    print(f"  2. {cloud_path}  (RECOMMENDED for RAR background cloud)")
else:
    print(f"  2. {cloud_path}  (not generated — no Rotmod_LTG)")
