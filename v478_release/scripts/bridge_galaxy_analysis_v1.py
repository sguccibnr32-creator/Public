# -*- coding: utf-8 -*-
"""
§0.ii ブリッジ銀河検証
  ESO444-G084, NGC2915, NGC1705, NGC3741 の g_obs(R), gc(R) プロファイル
  C15 -> Strigari 連続遷移テスト
  
ASCII互換表記
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fmng
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr, linregress

fmng.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
fmng.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
plt.rcParams["font.family"] = "IPAGothic"
plt.rcParams["axes.unicode_minus"] = False

COL_H = "#1a1a2e"
COL_S = "#16213e"
COL_RED = "#e94560"
COL_BLUE = "#4a90b8"
COL_GREEN = "#2a9d8f"
COL_ORANGE = "#f4a261"
COL_PURPLE = "#9b5de5"

# 定数
A0 = 1.2e-10
T_M = np.sqrt(6.0)
S0 = 1/(1 + np.exp(1.5/T_M))  # = 0.3515
G_BERN = S0 * (1 - S0)        # = 0.228
G_STRIG_A0 = G_BERN           # Bernoulli 予測 g_obs asymptote
GC_BERN = G_BERN**2           # dSph gc coefficient (= 0.052 a0^2 units)

BRIDGE = ["ESO444-G084", "NGC2915", "NGC1705", "NGC3741"]

# ========================================================
# データ
# ========================================================
cloud = pd.read_csv("/home/claude/rar_sparc_cloud.csv")
uni = pd.read_csv("/home/claude/rar_unified_galaxy_level.csv")

bridge = cloud[cloud.galaxy.isin(BRIDGE)].copy().sort_values(["galaxy", "R_kpc"])
bridge["g_obs_a0"] = 10**bridge["log_gobs_a0"]
bridge["g_bar_a0"] = 10**bridge["log_gbar_a0"]
bridge["upsilon_dyn"] = bridge["g_obs_a0"] / bridge["g_bar_a0"]
bridge["log_upsilon"] = np.log10(bridge["upsilon_dyn"])

# 2種類の gc 抽出
ratio = bridge["g_bar_a0"].values / bridge["g_obs_a0"].values
safe = (ratio > 0) & (ratio < 1)
with np.errstate(divide="ignore", invalid="ignore"):
    gc_exact = np.where(safe,
        bridge["g_bar_a0"].values / np.log(1 - np.clip(ratio, 0, 0.9999))**2,
        np.nan)
bridge["gc_exact_a0"] = gc_exact
bridge["gc_dmond_a0"] = bridge["g_obs_a0"]**2 / bridge["g_bar_a0"]
bridge["log_gc_exact"] = np.log10(gc_exact)
bridge["log_gc_dmond"] = np.log10(bridge["gc_dmond_a0"])

# dSph
dsph = uni[uni.source == "dSph"].copy()
dsph["g_obs_a0"] = 10**dsph["log_gobs_a0"]
dsph["g_bar_a0"] = 10**dsph["log_gbar_a0"]
dsph["gc_dmond_a0"] = dsph["g_obs_a0"]**2 / dsph["g_bar_a0"]

# ========================================================
# 1. 銀河別 プロファイル解析
# ========================================================
print("="*75)
print("銀河別 プロファイル解析")
print("="*75)

summary_rows = []
for gname in BRIDGE:
    g = bridge[bridge.galaxy == gname].sort_values("R_kpc").reset_index(drop=True)
    n = len(g)
    # Inner/Outer 分割 (外側 40%)
    n_inner = int(n * 0.6)
    inner = g.iloc[:n_inner]
    outer = g.iloc[n_inner:]
    
    # log10(R) vs log10(Upsilon) 回帰 (遷移の量化)
    if n >= 4:
        slope, intercept, r_value, p_value, stderr = linregress(
            np.log10(g["R_kpc"]), g["log_upsilon"]
        )
    else:
        slope, intercept, r_value, p_value, stderr = np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Inner/Outer 比較
    if len(outer) >= 2 and len(inner) >= 2:
        try:
            u_stat, p_mw = mannwhitneyu(outer["log_upsilon"], inner["log_upsilon"],
                                          alternative="greater")
        except Exception:
            u_stat, p_mw = np.nan, np.nan
    else:
        u_stat, p_mw = np.nan, np.nan
    
    row = {
        "galaxy": gname,
        "N_points": n,
        "R_min_kpc": g["R_kpc"].min(),
        "R_max_kpc": g["R_kpc"].max(),
        "inner_Ups_med": 10**inner["log_upsilon"].median(),
        "outer_Ups_med": 10**outer["log_upsilon"].median(),
        "ratio_outer_inner": 10**(outer["log_upsilon"].median() -
                                   inner["log_upsilon"].median()),
        "slope_logU_logR": slope,
        "slope_se": stderr,
        "slope_sig_z": slope/stderr if not np.isnan(stderr) else np.nan,
        "mannwhitney_p_outer_greater": p_mw,
        "outer_g_obs_med_a0": 10**outer["log_gobs_a0"].median(),
        "outer_gc_dmond_med_a0": outer["gc_dmond_a0"].median(),
    }
    summary_rows.append(row)
    
    print(f"\n--- {gname} (N={n}) ---")
    print(f"  R range:         {g['R_kpc'].min():.2f} - {g['R_kpc'].max():.2f} kpc")
    print(f"  Υ_dyn inner med: {row['inner_Ups_med']:.2f}")
    print(f"  Υ_dyn outer med: {row['outer_Ups_med']:.2f}")
    print(f"  outer/inner:     {row['ratio_outer_inner']:.2f}x")
    print(f"  slope(logU~logR): {slope:+.3f} +/- {stderr:.3f}  (z={row['slope_sig_z']:+.2f})")
    print(f"  MW test (outer>inner): p = {p_mw:.4f}")
    print(f"  outer g_obs med: {row['outer_g_obs_med_a0']:.3f} a0 (Bernoulli予測: 0.228 a0)")
    print(f"  outer gc  med:   {row['outer_gc_dmond_med_a0']:.2f} a0")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("/home/claude/bridge_profile_summary.csv", index=False)
print("\n[bridge_profile_summary.csv] saved")

# ========================================================
# 2. 統合: ブリッジ銀河 外側 vs dSph 比較
# ========================================================
print("\n" + "="*75)
print("ブリッジ銀河 外側 全点 vs dSph 全銀河 分布比較")
print("="*75)

# 外側40%の全点を集める
bridge_outer_all = []
for gname in BRIDGE:
    g = bridge[bridge.galaxy == gname].sort_values("R_kpc").reset_index(drop=True)
    n_inner = int(len(g) * 0.6)
    bridge_outer_all.append(g.iloc[n_inner:])
bridge_outer_all = pd.concat(bridge_outer_all, ignore_index=True)

# Upsilon 分布比較
print(f"\n外側点 (N={len(bridge_outer_all)}): "
      f"log_Upsilon median = {bridge_outer_all['log_upsilon'].median():.2f} "
      f"(Υ = {10**bridge_outer_all['log_upsilon'].median():.2f})")
print(f"dSph (N={len(dsph)}): "
      f"log_Upsilon median = {np.log10(dsph['g_obs_a0']/dsph['g_bar_a0']).median():.2f} "
      f"(Υ = {(dsph['g_obs_a0']/dsph['g_bar_a0']).median():.2f})")

ks_stat, ks_p = ks_2samp(bridge_outer_all["log_upsilon"],
                          np.log10(dsph["g_obs_a0"] / dsph["g_bar_a0"]))
print(f"\nKS test (ブリッジ外側 vs dSph): statistic={ks_stat:.3f}, p={ks_p:.4f}")

# g_obs 比較
outer_gobs_med = 10**bridge_outer_all["log_gobs_a0"].median()
outer_gobs_std = bridge_outer_all["log_gobs_a0"].std()
print(f"\n外側 g_obs: median = {outer_gobs_med:.3f} a0, std = {outer_gobs_std:.3f} dex")
print(f"Bernoulli 予測: 0.228 a0")
print(f"dSph 観測: median = {dsph['g_obs_a0'].median():.3f} a0")

# ========================================================
# 3. プロット: 4銀河 プロファイル
# ========================================================
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for i, gname in enumerate(BRIDGE):
    g = bridge[bridge.galaxy == gname].sort_values("R_kpc").reset_index(drop=True)
    n_inner = int(len(g) * 0.6)
    inner = g.iloc[:n_inner]
    outer = g.iloc[n_inner:]
    
    # (1) g_obs(R) と g_bar(R)
    ax = axes[i, 0]
    ax.plot(g["R_kpc"], g["g_obs_a0"], "o-", color=COL_BLUE, ms=5, lw=1,
            label="g_obs")
    ax.plot(g["R_kpc"], g["g_bar_a0"], "s--", color=COL_ORANGE, ms=4, lw=1,
            label="g_bar")
    ax.axhline(G_BERN, color=COL_RED, ls=":", lw=1.2,
               label=f"Bernoulli 0.228 a0")
    ax.set_yscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("g / a0")
    ax.set_title(f"{gname}: g_obs, g_bar プロファイル", color=COL_H, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")
    
    # (2) Upsilon_dyn(R) profile
    ax = axes[i, 1]
    # 閾値帯
    ax.axhspan(10, 30, alpha=0.15, color=COL_ORANGE, label="J3遷移帯 (10-30)")
    ax.axhline(10, color="#888", ls="--", lw=0.8)
    ax.axhline(30, color="#888", ls="--", lw=0.8)
    ax.axhline(3, color="#ccc", ls=":", lw=0.8)
    # データ
    ax.plot(g["R_kpc"], g["upsilon_dyn"], "o-", color=COL_H, ms=5, lw=1.2,
            label="Υ_dyn(R)")
    ax.plot(inner["R_kpc"], inner["upsilon_dyn"], "o", color=COL_BLUE, ms=7,
            label=f"inner (R<{inner['R_kpc'].max():.2f})")
    ax.plot(outer["R_kpc"], outer["upsilon_dyn"], "o", color=COL_RED, ms=7,
            label=f"outer (R>{inner['R_kpc'].max():.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("Υ_dyn")
    ax.set_title(f"{gname}: Υ_dyn(R) 遷移プロファイル", color=COL_H, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")
    
    # (3) gc(R) profile with transition zones
    ax = axes[i, 2]
    # C15 zone (SPARC median gc ~ 1.2 a0)
    ax.axhspan(0.3, 3, alpha=0.15, color=COL_GREEN, label="C15 (SPARC)")
    # dSph Strigari zone (gc ~ 3-20 a0)
    ax.axhspan(3, 30, alpha=0.15, color=COL_RED, label="dSph Strigari")
    # データ (exact)
    ax.plot(g["R_kpc"], g["gc_exact_a0"], "o-", color=COL_H, ms=5, lw=1.2,
            label="gc_exact(R)")
    ax.plot(g["R_kpc"], g["gc_dmond_a0"], "^--", color=COL_PURPLE, ms=4, lw=0.8,
            alpha=0.7, label="gc_dMOND(R)")
    ax.set_yscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("gc / a0")
    ax.set_title(f"{gname}: gc(R) プロファイル", color=COL_H, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")

plt.suptitle("§0.ii ブリッジ銀河 半径プロファイル解析: C15 -> Strigari 遷移",
             fontsize=14, color=COL_H, fontweight="bold", y=0.995)
plt.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_bridge_profiles_v1.pdf",
            bbox_inches="tight", dpi=170)
fig.savefig("/mnt/user-data/outputs/fig_bridge_profiles_v1.png",
            bbox_inches="tight", dpi=150)
plt.close(fig)
print("\n[fig_bridge_profiles_v1.pdf/png] saved")

# ========================================================
# 4. 統合プロット: RAR + Υ分布
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# (A) 統合RAR: bridge outer + SPARC cloud + dSph
ax = axes[0]
ax.scatter(cloud["log_gbar_a0"], cloud["log_gobs_a0"],
           s=3, c=COL_BLUE, alpha=0.1, label=f"SPARC cloud (N={len(cloud)})",
           rasterized=True)
# bridge 全点 (inner + outer)
for gname, col in zip(BRIDGE, [COL_GREEN, COL_PURPLE, COL_ORANGE, COL_H]):
    g = bridge[bridge.galaxy == gname].sort_values("R_kpc").reset_index(drop=True)
    n_inner = int(len(g) * 0.6)
    # Inner as small, outer as big highlighted
    ax.scatter(g.iloc[:n_inner]["log_gbar_a0"], g.iloc[:n_inner]["log_gobs_a0"],
               s=18, c=col, alpha=0.5, marker="o", edgecolors="black",
               linewidths=0.3, label=None)
    ax.scatter(g.iloc[n_inner:]["log_gbar_a0"], g.iloc[n_inner:]["log_gobs_a0"],
               s=55, c=col, marker="*", edgecolors="black", linewidths=0.5,
               label=f"{gname} 外側")
# dSph
ax.scatter(dsph["log_gbar_a0"], dsph["log_gobs_a0"],
           s=40, c=COL_RED, marker="o", alpha=0.65,
           edgecolors="black", linewidths=0.4,
           label=f"dSph (N={len(dsph)})", zorder=4)
# 理論線
x = np.linspace(-4.5, 3, 200)
ax.plot(x, x, "k:", lw=0.8, alpha=0.5, label="Newton 1:1")
# Bernoulli deep MOND line: g_obs = sqrt(g_bar * a0 * coeff?)
# Actually g_obs = const = G_Bern for Strigari
ax.axhline(np.log10(G_BERN), color=COL_RED, ls="-.", lw=1.2,
           label=f"g_obs=G_Bern=0.228 a0")
# MOND simple IF
y_mond = np.log10(10**x * 0.5 * (1 + np.sqrt(1 + 4.0/10**x)))
ax.plot(x, y_mond, "--", color="#888", lw=1, label="MOND simple IF")

ax.set_xlabel(r"$\log_{10}(g_{\mathrm{bar}}/a_0)$")
ax.set_ylabel(r"$\log_{10}(g_{\mathrm{obs}}/a_0)$")
ax.set_title("(A) 統合 RAR: ブリッジ銀河 (外側★) と dSph (赤丸)",
             color=COL_H, fontsize=11)
ax.legend(fontsize=7, loc="lower right", ncol=1)
ax.grid(alpha=0.3)
ax.set_xlim(-4.5, 3)
ax.set_ylim(-2.5, 3.5)

# (B) Upsilon 分布比較
ax = axes[1]
# SPARC cloud (全点)
cloud_ups = cloud["log_gobs_a0"] - cloud["log_gbar_a0"]
ax.hist(cloud_ups, bins=40, density=True, alpha=0.4, color=COL_BLUE,
        label=f"SPARC全点 (N={len(cloud)})")
# Bridge outer
ax.hist(bridge_outer_all["log_upsilon"], bins=15, density=True, alpha=0.7,
        color=COL_ORANGE, label=f"ブリッジ外側 (N={len(bridge_outer_all)})")
# dSph
dsph_ups = np.log10(dsph["g_obs_a0"] / dsph["g_bar_a0"])
ax.hist(dsph_ups, bins=15, density=True, alpha=0.7, color=COL_RED,
        label=f"dSph 銀河 (N={len(dsph)})")
# 閾値
for th, lab in [(3, "Υ=3"), (10, "Υ=10"), (30, "Υ=30")]:
    ax.axvline(np.log10(th), color="#555", ls="--", lw=0.8)
    ax.text(np.log10(th)+0.05, ax.get_ylim()[1] if False else 1.0,
            lab, fontsize=8, color="#555",
            transform=ax.get_xaxis_transform(), va="bottom")
ax.axvspan(np.log10(10), np.log10(30), alpha=0.12, color=COL_ORANGE)
ax.set_xlabel(r"$\log_{10}(\Upsilon_{\mathrm{dyn}})$")
ax.set_ylabel("正規化頻度")
ax.set_title("(B) Υ_dyn 分布: SPARC全点 vs ブリッジ外側 vs dSph",
             color=COL_H, fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

plt.suptitle("§0.ii ブリッジ銀河の集合的位置: SPARC -> dSph 連続遷移",
             fontsize=13, color=COL_H, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_bridge_integration_v1.pdf",
            bbox_inches="tight", dpi=200)
fig.savefig("/mnt/user-data/outputs/fig_bridge_integration_v1.png",
            bbox_inches="tight", dpi=170)
plt.close(fig)
print("[fig_bridge_integration_v1.pdf/png] saved")

# 出力CSV
bridge.to_csv("/home/claude/bridge_all_points.csv", index=False)
bridge_outer_all.to_csv("/home/claude/bridge_outer_points.csv", index=False)

# ========================================================
# 5. 連続遷移 A級判定
# ========================================================
print("\n" + "="*75)
print("A級 連続遷移エビデンス 判定")
print("="*75)

# 判定基準
# (a) 各銀河で Υ_dyn が R と共に増加 (slope > 0, 有意)
# (b) 外側 Υ が inner より有意に大 (Mann-Whitney p < 0.05)
# (c) 外側 Υ 分布が SPARC 典型 (~ 2-3) と dSph (~ 35) の中間
# (d) 外側 g_obs が Bernoulli 予測 0.228 a0 付近に収束

n_galaxies_with_positive_slope = (summary_df["slope_logU_logR"] > 0).sum()
n_galaxies_sig_slope = ((summary_df["slope_sig_z"].abs() > 2) &
                         (summary_df["slope_logU_logR"] > 0)).sum()
n_galaxies_mw_sig = (summary_df["mannwhitney_p_outer_greater"] < 0.05).sum()

print(f"\n(a) Υ_dyn が R と共に増加する銀河数:")
print(f"    slope > 0:      {n_galaxies_with_positive_slope} / 4")
print(f"    slope > 0 有意: {n_galaxies_sig_slope} / 4")
print(f"(b) 外側 > 内側 有意 (MW p<0.05): {n_galaxies_mw_sig} / 4")
print(f"\n(c) 外側 Υ_dyn 中央値分布:")
print(f"    ブリッジ外側  : {10**bridge_outer_all['log_upsilon'].median():.2f}")
print(f"    SPARC 全点典型: 2-3")
print(f"    dSph 全銀河    : 35.1")
print(f"(d) 外側 g_obs 中央値: {outer_gobs_med:.3f} a0 "
      f"(Bernoulli予測: 0.228 a0, 一致度: {outer_gobs_med/G_BERN:.2f})")

# まとめ
verdict_rows = []
verdict_rows.append(["(a) R-内での Υ 増加 (有意 slope > 0)", 
                     f"{n_galaxies_sig_slope}/4", 
                     "A級" if n_galaxies_sig_slope >= 3 else ("B級" if n_galaxies_sig_slope >= 2 else "C級")])
verdict_rows.append(["(b) inner < outer (MW test)",
                     f"{n_galaxies_mw_sig}/4",
                     "A級" if n_galaxies_mw_sig >= 3 else ("B級" if n_galaxies_mw_sig >= 2 else "C級")])
med_ratio = 10**bridge_outer_all['log_upsilon'].median()
verdict_c = "A級" if 5 < med_ratio < 20 else ("B級" if 3 < med_ratio < 30 else "C級")
verdict_rows.append(["(c) 外側 Υ が中間域 (5-20)",
                     f"{med_ratio:.2f}", verdict_c])
ratio_gobs = outer_gobs_med/G_BERN
verdict_d = "A級" if 0.7 < ratio_gobs < 1.5 else ("B級" if 0.5 < ratio_gobs < 2 else "C級")
verdict_rows.append(["(d) 外側 g_obs / Bernoulli予測",
                     f"{ratio_gobs:.2f}", verdict_d])

print("\n総合判定:")
for r in verdict_rows:
    print(f"  {r[0]:45s} {r[1]:<10s} -> {r[2]}")

# 保存
pd.DataFrame(verdict_rows, columns=["criterion","value","grade"]).to_csv(
    "/home/claude/bridge_verdict.csv", index=False)
print("\n[bridge_verdict.csv] saved")
