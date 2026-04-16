# -*- coding: utf-8 -*-
"""
§0 トートロジー分離: dSph gc_obs の真の物理的内容
  (0-a) gc_obs = g_obs^2/g_bar 定義構造の分解
  (0-b) gc_obs vs (sigma, r_h, M_bar) 個別・多変量回帰
  (0-c) g_obs 直接回帰 (真の Strigari テスト)
  (0-d) Mdyn^-1 (H4) の成分分解
  (0-e) 非トートロジー予測の同定

ASCII互換表記 (v4.2仕様)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fmng
from scipy.stats import pearsonr, spearmanr

# IPAGothic
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

# ========================================================
# 定数
# ========================================================
A0 = 1.2e-10         # m/s^2
G_NEWTON = 6.674e-11 # m^3/(kg s^2)
KPC_TO_M = 3.086e19
PC_TO_M = 3.086e16
KMS_TO_MS = 1000.0
M_SUN = 1.989e30     # kg

# ========================================================
# データ準備
# ========================================================
uni = pd.read_csv("/home/claude/rar_unified_galaxy_level.csv")
dsph = uni[uni.source == "dSph"].copy()

# 観測 scalar
dsph["g_obs_a0"] = 10**dsph["log_gobs_a0"]
dsph["g_bar_a0"] = 10**dsph["log_gbar_a0"]
dsph["g_obs_SI"] = dsph["g_obs_a0"] * A0
dsph["g_bar_SI"] = dsph["g_bar_a0"] * A0

# sigma [km/s], rh [pc]
sigma = dsph["sigma_kms"].values
rh_pc = dsph["rh_pc"].values
rh_m = rh_pc * PC_TO_M
sigma_ms = sigma * KMS_TO_MS

# M_bar を g_bar と r_h から逆算 (g_bar = G M_bar / r_h^2)
M_bar_kg = dsph["g_bar_SI"].values * rh_m**2 / G_NEWTON
M_bar_Msun = M_bar_kg / M_SUN
dsph["M_bar_Msun"] = M_bar_Msun
dsph["log_Mbar"] = np.log10(M_bar_Msun)
dsph["log_sigma"] = np.log10(sigma)
dsph["log_rh"] = np.log10(rh_pc)

# Walker 動力学質量 Mdyn = 2.5 sigma^2 r_h / G
Mdyn_kg = 2.5 * sigma_ms**2 * rh_m / G_NEWTON
Mdyn_Msun = Mdyn_kg / M_SUN
dsph["Mdyn_Msun"] = Mdyn_Msun
dsph["log_Mdyn"] = np.log10(Mdyn_Msun)

# gc_obs (deep MOND抽出)
dsph["gc_obs_a0"] = dsph["g_obs_a0"]**2 / dsph["g_bar_a0"]
dsph["log_gc"] = np.log10(dsph["gc_obs_a0"])

# 膜補間関数 正確解: gc = g_bar / [ln(1 - g_bar/g_obs)]^2
ratio = dsph["g_bar_a0"].values / dsph["g_obs_a0"].values
safe = (ratio < 1) & (ratio > 0)
ln_term = np.log(1 - np.clip(ratio, 0, 0.9999))
gc_exact = np.where(safe, dsph["g_bar_a0"].values / ln_term**2, np.nan)
dsph["gc_exact_a0"] = gc_exact
dsph["log_gc_exact"] = np.log10(gc_exact)

print(f"N = {len(dsph)} dSph 銀河")
print(f"sigma     range: [{sigma.min():.1f}, {sigma.max():.1f}] km/s")
print(f"r_h       range: [{rh_pc.min():.1f}, {rh_pc.max():.1f}] pc")
print(f"log M_bar range: [{dsph['log_Mbar'].min():.2f}, {dsph['log_Mbar'].max():.2f}] log Msun")
print(f"log M_dyn range: [{dsph['log_Mdyn'].min():.2f}, {dsph['log_Mdyn'].max():.2f}] log Msun")

# ========================================================
# (0-a) 定義的トートロジー構造
# ========================================================
print("\n" + "="*70)
print("(0-a) 定義的トートロジー構造の理論予測")
print("="*70)
print("""
dSph観測量の定義:
  g_obs = sigma^2 / r_h                [Wolf+2010]
  g_bar = G * M_bar / r_h^2            [Newton]
  gc_obs = g_obs^2/g_bar (deep MOND)

代入すると:
  gc_obs = (sigma^2/r_h)^2 / (G M_bar/r_h^2)
         = sigma^4 * r_h^2 / (r_h^2 * G M_bar)
         = sigma^4 / (G M_bar)      <-- r_h 依存が定義的に消失!

すなわち log gc_obs = 4 log sigma + 0 log r_h - 1 log M_bar + const

予測傾き:
  gc_obs vs sigma    : slope = +4    (完全トートロジー)
  gc_obs vs r_h      : slope =  0    (定義的中立 = 非トートロジー枠)
  gc_obs vs M_bar    : slope = -1    (完全トートロジー)
""")

# ========================================================
# (0-b) 多変量回帰で係数を測定
# ========================================================
print("="*70)
print("(0-b) 多変量回帰 log gc_obs ~ beta_s log sigma + beta_r log r_h + beta_m log M_bar")
print("="*70)

from numpy.linalg import lstsq

def multi_regression(y, X, names):
    """線形最小二乗 + 標準誤差"""
    X_full = np.column_stack([np.ones(len(y)), X])
    beta, res, rk, sv = lstsq(X_full, y, rcond=None)
    y_pred = X_full @ beta
    resid = y - y_pred
    # 標準誤差
    dof = len(y) - X_full.shape[1]
    sigma_hat2 = np.sum(resid**2) / dof
    cov = sigma_hat2 * np.linalg.inv(X_full.T @ X_full)
    se = np.sqrt(np.diag(cov))
    # R^2
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    # Print
    print(f"  intercept = {beta[0]:+.4f} +/- {se[0]:.4f}")
    for i, n in enumerate(names):
        print(f"  {n:15s} = {beta[i+1]:+.4f} +/- {se[i+1]:.4f}")
    print(f"  R^2 = {r2:.4f},  scatter = {resid.std():.3f} dex")
    return beta, se, resid, r2

X_full = np.column_stack([dsph["log_sigma"], dsph["log_rh"], dsph["log_Mbar"]])
print("\n[A] log gc_obs (deep MOND抽出) ~ log sigma + log r_h + log M_bar")
print("    理論予測 (完全トートロジー): (intercept_def, +4, 0, -1)")
beta_A, se_A, resid_A, r2_A = multi_regression(
    dsph["log_gc"].values, X_full, ["log sigma", "log r_h", "log M_bar"]
)

print("\n[B] log gc_exact (膜補間正確解) ~ log sigma + log r_h + log M_bar")
mask_exact = ~np.isnan(dsph["log_gc_exact"])
print(f"    N = {mask_exact.sum()} (NaN除く)")
beta_B, se_B, resid_B, r2_B = multi_regression(
    dsph.loc[mask_exact, "log_gc_exact"].values,
    X_full[mask_exact.values], ["log sigma", "log r_h", "log M_bar"]
)

# ========================================================
# (0-c) g_obs 直接回帰 (真の Strigari テスト)
# ========================================================
print("\n" + "="*70)
print("(0-c) g_obs 直接回帰: 真の Strigari 関係テスト")
print("="*70)
print("log g_obs = 2 log sigma - log r_h + 0 log M_bar + const (定義)")
print("予測傾き: (intercept, +2, -1, 0)")

print("\n[C] log g_obs ~ log sigma + log r_h + log M_bar")
beta_C, se_C, resid_C, r2_C = multi_regression(
    dsph["log_gobs_a0"].values, X_full, ["log sigma", "log r_h", "log M_bar"]
)

# g_obs 個別回帰
print("\n[D] log g_obs ~ log sigma のみ (単変量)")
X1 = np.column_stack([dsph["log_sigma"]])
beta_D, se_D, resid_D, r2_D = multi_regression(
    dsph["log_gobs_a0"].values, X1, ["log sigma"]
)

print("\n[E] log g_obs ~ log r_h のみ (単変量)")
X1 = np.column_stack([dsph["log_rh"]])
beta_E, se_E, resid_E, r2_E = multi_regression(
    dsph["log_gobs_a0"].values, X1, ["log r_h"]
)

print("\n[F] log g_obs ~ log M_bar のみ (単変量)")
X1 = np.column_stack([dsph["log_Mbar"]])
beta_F, se_F, resid_F, r2_F = multi_regression(
    dsph["log_gobs_a0"].values, X1, ["log M_bar"]
)

# g_obs vs 定数 (Strigari null テスト)
print("\n[G] log g_obs = const のみ (Strigari null 仮説)")
y = dsph["log_gobs_a0"].values
print(f"  mean = {y.mean():+.4f}  (=> g_obs = {10**y.mean():.3f} a0)")
print(f"  std  = {y.std():.4f} dex")
print(f"  g_obs SI = {10**y.mean() * A0:.3e} m/s^2")

# ========================================================
# (0-d) Mdyn^-1 decomposition
# ========================================================
print("\n" + "="*70)
print("(0-d) Mdyn^-1 (H4仮説) の σ, r_h 分解")
print("="*70)
print("Mdyn = 2.5 sigma^2 r_h / G より log Mdyn = 2 log sigma + log r_h + const")
print("\n[H] log gc_obs ~ log Mdyn の単回帰 (引継ぎメモの H4)")
X1 = np.column_stack([dsph["log_Mdyn"]])
beta_H, se_H, resid_H, r2_H = multi_regression(
    dsph["log_gc"].values, X1, ["log Mdyn"]
)

print("\n[I] log gc_obs ~ log sigma + log r_h (Mdynを分解)")
X2 = np.column_stack([dsph["log_sigma"], dsph["log_rh"]])
beta_I, se_I, resid_I, r2_I = multi_regression(
    dsph["log_gc"].values, X2, ["log sigma", "log r_h"]
)

# ========================================================
# (0-e) 候補D の真の予測性能 (g_obs const 仮説)
# ========================================================
print("\n" + "="*70)
print("(0-e) 候補D の真の予測内容")
print("="*70)
print("""
候補D: gc = G_fit^2 / g_bar  (G_fit = 0.240 a0)
    この「slope = -1 in log-log」は定義:
        gc = g_obs^2 / g_bar で g_obs = const  ならば自動的
    
    真の非トートロジー予測: g_obs = const = G_fit
""")

# Strigari null と g_obs vs 独立変数 の対比
print("[J] Strigari null (g_obs = const) vs g_obs 依存性検証:")
print(f"  null scatter:        {dsph['log_gobs_a0'].std():.3f} dex")
print(f"  vs sigma slope:      {beta_D[1]:+.3f} +/- {se_D[1]:.3f}  (null = 0)")
print(f"  vs r_h slope:        {beta_E[1]:+.3f} +/- {se_E[1]:.3f}  (null = 0)")
print(f"  vs M_bar slope:      {beta_F[1]:+.3f} +/- {se_F[1]:.3f}  (null = 0)")

# 有意性判定
def sig(b, s):
    z = b/s
    return f"{z:+.1f}σ", abs(z) > 2.0

for name, (b, s) in [("sigma", (beta_D[1], se_D[1])),
                      ("r_h", (beta_E[1], se_E[1])),
                      ("M_bar", (beta_F[1], se_F[1]))]:
    z_str, sig_bool = sig(b, s)
    marker = "*" if sig_bool else " "
    print(f"    {marker} g_obs vs {name:8s}: {z_str} {'(有意)' if sig_bool else '(null)'}")

# ========================================================
# 残差分析
# ========================================================
print("\n" + "="*70)
print("残差分析: トートロジー除去後の真の散布")
print("="*70)

# 完全トートロジー構造を差し引いた残差
# gc_obs_tauto = sigma^4 / (G M_bar)  (定義のみから期待される値)
gc_tauto_SI = sigma_ms**4 / (G_NEWTON * M_bar_kg)
gc_tauto_a0 = gc_tauto_SI / A0
dsph["gc_tautological_a0"] = gc_tauto_a0
dsph["log_gc_tauto"] = np.log10(gc_tauto_a0)
# 観測 - トートロジー = 純粋物理残差
dsph["physical_resid_dex"] = dsph["log_gc"] - dsph["log_gc_tauto"]
print(f"\n観測 gc - トートロジー予測 残差:")
print(f"  mean = {dsph['physical_resid_dex'].mean():+.3f} dex")
print(f"  std  = {dsph['physical_resid_dex'].std():.3f} dex")
print(f"  (完全ゼロなら100%トートロジー、有限なら物理成分あり)")

# Candidate D prediction
G_fit = 0.240
dsph["gc_D_pred"] = G_fit**2 / dsph["g_bar_a0"]
dsph["D_resid"] = dsph["log_gc"] - np.log10(dsph["gc_D_pred"])

# Bernoulli prediction  
T_m = np.sqrt(6)
s_0 = 1 / (1 + np.exp(1.5/T_m))
G_Bern = s_0 * (1 - s_0)
dsph["gc_Bern_pred"] = G_Bern**2 / dsph["g_bar_a0"]
dsph["Bern_resid"] = dsph["log_gc"] - np.log10(dsph["gc_Bern_pred"])

print(f"\n候補D (G_fit={G_fit}) 残差: mean={dsph['D_resid'].mean():+.3f}, std={dsph['D_resid'].std():.3f} dex")
print(f"Bernoulli (G={G_Bern:.3f}) 残差: mean={dsph['Bern_resid'].mean():+.3f}, std={dsph['Bern_resid'].std():.3f} dex")

# save
dsph.to_csv("/home/claude/tautology_analysis_dsph.csv", index=False)
print(f"\n[tautology_analysis_dsph.csv] saved")

# ========================================================
# プロット
# ========================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) gc_obs vs sigma (slope=4 予測)
ax = axes[0, 0]
ax.scatter(dsph["log_sigma"], dsph["log_gc"], s=40, c=COL_BLUE,
           alpha=0.75, edgecolor="black", linewidths=0.4)
# 理論線 slope=4
x_lin = np.linspace(dsph["log_sigma"].min()-0.1, dsph["log_sigma"].max()+0.1, 100)
intercept_fit = dsph["log_gc"].mean() - 4 * dsph["log_sigma"].mean()
ax.plot(x_lin, 4*x_lin + intercept_fit, "--", color=COL_RED, lw=1.2,
        label="slope=+4 (def tauto)")
# 実測フィット
from numpy.polynomial import polynomial as P
coef = np.polyfit(dsph["log_sigma"], dsph["log_gc"], 1)
ax.plot(x_lin, coef[0]*x_lin + coef[1], "-", color=COL_H, lw=1.2,
        label=f"fit slope={coef[0]:.2f}")
ax.set_xlabel("log sigma [km/s]")
ax.set_ylabel("log gc_obs [a0]")
ax.set_title("(a) gc_obs vs sigma\n予測 slope=+4 完全トートロジー", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) gc_obs vs r_h  (slope=0 ← 非トートロジー test!)
ax = axes[0, 1]
ax.scatter(dsph["log_rh"], dsph["log_gc"], s=40, c=COL_GREEN,
           alpha=0.75, edgecolor="black", linewidths=0.4)
x_lin = np.linspace(dsph["log_rh"].min()-0.1, dsph["log_rh"].max()+0.1, 100)
ax.axhline(dsph["log_gc"].mean(), color=COL_RED, ls="--", lw=1.2,
           label="slope=0 (def null)")
coef = np.polyfit(dsph["log_rh"], dsph["log_gc"], 1)
ax.plot(x_lin, coef[0]*x_lin + coef[1], "-", color=COL_H, lw=1.2,
        label=f"fit slope={coef[0]:.2f}")
ax.set_xlabel("log r_h [pc]")
ax.set_ylabel("log gc_obs [a0]")
ax.set_title("(b) gc_obs vs r_h\n★ 予測 slope=0 非トートロジーtest", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) gc_obs vs M_bar (slope=-1 予測)
ax = axes[0, 2]
ax.scatter(dsph["log_Mbar"], dsph["log_gc"], s=40, c=COL_ORANGE,
           alpha=0.75, edgecolor="black", linewidths=0.4)
x_lin = np.linspace(dsph["log_Mbar"].min()-0.1, dsph["log_Mbar"].max()+0.1, 100)
intercept_fit = dsph["log_gc"].mean() - (-1) * dsph["log_Mbar"].mean()
ax.plot(x_lin, -1*x_lin + intercept_fit, "--", color=COL_RED, lw=1.2,
        label="slope=-1 (def tauto)")
coef = np.polyfit(dsph["log_Mbar"], dsph["log_gc"], 1)
ax.plot(x_lin, coef[0]*x_lin + coef[1], "-", color=COL_H, lw=1.2,
        label=f"fit slope={coef[0]:.2f}")
ax.set_xlabel("log M_bar [Msun]")
ax.set_ylabel("log gc_obs [a0]")
ax.set_title("(c) gc_obs vs M_bar\n予測 slope=-1 完全トートロジー", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (d) g_obs vs sigma (Strigari null test)
ax = axes[1, 0]
ax.scatter(dsph["log_sigma"], dsph["log_gobs_a0"], s=40, c=COL_BLUE,
           alpha=0.75, edgecolor="black", linewidths=0.4)
x_lin = np.linspace(dsph["log_sigma"].min()-0.1, dsph["log_sigma"].max()+0.1, 100)
ax.axhline(dsph["log_gobs_a0"].mean(), color=COL_RED, ls="--", lw=1.2,
           label=f"Strigari: g_obs=const={10**dsph['log_gobs_a0'].mean():.3f}")
ax.plot(x_lin, beta_D[0] + beta_D[1]*x_lin, "-", color=COL_H, lw=1.2,
        label=f"fit slope={beta_D[1]:+.2f}+/-{se_D[1]:.2f}")
ax.set_xlabel("log sigma [km/s]")
ax.set_ylabel("log g_obs [a0]")
ax.set_title(f"(d) g_obs vs sigma\n非トートロジー: Strigari null test", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (e) g_obs vs r_h
ax = axes[1, 1]
ax.scatter(dsph["log_rh"], dsph["log_gobs_a0"], s=40, c=COL_GREEN,
           alpha=0.75, edgecolor="black", linewidths=0.4)
x_lin = np.linspace(dsph["log_rh"].min()-0.1, dsph["log_rh"].max()+0.1, 100)
ax.axhline(dsph["log_gobs_a0"].mean(), color=COL_RED, ls="--", lw=1.2,
           label=f"Strigari null")
ax.plot(x_lin, beta_E[0] + beta_E[1]*x_lin, "-", color=COL_H, lw=1.2,
        label=f"fit slope={beta_E[1]:+.2f}+/-{se_E[1]:.2f}")
ax.set_xlabel("log r_h [pc]")
ax.set_ylabel("log g_obs [a0]")
ax.set_title("(e) g_obs vs r_h\n非トートロジー: Strigari null test", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (f) g_obs vs M_bar
ax = axes[1, 2]
ax.scatter(dsph["log_Mbar"], dsph["log_gobs_a0"], s=40, c=COL_ORANGE,
           alpha=0.75, edgecolor="black", linewidths=0.4)
x_lin = np.linspace(dsph["log_Mbar"].min()-0.1, dsph["log_Mbar"].max()+0.1, 100)
ax.axhline(dsph["log_gobs_a0"].mean(), color=COL_RED, ls="--", lw=1.2,
           label="Strigari null")
ax.plot(x_lin, beta_F[0] + beta_F[1]*x_lin, "-", color=COL_H, lw=1.2,
        label=f"fit slope={beta_F[1]:+.2f}+/-{se_F[1]:.2f}")
ax.set_xlabel("log M_bar [Msun]")
ax.set_ylabel("log g_obs [a0]")
ax.set_title("(f) g_obs vs M_bar\n非トートロジー: Strigari null test", color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("§0 トートロジー分離: dSph gc_obs の真の物理的内容",
             fontsize=14, color=COL_H, fontweight="bold", y=1.01)
plt.tight_layout()

fig.savefig("/mnt/user-data/outputs/fig_tautology_analysis_v1.pdf",
            bbox_inches="tight", dpi=200)
fig.savefig("/mnt/user-data/outputs/fig_tautology_analysis_v1.png",
            bbox_inches="tight", dpi=170)
plt.close(fig)
print("\n[fig_tautology_analysis_v1.pdf/png] saved")

# ========================================================
# 結果まとめ JSONish
# ========================================================
results = {
    "N_dsph": len(dsph),
    "gc_multivariate": {
        "intercept": beta_A[0], "log_sigma": beta_A[1],
        "log_rh": beta_A[2], "log_Mbar": beta_A[3],
        "R2": r2_A, "scatter_dex": resid_A.std(),
    },
    "gobs_multivariate": {
        "intercept": beta_C[0], "log_sigma": beta_C[1],
        "log_rh": beta_C[2], "log_Mbar": beta_C[3],
        "R2": r2_C, "scatter_dex": resid_C.std(),
    },
    "gobs_univariate": {
        "vs_sigma_slope": beta_D[1], "vs_sigma_se": se_D[1], "vs_sigma_R2": r2_D,
        "vs_rh_slope": beta_E[1], "vs_rh_se": se_E[1], "vs_rh_R2": r2_E,
        "vs_Mbar_slope": beta_F[1], "vs_Mbar_se": se_F[1], "vs_Mbar_R2": r2_F,
    },
    "strigari_null": {
        "log_gobs_mean": dsph["log_gobs_a0"].mean(),
        "log_gobs_std": dsph["log_gobs_a0"].std(),
        "g_obs_const_a0": 10**dsph["log_gobs_a0"].mean(),
        "g_obs_const_SI": 10**dsph["log_gobs_a0"].mean() * A0,
    },
    "candidate_D_vs_Bernoulli": {
        "G_fit_a0": G_fit,
        "G_Bernoulli_a0": G_Bern,
        "D_resid_mean": dsph["D_resid"].mean(),
        "D_resid_std": dsph["D_resid"].std(),
        "Bern_resid_mean": dsph["Bern_resid"].mean(),
        "Bern_resid_std": dsph["Bern_resid"].std(),
    },
}
import json
with open("/home/claude/tautology_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("[tautology_results.json] saved")

print("\n" + "="*70)
print("核心判定")
print("="*70)

# g_obs が各独立変数に対して有意に依存するか
z_sigma = abs(beta_D[1]/se_D[1])
z_rh = abs(beta_E[1]/se_E[1])
z_Mbar = abs(beta_F[1]/se_F[1])

sigma_sig = z_sigma > 2
rh_sig = z_rh > 2
Mbar_sig = z_Mbar > 2

print(f"g_obs の Strigari null (const) からの有意偏差:")
print(f"  vs sigma : |z|={z_sigma:.2f} {'*有意*' if sigma_sig else '(null成立)'}")
print(f"  vs r_h   : |z|={z_rh:.2f} {'*有意*' if rh_sig else '(null成立)'}")
print(f"  vs M_bar : |z|={z_Mbar:.2f} {'*有意*' if Mbar_sig else '(null成立)'}")

if not (sigma_sig or rh_sig or Mbar_sig):
    print("\n=> g_obs は sigma, r_h, M_bar に対し統計的に一定 (Strigari成立)")
    print("=> Bernoulli予測 g_obs = 0.228 a0 は真に非トートロジーな予測")
else:
    print("\n=> g_obs に有意な従属変数依存あり (純粋 Strigari からの逸脱)")
    print("=> Strigari関係は近似のみ、膜理論の精緻化余地")
