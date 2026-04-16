# -*- coding: utf-8 -*-
"""
§9.1 dSph gc公式 再定式化: 理論解析
  (1) U(epsilon; c) 平衡解析
  (2) c->0 極限: 固有膜状態
  (3) SCE Q->0 極限: s_0, alpha_s
  (4) gc_dSph 候補式 4通り
  (5) 観測データ (dsph) との比較

ASCII互換表記 (v4.2仕様書準拠 + 拡張)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fmng
from scipy.optimize import brentq

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
# 理論定数 (memo v4.7.7)
# ========================================================
T_M = np.sqrt(6.0)    # 膜温度
A_SCE = 1.263
BETA = -0.331

# ========================================================
# Part 1: U(epsilon; c) 自由エネルギー
# ========================================================
def U_free(eps, c):
    """U(epsilon; c) = -eps - eps^2/2 - c*ln(1-eps)"""
    eps = np.asarray(eps, dtype=float)
    out = -eps - 0.5 * eps**2
    # c*ln(1-eps) term (eps=1 特異)
    mask = (1 - eps) > 1e-15
    out = np.where(mask, out - c * np.log(np.clip(1 - eps, 1e-15, None)), -np.inf)
    return out

def eps_eq_analytic(c):
    """dU/d(eps) = 0 の解: eps_eq = sqrt(1-c), c in [0, 1]"""
    c = np.asarray(c, dtype=float)
    out = np.where(c <= 1.0, np.sqrt(np.clip(1 - c, 0, None)), np.nan)
    return out

def U_eq(c):
    """U_eq(c) = U(eps_eq(c); c)"""
    eps = eps_eq_analytic(c)
    # c=0 では eps=1 で ln 項が 0*(-inf); 極限値は 0
    return -eps - 0.5 * eps**2 - c * np.where(1 - eps > 0,
                                               np.log(np.clip(1 - eps, 1e-15, 1)),
                                               0.0)

def dU_barrier_memo(c):
    """memo記載の Delta-U(c) = -3/2 + 2 sqrt(c) - c/2 - (c/2) ln(c)
       (独立定義、U_eq とは異なる)"""
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 1e-15, None)
    return -1.5 + 2.0 * np.sqrt(c) - 0.5 * c - 0.5 * c * np.log(c)

# ========================================================
# Part 2: SCE 数値解 s(Q; T_m)
# ========================================================
def sce_residual(s, Q, T_m, dU_func):
    """s - 1/(1 + exp(-dU(s*Q)/T_m)) = 0"""
    c_eff = s * Q
    dU = dU_func(c_eff)
    # オーバーフロー保護
    arg = -dU / T_m
    arg = np.clip(arg, -50, 50)
    return s - 1.0 / (1.0 + np.exp(arg))

def solve_s(Q, T_m, dU_func, s0=0.5):
    """SCE を固定点反復で解く"""
    s = s0
    for _ in range(500):
        c_eff = s * Q
        dU = dU_func(c_eff)
        arg = -dU / T_m
        arg = np.clip(arg, -50, 50)
        s_new = 1.0 / (1.0 + np.exp(arg))
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    return s

# ========================================================
# Part 3: 数値検証
# ========================================================
print("=" * 60)
print("Part 3: c->0 極限の解析値")
print("=" * 60)

# s_0 at Q=0
s0_memo = 1.0 / (1.0 + np.exp(1.5 / T_M))
print(f"s_0 = 1/(1 + exp(3/(2*T_m)))")
print(f"     T_m = sqrt(6) = {T_M:.6f}")
print(f"     exp(3/(2*T_m)) = {np.exp(1.5/T_M):.6f}")
print(f"     s_0 = {s0_memo:.6f}")

# 小さいQでの漸近
E = np.exp(1.5 / T_M)
F = 1 + E
alpha_s = (2 * E / (F * T_M)) * s0_memo**1.5
print(f"\n小Q展開: s(Q) ~ s_0 + alpha_s * sqrt(Q)")
print(f"  alpha_s = (2E / (F*T_m)) * s_0^(3/2) = {alpha_s:.6f}")
print(f"  E = exp(3/(2*T_m)) = {E:.4f}, F = 1+E = {F:.4f}")

# 数値SCE検証
print(f"\n数値SCE 検証:")
for Q in [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]:
    s_num = solve_s(Q, T_M, dU_barrier_memo, s0=s0_memo)
    s_asymp = s0_memo + alpha_s * np.sqrt(Q)
    print(f"  Q={Q:6.3f}: s_numerical={s_num:.6f}, s_asymptotic={s_asymp:.6f}, "
          f"relerr={abs(s_num-s_asymp)/s_num*100:.2f}%")

# ========================================================
# Part 4: gc_dSph 候補式
# ========================================================
print("\n" + "=" * 60)
print("Part 4: gc_dSph 候補式")
print("=" * 60)

# dSphデータ読込
uni = pd.read_csv("/home/claude/rar_unified_galaxy_level.csv")
dsph = uni[uni.source == "dSph"].copy()
dsph["g_obs_a0"] = 10**dsph["log_gobs_a0"]
dsph["g_bar_a0"] = 10**dsph["log_gbar_a0"]

# 観測 gc は deep-MOND 抽出 (cloud consistent)
# gc = g_obs^2 / g_bar
dsph["gc_obs_a0"] = dsph["g_obs_a0"]**2 / dsph["g_bar_a0"]
dsph["log_gc_obs"] = np.log10(dsph["gc_obs_a0"])

print(f"\n観測 dSph gc 統計 (N={len(dsph)}):")
print(f"  log10(gc/a0): median={dsph['log_gc_obs'].median():.3f}, "
      f"mean={dsph['log_gc_obs'].mean():.3f}, "
      f"std={dsph['log_gc_obs'].std():.3f}")
print(f"  gc/a0:        median={10**dsph['log_gc_obs'].median():.2f}, "
      f"range=[{dsph['gc_obs_a0'].min():.2f}, {dsph['gc_obs_a0'].max():.2f}]")

# 候補式 (全て gc を a0単位で出力)
# ---------------------------------------
# 候補A: 線形写像 gc = s_0 * a0 (定数)
gc_A = s0_memo  # [a0]

# 候補B: Fermi型 gc = (1-s_0)/s_0 * a0 = exp(3/(2T_m)) * a0
gc_B = (1 - s0_memo) / s0_memo

# 候補C: Strigari型 gc = G_Strigari^2 / g_bar (g_bar依存)
#   G_Strigari を s_0 * a0 と仮定
G_C = s0_memo  # [a0]
dsph["gc_C"] = G_C**2 / dsph["g_bar_a0"]

# 候補D: Strigari型 gc = G^2 / g_bar で G をフィット
#   最小二乗で G をフィット (log space)
from scipy.optimize import minimize_scalar
def loss_G(G):
    pred = np.log10(G**2 / dsph["g_bar_a0"].values)
    return np.sum((pred - dsph["log_gc_obs"].values)**2)
res = minimize_scalar(loss_G, bounds=(0.01, 2.0), method='bounded')
G_best = res.x
dsph["gc_D"] = G_best**2 / dsph["g_bar_a0"]

# 候補E: SCE 自己無撞着 gc = A * s(Q) * Q * Yd^beta (Yd=1 仮定)
#   Q = sqrt(g_obs/a0), s = s(Q; T_m)
def sce_gc(g_obs_a0, T_m=T_M, A=A_SCE, Yd=1.0, beta=BETA):
    Q = np.sqrt(np.asarray(g_obs_a0))
    s_vals = np.array([solve_s(q, T_m, dU_barrier_memo, s0=s0_memo) for q in np.atleast_1d(Q)])
    return A * s_vals * Q * (Yd**beta)

dsph["gc_E"] = sce_gc(dsph["g_obs_a0"].values)

# 残差統計
def residual_stats(label, pred_col_or_val):
    if isinstance(pred_col_or_val, (int, float)):
        pred = np.full(len(dsph), np.log10(pred_col_or_val))
    else:
        pred = np.log10(dsph[pred_col_or_val].values)
    resid = pred - dsph["log_gc_obs"].values
    print(f"  {label}: bias={resid.mean():+.3f} dex, "
          f"scatter={resid.std():.3f} dex")
    return resid

print(f"\n残差 (log10 dex, 観測 - 予測):")
print(f"  Predicted median observed = 10^{dsph['log_gc_obs'].median():.3f} = "
      f"{10**dsph['log_gc_obs'].median():.2f} a0")
print(f"\n  候補A  gc = s_0 * a0                = {gc_A:.3f} a0")
_ = residual_stats("   A", gc_A)
print(f"  候補B  gc = (1-s_0)/s_0 * a0        = {gc_B:.3f} a0")
_ = residual_stats("   B", gc_B)
print(f"  候補C  gc = (s_0*a0)^2 / g_bar  (Strigari型, G fixed)")
_ = residual_stats("   C", "gc_C")
print(f"  候補D  gc = G_fit^2 / g_bar     (G_fit = {G_best:.3f} a0)")
_ = residual_stats("   D", "gc_D")
print(f"  候補E  gc = A*s(Q)*Q*Yd^beta    (SCE SPARC式そのまま)")
_ = residual_stats("   E", "gc_E")

# 観測中央値マッチの係数を求める (候補AB用)
obs_med = 10**dsph["log_gc_obs"].median()
print(f"\n観測中央値 {obs_med:.2f} a0 に必要な無次元係数:")
print(f"  gc = alpha * a0 で alpha = {obs_med:.3f}")
print(f"  s_0 との比: alpha / s_0 = {obs_med/s0_memo:.2f}")
print(f"  s_0 * exp(T_m/T_m) との比較: {s0_memo * np.exp(T_M):.2f} a0 "
      f"(ratio: {obs_med / (s0_memo * np.exp(T_M)):.3f})")

# Strigari型 (候補D) の g_obs 予測
# g_obs = sqrt(gc * g_bar * a0 correction) の deep MOND で
# g_obs = sqrt(G_fit^2 / g_bar * g_bar) = G_fit (一定!)
print(f"\n候補D (Strigari型) の g_obs 予測: g_obs = G_fit = {G_best:.3f} a0")
print(f"  観測 g_obs median = {10**dsph['log_gobs_a0'].median():.3f} a0")
print(f"  両者の比: {G_best / 10**dsph['log_gobs_a0'].median():.3f}")

# CSV保存
dsph_out = dsph[["galaxy", "host", "log_gbar_a0", "log_gobs_a0", "log_gc_obs",
                  "gc_obs_a0", "gc_C", "gc_D", "gc_E"]].copy()
dsph_out.to_csv("/home/claude/dsph_gc_candidates_v1.csv", index=False)
print(f"\n[dsph_gc_candidates_v1.csv] saved")

# ========================================================
# Part 5: プロット
# ========================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) U(eps; c) landscape
ax = axes[0, 0]
eps_range = np.linspace(0, 0.98, 200)
for c, col, lw in [(0.0, COL_H, 1.5), (0.25, COL_BLUE, 1.2),
                    (0.5, COL_GREEN, 1.2), (0.75, COL_ORANGE, 1.2),
                    (1.0, COL_RED, 1.5)]:
    U = U_free(eps_range, c)
    ax.plot(eps_range, U, color=col, lw=lw, label=f"c={c}")
    # 平衡点
    if c < 1:
        e_eq = np.sqrt(1 - c)
        if e_eq < 0.98:
            U_at_eq = U_free(np.array([e_eq]), c)[0]
            ax.plot(e_eq, U_at_eq, "o", color=col, ms=7, mec="black", mew=0.5)
ax.set_xlabel("eps (膜変形量)")
ax.set_ylabel("U(eps; c)")
ax.set_title("(a) 自由エネルギー U(eps; c)\n  eps_eq(c) = sqrt(1-c) を点で表示",
             color=COL_H)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(-2.5, 0.5)

# (b) ΔU(c) memo formula vs U_eq
ax = axes[0, 1]
c_range = np.linspace(1e-4, 1.5, 300)
ax.plot(c_range, dU_barrier_memo(c_range), color=COL_H, lw=1.8,
        label="Delta-U(c) [memo式]")
ax.plot(c_range, U_eq(c_range), color=COL_RED, lw=1.3, ls="--",
        label="U_eq(c) = U(eps_eq; c)")
ax.axhline(-1.5, color="#888", ls=":", lw=0.8)
ax.axvline(0, color="#888", lw=0.5)
ax.text(0.05, -1.55, "Delta-U(0) = -3/2", fontsize=8, color="#555")
ax.set_xlabel("c")
ax.set_ylabel("Delta-U(c) または U_eq(c)")
ax.set_title("(b) c->0 極限: Delta-U(0) = -3/2\n  膜の固有負エネルギー状態",
             color=COL_H)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.5)

# (c) s(Q; T_m) SCE解
ax = axes[1, 0]
Q_range = np.logspace(-3, 1.5, 50)
s_num = np.array([solve_s(q, T_M, dU_barrier_memo, s0=s0_memo) for q in Q_range])
s_asymp = s0_memo + alpha_s * np.sqrt(Q_range)
ax.semilogx(Q_range, s_num, color=COL_H, lw=1.8, label="s(Q) 数値SCE")
ax.semilogx(Q_range, s_asymp, color=COL_RED, lw=1.2, ls="--",
            label=f"s_0 + alpha_s*sqrt(Q), s_0={s0_memo:.3f}")
ax.axhline(s0_memo, color="#888", ls=":", lw=0.8)
ax.text(1e-3, s0_memo + 0.012, f"s_0 = {s0_memo:.4f}",
        fontsize=9, color=COL_S)
ax.set_xlabel("Q = sqrt(g_obs/a0)")
ax.set_ylabel("s")
ax.set_title("(c) SCE 解: s(Q; T_m=sqrt(6))\n  小Q展開 s_0 + alpha_s*sqrt(Q)",
             color=COL_H)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, which="both")
ax.set_ylim(0.3, 1.05)

# (d) gc_dSph 候補 vs 観測
ax = axes[1, 1]
gbar_range = np.logspace(-4, 0.5, 100)
# 候補D Strigari
gc_D_line = G_best**2 / gbar_range
ax.loglog(gbar_range, gc_D_line, color=COL_H, lw=1.5,
          label=f"候補D: gc = ({G_best:.3f} a0)^2 / g_bar [Strigari型fit]")
# 候補A 定数
ax.axhline(gc_A, color=COL_BLUE, ls="--", lw=1.0,
           label=f"候補A: gc = s_0 * a0 = {gc_A:.3f}")
# 候補B
ax.axhline(gc_B, color=COL_GREEN, ls="--", lw=1.0,
           label=f"候補B: gc = (1-s_0)/s_0 * a0 = {gc_B:.3f}")
# 観測点
ax.scatter(dsph["g_bar_a0"], dsph["gc_obs_a0"],
           s=35, c=COL_RED, edgecolors="black", linewidths=0.5,
           alpha=0.85, label=f"dSph 観測 (N={len(dsph)})", zorder=3)
# 観測中央値
ax.axhline(obs_med, color="#888", ls=":", lw=0.8)
ax.text(1e-4, obs_med * 1.1, f"観測 median = {obs_med:.2f} a0",
        fontsize=8, color="#555")
ax.set_xlabel("log10(g_bar/a0)")
ax.set_ylabel("log10(gc/a0)")
ax.set_title("(d) gc_dSph 候補式 vs 観測\n  候補D (Strigari型) が最良フィット",
             color=COL_H)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3, which="both")

plt.suptitle("§9.1 dSph gc公式 理論解析: U(eps;c) c->0 極限",
             color=COL_H, fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()

fig.savefig("/mnt/user-data/outputs/fig_theory_dSph_gc_v1.pdf",
            bbox_inches="tight", dpi=200)
fig.savefig("/mnt/user-data/outputs/fig_theory_dSph_gc_v1.png",
            bbox_inches="tight", dpi=170)
plt.close(fig)
print("\n[fig_theory_dSph_gc_v1.pdf/png] saved")

# 追加の候補残差サマリCSV
summary = {
    "candidate": ["A: s_0*a0", "B: (1-s_0)/s_0*a0", "C: (s_0*a0)^2/gbar", 
                  "D: G_fit^2/gbar (fit)", "E: SCE SPARC式"],
    "bias_dex": [], "scatter_dex": [], "type": []
}
for label, val_or_col, typ in [
    ("A", gc_A, "const"), ("B", gc_B, "const"),
    ("C", "gc_C", "Strigari_fixed"), ("D", "gc_D", "Strigari_fit"),
    ("E", "gc_E", "SCE_SPARC")
]:
    if isinstance(val_or_col, (int, float)):
        pred = np.full(len(dsph), np.log10(val_or_col))
    else:
        pred = np.log10(dsph[val_or_col].values)
    resid = pred - dsph["log_gc_obs"].values
    summary["bias_dex"].append(resid.mean())
    summary["scatter_dex"].append(resid.std())
    summary["type"].append(typ)

summary_df = pd.DataFrame(summary)
summary_df.to_csv("/home/claude/gc_candidates_residuals.csv", index=False)
print("\n=== 候補式残差サマリ ===")
print(summary_df.to_string(index=False))

# 鍵となる物理量出力
key_values = {
    "T_m": T_M,
    "s_0": s0_memo,
    "alpha_s": alpha_s,
    "dU_at_c_0": -1.5,
    "obs_gc_median_a0": obs_med,
    "obs_gc_median_dex": dsph['log_gc_obs'].median(),
    "G_Strigari_fit_a0": G_best,
    "G_Strigari_fit_SI_ms2": G_best * 1.2e-10,
    "gap_alpha_over_s0": obs_med / s0_memo,
}
pd.DataFrame(list(key_values.items()), columns=["quantity", "value"]).to_csv(
    "/home/claude/theory_key_values.csv", index=False)
print("\n=== 鍵物理量 ===")
for k, v in key_values.items():
    print(f"  {k:30s} = {v}")
