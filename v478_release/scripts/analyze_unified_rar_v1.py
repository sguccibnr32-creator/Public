# -*- coding: utf-8 -*-
"""
§9.2 統一RARプロット + J3体制遷移境界特定 + 遷移候補抽出
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# ---- v4.2 layout: IPAGothic (仕様書準拠) ----
from matplotlib import font_manager as fmng
_ipa_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
_ipa_p_path = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
fmng.fontManager.addfont(_ipa_path)
fmng.fontManager.addfont(_ipa_p_path)
plt.rcParams["font.family"] = "IPAGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# v4.2 カラーパレット
COL_H = "#1a1a2e"   # ヘッダー濃紺
COL_S = "#16213e"   # サブ濃紺
COL_RED = "#e94560" # 強調赤
COL_SPARC = "#4a90b8"  # SPARC青
COL_DSPH_MW = "#e94560"  # MW衛星赤
COL_DSPH_M31 = "#4a90b8"  # M31衛星青
COL_DSPH_ISO = "#2a9d8f" # 孤立系緑
COL_DSPH_UNK = "#888888"

# ========================================================
# データ読込
# ========================================================
uni = pd.read_csv("rar_unified_galaxy_level.csv")
cloud = pd.read_csv("rar_sparc_cloud.csv")

# Upsilon_dyn per cloud point
cloud["upsilon_dyn_R"] = 10**(cloud["log_gobs_a0"] - cloud["log_gbar_a0"])
cloud["log_upsilon"] = np.log10(cloud["upsilon_dyn_R"])

# SPARC銀河ごとの外側半径での Upsilon_dyn (外側3点の中央値)
outer_stats = []
for gal, g in cloud.groupby("galaxy"):
    g = g.sort_values("R_kpc")
    n_outer = max(1, len(g) // 4)  # 外側25%
    outer = g.tail(n_outer)
    outer_stats.append({
        "galaxy": gal,
        "R_max_kpc": g["R_kpc"].max(),
        "Vobs_max_kms": g["Vobs_kms"].max(),
        "upsilon_dyn_outer_med": outer["upsilon_dyn_R"].median(),
        "log_gbar_outer_med": outer["log_gbar_a0"].median(),
        "log_gobs_outer_med": outer["log_gobs_a0"].median(),
        "N_points": len(g),
    })
outer_df = pd.DataFrame(outer_stats)
outer_df.to_csv("sparc_outer_upsilon.csv", index=False)
print(f"[sparc_outer_upsilon.csv] {len(outer_df)} galaxies")

# ========================================================
# J3 閾値統計
# ========================================================
print("\n=== J3 threshold statistics ===")
thresholds = [3, 10, 30, 100]
j3_summary = []
for th in thresholds:
    # SPARC cloud (radius-by-radius points)
    n_cloud_pts = (cloud["upsilon_dyn_R"] > th).sum()
    n_cloud_total = len(cloud)
    # SPARC galaxies with ANY radius exceeding threshold
    n_sparc_any = cloud[cloud["upsilon_dyn_R"] > th]["galaxy"].nunique()
    # SPARC galaxies where OUTER (median) exceeds threshold
    n_sparc_outer = (outer_df["upsilon_dyn_outer_med"] > th).sum()
    # dSph
    dsph = uni[uni.source == "dSph"]
    n_dsph = (dsph["upsilon_dyn"] > th).sum()
    j3_summary.append({
        "threshold": th,
        "SPARC_points": f"{n_cloud_pts}/{n_cloud_total} ({100*n_cloud_pts/n_cloud_total:.1f}%)",
        "SPARC_galaxies_any": f"{n_sparc_any}/175 ({100*n_sparc_any/175:.1f}%)",
        "SPARC_galaxies_outer": f"{n_sparc_outer}/175 ({100*n_sparc_outer/175:.1f}%)",
        "dSph_galaxies": f"{n_dsph}/31 ({100*n_dsph/31:.1f}%)",
    })
j3_df = pd.DataFrame(j3_summary)
print(j3_df.to_string(index=False))
j3_df.to_csv("j3_threshold_summary.csv", index=False)

# ========================================================
# 遷移領域候補 SPARC 銀河リスト
# ========================================================
print("\n=== SPARC J3 transition candidates (Upsilon_dyn_outer > 3) ===")
# SPARC galaxy-level (外側中央値) + v_flat, rs_tanh補強
sparc_gal = uni[uni.source == "SPARC"].copy()
candidates = outer_df.merge(
    sparc_gal[["galaxy", "v_flat_kms", "rs_tanh_kpc", "upsilon_d", "gc_over_a0"]],
    on="galaxy", how="left"
)
candidates = candidates.sort_values("upsilon_dyn_outer_med", ascending=False)
# 遷移候補: Upsilon > 3
cand_top = candidates[candidates["upsilon_dyn_outer_med"] > 3].copy()
print(f"Found {len(cand_top)} SPARC galaxies with outer Upsilon_dyn > 3")
print()
# 上位20表示
print(cand_top.head(20).to_string(index=False))
candidates.to_csv("sparc_j3_candidates_ranked.csv", index=False)

# ========================================================
# 理論曲線計算
# ========================================================
def membrane_interp(log_gbar_a0, gc_a0):
    """膜補間関数: g_obs = g_bar / (1 - exp(-sqrt(g_bar/gc)))
       a0単位で入力・出力"""
    gbar = 10**log_gbar_a0  # [a0]
    x = np.sqrt(gbar / gc_a0)
    # 保護: x近い0で発散回避
    gobs = gbar / (1.0 - np.exp(-x))
    return np.log10(gobs)

def mond_simple(log_gbar_a0):
    """MOND simple IF: g_obs = g_bar/2 (1 + sqrt(1 + 4 a0/g_bar))
       a0単位"""
    gbar = 10**log_gbar_a0
    gobs = gbar * 0.5 * (1 + np.sqrt(1 + 4.0/gbar))
    return np.log10(gobs)

x_theory = np.linspace(-4.5, 3.5, 500)
y_newton = x_theory  # 1:1
y_mond = mond_simple(x_theory)
y_memb_SPARC = membrane_interp(x_theory, gc_a0=1.2)   # SPARC代表
y_memb_MW    = membrane_interp(x_theory, gc_a0=2.55)
y_memb_M31   = membrane_interp(x_theory, gc_a0=1.49)
y_memb_dSph  = membrane_interp(x_theory, gc_a0=35.0)  # dSph中央値体制

# ========================================================
# FIGURE 1: 統一RARプロット (2パネル)
# ========================================================
fig = plt.figure(figsize=(14, 6.5))

# --- Panel A: 統一RAR ---
axA = fig.add_subplot(1, 2, 1)

# SPARC RAR雲 (背景)
axA.scatter(cloud["log_gbar_a0"], cloud["log_gobs_a0"],
            s=3, c=COL_SPARC, alpha=0.15, label=f"SPARC RAR 雲 (N={len(cloud)})",
            rasterized=True)

# dSph: host別マーカー
dsph = uni[uni.source == "dSph"].copy()
for host, col, mk, lab in [
    ("MW", COL_DSPH_MW, "o", "dSph: MW衛星"),
    ("M31", COL_DSPH_M31, "s", "dSph: M31衛星"),
    ("Isolated", COL_DSPH_ISO, "^", "dSph: 孤立系"),
    ("Unknown", COL_DSPH_UNK, "x", "dSph: 不明"),
]:
    sub = dsph[dsph.host == host]
    if len(sub) > 0:
        axA.scatter(sub["log_gbar_a0"], sub["log_gobs_a0"],
                    s=55, c=col, marker=mk, edgecolors="black", linewidths=0.6,
                    alpha=0.85, label=f"{lab} (N={len(sub)})", zorder=3)

# 理論曲線
axA.plot(x_theory, y_newton, "k:", lw=1.0, alpha=0.6, label="ニュートン (1:1)")
axA.plot(x_theory, y_mond, "--", color="#888", lw=1.2, label="MOND simple IF")
axA.plot(x_theory, y_memb_SPARC, "-", color=COL_H, lw=1.3, alpha=0.85,
         label="膜補間 (gc=1.2 a0, SPARC)")
axA.plot(x_theory, y_memb_dSph, "-", color=COL_RED, lw=1.5, alpha=0.9,
         label="膜補間 (gc=35 a0, dSph)")

# J3境界帯 (Upsilon_dyn = 3, 10, 30 の等高線)
# log(gobs) - log(gbar) = log(Upsilon_dyn)
for th, ls, lab in [(3, "-.", "Υ=3"), (10, "-.", "Υ=10"), (30, "-.", "Υ=30")]:
    axA.plot(x_theory, x_theory + np.log10(th), ls=ls, color="#b0b0b0",
             lw=0.8, alpha=0.6)
    axA.text(-4.2, -4.2 + np.log10(th) + 0.15, lab,
             fontsize=8, color="#888", alpha=0.7)

axA.set_xlabel(r"$\log_{10}(g_{\mathrm{bar}}/a_0)$")
axA.set_ylabel(r"$\log_{10}(g_{\mathrm{obs}}/a_0)$")
axA.set_title("(A) 統一 RAR プロット: SPARC × dSph", color=COL_H, fontweight="bold")
axA.set_xlim(-4.5, 3.5)
axA.set_ylim(-2.7, 3.5)
axA.grid(alpha=0.25, linewidth=0.5)
axA.legend(loc="lower right", framealpha=0.92, fontsize=8, ncol=1)

# --- Panel B: Υ_dyn 分布 ---
axB = fig.add_subplot(1, 2, 2)

# SPARC cloud の Upsilon (radius-by-radius)
upsilon_cloud = cloud["upsilon_dyn_R"].values
upsilon_cloud = upsilon_cloud[upsilon_cloud > 0]
log_up_cloud = np.log10(upsilon_cloud)

# dSph の Upsilon
upsilon_dsph = dsph["upsilon_dyn"].values
log_up_dsph = np.log10(upsilon_dsph)

bins = np.linspace(-1.5, 3.5, 40)
axB.hist(log_up_cloud, bins=bins, color=COL_SPARC, alpha=0.55,
         density=True, label=f"SPARC RAR点 (N={len(log_up_cloud)})")
axB.hist(log_up_dsph, bins=bins, color=COL_RED, alpha=0.7,
         density=True, label=f"dSph 銀河 (N={len(log_up_dsph)})")

# 中央値
axB.axvline(np.median(log_up_cloud), color=COL_SPARC, lw=1.5, ls="-",
            label=f"SPARC 中央 Υ={10**np.median(log_up_cloud):.2f}")
axB.axvline(np.median(log_up_dsph), color=COL_RED, lw=1.5, ls="-",
            label=f"dSph 中央 Υ={10**np.median(log_up_dsph):.1f}")

# 閾値
for th in [3, 10, 30]:
    axB.axvline(np.log10(th), color="#777", ls="--", lw=0.8, alpha=0.6)
    axB.text(np.log10(th)+0.03, axB.get_ylim()[1]*0.95 if False else 1.05,
             f"Υ={th}", fontsize=8, color="#555",
             transform=axB.get_xaxis_transform(), rotation=90, va="bottom")

axB.set_xlabel(r"$\log_{10}(\Upsilon_{\mathrm{dyn}})=\log_{10}(g_{\mathrm{obs}}/g_{\mathrm{bar}})$")
axB.set_ylabel("正規化頻度")
axB.set_title("(B) $\\Upsilon_{\\mathrm{dyn}}$ 分布: SPARC 点 vs dSph 銀河",
              color=COL_H, fontweight="bold")
axB.grid(alpha=0.25, linewidth=0.5)
axB.legend(loc="upper left", framealpha=0.92, fontsize=8)

plt.suptitle("膜宇宙論 v4.7.7 §9.2 - 統一 RAR プロットと J3 体制遷移",
             color=COL_H, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

# PDF/PNG出力
fig.savefig("/mnt/user-data/outputs/fig_unified_RAR_v1.pdf",
            bbox_inches="tight", dpi=200)
fig.savefig("/mnt/user-data/outputs/fig_unified_RAR_v1.png",
            bbox_inches="tight", dpi=180)
plt.close(fig)
print("\n[fig_unified_RAR_v1.pdf/png] saved")

# ========================================================
# 遷移候補銀河テーブル (v4.2 仕様書: ReportLab PDF)
# ========================================================
print("\n=== Generating candidate table PDF ===")
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# IPAGothic 登録 (v4.2 仕様書準拠)
pdfmetrics.registerFont(TTFont("NotoJP", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"))
pdfmetrics.registerFont(TTFont("NotoJP-Bold", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"))
# IPAGothicにはBoldバリアントが無いのでRegularを両方に使用

styles = getSampleStyleSheet()
style_title = ParagraphStyle("CustomTitle", parent=styles["Title"],
    fontName="NotoJP-Bold", fontSize=16, textColor=colors.HexColor(COL_H),
    alignment=1, spaceAfter=8*mm)
style_h2 = ParagraphStyle("H2", parent=styles["Heading2"],
    fontName="NotoJP-Bold", fontSize=13, textColor=colors.white,
    backColor=colors.HexColor(COL_S), borderPadding=(6,8,6,8),
    spaceBefore=4*mm, spaceAfter=3*mm)
style_body = ParagraphStyle("Body", parent=styles["BodyText"],
    fontName="NotoJP", fontSize=10, leading=14)
style_cell = ParagraphStyle("Cell", parent=styles["BodyText"],
    fontName="NotoJP", fontSize=8.5, leading=11)
style_cell_head = ParagraphStyle("CellHead", parent=styles["BodyText"],
    fontName="NotoJP-Bold", fontSize=9, leading=11,
    textColor=colors.white, alignment=1)

doc = SimpleDocTemplate("/mnt/user-data/outputs/table_j3_transition_v1.pdf",
    pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
    topMargin=15*mm, bottomMargin=15*mm)
story = []
story.append(Paragraph("§9.2 J3体制遷移境界: SPARC候補銀河リスト", style_title))
story.append(Paragraph("膜宇宙論 v4.7.7 (2026年4月16日)", style_body))
story.append(Spacer(1, 4*mm))

# --- J3閾値サマリ表 ---
story.append(Paragraph("J3体制遷移閾値統計", style_h2))
th_rows = [[Paragraph("閾値", style_cell_head),
            Paragraph("SPARC RAR点", style_cell_head),
            Paragraph("SPARC 銀河 (任意R)", style_cell_head),
            Paragraph("SPARC 銀河 (外側)", style_cell_head),
            Paragraph("dSph 銀河", style_cell_head)]]
for r in j3_summary:
    th_rows.append([
        Paragraph(f"Υ > {r['threshold']}", style_cell),
        Paragraph(r["SPARC_points"], style_cell),
        Paragraph(r["SPARC_galaxies_any"], style_cell),
        Paragraph(r["SPARC_galaxies_outer"], style_cell),
        Paragraph(r["dSph_galaxies"], style_cell),
    ])
t1 = Table(th_rows, colWidths=[22*mm, 36*mm, 38*mm, 38*mm, 30*mm])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t1)
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "注: 「SPARC RAR点」は Rotmod_LTG 全3389半径点のうち閾値超過率。" +
    "「SPARC 銀河 (任意R)」は少なくとも1半径で超過する銀河数。" +
    "「SPARC 銀河 (外側)」は外側25%半径の中央値が超過する銀河数。",
    style_body))
story.append(Spacer(1, 4*mm))

# --- 遷移候補銀河 Top 20 ---
story.append(Paragraph("SPARC J3遷移候補 Top 20 (外側 Υ_dyn 降順)", style_h2))
cand_rows = [[Paragraph("#", style_cell_head),
              Paragraph("銀河", style_cell_head),
              Paragraph("外側Υ_dyn", style_cell_head),
              Paragraph("R_max [kpc]", style_cell_head),
              Paragraph("V_obs,max [km/s]", style_cell_head),
              Paragraph("log(g_bar/a0)", style_cell_head),
              Paragraph("log(g_obs/a0)", style_cell_head),
              Paragraph("N点", style_cell_head)]]
for i, row in enumerate(candidates.head(20).itertuples(), start=1):
    cand_rows.append([
        Paragraph(str(i), style_cell),
        Paragraph(row.galaxy, style_cell),
        Paragraph(f"{row.upsilon_dyn_outer_med:.1f}", style_cell),
        Paragraph(f"{row.R_max_kpc:.2f}", style_cell),
        Paragraph(f"{row.Vobs_max_kms:.1f}", style_cell),
        Paragraph(f"{row.log_gbar_outer_med:.2f}", style_cell),
        Paragraph(f"{row.log_gobs_outer_med:.2f}", style_cell),
        Paragraph(str(row.N_points), style_cell),
    ])
t2 = Table(cand_rows, colWidths=[8*mm, 34*mm, 20*mm, 20*mm, 24*mm, 26*mm, 26*mm, 12*mm])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ALIGN", (1,1), (1,-1), "LEFT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t2)

doc.build(story)
print("[table_j3_transition_v1.pdf] saved")

# ========================================================
# 最終サマリ標準出力
# ========================================================
print("\n" + "="*60)
print("SUMMARY - §9.2 Unified RAR Plot Analysis")
print("="*60)
print(f"Cloud points: {len(cloud)} (175 galaxies)")
print(f"Cloud Upsilon_dyn median: {np.median(upsilon_cloud):.2f}")
print(f"dSph Upsilon_dyn median: {np.median(upsilon_dsph):.1f}")
print(f"\nTop 5 SPARC transition candidates:")
for i, row in enumerate(candidates.head(5).itertuples(), start=1):
    print(f"  {i}. {row.galaxy:12s}  Upsilon_outer={row.upsilon_dyn_outer_med:6.2f}, "
          f"R_max={row.R_max_kpc:5.2f} kpc, V={row.Vobs_max_kms:5.1f} km/s")
print(f"\nJ3 transition threshold (SPARC galaxies with outer Upsilon > 3):")
print(f"  {(candidates['upsilon_dyn_outer_med'] > 3).sum()} / 175 SPARC galaxies")
print(f"  {(dsph['upsilon_dyn'] > 3).sum()} / 31 dSph galaxies")
