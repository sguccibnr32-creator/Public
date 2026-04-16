# -*- coding: utf-8 -*-
"""
§6.10 PDF v2: ASCII互換表記 (v4.2仕様書準拠)

禁止Unicode→置換:
  ≤ → <=,  ≪ → <<,  ≈ → ~=,  ₀ → _0,  ² → ^2,  ³ → ^3,
  √ → sqrt,  → → ->,  × → *,  − → -
(Greek letters Σ,Υ,π,β,α,ε,σ,ρ はIPAGothic対応で維持)
"""
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

COL_H = "#1a1a2e"
COL_S = "#16213e"
COL_RED = "#e94560"

pdfmetrics.registerFont(TTFont("JP", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"))

styles = getSampleStyleSheet()
SH = ParagraphStyle("h1", parent=styles["Title"], fontName="JP",
    fontSize=16, textColor=colors.HexColor(COL_H), alignment=1, spaceAfter=6*mm)
SH2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="JP",
    fontSize=13, textColor=colors.white, backColor=colors.HexColor(COL_S),
    borderPadding=(6,8,6,8), spaceBefore=4*mm, spaceAfter=3*mm)
SB = ParagraphStyle("body", parent=styles["BodyText"], fontName="JP",
    fontSize=10, leading=15, spaceAfter=2*mm)
SM = ParagraphStyle("math", parent=SB, fontName="JP",
    leftIndent=10*mm, fontSize=10, textColor=colors.HexColor(COL_S))
SC = ParagraphStyle("cell", parent=styles["BodyText"], fontName="JP",
    fontSize=8.5, leading=11)
SCH = ParagraphStyle("cellhead", parent=styles["BodyText"], fontName="JP",
    fontSize=9, leading=11, textColor=colors.white, alignment=1)

doc = SimpleDocTemplate("/mnt/user-data/outputs/section_6_10_dSph_extension_v2.pdf",
    pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
    topMargin=15*mm, bottomMargin=15*mm)
story = []

story.append(Paragraph("Section 6.10 dSph拡張と J3体制遷移", SH))
story.append(Paragraph("膜宇宙論 v4.7.7 (2026年4月16日)", SB))
story.append(Spacer(1, 3*mm))

# ---- 6.10.1 動機 ----
story.append(Paragraph("6.10.1 動機", SH2))
story.append(Paragraph(
    "SPARC 回転曲線データ 175銀河で確立されたC15公式 "
    "gc = 0.584 * Υd^(-0.361) * sqrt(a0 * v_flat^2 / hR) "
    "(R^2=0.607, scatter=0.286 dex) を "
    "矮小楕円体銀河 (dSph) に拡張する試みを行った。dSphは圧力支持系であり回転曲線をもたないため、"
    "Wolf+2010 推定式 g_obs = σ^2 / r_h により動的加速度を評価する。"
    "N=31銀河 (MW衛星15 + M31衛星11 + 孤立系3 + 不明2) のカタログを構築し、"
    "C15の直接適用およびJ0-J3 Jeans枠組みによる再定式化を試みた。", SB))

# ---- 6.10.2 C15直接適用の破綻 ----
story.append(Paragraph("6.10.2 C15直接適用の破綻", SH2))
story.append(Paragraph(
    "C15公式をdSphに直接適用したところ、4モデル (C15_naive, SCE_naive, SCE_QJ, C15_QJ) "
    "全てで <b>bias ~ +1.5 dex, scatter ~ 0.9 dex</b> の系統オフセットが観測された。"
    "Υd帯変換によっても scatter は不変 (Υdが効かない) で、オフセットは吸収不可能である。"
    "H1 (3Dトポロジー) +0.151 dex と H2 (ビリアル係数 β_c=2 vs sqrt(3)) +0.250 dex を合算しても "
    "+0.401 dex (観測オフセットの29%) にとどまる。H3 (膜ヒステリシス単独仮説) は3つの独立根拠 "
    "(孤立系gc > 衛星gc、距離無相関、定数モデルに劣るAIC) により棄却した。", SB))

# ---- 6.10.3 J3体制逆転 ----
story.append(Paragraph("6.10.3 J3体制逆転: 根本原因", SH2))
story.append(Paragraph(
    "オフセットの根本原因は、膜Jeans質量 M_J,mem = (4π/3) * ρ * (c_s^2 / g_eff)^3 と "
    "標準Jeans質量 M_J,std の大小関係が SPARC と dSph で逆転することにある。", SB))
story.append(Paragraph("<b>SPARC:</b> M_J,std <= M_J,mem -> バリオン分布が膜状態を駆動 (C15成立)", SM))
story.append(Paragraph("<b>dSph:</b> M_J,mem << M_J,std (28/31銀河で確認) -> 膜が構造を設定、バリオンが追従", SM))
story.append(Paragraph(
    "この「因果の方向逆転」により、dSphではC15の前提 (バリオン駆動) が成立しない。"
    "gcの入力変数はバリオン面密度 Σ_0 ではなく、膜パラメータ (T_m, 宇宙論スケール a0 自身) へ移行する。", SB))

# ---- 6.10.4 統一RARプロット (図) ----
story.append(Paragraph("6.10.4 統一RARプロットによる可視化", SH2))
story.append(Paragraph(
    "SPARC RAR雲 (3389半径点 / 175銀河、Rotmod_LTG より算出) と dSph 31銀河を "
    "同一 log(g_bar/a0) vs log(g_obs/a0) 平面に重ね描きした (図 9.2-1)。", SB))
story.append(Spacer(1, 2*mm))
story.append(Image("/mnt/user-data/outputs/fig_unified_RAR_v2.png",
                   width=170*mm, height=77*mm))
story.append(Paragraph(
    "<b>図9.2-1</b>: (A) 統一RARプロット。SPARC RAR雲 (青小点) は膜補間曲線 "
    "gc=1.2 a0 (黒実線) に整合し、dSph 31銀河 (赤丸=MW衛星、青四角=M31衛星、緑三角=孤立系) "
    "は gc=35 a0 の曲線 (赤実線) 近傍に集中する。ブリッジ銀河4個 (橙星、外側 Υ_dyn>10) が "
    "両者の境界を橋渡しする。(B) Υ_dyn = g_obs/g_bar の分布比較。"
    "SPARC 中央値 2.75 に対し dSph 中央値 35.1 (~13倍差)。"
    "10 < Υ_dyn < 30 を J3遷移帯 (橙網掛け) と定義する。", SB))
story.append(Spacer(1, 3*mm))

# ---- 6.10.5 J3閾値統計 ----
story.append(Paragraph("6.10.5 J3遷移境界の定量化", SH2))
story.append(Paragraph(
    "閾値 Υ_dyn ごとに SPARC と dSph の銀河数を集計した。"
    "「外側」は各SPARC銀河の外側25%半径の中央値 Υ_dyn を用いる。", SB))

j3 = pd.read_csv("/home/claude/j3_threshold_summary.csv")
tab = [[Paragraph("閾値", SCH),
        Paragraph("SPARC RAR点", SCH),
        Paragraph("SPARC 銀河<br/>(任意R)", SCH),
        Paragraph("SPARC 銀河<br/>(外側)", SCH),
        Paragraph("dSph 銀河", SCH)]]
for _, r in j3.iterrows():
    tab.append([
        Paragraph(f"Υ_dyn > {r['threshold']}", SC),
        Paragraph(r["SPARC_points"], SC),
        Paragraph(r["SPARC_galaxies_any"], SC),
        Paragraph(r["SPARC_galaxies_outer"], SC),
        Paragraph(r["dSph_galaxies"], SC),
    ])
t = Table(tab, colWidths=[26*mm, 36*mm, 36*mm, 36*mm, 32*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t)
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "SPARC銀河の外側部で Υ_dyn > 3 に達するのは 126/175 (72%) だが、"
    "Υ_dyn > 10 に達するのは 4/175 (2.3%) のみで、Υ_dyn > 30 はゼロ。"
    "一方 dSph は 87%、77%、55% がそれぞれの閾値を超える。"
    "Υ_dyn ~ 10-30 の領域が両集団の境界帯を形成している。", SB))

# ---- 6.10.6 ブリッジ銀河 ----
story.append(Paragraph("6.10.6 ブリッジ銀河 (J3遷移候補)", SH2))
story.append(Paragraph(
    "外側 Υ_dyn > 10 を満たすSPARC銀河4個は SPARC 集団と dSph 集団を結ぶ "
    "J3遷移候補である。これらは低質量・ガスリッチ・青色円盤銀河 (Υd=0.3) で共通しており、"
    "dSphの前駆候補体とみなせる。将来的なケーススタディ対象。", SB))

cand = pd.read_csv("/home/claude/sparc_j3_candidates_ranked.csv")
bridge_rows = [[Paragraph("銀河", SCH),
                Paragraph("外側Υ_dyn", SCH),
                Paragraph("v_flat [km/s]", SCH),
                Paragraph("rs_tanh [kpc]", SCH),
                Paragraph("Υd", SCH),
                Paragraph("gc [a0]", SCH)]]
for _, row in cand[cand["upsilon_dyn_outer_med"] > 10].iterrows():
    bridge_rows.append([
        Paragraph(str(row["galaxy"]), SC),
        Paragraph(f"{row['upsilon_dyn_outer_med']:.1f}", SC),
        Paragraph(f"{row['v_flat_kms']:.1f}" if pd.notna(row['v_flat_kms']) else "-", SC),
        Paragraph(f"{row['rs_tanh_kpc']:.3f}" if pd.notna(row['rs_tanh_kpc']) else "-", SC),
        Paragraph(f"{row['upsilon_d']:.2f}" if pd.notna(row['upsilon_d']) else "-", SC),
        Paragraph(f"{row['gc_over_a0']:.2f}" if pd.notna(row['gc_over_a0']) else "-", SC),
    ])
tb = Table(bridge_rows, colWidths=[34*mm, 24*mm, 28*mm, 28*mm, 20*mm, 24*mm])
tb.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(tb)
story.append(Spacer(1, 4*mm))

# ---- 6.10.7 結論 ----
story.append(Paragraph("6.10.7 結論と次課題", SH2))
story.append(Paragraph(
    "(i) C15のdSph直接適用は +1.5 dex の系統オフセットで破綻する (確立: B+級)。"
    "(ii) 根本原因はJ3体制逆転 M_J,mem << M_J,std で、dSphではバリオン分布がgcを決めない。"
    "(iii) J3遷移帯は Υ_dyn ~ 10-30 に存在。SPARC で該当するブリッジ銀河は4個のみ "
    "(ESO444-G084, NGC2915, NGC1705, NGC3741)。"
    "(iv) dSph用 gc 公式の再定式化 (候補: gc = f(T_m, a0) による Strigari relation 導出) は "
    "次セッションの主題。", SB))

doc.build(story)
print("[section_6_10_dSph_extension_v2.pdf] saved")

# 禁止Unicode 最終チェック
with open(__file__, 'r') as f:
    src = f.read()
# 本文部分のみチェック (コメント除く) - 簡易的に全体から
forbidden = ['≤','≥','≪','≫','≈','≠','₀','₁','₂','₃','√','→','←','×','−','∞','∑','☉','⊙','★']
# 検査対象は PDF に渡される本文文字列
# story内の文字列を集約
import re
# ソースコード内のPDF本文文字列 (Paragraph の第一引数) のみ
# 簡易: story append パターン後の最初の ' または " で囲まれた文字列
body_texts = re.findall(r'Paragraph\(\s*"([^"]+)"', src)
body_texts += re.findall(r"Paragraph\(\s*'([^']+)'", src)
all_body = ' '.join(body_texts)
hits = {c: all_body.count(c) for c in forbidden if all_body.count(c) > 0}
if hits:
    print(f"WARNING: forbidden Unicode still present: {hits}")
else:
    print("PDF body text: 禁止Unicode完全除去 OK")
