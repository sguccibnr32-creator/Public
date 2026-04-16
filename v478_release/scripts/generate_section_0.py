# -*- coding: utf-8 -*-
"""
Section 0 トートロジー分離 PDF
"""
import pandas as pd
import numpy as np
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak)
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
SH3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="JP",
    fontSize=11, textColor=colors.HexColor(COL_H),
    spaceBefore=3*mm, spaceAfter=2*mm)
SB = ParagraphStyle("body", parent=styles["BodyText"], fontName="JP",
    fontSize=10, leading=15, spaceAfter=2*mm)
SM = ParagraphStyle("math", parent=SB, fontName="JP",
    leftIndent=10*mm, fontSize=10.5, textColor=colors.HexColor(COL_S),
    spaceBefore=1*mm, spaceAfter=1*mm)
SE = ParagraphStyle("emph", parent=SB, fontName="JP",
    textColor=colors.HexColor(COL_RED), leftIndent=8*mm)
SC = ParagraphStyle("cell", parent=styles["BodyText"], fontName="JP",
    fontSize=8.5, leading=11)
SCH = ParagraphStyle("cellhead", parent=styles["BodyText"], fontName="JP",
    fontSize=9, leading=11, textColor=colors.white, alignment=1)

doc = SimpleDocTemplate("/mnt/user-data/outputs/section_0_tautology_separation_v1.pdf",
    pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
    topMargin=15*mm, bottomMargin=15*mm)
story = []

story.append(Paragraph("Section 0 dSph gc 公式 トートロジー分離", SH))
story.append(Paragraph("膜宇宙論 v4.7.7 - §9.1 候補D の真の非トートロジー予測部分の同定",
                       SB))
story.append(Spacer(1, 3*mm))

# ========================================================
# 0.1 動機
# ========================================================
story.append(Paragraph("0.1 動機: 候補D の何が予測で、何が定義か", SH2))
story.append(Paragraph(
    "§9.1 で候補D gc = G^2/g_bar (G_fit = 0.240 a0) が最良フィット (bias=0, scatter=0.69 dex) "
    "を示した。しかし deep MOND抽出の定義から:", SB))
story.append(Paragraph(
    "gc_obs = g_obs^2 / g_bar  (定義)<br/>"
    "-&gt; g_obs = const (Strigari) ならば gc = const^2/g_bar が <b>自動的に</b>成立", SM))
story.append(Paragraph(
    "つまり「候補D の slope = -1」は <b>物理的予測ではなく、"
    "Strigari関係 + 抽出方法の合成的帰結</b>である可能性がある。"
    "本節では定義的構造を分離し、真の非トートロジー予測部分を同定する。", SB))

# ========================================================
# 0.2 定義的関係
# ========================================================
story.append(Paragraph("0.2 定義的関係の整理", SH2))
story.append(Paragraph("dSph 観測量の定義 (Wolf+2010 枠):", SB))
story.append(Paragraph(
    "g_obs = k * sigma^2 / r_h          (k ~ 3 for Wolf factor, 定義)<br/>"
    "g_bar = G * M_bar / r_h^2           (Newton, 定義)<br/>"
    "gc_obs = g_obs^2 / g_bar             (deep MOND 抽出)", SM))
story.append(Paragraph("これらを組合せると:", SB))
story.append(Paragraph(
    "<b>gc_obs = k^2 * sigma^4 / (G * M_bar)</b>  <- r_h 依存が定義的に消失!", SE))
story.append(Paragraph("log 表記で:", SB))
story.append(Paragraph(
    "log gc_obs = 4 * log sigma + 0 * log r_h - 1 * log M_bar + const", SM))

story.append(Paragraph("定義から期待される回帰傾き:", SH3))
pred_rows = [
    [Paragraph("回帰", SCH), Paragraph("予測傾き", SCH), Paragraph("性格", SCH)],
    [Paragraph("log gc vs log sigma",  SC), Paragraph("+4", SC),
     Paragraph("完全トートロジー", SC)],
    [Paragraph("log gc vs log r_h",    SC), Paragraph("0",  SC),
     Paragraph("<b>★ 非トートロジー試験</b>", SC)],
    [Paragraph("log gc vs log M_bar",  SC), Paragraph("-1", SC),
     Paragraph("完全トートロジー", SC)],
    [Paragraph("log g_obs vs sigma",   SC), Paragraph("+2", SC),
     Paragraph("完全トートロジー", SC)],
    [Paragraph("log g_obs vs r_h",     SC), Paragraph("-1", SC),
     Paragraph("完全トートロジー", SC)],
    [Paragraph("log g_obs vs M_bar",   SC), Paragraph("0",  SC),
     Paragraph("<b>★ 非トートロジー試験</b>", SC)],
]
t = Table(pred_rows, colWidths=[55*mm, 30*mm, 70*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ALIGN", (0,1), (0,-1), "LEFT"),
    ("ALIGN", (2,1), (2,-1), "LEFT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t)
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "★ の2つが <b>真に物理情報を担う回帰</b>。他は定義的に決まる。", SB))

# ========================================================
# 0.3 多変量回帰結果
# ========================================================
story.append(Paragraph("0.3 多変量回帰による傾き測定", SH2))

res_rows = [
    [Paragraph("回帰モデル", SCH),
     Paragraph("傾き (観測)", SCH),
     Paragraph("予測", SCH),
     Paragraph("R^2", SCH),
     Paragraph("scatter", SCH)],
    [Paragraph("[C] log g_obs ~ σ,r_h,M_bar", SC),
     Paragraph("(+2.000, -1.000, -0.000)", SC),
     Paragraph("(+2, -1, 0)", SC),
     Paragraph("1.0000", SC),
     Paragraph("0.000 dex", SC)],
    [Paragraph("[D] log g_obs ~ σ のみ (★)", SC),
     Paragraph("+0.75 +/- 0.33", SC),
     Paragraph("0 (null)", SC),
     Paragraph("0.15", SC),
     Paragraph("0.317 dex", SC)],
    [Paragraph("[E] log g_obs ~ r_h のみ (★)", SC),
     Paragraph("-0.47 +/- 0.14", SC),
     Paragraph("0 (null)", SC),
     Paragraph("0.28", SC),
     Paragraph("0.291 dex", SC)],
    [Paragraph("[F] log g_obs ~ M_bar のみ (★)", SC),
     Paragraph("-0.008 +/- 0.049", SC),
     Paragraph("0 (null)", SC),
     Paragraph("0.0008", SC),
     Paragraph("0.344 dex", SC)],
    [Paragraph("[G] log g_obs = const", SC),
     Paragraph("mean=-0.620, std=0.344", SC),
     Paragraph("Strigari", SC),
     Paragraph("-", SC),
     Paragraph("0.344 dex", SC)],
]
t2 = Table(res_rows, colWidths=[56*mm, 40*mm, 22*mm, 18*mm, 22*mm])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ALIGN", (0,1), (0,-1), "LEFT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t2)

story.append(Paragraph("有意性判定 (z = slope/se):", SH3))
story.append(Paragraph(
    "[D] g_obs vs sigma:  <b>z = +2.3σ</b> (marginal significant)<br/>"
    "[E] g_obs vs r_h:    <b>z = -3.4σ</b> (★ highly significant)<br/>"
    "[F] g_obs vs M_bar:  z = -0.2σ (<b>null成立</b>)", SM))

# ========================================================
# 図
# ========================================================
story.append(Spacer(1, 2*mm))
story.append(Image("/mnt/user-data/outputs/fig_tautology_analysis_v1.png",
                   width=170*mm, height=106*mm))
story.append(Paragraph(
    "<b>図0.1</b>: 上段 (a-c): gc_obs vs sigma, r_h, M_bar。"
    "赤点線は定義から期待される傾き。"
    "下段 (d-f): g_obs vs 各独立変数 (Strigari null test)。"
    "r_h 依存が -3.4σ で有意、M_bar 独立性は null 成立。", SB))

story.append(PageBreak())

# ========================================================
# 0.4 核心発見
# ========================================================
story.append(Paragraph("0.4 核心発見: Strigari 関係の部分的棄却と部分的承認", SH2))

story.append(Paragraph(
    "<b>[棄却] g_obs vs r_h: slope = -0.47 ± 0.14 (3.4σ)</b>", SE))
story.append(Paragraph(
    "dSph 集団内で g_obs は r_h に対し有意に減少する。"
    "これは「g_obs = 普遍定数」という厳密 Strigari 仮説の棄却を意味する。"
    "定義 g_obs = k sigma^2/r_h と組合わせると:<br/>"
    "  d(log sigma)/d(log r_h) = (-0.47 + 1)/2 = +0.265<br/>"
    "すなわち <b>dSph には構造的スケーリング関係 sigma ~ r_h^0.27 が存在</b>。"
    "McConnachie+2012, Brasseur+2011 の観測的 sigma-size 関係と整合。", SB))

story.append(Paragraph(
    "<b>[承認] g_obs vs M_bar: slope = -0.008 ± 0.049 (null)</b>", SE))
story.append(Paragraph(
    "dSph 集団内で g_obs は M_bar (=L_V*Yd) に対し統計的に独立。"
    "これは <b>膜理論の真の非トートロジー予測</b>である。"
    "c-&gt;0 極限の膜状態は baryon 量に依存せず、自発揺らぎのみで決まる "
    "という §9.1 の主張を支持する。", SB))

story.append(Paragraph(
    "<b>[承認] g_obs の中央値 = 0.240 a0 (log = -0.62)</b>", SE))
story.append(Paragraph(
    "Bernoulli 予測 G = s_0(1-s_0) * a0 = 0.228 a0 は観測 mean 0.240 a0 と "
    "5% 一致。これは集団平均としての予測であって、個別銀河の予測ではない。"
    "0.34 dex の集団内散布 (r_h スケーリング由来) は Bernoulli 予測の "
    "範囲外の構造的物理。", SB))

# ========================================================
# 0.5 Mdyn^-1 decomposition
# ========================================================
story.append(Paragraph("0.5 Mdyn^-1 (H4仮説) の真の正体", SH2))
story.append(Paragraph(
    "引継ぎメモ H4 仮説「gc ∝ Mdyn^(-1.02)」の構成を分析した。", SB))

h4_rows = [
    [Paragraph("回帰モデル", SCH),
     Paragraph("傾き", SCH),
     Paragraph("R^2", SCH),
     Paragraph("scatter", SCH)],
    [Paragraph("[H] log gc ~ log Mdyn", SC),
     Paragraph("-0.80 +/- 0.19", SC),
     Paragraph("0.38", SC),
     Paragraph("0.68 dex", SC)],
    [Paragraph("[I] log gc ~ log sigma + log r_h", SC),
     Paragraph("sigma: +0.00 +/- 0.81<br/>r_h: -1.51 +/- 0.37", SC),
     Paragraph("0.47", SC),
     Paragraph("0.62 dex", SC)],
]
t3 = Table(h4_rows, colWidths=[60*mm, 50*mm, 22*mm, 26*mm])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ALIGN", (0,1), (0,-1), "LEFT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t3)
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "Mdyn = 2.5 sigma^2 r_h / G の定義を使って分解すると、"
    "「Mdyn^-1 依存」の実体は <b>ほぼ純粋に r_h^(-1.5) 依存</b>であり、"
    "sigma 依存はほぼゼロ。引継ぎメモの「gc ∝ Mdyn^-1」は "
    "<b>真には「gc ∝ r_h^(-1.5)」</b> と解釈すべき。", SB))

# ========================================================
# 0.6 更新された予測枠組み
# ========================================================
story.append(Paragraph("0.6 膜理論 dSph 予測の更新された枠組み", SH2))

story.append(Paragraph("従来の枠組み (§9.1):", SH3))
story.append(Paragraph(
    "gc_dSph = [s_0(1-s_0)]^2 * a0^2 / g_bar  (個別銀河予測)<br/>"
    "scatter 0.69 dex を「測定誤差 + 個別進化」で説明", SM))

story.append(Paragraph("更新後の枠組み (§0解析に基づく):", SH3))
story.append(Paragraph(
    "<b>[予測A] g_obs &lt;M_bar&gt; = s_0(1-s_0) * a0 = 0.228 a0</b><br/>"
    "   M_bar-独立性が核心。集団平均 = Bernoulli予測と 5% 一致<br/>"
    "   これは非トートロジーな検証可能予測 (B+級 -&gt; A級昇格候補)", SE))
story.append(Paragraph(
    "<b>[予測B] g_obs に r_h 依存性: slope ~ -0.5</b><br/>"
    "   dSph の sigma-size 関係 (sigma ~ r_h^0.27) に由来<br/>"
    "   膜理論はこの構造を直接予測しない -&gt; 膜-構造結合の拡張が必要", SE))
story.append(Paragraph(
    "<b>[予測C] 個別 gc 予測は半径依存性込みで修正</b><br/>"
    "   gc_dSph(galaxy) = [s_0(1-s_0)]^2 * a0^2 / g_bar * f(r_h)<br/>"
    "   ここで f(r_h) ~ r_h^(-1) (経験的調整項)", SE))

# ========================================================
# 0.7 §9.1 の訂正
# ========================================================
story.append(Paragraph("0.7 §9.1 論点の訂正", SH2))

correct_rows = [
    [Paragraph("§9.1 主張", SCH),
     Paragraph("トートロジー後", SCH),
     Paragraph("格付け変更", SCH)],
    [Paragraph("候補D gc = G^2/g_bar 形状 (slope=-1)", SC),
     Paragraph("定義的帰結", SC),
     Paragraph("A級 -&gt; 格下げ", SC)],
    [Paragraph("G_fit = 0.240 a0", SC),
     Paragraph("g_obs 集団平均と同じ", SC),
     Paragraph("変更なし", SC)],
    [Paragraph("G_Bernoulli = 0.228 a0 予測", SC),
     Paragraph("集団平均の非トートロジー予測", SC),
     Paragraph("<b>B+級 -&gt; A級昇格候補</b>", SC)],
    [Paragraph("gc_dSph = [s_0(1-s_0)]^2 a0^2/g_bar", SC),
     Paragraph("集団平均として成立、個別銀河には r_h補正必要", SC),
     Paragraph("B級 -&gt; 集団平均はB+、個別はC級", SC)],
    [Paragraph("scatter 0.69 dex", SC),
     Paragraph("sigma-size 関係由来の構造散布", SC),
     Paragraph("C級 -&gt; B級 (原因特定)", SC)],
]
t4 = Table(correct_rows, colWidths=[64*mm, 66*mm, 36*mm])
t4.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COL_S)),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ALIGN", (0,1), (-2,-1), "LEFT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t4)

# ========================================================
# 0.8 残り優先順位への影響
# ========================================================
story.append(Paragraph("0.8 残りタスク優先順位への影響", SH2))

story.append(Paragraph(
    "§0 の結果を受けて優先順位を再評価:", SB))
story.append(Paragraph(
    "<b>(i) 散布分解</b>: 意義変化<br/>"
    "  - 測定誤差分 0.1 dex は残る<br/>"
    "  - host補正 ~0.2 dex は残る<br/>"
    "  - <b>残留 ~0.3 dex は sigma-size 関係由来</b> (測定誤差でない)<br/>"
    "  優先度: 中 (量化は必要だが、大半の起源は特定済み)", SB))
story.append(Paragraph(
    "<b>(ii) ブリッジ銀河検証</b>: 重要度増加<br/>"
    "  - dSph で g_obs が r_h 依存することが判明<br/>"
    "  - SPARC ブリッジ銀河 (ESO444-G084 等) でも g_obs 半径依存を測定し、"
    "dSph との連続性を確認する意義が増した<br/>"
    "  優先度: <b>高 (昇格)</b>", SB))
story.append(Paragraph(
    "<b>(iii) 高次揺らぎ項</b>: 意義低下<br/>"
    "  - sigma-size 構造依存性は膜内部物理では説明困難<br/>"
    "  - Bernoulli 平均予測は既に 5% 一致なので高次項の寄与は微小<br/>"
    "  優先度: 低 (保留)", SB))
story.append(Paragraph(
    "<b>(iv) mean-field 越え</b>: 意義変化<br/>"
    "  - sigma-size 関係は潮汐・進化史よりは内部構造の問題<br/>"
    "  - 膜-構造結合理論 (m3-ε結合拡張) が必要<br/>"
    "  優先度: 中 (v4.8)", SB))

# ========================================================
# 0.9 まとめ
# ========================================================
story.append(Paragraph("0.9 まとめ", SH2))
story.append(Paragraph(
    "dSph gc 解析からトートロジー成分を分離し、真の非トートロジー予測を同定した:", SB))
story.append(Paragraph(
    "<b>[1]</b> gc_obs の形状 (slope = -1 in log-log vs g_bar) は "
    "g_obs = const の帰結であり独立予測ではない。<br/>"
    "<b>[2]</b> 真に独立な予測は g_obs の <b>M_bar 非依存性</b> と <b>集団平均値 0.228 a0</b>。"
    "両方とも観測と一致。<br/>"
    "<b>[3]</b> g_obs の <b>r_h 依存性</b> (slope -0.47, 3.4σ) が検出された。"
    "純粋 Strigari は一部棄却、dSph sigma-size 関係と整合。<br/>"
    "<b>[4]</b> H4 仮説 gc ∝ Mdyn^-1 は真には gc ∝ r_h^(-1.5) であり、"
    "sigma 依存性はほぼゼロ。<br/>"
    "<b>[5]</b> §9.1 候補D は「形状予測」ではなく「M_bar独立 + 集団平均値」として "
    "再解釈されるべき。", SM))
story.append(Paragraph(
    "この訂正は膜理論を弱体化させるものではなく、むしろ "
    "<b>非トートロジーな予測部分を明確化し、arXiv 投稿での主張強度を定量化する</b>。"
    "v4.7.7 の §9.1 原稿は §0 の結果を反映して更新すべきである。", SB))

doc.build(story)
print("[section_0_tautology_separation_v1.pdf] saved")
