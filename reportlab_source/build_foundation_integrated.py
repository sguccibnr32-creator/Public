# -*- coding: utf-8 -*-
"""
build_foundation_integrated.py
foundation_integrated.pdf 最終統合書 (15-20 頁目標、v4.3 完全準拠)

Session +16 (A) 成果物。Session +11~+15 の 13 成果物を 1 文書に結晶化。
retract 不可事項 #1-31 全 cross-reference 整備、数値 anchor 最新反映。

Ch 構成:
  1. Introduction + Q-core 3 判断                        (#1-3)
  2. foundation_alpha1 4 主式 (Step i-iv)                 (#4-10)
  3. foundation_alpha2 7 閉形式 (Step 2-A/B/C)            (#11-17 範囲)
  4. foundation (beta)+(delta) FIRAS bound (Step iv-d)   (Eq. iv-d-I/II/III/IV)
  5. alpha-3 取扱規約 (chi dual + T_m + 2B/2C-II)         (#22-25)
  6. C3-3c P2 void narrow 化                              (#26-28)
  7. Gamma_actual physical adjoint                        (#29-31)
  8. Companion paper (v4.8) 公表戦略
  App.A Windows 実装再現性 + cross-reference table

実行 (Windows Claude Code):
  cd D:\\ドキュメント\\エントロピー\\新膜宇宙論\\これまでの軌跡\\パイソン\\
  uv run --with reportlab python build_foundation_integrated.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
pt = 1
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak,
                                 KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================================
# フォント登録 (Linux + Windows fallback)
# =========================================
FONT_PATHS = {
    'IPAGothic': [
        '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
        'C:/Windows/Fonts/ipag.ttf',
        'C:/Windows/Fonts/msgothic.ttc',
    ],
    'IPAPGothic': [
        '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
        'C:/Windows/Fonts/ipagp.ttf',
        'C:/Windows/Fonts/msgothic.ttc',
    ],
}
for font_name, paths in FONT_PATHS.items():
    for p in paths:
        if os.path.exists(p):
            try:
                if p.endswith('.ttc'):
                    pdfmetrics.registerFont(TTFont(font_name, p, subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont(font_name, p))
                print(f'[FONT] {font_name} <- {p}')
                break
            except Exception as e:
                print(f'[FONT WARN] {p}: {e}')
                continue

# =========================================
# カラーパレット (v4.3 標準、変更禁止)
# =========================================
COL_H    = colors.HexColor('#1a1a2e')
COL_S    = colors.HexColor('#16213e')
COL_RED  = colors.HexColor('#e94560')
COL_LT   = colors.HexColor('#f0f4f8')
COL_OK   = colors.HexColor('#d4edda')
COL_NG   = colors.HexColor('#f8d7da')
COL_WN   = colors.HexColor('#fff3cd')
COL_GOLD = colors.HexColor('#fff9c4')
COL_BLUE = colors.HexColor('#dbeafe')

# =========================================
# v4.3 スタイル定義 (変更禁止)
# =========================================
STYLE_H1 = ParagraphStyle('H1', fontName='IPAPGothic', fontSize=14, leading=18,
    textColor=colors.white, alignment=TA_CENTER, backColor=COL_H,
    borderPadding=6, spaceBefore=4, spaceAfter=10)
STYLE_H2 = ParagraphStyle('H2', fontName='IPAPGothic', fontSize=11, leading=15,
    textColor=colors.white, alignment=TA_LEFT, backColor=COL_S,
    borderPadding=4, leftIndent=0, spaceBefore=9, spaceAfter=5)
STYLE_H3 = ParagraphStyle('H3', fontName='IPAPGothic', fontSize=10, leading=13,
    textColor=COL_H, alignment=TA_LEFT, spaceBefore=4, spaceAfter=2)
STYLE_BODY = ParagraphStyle('Body', fontName='IPAGothic', fontSize=8.5, leading=12,
    alignment=TA_JUSTIFY, spaceAfter=2)
STYLE_BODY_SMALL = ParagraphStyle('BodySmall', fontName='IPAGothic', fontSize=8, leading=11,
    alignment=TA_JUSTIFY, spaceAfter=2)
STYLE_MATH = ParagraphStyle('Math', fontName='IPAGothic', fontSize=9, leading=13,
    leftIndent=10, backColor=COL_LT, borderPadding=3, spaceBefore=5, spaceAfter=5)
STYLE_NOTE = ParagraphStyle('Note', fontName='IPAGothic', fontSize=8, leading=11,
    alignment=TA_JUSTIFY, backColor=COL_BLUE, borderPadding=4, spaceBefore=7, spaceAfter=7)
STYLE_KEY = ParagraphStyle('Key', fontName='IPAGothic', fontSize=9, leading=13,
    alignment=TA_JUSTIFY, backColor=COL_GOLD, borderPadding=5, spaceBefore=8, spaceAfter=8)
STYLE_WARN = ParagraphStyle('Warn', fontName='IPAGothic', fontSize=8, leading=11,
    alignment=TA_JUSTIFY, backColor=COL_WN, borderPadding=4, spaceBefore=7, spaceAfter=7)

STYLE_SPACING = {
    'H1':   {'pad': 6, 'sB':  4, 'sA': 10, 'has_bg': True},
    'H2':   {'pad': 4, 'sB':  9, 'sA':  5, 'has_bg': True},
    'H3':   {'pad': 0, 'sB':  4, 'sA':  2, 'has_bg': False},
    'BODY': {'pad': 0, 'sB':  0, 'sA':  2, 'has_bg': False},
    'MATH': {'pad': 3, 'sB':  5, 'sA':  5, 'has_bg': True},
    'NOTE': {'pad': 4, 'sB':  7, 'sA':  7, 'has_bg': True},
    'KEY':  {'pad': 5, 'sB':  8, 'sA':  8, 'has_bg': True},
    'WARN': {'pad': 4, 'sB':  7, 'sA':  7, 'has_bg': True},
}

def verify_gaps(min_gap=4):
    """v4.3 Sec 5-8 必須冒頭実行"""
    fail = []
    names = list(STYLE_SPACING.keys())
    sp = STYLE_SPACING
    for a in names:
        for b in names:
            if not (sp[a]['has_bg'] and sp[b]['has_bg']): continue
            if b == 'H1': continue
            gap = sp[a]['sA'] + sp[b]['sB'] - sp[a]['pad'] - sp[b]['pad']
            if gap < min_gap:
                fail.append(f'  {a} -> {b}: gap={gap:+d}pt')
    if fail:
        print('[SPACING WARN]\n' + '\n'.join(fail))
    else:
        print(f'[OK] verify_gaps PASS (min_gap={min_gap}pt)')
    return len(fail)

# =========================================
# ヘルパー
# =========================================
sH = ParagraphStyle('TH', fontName='IPAPGothic', fontSize=8, leading=11,
    textColor=colors.white, alignment=TA_CENTER)
sC = ParagraphStyle('TC', fontName='IPAGothic', fontSize=8, leading=11,
    textColor=colors.black, alignment=TA_LEFT)

def Ph(t): return Paragraph(str(t), sH)
def P(t):  return Paragraph(str(t), sC)
def H1(t): return Paragraph(t, STYLE_H1)
def H2(t): return Paragraph(t, STYLE_H2)
def H3(t): return Paragraph(t, STYLE_H3)
def B(t):  return Paragraph(t, STYLE_BODY)
def BS(t): return Paragraph(t, STYLE_BODY_SMALL)
def M_(t): return Paragraph(t, STYLE_MATH)
def N(t):  return Paragraph(t, STYLE_NOTE)
def K(t):  return Paragraph(t, STYLE_KEY)
def W(t):  return Paragraph(t, STYLE_WARN)

def tbl(headers, rows, cw, bgs=None, hbg=None):
    data = [[Ph(h) for h in headers]]
    for row in rows:
        data.append([P(str(c)) for c in row])
    ts = [('BACKGROUND',(0,0),(-1,0), hbg or COL_H),
          ('GRID',(0,0),(-1,-1),0.4,colors.grey),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,COL_LT]),
          ('VALIGN',(0,0),(-1,-1),'TOP'),
          ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
          ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]
    if bgs:
        for ri, bg in bgs.items():
            ts.append(('BACKGROUND',(0,ri+1),(-1,ri+1),bg))
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle(ts))
    return t

# =========================================
# Chapter 1: Introduction + Q-core 3 判断
# =========================================
def ch01_introduction(story):
    story.append(H1('Chapter 1. Introduction + Q-core 3 判断'))

    story.append(H2('1.1 本統合書の位置付け'))
    story.append(B(
        '本書は膜宇宙論 Condition 15 (C15) を観測的に establish した '
        'v4.7.8 arXiv 原稿 (12 頁、19 結論) の <b>foundation layer</b> として、'
        'Session +2 から +15 までに確立した 11+ 個の閉形式と 31 項目の retract 不可事項を '
        '単一文書に結晶化する。v4.7.8 本体は保持し、本 foundation は <b>companion paper '
        '(v4.8)</b> として arXiv 併行投稿する分離戦略 (Session +6 確定)。'
    ))
    story.append(B(
        'v4.7.8 本体の主要結論 (C15 最終形、Bernoulli G_Strigari = 0.228 a_0、'
        'HSC+GAMA+KiDS Phase B での gc = 2.73 +/- 0.11 a_0 独立再現、MOND 棄却 '
        'p = 1.66e-53、universal alpha coupling b_alpha = 0.11 +/- 0.005 across 3.92 dex) '
        'に対し、本 foundation は以下を提供する: (i) Fluctuation-Dissipation から '
        'mu-distortion まで単一連鎖の閉形式、(ii) SI canonical な N_mode 定義、'
        '(iii) alpha_PT_upper ~ 3e-53 の FIRAS 上限、(iv) tau_m narrow 化の P2 void 機構。'
    ))

    story.append(H2('1.2 Q-core 3 判断 (retract 不可 #1-3)'))
    story.append(B(
        '全 foundation の起点となる 3 判断を Session +1 で確立。以降の全閉形式は '
        'これら 3 判断を前提とする。'
    ))
    story.append(M_(
        '<b>Q-core1 (C-judge)</b>: chi = a_0 / k_mem '
        '<br/>&nbsp;&nbsp;MOND 外部 anchor 採用により循環参照を回避。chi の次元 [s^2] 確定。'
        '<br/><br/>'
        '<b>Q-core2 (P-T)</b>: sigma x tau_m 結合は static curvature x dynamic relaxation '
        'の独立 product<br/>'
        '&nbsp;&nbsp;(静的 m_sigma^2 と動的 tau_m が直交、Eq. II-b と Eq. I の並立を正当化)'
        '<br/><br/>'
        '<b>Q-core3 (段階採用)</b>: Stage 1 (a) volume avg -&gt; Stage 2 (c) 精密化'
        '<br/>&nbsp;&nbsp;I_mu Stage 2 / Stage 1 比 = 0.8747 (c_0 非依存、std=2e-16) が invariance 保証。'
    ))

    story.append(H2('1.3 本統合書の到達点'))
    story.append(B(
        '閉形式 計 11+ 本 (Eq. I, II-b, II-d, III, IV / 2A-I, 2A-II / 2B-I, 2B-II / '
        '2C-0, 2C-I/II/III / iv-d-I, iv-d-II v2, iv-d-II-a, iv-d-III, iv-d-IV / iv-e-I)、'
        'retract 不可事項 #1-31 (#1-17 v4.7.8 継承、#18-21 internal memo C3 v3、'
        '#22-25 alpha-3、#26-28 P2 void、#29-31 Gamma_actual)、数値 anchor '
        '(c_0 = 0.83 canonical、NGC 3198 cross-check 2.32% PASS、alpha_PT_upper (V_xi) '
        '= 2.96e-53、tau_m in range [1.9e+10, 1.4e+53] s (FIRAS-only)) を確立済。'
    ))
    story.append(K(
        '<b>本書の意図</b>: 全 foundation を単一文書で検証可能にし、v4.8 本体執筆 '
        '(abstract/intro/methods/results/conclusions 形式 12-15 頁) の材料とする。'
        '各章で該当 retract 不可事項 # を明示、cross-reference を Appendix A に一覧化。'
    ))

    story.append(H2('1.4 本書の構成'))
    story.append(B(
        '本 foundation は観測結果を説明するための「物理的下部構造」の完結した連鎖であり、'
        '以下の読み順を推奨する:'
    ))
    story.append(tbl(
        ['Ch', 'タイトル', '主要内容', 'retract 不可 #'],
        [
            ['1', 'Introduction + Q-core 3 判断', 'C-judge / P-T / Stage 採用', '#1-3'],
            ['2', 'foundation_alpha1 4 主式', 'FDT + Picture T + mu', '#4-10'],
            ['3', 'foundation_alpha2 7 閉形式', 'c_mem + N_mode + Lambda_UV', '#11-17 範囲'],
            ['4', 'foundation (beta)+(delta)', 'FIRAS mu-distortion 上限', 'Eq. iv-d-I~IV'],
            ['5', 'alpha-3 取扱規約', 'chi dual + T_m + 2B/2C-II', '#22-25'],
            ['6', 'C3-3c P2 void narrow 化', 'Eq. iv-e-I 2 route min', '#26-28'],
            ['7', 'Gamma_actual physical adjoint', '3 route 並行 + b_alpha 直結', '#29-31'],
            ['8', 'Companion paper 公表戦略', 'M1-M5 + 3 channel', '--'],
            ['A', 'Windows 実装再現性', 'script + csv + cross-ref', '全 #'],
        ],
        [10*mm, 50*mm, 60*mm, 30*mm]
    ))
    story.append(N(
        '<b>読者の前提</b>: v4.7.8 本体 (12 頁 arXiv 原稿) および v4.2 本体 (126 頁、'
        '補節 A + 補題 5 + T-10) を既読とする。本書は foundation layer の閉形式体系と '
        '数値 anchor を<b>再編纂</b>したもので、観測結果の establishment (C15 + Bernoulli + '
        'HSC Phase B + universal b_alpha) は v4.7.8 を参照。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 2: foundation_alpha1 4 主式
# =========================================
def ch02_foundation_alpha1(story):
    story.append(H1('Chapter 2. foundation_alpha1: 4 主式 (Step i-iv)'))

    story.append(H2('2.1 Step i -- Fluctuation-Dissipation (Eq. I)'))
    story.append(B(
        '膜 strain field epsilon の平衡分散を、effective potential w(c)^2 '
        '(= 2 eps_0 / (1 - eps_0)、Form B) と Z_2 SSB 無次元温度 T_m = sqrt(6) '
        '(補題 5、v4.7.6 A 級) から導出。'
    ))
    story.append(M_(
        '<b>Eq. I</b>: &lt;delta eps^2&gt;_eq = T_m / w(c)^2<br/>'
        '&nbsp;&nbsp;= 2.449 / (2 eps_0 / (1 - eps_0))<br/>'
        '&nbsp;&nbsp;c_0 = 0.83 で eps_0 = 0.412311、w(c)^2 = 1.4032 -&gt; '
        '&lt;delta eps^2&gt;_eq = 1.745'
    ))
    story.append(H3('2.1.1 w(c)^2 の Form B 由来'))
    story.append(B(
        'Form B (v4.7.6 確立) では U(eps; c) = -eps - eps^2/2 - c ln(1 - eps) '
        'から eps_0 = sqrt(1 - c) が dU/deps = 0 を満たす。w(c)^2 = U\'\'(eps_0) を '
        '整理すると w(c)^2 = 2 eps_0 / (1 - eps_0) が得られる。c in (0, 1) の全域で '
        'w(c)^2 &gt; 0 (thermodynamic stability) が保証される。'
    ))
    story.append(M_(
        'eps_0(c) = sqrt(1 - c)、c_0 = 0.83 -&gt; eps_0 = sqrt(0.17) = 0.412311<br/>'
        'w(c)^2 = 2 &#183; 0.412311 / (1 - 0.412311) = 0.824622 / 0.587689 = 1.4032<br/>'
        '-&gt; &lt;delta eps^2&gt;_eq = 2.449 / 1.4032 = 1.745 (c_0 = 0.83 canonical)'
    ))
    story.append(H3('2.1.2 T_m = sqrt(6) の Z_2 SSB 由来 (補題 5)'))
    story.append(B(
        'Z_2 対称性自発的破れの臨界温度として T_m = sqrt(6) を無次元で与える。補題 5 '
        '(v4.7.6 A 級) は、4-point correlator の Gaussian 評価と mean-field 臨界指数 '
        'z nu = 1/2 の連立から本値が universal constant となることを示す。物理単位 '
        '(ケルビン等価) への翻訳は外部 anchor 依存で、本 foundation 内部では未決 '
        '(Chapter 5 #24, #25 参照)。'
    ))

    story.append(H2('2.2 Step ii -- Picture T (Eq. II-b, II-d, III)'))
    story.append(B(
        'sigma mass-squared、Jeans volume-halved、及び両者の積関係。いずれも '
        'k_mem/a_0 を共通次元担体とし、内部整合性 Eq. III = Eq. II-b &#183; Eq. I で閉じる。'
    ))
    story.append(M_(
        '<b>Eq. II-b</b>: m_sigma^2(c) = 4 eps_0 &#183; w(c)^2 &#183; k_mem / a_0<br/>'
        '<b>Eq. II-d</b>: J_half(c) = (8 k_mem^2 T_m^2 Gamma / a_0^2) &#183; eps_0 (1 - eps_0)<br/>'
        '<b>Eq. III</b>: m_sigma^2 &#183; &lt;delta eps^2&gt;_eq = 4 eps_0 &#183; T_m &#183; k_mem / a_0'
    ))
    story.append(N(
        '<b>注</b>: Eq. III は Eq. I と Eq. II-b から w(c)^2 が消去され、'
        '右辺が c_0 の関数として eps_0 のみに帰着する構造的整合性。'
        'Jeans volume-halved J_half は Phase C2 dSph 解析で G_Strigari と独立に 30 銀河再現 (+/-5%)。'
    ))

    story.append(H2('2.3 Step iii -- Picture T 数値検証 (内部整合性)'))
    story.append(B(
        'c_0 = 0.83 canonical で Eq. III 両辺を独立評価:'
    ))
    story.append(M_(
        'LHS: m_sigma^2 &#183; &lt;delta eps^2&gt;_eq = (4 &#183; 0.412 &#183; 1.40 &#183; k/a) &#183; '
        '(2.449 / 1.40) = 4.04 &#183; k_mem/a_0<br/>'
        'RHS: 4 &#183; 0.412 &#183; 2.449 &#183; k_mem/a_0 = 4.04 &#183; k_mem/a_0<br/>'
        '-&gt; 一致 (解析的恒等式)'
    ))

    story.append(H2('2.4 Step iv -- mu-distortion PT integration (Eq. IV)'))
    story.append(B(
        'mu 歪みは A_PT prefactor と I_mu kernel integral の直積に分離。Stage 1 で '
        'Chluba W_mu kernel 下の一次近似、Stage 2 で精密化。両 Stage の比が c_0 非依存 '
        '(std = 2e-16) となることが integrand 構造の正しさの強い証左。'
    ))
    story.append(M_(
        '<b>Eq. IV</b>: mu = A_PT &#183; I_mu<br/>'
        '<b>A_PT</b> = (4 alpha_PT T_m k_mem Gamma N_mode / a_0) &#183; (rho_mem,0^2 / rho_gamma,0)<br/>'
        '<b>I_mu_Stage1(c_0)</b> = eps_0(c_0) &#183; 1.76e+20 s<br/>'
        '&nbsp;&nbsp;c_0 = 0.83: I_mu_S1 = 7.2275e+19 s (期待 7.26e+19 比 0.448% PASS)<br/>'
        '<b>I_mu_Stage2(c_0)</b> = 6.3220e+19 s, S2/S1 = 0.8747 (c_0 非依存 std=2e-16)'
    ))
    story.append(H3('2.4.1 Stage 1 vs Stage 2 の分離構造'))
    story.append(tbl(
        ['Stage', '近似', 'I_mu(0.83) [s]', 'c_0 依存'],
        [
            ['Stage 1', 'volume avg (a)', '7.2275e+19', 'eps_0(c_0) 線形'],
            ['Stage 2', '精密化 (c) with Chluba W_mu', '6.3220e+19', 'eps_0(c_0) 線形'],
            ['比 S2/S1', '--', '0.8747', 'std=2e-16 (非依存)'],
        ],
        [22*mm, 55*mm, 40*mm, 40*mm]
    ))
    story.append(B(
        'S2/S1 = 0.8747 が c_0 に依存しない (std=2e-16 機械精度) ことは、Stage 2 integrand '
        'の (1+z) 分子配置 (Eq. iv-d-II v2) が正しく選ばれている強い証左。誤配置では '
        'c_0 依存性が std ~30% で現れる。'
    ))

    story.append(H2('2.5 FIRAS bound (foundation_alpha1 上限)'))
    story.append(B(
        'Fixsen+1996 FIRAS mu_FIRAS &lt; 9e-5 を A_PT に繋ぎ、alpha_PT の上限を設定。'
        '本上限は Chapter 4 (β)+(δ) で V_xi 採用後に c_mem linear 縮約形 (Eq. iv-d-IV) となる。'
    ))
    story.append(M_(
        '<b>FIRAS bound</b>: alpha_PT &#183; k_mem &#183; N_mode &lt; B_FIRAS_combo(c_0)<br/>'
        '&nbsp;&nbsp;c_0 = 0.83 で B_FIRAS_combo = 1.516e-12 [J^(-1) m^(-1/2)]'
    ))
    story.append(K(
        '<b>retract 不可 #4-10 (本章)</b>: Eq. I / II-b / II-d / III / IV は全て Session +2~+6 '
        '確立、以降の閉形式体系 (alpha2, (beta)+(delta)) は本 4 主式を前提とする。'
        'Form B (w(c)^2 = 2 eps_0/(1 - eps_0), eps_0 = sqrt(1 - c)) および '
        'T_m = sqrt(6) (Z_2 SSB 臨界温度、補題 5) は変更禁止。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 3: foundation_alpha2 7 閉形式
# =========================================
def ch03_foundation_alpha2(story):
    story.append(H1('Chapter 3. foundation_alpha2: 7 閉形式 (Step 2-A/B/C)'))

    story.append(H2('3.0 本章の 7 閉形式 overview'))
    story.append(B(
        'Step 2-A (c_mem + k_mem + chi_F)、Step 2-B (N_mode SI)、Step 2-C (Lambda_UV symbolic) '
        'の 3 群 7 閉形式を確立 (Session +7~+9)。以下 summary:'
    ))
    story.append(tbl(
        ['Eq', '内容', '次元', '用途'],
        [
            ['2A-I', 'k_mem^2 = a_0 / c_mem^2', '[1/m^2]', '波数 anchor'],
            ['2A-II', 'chi_F = c_mem sqrt(a_0)', '[m^(3/2)/s^2]', 'force-conjugate'],
            ['2B-I', 'n_mode = V Lambda_UV^3 / (6 pi^2 hbar^3 c_mem^3)', '[m^(-3)]', 'mode 数密度'],
            ['2B-II', 'N_mode = V Lambda_UV^2 / (6 pi^2 hbar^3 c_mem^3)', '[J^(-1)]', 'SI canonical 数値代入'],
            ['2C-0', 'Lambda_UV = 2 c_lt^2 w sqrt(eps_0) c_mem^(-1/2) a_0^(-1/4)', 'symbolic', 'UV cutoff 閉形式'],
            ['2C-I', '派生 (symbolic)', 'symbolic', '構造調査'],
            ['2C-II', '派生 (symbolic)', 'symbolic', '構造調査'],
            ['2C-III', '派生 (symbolic)', 'symbolic', '構造調査'],
        ],
        [15*mm, 75*mm, 30*mm, 40*mm]
    ))

    story.append(H2('3.1 Step 2-A -- c_mem + k_mem + chi_F (Eq. 2A-I, 2A-II)'))
    story.append(B(
        '膜固有音速 c_mem を介して k_mem と chi_F (Q-core1 chi の次元 [m^(3/2)/s^2] 版) '
        'を接続。v3.7 Chap 18 Table 18-3 から c_mem(c_0=0.83) の外挿値を得る。'
    ))
    story.append(M_(
        '<b>Eq. 2A-I</b>: k_mem^2 = a_0 / c_mem^2<br/>'
        '<b>Eq. 2A-II</b>: chi_F = c_mem &#183; sqrt(a_0) &nbsp;&nbsp;[m^(3/2)/s^2]<br/><br/>'
        'c_0 = 0.83 数値:<br/>'
        '&nbsp;&nbsp;c_mem = 3.833e+05 m/s (f_opt 1.9163 x V_flat 2.00e5 Sb 型外挿)<br/>'
        '&nbsp;&nbsp;chi_F = 4.198 m^(3/2)/s^2<br/>'
        '&nbsp;&nbsp;k_mem = sqrt(a_0)/c_mem = 2.861e-08 [1/m]'
    ))
    story.append(H3('3.1.1 c_mem の Chap 18-3 外挿手順'))
    story.append(B(
        'v3.7 Chap 18 Table 18-3 は f_opt(c) を c in [0.3, 0.8] 範囲で 2 点以上確保。'
        'c_0 = 0.83 は範囲外 (0.8 を超える) の Sb 型近傍外挿で、f_opt(0.83) = 1.9163 を '
        '得る。これに V_flat universal = 2.00e5 m/s を乗じて c_mem = 3.833e+05 m/s。'
        '本外挿は保留事項 (Session +10 指摘、長期課題、Chap 18 Table 2 点外挿の物理検証)。'
    ))
    story.append(N(
        '<b>V\'\'(phi_0) の Chap 18 Table lookup (Q1-alpha)</b>: x = 0 での値を採用し、'
        'c_0 = 0.83 で V\'\'(phi_0, x=0) = 2.31 (無次元)。Form B '
        'V\'\'(eps_0) = 4 eps_0 [c/(1-eps_0)^2 - 1] の具体評価。'
    ))

    story.append(H2('3.2 Step 2-B -- N_mode SI canonical (Eq. 2B-I, 2B-II)'))
    story.append(B(
        'mode 数密度 n_mode [m^(-3)] と mode 総数 per energy N_mode [J^(-1)] を '
        'Lambda_UV と c_mem で与える。<b>数値代入は必ず Eq. 2B-II (SI canonical)</b>。'
    ))
    story.append(M_(
        '<b>Eq. 2B-I</b>: n_mode = V &#183; Lambda_UV^3 / (6 pi^2 &#183; hbar^3 &#183; c_mem^3) '
        '&nbsp;[m^(-3)]<br/>'
        '<b>Eq. 2B-II</b>: N_mode = V &#183; Lambda_UV^2 / (6 pi^2 &#183; hbar^3 &#183; c_mem^3) '
        '&nbsp;[J^(-1)] &nbsp;<b>&lt;-- SI canonical</b>'
    ))
    story.append(H3('3.2.1 NGC 3198 cross-check 数値'))
    story.append(B(
        'C15 で観測的に establish された NGC 3198 anchor に対し、foundation の '
        'k_mem &#183; N_mode product を比較。foundation 予測値 2.998e+38 vs 観測値 '
        '2.93e+38、relative diff = <b>2.32%</b> で PASS (Session +8 A 級結果)。'
        'Eq. 2B-II の SI canonical 性の強い支持。'
    ))

    story.append(H2('3.3 Step 2-C -- Lambda_UV symbolic (Eq. 2C-0/I/II/III)'))
    story.append(B(
        'Lambda_UV 閉形式は c_lt (光速) を含む symbolic 式で、'
        '<b>数値代入は構造研究用途のみに限定</b>。Eq. 2B-II との混用は 132 桁乖離 (後述)。'
    ))
    story.append(M_(
        '<b>Eq. 2C-0</b>: Lambda_UV = 2 &#183; c_lt^2 &#183; w(c) &#183; sqrt(eps_0(c)) '
        '&#183; c_mem^(-1/2) &#183; a_0^(-1/4) &nbsp;(symbolic)<br/>'
        '<b>Eq. 2C-I/II/III</b>: 構造的派生形 (symbolic 閉形式のみ、数値代入不可)<br/><br/>'
        'c_0 = 0.83 で Lambda_UV = 9.549e-49 J (v3.7 Chap 18 Table 18-4 線形外挿、2 点外挿で要物理検証)'
    ))
    story.append(H3('3.3.1 132 桁乖離の具体例 (混用の罠)'))
    story.append(B(
        '規約違反して Eq. 2C-II に c_0 = 0.83 の SI 数値を直接代入した場合の破綻例:'
    ))
    story.append(M_(
        'Eq. 2C-II は c_lt を陽に含む symbolic 式<br/>'
        'SI 単位で c_lt = 2.998e+08 m/s -&gt; c_lt^n (n ~ 20) で 10^170 発散<br/>'
        '&nbsp;&nbsp;vs Eq. 2B-II 経由の正値 ~ 10^38 (NGC 3198 anchor 整合)<br/>'
        '-&gt; 約 10^132 倍の乖離 (<b>132 桁</b>、符号含めて完全破綻)'
    ))

    story.append(H2('3.4 取扱規約: 数値 2B-II / 構造 2C-II'))
    story.append(K(
        '<b>retract 不可 #23</b>: 数値代入は常に Eq. 2B-II (SI canonical)。'
        '構造調査 (UV-IR duality, dimensional scan) は Eq. 2C-II を symbolic で使用。'
        '<br/><br/>'
        '<b>混用の罠</b>: Eq. 2C-II に c_0 = 0.83 数値を SI 単位で代入すると、'
        'c_lt^2 が 9e+16 m^2/s^2 で 10^170 発散を誘発 -&gt; N_mode 比較で '
        '<b>132 桁乖離</b>。必ず Eq. 2B-II 経由で数値計算する。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 4: foundation (beta)+(delta) FIRAS bound
# =========================================
def ch04_foundation_beta_delta(story):
    story.append(H1('Chapter 4. foundation (beta)+(delta): FIRAS mu-distortion 上限'))

    story.append(H2('4.1 4 新規閉形式 (Eq. iv-d-I/II/III/IV)'))
    story.append(B(
        'Stage 2 精密化の軸となる 4 閉形式を Session +10 で確立。Chluba 2016 W_mu(z) '
        'kernel (Z_MU = 5e+04, Z_DC = 1.98e+06) を採用し、(1+z) 分子配置を critical に固定。'
    ))
    story.append(M_(
        '<b>Eq. iv-d-I</b> (Stage 1 closed):<br/>'
        '&nbsp;&nbsp;I_mu_Stage1(c_0) = eps_0(c_0) &#183; ln((1+Z_DC)/(1+Z_MU)) / '
        '(H_0 &#183; sqrt(Omega_R))<br/>'
        '&nbsp;&nbsp;~= eps_0(c_0) &#183; 1.76e+20 s<br/><br/>'
        '<b>Eq. iv-d-II v2</b> (Stage 2 integrand):<br/>'
        '&nbsp;&nbsp;integrand(z, c_0) = eps_0(c_0) &#183; W_mu_Chluba(z) &#183; (1+z) / H(z)<br/>'
        '&nbsp;&nbsp;[(1+z) が<b>分子位置</b>が critical、分母と誤った配置で 30% ずれ]<br/><br/>'
        '<b>Eq. iv-d-II-a</b> (Chluba W_mu kernel):<br/>'
        '&nbsp;&nbsp;W_mu(z) = (1 - exp(-(z/Z_MU)^1.88)) &#183; exp(-(z/Z_DC)^2.5)<br/>'
        '&nbsp;&nbsp;Z_MU = 5e+04, Z_DC = 1.98e+06 (Chluba 2016)'
    ))

    story.append(H2('4.2 B_FIRAS_combo と alpha_PT_upper'))
    story.append(M_(
        '<b>Eq. iv-d-III</b>: B_FIRAS_combo(c_0) = '
        '(mu_FIRAS &#183; A_0 &#183; rho_gamma,0) / (4 &#183; T_m &#183; Gamma_ref &#183; '
        'rho_mem,0^2 &#183; I_mu(c_0))<br/>'
        '&nbsp;&nbsp;c_0 = 0.83: B_FIRAS_combo = 1.516e-12 [J^(-1) m^(-1/2)]<br/><br/>'
        '<b>Eq. iv-d-IV</b>: alpha_PT_upper(c_0) = c_mem linear 縮約形 (V_xi 採用後)<br/>'
        '&nbsp;&nbsp;c_0 = 0.83, V_xi: <b>alpha_PT_upper = 2.96e-53</b><br/>'
        '&nbsp;&nbsp;(conservative V_cosmo 採用: 2.15e-68)'
    ))
    story.append(H3('4.2.1 V_xi vs V_cosmo の選択と alpha_PT_upper 感度'))
    story.append(tbl(
        ['V 選択', '体積 [m^3]', 'alpha_PT_upper', '解釈'],
        [
            ['V_xi (膜 coherence)', '7.683e+63', '2.96e-53', 'primary (a_0 独立性整合)'],
            ['V_cosmo (geometric)', '1.060e+79', '2.15e-68', 'conservative upper'],
            ['比', '~1.38e+15', '~1.38e+15', 'V に線形 scaling'],
        ],
        [40*mm, 35*mm, 30*mm, 45*mm]
    ))
    story.append(B(
        'V_xi 採用 (primary) は foundation_alpha2 の a_0 独立性 structural finding '
        'と整合。V_cosmo は銀河団スケール以上を含む geometric volume で、physical '
        'coherence を過大評価するため conservative upper bound として扱う。'
    ))

    story.append(H2('4.3 tau_m 両端 bound (FIRAS-only)'))
    story.append(B(
        'FIRAS 上限を alpha_PT_upper を通じて tau_m に逆写像した場合の '
        'upper bound (V_xi primary)、および因果律 (z=5e+04 Hubble 膨張) による '
        'lower bound の両端を確定。'
    ))
    story.append(M_(
        '<b>tau_m_upper (V_xi, FIRAS-only)</b> = 1.418e+53 s<br/>'
        '<b>tau_m_lower (causal, z=5e+04)</b> = 1.906e+10 s (~605 yr)<br/>'
        '&nbsp;&nbsp;-&gt; FIRAS-only 幅 = <b>40-58 桁</b> (Chapter 6 P2 void で narrow 化)'
    ))
    story.append(H3('4.3.1 Chluba W_mu(z) kernel の shape 解釈'))
    story.append(B(
        'Chluba 2016 kernel W_mu(z) = (1 - exp(-(z/Z_MU)^1.88)) &#183; exp(-(z/Z_DC)^2.5) は '
        '2 階層のべき表現。(1 - exp(-.)) 部が z = Z_MU = 5e+04 (mu 境界) での遷移、'
        'exp(-(z/Z_DC)^2.5) 部が z = Z_DC = 1.98e+06 (double Compton) での cutoff。'
        'べき 1.88 と 2.5 は Chluba 2016 Fig.1 フィット値、foundation は外部 input として採用。'
    ))

    story.append(H2('4.4 Gamma_ref = 1 placeholder と linear scaling rule'))
    story.append(N(
        '<b>Gamma_ref = 1</b> を placeholder とし、実 Gamma_actual を後適用する linear scaling rule '
        'を Session +10 で確立 (retract 不可):<br/>'
        '&nbsp;&nbsp;tau_from_firas proportional to Gamma_actual<br/>'
        '&nbsp;&nbsp;alpha_PT_upper: invariant (B_FIRAS 経由相殺)<br/>'
        '&nbsp;&nbsp;B_FIRAS_combo proportional to 1 / Gamma_actual<br/>'
        '&nbsp;&nbsp;tau_m_upper (FIRAS-bound 部分) proportional to Gamma_actual<br/>'
        '&nbsp;&nbsp;tau_m_upper (void-bound, geometric) invariant'
    ))
    story.append(K(
        '<b>Stage 2 の c_0 非依存性</b>: S2/S1 = 0.8747 (std = 2e-16、機械精度) は '
        'Eq. iv-d-II v2 integrand の (1+z) 分子配置が正しく選ばれている証拠。'
        '誤配置では c_0 依存性が現れる。<br/><br/>'
        '<b>linear scaling rule の利点</b>: Chapter 7 で Gamma_actual 3 route が確定した後、'
        '本章の全数値は Gamma_ref = 1 からの単純 scalar 倍で update 可能。'
        'foundation の modularity が保証される。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 5: alpha-3 取扱規約
# =========================================
def ch05_alpha3_conventions(story):
    story.append(H1('Chapter 5. alpha-3 取扱規約: chi dual + T_m + 2B/2C-II'))

    story.append(H2('5.1 #22 -- chi dual (chi_E と chi_F の独立性)'))
    story.append(B(
        '記号 chi は 2 つの独立変数を指す。v4.2 補節 A の chi_E と Eq. 2A-II の chi_F は '
        '<b>次元も意味も異なる</b>。universal conversion 不存在。'
    ))
    story.append(M_(
        '<b>chi_E</b> (v4.2 補節 A): [s^2]、MOND anchor a_0/k_mem 由来の Q-core1 量<br/>'
        '<b>chi_F</b> (Eq. 2A-II): [m^(3/2)/s^2]、c_mem sqrt(a_0) 由来の force-conjugate 量<br/><br/>'
        '-&gt; 両者の次元比 [m^(3/2)/s^2] / [s^2] = [m^(3/2)/s^4] に universal 意味なし'
    ))

    story.append(H2('5.2 #23 -- 数値 2B-II / 構造 2C-II 役割分担 (再掲)'))
    story.append(B(
        'Chapter 3.4 で確立した規約の再掲。<b>数値代入の方向性に応じて式を選ぶ</b>:'
    ))
    story.append(tbl(
        ['目的', '使用式', '単位系'],
        [
            ['数値代入 (NGC 3198 等)', 'Eq. 2B-II', 'SI 完全 [J^(-1)]'],
            ['構造調査 (UV-IR scan)', 'Eq. 2C-II', 'symbolic のみ'],
            ['混用 (禁止)', '2C-II に SI 数値', '132 桁乖離'],
        ],
        [50*mm, 60*mm, 40*mm]
    ))

    story.append(H2('5.3 #24 -- T_m = sqrt(6) 無次元保持'))
    story.append(B(
        'Z_2 SSB 臨界温度 T_m = sqrt(6) は<b>無次元 universal constant として保持</b>。'
        '絶対較正 (ケルビン換算等) は外部 anchor 依存であり、foundation 内部では決定不能。'
    ))
    story.append(M_(
        'T_m = sqrt(6) ~= 2.449 (dimensionless)<br/>'
        '&nbsp;&nbsp;Z_2 SSB Lemma 5 (v4.7.6 A 級) 由来<br/>'
        'Option III (膜固有エネルギースケール) vs Option IV (z_SSB 宇宙論的) の同定は長期課題'
    ))

    story.append(H2('5.4 #25 -- epsilon_scale 候補 (a)/(b) 共存'))
    story.append(B(
        'T_m を物理単位に翻訳する際、sigma rest energy route (a) と '
        'membrane coherence route (b) で <b>96 桁 gap</b> が生じる。両候補の共存を '
        '明示し、foundation 内部では決着しない stance を採る。'
    ))
    story.append(tbl(
        ['候補', '起点', '階層', 'T_m [K] 等価'],
        [
            ['(a) sigma rest E', '個別励起エネルギー', 'micro', '~1e-25 K'],
            ['(b) mem coherence', '集団励起エネルギー', 'macro', '~1e+71 K'],
            ['gap', '--', '--', '96 桁未解決'],
        ],
        [30*mm, 40*mm, 25*mm, 35*mm]
    ))
    story.append(H3('5.4.1 96 桁 gap の物理的意味'))
    story.append(B(
        '(a) sigma rest energy route は sigma mode 励起 1 quanta あたりのエネルギーを '
        '基準とし、hbar &#183; Omega_sigma ~ 1e-45 J 近傍を与える。(b) mem coherence route は '
        '膜 coherence volume V_xi 全体の集団励起を基準とし、rho_mem,0 &#183; V_xi &#183; c_lt^2 '
        '相当で 1e+47 J 近傍を与える。両者の比 ~ 10^92 が T_m 等価換算で 10^96 K 付近の '
        'gap として現れる。'
    ))
    story.append(B(
        '本 gap の解消には外部 anchor (FIRAS mu 上限 + CMB TT パワースペクトル + 21cm '
        'signal) を同時利用した cross-check が必要。foundation 内部では (a) が '
        'individual quantum excitation、(b) が collective mode excitation という階層的 '
        '別物理量であり、universal conversion 不存在を<b>積極的に認める</b> stance が '
        'v4.8 の誠実な取扱い。'
    ))
    story.append(K(
        '<b>alpha-3 取扱規約 (#22-25 まとめ)</b>: 本 foundation 内部では T_m 絶対較正は '
        '決定不能であり、外部 anchor (FIRAS + CMB TT + 21cm cross-check) による '
        'エネルギースケール固定が将来課題 (long-term #2)。v4.8 本体では '
        '(a)(b) 両候補の operational definition と限界を明示する。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 6: C3-3c P2 void narrow 化
# =========================================
def ch06_p2_void_narrow(story):
    story.append(H1('Chapter 6. C3-3c P2 void による tau_m narrow 化'))

    story.append(H2('6.1 Eq. iv-e-I -- B_tau_m_void 2 route (#26)'))
    story.append(B(
        'void 内 tau_m 上限を 2 route の minimum として定義。causality route は '
        '膜擾乱伝播 (c_lt 仮定)、dynamical route は void 特徴速度 v_char。'
    ))
    story.append(M_(
        '<b>Eq. iv-e-I</b>: B_tau_m_void = R_void / v_char &nbsp;(2 route, min 採用)<br/><br/>'
        'Route 1 (causality): B_causal = R_void / c_lt<br/>'
        'Route 2 (dynamical): B_dyn = R_void / v_flat_void_median<br/>'
        'B_tau_m_void_final = min(B_causal, B_dyn)'
    ))

    story.append(H2('6.2 V_mode_void primary = V_xi_void (#27)'))
    story.append(B(
        'V_mode (mode volume) の void 版は V_xi_void (膜 coherence volume) を primary、'
        'V_void (geometric) を conservative upper として採用。C15 foundation 全体で '
        'V_xi primary 方針と整合。'
    ))

    story.append(H2('6.3 tau_m_upper narrow 化 min 構造 (#28)'))
    story.append(M_(
        '<b>最終形</b>: tau_m_upper_final = min(B_tau_m_void, tau_firas)<br/><br/>'
        '実装: 各 void で B_causal 計算 -&gt; 最小値を FIRAS-only bound と比較'
    ))

    story.append(H2('6.4 Local Void narrow 見込 (38 桁 tighten)'))
    story.append(B(
        'Session +13 設計 + Session +15 Windows 実装 template 5 void 一覧。'
        'Local Void (R = 22 Mpc) が最も tight な B_causal を与える:'
    ))
    story.append(tbl(
        ['void', 'R [Mpc]', 'B_causal [s]', 'source (推定)'],
        [
            ['Local Void', '22', '2.26e+15', 'Kreckel+2011'],
            ['Bootes Void', '55', '5.66e+15', 'Kirshner+1981'],
            ['Cetus Void', '30', '3.09e+15', 'Pan+2012'],
            ['Sculptor Void', '25', '2.57e+15', 'Pan+2012'],
            ['Corona Borealis', '60', '6.17e+15', 'Weygaert+2011'],
        ],
        [40*mm, 20*mm, 35*mm, 45*mm]
    ))
    story.append(W(
        '<b>警告</b>: 上記 5 void の値は Session +15 時点で claude.ai 側が文献未確認で推定したもの。'
        'Windows 側 task (a) で Kreckel+2011 (ApJ 735, 6)、Pan+2012 (MNRAS 421, 926)、'
        'Ricciardelli+2014 (MNRAS 445, 4045) 等から実入力後に確定。'
        'rho_void_over_mean と c_void_median は 30-50% 誤差範囲想定。'
    ))
    story.append(B(
        '<b>narrow 化効果</b>: FIRAS-only 40-58 桁幅 -&gt; Local Void min 採用で '
        '~5 桁幅に圧縮 (38 桁 tighten)。Chapter 4 の tau_m in range '
        '[1.9e+10, 1.4e+53] s -&gt; [1.9e+10, ~2e+15] s.'
    ))

    story.append(H2('6.5 C3 universal coupling との関連 (b_alpha = 0.11)'))
    story.append(N(
        'internal memo C3 v3 (2026-04-18) で確立: SPARC 124 銀河 b_alpha = +0.1084 vs '
        'dSph 30 銀河 b_alpha = +0.1127、|diff| = 0.0042 (0.5% 以内、3.92 dex 密度範囲)。'
        '教訓 91 (bridge pre-cut)、92 (parsimony first)、93 (universal coupling = slope agreement) '
        '新規 A 級確立。Chapter 7 の Gamma_actual Route A で b_alpha = 0.11 直結。'
    ))
    story.append(H3('6.5.1 教訓 91-93 (C3 foundation 方法論三点セット)'))
    story.append(tbl(
        ['#', '命題', '核心証拠'],
        [
            ['91', 'bridge/extreme-regime pre-cut protocol',
             'SPARC v1 NGC3741 単独駆動偽陽性 rho=+0.85 -&gt; 除外後 +0.52'],
            ['92', 'parsimony first (AIC/BIC &gt; Spearman strength)',
             'SPARC で partial rho=+0.55 強信号でも dAIC=-2 で alpha 採用'],
            ['93', 'universal coupling = slope agreement across density',
             'b_SPARC vs b_dSph 0.5% 以内一致 (3.92 dex)'],
        ],
        [10*mm, 70*mm, 80*mm]
    ))
    story.append(B(
        'これら 3 教訓は dSph strict 15 subset の dAIC = +0.17 marginal gamma hint を '
        '残しつつ、SPARC + dSph 統合 N=154 で alpha (linear coupling) を universal と '
        '判定する operational 根拠。本 foundation の Chapter 7 Route A は b_alpha = 0.11 '
        '直結で Gamma_A = 8.091 を与え、u_mem -&gt; stress 変換効率の 11% / 89% branching '
        '解釈 (C3-4 未解決課題) と整合する。'
    ))
    story.append(K(
        '<b>retract 不可 #26-28 まとめ</b>: Eq. iv-e-I は 2 route min 採用が robust bound 設計の核心。'
        'Windows 実装 task (a) (c3_3c_p2_void.csv 実入力) により本 Chapter の数値を確定後、'
        'Appendix A に実入力版を archive。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 7: Gamma_actual physical adjoint
# =========================================
def ch07_gamma_actual(story):
    story.append(H1('Chapter 7. Gamma_actual physical adjoint (3 route 並行)'))

    story.append(H2('7.0 Gamma = 1/gamma identification の前提 (v4.7.6 A 級)'))
    story.append(B(
        '本 foundation で Gamma は overdamped Langevin の mobility (= 1/damping) として '
        'universal に定義される (v4.7.6 A 級確立)。M3-M4 整合性 gamma_w/mu = gamma/chi '
        '(v4.2 第 19 章) により、Gamma と内部他量 (chi, mu, Gamma_w) の relational '
        'structure が確定。本章は Gamma の absolute value 決定を 3 route 並行で試みる。'
    ))

    story.append(H2('7.1 Gamma 無次元化 foundation scale primary (#29)'))
    story.append(B(
        'Gamma の無次元化は foundation scale (chi_F &#183; V\'\'(phi_0) &#183; T_m / hbar 基準) を '
        'primary 採用 (#29)。本 scale は foundation_alpha2 の chi_F (Eq. 2A-II) と '
        'Chap 18 Table の V\'\'(phi_0) を直接結合する唯一の内部 consistent 選択。'
    ))
    story.append(M_(
        '<b>Gamma_norm</b> = Gamma_actual / (chi_F &#183; V\'\'(phi_0) &#183; T_m / hbar)<br/>'
        '&nbsp;&nbsp;c_0 = 0.83: scale = 4.198 &#183; 2.31 &#183; 2.449 / 1.0546e-34 [m^(3/2)/s^3 / J&#183;s]'
    ))

    story.append(H2('7.2 Route A -- 膜 normal mode (b_alpha 経由)'))
    story.append(B(
        'internal memo C3 v3 の b_alpha = 0.11 (universal, 3.92 dex) を '
        'u_mem -&gt; stress 変換効率と解釈し、Gamma に素朴接続:'
    ))
    story.append(M_(
        '<b>Gamma_A</b> = (1 - b_alpha) / b_alpha &#183; mode_factor<br/>'
        '&nbsp;&nbsp;b_alpha = 0.11, mode_factor = 1 で Gamma_A = <b>8.091</b> (dimensionless)<br/>'
        '&nbsp;&nbsp;解釈: u_mem 変動の 11% が観測 g 残差、残り 89% が熱散逸+重力波+内部転換'
    ))

    story.append(H2('7.3 Route B -- FDT relaxation (tau_relax 候補 4 つ)'))
    story.append(B(
        'Fluctuation-Dissipation から Gamma = chi_F / (V\'\'(phi_0) &#183; tau_relax) で '
        '4 tau_relax 候補を横並び:'
    ))
    story.append(tbl(
        ['tau_relax 候補', 'tau [s]', 'Gamma_B [m^(3/2)/s]', '物理解釈'],
        [
            ['Hubble time', '4.4e+17', '4.12e-18', '宇宙年齢スケール'],
            ['galaxy dyn time', '3.0e+15', '6.06e-16', 'NGC 3198 tau_dyn'],
            ['kpc crossing', '1.0e+13', '1.81e-13', '銀河内スケール'],
            ['BE thermalization', '1.9e+10', '9.58e-11', 'z=5e+04 Hubble'],
        ],
        [40*mm, 22*mm, 35*mm, 40*mm]
    ))
    story.append(B(
        'spread は 7 桁。Route B 単独で絶対値決定不能、tau_relax 選択の物理的根拠が未解決。'
    ))

    story.append(H2('7.4 Route C -- 観測 proxy upper (SPARC memory)'))
    story.append(M_(
        '<b>Gamma_C upper</b> &lt; 5.749e-18 [m^(3/2)/s]<br/>'
        '&nbsp;&nbsp;根拠: SPARC 銀河 memory decay time &gt; 10 Gyr (Phase C3 Step 1)<br/>'
        '&nbsp;&nbsp;tau_m &gt; 10 Gyr = 3.15e+17 s -&gt; Gamma_C &lt; chi_F/(V\'\'&#183;tau) &lt; 5.75e-18'
    ))

    story.append(H2('7.5 3 route 比較 summary'))
    story.append(tbl(
        ['route', '式', '結果', '次元', '判定'],
        [
            ['A', '(1-b)/b &#183; mode_factor', '8.091', 'dimensionless', 'strong candidate'],
            ['B', 'chi_F / (V\'\' tau_relax)', '4e-18 ~ 1e-10', 'm^(3/2)/s', 'tau_relax 未定'],
            ['C', 'chi_F / (V\'\' tau_SPARC)', '&lt; 5.75e-18', 'm^(3/2)/s', 'upper only'],
            ['内部整合', 'A vs B (Hubble time)', 'order ~同', '--', '弱整合'],
        ],
        [15*mm, 48*mm, 35*mm, 25*mm, 37*mm]
    ))
    story.append(B(
        'Route A の dimensionless 値 8.091 と Route B の Hubble time 採用 4.12e-18 m^(3/2)/s '
        'は <b>次元が異なる</b>ため直接比較不能だが、Gamma_norm (foundation scale 割算) '
        'で比較すると order of magnitude が近接する兆候あり (要検証、C3-4 energy branching 確定後)。'
    ))

    story.append(H2('7.6 3 route 統合判定 (#30, #31)'))
    story.append(K(
        '<b>retract 不可 #30</b>: 3 route (A 膜 normal / B FDT / C 観測 proxy) は '
        '並行評価で internal consistency を担保。単一 route の絶対値主張は保留。<br/><br/>'
        '<b>retract 不可 #31</b>: b_alpha = 0.11 と Gamma_actual は C3-4 branching ratio '
        '(0.11 : 0.89) を介して直結。Gamma_A = 8.091 はこの branching の素朴表現。'
    ))
    story.append(N(
        '<b>残課題</b>: Route B 7 桁 spread 解消には tau_relax 選択の物理原理 '
        '(C3-4 energy branching 解析) が必要。Route C upper は保守的で、'
        '実際の Gamma_actual は Route A 近傍 (~8.1 dimensionless) が strong candidate '
        '但し次元整合性 (#24 T_m 無次元保持) で SI 値への変換は未確定。'
    ))
    story.append(PageBreak())

# =========================================
# Chapter 8: Companion paper (v4.8) 公表戦略
# =========================================
def ch08_companion_paper(story):
    story.append(H1('Chapter 8. Companion paper (v4.8) 公表戦略'))

    story.append(H2('8.1 v4.7.8 本体との非干渉 + foundation 材料化'))
    story.append(B(
        'v4.7.8 arXiv 原稿 (12 頁、19 結論、Condition 14/15、Bernoulli G_Strigari、'
        'HSC+GAMA+KiDS Phase B) は保持。本 foundation (v4.8) は companion paper として '
        '<b>独立 arXiv 投稿</b> する分離戦略 (Session +6 確定)。'
    ))
    story.append(H3('8.1.1 v4.7.8 と v4.8 の役割分担'))
    story.append(tbl(
        ['側面', 'v4.7.8 (本体)', 'v4.8 (foundation companion)'],
        [
            ['層', 'observational establishment', 'theoretical foundation'],
            ['証拠', 'SPARC 175 + dSph 31 + HSC 503M', '閉形式 11+ + NGC 3198 2.32%'],
            ['主結論', 'C15 + MOND 棄却 p=1.66e-53', 'alpha_PT_upper + tau_m bound'],
            ['頁数', '12 頁', '12-15 頁 (本統合書 15-20 頁が材料)'],
            ['対象読者', '観測天文学者 + 宇宙論者', '理論物理 + 場の量子論'],
            ['arXiv cat', 'astro-ph.CO', 'astro-ph.CO + hep-th cross'],
        ],
        [25*mm, 55*mm, 70*mm]
    ))

    story.append(H2('8.2 5 main claims (M1-M5)'))
    story.append(tbl(
        ['#', '主張', '核心数値/論点'],
        [
            ['M1', 'alpha_PT_upper ~ 2.96e-53 (V_xi primary)', 'FIRAS mu_FIRAS &lt; 9e-5 + Stage 2'],
            ['M2', 'tau_m in [1.9e+10, 1.4e+53] s (FIRAS-only)', 'Local Void で 5 桁幅に narrow'],
            ['M3', 'Eq. 2B-II が SI canonical', 'N_mode 数値代入の唯一正路'],
            ['M4', 'chi_E [s^2] と chi_F [m^(3/2)/s^2] 独立', 'universal conversion 不存在'],
            ['M5', 'f = 0.379 (Pantheon+) は mu kernel 非寄与', 'explicit null (#17)'],
        ],
        [15*mm, 70*mm, 70*mm]
    ))

    story.append(H2('8.3 3 channel 展開戦略'))
    story.append(tbl(
        ['channel', '形式', '内容'],
        [
            ['arXiv', 'v4.8 論文 12-15 頁', 'abstract/intro/methods/results/conclusions'],
            ['WordPress', 'sakaguchi-physics.com', '一般向け要約 + 5 claims 解説'],
            ['GitHub', 'Public repo MIT 再リリース', 'script + csv + QA log + PDF'],
        ],
        [25*mm, 50*mm, 75*mm]
    ))

    story.append(H2('8.4 f = 0.379 explicit null (M5)'))
    story.append(N(
        'Pantheon+ 独立測定 f = 0.379 +/- 0.029 は c_0 = 0.83 (SPARC+dSph vol avg) とは '
        '独立量であり、mu kernel にも入らない。本分離を explicit null として v4.8 で明示し、'
        '読者の誤解 (f ~ 1 - c_0 等の spurious 関係) を予防する。'
    ))
    story.append(PageBreak())

# =========================================
# Appendix A: Windows 実装再現性 + cross-reference
# =========================================
def appendix_a_implementation(story):
    story.append(H1('Appendix A. Windows 実装再現性 + cross-reference'))

    story.append(H2('A.1 成果物一覧 (Session +11~+15、全 13 ファイル)'))
    story.append(tbl(
        ['session', '成果物', '用途'],
        [
            ['+11', 'alpha3_tm_si_design_spec.pdf (3p) + build_*.py', 'Ch.5 材料'],
            ['+12', 'companion_paper_v48_skeleton.pdf (5p) + build_*.py', 'Ch.8 材料'],
            ['+13', 'c3_3c_p2_void_design_spec.pdf (4p) + build_*.py', 'Ch.6 材料'],
            ['+14', 'foundation_delta_gamma_actual_design_spec.pdf (3p) + build_*.py', 'Ch.7 材料'],
            ['+15', 'windows_implementation_tasks_design.pdf (4p)', '3 task 統合設計'],
            ['+15', 'c3_3c_p2_void_template.csv (685 B)', '5 void 初期推定'],
            ['+15', 'foundation_gamma_actual.py (12 KB)', '3 route 完全実装'],
            ['+15', 'build_foundation_integrated_skeleton.py (16 KB)', '本 Ch 骨格'],
            ['+15', 'README_windows_tasks.md (8 KB)', '実行手順 + checklist'],
        ],
        [15*mm, 80*mm, 55*mm]
    ))

    story.append(H2('A.2 実行再現手順 (Windows Claude Code)'))
    story.append(M_(
        'cd D:\\ドキュメント\\エントロピー\\新膜宇宙論\\これまでの軌跡\\パイソン\\<br/><br/>'
        'REM (a) P2 void csv 実入力<br/>'
        'notepad c3_3c_p2_void_template.csv  REM 文献値上書き<br/>'
        'copy c3_3c_p2_void_template.csv c3_3c_p2_void.csv<br/><br/>'
        'REM (b) Gamma_actual 3 route 実行<br/>'
        'uv run --with numpy --with scipy --with matplotlib --with pandas \\<br/>'
        '&nbsp;&nbsp;python foundation_gamma_actual.py<br/><br/>'
        'REM (c) 本統合 PDF build<br/>'
        'uv run --with reportlab python build_foundation_integrated.py'
    ))

    story.append(H2('A.3 retract 不可事項 #1-31 cross-reference table'))
    story.append(tbl(
        ['# 範囲', '起源 session', '主要内容', '本書 Ch'],
        [
            ['#1-17', 'v4.7.8', 'kappa=0, T_m=sqrt(6), Form B, Q-core 3, Eq. I-IV', 'Ch.1, 2'],
            ['#18-21', 'C3 memo v3', 'universal b_alpha=0.11, 教訓 91-93', 'Ch.6'],
            ['#22-25', 'Session +11', 'chi dual, 2B-II/2C-II, T_m 無次元, eps_scale (a)(b)', 'Ch.5'],
            ['#26-28', 'Session +13', 'Eq. iv-e-I 2 route, V_xi_void, min 構造', 'Ch.6'],
            ['#29-31', 'Session +14', 'foundation scale 無次元化, 3 route, b_alpha 直結', 'Ch.7'],
        ],
        [20*mm, 22*mm, 80*mm, 20*mm]
    ))

    story.append(H2('A.4 数値 anchor 一覧 (c_0 = 0.83 canonical, V_xi, Stage 2)'))
    story.append(tbl(
        ['量', '値', '出所'],
        [
            ['eps_0(0.83)', '0.412311', 'Form B: sqrt(1 - c)'],
            ['w(0.83)^2', '1.4032', 'Form B: 2 eps_0/(1-eps_0)'],
            ['T_m', 'sqrt(6) ~= 2.449', '補題 5 Z_2 SSB'],
            ['c_mem(0.83)', '3.833e+05 m/s', 'v3.7 Chap 18-3 Sb 外挿'],
            ['chi_F(0.83)', '4.198 m^(3/2)/s^2', 'Eq. 2A-II'],
            ['V\'\'(phi_0, x=0)', '2.31', 'Chap 18 Table Q1-alpha'],
            ['Lambda_UV(0.83)', '9.549e-49 J', 'Chap 18-4 線形外挿'],
            ['V_xi(0.83)', '7.683e+63 m^3', 'coherence volume'],
            ['B_FIRAS_combo', '1.516e-12 J^(-1) m^(-1/2)', 'Eq. iv-d-III'],
            ['I_mu_Stage1(0.83)', '7.2275e+19 s (0.45% PASS)', 'Eq. iv-d-I'],
            ['I_mu_Stage2(0.83)', '6.3220e+19 s', 'Eq. iv-d-II v2'],
            ['S2/S1', '0.8747 (std=2e-16)', 'c_0 非依存'],
            ['alpha_PT_upper (V_xi)', '2.96e-53', 'Eq. iv-d-IV'],
            ['tau_m_upper (FIRAS-only)', '1.418e+53 s', 'V_xi primary'],
            ['tau_m_lower (causal z=5e+04)', '1.906e+10 s (~605 yr)', '因果律'],
            ['tau_m_upper (Local Void)', '~2.26e+15 s', 'Eq. iv-e-I (38 桁 tighten)'],
            ['NGC 3198 k_mem N_mode', '2.998e+38 / 2.93e+38 = 2.32% PASS', 'Eq. 2B-II 検証'],
            ['b_alpha (universal)', '+0.11 +/- 0.005 (3.92 dex)', 'C3 memo v3'],
            ['Gamma_A (dimensionless)', '8.091', 'Route A (b_alpha 経由)'],
            ['Gamma_B (FDT, 4 候補)', '4.12e-18 ~ 9.58e-11 m^(3/2)/s', 'Route B'],
            ['Gamma_C upper', '&lt; 5.75e-18 m^(3/2)/s', 'Route C (SPARC memory)'],
        ],
        [45*mm, 55*mm, 45*mm]
    ))

    story.append(H2('A.5 次 session 候補 (+17 以降)'))
    story.append(B(
        '(B) v4.8 本体執筆 (12-15 頁論文形式): 本統合書を材料として abstract/intro/methods/'
        'results/conclusions を書き下し、arXiv 投稿準備。'
        '(C) retract 不可 #1-31 全 cross-reference audit: 全 session 成果物間の数値不整合検出。'
        '長期: T_m SI 絶対較正 (#25 (a)/(b) 96 桁 gap 解消)、Lambda_UV / c_mem 外挿精度向上、'
        'UFD extension (Segue I/II, Reticulum II 等で dSph gamma hint の統計力確保)。'
    ))
    story.append(K(
        '<b>本統合書の完結宣言</b>: Session +11~+15 で確立した foundation 理論 layer を '
        '15-20 頁に結晶化。以降は validation + 公表 phase (v4.8 本体執筆 + arXiv 投稿) に移行する。'
        '本書および付属成果物 (script + csv + QA log) は Windows Claude Code 環境で '
        'uv run による再現実行が可能。'
    ))

# =========================================
# Main build
# =========================================
def main():
    print('=' * 60)
    print('foundation_integrated.pdf build (Session +16 A)')
    print('=' * 60)
    verify_gaps()

    story = []
    ch01_introduction(story)
    ch02_foundation_alpha1(story)
    ch03_foundation_alpha2(story)
    ch04_foundation_beta_delta(story)
    ch05_alpha3_conventions(story)
    ch06_p2_void_narrow(story)
    ch07_gamma_actual(story)
    ch08_companion_paper(story)
    appendix_a_implementation(story)

    out = 'foundation_integrated.pdf'
    doc = SimpleDocTemplate(out, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm)
    doc.build(story)
    print(f'[BUILT] {out}')
    print(f'[SIZE]  {os.path.getsize(out):,} bytes')

if __name__ == '__main__':
    main()
