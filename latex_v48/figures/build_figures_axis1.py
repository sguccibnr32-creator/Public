#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_figures_axis1.py
======================
arXiv v4.9 figure sub-round: 4 figures for axis_1 SPARC b_alpha.

Generates:
  - fig_axis1_jackknife.pdf  (J1 LOO histogram, 124 replicates)
  - fig_axis1_bootstrap.pdf  (J2 bootstrap histogram with 95% CI band)
  - fig_axis1_filter5.pdf    (J3 5-filter sensitivity bar chart)
  - fig_axis1_xmethod4.pdf   (Lesson 94 cross-method 5-scenario comparison)

Compliance:
  - v4.4 spec: IPAexGothic font, axes.unicode_minus=False, ASCII hyphen U+002D
  - rule 1:    SHA-256 verify of all 4 input CSVs (anchor IMMUTABLE preservation)
  - rule #26:  4 independent route visualization (multi-route minimum)
  - rule 92:   single-script parsimony

Usage (Claude Code / Windows):
  cd "E:/GitHub repo/github_workspace/Public/latex_v48/figures/"
  uv run --with matplotlib --with numpy --with pandas \\
      python build_figures_axis1.py

Source CSVs (forensic chain trace):
  jackknife: forensic_anchors/section2_5_v0_4a_validation/
             jackknife_axis_1_results.csv          SHA: 28886a65ca3de677...
  bootstrap: forensic_anchors/section2_5_v0_4a_validation/
             bootstrap_axis_1_results.csv          SHA: 5568321529c97f68...
  filter5:   forensic_anchors/section2_5_v0_4a_validation/
             per_filter_sensitivity.csv            SHA: 6cb3ee1524658782...
  xmethod4:  forensic_anchors/section2_axis_1_operational_closure/
             per_filter_sensitivity_extension.csv  SHA: af3809ffddde7325...
"""

import os
import sys
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ensure stdout UTF-8 (handoff memo §3.13 precedent)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# Font setup (v4.4 spec §4-2 compliance)
# ============================================================================

IPAGOTHIC_CANDIDATES = [
    # Windows / Claude Code primary (TeX Live 2026, verified by Discovery)
    "C:/texlive/2026/texmf-dist/fonts/truetype/public/ipaex/ipaexg.ttf",
    # Linux fallback (claude.ai container, optional dry-run)
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
]

IPAGOTHIC_PATH = next((p for p in IPAGOTHIC_CANDIDATES if os.path.exists(p)), None)
if IPAGOTHIC_PATH is None:
    print("[ERROR] IPAGothic / IPAexGothic font not found. Searched:")
    for p in IPAGOTHIC_CANDIDATES:
        print(f"  - {p}")
    sys.exit(1)

print(f"[OK] font: {IPAGOTHIC_PATH}")
ipa_font = fm.FontProperties(fname=IPAGOTHIC_PATH)

# v4.4 spec §4-2: avoid U+2212 (typographic minus) -> ASCII hyphen U+002D
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Paths and SHA verification (rule 1: anchor IMMUTABLE preservation)
# ============================================================================

ANCHOR_BASE = os.environ.get(
    'ANCHOR_BASE',
    'E:/GitHub repo/github_workspace/Public/forensic_anchors',
)
OUTDIR = os.environ.get(
    'OUTDIR',
    'E:/GitHub repo/github_workspace/Public/latex_v48/figures',
)
os.makedirs(OUTDIR, exist_ok=True)

# Note: 3 CSVs in section2_5_v0_4a_validation/, 1 in section2_axis_1_operational_closure/
CSV_FILES = {
    'jackknife': (
        'section2_5_v0_4a_validation/jackknife_axis_1_results.csv',
        '28886a65ca3de677',
    ),
    'bootstrap': (
        'section2_5_v0_4a_validation/bootstrap_axis_1_results.csv',
        '5568321529c97f68',
    ),
    'filter5': (
        'section2_5_v0_4a_validation/per_filter_sensitivity.csv',
        '6cb3ee1524658782',
    ),
    'xmethod4': (
        'section2_axis_1_operational_closure/per_filter_sensitivity_extension.csv',
        'af3809ffddde7325',
    ),
}


def verify_sha(rel_path, expected_prefix):
    full = os.path.join(ANCHOR_BASE, rel_path)
    if not os.path.exists(full):
        print(f"[ERROR] CSV not found: {full}")
        sys.exit(1)
    with open(full, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if not actual.startswith(expected_prefix):
        print(f"[ERROR] SHA mismatch: {rel_path}")
        print(f"  expected prefix: {expected_prefix}")
        print(f"  actual:          {actual[:16]}")
        sys.exit(1)
    print(f"[OK] SHA verify: {rel_path} ({actual[:16]}...)")
    return full


print("=" * 70)
print("Phase 0: SHA-256 verification (rule 1 compliance)")
print("=" * 70)
paths = {k: verify_sha(v[0], v[1]) for k, v in CSV_FILES.items()}
print()


# ============================================================================
# IMMUTABLE values (handoff memo frozen reference set)
# ============================================================================

B_ALPHA_BASELINE = 0.108442979149252  # G1 / J4 bit-exact
BOOTSTRAP_CI_LOW = 0.0790              # J2 95% CI low
BOOTSTRAP_CI_HIGH = 0.1388             # J2 95% CI high

# Common figure settings
FIGSIZE = (5.5, 3.5)  # inches; ~140mm wide, fits arXiv textwidth (0.9*textwidth)
DPI_PDF = 300

# Color palette (matches v4.4 spec convention)
COL_BAR = '#4a90d9'      # primary bar / histogram fill (blue)
COL_RED = '#e94560'      # baseline / highlight
COL_GREEN = '#28a745'    # CI band fill
COL_ORANGE = '#f08c00'   # secondary highlight (positive drift)
COL_GRAY = '#888888'     # secondary text / annotations


def apply_font(ax, title=None):
    """Apply IPAexGothic to all text elements of an Axes (v4.4 §4-2)."""
    if title is not None:
        ax.set_title(title, fontproperties=ipa_font, fontsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ipa_font)
    if ax.get_legend() is not None:
        for text in ax.get_legend().get_texts():
            text.set_fontproperties(ipa_font)


def save_fig(fig, name):
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=DPI_PDF)
    plt.close(fig)
    size_kb = os.path.getsize(out) / 1024
    print(f"[OK] saved: {out} ({size_kb:.1f} KB)")


# ============================================================================
# F1: jackknife LOO distribution
# ============================================================================

def plot_jackknife():
    print("--- F1: jackknife ---")
    df = pd.read_csv(paths['jackknife'])
    n = len(df)
    n_outlier = int(df['is_3sigma_outlier'].sum())
    std_loo = df['b_alpha_loo'].std()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI_PDF)
    ax.hist(df['b_alpha_loo'], bins=25, color=COL_BAR,
            edgecolor='black', linewidth=0.4, alpha=0.85)
    ax.axvline(B_ALPHA_BASELINE, color=COL_RED, linestyle='--',
               linewidth=1.5,
               label=f'baseline b_alpha = {B_ALPHA_BASELINE:.6f}')

    ax.set_xlabel('b_alpha (LOO replicate)', fontproperties=ipa_font)
    ax.set_ylabel('frequency', fontproperties=ipa_font)
    title = (f'axis_1 SPARC b_alpha jackknife LOO distribution\n'
             f'(N={n}, 3sigma outliers={n_outlier}, std_loo={std_loo:.6f})')
    apply_font(ax, title=title)
    ax.legend(prop=ipa_font, loc='upper right', fontsize=8)

    # SHA prefix annotation (forensic chain trace)
    ax.text(0.02, 0.97,
            'source: jackknife_axis_1_results.csv (SHA prefix: 28886a65)',
            transform=ax.transAxes, fontsize=6, color=COL_GRAY,
            verticalalignment='top', fontproperties=ipa_font)

    fig.tight_layout()
    save_fig(fig, 'fig_axis1_jackknife.pdf')


# ============================================================================
# F2: bootstrap distribution with 95% CI band
# ============================================================================

def plot_bootstrap():
    print("--- F2: bootstrap ---")
    df = pd.read_csv(paths['bootstrap'])
    n = len(df)
    median_b = df['b_alpha'].median()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI_PDF)
    counts, bins, _ = ax.hist(df['b_alpha'], bins=50, color=COL_BAR,
                              edgecolor='black', linewidth=0.3, alpha=0.85)

    ymax = counts.max() * 1.05
    # 95% CI band (J2 immutable: [0.0790, 0.1388])
    ax.axvspan(BOOTSTRAP_CI_LOW, BOOTSTRAP_CI_HIGH,
               alpha=0.2, color=COL_GREEN,
               label=f'95% CI [{BOOTSTRAP_CI_LOW:.4f}, {BOOTSTRAP_CI_HIGH:.4f}]')
    ax.axvline(B_ALPHA_BASELINE, color=COL_RED, linestyle='--',
               linewidth=1.5,
               label=f'baseline b_alpha = {B_ALPHA_BASELINE:.6f}')
    ax.set_ylim(0, ymax)

    ax.set_xlabel('b_alpha (bootstrap resample)', fontproperties=ipa_font)
    ax.set_ylabel('frequency', fontproperties=ipa_font)
    title = (f'axis_1 SPARC b_alpha bootstrap distribution\n'
             f'(N={n:,} resamples, median={median_b:.6f})')
    apply_font(ax, title=title)
    ax.legend(prop=ipa_font, loc='upper right', fontsize=8)

    ax.text(0.02, 0.97,
            'source: bootstrap_axis_1_results.csv (SHA prefix: 55683215)',
            transform=ax.transAxes, fontsize=6, color=COL_GRAY,
            verticalalignment='top', fontproperties=ipa_font)

    fig.tight_layout()
    save_fig(fig, 'fig_axis1_bootstrap.pdf')


# ============================================================================
# F3: 5-filter sensitivity bar chart
# ============================================================================

def plot_filter5():
    print("--- F3: 5-filter ---")
    df = pd.read_csv(paths['filter5'])
    # Restrict to executable scenarios only
    if 'executable' in df.columns:
        df = df[df['executable'] == True].copy()

    def short_label(s):
        s = s.replace('baseline_all_filters_active', 'baseline')
        s = s.replace('filter_', 'F')
        s = s.replace('_disable_', '_')
        return s[:22]

    labels = [short_label(s) for s in df['scenario']]
    drifts = df['drift_rel_pct'].astype(float).values

    # Color: baseline gray, positive drift orange, negative blue
    colors = []
    for s, d in zip(df['scenario'], drifts):
        if 'baseline' in s:
            colors.append(COL_GRAY)
        elif d > 0:
            colors.append(COL_ORANGE)
        else:
            colors.append(COL_BAR)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI_PDF)
    bars = ax.bar(range(len(labels)), drifts, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.6)

    # Label each bar with drift value
    drift_max = max(abs(d) for d in drifts) if len(drifts) else 1.0
    label_pad = drift_max * 0.04 if drift_max > 0 else 0.05
    for bar, val in zip(bars, drifts):
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = label_pad if h >= 0 else -label_pad
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f'{val:+.2f}%', ha='center', va=va, fontsize=7,
                fontproperties=ipa_font)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_xlabel('filter scenario', fontproperties=ipa_font)
    ax.set_ylabel('drift_rel_pct (%)', fontproperties=ipa_font)
    title = ('axis_1 SPARC J3 5-filter sensitivity\n'
             '(drift relative to baseline 124-row sample)')
    apply_font(ax, title=title)

    ax.text(0.02, 0.97,
            'source: per_filter_sensitivity.csv (SHA prefix: 6cb3ee15)',
            transform=ax.transAxes, fontsize=6, color=COL_GRAY,
            verticalalignment='top', fontproperties=ipa_font)

    fig.tight_layout()
    save_fig(fig, 'fig_axis1_filter5.pdf')


# ============================================================================
# F4: 4-method cross-method comparison (Lesson 94)
# ============================================================================

def plot_xmethod4():
    print("--- F4: 4-method cross-method ---")
    df = pd.read_csv(paths['xmethod4'])

    def short_label(s):
        return (s.replace('A0_', 'A0:').replace('A1_', 'A1:')
                 .replace('A2_', 'A2:').replace('A3_', 'A3:')
                 .replace('A4_', 'A4:').replace('_', ' '))

    labels = [short_label(s) for s in df['scenario']]
    b_alphas = df['b_alpha'].astype(float).values
    se_vals = df['SE_b_alpha'].astype(float).values
    delta_rel = df['delta_vs_baseline_rel'].astype(float).values * 100  # to percent

    # Color: baseline gray, A1 (aggregation) red highlight (Lesson 94 dominance),
    # A2/A3/A4 (imputation) blue
    colors = []
    for s in df['scenario']:
        if 'baseline' in s:
            colors.append(COL_GRAY)
        elif 'median_V2r' in s:
            colors.append(COL_RED)  # dominant aggregation effect
        else:
            colors.append(COL_BAR)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI_PDF)
    x = np.arange(len(labels))
    bars = ax.bar(x, b_alphas, yerr=se_vals, color=colors,
                  edgecolor='black', linewidth=0.5,
                  error_kw={'linewidth': 0.8, 'capsize': 3, 'ecolor': 'black'})

    # Baseline reference line
    ax.axhline(B_ALPHA_BASELINE, color=COL_RED, linestyle='--',
               linewidth=1.0, alpha=0.5,
               label=f'A0 baseline = {B_ALPHA_BASELINE:.6f}')

    # Annotate delta_rel
    bar_top_max = (b_alphas + se_vals).max()
    txt_offset = bar_top_max * 0.04
    for bar, b, se_v, d in zip(bars, b_alphas, se_vals, delta_rel):
        if abs(d) < 0.01:
            txt = '0%'
        else:
            txt = f'{d:+.2f}%'
        ax.text(bar.get_x() + bar.get_width() / 2, b + se_v + txt_offset,
                txt, ha='center', va='bottom', fontsize=7,
                fontproperties=ipa_font)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_xlabel('scenario (A0 baseline / A1 aggregation / A2-A4 imputation)',
                  fontproperties=ipa_font)
    ax.set_ylabel('b_alpha', fontproperties=ipa_font)
    title = ('axis_1 SPARC 5-scenario cross-method comparison (Lesson 94)\n'
             'aggregation choice (-34.62%) >> imputation choice (+/-20%)')
    apply_font(ax, title=title)
    ax.legend(prop=ipa_font, loc='upper right', fontsize=8)

    ax.text(0.02, 0.97,
            'source: per_filter_sensitivity_extension.csv (SHA prefix: af3809ff)',
            transform=ax.transAxes, fontsize=6, color=COL_GRAY,
            verticalalignment='top', fontproperties=ipa_font)

    fig.tight_layout()
    save_fig(fig, 'fig_axis1_xmethod4.pdf')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 1: figure generation (4 plots)")
    print("=" * 70)
    plot_jackknife()
    plot_bootstrap()
    plot_filter5()
    plot_xmethod4()
    print()
    print("=" * 70)
    print("Phase 2: post-build verification (manual checks below)")
    print("=" * 70)
    print("[1] visual inspection: open each PDF, verify no black-square chars")
    print("[2] grep markers (16 total): see latex_v49_figure_patch.md")
    print("[3] LaTeX integration: apply patches in latex_v49_figure_patch.md")
    print()
    print("[ALL OK] 4 figures generated.")
