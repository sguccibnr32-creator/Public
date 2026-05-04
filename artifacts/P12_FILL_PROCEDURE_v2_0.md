# P12 Fill Procedure v2.0 — `b_alpha_3axis_audit()` 公式 spec impl 手順書

作成日: 2026-05-04
発信元 session: T15+T16 受領 + T17/T18 抽出依頼確定 session
supersedes: P12_FILL_PROCEDURE_v1_0.md (SHA d9835d8e..., outdated post T15+T16 finding)
encoding: UTF-8 / LF only / no BOM
Unicode 禁止文字 audit: v4.4 PDF spec compliant (ALL CLEAN)

---

## 0. v1.0 -> v2.0 supersession 理由

v1.0 (T15+T16 受領前作成) は P12 の axis decomposition と estimator pattern を仮定で書いていた。T15+T16 verbatim 受領後、3 件の重大 finding が判明:

1. anchor 7 §2.5.5 公式 axis 1/2/3 semantic は v1.0.2 stub (`axis_1_SPARC` / `axis_2_dSph` / `axis_3_combined`) と完全 mismatch。公式 axes は continuity check / reversal trend / universal slope の 3 種類 audit
2. 真の estimator は **partial OLS** (numpy.linalg.lstsq、log_rh nuisance partialled out)。v1.0 で想定した simple linregress では baseline 0.1084/0.1127 reproduce 不可
3. b_α formal definition は anchor 7 §2.5.5 内に NOT FOUND。C3-A5 PDF SHA `69fb1a95...` §4.3 が canonical source

v2.0 では上記 3 件の finding を反映し、option 2 path (T17+T18 追加抽出 -> v1.0.3 で P12 公式 spec 完成) の executable spec として再構築する。

---

## 1. 目的

P12 を本 session 内で **公式 spec impl** として完成させる。これにより:

- `b_alpha_3axis_audit()` の NotImplementedError 解除
- Phase C3 v3 §4.3 universal coupling baseline (b_α SPARC=0.1084, dSph=0.1127, |Δ|=0.0042) の Python script 内 reproducibility 確立
- AC4 (b_α |Δ| <= 0.005) が deferred -> evaluated に昇格
- v1.0.4 への deferred 不要、v1.0.3 atomic completion で本 session 完結

---

## 2. 現状実装 (v1.0.2 line 番号付き、変更なし)

### 2.1 関数 stub (line 1424-1437)

```python
def b_alpha_3axis_audit(
    sparc_df: pd.DataFrame, dsph_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute b_alpha on 3 axes (per anchor 7 §2.5.5):
      axis 1: SPARC density-weighted
      axis 2: dSph density-weighted
      axis 3: combined (Phase C3 cross-paper coherence axis)
    Returns dict with all three plus diff statistics.
    """
    # TODO_USER_VERIFY: exact b_alpha estimator formula on each axis.
    raise NotImplementedError(
        "b_alpha 3-axis audit requires anchor 7 §2.5.5 formula."
    )
```

NOTE: docstring 内の "axis 1: SPARC density-weighted" 等は v1.0.2 当時の暫定記述。T15+T16 finding 反映後、実際の公式 axes は continuity / reversal / universal slope の 3 種類。

### 2.2 既配線済 constants (line 151-154)

```python
B_ALPHA_SPARC_BASELINE: float = 0.1084
B_ALPHA_DSPH_BASELINE: float = 0.1127
B_ALPHA_ABS_DIFF_BASELINE: float = 0.0042
```

### 2.3 caller expectation (line 1500-1517)

`run_dsph_audit` 内で return dict のキー `axis_1_SPARC` と `axis_2_dSph` を読む defensive guard あり。v1.0.3 では公式 axes (Axis 1/2/3) と互換 wrapper として `axis_1_SPARC` / `axis_2_dSph` を別 keys として返すか、caller side を更新するかの判断必要 (§7 で確定)。

### 2.4 AC4 dependency (line 1689, 1738-1747)

`abs_diff = dsph_block.get("b_alpha_abs_diff")` を読み取る。`abs_diff = |b_alpha_sparc_value - b_alpha_dsph_value|` で、AC4 threshold 0.005 と比較。

---

## 3. T15 finding summary (受領済 verbatim 内部化記録)

source: anchor 7 §2.5.5 (`J_system_paper_section2_5_v0.1.md`, SHA `9e03f53e`, L561-682)

### 3.1 公式 3 axes (anchor 7 §2.5.5 verbatim)

| 公式 axis | spec 記述 | operational test |
|---|---|---|
| Axis 1 | extreme regime continuity check | c<0.30 Strigari 領域で finite/divergent 判定 |
| Axis 2 | dSph 28/31 reversal trend reproduction | J0 minimal form baseline 比較 |
| Axis 3 | universal slope b_alpha=0.11 emergence audit | SPARC + dSph integrated, log-log slope fit, Phase C3 v3 §4.3 0.11±0.005 比較 |

### 3.2 v1.0.2 stub axis_{1,2,3} との mismatch

| v1.0.2 stub key | semantic | 公式 axis 対応 |
|---|---|---|
| axis_1_SPARC | SPARC-only b_alpha (anchor 19 §1.5 baseline 0.1084 reproducer) | 公式 Axis 3 の SPARC 側 component に相当 (公式 Axis 1 とは別物) |
| axis_2_dSph | dSph-only b_alpha (baseline 0.1127 reproducer) | 公式 Axis 3 の dSph 側 component (公式 Axis 2 とは別物) |
| axis_3_combined | SPARC + dSph integrated b_alpha | 公式 Axis 3 と semantic 整合 |

### 3.3 method 記述 verbatim (anchor 7 §2.5.5 L608-609)

```
method: selected form を SPARC + dSph 統合 sample に適用、
log-log slope fit で b_alpha を re-derive、Phase C3 v3 値との agreement check。
```

### 3.4 anchor 7 §2.5.5 内 NOT FOUND items

- b_alpha formal closed-form 数式定義 -> C3-A5 PDF §4.3 へ deferred (T17 抽出対象)
- axis 1 (continuity check) operational threshold -> T17.C
- axis 2 (reversal trend) baseline metric -> T17.C
- axis 3 (universal slope) significance criterion -> T17.D

---

## 4. T16 finding summary (受領済 verbatim 内部化記録)

source: phase_c3_step3_dsph_gamma_vs_alpha.py
SHA256: `c51c72f07d1b66636d695cc4cf69d8b4d0163a5fa3cc48836263a06666a295fb`
location: `D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\`

### 4.1 真の estimator pattern (T16.B+E verbatim)

```
target:    delta_primary = log10(g_obs / gc_C15)         # SPARC
           delta_dSph_primary = log10(g_obs / G_Strigari)  # dSph
feature:   log_umem_alpha = 2.0 * log10(rho_gal)
nuisance:  log_rh                                          # partialled out
estimator: numpy.linalg.lstsq([1, lu_a, log_rh], delta)[0][1]
         = b_alpha_partial (3-feature OLS slope at lu_a column index 1)
```

### 4.2 sample preparation (T16.F verbatim)

```python
# SPARC pipeline:
sparc = sparc[sparc['Q'] < 3]
sparc['Mstar'] = sparc['Upsilon_d'] * sparc['L36'] * 1e9
sparc['Mgas']  = 1.33 * sparc['MHI'] * 1e9
sparc['Mbar']  = sparc['Mstar'] + sparc['Mgas']
sparc['hR']    = sparc['Rdisk']
sparc['r_h']   = 1.68 * sparc['hR']
sparc['rho_gal'] = sparc['Mbar'] / (4.0/3.0 * np.pi * sparc['r_h']**3)
sparc = sparc[~sparc['Galaxy'].isin(BRIDGE_GALAXIES)]   # BRIDGE_GALAXIES: 4 galaxy
# -> SPARC 175 - bridge 4 - Q>=3 cut = 124 (推定)
```

```python
# dSph pipeline:
# Sgr excluded from 31 -> 30 (具体 line は T18.C で抽出)
```

### 4.3 dependencies (T16.D verbatim)

```python
import numpy as np
import pandas as pd
from scipy import stats   # LRT / Spearman 用のみ、b_alpha 自体は np.linalg.lstsq
import matplotlib.pyplot as plt
```

statsmodels 不要。numpy + scipy + pandas + matplotlib のみ。companion script v1.0.2 は既に numpy/pandas/scipy import 済 (line 73-86 程度) のため追加 import は scipy.stats のみ (もし未 import なら)。

### 4.4 constants (T16.F verbatim L80-101)

```python
A0 = 1.2e-10                                # m/s^2
A0_KPC = 1.2e-10 * 3.086e19 / 1e6           # (km/s)^2/kpc
G_STRIGARI = 0.228 * A0                     # m/s^2 = 2.736e-11
C15_COEF = 0.584
C15_UPSILON_EXP = -0.361
TWO_EPS_STAR = 2.0 * (1.0 - np.sqrt(1e-6))  # ~= 2.0
LAM_GRID = np.logspace(-11, -2, 500)        # 500 points (gamma fit のみ使用)
BRIDGE_GALAXIES = {'ESO444-G084', 'NGC2915', 'NGC1705', 'NGC3741'}
```

NOTE: `BRIDGE_GALAXIES` は v1.0.2 内 `EXCLUDED_4_SPARC_GALAXIES` (line ~136 周辺) と同一 set。constant 名統一は P12 patch で実施。

### 4.5 axis 3 (combo separate-intercept) verbatim (T16.E referenced from L621-622)

```python
# verbatim:
X_combo = np.column_stack([is_sparc, is_dsph,
                            is_sparc * lu_a, is_dsph * lu_a,
                            is_sparc * log_rh, is_dsph * log_rh])
b_a_p_combo = np.linalg.lstsq(X_combo, delta_combined, rcond=None)[0]
b_alpha_sparc = b_a_p_combo[2]   # column index 2 = is_sparc * lu_a slope
b_alpha_dsph  = b_a_p_combo[3]   # column index 3 = is_dsph  * lu_a slope
```

これが axis 3 の真の operational form (公式 Axis 3 と整合)。

---

## 5. T17 抽出依頼仕様 — C3-A5 PDF §4.3 universal coupling verbatim

source: `internal_memo_c3_extension_v3.pdf`
SHA256 prefix: `69fb1a95...`
推定所在 priority order:

1. Windows 側 master root: `D:\ドキュメント\エントロピー\膜宇宙論再考察AB効果有り\C3 拡張版仮説関連2\`
2. Windows 側 release build root: `E:\GitHub repo\build_2026-05-03\companion-v0.1-2026-05-03\companion\`
3. WordPress public docs (sakaguchi-physics.com の publication asset 配下、最終手段)

推定 page range: §4.3 universal coupling section、PDF 全 11 ページ中の中盤あたり (handoff memo 記載「11 ページ」より逆算)

### T17.A: §4.3 全文 verbatim (前後 ±5 行のコンテキスト含む)

PDF page 範囲を明示し、§4.3 開始 line から §4.4 開始直前までを **改変なし transcription**。式 block / table / 注釈すべて含む。

抽出 command 例:

```bash
# Windows + Claude Code:
uv run --with pdfplumber python -c "
import pdfplumber
path = 'D:/.../internal_memo_c3_extension_v3.pdf'
with pdfplumber.open(path) as pdf:
    for i, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text and ('4.3' in text or 'universal' in text.lower()):
            print(f'=== page {i} ===')
            print(text)
            print()
"
```

### T17.B: b_alpha formal definition (closed-form 数式 + 単位 + 物理意味)

- closed-form 数式 (例: `b_alpha = d log10(g_obs/gc_C15) / d log10(rho_baryon)`) の verbatim
- 単位 (dimensionless / log-log slope)
- 物理意味 (universal coupling index、3.92 dex 密度範囲 spanning 確認)
- alpha=0.5 deep-MOND limit + gamma=2 alternative model との関係

### T17.C: anchor 7 §2.5.5 公式 axis 1/2/3 の operational definition

T15 で確認した 3 axes の operational test specification:

| axis | spec 記述 | T17.C 抽出必須 |
|---|---|---|
| Axis 1 | extreme regime continuity check | finite/divergent test の closed-form criterion + threshold |
| Axis 2 | dSph 28/31 reversal trend reproduction | J0 minimal form baseline と比較する具体 metric + pass/fail rule |
| Axis 3 | universal slope b_alpha=0.11 emergence audit | partial OLS estimator + 0.11±0.005 test の significance criterion |

PDF 内に operational test の closed-form 記載があれば verbatim、なければ「anchor 7 §2.5.5 cite reference のみで PDF 内 implicit」と明示。

### T17.D: abs_tol / agreement criterion

- `0.11 ± 0.005` (3.92 dex spanning) と anchor 19 §1.5 baseline `0.1084` / `0.1127` の関係
- AC4 threshold `|Δ| <= 0.005` (v1.0.2 hardcoded、line 1740-1747) と整合する spec source
- bootstrap / jackknife confidence interval が PDF 内に記載されていれば verbatim

---

## 6. T18 抽出依頼仕様 — phase_c3_step3 完全 verbatim

source: `phase_c3_step3_dsph_gamma_vs_alpha.py`
SHA256: `c51c72f07d1b66636d695cc4cf69d8b4d0163a5fa3cc48836263a06666a295fb`

### T18.A: analyze_dataset() 全 body (L320-432 全 113 行 verbatim)

T16.C で omit (`# ... gamma fit, AIC computation, LRT, Spearman, etc. ...`) されていた部分を全行 verbatim:

```
L320-432 全 113 行を改変なし transcription。
特に以下の中間部分が必須:
  - gamma fit block (lambda grid 経由、500 points 探索)
  - AIC / BIC computation 全 4 模型
    (alpha-direct, alpha-partial, gamma-direct, gamma-partial)
  - dAIC / dBIC delta computation
  - LRT (likelihood ratio test) statistic
  - Spearman rho (scipy.stats.spearmanr)
  - return dict 全 keys verbatim
```

### T18.B: combo partial fit (separate-intercept design) verbatim

T16.E で言及された L621-622 周辺 ±10 行:

```
combo partial fit の verbatim:
  X_combo = column_stack([is_sparc, is_dsph,
                          is_sparc * lu_a, is_dsph * lu_a,
                          is_sparc * log_rh, is_dsph * log_rh])
  b_a_p_combo = OLS(delta_combined, X_combo)
  b_alpha_sparc = b_a_p_combo[2]
  b_alpha_dsph  = b_a_p_combo[3]

の前後 ±10 行 (function 名 + caller / output dict への代入箇所)。
特に combo function の signature と
delta_combined / is_sparc / is_dsph 構築 logic 必須。
```

### T18.C: Sgr exclusion criterion (dSph 31 -> 30 baseline rule)

```
sample preparation pipeline 内の Sgr 除外判定 verbatim:
  - 列名 (例: dwarf_name, name 等)
  - 除外条件 (例: dwarf_name != "Sgr" or filter rule)
  - 31 -> 30 reduction の地点 (line range)
  - 関連 input file (CSV / parquet) の path
```

### T18.D: output dict 全 keys + 用途分類

```
analyze_dataset() return dict の verbatim 全 keys + 各 key の意味:
  REQUIRED (baseline 値 reproducibility 必須):
    - b_alpha_direct  (axis 1/2 simple slope)
    - b_alpha_partial (axis 1/2 partial OLS slope = baseline 0.1084/0.1127 reproducer)
    - n_loc           (sample size verification)
  OPTIONAL (audit のみ):
    - nll_*           (negative log likelihood, 4 models)
    - AIC_*, BIC_*    (information criteria, 4 models)
    - dAIC_vs_d, dAIC_vs_p, dBIC_vs_d, dBIC_vs_p (model selection delta)
    - LRT, Spearman_* (auxiliary stats)

REQUIRED keys のみ companion script v1.0.3 に移植、
OPTIONAL keys は v1.0.4 round 候補として deferred 可能。
```

---

## 7. P12 patch 適用 plan (T17+T18 受領後)

### 7.1 patch 構造 overview

| sub-patch | scope | LOC 増 |
|---|---|---|
| P12-1 | constants 拡張 (T16.F + T17.D verbatim) | +20 |
| P12-2 | helper functions 追加 (log_umem_alpha, fit_ols, sample preparation) | +50 |
| P12-3 | b_alpha_3axis_audit() 本体 fill (3 axes 公式 spec) | +120 |
| P12-4 | run_dsph_audit caller-side 互換 wrapper | +15 |
| P12-5 | docstring + changelog | +30 |

P12 sub-total: +235 LOC (前回見積もり +195-275 の中央値)

### 7.2 P12-1: constants 拡張 (line ~155 周辺 additive insertion)

```python
# Phase C3 v3 §4.3 universal coupling constants (T16.F + T17.D verbatim)
A0_KPC: float = 1.2e-10 * 3.086e19 / 1e6           # (km/s)^2/kpc
G_STRIGARI: float = 0.228 * A_0                     # m/s^2 (~2.736e-11)
C15_COEF: float = 0.584                             # eta_0 prefactor
C15_UPSILON_EXP: float = -0.361                     # beta_Y exponent

# axis 3 universal coupling baseline (T17.D verbatim から確定)
B_ALPHA_AXIS3_BASELINE: float = 0.11                # universal slope
B_ALPHA_AXIS3_TOLERANCE: float = 0.005              # ±0.005 from C3-A5 §4.3

# axis 1/2 sample sizes (T16+T18.C verbatim から確定)
N_AXIS_1_SPARC_EXPECTED: int = 124                  # SPARC Q<3 + bridge 4 excluded
N_AXIS_2_DSPH_EXPECTED: int = 30                    # dSph 31 - Sgr = 30
N_AXIS_3_COMBINED_EXPECTED: int = 154               # 124 + 30

# Phase C3 v3 dwarf exclusion (T18.C verbatim から確定)
DSPH_EXCLUDED_NAMES: Set[str] = {"Sgr"}             # Sgr from dSph baseline
```

NOTE: `EXCLUDED_4_SPARC_GALAXIES` (v1.0.2 既存) を `BRIDGE_GALAXIES` の alias として確認、constant 名は v1.0.2 の `EXCLUDED_4_SPARC_GALAXIES` 維持 (retroactive change 避ける)。

### 7.3 P12-2: helper functions 追加 (line ~1420 周辺 additive insertion)

```python
def log_umem_alpha(rho_gal: np.ndarray) -> np.ndarray:
    """log u_mem under alpha hypothesis (Phase C3 v3 §4.3, T16.E verbatim)."""
    return 2.0 * np.log10(rho_gal)


def _fit_ols_partial(
    y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    OLS via numpy.linalg.lstsq with NLL computation.
    Returns (beta, nll). Phase C3 v3 §4.3 estimator (T16.C verbatim).
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    sigma2 = float(np.sum(resid ** 2) / n)
    if sigma2 <= 0:
        return beta, 1e10
    nll = 0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return beta, float(nll)


def _prepare_sparc_phase_c3_sample(
    sparc_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Phase C3 v3 §4.3 SPARC sample preparation (T16.F verbatim).
    Returns 124-galaxy sample (Q<3, bridge 4 excluded).
    """
    df = sparc_df.copy()
    # Q cut (T16.F: sparc[sparc['Q'] < 3])
    if "Q" in df.columns:
        df = df[df["Q"] < 3].reset_index(drop=True)
    # Mass / radius derivation
    df["Mstar"] = df["Upsilon_d"] * df["L36"] * 1e9
    df["Mgas"]  = HELIUM_FACTOR * df["MHI"] * 1e9
    df["Mbar"]  = df["Mstar"] + df["Mgas"]
    df["hR"]    = df["Rdisk"]
    df["r_h"]   = 1.68 * df["hR"]
    df["rho_gal"] = df["Mbar"] / (4.0 / 3.0 * np.pi * df["r_h"] ** 3)
    df["v_flat"] = df["Vflat"]
    df["gc_C15"] = (
        C15_COEF * df["Upsilon_d"] ** C15_UPSILON_EXP
        * np.sqrt(A0_KPC * df["v_flat"] ** 2 / df["hR"])
    )
    # Bridge exclusion (T16.F: BRIDGE_GALAXIES 4 件)
    df = df[~df["Galaxy"].isin(EXCLUDED_4_SPARC_GALAXIES)].reset_index(drop=True)
    df["delta_primary"] = (
        np.log10(df["g_obs"]) - np.log10(df["gc_C15"])
    )
    df["log_rh"] = np.log10(df["r_h"])
    return df


def _prepare_dsph_phase_c3_sample(
    dsph_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Phase C3 v3 §4.3 dSph sample preparation (T18.C verbatim).
    Returns 30-galaxy sample (31 - Sgr).
    """
    df = dsph_df.copy()
    # Sgr exclusion (T18.C verbatim)
    name_col = <T18.C verbatim 列名>   # e.g., "dwarf_name" or "name"
    df = df[~df[name_col].isin(DSPH_EXCLUDED_NAMES)].reset_index(drop=True)
    # delta_dSph_primary (T16.F + T18.A verbatim)
    df["delta_primary"] = np.log10(df["g_obs"]) - np.log10(G_STRIGARI)
    df["log_rh"] = np.log10(df["r_h"])  # or equivalent column per T18.C
    df["rho_gal"] = <T18.C verbatim density derivation>
    return df
```

NOTE: `<T18.C verbatim ...>` 部分は T18.C 受領後に literal で fill。

### 7.4 P12-3: b_alpha_3axis_audit() 本体 fill (line 1424-1437 replace)

```python
def b_alpha_3axis_audit(
    sparc_df: pd.DataFrame, dsph_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute b_alpha on Phase C3 v3 §4.3 universal coupling 3 axes.

    spec sources:
        anchor 7 §2.5.5 L561-682 (cite reference, formal definition deferred)
        C3-A5 internal_memo_c3_extension_v3.pdf §4.3 (canonical definition,
                                                       SHA 69fb1a95)
        phase_c3_step3_dsph_gamma_vs_alpha.py L320-432 (operational impl,
                                                        SHA c51c72f0)

    Three axes (公式 anchor 7 §2.5.5):
        Axis 1: extreme regime continuity check
                  (operational: SPARC partial OLS, 124 sample,
                   target=delta_primary, feature=log_umem_alpha,
                   nuisance=log_rh)
        Axis 2: dSph 28/31 reversal trend reproduction
                  (operational: dSph partial OLS, 30 sample,
                   target=delta_primary, feature=log_umem_alpha,
                   nuisance=log_rh)
        Axis 3: universal slope b_alpha=0.11 emergence audit
                  (operational: combo separate-intercept design,
                   154 = 124 + 30 sample,
                   features = [is_sparc, is_dsph, is_sparc*lu, is_dsph*lu,
                               is_sparc*log_rh, is_dsph*log_rh])

    Returns
    -------
    dict with keys:
        # Axis 1 (anchor 7 §2.5.5 公式)
        axis_1_continuity_status : str  # 'finite' / 'divergent' (T17.C)
        # Axis 2 (公式)
        axis_2_reversal_status   : str  # 'reproduced' / 'failed' (T17.C)
        # Axis 3 (公式 + universal slope reproducibility)
        axis_3_universal_slope   : float        # combo partial OLS (T17.B)
        axis_3_within_tolerance  : bool         # |slope - 0.11| <= 0.005
        # Sub-axis breakdowns (caller-side compatibility)
        axis_1_SPARC             : float        # SPARC partial OLS slope
        axis_2_dSph              : float        # dSph  partial OLS slope
        axis_3_combined_sparc    : float        # combo b_a_p_combo[2]
        axis_3_combined_dsph     : float        # combo b_a_p_combo[3]
        # Diff statistics
        abs_diff_axis12          : float        # |axis_1 - axis_2|
        # Sample sizes (audit trail)
        sample_n_axis_1          : int
        sample_n_axis_2          : int
        sample_n_axis_3          : int
        # Estimator metadata
        estimator                : str  # "numpy.linalg.lstsq partial OLS"
    """
    # Sample preparation (T16.F + T18.C verbatim)
    sparc_prep = _prepare_sparc_phase_c3_sample(sparc_df)
    dsph_prep  = _prepare_dsph_phase_c3_sample(dsph_df)
    n1 = len(sparc_prep)
    n2 = len(dsph_prep)

    # Axis 1: SPARC partial OLS (T16.B+E verbatim)
    rho_s = sparc_prep["rho_gal"].values
    lu_s  = log_umem_alpha(rho_s)
    log_rh_s = sparc_prep["log_rh"].values
    delta_s = sparc_prep["delta_primary"].values
    X1 = np.column_stack([np.ones(n1), lu_s, log_rh_s])
    beta_1, _ = _fit_ols_partial(delta_s, X1)
    axis_1_value = float(beta_1[1])

    # Axis 2: dSph partial OLS (T18.A verbatim, same form)
    rho_d = dsph_prep["rho_gal"].values
    lu_d  = log_umem_alpha(rho_d)
    log_rh_d = dsph_prep["log_rh"].values
    delta_d = dsph_prep["delta_primary"].values
    X2 = np.column_stack([np.ones(n2), lu_d, log_rh_d])
    beta_2, _ = _fit_ols_partial(delta_d, X2)
    axis_2_value = float(beta_2[1])

    # Axis 3: combo separate-intercept (T18.B verbatim)
    n3 = n1 + n2
    is_sparc = np.concatenate([np.ones(n1), np.zeros(n2)])
    is_dsph  = np.concatenate([np.zeros(n1), np.ones(n2)])
    lu_combined = np.concatenate([lu_s, lu_d])
    log_rh_combined = np.concatenate([log_rh_s, log_rh_d])
    delta_combined  = np.concatenate([delta_s, delta_d])
    X3 = np.column_stack([
        is_sparc, is_dsph,
        is_sparc * lu_combined, is_dsph * lu_combined,
        is_sparc * log_rh_combined, is_dsph * log_rh_combined,
    ])
    beta_3, _ = _fit_ols_partial(delta_combined, X3)
    axis_3_combined_sparc = float(beta_3[2])
    axis_3_combined_dsph  = float(beta_3[3])
    # universal slope = mean of two (or weighted, per T17.B verbatim)
    axis_3_universal = 0.5 * (axis_3_combined_sparc + axis_3_combined_dsph)
    axis_3_within = abs(axis_3_universal - B_ALPHA_AXIS3_BASELINE) <= B_ALPHA_AXIS3_TOLERANCE

    # Axis 1 continuity check (T17.C operational)
    axis_1_continuity = "finite" if np.isfinite(axis_1_value) else "divergent"

    # Axis 2 reversal trend (T17.C operational)
    # NOTE: J0 minimal form baseline 比較は T17.C verbatim から確定
    axis_2_reversal = <T17.C verbatim baseline metric>

    # Diff statistics
    abs_diff = abs(axis_1_value - axis_2_value)

    return {
        "axis_1_continuity_status": axis_1_continuity,
        "axis_2_reversal_status":   axis_2_reversal,
        "axis_3_universal_slope":   axis_3_universal,
        "axis_3_within_tolerance":  axis_3_within,
        "axis_1_SPARC":             axis_1_value,
        "axis_2_dSph":              axis_2_value,
        "axis_3_combined_sparc":    axis_3_combined_sparc,
        "axis_3_combined_dsph":     axis_3_combined_dsph,
        "abs_diff_axis12":          abs_diff,
        "sample_n_axis_1":          n1,
        "sample_n_axis_2":          n2,
        "sample_n_axis_3":          n3,
        "estimator":                "numpy.linalg.lstsq partial OLS (Phase C3 v3 §4.3)",
    }
```

### 7.5 P12-4: run_dsph_audit caller-side 互換 wrapper

caller (line 1500-1517) は既存 `axis_1_SPARC` / `axis_2_dSph` キーを読むため、§7.4 の return dict は両キーを含む形 (公式 axes + sub-axis breakdowns) で互換性維持。caller side コード変更不要。

NOTE: `b_alpha_abs_diff` の計算 (line 1507) は caller 側で `axis_1_SPARC - axis_2_dSph` で実施されるため、dict の `abs_diff_axis12` とは独立に計算される (二重計算だが整合する想定)。

### 7.6 P12-5: docstring + changelog (header + module-level)

```python
# v1.0.3 changelog (P12 公式 spec impl 込み):
#  P1  : header v1.0.2 -> v1.0.3 + changelog (本 patch)
#  P2  : load_rotation_curve() T12.A+D fill
#  P3  : compute_g_obs_g_bar() T12.B+C fill
#  P4  : f_opt(x!=0.5, c) NotImplementedError 維持 + docstring 強化 (T13.B)
#  P5  : _backsolve_c() x=0.5 operational projection (T13.B finding)
#  P6  : algorithm_b_step() docstring 更新 (x=0.5 projection)
#  P7  : e_pipeline_score() T14.A+F fill (first concrete impl)
#  P8  : b_pipeline_score() T14.B+F fill (first concrete impl)
#  P9  : NU_CANONICAL_REFERENCE_PAIRS populate (T13.D 5+2 anchor)
#  P10 : NLL_REFERENCE_PAIRS structural invariants (T14.E finding)
#  P11 : dsph_j3_check() 既存維持
#  P12 : b_alpha_3axis_audit() 公式 spec impl
#         (anchor 7 §2.5.5 axis 1/2/3 + Phase C3 v3 §4.3 partial OLS pattern
#          + phase_c3_step3 SHA c51c72f0 reproduction
#          + Sgr-excluded dSph 30 + bridge-excluded SPARC 124
#          + axis 3 combo separate-intercept design)
#  P13 : self-check expansion (b_alpha_self_check + nu_canonical_self_check
#         + nll_self_check) + manifest update
```

---

## 8. 検証 protocol (P12 fill 後実行)

### 8.1 静的検証 (claude.ai container 内、input-free)

| test | expected |
|---|---|
| AST parse OK | syntax valid |
| 関数定義数 | 48 -> 50 (b_alpha_3axis_audit + helper 2 関数 追加) |
| import 整合 | numpy, pandas, scipy.stats 全て v1.0.2 で済 import、追加なし |
| Unicode 禁止文字 audit | ALL CLEAN (PDF v4.4 spec) |
| cascade SSoT self-check (vpp_x05(0.83), f_opt_v3_cascade(0.83)) | 値変化なし (P12 patch は cascade SSoT 不変領域) |

### 8.2 structural test (mock data)

```python
# 5 行 mock SPARC, 5 行 mock dSph DataFrame で raise しないこと
import pandas as pd, numpy as np
from run_section2_5_v0_2 import b_alpha_3axis_audit

mock_sparc = pd.DataFrame({
    "Galaxy": ["Mock1", "Mock2", "Mock3", "Mock4", "Mock5"],
    "Q": [1, 1, 1, 1, 2],
    "Upsilon_d": [0.5] * 5,
    "L36": [10.0, 8.0, 12.0, 9.0, 11.0],
    "MHI": [3.0, 2.5, 4.0, 3.5, 3.2],
    "Rdisk": [2.5, 3.0, 3.5, 2.8, 3.2],
    "Vflat": [150, 140, 160, 145, 155],
    "g_obs": np.linspace(1e-11, 5e-11, 5),
})
mock_dsph = pd.DataFrame({
    "name": ["Draco", "Fornax", "Sculptor", "Carina", "Leo I"],
    "g_obs": np.linspace(2e-11, 1e-10, 5),
    "rho_gal": np.linspace(1e6, 1e7, 5),
    "r_h": np.linspace(0.1, 1.0, 5),
})

result = b_alpha_3axis_audit(mock_sparc, mock_dsph)
required_keys = ["axis_1_SPARC", "axis_2_dSph", "axis_3_universal_slope",
                 "abs_diff_axis12", "sample_n_axis_1", "sample_n_axis_2",
                 "estimator"]
for k in required_keys:
    assert k in result, f"missing key: {k}"
print("structural test: PASS")
```

### 8.3 numerical reproducibility test (Windows + Claude Code 実行)

```bash
uv run --with numpy --with scipy --with pandas python -c "
import pandas as pd
from run_section2_5_v0_2 import (
    b_alpha_3axis_audit,
    B_ALPHA_SPARC_BASELINE, B_ALPHA_DSPH_BASELINE,
    B_ALPHA_ABS_DIFF_BASELINE, B_ALPHA_AXIS3_TOLERANCE,
)

sparc = pd.read_csv('<phase_c3_v3_sparc_input_path>')
dsph  = pd.read_csv('<phase_c3_v3_dsph_input_path>')

result = b_alpha_3axis_audit(sparc, dsph)

# 確認 1: axis 1 SPARC = 0.1084 ± 1e-3
delta_1 = abs(result['axis_1_SPARC'] - B_ALPHA_SPARC_BASELINE)
assert delta_1 < 1e-3, f'axis 1 mismatch: delta={delta_1}'

# 確認 2: axis 2 dSph = 0.1127 ± 1e-3
delta_2 = abs(result['axis_2_dSph'] - B_ALPHA_DSPH_BASELINE)
assert delta_2 < 1e-3, f'axis 2 mismatch: delta={delta_2}'

# 確認 3: |axis 1 - axis 2| = 0.0042 ± 1e-3
delta_diff = abs(result['abs_diff_axis12'] - B_ALPHA_ABS_DIFF_BASELINE)
assert delta_diff < 1e-3, f'diff mismatch: delta={delta_diff}'

# 確認 4: axis 3 universal slope within tolerance
assert result['axis_3_within_tolerance'], (
    f'axis 3 universal slope outside tolerance: {result[\"axis_3_universal_slope\"]}'
)

# 確認 5: sample sizes
assert result['sample_n_axis_1'] == 124, f'n_1 mismatch: {result[\"sample_n_axis_1\"]}'
assert result['sample_n_axis_2'] == 30,  f'n_2 mismatch: {result[\"sample_n_axis_2\"]}'
assert result['sample_n_axis_3'] == 154, f'n_3 mismatch: {result[\"sample_n_axis_3\"]}'

print('numerical reproducibility test: PASS')
print(result)
"
```

### 8.4 AC4 evaluation 確認

`python run_section2_5_v0_2.py --no-rotation-curve` で:

- `dsph_block.b_alpha_status == "computed"` (deferred ではない)
- `dsph_block.b_alpha_abs_diff` が float (None ではない)
- `ac.AC4_b_alpha_abs_diff_le_0_005.pass == True` (|Δ|=0.0042 <= 0.005 のため)

### 8.5 forensic chain confirmation

| item | confirmation |
|---|---|
| 1. Anchor IMMUTABLE preservation | anchor 5/6/7/8/14/16/17/19/21 全て modify 0、retroactive change 0 |
| 2. R-1 LOCK | k_B=0 維持、parameter-free canonical 維持 |
| 3. R-2 LOCK | Algorithm B simultaneous self-consistency loop 維持 |
| 4. Q-C1 LOCK | k_E=2 default 維持 |
| 5. cascade SSoT | V''(x=0.5, c) 5-anchor + foundation b0cb36d7 不変 |
| 6. L-1 forward-ref 0 | parent v4.8 forward-ref 0 strict |
| 7. companion additive | parent v4.8 NULL impact (純 additive supersession) |

---

## 9. Delivery convention (Route A pattern)

### 9.1 ファイル生成

- 本 procedure 自身: `P12_FILL_PROCEDURE_v2_0.md` (本ファイル)
- v1.0.3 fill 後 script: `run_section2_5_v0_2_v1_0_3.py` (P1-P13、P12 公式 spec 込み)

### 9.2 SHA verify + canonical promotion (Windows side)

```powershell
# 本 v2.0 procedure download 後:
$expected_sha = "<本 chat で提示された SHA256>"
$actual_sha = (Get-FileHash -Algorithm SHA256 .\P12_FILL_PROCEDURE_v2_0.md).Hash.ToLower()
if ($actual_sha -eq $expected_sha) {
    Write-Host "v2.0 SHA MATCH: bit-exact" -ForegroundColor Green
} else {
    Write-Host "v2.0 SHA MISMATCH -- re-download" -ForegroundColor Red
    exit 1
}

# v1.0.3 script download 後:
$expected_v103_sha = "<v1.0.3 final SHA from claude.ai session>"
$actual_v103_sha = (Get-FileHash -Algorithm SHA256 .\run_section2_5_v0_2_v1_0_3.py).Hash.ToLower()
if ($actual_v103_sha -eq $expected_v103_sha) {
    # canonical promotion sequence:
    Move-Item .\run_section2_5_v0_2.py .\run_section2_5_v0_2_v1_0_2.bak.py
    Rename-Item .\run_section2_5_v0_2_v1_0_3.py .\run_section2_5_v0_2.py
    Write-Host "v1.0.3 canonical promotion: complete" -ForegroundColor Green
}
```

### 9.3 backup chain (forensic compliance)

| state | Windows side filename | role |
|---|---|---|
| pre-v1.0.3 | run_section2_5_v0_2.py (95,914 B / dd762fd2...) | v1.0.2 canonical (operational) |
| post-promotion | run_section2_5_v0_2.py (~115,000 B / <new SHA>) | v1.0.3 canonical (operational) |
| backup-1 | run_section2_5_v0_2_v1_0_2.bak.py | v1.0.2 historical |
| backup-0 | run_section2_5_v0_2_v1_0_1.bak.py | v1.0.1 historical (既存) |

---

## 10. Scope clarification (P12 atomic completion 確定)

v1.0 の §8 (P12 deferred vs fill 比較) は v2.0 では不要 (既に option 2 = P12 atomic completion 確定済)。代わりに本 §10 では本 session で P12 完了する場合の確実性 / risk を整理。

### 10.1 P12 公式 spec impl 完了 確実性

| 項目 | confidence | 根拠 |
|---|---|---|
| axis 1 SPARC partial OLS impl | HIGH | T16.B+C+E+F verbatim 受領済、estimator pattern 完全特定 |
| axis 2 dSph partial OLS impl | HIGH | T16 verbatim + T18.C で sample preparation 完成 |
| axis 3 combo separate-intercept impl | HIGH | T16.E reference + T18.B verbatim で確定 |
| 0.1084 reproducibility | HIGH | phase_c3_step3 (SHA c51c72f0...) が baseline producer の真の source |
| 0.1127 reproducibility | HIGH | 同上 |
| anchor 7 §2.5.5 公式 axes 全 PASS | MEDIUM | T17.C verbatim 必要、特に axis 1 continuity threshold + axis 2 reversal metric |
| AC4 (|Δ| <= 0.005) PASS | HIGH | baseline 値が reproduce されれば |Δ|=0.0042 < 0.005 自動 PASS |

### 10.2 P12 公式 spec impl の latent risk

| risk | impact | mitigation |
|---|---|---|
| C3-A5 PDF が見つからない | HIGH | T17 抽出不可 -> §10.3 fallback path |
| T17.C 内 axis 1 continuity threshold が implicit のみ | MEDIUM | implementation で defensive `np.isfinite()` check のみ採用 + comment 記載 |
| T17.C 内 axis 2 reversal metric が implicit のみ | MEDIUM | J0 minimal form baseline = 0 比較で polarity check |
| T18.A の gamma fit / LRT 部分が長大 | LOW | OPTIONAL key 扱いで baseline reproducibility には不要 |
| sample preparation 内 column 名が現 v1.0.2 schema と不整合 | MEDIUM | adapter layer で column rename / conditional handling |

### 10.3 fallback path (T17 不可時)

C3-A5 PDF が見つからない / 抽出不可の場合:

- **fallback A**: anchor 7 §2.5.5 cite reference のみで b_alpha formal definition placeholder、operational impl は phase_c3_step3 pattern (T16+T18) 完全 transcribe -> 0.1084/0.1127 reproducibility は確保、formal definition 部分は v1.0.4 で C3-A5 verbatim 取得後 hardening
- **fallback B**: P12 を deferred に戻し、v1.0.3 では P12 docstring 強化のみで本 session 完結 -> v1.0.4 で full impl

fallback A 採択時の impact: P12 機能は完全動作、ただし docstring 内 "T17.B formal definition" 部分が cite reference 表記になる (next round で hardening)。

---

## 11. 想定 timeline (option 2 atomic completion path)

| Phase | 想定所要時間 | 担当 | 並列可否 |
|---|---|---|---|
| 本 v2.0 procedure review | 5-10 分 | ユーザー | - |
| T17 抽出 (C3-A5 PDF §4.3) | 15-30 分 | ユーザー (Windows + Claude Code) | T18 と並列可 |
| T18 抽出 (phase_c3_step3 完全 verbatim) | 10-20 分 | ユーザー (Windows + Claude Code) | T17 と並列可 |
| T17 + T18 投稿 | 5 分 | ユーザー | - |
| **抽出 phase 小計** | **15-30 分** (並列実施前提) | - | - |
| v1.0.3 fill patch 適用 (P1-P13、P12 公式 spec 込み) | 30-60 分 | claude.ai (本 chat) | - |
| 静的検証 (§8.1, §8.2) | 5-10 分 | claude.ai (本 chat) | - |
| Route A delivery (.py 生成 + SHA 提示) | 5 分 | claude.ai (本 chat) | - |
| Windows 側 SHA verify + canonical promotion | 5-10 分 | ユーザー | - |
| numerical reproducibility test (§8.3) | 10-30 分 | ユーザー (Windows + Claude Code) | - |
| AC4 evaluation 確認 (§8.4) | 5-10 分 | ユーザー (Windows + Claude Code) | - |
| forensic chain confirmation (§8.5) | 5 分 | claude.ai + ユーザー 共同 | - |
| **本 session 総計** | **80-160 分** | - | - |

---

## 12. 次の action

1. **本 v2.0 procedure review** (5-10 分) -> ユーザーが §5 (T17) + §6 (T18) の抽出依頼項目を確認、不足項目があれば追加要求
2. **T17 + T18 並列抽出** -> Windows + Claude Code で 2 verbatim block を取得
3. **本 chat に投稿** -> 付録 A の最小 template に従う
4. **claude.ai 側 v1.0.3 fill** -> P1-P13 atomic 適用 + 静的検証
5. **Route A delivery** -> `run_section2_5_v0_2_v1_0_3.py` SHA 提示
6. **Windows 側 promotion + numerical test** -> baseline reproducibility 確認 + AC4 PASS
7. **本 session 完結** -> handoff memo 不要 (atomic completion)

---

## 付録 A: T17 + T18 投稿時の最小 template

```
=== T17 (C3-A5 PDF §4.3 universal coupling verbatim) ===

T17.A: <§4.3 全文 verbatim, page range XXX-YYY>

T17.B: <b_alpha formal definition (closed-form 数式 + 単位 + 物理意味)>

T17.C: <anchor 7 §2.5.5 公式 axis 1/2/3 operational definition>
  axis 1 continuity threshold: ...
  axis 2 reversal trend metric: ...
  axis 3 universal slope significance: ...

T17.D: <abs_tol / agreement criterion>
  0.11 ± 0.005 と anchor 19 §1.5 baseline (0.1084/0.1127) の関係
  AC4 threshold |Δ| <= 0.005 spec source

=== T18 (phase_c3_step3 完全 verbatim) ===

T18.A: <analyze_dataset() L320-432 全 113 行 verbatim>

T18.B: <combo partial fit (L621-622 周辺 ±10 行) verbatim>

T18.C: <Sgr exclusion criterion verbatim>
  列名: dwarf_name / name / etc.
  除外条件 verbatim
  31 -> 30 reduction 地点 (line range)
  関連 input file path

T18.D: <output dict 全 keys + REQUIRED/OPTIONAL 分類>
```

---

## 付録 B: v1.0 -> v2.0 主要差分対照表

| 項目 | v1.0 (T15+T16 受領前) | v2.0 (T15+T16 finding 反映 + T17+T18 抽出依頼) |
|---|---|---|
| axis decomposition | density-weighted SPARC/dSph/combined | 公式 axes (continuity/reversal/universal) + sub-axis (SPARC/dSph/combined partial OLS) |
| estimator | scipy.stats.linregress (simple) | numpy.linalg.lstsq (3-feature partial OLS, log_rh nuisance) |
| target variable | log10(c_membrane) 想定 | delta_primary = log10(g_obs/gc_C15) (SPARC) / log10(g_obs/G_Strigari) (dSph) |
| feature variable | log10(GS0/a_0) 想定 | log_umem_alpha = 2.0 * log10(rho_gal) |
| sample size | SPARC 163 / dSph 30 | SPARC 124 / dSph 30 / combined 154 |
| import dependency | statsmodels.api 想定 | numpy のみで実装可 (scipy.stats は LRT 用、optional) |
| constants 追加 | B_ALPHA_AXIS3_BASELINE 単発 | A0_KPC, G_STRIGARI, C15_COEF, C15_UPSILON_EXP, TWO_EPS_STAR, BRIDGE_GALAXIES (alias), B_ALPHA_AXIS3_TOLERANCE, sample size constants |
| b_alpha definition source | anchor 7 §2.5.5 内想定 | C3-A5 PDF §4.3 (anchor 7 §2.5.5 は cite のみ) |
| LOC 増 | +60-120 (P12 のみ) | +195-275 (P12 + 周辺、P12-1〜P12-5 sub-patch) |
| 抽出依頼項目 | T15 + T16 | T17 + T18 (T15+T16 受領済) |
| timeline | 75-160 分 (P12 deferred 含む選択肢あり) | 80-160 分 (P12 atomic completion 確定) |
| risk | medium (semantic mismatch 未判明) | low (semantic 完全特定済、operational pattern verbatim 確認済) |

---

## 付録 C: phase_c3_step3 pattern reproduction guarantee specification

P12 公式 spec impl が phase_c3_step3 (SHA `c51c72f0...`) の baseline values 0.1084 / 0.1127 を bit-exact (within 1e-3 tolerance) reproduce することを保証する specification:

### C.1 reproduction guarantee の必要十分条件

| 条件 | 検証方法 |
|---|---|
| 1. sample 構築一致 | T16.F + T18.C verbatim 完全 transcribe -> SPARC 124 / dSph 30 sample が phase_c3_step3 と bit-exact 一致 |
| 2. estimator 一致 | numpy.linalg.lstsq([1, log_umem_alpha, log_rh], delta_primary) -> phase_c3_step3 と bit-exact 一致 (numpy version 依存性 low) |
| 3. constants 一致 | C15_COEF=0.584, C15_UPSILON_EXP=-0.361, A0_KPC, G_STRIGARI, HELIUM_FACTOR=1.33, BRIDGE_GALAXIES = phase_c3_step3 と完全一致 |
| 4. column derivation 一致 | rho_gal, gc_C15, delta_primary, log_rh の derivation 順序 + 計算式 一致 |
| 5. exclusion order 一致 | Q<3 -> bridge -> derivation 順序 一致 (順序入れ替えると微小差異発生可能性) |

### C.2 reproduction failure 時の triage

| failure pattern | 想定原因 | mitigation |
|---|---|---|
| axis 1 = 0.0XX (大幅違い) | sample 数 mismatch (Q cut or bridge order) | _prepare_sparc_phase_c3_sample の中で intermediate len(df) print + phase_c3_step3 と比較 |
| axis 1 = 0.10XX (微小違い) | numpy version dependence / NaN handling 差 | rcond=None 明示 + pre-NaN drop |
| axis 2 = NaN | dSph column 名 mismatch | T18.C verbatim 列名で v1.0.2 schema 不整合確認 |
| axis 3 = 不整合 | combo column_stack 順序 mismatch | T18.B verbatim 通り [is_sparc, is_dsph, is_sparc*lu, is_dsph*lu, is_sparc*log_rh, is_dsph*log_rh] 順序厳守 |

### C.3 acceptance gate

P12 fill 後、以下 5 conditions ALL PASS で本 session 完結:

```
COND_1: |axis_1_SPARC - 0.1084| <= 1e-3
COND_2: |axis_2_dSph  - 0.1127| <= 1e-3
COND_3: |abs_diff_axis12 - 0.0042| <= 1e-3
COND_4: axis_3_within_tolerance == True (|axis_3_universal_slope - 0.11| <= 0.005)
COND_5: AC4 evaluation in run_summary.json: pass == True
```

5 of 5 PASS で v1.0.3 を canonical promote、本 session 完結。
4 of 5 以下なら fallback path (§10.3) 検討。

---

END OF P12_FILL_PROCEDURE_v2.0
