# Membrane Cosmology

> Galactic rotation curves without dark matter.

arXiv 論文 **v4.7.6** (2026-04-15) の全再現スクリプト。総 255 ファイル。

## ステータス

- **有効 baseline**: v4.7.6 (§6.8.1 Yd 部分導出 + §6.9 f_2h 実測)
- **撤回**: v4.7.5 (r_s=Rdisk/2 バグ依存)。詳細: `archive/retracted/README.md`
- **PDF**: `../../膜宇宙論再考察AB効果有り/arxiv/membrane_arxiv_v476_rebuild.pdf` (33KB, 8pages)

## 論文対応表

| 論文箇所 | 主要スクリプト | 出力 |
|---|---|---|
| §3 C15 最終形 (gc = 0.584·Yd^{-0.361}·√(a0v²/hR)) | `core/sparc/sparc_eta_derivation.py` | TA3_gc_independent.csv |
| §3.3 MOND 棄却 p=1.66e-53 | `core/sparc/sparc_cond15_*.py` | |
| §4 C15 minimal sufficient (6 Sgal models) | `core/sparc/sparc_cond15_sgal_model.py` | |
| §5 BTFR Simpson's paradox | `core/sparc/sparc_btfr_withinbin.py` | |
| §5 deep-MOND Σdyn 応答 | `core/sparc/sparc_deep_mond_inversion.py` | |
| §6.7 LITTLE THINGS dwarf | `core/dwarfs/little_things_step2.py` | |
| **§6.8.1 Yd 自己無撞着 (v4.7.6 NEW)** | `theory/plasticity_direct_gc.py` | plasticity_direct_gc.png |
| §6.9 KiDS-1000 | `core/sparc/kids_extract.py` | |
| §6.9 HSC-SSP Y3 3視野 | `core/phase_b/phase_b_step3_three_fields.py` | phase_b_output/ |
| **§6.9 P15 f_2h empirical (v4.7.6 NEW)** | `twohalo/f2h_empirical_slope_v2.py` | f2h_empirical_slope_v2.png |
| §7 Item 13 結論 | (複合) | |

## rs_tanh 定義 (重要)

TA3_gc_independent.csv の `rs_tanh` 列は **回転曲線の tanh フィット** から得られる。
**生成スクリプト**: `core/sparc_fit/fit_tanh.py`, `fit_tanh2_all.py`。

- **正しい使用**: C15/Yd 関連の r_s (median ~ 2.6 kpc, range 0.1-20 kpc)
- **誤用 (v4.7.5 撤回原因)**: `Rdisk/2` (MRT Table 1 の disk scale length の半分、median ~ 1.2 kpc)

## 実行順序 (v4.7.6 再現)

```bash
# 1. SPARC 基盤
python core/sparc_fit/fit_tanh2_all.py             # rs_tanh
python data_prep/TA3_gc_measurement.py             # gc_a0, Yd
python core/sparc/sparc_eta_derivation.py          # eta0, beta

# 2. HSC-SSP Y3 Pipeline (§6.9)
python core/phase_b/phase_a_step1_gama_inspect.py
python core/phase_b/phase_a_step2_lens_sample.py
python core/phase_b/phase_a_step34_esd.py
python core/phase_b/phase_b_step1_gbar_fix.py
python core/phase_b/phase_b_step2_errors.py
python core/phase_b/phase_b_step3_three_fields.py  # gc=2.73 a0, slope +0.166

# 3. 本セッション追加 (v4.7.6)
python phase_b_output/f2h_empirical_slope_v2.py    # Delta_slope = +0.034
python theory/plasticity_direct_gc.py              # beta = -0.33 (Yd 部分導出)

# 4. arXiv ビルド
cd ../../../../膜宇宙論再考察AB効果有り/arxiv/
python arxiv_v476_patch.py
python membrane_arxiv_v476_submission.py
```

## 環境

- **Python 3.12** + numpy 2.x (`np.trapz` → `np.trapezoid` 必須)
- **Windows cp932**: `PYTHONIOENCODING=utf-8` 設定必要
- **依存**: scipy, matplotlib, astropy, reportlab, pypdf, scikit-learn
- **オプション**: colossus (2-halo 計算用)

## データファイル (外部)

- `TA3_gc_independent.csv` — SPARC gc/a0 測定結果 (175 銀河)
- `SPARC_Lelli2016c.mrt` — SPARC Table 1 (列は whitespace-split 必須)
- `Rotmod_LTG/*.dat` — 回転曲線 (V(r), tanh フィット元)
- `phase1/` — Phase 1 中間ファイル
- `phase_b_output/` — Phase B 最終出力 (ESD, jackknife cov)
- GAMA DR4 FITS: 別ディレクトリ
- HSC Y3 shape catalog: 別ディレクトリ

## データ解析の注意点

- GAMA 列: `RAcen`, `Deccen` (大文字 `RA/DEC` ではない)
- StellarMasses: `StellarMassesGKV_v24.fits` (v02/v06 ではない)
- MRT parsing: **whitespace split** を使用 (固定幅 byte position は誤り)
  - `parts[9]` = Reff, `parts[11]` = Rdisk (NOT Rflat), `parts[15]` = Vflat
- HSC `.gz.1` は実 plaintext (magic number 判定)

## 撤回・無効化記録

### v4.7.5 (2026-04-15 撤回)
- 原因: `model_b_slope_contribution.py` で r_s = Rdisk/2 を使用 (本来は rs_tanh)
- 影響: Model B slope contribution が +0.089 と過大評価 (真値 +0.017)
- 対応: v4.7.4 に復帰 → v4.7.6 で正しい rs_tanh で再評価
- 詳細: `archive/retracted/model_b_slope_contribution.py` (元版保存)

### Dead / broken files
- `broken_names/` (4 files): 名前破損 (python プレフィックス, コロン, 日本語名)
- 0-byte files: `get_G223_G228_G231_nfw.py`, `check_catalog.py`, `psz2_step4_refit_v2.py`

## 参照

- 引き継ぎメモ: `../../膜宇宙論再考察AB効果有り/handoff_memo_*.txt`
- プロジェクトメモリ: `~/.claude/projects/C--Users-sgucc/memory/project_membrane_cosmology.md`
- arXiv PDF: `../../膜宇宙論再考察AB効果有り/arxiv/membrane_arxiv_v476_rebuild.pdf`
