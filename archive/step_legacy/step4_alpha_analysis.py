# -*- coding: utf-8 -*-
"""
Step 4 v3: P(y) べき指数 α の解析
修正①：r_s を F_tanh フィットで最適化
修正②：|dF/dx| > 0.05 の有意な点のみ
修正③：v_inf 漸近フィット
修正④：r_d Freeman フィット + ratio_theory = 1 + 2η(rs/r_d) の検証
修正⑤：alpha 誤差棒 + 採用基準フィルタ

使用例:
  python -X utf8 step4_alpha_analysis.py --rotmod-dir Rotmod_LTG
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import i0, i1, k0, k1  # Freeman disk
import argparse

G_c = 1.2e-10   # m/s^2 = a0
kpc_to_m = 3.086e19
RS_CORRECTION = 1.35  # tanh vs P(y²) モデルの rs 定義不整合補正

# ============================================================
# データ読み込み
# ============================================================
def load_rotmod(filepath):
    cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', names=cols)
    return df

def compute_vflat_old(df, frac=0.3):
    n = max(3, int(len(df) * frac))
    return df['Vobs'].iloc[-n:].median()

def v_asymptotic(r, v_inf, r_turn, beta):
    return v_inf * (1.0 - np.exp(-(r / r_turn)**beta))

def compute_vflat(df):
    r = df['Rad'].values
    v = df['Vobs'].values
    v_old = compute_vflat_old(df)
    try:
        popt, _ = curve_fit(
            v_asymptotic, r, v,
            p0=[v_old, r.max() * 0.3, 1.5],
            bounds=([5, 0.01, 0.3], [500, r.max() * 5, 5.0]),
            maxfev=5000
        )
        v_inf = popt[0]
        if v_inf < v_old * 0.5 or v_inf > v_old * 3.0:
            return v_old, v_old, True
        return v_inf, v_old, False
    except (RuntimeError, ValueError):
        return v_old, v_old, True

def compute_rs_pred(vflat_kms):
    vflat_ms = vflat_kms * 1e3
    return (vflat_ms**2 / G_c) / kpc_to_m

# ============================================================
# F_tanh フィットで r_s を最適化
# ============================================================
def F_tanh(r, rs, w):
    x = r / rs
    return 0.5 * (1.0 + np.tanh(w * (x - 1.0)))

def fit_rs(df, vflat):
    v_bar2 = df['Vdisk']**2 + df['Vgas']**2 + df['Vbul']**2
    F_obs = (df['Vobs']**2 - v_bar2) / vflat**2
    r = df['Rad'].values
    F_val = np.clip(F_obs.values, -0.5, 2.0)
    rs_init = compute_rs_pred(vflat)
    try:
        popt, _ = curve_fit(
            F_tanh, r, F_val,
            p0=[rs_init, 1.0],
            bounds=([0.05, 0.01], [50.0, 20.0]),
            maxfev=5000
        )
        return popt[0], popt[1], F_val
    except RuntimeError:
        return rs_init, 1.0, F_val

# ============================================================
# r_d Freeman ディスクスケール半径フィット
# ============================================================
def sb_exp(r, sigma0, r_d):
    return sigma0 * np.exp(-r / r_d)

def fit_rd(df):
    """SBdisk(r) に指数関数フィットして r_d を推定"""
    r = df['Rad'].values
    sb = df['SBdisk'].values
    mask = (sb > 0) & np.isfinite(sb)
    if mask.sum() < 3:
        return np.nan, np.nan
    r_m, sb_m = r[mask], sb[mask]
    try:
        popt, _ = curve_fit(
            sb_exp, r_m, sb_m,
            p0=[sb_m[0], r_m.max() * 0.3],
            bounds=([1e-5, 0.01], [1e6, r_m.max() * 3]),
            maxfev=5000
        )
        return popt[1], popt[0]  # r_d, Sigma_0
    except (RuntimeError, ValueError):
        return np.nan, np.nan

# ============================================================
# Freeman disk η(u) = v²_disk(r) / v²_disk(∞)
# u = r / (2*r_d) として
# v²_disk ∝ u² [I0(u)K0(u) - I1(u)K1(u)]
# ============================================================
def freeman_eta(rs, r_d):
    """Freeman 指数ディスクの v²(rs)/v²_flat 近似"""
    if r_d <= 0 or np.isnan(r_d):
        return np.nan
    u = rs / (2.0 * r_d)
    if u < 0.01:
        return 0.0
    if u > 20:
        return 1.0
    val = u**2 * (i0(u)*k0(u) - i1(u)*k1(u))
    # 正規化（Freeman disk のピーク付近で ~0.45）
    # v²_flat ≈ v²_disk_max として
    u_peak = 1.1  # Freeman disk のピーク u ≈ 2.2*r_d → u=1.1
    val_peak = u_peak**2 * (i0(u_peak)*k0(u_peak) - i1(u_peak)*k1(u_peak))
    if val_peak > 0:
        return float(val / val_peak)
    return np.nan

# ============================================================
# F(x)、P(y) 計算
# ============================================================
def compute_F(df, vflat, rs):
    v_bar2 = df['Vdisk']**2 + df['Vgas']**2 + df['Vbul']**2
    F = (df['Vobs']**2 - v_bar2) / vflat**2
    x = df['Rad'].values / rs
    return x, F.values

def compute_P(x, F):
    mask = x > 1.0
    xm = x[mask]
    Fm = F[mask]
    if len(xm) < 3:
        return None, None, None
    dFdx = np.gradient(Fm, xm)
    y = 1.0 / xm
    denom = (1.0 - y) * y**2
    valid = (np.abs(dFdx) > 0.05) & (np.abs(denom) > 0.01)
    if valid.sum() < 2:
        return None, None, None
    P = dFdx[valid] / denom[valid]
    return y[valid], P, dFdx[valid]

def fit_power_law(y, P):
    """P > 0 の点で log P = alpha * log y + const（誤差付き）"""
    absP = np.abs(P)
    mask = (y > 0.1) & (y < 0.95) & (absP > 1e-10)
    if mask.sum() < 3:
        return None, None, None, None
    log_y = np.log(y[mask])
    log_absP = np.log(absP[mask])
    # np.polyfit で共分散を取得
    alpha, const = np.polyfit(log_y, log_absP, 1)
    # 残差から alpha の標準誤差を推定
    residuals = log_absP - (alpha * log_y + const)
    n = len(log_y)
    if n > 2:
        s2 = np.sum(residuals**2) / (n - 2)
        Sxx = np.sum((log_y - log_y.mean())**2)
        alpha_err = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan
    else:
        alpha_err = np.nan
    sign_P = np.sign(P[mask]).mean()
    return alpha, alpha_err, const, sign_P

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotmod-dir', default='Rotmod_LTG')
    args = parser.parse_args()

    data_dir = Path(args.rotmod_dir)
    if not data_dir.exists():
        alt = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
        if alt.exists():
            data_dir = alt
        else:
            print(f"ERROR: {data_dir} が見つかりません"); return

    targets = [
        'DDO154', 'DDO168', 'NGC2366', 'NGC1569', 'IC1613',
        'DDO170', 'DDO161', 'DDO064',
        'UGC07577', 'UGC05918', 'UGC04305',
        'CamB', 'NGC3741', 'UGCA442',
        'UGC07866', 'KK98-251', 'ESO444-G084',
        'NGC4214', 'NGC2976', 'NGC1705',
    ]

    available = {f.stem.replace('_rotmod',''): f
                 for f in data_dir.glob('*_rotmod.dat')}
    print(f"利用可能な .dat: {len(available)} 件")

    results = []
    for name in targets:
        fp = available.get(name)
        if fp is None:
            matches = [k for k in available
                       if name.replace(' ','') in k.replace(' ','')]
            fp = available[matches[0]] if matches else None
        if fp is None:
            print(f"  {name}: not found"); continue

        df = load_rotmod(fp)
        if len(df) < 5:
            print(f"  {name}: データ点不足 ({len(df)})"); continue

        v_inf, v_old, fallback = compute_vflat(df)
        rs_pred_new = compute_rs_pred(v_inf)
        rs_fit, w_fit, _ = fit_rs(df, v_inf)

        # r_d フィット
        r_d, sigma0 = fit_rd(df)

        x, F = compute_F(df, v_inf, rs_fit)
        y, P, dFdx = compute_P(x, F)

        if y is None:
            print(f"  {name}: x > 1 のデータ点不足 (rs_fit={rs_fit:.2f})"); continue

        alpha_result = fit_power_law(y, P)
        if alpha_result[0] is None:
            alpha, alpha_err, sign_P = None, None, None
        else:
            alpha, alpha_err, const, sign_P = alpha_result

        Sigma_b = df['SBdisk'].median()
        ratio_obs = rs_fit / rs_pred_new if rs_pred_new > 0 else np.nan

        # rs 定義補正
        rs_corr = rs_fit / RS_CORRECTION
        ratio_corr = rs_corr / rs_pred_new if rs_pred_new > 0 else np.nan

        # ratio_theory = 1 + 2*η(rs_corr/r_d)  ← 補正後 rs で計算
        eta = freeman_eta(rs_corr, r_d)
        ratio_theory = 1.0 + 2.0 * eta if not np.isnan(eta) else np.nan

        n_x_gt_1 = int((x > 1).sum())
        alpha_str = f"{alpha:.2f}±{alpha_err:.2f}" if alpha is not None else "N/A"
        rd_str = f"{r_d:.2f}" if not np.isnan(r_d) else "N/A"
        print(f"  {name:12s}: v_inf={v_inf:5.1f}, rs_fit={rs_fit:.2f}, "
              f"rs_corr={rs_corr:.2f}, r_d={rd_str}, "
              f"ratio_obs={ratio_obs:.2f}, ratio_corr={ratio_corr:.2f}, "
              f"ratio_th={ratio_theory:.2f}, "
              f"alpha={alpha_str}, n={n_x_gt_1}"
              if not np.isnan(r_d) else
              f"  {name:12s}: v_inf={v_inf:5.1f}, rs_fit={rs_fit:.2f}, "
              f"rs_corr={rs_corr:.2f}, r_d=N/A, "
              f"alpha={alpha_str}, n={n_x_gt_1}")

        results.append({
            'galaxy': name,
            'v_inf': round(v_inf, 1),
            'v_old': round(v_old, 1),
            'rs_pred': round(rs_pred_new, 2),
            'rs_fit': round(rs_fit, 2),
            'rs_corr': round(rs_corr, 2),
            'r_d': round(r_d, 2) if not np.isnan(r_d) else np.nan,
            'rs_over_rd': round(rs_corr/r_d, 2) if (not np.isnan(r_d) and r_d > 0) else np.nan,
            'ratio_obs': round(ratio_obs, 2),
            'ratio_corr': round(ratio_corr, 2),
            'ratio_theory': round(ratio_theory, 2) if not np.isnan(ratio_theory) else np.nan,
            'w_fit': round(w_fit, 2),
            'alpha': round(alpha, 2) if alpha is not None else np.nan,
            'alpha_err': round(alpha_err, 2) if alpha_err is not None else np.nan,
            'sign_P': round(sign_P, 2) if sign_P is not None else np.nan,
            'Sigma_b': round(Sigma_b, 2),
            'Sigma_0': round(sigma0, 2) if not np.isnan(sigma0) else np.nan,
            'n_pts_x_gt_1': n_x_gt_1,
        })

    if not results:
        print("解析対象なし"); return

    results_df = pd.DataFrame(results)

    # ============================================================
    # 全銀河テーブル
    # ============================================================
    print(f"\n{'='*130}")
    print(f"全銀河結果 (RS_CORRECTION = {RS_CORRECTION})")
    print(results_df[['galaxy','v_inf','rs_pred','rs_fit','rs_corr','r_d','rs_over_rd',
                       'ratio_obs','ratio_corr','ratio_theory','alpha','alpha_err',
                       'sign_P','Sigma_b','n_pts_x_gt_1']].to_string(index=False))
    print(f"{'='*130}")

    # ============================================================
    # ratio 比較（3段階）
    # ============================================================
    valid_ratio = results_df.dropna(subset=['ratio_corr','ratio_theory'])
    if len(valid_ratio) >= 3:
        from scipy import stats

        print(f"\n--- ratio 比較（3段階）---")
        med_obs  = valid_ratio['ratio_obs'].median()
        med_corr = valid_ratio['ratio_corr'].median()
        med_th   = valid_ratio['ratio_theory'].median()
        print(f"ratio_obs   中央値: {med_obs:.2f}  (補正前)")
        print(f"ratio_corr  中央値: {med_corr:.2f}  (÷{RS_CORRECTION} 補正後)")
        print(f"ratio_theory中央値: {med_th:.2f}  (1+2η Freeman)")
        print(f"")
        print(f"|ratio-1| 中央値:")
        dev_obs  = (valid_ratio['ratio_obs'] - 1).abs().median()
        dev_corr = (valid_ratio['ratio_corr'] - 1).abs().median()
        dev_th   = (valid_ratio['ratio_theory'] - 1).abs().median()
        print(f"  obs:  {dev_obs:.2f}")
        print(f"  corr: {dev_corr:.2f}  (改善 {(1-dev_corr/dev_obs)*100:.0f}%)")
        print(f"  theory: {dev_th:.2f}")

        # ratio_corr vs ratio_theory の相関
        r_val, p_val = stats.pearsonr(
            valid_ratio['ratio_corr'], valid_ratio['ratio_theory'])
        print(f"\nratio_corr vs ratio_theory: Pearson r={r_val:.3f}, p={p_val:.4f}")

        # ratio_corr / ratio_theory の残差
        residual = valid_ratio['ratio_corr'] / valid_ratio['ratio_theory']
        print(f"ratio_corr / ratio_theory 中央値: {residual.median():.2f} (=1 なら完全一致)")

    # ============================================================
    # 採用基準フィルタ：n >= 8, sign_P > 0.5, alpha_err < |alpha|
    # ============================================================
    print(f"\n{'='*120}")
    print("採用基準フィルタ: n_pts_x_gt_1 >= 6 & sign_P > 0.5 & alpha_err < |alpha|")
    print(f"{'='*120}")

    sel = results_df[
        (results_df['n_pts_x_gt_1'] >= 6) &
        (results_df['sign_P'] > 0.5) &
        (results_df['alpha_err'] < results_df['alpha'].abs())
    ].copy()

    if len(sel) == 0:
        # 緩和基準
        print("厳密基準で 0 件。緩和基準 (n >= 5, sign_P > 0) を適用:")
        sel = results_df[
            (results_df['n_pts_x_gt_1'] >= 5) &
            (results_df['sign_P'] > 0)
        ].copy()

    if len(sel) > 0:
        print(sel[['galaxy','alpha','alpha_err','sign_P',
                    'ratio_obs','ratio_theory','Sigma_b',
                    'n_pts_x_gt_1']].to_string(index=False))
        print(f"\nalpha 加重平均: {np.average(sel['alpha'].dropna(), weights=1.0/(sel['alpha_err'].dropna().clip(lower=0.1)**2)):.2f}" if sel['alpha_err'].notna().sum() > 0 else "")
        print(f"alpha 中央値:   {sel['alpha'].median():.2f}")
    else:
        print("該当なし")

    # ============================================================
    # プロット（3パネル）
    # ============================================================
    valid_df = results_df.dropna(subset=['alpha'])
    if len(valid_df) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 左：alpha vs Sigma_b（誤差棒付き）
        ax = axes[0]
        has_err = valid_df['alpha_err'].notna()
        ax.errorbar(valid_df.loc[has_err, 'Sigma_b'],
                    valid_df.loc[has_err, 'alpha'],
                    yerr=valid_df.loc[has_err, 'alpha_err'],
                    fmt='o', capsize=3, markersize=6, zorder=5)
        if (~has_err).any():
            ax.scatter(valid_df.loc[~has_err, 'Sigma_b'],
                      valid_df.loc[~has_err, 'alpha'],
                      s=40, marker='x', color='grey', zorder=4)
        # 採用銀河をハイライト
        if len(sel) > 0:
            ax.scatter(sel['Sigma_b'], sel['alpha'],
                      s=120, facecolors='none', edgecolors='red', lw=2,
                      zorder=6, label='adopted')
        for _, row in valid_df.iterrows():
            ax.annotate(row['galaxy'],
                       (row['Sigma_b'], row['alpha']),
                       textcoords='offset points', xytext=(6, 4), fontsize=6)
        ax.axhline(2.0, color='grey', ls='--', lw=1, alpha=0.5,
                   label='alpha=2 (theory)')
        ax.set_xlabel('Sigma_b [L_sun/pc^2]', fontsize=11)
        ax.set_ylabel('alpha +/- err', fontsize=11)
        ax.set_title('alpha vs Sigma_b', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 中：ratio_corr vs ratio_theory
        ax2 = axes[1]
        vr = results_df.dropna(subset=['ratio_corr','ratio_theory'])
        if len(vr) > 0:
            ax2.scatter(vr['ratio_theory'], vr['ratio_corr'], s=60, zorder=5,
                       label=f'corrected (/{RS_CORRECTION})')
            ax2.scatter(vr['ratio_theory'], vr['ratio_obs'], s=40, marker='x',
                       color='grey', alpha=0.5, label='uncorrected')
            for _, row in vr.iterrows():
                ax2.annotate(row['galaxy'],
                            (row['ratio_theory'], row['ratio_corr']),
                            textcoords='offset points', xytext=(5, 3), fontsize=6)
            lim = max(vr['ratio_obs'].max(), vr['ratio_theory'].max()) * 1.1
            ax2.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='1:1 line')
            ax2.set_xlabel('ratio_theory = 1 + 2*eta(rs_corr/r_d)', fontsize=10)
            ax2.set_ylabel('ratio = rs / rs_pred', fontsize=10)
            ax2.set_title(f'ratio comparison (correction={RS_CORRECTION})', fontsize=11)
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)

        # 右：alpha（採用銀河のみ）誤差棒付き棒グラフ
        ax3 = axes[2]
        if len(sel) > 0:
            x_pos = np.arange(len(sel))
            ax3.bar(x_pos, sel['alpha'].values, yerr=sel['alpha_err'].values,
                   capsize=4, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(sel['galaxy'].values, rotation=30, ha='right', fontsize=8)
            ax3.axhline(2.0, color='red', ls='--', lw=1.5, label='alpha=2')
            ax3.set_ylabel('alpha', fontsize=11)
            ax3.set_title('Adopted galaxies: alpha +/- err', fontsize=11)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out_fig = 'alpha_vs_Sigma_b.png'
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        print(f"\n図を {out_fig} に保存しました。")

    out_csv = 'step4_alpha_result.csv'
    results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"結果を {out_csv} に保存しました。")

if __name__ == '__main__':
    main()
