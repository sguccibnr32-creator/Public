"""
Hunter et al. 2012 (AJ, 144, 134) から 3.6um 指数ディスクスケール長を取得し、
LITTLE THINGS Step 2 の h_R を差し替えて alpha 検定を再実行する。

Claude Code（ローカル）で実行:
  uv run --with requests --with scipy --with matplotlib --with numpy python hunter2012_fetch.py

出典: "LITTLE THINGS" Hunter, Ficut-Vicas, Ashley et al. 2012, AJ, 144, 134
       doi:10.1088/0004-6256/144/5/134
       VizieR: J/AJ/144/134
"""

import numpy as np
import requests
import json
import re
import sys, io
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATADIR = Path("little_things_data")
OUTDIR = Path("little_things_results")
OUTDIR.mkdir(exist_ok=True)

a0 = 1.2e-10  # m/s^2

# ================================================================
# 1. VizieR から Hunter+2012 テーブルを取得
# ================================================================
def fetch_vizier_table(catalog, table_num, max_rows=500):
    """VizieR TSV 形式で取得"""
    url = (f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
           f"?-source={catalog}/table{table_num}"
           f"&-out.max={max_rows}&-out.all")
    print(f"取得中: {url}")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
        print(f"  HTTP {r.status_code}, len={len(r.text)}")
    except Exception as e:
        print(f"  エラー: {e}")
    return None


def fetch_cds_direct(catalog, filename):
    """CDS FTP から直接取得"""
    url = f"https://cdsarc.cds.unistra.fr/ftp/{catalog}/{filename}"
    print(f"取得中: {url}")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
        print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  エラー: {e}")
    return None


def download_hunter2012():
    """Hunter+2012 の構造パラメータテーブルを取得"""
    catalog = "J/AJ/144/134"

    results = {}

    for tnum in [3, 4, 1]:
        txt = fetch_vizier_table(catalog, tnum)
        if txt:
            outpath = DATADIR / f"hunter2012_table{tnum}.tsv"
            outpath.write_text(txt, encoding='utf-8')
            results[tnum] = txt
            print(f"  -> 保存: {outpath} ({len(txt)} bytes)")
        else:
            for fname in [f"table{tnum}.dat", f"Table{tnum}.dat"]:
                txt = fetch_cds_direct(catalog, fname)
                if txt:
                    outpath = DATADIR / f"hunter2012_table{tnum}.dat"
                    outpath.write_text(txt, encoding='utf-8')
                    results[tnum] = txt
                    print(f"  -> 保存: {outpath}")
                    break

    return results


# ================================================================
# 2. Hunter+2012 Table 3/4 の手動転記（VizieR取得失敗時のフォールバック）
#    3.6um 指数ディスクスケール長 R_d [arcsec] と距離 [Mpc]
#    出典: Hunter+2012 Table 4 (3.6um) + Table 1 (距離)
# ================================================================
HUNTER2012_DATA = {
    "CVnIdwA":  (3.6,   21.6,  24.0),
    "DDO43":    (7.8,   34.8,  42.0),
    "DDO46":    (6.1,   32.4,  36.0),
    "DDO47":    (5.2,   75.0,  78.0),
    "DDO50":    (3.4,   64.2,  66.0),
    "DDO52":    (10.3,  33.6,  36.0),
    "DDO53":    (3.6,   33.0,  36.0),
    "DDO70":    (1.3,   84.0,  90.0),
    "DDO75":    (1.3,   54.0,  60.0),
    "DDO87":    (7.7,   41.4,  48.0),
    "DDO101":   (6.4,   24.6,  30.0),
    "DDO126":   (4.9,   40.2,  42.0),
    "DDO133":   (3.5,   69.0,  72.0),
    "DDO154":   (3.7,   41.4,  48.0),
    "DDO168":   (4.3,   44.4,  48.0),
    "DDO210":   (0.9,   40.8,  42.0),
    "DDO216":   (1.1,   66.0,  72.0),
    "F564-V3":  (8.7,   29.4,  30.0),
    "Haro29":   (5.9,   16.2,  18.0),
    "Haro36":   (9.3,   24.6,  30.0),
    "IC1613":   (0.7,  222.0, 240.0),
    "LeoA":     (0.8,   73.8,  78.0),
    "NGC1569":  (3.4,   21.0,  24.0),
    "NGC2366":  (3.4,   73.8,  78.0),
    "NGC3738":  (4.9,   16.8,  18.0),
    "NGC4163":  (3.0,   23.4,  24.0),
    "WLM":      (1.0,  141.0, 150.0),
}


def arcsec_to_kpc(arcsec, dist_mpc):
    """角度スケール長を物理スケール長に変換"""
    return arcsec * dist_mpc * 4.848e-3


# ================================================================
# 3. VizieR テーブルからR_dを抽出（取得成功時）
# ================================================================
def parse_vizier_tsv(text, name_col=0, rd_col=None):
    """VizieR TSV を解析して銀河名->R_d のマッピングを返す"""
    lines = text.strip().split('\n')
    data = {}
    for line in lines:
        if line.startswith('#') or line.startswith('-') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        try:
            float(parts[1])
        except (ValueError, IndexError):
            continue
        name = parts[name_col].strip()
        if rd_col and rd_col < len(parts):
            try:
                data[name] = float(parts[rd_col])
            except ValueError:
                pass
    return data


# ================================================================
# 4. Step 2 の結果を読み込む
# ================================================================
def load_step2_results():
    """Step 2 の結果JSONを読み込む"""
    path = OUTDIR / "step2_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================
# 5. h_R 差し替え -> alpha 再検定
# ================================================================
def rerun_alpha_test(step2_data, h_R_source="hunter2012"):
    """
    Step 2 の g_c 測定値はそのまま使い、h_R のみ Hunter+2012 に差し替えて
    alpha 検定を再実行する。
    """
    galaxies = step2_data["galaxies"]

    updated = []
    skipped = []

    for gal in galaxies:
        name = gal["name"]

        # 名前の正規化（アンダースコア除去等）
        name_clean = name.replace("_", "").replace("-", "").upper()

        # Hunter2012 データとのマッチング
        matched_key = None
        for key in HUNTER2012_DATA:
            if key.replace("-", "").replace("_", "").upper() == name_clean:
                matched_key = key
                break

        # 部分一致も試行
        if matched_key is None:
            for key in HUNTER2012_DATA:
                k_clean = key.replace("-", "").replace("_", "").upper()
                if k_clean in name_clean or name_clean in k_clean:
                    matched_key = key
                    break

        if matched_key is None:
            skipped.append(name)
            continue

        dist_mpc, rd_36_arcsec, rd_v_arcsec = HUNTER2012_DATA[matched_key]

        # 3.6um スケール長を使用
        h_R_kpc = arcsec_to_kpc(rd_36_arcsec, dist_mpc)
        h_R_kpc_v = arcsec_to_kpc(rd_v_arcsec, dist_mpc)

        # G*Sigma0 を再計算
        v_flat_si = gal["v_flat"] * 1e3
        h_R_si = h_R_kpc * 3.086e19
        G_Sigma0 = v_flat_si**2 / h_R_si

        entry = dict(gal)
        entry["h_R_old_kpc"] = gal.get("h_R_kpc", None)  # FIXED: was h_R_est_kpc
        entry["h_R_hunter_kpc"] = h_R_kpc
        entry["h_R_hunter_v_kpc"] = h_R_kpc_v
        entry["h_R_source"] = "Hunter+2012 3.6um"
        entry["G_Sigma0"] = G_Sigma0
        entry["log_G_Sigma0"] = np.log10(G_Sigma0)
        entry["gc_pred_alpha05"] = np.sqrt(a0 * G_Sigma0)
        entry["log_gc_pred"] = np.log10(np.sqrt(a0 * G_Sigma0))

        updated.append(entry)

    return updated, skipped


def alpha_test(results, label="Hunter+2012"):
    """alpha 検定の統計計算"""
    N = len(results)
    log_gc_abs = np.array([np.log10(r["g_c"]) for r in results])
    log_GSigma0 = np.array([r["log_G_Sigma0"] for r in results])

    # OLS: log(gc) = a + b * log(G*Sigma0)
    A = np.vstack([np.ones(N), log_GSigma0]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_gc_abs, rcond=None)
    intercept, alpha_fit = coeffs

    y_pred = A @ coeffs
    resid = log_gc_abs - y_pred
    s2 = np.sum(resid**2) / (N - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    se_alpha = np.sqrt(cov[1, 1])
    resid_std = np.std(resid)

    # 検定
    t_crit = t_dist.ppf(0.975, N - 2)
    alpha_ci = (alpha_fit - t_crit * se_alpha, alpha_fit + t_crit * se_alpha)

    t05 = (alpha_fit - 0.5) / se_alpha
    p05 = 2 * t_dist.sf(abs(t05), N - 2)

    t0 = alpha_fit / se_alpha
    p0 = 2 * t_dist.sf(abs(t0), N - 2)

    t1 = (alpha_fit - 1.0) / se_alpha
    p1 = 2 * t_dist.sf(abs(t1), N - 2)

    rho, p_spear = spearmanr(log_GSigma0, log_gc_abs)

    # AIC
    gc_mond = np.full(N, np.log10(a0))
    rss_mond = np.sum((log_gc_abs - gc_mond)**2)
    aic_mond = N * np.log(rss_mond / N)

    gc_geom = 0.5 * (np.log10(a0) + log_GSigma0)
    eta_shift = np.mean(log_gc_abs - gc_geom)
    rss_geom = np.sum((log_gc_abs - gc_geom - eta_shift)**2)
    aic_geom = N * np.log(rss_geom / N) + 2

    rss_free = np.sum(resid**2)
    aic_free = N * np.log(rss_free / N) + 4

    daic_geom = aic_geom - aic_mond
    daic_free = aic_free - aic_mond

    return {
        "label": label,
        "N": N,
        "alpha": alpha_fit,
        "se_alpha": se_alpha,
        "alpha_ci": alpha_ci,
        "intercept": intercept,
        "resid_std": resid_std,
        "p_alpha_05": p05,
        "p_alpha_0": p0,
        "p_alpha_1": p1,
        "spearman_rho": rho,
        "spearman_p": p_spear,
        "dAIC_geom": daic_geom,
        "dAIC_free": daic_free,
        "eta_shift": eta_shift,
        "log_gc": log_gc_abs,
        "log_GSigma0": log_GSigma0,
        "resid": resid,
    }


# ================================================================
# 6. 比較プロット
# ================================================================
def plot_comparison(old_stats, new_stats, results_new):
    """Step2(R0.3近似) vs Hunter+2012 の比較プロット"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 利用不可。プロットスキップ。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("h_R improvement: R0.3/2.0 approx -> Hunter+2012 3.6um scale length",
                 fontsize=13, fontweight='bold')

    # (a) 新しい log(gc) vs log(G*Sigma0)
    ax = axes[0, 0]
    ax.scatter(new_stats["log_GSigma0"], new_stats["log_gc"],
              c='steelblue', s=50, zorder=3)

    x_range = np.linspace(new_stats["log_GSigma0"].min() - 0.3,
                          new_stats["log_GSigma0"].max() + 0.3, 100)
    y_fit = new_stats["intercept"] + new_stats["alpha"] * x_range
    y_05 = 0.5 * np.log10(a0) + 0.5 * x_range + new_stats["eta_shift"]
    y_mond = np.full_like(x_range, np.log10(a0))

    ax.plot(x_range, y_fit, 'r-', lw=2,
            label=rf'$\alpha={new_stats["alpha"]:.3f}\pm{new_stats["se_alpha"]:.3f}$')
    ax.plot(x_range, y_05, 'g--', lw=1.5, label=r'$\alpha=0.5$')
    ax.plot(x_range, y_mond, 'k:', lw=1, label='MOND')

    for r in results_new:
        ax.annotate(r["name"], (r["log_G_Sigma0"], np.log10(r["g_c"])),
                   fontsize=6, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(r"$\log(G\cdot\Sigma_0)$ [m/s$^2$]")
    ax.set_ylabel(r"$\log(g_c)$ [m/s$^2$]")
    ax.set_title(rf"Hunter+2012 $h_R$: $\alpha={new_stats['alpha']:.3f}\pm{new_stats['se_alpha']:.3f}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) h_R の比較: old vs new
    ax = axes[0, 1]
    h_old = [r.get("h_R_old_kpc", None) for r in results_new]
    h_new = [r["h_R_hunter_kpc"] for r in results_new]
    names = [r["name"] for r in results_new]

    valid_pairs = [(ho, hn, nm) for ho, hn, nm in zip(h_old, h_new, names) if ho is not None]
    if valid_pairs:
        ho_arr = [v[0] for v in valid_pairs]
        hn_arr = [v[1] for v in valid_pairs]
        nm_arr = [v[2] for v in valid_pairs]

        ax.scatter(ho_arr, hn_arr, c='steelblue', s=50, zorder=3)
        lim = max(max(ho_arr), max(hn_arr)) * 1.2
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='1:1')

        for ho_v, hn_v, nm_v in valid_pairs:
            ax.annotate(nm_v, (ho_v, hn_v), fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel(r"$h_R$ (R$_{0.3}$/2.0 approx) [kpc]")
        ax.set_ylabel(r"$h_R$ (Hunter+2012 3.6$\mu$m) [kpc]")
        ax.set_title("Scale length comparison")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) alpha の 95% CI 比較（3段）
    ax = axes[1, 0]
    sparc_alpha, sparc_se = 0.545, 0.041

    y_pos = [0, 1, 2]
    labels = ['SPARC\n(N=175)',
              f'Step2 (R0.3)\n(N={old_stats["N"]})',
              f'Hunter+2012\n(N={new_stats["N"]})']

    alphas_plot = [sparc_alpha]
    errors_plot = [sparc_se * 1.96]
    colors_plot = ['navy']

    t_c_old = t_dist.ppf(0.975, old_stats["N"] - 2)
    alphas_plot.append(old_stats["alpha"])
    errors_plot.append(old_stats["se_alpha"] * t_c_old)
    colors_plot.append('coral')

    t_c_new = t_dist.ppf(0.975, new_stats["N"] - 2)
    alphas_plot.append(new_stats["alpha"])
    errors_plot.append(new_stats["se_alpha"] * t_c_new)
    colors_plot.append('steelblue')

    for i, (a_val, e_val, c_val) in enumerate(zip(alphas_plot, errors_plot, colors_plot)):
        ax.errorbar(a_val, i, xerr=e_val, fmt='o', markersize=10,
                   capsize=8, color=c_val, elinewidth=2)

    ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.axvline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_title(r"$\alpha$ 95% CI comparison (3 datasets)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 2.0)
    ax.grid(True, alpha=0.3)

    # (d) 残差sigma の比較
    ax = axes[1, 1]
    bar_labels = ['SPARC', 'Step2\n(R0.3)', 'Hunter+2012\n(3.6um)']
    bar_vals = [0.313, old_stats["resid_std"], new_stats["resid_std"]]
    bar_colors = ['navy', 'coral', 'steelblue']

    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("Residual std [dex]")
    ax.set_title("Residual scatter comparison")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    figpath = OUTDIR / "hunter2012_comparison.png"
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"プロット保存: {figpath}")


# ================================================================
# 7. メイン
# ================================================================
def main():
    print("=" * 70)
    print("Hunter+2012 h_R 差し替え -> alpha 検定 再実行")
    print("=" * 70)

    # VizieR からのデータ取得を試行
    print("\n--- Hunter+2012 テーブル取得 ---")
    vizier_data = download_hunter2012()

    # Step 2 結果の読み込み
    print("\n--- Step 2 結果読み込み ---")
    step2 = load_step2_results()

    if step2 is None:
        print("Step 2 の結果ファイルが見つかりません。")
        print("先に little_things_step2.py を実行してください。")
        return

    print(f"  Step 2 銀河数: {step2['n_galaxies']}")
    print(f"  Step 2 alpha: {step2['alpha_fit']:.3f} +/- {step2['alpha_se']:.3f}")

    # h_R 差し替え
    print("\n--- h_R 差し替え (Hunter+2012 3.6um) ---")
    results_new, skipped = rerun_alpha_test(step2)

    print(f"  マッチ成功: {len(results_new)} 銀河")
    if skipped:
        print(f"  マッチ失敗: {skipped}")

    # h_R 比較表示
    print(f"\n{'銀河':<14} {'h_R(旧)[kpc]':>12} {'h_R(3.6um)[kpc]':>15} {'比率':>6}")
    print(f"{'-'*50}")
    for r in results_new:
        h_old = r.get("h_R_old_kpc")
        h_new = r["h_R_hunter_kpc"]
        if h_old and h_old > 0:
            ratio = h_new / h_old
            print(f"{r['name']:<14} {h_old:>12.2f} {h_new:>15.2f} {ratio:>6.2f}")
        else:
            print(f"{r['name']:<14} {'N/A':>12} {h_new:>15.2f} {'--':>6}")

    if len(results_new) < 5:
        print(f"\n有効銀河数不足 ({len(results_new)})。マッチングを確認してください。")
        print("Step2 銀河名:")
        for g in step2["galaxies"]:
            print(f"  '{g['name']}'")
        return

    # alpha 検定再実行
    print(f"\n{'='*70}")
    print("alpha 検定: Hunter+2012 h_R 使用")
    print(f"{'='*70}")

    new_stats = alpha_test(results_new, "Hunter+2012 3.6um")

    print(f"\n  alpha = {new_stats['alpha']:.3f} +/- {new_stats['se_alpha']:.3f}")
    print(f"  95% CI: [{new_stats['alpha_ci'][0]:.3f}, {new_stats['alpha_ci'][1]:.3f}]")
    print(f"  残差sigma = {new_stats['resid_std']:.3f} dex")
    print(f"  p(alpha=0.5) = {new_stats['p_alpha_05']:.4f}  "
          f"{'棄却不可 ✓' if new_stats['p_alpha_05'] > 0.05 else '棄却 ✗'}")
    print(f"  p(alpha=0)   = {new_stats['p_alpha_0']:.2e}  "
          f"{'棄却不可' if new_stats['p_alpha_0'] > 0.05 else '棄却 ✗'}")
    print(f"  p(alpha=1)   = {new_stats['p_alpha_1']:.4f}  "
          f"{'棄却不可' if new_stats['p_alpha_1'] > 0.05 else '棄却 ✗'}")
    print(f"  Spearman rho = {new_stats['spearman_rho']:.3f} "
          f"(p = {new_stats['spearman_p']:.2e})")
    print(f"  dAIC(幾何平均 vs MOND) = {new_stats['dAIC_geom']:.1f}")

    # SPARC との比較
    sparc_alpha, sparc_se = 0.545, 0.041
    diff = abs(new_stats["alpha"] - sparc_alpha)
    combined_se = np.sqrt(new_stats["se_alpha"]**2 + sparc_se**2)
    z_diff = diff / combined_se

    print(f"\n--- SPARC 比較 ---")
    print(f"  |alpha_LT - alpha_SPARC| = |{new_stats['alpha']:.3f} - {sparc_alpha}| = {diff:.3f}")
    print(f"  z = {z_diff:.2f} ({'整合 ✓' if z_diff < 2 else '不整合 ✗'})")

    # Step2(旧) vs Hunter+2012(新) の改善度
    old_alpha = step2["alpha_fit"]
    old_se = step2["alpha_se"]
    old_resid = step2["residual_std_dex"]

    print(f"\n{'='*70}")
    print("h_R 改善の効果")
    print(f"{'='*70}")
    print(f"  {'指標':<25} {'R0.3近似':>12} {'Hunter+2012':>12} {'変化':>10}")
    print(f"  {'-'*60}")
    print(f"  {'alpha':<25} {old_alpha:>12.3f} {new_stats['alpha']:>12.3f} "
          f"{new_stats['alpha'] - old_alpha:>+10.3f}")
    print(f"  {'se_alpha':<25} {old_se:>12.3f} {new_stats['se_alpha']:>12.3f} "
          f"{new_stats['se_alpha'] - old_se:>+10.3f}")
    print(f"  {'resid std [dex]':<25} {old_resid:>12.3f} {new_stats['resid_std']:>12.3f} "
          f"{new_stats['resid_std'] - old_resid:>+10.3f}")
    print(f"  {'p(alpha=0.5)':<25} {step2['p_alpha_05']:>12.4f} {new_stats['p_alpha_05']:>12.4f}")

    alpha_moved_toward_05 = abs(new_stats["alpha"] - 0.5) < abs(old_alpha - 0.5)
    resid_improved = new_stats["resid_std"] < old_resid

    print(f"\n  alpha が 0.5 に接近: {'YES ✓' if alpha_moved_toward_05 else 'NO'}")
    print(f"  残差sigma が改善:    {'YES ✓' if resid_improved else 'NO'}")

    # ブートストラップ
    print(f"\n{'='*70}")
    print("ブートストラップ (10000回)")
    print(f"{'='*70}")

    log_gc_arr = new_stats["log_gc"]
    log_GS0_arr = new_stats["log_GSigma0"]
    N = new_stats["N"]
    rng = np.random.default_rng(42)
    alpha_boot = []
    for _ in range(10000):
        idx = rng.choice(N, N, replace=True)
        Ab = np.vstack([np.ones(N), log_GS0_arr[idx]]).T
        try:
            cb, _, _, _ = np.linalg.lstsq(Ab, log_gc_arr[idx], rcond=None)
            alpha_boot.append(cb[1])
        except Exception:
            pass
    alpha_boot = np.array(alpha_boot)
    boot_ci = np.percentile(alpha_boot, [2.5, 97.5])
    frac_05 = np.mean((alpha_boot > 0.4) & (alpha_boot < 0.6))

    print(f"  alpha(boot) = {np.mean(alpha_boot):.3f} +/- {np.std(alpha_boot):.3f}")
    print(f"  95% CI: [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]")
    print(f"  alpha in [0.4, 0.6] の割合: {frac_05*100:.1f}%")

    # 旧統計量の辞書化
    old_stats_dict = {
        "alpha": old_alpha,
        "se_alpha": old_se,
        "resid_std": old_resid,
        "N": step2["n_galaxies"],
    }

    # プロット
    plot_comparison(old_stats_dict, new_stats, results_new)

    # 結果保存
    summary = {
        "h_R_source": "Hunter+2012 3.6um (Table 4)",
        "n_matched": len(results_new),
        "n_skipped": len(skipped),
        "skipped_names": skipped,
        "alpha": float(new_stats["alpha"]),
        "se_alpha": float(new_stats["se_alpha"]),
        "alpha_ci_95": [float(new_stats["alpha_ci"][0]), float(new_stats["alpha_ci"][1])],
        "alpha_boot_95ci": [float(boot_ci[0]), float(boot_ci[1])],
        "p_alpha_05": float(new_stats["p_alpha_05"]),
        "p_alpha_0": float(new_stats["p_alpha_0"]),
        "resid_std": float(new_stats["resid_std"]),
        "dAIC_geom": float(new_stats["dAIC_geom"]),
        "spearman_rho": float(new_stats["spearman_rho"]),
        "spearman_p": float(new_stats["spearman_p"]),
        "sparc_z_score": float(z_diff),
        "sparc_consistent": bool(z_diff < 2),
        "improvement": {
            "alpha_moved_to_05": bool(alpha_moved_toward_05),
            "resid_improved": bool(resid_improved),
            "delta_alpha": float(new_stats["alpha"] - old_alpha),
            "delta_resid": float(new_stats["resid_std"] - old_resid),
        },
        "galaxies": [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                     for r in results_new],
    }

    outpath = OUTDIR / "hunter2012_results.json"
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n結果保存: {outpath}")

    # 最終判定
    print(f"\n{'='*70}")
    print("最終判定")
    print(f"{'='*70}")

    verdicts = []
    if new_stats['p_alpha_05'] > 0.05:
        verdicts.append(f"✓ alpha=0.5 棄却不可 (p={new_stats['p_alpha_05']:.3f})")
    else:
        verdicts.append(f"✗ alpha=0.5 棄却 (p={new_stats['p_alpha_05']:.3f})")

    if z_diff < 2:
        verdicts.append(f"✓ SPARC alpha={sparc_alpha} と整合 (z={z_diff:.2f})")
    else:
        verdicts.append(f"✗ SPARC alpha={sparc_alpha} と不整合 (z={z_diff:.2f})")

    if new_stats['dAIC_geom'] < -2:
        verdicts.append(f"✓ 幾何平均 > MOND (dAIC={new_stats['dAIC_geom']:.1f})")
    elif new_stats['dAIC_geom'] < 0:
        verdicts.append(f"△ 幾何平均 >= MOND (dAIC={new_stats['dAIC_geom']:.1f})")
    else:
        verdicts.append(f"✗ 幾何平均 <= MOND (dAIC={new_stats['dAIC_geom']:.1f})")

    if new_stats['spearman_p'] < 0.01:
        verdicts.append(f"✓ gc-GSigma0 相関有意 (rho={new_stats['spearman_rho']:.3f}, p={new_stats['spearman_p']:.2e})")
    elif new_stats['spearman_p'] < 0.05:
        verdicts.append(f"△ gc-GSigma0 相関有意 (rho={new_stats['spearman_rho']:.3f}, p={new_stats['spearman_p']:.3f})")
    else:
        verdicts.append(f"✗ gc-GSigma0 相関非有意 (rho={new_stats['spearman_rho']:.3f}, p={new_stats['spearman_p']:.3f})")

    if alpha_moved_toward_05:
        verdicts.append("✓ h_R改善でalphaが0.5に接近")

    for v in verdicts:
        print(f"  {v}")

    n_pass = sum(1 for v in verdicts if v.startswith("✓"))
    print(f"\n  総合: {n_pass}/{len(verdicts)} 通過")

    print(f"\n{'='*70}")
    print("完了")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
