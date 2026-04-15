"""
Hunter+2012 の正しい V-band R_d (kpc) で α 検定を再実行。
VizieR J/AJ/144/134 table1.dat から取得した実測値を使用。
3.6μm スケール長はこのカタログに含まれないため V-band を使用。

uv run --with scipy --with matplotlib --with numpy python hunter2012_correct.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("little_things_results"); OUTDIR.mkdir(exist_ok=True)
a0 = 1.2e-10  # m/s²
kpc_m = 3.086e19

# ================================================================
# Hunter+2012 V-band R_d (kpc) — VizieR J/AJ/144/134 table1.dat
# ================================================================
# {name: (dist_Mpc, R_d_kpc, e_Rd_kpc)}
HUNTER2012_RD = {
    "CVnIdwA":  (3.6, 0.57, 0.12),
    "DDO43":    (7.8, 0.41, 0.03),
    "DDO46":    (6.1, 1.14, 0.06),
    "DDO47":    (5.2, 1.37, 0.06),
    "DDO50":    (3.4, 1.10, 0.05),
    "DDO52":    (10.3, 1.30, 0.13),
    "DDO53":    (3.6, 0.72, 0.06),
    "DDO70":    (1.3, 0.48, 0.01),
    "DDO75":    (1.3, 0.22, 0.01),
    "DDO87":    (7.7, 1.31, 0.12),
    "DDO101":   (6.4, 0.94, 0.03),
    "DDO126":   (4.9, 0.87, 0.03),
    "DDO133":   (3.5, 1.24, 0.09),
    "DDO154":   (3.7, 0.59, 0.03),
    "DDO168":   (4.3, 0.82, 0.01),
    "DDO210":   (0.9, 0.17, 0.01),
    "DDO216":   (1.1, 0.54, 0.01),
    "F564-V3":  (8.7, 0.53, 0.03),
    "Haro29":   (5.8, 0.29, 0.01),
    "Haro36":   (9.3, 0.69, 0.01),
    "IC10":     (0.7, 0.40, 0.01),
    "IC1613":   (0.7, 0.58, 0.02),
    "NGC1569":  (3.4, 0.38, 0.02),
    "NGC2366":  (3.4, 1.36, 0.04),
    "NGC3738":  (4.9, 0.78, 0.01),
    "UGC8508":  (2.6, 0.27, 0.01),
    "WLM":      (1.0, 0.57, 0.03),
}

def load_step2():
    path = OUTDIR / "step2_results.json"
    with open(path) as f:
        return json.load(f)

def match_name(name):
    """Oh+2015 名 → Hunter+2012 名のマッチング"""
    clean = name.replace("_", "").replace("-", "").replace(" ", "").upper()
    for key in HUNTER2012_RD:
        if key.replace("-","").replace("_","").replace(" ","").upper() == clean:
            return key
    # 部分一致
    for key in HUNTER2012_RD:
        kc = key.replace("-","").replace("_","").replace(" ","").upper()
        if kc in clean or clean in kc:
            return key
    return None

def main():
    print("=" * 70)
    print("Hunter+2012 正しい V-band R_d で α 検定")
    print("=" * 70)

    step2 = load_step2()
    print(f"Step 2: {step2['n_galaxies']} 銀河, α={step2['alpha_fit']:.3f}±{step2['alpha_se']:.3f}")

    # h_R 差し替え
    results = []
    skipped = []

    for gal in step2["galaxies"]:
        name = gal["name"]
        key = match_name(name)
        if key is None:
            skipped.append(name)
            continue

        dist, Rd_kpc, eRd = HUNTER2012_RD[key]
        h_R_si = Rd_kpc * kpc_m
        v_flat_si = gal["v_flat"] * 1e3
        GS0 = v_flat_si**2 / h_R_si

        entry = dict(gal)
        entry["h_R_R03_kpc"] = gal.get("h_R_kpc", None)
        entry["h_R_hunter_kpc"] = Rd_kpc
        entry["e_h_R_hunter"] = eRd
        entry["G_Sigma0"] = GS0
        entry["log_G_Sigma0"] = np.log10(GS0)
        results.append(entry)

    print(f"マッチ: {len(results)} 銀河, スキップ: {skipped}")

    # h_R 比較テーブル
    print(f"\n{'銀河':<12} {'h_R(R03)':>8} {'h_R(H12)':>8} {'比率':>6} {'V_flat':>6}")
    print("-" * 50)
    for r in results:
        h_old = r.get("h_R_R03_kpc")
        h_new = r["h_R_hunter_kpc"]
        ratio = h_new / h_old if h_old and h_old > 0 else float('nan')
        print(f"{r['name']:<12} {h_old:>8.2f} {h_new:>8.2f} {ratio:>6.2f} {r['v_flat']:>6.1f}")

    # α 検定
    N = len(results)
    log_gc = np.array([np.log10(r["g_c"]) for r in results])
    log_GS0 = np.array([r["log_G_Sigma0"] for r in results])

    A = np.vstack([np.ones(N), log_GS0]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_gc, rcond=None)
    intercept, alpha_fit = coeffs

    y_pred = A @ coeffs
    resid = log_gc - y_pred
    s2 = np.sum(resid**2) / (N - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    se_alpha = np.sqrt(cov[1, 1])
    resid_std = np.std(resid)

    t_crit = t_dist.ppf(0.975, N - 2)
    ci = (alpha_fit - t_crit * se_alpha, alpha_fit + t_crit * se_alpha)

    t05 = (alpha_fit - 0.5) / se_alpha
    p05 = 2 * t_dist.sf(abs(t05), N - 2)
    t0 = alpha_fit / se_alpha
    p0 = 2 * t_dist.sf(abs(t0), N - 2)
    t1 = (alpha_fit - 1.0) / se_alpha
    p1 = 2 * t_dist.sf(abs(t1), N - 2)

    rho, p_sp = spearmanr(log_GS0, log_gc)

    # AIC
    gc_mond = np.full(N, np.log10(a0))
    rss_mond = np.sum((log_gc - gc_mond)**2)
    aic_mond = N * np.log(rss_mond / N)

    gc_geom = 0.5 * (np.log10(a0) + log_GS0)
    eta_sh = np.mean(log_gc - gc_geom)
    rss_geom = np.sum((log_gc - gc_geom - eta_sh)**2)
    aic_geom = N * np.log(rss_geom / N) + 2
    d_geom = aic_geom - aic_mond

    rss_free = np.sum(resid**2)
    aic_free = N * np.log(rss_free / N) + 4
    d_free = aic_free - aic_mond

    print(f"\n{'='*70}")
    print(f"α 検定  N = {N}")
    print(f"{'='*70}")
    print(f"  log(gc) = {intercept:.3f} + {alpha_fit:.3f} * log(G·Σ₀)")
    print(f"  α = {alpha_fit:.3f} ± {se_alpha:.3f}")
    print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  残差σ = {resid_std:.3f} dex")
    print(f"\n  p(α=0.5) = {p05:.4f}  {'棄却不可 ✓' if p05 > 0.05 else '棄却 ✗'}")
    print(f"  p(α=0)   = {p0:.2e}  {'棄却不可' if p0 > 0.05 else '棄却 ✗'}")
    print(f"  p(α=1)   = {p1:.4f}  {'棄却不可' if p1 > 0.05 else '棄却 ✗'}")
    print(f"  Spearman ρ = {rho:.3f} (p = {p_sp:.2e})")
    print(f"  ΔAIC(幾何平均) = {d_geom:.1f}")
    print(f"  ΔAIC(α自由)   = {d_free:.1f}")
    print(f"  η(α=0.5) = 10^{eta_sh:.3f} = {10**eta_sh:.3f}")

    # SPARC比較
    sp_a, sp_se = 0.545, 0.041
    diff = abs(alpha_fit - sp_a)
    comb = np.sqrt(se_alpha**2 + sp_se**2)
    z = diff / comb
    print(f"\n  SPARC比較: z = {z:.2f} ({'整合 ✓' if z < 2 else '不整合 ✗'})")

    # ブートストラップ
    rng = np.random.default_rng(42)
    ab = []
    for _ in range(10000):
        idx = rng.choice(N, N, replace=True)
        Ab = np.vstack([np.ones(N), log_GS0[idx]]).T
        try:
            cb, _, _, _ = np.linalg.lstsq(Ab, log_gc[idx], rcond=None)
            ab.append(cb[1])
        except:
            pass
    ab = np.array(ab)
    bci = np.percentile(ab, [2.5, 97.5])
    print(f"\n  ブートストラップ: α = {np.mean(ab):.3f} ± {np.std(ab):.3f}")
    print(f"  Boot 95%CI: [{bci[0]:.3f}, {bci[1]:.3f}]")
    print(f"  0.5 in Boot CI: {'YES ✓' if bci[0] <= 0.5 <= bci[1] else 'NO ✗'}")

    # 3方法比較テーブル
    print(f"\n{'='*70}")
    print("3方法比較")
    print(f"{'='*70}")
    old = step2
    print(f"  {'':25} {'R03/2.0':>10} {'H12(V)':>10} {'SPARC':>10}")
    print(f"  {'-'*55}")
    print(f"  {'α':25} {old['alpha_fit']:>10.3f} {alpha_fit:>10.3f} {sp_a:>10.3f}")
    print(f"  {'σ_α':25} {old['alpha_se']:>10.3f} {se_alpha:>10.3f} {sp_se:>10.3f}")
    print(f"  {'残差σ':25} {old['residual_std_dex']:>10.3f} {resid_std:>10.3f} {'0.313':>10}")
    print(f"  {'p(α=0.5)':25} {old['p_alpha_05']:>10.4f} {p05:>10.4f} {'0.27':>10}")
    print(f"  {'ρ(Spearman)':25} {old['spearman_rho']:>10.3f} {rho:>10.3f} {'---':>10}")
    print(f"  {'ΔAIC':25} {old['dAIC_geom_vs_mond']:>10.1f} {d_geom:>10.1f} {'---':>10}")

    # アウトライヤー解析
    print(f"\n{'='*70}")
    print("アウトライヤー解析")
    print(f"{'='*70}")
    abs_resid = np.abs(resid)
    idx_sort = np.argsort(abs_resid)[::-1]
    print(f"  {'銀河':<12} {'残差[dex]':>10} {'log(gc/a0)':>10} {'h_R[kpc]':>8}")
    for i in idx_sort[:5]:
        r = results[i]
        print(f"  {r['name']:<12} {resid[i]:>+10.3f} {r['log_gc_a0']:>10.3f} "
              f"{r['h_R_hunter_kpc']:>8.2f}")

    # アウトライヤー除外テスト (|resid| > 2σ)
    mask = abs_resid < 2 * resid_std
    if np.sum(mask) < N:
        N2 = np.sum(mask)
        A2 = np.vstack([np.ones(N2), log_GS0[mask]]).T
        c2, _, _, _ = np.linalg.lstsq(A2, log_gc[mask], rcond=None)
        r2 = log_gc[mask] - A2 @ c2
        s2_2 = np.sum(r2**2) / (N2 - 2)
        se2 = np.sqrt(s2_2 * np.linalg.inv(A2.T @ A2)[1, 1])
        t05_2 = (c2[1] - 0.5) / se2
        p05_2 = 2 * t_dist.sf(abs(t05_2), N2 - 2)
        removed = [results[i]["name"] for i in range(N) if not mask[i]]
        print(f"\n  2σクリップ後 (除外: {removed}):")
        print(f"  α = {c2[1]:.3f} ± {se2:.3f}, p(α=0.5) = {p05_2:.4f}")

    # プロット
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Hunter+2012 V-band $R_d$ (correct values from VizieR)",
                     fontsize=14, fontweight='bold')

        # (a) log(gc) vs log(G·Σ₀)
        ax = axes[0, 0]
        ax.scatter(log_GS0, log_gc, c='steelblue', s=60, zorder=3,
                   edgecolors='navy', lw=0.5)
        xr = np.linspace(log_GS0.min()-0.3, log_GS0.max()+0.3, 100)
        ax.plot(xr, intercept + alpha_fit * xr, 'r-', lw=2,
                label=rf'$\alpha={alpha_fit:.3f}\pm{se_alpha:.3f}$')
        ax.plot(xr, 0.5*np.log10(a0) + 0.5*xr + eta_sh, 'g--', lw=1.5,
                label=r'$\alpha=0.5$')
        ax.axhline(np.log10(a0), color='gray', ls=':', lw=1, label='MOND')
        for r in results:
            ax.annotate(r["name"], (r["log_G_Sigma0"], np.log10(r["g_c"])),
                       fontsize=6, alpha=0.7, xytext=(3,3), textcoords='offset points')
        ax.set_xlabel(r"$\log(G\cdot\Sigma_0)$ [m/s$^2$]")
        ax.set_ylabel(r"$\log(g_c)$ [m/s$^2$]")
        ax.set_title(rf"$\alpha={alpha_fit:.3f}\pm{se_alpha:.3f}$, $p(\alpha=0.5)={p05:.3f}$")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (b) h_R old vs new
        ax = axes[0, 1]
        h_old = np.array([r["h_R_R03_kpc"] for r in results])
        h_new = np.array([r["h_R_hunter_kpc"] for r in results])
        ax.scatter(h_old, h_new, c='steelblue', s=50)
        lim = max(h_old.max(), h_new.max()) * 1.2
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='1:1')
        for r in results:
            ax.annotate(r["name"], (r["h_R_R03_kpc"], r["h_R_hunter_kpc"]),
                       fontsize=6, alpha=0.7, xytext=(3,3), textcoords='offset points')
        ax.set_xlabel(r"$h_R$ ($R_{0.3}/2.0$) [kpc]"); ax.set_ylabel(r"$h_R$ (Hunter+2012 V) [kpc]")
        ax.set_title("Scale length comparison"); ax.legend(); ax.grid(True, alpha=0.3)

        # (c) α CI comparison
        ax = axes[1, 0]
        yp = [0, 1, 2]
        labs = ['SPARC\n(N=175)', f'Step2 ($R_{{0.3}}$)\n(N={old["n_galaxies"]})',
                f'Hunter+2012 V\n(N={N})']
        als = [sp_a, old["alpha_fit"], alpha_fit]
        ers = [sp_se*1.96,
               old["alpha_se"]*t_dist.ppf(0.975, old["n_galaxies"]-2),
               se_alpha*t_crit]
        cols = ['navy', 'coral', 'steelblue']
        for i in range(3):
            ax.errorbar(als[i], yp[i], xerr=ers[i], fmt='o', ms=10,
                       capsize=8, color=cols[i], elinewidth=2)
        ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
        ax.axvline(0, color='gray', ls=':'); ax.axvline(1, color='gray', ls=':')
        ax.set_xlabel(r'$\alpha$'); ax.set_yticks(yp); ax.set_yticklabels(labs)
        ax.set_title(r"$\alpha$ 95% CI"); ax.legend(fontsize=9)
        ax.set_xlim(-0.5, 2.5); ax.grid(True, alpha=0.3)

        # (d) ブートストラップ
        ax = axes[1, 1]
        ax.hist(ab, bins=50, color='steelblue', edgecolor='white', alpha=0.7, density=True)
        ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
        ax.axvline(alpha_fit, color='red', lw=2, label=f'OLS')
        ax.axvline(sp_a, color='navy', ls=':', lw=1.5, label=f'SPARC')
        ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap: $\\alpha={np.mean(ab):.3f}\\pm{np.std(ab):.3f}$')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figpath = OUTDIR / "hunter2012_correct.png"
        plt.savefig(figpath, dpi=150, bbox_inches='tight')
        print(f"\nプロット: {figpath}")
    except ImportError:
        pass

    # 結果保存
    summary = {
        "h_R_source": "Hunter+2012 V-band R_d (VizieR J/AJ/144/134)",
        "n_galaxies": N,
        "alpha": float(alpha_fit), "se_alpha": float(se_alpha),
        "alpha_95ci": [float(ci[0]), float(ci[1])],
        "alpha_boot_95ci": [float(bci[0]), float(bci[1])],
        "p_alpha_05": float(p05), "p_alpha_0": float(p0), "p_alpha_1": float(p1),
        "resid_std": float(resid_std),
        "spearman_rho": float(rho), "spearman_p": float(p_sp),
        "dAIC_geom": float(d_geom), "dAIC_free": float(d_free),
        "sparc_z": float(z), "sparc_consistent": bool(z < 2),
    }
    with open(OUTDIR / "hunter2012_correct_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"結果: {OUTDIR / 'hunter2012_correct_results.json'}")

    # 最終判定
    print(f"\n{'='*70}")
    print("最終判定")
    print(f"{'='*70}")
    vd = []
    vd.append(f"{'✓' if p05>0.05 else '✗'} α=0.5: p={p05:.4f}")
    vd.append(f"{'✓' if z<2 else '✗'} SPARC整合: z={z:.2f}")
    vd.append(f"{'✓' if d_geom<-2 else '△' if d_geom<0 else '✗'} ΔAIC={d_geom:.1f}")
    vd.append(f"{'✓' if p_sp<0.01 else '△' if p_sp<0.05 else '✗'} ρ={rho:.3f}, p={p_sp:.2e}")
    vd.append(f"{'✓' if bci[0]<=0.5<=bci[1] else '✗'} Boot CI に 0.5 含む")
    for v in vd:
        print(f"  {v}")
    n_pass = sum(1 for v in vd if v.startswith("✓"))
    print(f"\n  {n_pass}/{len(vd)} 通過")

if __name__ == "__main__":
    main()
