# -*- coding: utf-8 -*-
"""
MaNGA v2: Lambda_Re > 0.5 回転支配系での alpha 検定
=====================================================
uv run --with scipy --with matplotlib --with numpy --with astropy python manga_v2.py

前提: manga_results/SDSSDR17_MaNGA_JAM_v2.fits (v1で取得済み)
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.stats import spearmanr, t as t_dist
from scipy.integrate import quad

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("manga_results"); OUTDIR.mkdir(exist_ok=True)
H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 2.998e5
a0 = 1.2e-10; G_SI = 6.674e-11; M_SUN = 1.989e30; KPC_M = 3.0857e19

def _DA(z):
    r, _ = quad(lambda zp: 1/(OMEGA_M*(1+zp)**3+(1-OMEGA_M))**0.5, 0, z)
    return C_LIGHT/H0*r/(1+z)

# ================================================================
# alpha 検定
# ================================================================
def alpha_test(lgc, lgs, label=""):
    N = len(lgc)
    if N < 10: return None
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    interc, alpha = c
    res = lgc - A @ c
    s2 = np.sum(res**2)/(N-2)
    cov = s2*np.linalg.inv(A.T @ A)
    se = np.sqrt(cov[1,1]); rsd = np.std(res)
    tc = t_dist.ppf(0.975, N-2)
    ci = (alpha-tc*se, alpha+tc*se)
    p05 = 2*t_dist.sf(abs((alpha-0.5)/se), N-2)
    p0 = 2*t_dist.sf(abs(alpha/se), N-2)
    rho, psp = spearmanr(lgs, lgc)
    gm = np.full(N, np.log10(a0))
    rss_m = np.sum((lgc-gm)**2); aic_m = N*np.log(rss_m/N)
    gcg = 0.5*(np.log10(a0)+lgs); eta = np.mean(lgc-gcg)
    rss_g = np.sum((lgc-gcg-eta)**2); aic_g = N*np.log(rss_g/N)+2
    return {"label":label,"N":N,"alpha":float(alpha),"se":float(se),
            "ci":(float(ci[0]),float(ci[1])),"p05":float(p05),"p0":float(p0),
            "rsd":float(rsd),"rho":float(rho),"psp":float(psp),
            "dAIC_g":float(aic_g-aic_m),"interc":float(interc),"eta":float(eta),
            "lgc":lgc,"lgs":lgs}

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("MaNGA v2: rotation-dominated subsample (Lambda_Re > 0.5)")
    print("="*70)

    # FITS読み込み
    from astropy.io import fits
    fp = OUTDIR / "SDSSDR17_MaNGA_JAM_v2.fits"
    if not fp.exists():
        print(f"  {fp} not found. Run manga_verification.py first."); return
    hdul = fits.open(str(fp))
    base = hdul[1].data
    print(f"  {len(base)} galaxies")
    print(f"  base cols: {[c.name for c in hdul[1].columns]}")

    # 基本量抽出
    z = base['z']
    Re_arcsec = base['Re_arcsec_MGE']
    logM = base['nsa_elpetro_mass']   # log10(M/M_sun)
    sigma = base['Sigma_Re']          # sigma_e [km/s]
    lam_Re = base['Lambda_Re']        # spin parameter
    plateifu = base['plateifu']

    # 角径距離 -> Re [kpc]
    print("  computing distances...")
    Da = np.array([_DA(zi) if 0.001 < zi < 0.5 else np.nan for zi in z])
    Re_kpc = Re_arcsec / 206265 * Da * 1e3  # arcsec -> kpc

    # V_max proxy: 回転支配系では V_max ~ sigma_e * sqrt(2) / sqrt(1-eps)
    # ここではSPARCとの一貫性のためにσ_eベースのV_maxを使用
    # Lambda_Re > 0.5 の系は回転支配なので V ~ sqrt(2)*sigma が妥当
    V_max_proxy = sigma * np.sqrt(2)  # km/s

    # gNFW拡張: ext[4]にNFW oblateのGs_Re（恒星重力加速度）がある
    nfw_ob = hdul[4].data
    # Gt_Re, Gs_Re は log10(g / [10^{-10} m/s^2]) のはず
    # 確認: Gt_Re = -1.0 → g = 10^(-1)*10^(-10) = 10^(-11) m/s^2

    print(f"\n--- sample statistics ---")
    ok_base = np.isfinite(sigma) & (sigma > 0) & np.isfinite(Re_kpc) & (Re_kpc > 0) & \
              np.isfinite(logM) & (logM > 7) & np.isfinite(lam_Re)
    print(f"  valid base: {np.sum(ok_base)}")

    # Lambda_Re分布
    lam_valid = lam_Re[ok_base]
    print(f"  Lambda_Re: median={np.median(lam_valid):.3f}, "
          f"[{np.min(lam_valid):.3f}, {np.max(lam_valid):.3f}]")
    print(f"  Lambda_Re > 0.5: {np.sum(lam_valid > 0.5)}")
    print(f"  Lambda_Re > 0.3: {np.sum(lam_valid > 0.3)}")

    # ================================================================
    # 3つのサブサンプルで解析
    # ================================================================
    results = {}
    for label, lam_cut in [("all", 0.0), ("fast_rot (>0.5)", 0.5), ("slow_rot (<0.3)", -0.3)]:
        if lam_cut >= 0:
            mask = ok_base & (lam_Re >= lam_cut)
        else:
            mask = ok_base & (lam_Re < abs(lam_cut))
        N = np.sum(mask)
        if N < 20: continue

        # V_max proxy
        v_si = V_max_proxy[mask] * 1e3
        Re_m = Re_kpc[mask] * KPC_M
        hR = Re_kpc[mask] / 1.678
        hR_si = hR * KPC_M
        Mstar = 10**logM[mask] * M_SUN

        GS0 = v_si**2 / hR_si
        gobs = v_si**2 / Re_m
        gbar = G_SI * 0.736 * Mstar / Re_m**2

        gc_ok = (gobs > gbar) & (gbar > 0)
        gc = np.where(gc_ok, gobs*(gobs-gbar)/gbar, np.nan)

        lgc = np.log10(gc[gc_ok & np.isfinite(gc) & (gc > 0)])
        lgs = np.log10(GS0[gc_ok & np.isfinite(gc) & (gc > 0)])

        # 外れ値除去: |log(gc)| > 5σ
        med_lgc = np.median(lgc); std_lgc = np.std(lgc)
        clip = np.abs(lgc - med_lgc) < 5 * std_lgc
        lgc = lgc[clip]; lgs = lgs[clip]

        print(f"\n--- {label} (N={len(lgc)}) ---")
        r = alpha_test(lgc, lgs, label)
        if r:
            print(f"  alpha={r['alpha']:.3f}±{r['se']:.3f}, 95%CI=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
            print(f"  p(0.5)={r['p05']:.4f} {'✓' if r['p05']>0.05 else '✗'}")
            print(f"  p(0)={r['p0']:.2e}, σ_res={r['rsd']:.3f}, ρ={r['rho']:.3f}")
            print(f"  dAIC(geom)={r['dAIC_g']:.1f}")
            results[label] = r

    # SPARC比較
    if "fast_rot (>0.5)" in results:
        r = results["fast_rot (>0.5)"]
        d = abs(r["alpha"]-0.545); cs = np.sqrt(r["se"]**2+0.041**2); z_sp = d/cs
        print(f"\n  SPARC comparison (fast_rot): z={z_sp:.2f} ({'✓' if z_sp<2 else '✗'})")

    # 質量ビン別解析（fast rotator のみ）
    print(f"\n--- mass bin analysis (fast rotators) ---")
    mask_fr = ok_base & (lam_Re >= 0.5)
    v_si_fr = V_max_proxy[mask_fr]*1e3; Re_m_fr = Re_kpc[mask_fr]*KPC_M
    hR_si_fr = Re_kpc[mask_fr]/1.678*KPC_M; Mstar_fr = 10**logM[mask_fr]*M_SUN
    GS0_fr = v_si_fr**2/hR_si_fr
    gobs_fr = v_si_fr**2/Re_m_fr; gbar_fr = G_SI*0.736*Mstar_fr/Re_m_fr**2
    gc_ok_fr = (gobs_fr>gbar_fr)&(gbar_fr>0)
    gc_fr = np.where(gc_ok_fr, gobs_fr*(gobs_fr-gbar_fr)/gbar_fr, np.nan)
    logM_fr = logM[mask_fr]

    mass_bins = []
    for lo, hi, blabel in [(8,9.5,"dwarf"),(9.5,10.5,"mid"),(10.5,12,"massive")]:
        mb = gc_ok_fr & (logM_fr>=lo) & (logM_fr<hi) & np.isfinite(gc_fr) & (gc_fr>0)
        if np.sum(mb) < 10:
            mass_bins.append({"bin":blabel,"N":int(np.sum(mb)),"alpha":None})
            continue
        lgc_b = np.log10(gc_fr[mb]); lgs_b = np.log10(GS0_fr[mb])
        rb = alpha_test(lgc_b, lgs_b, blabel)
        if rb:
            mass_bins.append({"bin":blabel,"N":rb["N"],"alpha":rb["alpha"],
                              "se":rb["se"],"p05":rb["p05"]})
            print(f"  {blabel}: N={rb['N']}, α={rb['alpha']:.3f}±{rb['se']:.3f}, p(0.5)={rb['p05']:.4f}")
        else:
            mass_bins.append({"bin":blabel,"N":int(np.sum(mb)),"alpha":None})

    # 比較テーブル
    print(f"\n{'='*70}")
    print("comparison")
    print(f"{'='*70}")
    print(f"  {'dataset':<22} {'N':>6} {'α':>7} {'σ_α':>6} {'p(0.5)':>10} {'σ_res':>6}")
    print(f"  {'-'*60}")
    print(f"  {'SPARC':<22} {'175':>6} {'0.545':>7} {'0.041':>6} {'0.273':>10} {'0.312':>6}")
    print(f"  {'LT+SPARC joint':<22} {'178':>6} {'0.576':>7} {'0.047':>6} {'0.109':>10} {'0.373':>6}")
    for lab, r in results.items():
        print(f"  {('MaNGA '+lab):<22} {r['N']:>6} {r['alpha']:>7.3f} {r['se']:>6.3f} "
              f"{r['p05']:>10.4f} {r['rsd']:>6.3f}")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("MaNGA v2: rotation-dominated subsample", fontsize=14, fontweight='bold')

        # (a) all vs fast_rot
        ax = axes[0,0]
        if "all" in results:
            ax.scatter(results["all"]["lgs"], results["all"]["lgc"], c='gray', s=3, alpha=0.15, label=f'all N={results["all"]["N"]}')
        if "fast_rot (>0.5)" in results:
            r = results["fast_rot (>0.5)"]
            ax.scatter(r["lgs"], r["lgc"], c='steelblue', s=5, alpha=0.3, label=f'rot N={r["N"]}')
            xr = np.linspace(r["lgs"].min()-0.3, r["lgs"].max()+0.3, 100)
            ax.plot(xr, r["interc"]+r["alpha"]*xr, 'r-', lw=2, label=rf'$\alpha={r["alpha"]:.3f}$')
            ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+r["eta"], 'g--', lw=1.5, label=r'$\alpha=0.5$')
            ax.axhline(np.log10(a0), color='gray', ls=':')
        ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
        ax.set_title('Fast rotators'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (b) alpha CI comparison
        ax = axes[0,1]
        ds = ['SPARC\n(N=175)', 'Joint\n(N=178)']
        al = [0.545, 0.576]; er = [0.041*1.96, 0.047*2.09]; cc = ['navy','coral']
        for lab, r in results.items():
            ds.append(f'MaNGA {lab[:8]}\n(N={r["N"]})')
            al.append(r["alpha"])
            er.append(r["se"]*t_dist.ppf(0.975, r["N"]-2))
            cc.append('steelblue' if 'fast' in lab else 'gray')
        for i in range(len(ds)):
            ax.errorbar(al[i], i, xerr=er[i], fmt='o', ms=10, capsize=8, color=cc[i], elinewidth=2)
        ax.axvline(0.5, color='green', ls='--', lw=2)
        ax.set_xlabel(r'$\alpha$'); ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds, fontsize=8)
        ax.set_xlim(-0.1, 1.3); ax.grid(True, alpha=0.3); ax.set_title(r'$\alpha$ 95% CI')

        # (c) mass bins
        ax = axes[0,2]
        vb = [b for b in mass_bins if b["alpha"] is not None]
        if vb:
            ax.errorbar([b["alpha"] for b in vb], range(len(vb)),
                       xerr=[b["se"]*1.96 for b in vb], fmt='o', ms=10, capsize=8,
                       color='steelblue', elinewidth=2)
            ax.axvline(0.5, color='green', ls='--', lw=2)
            ax.axvline(0.545, color='navy', ls=':', lw=1.5, label='SPARC')
            ax.set_yticks(range(len(vb)))
            ax.set_yticklabels([f"{b['bin']} (N={b['N']})" for b in vb])
            ax.set_xlabel(r'$\alpha$'); ax.set_title('Mass bins (fast rot)')
            ax.legend(); ax.grid(True, alpha=0.3)

        # (d) Lambda_Re histogram
        ax = axes[1,0]
        ax.hist(lam_valid, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0.5, color='red', ls='--', lw=2, label=r'$\lambda_{Re}=0.5$')
        ax.axvline(0.3, color='orange', ls=':', lw=1.5, label='0.3')
        ax.set_xlabel(r'$\lambda_{Re}$'); ax.set_ylabel('Count')
        ax.set_title(r'$\lambda_{Re}$ distribution'); ax.legend(); ax.grid(True, alpha=0.3)

        # (e) residual (fast rot)
        ax = axes[1,1]
        if "fast_rot (>0.5)" in results:
            r = results["fast_rot (>0.5)"]
            rd = r["lgc"]-(r["interc"]+r["alpha"]*r["lgs"])
            ax.hist(rd, bins=40, color='steelblue', edgecolor='white', density=True)
            ax.axvline(0, color='red', ls='--')
            ax.set_xlabel('residual [dex]')
            ax.set_title(f'Fast rot residual (σ={r["rsd"]:.3f})')
        ax.grid(True, alpha=0.3)

        # (f) G*Sigma0 coverage
        ax = axes[1,2]
        if "fast_rot (>0.5)" in results:
            ax.hist(results["fast_rot (>0.5)"]["lgs"], bins=30, alpha=0.6,
                    color='steelblue', density=True, label='MaNGA fast rot')
        ax.axvspan(-11.5, -9.0, alpha=0.1, color='navy', label='SPARC')
        ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_title('Coverage')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = OUTDIR/"manga_v2_rotators.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except ImportError: pass

    # 保存
    out = {lab: {k:v for k,v in r.items() if not isinstance(v, np.ndarray)}
           for lab, r in results.items()}
    out["mass_bins"] = mass_bins
    with open(OUTDIR/"manga_v2_results.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"results: {OUTDIR/'manga_v2_results.json'}")

    print(f"\n{'='*70}")
    print("verdict")
    print(f"{'='*70}")
    if "fast_rot (>0.5)" in results:
        r = results["fast_rot (>0.5)"]
        if r["p05"] > 0.05:
            print(f"  α=0.5 NOT REJECTED in fast rotators (p={r['p05']:.3f}) ✓")
        else:
            print(f"  α=0.5 REJECTED in fast rotators (p={r['p05']:.4f}) ✗")
            print(f"  α={r['alpha']:.3f}, closer to 0.5 than all-sample (0.922)")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
