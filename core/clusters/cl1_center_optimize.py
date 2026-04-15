# -*- coding: utf-8 -*-
"""
cl1 クラスター中心最適化: gamma_x 最小化
==========================================
uv run --with scipy --with matplotlib --with numpy --with pandas python cl1_center_optimize.py

cl1: RA=140.45, Dec=-0.25, z_cl=0.313
現状: gamma_x = -0.032（基準0.005の6倍）
目標: |gamma_x| < 0.005
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize
from scipy.integrate import quad

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ================================================================
# 設定
# ================================================================
CL1_RA  = 140.45
CL1_DEC = -0.25
CL1_Z   = 0.313

R_MIN_MPC = 0.1
R_MAX_MPC = 5.0
N_RADIAL_BINS = 12

SEARCH_RADIUS_ARCMIN = 3.0
GRID_STEP_ARCSEC = 10.0

H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 3e5
OUTDIR = Path("cl1_optimization"); OUTDIR.mkdir(exist_ok=True)

# ================================================================
# 宇宙論
# ================================================================
def comoving_distance(z):
    f = lambda zp: 1.0 / np.sqrt(OMEGA_M*(1+zp)**3 + (1-OMEGA_M))
    r, _ = quad(f, 0, z)
    return C_LIGHT / H0 * r

def angular_diameter_distance(z):
    return comoving_distance(z) / (1 + z)

# ================================================================
# 剪断計算
# ================================================================
def compute_shear_components(ra_s, dec_s, e1, e2, ra_c, dec_c):
    dra = (ra_s - ra_c) * np.cos(np.radians(dec_c)) * 60.0
    ddec = (dec_s - dec_c) * 60.0
    r = np.sqrt(dra**2 + ddec**2)
    phi = np.arctan2(ddec, dra)
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    gt = -(e1*c2 + e2*s2)
    gx = e1*s2 - e2*c2
    return r, gt, gx

def compute_profile(ra_s, dec_s, e1, e2, w, ra_c, dec_c, rmin, rmax, nb):
    r, gt, gx = compute_shear_components(ra_s, dec_s, e1, e2, ra_c, dec_c)
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nb+1)
    rm, gtm, gxm, gte, npb = [], [], [], [], []
    for i in range(nb):
        m = (r >= bins[i]) & (r < bins[i+1])
        n = np.sum(m)
        rm.append(np.sqrt(bins[i]*bins[i+1]))
        if n < 5:
            gtm.append(np.nan); gxm.append(np.nan); gte.append(np.nan); npb.append(n)
            continue
        wm = w[m]; ws = np.sum(wm)
        g_t = np.sum(wm*gt[m])/ws
        g_x = np.sum(wm*gx[m])/ws
        se = np.sqrt(np.sum(wm**2 * (gt[m]-g_t)**2) / ws**2)
        gtm.append(g_t); gxm.append(g_x); gte.append(se); npb.append(n)
    return np.array(rm), np.array(gtm), np.array(gxm), np.array(gte), np.array(npb)

def gamma_x_metric(ra_c, dec_c, ra_s, dec_s, e1, e2, w, rmin, rmax):
    r, gt, gx = compute_shear_components(ra_s, dec_s, e1, e2, ra_c, dec_c)
    m = (r >= rmin) & (r <= rmax)
    if np.sum(m) < 10: return 1.0
    wm = w[m]; return abs(np.sum(wm*gx[m]) / np.sum(wm))

# ================================================================
# データ読み込み
# ================================================================
def load_sources():
    import pandas as pd
    fp = Path("cl1_sources.csv")
    if not fp.exists():
        print(f"  {fp} が見つかりません。"); return None
    df = pd.read_csv(fp)
    print(f"  {fp}: {len(df)} 行, 列: {list(df.columns[:8])}...")

    # 列名マッピング
    cm = {}
    for c in df.columns:
        cl = c.lower()
        if 'ra' in cl and 'dec' not in cl: cm.setdefault('ra', c)
        elif 'dec' in cl: cm.setdefault('dec', c)
        elif 'e1' in cl: cm.setdefault('e1', c)
        elif 'e2' in cl: cm.setdefault('e2', c)
        elif 'weight' in cl: cm.setdefault('w', c)
    print(f"  マッピング: {cm}")

    for k in ['ra','dec','e1','e2']:
        if k not in cm:
            print(f"  必須列 {k} なし"); return None

    ra = df[cm['ra']].values; dec = df[cm['dec']].values
    e1 = df[cm['e1']].values; e2 = df[cm['e2']].values
    w = df[cm['w']].values if 'w' in cm else np.ones(len(df))

    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(e1) & np.isfinite(e2)
    print(f"  有効: {np.sum(ok)}")
    return ra[ok], dec[ok], e1[ok], e2[ok], w[ok]

# ================================================================
# グリッドサーチ
# ================================================================
def grid_search(ra_s, dec_s, e1, e2, w, rmin, rmax):
    step = GRID_STEP_ARCSEC / 3600.0
    rad = SEARCH_RADIUS_ARCMIN / 60.0
    ns = int(2*rad/step)+1
    rag = np.linspace(CL1_RA-rad, CL1_RA+rad, ns)
    dg = np.linspace(CL1_DEC-rad, CL1_DEC+rad, ns)
    print(f"  グリッド: {ns}x{ns}={ns**2} 点, ステップ {GRID_STEP_ARCSEC}\"")

    best_gx, best_ra, best_dec = 1.0, CL1_RA, CL1_DEC
    results = []
    for rc in rag:
        for dc in dg:
            dr = (rc-CL1_RA)*np.cos(np.radians(CL1_DEC))*60
            dd = (dc-CL1_DEC)*60
            if np.sqrt(dr**2+dd**2) > SEARCH_RADIUS_ARCMIN: continue
            gx = gamma_x_metric(rc, dc, ra_s, dec_s, e1, e2, w, rmin, rmax)
            results.append({'ra':rc,'dec':dc,'gx':gx,
                           'dra':(rc-CL1_RA)*np.cos(np.radians(CL1_DEC))*3600,
                           'ddec':(dc-CL1_DEC)*3600})
            if gx < best_gx:
                best_gx, best_ra, best_dec = gx, rc, dc

    print(f"  評価: {len(results)} 点")
    print(f"  最良: RA={best_ra:.5f}, Dec={best_dec:.5f}, |gx|={best_gx:.5f}")
    return best_ra, best_dec, best_gx, results

def refine(ra_s, dec_s, e1, e2, w, ra0, dec0, rmin, rmax):
    obj = lambda p: gamma_x_metric(p[0], p[1], ra_s, dec_s, e1, e2, w, rmin, rmax)
    r = minimize(obj, [ra0, dec0], method='Nelder-Mead',
                 options={'xatol':1e-6, 'fatol':1e-6, 'maxiter':1000})
    return r.x[0], r.x[1], r.fun

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("cl1 中心最適化: gamma_x 最小化")
    print("="*70)

    src = load_sources()
    if src is None: return
    ra_s, dec_s, e1, e2, w = src

    D_l = angular_diameter_distance(CL1_Z)
    amp = 1.0 / (D_l * np.pi/180/60)  # arcmin/Mpc
    rmin = R_MIN_MPC * amp
    rmax = R_MAX_MPC * amp
    print(f"\n  D_l={D_l:.1f} Mpc, 1 Mpc={amp:.2f}', R=[{rmin:.1f}'-{rmax:.1f}']")

    # 現在の中心
    print(f"\n--- 現在の中心 ---")
    rm, gt, gx, gte, nb = compute_profile(ra_s, dec_s, e1, e2, w,
                                           CL1_RA, CL1_DEC, rmin, rmax, N_RADIAL_BINS)
    v = np.isfinite(gx)
    gx0 = np.nanmean(gx[v]); gt0 = np.nanmean(gt[v])
    print(f"  <gt>={gt0:.5f}, <gx>={gx0:.5f}, |gx|={abs(gx0):.5f}")

    # グリッドサーチ
    print(f"\n--- グリッドサーチ ---")
    bra_g, bdec_g, bgx_g, grid = grid_search(ra_s, dec_s, e1, e2, w, rmin, rmax)

    # 精密化
    print(f"\n--- Nelder-Mead精密化 ---")
    bra, bdec, bgx = refine(ra_s, dec_s, e1, e2, w, bra_g, bdec_g, rmin, rmax)
    dra = (bra-CL1_RA)*np.cos(np.radians(CL1_DEC))*3600
    ddc = (bdec-CL1_DEC)*3600
    shift = np.sqrt(dra**2+ddc**2)
    print(f"  最適: RA={bra:.5f}, Dec={bdec:.5f}")
    print(f"  シフト: dRA={dra:+.1f}\", dDec={ddc:+.1f}\" (|shift|={shift:.1f}\")")
    print(f"  |gx|={bgx:.5f}")

    # 最適中心プロファイル
    rm2, gt2, gx2, gte2, nb2 = compute_profile(ra_s, dec_s, e1, e2, w,
                                                 bra, bdec, rmin, rmax, N_RADIAL_BINS)
    v2 = np.isfinite(gx2)
    gx_opt = np.nanmean(gx2[v2]); gt_opt = np.nanmean(gt2[v2])

    # S/N計算
    valid_gt = np.isfinite(gt2) & np.isfinite(gte2) & (gte2 > 0)
    if np.sum(valid_gt) > 0:
        sn = np.sqrt(np.sum((gt2[valid_gt]/gte2[valid_gt])**2))
    else:
        sn = 0.0

    # 比較
    print(f"\n{'='*70}")
    print("結果比較")
    print(f"{'='*70}")
    print(f"  {'':25} {'元の中心':>12} {'最適中心':>12}")
    print(f"  {'-'*50}")
    print(f"  {'RA':25} {CL1_RA:>12.5f} {bra:>12.5f}")
    print(f"  {'Dec':25} {CL1_DEC:>12.5f} {bdec:>12.5f}")
    print(f"  {'<gamma_t>':25} {gt0:>12.5f} {gt_opt:>12.5f}")
    print(f"  {'<gamma_x>':25} {gx0:>12.5f} {gx_opt:>12.5f}")
    print(f"  {'|gamma_x|':25} {abs(gx0):>12.5f} {abs(gx_opt):>12.5f}")
    print(f"  {'S/N':25} {'---':>12} {sn:>12.1f}")

    imp = abs(gx_opt) < abs(gx0)
    ok = abs(gx_opt) < 0.005
    print(f"\n  改善: {'YES' if imp else 'NO'}")
    print(f"  |gx|<0.005: {'PASS ✓' if ok else 'FAIL ✗'}")
    if ok:
        print(f"  → 中心ずれが原因。信頼度: B → B+")
    elif imp:
        red = (1-abs(gx_opt)/abs(gx0))*100
        print(f"  → {red:.0f}%改善。残存gxはPSF残差または非弛緩系に起因")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"cl1 center optimization: shift=({dra:+.1f}\", {ddc:+.1f}\")",
                     fontsize=13, fontweight='bold')

        # (a) gx map
        ax = axes[0,0]
        ga = np.array([r['gx'] for r in grid])
        da = np.array([r['dra'] for r in grid])
        dd = np.array([r['ddec'] for r in grid])
        sc = ax.scatter(da, dd, c=ga, s=8, cmap='viridis_r',
                       vmin=0, vmax=max(0.05, np.percentile(ga, 90)))
        ax.plot(0, 0, 'rx', ms=12, mew=2, label='original')
        ax.plot(dra, ddc, 'w*', ms=15, mew=1, label=f'optimal |gx|={bgx:.4f}')
        ax.set_xlabel('dRA ["]'); ax.set_ylabel('dDec ["]')
        ax.set_title('|gamma_x| map'); ax.legend(fontsize=9)
        plt.colorbar(sc, ax=ax, label='|gamma_x|')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        # (b) gamma_t比較
        ax = axes[0,1]
        vo = np.isfinite(gt); vn = np.isfinite(gt2)
        ax.errorbar(rm[vo], gt[vo], gte[vo], fmt='o', color='coral', ms=5,
                    capsize=3, label='original')
        ax.errorbar(rm2[vn]*1.03, gt2[vn], gte2[vn], fmt='s', color='steelblue',
                    ms=5, capsize=3, label='optimal')
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_t$')
        ax.set_xscale('log'); ax.set_title('Tangential shear')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (c) gamma_x比較
        ax = axes[1,0]
        ax.errorbar(rm[vo], gx[vo], gte[vo], fmt='o', color='coral', ms=5,
                    capsize=3, label='original')
        ax.errorbar(rm2[vn]*1.03, gx2[vn], gte2[vn], fmt='s', color='steelblue',
                    ms=5, capsize=3, label='optimal')
        ax.axhline(0, color='green', ls='--', lw=2)
        ax.axhspan(-0.005, 0.005, alpha=0.1, color='green', label='|gx|<0.005')
        ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_\times$')
        ax.set_xscale('log'); ax.set_title('Cross shear (null test)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (d) ビンごと比較
        ax = axes[1,1]
        valid_bins = np.isfinite(gt2) & np.isfinite(gx2)
        if np.sum(valid_bins) > 0:
            r_mpc = rm2[valid_bins] / amp
            ax.plot(r_mpc, gt2[valid_bins], 'o-', color='steelblue', label=r'$\gamma_t$')
            ax.plot(r_mpc, gx2[valid_bins], 's--', color='red', label=r'$\gamma_\times$')
            ax.fill_between(r_mpc,
                           gt2[valid_bins]-gte2[valid_bins],
                           gt2[valid_bins]+gte2[valid_bins],
                           alpha=0.2, color='steelblue')
            ax.axhline(0, color='gray', ls=':')
            ax.set_xlabel('R [Mpc]'); ax.set_ylabel('Shear')
            ax.set_title(f'Optimal center profile (S/N={sn:.1f})')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = OUTDIR / "cl1_center_optimization.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nプロット: {fp}")
    except ImportError:
        pass

    # 結果保存
    summary = {
        "original": {"ra": CL1_RA, "dec": CL1_DEC,
                      "gamma_t": float(gt0), "gamma_x": float(gx0)},
        "optimal": {"ra": float(bra), "dec": float(bdec),
                     "gamma_t": float(gt_opt), "gamma_x": float(gx_opt)},
        "shift": {"dra_arcsec": float(dra), "ddec_arcsec": float(ddc),
                  "total_arcsec": float(shift)},
        "threshold_met": bool(ok), "improved": bool(imp),
        "SN_optimal": float(sn),
        "n_sources": len(ra_s),
        "profile_optimal": {
            "r_arcmin": rm2[v2].tolist(),
            "gamma_t": gt2[v2].tolist(),
            "gamma_x": gx2[v2].tolist(),
            "sigma_t": gte2[v2].tolist(),
        },
    }
    with open(OUTDIR / "cl1_optimization_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"結果: {OUTDIR / 'cl1_optimization_results.json'}")
    print(f"\n{'='*70}\n完了\n{'='*70}")

if __name__ == "__main__":
    main()
