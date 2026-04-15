# -*- coding: utf-8 -*-
"""
cl1 中心最適化 v2: chi^2 ベース gamma_x 最小化
================================================
uv run --with scipy --with matplotlib --with numpy --with pandas python cl1_center_v2.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize

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
GRID_STEP_ARCSEC = 15.0

H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 3e5
OUTDIR = Path("cl1_optimization"); OUTDIR.mkdir(exist_ok=True)

# ================================================================
# 宇宙論
# ================================================================
def angular_diameter_distance(z):
    from scipy.integrate import quad
    f = lambda zp: 1.0 / np.sqrt(OMEGA_M*(1+zp)**3 + (1-OMEGA_M))
    dc, _ = quad(f, 0, z)
    return (C_LIGHT/H0 * dc) / (1+z)

# ================================================================
# 剪断
# ================================================================
def compute_shear(ra_s, dec_s, e1, e2, ra_c, dec_c):
    dra = (ra_s - ra_c) * np.cos(np.radians(dec_c)) * 60.0
    ddec = (dec_s - dec_c) * 60.0
    r = np.sqrt(dra**2 + ddec**2)
    phi = np.arctan2(ddec, dra)
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    gt = -(e1*c2 + e2*s2)
    gx = e1*s2 - e2*c2
    return r, gt, gx

def binned_profile(ra_s, dec_s, e1, e2, w, ra_c, dec_c, rmin, rmax, nb):
    r, gt, gx = compute_shear(ra_s, dec_s, e1, e2, ra_c, dec_c)
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nb+1)
    rm = np.full(nb, np.nan); gtm = np.full(nb, np.nan)
    gxm = np.full(nb, np.nan); gte = np.full(nb, np.nan)
    nbin = np.zeros(nb, dtype=int)
    sigma_e = 0.25
    for i in range(nb):
        m = (r >= bins[i]) & (r < bins[i+1])
        n = np.sum(m); nbin[i] = n
        if n < 5: continue
        wm = w[m]; ws = np.sum(wm)
        rm[i] = np.sqrt(bins[i]*bins[i+1])
        gtm[i] = np.sum(wm*gt[m])/ws
        gxm[i] = np.sum(wm*gx[m])/ws
        n_eff = ws**2 / np.sum(wm**2)
        gte[i] = sigma_e / np.sqrt(max(n_eff, 1))
    return rm, gtm, gxm, gte, nbin

# ================================================================
# v2 メトリック: chi^2(gx) + gt>0 制約
# ================================================================
def center_metric(ra_c, dec_c, ra_s, dec_s, e1, e2, w, rmin, rmax):
    rm, gtm, gxm, gte, nb = binned_profile(ra_s, dec_s, e1, e2, w,
                                             ra_c, dec_c, rmin, rmax, N_RADIAL_BINS)
    v = np.isfinite(gxm) & np.isfinite(gte) & (gte > 0)
    if np.sum(v) < 3: return 1e6
    chi2_gx = np.sum(gxm[v]**2 / gte[v]**2)
    wv = 1.0/gte[v]**2
    gt_mean = np.sum(wv*gtm[v])/np.sum(wv)
    penalty = 0.0
    if gt_mean <= 0:
        penalty = 1000.0 + 100*abs(gt_mean)
    return chi2_gx + penalty

# ================================================================
# 中心診断
# ================================================================
def diagnose(label, ra_c, dec_c, src, rmin, rmax):
    rm, gtm, gxm, gte, nb = binned_profile(
        src['ra'], src['dec'], src['e1'], src['e2'], src['weight'],
        ra_c, dec_c, rmin, rmax, N_RADIAL_BINS)
    v = np.isfinite(gtm) & np.isfinite(gte) & (gte > 0)
    nv = np.sum(v)
    if nv < 2:
        return {"label": label, "ra": ra_c, "dec": dec_c, "valid": False}
    w = 1.0/gte[v]**2; ws = np.sum(w)
    gt_m = np.sum(w*gtm[v])/ws
    gx_m = np.sum(w*gxm[v])/ws
    gt_e = 1.0/np.sqrt(ws)
    sn = gt_m/gt_e if gt_e > 0 else 0
    chi2_gx = np.sum(gxm[v]**2/gte[v]**2)
    gx_rms = np.sqrt(np.sum(w*gxm[v]**2)/ws)
    return {
        "label": label, "ra": float(ra_c), "dec": float(dec_c), "valid": True,
        "nv": int(nv), "gt_mean": float(gt_m), "gx_mean": float(gx_m),
        "gt_err": float(gt_e), "sn": float(sn),
        "chi2_gx": float(chi2_gx), "chi2_gx_dof": float(chi2_gx/max(nv,1)),
        "gx_rms": float(gx_rms), "abs_gx": float(abs(gx_m)),
        "rm": rm.tolist(), "gt_p": gtm.tolist(), "gx_p": gxm.tolist(),
        "gt_e_p": gte.tolist(),
    }

# ================================================================
# グリッドサーチ + 精密化
# ================================================================
def grid_search(src, rmin, rmax):
    ra_s, dec_s, e1, e2, w = src['ra'], src['dec'], src['e1'], src['e2'], src['weight']
    step = GRID_STEP_ARCSEC/3600.0; rad = SEARCH_RADIUS_ARCMIN/60.0
    ns = int(2*rad/step)+1
    rag = np.linspace(CL1_RA-rad, CL1_RA+rad, ns)
    dg = np.linspace(CL1_DEC-rad, CL1_DEC+rad, ns)
    print(f"  grid: {ns}x{ns}, step={GRID_STEP_ARCSEC}\"")
    best_m, best_ra, best_dec = 1e9, CL1_RA, CL1_DEC
    pts = []
    for rc in rag:
        for dc in dg:
            dr = (rc-CL1_RA)*np.cos(np.radians(CL1_DEC))*60
            dd = (dc-CL1_DEC)*60
            if np.sqrt(dr**2+dd**2) > SEARCH_RADIUS_ARCMIN: continue
            m = center_metric(rc, dc, ra_s, dec_s, e1, e2, w, rmin, rmax)
            pts.append({"ra":rc, "dec":dc, "m":m, "dra":dr*60, "ddec":dd*60})
            if m < best_m: best_m, best_ra, best_dec = m, rc, dc
    print(f"  {len(pts)} pts, best: RA={best_ra:.5f} Dec={best_dec:.5f} metric={best_m:.1f}")
    return best_ra, best_dec, best_m, pts

def refine(src, ra0, dec0, rmin, rmax):
    ra_s, dec_s, e1, e2, w = src['ra'], src['dec'], src['e1'], src['e2'], src['weight']
    obj = lambda p: center_metric(p[0], p[1], ra_s, dec_s, e1, e2, w, rmin, rmax)
    r = minimize(obj, [ra0, dec0], method='Nelder-Mead',
                 options={'xatol':1e-6, 'fatol':1e-2, 'maxiter':500})
    return r.x[0], r.x[1], r.fun

# ================================================================
# データ読み込み
# ================================================================
def load_sources():
    import pandas as pd
    fp = Path("cl1_sources.csv")
    if not fp.exists():
        print(f"  {fp} not found"); return None
    df = pd.read_csv(fp)
    print(f"  {fp}: {len(df)} rows")
    cm = {}
    for c in df.columns:
        cl = c.lower()
        if 'ra' in cl and 'dec' not in cl: cm.setdefault('ra', c)
        elif 'dec' in cl: cm.setdefault('dec', c)
        elif 'e1' in cl: cm.setdefault('e1', c)
        elif 'e2' in cl: cm.setdefault('e2', c)
        elif 'weight' in cl: cm.setdefault('w', c)
        elif 'photo' in cl and 'z' in cl: cm.setdefault('zp', c)
        elif cl in ['z','z_best','photoz_best']: cm.setdefault('zp', c)
    for k in ['ra','dec','e1','e2']:
        if k not in cm: print(f"  missing {k}"); return None
    out = {k: df[cm[k]].values for k in ['ra','dec','e1','e2']}
    out['weight'] = df[cm['w']].values if 'w' in cm else np.ones(len(df))
    if 'zp' in cm: out['zp'] = df[cm['zp']].values
    ok = np.isfinite(out['ra']) & np.isfinite(out['dec']) & \
         np.isfinite(out['e1']) & np.isfinite(out['e2'])
    for k in out: out[k] = out[k][ok]
    print(f"  valid: {len(out['ra'])}")
    return out

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("cl1 center optimization v2 (chi^2-based)")
    print("="*70)

    src = load_sources()
    if src is None: return
    if 'zp' in src:
        m = src['zp'] > CL1_Z+0.1
        n0 = len(src['ra'])
        for k in src: src[k] = src[k][m]
        print(f"  z>{CL1_Z+0.1}: {n0}->{len(src['ra'])}")

    D_l = angular_diameter_distance(CL1_Z)
    amp = 1.0/(D_l*np.pi/180/60)
    rmin = R_MIN_MPC*amp; rmax = R_MAX_MPC*amp
    print(f"  D_l={D_l:.1f} Mpc, R=[{rmin:.1f}'-{rmax:.1f}']")

    # 元の中心
    print(f"\n--- original ---")
    d0 = diagnose("original", CL1_RA, CL1_DEC, src, rmin, rmax)
    if d0["valid"]:
        print(f"  <gt>={d0['gt_mean']:.5f}±{d0['gt_err']:.5f}, S/N={d0['sn']:.1f}")
        print(f"  <gx>={d0['gx_mean']:.5f}, chi2(gx)/dof={d0['chi2_gx_dof']:.2f}")

    # 候補中心
    cos_dec = np.cos(np.radians(CL1_DEC))
    cands = [
        ("original", CL1_RA, CL1_DEC),
        ("+30\"E", CL1_RA+30/3600/cos_dec, CL1_DEC),
        ("-30\"E", CL1_RA-30/3600/cos_dec, CL1_DEC),
        ("+30\"N", CL1_RA, CL1_DEC+30/3600),
        ("-30\"N", CL1_RA, CL1_DEC-30/3600),
        ("+60\"NE", CL1_RA+42/3600/cos_dec, CL1_DEC+42/3600),
        ("-60\"SW", CL1_RA-42/3600/cos_dec, CL1_DEC-42/3600),
    ]
    print(f"\n--- candidates ---")
    print(f"  {'label':<16} {'<gt>':>8} {'<gx>':>8} {'S/N':>6} {'chi2/dof':>9} {'|gx|':>8}")
    print(f"  {'-'*58}")
    diags = []
    for lab, rc, dc in cands:
        d = diagnose(lab, rc, dc, src, rmin, rmax)
        diags.append(d)
        if d["valid"]:
            print(f"  {lab:<16} {d['gt_mean']:>8.5f} {d['gx_mean']:>8.5f} "
                  f"{d['sn']:>6.1f} {d['chi2_gx_dof']:>9.2f} {d['abs_gx']:>8.5f}")

    # グリッドサーチ
    print(f"\n--- grid search ---")
    bra_g, bdec_g, bm_g, pts = grid_search(src, rmin, rmax)

    # 精密化
    print(f"\n--- refine ---")
    bra, bdec, bm = refine(src, bra_g, bdec_g, rmin, rmax)
    dra = (bra-CL1_RA)*cos_dec*3600
    ddc = (bdec-CL1_DEC)*3600
    shift = np.sqrt(dra**2+ddc**2)
    print(f"  optimal: RA={bra:.5f} Dec={bdec:.5f}")
    print(f"  shift: {dra:+.1f}\"E, {ddc:+.1f}\"N (|{shift:.1f}\")")

    dopt = diagnose("optimal", bra, bdec, src, rmin, rmax)
    diags.append(dopt)

    # 比較
    print(f"\n{'='*70}")
    print("comparison")
    print(f"{'='*70}")
    if d0["valid"] and dopt["valid"]:
        print(f"\n  {'':25} {'original':>12} {'optimal':>12}")
        print(f"  {'-'*50}")
        for k, fmt in [("gt_mean",".5f"),("gx_mean",".5f"),("abs_gx",".5f"),
                        ("sn",".1f"),("chi2_gx_dof",".2f"),("gx_rms",".5f")]:
            print(f"  {k:<25} {d0[k]:>12{fmt}} {dopt[k]:>12{fmt}}")

        gt_pos = dopt["gt_mean"] > 0
        gx_imp = dopt["abs_gx"] < d0["abs_gx"]
        sn_ok = dopt["sn"] > 3
        gx_ok = dopt["abs_gx"] < 0.005

        print(f"\n  gt > 0:       {'PASS' if gt_pos else 'FAIL'}")
        print(f"  |gx| improved: {'PASS' if gx_imp else 'FAIL'}")
        print(f"  S/N > 3:       {'PASS' if sn_ok else 'FAIL'} ({dopt['sn']:.1f})")
        print(f"  |gx| < 0.005:  {'PASS' if gx_ok else 'FAIL'} ({dopt['abs_gx']:.5f})")

        if shift > 60:
            print(f"\n  [WARNING] shift={shift:.0f}\">60\"")
            print(f"  原因候補: (a)元座標が粗い丸め (b)非弛緩系 (c)ソース空間範囲の非対称")
            print(f"  推奨: 分光メンバー22銀河の光度加重重心を使用")
        if not gt_pos:
            print(f"\n  [CRITICAL] gt<0 持続 → e1/e2符号規約確認、既知クラスターで検証")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"cl1 center v2: shift=({dra:+.1f}\", {ddc:+.1f}\")",
                     fontsize=13, fontweight='bold')

        # (a) chi2 map
        ax = axes[0,0]
        ms_arr = np.array([p['m'] for p in pts])
        da = np.array([p['dra'] for p in pts])
        dd_arr = np.array([p['ddec'] for p in pts])
        ok = ms_arr < 500
        if np.sum(ok)>10:
            vmax = np.percentile(ms_arr[ok], 90)
            sc = ax.scatter(da[ok], dd_arr[ok], c=ms_arr[ok], s=10, cmap='viridis_r',
                           vmin=np.min(ms_arr[ok]), vmax=vmax)
            plt.colorbar(sc, ax=ax, label='chi2(gx)')
        if np.sum(~ok)>0:
            ax.scatter(da[~ok], dd_arr[~ok], c='gray', s=5, alpha=0.3, label='gt<0')
        ax.plot(0, 0, 'rx', ms=12, mew=2, label='original')
        ax.plot(dra, ddc, 'w*', ms=15, mew=1, label='optimal')
        ax.set_xlabel('dRA ["]'); ax.set_ylabel('dDec ["]')
        ax.set_title('chi2(gx) map'); ax.legend(fontsize=8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        # (b) gamma_t
        ax = axes[0,1]
        for d, col, mk, lab in [(d0,'coral','o','original'),(dopt,'steelblue','s','optimal')]:
            if not d["valid"]: continue
            rm = np.array(d["rm"]); gtp = np.array(d["gt_p"]); gep = np.array(d["gt_e_p"])
            v = np.isfinite(gtp)
            off = 1.03 if d is dopt else 1.0
            ax.errorbar(rm[v]*off, gtp[v], gep[v], fmt=mk, color=col, ms=5, capsize=3, label=lab)
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_t$')
        ax.set_xscale('log'); ax.set_title('Tangential shear')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (c) gamma_x
        ax = axes[1,0]
        for d, col, mk, lab in [(d0,'coral','o','original'),(dopt,'steelblue','s','optimal')]:
            if not d["valid"]: continue
            rm = np.array(d["rm"]); gxp = np.array(d["gx_p"]); gep = np.array(d["gt_e_p"])
            v = np.isfinite(gxp)
            off = 1.03 if d is dopt else 1.0
            ax.errorbar(rm[v]*off, gxp[v], gep[v], fmt=mk, color=col, ms=5, capsize=3, label=lab)
        ax.axhline(0, color='green', ls='--', lw=2)
        ax.axhspan(-0.005, 0.005, alpha=0.1, color='green', label='|gx|<0.005')
        ax.set_xlabel('R [arcmin]'); ax.set_ylabel(r'$\gamma_\times$')
        ax.set_xscale('log'); ax.set_title('Cross shear (null)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (d) S/N bar
        ax = axes[0,2]
        vd = [d for d in diags if d["valid"]]
        labs = [d["label"][:15] for d in vd]
        sns = [d["sn"] for d in vd]
        bc = ['green' if s>3 else 'orange' if s>0 else 'red' for s in sns]
        ax.barh(range(len(labs)), sns, color=bc, edgecolor='white', height=0.6)
        ax.axvline(3, color='gray', ls='--', label='S/N=3')
        ax.axvline(0, color='red', lw=0.5)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel('S/N'); ax.set_title('S/N by candidate')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (e) |gx| bar
        ax = axes[1,1]
        gxs = [d["abs_gx"] for d in vd]
        bc2 = ['green' if g<0.005 else 'orange' if g<0.02 else 'red' for g in gxs]
        ax.barh(range(len(labs)), gxs, color=bc2, edgecolor='white', height=0.6)
        ax.axvline(0.005, color='green', ls='--', lw=2, label='0.005')
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel('|<gx>|'); ax.set_title('|gamma_x| by candidate')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (f) summary
        ax = axes[1,2]; ax.axis('off')
        txt = f"cl1 (z={CL1_Z}), N={len(src['ra'])}\n\n"
        if d0["valid"]:
            txt += f"Original:\n  gt={d0['gt_mean']:.5f}, gx={d0['gx_mean']:.5f}\n  S/N={d0['sn']:.1f}\n\n"
        if dopt["valid"]:
            txt += f"Optimal (shift={shift:.0f}\"):\n  gt={dopt['gt_mean']:.5f}, gx={dopt['gx_mean']:.5f}\n  S/N={dopt['sn']:.1f}\n\n"
            txt += f"|gx|<0.005: {'PASS' if dopt['abs_gx']<0.005 else 'FAIL'}\n"
            txt += f"gt>0: {'PASS' if dopt['gt_mean']>0 else 'FAIL'}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
               fontfamily='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        fp = OUTDIR / "cl1_center_v2.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except ImportError: pass

    # 保存
    result = {
        "original": {k:v for k,v in d0.items() if k not in ("rm","gt_p","gx_p","gt_e_p")},
        "optimal": {k:v for k,v in dopt.items() if k not in ("rm","gt_p","gx_p","gt_e_p")},
        "shift_arcsec": float(shift),
        "candidates": [{k:v for k,v in d.items() if k not in ("rm","gt_p","gx_p","gt_e_p")}
                       for d in diags if d["valid"]],
    }
    with open(OUTDIR / "cl1_center_v2_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"results: {OUTDIR / 'cl1_center_v2_results.json'}")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
