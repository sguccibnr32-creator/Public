#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_jackknife_extended.py — SPARC拡張ジャックナイフ検証
=========================================================
uv run --with scipy --with matplotlib --with numpy python sparc_jackknife_extended.py
"""

import numpy as np
import csv, json, sys, io, os
from pathlib import Path
from scipy.stats import linregress, t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SPARC_DIR = Path("Rotmod_LTG")
OUTDIR = Path("jackknife_output"); OUTDIR.mkdir(exist_ok=True)
G_SI = 6.674e-11; Msun = 1.989e30; pc = 3.086e16; kpc = 1e3*pc; a0 = 1.2e-10
N_JACK = 1000; RNG_SEED = 42

# ================================================================
# SPARCデータ: 元パイプラインと同一手法で読み込み
# ================================================================
def load_sparc():
    """gc_geometric_mean_test.py と同一手法でSPARC g_c/G*Sigma0を計算"""
    # CSVから ud, vflat を取得
    csv_path = Path("phase1/sparc_results.csv")
    gc_path = Path("TA3_gc_independent.csv")
    if not csv_path.exists() or not gc_path.exists():
        print(f"  Required: {csv_path} and {gc_path}")
        return None

    gal_info = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy','').strip()
            ud = float(row.get('ud','nan')); vf = float(row.get('vflat','nan'))
            if name and np.isfinite(ud) and np.isfinite(vf) and ud > 0 and vf > 0:
                gal_info[name] = {'ud': ud, 'vflat': vf}

    gc_dict = {}
    with open(gc_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy','').strip()
            gc = float(row.get('gc_over_a0','nan'))
            if n and np.isfinite(gc) and gc > 0:
                gc_dict[n] = gc

    results = []
    for name, info in gal_info.items():
        if name not in gc_dict: continue
        fp = SPARC_DIR / f"{name}_rotmod.dat"
        if not fp.exists(): continue
        try: data = np.loadtxt(fp, comments='#')
        except: continue
        if data.ndim != 2 or data.shape[0] < 5: continue

        rad = data[:,0]; v_disk = data[:,4]; v_gas = data[:,3]
        v_bul = data[:,5] if data.shape[1] > 5 else np.zeros_like(rad)
        ud = info['ud']

        mask = rad > 0.01; r = rad[mask]; vd = v_disk[mask]; vg = v_gas[mask]; vb = v_bul[mask]
        if len(r) < 5: continue

        vds = np.sqrt(ud)*np.abs(vd)
        V_pk = np.max(vds); i_pk = np.argmax(vds); r_pk = r[i_pk]
        if r_pk >= r.max()*0.9 or r_pk < 0.01: continue
        h_R = r_pk / 2.15

        gc_a0 = gc_dict[name]
        gc_si = gc_a0 * a0
        vflat_si = info['vflat'] * 1e3
        hR_si = h_R * kpc
        GS0 = vflat_si**2 / hR_si

        results.append({
            'name': name, 'gc': gc_si, 'gc_a0': gc_a0,
            'GS0': GS0, 'h_R': h_R, 'vflat': info['vflat'], 'ud': ud,
        })

    print(f"  SPARC galaxies: {len(results)}")
    return results

# ================================================================
# α回帰
# ================================================================
def alpha_fit(gc, GS0):
    lgc = np.log10(gc); lgs = np.log10(GS0)
    ok = np.isfinite(lgc) & np.isfinite(lgs)
    if np.sum(ok) < 5: return None
    x = lgs[ok]; y = lgc[ok]
    slope, intercept, r, p, se = linregress(x, y)
    t_stat = (slope - 0.5) / se
    p05 = 2 * tdist.sf(abs(t_stat), df=len(x)-2)
    return {"alpha": slope, "se": se, "p05": p05, "r": r, "N": len(x), "intercept": intercept}

# ================================================================
# メイン
# ================================================================
def main():
    print("="*60)
    print("SPARC拡張ジャックナイフ検証")
    print(f"N_iterations = {N_JACK}")
    print("="*60)

    gals = load_sparc()
    if not gals: return

    gc_all = np.array([g['gc'] for g in gals])
    GS_all = np.array([g['GS0'] for g in gals])
    N = len(gals)

    # 全体フィット
    print("\n--- 全体フィット ---")
    full = alpha_fit(gc_all, GS_all)
    if full:
        print(f"  α = {full['alpha']:.4f} ± {full['se']:.4f}")
        print(f"  p(α=0.5) = {full['p05']:.4f}")
        print(f"  N = {full['N']}, r = {full['r']:.3f}")

    # 半分割ジャックナイフ
    print(f"\n--- 半分割ジャックナイフ ×{N_JACK} ---")
    rng = np.random.RandomState(RNG_SEED)
    a_tr, a_te, p_tr, p_te, da = [], [], [], [], []

    for _ in range(N_JACK):
        idx = rng.permutation(N); h = N//2
        ft = alpha_fit(gc_all[idx[:h]], GS_all[idx[:h]])
        fe = alpha_fit(gc_all[idx[h:]], GS_all[idx[h:]])
        if ft and fe:
            a_tr.append(ft["alpha"]); a_te.append(fe["alpha"])
            p_tr.append(ft["p05"]); p_te.append(fe["p05"])
            da.append(ft["alpha"]-fe["alpha"])

    a_tr = np.array(a_tr); a_te = np.array(a_te)
    p_tr = np.array(p_tr); p_te = np.array(p_te); da = np.array(da)
    nv = len(a_tr)

    print(f"  有効: {nv}/{N_JACK}")
    print(f"  α(train) = {np.mean(a_tr):.4f} ± {np.std(a_tr):.4f}")
    print(f"  α(test)  = {np.mean(a_te):.4f} ± {np.std(a_te):.4f}")
    print(f"  Δα = {np.mean(da):.5f} ± {np.std(da):.5f}")
    print(f"  |Δα|<0.05: {np.mean(np.abs(da)<0.05)*100:.1f}%")
    print(f"  α=0.5 棄却率(train): {np.mean(p_tr<0.05)*100:.1f}%")
    print(f"  α=0.5 棄却率(test):  {np.mean(p_te<0.05)*100:.1f}%")

    # Leave-N-out
    print(f"\n--- Leave-N-out ---")
    for Nl in [10, 20, 50]:
        if Nl >= N: continue
        als = []
        for _ in range(500):
            keep = rng.choice(N, N-Nl, replace=False)
            f = alpha_fit(gc_all[keep], GS_all[keep])
            if f: als.append(f["alpha"])
        als = np.array(als)
        print(f"  Leave-{Nl}: α = {np.mean(als):.4f} ± {np.std(als):.4f}, "
              f"[{np.min(als):.3f}, {np.max(als):.3f}]")

    # Bootstrap
    print(f"\n--- Bootstrap ×{N_JACK} ---")
    a_boot = []
    for _ in range(N_JACK):
        idx = rng.choice(N, N, replace=True)
        f = alpha_fit(gc_all[idx], GS_all[idx])
        if f: a_boot.append(f["alpha"])
    a_boot = np.array(a_boot)
    ci = np.percentile(a_boot, [2.5, 97.5])
    print(f"  α = {np.mean(a_boot):.4f} ± {np.std(a_boot):.4f}")
    print(f"  95%CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  0.5 in CI: {'YES ✓' if ci[0]<=0.5<=ci[1] else 'NO ✗'}")

    # プロット
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"SPARC Extended Jackknife (N={N}, {N_JACK} iter)", fontsize=14, fontweight='bold')

    ax = axes[0,0]
    ax.scatter(a_tr, a_te, alpha=0.15, s=5, c='steelblue')
    ax.axhline(0.5, color='red', ls='--'); ax.axvline(0.5, color='red', ls='--')
    ax.plot([0.2,0.9],[0.2,0.9],'k--',alpha=0.3)
    ax.set_xlabel(r'$\alpha$ (train)'); ax.set_ylabel(r'$\alpha$ (test)')
    ax.set_title('Half-split consistency'); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.hist(da, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_xlabel(r'$\Delta\alpha$'); ax.set_title(f'mean={np.mean(da):.4f}, std={np.std(da):.4f}')
    ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    ax.hist(a_tr, bins=50, alpha=0.6, color='navy', label='train')
    ax.hist(a_te, bins=50, alpha=0.6, color='coral', label='test')
    ax.axvline(0.5, color='green', ls='--', lw=2, label=r'$\alpha=0.5$')
    ax.set_xlabel(r'$\alpha$'); ax.set_title('Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.hist(a_boot, bins=50, color='green', edgecolor='white', alpha=0.7)
    ax.axvline(0.5, color='red', ls='--', lw=2)
    ax.axvline(ci[0], color='red', ls=':', alpha=0.5)
    ax.axvline(ci[1], color='red', ls=':', alpha=0.5)
    ax.set_xlabel(r'$\alpha$ (bootstrap)'); ax.set_title(f'95%CI: [{ci[0]:.3f},{ci[1]:.3f}]')
    ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.hist(p_tr, bins=50, alpha=0.6, color='navy', label='train p(0.5)')
    ax.hist(p_te, bins=50, alpha=0.6, color='coral', label='test p(0.5)')
    ax.axvline(0.05, color='red', ls='--', lw=2, label='p=0.05')
    ax.set_xlabel('p(α=0.5)'); ax.set_title(f'Rejection rate: {np.mean(p_tr<0.05)*100:.0f}%/{np.mean(p_te<0.05)*100:.0f}%')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (f) log(gc) vs log(GS0) scatter
    ax = axes[1,2]
    lgc = np.log10(gc_all); lgs = np.log10(GS_all)
    ok = np.isfinite(lgc)&np.isfinite(lgs)
    ax.scatter(lgs[ok], lgc[ok], c='steelblue', s=10, alpha=0.5)
    xr = np.linspace(lgs[ok].min()-0.3, lgs[ok].max()+0.3, 100)
    ax.plot(xr, full['intercept']+full['alpha']*xr, 'r-', lw=2,
            label=rf"$\alpha={full['alpha']:.3f}\pm{full['se']:.3f}$")
    eta = np.mean(lgc[ok]-0.5*(np.log10(a0)+lgs[ok]))
    ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+eta, 'g--', lw=1.5, label=r'$\alpha=0.5$')
    ax.axhline(np.log10(a0), color='gray', ls=':')
    ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
    ax.set_title(f'SPARC N={full["N"]}'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = OUTDIR / "jackknife_extended.png"
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    print(f"\nplot: {fp}")

    # JSON
    summary = {
        "full": full,
        "jackknife": {
            "N_iter": int(nv), "alpha_train": [float(np.mean(a_tr)),float(np.std(a_tr))],
            "alpha_test": [float(np.mean(a_te)),float(np.std(a_te))],
            "delta_alpha": [float(np.mean(da)),float(np.std(da))],
            "reject_rate_train": float(np.mean(p_tr<0.05)),
            "reject_rate_test": float(np.mean(p_te<0.05)),
        },
        "bootstrap": {
            "alpha": [float(np.mean(a_boot)),float(np.std(a_boot))],
            "CI95": [float(ci[0]),float(ci[1])],
        },
    }
    with open(OUTDIR/"jackknife_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"results: {OUTDIR/'jackknife_summary.json'}")
    print(f"\n{'='*60}\ndone\n{'='*60}")

if __name__ == "__main__":
    main()
