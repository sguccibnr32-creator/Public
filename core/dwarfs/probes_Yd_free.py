#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probes_Yd_free.py — Υ_d フリーフィットで α=0.5 再現を検証
============================================================
uv run --with scipy --with matplotlib --with numpy --with pandas python probes_Yd_free.py

前提: probes_results/ に Zenodo データ + S4G TSV が存在
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import i0, i1, k0, k1
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

G_SI = 6.674e-11; Msun = 1.989e30; pc = 3.086e16; kpc_m = 1e3*pc
a0 = 1.2e-10; M_sun_36 = 3.24
OUTDIR = Path("probes_vbar_output"); OUTDIR.mkdir(exist_ok=True)

SPARC_NAMES = {"NGC3198","NGC2841","NGC2403","NGC3031","NGC2903","NGC5055",
    "NGC7331","NGC6946","NGC3521","NGC4736","NGC5585","NGC4258","NGC925",
    "NGC2366","NGC4163","NGC1569","NGC3738","DDO154","DDO168","DDO50",
    "DDO126","DDO133","DDO87","DDO47","DDO52","DDO46","DDO43","DDO70",
    "DDO53","DDO210","DDO216","DDO101","IC2574","IC1613","UGC2885","WLM"}

def norm_name(n):
    import re
    s = str(n).strip().upper()
    s = re.sub(r'[\s_\-]+', '', s)
    for p in ["NGC","DDO","UGC","IC"]:
        if s.startswith(p): s = p+s[len(p):].lstrip("0")
    return s

# ================================================================
# データ読み込み（ローカルファイル優先）
# ================================================================
def load_data():
    import pandas as pd
    print("=== データ読み込み ===")

    # PROBES RC (Zenodoからダウンロード済み)
    rc_dir = Path("probes_results/profiles/profiles")
    if not rc_dir.exists():
        print(f"  {rc_dir} not found"); return None, None, None
    rc_files = sorted(rc_dir.glob("*_rc.prof"))
    print(f"  PROBES RC files: {len(rc_files)}")

    rc_data = {}
    for fp in rc_files:
        name = fp.stem.replace("_rc","")
        lines = fp.read_text(encoding='utf-8', errors='replace').strip().split('\n')
        r, v, e = [], [], []
        for line in lines:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    ri = float(parts[0]); vi = float(parts[1]); ei = float(parts[2])
                    if ri > 0:  # 正側のみ
                        r.append(ri); v.append(abs(vi)); e.append(abs(ei))
                except: pass
        if len(r) >= 5:
            rc_data[name] = {"R": np.array(r), "V": np.array(v), "eV": np.array(e)}

    # 距離 (PROBES main table)
    dist_map = {}
    mfp = Path("probes_results/probes_main.csv")
    if mfp.exists():
        main = pd.read_csv(mfp, comment='#')
        for _, row in main.iterrows():
            dist_map[row['name']] = row['distance']
    print(f"  距離: {len(dist_map)}")

    # S4G (前回パイプラインで取得済み)
    s4g = {}
    sfp = OUTDIR / "s4g_salo2015.tsv"
    if sfp.exists():
        lines = sfp.read_text(encoding='utf-8', errors='replace').strip().split('\n')
        header = None; ds = 0
        for i, line in enumerate(lines):
            if line.startswith('#') or line.startswith('-'): continue
            parts = [p.strip() for p in line.split('\t')]
            if len(parts) >= 3 and any(not p.replace('.','').replace('-','').replace('+','').isdigit() for p in parts[:3] if p):
                header = parts; ds = i+1
                while ds < len(lines) and lines[ds].startswith('-'): ds += 1
                break
        if header:
            # Re, Tmag, n のインデックスを探す
            re_idx = next((j for j, h in enumerate(header) if h.strip() == 'Re'), None)
            tm_idx = next((j for j, h in enumerate(header) if h.strip() == 'Tmag'), None)
            n_idx = next((j for j, h in enumerate(header) if h.strip() == 'n'), None)
            nm_idx = next((j for j, h in enumerate(header) if 'Name' in h), None)
            if re_idx is not None and nm_idx is not None:
                for line in lines[ds:]:
                    if line.startswith('#') or line.startswith('-') or not line.strip(): continue
                    parts = [p.strip() for p in line.split('\t')]
                    if len(parts) > max(re_idx, nm_idx):
                        nm = parts[nm_idx].strip()
                        try:
                            Re = float(parts[re_idx])
                            Tm = float(parts[tm_idx]) if tm_idx and parts[tm_idx].strip() else 99
                            ns = float(parts[n_idx]) if n_idx and parts[n_idx].strip() else 1.0
                            if Re > 0 and Tm < 90:
                                s4g[nm] = {"Re": Re, "Tmag": Tm, "n": ns}
                        except: pass
    print(f"  S4G: {len(s4g)}")
    print(f"  PROBES RC: {len(rc_data)}")

    return rc_data, dist_map, s4g

# ================================================================
# V_bar構築（Υ_d パラメータ付き）
# ================================================================
def compute_vbar(R_arcsec, dist_Mpc, Re_arcsec, Tmag, n_sersic, Yd, f_gas=0.20):
    """Freeman disk V_bar with explicit Υ_d"""
    # Re → h
    bn = max(2*n_sersic - 1./3 + 4./(405*n_sersic), 0.5) if n_sersic > 0.3 else 1.678
    h_as = Re_arcsec / bn
    h_kpc = h_as * dist_Mpc * 1e3 * np.pi / (180*3600)
    R_kpc = R_arcsec * dist_Mpc * 1e3 * np.pi / (180*3600)

    if h_kpc < 0.01 or h_kpc > 100: return None

    # mu0 from Tmag
    area = 2*np.pi*Re_arcsec**2
    if area <= 0: return None
    mu0 = Tmag + 2.5*np.log10(area) + 0.7

    I0 = 10**(-0.4*(mu0 - M_sun_36 - 21.572))  # L_sun/pc^2
    S0_SI = I0 * Msun / (pc**2)  # kg/m^2 (光度密度、Υ不含)

    y = np.clip(R_kpc/(2*h_kpc), 1e-6, 50)
    bt = i0(y)*k0(y) - i1(y)*k1(y)
    h_m = h_kpc * kpc_m

    V2_disk_unit = 4*np.pi*G_SI*S0_SI*h_m * y**2 * bt  # ∝ I0 (Υ不含)
    V2_disk_unit = np.maximum(V2_disk_unit, 0)

    # V_bar² = Υ_d × V2_disk_unit + f_gas × Υ_d × V2_disk_unit
    V2_bar = Yd * (1 + f_gas) * V2_disk_unit
    V_bar = np.sqrt(V2_bar) / 1e3  # km/s

    return {"V_bar": V_bar, "R_kpc": R_kpc, "h_kpc": h_kpc,
            "Sigma0": Yd*I0, "I0": I0}

# ================================================================
# g_c フィット（SPARC同一手法）
# ================================================================
def fit_gc(R_kpc, V_obs, eV, V_bar):
    rm = R_kpc*kpc_m; vm = V_obs*1e3; em = np.maximum(eV,1)*1e3; vbm = V_bar*1e3
    ok = (rm > 0)&(vbm > 0)&(vm > 0)
    if np.sum(ok) < 3: return None
    gN = vbm[ok]**2/rm[ok]; go = vm[ok]**2/rm[ok]
    eg = np.maximum(2*vm[ok]*em[ok]/rm[ok], 1e-12)
    def c2(lgc):
        gc = 10**lgc; gp = (gN+np.sqrt(gN**2+4*gc*gN))/2
        return np.sum(((go-gp)/eg)**2)
    try:
        res = minimize_scalar(c2, bounds=(-12,-8), method='bounded')
        gc = 10**res.x
        return {"g_c": float(gc), "log_gc": float(np.log10(gc)),
                "gc_a0": float(gc/a0), "chi2_dof": float(res.fun/max(np.sum(ok)-1,1))}
    except: return None

# ================================================================
# Υ_d最適化
# ================================================================
def optimize_Yd(R_as, V_obs, eV, dist, Re, Tmag, ns, f_gas=0.2):
    def chi2(Yd):
        r = compute_vbar(R_as, dist, Re, Tmag, ns, Yd, f_gas)
        if r is None or len(r["V_bar"]) != len(V_obs): return 1e10
        return np.sum(((V_obs - r["V_bar"])/np.maximum(eV, 1.0))**2)
    try:
        res = minimize_scalar(chi2, bounds=(0.05, 2.0), method='bounded')
        if res.fun < 1e9: return res.x, res.fun/max(len(V_obs)-1,1)
    except: pass
    return None, None

# ================================================================
# α検定
# ================================================================
def alpha_test(gals, label=""):
    v = [g for g in gals if g.get("chi2_dof",0) < 10 and g["log_gc"] > -15]
    N = len(v)
    if N < 10: return None
    lgc = np.array([g["log_gc"] for g in v])
    lgs = np.array([g["log_GS0"] for g in v])
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    r = lgc-A@c; s2 = np.sum(r**2)/(N-2)
    cov = s2*np.linalg.inv(A.T@A); se = np.sqrt(cov[1,1])
    tc = t_dist.ppf(0.975, N-2)
    p05 = 2*t_dist.sf(abs((c[1]-0.5)/se), N-2)
    rho, _ = spearmanr(lgs, lgc)
    zs = abs(c[1]-0.545)/np.sqrt(se**2+0.041**2)
    print(f"  {label}: N={N}, α={c[1]:.3f}±{se:.3f}, p(0.5)={p05:.4f}, "
          f"SPARC z={zs:.2f} {'✓' if zs<2 else '✗'}")
    return {"N":N,"alpha":float(c[1]),"se":float(se),"p05":float(p05),
            "zs":float(zs),"rsd":float(np.std(r)),"lgc":lgc,"lgs":lgs,
            "interc":float(c[0]),"eta":float(np.mean(lgc-0.5*(np.log10(a0)+lgs)))}

# ================================================================
# メイン
# ================================================================
def main():
    print("="*60)
    print("PROBES Υ_d フリーフィット検証")
    print("="*60)

    rc_data, dist_map, s4g = load_data()
    if not rc_data: return

    # クロスマッチ
    s4g_n = {norm_name(k): k for k in s4g}
    sparc_n = {norm_name(s) for s in SPARC_NAMES}

    all_results = {yd: [] for yd in [0.3, 0.5, 0.7, "free"]}
    Yd_opts = []
    n_match = 0

    for rc_name, rc in rc_data.items():
        nn = norm_name(rc_name)
        if nn not in s4g_n: continue
        s_name = s4g_n[nn]
        dist = dist_map.get(rc_name)
        if not dist or dist <= 0: continue
        n_match += 1
        is_sp = nn in sparc_n

        R = rc["R"]; V = rc["V"]; eV = rc["eV"]
        Re = s4g[s_name]["Re"]; Tm = s4g[s_name]["Tmag"]; ns = s4g[s_name]["n"]

        # 固定Υ_d
        for Yd in [0.3, 0.5, 0.7]:
            res = compute_vbar(R, dist, Re, Tm, ns, Yd)
            if res is None or len(res["V_bar"]) != len(V): continue
            fit = fit_gc(res["R_kpc"], V, eV, res["V_bar"])
            if fit and fit["g_c"] > 0:
                GS0 = (np.median(V[-max(3,len(V)//4):])*1e3)**2/(res["h_kpc"]*kpc_m)
                all_results[Yd].append({**fit, "name": rc_name, "is_sparc": is_sp,
                    "log_GS0": float(np.log10(GS0)), "Yd": Yd})

        # フリーΥ_d
        Yd_best, c2d = optimize_Yd(R, V, eV, dist, Re, Tm, ns)
        if Yd_best and c2d and c2d < 20:
            res = compute_vbar(R, dist, Re, Tm, ns, Yd_best)
            if res and len(res["V_bar"]) == len(V):
                fit = fit_gc(res["R_kpc"], V, eV, res["V_bar"])
                if fit and fit["g_c"] > 0:
                    GS0 = (np.median(V[-max(3,len(V)//4):])*1e3)**2/(res["h_kpc"]*kpc_m)
                    all_results["free"].append({**fit, "name": rc_name, "is_sparc": is_sp,
                        "log_GS0": float(np.log10(GS0)), "Yd": float(Yd_best)})
                    Yd_opts.append(Yd_best)

    print(f"\nマッチ: {n_match}")
    for k, v in all_results.items():
        print(f"  Yd={k}: {len(v)} galaxies ({sum(1 for g in v if not g['is_sparc'])} independent)")

    # α検定
    print(f"\n{'='*60}")
    print("α検定")
    print(f"{'='*60}")
    test_results = {}
    for Yd_key, label in [(0.3,"Yd=0.3"),(0.5,"Yd=0.5"),(0.7,"Yd=0.7"),("free","Yd free")]:
        gals = all_results[Yd_key]
        if not gals: continue
        # 全体
        r = alpha_test(gals, f"{label} (all)")
        if r: test_results[f"{label}_all"] = r
        # SPARC除外
        indep = [g for g in gals if not g["is_sparc"]]
        if len(indep) >= 10:
            r2 = alpha_test(indep, f"{label} (indep)")
            if r2: test_results[f"{label}_indep"] = r2

    # Υ_d分布
    if Yd_opts:
        ya = np.array(Yd_opts)
        print(f"\n{'='*60}")
        print(f"Υ_d 最適値分布 (N={len(ya)})")
        print(f"  median={np.median(ya):.3f}, mean={np.mean(ya):.3f}, std={np.std(ya):.3f}")
        print(f"  IQR=[{np.percentile(ya,25):.3f}, {np.percentile(ya,75):.3f}]")
        print(f"  SPARC範囲(0.3-0.9)内: {np.sum((ya>=0.3)&(ya<=0.9))}/{len(ya)}")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("PROBES Yd optimization", fontsize=14, fontweight='bold')

        # (a) Yd=0.5 vs free
        for ax_idx, (key, lab, col) in enumerate([(0.5,"Yd=0.5",'coral'),("free","Yd free",'steelblue')]):
            ax = axes[0, ax_idx]
            k = f"{lab}_indep" if f"{lab}_indep" in test_results else f"{lab}_all"
            if k in test_results:
                r = test_results[k]
                ax.scatter(r['lgs'], r['lgc'], c=col, s=10, alpha=0.4)
                xr = np.linspace(r['lgs'].min()-0.3, r['lgs'].max()+0.3, 100)
                ax.plot(xr, r['interc']+r['alpha']*xr, 'r-', lw=2,
                        label=rf'$\alpha={r["alpha"]:.3f}\pm{r["se"]:.3f}$')
                ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+r['eta'], 'g--', lw=1.5, label=r'$\alpha=0.5$')
                ax.axhline(np.log10(a0), color='gray', ls=':')
                ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
                ax.set_title(f'{lab}: p(0.5)={r["p05"]:.3f}')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (c) Yd distribution
        ax = axes[1,0]
        if Yd_opts:
            ax.hist(Yd_opts, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(0.5, color='red', ls='--', lw=2, label='SPARC default')
            ax.axvspan(0.3, 0.9, alpha=0.1, color='green', label='SPARC range')
            ax.set_xlabel(r'$\Upsilon_d$ [M$_\odot$/L$_\odot$]')
            ax.set_title(f'Optimal Yd (N={len(Yd_opts)})')
            ax.legend(); ax.grid(True, alpha=0.3)

        # (d) alpha comparison
        ax = axes[1,1]
        labels = ['SPARC\n(N=175)']; alphas = [0.545]; errors = [0.041*1.96]
        colors = ['navy']
        for k, lab, col in [("Yd=0.3_indep","Yd=0.3",'orange'),
                             ("Yd=0.5_indep","Yd=0.5",'coral'),
                             ("Yd=0.7_indep","Yd=0.7",'brown'),
                             ("Yd free_indep","Yd free",'steelblue')]:
            if k in test_results:
                r = test_results[k]
                labels.append(f'{lab}\n(N={r["N"]})')
                alphas.append(r['alpha']); errors.append(r['se']*t_dist.ppf(0.975,r['N']-2))
                colors.append(col)
        for i,(a,e,c) in enumerate(zip(alphas,errors,colors)):
            ax.errorbar(a, i, xerr=e, fmt='o', ms=10, capsize=8, color=c, elinewidth=2)
        ax.axvline(0.5, color='green', ls='--', lw=2)
        ax.set_xlabel(r'$\alpha$'); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(-0.2, 2.0); ax.grid(True, alpha=0.3); ax.set_title(r'$\alpha$ 95% CI')

        plt.tight_layout()
        fp = OUTDIR/"probes_Yd_free.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except: pass

    # 保存
    out = {k: {kk:vv for kk,vv in v.items() if not isinstance(vv, np.ndarray)}
           for k, v in test_results.items()}
    if Yd_opts:
        out["Yd_stats"] = {"median":float(np.median(Yd_opts)),"mean":float(np.mean(Yd_opts)),
                           "std":float(np.std(Yd_opts))}
    with open(OUTDIR/"probes_Yd_free.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"results: {OUTDIR/'probes_Yd_free.json'}")
    print(f"\n{'='*60}\ndone\n{'='*60}")

if __name__ == "__main__":
    main()
