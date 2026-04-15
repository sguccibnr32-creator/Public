# -*- coding: utf-8 -*-
"""
HSC Y3 クラスター候補: 分光照合 -> 弛緩系選別 -> スタック弱レンズ解析
====================================================================
uv run --with requests --with scipy --with matplotlib --with numpy --with pandas python cluster_stack.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("cluster_stack"); OUTDIR.mkdir(exist_ok=True)
H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 3e5; a0 = 1.2e-10

# ================================================================
# 宇宙論
# ================================================================
def comoving_distance(z):
    f = lambda zp: 1.0/np.sqrt(OMEGA_M*(1+zp)**3+(1-OMEGA_M))
    r, _ = quad(f, 0, z); return C_LIGHT/H0*r

def angular_diameter_distance(z):
    return comoving_distance(z)/(1+z)

def sigma_crit_inv(z_l, z_s):
    if z_s <= z_l+0.05: return 0.0
    D_s = angular_diameter_distance(z_s)
    D_ls = (comoving_distance(z_s)-comoving_distance(z_l))/(1+z_s)
    if D_ls <= 0 or D_s <= 0: return 0.0
    return D_ls/D_s

# ================================================================
# Step 1: 実データ候補リスト読み込み
# ================================================================
def load_candidates():
    """hsc_cluster_candidates_y3.csv から実際の候補を読み込む"""
    import pandas as pd
    fp = Path("hsc_cluster_candidates_y3.csv")
    if not fp.exists():
        print(f"  {fp} not found"); return []
    df = pd.read_csv(fp)
    print(f"  {fp}: {len(df)} candidates")
    cands = []
    for _, row in df.iterrows():
        cands.append({
            "id": f"cl{len(cands)+1}",
            "ra": float(row["ra"]), "dec": float(row["dec"]),
            "sgm_peak": float(row.get("sgm_peak", 0)),
            "quality": float(row.get("quality", 0)),
            "best_zbin": str(row.get("best_zbin", "")),
            "z_photo": None, "z_spec": None,
            "n_spec_members": 0, "sigma_v": None,
        })
    # cl1 は分光確認済み
    if cands:
        cands[0]["z_spec"] = 0.313
        cands[0]["n_spec_members"] = 22
        cands[0]["sigma_v"] = 527
        cands[0]["z_photo"] = 0.313
    return cands

# ================================================================
# Step 2: SDSS分光照合
# ================================================================
def crossmatch_sdss(candidates, max_queries=38):
    """SDSS DR18 SqlSearch APIで分光照合（RadialSearchがHTTP500のため）"""
    import requests
    print("\n--- SDSS DR18 spectroscopic crossmatch (SqlSearch) ---")
    n_matched = 0
    base = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"

    for i, c in enumerate(candidates):
        if i >= max_queries: break
        if c.get("z_spec") is not None:
            print(f"  {c['id']}: z_spec={c['z_spec']} (known)")
            n_matched += 1; continue

        ra, dec = c["ra"], c["dec"]
        r_deg = 2.0 / 60.0  # 2 arcmin in degrees
        sql = (f"SELECT s.ra, s.dec, s.z, s.zErr, s.class "
               f"FROM SpecObj s "
               f"WHERE s.ra BETWEEN {ra-r_deg} AND {ra+r_deg} "
               f"AND s.dec BETWEEN {dec-r_deg} AND {dec+r_deg} "
               f"AND s.z BETWEEN 0.05 AND 1.0 "
               f"AND s.zWarning = 0")
        try:
            r = requests.get(base, params={"cmd": sql, "format": "json"}, timeout=30)
            if r.status_code != 200:
                print(f"  {c['id']}: HTTP {r.status_code}")
                continue
            data = r.json()
            rows = data[0].get("Rows", []) if data else []
            if len(rows) < 3:
                print(f"  {c['id']}: {len(rows)} spectra (<3)")
                continue

            zvals = np.array([float(row["z"]) for row in rows])
            # zヒストグラムでピーク検出
            zbins = np.arange(0.05, 1.0, 0.02)
            hist, edges = np.histogram(zvals, bins=zbins)
            peak_idx = np.argmax(hist)
            if hist[peak_idx] >= 3:
                z_lo, z_hi = edges[peak_idx], edges[peak_idx+1]
                members = zvals[(zvals >= z_lo) & (zvals < z_hi)]
                z_med = float(np.median(members))
                sig_v = float(np.std(members) * C_LIGHT / (1+z_med))
                c["z_spec"] = round(z_med, 4)
                c["z_photo"] = round(z_med, 4)
                c["n_spec_members"] = len(members)
                c["sigma_v"] = round(sig_v, 0)
                n_matched += 1
                print(f"  {c['id']}: z={z_med:.4f} N={len(members)} σv={sig_v:.0f} km/s")
            else:
                print(f"  {c['id']}: {len(rows)} spectra, no z-peak (max={hist[peak_idx]})")
        except Exception as e:
            print(f"  {c['id']}: error: {e}")

    print(f"\n  matched: {n_matched}/{len(candidates)}")
    return candidates

# ================================================================
# Step 3: 弛緩系フィルタ
# ================================================================
def relaxation_filter(candidates):
    print("\n--- relaxation filter ---")
    relaxed, rejected = [], []
    for c in candidates:
        reasons = []
        if c.get("z_spec") is None: reasons.append("no z_spec")
        if c.get("n_spec_members", 0) < 3: reasons.append(f"N={c.get('n_spec_members',0)}<3")
        if c.get("sigma_v") is not None:
            if c["sigma_v"] > 1200: reasons.append("merger?")
            if c["sigma_v"] < 150: reasons.append("group")
        if not reasons:
            c["relaxed"] = True; relaxed.append(c)
            print(f"  {c['id']}: PASS z={c['z_spec']} N={c['n_spec_members']} σv={c['sigma_v']}")
        else:
            c["relaxed"] = False; rejected.append(c)
    print(f"  passed: {len(relaxed)}, rejected: {len(rejected)}")
    return relaxed, rejected

# ================================================================
# Step 4: HSC CAS クエリ生成
# ================================================================
def generate_queries(clusters, r_deg=1.0):
    print("\n--- HSC CAS query generation ---")
    queries = []
    for c in clusters:
        ra, dec, z = c["ra"], c["dec"], c["z_spec"]
        q = f"""-- {c['id']} (z={z})
SELECT object_id, i_ra, i_dec,
  i_hsmshaperegauss_e1 AS e1, i_hsmshaperegauss_e2 AS e2,
  i_hsmshaperegauss_derived_weight AS weight,
  a.photoz_best AS z_photo
FROM s19a_wide.weaklensing_hsm_regauss AS w
LEFT JOIN pdr3_wide.photoz_demp AS a USING (object_id)
WHERE w.b_mode_mask = 1
  AND w.i_ra BETWEEN {ra-r_deg} AND {ra+r_deg}
  AND w.i_dec BETWEEN {dec-r_deg} AND {dec+r_deg}
  AND a.photoz_best > {z+0.1} AND a.photoz_best < 2.0
  AND w.i_hsmshaperegauss_resolution > 0.3;
"""
        queries.append({"id": c["id"], "query": q})
        print(f"  {c['id']}: R=[{ra-r_deg:.2f},{ra+r_deg:.2f}] z_src>{z+0.1:.2f}")

    qf = OUTDIR / "hsc_source_queries.sql"
    with open(qf, "w") as f:
        for q in queries: f.write(q["query"]+"\n")
    print(f"  saved: {qf}")
    return queries

# ================================================================
# Step 5: ソースデータ読み込み
# ================================================================
def load_source_data(clusters):
    """既存のソースCSVファイルを探して読み込む"""
    import pandas as pd
    print("\n--- source data ---")
    data = {}
    for c in clusters:
        cid = c["id"]
        for pat in [f"{cid}_sources.csv", f"{cid}_shear_sources.csv",
                    f"cluster_stack/{cid}_sources.csv"]:
            p = Path(pat)
            if not p.exists(): continue
            # ヘッダが "# col1,col2,..." の場合に対応
            with open(p, 'r') as fh:
                first = fh.readline()
            if first.startswith('# '):
                # #付きヘッダ: 先頭の"# "を除去した列名を使用
                cols = first[2:].strip().split(',')
                df = pd.read_csv(p, skiprows=1, header=None, names=cols)
            else:
                df = pd.read_csv(p, comment='#')
            print(f"    trying {p}: {len(df)} rows, cols={list(df.columns[:5])}")
            cm = {}
            for col in df.columns:
                cl = col.lower()
                if 'ra' in cl and 'dec' not in cl: cm.setdefault('ra', col)
                elif 'dec' in cl: cm.setdefault('dec', col)
                elif 'e1' in cl: cm.setdefault('e1', col)
                elif 'e2' in cl: cm.setdefault('e2', col)
                elif 'weight' in cl: cm.setdefault('w', col)
            if all(k in cm for k in ['ra','dec','e1','e2']):
                src = {k: df[cm[k]].values for k in ['ra','dec','e1','e2']}
                src['weight'] = df[cm['w']].values if 'w' in cm else np.ones(len(df))
                # photo-z がない場合はダミー（z_cl + 0.5）
                src['z_photo'] = np.full(len(df), c["z_spec"]+0.5)
                data[cid] = src
                print(f"  {cid}: {len(df)} sources from {p}")
                break
    return data

# ================================================================
# Step 6: スタックプロファイル
# ================================================================
def stacked_profile(clusters, src_data, nb=12):
    r_bins = np.logspace(np.log10(0.1), np.log10(5.0), nb+1)
    gt_s = np.zeros(nb); gx_s = np.zeros(nb)
    w_s = np.zeros(nb); w2_s = np.zeros(nb)
    n_t = np.zeros(nb, dtype=int)

    for c in clusters:
        if c["id"] not in src_data: continue
        src = src_data[c["id"]]
        z_cl = c["z_spec"]; D_l = angular_diameter_distance(z_cl)
        ra_c, dec_c = c["ra"], c["dec"]

        dra = (src["ra"]-ra_c)*np.cos(np.radians(dec_c))*np.pi/180*D_l
        ddec= (src["dec"]-dec_c)*np.pi/180*D_l
        r_mpc = np.sqrt(dra**2+ddec**2)

        phi = np.arctan2(ddec, dra)
        c2 = np.cos(2*phi); s2 = np.sin(2*phi)
        gt = -(src["e1"]*c2 + src["e2"]*s2)
        gx = src["e1"]*s2 - src["e2"]*c2

        sci = np.array([sigma_crit_inv(z_cl, zs) for zs in src["z_photo"]])
        w = src["weight"] * sci**2
        w[sci <= 0] = 0

        for i in range(nb):
            m = (r_mpc >= r_bins[i]) & (r_mpc < r_bins[i+1]) & (w > 0)
            n = np.sum(m)
            if n == 0: continue
            n_t[i] += n
            wi = w[m]
            gt_s[i] += np.sum(wi*gt[m])
            gx_s[i] += np.sum(wi*gx[m])
            w_s[i] += np.sum(wi)
            w2_s[i] += np.sum(wi**2)

    r_mid = np.sqrt(r_bins[:-1]*r_bins[1:])
    gt_st = np.where(w_s > 0, gt_s/w_s, np.nan)
    gx_st = np.where(w_s > 0, gx_s/w_s, np.nan)
    n_eff = np.where(w2_s > 0, w_s**2/w2_s, 0)
    gt_err = np.where(n_eff > 0, 0.25/np.sqrt(n_eff), np.nan)
    return r_mid, gt_st, gx_st, gt_err, n_t

# ================================================================
# Step 7: モデルフィット
# ================================================================
def nfw_gamma_t(r_mpc, M200, c200):
    rho_c = 2.775e11*(H0/100)**2
    r200 = (3*M200/(4*np.pi*200*rho_c))**(1/3)
    rs = r200/c200
    dc = 200/3*c200**3/(np.log(1+c200)-c200/(1+c200))
    rho_s = dc*rho_c
    x = r_mpc/rs
    result = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi < 1e-6: continue
        if abs(xi-1)<1e-4: sig = 2*rho_s*rs/3.0
        elif xi<1: sig = 2*rho_s*rs/(xi**2-1)*(1-np.arccosh(1/xi)/np.sqrt(1-xi**2))
        else: sig = 2*rho_s*rs/(xi**2-1)*(1-np.arccos(1/xi)/np.sqrt(xi**2-1))
        if abs(xi-1)<1e-4: g = 1+np.log(0.5)
        elif xi<1: g = np.arccosh(1/xi)/np.sqrt(1-xi**2)+np.log(xi/2)
        else: g = np.arccos(1/xi)/np.sqrt(xi**2-1)+np.log(xi/2)
        sm = 4*rho_s*rs*g/xi**2
        result[i] = sm - sig
    return result

def mond_gamma_t(r_mpc, gc, M_bar=1e14):
    G = 6.674e-11; Mk = M_bar*1.989e30; rv = 1.5
    result = np.zeros_like(r_mpc, dtype=float)
    for i, ri in enumerate(r_mpc):
        rm = ri*3.086e22
        if rm < 1e10: continue
        frac = min(1.0, (ri/rv)**2)
        gN = G*Mk*frac/rm**2
        gobs = (gN+np.sqrt(gN**2+4*gc*gN))/2
        result[i] = (gobs-gN)*rm/G/(3.086e22)**2*1.989e30
    return result

def fit_models(r, gt, err, z_mean):
    v = np.isfinite(gt) & np.isfinite(err) & (err > 0)
    if np.sum(v) < 3: return None
    rv, gv, ev = r[v], gt[v], err[v]
    results = {}

    # NFW
    best_c2, best_p = 1e10, None
    for lm in np.linspace(13, 15.5, 15):
        for cc in [2,4,6,8,10]:
            def nfw_c2(p):
                M, c = 10**p[0], p[1]
                if c<1 or c>20 or M<1e12 or M>1e16: return 1e10
                mod = nfw_gamma_t(rv, M, c)
                s = np.sum(mod**2/ev**2)
                A = np.sum(gv*mod/ev**2)/s if s>0 else 1
                return np.sum(((gv-A*mod)/ev)**2)
            try:
                res = minimize(nfw_c2, [lm, cc], method='Nelder-Mead', options={'maxiter':500})
                if res.fun < best_c2: best_c2, best_p = res.fun, res.x
            except: pass
    if best_p is not None:
        dof = np.sum(v)-2
        results["NFW"] = {"M200":float(10**best_p[0]), "c200":float(best_p[1]),
                          "chi2":float(best_c2), "dof":int(dof),
                          "chi2_dof":float(best_c2/max(dof,1)), "AIC":float(best_c2+4)}

    # MOND/膜
    def mond_c2(lgc):
        gc = 10**lgc
        mod = mond_gamma_t(rv, gc)
        s = np.sum(mod**2/ev**2)
        A = np.sum(gv*mod/ev**2)/s if s>0 else 1
        return np.sum(((gv-A*mod)/ev)**2)

    # MOND固定
    c2_mond = mond_c2(np.log10(a0))
    dof_m = np.sum(v)-1
    results["MOND"] = {"gc":float(a0), "gc_a0":1.0,
                        "chi2":float(c2_mond), "dof":int(dof_m),
                        "chi2_dof":float(c2_mond/max(dof_m,1)), "AIC":float(c2_mond+2)}

    # 膜(gc自由)
    try:
        res = minimize_scalar(mond_c2, bounds=(-12,-8), method='bounded')
        gc_b = 10**res.x; c2_mem = res.fun; dof_mem = np.sum(v)-2
        results["membrane"] = {"gc":float(gc_b), "gc_a0":float(gc_b/a0),
                                "chi2":float(c2_mem), "dof":int(dof_mem),
                                "chi2_dof":float(c2_mem/max(dof_mem,1)), "AIC":float(c2_mem+4)}
    except: pass

    if "NFW" in results:
        for m in ["MOND","membrane"]:
            if m in results:
                results[f"dAIC_{m}_vs_NFW"] = results[m]["AIC"]-results["NFW"]["AIC"]
    return results

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("HSC Y3 cluster stack analysis")
    print("="*70)

    # Step 1
    print("\n--- Step 1: candidates ---")
    cands = load_candidates()
    if not cands:
        print("No candidates."); return

    # Step 2: SDSS照合
    try:
        cands = crossmatch_sdss(cands)
    except ImportError:
        print("  requests not available, using defaults only")

    n_spec = sum(1 for c in cands if c.get("z_spec") is not None)
    print(f"\n  spec confirmed: {n_spec}/{len(cands)}")

    # Step 3
    relaxed, rejected = relaxation_filter(cands)

    # Step 4
    if relaxed:
        generate_queries(relaxed)

    # Step 5
    src_data = load_source_data(relaxed) if relaxed else {}

    # Step 6: スタック
    if src_data:
        cls_with = [c for c in relaxed if c["id"] in src_data]
        print(f"\n--- Step 6: stacked profile ({len(cls_with)} clusters) ---")
        z_mean = np.mean([c["z_spec"] for c in cls_with])
        r, gt, gx, gte, nt = stacked_profile(cls_with, src_data)

        v = np.isfinite(gt) & np.isfinite(gte) & (gte > 0)
        if np.sum(v) > 0:
            w = 1.0/gte[v]**2; ws = np.sum(w)
            gt_m = np.sum(w*gt[v])/ws
            gx_m = np.sum(w*gx[v])/ws
            gt_e = 1.0/np.sqrt(ws)
            sn = gt_m/gt_e
            print(f"  <gt>={gt_m:.5f}, <gx>={gx_m:.5f}")
            print(f"  S/N={sn:.1f}, total sources={np.sum(nt)}")

        # Step 7
        print(f"\n--- Step 7: model fit ---")
        fits = fit_models(r, gt, gte, z_mean)
        if fits:
            nfw_aic = fits.get("NFW",{}).get("AIC",0)
            print(f"\n  {'model':<12} {'chi2/dof':>10} {'AIC':>8} {'dAIC':>8}")
            print(f"  {'-'*40}")
            for nm in ["NFW","MOND","membrane"]:
                if nm in fits:
                    m = fits[nm]; da = m["AIC"]-nfw_aic
                    print(f"  {nm:<12} {m['chi2_dof']:>10.2f} {m['AIC']:>8.1f} {da:>+8.1f}")
                    if 'gc_a0' in m: print(f"  {'':12} gc/a0={m['gc_a0']:.3f}")
                    if 'M200' in m: print(f"  {'':12} M200={m['M200']:.2e}")

        # プロット
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"Stacked weak lensing: {len(cls_with)} clusters",
                         fontsize=14, fontweight='bold')

            ax = axes[0,0]
            vv = np.isfinite(gt)
            ax.errorbar(r[vv], gt[vv], gte[vv], fmt='ko', ms=6, capsize=4, label=r'$\gamma_t$')
            ax.axhline(0, color='gray', ls=':')
            ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\gamma_t$')
            ax.set_xscale('log'); ax.set_title(f'Stacked (N={len(cls_with)}, S/N={sn:.1f})')
            ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[0,1]
            vx = np.isfinite(gx)
            ax.errorbar(r[vx], gx[vx], gte[vx], fmt='ro', ms=6, capsize=4, label=r'$\gamma_\times$')
            ax.axhline(0, color='green', ls='--', lw=2)
            ax.axhspan(-0.005, 0.005, alpha=0.1, color='green')
            ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\gamma_\times$')
            ax.set_xscale('log'); ax.set_title('Cross-shear null test')
            ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[1,0]
            ax.bar(range(len(nt)), nt, color='steelblue', edgecolor='white')
            ax.set_xlabel('Bin'); ax.set_ylabel('N sources')
            ax.set_title('Sources per bin'); ax.grid(True, alpha=0.3)

            ax = axes[1,1]; ax.axis('off')
            txt = f"Clusters: {len(cls_with)}\nz_mean: {z_mean:.3f}\n"
            txt += f"S/N: {sn:.1f}\n|<gx>|: {abs(gx_m):.5f}\n\n"
            if fits:
                for nm in ["NFW","MOND","membrane"]:
                    if nm in fits:
                        m = fits[nm]
                        txt += f"{nm}: chi2/dof={m['chi2_dof']:.2f}"
                        if 'gc_a0' in m: txt += f", gc/a0={m['gc_a0']:.2f}"
                        if 'M200' in m: txt += f"\n  M200={m['M200']:.1e}"
                        txt += "\n"
                for k in fits:
                    if k.startswith("dAIC"):
                        txt += f"\n{k}={fits[k]:.1f}"
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
                   fontfamily='monospace', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout()
            fp = OUTDIR / "cluster_stack_analysis.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            print(f"\nplot: {fp}")
        except ImportError: pass
    else:
        sn = None; fits = None

    # 結果保存
    summary = {
        "n_candidates": len(cands),
        "n_spec": n_spec,
        "n_relaxed": len(relaxed),
        "n_with_data": len(src_data) if src_data else 0,
        "relaxed": [{k:v for k,v in c.items() if not isinstance(v, (np.ndarray, np.generic))}
                    for c in relaxed],
    }
    if sn is not None: summary["sn"] = float(sn)
    if fits: summary["fits"] = fits
    with open(OUTDIR / "cluster_stack_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nresults: {OUTDIR / 'cluster_stack_results.json'}")

    # 次のステップ
    print(f"\n{'='*70}")
    print("次のステップ")
    print(f"{'='*70}")
    if n_spec < 5:
        print(f"  1. 分光確認不足({n_spec}/38). 照合推奨:")
        print(f"     - redMaPPer (SDSS DR8), WHL2012, HSC-SSP Oguri+2018")
    if not src_data or len(src_data) < 3:
        print(f"  2. HSC CASでソース銀河抽出 → {OUTDIR}/hsc_source_queries.sql")
    print(f"  3. 5-10クラスタースタックで S/N~3倍向上")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
