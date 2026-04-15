#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probes_vbar_pipeline.py — PROBES V_obs + S4G 3.6μm + THINGS HI → V_bar構築 → α検定
=====================================================================================
uv run --with scipy --with matplotlib --with astropy --with requests python probes_vbar_pipeline.py
"""

import numpy as np
import json, sys, io, time
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize_scalar
from scipy.special import i0, i1, k0, k1
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("probes_vbar_output"); OUTDIR.mkdir(exist_ok=True)
G_SI = 6.674e-11; Msun = 1.989e30; pc = 3.086e16; kpc_m = 1e3*pc
a0 = 1.2e-10; Y_disk = 0.5; M_sun_36 = 3.24

# ================================================================
# VizieR取得ユーティリティ
# ================================================================
def vizier_get(source, label, max_rows=50000, timeout=120):
    import requests
    url = (f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
           f"?-source={source}&-out.max={max_rows}&-out.all")
    for attempt in range(3):
        try:
            print(f"  {label} (try {attempt+1})...", end=" ", flush=True)
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and 'not found' not in r.text.lower()[:500] and len(r.text) > 800:
                fp = OUTDIR / f"{label}.tsv"
                fp.write_text(r.text, encoding='utf-8')
                print(f"OK ({len(r.text)} bytes)")
                return r.text
            print(f"{r.status_code}/{len(r.text)}")
        except Exception as e:
            print(f"{e}")
        if attempt < 2: time.sleep(5*(attempt+1))
    return None

def parse_tsv(text):
    if not text: return None
    lines = text.strip().split('\n')
    header = None; ds = 0
    for i, line in enumerate(lines):
        if line.startswith('#') or line.startswith('-'): continue
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) >= 3:
            nn = sum(1 for p in parts[:5] if p and not p.replace('.','').replace('-','').replace('+','').replace('e','').isdigit())
            if nn >= 2:
                header = parts; ds = i+1
                while ds < len(lines) and lines[ds].startswith('-'): ds += 1
                break
    if not header: return None
    rows = []
    for line in lines[ds:]:
        if not line.strip() or line.startswith('#') or line.startswith('-'): continue
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) >= len(header):
            row = {}
            for j, h in enumerate(header):
                try: row[h] = float(parts[j])
                except: row[h] = parts[j]
            rows.append(row)
    return {"header": header, "rows": rows} if rows else None

def norm_name(n):
    import re
    s = str(n).strip().upper()
    s = re.sub(r'[\s_\-]+', '', s)
    for p in ["NGC","DDO","UGC","IC"]:
        if s.startswith(p): s = p + s[len(p):].lstrip("0")
    return s

# ================================================================
# Step 0: PROBES RC (Zenodoから取得済みprofiles/を使用)
# ================================================================
def load_probes_rc():
    """Zenodoからダウンロード済みのPROBES RCを読み込む"""
    print("="*60)
    print("Step 0: PROBES回転曲線")
    print("="*60)

    rc_dir = Path("probes_results/profiles/profiles")
    if not rc_dir.exists():
        print(f"  {rc_dir} not found")
        return None, None

    # model_fits.csv から V_flat, distance
    main = None; fits = None
    mfp = Path("probes_results/probes_main.csv")
    ffp = Path("probes_results/probes_model_fits.csv")
    if mfp.exists() and ffp.exists():
        import pandas as pd
        main = pd.read_csv(mfp, comment='#')
        fits = pd.read_csv(ffp, comment='#')

    # RC files
    rc_files = sorted(rc_dir.glob("*_rc.prof"))
    print(f"  RC files: {len(rc_files)}")

    rc_data = {}
    for fp in rc_files:
        name = fp.stem.replace("_rc", "")
        lines = fp.read_text(encoding='utf-8', errors='replace').strip().split('\n')
        r_list, v_list, e_list = [], [], []
        for line in lines:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    r = float(parts[0]); v = float(parts[1]); e = float(parts[2])
                    r_list.append(r); v_list.append(v); e_list.append(e)
                except: pass
        if len(r_list) >= 5:
            # R is in arcsec, V in km/s — need distance to convert
            rc_data[name] = {
                "R_arcsec": np.array(r_list),
                "V_obs": np.array(v_list),
                "e_V": np.array(e_list),
            }

    print(f"  Parsed: {len(rc_data)} galaxies with RC")

    # 距離情報をマージ
    dist_map = {}
    if main is not None:
        for _, row in main.iterrows():
            dist_map[row['name']] = row['distance']

    return rc_data, dist_map

# ================================================================
# Step 1: S4G 3.6μm ディスクパラメータ
# ================================================================
def fetch_s4g():
    print("\n"+"="*60)
    print("Step 1: S4G 3.6μm分解パラメータ (Salo+2015)")
    print("="*60)

    # Salo+2015 J/ApJS/219/4 — S4G decomposition
    txt = vizier_get("J/ApJS/219/4", "s4g_salo2015")
    if not txt:
        # 代替: Muñoz-Mateos+2015 J/ApJ/799/213
        txt = vizier_get("J/ApJ/799/213", "s4g_munoz2015")
    d = parse_tsv(txt) if txt else None
    if d:
        print(f"  {len(d['rows'])} rows, cols={d['header'][:15]}")
        # h_R (disk scale length) と mu0 を抽出
        s4g = {}
        for row in d["rows"]:
            nm = None; Re = None; n_s = None; Tmag = None
            for k, v in row.items():
                kl = k.lower().strip()
                if 'name' in kl or 'galaxy' in kl: nm = str(v).strip()
                elif kl == 're':
                    try: Re = float(v)
                    except: pass
                elif kl == 'n':
                    try: n_s = float(v)
                    except: pass
                elif kl == 'tmag':
                    try: Tmag = float(v)
                    except: pass
            if nm and Re and Re > 0:
                # 指数ディスク近似: h = Re / 1.678
                h_arcsec = Re / 1.678
                # mu0概算: Tmag + 2.5*log10(2*pi*Re^2) + 0.75 (Sersic n~1近似)
                mu0 = Tmag + 2.5*np.log10(2*np.pi*Re**2) if Tmag and Tmag < 90 else None
                s4g[nm] = {"h_arcsec": h_arcsec, "Re_arcsec": Re, "mu0": mu0,
                           "n_sersic": n_s, "Tmag": Tmag}
        print(f"  S4G galaxies with Re->h: {len(s4g)}")
        return s4g
    return None

# ================================================================
# Step 2: THINGS / de Blok+2008 バリオン分解RC
# ================================================================
def fetch_things_decomposed():
    print("\n"+"="*60)
    print("Step 2: THINGS分解回転曲線 (de Blok+2008)")
    print("="*60)

    # de Blok+2008 J/AJ/136/2648 — 19銀河のバリオン分解RC
    txt = vizier_get("J/AJ/136/2648", "things_deblok2008")
    if txt:
        d = parse_tsv(txt)
        if d:
            print(f"  {len(d['rows'])} rows, cols={d['header'][:15]}")
            return d

    # 代替テーブル
    for sfx in ["/table5","/table4","/table3","/table2","/table1"]:
        txt2 = vizier_get(f"J/AJ/136/2648{sfx}", f"things{sfx.replace('/','_')}")
        if txt2:
            d2 = parse_tsv(txt2)
            if d2 and len(d2["rows"]) > 10:
                print(f"  {sfx}: {len(d2['rows'])} rows, cols={d2['header'][:15]}")
                return d2
    return None

# ================================================================
# Step 3: クロスマッチ + V_bar構築 + g_cフィット
# ================================================================
def build_vbar_and_fit(rc_data, dist_map, s4g, things):
    print("\n"+"="*60)
    print("Step 3-7: V_bar構築 + g_cフィット")
    print("="*60)

    # PROBES名とS4G名のクロスマッチ
    probes_norm = {norm_name(n): n for n in rc_data.keys()}
    s4g_norm = {norm_name(n): n for n in s4g.keys()} if s4g else {}

    matched = set(probes_norm.keys()) & set(s4g_norm.keys())
    print(f"  PROBES: {len(probes_norm)}")
    print(f"  S4G: {len(s4g_norm)}")
    print(f"  Matched: {len(matched)}")

    if len(matched) < 5:
        print("  Insufficient matches.")
        # S4Gなしでも、distance + 指数ディスク仮定でV_diskを概算できる
        print("  Falling back to exponential disk assumption from V_obs profile...")
        return None

    results = []
    for nm in matched:
        p_name = probes_norm[nm]
        s_name = s4g_norm[nm]
        dist = dist_map.get(p_name)
        if not dist or dist <= 0: continue

        rc = rc_data[p_name]
        s4g_p = s4g[s_name]

        h_arcsec = s4g_p["h_arcsec"]
        mu0 = s4g_p.get("mu0")
        if not mu0 or mu0 <= 0: continue

        # arcsec → kpc
        h_kpc = h_arcsec * dist * 1e3 * np.pi / (180*3600)
        R_arcsec = rc["R_arcsec"]
        # 負のRを折り返して正側のみ使用
        pos = R_arcsec > 0
        R_as = R_arcsec[pos]
        V_obs = rc["V_obs"][pos]
        e_V = rc["e_V"][pos]

        if len(R_as) < 5: continue

        R_kpc = R_as * dist * 1e3 * np.pi / (180*3600)

        # V_disk (Freeman disk)
        try:
            V_disk, Sigma0 = compute_vdisk(R_kpc, h_kpc, mu0)
        except:
            continue

        # V_gas: S4Gにはガス情報がないので、典型的なgas fraction (20%)で近似
        # V_gas ≈ sqrt(f_gas/(1-f_gas)) × V_disk
        f_gas = 0.2
        V_gas = V_disk * np.sqrt(f_gas / max(1-f_gas, 0.01))

        V_bar = np.sqrt(V_disk**2 + V_gas**2)

        # g_c フィット (SPARC同一手法)
        fit = fit_gc(R_kpc, np.abs(V_obs), e_V, V_bar)
        if fit is None: continue

        # G*Sigma0
        n_out = max(3, len(V_obs)//4)
        v_flat = np.median(np.abs(V_obs[-n_out:]))
        GS0 = (v_flat*1e3)**2 / (h_kpc*kpc_m)

        results.append({
            "name": p_name, **fit,
            "G_Sigma0": float(GS0), "log_GS0": float(np.log10(GS0)),
            "h_R_kpc": float(h_kpc), "V_flat": float(v_flat),
            "Sigma0_disk": float(Sigma0),
        })

    print(f"  g_c fit success: {len(results)}")
    return results

def compute_vdisk(R_kpc, h_kpc, mu0_36):
    """Freeman disk V_disk(R) from h and mu0"""
    I0_Lpc2 = 10**(-0.4*(mu0_36 - M_sun_36 - 21.572))
    Sigma0 = Y_disk * I0_Lpc2  # M_sun/pc^2
    y = np.clip(R_kpc / (2.0*h_kpc), 1e-6, 50)
    bt = i0(y)*k0(y) - i1(y)*k1(y)
    h_m = h_kpc * kpc_m
    S0_SI = Sigma0 * Msun / (pc**2)
    V2 = 4*np.pi*G_SI*S0_SI*h_m * y**2 * bt
    return np.sqrt(np.maximum(V2, 0))/1e3, Sigma0

def fit_gc(r_kpc, v_obs, e_v, v_bar):
    rm = r_kpc*kpc_m; vm = v_obs*1e3; em = np.maximum(e_v,1)*1e3; vbm = v_bar*1e3
    ok = (rm>0)&(vbm>0)&(vm>0)
    if np.sum(ok) < 3: return None
    gN = vbm[ok]**2/rm[ok]; go = vm[ok]**2/rm[ok]
    eg = np.maximum(2*vm[ok]*em[ok]/rm[ok], 1e-12)
    def c2(lgc):
        gc = 10**lgc; gp = (gN+np.sqrt(gN**2+4*gc*gN))/2
        return np.sum(((go-gp)/eg)**2)
    try:
        res = minimize_scalar(c2, bounds=(-12,-8), method='bounded')
        gc = 10**res.x
        return {"g_c":float(gc),"log_gc":float(np.log10(gc)),"gc_a0":float(gc/a0),
                "chi2_dof":float(res.fun/max(np.sum(ok)-1,1)),"n_pts":int(np.sum(ok))}
    except: return None

# ================================================================
# alpha検定
# ================================================================
def alpha_test(gals):
    v = [g for g in gals if "log_gc" in g and "log_GS0" in g
         and np.isfinite(g["log_gc"]) and np.isfinite(g["log_GS0"])
         and g.get("chi2_dof",0) < 10]
    N = len(v)
    if N < 5: return None
    lgc = np.array([g["log_gc"] for g in v])
    lgs = np.array([g["log_GS0"] for g in v])
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    r = lgc-A@c; s2 = np.sum(r**2)/(N-2)
    cov = s2*np.linalg.inv(A.T@A); se = np.sqrt(cov[1,1])
    tc = t_dist.ppf(0.975, N-2)
    p05 = 2*t_dist.sf(abs((c[1]-0.5)/se), N-2)
    p0 = 2*t_dist.sf(abs(c[1]/se), N-2)
    rho, psp = spearmanr(lgs, lgc)
    zs = abs(c[1]-0.545)/np.sqrt(se**2+0.041**2)
    return {"N":N,"alpha":float(c[1]),"se":float(se),
            "ci":(float(c[1]-tc*se),float(c[1]+tc*se)),
            "p05":float(p05),"p0":float(p0),"rsd":float(np.std(r)),
            "rho":float(rho),"zs":float(zs),"lgc":lgc,"lgs":lgs,
            "interc":float(c[0]),"eta":float(np.mean(lgc-0.5*(np.log10(a0)+lgs)))}

# ================================================================
# SPARC重複除去
# ================================================================
SPARC_NAMES = {"NGC3198","NGC2841","NGC2403","NGC3031","NGC2903","NGC5055",
    "NGC7331","NGC6946","NGC3521","NGC4736","NGC5585","NGC4258","NGC925",
    "NGC2366","NGC4163","NGC1569","NGC3738","DDO154","DDO168","DDO50",
    "DDO126","DDO133","DDO87","DDO47","DDO52","DDO46","DDO43","DDO70",
    "DDO53","DDO210","DDO216","DDO101","IC2574","IC1613","UGC2885","WLM"}

# ================================================================
# メイン
# ================================================================
def main():
    print("="*60)
    print("PROBES V_bar構築パイプライン")
    print("="*60)

    rc_data, dist_map = load_probes_rc()
    if not rc_data: return

    s4g = fetch_s4g()
    things = fetch_things_decomposed()

    results = build_vbar_and_fit(rc_data, dist_map, s4g, things)

    if results and len(results) >= 5:
        # SPARC重複除去
        sparc_n = {norm_name(s) for s in SPARC_NAMES}
        indep = [g for g in results if norm_name(g["name"]) not in sparc_n]
        nov = len(results) - len(indep)
        print(f"\n  Total: {len(results)}, SPARC overlap: {nov}, Independent: {len(indep)}")

        print(f"\n{'='*60}")
        print("alpha test (all)")
        r_all = alpha_test(results)
        if r_all:
            print(f"  N={r_all['N']}, α={r_all['alpha']:.3f}±{r_all['se']:.3f}")
            print(f"  p(0.5)={r_all['p05']:.4f}, SPARC z={r_all['zs']:.2f}")

        if len(indep) >= 5:
            print(f"\nalpha test (SPARC-independent)")
            r_ind = alpha_test(indep)
            if r_ind:
                print(f"  N={r_ind['N']}, α={r_ind['alpha']:.3f}±{r_ind['se']:.3f}")
                print(f"  p(0.5)={r_ind['p05']:.4f}, SPARC z={r_ind['zs']:.2f}")
                print(f"  {'✓ NOT REJECTED' if r_ind['p05']>0.05 else '✗ REJECTED'}")

        # プロット
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            r = r_all or r_ind
            if r:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.scatter(r['lgs'], r['lgc'], c='steelblue', s=15, alpha=0.5)
                xr = np.linspace(r['lgs'].min()-0.3, r['lgs'].max()+0.3, 100)
                ax.plot(xr, r['interc']+r['alpha']*xr, 'r-', lw=2,
                        label=rf'$\alpha={r["alpha"]:.3f}\pm{r["se"]:.3f}$')
                ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+r['eta'], 'g--', lw=1.5, label=r'$\alpha=0.5$')
                ax.axhline(np.log10(a0), color='gray', ls=':')
                ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
                ax.set_title(f'PROBES V_bar: N={r["N"]}, p(0.5)={r["p05"]:.3f}')
                ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(OUTDIR/"probes_vbar_alpha.png", dpi=150)
                print(f"\nplot: {OUTDIR/'probes_vbar_alpha.png'}")
        except: pass

        # 保存
        out = {}
        if r_all: out["all"] = {k:v for k,v in r_all.items() if not isinstance(v, np.ndarray)}
        if len(indep) >= 5 and r_ind:
            out["independent"] = {k:v for k,v in r_ind.items() if not isinstance(v, np.ndarray)}
        out["n_overlap"] = nov
        with open(OUTDIR/"probes_vbar_results.json","w") as f:
            json.dump(out, f, indent=2)
        print(f"results: {OUTDIR/'probes_vbar_results.json'}")
    else:
        print("\n  Insufficient results for alpha test.")
        print("  Data availability summary:")
        if rc_data: print(f"    PROBES RC: {len(rc_data)}")
        if s4g: print(f"    S4G: {len(s4g)}")
        if things: print(f"    THINGS: available")

    print(f"\n{'='*60}\ndone\n{'='*60}")

if __name__ == "__main__":
    main()
