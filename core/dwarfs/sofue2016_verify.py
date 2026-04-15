# -*- coding: utf-8 -*-
"""
Sofue 2016 (PASJ 68, 2) 回転曲線コンパイル → alpha 検定
=========================================================
uv run --with requests --with scipy --with matplotlib --with numpy python sofue2016_verify.py
"""

import numpy as np
import json, sys, io, time
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("sofue_results"); OUTDIR.mkdir(exist_ok=True)
a0 = 1.2e-10; KPC_M = 3.0857e19

# ================================================================
# 1. VizieR/CDS取得
# ================================================================
def fetch(url, label, timeout=120, retries=3):
    import requests
    for i in range(retries):
        try:
            print(f"  {label} (try {i+1})...", end=" ", flush=True)
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and 'not found' not in r.text.lower()[:500] and len(r.text) > 500:
                print(f"OK ({len(r.text)} bytes)")
                return r.text
            print(f"{r.status_code}/{len(r.text)}")
        except Exception as e:
            print(f"{e}")
        if i < retries-1: time.sleep(5*(i+1))
    return None

def fetch_all():
    results = {}
    base = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    # Sofue 2016
    for cat, label in [("J/PASJ/68/2","Sofue2016"),("J/PASJ/69/R1","Sofue2017")]:
        for sfx in ["","table1","table2","table3","rctab"]:
            src = f"{cat}/{sfx}" if sfx else cat
            url = f"{base}?-source={src}&-out.max=50000&-out.all"
            txt = fetch(url, f"{label}/{sfx or 'main'}")
            if txt:
                fp = OUTDIR / f"{label}_{sfx or 'main'}.tsv"
                fp.write_text(txt, encoding='utf-8')
                results[f"{label}_{sfx or 'main'}"] = txt
    # CDS direct
    for fname in ["table1.dat","table2.dat","ReadMe"]:
        url = f"https://cdsarc.cds.unistra.fr/ftp/J/PASJ/68/2/{fname}"
        txt = fetch(url, f"CDS/{fname}", timeout=60, retries=2)
        if txt:
            fp = OUTDIR / f"sofue_cds_{fname}"
            fp.write_text(txt, encoding='utf-8')
            results[f"cds_{fname}"] = txt
    return results

# ================================================================
# 2. パーサー
# ================================================================
def parse_tsv(text):
    lines = text.strip().split('\n')
    header = None; ds = 0
    for i, line in enumerate(lines):
        if line.startswith('#') or line.startswith('-'): continue
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) < 3: parts = line.split()
        if len(parts) >= 3:
            nn = sum(1 for p in parts[:4] if p and not p.replace('.','').replace('-','').replace('+','').replace('e','').isdigit())
            if nn >= 2:
                header = parts; ds = i+1
                while ds < len(lines) and lines[ds].startswith('-'): ds += 1
                break
    if not header: return None
    rows = []
    for line in lines[ds:]:
        if not line.strip() or line.startswith('#') or line.startswith('-'): continue
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) < len(header): parts = line.split()
        if len(parts) >= len(header):
            row = {}
            for j, h in enumerate(header):
                try: row[h] = float(parts[j])
                except: row[h] = parts[j]
            rows.append(row)
    return {"header": header, "rows": rows} if rows else None

# ================================================================
# 3. g_c フィット
# ================================================================
def extract_and_fit(data):
    header = data["header"]; rows = data["rows"]
    print(f"  cols ({len(header)}): {header[:15]}")
    col = {}
    for h in header:
        hl = h.lower().replace(' ','').replace('_','')
        if 'name' in hl or 'galaxy' in hl: col.setdefault('nm', h)
        elif hl in ['r','rad','rkpc'] or 'rkpc' in hl: col.setdefault('r', h)
        elif 'vobs' in hl or 'vrot' in hl or 'vtot' in hl: col.setdefault('vo', h)
        elif 'verr' in hl or 'evobs' in hl or 'dv' in hl: col.setdefault('ev', h)
        elif 'vbar' in hl: col.setdefault('vb', h)
        elif 'vgas' in hl or 'vhi' in hl: col.setdefault('vg', h)
        elif 'vdisk' in hl or 'vstar' in hl: col.setdefault('vd', h)
        elif 'vbul' in hl: col.setdefault('vbl', h)
        elif 'vflat' in hl or 'vmax' in hl: col.setdefault('vf', h)
        elif 'rd' in hl or 'scalelength' in hl: col.setdefault('hr', h)
    print(f"  map: {col}")

    if 'r' in col and 'vo' in col and 'nm' in col:
        return fit_rc(rows, col)
    elif 'nm' in col and ('vf' in col or 'vo' in col):
        return fit_params(rows, col)
    return None

def fit_rc(rows, col):
    from collections import defaultdict
    gd = defaultdict(lambda: {"r":[],"v":[],"e":[],"vb":[]})
    for row in rows:
        nm = str(row.get(col['nm'],''))
        if not nm: continue
        try: r = float(row[col['r']]); v = float(row[col['vo']])
        except: continue
        gd[nm]["r"].append(r); gd[nm]["v"].append(v)
        gd[nm]["e"].append(float(row.get(col.get('ev',''),v*0.1)) if 'ev' in col else max(v*0.1,2))
        vb = None
        if 'vb' in col:
            try: vb = float(row[col['vb']])
            except: pass
        elif 'vg' in col and 'vd' in col:
            try:
                vg = float(row.get(col['vg'],0)); vd = float(row.get(col['vd'],0))
                vbl = float(row.get(col.get('vbl',''),0)) if 'vbl' in col else 0
                vb = np.sqrt(abs(vg)**2+abs(vd)**2+abs(vbl)**2)
            except: pass
        gd[nm]["vb"].append(vb)
    print(f"  RC galaxies: {len(gd)}")
    results = []; nf = 0; nnv = 0
    for nm, d in gd.items():
        if None in d["vb"] or len(d["r"]) < 5: nnv += 1; continue
        r = np.array(d["r"]); v = np.array(d["v"]); e = np.array(d["e"]); vb = np.array(d["vb"])
        fit = _fit_gc(r, v, e, vb)
        if fit: fit["name"] = nm; results.append(fit)
        else: nf += 1
    print(f"  fit: {len(results)}, no_vbar: {nnv}, fail: {nf}")
    return results

def fit_params(rows, col):
    results = []
    for row in rows:
        nm = str(row.get(col['nm'],''))
        vf = None
        for k in ['vf','vo']:
            if k in col:
                try: vf = float(row[col[k]]); break
                except: pass
        hr = None
        if 'hr' in col:
            try: hr = float(row[col['hr']])
            except: pass
        if not vf or vf <= 10 or not hr or hr <= 0: continue
        GS = (vf*1e3)**2/(hr*KPC_M)
        results.append({"name":nm,"V_flat":vf,"h_R_kpc":hr,
                        "G_Sigma0":float(GS),"log_G_Sigma0":float(np.log10(GS)),
                        "method":"params"})
    print(f"  params galaxies: {len(results)}")
    return results

def _fit_gc(r, v, e, vb):
    rm = r*KPC_M; vm = v*1e3; em = np.maximum(e,1)*1e3; vbm = vb*1e3
    ok = (rm>0)&(vbm>0)&(vm>0)
    if np.sum(ok) < 3: return None
    gN = vbm[ok]**2/rm[ok]; go = vm[ok]**2/rm[ok]
    eg = np.maximum(2*vm[ok]*em[ok]/rm[ok], 1e-12)
    def c2(lgc):
        gc = 10**lgc; gp = (gN+np.sqrt(gN**2+4*gc*gN))/2
        return np.sum(((go-gp)/eg)**2)
    try:
        res = minimize_scalar(c2, bounds=(-12,-8), method='bounded')
        gc = 10**res.x; dof = np.sum(ok)-1
        no = max(3, len(v)//4); vf = np.median(v[-no:])
        idx = np.argmax(np.abs(vb)); hr = max(r[idx]/2.2, 0.05)
        GS = (vf*1e3)**2/(hr*KPC_M)
        return {"g_c":float(gc),"log_gc":float(np.log10(gc)),"gc_a0":float(gc/a0),
                "chi2_dof":float(res.fun/max(dof,1)),"n_points":int(np.sum(ok)),
                "V_flat":float(vf),"h_R_kpc":float(hr),
                "G_Sigma0":float(GS),"log_G_Sigma0":float(np.log10(GS)),
                "method":"full_RC_fit"}
    except: return None

# ================================================================
# 4. SPARC重複除去 + alpha検定
# ================================================================
SPARC_NAMES = {"NGC3198","NGC2841","NGC2403","NGC3031","NGC2903","NGC5055",
    "NGC7331","NGC6946","NGC3521","NGC4736","NGC5585","NGC4258","NGC1003",
    "NGC4395","NGC3109","NGC300","NGC55","NGC247","NGC4559","NGC6503",
    "NGC3621","NGC2976","NGC4214","NGC4449","NGC925","NGC2366","NGC4163",
    "NGC1569","NGC3738","DDO154","DDO168","DDO50","DDO126","DDO133",
    "DDO87","DDO47","DDO52","DDO46","DDO43","DDO70","DDO53","DDO210",
    "DDO216","DDO101","IC2574","IC1613","UGC2885","UGC128","WLM","CVNIDWA","LEOA"}
def norm(n):
    s = str(n).upper().replace(" ","").replace("_","").replace("-","")
    for p in ["NGC","DDO","UGC","IC"]:
        if s.startswith(p): s = p+s[len(p):].lstrip("0")
    return s
def is_sparc(n): return norm(n) in {norm(s) for s in SPARC_NAMES}

def alpha_test(gals, label=""):
    v = [g for g in gals if "log_gc" in g and "log_G_Sigma0" in g
         and np.isfinite(g["log_gc"]) and np.isfinite(g["log_G_Sigma0"])
         and g.get("chi2_dof",0) < 10]
    N = len(v)
    if N < 10: return None
    lgc = np.array([g["log_gc"] for g in v])
    lgs = np.array([g["log_G_Sigma0"] for g in v])
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    r = lgc-A@c; s2 = np.sum(r**2)/(N-2)
    cov = s2*np.linalg.inv(A.T@A); se = np.sqrt(cov[1,1])
    tc = t_dist.ppf(0.975, N-2)
    p05 = 2*t_dist.sf(abs((c[1]-0.5)/se), N-2)
    p0 = 2*t_dist.sf(abs(c[1]/se), N-2)
    rho, psp = spearmanr(lgs, lgc)
    rss_m = np.sum((lgc-np.log10(a0))**2); aic_m = N*np.log(rss_m/N)
    gcg = 0.5*(np.log10(a0)+lgs); eta = np.mean(lgc-gcg)
    rss_g = np.sum((lgc-gcg-eta)**2); aic_g = N*np.log(rss_g/N)+2
    zs = abs(c[1]-0.545)/np.sqrt(se**2+0.041**2)
    return {"label":label,"N":N,"alpha":float(c[1]),"se":float(se),
            "ci":(float(c[1]-tc*se),float(c[1]+tc*se)),
            "p05":float(p05),"p0":float(p0),"rsd":float(np.std(r)),
            "rho":float(rho),"dAIC":float(aic_g-aic_m),
            "zs":float(zs),"interc":float(c[0]),"eta":float(eta),
            "lgc":lgc,"lgs":lgs}

# ================================================================
# 5. メイン
# ================================================================
def main():
    print("="*70)
    print("Sofue 2016 rotation curve compilation")
    print("="*70)

    print("\n--- fetch ---")
    data = fetch_all()
    if not data:
        # ローカルチェック
        local = list(OUTDIR.glob("*.tsv")) + list(OUTDIR.glob("*.dat"))
        if local:
            print(f"  local files: {[f.name for f in local]}")
            for f in local: data[f.stem] = f.read_text(encoding='utf-8', errors='replace')
        else:
            print("  No data."); return

    print(f"\n--- processing {len(data)} tables ---")
    all_results = []
    for key, text in data.items():
        print(f"\n  [{key}]")
        parsed = parse_tsv(text)
        if not parsed or len(parsed["rows"]) < 5:
            print(f"    skip"); continue
        gals = extract_and_fit(parsed)
        if gals: all_results.extend(gals)

    rc = [g for g in all_results if g.get("method") == "full_RC_fit"]
    prm = [g for g in all_results if g.get("method") == "params"]
    print(f"\n  RC fit: {len(rc)}, params: {len(prm)}")

    # alpha検定
    for subset, label in [(rc,"Sofue_RC"),(prm,"Sofue_params"),(all_results,"Sofue_all")]:
        if len(subset) < 10: continue
        ns = [g for g in subset if not is_sparc(g.get("name",""))]
        nov = len(subset)-len(ns)
        print(f"\n{'='*50}")
        print(f"  {label}: total={len(subset)}, overlap={nov}, indep={len(ns)}")
        res = alpha_test(ns, label)
        if not res:
            if len(ns) < 10: print(f"    N={len(ns)}<10")
            continue
        print(f"  alpha = {res['alpha']:.3f} ± {res['se']:.3f}")
        print(f"  95%CI = [{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]")
        print(f"  p(0.5) = {res['p05']:.4f} {'✓' if res['p05']>0.05 else '✗'}")
        print(f"  p(0) = {res['p0']:.2e}")
        print(f"  σ = {res['rsd']:.3f}, ρ = {res['rho']:.3f}")
        print(f"  SPARC z = {res['zs']:.2f} {'✓' if res['zs']<2 else '✗'}")
        print(f"  dAIC = {res['dAIC']:.1f}")

        # プロット
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(res['lgs'], res['lgc'], c='steelblue', s=15, alpha=0.5)
            xr = np.linspace(res['lgs'].min()-0.3, res['lgs'].max()+0.3, 100)
            ax.plot(xr, res['interc']+res['alpha']*xr, 'r-', lw=2,
                    label=rf'$\alpha={res["alpha"]:.3f}\pm{res["se"]:.3f}$')
            ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+res['eta'], 'g--', lw=1.5, label=r'$\alpha=0.5$')
            ax.axhline(np.log10(a0), color='gray', ls=':')
            ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
            ax.set_title(f'{label}: N={res["N"]}, p(0.5)={res["p05"]:.3f}')
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(OUTDIR/f"{label.lower()}.png", dpi=150); plt.close()
        except: pass

        out = {k:v for k,v in res.items() if not isinstance(v, np.ndarray)}
        out["n_overlap"] = nov
        with open(OUTDIR/f"{label.lower()}.json","w") as f:
            json.dump(out, f, indent=2)

    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
