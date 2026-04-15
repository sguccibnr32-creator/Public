# -*- coding: utf-8 -*-
"""
N-2a: PROBES + 代替RCカタログによる幾何平均法則の独立検証
=========================================================
uv run --with requests --with scipy --with matplotlib --with numpy python probes_verification.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("probes_results"); OUTDIR.mkdir(exist_ok=True)
a0 = 1.2e-10; G_SI = 6.674e-11; M_SUN = 1.989e30; KPC_M = 3.0857e19

# ================================================================
# VizieR取得
# ================================================================
def fetch_vizier(catalog, label, max_rows=50000):
    import requests
    for suffix in ["", "/table1", "/table2", "/catalog", "/tableb1"]:
        url = (f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
               f"?-source={catalog}{suffix}&-out.max={max_rows}&-out.all")
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and 'not found' not in r.text.lower() and len(r.text) > 1000:
                fp = OUTDIR / f"{label}{suffix.replace('/','_') or ''}.tsv"
                fp.write_text(r.text, encoding='utf-8')
                print(f"  {label}{suffix}: {len(r.text)} bytes OK")
                return r.text
        except: pass
    print(f"  {label}: not found")
    return None

def parse_tsv(text):
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

# ================================================================
# 銀河物性量抽出
# ================================================================
def extract_properties(data):
    header = data["header"]; rows = data["rows"]
    cm = {}
    for h in header:
        hl = h.lower().replace(' ','').replace('_','')
        if any(k in hl for k in ['vflat','vrot','vmax','vc','vmean']): cm.setdefault('vf', h)
        elif any(k in hl for k in ['hrkpc','rdkpc','rscale','hrdisk','scalelength']): cm.setdefault('hr', h)
        elif any(k in hl for k in ['rekpc','reff','rhlkpc']): cm.setdefault('re', h)
        elif any(k in hl for k in ['logmstar','logm','mstar','stellarmass']): cm.setdefault('lm', h)
        elif any(k in hl for k in ['name','galaxy','id','source']): cm.setdefault('nm', h)
        elif any(k in hl for k in ['upsilon','ml','masstolight']): cm.setdefault('ud', h)
    print(f"  cols: {header[:15]}")
    print(f"  map: {cm}")
    gals = []
    for row in rows:
        vf = None
        if cm.get('vf') and cm['vf'] in row:
            try: vf = float(row[cm['vf']])
            except: pass
        if not vf or vf <= 10: continue
        hr = None
        if cm.get('hr') and cm['hr'] in row:
            try: hr = float(row[cm['hr']])
            except: pass
        if not hr and cm.get('re') and cm['re'] in row:
            try: hr = float(row[cm['re']]) / 1.678
            except: pass
        if not hr or hr <= 0: continue
        lm = None
        if cm.get('lm') and cm['lm'] in row:
            try:
                lm = float(row[cm['lm']])
                if lm < 20: lm = lm  # already log
            except: pass
        nm = str(row.get(cm.get('nm',''), 'unknown'))
        vsi = vf*1e3; hsi = hr*KPC_M
        GS0 = vsi**2/hsi
        gobs = vsi**2/(2.2*hr*KPC_M)
        gc = None
        if lm and lm > 5:
            ms = 10**lm
            gbar = G_SI*0.65*ms*M_SUN/(2.2*hr*KPC_M)**2
            if gbar > 0 and gobs > gbar:
                gc = gobs*(gobs-gbar)/gbar
        e = {"name":nm,"V_flat":vf,"h_R":hr,"G_Sigma0":GS0,"log_GS0":np.log10(GS0)}
        if lm: e["log_Mstar"] = lm
        if gc and gc > 0: e["g_c"] = gc; e["log_gc"] = np.log10(gc); e["gc_a0"] = gc/a0
        gals.append(e)
    return gals

def extract_rc_fit(data):
    """回転曲線テーブルから銀河ごとにg_cフィット"""
    header = data["header"]; rows = data["rows"]
    nc = rc = vc = bc = None
    for h in header:
        hl = h.lower()
        if 'name' in hl or 'galaxy' in hl: nc = nc or h
        elif hl in ['r','rad','radius'] or 'rkpc' in hl: rc = rc or h
        elif 'vobs' in hl or 'vrot' in hl or hl == 'vc': vc = vc or h
        elif 'vbar' in hl: bc = bc or h
    if not (nc and rc and vc): return None
    from collections import defaultdict
    gd = defaultdict(lambda: {"r":[],"vo":[],"vb":[]})
    for row in rows:
        nm = str(row.get(nc,""));
        if not nm: continue
        try: r = float(row[rc]); v = float(row[vc])
        except: continue
        gd[nm]["r"].append(r); gd[nm]["vo"].append(v)
        if bc and bc in row:
            try: gd[nm]["vb"].append(float(row[bc]))
            except: gd[nm]["vb"].append(0)
    results = []
    for nm, d in gd.items():
        r = np.array(d["r"]); vo = np.array(d["vo"]); vb = np.array(d["vb"])
        if len(r) < 5 or len(vb) != len(r): continue
        rm = r*KPC_M; ok = (rm > 0)&(vo > 0)&(vb > 0)
        if np.sum(ok) < 3: continue
        gN = (vb[ok]*1e3)**2/rm[ok]; gobs = (vo[ok]*1e3)**2/rm[ok]
        def c2(lgc):
            gc = 10**lgc; gp = (gN+np.sqrt(gN**2+4*gc*gN))/2
            return np.sum(((gobs-gp)/(gobs*0.1))**2)
        try:
            res = minimize_scalar(c2, bounds=(-12,-8), method='bounded')
            gc = 10**res.x; no = max(3, len(vo)//4); vf = np.median(vo[-no:])
            ipk = np.argmax(np.abs(vb)); hr = max(r[ipk]/2.2, 0.1)
            GS0 = (vf*1e3)**2/(hr*KPC_M)
            results.append({"name":nm,"g_c":float(gc),"log_gc":float(np.log10(gc)),
                            "gc_a0":float(gc/a0),"V_flat":float(vf),"h_R":float(hr),
                            "G_Sigma0":float(GS0),"log_GS0":float(np.log10(GS0)),
                            "n_pts":int(np.sum(ok)),"method":"RC_fit"})
        except: pass
    return results

# ================================================================
# alpha 検定
# ================================================================
def alpha_test(gals, label=""):
    vg = [g for g in gals if "log_gc" in g and "log_GS0" in g
          and np.isfinite(g["log_gc"]) and np.isfinite(g["log_GS0"])
          and -15 < g["log_gc"] < -5]
    N = len(vg)
    if N < 10: return None
    lgc = np.array([g["log_gc"] for g in vg])
    lgs = np.array([g["log_GS0"] for g in vg])
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    interc, alpha = c; res = lgc-A@c
    s2 = np.sum(res**2)/(N-2); cov = s2*np.linalg.inv(A.T@A)
    se = np.sqrt(cov[1,1]); rsd = np.std(res)
    tc = t_dist.ppf(0.975, N-2); ci = (alpha-tc*se, alpha+tc*se)
    p05 = 2*t_dist.sf(abs((alpha-0.5)/se), N-2)
    p0 = 2*t_dist.sf(abs(alpha/se), N-2)
    rho, psp = spearmanr(lgs, lgc)
    gm = np.full(N, np.log10(a0))
    rss_m = np.sum((lgc-gm)**2); aic_m = N*np.log(rss_m/N)
    gcg = 0.5*(np.log10(a0)+lgs); eta = np.mean(lgc-gcg)
    rss_g = np.sum((lgc-gcg-eta)**2); aic_g = N*np.log(rss_g/N)+2
    d = abs(alpha-0.545); cs = np.sqrt(se**2+0.041**2); zs = d/cs
    return {"label":label,"N":N,"alpha":float(alpha),"se":float(se),
            "ci":(float(ci[0]),float(ci[1])),"p05":float(p05),"p0":float(p0),
            "rsd":float(rsd),"rho":float(rho),"psp":float(psp),
            "dAIC_g":float(aic_g-aic_m),"interc":float(interc),"eta":float(eta),
            "zs":float(zs),"sparc_ok":bool(zs<2),"lgc":lgc,"lgs":lgs}

# SPARC重複除去
SPARC_NAMES = {"NGC3198","NGC2841","NGC2403","NGC3031","NGC2903","NGC5055",
    "NGC7331","NGC6946","NGC3521","NGC4736","NGC5585","NGC4258","NGC1003",
    "NGC4395","NGC3109","NGC300","NGC55","NGC247","NGC4559","NGC6503",
    "NGC3621","NGC2976","NGC4214","NGC4449","NGC925","NGC2366","NGC4163",
    "NGC1569","NGC3738","DDO154","DDO168","DDO50","DDO126","DDO133",
    "DDO87","DDO47","DDO52","DDO46","DDO43","DDO70","DDO53","DDO210",
    "DDO216","DDO101","IC2574","IC1613","UGC2885","UGC128","WLM","CVnIdwA","LeoA"}
def norm_name(n):
    s = str(n).upper().replace(" ","").replace("_","").replace("-","")
    for p in ["NGC","DDO","UGC","IC"]:
        if s.startswith(p): s = p+s[len(p):].lstrip("0")
    return s
def remove_sparc(gals):
    sn = {norm_name(s) for s in SPARC_NAMES}
    nv = [g for g in gals if norm_name(g.get("name","")) not in sn]
    ov = [g["name"] for g in gals if norm_name(g.get("name","")) in sn]
    return nv, ov

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("PROBES + alternative RC catalogs")
    print("="*70)

    all_gals = {}

    # PROBES
    print("\n--- PROBES (Stone+2021) ---")
    txt = fetch_vizier("J/ApJS/256/19", "probes")
    if txt:
        d = parse_tsv(txt)
        if d:
            g = extract_properties(d)
            gc_g = [x for x in g if "g_c" in x]
            print(f"  {len(g)} total, {len(gc_g)} with g_c")
            if gc_g: all_gals["PROBES"] = gc_g
            # RC fit も試行
            rc = extract_rc_fit(d)
            if rc: all_gals["PROBES_RC"] = rc; print(f"  RC fit: {len(rc)}")

    # 代替カタログ
    print("\n--- alternative catalogs ---")
    alts = [
        ("J/AJ/152/157", "SPARC_Lelli2016"),
        ("J/PASJ/68/2", "Sofue2016"),
        ("J/MNRAS/465/4703", "Karukes2017"),
        ("J/MNRAS/474/4366", "Ponomareva2018"),
        ("J/A+A/641/A31", "Shelest2020"),
    ]
    for cat, label in alts:
        txt = fetch_vizier(cat, label)
        if not txt: continue
        d = parse_tsv(txt)
        if not d or len(d["rows"]) < 10: continue
        print(f"  {label}: {len(d['rows'])} rows")
        # RC fit
        rc = extract_rc_fit(d)
        if rc and len(rc) > 5:
            all_gals[f"{label}_RC"] = rc
            print(f"    RC fit: {len(rc)}")
        # Properties
        g = extract_properties(d)
        gc_g = [x for x in g if "g_c" in x]
        if gc_g and len(gc_g) > 5:
            all_gals[f"{label}_prop"] = gc_g
            print(f"    properties: {len(gc_g)} with g_c")

    # 結果
    print(f"\n{'='*70}")
    print("results")
    print(f"{'='*70}")

    print(f"\n  {'catalog':<28} {'N':>5} {'α':>7} {'σ_α':>6} {'p(.5)':>10} {'z_SP':>6}")
    print(f"  {'-'*64}")
    print(f"  {'SPARC (reference)':<28} {'175':>5} {'0.545':>7} {'0.041':>6} {'0.273':>10} {'--':>6}")
    print(f"  {'LT+SPARC joint':<28} {'178':>5} {'0.576':>7} {'0.047':>6} {'0.109':>10} {'--':>6}")

    for key, gals in all_gals.items():
        nv, ov = remove_sparc(gals)
        r = alpha_test(nv, key) if len(nv) >= 10 else alpha_test(gals, key)
        used = nv if len(nv) >= 10 else gals
        tag = f" (-{len(ov)}SP)" if len(nv) >= 10 and ov else ""
        if r:
            print(f"  {(key+tag):<28} {r['N']:>5} {r['alpha']:>7.3f} {r['se']:>6.3f} "
                  f"{r['p05']:>10.4f} {r['zs']:>6.2f}")
            # プロット
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8,6))
                ax.scatter(r['lgs'], r['lgc'], c='steelblue', s=15, alpha=0.5)
                xr = np.linspace(r['lgs'].min()-0.3, r['lgs'].max()+0.3, 100)
                ax.plot(xr, r['interc']+r['alpha']*xr, 'r-', lw=2,
                        label=rf"$\alpha={r['alpha']:.3f}\pm{r['se']:.3f}$")
                ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+r['eta'], 'g--', lw=1.5, label=r'$\alpha=0.5$')
                ax.axhline(np.log10(a0), color='gray', ls=':')
                ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
                ax.set_title(f"{key}: N={r['N']}, p(0.5)={r['p05']:.3f}")
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fp = OUTDIR/f"{key.replace(' ','_')}.png"
                plt.savefig(fp, dpi=150); plt.close()
            except: pass
            # 保存
            out = {k:v for k,v in r.items() if not isinstance(v, np.ndarray)}
            out["n_sparc_overlap"] = len(ov)
            with open(OUTDIR/f"{key.replace(' ','_')}.json","w") as f:
                json.dump(out, f, indent=2)

    if not all_gals:
        print("\n  データ取得不可。VizieRを確認してください。")
        print("  PROBES: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/PROBES/")

    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
