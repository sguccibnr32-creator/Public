# -*- coding: utf-8 -*-
"""
N-2a: MaNGA DynPop による幾何平均法則の独立検証
================================================
uv run --with requests --with scipy --with matplotlib --with numpy --with pandas python manga_verification.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("manga_results"); OUTDIR.mkdir(exist_ok=True)
H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 2.998e5
a0 = 1.2e-10
G_SI = 6.674e-11; M_SUN = 1.989e30; KPC_M = 3.0857e19

# ================================================================
# 1. データ取得
# ================================================================
def fetch_vizier(catalog, label, max_rows=15000):
    import requests
    # まずカタログ直接
    url = (f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
           f"?-source={catalog}&-out.max={max_rows}&-out.all")
    print(f"  {label}: {catalog} ...", end="", flush=True)
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.text) > 1000:
            fp = OUTDIR / f"{label}_raw.tsv"
            fp.write_text(r.text, encoding='utf-8')
            print(f" {len(r.text)} bytes")
            return r.text
        print(f" {r.status_code}/{len(r.text)}")
    except Exception as e:
        print(f" {e}")
    # テーブル名つき
    for t in ["table1","tableb1","catalog","table2","tablea1"]:
        url2 = f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={catalog}/{t}&-out.max={max_rows}&-out.all"
        try:
            r = requests.get(url2, timeout=60)
            if r.status_code == 200 and len(r.text) > 1000:
                fp = OUTDIR / f"{label}_{t}.tsv"
                fp.write_text(r.text, encoding='utf-8')
                print(f"  -> {t}: {len(r.text)} bytes")
                return r.text
        except: pass
    return None

def parse_tsv(text):
    lines = text.strip().split('\n')
    header = None; dstart = 0
    for i, line in enumerate(lines):
        if line.startswith('#') or line.startswith('-'): continue
        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                float(parts[0].strip())
            except ValueError:
                header = [p.strip() for p in parts]
                dstart = i + 1
                while dstart < len(lines) and lines[dstart].startswith('-'):
                    dstart += 1
                break
    if header is None:
        print("  no header"); return None
    rows = []
    for line in lines[dstart:]:
        if not line.strip() or line.startswith('#') or line.startswith('-'): continue
        parts = line.split('\t')
        if len(parts) < len(header): continue
        row = {}
        for j, h in enumerate(header):
            v = parts[j].strip()
            try: row[h] = float(v)
            except: row[h] = v
        rows.append(row)
    print(f"  parsed: {len(rows)} rows, cols={header[:12]}")
    return {"header": header, "rows": rows}

# ================================================================
# 2. 物性量計算
# ================================================================
def compute_props(galaxies):
    results = []
    for g in galaxies:
        vmax = None; re = None; mstar = None; name = "?"
        for k, v in g.items():
            kl = k.lower()
            if ('vmax' in kl or 'v_max' in kl or kl == 'vc') and vmax is None:
                try: vmax = float(v)
                except: pass
            elif ('re' in kl or 'r_e' in kl) and 'kpc' in kl and re is None:
                try: re = float(v)
                except: pass
            elif kl in ('re','r_e','reff','r_eff') and re is None:
                try: re = float(v)
                except: pass
            elif ('mstar' in kl or 'logm' in kl or 'm_star' in kl) and mstar is None:
                try:
                    mv = float(v)
                    mstar = 10**mv if mv < 20 else mv
                except: pass
            elif ('plate' in kl or 'manga' in kl or 'name' in kl) and name == "?":
                name = str(v)
            elif 'sigma' in kl and 'e' in kl and vmax is None:
                # 速度分散からV_max推定: V_max ~ sqrt(2)*sigma (等方近似)
                try: vmax = float(v) * np.sqrt(2)
                except: pass
        if vmax is None or vmax <= 0: continue
        if re is None or re <= 0: continue
        hR = re / 1.678
        vsi = vmax * 1e3; hsi = hR * KPC_M
        GS0 = vsi**2 / hsi
        gobs = vsi**2 / (re * KPC_M)
        entry = {"name": name, "V_max": vmax, "R_e": re, "h_R": hR,
                 "G_Sigma0": GS0, "log_GS0": np.log10(GS0), "g_obs": gobs}
        if mstar and mstar > 0:
            gbar = G_SI * 0.736 * mstar * M_SUN / (re * KPC_M)**2
            entry["M_star"] = mstar; entry["g_bar"] = gbar
            if gbar > 0 and gobs > gbar:
                gc = gobs*(gobs - gbar)/gbar
                entry["g_c"] = gc; entry["log_gc"] = np.log10(gc)
                entry["gc_a0"] = gc/a0
        results.append(entry)
    return results

# ================================================================
# 3. alpha 検定
# ================================================================
def alpha_test(gals, label=""):
    vg = [g for g in gals if "log_gc" in g and "log_GS0" in g
          and np.isfinite(g["log_gc"]) and np.isfinite(g["log_GS0"])]
    N = len(vg)
    if N < 5: print(f"  {label}: N={N}<5"); return None
    lgc = np.array([g["log_gc"] for g in vg])
    lgs = np.array([g["log_GS0"] for g in vg])
    A = np.vstack([np.ones(N), lgs]).T
    c, _, _, _ = np.linalg.lstsq(A, lgc, rcond=None)
    interc, alpha = c
    yp = A @ c; res = lgc - yp
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
    rss_f = np.sum(res**2); aic_f = N*np.log(rss_f/N)+4
    return {"label":label,"N":N,"alpha":float(alpha),"se":float(se),
            "ci":(float(ci[0]),float(ci[1])),"p05":float(p05),"p0":float(p0),
            "rsd":float(rsd),"rho":float(rho),"psp":float(psp),
            "dAIC_g":float(aic_g-aic_m),"dAIC_f":float(aic_f-aic_m),
            "interc":float(interc),"lgc":lgc,"lgs":lgs}

# ================================================================
# 4. メイン
# ================================================================
def main():
    print("="*70)
    print("N-2a: MaNGA independent verification")
    print("="*70)

    # データ取得: Zenodo DynPop FITS
    print("\n--- data: MaNGA DynPop (Zhu+2023, Zenodo) ---")
    fits_path = OUTDIR / "SDSSDR17_MaNGA_JAM_v2.fits"
    if not fits_path.exists():
        import requests
        url = "https://zenodo.org/records/17518315/files/SDSSDR17_MaNGA_JAM_v2.fits?download=1"
        print(f"  downloading {url}...")
        r = requests.get(url, timeout=300)
        fits_path.write_bytes(r.content)
        print(f"  -> {fits_path} ({len(r.content)} bytes)")
    else:
        print(f"  {fits_path} exists")

    from astropy.io import fits as afits
    from scipy.integrate import quad as _quad
    hdul = afits.open(str(fits_path))
    base = hdul[1].data  # 基本パラメータ
    print(f"  {len(base)} galaxies")

    # 宇宙論距離（グローバル定数を明示キャプチャ）
    _H0 = H0; _OM = OMEGA_M; _CL = C_LIGHT
    def _DA(z):
        r, _ = _quad(lambda zp: 1/(_OM*(1+zp)**3+(1-_OM))**0.5, 0, z)
        return _CL/_H0*r/(1+z)

    zz = base['z']
    Re_arcsec = base['Re_arcsec_MGE']
    logM = base['nsa_elpetro_mass']  # log10(M_star/M_sun)
    sigma = base['Sigma_Re']  # sigma_e [km/s]

    Da_mpc = np.array([_DA(zi) if 0.001 < zi < 0.5 else np.nan for zi in zz])
    Re_kpc = Re_arcsec / 206265 * Da_mpc * 1e3

    # 物性量計算
    print(f"\n--- computing properties ---")
    hR = Re_kpc / 1.678
    v_si = sigma * 1e3
    hR_si = hR * KPC_M

    ok = (sigma > 0) & (hR_si > 0) & np.isfinite(logM) & (logM > 7) & \
         np.isfinite(Re_kpc) & (Re_kpc > 0)
    print(f"  valid base: {np.sum(ok)}")

    GS0 = v_si[ok]**2 / hR_si[ok]
    Re_m = Re_kpc[ok] * KPC_M
    Mstar = 10**logM[ok] * M_SUN
    gobs = v_si[ok]**2 / Re_m
    gbar = G_SI * 0.736 * Mstar / Re_m**2

    gc_valid = (gobs > gbar) & (gbar > 0)
    gc = np.where(gc_valid, gobs*(gobs-gbar)/gbar, np.nan)

    # 結果をリストに整理
    names_ok = base['plateifu'][ok]
    with_gc = []
    for i in range(np.sum(ok)):
        if not np.isfinite(gc[i]) or gc[i] <= 0: continue
        with_gc.append({
            "name": str(names_ok[i]),
            "log_gc": float(np.log10(gc[i])),
            "log_GS0": float(np.log10(GS0[i])),
            "g_c": float(gc[i]),
            "gc_a0": float(gc[i]/a0),
        })
    print(f"  with g_c: {len(with_gc)}")
    print(f"  log(GS0): [{np.log10(GS0.min()):.2f}, {np.log10(GS0.max()):.2f}]")
    print(f"  median gc/a0: {np.median([g['gc_a0'] for g in with_gc]):.3f}")

    # alpha 検定
    print(f"\n{'='*70}")
    print(f"alpha test (N={len(with_gc)})")
    print(f"{'='*70}")

    res = alpha_test(with_gc, "MaNGA")
    if res is None: return

    print(f"  alpha = {res['alpha']:.3f} +/- {res['se']:.3f}")
    print(f"  95%CI: [{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]")
    print(f"  p(alpha=0.5) = {res['p05']:.4f} {'✓' if res['p05']>0.05 else '✗'}")
    print(f"  p(alpha=0) = {res['p0']:.2e}")
    print(f"  sigma_res = {res['rsd']:.3f} dex")
    print(f"  rho = {res['rho']:.3f} (p={res['psp']:.2e})")
    print(f"  dAIC(geom vs MOND) = {res['dAIC_g']:.1f}")

    sa, sse = 0.545, 0.041
    d = abs(res['alpha']-sa); cs = np.sqrt(res['se']**2+sse**2); z = d/cs
    print(f"\n  SPARC: z={z:.2f} ({'consistent ✓' if z<2 else 'inconsistent ✗'})")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"MaNGA verification: N={res['N']}", fontsize=14, fontweight='bold')

        ax = axes[0,0]
        ax.scatter(res['lgs'], res['lgc'], c='steelblue', s=10, alpha=0.3)
        xr = np.linspace(res['lgs'].min()-0.3, res['lgs'].max()+0.3, 100)
        ax.plot(xr, res['interc']+res['alpha']*xr, 'r-', lw=2,
                label=rf"$\alpha={res['alpha']:.3f}\pm{res['se']:.3f}$")
        eta = np.mean(res['lgc']-0.5*(np.log10(a0)+res['lgs']))
        ax.plot(xr, 0.5*np.log10(a0)+0.5*xr+eta, 'g--', lw=1.5, label=r'$\alpha=0.5$')
        ax.axhline(np.log10(a0), color='gray', ls=':', label='MOND')
        ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_ylabel(r'$\log(g_c)$')
        ax.set_title(rf"$\alpha={res['alpha']:.3f}$, p(0.5)={res['p05']:.3f}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[0,1]
        ds = ['SPARC\n(N=175)', 'Joint\n(N=178)', f'MaNGA\n(N={res["N"]})']
        al = [sa, 0.576, res['alpha']]
        er = [sse*1.96, 0.047*2.09, res['se']*t_dist.ppf(0.975, res['N']-2)]
        cc = ['navy','coral','steelblue']
        for i in range(3):
            ax.errorbar(al[i], i, xerr=er[i], fmt='o', ms=10, capsize=8, color=cc[i], elinewidth=2)
        ax.axvline(0.5, color='green', ls='--', lw=2)
        ax.set_xlabel(r'$\alpha$'); ax.set_yticks(range(3)); ax.set_yticklabels(ds)
        ax.set_xlim(-0.3, 1.5); ax.grid(True, alpha=0.3); ax.set_title(r'$\alpha$ 95% CI')

        ax = axes[1,0]
        rd = res['lgc']-(res['interc']+res['alpha']*res['lgs'])
        ax.hist(rd, bins=min(50, len(rd)//3), color='steelblue', edgecolor='white', density=True)
        ax.axvline(0, color='red', ls='--')
        ax.set_xlabel('residual [dex]'); ax.set_title(f'sigma={res["rsd"]:.3f} dex')
        ax.grid(True, alpha=0.3)

        ax = axes[1,1]
        ax.hist(res['lgs'], bins=30, alpha=0.6, color='steelblue', density=True, label='MaNGA')
        ax.axvspan(-11.5, -9.0, alpha=0.1, color='navy', label='SPARC')
        ax.set_xlabel(r'$\log(G\cdot\Sigma_0)$'); ax.set_title('Coverage')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = OUTDIR/"manga_verification.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except ImportError: pass

    # 保存
    out = {"N":res['N'],"alpha":res['alpha'],"se":res['se'],"ci":res['ci'],
           "p05":res['p05'],"p0":res['p0'],"rsd":res['rsd'],
           "dAIC_g":res['dAIC_g'],"sparc_z":float(z),"sparc_ok":bool(z<2)}
    with open(OUTDIR/"manga_results.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"results: {OUTDIR/'manga_results.json'}")

    print(f"\n{'='*70}")
    print("verdict")
    print(f"{'='*70}")
    if res['p05'] > 0.05 and z < 2:
        print(f"  alpha=0.5 NOT REJECTED, SPARC consistent -> Level A candidate")
    elif res['p05'] > 0.05:
        print(f"  alpha=0.5 not rejected but SPARC inconsistent")
    else:
        print(f"  alpha=0.5 REJECTED (p={res['p05']:.4f})")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
