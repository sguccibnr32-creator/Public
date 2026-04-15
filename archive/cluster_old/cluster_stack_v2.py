# -*- coding: utf-8 -*-
"""
スタック弱レンズ v2: Sigma_crit正規化 + ミスセンタリング補正
============================================================
uv run --with scipy --with matplotlib --with numpy --with pandas python cluster_stack_v2.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("cluster_stack"); OUTDIR.mkdir(exist_ok=True)

# 物理定数・宇宙論
H0 = 70.0; OMEGA_M = 0.3; C_LIGHT = 2.998e5
G_SI = 6.674e-11; M_SUN = 1.989e30; MPC_M = 3.0857e22; PC_M = 3.0857e16
a0 = 1.2e-10

def E_z(z): return np.sqrt(OMEGA_M*(1+z)**3+(1-OMEGA_M))
def comoving_distance(z):
    r, _ = quad(lambda zp: 1.0/E_z(zp), 0, z)
    return C_LIGHT/H0*r
def angular_diameter_distance(z): return comoving_distance(z)/(1+z)
def D_ls_func(z_l, z_s):
    return (comoving_distance(z_s)-comoving_distance(z_l))/(1+z_s)

def sigma_crit(z_l, z_s):
    """Sigma_crit [M_sun/pc^2]"""
    Dl = angular_diameter_distance(z_l)*MPC_M
    Ds = angular_diameter_distance(z_s)*MPC_M
    Dls = D_ls_func(z_l, z_s)*MPC_M
    if Dls <= 0 or Ds <= 0 or Dl <= 0: return np.inf
    pf = (C_LIGHT*1e3)**2/(4*np.pi*G_SI)
    return pf*Ds/(Dl*Dls) / M_SUN * PC_M**2

def rho_crit(z):
    Hz = H0*E_z(z)*1e3/MPC_M
    return 3*Hz**2/(8*np.pi*G_SI) / M_SUN * MPC_M**3

# ================================================================
# NFW DeltaSigma (Wright & Brainerd 2000)
# ================================================================
def nfw_delta_sigma(R_mpc, M200, c200, z_l):
    """DeltaSigma [M_sun/pc^2]"""
    rc = rho_crit(z_l)
    r200 = (3*M200/(4*np.pi*200*rc))**(1./3.)
    rs = r200/c200
    dc = (200./3.)*c200**3/(np.log(1+c200)-c200/(1+c200))
    rho_s = dc*rc
    x = np.atleast_1d(R_mpc/rs).astype(float)
    sig = np.zeros_like(x); bsig = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 1e-6: continue
        if abs(xi-1)<1e-4: sig[i] = 2*rho_s*rs/3.0
        elif xi<1: sig[i] = 2*rho_s*rs/(xi**2-1)*(1-np.arccosh(1/xi)/np.sqrt(1-xi**2))
        else: sig[i] = 2*rho_s*rs/(xi**2-1)*(1-np.arccos(1/xi)/np.sqrt(xi**2-1))
        if abs(xi-1)<1e-4: h = 1+np.log(0.5)
        elif xi<1: h = np.arccosh(1/xi)/np.sqrt(1-xi**2)+np.log(xi/2)
        else: h = np.arccos(1/xi)/np.sqrt(xi**2-1)+np.log(xi/2)
        bsig[i] = 4*rho_s*rs*h/xi**2
    return (bsig - sig)*1e-12  # M_sun/Mpc^2 -> M_sun/pc^2

def nfw_ds_miscentered(R_mpc, M200, c200, z_l, R_misc):
    if R_misc < 1e-4: return nfw_delta_sigma(R_mpc, M200, c200, z_l)
    R = np.atleast_1d(R_mpc)
    ds_c = nfw_delta_sigma(R, M200, c200, z_l)
    # 方位角平均によるミスセンタリング
    n_phi = 36; phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    ds_m = np.zeros_like(R)
    for i, Ri in enumerate(R):
        s = 0.0
        for phi in phis:
            Re = max(np.sqrt(Ri**2+R_misc**2-2*Ri*R_misc*np.cos(phi)), 1e-4)
            s += nfw_delta_sigma(np.array([Re]), M200, c200, z_l)[0]
        ds_m[i] = s/n_phi
    f = np.exp(-0.5*(R_misc/np.maximum(R, 1e-4))**2)
    return f*ds_c + (1-f)*ds_m

# ================================================================
# MOND/膜 DeltaSigma (Abel変換)
# ================================================================
def mond_delta_sigma_abel(R_mpc, M_bar, gc, z_l):
    """DeltaSigma [M_sun/pc^2] via Abel transform"""
    c_bar = 5.0; rc = rho_crit(z_l)
    r200 = (3*M_bar/(4*np.pi*200*rc))**(1./3.)
    rs = r200/c_bar; norm = np.log(1+c_bar)-c_bar/(1+c_bar)
    R = np.atleast_1d(R_mpc)
    sig = np.zeros_like(R)
    n_los = 80
    for i, Ri in enumerate(R):
        if Ri < 1e-4: continue
        z_max = 10*r200
        zs = np.linspace(0, z_max, n_los); dz = zs[1]-zs[0]
        integ = np.zeros(n_los)
        for j, zj in enumerate(zs):
            r3d = np.sqrt(Ri**2+zj**2)
            if r3d < 1e-6: continue
            Me = M_bar*(np.log(1+r3d/rs)-(r3d/rs)/(1+r3d/rs))/norm
            gN = G_SI*Me*M_SUN/(r3d*MPC_M)**2
            gobs = (gN+np.sqrt(gN**2+4*gc*abs(gN)))/2
            integ[j] = (gobs-gN)/(4*np.pi*G_SI)/(r3d*MPC_M)
        integ_msun = integ/M_SUN*MPC_M**3
        sig[i] = 2*np.trapezoid(integ_msun, zs)  # M_sun/Mpc^2 (zs in Mpc)
    bsig = np.zeros_like(R)
    for i, Ri in enumerate(R):
        if Ri < 1e-4 or i == 0: bsig[i] = sig[i]; continue
        Rs = R[:i+1]; Ss = sig[:i+1]
        m = Rs <= Ri
        if np.sum(m) < 2: bsig[i] = sig[i]; continue
        bsig[i] = 2/Ri**2*np.trapezoid(Ss[m]*Rs[m], Rs[m])
    return (bsig - sig)*1e-12

# ================================================================
# スタック DeltaSigma
# ================================================================
def stacked_ds(clusters, src_data, nb=12):
    bins = np.logspace(np.log10(0.3), np.log10(5.0), nb+1)  # R_min=0.3 Mpc to avoid miscentering noise
    ds_s = np.zeros(nb); dx_s = np.zeros(nb)
    w_s = np.zeros(nb); w2_s = np.zeros(nb)
    nt = np.zeros(nb, dtype=int)

    for c in clusters:
        if c["id"] not in src_data: continue
        src = src_data[c["id"]]; z_l = c["z_spec"]
        Dl = angular_diameter_distance(z_l)
        ra_c, dec_c = c["ra"], c["dec"]

        dra = (src["ra"]-ra_c)*np.cos(np.radians(dec_c))*np.pi/180*Dl
        ddec = (src["dec"]-dec_c)*np.pi/180*Dl
        Rm = np.sqrt(dra**2+ddec**2)

        phi = np.arctan2(ddec, dra)
        c2 = np.cos(2*phi); s2 = np.sin(2*phi)
        et = -(src["e1"]*c2+src["e2"]*s2)
        ex = src["e1"]*s2-src["e2"]*c2

        sc = np.array([sigma_crit(z_l, zs) for zs in src["z_photo"]])
        ok = np.isfinite(sc) & (sc > 0) & (sc < 1e6)

        ds_i = et*sc; dx_i = ex*sc
        ws = src.get("weight", np.ones(len(src["ra"])))
        wi = ws/sc**2; wi[~ok] = 0

        for b in range(nb):
            m = (Rm >= bins[b])&(Rm < bins[b+1])&ok&(wi > 0)
            n = np.sum(m)
            if n == 0: continue
            nt[b] += n; wm = wi[m]
            ds_s[b] += np.sum(wm*ds_i[m])
            dx_s[b] += np.sum(wm*dx_i[m])
            w_s[b] += np.sum(wm); w2_s[b] += np.sum(wm**2)

    Rmid = np.sqrt(bins[:-1]*bins[1:])
    ds_st = np.where(w_s > 0, ds_s/w_s, np.nan)
    dx_st = np.where(w_s > 0, dx_s/w_s, np.nan)
    neff = np.where(w2_s > 0, w_s**2/w2_s, 0)
    msc = np.where(w_s > 0, np.sqrt(neff/w_s), 1e4)
    ds_err = 0.25*msc/np.sqrt(np.maximum(neff, 1))
    return Rmid, ds_st, dx_st, ds_err, nt

# ================================================================
# モデルフィット
# ================================================================
def fit_nfw(R, ds, err, z, misc=False):
    v = np.isfinite(ds)&np.isfinite(err)&(err > 0)
    if np.sum(v) < 3: return None
    Rv, Dv, Ev = R[v], ds[v], err[v]
    def c2(p):
        M = 10**p[0]; c = p[1]
        if c<1 or c>20 or M<1e12 or M>1e16: return 1e10
        if misc and len(p)>2:
            mod = nfw_ds_miscentered(Rv, M, c, z, 10**p[2])
        else:
            mod = nfw_delta_sigma(Rv, M, c, z)
        return np.sum(((Dv-mod)/Ev)**2)
    best_c, best_p = 1e10, None
    pars = [(lm, cc, lr) for lm in np.linspace(13.5,15.5,8)
            for cc in [3,5,8,12] for lr in ([-2,-1.5,-1,-0.5] if misc else [0])]
    for p0 in pars:
        x0 = list(p0[:2]) + (list(p0[2:]) if misc else [])
        try:
            r = minimize(c2, x0, method='Nelder-Mead', options={'maxiter':500})
            if r.fun < best_c: best_c, best_p = r.fun, r.x
        except: pass
    if best_p is None: return None
    np_ = 3 if misc else 2
    d = {"M200":float(10**best_p[0]),"c200":float(best_p[1]),
         "chi2":float(best_c),"dof":int(np.sum(v)-np_),
         "chi2_dof":float(best_c/max(np.sum(v)-np_,1)),"AIC":float(best_c+2*np_),"np":np_}
    if misc: d["R_misc"]=float(10**best_p[2])
    return d

def fit_mond(R, ds, err, z, gc_fixed=None):
    v = np.isfinite(ds)&np.isfinite(err)&(err > 0)
    if np.sum(v) < 3: return None
    Rv, Dv, Ev = R[v], ds[v], err[v]
    def c2(p):
        M = 10**p[0]; gc = gc_fixed if gc_fixed else 10**p[1]
        if M<1e11 or M>1e15: return 1e10
        mod = mond_delta_sigma_abel(Rv, M, gc, z)
        if np.any(~np.isfinite(mod)): return 1e10
        return np.sum(((Dv-mod)/Ev)**2)
    best_c, best_p = 1e10, None
    for lm in np.linspace(12,15,15):
        gcs = [np.log10(gc_fixed)] if gc_fixed else np.linspace(-11,-9,8)
        for lgc in gcs:
            try:
                r = minimize(c2, [lm, lgc], method='Nelder-Mead', options={'maxiter':300})
                if r.fun < best_c: best_c, best_p = r.fun, r.x
            except: pass
    if best_p is None: return None
    np_ = 1 if gc_fixed else 2
    gc_out = gc_fixed if gc_fixed else 10**best_p[1]
    return {"M_bar":float(10**best_p[0]),"gc":float(gc_out),"gc_a0":float(gc_out/a0),
            "chi2":float(best_c),"dof":int(np.sum(v)-np_),
            "chi2_dof":float(best_c/max(np.sum(v)-np_,1)),"AIC":float(best_c+2*np_),"np":np_}

# ================================================================
# データ読み込み
# ================================================================
def load_clusters():
    """v1の結果JSONから分光確認済みクラスターを読む"""
    p = OUTDIR/"cluster_stack_results.json"
    if not p.exists(): return None
    with open(p) as f: d = json.load(f)
    return d.get("relaxed", [])

def load_src(cid):
    import pandas as pd
    for pat in [f"{cid}_sources_photoz.csv", f"{cid}_sources.csv"]:
        p = Path(pat)
        if not p.exists(): continue
        # #付きヘッダ対応
        with open(p) as fh: first = fh.readline()
        if first.startswith('# '):
            cols = first[2:].strip().split(',')
            df = pd.read_csv(p, skiprows=1, header=None, names=cols)
        else:
            df = pd.read_csv(p, comment='#')
        cm = {}
        for c in df.columns:
            cl = c.lower()
            if 'ra' in cl and 'dec' not in cl: cm.setdefault('ra', c)
            elif 'dec' in cl: cm.setdefault('dec', c)
            elif 'e1' in cl: cm.setdefault('e1', c)
            elif 'e2' in cl: cm.setdefault('e2', c)
            elif 'weight' in cl: cm.setdefault('w', c)
            elif 'photoz_best' in cl or ('photo' in cl and 'z' in cl): cm.setdefault('zp', c)
        if not all(k in cm for k in ['ra','dec','e1','e2']): continue
        src = {k: df[cm[k]].values for k in ['ra','dec','e1','e2']}
        src['weight'] = df[cm['w']].values if 'w' in cm else np.ones(len(df))
        # photo-z: なければ z_s=0.8 (z_l~0.32に対する典型背景)
        src['z_photo'] = df[cm['zp']].values if 'zp' in cm else np.full(len(df), 0.8)
        ok = np.isfinite(src['ra'])&np.isfinite(src['e1'])
        for k in src: src[k] = src[k][ok]
        return src
    return None

# ================================================================
# メイン
# ================================================================
def main():
    print("="*70)
    print("cluster stack v2: Sigma_crit + miscentering")
    print("="*70)

    cands = load_clusters()
    if not cands:
        print("No cluster data. Run cluster_stack.py first."); return

    # N>=3, z>0.15 でフィルタ（cl2 z=0.11は低すぎるので除外）
    confirmed = [c for c in cands
                 if c.get("z_spec") and c["z_spec"] > 0.15
                 and c.get("n_spec_members", 0) >= 3]
    print(f"\nConfirmed clusters: {len(confirmed)}")
    for c in confirmed:
        print(f"  {c['id']}: z={c['z_spec']}, N={c.get('n_spec_members')}, σv={c.get('sigma_v')}")

    # ソース読み込み
    src_data = {}
    for c in confirmed:
        s = load_src(c["id"])
        if s:
            src_data[c["id"]] = s
            print(f"  {c['id']}: {len(s['ra'])} sources")

    used = [c for c in confirmed if c["id"] in src_data]
    print(f"\nClusters with data: {len(used)}")
    if not used: print("No data."); return

    # スタック
    z_mean = np.mean([c["z_spec"] for c in used])
    print(f"\n--- Stacked DeltaSigma (z_mean={z_mean:.3f}) ---")
    R, ds, dx, dse, nt = stacked_ds(used, src_data)

    v = np.isfinite(ds)&np.isfinite(dse)&(dse > 0)
    if np.sum(v) > 0:
        w = 1.0/dse[v]**2
        ds_m = np.sum(w*ds[v])/np.sum(w)
        dx_m = np.sum(w*dx[v])/np.sum(w)
        sn = ds_m/(1.0/np.sqrt(np.sum(w)))
        print(f"  <DS> = {ds_m:.2f} M_sun/pc^2")
        print(f"  <DS_x> = {dx_m:.2f} M_sun/pc^2")
        print(f"  S/N = {sn:.1f}")
        print(f"  sources: {np.sum(nt)}")

    print(f"\n  {'R[Mpc]':>8} {'DS':>10} {'err':>10} {'DS_x':>10} {'N':>6}")
    for i in range(len(R)):
        if np.isfinite(ds[i]):
            print(f"  {R[i]:>8.3f} {ds[i]:>10.2f} {dse[i]:>10.2f} {dx[i]:>10.2f} {nt[i]:>6}")

    # フィット
    print(f"\n--- Model fitting ---")
    print("  NFW centered..."); nfw_c = fit_nfw(R, ds, dse, z_mean, misc=False)
    if nfw_c: print(f"    M200={nfw_c['M200']:.2e}, c={nfw_c['c200']:.1f}, chi2/dof={nfw_c['chi2_dof']:.2f}")

    print("  NFW miscentered..."); nfw_m = fit_nfw(R, ds, dse, z_mean, misc=True)
    if nfw_m: print(f"    M200={nfw_m['M200']:.2e}, c={nfw_m['c200']:.1f}, R_misc={nfw_m['R_misc']:.3f}, chi2/dof={nfw_m['chi2_dof']:.2f}")

    print("  MOND (gc=a0)...")
    # デバッグ: Abel変換のテスト
    v_test = np.isfinite(ds) & np.isfinite(dse) & (dse > 0)
    Rv_test = R[v_test]
    print(f"    test Abel: R={Rv_test[:3]}, M=1e14, gc=a0")
    ds_test = mond_delta_sigma_abel(Rv_test[:3], 1e14, a0, z_mean)
    print(f"    result: {ds_test}")
    mond = fit_mond(R, ds, dse, z_mean, gc_fixed=a0)
    if mond: print(f"    M_bar={mond['M_bar']:.2e}, chi2/dof={mond['chi2_dof']:.2f}")
    else: print(f"    fit failed (returned None)")

    print("  Membrane (gc free)..."); mem = fit_mond(R, ds, dse, z_mean)
    if mem: print(f"    M_bar={mem['M_bar']:.2e}, gc/a0={mem['gc_a0']:.3f}, chi2/dof={mem['chi2_dof']:.2f}")

    # 比較テーブル
    models = {}
    if nfw_c: models["NFW"] = nfw_c
    if nfw_m: models["NFW+misc"] = nfw_m
    if mond: models["MOND"] = mond
    if mem: models["membrane"] = mem

    if models:
        aic_ref = min(m["AIC"] for m in models.values())
        print(f"\n{'='*70}")
        print(f"  {'model':<16} {'chi2/dof':>9} {'AIC':>8} {'dAIC':>8}")
        print(f"  {'-'*44}")
        for nm, m in models.items():
            da = m["AIC"]-aic_ref
            print(f"  {nm:<16} {m['chi2_dof']:>9.2f} {m['AIC']:>8.1f} {da:>+8.1f}")

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Stacked WL v2: {len(used)} clusters, $\\Sigma_{{crit}}$ normalized",
                     fontsize=13, fontweight='bold')

        ax = axes[0,0]
        vv = np.isfinite(ds)
        ax.errorbar(R[vv], ds[vv], dse[vv], fmt='ko', ms=6, capsize=4, label='data', zorder=5)
        Rf = np.logspace(np.log10(0.08), np.log10(6), 80)
        if nfw_c:
            ax.plot(Rf, nfw_delta_sigma(Rf, nfw_c["M200"], nfw_c["c200"], z_mean),
                   'b-', lw=2, label=f'NFW c={nfw_c["c200"]:.1f}')
        if nfw_m:
            ax.plot(Rf, nfw_ds_miscentered(Rf, nfw_m["M200"], nfw_m["c200"], z_mean, nfw_m["R_misc"]),
                   'b--', lw=1.5, label=f'NFW+misc R={nfw_m["R_misc"]:.2f}')
        if mond:
            ax.plot(Rf, mond_delta_sigma_abel(Rf, mond["M_bar"], a0, z_mean),
                   'r-', lw=2, label='MOND')
        if mem:
            ax.plot(Rf, mond_delta_sigma_abel(Rf, mem["M_bar"], mem["gc"], z_mean),
                   'g--', lw=2, label=f'membrane gc={mem["gc_a0"]:.2f}a0')
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]')
        ax.set_xscale('log'); ax.set_title(f'DeltaSigma (S/N={sn:.1f})')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0,1]
        vx = np.isfinite(dx)
        ax.errorbar(R[vx], dx[vx], dse[vx], fmt='ro', ms=6, capsize=4)
        ax.axhline(0, color='green', ls='--', lw=2)
        ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\Delta\Sigma_\times$')
        ax.set_xscale('log'); ax.set_title('Cross (null test)')
        ax.grid(True, alpha=0.3)

        ax = axes[1,0]
        ax.bar(range(len(nt)), nt, color='steelblue', edgecolor='white')
        ax.set_xlabel('bin'); ax.set_ylabel('N')
        ax.set_title(f'Sources (total={np.sum(nt)})'); ax.grid(True, alpha=0.3)

        ax = axes[1,1]
        if models:
            nms = list(models.keys())
            daics = [models[n]["AIC"]-aic_ref for n in nms]
            bc = ['green' if d<=0 else 'steelblue' if d<5 else 'coral' for d in daics]
            bars = ax.barh(range(len(nms)), daics, color=bc, edgecolor='white', height=0.6)
            ax.set_yticks(range(len(nms))); ax.set_yticklabels(nms, fontsize=9)
            ax.set_xlabel('dAIC'); ax.set_title('Model comparison')
            ax.axvline(0, color='k', lw=0.5)
            for b, d in zip(bars, daics):
                ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                       f'{d:+.1f}', va='center', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = OUTDIR/"cluster_stack_v2.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except ImportError: pass

    # 保存
    result = {"v":"v2","n_cl":len(used),"z_mean":float(z_mean),
              "sn":float(sn) if 'sn' in dir() else None, "models":models,
              "clusters":[{"id":c["id"],"z":c["z_spec"]} for c in used]}
    with open(OUTDIR/"cluster_stack_v2_results.json","w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"results: {OUTDIR/'cluster_stack_v2_results.json'}")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
