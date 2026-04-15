# -*- coding: utf-8 -*-
"""
スタック弱レンズ v2b: 内側カット R > 0.5 Mpc で公平比較
=============================================================
uv run --with scipy --with matplotlib --with numpy --with pandas python cluster_stack_v2b.py
"""

import numpy as np
import json, sys, io
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTDIR = Path("cluster_stack"); OUTDIR.mkdir(exist_ok=True)

H0=70.0; OMEGA_M=0.3; C_LIGHT=2.998e5; G_SI=6.674e-11
M_SUN=1.989e30; MPC_M=3.0857e22; PC_M=3.0857e16; a0=1.2e-10
R_MIN_FIT = 0.5  # Mpc

def E_z(z): return np.sqrt(OMEGA_M*(1+z)**3+(1-OMEGA_M))
def comoving_distance(z):
    r, _ = quad(lambda zp: 1.0/E_z(zp), 0, z); return C_LIGHT/H0*r
def angular_diameter_distance(z): return comoving_distance(z)/(1+z)
def D_ls_func(z_l, z_s):
    return (comoving_distance(z_s)-comoving_distance(z_l))/(1+z_s)
def rho_crit(z):
    Hz = H0*E_z(z)*1e3/MPC_M; return 3*Hz**2/(8*np.pi*G_SI)/M_SUN*MPC_M**3
def sigma_crit(z_l, z_s):
    Dl = angular_diameter_distance(z_l)*MPC_M
    Ds = angular_diameter_distance(z_s)*MPC_M
    Dls = D_ls_func(z_l, z_s)*MPC_M
    if Dls <= 0 or Ds <= 0 or Dl <= 0: return np.inf
    return (C_LIGHT*1e3)**2/(4*np.pi*G_SI)*Ds/(Dl*Dls)/M_SUN*PC_M**2

# NFW DeltaSigma (Wright & Brainerd 2000)
def nfw_delta_sigma(R_mpc, M200, c200, z_l):
    rc = rho_crit(z_l); r200 = (3*M200/(4*np.pi*200*rc))**(1./3.)
    rs = r200/c200; dc = (200./3.)*c200**3/(np.log(1+c200)-c200/(1+c200))
    rho_s = dc*rc; x = np.atleast_1d(R_mpc/rs).astype(float)
    sig = np.zeros_like(x); bsig = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 1e-6: continue
        if abs(xi-1)<1e-4: sig[i] = 2*rho_s*rs/3.
        elif xi<1: sig[i] = 2*rho_s*rs/(xi**2-1)*(1-np.arccosh(1/xi)/np.sqrt(1-xi**2))
        else: sig[i] = 2*rho_s*rs/(xi**2-1)*(1-np.arccos(1/xi)/np.sqrt(xi**2-1))
        if abs(xi-1)<1e-4: h = 1+np.log(0.5)
        elif xi<1: h = np.arccosh(1/xi)/np.sqrt(1-xi**2)+np.log(xi/2)
        else: h = np.arccos(1/xi)/np.sqrt(xi**2-1)+np.log(xi/2)
        bsig[i] = 4*rho_s*rs*h/xi**2
    return (bsig-sig)*1e-12

def nfw_ds_misc(R_mpc, M200, c200, z_l, Rm):
    if Rm < 1e-4: return nfw_delta_sigma(R_mpc, M200, c200, z_l)
    R = np.atleast_1d(R_mpc); ds_c = nfw_delta_sigma(R, M200, c200, z_l)
    phis = np.linspace(0, 2*np.pi, 36, endpoint=False)
    ds_m = np.zeros_like(R)
    for i, Ri in enumerate(R):
        s = 0.0
        for phi in phis:
            Re = max(np.sqrt(Ri**2+Rm**2-2*Ri*Rm*np.cos(phi)), 1e-4)
            s += nfw_delta_sigma(np.array([Re]), M200, c200, z_l)[0]
        ds_m[i] = s/36
    f = np.exp(-0.5*(Rm/np.maximum(R, 1e-4))**2)
    return f*ds_c + (1-f)*ds_m

# MOND/膜 Abel変換
def mond_ds_abel(R_mpc, M_bar, gc, z_l):
    cb = 5.0; rc = rho_crit(z_l); r200 = (3*M_bar/(4*np.pi*200*rc))**(1./3.)
    rs = r200/cb; norm = np.log(1+cb)-cb/(1+cb)
    R = np.atleast_1d(R_mpc); sig = np.zeros_like(R)
    for i, Ri in enumerate(R):
        if Ri < 1e-4: continue
        zs = np.linspace(0, 10*r200, 80); integ = np.zeros(80)
        for j, zj in enumerate(zs):
            r3d = np.sqrt(Ri**2+zj**2)
            if r3d < 1e-6: continue
            Me = M_bar*(np.log(1+r3d/rs)-(r3d/rs)/(1+r3d/rs))/norm
            gN = G_SI*Me*M_SUN/(r3d*MPC_M)**2
            gobs = (gN+np.sqrt(gN**2+4*gc*abs(gN)))/2
            integ[j] = (gobs-gN)/(4*np.pi*G_SI)/(r3d*MPC_M)
        integ_m = integ/M_SUN*MPC_M**3
        sig[i] = 2*np.trapezoid(integ_m, zs)  # M_sun/Mpc^2
    bsig = np.zeros_like(R)
    for i, Ri in enumerate(R):
        if Ri < 1e-4 or i == 0: bsig[i] = sig[i]; continue
        Rs = R[:i+1]; Ss = sig[:i+1]; m = Rs <= Ri
        if np.sum(m) < 2: bsig[i] = sig[i]; continue
        bsig[i] = 2/Ri**2*np.trapezoid(Ss[m]*Rs[m], Rs[m])
    return (bsig-sig)*1e-12

# スタック DeltaSigma
def stacked_ds(clusters, src_data, nb=12):
    bins = np.logspace(np.log10(0.1), np.log10(5.0), nb+1)
    ds_s = np.zeros(nb); dx_s = np.zeros(nb)
    w_s = np.zeros(nb); w2_s = np.zeros(nb)
    nt = np.zeros(nb, dtype=int)
    for c in clusters:
        if c["id"] not in src_data: continue
        src = src_data[c["id"]]; z_l = c["z_spec"]
        Dl = angular_diameter_distance(z_l); ra_c, dec_c = c["ra"], c["dec"]
        dra = (src["ra"]-ra_c)*np.cos(np.radians(dec_c))*np.pi/180*Dl
        ddec = (src["dec"]-dec_c)*np.pi/180*Dl
        Rm = np.sqrt(dra**2+ddec**2)
        phi = np.arctan2(ddec, dra)
        c2 = np.cos(2*phi); s2 = np.sin(2*phi)
        et = -(src["e1"]*c2+src["e2"]*s2); ex = src["e1"]*s2-src["e2"]*c2
        sc = np.array([sigma_crit(z_l, zs) for zs in src["z_photo"]])
        ok = np.isfinite(sc)&(sc > 0)&(sc < 1e6)
        ds_i = et*sc; dx_i = ex*sc
        ws = src.get("weight", np.ones(len(src["ra"])))
        wi = ws/sc**2; wi[~ok] = 0
        for b in range(nb):
            m = (Rm >= bins[b])&(Rm < bins[b+1])&ok&(wi > 0)
            n = np.sum(m)
            if n == 0: continue
            nt[b] += n; wm = wi[m]
            ds_s[b] += np.sum(wm*ds_i[m]); dx_s[b] += np.sum(wm*dx_i[m])
            w_s[b] += np.sum(wm); w2_s[b] += np.sum(wm**2)
    Rmid = np.sqrt(bins[:-1]*bins[1:])
    ds_st = np.where(w_s > 0, ds_s/w_s, np.nan)
    dx_st = np.where(w_s > 0, dx_s/w_s, np.nan)
    neff = np.where(w2_s > 0, w_s**2/w2_s, 0)
    msc = np.where(w_s > 0, np.sqrt(neff/w_s), 1e4)
    ds_err = 0.25*msc/np.sqrt(np.maximum(neff, 1))
    return Rmid, ds_st, dx_st, ds_err, nt

# データ読み込み
def load_data():
    import pandas as pd
    p = OUTDIR/"cluster_stack_results.json"
    if not p.exists(): return None, {}
    with open(p) as f: d = json.load(f)
    cands = d.get("relaxed", [])
    confirmed = [c for c in cands if c.get("z_spec") and c["z_spec"] > 0.15
                 and c.get("n_spec_members", 0) >= 3]
    src_data = {}
    for c in confirmed:
        cid = c["id"]
        for pat in [f"{cid}_sources_photoz.csv", f"{cid}_sources.csv"]:
            p2 = Path(pat)
            if not p2.exists(): continue
            with open(p2) as fh: first = fh.readline()
            if first.startswith('# '):
                cols = first[2:].strip().split(',')
                df = pd.read_csv(p2, skiprows=1, header=None, names=cols)
            else:
                df = pd.read_csv(p2, comment='#')
            cm = {}
            for col in df.columns:
                cl = col.lower()
                if 'ra' in cl and 'dec' not in cl: cm.setdefault('ra', col)
                elif 'dec' in cl: cm.setdefault('dec', col)
                elif 'e1' in cl: cm.setdefault('e1', col)
                elif 'e2' in cl: cm.setdefault('e2', col)
                elif 'weight' in cl: cm.setdefault('w', col)
                elif 'photoz_best' in cl or ('photo' in cl and 'z' in cl): cm.setdefault('zp', col)
            if not all(k in cm for k in ['ra','dec','e1','e2']): continue
            src = {k: df[cm[k]].values for k in ['ra','dec','e1','e2']}
            src['weight'] = df[cm['w']].values if 'w' in cm else np.ones(len(df))
            src['z_photo'] = df[cm['zp']].values if 'zp' in cm else np.full(len(df), 0.8)
            ok = np.isfinite(src['ra'])&np.isfinite(src['e1'])&(src['z_photo'] > c['z_spec']+0.1)
            for k in src: src[k] = src[k][ok]
            src_data[cid] = src
            print(f"  {cid}: {len(src['ra'])} sources ({pat})")
            break
    return confirmed, src_data

# 感度テスト
def sensitivity(R, ds, err, z):
    print(f"\n--- R_min sensitivity ---")
    results = []
    for Rc in [0.3, 0.4, 0.5, 0.7, 1.0, 1.5]:
        v = (R >= Rc)&np.isfinite(ds)&np.isfinite(err)&(err > 0)
        n = np.sum(v)
        if n < 3: continue
        Rv, Dv, Ev = R[v], ds[v], err[v]
        # NFW
        bc_n = 1e10
        for lm in np.linspace(13.5,15.5,8):
            for cc in [4,8,12]:
                try:
                    def fn(p):
                        M=10**p[0];c=p[1]
                        if c<1 or c>20 or M<1e12 or M>1e16: return 1e10
                        return np.sum(((Dv-nfw_delta_sigma(Rv,M,c,z))/Ev)**2)
                    r=minimize(fn,[lm,cc],method='Nelder-Mead',options={'maxiter':300})
                    if r.fun<bc_n: bc_n=r.fun
                except: pass
        # MOND
        bc_m = 1e10
        for lm in np.linspace(12,15,12):
            try:
                def fm(p):
                    M=10**p[0]
                    if M<1e11 or M>1e15: return 1e10
                    m=mond_ds_abel(Rv,M,a0,z)
                    if np.any(~np.isfinite(m)): return 1e10
                    return np.sum(((Dv-m)/Ev)**2)
                r=minimize(fm,[lm],method='Nelder-Mead',options={'maxiter':200})
                if r.fun<bc_m: bc_m=r.fun
            except: pass
        da = (bc_m+2)-(bc_n+4)
        fav = "MOND" if da < -2 else "NFW" if da > 2 else "~"
        results.append({"Rc":Rc,"n":int(n),"c2n":float(bc_n),"c2m":float(bc_m),"dAIC":float(da)})
        print(f"  R>{Rc:.1f} ({n} bins): dAIC(MOND-NFW)={da:+.1f} -> {fav}")
    return results

# メイン
def main():
    print("="*70)
    print("cluster stack v2b: R > 0.5 Mpc inner cut")
    print("="*70)

    confirmed, src_data = load_data()
    if not confirmed: print("No clusters."); return
    used = [c for c in confirmed if c["id"] in src_data]
    print(f"\nClusters with data: {len(used)}")
    if not used: return

    z_mean = np.mean([c["z_spec"] for c in used])
    R, ds, dx, dse, nt = stacked_ds(used, src_data)

    # プロファイル
    print(f"\n  {'R':>6} {'DS':>8} {'err':>8} {'DSx':>8} {'N':>6} {'fit':>4}")
    for i in range(len(R)):
        if np.isfinite(ds[i]):
            f = "***" if R[i] >= R_MIN_FIT else ""
            print(f"  {R[i]:>6.3f} {ds[i]:>8.1f} {dse[i]:>8.1f} {dx[i]:>8.1f} {nt[i]:>6} {f:>4}")

    # S/N
    v = np.isfinite(ds)&np.isfinite(dse)&(dse > 0)
    w = 1.0/dse[v]**2; ds_m = np.sum(w*ds[v])/np.sum(w)
    sn = ds_m/(1.0/np.sqrt(np.sum(w)))
    print(f"\n  全体 S/N={sn:.1f}")

    # フィット (R > R_MIN_FIT)
    print(f"\n{'='*70}")
    print(f"Fitting R > {R_MIN_FIT} Mpc")
    print(f"{'='*70}")

    cut = (R >= R_MIN_FIT)&np.isfinite(ds)&np.isfinite(dse)&(dse > 0)
    nv = np.sum(cut); Rv = R[cut]; Dv = ds[cut]; Ev = dse[cut]
    print(f"  {nv} bins, R=[{Rv[0]:.2f},{Rv[-1]:.2f}]")

    models = {}

    # NFW
    print("  NFW...", end="", flush=True)
    bc, bp = 1e10, None
    for lm in np.linspace(13.5,15.5,12):
        for cc in [2,4,6,8,10,15]:
            try:
                def fn(p):
                    M=10**p[0];c=p[1]
                    if c<1 or c>20 or M<1e12 or M>1e16: return 1e10
                    return np.sum(((Dv-nfw_delta_sigma(Rv,M,c,z_mean))/Ev)**2)
                r=minimize(fn,[lm,cc],method='Nelder-Mead',options={'maxiter':500})
                if r.fun<bc: bc=r.fun; bp=r.x
            except: pass
    if bp is not None:
        models["NFW"]={"M200":float(10**bp[0]),"c200":float(bp[1]),
            "chi2":float(bc),"dof":int(nv-2),"chi2_dof":float(bc/max(nv-2,1)),"AIC":float(bc+4)}
        print(f" M200={models['NFW']['M200']:.2e}, c={models['NFW']['c200']:.1f}, chi2/dof={models['NFW']['chi2_dof']:.2f}")
    else: print(" fail")

    # NFW+misc
    print("  NFW+misc...", end="", flush=True)
    bc2, bp2 = 1e10, None
    for lm in np.linspace(13.5,15.5,8):
        for cc in [3,5,8,12]:
            for lr in [-2,-1.5,-1,-0.5]:
                try:
                    def fn2(p):
                        M=10**p[0];c=p[1];rm=10**p[2]
                        if c<1 or c>20 or M<1e12 or M>1e16 or rm>2: return 1e10
                        return np.sum(((Dv-nfw_ds_misc(Rv,M,c,z_mean,rm))/Ev)**2)
                    r=minimize(fn2,[lm,cc,lr],method='Nelder-Mead',options={'maxiter':500})
                    if r.fun<bc2: bc2=r.fun; bp2=r.x
                except: pass
    if bp2 is not None:
        models["NFW+misc"]={"M200":float(10**bp2[0]),"c200":float(bp2[1]),"R_misc":float(10**bp2[2]),
            "chi2":float(bc2),"dof":int(nv-3),"chi2_dof":float(bc2/max(nv-3,1)),"AIC":float(bc2+6)}
        print(f" M200={models['NFW+misc']['M200']:.2e}, c={models['NFW+misc']['c200']:.1f}, Rm={models['NFW+misc']['R_misc']:.3f}, chi2/dof={models['NFW+misc']['chi2_dof']:.2f}")
    else: print(" fail")

    # MOND
    print("  MOND...", end="", flush=True)
    bc3, bp3 = 1e10, None
    for lm in np.linspace(12,15,20):
        try:
            def fm(p):
                M=10**p[0]
                if M<1e11 or M>1e15: return 1e10
                m=mond_ds_abel(Rv,M,a0,z_mean)
                if np.any(~np.isfinite(m)): return 1e10
                return np.sum(((Dv-m)/Ev)**2)
            r=minimize(fm,[lm],method='Nelder-Mead',options={'maxiter':300})
            if r.fun<bc3: bc3=r.fun; bp3=r.x
        except: pass
    if bp3 is not None:
        models["MOND"]={"M_bar":float(10**bp3[0]),"gc":float(a0),"gc_a0":1.0,
            "chi2":float(bc3),"dof":int(nv-1),"chi2_dof":float(bc3/max(nv-1,1)),"AIC":float(bc3+2)}
        print(f" M_bar={models['MOND']['M_bar']:.2e}, chi2/dof={models['MOND']['chi2_dof']:.2f}")
    else: print(" fail")

    # 膜
    print("  membrane...", end="", flush=True)
    bc4, bp4 = 1e10, None
    for lm in np.linspace(12,15,10):
        for lgc in np.linspace(-11,-9,10):
            try:
                def fe(p):
                    M=10**p[0];gc=10**p[1]
                    if M<1e11 or M>1e15: return 1e10
                    m=mond_ds_abel(Rv,M,gc,z_mean)
                    if np.any(~np.isfinite(m)): return 1e10
                    return np.sum(((Dv-m)/Ev)**2)
                r=minimize(fe,[lm,lgc],method='Nelder-Mead',options={'maxiter':300})
                if r.fun<bc4: bc4=r.fun; bp4=r.x
            except: pass
    if bp4 is not None:
        gc_v = 10**bp4[1]
        models["membrane"]={"M_bar":float(10**bp4[0]),"gc":float(gc_v),"gc_a0":float(gc_v/a0),
            "chi2":float(bc4),"dof":int(nv-2),"chi2_dof":float(bc4/max(nv-2,1)),"AIC":float(bc4+4)}
        print(f" M_bar={models['membrane']['M_bar']:.2e}, gc/a0={models['membrane']['gc_a0']:.3f}, chi2/dof={models['membrane']['chi2_dof']:.2f}")
    else: print(" fail")

    # 比較テーブル
    if models:
        aic_min = min(m["AIC"] for m in models.values())
        print(f"\n{'='*70}")
        print(f"  {'model':<16} {'k':>3} {'chi2/dof':>9} {'AIC':>8} {'dAIC':>8}")
        print(f"  {'-'*48}")
        for nm in ["NFW","NFW+misc","MOND","membrane"]:
            if nm not in models: continue
            m = models[nm]; da = m["AIC"]-aic_min
            k = 2 if nm in ["NFW","membrane"] else 3 if nm=="NFW+misc" else 1
            print(f"  {nm:<16} {k:>3} {m['chi2_dof']:>9.2f} {m['AIC']:>8.1f} {da:>+8.1f}")

    # 感度テスト
    scan = sensitivity(R, ds, dse, z_mean)

    # プロット
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Stacked WL v2b: R>{R_MIN_FIT} Mpc, {len(used)} clusters, z~{z_mean:.3f}",
                     fontsize=13, fontweight='bold')

        # (a) Profile + models
        ax = axes[0,0]
        va = np.isfinite(ds); vf = va&(R >= R_MIN_FIT)
        ax.errorbar(R[va&~vf], ds[va&~vf], dse[va&~vf], fmt='o', color='gray', ms=5,
                    capsize=3, alpha=0.5, label=f'R<{R_MIN_FIT}')
        ax.errorbar(R[vf], ds[vf], dse[vf], fmt='ko', ms=6, capsize=4, label=f'R>{R_MIN_FIT} (fit)')
        Rf = np.logspace(np.log10(0.08), np.log10(6), 80)
        if "NFW" in models:
            m = models["NFW"]
            ax.plot(Rf, nfw_delta_sigma(Rf, m["M200"], m["c200"], z_mean), 'b-', lw=2,
                    label=f'NFW c={m["c200"]:.1f}')
        if "NFW+misc" in models:
            m = models["NFW+misc"]
            ax.plot(Rf, nfw_ds_misc(Rf, m["M200"], m["c200"], z_mean, m["R_misc"]), 'b--', lw=1.5,
                    label=f'NFW+misc Rm={m["R_misc"]:.2f}')
        if "MOND" in models:
            m = models["MOND"]
            ax.plot(Rf, mond_ds_abel(Rf, m["M_bar"], a0, z_mean), 'r-', lw=2, label='MOND')
        if "membrane" in models:
            m = models["membrane"]
            ax.plot(Rf, mond_ds_abel(Rf, m["M_bar"], m["gc"], z_mean), 'g--', lw=2,
                    label=f'membrane gc={m["gc_a0"]:.2f}a0')
        ax.axvline(R_MIN_FIT, color='orange', ls=':', lw=2, alpha=0.7)
        ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$/pc$^2$]')
        ax.set_xscale('log'); ax.set_title(f'S/N={sn:.1f}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (b) dAIC vs R_min
        ax = axes[0,1]
        if scan:
            rcs = [s["Rc"] for s in scan]; das = [s["dAIC"] for s in scan]
            ax.plot(rcs, das, 'o-', color='steelblue', ms=8, lw=2)
            ax.axhline(0, color='k', lw=0.5)
            ax.axhline(-2, color='green', ls='--', alpha=0.5, label='MOND pref')
            ax.axhline(2, color='red', ls='--', alpha=0.5, label='NFW pref')
            ax.fill_between([0.2,1.6], -2, 2, alpha=0.08, color='gray')
            ax.set_xlabel('R_min [Mpc]'); ax.set_ylabel('dAIC (MOND-NFW)')
            ax.set_title('Sensitivity'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (c) AIC bar
        ax = axes[1,0]
        if models:
            nms = list(models.keys())
            das_bar = [models[n]["AIC"]-aic_min for n in nms]
            bc_bar = ['green' if d<=0 else 'steelblue' if d<5 else 'coral' for d in das_bar]
            bars = ax.barh(range(len(nms)), das_bar, color=bc_bar, edgecolor='white', height=0.6)
            ax.set_yticks(range(len(nms))); ax.set_yticklabels(nms, fontsize=10)
            ax.set_xlabel('dAIC'); ax.set_title(f'R>{R_MIN_FIT} Mpc')
            ax.axvline(0, color='k', lw=0.5)
            for b, d in zip(bars, das_bar):
                ax.text(max(b.get_width(),0)+0.3, b.get_y()+b.get_height()/2,
                       f'{d:+.1f}', va='center', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # (d) Cross component
        ax = axes[1,1]
        vx = np.isfinite(dx)
        ax.errorbar(R[vx], dx[vx], dse[vx], fmt='ro', ms=6, capsize=4)
        ax.axhline(0, color='green', ls='--', lw=2)
        ax.set_xlabel('R [Mpc]'); ax.set_ylabel(r'$\Delta\Sigma_\times$')
        ax.set_xscale('log'); ax.set_title('Cross (null test)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = OUTDIR/"cluster_stack_v2b.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nplot: {fp}")
    except ImportError: pass

    # 保存
    out = {"v":"v2b","R_min":R_MIN_FIT,"n_cl":len(used),"z_mean":float(z_mean),
           "sn":float(sn),"models":models,"scan":scan}
    with open(OUTDIR/"cluster_stack_v2b_results.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"results: {OUTDIR/'cluster_stack_v2b_results.json'}")

    # 判定
    print(f"\n{'='*70}")
    print("verdict")
    print(f"{'='*70}")
    if "MOND" in models and "NFW" in models:
        da = models["MOND"]["AIC"]-models["NFW"]["AIC"]
        if da < -2: print(f"  MOND > NFW (dAIC={da:.1f})")
        elif da > 2: print(f"  NFW > MOND (dAIC={da:.1f})")
        else: print(f"  inconclusive (dAIC={da:.1f})")
    if "membrane" in models and "NFW" in models:
        da2 = models["membrane"]["AIC"]-models["NFW"]["AIC"]
        print(f"  membrane vs NFW: dAIC={da2:.1f}")
    print(f"\n{'='*70}\ndone\n{'='*70}")

if __name__ == "__main__":
    main()
