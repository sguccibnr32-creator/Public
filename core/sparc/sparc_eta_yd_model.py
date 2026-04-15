#!/usr/bin/env python3
"""
sparc_eta_yd_model.py (TA3+phase1+MRT adapted)
Non-circular eta(Yd) model for Condition 15.
"""
import os, csv, warnings
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.0857e19

def load_pipeline():
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try: data[n] = {'vflat': float(row.get('vflat', '0')), 'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                g = float(row.get('gc_over_a0', '0'))
                if n in data and g > 0: data[n]['gc'] = g * a0
            except: pass
    return {k: v for k, v in data.items() if 'gc' in v and v['vflat'] > 0}

def parse_mrt():
    data = {}; in_data = False; sep = 0
    with open(MRT, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4: in_data = True
                continue
            if not in_data: continue
            p = line.split()
            if len(p) < 18: continue
            try:
                data[p[0]] = {'T': int(p[1]), 'L': float(p[7]), 'Rdisk': float(p[11]),
                              'SBdisk0': float(p[12]), 'MHI': float(p[13]),
                              'Inc': float(p[5]), 'Q': int(p[17])}
            except: continue
    return data

def partial_corr(x, y, z):
    if z.ndim == 1: z = z.reshape(-1, 1)
    Z = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(x - Z@bx, y - Z@by)

def main():
    print("=" * 70)
    print("eta(Yd) non-circular model & Condition 15")
    print("=" * 70)

    pipe = load_pipeline()
    mrt_d = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt_d)}")

    # Use MRT Rdisk for hR
    names, gc_a, vf_a, hR_a, Yd_a = [], [], [], [], []
    for n in sorted(pipe.keys()):
        gd = pipe[n]; gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vf <= 0: continue
        m = mrt_d.get(n, {})
        hR = m.get('Rdisk', 0)
        if hR <= 0: continue
        names.append(n); gc_a.append(gc); vf_a.append(vf); hR_a.append(hR); Yd_a.append(Yd)

    gc = np.array(gc_a); vf = np.array(vf_a); hR = np.array(hR_a); Yd = np.array(Yd_a)
    N = len(gc)
    log_gc = np.log10(gc); log_vf = np.log10(vf); log_hR = np.log10(hR)
    log_Yd = np.log10(Yd)
    Sd = (vf*1e3)**2 / (hR*kpc_m); log_Sd = np.log10(Sd)
    log_a0Sd = np.log10(a0 * Sd)
    SS_tot = np.sum((log_gc - log_gc.mean())**2)
    print(f"N = {N}")

    # T1: Yd independence
    print("\n" + "=" * 70)
    print("T1: Yd independence check")
    print("=" * 70)
    r_yg, p_yg = pearsonr(log_Yd, log_gc)
    r_ys, p_ys = pearsonr(log_Yd, log_Sd)
    r_yv, p_yv = pearsonr(log_Yd, log_vf)
    r_yh, p_yh = pearsonr(log_Yd, log_hR)
    print(f"  r(Yd, gc)   = {r_yg:+.3f} (p={p_yg:.2e})")
    print(f"  r(Yd, Sd)   = {r_ys:+.3f} (p={p_ys:.2e})")
    print(f"  r(Yd, vflat)= {r_yv:+.3f} (p={p_yv:.2e})")
    print(f"  r(Yd, hR)   = {r_yh:+.3f} (p={p_yh:.2e})")
    rp, pp = partial_corr(log_Yd, log_gc, log_Sd)
    rp2, pp2 = partial_corr(log_Yd, log_gc, np.column_stack([log_vf, log_hR]))
    print(f"  partial r(Yd, gc | Sd)      = {rp:+.3f} (p={pp:.2e})")
    print(f"  partial r(Yd, gc | vflat,hR)= {rp2:+.3f} (p={pp2:.2e})")

    # T2: baseline vs eta(Yd)
    print("\n" + "=" * 70)
    print("T2: Model comparison")
    print("=" * 70)
    # M0: baseline
    sl0, ic0, r0, _, se0 = linregress(log_Sd, log_gc)
    pred0 = ic0 + sl0*log_Sd; resid0 = log_gc - pred0
    scat0 = np.std(resid0); R2_0 = r0**2
    AIC_0 = N*np.log(np.sum(resid0**2)/N) + 2*2
    print(f"\n  M0 (Sd only): alpha={sl0:.4f}+/-{se0:.4f}, eta={10**ic0:.4f}")
    print(f"    R2={R2_0:.4f}, scatter={scat0:.4f}, AIC={AIC_0:.1f}")

    # M1: Sd + Yd
    X1 = np.column_stack([np.ones(N), log_a0Sd, log_Yd])
    b1 = np.linalg.lstsq(X1, log_gc, rcond=None)[0]
    pred1 = X1 @ b1; resid1 = log_gc - pred1
    scat1 = np.std(resid1); R2_1 = 1 - np.sum(resid1**2)/SS_tot
    AIC_1 = N*np.log(np.sum(resid1**2)/N) + 2*3
    print(f"\n  M1 (+Yd): alpha={b1[1]:.4f}, beta_Yd={b1[2]:.4f}, eta0={10**b1[0]:.4f}")
    print(f"    R2={R2_1:.4f} (dR2={R2_1-R2_0:+.4f}), scatter={scat1:.4f}, AIC={AIC_1:.1f} (dAIC={AIC_1-AIC_0:+.1f})")

    # M1b: vflat + hR + Yd
    X1b = np.column_stack([np.ones(N), log_vf, log_hR, log_Yd])
    b1b = np.linalg.lstsq(X1b, log_gc, rcond=None)[0]
    pred1b = X1b @ b1b; resid1b = log_gc - pred1b
    scat1b = np.std(resid1b); R2_1b = 1 - np.sum(resid1b**2)/SS_tot
    AIC_1b = N*np.log(np.sum(resid1b**2)/N) + 2*4
    print(f"\n  M1b (vf+hR+Yd): b_vf={b1b[1]:.3f}, b_hR={b1b[2]:.3f}, b_Yd={b1b[3]:.3f}")
    print(f"    R2={R2_1b:.4f}, scatter={scat1b:.4f}, AIC={AIC_1b:.1f}")
    print(f"    b_vf/(-b_hR)={b1b[1]/(-b1b[2]):.3f} (theory 2.0)")

    # T3: + T-type
    print("\n" + "=" * 70)
    print("T3: +T-type")
    print("=" * 70)
    T_arr = np.array([mrt_d.get(n, {}).get('T', np.nan) for n in names])
    vT = np.isfinite(T_arr)
    if vT.sum() > 30:
        Nt = vT.sum()
        ti = np.where(vT)[0]; tt = T_arr[ti]
        lg_t = log_gc[ti]; ls_t = log_a0Sd[ti]; ly_t = log_Yd[ti]
        SS_t = np.sum((lg_t - lg_t.mean())**2)
        # baseline
        s0, i0, r0t, _, _ = linregress(log_Sd[ti], lg_t)
        R2_t0 = r0t**2
        # +Yd
        Xy = np.column_stack([np.ones(Nt), ls_t, ly_t])
        by = np.linalg.lstsq(Xy, lg_t, rcond=None)[0]
        R2_ty = 1 - np.sum((lg_t - Xy@by)**2)/SS_t
        # +T
        Xt = np.column_stack([np.ones(Nt), log_Sd[ti], tt])
        bt = np.linalg.lstsq(Xt, lg_t, rcond=None)[0]
        R2_tt = 1 - np.sum((lg_t - Xt@bt)**2)/SS_t
        # +Yd+T
        Xyt = np.column_stack([np.ones(Nt), ls_t, ly_t, tt])
        byt = np.linalg.lstsq(Xyt, lg_t, rcond=None)[0]
        ryt = lg_t - Xyt@byt
        R2_yt = 1 - np.sum(ryt**2)/SS_t
        print(f"  N_T={Nt}")
        print(f"  baseline:  R2={R2_t0:.3f}")
        print(f"  +Yd:       R2={R2_ty:.3f} (beta_Yd={by[2]:+.3f})")
        print(f"  +T:        R2={R2_tt:.3f} (b_T={bt[2]:+.4f})")
        print(f"  +Yd+T:     R2={R2_yt:.3f} (beta_Yd={byt[2]:+.3f}, b_T={byt[3]:+.4f})")

    # T4: eta(Yd) interpretation
    print("\n" + "=" * 70)
    print("T4: eta(Yd) interpretation")
    print("=" * 70)
    alpha_fit = b1[1]
    log_eta_pg = log_gc - alpha_fit * log_a0Sd
    sl_ey, ic_ey, r_ey, p_ey, se_ey = linregress(log_Yd, log_eta_pg)
    eta_corr = log_eta_pg - (ic_ey + sl_ey*log_Yd)
    print(f"  eta ~ Yd^({sl_ey:.3f}+/-{se_ey:.3f}), r={r_ey:.3f}")
    print(f"  eta scatter: {np.std(log_eta_pg):.4f} -> {np.std(eta_corr):.4f} dex")
    print(f"  Yd explains {(1 - np.var(eta_corr)/np.var(log_eta_pg))*100:.1f}% of eta variance")

    for prop in ['T', 'L', 'SBdisk0', 'Inc']:
        vals, yd_sub = [], []
        for n in names:
            if n in mrt_d and prop in mrt_d[n]:
                v = mrt_d[n][prop]
                if np.isfinite(v) and (prop not in ('L', 'SBdisk0') or v > 0):
                    vals.append(np.log10(v) if prop in ('L', 'SBdisk0') else v)
                    yd_sub.append(np.log10(pipe[n]['Yd']))
        if len(vals) > 20:
            rho, p = spearmanr(vals, yd_sub)
            print(f"  r(Yd, {prop:8s}) = {rho:+.3f} (p={p:.2e})")

    print(f"\n  beta_Yd = {b1[2]:.3f}: {'gc decreases with Yd (membrane suppressed by heavy baryons)' if b1[2] < 0 else 'gc increases with Yd'}")

    # T5: LOO cross-validation
    print("\n" + "=" * 70)
    print("T5: Leave-One-Out CV")
    print("=" * 70)
    pred_loo_0, pred_loo_1 = np.zeros(N), np.zeros(N)
    for i in range(N):
        m = np.ones(N, dtype=bool); m[i] = False
        s, ic, _, _, _ = linregress(log_Sd[m], log_gc[m])
        pred_loo_0[i] = ic + s*log_Sd[i]
        Xtr = np.column_stack([np.ones(N-1), log_a0Sd[m], log_Yd[m]])
        btr = np.linalg.lstsq(Xtr, log_gc[m], rcond=None)[0]
        pred_loo_1[i] = btr[0] + btr[1]*log_a0Sd[i] + btr[2]*log_Yd[i]
    rl0 = log_gc - pred_loo_0; rl1 = log_gc - pred_loo_1
    sl0_loo = np.std(rl0); sl1_loo = np.std(rl1)
    R2l0 = 1 - np.sum(rl0**2)/SS_tot; R2l1 = 1 - np.sum(rl1**2)/SS_tot
    print(f"  M0: train R2={R2_0:.4f} sc={scat0:.4f} | LOO R2={R2l0:.4f} sc={sl0_loo:.4f}")
    print(f"  M1: train R2={R2_1:.4f} sc={scat1:.4f} | LOO R2={R2l1:.4f} sc={sl1_loo:.4f}")
    print(f"  LOO improvement: dR2={R2l1-R2l0:+.4f}, dsc={sl1_loo-sl0_loo:+.4f} ({(sl0_loo-sl1_loo)/sl0_loo*100:.1f}%)")
    if R2l1 > R2l0:
        print(f"  -> Yd contribution validated by LOO (not overfitting)")

    # T6: Yd bin stability
    print("\n" + "=" * 70)
    print("T6: Yd bin stability")
    print("=" * 70)
    n_bins = 4; sort_yd = np.argsort(log_Yd); bsz = N // n_bins
    print(f"  {'bin':>3s} {'Yd range':>16s} {'N':>4s} {'alpha':>10s} {'eta':>8s} {'R2':>7s}")
    for b in range(n_bins):
        i0 = b*bsz; i1 = (b+1)*bsz if b < n_bins-1 else N
        idx = sort_yd[i0:i1]
        s, ic, r, _, se = linregress(log_Sd[idx], log_gc[idx])
        print(f"  {b+1:3d} [{Yd[idx].min():.3f},{Yd[idx].max():.3f}] {len(idx):4d} {s:7.3f}+/-{se:.3f} {10**ic:8.4f} {r**2:7.3f}")

    # T7: Condition 15 final form
    print("\n" + "=" * 70)
    print("T7: Condition 15 final form")
    print("=" * 70)

    # alpha=0.5 fixed version
    lg_minus = log_gc - 0.5*log_a0Sd
    sl_f, ic_f, r_f, _, se_f = linregress(log_Yd, lg_minus)
    pred_f = 0.5*log_a0Sd + ic_f + sl_f*log_Yd
    resid_f = log_gc - pred_f; scat_f = np.std(resid_f)
    R2_f = 1 - np.sum(resid_f**2)/SS_tot

    print(f"""
  Summary table:
  {'Model':<25s} {'R2':>7s} {'scatter':>8s} {'AIC':>8s} {'R2_LOO':>7s} {'sc_LOO':>8s}
  {'M0 Sd only':<25s} {R2_0:7.4f} {scat0:8.4f} {AIC_0:8.1f} {R2l0:7.4f} {sl0_loo:8.4f}
  {'M1 Sd+Yd':<25s} {R2_1:7.4f} {scat1:8.4f} {AIC_1:8.1f} {R2l1:7.4f} {sl1_loo:8.4f}
  {'M1b vf+hR+Yd':<25s} {R2_1b:7.4f} {scat1b:8.4f} {AIC_1b:8.1f} {'---':>7s} {'---':>8s}
  {'alpha=0.5+Yd':<25s} {R2_f:7.4f} {scat_f:8.4f} {'---':>8s} {'---':>7s} {'---':>8s}

  Condition 15:
    gc = eta0 * Yd^beta * (a0 * Sigma_dyn)^alpha

    alpha  = {b1[1]:.4f}
    beta   = {b1[2]:.4f}  (eta ~ Yd^{b1[2]:.2f})
    eta0   = {10**b1[0]:.4f}
    scatter= {scat1:.3f} dex (LOO: {sl1_loo:.3f})

  alpha=0.5 fixed:
    beta_Yd = {sl_f:.4f}+/-{se_f:.4f}
    eta0    = {10**ic_f:.4f}
    R2      = {R2_f:.4f}, scatter = {scat_f:.4f}

  Condition 14+15 integrated:
    gc_eff(r) = eta0 * Yd^beta * (a0*Sd)^alpha * [1 + b14*max(0, eps(r)-eps_c)]
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        ax.scatter(log_gc, pred0, s=10, alpha=0.3, label=f'M0 R2={R2_0:.3f}')
        ax.scatter(log_gc, pred1, s=10, alpha=0.3, label=f'M1 R2={R2_1:.3f}')
        lim = [log_gc.min()-0.1, log_gc.max()+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('obs'); ax.set_ylabel('pred')
        ax.set_title('(a) model comparison'); ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.scatter(log_Yd, log_eta_pg, s=12, alpha=0.5)
        xl = np.linspace(log_Yd.min(), log_Yd.max(), 50)
        ax.plot(xl, ic_ey+sl_ey*xl, 'r-', lw=2, label=f'Yd^{sl_ey:.2f}')
        ax.set_xlabel('log Yd'); ax.set_ylabel('log eta')
        ax.set_title('(b) eta vs Yd'); ax.legend()

        ax = axes[0, 2]
        ax.scatter(log_gc, resid0, s=10, alpha=0.3, label=f'M0 s={scat0:.3f}')
        ax.scatter(log_gc, resid1, s=10, alpha=0.3, label=f'M1 s={scat1:.3f}')
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel('log gc'); ax.set_ylabel('resid')
        ax.set_title('(c) residuals'); ax.legend(fontsize=7)

        ax = axes[1, 0]
        ax.scatter(log_gc, rl0, s=10, alpha=0.3, label=f'M0 LOO s={sl0_loo:.3f}')
        ax.scatter(log_gc, rl1, s=10, alpha=0.3, label=f'M1 LOO s={sl1_loo:.3f}')
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel('log gc'); ax.set_ylabel('LOO resid')
        ax.set_title('(d) LOO validation'); ax.legend(fontsize=7)

        ax = axes[1, 1]
        ax.hist(log_eta_pg, bins=25, alpha=0.5, label=f'before s={np.std(log_eta_pg):.3f}')
        ax.hist(eta_corr, bins=25, alpha=0.5, label=f'after Yd s={np.std(eta_corr):.3f}')
        ax.set_xlabel('log eta'); ax.set_title('(e) eta dist'); ax.legend(fontsize=7)

        ax = axes[1, 2]
        ax.scatter(log_Yd, log_gc, c=log_Sd, cmap='viridis', s=12, alpha=0.5)
        plt.colorbar(ax.collections[0], ax=ax, label='log Sd')
        ax.set_xlabel('log Yd'); ax.set_ylabel('log gc')
        ax.set_title('(f) gc vs Yd colored by Sd')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'eta_yd_model.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
