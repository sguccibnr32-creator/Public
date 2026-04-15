#!/usr/bin/env python3
"""sparc_alpha_bar_reanalysis.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19; pc_m = 3.0857e16
Msun = 1.989e30; Lsun = 3.828e26

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
                              'SBdisk0': float(p[12]), 'MHI': float(p[13])}
            except: continue
    return data

def main():
    print("=" * 70)
    print("alpha_bar gap re-analysis (post Simpson fix)")
    print("=" * 70)
    pipe = load_pipeline(); mrt = parse_mrt()
    names = sorted([n for n in pipe if n in mrt and mrt[n].get('L', 0) > 0 and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    gc = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    L36 = np.array([mrt[n]['L'] for n in names])
    MHI = np.array([mrt[n].get('MHI', 0) for n in names])
    Rdisk = np.array([mrt[n]['Rdisk'] for n in names])

    vf_ms = vf*1e3; Rd_m = Rdisk*kpc_m
    M_star = Yd*L36*1e9*Msun; M_gas = 1.33*MHI*1e9*Msun; M_bar = M_star + M_gas
    Sd = vf_ms**2/(Rd_m)

    log_gc = np.log10(gc); log_vf = np.log10(vf); log_Yd = np.log10(Yd)
    log_Sd = np.log10(Sd); log_Mbar = np.log10(M_bar/Msun)

    # Sigma_bar definitions
    GSb_A = G_SI * M_bar / (2*np.pi*Rd_m**2)  # M_bar/(2piR^2)
    L_SI = L36*1e9*Lsun; GSL = G_SI * L_SI/(2*np.pi*Rd_m**2)

    alpha = 0.537; beta = -0.361
    ab_pred = alpha/(2-alpha); gy_pred = 2*beta/(2-alpha)

    # T1
    print(f"\nT1: Theory: alpha_bar={ab_pred:.4f}, gamma_Yd={gy_pred:.4f}")

    # T2: Definitions
    print("\n" + "=" * 70)
    print("T2: Sigma_bar definitions")
    print("=" * 70)
    sl_dyn, _, r_dyn, _, se_dyn = linregress(log_Sd, log_gc)
    print(f"  alpha_dyn = {sl_dyn:.4f}+/-{se_dyn:.4f}")

    defs = {'A: Mbar/(2piR2)': GSb_A, 'E: GL/(2piR2)': GSL}
    print(f"\n  {'def':<20s} {'alpha_bar':>12s} {'R2':>7s} {'gap':>8s}")
    for label, gs in defs.items():
        v = gs > 0
        sl, _, r, _, se = linregress(np.log10(gs[v]), log_gc[v])
        print(f"  {label:<20s} {sl:8.4f}+/-{se:.4f} {r**2:7.4f} {ab_pred-sl:+8.4f}")

    # Univariate baseline
    log_GSbA = np.log10(GSb_A); log_GSL = np.log10(GSL)
    sl_uni, ic_uni, r_uni, _, se_uni = linregress(log_GSbA, log_gc)

    # T3: Deep MOND subset
    print("\n" + "=" * 70)
    print("T3: Deep MOND subsets")
    print("=" * 70)
    x = gc/a0
    print(f"  {'x_max':>6s} {'N':>4s} {'ab(A)':>10s} {'ab_Yd_ctrl':>12s} {'theory':>8s}")
    for xmax in [0.3, 0.5, 1.0, 2.0, 999]:
        mask = x < xmax; nm = mask.sum()
        if nm < 15: continue
        v = mask & (GSb_A > 0)
        sl_a, _, _, _, se_a = linregress(log_GSbA[v], log_gc[v])
        X = np.column_stack([np.ones(v.sum()), log_GSbA[v], log_Yd[v]])
        b = np.linalg.lstsq(X, log_gc[v], rcond=None)[0]
        sl_d, _, _, _, _ = linregress(log_Sd[mask], log_gc[mask])
        ab_d = sl_d/(2-sl_d)
        lbl = f"<{xmax}" if xmax < 999 else "all"
        print(f"  {lbl:>6s} {nm:4d} {sl_a:7.4f}+/-{se_a:.4f} {b[1]:12.4f} {ab_d:8.4f}")

    # T4: Nonlinearity
    print("\n" + "=" * 70)
    print("T4: Nonlinearity test")
    print("=" * 70)
    X2 = np.column_stack([np.ones(N), log_GSbA, log_GSbA**2])
    b2 = np.linalg.lstsq(X2, log_gc, rcond=None)[0]
    pred2 = X2 @ b2
    R2_2 = 1 - np.sum((log_gc-pred2)**2)/np.sum((log_gc-log_gc.mean())**2)
    print(f"  Quadratic: c={b2[2]:.4f}, R2={R2_2:.4f} (linear R2={r_uni**2:.4f})")

    # T5: Sigma_L + Yd decomposition
    print("\n" + "=" * 70)
    print("T5: Sigma_L + Yd decomposition")
    print("=" * 70)
    sl_l, _, r_l, _, se_l = linregress(log_GSL, log_gc)
    print(f"  gc vs Sigma_L: slope={sl_l:.4f}, R2={r_l**2:.4f}")
    X_ly = np.column_stack([np.ones(N), log_GSL, log_Yd])
    b_ly = np.linalg.lstsq(X_ly, log_gc, rcond=None)[0]
    R2_ly = 1 - np.sum((log_gc - X_ly@b_ly)**2)/np.sum((log_gc-log_gc.mean())**2)
    print(f"  gc vs Sigma_L + Yd: b_SL={b_ly[1]:.4f}, b_Yd={b_ly[2]:.4f}, R2={R2_ly:.4f}")
    print(f"  Theory: b_SL={ab_pred:.4f}, b_Yd={gy_pred+ab_pred:.4f}")

    X_by = np.column_stack([np.ones(N), log_GSbA, log_Yd])
    b_by = np.linalg.lstsq(X_by, log_gc, rcond=None)[0]
    print(f"\n  gc vs Sigma_bar + Yd: b_Sb={b_by[1]:.4f}, b_Yd={b_by[2]:.4f}")
    print(f"  Theory: b_Sb={ab_pred:.4f}, b_Yd={gy_pred:.4f}")

    # T6: Monte Carlo
    print("\n" + "=" * 70)
    print("T6: Monte Carlo error propagation")
    print("=" * 70)
    rng = np.random.RandomState(42)
    ab_mc, ab_mc_yd = [], []
    for _ in range(1000):
        Yd_p = Yd*10**(rng.normal(0, 0.15, N))
        L_p = L36*10**(rng.normal(0, 0.10, N))
        MHI_p = MHI*10**(rng.normal(0, 0.10, N))
        Mb_p = Yd_p*L_p*1e9*Msun + 1.33*MHI_p*1e9*Msun
        GSb_p = G_SI*Mb_p/(2*np.pi*Rd_m**2)
        v = GSb_p > 0
        sl_p, _, _, _, _ = linregress(np.log10(GSb_p[v]), log_gc[v])
        ab_mc.append(sl_p)
        X_p = np.column_stack([np.ones(N), np.log10(GSb_p), np.log10(Yd_p)])
        b_p = np.linalg.lstsq(X_p, log_gc, rcond=None)[0]
        ab_mc_yd.append(b_p[1])
    ab_mc = np.array(ab_mc); ab_mc_yd = np.array(ab_mc_yd)
    print(f"  Univar: {np.mean(ab_mc):.4f}+/-{np.std(ab_mc):.4f}, 95%CI=[{np.percentile(ab_mc,2.5):.4f},{np.percentile(ab_mc,97.5):.4f}]")
    print(f"  Yd ctrl: {np.mean(ab_mc_yd):.4f}+/-{np.std(ab_mc_yd):.4f}, 95%CI=[{np.percentile(ab_mc_yd,2.5):.4f},{np.percentile(ab_mc_yd,97.5):.4f}]")
    in_ci = np.percentile(ab_mc_yd, 2.5) <= ab_pred <= np.percentile(ab_mc_yd, 97.5)
    print(f"  Theory {ab_pred:.4f} {'IN' if in_ci else 'OUT of'} 95% CI")

    # T7: per-galaxy self-ref
    print("\n" + "=" * 70)
    print("T7: Per-galaxy gc prediction")
    print("=" * 70)
    gc_gml = 0.584 * Yd**(-0.361) * np.sqrt(a0*Sd)
    gc_btfr = vf_ms**4 / (G_SI*M_bar)
    gc_sr = 10**(b_by[0] + b_by[1]*log_GSbA + b_by[2]*log_Yd)
    print(f"  {'method':<25s} {'r':>7s} {'scatter':>8s} {'bias':>8s}")
    for lbl, gp in [("GML (Sd+Yd)", gc_gml), ("BTFR (vf4/GMb)", gc_btfr), ("Self-ref (Sb+Yd)", gc_sr)]:
        r_ = pearsonr(log_gc, np.log10(gp))[0]
        sc = np.std(log_gc - np.log10(gp))
        bi = np.mean(log_gc - np.log10(gp))
        print(f"  {lbl:<25s} {r_:7.3f} {sc:8.3f} {bi:+8.3f}")
    r_gs = pearsonr(np.log10(gc_gml), np.log10(gc_sr))[0]
    print(f"  GML vs Self-ref: r={r_gs:.3f}, scatter={np.std(np.log10(gc_gml)-np.log10(gc_sr)):.3f}")

    # T8: Gap decomposition
    print("\n" + "=" * 70)
    print("T8: Final gap decomposition")
    print("=" * 70)
    gap_total = ab_pred - sl_uni
    gap_yd = b_by[1] - sl_uni
    gap_remaining = ab_pred - b_by[1]
    print(f"  Theory: {ab_pred:.4f}, Univar: {sl_uni:.4f}, Yd-ctrl: {b_by[1]:.4f}")
    print(f"  Total gap:     {gap_total:+.4f}")
    print(f"  Yd effect:     {gap_yd:+.4f} ({gap_yd/gap_total*100:.0f}%)")
    print(f"  Remaining:     {gap_remaining:+.4f} ({gap_remaining/gap_total*100:.0f}%)")

    # Deep MOND check
    mask_d = x < 0.5; v_d = mask_d & (GSb_A > 0)
    if v_d.sum() > 15:
        X_d = np.column_stack([np.ones(v_d.sum()), log_GSbA[v_d], log_Yd[v_d]])
        b_d = np.linalg.lstsq(X_d, log_gc[v_d], rcond=None)[0]
        print(f"\n  Deep MOND (x<0.5, N={v_d.sum()}): ab_Yd_ctrl={b_d[1]:.4f}")
        print(f"  Theory gap in deep MOND: {ab_pred - b_d[1]:+.4f}")

    f_gas = M_gas/M_bar
    resid_btfr = np.log10(vf_ms**4) - np.log10(G_SI*M_bar*gc)
    print(f"\n  Remaining gap candidates:")
    print(f"    BTFR scatter: {np.std(resid_btfr):.3f} dex")
    print(f"    <f_gas>: {np.mean(f_gas):.3f}")
    print(f"    MC 95%CI covers theory: {'YES' if in_ci else 'NO'}")

    print(f"""
  CONCLUSION:
    alpha_bar gap = {gap_total:+.4f}
    Yd explains: {gap_yd/gap_total*100:.0f}%
    Remaining: {gap_remaining/gap_total*100:.0f}%
    MC error covers theory: {'YES -> gap is within measurement uncertainty' if in_ci else 'NO -> physical gap remains'}
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        sc_ = ax.scatter(log_GSbA, log_gc, c=log_Yd, cmap='viridis', s=12, alpha=0.5)
        xf = np.linspace(log_GSbA.min(), log_GSbA.max(), 50)
        ax.plot(xf, ic_uni+sl_uni*xf, 'r-', lw=2, label=f'obs {sl_uni:.3f}')
        ax.plot(xf, ab_pred*xf+(log_gc.mean()-ab_pred*log_GSbA.mean()), 'b--', lw=2, label=f'theory {ab_pred:.3f}')
        ax.set_xlabel('log(G*Sb)'); ax.set_ylabel('log gc'); ax.set_title('(a) gap')
        ax.legend(fontsize=7); plt.colorbar(sc_, ax=ax, label='log Yd')

        ax = axes[0, 1]
        ax.scatter(log_GSL, log_gc, s=12, alpha=0.5)
        ax.set_xlabel('log(G*SL)'); ax.set_ylabel('log gc')
        ax.set_title(f'(b) gc vs SigmaL slope={sl_l:.3f}')

        ax = axes[0, 2]
        ax.scatter(log_gc, np.log10(gc_gml), s=10, alpha=0.3, label='GML')
        ax.scatter(log_gc, np.log10(gc_sr), s=10, alpha=0.3, label='Self-ref')
        lim = [log_gc.min()-0.1, log_gc.max()+0.1]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('obs'); ax.set_ylabel('pred'); ax.set_title('(c) gc predictions')
        ax.legend(fontsize=7)

        ax = axes[1, 0]
        for xm, c in [(0.5, 'blue'), (999, 'red')]:
            m = (x < xm) & (GSb_A > 0)
            if m.sum() < 10: continue
            ax.scatter(log_GSbA[m], log_gc[m], s=8, alpha=0.3, c=c, label=f'x<{xm}')
        ax.set_xlabel('log(G*Sb)'); ax.set_ylabel('log gc'); ax.set_title('(d) deep MOND')
        ax.legend()

        ax = axes[1, 1]
        ax.hist(ab_mc, bins=30, alpha=0.5, label='univar')
        ax.hist(ab_mc_yd, bins=30, alpha=0.5, label='Yd ctrl')
        ax.axvline(ab_pred, color='r', ls='--', label=f'theory {ab_pred:.3f}')
        ax.set_xlabel('alpha_bar'); ax.set_title('(e) MC'); ax.legend(fontsize=7)

        ax = axes[1, 2]
        labs = ['Total', 'Yd', 'Remain']
        vals = [gap_total, gap_yd, gap_remaining]
        ax.bar(labs, vals, color=['gray', 'steelblue', 'coral'], alpha=0.7)
        ax.axhline(0, color='k'); ax.set_ylabel('gap'); ax.set_title('(f) decomposition')
        for i, v in enumerate(vals): ax.text(i, v+0.003*np.sign(v), f'{v:+.3f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'alpha_bar_reanalysis.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
