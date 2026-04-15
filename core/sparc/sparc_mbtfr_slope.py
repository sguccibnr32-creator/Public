#!/usr/bin/env python3
"""
sparc_mbtfr_slope.py (TA3+phase1+MRT adapted)
Membrane BTFR slope!=1 analysis.
"""
import os, csv, warnings
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr, t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19
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
                data[p[0]] = {'T': int(p[1]), 'L': float(p[7]),
                              'Rdisk': float(p[11]), 'MHI': float(p[13]),
                              'Q': int(p[17])}
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
    print("Membrane BTFR slope analysis")
    print("=" * 70)
    pipe = load_pipeline(); mrt = parse_mrt()
    names = sorted([n for n in pipe if n in mrt and mrt[n].get('L', 0) > 0
                     and mrt[n].get('Rdisk', 0) > 0])
    N = len(names)
    print(f"N={N}")

    gc = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    L36 = np.array([mrt[n]['L'] for n in names])
    MHI = np.array([mrt[n].get('MHI', 0) for n in names])
    Rdisk = np.array([mrt[n]['Rdisk'] for n in names])
    T_type = np.array([mrt[n].get('T', np.nan) for n in names])

    vf_ms = vf*1e3; hR_m = Rdisk*kpc_m
    M_star = Yd*L36*1e9*Msun; M_gas = 1.33*MHI*1e9*Msun; M_bar = M_star + M_gas
    f_gas = np.where(M_bar > 0, M_gas/M_bar, 0)
    Sd = vf_ms**2/hR_m

    log_gc = np.log10(gc); log_vf = np.log10(vf); log_vf4 = 4*np.log10(vf_ms)
    log_GMbar = np.log10(G_SI*M_bar); log_GMbar_gc = np.log10(G_SI*M_bar*gc)
    log_GMbar_a0 = np.log10(G_SI*M_bar*a0)
    log_Mbar = np.log10(M_bar/Msun); log_Yd = np.log10(Yd); log_Sd = np.log10(Sd)

    # T1
    print("\n" + "=" * 70)
    print("T1: Membrane BTFR diagnostics")
    print("=" * 70)
    sl, ic, r, _, se = linregress(log_GMbar_gc, log_vf4)
    pred = ic + sl*log_GMbar_gc; resid = log_vf4 - pred
    print(f"  slope={sl:.4f}+/-{se:.4f}, R2={r**2:.4f}, sc={np.std(resid):.4f}")
    print(f"  slope=1 rejection: {abs(sl-1)/se:.1f}sigma")

    resid_s1 = log_vf4 - log_GMbar_gc
    offset = np.mean(resid_s1)
    print(f"  slope=1 forced: offset={offset:+.4f} -> vf4={10**offset:.3f}*G*Mbar*gc")
    print(f"  scatter(slope=1)={np.std(resid_s1):.4f}")

    sl_std, ic_std, r_std, _, se_std = linregress(log_GMbar_a0, log_vf4)
    res_std = log_vf4 - (ic_std + sl_std*log_GMbar_a0)
    print(f"\n  Standard BTFR: slope={sl_std:.4f}+/-{se_std:.4f}, R2={r_std**2:.4f}, sc={np.std(res_std):.4f}")

    # T2
    print("\n" + "=" * 70)
    print("T2: gc mass dependence")
    print("=" * 70)
    sl_gm, _, r_gm, p_gm, se_gm = linregress(log_Mbar, log_gc)
    print(f"  gc vs Mbar: slope={sl_gm:.4f}+/-{se_gm:.4f}, r={r_gm:.3f}")
    sl_gv, _, r_gv, _, _ = linregress(log_vf, log_gc)
    print(f"  gc vs vflat: slope={sl_gv:.4f}, r={r_gv:.3f}")
    sl_gs, _, r_gs, _, _ = linregress(log_Sd, log_gc)
    print(f"  gc vs Sd: slope={sl_gs:.4f}, r={r_gs:.3f}")

    # T3
    print("\n" + "=" * 70)
    print("T3: Yd sensitivity")
    print("=" * 70)
    for yd_fix, label in [(0.3, "0.3"), (0.5, "0.5"), (0.8, "0.8")]:
        Mb = yd_fix*L36*1e9*Msun + M_gas
        lx = np.log10(G_SI*Mb*gc)
        s, _, r_, _, se_ = linregress(lx, log_vf4)
        print(f"  Yd={label}: slope={s:.4f}+/-{se_:.4f} ({abs(s-1)/se_:.1f}sig)")
    print(f"  per-gal Yd: slope={sl:.4f}+/-{se:.4f}")

    rho_yd_m, _ = spearmanr(log_Mbar, log_Yd)
    print(f"  rho(Yd, Mbar) = {rho_yd_m:+.3f}")

    # T4
    print("\n" + "=" * 70)
    print("T4: Gas fraction effect")
    print("=" * 70)
    fg_med = np.median(f_gas[f_gas > 0])
    for label, mask in [("Gas-rich", f_gas > fg_med), ("Gas-poor", f_gas <= fg_med)]:
        nm = mask.sum()
        if nm < 15: continue
        s_, _, r_, _, se_ = linregress(log_GMbar_gc[mask], log_vf4[mask])
        print(f"  {label} N={nm}: slope={s_:.4f}+/-{se_:.4f} ({abs(s_-1)/se_:.1f}sig)")

    # T5
    print("\n" + "=" * 70)
    print("T5: vflat regime split")
    print("=" * 70)
    pcts = np.percentile(log_vf, [33, 67])
    for label, mask in [("Low vf", log_vf < pcts[0]),
                         ("Mid vf", (log_vf >= pcts[0]) & (log_vf < pcts[1])),
                         ("High vf", log_vf >= pcts[1])]:
        nm = mask.sum()
        if nm < 10: continue
        s_, _, r_, _, se_ = linregress(log_GMbar_gc[mask], log_vf4[mask])
        x_med = np.median(gc[mask]/a0)
        print(f"  {label:8s} N={nm:3d} slope={s_:.3f}+/-{se_:.3f} <gc/a0>={x_med:.2f}")

    # T6
    print("\n" + "=" * 70)
    print("T6: Generalized BTFR")
    print("=" * 70)
    X_gen = np.column_stack([np.ones(N), log_GMbar, log_gc])
    b_gen = np.linalg.lstsq(X_gen, log_vf4, rcond=None)[0]
    pred_gen = X_gen @ b_gen
    SS_vf = np.sum((log_vf4 - log_vf4.mean())**2)
    R2_gen = 1 - np.sum((log_vf4 - pred_gen)**2)/SS_vf
    print(f"  log(vf4) = {b_gen[0]:.3f} + {b_gen[1]:.3f}*log(GMbar) + {b_gen[2]:.3f}*log(gc)")
    print(f"  R2={R2_gen:.4f}")
    print(f"  b_Mbar/b_gc = {b_gen[1]/b_gen[2]:.3f} (theory 1.0)")

    y_d = log_vf4 - log_GMbar
    sl_d, ic_d, r_d, _, se_d = linregress(log_gc, y_d)
    print(f"\n  gc power free: delta={sl_d:.4f}+/-{se_d:.4f} ({abs(sl_d-1)/se_d:.1f}sig from 1)")

    # T7
    print("\n" + "=" * 70)
    print("T7: gc mass dependence (partial)")
    print("=" * 70)
    rp1, pp1 = partial_corr(log_gc, log_Mbar, log_Sd)
    rp2, pp2 = partial_corr(log_gc, log_Mbar, log_vf.reshape(-1, 1))
    rp3, pp3 = partial_corr(log_gc, log_Mbar, np.column_stack([log_vf, np.log10(Rdisk)]))
    print(f"  r(gc, Mbar | Sd)       = {rp1:+.3f} (p={pp1:.2e})")
    print(f"  r(gc, Mbar | vflat)    = {rp2:+.3f} (p={pp2:.2e})")
    print(f"  r(gc, Mbar | vflat,hR) = {rp3:+.3f} (p={pp3:.2e})")

    # T8
    print("\n" + "=" * 70)
    print("T8: Required gc correction for slope=1")
    print("=" * 70)
    C_s1 = 10**offset
    gc_needed = vf_ms**4 / (C_s1*G_SI*M_bar)
    delta_n = np.log10(gc_needed) - log_gc
    sl_dn, ic_dn, r_dn, _, se_dn = linregress(log_Mbar, delta_n)
    print(f"  gc_needed/gc_obs: mean offset={np.mean(delta_n):+.3f}, std={np.std(delta_n):.3f}")
    print(f"  vs Mbar: slope={sl_dn:.4f}+/-{se_dn:.4f} -> gc_corr ~ Mbar^{sl_dn:.3f}")
    for vn, va in [("vflat", log_vf), ("gc", log_gc), ("Yd", log_Yd), ("Sd", log_Sd)]:
        rho, _ = spearmanr(va, delta_n)
        print(f"  rho(delta, {vn:6s}) = {rho:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Membrane BTFR slope = {sl:.3f}+/-{se:.3f} ({abs(sl-1)/se:.1f}sig from 1.0)
  Generalized: b_Mbar={b_gen[1]:.3f}, b_gc={b_gen[2]:.3f}
  gc power free: delta={sl_d:.3f}+/-{se_d:.3f}
  gc vs Mbar: r={r_gm:.3f}, slope={sl_gm:.3f}
  For slope=1: gc needs Mbar^{sl_dn:.3f} correction

  Yd sensitivity: slope ranges {sl:.3f} (per-gal) to ~same (fixed Yd)
  -> Yd is NOT the main cause of slope!=1

  Main cause: gc has weak mass dependence (r={r_gm:.3f})
  -> massive galaxies have slightly higher gc
  -> inflates BTFR slope above 1.0
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        sc_ = ax.scatter(log_GMbar_gc, log_vf4, c=log_gc, cmap='viridis', s=12, alpha=0.5)
        xf = np.linspace(log_GMbar_gc.min(), log_GMbar_gc.max(), 50)
        ax.plot(xf, ic+sl*xf, 'r-', lw=2, label=f's={sl:.3f}')
        ax.plot(xf, xf+offset, 'b--', lw=2, label='s=1')
        ax.set_xlabel('log(G*Mbar*gc)'); ax.set_ylabel('log(vf^4)')
        ax.set_title(f'(a) mBTFR s={sl:.3f}'); ax.legend(); plt.colorbar(sc_, ax=ax)

        ax = axes[0, 1]
        ax.scatter(log_Mbar, log_gc, s=12, alpha=0.5)
        xf = np.linspace(log_Mbar.min(), log_Mbar.max(), 50)
        ax.plot(xf, sl_gm*xf + (log_gc.mean()-sl_gm*log_Mbar.mean()), 'r-', lw=2)
        ax.set_xlabel('log Mbar'); ax.set_ylabel('log gc')
        ax.set_title(f'(b) gc vs Mbar r={r_gm:.3f}')

        ax = axes[0, 2]
        for yd_f, s_f, c in [(0.3, None, 'gray'), (0.5, None, 'orange'), (0.8, None, 'green')]:
            Mb = yd_f*L36*1e9*Msun + M_gas
            lx = np.log10(G_SI*Mb*gc)
            s_f, _, _, _, _ = linregress(lx, log_vf4)
            ax.scatter([], [], c=c, label=f'Yd={yd_f}: s={s_f:.3f}')
        ax.scatter(log_GMbar_gc, log_vf4, s=8, alpha=0.3, c='steelblue')
        ax.set_xlabel('log(G*Mbar*gc)'); ax.set_ylabel('log(vf^4)')
        ax.set_title('(c) Yd sensitivity'); ax.legend(fontsize=7)

        ax = axes[1, 0]
        ax.scatter(log_Mbar, delta_n, s=12, alpha=0.5)
        xf = np.linspace(log_Mbar.min(), log_Mbar.max(), 50)
        ax.plot(xf, ic_dn+sl_dn*xf, 'r-', lw=2, label=f'~Mbar^{sl_dn:.2f}')
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel('log Mbar'); ax.set_ylabel('log(gc_need/gc_obs)')
        ax.set_title('(d) correction for s=1'); ax.legend()

        ax = axes[1, 1]
        ax.scatter(log_gc, resid, s=12, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        rho_r, _ = spearmanr(log_gc, resid)
        ax.set_xlabel('log gc'); ax.set_ylabel('BTFR resid')
        ax.set_title(f'(e) resid vs gc rho={rho_r:+.3f}')

        ax = axes[1, 2]
        labs = ['b_Mbar', 'b_gc', 'delta']
        vals = [b_gen[1], b_gen[2], sl_d]
        ax.bar(labs, vals, color=['steelblue', 'coral', 'green'], alpha=0.7)
        ax.axhline(1, color='r', ls='--')
        for i, v in enumerate(vals):
            ax.text(i, v+0.02, f'{v:.3f}', ha='center')
        ax.set_title('(f) generalized decomposition')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'mbtfr_slope.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
