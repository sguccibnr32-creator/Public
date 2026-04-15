#!/usr/bin/env python3
"""
sparc_alpha_bar_gap.py (TA3+phase1+MRT adapted)
alpha_bar gap quantification: theory 1/3 vs observed.
"""
import os, csv, warnings
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr, t as tdist
from scipy.optimize import minimize
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
                data[p[0]] = {'T': int(p[1]), 'L': float(p[7]),
                              'Rdisk': float(p[11]), 'SBdisk0': float(p[12]),
                              'MHI': float(p[13]), 'RHI': float(p[14])}
            except: continue
    return data

def main():
    print("=" * 70)
    print("alpha_bar gap: theory 1/3 vs observed")
    print("=" * 70)

    pipe = load_pipeline(); mrt = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}")

    names = sorted([n for n in pipe if n in mrt
                     and mrt[n].get('L', 0) > 0 and mrt[n].get('Rdisk', 0) > 0])
    N = len(names)
    print(f"Full data: N={N}")

    gc = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    L36 = np.array([mrt[n]['L'] for n in names])
    Rdisk = np.array([mrt[n]['Rdisk'] for n in names])
    MHI = np.array([mrt[n].get('MHI', 0) for n in names])
    SBdisk0 = np.array([mrt[n].get('SBdisk0', 0) for n in names])

    vf_ms = vf * 1e3; Rd_m = Rdisk * kpc_m
    log_gc = np.log10(gc); log_vf = np.log10(vf); log_Yd = np.log10(Yd)
    Sd = vf_ms**2 / Rd_m; log_Sd = np.log10(Sd)

    M_star = Yd * L36 * 1e9 * Msun
    M_gas = 1.33 * MHI * 1e9 * Msun
    M_bar = M_star + M_gas
    Sigma_bar = M_bar / (2*np.pi*Rd_m**2)
    G_Sigma_bar = G_SI * Sigma_bar

    # Luminosity surface density (Yd-free)
    L_SI = L36 * 1e9 * Lsun
    Sigma_L = L_SI / (2*np.pi*Rd_m**2)
    G_Sigma_L = G_SI * Sigma_L

    SS_tot = np.sum((log_gc - log_gc.mean())**2)

    # T1: algebraic derivation
    print("\n" + "=" * 70)
    print("T1: Self-referential algebra with eta(Yd)")
    print("=" * 70)
    print("""
  gc = eta0 * Yd^beta * (a0*Sd)^alpha,  Sd = vf^2/hR
  vf^4 = G*M_bar*gc  (membrane BTFR)
  => Sd = sqrt(G*M_bar*gc)/hR
  => gc^(1-alpha/2) = eta0*Yd^beta * a0^alpha * (G*Sigma_bar)^(alpha/2)
  => gc = [...]^(2/(2-alpha))
  => alpha_bar = alpha/(2-alpha)

  With alpha=0.5: alpha_bar = 1/3 = 0.333
  With beta=-0.36: gc ~ Yd^(2*beta/(2-alpha)) * (G*Sigma_bar)^(alpha/(2-alpha))
                      = Yd^(-0.48) * (G*Sigma_bar)^(1/3)
""")

    # T2: direct alpha_bar measurement
    print("=" * 70)
    print("T2: alpha_bar direct measurement")
    print("=" * 70)
    v_bar = G_Sigma_bar > 0
    results = {}
    for label, gs, valid in [
        ("Sigma_dyn(MRT)", Sd, np.ones(N, dtype=bool)),
        ("Sigma_bar(MRT)", G_Sigma_bar, v_bar),
        ("Sigma_L(MRT)", G_Sigma_L, G_Sigma_L > 0),
    ]:
        nv = valid.sum()
        if nv < 20: continue
        lg = log_gc[valid]; ls = np.log10(gs[valid])
        sl, ic, r, _, se = linregress(ls, lg)
        res = lg - (ic + sl*ls)
        results[label] = {'alpha': sl, 'se': se, 'R2': r**2, 'sc': np.std(res), 'N': nv}
        print(f"  {label:20s} N={nv:3d} alpha={sl:.3f}+/-{se:.3f} R2={r**2:.3f}")

    a_dyn = results["Sigma_dyn(MRT)"]['alpha']
    a_bar_obs = results.get("Sigma_bar(MRT)", {}).get('alpha', np.nan)
    a_bar_pred = a_dyn / (2 - a_dyn)
    print(f"\n  alpha_dyn={a_dyn:.3f} -> predicted alpha_bar={a_bar_pred:.3f}")
    print(f"  Observed alpha_bar(Sigma_bar) = {a_bar_obs:.3f}")
    print(f"  Gap = {a_bar_pred - a_bar_obs:+.3f}")

    # T3: membrane BTFR
    print("\n" + "=" * 70)
    print("T3: Membrane BTFR verification")
    print("=" * 70)
    log_vf4 = 4*np.log10(vf_ms)
    log_GMbar = np.log10(G_SI * M_bar)
    log_GMbar_gc = np.log10(G_SI * M_bar * gc)

    sl_bt, ic_bt, r_bt, _, se_bt = linregress(log_GMbar, log_vf4)
    sl_mb, ic_mb, r_mb, _, se_mb = linregress(log_GMbar_gc, log_vf4)
    res_bt = log_vf4 - (ic_bt + sl_bt*log_GMbar)
    res_mb = log_vf4 - (ic_mb + sl_mb*log_GMbar_gc)
    print(f"  Standard BTFR: slope={sl_bt:.3f}+/-{se_bt:.3f}, R2={r_bt**2:.4f}, sc={np.std(res_bt):.4f}")
    print(f"  Membrane BTFR: slope={sl_mb:.3f}+/-{se_mb:.3f}, R2={r_mb**2:.4f}, sc={np.std(res_mb):.4f}")
    t_s1 = (sl_mb - 1.0)/se_mb
    p_s1 = 2*(1 - tdist.cdf(abs(t_s1), N-2))
    print(f"  slope=1 test: t={t_s1:.2f}, p={p_s1:.4e}")

    gc_btfr = vf_ms**4 / (G_SI * M_bar)
    log_gc_btfr = np.log10(gc_btfr)
    r_gg, _ = pearsonr(log_gc, log_gc_btfr)
    delta_gc = log_gc_btfr - log_gc
    print(f"\n  gc(BTFR) vs gc(tanh): r={r_gg:.4f}")
    print(f"  offset={np.mean(delta_gc):+.3f}, scatter={np.std(delta_gc):.3f} dex")
    print(f"  median ratio={10**np.median(delta_gc):.3f}")

    # T4: Yd-controlled alpha_bar
    print("\n" + "=" * 70)
    print("T4: Yd-controlled alpha_bar")
    print("=" * 70)
    nv = v_bar.sum()
    lg = log_gc[v_bar]; log_GSb = np.log10(G_Sigma_bar[v_bar]); ly = log_Yd[v_bar]
    log_GSL = np.log10(G_Sigma_L[v_bar])
    SS_v = np.sum((lg - lg.mean())**2)

    # Univariate
    sl_uni, _, r_uni, _, se_uni = linregress(log_GSb, lg)
    print(f"\n  Univariate: alpha_bar={sl_uni:.4f}+/-{se_uni:.4f}")

    # +Yd
    X2 = np.column_stack([np.ones(nv), log_GSb, ly])
    b2 = np.linalg.lstsq(X2, lg, rcond=None)[0]
    res2 = lg - X2@b2
    R2_2 = 1 - np.sum(res2**2)/SS_v
    print(f"  +Yd: alpha_bar={b2[1]:.4f}, gamma_Yd={b2[2]:.4f}, R2={R2_2:.4f}")

    # Sigma_L + Yd decomposition
    X3 = np.column_stack([np.ones(nv), log_GSL, ly])
    b3 = np.linalg.lstsq(X3, lg, rcond=None)[0]
    res3 = lg - X3@b3
    R2_3 = 1 - np.sum(res3**2)/SS_v
    print(f"  Sigma_L+Yd: b_SigmaL={b3[1]:.4f}, b_Yd={b3[2]:.4f}, R2={R2_3:.4f}")

    beta_yd = -0.361
    for alpha in [0.5, 0.537]:
        ab_p = alpha/(2-alpha)
        yd_p = 2*beta_yd/(2-alpha) + ab_p  # b_Yd prediction = ab + 2*beta/(2-alpha)
        yd_p_pure = 2*beta_yd/(2-alpha)
        print(f"\n  alpha={alpha}:")
        print(f"    alpha_bar pred={ab_p:.3f} vs b_SigmaL obs={b3[1]:.3f} (delta={b3[1]-ab_p:+.3f})")
        print(f"    b_Yd pred={yd_p_pure + ab_p:.3f} vs obs={b3[2]:.3f}")

    # T5: Gap decomposition
    print("\n" + "=" * 70)
    print("T5: Gap decomposition")
    print("=" * 70)
    r_yd_sb, _ = pearsonr(ly, log_GSb)
    print(f"  r(Yd, Sigma_bar) = {r_yd_sb:.3f}")
    sl_yd_sb, _, _, _, _ = linregress(log_GSb, ly)
    correction = b2[2] * sl_yd_sb
    print(f"  d(log Yd)/d(log Sigma_bar) = {sl_yd_sb:.4f}")
    print(f"  alpha_bar correction = gamma*slope = {correction:+.4f}")
    print(f"  alpha_bar(true) approx {sl_uni:.3f} - ({correction:+.3f}) = {sl_uni - correction:.3f}")
    print(f"  vs theory 1/3={0.333:.3f}, gap={abs(sl_uni - correction - 0.333):.3f}")

    # T6: per-galaxy
    print("\n" + "=" * 70)
    print("T6: gc(BTFR) analysis")
    print("=" * 70)
    print(f"  gc(BTFR) median/a0 = {np.median(gc_btfr)/a0:.3f}")
    print(f"  gc(BTFR) scatter = {np.std(log_gc_btfr):.3f} dex")
    for vn, va in [("vflat", log_vf), ("Yd", log_Yd)]:
        rho, p = spearmanr(va, delta_gc)
        print(f"  rho(delta_gc, {vn}) = {rho:+.3f} (p={p:.2e})")

    # T7: Summary
    print("\n" + "=" * 70)
    print("T7: Final assessment")
    print("=" * 70)
    print(f"""
  alpha_bar gap structure:
    Observed (univariate):    {sl_uni:.3f}
    Theory (alpha=0.5):       0.333
    Gap:                      {0.333 - sl_uni:+.3f}

  Decomposition:
    Yd-Sigma_bar correlation: r={r_yd_sb:.3f}
    Yd correction:            {correction:+.3f}
    Corrected alpha_bar:      {sl_uni - correction:.3f}
    Residual gap from 1/3:    {abs(sl_uni - correction - 0.333):.3f}

  Yd-controlled (multivar):  {b2[1]:.3f}
  Sigma_L decomposed:        b_SigmaL={b3[1]:.3f}

  Membrane BTFR:
    slope={sl_mb:.3f} (theory 1.0), R2={r_mb**2:.3f}
    gc(BTFR)/gc(tanh) scatter={np.std(delta_gc):.3f} dex

  Conclusion:
    Main gap cause: Yd is component of Sigma_bar.
    Negative eta(Yd) pushes apparent alpha_bar below 1/3.
    After Yd control: alpha_bar ~ {b2[1]:.3f} (closer to 1/3).
    Residual gap from imperfect membrane BTFR.
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        sc_ = ax.scatter(log_GSb, lg, c=ly, cmap='viridis', s=12, alpha=0.5)
        xf = np.linspace(log_GSb.min(), log_GSb.max(), 50)
        ax.plot(xf, sl_uni*xf + (lg.mean()-sl_uni*log_GSb.mean()), 'r-', lw=2, label=f'obs {sl_uni:.3f}')
        ax.plot(xf, 0.333*xf + (lg.mean()-0.333*log_GSb.mean()), 'b--', lw=2, label='theory 1/3')
        ax.set_xlabel('log(G*Sigma_bar)'); ax.set_ylabel('log gc')
        ax.set_title(f'(a) alpha_bar={sl_uni:.3f}'); ax.legend(fontsize=7)
        plt.colorbar(sc_, ax=ax, label='log Yd')

        ax = axes[0, 1]
        ax.scatter(log_GMbar_gc, log_vf4, s=12, alpha=0.5)
        lim = [log_GMbar_gc.min()-0.2, log_GMbar_gc.max()+0.2]
        ax.plot(lim, lim, 'k--'); ax.plot(lim, [ic_mb+sl_mb*x for x in lim], 'r-')
        ax.set_xlabel('log(G*Mbar*gc)'); ax.set_ylabel('log(vf^4)')
        ax.set_title(f'(b) BTFR slope={sl_mb:.3f}')

        ax = axes[0, 2]
        ax.scatter(log_gc, log_gc_btfr, s=12, alpha=0.5)
        lim2 = [log_gc.min()-0.2, log_gc.max()+0.2]
        ax.plot(lim2, lim2, 'r--')
        ax.set_xlabel('gc(tanh)'); ax.set_ylabel('gc(BTFR)')
        ax.set_title(f'(c) r={r_gg:.3f}')

        ax = axes[1, 0]
        labs = list(results.keys()) + ['Yd-ctrl', 'b_SigmaL']
        als = [results[l]['alpha'] for l in results] + [b2[1], b3[1]]
        ax.barh(range(len(labs)), als, alpha=0.7)
        ax.axvline(0.333, color='r', ls='--', label='1/3')
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel('alpha'); ax.set_title('(d) alpha comparison'); ax.legend()

        ax = axes[1, 1]
        ax.hist(delta_gc, bins=25, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', ls='--')
        ax.set_xlabel('log(gc_BTFR/gc_tanh)'); ax.set_title('(e) BTFR discrepancy')

        ax = axes[1, 2]
        ax.scatter(ly, log_GSb, s=12, alpha=0.5)
        ax.set_xlabel('log Yd'); ax.set_ylabel('log(G*Sigma_bar)')
        ax.set_title(f'(f) Yd-Sigma_bar r={r_yd_sb:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'alpha_bar_gap.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
