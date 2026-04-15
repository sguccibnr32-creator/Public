#!/usr/bin/env python3
"""
sparc_self_referential.py (TA3+phase1+MRT adapted)
Self-referential structure & Condition 15 quantitative analysis.
"""
import os, csv, warnings, glob
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19; pc_m = 3.0857e16; Msun = 1.989e30

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
                data[p[0]] = {'T_type': int(p[1]), 'Rdisk': float(p[11]),
                              'SBdisk0': float(p[12]), 'MHI': float(p[13]), 'RHI': float(p[14])}
            except: continue
    return data

def get_sbdisk_central(name):
    fp = os.path.join(ROTMOD, f"{name}_rotmod.dat")
    if not os.path.exists(fp): return np.nan
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'): continue
            pp = line.split()
            if len(pp) >= 7:
                try:
                    sb = float(pp[6])
                    if sb > 0: return sb
                except: pass
    return np.nan

def fit_alpha(log_gc, g_proxy, n_boot=500):
    v = np.isfinite(g_proxy) & (g_proxy > 0) & np.isfinite(log_gc)
    if v.sum() < 20: return np.nan, np.nan, np.nan, np.nan, v.sum()
    def resid(p, gp, lgc): return p[0] + p[1]*np.log10(a0*gp) - lgc
    res = optimize.least_squares(resid, [0, 0.5], args=(g_proxy[v], log_gc[v]),
                                  bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha = res.x[1]; eta = 10**res.x[0]
    pred = res.x[0] + alpha*np.log10(a0*g_proxy[v])
    R2 = 1 - np.sum((log_gc[v]-pred)**2)/np.sum((log_gc[v]-log_gc[v].mean())**2)
    np.random.seed(42); boots = []
    ng = v.sum()
    for _ in range(n_boot):
        idx = np.random.choice(ng, ng, replace=True)
        try:
            rb = optimize.least_squares(resid, res.x, args=(g_proxy[v][idx], log_gc[v][idx]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            boots.append(rb.x[1])
        except: pass
    err = np.std(boots) if boots else np.nan
    return alpha, err, eta, R2, v.sum()

def main():
    print("=" * 70)
    print("Self-referential structure & Condition 15")
    print("=" * 70)

    pipe = load_pipeline()
    mrt = parse_mrt()
    print(f"Pipeline: {len(pipe)}, MRT: {len(mrt)}")

    records = []
    for n in pipe:
        gd = pipe[n]
        gc = gd['gc']; vf = gd['vflat']; Yd = gd.get('Yd', 0.5)
        if gc <= 0 or vf <= 0: continue
        m = mrt.get(n, {})
        hR = m.get('Rdisk', 0)
        if hR <= 0: continue
        sb = get_sbdisk_central(n)
        if not np.isfinite(sb): continue
        sb_mrt = m.get('SBdisk0', np.nan)
        MHI = m.get('MHI', np.nan); RHI = m.get('RHI', np.nan)
        sg = (MHI*1.33e9)/(np.pi*(RHI*1e3)**2) if MHI > 0 and RHI > 0 else 0
        sigma_dyn = (vf*1e3)**2 / (hR*kpc_m)
        sigma_YdSB = G_SI * Yd * sb * Msun / pc_m**2
        sigma_YdSBg = G_SI * (Yd*sb + sg) * Msun / pc_m**2
        sigma_SB = G_SI * sb * Msun / pc_m**2
        records.append({'name': n, 'gc': gc, 'vflat': vf, 'hR': hR, 'Yd': Yd,
                        'sb': sb, 'sg': sg, 'sigma_dyn': sigma_dyn,
                        'sigma_YdSB': sigma_YdSB, 'sigma_YdSBg': sigma_YdSBg,
                        'sigma_SB': sigma_SB})

    N = len(records)
    print(f"Integrated: {N}")
    gc = np.array([r['gc'] for r in records]); log_gc = np.log10(gc)
    vf = np.array([r['vflat'] for r in records]); hR = np.array([r['hR'] for r in records])
    Yd = np.array([r['Yd'] for r in records]); sb = np.array([r['sb'] for r in records])
    sd = np.array([r['sigma_dyn'] for r in records])
    s_YdSB = np.array([r['sigma_YdSB'] for r in records])
    s_YdSBg = np.array([r['sigma_YdSBg'] for r in records])
    s_SB = np.array([r['sigma_SB'] for r in records])

    # T1
    print("\n" + "=" * 70)
    print("T1: alpha for 4 Sigma definitions")
    print("=" * 70)
    defs = {"(a) Sigma_dyn (vflat^2/hR)": sd, "(b) Yd*SBdisk": s_YdSB,
            "(c) Yd*SBdisk+gas": s_YdSBg, "(d) SBdisk (Yd=1)": s_SB}
    print(f"\n  {'def':<35s} {'N':>5s} {'alpha':>7s} {'err':>7s} {'R2':>7s}")
    res = {}
    for label, sig in defs.items():
        a, e, eta, r2, nn = fit_alpha(log_gc, sig)
        res[label] = {'alpha': a, 'err': e, 'R2': r2}
        print(f"  {label:<35s} {nn:>5d} {a:>7.3f} {e:>7.3f} {r2:>7.3f}")
    a_dyn = res["(a) Sigma_dyn (vflat^2/hR)"]['alpha']
    a_bar = res["(c) Yd*SBdisk+gas"]['alpha']
    a_star = res["(b) Yd*SBdisk"]['alpha']
    a_SB = res["(d) SBdisk (Yd=1)"]['alpha']

    # T2
    print("\n" + "=" * 70)
    print("T2: alpha_bar = alpha_dyn/(2-alpha_dyn)")
    print("=" * 70)
    a_pred = a_dyn / (2 - a_dyn)
    print(f"  alpha_dyn = {a_dyn:.4f}")
    print(f"  Predicted alpha_bar = {a_pred:.4f}")
    print(f"  Observed:  bar={a_bar:.4f}, star={a_star:.4f}, SB={a_SB:.4f}")
    print(f"  Theory(a_dyn=0.5): alpha_bar = 1/3 = 0.333")

    # T3
    print("\n" + "=" * 70)
    print("T3: Yd variance attenuation")
    print("=" * 70)
    log_SB = np.log10(sb); log_Yd = np.log10(Yd)
    log_YdSB = np.log10(Yd*sb)
    v_bar = s_YdSBg > 0
    log_bar = np.log10(s_YdSBg[v_bar])
    print(f"  var(log SB)       = {np.var(log_SB):.4f}")
    print(f"  var(log Yd)       = {np.var(log_Yd):.4f}")
    print(f"  var(log Yd*SB)    = {np.var(log_YdSB):.4f}")
    print(f"  var(log Sigma_bar)= {np.var(log_bar):.4f}")
    print(f"  var(log Sigma_dyn)= {np.var(np.log10(sd)):.4f}")
    print(f"  var(log gc)       = {np.var(log_gc):.4f}")
    ratio_var = np.var(log_SB) / np.var(log_YdSB)
    print(f"  attenuation: var(SB)/var(Yd*SB) = {ratio_var:.3f}")
    rho_yd_sb, _ = stats.pearsonr(log_Yd, log_SB)
    print(f"  rho(Yd, SBdisk) = {rho_yd_sb:.3f}")

    # T4
    print("\n" + "=" * 70)
    print("T4: Deep-MOND identity Sigma_dyn = sqrt(2pi*Sigma_bar*gc)")
    print("=" * 70)
    pred_sd = 0.5*np.log10(2*np.pi) + 0.5*log_bar + 0.5*log_gc[v_bar]
    act_sd = np.log10(sd[v_bar])
    rho_s, _ = stats.pearsonr(pred_sd, act_sd)
    sc_s = np.std(act_sd - pred_sd)
    off_s = np.mean(act_sd - pred_sd)
    print(f"  r(pred, actual) = {rho_s:.4f}")
    print(f"  scatter = {sc_s:.3f} dex, offset = {off_s:.3f}")

    # T5
    print("\n" + "=" * 70)
    print("T5: Variance decomposition of log(Sigma_dyn)")
    print("=" * 70)
    var_gc = 0.25*np.var(log_gc[v_bar])
    var_bar = 0.25*np.var(log_bar)
    cov_t = 0.5*np.cov(log_gc[v_bar], log_bar)[0, 1]
    var_act = np.var(act_sd)
    print(f"  gc:        {var_gc:.4f} ({var_gc/var_act*100:.1f}%)")
    print(f"  Sigma_bar: {var_bar:.4f} ({var_bar/var_act*100:.1f}%)")
    print(f"  covar:     {cov_t:.4f} ({cov_t/var_act*100:.1f}%)")
    print(f"  sum:       {var_gc+var_bar+cov_t:.4f}")
    print(f"  actual:    {var_act:.4f}")
    frac_explained = (var_gc+var_bar+cov_t)/var_act*100
    print(f"  explained: {frac_explained:.1f}%")

    # T6
    print("\n" + "=" * 70)
    print("T6: alpha=1/3, free, 0.5 on baryonic Sigma")
    print("=" * 70)
    log_x = np.log10(a0 * s_YdSBg[v_bar])
    lgc_v = log_gc[v_bar]
    for alabel, aval in [("1/3 (theory)", 1/3), ("free", a_bar), ("0.5", 0.5)]:
        log_eta = np.median(lgc_v - aval*log_x)
        pred = log_eta + aval*log_x
        R2 = 1 - np.sum((lgc_v-pred)**2)/np.sum((lgc_v-lgc_v.mean())**2)
        sc = np.std(lgc_v - pred)
        n_aic = v_bar.sum()
        k = 1 if alabel != "free" else 2
        aic = n_aic*np.log(np.sum((lgc_v-pred)**2)/n_aic) + 2*k
        print(f"  alpha={alabel:12s}: R2={R2:.4f}, scatter={sc:.3f}, eta={10**log_eta:.4f}")

    # T7
    print("\n" + "=" * 70)
    print("T7: Dual eta")
    print("=" * 70)
    eta_dyn = gc / np.sqrt(a0 * sd)
    eta_bar = np.full(N, np.nan)
    for i in range(N):
        if s_YdSBg[i] > 0:
            base = gc[i] / (a0**(2/3) * (2*np.pi*s_YdSBg[i])**(1/3))
            if base > 0: eta_bar[i] = base**(3/4)
    ve = np.isfinite(eta_bar) & (eta_bar > 0)
    print(f"  eta_dyn: median={np.median(eta_dyn):.4f}, scatter={np.std(np.log10(eta_dyn)):.3f}")
    if ve.sum() > 10:
        print(f"  eta_bar: median={np.median(eta_bar[ve]):.4f}, scatter={np.std(np.log10(eta_bar[ve])):.3f}")
        rho_e, _ = stats.pearsonr(np.log10(eta_dyn[ve]), np.log10(eta_bar[ve]))
        print(f"  r(eta_dyn, eta_bar) = {rho_e:.3f}")
        print(f"  eta_bar^(4/3) / eta_dyn^2 = {np.median(eta_bar[ve])**(4/3)/np.median(eta_dyn)**2:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("CONDITION 15 Summary")
    print("=" * 70)
    print(f"""
  Self-referential loop:
    gc = eta * sqrt(a0 * Sigma_dyn)   [geometric mean, alpha_dyn~0.59]
    Sigma_dyn = sqrt(2pi * G*Sigma_bar * gc)  [deep MOND]
    => gc = eta^(4/3) * a0^(2/3) * (2pi*G*Sigma_bar)^(1/3)

  Predictions:
    (C15a) alpha_bar = 1/3          observed: {a_bar:.3f}  (gap {abs(1/3-a_bar):.3f})
    (C15b) alpha_bar = a_dyn/(2-a_dyn) = {a_pred:.3f}  observed: {a_bar:.3f}  (gap {abs(a_pred-a_bar):.3f})

  Deep-MOND identity:
    r = {rho_s:.3f}, scatter = {sc_s:.3f} dex -> explains {frac_explained:.0f}% of Sigma_dyn variance

  Variance decomposition:
    gc contributes {var_gc/var_act*100:.0f}%, Sigma_bar {var_bar/var_act*100:.0f}%
    => Self-reference is real but not dominant

  Implication:
    alpha_dyn~0.5 is partly self-referential (Sigma_dyn encodes gc).
    True baryonic exponent alpha_bar~0.14-0.22 << 1/3.
    Gap from 1/3: Yd variance + non-deep-MOND + disk geometry.
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Self-Referential Structure & Condition 15", fontsize=14)

        ax = axes[0, 0]
        adr = np.linspace(0.1, 0.9, 50)
        ax.plot(adr, adr/(2-adr), 'b-', lw=2, label='theory')
        ax.scatter([a_dyn], [a_bar], c='red', s=100, zorder=5, label=f'bar: {a_bar:.2f}')
        ax.scatter([a_dyn], [a_star], c='orange', s=80, zorder=5, label=f'star: {a_star:.2f}')
        ax.scatter([a_dyn], [a_SB], c='gray', s=60, zorder=5, label=f'SB: {a_SB:.2f}')
        ax.axhline(1/3, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('alpha_dyn'); ax.set_ylabel('alpha_bar')
        ax.set_title('(a) Theory curve'); ax.legend(fontsize=6)

        ax = axes[0, 1]
        ax.scatter(pred_sd, act_sd, s=10, alpha=0.5, c='steelblue')
        lim = [min(pred_sd.min(), act_sd.min())-0.2, max(pred_sd.max(), act_sd.max())+0.2]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('0.5*log(2pi*Sb*gc)'); ax.set_ylabel('log(Sd)')
        ax.set_title(f'(b) Deep-MOND r={rho_s:.3f}')

        ax = axes[0, 2]
        labs = ['gc', 'Sb', 'cov', 'actual']
        vals = [var_gc, var_bar, cov_t, var_act]
        ax.bar(range(4), vals, color=['steelblue', 'salmon', 'gray', 'black'])
        ax.set_xticks(range(4)); ax.set_xticklabels(labs)
        ax.set_ylabel('variance'); ax.set_title('(c) decomposition')

        ax = axes[1, 0]
        ax.scatter(log_x, lgc_v, s=10, alpha=0.5, c='salmon')
        xf = np.linspace(log_x.min(), log_x.max(), 50)
        le13 = np.median(lgc_v - (1/3)*log_x)
        ax.plot(xf, le13+(1/3)*xf, 'r-', lw=2, label='1/3')
        lef = np.median(lgc_v - a_bar*log_x)
        ax.plot(xf, lef+a_bar*xf, 'b--', lw=1, label=f'{a_bar:.2f}')
        ax.set_xlabel('log(a0*G*Sigma_bar)'); ax.set_ylabel('log gc')
        ax.set_title('(d) baryonic fit'); ax.legend()

        ax = axes[1, 1]
        ax.scatter(log_Yd, log_SB, s=10, alpha=0.5, c='steelblue')
        ax.set_xlabel('log Yd'); ax.set_ylabel('log SBdisk')
        ax.set_title(f'(e) Yd vs SB r={rho_yd_sb:.3f}')

        ax = axes[1, 2]
        if ve.sum() > 10:
            ax.scatter(np.log10(eta_dyn[ve]), np.log10(eta_bar[ve]), s=10, alpha=0.5, c='steelblue')
            lim = [min(np.log10(eta_dyn[ve]).min(), np.log10(eta_bar[ve]).min())-0.1,
                   max(np.log10(eta_dyn[ve]).max(), np.log10(eta_bar[ve]).max())+0.1]
            ax.plot(lim, lim, 'k--')
            ax.set_xlabel('log eta_dyn'); ax.set_ylabel('log eta_bar')
            ax.set_title(f'(f) dual eta r={rho_e:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'self_referential.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
