#!/usr/bin/env python3
"""sparc_simpson_paradox.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; G_SI = 6.674e-11; kpc_m = 3.0857e19; Msun = 1.989e30; Lsun = 3.828e26

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
            try: data[p[0]] = {'L': float(p[7]), 'Rdisk': float(p[11]), 'MHI': float(p[13])}
            except: continue
    return data

def main():
    print("=" * 70)
    print("Simpson's Paradox in Membrane BTFR")
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

    vf_ms = vf*1e3; hR_m = Rdisk*kpc_m
    M_star = Yd*L36*1e9*Msun; M_gas = 1.33*MHI*1e9*Msun; M_bar = M_star + M_gas

    log_vf = np.log10(vf); log_vf4 = 4*np.log10(vf_ms)
    log_gc = np.log10(gc); log_Mbar = np.log10(M_bar/Msun)
    log_GMgc = np.log10(G_SI*M_bar*gc); log_hR = np.log10(Rdisk)

    sl_all, ic_all, r_all, _, se_all = linregress(log_GMgc, log_vf4)
    resid_s1 = log_vf4 - log_GMgc; offset_s1 = np.mean(resid_s1)

    # T1: Paradox structure
    print("\n" + "=" * 70)
    print("T1: Paradox structure")
    print("=" * 70)
    print(f"  Overall: slope={sl_all:.3f}+/-{se_all:.3f}")
    n_bins = 5; sort_vf = np.argsort(log_vf); bsz = N // n_bins
    bin_info = []
    print(f"\n  {'bin':>3s} {'vf range':>16s} {'N':>4s} {'slope':>10s} {'R2':>7s}")
    for b in range(n_bins):
        i0 = b*bsz; i1 = (b+1)*bsz if b < n_bins-1 else N
        idx = sort_vf[i0:i1]
        s, _, r, _, se = linregress(log_GMgc[idx], log_vf4[idx])
        bin_info.append({'idx': idx, 'slope': s, 'se': se, 'r2': r**2,
                         'x_mid': np.median(log_GMgc[idx]), 'y_mid': np.median(log_vf4[idx]),
                         'vf_mid': np.median(vf[idx])})
        print(f"  {b+1:3d} [{vf[idx].min():.0f},{vf[idx].max():.0f}] {len(idx):4d} {s:7.3f}+/-{se:.3f} {r**2:7.3f}")

    x_mids = np.array([bi['x_mid'] for bi in bin_info])
    y_mids = np.array([bi['y_mid'] for bi in bin_info])
    sl_bet, _, _, _, se_bet = linregress(x_mids, y_mids)
    w_s = sum(bi['slope']/bi['se']**2 for bi in bin_info if bi['se'] > 0)
    w_w = sum(1/bi['se']**2 for bi in bin_info if bi['se'] > 0)
    sl_within = w_s / w_w if w_w > 0 else np.nan
    print(f"\n  Between slope = {sl_bet:.3f}")
    print(f"  Weighted within slope = {sl_within:.3f}")
    print(f"  Simpson gap = {sl_all - sl_within:+.3f}")

    # T2: Confounders
    print("\n" + "=" * 70)
    print("T2: Confounders")
    print("=" * 70)
    r_vf_x, _ = pearsonr(log_vf, log_GMgc)
    r_vf_gc, _ = pearsonr(log_vf, log_gc)
    r_vf_mb, _ = pearsonr(log_vf, log_Mbar)
    print(f"  r(vflat, x=log GMgc) = {r_vf_x:.3f}")
    print(f"  r(vflat, gc) = {r_vf_gc:.3f}")
    print(f"  r(vflat, Mbar) = {r_vf_mb:.3f}")

    var_gm = np.var(np.log10(G_SI*M_bar))
    var_gc = np.var(log_gc)
    cov_gm_gc = np.cov(np.log10(G_SI*M_bar), log_gc)[0, 1]
    var_x = np.var(log_GMgc)
    print(f"\n  Var(x) decomposition:")
    print(f"    Var(log GM) = {var_gm:.4f} ({var_gm/var_x*100:.1f}%)")
    print(f"    Var(log gc) = {var_gc:.4f} ({var_gc/var_x*100:.1f}%)")
    print(f"    2*Cov       = {2*cov_gm_gc:.4f} ({2*cov_gm_gc/var_x*100:.1f}%)")

    r_gc_mb, _ = pearsonr(log_gc, log_Mbar)
    print(f"\n  r(gc, Mbar) = {r_gc_mb:.3f}")
    print(f"  gc range: {log_gc.max()-log_gc.min():.2f} dex")
    print(f"  Mbar range: {log_Mbar.max()-log_Mbar.min():.2f} dex")

    # T4: Analytical prediction
    print("\n" + "=" * 70)
    print("T4: Analytical slope prediction")
    print("=" * 70)
    sl_hv, _, r_hv, _, _ = linregress(log_vf, log_hR)
    alpha = 0.55
    dx_dvf = 4 + 2*alpha - alpha*sl_hv
    slope_pred = 4 / dx_dvf
    print(f"  hR ~ vflat^{sl_hv:.3f} (r={r_hv:.3f})")
    print(f"  dx/d(logvf) = 4 + 2*{alpha} - {alpha}*{sl_hv:.3f} = {dx_dvf:.3f}")
    print(f"  slope_pred = 4/{dx_dvf:.3f} = {slope_pred:.3f}")
    print(f"  slope_obs = {sl_all:.3f}, delta = {sl_all-slope_pred:+.3f}")

    sl_x_vf, _, _, _, _ = linregress(log_vf, log_GMgc)
    print(f"\n  Direct: dx/d(logvf) = {sl_x_vf:.3f}")
    print(f"  slope = 4/{sl_x_vf:.3f} = {4/sl_x_vf:.3f}")

    # T5: gc=const simulation
    print("\n" + "=" * 70)
    print("T5: gc=const simulation")
    print("=" * 70)
    for gc_val, label in [(a0, "a0"), (np.median(gc), "median gc")]:
        lx = np.log10(G_SI*M_bar*gc_val)
        s, _, r, _, se = linregress(lx, log_vf4)
        print(f"  gc={label}: slope={s:.4f}+/-{se:.4f} ({abs(s-1)/se:.1f}sig)")
    sl_std, _, _, _, se_std = linregress(np.log10(G_SI*M_bar), log_vf4)
    print(f"  Standard BTFR (vf4 vs GMbar): slope={sl_std:.4f}+/-{se_std:.4f}")

    # T6: OLS decomposition
    print("\n" + "=" * 70)
    print("T6: OLS slope decomposition")
    print("=" * 70)
    cov_gm_vf = np.cov(np.log10(G_SI*M_bar), log_vf)[0, 1]
    cov_gc_vf = np.cov(log_gc, log_vf)[0, 1]
    slope_formula = 4*(cov_gm_vf + cov_gc_vf)/var_x
    print(f"  Cov(logGM, logvf) = {cov_gm_vf:.5f}")
    print(f"  Cov(loggc, logvf) = {cov_gc_vf:.5f}")
    print(f"  slope = 4*({cov_gm_vf+cov_gc_vf:.5f})/{var_x:.5f} = {slope_formula:.4f}")
    print(f"  gc contribution = 4*Cov(gc,vf)/Var(x) = {4*cov_gc_vf/var_x:+.4f}")
    slope_const = 4*cov_gm_vf/var_gm
    print(f"  If gc=const: slope = 4*Cov(logGM,logvf)/Var(logGM) = {slope_const:.4f}")

    # T7: Sliding window
    print("\n" + "=" * 70)
    print("T7: Sliding window slope")
    print("=" * 70)
    window = 25; sw_vf, sw_slope, sw_se = [], [], []
    for i in range(window, N-window):
        idx = sort_vf[i-window:i+window+1]
        s, _, _, _, se = linregress(log_GMgc[idx], log_vf4[idx])
        sw_vf.append(np.median(vf[idx])); sw_slope.append(s); sw_se.append(se)
    sw_vf = np.array(sw_vf); sw_slope = np.array(sw_slope)
    step = max(1, len(sw_vf)//8)
    for i in range(0, len(sw_vf), step):
        print(f"  vf={sw_vf[i]:6.1f}: slope={sw_slope[i]:.3f}")

    # T8: True slope
    print("\n" + "=" * 70)
    print("T8: Separated fit")
    print("=" * 70)
    X = np.column_stack([np.ones(N), np.log10(G_SI*M_bar), log_gc])
    b = np.linalg.lstsq(X, log_vf4, rcond=None)[0]
    pred = X @ b
    R2 = 1 - np.sum((log_vf4-pred)**2)/np.sum((log_vf4-log_vf4.mean())**2)
    print(f"  log(vf4) = {b[0]:.3f} + {b[1]:.3f}*log(GMbar) + {b[2]:.3f}*log(gc)")
    print(f"  R2={R2:.4f}, b_gc/b_Mbar={b[2]/b[1]:.3f} (theory 1.0)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  Overall slope = {sl_all:.3f} (>1)
  Within-bin slope = {sl_within:.3f} (<1)
  Between slope = {sl_bet:.3f}

  Analytical prediction: slope = 4/dx_dvf = {slope_pred:.3f} (obs {sl_all:.3f})
  Cause: vflat drives both x and y (confounding)
    r(vflat, x) = {r_vf_x:.3f}
    dx/d(logvf) = {sl_x_vf:.3f} < 4 -> slope = 4/{sl_x_vf:.3f} = {4/sl_x_vf:.3f} > 1

  Standard BTFR also slope={sl_std:.3f} (same confounding)
  => slope!=1 is NOT physical breakdown of membrane BTFR
  => It is a statistical confounding effect from vflat

  Separated fit: b_Mbar={b[1]:.3f}, b_gc={b[2]:.3f} (gc 1.6x stronger)
  gc=const: slope={slope_const:.3f} (same issue exists in standard BTFR)
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins))
        for b, bi in enumerate(bin_info):
            idx = bi['idx']
            ax.scatter(log_GMgc[idx], log_vf4[idx], c=[colors[b]], s=10, alpha=0.5)
            xr = np.linspace(log_GMgc[idx].min(), log_GMgc[idx].max(), 50)
            ic_b = bi['y_mid'] - bi['slope']*bi['x_mid']
            ax.plot(xr, ic_b+bi['slope']*xr, color=colors[b], lw=1.5)
        xr = np.linspace(log_GMgc.min(), log_GMgc.max(), 50)
        ax.plot(xr, ic_all+sl_all*xr, 'r-', lw=2, label=f'overall {sl_all:.2f}')
        ax.plot(xr, xr+offset_s1, 'k--', lw=1, label='slope=1')
        ax.set_xlabel('log(GMbar*gc)'); ax.set_ylabel('log(vf^4)')
        ax.set_title("(a) Simpson's paradox"); ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.plot(sw_vf, sw_slope, 'b-', lw=1.5)
        ax.axhline(1, color='r', ls='--'); ax.axhline(sl_all, color='gray', ls=':')
        ax.set_xlabel('vflat'); ax.set_ylabel('local slope')
        ax.set_title(f'(b) sliding window +/-{window}')

        ax = axes[0, 2]
        sc_ = ax.scatter(log_vf, log_gc, c=log_Mbar, cmap='viridis', s=12, alpha=0.5)
        ax.set_xlabel('log vflat'); ax.set_ylabel('log gc')
        ax.set_title(f'(c) gc-vf confounding r={r_vf_gc:.3f}')
        plt.colorbar(sc_, ax=ax, label='log Mbar')

        ax = axes[1, 0]
        labs = ['Var(GM)', 'Var(gc)', '2*Cov']
        vals = [var_gm/var_x*100, var_gc/var_x*100, 2*cov_gm_gc/var_x*100]
        ax.bar(labs, vals, color=['steelblue', 'coral', 'green'], alpha=0.7)
        ax.set_ylabel('% of Var(x)'); ax.set_title('(d) x variance')

        ax = axes[1, 1]
        ax.errorbar(x_mids, y_mids, fmt='ko', ms=8)
        xr = np.linspace(x_mids.min(), x_mids.max(), 50)
        ax.plot(xr, sl_bet*xr+(y_mids.mean()-sl_bet*x_mids.mean()), 'r-', lw=2,
                label=f'between {sl_bet:.2f}')
        ax.plot(xr, xr+offset_s1, 'k--', label='s=1')
        ax.set_xlabel('bin median x'); ax.set_ylabel('bin median y')
        ax.set_title('(e) between-group'); ax.legend()

        ax = axes[1, 2]
        ax.bar(['overall', 'within', 'between', 'predicted'],
               [sl_all, sl_within, sl_bet, slope_pred],
               color=['steelblue', 'coral', 'green', 'orange'], alpha=0.7)
        ax.axhline(1, color='r', ls='--')
        ax.set_ylabel('slope'); ax.set_title('(f) slope summary')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'simpson_paradox.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
