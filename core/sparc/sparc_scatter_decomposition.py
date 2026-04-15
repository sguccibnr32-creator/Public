#!/usr/bin/env python3
"""sparc_scatter_decomposition.py (TA3+phase1+MRT adapted)"""
import os, csv, warnings
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
a0 = 1.2e-10; kpc_m = 3.086e19
ETA0 = 0.584; BETA_YD = -0.361; ALPHA = 0.5

def gc_c15_log(log_vf, log_hR, log_Yd):
    log_vf_SI = log_vf + 3; log_hR_SI = log_hR + np.log10(kpc_m)
    log_Sd = 2*log_vf_SI - log_hR_SI
    return np.log10(ETA0) + BETA_YD*log_Yd + ALPHA*(np.log10(a0) + log_Sd)

def load_data():
    pipe = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try: pipe[n] = {'vflat': float(row.get('vflat', '0')), 'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                g = float(row.get('gc_over_a0', '0'))
                if n in pipe and g > 0: pipe[n]['gc'] = g * a0
            except: pass
    mrt = {}; in_data = False; sep = 0
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
                mrt[p[0]] = {'Rdisk': float(p[11]), 'e_Vf': float(p[16]),
                              'T': int(p[1]), 'Q': int(p[17])}
            except: continue
    return pipe, mrt

def main():
    print("=" * 70)
    print("C15 scatter decomposition: measurement vs intrinsic")
    print("=" * 70)
    pipe, mrt = load_data()
    names = sorted([n for n in pipe if 'gc' in pipe[n] and n in mrt and mrt[n].get('Rdisk', 0) > 0])
    N = len(names); print(f"N={N}")

    gc = np.array([pipe[n]['gc'] for n in names])
    vf = np.array([pipe[n]['vflat'] for n in names])
    Yd = np.array([pipe[n]['Yd'] for n in names])
    hR = np.array([mrt[n]['Rdisk'] for n in names])
    e_vf_mrt = np.array([mrt[n].get('e_Vf', 0) for n in names])
    T_type = np.array([mrt[n].get('T', np.nan) for n in names])
    Q = np.array([mrt[n].get('Q', 3) for n in names])

    lgc = np.log10(gc); lvf = np.log10(vf); lhr = np.log10(hR); lyd = np.log10(Yd)
    lgc_pred = gc_c15_log(lvf, lhr, lyd)
    resid = lgc - lgc_pred
    scatter_total = np.std(resid)
    print(f"Baseline: scatter={scatter_total:.4f}, bias={np.mean(resid):+.4f}")

    # T1: Error propagation
    print("\n" + "=" * 60)
    print("T1: Error propagation")
    print("=" * 60)
    # Use MRT e_Vflat where available, else 7%
    frac_vf = np.where(e_vf_mrt > 0, e_vf_mrt/vf, 0.07)
    frac_hr = 0.15  # typical photometric
    frac_yd = 0.40  # SPS
    frac_gc = 0.35  # RAR fit

    s_lvf = frac_vf / np.log(10)
    s_lhr = frac_hr / np.log(10)
    s_lyd = frac_yd / np.log(10)
    s_lgc = frac_gc / np.log(10)

    s2_pred = BETA_YD**2*s_lyd**2 + ALPHA**2*(4*s_lvf**2 + s_lhr**2)
    s2_obs = s_lgc**2
    s2_meas = s2_pred + s2_obs
    s_meas = np.sqrt(np.mean(s2_meas))

    s_vf_c = ALPHA*2*np.sqrt(np.mean(s_lvf**2))
    s_hr_c = ALPHA*np.sqrt(np.mean(s_lhr**2))
    s_yd_c = abs(BETA_YD)*np.sqrt(np.mean(s_lyd**2))
    s_gc_c = np.sqrt(np.mean(s2_obs))

    print(f"  vflat: median frac={np.median(frac_vf)*100:.1f}%, sigma contrib={s_vf_c:.4f}")
    print(f"  hR:    frac={frac_hr*100:.0f}%, sigma contrib={s_hr_c:.4f}")
    print(f"  Yd:    frac={frac_yd*100:.0f}%, sigma contrib={s_yd_c:.4f}")
    print(f"  gc_obs: frac={frac_gc*100:.0f}%, sigma contrib={s_gc_c:.4f}")
    print(f"  Total meas: {s_meas:.4f}")

    # T2: Intrinsic
    print("\n" + "=" * 60)
    print("T2: Intrinsic scatter")
    print("=" * 60)
    s2_intr = max(0, scatter_total**2 - s_meas**2)
    s_intr = np.sqrt(s2_intr)
    print(f"  scatter^2_total = {scatter_total**2:.6f}")
    print(f"  scatter^2_meas  = {s_meas**2:.6f}")
    print(f"  scatter^2_intr  = {s2_intr:.6f}")
    print(f"  scatter_intr    = {s_intr:.4f}")
    print(f"  Intrinsic fraction: {s2_intr/scatter_total**2*100:.1f}%")

    # T3: Sensitivity
    print("\n" + "=" * 60)
    print("T3: Sensitivity analysis")
    print("=" * 60)
    scenarios = [("Optimistic", 0.03, 0.08, 0.20, 0.15),
                  ("SPARC typical", 0.05, 0.10, 0.30, 0.25),
                  ("Default", np.median(frac_vf), frac_hr, frac_yd, frac_gc),
                  ("Conservative", 0.10, 0.20, 0.50, 0.40),
                  ("Pessimistic", 0.15, 0.30, 0.60, 0.50)]
    print(f"  {'scenario':<16s} {'s_meas':>8s} {'s_intr':>8s} {'%intr':>7s}")
    for nm, fv, fh, fy, fg in scenarios:
        slv = fv/np.log(10); slh = fh/np.log(10); sly = fy/np.log(10); slg = fg/np.log(10)
        s2p = BETA_YD**2*sly**2 + ALPHA**2*(4*slv**2 + slh**2)
        s2m = s2p + slg**2; sm = np.sqrt(s2m)
        s2i = max(0, scatter_total**2 - s2m); si = np.sqrt(s2i)
        print(f"  {nm:<16s} {sm:8.4f} {si:8.4f} {s2i/scatter_total**2*100:7.1f}")

    # T4: Rotmod comparison
    print("\n" + "=" * 60)
    print("T4: Rotmod lower bound")
    print("=" * 60)
    sr = 0.244
    print(f"  scatter_total={scatter_total:.4f}, scatter_Rotmod={sr:.4f}")
    print(f"  global-vs-local = sqrt(total^2-rotmod^2) = {np.sqrt(max(0,scatter_total**2-sr**2)):.4f}")

    # T5: MC
    print("\n" + "=" * 60)
    print("T5: Monte Carlo (1000x)")
    print("=" * 60)
    np.random.seed(42)
    mc_sc = []
    evf = frac_vf * vf; ehr = frac_hr * hR; eyd = frac_yd * Yd; egc = frac_gc * gc
    for _ in range(1000):
        vn = np.maximum(vf + np.random.normal(0, evf), 1)
        hn = np.maximum(hR + np.random.normal(0, ehr), 0.01)
        yn = np.maximum(Yd + np.random.normal(0, eyd), 0.01)
        gn = np.maximum(gc + np.random.normal(0, egc), 1e-15)
        lp = gc_c15_log(np.log10(vn), np.log10(hn), np.log10(yn))
        mc_sc.append(np.std(np.log10(gn) - lp))
    mc_sc = np.array(mc_sc)
    s_noise_mc = np.sqrt(max(0, np.median(mc_sc)**2 - scatter_total**2))
    print(f"  Noiseless: {scatter_total:.4f}, MC median: {np.median(mc_sc):.4f}")
    print(f"  MC noise: {s_noise_mc:.4f} (analytic: {s_meas:.4f})")

    # T6: chi2
    print("\n" + "=" * 60)
    print("T6: chi^2 analysis")
    print("=" * 60)
    s_pg = np.sqrt(s2_meas)
    chi2_red = np.mean((resid/s_pg)**2)
    print(f"  chi^2/N = {chi2_red:.3f}")
    if chi2_red > 1:
        try:
            s_int = brentq(lambda si: np.mean(resid**2/(s2_meas + si**2)) - 1, 0, 1)
            print(f"  Optimal sigma_intrinsic (chi2/N=1): {s_int:.4f}")
        except:
            print("  Could not find optimal sigma_intrinsic")
    else:
        print("  chi2/N < 1: measurement errors explain all scatter")

    # T7: By type
    print("\n" + "=" * 60)
    print("T7: By galaxy type")
    print("=" * 60)
    for lbl, mask in [("T<=3", T_type <= 3), ("T=4-7", (T_type >= 4) & (T_type <= 7)), ("T>=8", T_type >= 8)]:
        mt = mask & np.isfinite(T_type)
        if mt.sum() < 10: continue
        st = np.std(resid[mt]); sm_ = np.sqrt(np.mean(s2_meas[mt]))
        si_ = np.sqrt(max(0, st**2-sm_**2))
        print(f"  {lbl}: N={mt.sum()}, total={st:.4f}, meas={sm_:.4f}, intr={si_:.4f}")

    # T8: By quality
    print("\n" + "=" * 60)
    print("T8: By quality flag")
    print("=" * 60)
    for q in [1, 2, 3]:
        mq = Q == q
        if mq.sum() < 10: continue
        st = np.std(resid[mq]); sm_ = np.sqrt(np.mean(s2_meas[mq]))
        si_ = np.sqrt(max(0, st**2-sm_**2))
        print(f"  Q={q}: N={mq.sum()}, total={st:.4f}, meas={sm_:.4f}, intr={si_:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  scatter_total   = {scatter_total:.4f} dex (100%)
  scatter_meas    = {s_meas:.4f} dex ({s_meas**2/scatter_total**2*100:.0f}%)
  scatter_intrinsic = {s_intr:.4f} dex ({s2_intr/scatter_total**2*100:.0f}%)

  Components (sigma^2 basis):
    vflat: {s_vf_c**2/scatter_total**2*100:.1f}%
    hR:    {s_hr_c**2/scatter_total**2*100:.1f}%
    Yd:    {s_yd_c**2/scatter_total**2*100:.1f}%
    gc:    {s_gc_c**2/scatter_total**2*100:.1f}%
    intrinsic: {s2_intr/scatter_total**2*100:.1f}%
""")

    # Figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes[0, 0]
        comps = {'vflat': s_vf_c**2, 'hR': s_hr_c**2, 'Yd': s_yd_c**2, 'gc': s_gc_c**2, 'intrinsic': s2_intr}
        ax.pie([v/sum(comps.values())*100 for v in comps.values()],
               labels=[f'{k}\n{v/sum(comps.values())*100:.0f}%' for k, v in comps.items()],
               colors=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'])
        ax.set_title(f'(a) scatter budget ({scatter_total:.3f} dex)')

        ax = axes[0, 1]
        ax.scatter(s_pg, np.abs(resid), s=12, alpha=0.5)
        ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5)
        ax.set_xlabel('sigma_meas'); ax.set_ylabel('|resid|')
        ax.set_title('(b) meas err vs |resid|')

        ax = axes[0, 2]
        ax.hist(mc_sc, bins=40, color='steelblue', alpha=0.7)
        ax.axvline(scatter_total, color='red', ls='--', label=f'noiseless {scatter_total:.4f}')
        ax.axvline(np.median(mc_sc), color='green', label=f'MC {np.median(mc_sc):.4f}')
        ax.set_xlabel('scatter'); ax.set_title('(c) MC noise'); ax.legend(fontsize=7)

        ax = axes[1, 0]
        labs = ['vflat', 'hR', 'Yd', 'gc', 'intrinsic']
        vals = [s_vf_c, s_hr_c, s_yd_c, s_gc_c, s_intr]
        ax.bar(labs, vals, color=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'])
        ax.axhline(scatter_total, color='red', ls='--')
        ax.set_ylabel('sigma'); ax.set_title('(d) components')
        for i, v in enumerate(vals): ax.text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=8)

        ax = axes[1, 1]
        for nm, fv, fh, fy, fg in scenarios:
            slv = fv/np.log(10); slh = fh/np.log(10); sly = fy/np.log(10); slg = fg/np.log(10)
            s2p = BETA_YD**2*sly**2 + ALPHA**2*(4*slv**2+slh**2)
            s2m = s2p+slg**2; si = np.sqrt(max(0, scatter_total**2-s2m))
            ax.scatter(np.sqrt(s2m), si, s=60, label=nm[:8])
        ax.set_xlabel('sigma_meas'); ax.set_ylabel('sigma_intrinsic')
        ax.set_title('(e) sensitivity'); ax.legend(fontsize=6)

        ax = axes[1, 2]
        sc_ = ax.scatter(lgc, resid, c=s_pg, cmap='YlOrRd', s=15, alpha=0.6, vmin=0, vmax=0.4)
        ax.axhline(0, ls='--', color='grey')
        ax.set_xlabel('log gc'); ax.set_ylabel('resid')
        ax.set_title('(f) resid colored by sigma_meas')
        plt.colorbar(sc_, ax=ax, label='sigma_meas')

        plt.suptitle('Scatter Decomposition', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'scatter_decomposition.png'), dpi=150)
        print("Figure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == "__main__":
    main()
