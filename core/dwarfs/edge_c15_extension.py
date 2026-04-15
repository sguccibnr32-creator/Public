# -*- coding: utf-8 -*-
"""
EDGE dwarf sample (Julio+2025, A&A 703, A106) extension of C15.

EDGE 12 dwarfs (8 classical dSph + 4 MUSE-Faint): Table 1 of Julio+2025
plus velocity dispersions from literature compilations.

Virial estimator (Wolf+2010):
    M(r_1/2) = 4 * sigma_los^2 * r_1/2 / G
    g_obs(r_1/2) = G * M / r_1/2^2 = 4 * sigma^2 / r_1/2
    g_bar(r_1/2) = G * (M_star/2 + M_gas/2) / r_1/2^2

C15 in rotation-supported form:
    gc = 0.584 * Yd^{-0.361} * sqrt(a0 * vflat^2 / hR)
For dispersion-supported systems, substitute:
    vflat^2 -> sigma_los^2 (or 3*sigma^2 for 3D)
    hR -> r_1/2
    Yd (=M*/L_3.6um) -> M*/L_V (dSph use V-band)
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Constants ----------
G = 6.674e-11                    # m^3 kg^-1 s^-2
G_pc = 4.30091e-3                # pc M_sun^-1 (km/s)^2
a0 = 1.2e-10                     # m/s^2 (MOND)
Msun = 1.989e30                  # kg
pc = 3.0857e16                   # m
kpc_m = 3.086e19

BASE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン")
TA3  = BASE / "TA3_gc_independent.csv"
PHASE1 = BASE / "phase1" / "sparc_results.csv"
OUT_CSV = BASE / "edge_c15_extension.csv"
OUT_PNG = BASE / "edge_c15_extension.png"

# ---------- EDGE Table 1 (Julio+2025) + sigma_los literature ----------
# sigma_los [km/s], r_1/2 [pc], M_star [1e6 Msun], M_gas [1e6 Msun]
EDGE = [
    # name,        sigma, s_err, r12_pc, r12_err, Mstar_6, Mgas_6, Mv,   D_kpc
    ("Carina",     6.6,   1.2,   250,   39,   0.38,  0.0,   -9.1,  106),
    ("Fornax",    11.7,   0.9,   710,   77,   43.0,  0.0,  -13.4,  138),
    ("Draco",      9.1,   1.2,   221,   19,   0.29,  0.0,   -8.8,   76),
    ("Leo I",      9.2,   1.4,   251,   27,   5.50,  0.0,  -12.0,  254),
    ("Leo II",     6.6,   0.7,   176,   42,   0.74,  0.0,   -9.8,  233),
    ("Sculptor",   9.2,   1.4,   283,   45,   2.30,  0.0,  -11.1,   86),
    ("Sextans",    7.9,   1.3,   695,   44,   0.44,  0.0,   -9.3,   86),
    ("Ursa Minor", 9.5,   1.2,   181,   27,   0.29,  0.0,   -8.8,   76),
    ("Antlia B",   5.8,   1.6,   273,   29,   0.76,  0.28,  -9.7, 1350),
    ("Eridanus II",10.3,  3.5,   277,   14,   0.09,  0.0,   -7.1,  366),
    ("Grus 1",     2.5,   1.0,   151,   25,   0.01,  0.0,   -4.1,  125),
    ("Leo T",      7.5,   1.6,   115,   17,   0.14,  0.41,  -7.6,  409),
]

# ---------- Virial accelerations ----------
def virial_accelerations(sigma_kms, r12_pc, Mstar_6, Mgas_6):
    """Wolf+2010: g_obs = 4*sigma^2/r_1/2 ; g_bar = G*M_bar(r12)/r12^2.
    Assume mass within r_1/2 is half of total stellar + gas mass."""
    # SI conversion
    sigma = sigma_kms * 1e3          # m/s
    r12   = r12_pc * pc              # m
    # g_obs (Wolf virial)
    g_obs = 4.0 * sigma**2 / r12      # m/s^2
    # g_bar (half of Mstar + Mgas enclosed at r_1/2 by definition of half-light)
    Mbar_half = 0.5 * (Mstar_6 + Mgas_6) * 1e6 * Msun
    g_bar = G * Mbar_half / r12**2
    return g_obs, g_bar

# ---------- MOND / McGaugh RAR prediction ----------
def rar_mcgaugh(g_bar):
    """Lelli+2017 RAR fit: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))

# ---------- C15 virial reformulation ----------
def c15_virial(sigma_kms, r12_pc, Mstar_6):
    """Analog of C15 for dispersion-supported systems.
       gc_pred = eta0 * (Mstar/M0)^beta * sqrt(a0 * sigma^2 / r_1/2)
       eta0 = 0.584 (SPARC C15), beta = -0.361 (absorbing Yd dependence)
       Use Mstar as Yd proxy via Mstar ~ Yd * L; for fixed M/L variation,
       Mstar^beta = (Yd*L)^beta gives extra L^beta term.  For first-pass
       use dimensional scaling only."""
    sigma_m = sigma_kms * 1e3
    r12_m = r12_pc * pc
    # Yd proxy: SPARC median Yd = 0.5.  For dSph V-band, M*/L_V ~ 1-3.
    # Use normalized Mstar as rough proxy (larger Mstar -> older, more plastic?)
    # Simplest: sqrt(a0 * sigma^2 / r12) with eta=0.584
    g_scale = math.sqrt(a0 * sigma_m**2 / r12_m)
    return 0.584 * g_scale

# ---------- SPARC overlay ----------
def load_sparc_rar():
    """Load SPARC galaxy gc and vflat/rdisk to compute g_bar analog."""
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                data[n] = {
                    'vflat': float(row.get('vflat', '0')),
                    'Yd':    float(row.get('ud', '0.5')),
                }
            except:
                pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                rs = float(row.get('rs_tanh', '0'))
                if n in data and gc_a0 > 0 and rs > 0:
                    data[n]['gc']  = gc_a0 * a0
                    data[n]['hR']  = rs
                    # v^2/R (dynamical acceleration) as g_obs proxy
                    v_m = data[n]['vflat'] * 1e3
                    R_m = rs * kpc_m
                    data[n]['gbar_dyn'] = v_m**2 / R_m
            except:
                pass
    return {k: v for k, v in data.items() if 'gc' in v}

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EDGE dwarf sample C15 extension (N=12)")
    print("="*60)

    results = []
    for row in EDGE:
        name, s, serr, r12, r12e, Ms, Mg, Mv, D = row
        g_obs, g_bar = virial_accelerations(s, r12, Ms, Mg)
        gc_c15_pred = c15_virial(s, r12, Ms)
        g_rar = rar_mcgaugh(np.array([g_bar]))[0]
        results.append({
            'name':    name, 'sigma':  s,  'r12_pc': r12, 'Mstar_6': Ms,
            'Mgas_6':  Mg, 'Mv': Mv, 'D_kpc': D,
            'g_obs':   g_obs, 'g_bar':  g_bar,
            'g_obs_a0': g_obs/a0, 'g_bar_a0': g_bar/a0,
            'gc_c15_pred': gc_c15_pred,
            'gc_c15_a0':   gc_c15_pred/a0,
            'g_rar_pred':  g_rar,
            'g_rar_a0':    g_rar/a0,
            'ratio_obs_over_rar': g_obs/g_rar,
            'ratio_obs_over_c15': g_obs/gc_c15_pred,
        })

    # Print table
    hdr = f"{'name':14s} {'sig':>5s} {'r12':>5s} {'M*6':>6s}  {'lg_gbar':>8s} {'lg_gobs':>8s} {'gobs/a0':>8s} {'gRAR/gobs':>10s} {'gC15/gobs':>10s}"
    print(hdr)
    for r in results:
        print(f"{r['name']:14s} {r['sigma']:5.1f} {r['r12_pc']:5.0f} {r['Mstar_6']:6.2f}  "
              f"{math.log10(r['g_bar']):8.2f} {math.log10(r['g_obs']):8.2f} "
              f"{r['g_obs_a0']:8.4f} {1.0/r['ratio_obs_over_rar']:10.3f} {1.0/r['ratio_obs_over_c15']:10.3f}")

    # Save CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nCSV saved: {OUT_CSV}")

    # ---------- Statistics ----------
    gobs = np.array([r['g_obs'] for r in results])
    gbar = np.array([r['g_bar'] for r in results])
    gRAR = np.array([r['g_rar_pred'] for r in results])
    gC15 = np.array([r['gc_c15_pred'] for r in results])

    # log residuals
    res_RAR = np.log10(gobs / gRAR)
    res_C15 = np.log10(gobs / gC15)
    print("\n--- Residual statistics (log10 g_obs/g_pred) ---")
    print(f"vs McGaugh RAR   : median={np.median(res_RAR):+.3f}  scatter={np.std(res_RAR):.3f}")
    print(f"vs C15 virial    : median={np.median(res_C15):+.3f}  scatter={np.std(res_C15):.3f}")

    # Deep-MOND regime (g_bar << a0)
    deep = gbar < 0.01 * a0
    print(f"\nDeep-MOND subsample (g_bar < 0.01 a0): N={deep.sum()}")
    if deep.sum() > 3:
        print(f"  median g_obs/g_RAR = {np.median(gobs[deep]/gRAR[deep]):.3f}")
        print(f"  median g_obs/g_C15 = {np.median(gobs[deep]/gC15[deep]):.3f}")

    # Fornax outlier check (paper reports Fornax below RAR)
    idx_fornax = [i for i,r in enumerate(results) if r['name']=='Fornax'][0]
    print(f"\nFornax: g_obs/g_RAR = {gobs[idx_fornax]/gRAR[idx_fornax]:.3f} "
          f"(paper reports systematic deviation)")

    # Gas-bearing vs gas-less (S_plastic analog: f_gas > 0)
    has_gas = np.array([r['Mgas_6']>0 for r in results])
    print(f"\nGas-bearing (Antlia B, Leo T): N={has_gas.sum()}")
    print(f"  median g_obs/g_RAR = {np.median(gobs[has_gas]/gRAR[has_gas]):.3f}")
    print(f"Gas-less (dSph):         N={(~has_gas).sum()}")
    print(f"  median g_obs/g_RAR = {np.median(gobs[~has_gas]/gRAR[~has_gas]):.3f}")

    # ---------- Plot ----------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: RAR
    ax = axs[0]
    # SPARC overlay (g_bar_dyn, gc)
    sparc = load_sparc_rar()
    sp_gbar = np.array([v['gbar_dyn'] for v in sparc.values()])
    sp_gc   = np.array([v['gc']       for v in sparc.values()])
    ax.loglog(sp_gbar/a0, sp_gc/a0, '.', color='lightgray', ms=3,
              label=f'SPARC N={len(sparc)} (v^2/R, gc)')

    # EDGE points
    for r in results:
        marker = 's' if r['Mgas_6']>0 else 'o'
        col = 'red' if r['name']=='Fornax' else ('blue' if r['Mgas_6']>0 else 'black')
        ax.errorbar(r['g_bar_a0'], r['g_obs_a0'],
                    xerr=0, yerr=r['g_obs_a0']*2*0.2,
                    fmt=marker, color=col, ms=7, capsize=2)
        ax.annotate(r['name'][:6], (r['g_bar_a0'], r['g_obs_a0']),
                    xytext=(4,4), textcoords='offset points', fontsize=7)

    # RAR curve
    gb_range = np.logspace(-6, 2, 200) * a0
    ax.loglog(gb_range/a0, rar_mcgaugh(gb_range)/a0, 'g-', lw=1.5,
              label='McGaugh RAR')
    ax.loglog(gb_range/a0, gb_range/a0, 'k--', lw=0.7, label='1:1 (Newtonian)')
    ax.loglog(gb_range/a0, np.sqrt(gb_range*a0)/a0, 'b:', lw=0.7, label='MOND deep')

    ax.set_xlabel('g_bar / a0')
    ax.set_ylabel('g_obs / a0')
    ax.set_title('EDGE dwarfs + SPARC in RAR space')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-6, 1e2)
    ax.set_ylim(1e-2, 1e2)

    # Panel 2: g_obs/g_RAR vs log M_star
    ax = axs[1]
    Ms_arr = np.array([r['Mstar_6'] for r in results])
    ratio = gobs/gRAR
    for i, r in enumerate(results):
        marker = 's' if r['Mgas_6']>0 else 'o'
        col = 'red' if r['name']=='Fornax' else ('blue' if r['Mgas_6']>0 else 'black')
        ax.semilogx(Ms_arr[i]*1e6, ratio[i], marker, color=col, ms=8)
        ax.annotate(r['name'][:6], (Ms_arr[i]*1e6, ratio[i]),
                    xytext=(4,4), textcoords='offset points', fontsize=7)
    ax.axhline(1.0, color='g', ls='-', lw=1, label='RAR match')
    ax.set_xlabel('M_star [Msun]')
    ax.set_ylabel('g_obs / g_RAR')
    ax.set_title('Deviation from McGaugh RAR vs stellar mass')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"\nPlot saved: {OUT_PNG}")

if __name__ == '__main__':
    main()
