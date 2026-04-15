# -*- coding: utf-8 -*-
"""
phase_a_step5_rar_v2.py

Step 5 v2: HSC ESD -> RAR with g_bar-axis + per-lens M*.
"""

import os, sys, time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy import stats
from astropy.table import Table

OUTPUT_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase_a_output")
BROUWER_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\brouwer2021")
LENS_FILE = OUTPUT_DIR / 'gama_lenses_g12_isolated.fits'
HSC_FILE = Path(r"E:\スバル望遠鏡データ\931720.csv.gz.1")

LENS_COLS = {
    'id':   'uberID',
    'ra':   'RAcen',
    'dec':  'Deccen',
    'z':    'Z',
    'logm': 'logmstar',
}

HSC_COLS = {
    'ra':     'i_ra',
    'dec':    'i_dec',
    'e1':     'i_hsmshaperegauss_e1',
    'e2':     'i_hsmshaperegauss_e2',
    'weight': 'i_hsmshaperegauss_derived_weight',
    'zbin':   'hsc_y3_zbin',
}

N_GBINS = 15
GBAR_MIN = 1e-16
GBAR_MAX = 1e-10
GBAR_EDGES = np.logspace(np.log10(GBAR_MIN), np.log10(GBAR_MAX), N_GBINS + 1)
GBAR_MID = np.sqrt(GBAR_EDGES[:-1] * GBAR_EDGES[1:])

R_MIN_MPC = 0.02
R_MAX_MPC = 3.0

MSTAR_BINS = [8.5, 10.3, 10.6, 10.8, 11.0, 11.5]
N_MBINS = len(MSTAR_BINS) - 1

ZBIN_ZEFF = {0: 0.0, 1: 0.44, 2: 0.75, 3: 1.01, 4: 1.30}
DZ_MIN = 0.1

G12_RA_MIN, G12_RA_MAX = 172.0, 188.0
G12_DEC_MIN, G12_DEC_MAX = -4.0, 3.0

H0 = 70.0
c_km = 2.998e5
c_m = 2.998e8
G_SI = 6.674e-11
Msun_kg = 1.989e30
pc_m = 3.0857e16
Mpc_m = 3.0857e22
Om0 = 0.3

a0 = 1.2e-10
BIAS_HSC = 1.0

REPORT_INTERVAL = 2_000_000


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def comoving_distance(z):
    from scipy.integrate import quad
    def integrand(zp):
        return 1.0 / np.sqrt(Om0 * (1 + zp)**3 + (1 - Om0))
    result, _ = quad(integrand, 0, z)
    return (c_km / H0) * result


def angular_diameter_distance(z):
    return comoving_distance(z) / (1 + z)


def sigma_crit(z_l, z_s):
    if z_s <= z_l + 0.01:
        return np.inf
    D_l = angular_diameter_distance(z_l)
    D_s = angular_diameter_distance(z_s)
    chi_l = comoving_distance(z_l)
    chi_s = comoving_distance(z_s)
    D_ls = (chi_s - chi_l) / (1 + z_s)
    if D_ls <= 0 or D_l <= 0:
        return np.inf
    prefactor = c_m**2 / (4 * np.pi * G_SI)
    D_ratio = D_s / (D_l * D_ls)
    Sc_SI = prefactor * D_ratio / Mpc_m
    Sc = Sc_SI / Msun_kg * pc_m**2
    return Sc


def precompute_sigma_crit(z_lens_arr):
    n = len(z_lens_arr)
    sc = np.full((n, 5), np.inf)
    for i in range(n):
        for zb in range(1, 5):
            z_s = ZBIN_ZEFF[zb]
            if z_s > z_lens_arr[i] + DZ_MIN:
                sc[i, zb] = sigma_crit(z_lens_arr[i], z_s)
    return sc


def compute_gbar(logMstar, R_Mpc):
    M_bar = 1.33 * 10**logMstar * Msun_kg
    R_m = R_Mpc * Mpc_m
    return G_SI * M_bar / R_m**2


def hsc_distortion_to_shear(e1, e2):
    return e1 / 2.0, e2 / 2.0


class RARAccumulator:
    def __init__(self):
        self.sum_w = np.zeros(N_GBINS)
        self.sum_w_gt = np.zeros(N_GBINS)
        self.sum_w_gx = np.zeros(N_GBINS)
        self.sum_w_gbar = np.zeros(N_GBINS)
        self.n_pairs = np.zeros(N_GBINS, dtype=np.int64)

        self.mb_sum_w = np.zeros((N_MBINS, N_GBINS))
        self.mb_sum_w_gt = np.zeros((N_MBINS, N_GBINS))
        self.mb_sum_w_gx = np.zeros((N_MBINS, N_GBINS))
        self.mb_sum_w_gbar = np.zeros((N_MBINS, N_GBINS))
        self.mb_n_pairs = np.zeros((N_MBINS, N_GBINS), dtype=np.int64)

    def add(self, g_bar, gamma_t, gamma_x, w_src, sc, mbin):
        if g_bar < GBAR_MIN or g_bar >= GBAR_MAX:
            return
        if not np.isfinite(sc) or sc <= 0 or np.isinf(sc):
            return

        gb_idx = np.searchsorted(GBAR_EDGES, g_bar) - 1
        if gb_idx < 0 or gb_idx >= N_GBINS:
            return

        sc_inv = 1.0 / sc
        w_eff = w_src * sc_inv**2

        self.sum_w[gb_idx] += w_eff
        self.sum_w_gt[gb_idx] += w_eff * gamma_t
        self.sum_w_gx[gb_idx] += w_eff * gamma_x
        self.sum_w_gbar[gb_idx] += w_eff * g_bar
        self.n_pairs[gb_idx] += 1

        if 0 <= mbin < N_MBINS:
            self.mb_sum_w[mbin, gb_idx] += w_eff
            self.mb_sum_w_gt[mbin, gb_idx] += w_eff * gamma_t
            self.mb_sum_w_gx[mbin, gb_idx] += w_eff * gamma_x
            self.mb_sum_w_gbar[mbin, gb_idx] += w_eff * g_bar
            self.mb_n_pairs[mbin, gb_idx] += 1

    def get_rar(self, which='all', mbin=None):
        if which == 'all':
            sw = self.sum_w
            sgt = self.sum_w_gt
            sgx = self.sum_w_gx
            sgb = self.sum_w_gbar
            np_arr = self.n_pairs
        else:
            sw = self.mb_sum_w[mbin]
            sgt = self.mb_sum_w_gt[mbin]
            sgx = self.mb_sum_w_gx[mbin]
            sgb = self.mb_sum_w_gbar[mbin]
            np_arr = self.mb_n_pairs[mbin]

        mean_gt = np.where(sw > 0, sgt / sw, 0)
        mean_gx = np.where(sw > 0, sgx / sw, 0)
        mean_gbar = np.where(sw > 0, sgb / sw, GBAR_MID)

        SC_EFF = 4000.0  # rough h70 Msun/pc^2

        ESD_t = mean_gt * SC_EFF
        GOBS_FACTOR = 4 * G_SI * Msun_kg / pc_m**2  # FIXED: pc_m^2 not pc_m
        g_obs = GOBS_FACTOR * ESD_t / BIAS_HSC

        sigma_e = 0.26
        err_gt = np.where(np_arr > 0, sigma_e / np.sqrt(np_arr.astype(float)), 1.0)
        g_err = GOBS_FACTOR * err_gt * SC_EFF / BIAS_HSC

        return mean_gbar, g_obs, g_err, ESD_t, mean_gx * SC_EFF, np_arr


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4 * gc * g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def c15_gc(logMstar, Yd=0.5):
    import math
    kpc_m = 3.0857e19
    Mbar = 10**logMstar * 1.4
    vflat = (G_SI * a0 * Mbar * Msun_kg)**0.25 / 1e3
    hR = 10**(0.35 * (logMstar - 10.0) + 0.5)
    return 0.584 * Yd**(-0.361) * math.sqrt(a0 * (vflat * 1e3)**2 / (hR * kpc_m))


def fit_gc_robust(g_bar, g_obs, g_err):
    valid = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_obs) & (g_err > 0)
    gb, go, ge = g_bar[valid], g_obs[valid], g_err[valid]

    if len(gb) < 3:
        return np.nan

    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(gb, gc)
        res = np.log10(go) - np.log10(np.maximum(gp, 1e-30))
        w = 1.0 / np.maximum((ge / (go * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)

    res1 = minimize_scalar(chi2, bounds=(-13, -9), method='bounded')
    gc1 = 10**res1.x

    gp1 = membrane_rar(gb, gc1)
    log_res = np.log10(go) - np.log10(np.maximum(gp1, 1e-30))
    med_res = np.median(log_res)
    mad_res = np.median(np.abs(log_res - med_res)) * 1.4826
    if mad_res < 0.01:
        mad_res = 0.3

    keep = np.abs(log_res - med_res) < 3 * mad_res
    if keep.sum() < 3:
        return gc1

    gb2, go2, ge2 = gb[keep], go[keep], ge[keep]

    def chi2_clip(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(gb2, gc)
        res = np.log10(go2) - np.log10(np.maximum(gp, 1e-30))
        w = 1.0 / np.maximum((ge2 / (go2 * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)

    res2 = minimize_scalar(chi2_clip, bounds=(-13, -9), method='bounded')
    return 10**res2.x


def main():
    t_start = time.time()
    section("STEP 5 v2: HSC LENSING RAR (g_bar-axis, per-lens M*)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lenses = Table.read(LENS_FILE)
    n_lens = len(lenses)
    lens_ra = np.array(lenses[LENS_COLS['ra']], dtype=np.float64)
    lens_dec = np.array(lenses[LENS_COLS['dec']], dtype=np.float64)
    lens_z = np.array(lenses[LENS_COLS['z']], dtype=np.float64)
    lens_logm = np.array(lenses[LENS_COLS['logm']], dtype=np.float64)

    lens_mbin = np.full(n_lens, -1, dtype=int)
    for mb in range(N_MBINS):
        mask = (lens_logm >= MSTAR_BINS[mb]) & (lens_logm < MSTAR_BINS[mb + 1])
        lens_mbin[mask] = mb

    print(f"  Lenses: {n_lens:,}")
    print(f"  M* bins: {[int((lens_mbin==mb).sum()) for mb in range(N_MBINS)]}")

    print(f"  Pre-computing Sigma_crit...", flush=True)
    sc_table = precompute_sigma_crit(lens_z)

    cos_dec_ref = np.cos(np.radians(np.median(lens_dec)))
    lens_xy = np.column_stack([lens_ra * cos_dec_ref, lens_dec])
    from scipy.spatial import cKDTree
    lens_tree = cKDTree(lens_xy)

    D_A_min = angular_diameter_distance(np.min(lens_z))
    theta_max_deg = np.degrees(R_MAX_MPC / D_A_min)
    print(f"  Max search: {theta_max_deg:.4f} deg")

    acc = RARAccumulator()

    print(f"\n  Reading HSC: {HSC_FILE.name} ({HSC_FILE.stat().st_size/1e9:.1f} GB)")

    with open(HSC_FILE, 'rb') as f:
        is_gz = (f.read(2) == b'\x1f\x8b')

    if is_gz:
        import gzip
        open_func = lambda: gzip.open(HSC_FILE, 'rt', errors='replace')
    else:
        open_func = lambda: open(HSC_FILE, 'r', encoding='utf-8', errors='replace')

    col_idx = {}
    n_read = 0
    n_g12 = 0
    total_pairs = 0
    header_candidate = None

    with open_func() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                header_candidate = line.lstrip('#').strip()
                continue

            if not col_idx:
                parts = [c.strip() for c in line.split(',')]
                if len(parts) < 5:
                    parts = line.split()
                try:
                    float(parts[0])
                    is_data_first = True
                    if header_candidate:
                        hdr = [c.strip() for c in header_candidate.split(',')]
                        if len(hdr) < 5:
                            hdr = header_candidate.split()
                    else:
                        print("  ERROR: No header")
                        return
                except ValueError:
                    hdr = parts
                    is_data_first = False

                for key, col_name in HSC_COLS.items():
                    for j, h in enumerate(hdr):
                        if h == col_name:
                            col_idx[key] = j
                            break

                if len(col_idx) < 4:
                    print(f"  ERROR: Insufficient columns: {col_idx}")
                    return
                print(f"  Columns mapped: {col_idx}")

                if not is_data_first:
                    continue

            n_read += 1
            parts = [c.strip() for c in line.split(',')]

            try:
                sra = float(parts[col_idx['ra']])
                sdec = float(parts[col_idx['dec']])
            except (ValueError, IndexError):
                continue

            if sra < G12_RA_MIN or sra > G12_RA_MAX or sdec < G12_DEC_MIN or sdec > G12_DEC_MAX:
                continue

            n_g12 += 1

            try:
                se1 = float(parts[col_idx['e1']])
                se2 = float(parts[col_idx['e2']])
                sw = float(parts[col_idx['weight']])
                szb = int(float(parts[col_idx['zbin']]))
            except (ValueError, IndexError):
                continue

            if sw <= 0 or szb < 1 or szb > 4:
                continue

            g1, g2 = hsc_distortion_to_shear(se1, se2)

            src_xy = np.array([sra * cos_dec_ref, sdec])
            nearby = lens_tree.query_ball_point(src_xy, theta_max_deg)
            if not nearby:
                continue

            z_s_eff = ZBIN_ZEFF[szb]

            for li in nearby:
                if z_s_eff <= lens_z[li] + DZ_MIN:
                    continue

                sc_val = sc_table[li, szb]
                if np.isinf(sc_val) or sc_val <= 0:
                    continue

                dra = (sra - lens_ra[li]) * cos_dec_ref * np.pi / 180
                ddec = (sdec - lens_dec[li]) * np.pi / 180
                theta = np.sqrt(dra**2 + ddec**2)
                D_A = angular_diameter_distance(lens_z[li])
                r_mpc = theta * D_A

                if r_mpc < R_MIN_MPC or r_mpc >= R_MAX_MPC:
                    continue

                g_bar = compute_gbar(lens_logm[li], r_mpc)

                phi = np.arctan2(ddec, dra)
                gamma_t = -(g1 * np.cos(2 * phi) + g2 * np.sin(2 * phi))
                gamma_x =  (g1 * np.sin(2 * phi) - g2 * np.cos(2 * phi))

                acc.add(g_bar, gamma_t, gamma_x, sw, sc_val, lens_mbin[li])
                total_pairs += 1

            if n_read % REPORT_INTERVAL == 0:
                elapsed = time.time() - t_start
                print(f"    {n_read/1e6:.1f}M read, {n_g12:,} G12, "
                      f"{total_pairs:,} pairs, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Complete: {n_read:,} lines, {n_g12:,} G12, "
          f"{total_pairs:,} pairs, {elapsed:.0f}s")

    section("RAR PROFILES")

    gbar_all, gobs_all, gerr_all, esd_t_all, esd_x_all, np_all = acc.get_rar('all')

    print(f"\n  All lenses:")
    print(f"  {'g_bar':>12s}  {'g_obs':>12s}  {'ESD_t':>12s}  {'ESD_x':>12s}  {'N_pairs':>10s}")
    for b in range(N_GBINS):
        if np_all[b] > 0:
            print(f"  {gbar_all[b]:12.3e}  {gobs_all[b]:12.3e}  "
                  f"{esd_t_all[b]:12.4f}  {esd_x_all[b]:12.4f}  {np_all[b]:10,d}")

    out_all = OUTPUT_DIR / 'esd_profile_all_v2.txt'
    np.savetxt(out_all,
               np.column_stack([gbar_all, gobs_all, gerr_all, esd_t_all, esd_x_all, np_all]),
               header='g_bar  g_obs  g_err  ESD_t  ESD_x  N_pairs',
               fmt=['%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%d'])
    print(f"  Saved: {out_all}")

    rar_results = []
    for mb in range(N_MBINS):
        gbar_mb, gobs_mb, gerr_mb, esd_mb, esdx_mb, np_mb = acc.get_rar('mbin', mbin=mb)

        n_lens_mb = (lens_mbin == mb).sum()
        logM_med = 0.5 * (MSTAR_BINS[mb] + MSTAR_BINS[mb + 1])

        valid = (np_mb > 10) & (gobs_mb > 0) & np.isfinite(gobs_mb)

        if valid.sum() < 3:
            print(f"\n  Bin {mb+1}: insufficient valid points")
            continue

        gc_fit = fit_gc_robust(gbar_mb[valid], gobs_mb[valid], gerr_mb[valid])
        gc_c15_val = c15_gc(logM_med)

        if not np.isnan(gc_fit):
            g_memb = membrane_rar(gbar_mb[valid], gc_fit)
            g_mond = mond_rar(gbar_mb[valid])
            g_c15 = membrane_rar(gbar_mb[valid], gc_c15_val)

            w = 1.0 / np.maximum(gerr_mb[valid]**2, 1e-30)
            chi2_mond = np.sum(w * (gobs_mb[valid] - g_mond)**2)
            chi2_c15 = np.sum(w * (gobs_mb[valid] - g_c15)**2)
            chi2_fit = np.sum(w * (gobs_mb[valid] - g_memb)**2)
        else:
            chi2_mond = chi2_c15 = chi2_fit = np.nan

        result = {
            'mbin': mb + 1, 'logM': logM_med,
            'gc_fit': gc_fit / a0 if not np.isnan(gc_fit) else np.nan,
            'gc_c15': gc_c15_val / a0,
            'chi2_mond': chi2_mond, 'chi2_c15': chi2_c15, 'chi2_fit': chi2_fit,
            'dof': int(valid.sum()),
            'g_bar': gbar_mb, 'g_obs': gobs_mb, 'g_err': gerr_mb,
            'valid': valid,
        }
        rar_results.append(result)

        dchi = chi2_mond - chi2_c15 if np.isfinite(chi2_mond) else np.nan
        print(f"\n  Bin {mb+1} (logM*={logM_med:.2f}, N_lens={n_lens_mb:,}):")
        print(f"    gc_fit={result['gc_fit']:.3f} a0, "
              f"gc_C15={result['gc_c15']:.3f} a0, "
              f"ratio={result['gc_fit']/result['gc_c15']:.2f}")
        print(f"    chi2(MOND)={chi2_mond:.1f}, chi2(C15)={chi2_c15:.1f}, "
              f"Dchi2={dchi:+.1f}")

        out_mb = OUTPUT_DIR / f'hsc_rar_mbin_{mb+1}_v2.txt'
        np.savetxt(out_mb,
                   np.column_stack([gbar_mb, gobs_mb, gerr_mb, np_mb]),
                   header=f'g_bar  g_obs  g_err  N_pairs  [logM*={logM_med:.2f}]',
                   fmt=['%.6e', '%.6e', '%.6e', '%d'])

    section("SUMMARY TABLE")
    if rar_results:
        print(f"\n  {'Bin':>4s} {'logM*':>6s} {'gc_fit/a0':>10s} {'gc_C15/a0':>10s} "
              f"{'ratio':>8s} {'Dchi2(M-C)':>12s}")
        for r in rar_results:
            dchi = r['chi2_mond'] - r['chi2_c15'] if np.isfinite(r['chi2_mond']) else np.nan
            ratio = r['gc_fit'] / r['gc_c15'] if not np.isnan(r['gc_fit']) else np.nan
            print(f"  {r['mbin']:>4d} {r['logM']:6.2f} {r['gc_fit']:10.3f} "
                  f"{r['gc_c15']:10.3f} {ratio:8.2f} {dchi:+12.1f}")

        logM_arr = np.array([r['logM'] for r in rar_results if not np.isnan(r['gc_fit'])])
        log_gc = np.log10([r['gc_fit'] for r in rar_results if not np.isnan(r['gc_fit'])])
        log_gc_c15 = np.log10([r['gc_c15'] for r in rar_results if not np.isnan(r['gc_fit'])])

        if len(logM_arr) >= 3:
            s_fit = np.polyfit(logM_arr, log_gc, 1)[0]
            s_c15 = np.polyfit(logM_arr, log_gc_c15, 1)[0]
            print(f"\n  gc-M* slopes:")
            print(f"    HSC observed:  {s_fit:+.3f}")
            print(f"    C15 predicted: {s_c15:+.3f}")
            print(f"    MOND expected: 0.000")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

        ax = axs[0, 0]
        for r in rar_results:
            v = r['valid']
            ax.loglog(r['g_bar'][v], r['g_obs'][v], 'o', ms=5,
                      color=colors[(r['mbin']-1) % 5],
                      label=f"Bin {r['mbin']} ({r['logM']:.1f})")
        gb_line = np.logspace(-16, -10, 100)
        ax.loglog(gb_line, mond_rar(gb_line), 'k--', lw=1.5, label='MOND')
        ax.loglog(gb_line, gb_line, 'k:', lw=0.5, alpha=0.3)
        ax.set_xlabel('g_bar (m/s^2)')
        ax.set_ylabel('g_obs (m/s^2)')
        ax.set_title('HSC Lensing RAR v2 (g_bar-axis)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        ax = axs[0, 1]
        if rar_results:
            lm = [r['logM'] for r in rar_results]
            gc_f = [r['gc_fit'] for r in rar_results]
            gc_c = [r['gc_c15'] for r in rar_results]
            ax.semilogy(lm, gc_f, 'rs-', ms=8, lw=2, label='HSC gc (fit)')
            ax.semilogy(lm, gc_c, 'b^--', ms=8, lw=1.5, label='C15 prediction')
            ax.axhline(1.0, color='gray', ls=':', label='MOND')
            ax.set_xlabel('log10(M*/Msun)')
            ax.set_ylabel('gc / a0')
            ax.set_title('gc per M* bin: HSC v2 vs C15')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axs[0, 2]
        if rar_results:
            valid_r = [r for r in rar_results if np.isfinite(r['chi2_mond'])]
            bx = [r['mbin'] for r in valid_r]
            dc = [r['chi2_mond'] - r['chi2_c15'] for r in valid_r]
            ax.bar(bx, dc, color=['green' if d > 0 else 'red' for d in dc])
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('M* bin')
            ax.set_ylabel('Delta-chi2 (MOND - C15)')
            ax.set_title('+ve = C15 preferred')
            ax.grid(True, alpha=0.3)

        for pi, r in enumerate(rar_results[:3]):
            ax = axs[1, pi]
            v = r['valid']
            ax.errorbar(r['g_bar'][v], r['g_obs'][v], yerr=r['g_err'][v],
                        fmt='ko', ms=4, capsize=2, label='HSC data')
            gb_l = np.logspace(np.log10(r['g_bar'][v].min() * 0.3),
                               np.log10(r['g_bar'][v].max() * 3), 50)
            ax.loglog(gb_l, mond_rar(gb_l), 'b--', lw=1.5, label='MOND')
            if not np.isnan(r['gc_fit']):
                ax.loglog(gb_l, membrane_rar(gb_l, r['gc_fit'] * a0),
                          'r-', lw=1.5, label=f"fit gc={r['gc_fit']:.2f}")
            ax.loglog(gb_l, membrane_rar(gb_l, r['gc_c15'] * a0),
                      'g:', lw=1.5, label=f"C15 gc={r['gc_c15']:.2f}")
            ax.loglog(gb_l, gb_l, 'k:', lw=0.5, alpha=0.3)
            ax.set_xlabel('g_bar')
            ax.set_ylabel('g_obs')
            ax.set_title(f"Bin {r['mbin']} (logM*={r['logM']:.1f})")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out_png = OUTPUT_DIR / 'hsc_lensing_rar_v2.png'
        plt.savefig(out_png, dpi=120)
        print(f"\n  Plot saved: {out_png}")
    except Exception as e:
        print(f"\n  [Plot error: {e}]")

    total = time.time() - t_start
    section("STEP 5 v2 COMPLETE")
    print(f"  Elapsed: {total:.0f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
