#!/usr/bin/env python3
"""
Phase B Step 2: Error bars + mult bias + chi^2.
"""

import numpy as np
import os
import sys
import time

G_SI    = 6.67430e-11
Msun    = 1.98892e30
Mpc_m   = 3.08568e22
c_ms    = 2.99792e8
H0      = 70.0
a0_si   = 1.2e-10
deg2rad = np.pi / 180.0

N_GBAR_BINS = 15
GBAR_MIN = 1e-15
GBAR_MAX = 5e-12
MSTAR_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0, 12.0]

ZBIN_ZEFF = {1: 0.44, 2: 0.75, 3: 1.01, 4: 1.30}

MULT_BIAS = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
MULT_BIAS_ERR = 0.01

N_JK_RA  = 8
N_JK_DEC = 4
N_JK = N_JK_RA * N_JK_DEC
G12_RA_MIN, G12_RA_MAX = 174.0, 186.0
G12_DEC_MIN, G12_DEC_MAX = -3.0, 3.0

Om0 = 0.3
OL0 = 0.7


def _E(z):
    return np.sqrt(Om0 * (1+z)**3 + OL0)


def comoving_dist(z, nstep=500):
    if z <= 0:
        return 0.0
    zz = np.linspace(0, z, nstep)
    return (c_ms / 1e3 / H0) * np.trapezoid(1.0 / _E(zz), zz)


_z_table = np.linspace(0.0, 2.0, 2000)
_chi_table = np.array([comoving_dist(z) for z in _z_table])


def _chi_interp(z):
    return np.interp(z, _z_table, _chi_table)


def angular_diam_dist(z):
    return _chi_interp(z) / (1 + z)


def Sigma_crit_inv(zl, zs):
    if zs <= zl + 0.05:
        return 0.0
    chi_l = _chi_interp(zl)
    chi_s = _chi_interp(zs)
    Dl  = chi_l / (1 + zl) * Mpc_m
    Ds  = chi_s / (1 + zs) * Mpc_m
    Dls = (chi_s - chi_l) / (1 + zs) * Mpc_m
    if Dls <= 0 or Ds <= 0:
        return 0.0
    return 4 * np.pi * G_SI / c_ms**2 * Dl * Dls / Ds


BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
HSC_FILE  = r"E:\スバル望遠鏡データ\931720.csv.gz.1"
LENS_FILE = os.path.join(BASE, "phase_a_output", "gama_lenses_g12_isolated.fits")
OUT_DIR   = os.path.join(BASE, "phase_b_output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_lenses():
    from astropy.io import fits as afits
    print(f"Loading lenses: {LENS_FILE}")
    hdu = afits.open(LENS_FILE)
    d = hdu[1].data
    ra   = np.array(d['RAcen'], dtype=np.float64)
    dec  = np.array(d['Deccen'], dtype=np.float64)
    z    = np.array(d['Z'], dtype=np.float64)
    logm = np.array(d['logmstar'], dtype=np.float64)
    Mgal_kg = 10**logm * Msun
    hdu.close()
    print(f"  {len(ra)} lenses, logM*: {logm.min():.1f} - {logm.max():.1f}")
    return ra, dec, z, logm, Mgal_kg


def load_hsc_sources():
    print(f"Loading HSC: {HSC_FILE}")
    RA_MIN, RA_MAX = G12_RA_MIN, G12_RA_MAX
    DEC_MIN, DEC_MAX = G12_DEC_MIN, G12_DEC_MAX

    ra_l, dec_l, e1_l, e2_l, w_l, zbin_l = [], [], [], [], [], []
    chunk = 2_000_000
    row_count = 0
    g12_count = 0

    # Detect gzip
    with open(HSC_FILE, 'rb') as fb:
        is_gz = (fb.read(2) == b'\x1f\x8b')

    if is_gz:
        import gzip
        fh = gzip.open(HSC_FILE, 'rt', errors='replace')
    else:
        fh = open(HSC_FILE, 'r', encoding='utf-8', errors='replace')

    with fh as f:
        header = None
        first_data = None
        for line in f:
            if line.startswith('#'):
                header = line.lstrip('#').strip().split(',')
                continue
            else:
                if header is None:
                    header = line.strip().split(',')
                else:
                    first_data = line
                break

        col = {name.strip(): i for i, name in enumerate(header)}
        idx_ra   = col.get('i_ra')
        idx_dec  = col.get('i_dec')
        idx_e1   = col.get('i_hsmshaperegauss_e1')
        idx_e2   = col.get('i_hsmshaperegauss_e2')
        idx_w    = col.get('i_hsmshaperegauss_derived_weight')
        idx_zbin = col.get('hsc_y3_zbin')

        if any(x is None for x in [idx_ra, idx_dec, idx_e1, idx_e2, idx_w, idx_zbin]):
            print("  ERROR: Column mapping failed")
            sys.exit(1)

        buf_ra  = np.empty(chunk, dtype=np.float64)
        buf_dec = np.empty(chunk, dtype=np.float64)
        buf_e1  = np.empty(chunk, dtype=np.float64)
        buf_e2  = np.empty(chunk, dtype=np.float64)
        buf_w   = np.empty(chunk, dtype=np.float64)
        buf_zb  = np.empty(chunk, dtype=np.int32)
        bi = 0
        t0 = time.time()

        def process(line):
            nonlocal bi, g12_count
            parts = line.strip().split(',')
            try:
                ra = float(parts[idx_ra])
                dec = float(parts[idx_dec])
            except (ValueError, IndexError):
                return
            if not (RA_MIN <= ra <= RA_MAX and DEC_MIN <= dec <= DEC_MAX):
                return
            try:
                e1 = float(parts[idx_e1])
                e2 = float(parts[idx_e2])
                w  = float(parts[idx_w])
                zb = int(float(parts[idx_zbin]))
            except (ValueError, IndexError):
                return
            if w <= 0 or zb < 1 or zb > 4:
                return
            buf_ra[bi]  = ra
            buf_dec[bi] = dec
            buf_e1[bi]  = e1
            buf_e2[bi]  = e2
            buf_w[bi]   = w
            buf_zb[bi]  = zb
            bi += 1
            g12_count += 1

        if first_data is not None:
            row_count += 1
            process(first_data)

        for line in f:
            row_count += 1
            process(line)

            if bi >= chunk:
                ra_l.append(buf_ra[:bi].copy())
                dec_l.append(buf_dec[:bi].copy())
                e1_l.append(buf_e1[:bi].copy())
                e2_l.append(buf_e2[:bi].copy())
                w_l.append(buf_w[:bi].copy())
                zbin_l.append(buf_zb[:bi].copy())
                bi = 0

            if row_count % 5_000_000 == 0:
                dt = time.time() - t0
                print(f"  {row_count/1e6:.0f}M rows, {g12_count/1e6:.2f}M G12, {dt:.0f}s", flush=True)

        if bi > 0:
            ra_l.append(buf_ra[:bi].copy())
            dec_l.append(buf_dec[:bi].copy())
            e1_l.append(buf_e1[:bi].copy())
            e2_l.append(buf_e2[:bi].copy())
            w_l.append(buf_w[:bi].copy())
            zbin_l.append(buf_zb[:bi].copy())

    s_ra   = np.concatenate(ra_l)
    s_dec  = np.concatenate(dec_l)
    s_e1   = np.concatenate(e1_l)
    s_e2   = np.concatenate(e2_l)
    s_w    = np.concatenate(w_l)
    s_zbin = np.concatenate(zbin_l)

    dt = time.time() - t0
    print(f"  Done: {row_count/1e6:.1f}M total, {len(s_ra)/1e6:.2f}M G12, {dt:.0f}s")

    e_sq = s_e1**2 + s_e2**2
    e_rms_sq = np.sum(s_w * e_sq) / np.sum(s_w)
    R_resp = 1.0 - e_rms_sq
    print(f"  Responsivity: R = {R_resp:.4f}")

    return s_ra, s_dec, s_e1, s_e2, s_w, s_zbin, R_resp


def assign_jk_patch(ra, dec):
    ira  = np.clip(((ra  - G12_RA_MIN) / (G12_RA_MAX - G12_RA_MIN) * N_JK_RA).astype(int),
                   0, N_JK_RA - 1)
    idec = np.clip(((dec - G12_DEC_MIN) / (G12_DEC_MAX - G12_DEC_MIN) * N_JK_DEC).astype(int),
                   0, N_JK_DEC - 1)
    return idec * N_JK_RA + ira


def compute_esd_with_errors(l_ra, l_dec, l_z, l_logm, l_Mgal,
                             s_ra, s_dec, s_e1, s_e2, s_w, s_zbin,
                             R_resp):
    from scipy.spatial import cKDTree

    print("\n=== Phase B Step 2: ESD with error estimation ===")

    gbar_edges = np.logspace(np.log10(GBAR_MIN), np.log10(GBAR_MAX), N_GBAR_BINS + 1)
    gbar_centers = np.sqrt(gbar_edges[:-1] * gbar_edges[1:])

    n_mbins = len(MSTAR_EDGES) - 1

    esd_num     = np.zeros(N_GBAR_BINS)
    esd_x_num   = np.zeros(N_GBAR_BINS)
    esd_den     = np.zeros(N_GBAR_BINS)
    esd_npair   = np.zeros(N_GBAR_BINS, dtype=np.int64)
    esd_var_num = np.zeros(N_GBAR_BINS)
    esd_w2_sum  = np.zeros(N_GBAR_BINS)

    esd_num_m     = np.zeros((n_mbins, N_GBAR_BINS))
    esd_x_num_m   = np.zeros((n_mbins, N_GBAR_BINS))
    esd_den_m     = np.zeros((n_mbins, N_GBAR_BINS))
    esd_npair_m   = np.zeros((n_mbins, N_GBAR_BINS), dtype=np.int64)
    esd_var_num_m = np.zeros((n_mbins, N_GBAR_BINS))
    esd_w2_sum_m  = np.zeros((n_mbins, N_GBAR_BINS))

    jk_esd_num = np.zeros((N_JK, N_GBAR_BINS))
    jk_esd_den = np.zeros((N_JK, N_GBAR_BINS))
    jk_esd_num_m = np.zeros((N_JK, n_mbins, N_GBAR_BINS))
    jk_esd_den_m = np.zeros((N_JK, n_mbins, N_GBAR_BINS))

    bin5_lens_logm  = []
    bin5_lens_z     = []
    bin5_lens_npair = []

    l_patch = assign_jk_patch(l_ra, l_dec)

    print("Building source KDTree...")
    s_xyz = np.column_stack([
        np.cos(s_dec * deg2rad) * np.cos(s_ra * deg2rad),
        np.cos(s_dec * deg2rad) * np.sin(s_ra * deg2rad),
        np.sin(s_dec * deg2rad)
    ])
    tree = cKDTree(s_xyz)

    # Precompute Sigma_crit_inv per (lens, zbin)
    print("Precomputing Sigma_crit...")
    sc_inv_table = np.zeros((len(l_ra), 5))
    for i in range(len(l_ra)):
        for zb in range(1, 5):
            sc_inv_table[i, zb] = Sigma_crit_inv(l_z[i], ZBIN_ZEFF[zb])

    n_lens = len(l_ra)
    total_pairs = 0
    t0 = time.time()

    for i in range(n_lens):
        zl = l_z[i]
        Dl = angular_diam_dist(zl)
        Mgal = l_Mgal[i]
        lm = l_logm[i]
        patch = l_patch[i]

        mbin = -1
        for mb in range(n_mbins):
            if MSTAR_EDGES[mb] <= lm < MSTAR_EDGES[mb+1]:
                mbin = mb
                break

        l_xyz_i = np.array([
            np.cos(l_dec[i]*deg2rad) * np.cos(l_ra[i]*deg2rad),
            np.cos(l_dec[i]*deg2rad) * np.sin(l_ra[i]*deg2rad),
            np.sin(l_dec[i]*deg2rad)
        ])

        theta_max = min(3.0 / Dl, np.pi / 2)
        chord_max = 2 * np.sin(theta_max / 2)
        idx = tree.query_ball_point(l_xyz_i, chord_max)
        if len(idx) == 0:
            continue
        idx = np.array(idx)

        cos_th = np.clip(
            s_xyz[idx, 0]*l_xyz_i[0] + s_xyz[idx, 1]*l_xyz_i[1] + s_xyz[idx, 2]*l_xyz_i[2],
            -1, 1)
        theta = np.arccos(cos_th)
        R_Mpc = theta * Dl
        R_m   = R_Mpc * Mpc_m

        mask_R = (R_Mpc >= 0.03) & (R_Mpc <= 3.0)
        if not np.any(mask_R):
            continue

        idx_g = idx[mask_R]
        R_m_g = R_m[mask_R]

        gbar = G_SI * Mgal / R_m_g**2
        gbar_bin = np.searchsorted(gbar_edges, gbar) - 1
        valid = (gbar_bin >= 0) & (gbar_bin < N_GBAR_BINS)
        if not np.any(valid):
            continue

        idx_v = idx_g[valid]
        gb_v  = gbar_bin[valid]

        se1 = s_e1[idx_v]
        se2 = s_e2[idx_v]
        sw  = s_w[idx_v]
        szb = s_zbin[idx_v]

        dra  = (s_ra[idx_v] - l_ra[i]) * np.cos(l_dec[i]*deg2rad) * deg2rad
        ddec = (s_dec[idx_v] - l_dec[i]) * deg2rad
        phi  = np.arctan2(ddec, dra)
        cos2p = np.cos(2*phi)
        sin2p = np.sin(2*phi)

        e_t = -(se1 * cos2p + se2 * sin2p) / 2.0
        e_x =  (se1 * sin2p - se2 * cos2p) / 2.0

        # Vectorized Sigma_crit lookup
        sc_inv_j = sc_inv_table[i, szb]
        valid_sc = sc_inv_j > 0
        if not np.any(valid_sc):
            continue

        e_t = e_t[valid_sc]
        e_x = e_x[valid_sc]
        sw = sw[valid_sc]
        szb = szb[valid_sc]
        sc_inv_v = sc_inv_j[valid_sc]
        gb_v = gb_v[valid_sc]

        # m_bias vector (zero for all currently)
        m_bias = np.array([MULT_BIAS.get(int(z), 0.0) for z in szb])
        corr = (1.0 + m_bias) * R_resp

        Scrit_v = 1.0 / sc_inv_v
        W_ls = sw * sc_inv_v**2
        et_corr = e_t / corr
        ex_corr = e_x / corr

        val_t = W_ls * et_corr * Scrit_v
        val_x = W_ls * ex_corr * Scrit_v
        var_contrib = W_ls**2 * (et_corr * Scrit_v)**2

        # Vectorized accumulation
        np.add.at(esd_num, gb_v, val_t)
        np.add.at(esd_x_num, gb_v, val_x)
        np.add.at(esd_den, gb_v, W_ls)
        np.add.at(esd_npair, gb_v, 1)
        np.add.at(esd_var_num, gb_v, var_contrib)
        np.add.at(esd_w2_sum, gb_v, W_ls**2)

        np.add.at(jk_esd_num[patch], gb_v, val_t)
        np.add.at(jk_esd_den[patch], gb_v, W_ls)

        if mbin >= 0:
            np.add.at(esd_num_m[mbin], gb_v, val_t)
            np.add.at(esd_x_num_m[mbin], gb_v, val_x)
            np.add.at(esd_den_m[mbin], gb_v, W_ls)
            np.add.at(esd_npair_m[mbin], gb_v, 1)
            np.add.at(esd_var_num_m[mbin], gb_v, var_contrib)
            np.add.at(esd_w2_sum_m[mbin], gb_v, W_ls**2)

            np.add.at(jk_esd_num_m[patch, mbin], gb_v, val_t)
            np.add.at(jk_esd_den_m[patch, mbin], gb_v, W_ls)

        total_pairs += len(e_t)

        if mbin == 4 and len(e_t) > 0:
            bin5_lens_logm.append(lm)
            bin5_lens_z.append(zl)
            bin5_lens_npair.append(len(e_t))

        if (i+1) % 1000 == 0:
            dt = time.time() - t0
            print(f"  Lens {i+1}/{n_lens}, {total_pairs/1e6:.2f}M pairs, {dt:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"\n  Done: {n_lens} lenses, {total_pairs/1e6:.2f}M pairs, {dt:.0f}s")

    results = {}

    mv = esd_den > 0
    esd = np.full(N_GBAR_BINS, np.nan)
    esd_x = np.full(N_GBAR_BINS, np.nan)
    esd[mv]   = esd_num[mv] / esd_den[mv]
    esd_x[mv] = esd_x_num[mv] / esd_den[mv]

    esd_err = np.full(N_GBAR_BINS, np.nan)
    esd_err[mv] = np.sqrt(esd_var_num[mv]) / esd_den[mv]

    # Jackknife
    jk_esd_full = np.zeros((N_JK, N_GBAR_BINS))
    for p in range(N_JK):
        num_loo = esd_num - jk_esd_num[p]
        den_loo = esd_den - jk_esd_den[p]
        good = den_loo > 0
        jk_esd_full[p, good] = num_loo[good] / den_loo[good]
        jk_esd_full[p, ~good] = np.nan

    jk_mean = np.nanmean(jk_esd_full, axis=0)
    jk_cov = np.zeros((N_GBAR_BINS, N_GBAR_BINS))
    for p in range(N_JK):
        diff = jk_esd_full[p] - jk_mean
        diff[np.isnan(diff)] = 0
        jk_cov += np.outer(diff, diff)
    jk_cov *= (N_JK - 1) / N_JK
    jk_err = np.sqrt(np.diag(jk_cov))

    gobs     = 4 * G_SI * esd
    gobs_x   = 4 * G_SI * esd_x
    gobs_err_ana = 4 * G_SI * esd_err
    gobs_err_jk  = 4 * G_SI * jk_err

    gobs_err = gobs_err_jk.copy()
    bad = (gobs_err <= 0) | np.isnan(gobs_err)
    gobs_err[bad] = gobs_err_ana[bad]

    results['all'] = {
        'gbar': gbar_centers, 'gobs': gobs, 'gobs_x': gobs_x,
        'gobs_err': gobs_err, 'gobs_err_ana': gobs_err_ana,
        'gobs_err_jk': gobs_err_jk, 'npair': esd_npair,
        'jk_cov_gobs': jk_cov * (4*G_SI)**2
    }

    results['mbins'] = []
    for mb in range(n_mbins):
        mv_m = esd_den_m[mb] > 0
        esd_mb = np.full(N_GBAR_BINS, np.nan)
        esd_err_mb = np.full(N_GBAR_BINS, np.nan)
        esd_mb[mv_m] = esd_num_m[mb, mv_m] / esd_den_m[mb, mv_m]
        esd_err_mb[mv_m] = np.sqrt(esd_var_num_m[mb, mv_m]) / esd_den_m[mb, mv_m]

        jk_esd_mb = np.zeros((N_JK, N_GBAR_BINS))
        for p in range(N_JK):
            num_loo = esd_num_m[mb] - jk_esd_num_m[p, mb]
            den_loo = esd_den_m[mb] - jk_esd_den_m[p, mb]
            good = den_loo > 0
            jk_esd_mb[p, good] = num_loo[good] / den_loo[good]
            jk_esd_mb[p, ~good] = np.nan

        jk_mean_mb = np.nanmean(jk_esd_mb, axis=0)
        jk_var_mb = np.zeros(N_GBAR_BINS)
        for p in range(N_JK):
            diff = jk_esd_mb[p] - jk_mean_mb
            diff[np.isnan(diff)] = 0
            jk_var_mb += diff**2
        jk_var_mb *= (N_JK - 1) / N_JK
        jk_err_mb = np.sqrt(jk_var_mb)

        go_mb     = 4 * G_SI * esd_mb
        go_err_a  = 4 * G_SI * esd_err_mb
        go_err_jk = 4 * G_SI * jk_err_mb
        go_err = go_err_jk.copy()
        bad = (go_err <= 0) | np.isnan(go_err)
        go_err[bad] = go_err_a[bad]

        results['mbins'].append({
            'mbin': mb, 'mlo': MSTAR_EDGES[mb], 'mhi': MSTAR_EDGES[mb+1],
            'gobs': go_mb, 'gobs_err': go_err,
            'gobs_err_ana': go_err_a, 'gobs_err_jk': go_err_jk,
            'npair': esd_npair_m[mb]
        })

    results['bin5_diag'] = {
        'logm': np.array(bin5_lens_logm),
        'z': np.array(bin5_lens_z),
        'npair_per_lens': np.array(bin5_lens_npair)
    }

    return results


def fit_gc_with_error(gbar, gobs, gobs_err):
    from scipy.optimize import minimize_scalar

    mask = np.isfinite(gbar) & np.isfinite(gobs) & np.isfinite(gobs_err)
    mask &= (gobs > 0) & (gbar > 0) & (gobs_err > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.inf

    gb = gbar[mask]
    go = gobs[mask]
    ge = gobs_err[mask]

    def chi2(log_gc):
        gc = 10**log_gc
        model = gb / (1 - np.exp(-np.sqrt(gb / gc)))
        return np.sum(((go - model) / ge)**2)

    res = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    gc_best = 10**res.x
    chi2_min = res.fun
    ndof = np.sum(mask) - 1

    log_gc_best = res.x
    chi2_target = chi2_min + 1.0

    from scipy.optimize import brentq
    try:
        log_gc_hi = brentq(lambda x: chi2(x) - chi2_target,
                           log_gc_best, log_gc_best + 2)
    except (ValueError, RuntimeError):
        log_gc_hi = log_gc_best + 0.5

    try:
        log_gc_lo = brentq(lambda x: chi2(x) - chi2_target,
                           log_gc_best - 2, log_gc_best)
    except (ValueError, RuntimeError):
        log_gc_lo = log_gc_best - 0.5

    gc_err = (10**log_gc_hi - 10**log_gc_lo) / 2

    return gc_best, gc_err, chi2_min / max(ndof, 1)


def chi2_model_comparison(gbar, gobs, gobs_err, jk_cov_gobs=None):
    mask = np.isfinite(gbar) & np.isfinite(gobs) & np.isfinite(gobs_err)
    mask &= (gobs > 0) & (gbar > 0) & (gobs_err > 0)
    if np.sum(mask) < 3:
        return {}

    gb = gbar[mask]
    go = gobs[mask]
    ge = gobs_err[mask]
    idx_mask = np.where(mask)[0]

    if jk_cov_gobs is not None:
        cov = jk_cov_gobs[np.ix_(idx_mask, idx_mask)]
        diag = np.diag(cov).copy()
        bad_diag = diag <= 0
        if np.any(bad_diag):
            cov = cov.copy()
            for k in np.where(bad_diag)[0]:
                cov[k, k] = ge[k]**2
        try:
            cov_inv = np.linalg.inv(cov)
            use_cov = True
        except np.linalg.LinAlgError:
            use_cov = False
    else:
        use_cov = False

    def compute_chi2(model):
        residual = go - model
        if use_cov:
            return float(residual @ cov_inv @ residual)
        else:
            return float(np.sum((residual / ge)**2))

    mond_model = gb / (1 - np.exp(-np.sqrt(gb / a0_si)))
    chi2_mond = compute_chi2(mond_model)

    gc_fit, gc_err, _ = fit_gc_with_error(gbar, gobs, gobs_err)
    if np.isfinite(gc_fit):
        c15_model = gb / (1 - np.exp(-np.sqrt(gb / gc_fit)))
        chi2_c15 = compute_chi2(c15_model)
    else:
        chi2_c15 = np.inf

    n_pts = int(np.sum(mask))
    return {
        'chi2_mond': chi2_mond, 'chi2_c15': chi2_c15,
        'ndof_mond': n_pts, 'ndof_c15': n_pts - 1,
        'gc_fit': gc_fit, 'gc_err': gc_err,
        'delta_chi2': chi2_mond - chi2_c15,
        'delta_AIC': (chi2_mond - chi2_c15) - 2,
        'n_pts': n_pts, 'use_cov': use_cov
    }


def gc_slope_analysis(gbar, results_mbins):
    print("\n=== gc vs M* slope analysis ===")

    logm_c_list = []
    gc_list = []
    gc_err_list = []

    for r in results_mbins:
        mlo, mhi = r['mlo'], r['mhi']
        logm_c = (mlo + mhi) / 2
        mask = np.isfinite(r['gobs']) & np.isfinite(r['gobs_err'])
        mask &= (r['gobs'] > 0) & (r['gobs_err'] > 0) & (r['npair'] > 50)
        if np.sum(mask) < 3:
            print(f"  M*-bin {r['mbin']+1} [{mlo:.1f},{mhi:.1f}): skip")
            continue

        gc, gc_err, chi2r = fit_gc_with_error(gbar[mask], r['gobs'][mask], r['gobs_err'][mask])
        gc_a0 = gc / a0_si
        gc_err_a0 = gc_err / a0_si
        print(f"  M*-bin {r['mbin']+1} [{mlo:.1f},{mhi:.1f}): "
              f"gc = {gc_a0:.3f} +/- {gc_err_a0:.3f} a0, "
              f"chi2/dof = {chi2r:.2f}, N_pairs = {np.sum(r['npair']):.0f}")

        if np.isfinite(gc) and gc > 0 and np.isfinite(gc_err_a0) and gc_err_a0 > 0:
            logm_c_list.append(logm_c)
            gc_list.append(gc_a0)
            gc_err_list.append(gc_err_a0)

    if len(logm_c_list) < 2:
        print("  Cannot compute slope")
        return

    x = np.array(logm_c_list)
    y = np.log10(gc_list)
    ye = np.array(gc_err_list) / (np.array(gc_list) * np.log(10))

    w = 1.0 / ye**2
    S   = np.sum(w)
    Sx  = np.sum(w * x)
    Sy  = np.sum(w * y)
    Sxx = np.sum(w * x**2)
    Sxy = np.sum(w * x * y)
    det = S * Sxx - Sx**2

    if det <= 0:
        print("  Degenerate fit")
        return

    slope = (S * Sxy - Sx * Sy) / det
    slope_err = np.sqrt(S / det)

    print(f"\n  >> gc-M* slope (Phase B Step 2): {slope:+.3f} +/- {slope_err:.3f}")
    print(f"     C15 prediction: +0.075")
    print(f"     MOND prediction: 0.000")

    t_c15  = (slope - 0.075) / slope_err
    t_mond = (slope - 0.000) / slope_err
    from scipy.stats import norm
    p_c15  = 2 * norm.sf(abs(t_c15))
    p_mond = 2 * norm.sf(abs(t_mond))
    print(f"     Consistent with C15?  t={t_c15:+.2f}, p={p_c15:.3f}")
    print(f"     Consistent with MOND? t={t_mond:+.2f}, p={p_mond:.3f}")

    return slope, slope_err


def bin5_diagnostics(diag):
    print("\n=== Bin 5 (logM* 11.0-12.0) diagnostics ===")
    logm = diag['logm']
    z = diag['z']
    np_lens = diag['npair_per_lens']

    if len(logm) == 0:
        print("  No lenses in Bin 5")
        return

    print(f"  N_lenses: {len(logm)}")
    print(f"  logM* range: {logm.min():.2f} - {logm.max():.2f}")
    print(f"  logM* median: {np.median(logm):.2f}")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    print(f"  Pairs/lens: median={np.median(np_lens):.0f}, max={np.max(np_lens)}, total={np.sum(np_lens)}")

    sorted_np = np.sort(np_lens)[::-1]
    cumsum = np.cumsum(sorted_np) / np.sum(np_lens) * 100
    n10 = min(10, len(sorted_np))
    print(f"  Top {n10} lenses contribute {cumsum[n10-1]:.1f}% of pairs")

    n_above_11 = np.sum(logm >= 11.0)
    print(f"  Lenses with logM* >= 11.0: {n_above_11}")
    print(f"  NOTE: Brouwer+2021 restricts to M* < 10^11 h70^-2 Msun")


def save_and_plot(gbar, results):
    print("\n=== Saving results ===")
    r = results['all']

    fname = os.path.join(OUT_DIR, "phase_b_rar_all_errors.txt")
    with open(fname, 'w') as f:
        f.write("# Phase B Step 2 RAR\n")
        f.write("# g_bar  g_obs  g_obs_err  err_ana  err_jk  g_x  N_pairs\n")
        for k in range(N_GBAR_BINS):
            f.write(f"{gbar[k]:.6e}  {r['gobs'][k]:.6e}  {r['gobs_err'][k]:.6e}  "
                    f"{r['gobs_err_ana'][k]:.6e}  {r['gobs_err_jk'][k]:.6e}  "
                    f"{r['gobs_x'][k]:.6e}  {r['npair'][k]}\n")
    print(f"  Saved: {fname}")

    for rb in results['mbins']:
        mb = rb['mbin']
        fname = os.path.join(OUT_DIR, f"phase_b_rar_mbin_{mb+1}_errors.txt")
        with open(fname, 'w') as f:
            f.write(f"# M*-bin {mb+1} [{rb['mlo']:.1f},{rb['mhi']:.1f})\n")
            f.write("# g_bar  g_obs  g_obs_err  N_pairs\n")
            for k in range(N_GBAR_BINS):
                f.write(f"{gbar[k]:.6e}  {rb['gobs'][k]:.6e}  "
                        f"{rb['gobs_err'][k]:.6e}  {rb['npair'][k]}\n")
        print(f"  Saved: {fname}")

    fname = os.path.join(OUT_DIR, "phase_b_jk_covariance.txt")
    np.savetxt(fname, r['jk_cov_gobs'], header="JK covariance of g_obs, 15x15")
    print(f"  Saved: {fname}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    gb_plot = np.logspace(-5, 2, 200) * a0_si
    g_mond = gb_plot / (1 - np.exp(-np.sqrt(gb_plot / a0_si)))

    ax = axes[0, 0]
    mask = np.isfinite(r['gobs']) & (r['npair'] > 100) & (r['gobs_err'] > 0)
    if np.any(mask):
        ax.errorbar(gbar[mask]/a0_si, r['gobs'][mask]/a0_si,
                    yerr=r['gobs_err'][mask]/a0_si,
                    fmt='o', ms=5, color='blue', ecolor='blue', alpha=0.8,
                    capsize=3, label='g_obs')
        mask_x = np.isfinite(r['gobs_x']) & (r['npair'] > 100)
        ax.scatter(gbar[mask_x]/a0_si, np.abs(r['gobs_x'][mask_x])/a0_si,
                   c='gray', s=15, alpha=0.4, marker='x', label='|g_x|')

    ax.plot(gb_plot/a0_si, g_mond/a0_si, 'r-', lw=2, label='MOND')
    ax.plot(gb_plot/a0_si, gb_plot/a0_si, 'k--', lw=1, alpha=0.3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar / a0'); ax.set_ylabel('g_obs / a0')
    ax.set_title('All isolated')
    ax.legend(fontsize=7); ax.set_xlim(1e-5, 1e2); ax.set_ylim(1e-3, 1e2)

    colors = ['purple', 'blue', 'green', 'orange', 'red']
    for mb_i, rb in enumerate(results['mbins']):
        if mb_i >= 5:
            break
        ax = axes[(mb_i+1)//3, (mb_i+1)%3]
        mask = np.isfinite(rb['gobs']) & (rb['npair'] > 30) & (rb['gobs_err'] > 0)

        gc_fit, gc_err, chi2r = fit_gc_with_error(gbar[mask], rb['gobs'][mask], rb['gobs_err'][mask])
        gc_a0 = gc_fit / a0_si
        gc_err_a0 = gc_err / a0_si

        if np.any(mask):
            lab = f"gc={gc_a0:.2f}+/-{gc_err_a0:.2f} a0" if np.isfinite(gc_fit) else "fit failed"
            ax.errorbar(gbar[mask]/a0_si, rb['gobs'][mask]/a0_si,
                        yerr=rb['gobs_err'][mask]/a0_si,
                        fmt='o', ms=4, color=colors[mb_i], ecolor=colors[mb_i],
                        alpha=0.8, capsize=2, label=lab)

            if np.isfinite(gc_fit):
                g_c15 = gb_plot / (1 - np.exp(-np.sqrt(gb_plot / gc_fit)))
                ax.plot(gb_plot/a0_si, g_c15/a0_si, '-', color=colors[mb_i],
                        lw=1, alpha=0.5)

        ax.plot(gb_plot/a0_si, g_mond/a0_si, 'r-', lw=1.5, alpha=0.5)
        ax.plot(gb_plot/a0_si, gb_plot/a0_si, 'k--', lw=1, alpha=0.2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('g_bar / a0'); ax.set_ylabel('g_obs / a0')
        ax.set_title(f"M*-bin {mb_i+1}: [{rb['mlo']:.1f},{rb['mhi']:.1f})")
        ax.legend(fontsize=7)
        ax.set_xlim(1e-5, 1e2); ax.set_ylim(1e-3, 1e2)

    plt.tight_layout()
    figpath = os.path.join(OUT_DIR, "phase_b_rar_errors.png")
    plt.savefig(figpath, dpi=150)
    print(f"  Saved: {figpath}")
    plt.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    cov = r['jk_cov_gobs']
    diag = np.sqrt(np.diag(cov))
    corr = np.zeros_like(cov)
    for i in range(N_GBAR_BINS):
        for j in range(N_GBAR_BINS):
            if diag[i] > 0 and diag[j] > 0:
                corr[i, j] = cov[i, j] / (diag[i] * diag[j])
    im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                    origin='lower', aspect='equal')
    ax2.set_xlabel('g_bar bin'); ax2.set_ylabel('g_bar bin')
    ax2.set_title('Jackknife correlation matrix (g_obs)')
    plt.colorbar(im, ax=ax2, label='Correlation')
    figpath2 = os.path.join(OUT_DIR, "phase_b_jk_correlation.png")
    plt.savefig(figpath2, dpi=150)
    print(f"  Saved: {figpath2}")
    plt.close()


def main():
    print("=" * 64)
    print("Phase B Step 2: Error bars + chi^2")
    print("=" * 64)

    l_ra, l_dec, l_z, l_logm, l_Mgal = load_lenses()
    s_ra, s_dec, s_e1, s_e2, s_w, s_zbin, R_resp = load_hsc_sources()

    results = compute_esd_with_errors(
        l_ra, l_dec, l_z, l_logm, l_Mgal,
        s_ra, s_dec, s_e1, s_e2, s_w, s_zbin, R_resp)

    gbar = results['all']['gbar']

    print("\n=== ESD summary with errors ===")
    r = results['all']
    for k in range(N_GBAR_BINS):
        if r['npair'][k] > 0:
            snr = r['gobs'][k] / r['gobs_err'][k] if r['gobs_err'][k] > 0 else 0
            print(f"  bin {k+1:2d}: g_bar={gbar[k]/a0_si:.3e} a0, "
                  f"g_obs={r['gobs'][k]/a0_si:.3e} +/- {r['gobs_err'][k]/a0_si:.3e} a0, "
                  f"S/N={snr:.1f}, N={r['npair'][k]}")

    print("\n=== Chi^2: C15 vs MOND ===")
    chi2_res = chi2_model_comparison(gbar, r['gobs'], r['gobs_err'], r['jk_cov_gobs'])
    if chi2_res:
        print(f"  MOND: chi2={chi2_res['chi2_mond']:.1f} (dof={chi2_res['ndof_mond']})")
        print(f"  C15:  chi2={chi2_res['chi2_c15']:.1f} (dof={chi2_res['ndof_c15']}, "
              f"gc={chi2_res['gc_fit']/a0_si:.3f}+/-{chi2_res['gc_err']/a0_si:.3f} a0)")
        print(f"  Delta-chi2 = {chi2_res['delta_chi2']:+.1f}")
        print(f"  Delta-AIC  = {chi2_res['delta_AIC']:+.1f}")
        print(f"  Cov: {'full jackknife' if chi2_res['use_cov'] else 'diagonal'}")
        if abs(chi2_res['delta_AIC']) < 2:
            print("  >> |dAIC| < 2: INDISTINGUISHABLE")
        elif chi2_res['delta_AIC'] > 2:
            print("  >> dAIC > 2: C15 preferred")
        else:
            print("  >> dAIC < -2: MOND preferred")

    gc_slope_analysis(gbar, results['mbins'])

    bin5_diagnostics(results['bin5_diag'])

    save_and_plot(gbar, results)

    print("\n=== Phase B Step 2 complete ===")


if __name__ == '__main__':
    main()
