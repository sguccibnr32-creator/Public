#!/usr/bin/env python3
"""
Phase B Step 1: Corrected g_bar (baryonic point mass per lens, Brouwer Eq.2).
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

G_SI   = 6.67430e-11
Msun   = 1.98892e30
Mpc_m  = 3.08568e22
kpc_m  = 3.08568e19
c_ms   = 2.99792e8
H0     = 70.0
a0_si  = 1.2e-10
deg2rad = np.pi / 180.0

N_GBAR_BINS = 15
GBAR_MIN = 1e-15
GBAR_MAX = 5e-12

MSTAR_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0, 12.0]

ZBIN_ZEFF = {1: 0.44, 2: 0.75, 3: 1.01, 4: 1.30}

Om0 = 0.3
OL0 = 0.7


def _E(z):
    return np.sqrt(Om0 * (1+z)**3 + OL0)


def comoving_dist(z, nstep=500):
    if z <= 0:
        return 0.0
    zz = np.linspace(0, z, nstep)
    integrand = 1.0 / _E(zz)
    return (c_ms / 1e3 / H0) * np.trapezoid(integrand, zz)


def angular_diam_dist(z):
    return comoving_dist(z) / (1 + z)


_z_table = np.linspace(0.0, 2.0, 2000)
_chi_table = np.array([comoving_dist(z) for z in _z_table])


def _chi_interp(z):
    return np.interp(z, _z_table, _chi_table)


def Sigma_crit_inv(zl, zs):
    if zs <= zl + 0.05:
        return 0.0
    chi_l = _chi_interp(zl)
    chi_s = _chi_interp(zs)
    Dl = chi_l / (1 + zl) * Mpc_m
    Ds = chi_s / (1 + zs) * Mpc_m
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
    print(f"  Loaded {len(ra)} lenses, M* range: {logm.min():.1f} - {logm.max():.1f}")
    return ra, dec, z, logm, Mgal_kg


def load_hsc_sources():
    print(f"Loading HSC sources: {HSC_FILE}")

    RA_MIN, RA_MAX = 174.0, 186.0
    DEC_MIN, DEC_MAX = -3.0, 3.0

    ra_list, dec_list, e1_list, e2_list, w_list, zbin_list = \
        [], [], [], [], [], []

    chunk_size = 2_000_000
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
        # Skip comment lines; find header
        header = None
        for line in f:
            if line.startswith('#'):
                header = line.lstrip('#').strip().split(',')
                continue
            else:
                # First data line
                if header is None:
                    header = line.strip().split(',')
                    col = {name.strip(): i for i, name in enumerate(header)}
                    first_data = None
                else:
                    col = {name.strip(): i for i, name in enumerate(header)}
                    first_data = line
                break

        idx_ra   = col.get('i_ra')
        idx_dec  = col.get('i_dec')
        idx_e1   = col.get('i_hsmshaperegauss_e1')
        idx_e2   = col.get('i_hsmshaperegauss_e2')
        idx_w    = col.get('i_hsmshaperegauss_derived_weight')
        idx_zbin = col.get('hsc_y3_zbin')

        if any(x is None for x in [idx_ra, idx_dec, idx_e1, idx_e2, idx_w, idx_zbin]):
            print(f"  WARNING: Column mapping issue:")
            for k, v in list(col.items())[:20]:
                print(f"    {k}: {v}")
            sys.exit(1)

        print(f"  Column indices: ra={idx_ra}, dec={idx_dec}, "
              f"e1={idx_e1}, e2={idx_e2}, w={idx_w}, zbin={idx_zbin}")

        batch_ra = np.empty(chunk_size, dtype=np.float64)
        batch_dec = np.empty(chunk_size, dtype=np.float64)
        batch_e1 = np.empty(chunk_size, dtype=np.float64)
        batch_e2 = np.empty(chunk_size, dtype=np.float64)
        batch_w = np.empty(chunk_size, dtype=np.float64)
        batch_zbin = np.empty(chunk_size, dtype=np.int32)
        bi = 0

        t0 = time.time()

        def process_line(line):
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
            batch_ra[bi] = ra
            batch_dec[bi] = dec
            batch_e1[bi] = e1
            batch_e2[bi] = e2
            batch_w[bi] = w
            batch_zbin[bi] = zb
            bi += 1
            g12_count += 1

        # Process first line if data
        if first_data is not None:
            row_count += 1
            process_line(first_data)

        for line in f:
            row_count += 1
            process_line(line)

            if bi >= chunk_size:
                ra_list.append(batch_ra[:bi].copy())
                dec_list.append(batch_dec[:bi].copy())
                e1_list.append(batch_e1[:bi].copy())
                e2_list.append(batch_e2[:bi].copy())
                w_list.append(batch_w[:bi].copy())
                zbin_list.append(batch_zbin[:bi].copy())
                bi = 0

            if row_count % 5_000_000 == 0:
                dt = time.time() - t0
                print(f"  {row_count/1e6:.0f}M rows, "
                      f"{g12_count/1e6:.2f}M G12, {dt:.0f}s", flush=True)

        if bi > 0:
            ra_list.append(batch_ra[:bi].copy())
            dec_list.append(batch_dec[:bi].copy())
            e1_list.append(batch_e1[:bi].copy())
            e2_list.append(batch_e2[:bi].copy())
            w_list.append(batch_w[:bi].copy())
            zbin_list.append(batch_zbin[:bi].copy())

    s_ra   = np.concatenate(ra_list)
    s_dec  = np.concatenate(dec_list)
    s_e1   = np.concatenate(e1_list)
    s_e2   = np.concatenate(e2_list)
    s_w    = np.concatenate(w_list)
    s_zbin = np.concatenate(zbin_list)

    dt = time.time() - t0
    print(f"  Done: {row_count/1e6:.1f}M total, "
          f"{len(s_ra)/1e6:.2f}M G12, {dt:.0f}s")

    e_sq = s_e1**2 + s_e2**2
    w_sum = np.sum(s_w)
    e_rms_sq = np.sum(s_w * e_sq) / w_sum
    R_resp = 1.0 - e_rms_sq
    print(f"  Responsivity: R = 1 - <e_rms^2> = 1 - {e_rms_sq:.4f} = {R_resp:.4f}")

    return s_ra, s_dec, s_e1, s_e2, s_w, s_zbin, R_resp


def compute_esd_gbar_binned(l_ra, l_dec, l_z, l_logm, l_Mgal,
                             s_ra, s_dec, s_e1, s_e2, s_w, s_zbin,
                             R_resp):
    from scipy.spatial import cKDTree

    print("\n=== Phase B: g_bar-binned ESD computation ===")

    gbar_edges = np.logspace(np.log10(GBAR_MIN), np.log10(GBAR_MAX), N_GBAR_BINS + 1)
    gbar_centers = np.sqrt(gbar_edges[:-1] * gbar_edges[1:])

    print("Building source KDTree...")
    s_xyz = np.column_stack([
        np.cos(s_dec * deg2rad) * np.cos(s_ra * deg2rad),
        np.cos(s_dec * deg2rad) * np.sin(s_ra * deg2rad),
        np.sin(s_dec * deg2rad)
    ])
    tree = cKDTree(s_xyz)

    zl_min = l_z.min()
    Dl_min = angular_diam_dist(zl_min)
    theta_max = 3.0 / Dl_min
    print(f"  Max search angle: {np.degrees(theta_max):.2f} deg at z={zl_min:.3f}")

    n_mbins = len(MSTAR_EDGES) - 1

    esd_num   = np.zeros(N_GBAR_BINS)
    esd_den   = np.zeros(N_GBAR_BINS)
    esd_npair = np.zeros(N_GBAR_BINS, dtype=np.int64)
    esd_x_num = np.zeros(N_GBAR_BINS)

    esd_num_m   = np.zeros((n_mbins, N_GBAR_BINS))
    esd_den_m   = np.zeros((n_mbins, N_GBAR_BINS))
    esd_npair_m = np.zeros((n_mbins, N_GBAR_BINS), dtype=np.int64)
    esd_x_num_m = np.zeros((n_mbins, N_GBAR_BINS))

    # Precompute Sigma_crit_inv table for each lens x zbin
    print("  Precomputing Sigma_crit(lens, zbin)...")
    sc_inv_table = np.zeros((len(l_ra), 5))  # zbin 1-4
    for i in range(len(l_ra)):
        for zb in range(1, 5):
            sc_inv_table[i, zb] = Sigma_crit_inv(l_z[i], ZBIN_ZEFF[zb])

    n_lens = len(l_ra)
    t0 = time.time()
    total_pairs = 0

    for i in range(n_lens):
        zl = l_z[i]
        Dl = angular_diam_dist(zl)
        Mgal = l_Mgal[i]
        lm = l_logm[i]

        mbin = -1
        for mb in range(n_mbins):
            if MSTAR_EDGES[mb] <= lm < MSTAR_EDGES[mb+1]:
                mbin = mb
                break

        l_xyz = np.array([
            np.cos(l_dec[i] * deg2rad) * np.cos(l_ra[i] * deg2rad),
            np.cos(l_dec[i] * deg2rad) * np.sin(l_ra[i] * deg2rad),
            np.sin(l_dec[i] * deg2rad)
        ])

        theta_this = 3.0 / Dl
        chord_this = 2 * np.sin(min(theta_this, np.pi/2) / 2)

        idx = tree.query_ball_point(l_xyz, chord_this)
        if len(idx) == 0:
            continue
        idx = np.array(idx)

        cos_theta = np.clip(
            s_xyz[idx, 0]*l_xyz[0] + s_xyz[idx, 1]*l_xyz[1] + s_xyz[idx, 2]*l_xyz[2],
            -1, 1
        )
        theta = np.arccos(cos_theta)

        R_Mpc = theta * Dl
        R_m = R_Mpc * Mpc_m

        mask_R = (R_Mpc >= 0.03) & (R_Mpc <= 3.0)
        if not np.any(mask_R):
            continue

        idx_good = idx[mask_R]
        R_m_good = R_m[mask_R]

        # Per-pair g_bar (Brouwer Eq.2)
        gbar = G_SI * Mgal / R_m_good**2

        gbar_bin = np.searchsorted(gbar_edges, gbar) - 1
        valid_bin = (gbar_bin >= 0) & (gbar_bin < N_GBAR_BINS)
        if not np.any(valid_bin):
            continue

        idx_valid = idx_good[valid_bin]
        gbar_bin_valid = gbar_bin[valid_bin]

        se1 = s_e1[idx_valid]
        se2 = s_e2[idx_valid]
        sw  = s_w[idx_valid]
        szb = s_zbin[idx_valid]

        dra  = (s_ra[idx_valid] - l_ra[i]) * np.cos(l_dec[i] * deg2rad) * deg2rad
        ddec = (s_dec[idx_valid] - l_dec[i]) * deg2rad
        phi  = np.arctan2(ddec, dra)

        cos2phi = np.cos(2 * phi)
        sin2phi = np.sin(2 * phi)

        e_t = -(se1 * cos2phi + se2 * sin2phi) / 2.0
        e_x =  (se1 * sin2phi - se2 * cos2phi) / 2.0

        # Vectorized Sigma_crit lookup per source
        sc_inv_j = sc_inv_table[i, szb]  # array indexed by zbin
        valid_sc = sc_inv_j > 0
        if not np.any(valid_sc):
            continue

        # Apply valid_sc mask
        e_t_v = e_t[valid_sc]
        e_x_v = e_x[valid_sc]
        sw_v = sw[valid_sc]
        sc_inv_v = sc_inv_j[valid_sc]
        gbar_bin_v = gbar_bin_valid[valid_sc]

        W_ls = sw_v * sc_inv_v**2
        Scrit_v = 1.0 / sc_inv_v

        # Vectorized accumulation using np.add.at (atomic-style)
        contribs_num = W_ls * e_t_v * Scrit_v
        contribs_x = W_ls * e_x_v * Scrit_v

        np.add.at(esd_num, gbar_bin_v, contribs_num)
        np.add.at(esd_x_num, gbar_bin_v, contribs_x)
        np.add.at(esd_den, gbar_bin_v, W_ls)
        np.add.at(esd_npair, gbar_bin_v, 1)

        if mbin >= 0:
            np.add.at(esd_num_m[mbin], gbar_bin_v, contribs_num)
            np.add.at(esd_x_num_m[mbin], gbar_bin_v, contribs_x)
            np.add.at(esd_den_m[mbin], gbar_bin_v, W_ls)
            np.add.at(esd_npair_m[mbin], gbar_bin_v, 1)

        total_pairs += len(e_t_v)

        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (n_lens - i - 1) / rate / 60
            print(f"  Lens {i+1}/{n_lens}, {total_pairs/1e6:.2f}M pairs, "
                  f"{dt:.0f}s, ETA {eta:.1f}min", flush=True)

    dt = time.time() - t0
    print(f"\n  Completed: {n_lens} lenses, {total_pairs/1e6:.2f}M pairs, {dt:.0f}s")

    mu_corr = 0.0  # Phase B placeholder (set to 0 for simplicity)

    mask_valid = esd_den > 0
    esd = np.full(N_GBAR_BINS, np.nan)
    esd_x = np.full(N_GBAR_BINS, np.nan)
    esd[mask_valid] = esd_num[mask_valid] / esd_den[mask_valid] / (1 + mu_corr) / R_resp
    esd_x[mask_valid] = esd_x_num[mask_valid] / esd_den[mask_valid] / (1 + mu_corr) / R_resp

    gobs = 4 * G_SI * esd
    gobs_x = 4 * G_SI * esd_x

    esd_m = np.full((n_mbins, N_GBAR_BINS), np.nan)
    gobs_m = np.full((n_mbins, N_GBAR_BINS), np.nan)
    for mb in range(n_mbins):
        mv = esd_den_m[mb] > 0
        esd_m[mb, mv] = esd_num_m[mb, mv] / esd_den_m[mb, mv] / (1+mu_corr) / R_resp
        gobs_m[mb] = 4 * G_SI * esd_m[mb]

    return (gbar_centers, gbar_edges, esd, esd_x, gobs, gobs_x, esd_npair,
            esd_m, gobs_m, esd_npair_m)


def fit_gc(gbar, gobs, a0=a0_si):
    from scipy.optimize import minimize_scalar

    mask = np.isfinite(gbar) & np.isfinite(gobs) & (gobs > 0) & (gbar > 0)
    if np.sum(mask) < 3:
        return np.nan

    gb = gbar[mask]
    go = gobs[mask]
    log_go = np.log10(go)

    def chi2(log_gc):
        gc = 10**log_gc
        model = gb / (1 - np.exp(-np.sqrt(gb / gc)))
        log_model = np.log10(np.maximum(model, 1e-30))
        return np.sum((log_go - log_model)**2)

    result = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**result.x


def save_results(gbar_centers, esd, esd_x, gobs, gobs_x, npair,
                 esd_m, gobs_m, npair_m):
    fname = os.path.join(OUT_DIR, "phase_b_rar_all.txt")
    with open(fname, 'w') as f:
        f.write("# Phase B: Corrected g_bar definition (baryonic point mass per lens)\n")
        f.write("# g_bar [m/s^2]  g_obs [m/s^2]  g_obs_x [m/s^2]  "
                "ESD [kg/m^2]  N_pairs\n")
        for k in range(len(gbar_centers)):
            f.write(f"{gbar_centers[k]:.6e}  {gobs[k]:.6e}  {gobs_x[k]:.6e}  "
                    f"{esd[k]:.6e}  {npair[k]}\n")
    print(f"  Saved: {fname}")

    n_mbins = len(MSTAR_EDGES) - 1
    for mb in range(n_mbins):
        mlo, mhi = MSTAR_EDGES[mb], MSTAR_EDGES[mb+1]
        fname = os.path.join(OUT_DIR, f"phase_b_rar_mbin_{mb+1}.txt")
        with open(fname, 'w') as f:
            f.write(f"# Phase B RAR: M* bin [{mlo:.1f}, {mhi:.1f})\n")
            f.write("# g_bar [m/s^2]  g_obs [m/s^2]  ESD [kg/m^2]  N_pairs\n")
            for k in range(len(gbar_centers)):
                f.write(f"{gbar_centers[k]:.6e}  {gobs_m[mb,k]:.6e}  "
                        f"{esd_m[mb,k]:.6e}  {npair_m[mb,k]}\n")
        print(f"  Saved: {fname}")


def plot_rar(gbar_centers, gobs, gobs_x, gobs_m, npair, npair_m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_mbins = len(MSTAR_EDGES) - 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    ax = axes[0]
    mask = np.isfinite(gobs) & (npair > 100)
    if np.any(mask):
        ax.scatter(gbar_centers[mask] / a0_si, gobs[mask] / a0_si,
                   c='blue', s=40, zorder=3, label='g_obs (Phase B)')
        mask_x = np.isfinite(gobs_x) & (npair > 100)
        ax.scatter(gbar_centers[mask_x] / a0_si, np.abs(gobs_x[mask_x]) / a0_si,
                   c='gray', s=20, alpha=0.5, marker='x', label='|g_x| (null)')

    gb_plot = np.logspace(-5, 2, 200) * a0_si
    g_mond = gb_plot / (1 - np.exp(-np.sqrt(gb_plot / a0_si)))
    ax.plot(gb_plot / a0_si, g_mond / a0_si, 'r-', lw=2, label='MOND (a0)')
    ax.plot(gb_plot / a0_si, gb_plot / a0_si, 'k--', lw=1, alpha=0.5, label='g_obs=g_bar')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar / a0'); ax.set_ylabel('g_obs / a0')
    ax.set_title('All isolated galaxies (Phase B)')
    ax.legend(fontsize=8)
    ax.set_xlim(1e-5, 1e2); ax.set_ylim(1e-3, 1e2)

    colors = ['purple', 'blue', 'green', 'orange', 'red']
    for mb in range(min(n_mbins, 5)):
        ax = axes[mb + 1]
        mlo, mhi = MSTAR_EDGES[mb], MSTAR_EDGES[mb+1]
        mask = np.isfinite(gobs_m[mb]) & (npair_m[mb] > 50)

        if np.any(mask):
            gc_fit = fit_gc(gbar_centers[mask], gobs_m[mb, mask])
            gc_label = f"gc = {gc_fit/a0_si:.3f} a0" if np.isfinite(gc_fit) else "gc: fit failed"

            ax.scatter(gbar_centers[mask] / a0_si, gobs_m[mb, mask] / a0_si,
                       c=colors[mb], s=40, zorder=3, label=gc_label)

        ax.plot(gb_plot / a0_si, g_mond / a0_si, 'r-', lw=1.5, alpha=0.7)
        ax.plot(gb_plot / a0_si, gb_plot / a0_si, 'k--', lw=1, alpha=0.3)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('g_bar / a0'); ax.set_ylabel('g_obs / a0')
        ax.set_title(f'M* bin {mb+1}: [{mlo:.1f}, {mhi:.1f})')
        ax.legend(fontsize=8)
        ax.set_xlim(1e-5, 1e2); ax.set_ylim(1e-3, 1e2)

    plt.tight_layout()
    figpath = os.path.join(OUT_DIR, "phase_b_rar.png")
    plt.savefig(figpath, dpi=150)
    print(f"  Saved: {figpath}")
    plt.close()


def analyze_gc_slope(gbar_centers, gobs_m, npair_m):
    print("\n=== gc vs M* analysis ===")

    n_mbins = len(MSTAR_EDGES) - 1
    logm_centers = []
    gc_values = []

    for mb in range(n_mbins):
        mlo, mhi = MSTAR_EDGES[mb], MSTAR_EDGES[mb+1]
        logm_c = (mlo + mhi) / 2

        mask = np.isfinite(gobs_m[mb]) & (npair_m[mb] > 50)
        if np.sum(mask) < 3:
            print(f"  M*-bin {mb+1} [{mlo:.1f},{mhi:.1f}): too few valid points")
            continue

        gc = fit_gc(gbar_centers[mask], gobs_m[mb, mask])
        gc_a0 = gc / a0_si
        print(f"  M*-bin {mb+1} [{mlo:.1f},{mhi:.1f}): gc = {gc_a0:.4f} a0 "
              f"({np.sum(mask)} pts, {np.sum(npair_m[mb]):.0f} pairs)")

        if np.isfinite(gc):
            logm_centers.append(logm_c)
            gc_values.append(gc_a0)

    if len(logm_centers) >= 2:
        logm_arr = np.array(logm_centers)
        loggc_arr = np.log10(gc_values)
        coeffs = np.polyfit(logm_arr, loggc_arr, 1)
        slope = coeffs[0]
        print(f"\n  >> gc-M* slope (Phase B): {slope:+.3f}")
        print(f"    (Phase A v1: +0.135, Phase A v2: -0.240)")
        print(f"    C15 prediction: +0.075")

    return logm_centers, gc_values


def main():
    print("=" * 60)
    print("Phase B Step 1: Corrected g_bar definition")
    print("  g_bar = G * M_gal / R^2 (baryonic point mass per lens)")
    print("  Binning in g_bar space (Brouwer+2021 methodology)")
    print("=" * 60)

    l_ra, l_dec, l_z, l_logm, l_Mgal = load_lenses()
    s_ra, s_dec, s_e1, s_e2, s_w, s_zbin, R_resp = load_hsc_sources()

    (gbar_centers, gbar_edges, esd, esd_x, gobs, gobs_x, npair,
     esd_m, gobs_m, npair_m) = \
        compute_esd_gbar_binned(l_ra, l_dec, l_z, l_logm, l_Mgal,
                                 s_ra, s_dec, s_e1, s_e2, s_w, s_zbin,
                                 R_resp)

    print("\n=== Phase B ESD summary ===")
    print(f"  g_bar bins: {N_GBAR_BINS} ({GBAR_MIN:.0e} to {GBAR_MAX:.0e} m/s^2)")
    print(f"  Responsivity: {R_resp:.4f}")
    for k in range(N_GBAR_BINS):
        if npair[k] > 0:
            print(f"  bin {k+1:2d}: g_bar={gbar_centers[k]/a0_si:.4e} a0, "
                  f"g_obs={gobs[k]/a0_si:.4e} a0, "
                  f"g_x={gobs_x[k]/a0_si:+.4e} a0, "
                  f"N={npair[k]}")

    save_results(gbar_centers, esd, esd_x, gobs, gobs_x, npair,
                 esd_m, gobs_m, npair_m)

    plot_rar(gbar_centers, gobs, gobs_x, gobs_m, npair, npair_m)

    analyze_gc_slope(gbar_centers, gobs_m, npair_m)

    print("\n=== Phase B Step 1 complete ===")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == '__main__':
    main()
