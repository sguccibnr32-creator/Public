# -*- coding: utf-8 -*-
"""
phase_a_step34_esd.py

Step 3+4: HSC Y3 source-lens pairing + ESD calculation (single pass).
"""

import os, sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.table import Table

HSC_FILE = Path(r"E:\スバル望遠鏡データ\931720.csv.gz.1")
LENS_FILE = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase_a_output\gama_lenses_g12_isolated.fits")
OUTPUT_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase_a_output")

HSC_COLS = {
    'ra':     'i_ra',
    'dec':    'i_dec',
    'e1':     'i_hsmshaperegauss_e1',
    'e2':     'i_hsmshaperegauss_e2',
    'weight': 'i_hsmshaperegauss_derived_weight',
    'zbin':   'hsc_y3_zbin',
}

# Step 2 saved lenses use uberID / RAcen / Deccen / Z / logmstar
LENS_COLS = {
    'id':     'uberID',
    'ra':     'RAcen',
    'dec':    'Deccen',
    'z':      'Z',
    'logm':   'logmstar',
}

N_RBINS = 15
R_MIN_MPC = 0.02
R_MAX_MPC = 3.0
R_EDGES = np.logspace(np.log10(R_MIN_MPC), np.log10(R_MAX_MPC), N_RBINS + 1)
R_MID = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])

MSTAR_BINS = [8.5, 10.3, 10.6, 10.8, 11.0, 11.5]
N_MBINS = len(MSTAR_BINS) - 1

ZBIN_ZEFF = {0: 0.0, 1: 0.44, 2: 0.75, 3: 1.01, 4: 1.30}

G12_RA_MIN, G12_RA_MAX = 172.0, 188.0
G12_DEC_MIN, G12_DEC_MAX = -4.0, 3.0

DZ_MIN = 0.1

H0 = 70.0
Om0 = 0.3
c_km = 2.998e5
c_m = 2.998e8
G_SI = 6.674e-11
Msun_kg = 1.989e30
Mpc_m = 3.0857e22
pc_m = 3.0857e16

REPORT_INTERVAL = 2_000_000


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
    n_lens = len(z_lens_arr)
    sc_table = np.zeros((n_lens, 5))
    for i in range(n_lens):
        for zb in range(5):
            z_s = ZBIN_ZEFF[zb]
            if z_s > z_lens_arr[i] + DZ_MIN:
                sc_table[i, zb] = sigma_crit(z_lens_arr[i], z_s)
            else:
                sc_table[i, zb] = np.inf
    return sc_table


class ESDAccumulator:
    def __init__(self):
        self.sum_wgt = np.zeros(N_RBINS)
        self.sum_wgt_gt = np.zeros(N_RBINS)
        self.sum_wgt_gx = np.zeros(N_RBINS)
        self.n_pairs = np.zeros(N_RBINS, dtype=np.int64)
        self.mbin_sum_wgt = np.zeros((N_MBINS, N_RBINS))
        self.mbin_sum_wgt_gt = np.zeros((N_MBINS, N_RBINS))
        self.mbin_sum_wgt_gx = np.zeros((N_MBINS, N_RBINS))
        self.mbin_n_pairs = np.zeros((N_MBINS, N_RBINS), dtype=np.int64)

    def add(self, r_mpc, gamma_t, gamma_x, w_src, sc_inv, mbin_idx):
        if r_mpc < R_MIN_MPC or r_mpc >= R_MAX_MPC:
            return
        rbin = np.searchsorted(R_EDGES, r_mpc) - 1
        if rbin < 0 or rbin >= N_RBINS:
            return
        w_eff = w_src * sc_inv**2
        self.sum_wgt[rbin] += w_eff
        self.sum_wgt_gt[rbin] += w_src * sc_inv * gamma_t
        self.sum_wgt_gx[rbin] += w_src * sc_inv * gamma_x
        self.n_pairs[rbin] += 1
        if 0 <= mbin_idx < N_MBINS:
            self.mbin_sum_wgt[mbin_idx, rbin] += w_eff
            self.mbin_sum_wgt_gt[mbin_idx, rbin] += w_src * sc_inv * gamma_t
            self.mbin_sum_wgt_gx[mbin_idx, rbin] += w_src * sc_inv * gamma_x
            self.mbin_n_pairs[mbin_idx, rbin] += 1

    def get_esd(self, which='all', mbin=None):
        if which == 'all':
            sw = self.sum_wgt
            sgt = self.sum_wgt_gt
            sgx = self.sum_wgt_gx
            np_arr = self.n_pairs
        else:
            sw = self.mbin_sum_wgt[mbin]
            sgt = self.mbin_sum_wgt_gt[mbin]
            sgx = self.mbin_sum_wgt_gx[mbin]
            np_arr = self.mbin_n_pairs[mbin]
        esd_t = np.where(sw > 0, sgt / sw, 0)
        esd_x = np.where(sw > 0, sgx / sw, 0)
        return R_MID, esd_t, esd_x, np_arr


def hsc_distortion_to_shear(e1_dist, e2_dist):
    return e1_dist / 2.0, e2_dist / 2.0


def main():
    t_start = time.time()
    print("=" * 72)
    print("  STEP 3+4: HSC SOURCE-LENS PAIRING + ESD")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading lenses: {LENS_FILE.name}")
    lenses = Table.read(LENS_FILE)
    n_lens = len(lenses)
    print(f"  N_lenses = {n_lens:,}")

    lens_ra = np.array(lenses[LENS_COLS['ra']], dtype=np.float64)
    lens_dec = np.array(lenses[LENS_COLS['dec']], dtype=np.float64)
    lens_z = np.array(lenses[LENS_COLS['z']], dtype=np.float64)
    lens_logm = np.array(lenses[LENS_COLS['logm']], dtype=np.float64)

    lens_mbin = np.full(n_lens, -1, dtype=int)
    for mb in range(N_MBINS):
        mask = (lens_logm >= MSTAR_BINS[mb]) & (lens_logm < MSTAR_BINS[mb + 1])
        lens_mbin[mask] = mb
    print(f"  M* bin counts: {[int((lens_mbin==mb).sum()) for mb in range(N_MBINS)]}")

    print(f"  Pre-computing Sigma_crit...", flush=True)
    sc_table = precompute_sigma_crit(lens_z)
    sci_table = np.where(np.isinf(sc_table), 0, 1.0 / sc_table)
    print(f"  Done.")

    cos_dec_ref = np.cos(np.radians(np.median(lens_dec)))
    lens_xy = np.column_stack([lens_ra * cos_dec_ref, lens_dec])
    lens_tree = cKDTree(lens_xy)

    D_A_min = angular_diameter_distance(np.min(lens_z))
    theta_max_deg = np.degrees(R_MAX_MPC / D_A_min)
    print(f"  Max search radius: {theta_max_deg:.4f} deg "
          f"(R_max={R_MAX_MPC} Mpc at z_min={np.min(lens_z):.3f})")

    acc = ESDAccumulator()

    print(f"\n  Reading HSC sources: {HSC_FILE.name}")
    print(f"  File size: {HSC_FILE.stat().st_size/1e9:.2f} GB")

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
    n_paired = 0
    total_pairs = 0
    header = None
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
                        hdr_parts = [c.strip() for c in header_candidate.split(',')]
                        if len(hdr_parts) < 5:
                            hdr_parts = header_candidate.split()
                        header = hdr_parts
                    else:
                        print("  ERROR: Cannot identify header")
                        return
                except ValueError:
                    header = parts
                    is_data_first = False

                for key, col_name in HSC_COLS.items():
                    for j, h in enumerate(header):
                        if h == col_name:
                            col_idx[key] = j
                            break

                if len(col_idx) < 4:
                    print(f"  ERROR: Insufficient column matches: {col_idx}")
                    print(f"  Header sample: {header[:10]}")
                    return

                print(f"  Column mapping: {col_idx}")
                print(f"  Processing sources...", flush=True)

                if not is_data_first:
                    continue

            n_read += 1
            parts = [c.strip() for c in line.split(',')]

            try:
                src_ra = float(parts[col_idx['ra']])
                src_dec = float(parts[col_idx['dec']])
            except (ValueError, IndexError):
                continue

            if (src_ra < G12_RA_MIN or src_ra > G12_RA_MAX or
                src_dec < G12_DEC_MIN or src_dec > G12_DEC_MAX):
                continue

            n_g12 += 1

            try:
                src_e1 = float(parts[col_idx['e1']])
                src_e2 = float(parts[col_idx['e2']])
                src_w = float(parts[col_idx['weight']])
                src_zbin = int(float(parts[col_idx['zbin']]))
            except (ValueError, IndexError):
                continue

            if src_w <= 0 or src_zbin < 1 or src_zbin > 4:
                continue

            g1, g2 = hsc_distortion_to_shear(src_e1, src_e2)

            src_xy = np.array([src_ra * cos_dec_ref, src_dec])
            nearby = lens_tree.query_ball_point(src_xy, theta_max_deg)

            if not nearby:
                continue

            n_paired += 1

            for li in nearby:
                z_s_eff = ZBIN_ZEFF[src_zbin]
                if z_s_eff <= lens_z[li] + DZ_MIN:
                    continue

                sc_inv = sci_table[li, src_zbin]
                if sc_inv <= 0:
                    continue

                dra = (src_ra - lens_ra[li]) * cos_dec_ref * np.pi / 180
                ddec = (src_dec - lens_dec[li]) * np.pi / 180
                theta_rad = np.sqrt(dra**2 + ddec**2)
                D_A = angular_diameter_distance(lens_z[li])
                r_mpc = theta_rad * D_A

                if r_mpc < R_MIN_MPC or r_mpc >= R_MAX_MPC:
                    continue

                phi = np.arctan2(ddec, dra)
                gamma_t = -(g1 * np.cos(2 * phi) + g2 * np.sin(2 * phi))
                gamma_x =  (g1 * np.sin(2 * phi) - g2 * np.cos(2 * phi))

                acc.add(r_mpc, gamma_t, gamma_x, src_w, sc_inv, lens_mbin[li])
                total_pairs += 1

            if n_read % REPORT_INTERVAL == 0:
                elapsed = time.time() - t_start
                rate = n_read / elapsed
                print(f"    {n_read/1e6:.1f}M read, {n_g12:,} in G12, "
                      f"{n_paired:,} paired, {total_pairs:,} ESD pairs, "
                      f"{rate:.0f} lines/s, {elapsed:.0f}s elapsed",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"\n  READING COMPLETE:")
    print(f"    Total lines read: {n_read:,}")
    print(f"    Sources in G12:   {n_g12:,}")
    print(f"    Sources paired:   {n_paired:,}")
    print(f"    Total ESD pairs:  {total_pairs:,}")
    print(f"    Elapsed time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n{'='*72}\n  ESD PROFILES\n{'='*72}")

    R, esd_t, esd_x, n_pairs = acc.get_esd('all')
    print(f"\n  All lenses (N={n_lens:,}):")
    print(f"  {'R(Mpc)':>10s}  {'ESD_t':>14s}  {'ESD_x':>14s}  {'N_pairs':>12s}")
    for b in range(N_RBINS):
        print(f"  {R[b]:10.4f}  {esd_t[b]:14.4f}  {esd_x[b]:14.4f}  "
              f"{n_pairs[b]:12,d}")

    out_all = OUTPUT_DIR / 'esd_profile_all.txt'
    np.savetxt(out_all,
               np.column_stack([R, esd_t, esd_x, n_pairs]),
               header='R_Mpc  ESD_t(h70Msun/pc2)  ESD_x  N_pairs',
               fmt=['%.6e', '%.6e', '%.6e', '%d'])
    print(f"  Saved: {out_all}")

    for mb in range(N_MBINS):
        R, esd_t, esd_x, n_pairs = acc.get_esd('mbin', mbin=mb)
        n_lens_mb = (lens_mbin == mb).sum()
        print(f"\n  M* bin {mb+1} [{MSTAR_BINS[mb]:.1f}, {MSTAR_BINS[mb+1]:.1f}) "
              f"(N_lens={n_lens_mb:,}):")
        print(f"  {'R(Mpc)':>10s}  {'ESD_t':>14s}  {'ESD_x':>14s}  {'N_pairs':>12s}")
        for b in range(N_RBINS):
            print(f"  {R[b]:10.4f}  {esd_t[b]:14.4f}  {esd_x[b]:14.4f}  "
                  f"{n_pairs[b]:12,d}")

        out_mb = OUTPUT_DIR / f'esd_profile_mbin_{mb+1}.txt'
        np.savetxt(out_mb,
                   np.column_stack([R, esd_t, esd_x, n_pairs]),
                   header=f'R_Mpc  ESD_t  ESD_x  N_pairs  '
                          f'[M*bin {MSTAR_BINS[mb]:.1f}-{MSTAR_BINS[mb+1]:.1f}]',
                   fmt=['%.6e', '%.6e', '%.6e', '%d'])
        print(f"  Saved: {out_mb}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"\n{'='*72}\n  STEP 3+4 COMPLETE\n{'='*72}")


if __name__ == '__main__':
    main()
