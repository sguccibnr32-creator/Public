#!/usr/bin/env python3
"""
cluster_stack_v3_fix.py
=======================
v3の2つのバグを修正:
  1. BCG同定: z_spec NULL除外 + 絶対等級フィルタ (M_r < -21)
  2. シェア計算: CSVカラム自動検出 + デバッグ出力
"""

import numpy as np
import csv
import sys
import json
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

OUTDIR = Path("cluster_stack_v3_output")
OUTDIR.mkdir(exist_ok=True)

c_light = 2.998e5
G_SI = 6.674e-11
Msun = 1.989e30
Mpc = 3.086e22
H0 = 70.0
Omega_m = 0.3
Omega_L = 0.7

CLUSTERS = {
    'cl1':  {'ra': 140.45, 'dec': -0.25, 'z': 0.313},
    'cl3':  {'ra': 139.80, 'dec': -0.10, 'z': 0.318},
    'cl4':  {'ra': 140.90, 'dec': -0.50, 'z': 0.315},
    'cl27': {'ra': 141.20, 'dec':  0.10, 'z': 0.322},
}


def bcg_identification():
    import requests

    print('=' * 60)
    print('BCG Identification (z_spec required, M_r < -21)')
    print('=' * 60)

    results = {}

    for cl_name, cl in CLUSTERS.items():
        z = cl['z']
        ra, dec = cl['ra'], cl['dec']
        search_r = 5.0

        D_L_Mpc = c_light * z / H0 * (1 + z/2)
        dist_mod = 5 * np.log10(D_L_Mpc * 1e6 / 10)

        r_limit = dist_mod - 21
        print(f'\n  {cl_name}: z={z:.3f}, D_L={D_L_Mpc:.0f} Mpc, '
              f'dist_mod={dist_mod:.1f}, r_limit={r_limit:.1f}')

        sql = f"""
        SELECT TOP 10
            p.objID, p.ra, p.dec, p.r, p.g, p.i,
            s.z as z_spec, s.zErr,
            p.petroRad_r, p.petroR50_r
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON p.objID = s.bestObjID
        WHERE
            p.ra BETWEEN {ra - search_r/60} AND {ra + search_r/60}
            AND p.dec BETWEEN {dec - search_r/60} AND {dec + search_r/60}
            AND p.type = 3
            AND s.z BETWEEN {z - 0.015} AND {z + 0.015}
            AND s.zWarning = 0
            AND p.r < {r_limit}
        ORDER BY p.r ASC
        """

        url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
        try:
            r_resp = requests.get(url, params={'cmd': sql, 'format': 'json'}, timeout=60)
            data = r_resp.json()

            rows = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'Rows' in item:
                        rows = item['Rows']
                        break
                    elif isinstance(item, dict) and 'ra' in item:
                        rows = data
                        break

            if rows:
                bcg = rows[0]
                bcg_ra = float(bcg['ra'])
                bcg_dec = float(bcg['dec'])
                bcg_r = float(bcg['r'])
                bcg_z = float(bcg['z_spec'])
                M_r = bcg_r - dist_mod

                offset_arcmin = np.sqrt(
                    ((bcg_ra - ra) * np.cos(np.radians(dec)) * 60)**2 +
                    ((bcg_dec - dec) * 60)**2
                )
                offset_Mpc = offset_arcmin / 60 * np.pi / 180 * D_L_Mpc / (1+z)

                results[cl_name] = {
                    'bcg_ra': bcg_ra, 'bcg_dec': bcg_dec,
                    'bcg_r_mag': bcg_r, 'bcg_z_spec': bcg_z,
                    'M_r': M_r,
                    'offset_arcmin': offset_arcmin,
                    'offset_Mpc': offset_Mpc,
                    'n_candidates': len(rows),
                }
                print(f'    BCG: RA={bcg_ra:.5f} Dec={bcg_dec:.5f}')
                print(f'    r={bcg_r:.2f}, M_r={M_r:.2f}, z_spec={bcg_z:.4f}')
                print(f'    Offset: {offset_arcmin:.2f} arcmin = {offset_Mpc:.3f} Mpc')
                print(f'    Candidates: {len(rows)}')

                for i, row in enumerate(rows[:5]):
                    off = np.sqrt(
                        ((float(row['ra'])-ra)*np.cos(np.radians(dec))*60)**2 +
                        ((float(row['dec'])-dec)*60)**2)
                    print(f'      #{i+1}: r={float(row["r"]):.2f}, '
                          f'z={float(row["z_spec"]):.4f}, '
                          f'offset={off:.2f}\'')
            else:
                print(f'    No spectroscopic members found')
                print(f'    -> Using original center as fallback')
                results[cl_name] = {
                    'bcg_ra': ra, 'bcg_dec': dec,
                    'offset_arcmin': 0, 'offset_Mpc': 0,
                    'fallback': True,
                }

        except Exception as e:
            print(f'    SDSS query failed: {e}')
            results[cl_name] = {
                'bcg_ra': ra, 'bcg_dec': dec,
                'offset_arcmin': 0, 'offset_Mpc': 0,
                'error': str(e),
            }

    return results


def find_shear_files():
    candidates = []
    for d in [OUTDIR, Path('.'), Path('..'), Path('cluster_stack_output'),
              Path('probes_vbar_output')]:
        if d.exists():
            for f in d.glob('*.csv'):
                if any(k in f.name.lower() for k in ['shear', 'source', 'photoz',
                                                       'background', 'cl1', 'weak']):
                    candidates.append(f)
    return candidates


def inspect_csv(filepath):
    print(f'\n  Inspecting: {filepath}')
    print(f'    Size: {filepath.stat().st_size:,} bytes')

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            print(f'    Columns ({len(header)}): {header[:15]}')
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f'    Row {i}: {row[:10]}')
            return header
    return None


def compute_shear_profile_v2(filepath, ra_c, dec_c, z_cl):
    col_maps = {
        'ra':  ['i_ra', 'ra', 'RA', 'RAJ2000', 'ra_gal', 'alpha'],
        'dec': ['i_dec', 'dec', 'DEC', 'DEJ2000', 'dec_gal', 'delta'],
        'e1':  ['e1', 'e1_regauss', 'g1', 'e1_hsm', 'ishape_hsm_regauss_e1'],
        'e2':  ['e2', 'e2_regauss', 'g2', 'e2_hsm', 'ishape_hsm_regauss_e2'],
        'w':   ['weight', 'w', 'ishape_hsm_regauss_derived_shape_weight', 'wt'],
        'zph': ['photoz_mean', 'z_phot', 'photo_z', 'z_best', 'photoz_best',
                'pz_best_z', 'mizuki_photoz_best'],
    }

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f'    CSV columns: {headers[:15]}')

        matched = {}
        for key, candidates in col_maps.items():
            for c in candidates:
                if c in headers:
                    matched[key] = c
                    break
            if key not in matched:
                for c in candidates:
                    for h in headers:
                        if c.lower() in h.lower():
                            matched[key] = h
                            break
                    if key in matched:
                        break

        print(f'    Matched columns: {matched}')

        missing = [k for k in ['ra', 'dec', 'e1', 'e2'] if k not in matched]
        if missing:
            print(f'    ERROR: Missing required columns: {missing}')
            print(f'    Available: {headers}')
            return None

        if 'w' not in matched:
            print(f'    Warning: no weight column, using w=1')
        if 'zph' not in matched:
            print(f'    Warning: no photo-z column, using z=1.0')

        f.seek(0)
        reader = csv.DictReader(f)

        gal_data = []
        n_read = 0
        n_skip_parse = 0
        n_skip_zph = 0

        for row in reader:
            n_read += 1
            try:
                ra_g = float(row[matched['ra']])
                dec_g = float(row[matched['dec']])
                e1 = float(row[matched['e1']])
                e2 = float(row[matched['e2']])
                w = float(row[matched['w']]) if 'w' in matched else 1.0
                zph = float(row[matched['zph']]) if 'zph' in matched else 1.0

                if w <= 0:
                    continue
                if zph <= z_cl + 0.1:
                    n_skip_zph += 1
                    continue

                gal_data.append((ra_g, dec_g, e1, e2, w, zph))
            except (ValueError, KeyError):
                n_skip_parse += 1

    print(f'    Read: {n_read}, Valid: {len(gal_data)}, '
          f'Skip(z): {n_skip_zph}, Skip(parse): {n_skip_parse}')

    if len(gal_data) < 50:
        print(f'    ERROR: Too few galaxies ({len(gal_data)})')
        return None

    from scipy.integrate import quad
    def D_A(z):
        f = lambda zp: 1.0 / np.sqrt(Omega_m*(1+zp)**3 + Omega_L)
        chi, _ = quad(f, 0, z)
        return c_light / H0 * chi / (1+z)

    D_l = D_A(z_cl)
    print(f'    D_A(z={z_cl}) = {D_l:.1f} Mpc')
    arcmin_to_Mpc = D_l * np.pi / (180 * 60)
    print(f'    1 arcmin = {arcmin_to_Mpc:.4f} Mpc')

    R_bins_Mpc = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    R_mid = np.sqrt(R_bins_Mpc[:-1] * R_bins_Mpc[1:])
    n_bins = len(R_mid)

    gt_sum = np.zeros(n_bins)
    gx_sum = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    n_count = np.zeros(n_bins, dtype=int)

    for ra_g, dec_g, e1, e2, w, zph in gal_data:
        dra = (ra_g - ra_c) * np.cos(np.radians(dec_c))
        ddec = dec_g - dec_c
        theta_deg = np.sqrt(dra**2 + ddec**2)
        R_Mpc = theta_deg * D_l * np.pi / 180

        if R_Mpc < R_bins_Mpc[0] or R_Mpc >= R_bins_Mpc[-1]:
            continue

        phi = np.arctan2(ddec, dra)
        gt = -(e1 * np.cos(2*phi) + e2 * np.sin(2*phi))
        gx = +(e2 * np.cos(2*phi) - e1 * np.sin(2*phi))

        idx = np.searchsorted(R_bins_Mpc, R_Mpc) - 1
        if 0 <= idx < n_bins:
            gt_sum[idx] += gt * w
            gx_sum[idx] += gx * w
            w_sum[idx] += w
            n_count[idx] += 1

    mask = w_sum > 0
    gamma_t = np.zeros(n_bins)
    gamma_x = np.zeros(n_bins)
    e_gamma = np.full(n_bins, np.inf)

    gamma_t[mask] = gt_sum[mask] / w_sum[mask]
    gamma_x[mask] = gx_sum[mask] / w_sum[mask]

    sigma_e = 0.26
    e_gamma[mask] = sigma_e / np.sqrt(n_count[mask])

    print(f'\n    Radial profile:')
    print(f'    {"R[Mpc]":>8} {"N":>6} {"gamma_t":>10} {"gamma_x":>10} {"S/N":>8}')
    for i in range(n_bins):
        sn = gamma_t[i] / e_gamma[i] if e_gamma[i] < np.inf else 0
        print(f'    {R_mid[i]:8.3f} {n_count[i]:6d} {gamma_t[i]:10.5f} '
              f'{gamma_x[i]:10.5f} {sn:8.2f}')

    total_sn = np.sqrt(np.sum((gamma_t[mask] / e_gamma[mask])**2))
    gx_median = np.median(np.abs(gamma_x[mask])) if np.any(mask) else 0

    print(f'\n    Total S/N = {total_sn:.1f}')
    print(f'    |gamma_x| median = {gx_median:.5f}')

    return {
        'R_Mpc': R_mid,
        'gamma_t': gamma_t,
        'gamma_x': gamma_x,
        'e_gamma': e_gamma,
        'n_count': n_count,
        'S_N': total_sn,
    }


def main():
    print('=' * 60)
    print('Cluster Stack v3 Fix')
    print('=' * 60)

    bcg = bcg_identification()

    print(f'\n{"="*60}')
    print('Shear File Search')
    print(f'{"="*60}')

    shear_files = find_shear_files()
    print(f'  Found {len(shear_files)} candidate files:')
    for f in shear_files:
        print(f'    {f}')

    if shear_files:
        for sf in shear_files[:5]:
            inspect_csv(sf)

    cl1_file = None
    for sf in shear_files:
        if 'cl1' in sf.name.lower():
            cl1_file = sf
            break
    if cl1_file is None and shear_files:
        cl1_file = shear_files[0]

    if cl1_file:
        cl1_bcg = bcg.get('cl1', {})

        print(f'\n{"="*60}')
        print(f'cl1 Shear Profile: Original Center ({CLUSTERS["cl1"]["ra"]}, {CLUSTERS["cl1"]["dec"]})')
        print(f'{"="*60}')
        prof_orig = compute_shear_profile_v2(
            cl1_file, CLUSTERS['cl1']['ra'], CLUSTERS['cl1']['dec'], CLUSTERS['cl1']['z'])

        if not cl1_bcg.get('fallback') and cl1_bcg.get('bcg_ra'):
            print(f'\n{"="*60}')
            print(f'cl1 Shear Profile: BCG Center ({cl1_bcg["bcg_ra"]:.5f}, {cl1_bcg["bcg_dec"]:.5f})')
            print(f'{"="*60}')
            prof_bcg = compute_shear_profile_v2(
                cl1_file, cl1_bcg['bcg_ra'], cl1_bcg['bcg_dec'], CLUSTERS['cl1']['z'])

            if prof_orig and prof_bcg:
                print(f'\n  Comparison:')
                print(f'    Original center: S/N = {prof_orig["S_N"]:.1f}')
                print(f'    BCG center:      S/N = {prof_bcg["S_N"]:.1f}')
                print(f'    -> {"BCG is better" if prof_bcg["S_N"] > prof_orig["S_N"] else "Original is better"}')
    else:
        print('\n  No shear data file found.')

    summary = {'bcg': {k: v for k, v in bcg.items()}}
    with open(OUTDIR / 'v3_fix_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    main()
