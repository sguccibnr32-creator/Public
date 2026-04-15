#!/usr/bin/env python3
"""
cluster_stack_v3.py
===================
スタック弱レンズ解析の改善版 (v3)
BCG同定 + NFW+2halo モデル
"""

import numpy as np
import sys
import json
from pathlib import Path
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2 as chi2_dist
from scipy.integrate import quad
from scipy.interpolate import interp1d

OUTDIR = Path("cluster_stack_v3_output")
OUTDIR.mkdir(exist_ok=True)

# 物理定数・宇宙論パラメータ
c_light = 2.998e5  # km/s
G_SI = 6.674e-11
Msun = 1.989e30
pc = 3.086e16
Mpc = 1e6 * pc
H0 = 70.0
Omega_m = 0.3
Omega_L = 0.7
rho_crit_0 = 3 * H0**2 / (8 * np.pi * G_SI) * (1e3/(Mpc))**2 / Msun * Mpc**3

CLUSTERS = {
    'cl1': {'ra': 140.45, 'dec': -0.25, 'z': 0.313, 'sigma_v': 527,
            'n_member': 22, 'notes': 'primary cluster'},
    'cl3': {'ra': 139.80, 'dec': -0.10, 'z': 0.318, 'sigma_v': None,
            'n_member': None, 'notes': 'stack candidate'},
    'cl4': {'ra': 140.90, 'dec': -0.50, 'z': 0.315, 'sigma_v': None,
            'n_member': None, 'notes': 'stack candidate'},
    'cl27': {'ra': 141.20, 'dec': 0.10, 'z': 0.322, 'sigma_v': None,
             'n_member': None, 'notes': 'stack candidate'},
}


def step1_bcg_identification():
    import requests

    print('=' * 60)
    print('Step 1: BCG Identification')
    print('=' * 60)

    bcg_results = {}

    for cl_name, cl in CLUSTERS.items():
        print(f'\n  {cl_name}: (RA={cl["ra"]}, Dec={cl["dec"]}, z={cl["z"]})')

        search_radius_arcmin = 5.0
        z_lo = cl['z'] - 0.01
        z_hi = cl['z'] + 0.01

        sql = f"""
        SELECT TOP 20
            p.objID, p.ra, p.dec, p.r, p.g, p.i,
            s.z as z_spec, s.zErr,
            p.type, p.petroRad_r
        FROM PhotoObj AS p
        LEFT JOIN SpecObj AS s ON p.objID = s.bestObjID
        WHERE
            p.ra BETWEEN {cl['ra'] - search_radius_arcmin/60} AND {cl['ra'] + search_radius_arcmin/60}
            AND p.dec BETWEEN {cl['dec'] - search_radius_arcmin/60} AND {cl['dec'] + search_radius_arcmin/60}
            AND p.type = 3
            AND p.r < 19.0
            AND (s.z BETWEEN {z_lo} AND {z_hi} OR s.z IS NULL)
        ORDER BY p.r ASC
        """

        url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
        try:
            r = requests.get(url, params={'cmd': sql, 'format': 'json'}, timeout=60)
            data = r.json()

            rows = []
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'Rows' in data[0]:
                    rows = data[0]['Rows']
                elif isinstance(data[0], dict):
                    rows = data

            if rows:
                bcg = rows[0]
                bcg_ra = float(bcg.get('ra', cl['ra']))
                bcg_dec = float(bcg.get('dec', cl['dec']))
                bcg_r = float(bcg.get('r', 99))
                bcg_z = bcg.get('z_spec', None)

                offset_arcmin = np.sqrt(
                    ((bcg_ra - cl['ra']) * np.cos(np.radians(cl['dec'])))**2 +
                    (bcg_dec - cl['dec'])**2
                ) * 60

                bcg_results[cl_name] = {
                    'bcg_ra': bcg_ra,
                    'bcg_dec': bcg_dec,
                    'bcg_r_mag': bcg_r,
                    'bcg_z_spec': bcg_z,
                    'offset_arcmin': offset_arcmin,
                    'n_candidates': len(rows),
                }
                print(f'    BCG: RA={bcg_ra:.4f}, Dec={bcg_dec:.4f}, r={bcg_r:.2f}')
                print(f'    Offset: {offset_arcmin:.2f} arcmin')
                if bcg_z:
                    print(f'    z_spec: {bcg_z}')
            else:
                print(f'    No candidates found')
                bcg_results[cl_name] = {
                    'bcg_ra': cl['ra'], 'bcg_dec': cl['dec'],
                    'bcg_r_mag': None, 'bcg_z_spec': None,
                    'offset_arcmin': 0.0, 'n_candidates': 0, 'fallback': True,
                }

        except Exception as e:
            print(f'    SDSS query failed: {e}')
            bcg_results[cl_name] = {
                'bcg_ra': cl['ra'], 'bcg_dec': cl['dec'],
                'bcg_r_mag': None, 'bcg_z_spec': None,
                'offset_arcmin': 0.0, 'n_candidates': 0, 'fallback': True,
            }

    return bcg_results


def nfw_sigma(R_Mpc, M200, c200, z):
    Ez2 = Omega_m * (1+z)**3 + Omega_L
    rho_c = rho_crit_0 * Ez2

    r200 = (3 * M200 / (4 * np.pi * 200 * rho_c))**(1.0/3)
    rs = r200 / c200

    x = R_Mpc / rs
    x = np.clip(x, 1e-6, 1e4)

    sigma = np.zeros_like(x)

    m1 = x < 1
    if np.any(m1):
        t = np.sqrt(1 - x[m1]**2)
        sigma[m1] = 1.0 / (x[m1]**2 - 1) * (1 - np.arctanh(t) / t)

    m2 = x > 1
    if np.any(m2):
        t = np.sqrt(x[m2]**2 - 1)
        sigma[m2] = 1.0 / (x[m2]**2 - 1) * (1 - np.arctan(t) / t)

    m3 = np.abs(x - 1) < 1e-6
    sigma[m3] = 1.0 / 3

    rho_s = M200 / (4 * np.pi * rs**3 * (np.log(1+c200) - c200/(1+c200)))
    Sigma_phys = 2 * rs * rho_s * sigma

    return Sigma_phys


def nfw_delta_sigma(R_Mpc, M200, c200, z):
    Sigma_R = nfw_sigma(R_Mpc, M200, c200, z)

    Sigma_mean = np.zeros_like(R_Mpc)
    for i, Ri in enumerate(R_Mpc):
        r_int = np.linspace(0.001, Ri, 200)
        S_int = nfw_sigma(r_int, M200, c200, z)
        Sigma_mean[i] = 2 * np.trapezoid(r_int * S_int, r_int) / Ri**2

    return Sigma_mean - Sigma_R


def two_halo_delta_sigma(R_Mpc, M200, b_lin, z):
    Ez2 = Omega_m * (1+z)**3 + Omega_L
    rho_m = rho_crit_0 * Omega_m * (1+z)**3

    R0 = 5.0
    xi_mm = (R_Mpc / R0)**(-1.8)

    Delta_Sigma_2h = b_lin * rho_m * R0 * xi_mm / 1e6

    return Delta_Sigma_2h


def model_nfw_2halo(R_Mpc, M200, c200, b_lin, z):
    ds_1h = nfw_delta_sigma(R_Mpc, M200, c200, z)
    ds_2h = two_halo_delta_sigma(R_Mpc, M200, b_lin, z)
    return ds_1h + ds_2h


def step3_hsc_shear(bcg_results):
    print('\n' + '=' * 60)
    print('Step 3: HSC Shear Data')
    print('=' * 60)

    for cl_name, bcg in bcg_results.items():
        ra_c = bcg['bcg_ra']
        dec_c = bcg['bcg_dec']
        z_cl = CLUSTERS[cl_name]['z']

        sql = f"""
        SELECT
            object_id, i_ra, i_dec,
            e1, e2, weight, m_bias,
            photoz_mean, photoz_err,
            b_mode_mask
        FROM s21a_wide.weaklensing_hsm_regauss
        WHERE
            b_mode_mask = 1
            AND coneSearch(i_ra, i_dec, {ra_c}, {dec_c}, 30.0)
            AND photoz_mean > {z_cl + 0.1}
            AND weight > 0
        """

        print(f'\n  {cl_name} (center: BCG at {ra_c:.4f}, {dec_c:.4f}):')
        (OUTDIR / f"hsc_query_{cl_name}.sql").write_text(sql)
        print(f'  HSC SQL query saved to: {OUTDIR / f"hsc_query_{cl_name}.sql"}')

    existing = {}
    for cl_name in CLUSTERS:
        for pattern in [f'hsc_shear_{cl_name}.csv', f'cl1_shear.csv',
                       f'{cl_name}_background.csv', f'{cl_name}_sources_photoz.csv']:
            for search_dir in [OUTDIR, Path('.'), Path('..'),
                              Path('cluster_stack_output'), Path('cluster_stack_v2_output')]:
                f = search_dir / pattern
                if f.exists():
                    print(f'  Found existing: {f}')
                    existing[cl_name] = f
                    break
            if cl_name in existing:
                break

    return existing


def compute_shear_profile(shear_file, ra_c, dec_c, z_cl, z_s_min=None):
    import csv

    data = []
    with open(shear_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ra = float(row.get('i_ra', row.get('ra', '0')))
                dec = float(row.get('i_dec', row.get('dec', '0')))
                e1 = float(row.get('e1', '0'))
                e2 = float(row.get('e2', '0'))
                w = float(row.get('weight', '1'))
                z_ph = float(row.get('photoz_mean', row.get('z_phot', '1.0')))

                if w > 0 and z_ph > z_cl + 0.1:
                    data.append({
                        'ra': ra, 'dec': dec, 'e1': e1, 'e2': e2,
                        'w': w, 'z_ph': z_ph,
                    })
            except (ValueError, TypeError):
                pass

    if len(data) < 50:
        print(f'  Warning: only {len(data)} background galaxies')
        return None

    print(f'  {len(data)} background galaxies')

    def D_A(z):
        f = lambda zp: 1.0 / np.sqrt(Omega_m*(1+zp)**3 + Omega_L)
        chi, _ = quad(f, 0, z)
        return c_light / H0 * chi / (1+z)

    D_l = D_A(z_cl)

    R_bins = np.logspace(np.log10(0.1), np.log10(5.0), 12)
    R_mid = np.sqrt(R_bins[:-1] * R_bins[1:])

    gamma_t = np.zeros(len(R_mid))
    gamma_x = np.zeros(len(R_mid))
    w_sum = np.zeros(len(R_mid))
    n_count = np.zeros(len(R_mid), dtype=int)

    for gal in data:
        dra = (gal['ra'] - ra_c) * np.cos(np.radians(dec_c))
        ddec = gal['dec'] - dec_c
        theta = np.sqrt(dra**2 + ddec**2) * np.pi / 180
        R_phys = theta * D_l

        phi = np.arctan2(ddec, dra)

        gt = -(gal['e1'] * np.cos(2*phi) + gal['e2'] * np.sin(2*phi))
        gx = -(gal['e2'] * np.cos(2*phi) - gal['e1'] * np.sin(2*phi))

        D_s = D_A(gal['z_ph'])
        D_ls = D_A(gal['z_ph']) - D_A(z_cl)
        if D_ls <= 0:
            continue

        idx = np.searchsorted(R_bins, R_phys) - 1
        if 0 <= idx < len(R_mid):
            gamma_t[idx] += gt * gal['w']
            gamma_x[idx] += gx * gal['w']
            w_sum[idx] += gal['w']
            n_count[idx] += 1

    mask = w_sum > 0
    gamma_t[mask] /= w_sum[mask]
    gamma_x[mask] /= w_sum[mask]
    e_gamma = np.where(n_count > 1, 1.0 / np.sqrt(n_count * w_sum / np.maximum(w_sum, 1)), 1e10)

    return {
        'R_Mpc': R_mid,
        'gamma_t': gamma_t,
        'gamma_x': gamma_x,
        'e_gamma': e_gamma,
        'n_count': n_count,
        'n_total': len(data),
    }


def fit_models(profile, z_cl):
    R = profile['R_Mpc']
    gt = profile['gamma_t']
    eg = profile['e_gamma']

    mask = (gt != 0) & (eg < 1)
    R_fit = R[mask]
    gt_fit = gt[mask]
    eg_fit = eg[mask]

    if len(R_fit) < 4:
        return None

    results = {}

    try:
        def nfw_model(R, logM, c):
            return nfw_delta_sigma(R, 10**logM, c, z_cl) * 1e-15

        popt, pcov = curve_fit(nfw_model, R_fit, gt_fit, p0=[14.5, 5],
                               sigma=eg_fit, maxfev=5000)
        gt_pred = nfw_model(R_fit, *popt)
        chi2_nfw = np.sum(((gt_fit - gt_pred) / eg_fit)**2)
        dof_nfw = len(R_fit) - 2
        results['NFW'] = {
            'logM200': popt[0], 'c200': popt[1],
            'chi2': chi2_nfw, 'dof': dof_nfw,
            'chi2_dof': chi2_nfw / dof_nfw,
            'AIC': chi2_nfw + 2*2,
        }
    except Exception as e:
        print(f'  NFW fit failed: {e}')

    try:
        def nfw_2h_model(R, logM, c, b):
            return model_nfw_2halo(R, 10**logM, c, b, z_cl) * 1e-15

        popt2, pcov2 = curve_fit(nfw_2h_model, R_fit, gt_fit, p0=[14.5, 5, 3],
                                 sigma=eg_fit, maxfev=5000)
        gt_pred2 = nfw_2h_model(R_fit, *popt2)
        chi2_2h = np.sum(((gt_fit - gt_pred2) / eg_fit)**2)
        dof_2h = len(R_fit) - 3
        results['NFW+2halo'] = {
            'logM200': popt2[0], 'c200': popt2[1], 'b_lin': popt2[2],
            'chi2': chi2_2h, 'dof': dof_2h,
            'chi2_dof': chi2_2h / dof_2h,
            'AIC': chi2_2h + 2*3,
        }
    except Exception as e:
        print(f'  NFW+2halo fit failed: {e}')

    if results:
        aic_min = min(r['AIC'] for r in results.values())
        for name, r in results.items():
            r['DAIC'] = r['AIC'] - aic_min

    return results


def main():
    print('=' * 60)
    print('Cluster Stack Weak Lensing v3')
    print('BCG identification + NFW+2halo model')
    print('=' * 60)

    bcg = step1_bcg_identification()

    print(f'\n--- BCG Summary ---')
    for cl_name, b in bcg.items():
        offset = b.get('offset_arcmin', 0)
        print(f'  {cl_name}: offset={offset:.2f} arcmin, '
              f'r_mag={b.get("bcg_r_mag", "N/A")}')

    existing = step3_hsc_shear(bcg)

    if existing:
        print(f'\n--- Processing existing shear data ---')
        for cl_name, shear_file in existing.items():
            b = bcg.get(cl_name, {})
            ra_c = b.get('bcg_ra', CLUSTERS[cl_name]['ra'])
            dec_c = b.get('bcg_dec', CLUSTERS[cl_name]['dec'])
            z_cl = CLUSTERS[cl_name]['z']

            print(f'\n  {cl_name}:')
            profile = compute_shear_profile(shear_file, ra_c, dec_c, z_cl)
            if profile:
                sn = np.sum(profile["gamma_t"]**2 / np.maximum(profile["e_gamma"]**2, 1e-20))**0.5
                print(f'    S/N = {sn:.1f}')
                print(f'    |gamma_x| = {np.median(np.abs(profile["gamma_x"])):.4f}')

                results = fit_models(profile, z_cl)
                if results:
                    print(f'\n    Model fits:')
                    for name, r in results.items():
                        print(f'      {name}: chi2/dof={r["chi2_dof"]:.2f}, '
                              f'DAIC={r["DAIC"]:.1f}')
    else:
        print('\n  No existing shear data found.')
        print('  Run HSC queries manually, then re-run this script.')

    summary = {
        'bcg': {k: {kk: vv for kk, vv in v.items()} for k, v in bcg.items()},
        'existing_data': {k: str(v) for k, v in existing.items()} if existing else {},
    }
    with open(OUTDIR / 'stack_v3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\nResults: {OUTDIR}')


if __name__ == '__main__':
    main()
