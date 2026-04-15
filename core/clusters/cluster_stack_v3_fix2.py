#!/usr/bin/env python3
"""
cluster_stack_v3_fix2.py
========================
cl1_sources_photoz.csv を正しく読み込み、
元中心 vs BCG中心 vs BCG候補#3 のシェアプロファイルを比較する。
"""

import numpy as np
import csv
import json
from pathlib import Path
from scipy.integrate import quad

OUTDIR = Path("cluster_stack_v3_output")
OUTDIR.mkdir(exist_ok=True)

c_light = 2.998e5; H0 = 70.0; Omega_m = 0.3; Omega_L = 0.7

CENTERS = {
    'original':   {'ra': 140.45,   'dec': -0.25,    'label': 'Original (140.45, -0.25)'},
    'bcg_top1':   {'ra': None,     'dec': None,     'label': 'BCG #1 (from SDSS)'},
    'bcg_top3':   {'ra': None,     'dec': None,     'label': 'BCG #3 (offset=1.16)'},
}
Z_CL = 0.313


def D_A(z):
    f = lambda zp: 1.0 / np.sqrt(Omega_m*(1+zp)**3 + Omega_L)
    chi, _ = quad(f, 0, z)
    return c_light / H0 * chi / (1+z)


def find_sources_file():
    candidates = []
    for d in [Path('.'), Path('..'), OUTDIR, Path('cluster_stack_output')]:
        if not d.exists():
            continue
        for f in d.rglob('*sources_photoz*.csv'):
            candidates.append(f)
        for f in d.rglob('*sources*.csv'):
            if 'individual' not in f.name.lower():
                candidates.append(f)

    candidates = list(set(candidates))
    # cl1優先
    candidates.sort(key=lambda f: ('cl1' not in f.name.lower(), -f.stat().st_size))

    print(f'Source file candidates:')
    for f in candidates[:5]:
        print(f'  {f}: {f.stat().st_size/1e6:.1f} MB')

    # cl1_sources_photoz.csv を優先
    for f in candidates:
        if 'cl1' in f.name.lower() and 'photoz' in f.name.lower():
            return f
    return candidates[0] if candidates else None


def load_sources(filepath):
    print(f'\nLoading: {filepath}')
    print(f'  Size: {filepath.stat().st_size/1e6:.1f} MB')

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()

    if first_line.startswith('#'):
        first_line = first_line[1:].strip()

    raw_headers = [h.strip() for h in first_line.split(',')]
    print(f'  Raw headers ({len(raw_headers)}): {raw_headers[:10]}...')

    col_map = {}
    for h in raw_headers:
        hl = h.lower()
        if hl in ('i_ra', 'ra'): col_map['ra'] = h
        elif hl in ('i_dec', 'dec'): col_map['dec'] = h
        elif 'e1' in hl and 'hsm' in hl: col_map['e1'] = h
        elif 'e2' in hl and 'hsm' in hl: col_map['e2'] = h
        elif 'derived_shape_weight' in hl or hl == 'weight': col_map['w'] = h
        elif 'derived_weight' in hl: col_map.setdefault('w', h)
        elif hl in ('photoz_best', 'photoz_mean', 'photo_z', 'z_phot'):
            col_map['zph'] = h
        elif 'photoz' in hl and 'zph' not in col_map and 'err' not in hl:
            col_map.setdefault('zph', h)

    print(f'  Column map: {col_map}')

    missing = [k for k in ['ra', 'dec', 'e1', 'e2'] if k not in col_map]
    if missing:
        print(f'  ERROR: Missing columns: {missing}')
        print(f'  All headers: {raw_headers}')
        return None

    col_idx = {}
    for key, col_name in col_map.items():
        for i, h in enumerate(raw_headers):
            if h.strip() == col_name.strip():
                col_idx[key] = i
                break

    print(f'  Column indices: {col_idx}')

    data = []
    n_total = 0
    n_bad = 0

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        f.readline()  # ヘッダスキップ
        for line in f:
            n_total += 1
            cols = line.strip().split(',')
            try:
                ra = float(cols[col_idx['ra']])
                dec = float(cols[col_idx['dec']])
                e1 = float(cols[col_idx['e1']])
                e2 = float(cols[col_idx['e2']])
                w = float(cols[col_idx['w']]) if 'w' in col_idx else 1.0
                zph = float(cols[col_idx['zph']]) if 'zph' in col_idx else 1.0

                if w > 0 and zph > Z_CL + 0.1:
                    data.append((ra, dec, e1, e2, w, zph))
            except (ValueError, IndexError):
                n_bad += 1

    print(f'  Total rows: {n_total}, Valid bg: {len(data)}, Bad: {n_bad}')
    return data


def shear_profile(sources, ra_c, dec_c, z_cl, label=''):
    D_l = D_A(z_cl)

    R_bins = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    R_mid = np.sqrt(R_bins[:-1] * R_bins[1:])
    n_bins = len(R_mid)

    gt_sum = np.zeros(n_bins)
    gx_sum = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    n_count = np.zeros(n_bins, dtype=int)

    for ra_g, dec_g, e1, e2, w, zph in sources:
        dra = (ra_g - ra_c) * np.cos(np.radians(dec_c))
        ddec = dec_g - dec_c
        R_Mpc = np.sqrt(dra**2 + ddec**2) * np.pi / 180 * D_l

        if R_Mpc < R_bins[0] or R_Mpc >= R_bins[-1]:
            continue

        phi = np.arctan2(ddec, dra)
        gt = -(e1 * np.cos(2*phi) + e2 * np.sin(2*phi))
        gx = +(e2 * np.cos(2*phi) - e1 * np.sin(2*phi))

        idx = np.searchsorted(R_bins, R_Mpc) - 1
        if 0 <= idx < n_bins:
            gt_sum[idx] += gt * w
            gx_sum[idx] += gx * w
            w_sum[idx] += w
            n_count[idx] += 1

    mask = w_sum > 0
    gamma_t = np.zeros(n_bins)
    gamma_x = np.zeros(n_bins)
    gamma_t[mask] = gt_sum[mask] / w_sum[mask]
    gamma_x[mask] = gx_sum[mask] / w_sum[mask]

    sigma_e = 0.26
    e_gamma = np.full(n_bins, np.inf)
    e_gamma[mask] = sigma_e / np.sqrt(n_count[mask])

    print(f'\n  Profile [{label}]:')
    print(f'  {"R[Mpc]":>8} {"N":>7} {"gamma_t":>10} {"gamma_x":>10} {"S/N":>7}')
    for i in range(n_bins):
        sn = gamma_t[i] / e_gamma[i] if e_gamma[i] < np.inf else 0
        print(f'  {R_mid[i]:8.3f} {n_count[i]:7d} {gamma_t[i]:10.6f} '
              f'{gamma_x[i]:10.6f} {sn:7.2f}')

    total_sn = np.sqrt(np.sum(np.where(mask, (gamma_t/e_gamma)**2, 0)))
    gx_med = np.median(np.abs(gamma_x[mask])) if np.any(mask) else 0
    print(f'  Total S/N = {total_sn:.1f}, |gamma_x| median = {gx_med:.6f}')

    return {
        'R': R_mid, 'gt': gamma_t, 'gx': gamma_x, 'eg': e_gamma,
        'n': n_count, 'SN': total_sn, 'gx_med': gx_med,
    }


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('=' * 60)
    print('cl1 Shear Profile: Multi-Center Comparison')
    print('=' * 60)

    src_file = find_sources_file()
    if src_file is None:
        print('ERROR: No source file found.')
        return

    sources = load_sources(src_file)
    if sources is None:
        return

    try:
        import requests
        sql = """
        SELECT TOP 5 p.ra, p.dec, p.r, s.z as z_spec
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON p.objID = s.bestObjID
        WHERE p.ra BETWEEN 140.3 AND 140.6
          AND p.dec BETWEEN -0.35 AND -0.15
          AND p.type = 3
          AND s.z BETWEEN 0.298 AND 0.328
          AND s.zWarning = 0
          AND p.r < 20
        ORDER BY p.r ASC
        """
        r = requests.get(
            "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
            params={'cmd': sql, 'format': 'json'}, timeout=30)
        data = r.json()
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
            print(f'\nSDSS BCG candidates for cl1:')
            for i, row in enumerate(rows):
                off = np.sqrt(
                    ((float(row['ra'])-140.45)*np.cos(np.radians(-0.25))*60)**2 +
                    ((float(row['dec'])+0.25)*60)**2)
                print(f'  #{i+1}: RA={float(row["ra"]):.5f} Dec={float(row["dec"]):.5f} '
                      f'r={float(row["r"]):.2f} z={float(row["z_spec"]):.4f} '
                      f'offset={off:.2f}')

            CENTERS['bcg_top1']['ra'] = float(rows[0]['ra'])
            CENTERS['bcg_top1']['dec'] = float(rows[0]['dec'])
            if len(rows) >= 3:
                CENTERS['bcg_top3']['ra'] = float(rows[2]['ra'])
                CENTERS['bcg_top3']['dec'] = float(rows[2]['dec'])
                CENTERS['bcg_top3']['label'] = f'BCG #3 ({float(rows[2]["ra"]):.4f}, {float(rows[2]["dec"]):.4f})'
    except Exception as e:
        print(f'SDSS query failed: {e}')

    profiles = {}
    for key, center in CENTERS.items():
        if center['ra'] is None:
            continue
        print(f'\n{"="*60}')
        print(f'Center: {center["label"]}')
        print(f'  RA={center["ra"]:.5f}, Dec={center["dec"]:.5f}')
        print(f'{"="*60}')
        prof = shear_profile(sources, center['ra'], center['dec'], Z_CL, center['label'])
        profiles[key] = prof

    print(f'\n{"="*60}')
    print('Summary: S/N and |gamma_x| by center')
    print(f'{"="*60}')
    print(f'{"Center":<45} {"S/N":>7} {"|gx|":>10} {"Best?":>6}')
    if profiles:
        best_sn = max(p['SN'] for p in profiles.values())
        for key, prof in profiles.items():
            is_best = 'YES' if prof['SN'] == best_sn else ''
            print(f'{CENTERS[key]["label"]:<45} {prof["SN"]:7.1f} {prof["gx_med"]:10.6f} {is_best:>6}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'original': '#1a1a2e', 'bcg_top1': '#e94560', 'bcg_top3': '#2ecc71'}

    ax = axes[0]
    for key, prof in profiles.items():
        mask = prof['n'] > 0
        c = colors.get(key, 'grey')
        ax.errorbar(prof['R'][mask], prof['gt'][mask], yerr=prof['eg'][mask],
                    fmt='o-', color=c, capsize=3, markersize=5,
                    label=f'{CENTERS[key]["label"]} (S/N={prof["SN"]:.1f})')
    ax.axhline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('gamma_t')
    ax.set_xscale('log')
    ax.set_title('(a) Tangential shear profile')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, prof in profiles.items():
        mask = prof['n'] > 0
        c = colors.get(key, 'grey')
        ax.errorbar(prof['R'][mask], prof['gx'][mask], yerr=prof['eg'][mask],
                    fmt='o-', color=c, capsize=3, markersize=5,
                    label=CENTERS[key]['label'])
    ax.axhline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('gamma_x (null test)')
    ax.set_xscale('log')
    ax.set_title('(b) Cross shear (should be ~0)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('cl1 Weak Lensing: Original vs BCG Center', fontsize=13)
    plt.tight_layout()
    fig_path = OUTDIR / 'cl1_center_comparison.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\nFigure: {fig_path}')

    summary = {k: {'SN': p['SN'], 'gx_med': p['gx_med'],
                    'gt_max': float(np.max(p['gt'])),
                    'center': CENTERS[k]}
               for k, p in profiles.items()}
    with open(OUTDIR / 'center_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    main()
