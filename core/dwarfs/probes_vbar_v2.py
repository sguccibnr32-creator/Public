#!/usr/bin/env python3
"""
probes_vbar_v2.py
=================
PROBES自前V_bar構築パイプライン v2
Yd 個別最適化 + 実測HI面密度プロファイル
"""

import numpy as np
import sys
import json
import os
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import i0, i1, k0, k1
from scipy.stats import linregress, t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _fp in ['/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
            '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf']:
    try: _fm.fontManager.addfont(_fp)
    except: pass
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# === 物理定数 ===
G_SI = 6.674e-11
Msun = 1.989e30
pc = 3.086e16
kpc = 1e3 * pc
a0 = 1.2e-10
M_sun_W1 = 3.24

OUTDIR = Path("probes_vbar_v2_output")
OUTDIR.mkdir(exist_ok=True)


def fetch_vizier_tsv(source, out_params, max_rows=999999, outfile=None):
    import requests
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    params = {
        "-source": source,
        "-out.max": str(max_rows),
        "-out": out_params,
        "-out.form": "tsv",
    }
    print(f'  VizieR: {source} ...')
    r = requests.get(url, params=params, timeout=180)
    if outfile:
        Path(outfile).write_text(r.text, encoding='utf-8')
    return r.text


def parse_tsv(text, required_cols=None):
    lines = [l for l in text.strip().split('\n') if l and not l.startswith('#')]
    header_idx = None
    for i, l in enumerate(lines):
        if required_cols and all(c in l for c in required_cols[:2]):
            header_idx = i
            break
        elif not l.startswith('-') and '\t' in l and i < 10:
            header_idx = i
            break
    if header_idx is None:
        return [], []
    headers = [h.strip() for h in lines[header_idx].split('\t')]
    data = []
    for l in lines[header_idx+1:]:
        if l.startswith('-') or not l.strip():
            continue
        cols = l.split('\t')
        if len(cols) >= len(headers):
            data.append(dict(zip(headers, [c.strip() for c in cols])))
    return headers, data


def step0_fetch_all():
    print('=' * 60)
    print('Step 0: Data Retrieval')
    print('=' * 60)

    results = {}

    # --- PROBES RC ---
    print('\n[PROBES RC]')
    txt = fetch_vizier_tsv(
        "J/ApJS/256/33/table2",
        "Name,Rad,Vrot,e_Vrot",
        outfile=OUTDIR / "probes_rc.tsv")
    _, rc_rows = parse_tsv(txt, ['Name', 'Rad'])

    rc_data = {}
    for row in rc_rows:
        try:
            name = row.get('Name', '').strip()
            rad = float(row.get('Rad', '0'))
            vrot = float(row.get('Vrot', '0'))
            ev = float(row.get('e_Vrot', '5')) if row.get('e_Vrot', '').strip() else 5.0
            if name:
                if name not in rc_data:
                    rc_data[name] = {'R': [], 'V': [], 'eV': []}
                rc_data[name]['R'].append(rad)
                rc_data[name]['V'].append(vrot)
                rc_data[name]['eV'].append(ev)
        except ValueError:
            pass

    for name in rc_data:
        for k in rc_data[name]:
            rc_data[name][k] = np.array(rc_data[name][k])
    print(f'  -> {len(rc_data)} galaxies')
    results['probes_rc'] = rc_data

    # --- PROBES master ---
    print('\n[PROBES Master]')
    txt = fetch_vizier_tsv(
        "J/ApJS/256/33/table1",
        "Name,Dist,incl,Vmax",
        outfile=OUTDIR / "probes_master.tsv")
    _, master_rows = parse_tsv(txt, ['Name', 'Dist'])

    dist_map = {}
    for row in master_rows:
        try:
            name = row.get('Name', '').strip()
            dist = float(row.get('Dist', '0'))
            if name and dist > 0:
                dist_map[name] = dist
        except ValueError:
            pass
    print(f'  -> {len(dist_map)} galaxies with distance')
    results['dist'] = dist_map

    # --- S4G ---
    print('\n[S4G Salo+2015]')
    txt = fetch_vizier_tsv(
        "J/ApJS/219/4/table1",
        "Name,Re,Tmag,n,T,PA,b/a",
        outfile=OUTDIR / "s4g_params.tsv")
    _, s4g_rows = parse_tsv(txt, ['Name', 'Re'])

    s4g = {}
    for row in s4g_rows:
        try:
            name = row.get('Name', '').strip()
            Re = float(row.get('Re', '0'))
            Tmag = float(row.get('Tmag', '99'))
            n = float(row.get('n', '1')) if row.get('n', '').strip() else 1.0
            if name and Re > 0 and Tmag < 30:
                s4g[name] = {'Re': Re, 'Tmag': Tmag, 'n': n}
        except ValueError:
            pass
    print(f'  -> {len(s4g)} galaxies')
    results['s4g'] = s4g

    # --- WHISP ---
    print('\n[WHISP HI]')
    whisp_sources = [
        ("J/A+A/390/829", "Name,Rad,Vrot,e_Vrot,Vgas"),
        ("J/AJ/141/193", "Name,Rad,Vobs,e_Vobs,Vgas,Vdisk"),
    ]

    whisp_rc = {}
    for src, cols in whisp_sources:
        try:
            txt = fetch_vizier_tsv(src, cols, outfile=OUTDIR / f"whisp_{src.replace('/','-')}.tsv")
            _, rows = parse_tsv(txt)
            if rows:
                print(f'  {src}: {len(rows)} rows')
                for row in rows:
                    name = (row.get('Name', '') or row.get('Galaxy', '')).strip()
                    try:
                        rad = float(row.get('Rad', row.get('R', '0')))
                        vgas_str = row.get('Vgas', row.get('VHI', ''))
                        if name and vgas_str.strip():
                            vgas = float(vgas_str)
                            if name not in whisp_rc:
                                whisp_rc[name] = {'R': [], 'Vgas': []}
                            whisp_rc[name]['R'].append(rad)
                            whisp_rc[name]['Vgas'].append(vgas)
                    except ValueError:
                        pass
        except Exception as e:
            print(f'  {src}: failed ({e})')

    for name in whisp_rc:
        for k in whisp_rc[name]:
            whisp_rc[name][k] = np.array(whisp_rc[name][k])
    print(f'  -> {len(whisp_rc)} galaxies with HI data')
    results['whisp'] = whisp_rc

    # --- THINGS ---
    print('\n[THINGS de Blok+2008]')
    txt = fetch_vizier_tsv(
        "J/AJ/136/2648/table5",
        "Name,Rad,Vobs,Vgas,Vdis,Vbul",
        outfile=OUTDIR / "things_rc.tsv")
    _, things_rows = parse_tsv(txt)

    things_rc = {}
    for row in things_rows:
        name = (row.get('Name', '') or row.get('Galaxy', '')).strip()
        try:
            rad = float(row.get('Rad', '0'))
            vgas_str = row.get('Vgas', '')
            vdisk_str = row.get('Vdis', row.get('Vdisk', ''))
            if name and vgas_str.strip():
                vgas = float(vgas_str)
                vdisk = float(vdisk_str) if vdisk_str.strip() else 0.0
                if name not in things_rc:
                    things_rc[name] = {'R': [], 'Vgas': [], 'Vdisk': []}
                things_rc[name]['R'].append(rad)
                things_rc[name]['Vgas'].append(vgas)
                things_rc[name]['Vdisk'].append(vdisk)
        except ValueError:
            pass

    for name in things_rc:
        for k in things_rc[name]:
            things_rc[name][k] = np.array(things_rc[name][k])
    print(f'  -> {len(things_rc)} galaxies with decomposed RC')
    results['things'] = things_rc

    return results


def compute_vdisk_freeman(R_kpc, h_kpc, I0_Lpc2, Yd):
    y = R_kpc / (2.0 * h_kpc)
    y = np.clip(y, 1e-6, 50)
    bessel = i0(y)*k0(y) - i1(y)*k1(y)

    Sigma0_SI = Yd * I0_Lpc2 * Msun / (pc**2)
    h_m = h_kpc * kpc
    V2 = 4 * np.pi * G_SI * Sigma0_SI * h_m * y**2 * bessel
    return np.sqrt(np.maximum(V2, 0)) / 1e3


def s4g_to_disk_params(s4g_entry, dist_Mpc):
    Re_arcsec = s4g_entry['Re']
    Tmag = s4g_entry['Tmag']
    n = s4g_entry['n']

    bn = 2*n - 1.0/3 + 4.0/(405*n) if n > 0.5 else 1.678
    h_arcsec = Re_arcsec / bn

    dist_kpc = dist_Mpc * 1e3
    h_kpc = h_arcsec * dist_kpc * np.pi / (180 * 3600)

    if h_kpc < 0.01 or h_kpc > 100:
        return None

    area = 2 * np.pi * Re_arcsec**2
    if area <= 0:
        return None
    mu0 = Tmag + 2.5 * np.log10(area) + 0.7
    I0_Lpc2 = 10**(-0.4 * (mu0 - M_sun_W1 - 21.572))

    return {'h_kpc': h_kpc, 'I0_Lpc2': I0_Lpc2}


def optimize_Yd(R_kpc, V_obs, eV, V_disk_unit, V_gas):
    def chi2(Yd):
        V_bar = np.sqrt(np.maximum(Yd * V_disk_unit**2 + V_gas**2, 0))
        resid = (V_obs - V_bar) / np.maximum(eV, 1.0)
        return np.sum(resid**2) / max(len(V_obs) - 1, 1)

    try:
        res = minimize_scalar(chi2, bounds=(0.05, 2.0), method='bounded')
        if res.fun < 50:
            return res.x, res.fun
    except:
        pass
    return None, None


def measure_gc_gs0(R_kpc, V_obs, V_bar, V_disk_unit, Yd_opt):
    n = len(R_kpc)
    if n < 5:
        return None

    outer = slice(2*n//3, n)
    R_m = R_kpc[outer] * kpc
    gc_vals = (V_obs[outer]*1e3)**2/R_m - (V_bar[outer]*1e3)**2/R_m
    gc = np.median(gc_vals)

    if gc <= 0:
        return None

    V_flat = np.median(V_obs[outer])

    vds = np.sqrt(Yd_opt) * np.abs(V_disk_unit)
    i_pk = np.argmax(vds)
    if i_pk == 0:
        i_pk = 1
    r_pk = R_kpc[i_pk]
    if r_pk >= R_kpc.max() * 0.9 or r_pk < 0.01:
        return None
    h_R = r_pk / 2.15

    GS0 = (V_flat * 1e3)**2 / (h_R * kpc)

    return {'gc': gc, 'GS0': GS0, 'V_flat': V_flat, 'h_R': h_R, 'Yd': Yd_opt}


def alpha_fit(gc_arr, gs0_arr):
    log_gc = np.log10(gc_arr)
    log_gs = np.log10(gs0_arr)
    x = log_gs - np.log10(a0)
    y = log_gc
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None
    sl, ic, r, p, se = linregress(x, y)
    t_stat = (sl - 0.5) / se
    p05 = 2 * tdist.sf(abs(t_stat), df=len(x)-2)
    return {'alpha': sl, 'e_alpha': se, 'p05': p05, 'r': r, 'N': len(x)}


def norm_name(n):
    import re
    return re.sub(r'[\s\-_]', '', n.upper())


SPARC_NAMES = {norm_name(n) for n in [
    "NGC7331","NGC2403","NGC3198","NGC2841","NGC6946","NGC3521",
    "NGC925","NGC2976","NGC4736","NGC5055","NGC7793","NGC3031",
    "NGC4826","NGC2903","NGC4559","NGC1003","NGC3109","NGC4395",
    "DDO154","DDO168","IC2574","NGC1560","UGC2259","DDO52","DDO87",
    "DDO101","DDO126","DDO133","DDO47","DDO50","DDO53","DDO46",
    "CVnIdwA","Haro29","Haro36","WLM","NGC6822","NGC3738","NGC4214",
    "NGC1569","NGC2366","SagDIG","DDO210","DDO216","UGC8508","UGCA281",
]}


def main():
    print('=' * 60)
    print('PROBES V_bar Pipeline v2')
    print('(Yd individual optimization + HI profiles)')
    print('=' * 60)

    data = step0_fetch_all()
    rc_data = data['probes_rc']
    dist_map = data['dist']
    s4g = data['s4g']
    whisp = data['whisp']
    things = data['things']

    print(f'\n{"="*60}')
    print('Step 1: Cross-match')
    print(f'{"="*60}')

    rc_norm = {norm_name(n): n for n in rc_data}
    s4g_norm = {norm_name(n): n for n in s4g}
    whisp_norm = {norm_name(n): n for n in whisp}
    things_norm = {norm_name(n): n for n in things}
    dist_norm = {norm_name(n): n for n in dist_map}

    hi_all = {}
    for nn, orig in whisp_norm.items():
        hi_all[nn] = {'source': 'WHISP', 'data': whisp[orig]}
    for nn, orig in things_norm.items():
        hi_all[nn] = {'source': 'THINGS', 'data': things[orig]}

    match_3 = set(rc_norm.keys()) & set(s4g_norm.keys()) & set(hi_all.keys()) & set(dist_norm.keys())
    match_2 = (set(rc_norm.keys()) & set(s4g_norm.keys()) & set(dist_norm.keys())) - match_3

    print(f'  PROBES: {len(rc_norm)}')
    print(f'  S4G: {len(s4g_norm)}')
    print(f'  HI (WHISP+THINGS): {len(hi_all)}')
    print(f'  PROBES x S4G x HI x Dist: {len(match_3)}')
    print(f'  PROBES x S4G x Dist (no HI): {len(match_2)}')

    match_3_indep = match_3 - SPARC_NAMES
    match_2_indep = match_2 - SPARC_NAMES
    print(f'  3-way independent of SPARC: {len(match_3_indep)}')
    print(f'  2-way independent of SPARC: {len(match_2_indep)}')

    print(f'\n{"="*60}')
    print('Step 2-6: V_bar construction + alpha test')
    print(f'{"="*60}')

    results_3way = []
    results_2way = []
    Yd_list = []

    for nn in sorted(match_3_indep | match_2_indep):
        rc_name = rc_norm[nn]
        s4g_name = s4g_norm[nn]
        dist_name = dist_norm[nn]
        dist_Mpc = dist_map[dist_name]

        rc = rc_data[rc_name]
        R_arcsec = rc['R']
        V_obs = rc['V']
        eV = rc['eV']

        dist_kpc = dist_Mpc * 1e3
        R_kpc = R_arcsec * dist_kpc * np.pi / (180 * 3600)

        if len(R_kpc) < 8:
            continue

        dp = s4g_to_disk_params(s4g[s4g_name], dist_Mpc)
        if dp is None:
            continue

        V_disk_unit = compute_vdisk_freeman(R_kpc, dp['h_kpc'], dp['I0_Lpc2'], Yd=1.0)

        if len(V_disk_unit) != len(V_obs):
            continue

        has_hi = nn in hi_all
        if has_hi:
            hi = hi_all[nn]['data']
            hi_R = hi.get('R', np.array([]))
            hi_Vgas = hi.get('Vgas', np.array([]))
            if len(hi_R) >= 3 and len(hi_Vgas) >= 3:
                from scipy.interpolate import interp1d
                try:
                    f_gas = interp1d(hi_R, hi_Vgas, kind='linear',
                                    fill_value='extrapolate', bounds_error=False)
                    V_gas = np.maximum(f_gas(R_kpc), 0)
                except:
                    V_gas = np.zeros_like(R_kpc)
                    has_hi = False
            else:
                V_gas = np.zeros_like(R_kpc)
                has_hi = False

        if not has_hi:
            V_flat_est = np.median(V_obs[2*len(V_obs)//3:])
            if V_flat_est < 80:
                f_gas = 0.30
            elif V_flat_est < 150:
                f_gas = 0.15
            else:
                f_gas = 0.05
            V_gas = np.sqrt(f_gas) * V_disk_unit

        Yd_opt, chi2_dof = optimize_Yd(R_kpc, V_obs, eV, V_disk_unit, V_gas)
        if Yd_opt is None:
            continue

        V_bar = np.sqrt(np.maximum(Yd_opt * V_disk_unit**2 + V_gas**2, 0))

        result = measure_gc_gs0(R_kpc, V_obs, V_bar, V_disk_unit, Yd_opt)
        if result is None:
            continue

        result['name'] = rc_name
        result['has_hi'] = has_hi
        result['chi2_dof'] = chi2_dof
        result['dist'] = dist_Mpc

        if has_hi:
            results_3way.append(result)
        else:
            results_2way.append(result)
        Yd_list.append(Yd_opt)

    print(f'\n  3-way results (real HI): {len(results_3way)}')
    print(f'  2-way results (est HI):  {len(results_2way)}')
    print(f'  Total: {len(results_3way) + len(results_2way)}')

    if Yd_list:
        Yd_arr = np.array(Yd_list)
        print(f'\n  Yd distribution:')
        print(f'    median={np.median(Yd_arr):.3f}, mean={np.mean(Yd_arr):.3f}, '
              f'std={np.std(Yd_arr):.3f}')
        print(f'    range=[{np.min(Yd_arr):.3f}, {np.max(Yd_arr):.3f}]')
        print(f'    IQR=[{np.percentile(Yd_arr,25):.3f}, {np.percentile(Yd_arr,75):.3f}]')

    print(f'\n{"="*60}')
    print('Step 7: Alpha Test')
    print(f'{"="*60}')

    for label, res_list in [
        ('3-way (real HI)', results_3way),
        ('2-way (est HI)', results_2way),
        ('All combined', results_3way + results_2way),
    ]:
        if len(res_list) < 5:
            print(f'  {label}: N={len(res_list)} < 5, skip')
            continue
        gc_arr = np.array([r['gc'] for r in res_list])
        gs0_arr = np.array([r['GS0'] for r in res_list])
        fit = alpha_fit(gc_arr, gs0_arr)
        if fit:
            ok = 'YES' if fit['p05'] > 0.05 else 'no'
            print(f'  {label}: N={fit["N"]}, alpha={fit["alpha"]:.3f}+/-{fit["e_alpha"]:.3f}, '
                  f'p(0.5)={fit["p05"]:.4f}, r={fit["r"]:.3f}, 0.5棄却不可={ok}')

    summary = {
        'n_3way': len(results_3way),
        'n_2way': len(results_2way),
        'Yd_stats': {
            'median': float(np.median(Yd_arr)) if Yd_list else None,
            'mean': float(np.mean(Yd_arr)) if Yd_list else None,
        },
        'results_3way': [{'name': r['name'], 'gc': float(r['gc']),
                          'GS0': float(r['GS0']), 'Yd': float(r['Yd'])}
                         for r in results_3way],
        'results_2way_count': len(results_2way),
    }
    with open(OUTDIR / 'vbar_v2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults: {OUTDIR / "vbar_v2_summary.json"}')


if __name__ == '__main__':
    main()
