#!/usr/bin/env python3
"""
cl1_cl3_stack.py
================
cl1 + cl3 のみのスタック弱レンズ解析。
cl4/cl27 はSDSS分光メンバーなし → 除外。
"""

import numpy as np
import csv
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.integrate import quad

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

OUTDIR = Path("cluster_stack_v3_output")
OUTDIR.mkdir(exist_ok=True)

c_light = 2.998e5; H0 = 70.0; Omega_m = 0.3; Omega_L = 0.7
RHO_CRIT_0 = 1.36e11

CLUSTERS = {
    'cl1': {'ra': 140.45, 'dec': -0.25, 'z': 0.313},
    'cl3': {'ra': 139.80, 'dec': -0.10, 'z': 0.318},
}


def E(z):
    return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def D_A(z):
    f = lambda zp: 1.0 / E(zp)
    chi, _ = quad(f, 0, z)
    return c_light / H0 * chi / (1+z)

def Sigma_crit(z_l, z_s):
    D_l = D_A(z_l)
    D_s = D_A(z_s)
    f = lambda zp: 1.0 / E(zp)
    chi_ls, _ = quad(f, z_l, z_s)
    D_ls = c_light / H0 * chi_ls / (1+z_s)

    if D_ls <= 0 or D_l <= 0 or D_s <= 0:
        return None

    c2_over_4piG = (c_light * 1e3)**2 / (4 * np.pi * 6.674e-11) / (1.989e30) * (3.086e22)
    return c2_over_4piG * D_s / (D_l * D_ls)


def find_source_file(cl_name):
    patterns = [
        f'{cl_name}_sources_photoz.csv',
        f'{cl_name}_sources.csv',
        f'{cl_name}_background.csv',
        f'{cl_name}_photoz.csv',
    ]
    for d in [Path('.'), Path('..'), OUTDIR, Path('cluster_stack_output')]:
        if not d.exists():
            continue
        for p in patterns:
            f = d / p
            if f.exists():
                return f
        for f in d.glob(f'*{cl_name}*source*photoz*.csv'):
            return f
        for f in d.glob(f'*{cl_name}*source*.csv'):
            if 'individual' not in f.name.lower():
                return f
    return None


def load_sources(filepath, z_cl):
    print(f'  Loading: {filepath} ({filepath.stat().st_size/1e6:.1f} MB)')

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first = f.readline().strip()

    if first.startswith('#'):
        first = first[1:].strip()
    headers = [h.strip() for h in first.split(',')]

    def find_col(candidates):
        for c in candidates:
            for i, h in enumerate(headers):
                if h.lower() == c.lower():
                    return i
        for c in candidates:
            for i, h in enumerate(headers):
                if c.lower() in h.lower():
                    return i
        return None

    i_ra = find_col(['i_ra', 'ra'])
    i_dec = find_col(['i_dec', 'dec'])
    i_e1 = find_col(['i_hsmshaperegauss_e1', 'e1'])
    i_e2 = find_col(['i_hsmshaperegauss_e2', 'e2'])
    i_w = find_col(['i_hsmshaperegauss_derived_weight', 'derived_weight', 'weight'])
    i_zph = find_col(['photoz_best', 'photoz_mean', 'photo_z'])

    print(f'    Columns: ra={i_ra}, dec={i_dec}, e1={i_e1}, e2={i_e2}, w={i_w}, zph={i_zph}')

    if i_ra is None or i_dec is None or i_e1 is None or i_e2 is None:
        print(f'    ERROR: Missing columns. Headers: {headers[:15]}')
        return None

    sources = []
    n_total = 0
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        f.readline()
        for line in f:
            n_total += 1
            cols = line.strip().split(',')
            try:
                ra = float(cols[i_ra])
                dec = float(cols[i_dec])
                e1 = float(cols[i_e1])
                e2 = float(cols[i_e2])
                w = float(cols[i_w]) if i_w is not None else 1.0
                zph = float(cols[i_zph]) if i_zph is not None else 1.0

                if w > 0 and zph > z_cl + 0.1:
                    sources.append((ra, dec, e1, e2, w, zph))
            except (ValueError, IndexError):
                pass

    print(f'    Total: {n_total}, Background: {len(sources)}')
    return sources


def compute_DeltaSigma_profile(sources, ra_c, dec_c, z_cl, R_bins_Mpc):
    D_l = D_A(z_cl)
    n_bins = len(R_bins_Mpc) - 1
    R_mid = np.sqrt(R_bins_Mpc[:-1] * R_bins_Mpc[1:])

    DS_sum = np.zeros(n_bins)
    DX_sum = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    n_count = np.zeros(n_bins, dtype=int)
    Sc_sum = np.zeros(n_bins)

    for ra_g, dec_g, e1, e2, w, zph in sources:
        dra = (ra_g - ra_c) * np.cos(np.radians(dec_c))
        ddec = dec_g - dec_c
        R_Mpc = np.sqrt(dra**2 + ddec**2) * np.pi / 180 * D_l

        if R_Mpc < R_bins_Mpc[0] or R_Mpc >= R_bins_Mpc[-1]:
            continue

        Sc = Sigma_crit(z_cl, zph)
        if Sc is None or Sc <= 0:
            continue

        phi = np.arctan2(ddec, dra)
        gt = -(e1 * np.cos(2*phi) + e2 * np.sin(2*phi))
        gx = +(e2 * np.cos(2*phi) - e1 * np.sin(2*phi))

        DS = gt * Sc
        DX = gx * Sc

        w_ls = w / Sc**2

        idx = np.searchsorted(R_bins_Mpc, R_Mpc) - 1
        if 0 <= idx < n_bins:
            DS_sum[idx] += DS * w_ls
            DX_sum[idx] += DX * w_ls
            w_sum[idx] += w_ls
            Sc_sum[idx] += Sc * w
            n_count[idx] += 1

    mask = w_sum > 0
    DeltaSigma = np.zeros(n_bins)
    DeltaSigma_x = np.zeros(n_bins)
    DeltaSigma[mask] = DS_sum[mask] / w_sum[mask]
    DeltaSigma_x[mask] = DX_sum[mask] / w_sum[mask]

    sigma_e = 0.26
    e_DS = np.full(n_bins, np.inf)
    for i in range(n_bins):
        if n_count[i] > 0 and w_sum[i] > 0:
            # 適切な誤差: sigma_e / sqrt(sum w_ls)  (DSの単位で)
            e_DS[i] = sigma_e / np.sqrt(w_sum[i])

    return {
        'R': R_mid, 'DS': DeltaSigma, 'DS_x': DeltaSigma_x,
        'e_DS': e_DS, 'n': n_count, 'w': w_sum,
    }


def stack_profiles(profiles):
    R = profiles[0]['R']
    n_bins = len(R)

    DS_sum = np.zeros(n_bins)
    DX_sum = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    n_total = np.zeros(n_bins, dtype=int)

    for p in profiles:
        for i in range(n_bins):
            if p['w'][i] > 0:
                DS_sum[i] += p['DS'][i] * p['w'][i]
                DX_sum[i] += p['DS_x'][i] * p['w'][i]
                w_sum[i] += p['w'][i]
                n_total[i] += p['n'][i]

    mask = w_sum > 0
    DS_stack = np.zeros(n_bins)
    DX_stack = np.zeros(n_bins)
    DS_stack[mask] = DS_sum[mask] / w_sum[mask]
    DX_stack[mask] = DX_sum[mask] / w_sum[mask]

    sigma_e = 0.26
    e_DS = np.full(n_bins, np.inf)
    e_DS[mask] = sigma_e / np.sqrt(w_sum[mask])

    return {
        'R': R, 'DS': DS_stack, 'DS_x': DX_stack,
        'e_DS': e_DS, 'n': n_total, 'w': w_sum,
    }


def nfw_DeltaSigma(R, M200, c, z):
    rho_c = RHO_CRIT_0 * E(z)**2
    r200 = (3*M200/(4*np.pi*200*rho_c))**(1./3)
    rs = r200/c
    rho_s = M200/(4*np.pi*rs**3*(np.log(1+c)-c/(1+c)))

    def sigma_at(R_arr):
        x = np.atleast_1d(R_arr/rs).astype(float)
        Sig = np.zeros_like(x)
        lt = (x>0)&(x<1); gt = x>1; eq = np.abs(x-1)<1e-4
        if np.any(lt):
            t = np.sqrt(1-x[lt]**2)
            Sig[lt] = 1/(x[lt]**2-1)*(1-np.arctanh(t)/t)
        if np.any(gt):
            t = np.sqrt(x[gt]**2-1)
            Sig[gt] = 1/(x[gt]**2-1)*(1-np.arctan(t)/t)
        if np.any(eq):
            Sig[eq] = 1./3
        return 2*rs*rho_s*Sig

    Sigma = sigma_at(R)
    Sigma_mean = np.zeros_like(R)
    for i, Ri in enumerate(R):
        r_int = np.linspace(0.001*Ri, Ri, 300)
        S_int = sigma_at(r_int)
        Sigma_mean[i] = 2*np.trapezoid(r_int*S_int, r_int)/Ri**2

    return Sigma_mean - Sigma


def two_halo_DS(R, b, z):
    from scipy.special import gamma as Gamma
    rho_m = RHO_CRIT_0*Omega_m*(1+z)**3
    r0 = 7.0; gam = 1.8
    pf = rho_m*r0*np.sqrt(np.pi)*Gamma((gam-1)/2)/Gamma(gam/2)
    S2h = b*pf*(R/r0)**(1-gam)
    return (gam-1)/(3-gam)*S2h


def fit_all(R, DS, eDS, z_stack):
    mask = np.isfinite(DS) & np.isfinite(eDS) & (eDS > 0) & (eDS < 1e20)
    R_f, DS_f, eDS_f = R[mask], DS[mask], eDS[mask]

    if len(R_f) < 4:
        print('  Too few valid bins')
        return {}

    results = {}

    def chi2_nfw(p):
        logM, c = p
        pred = nfw_DeltaSigma(R_f, 10**logM, c, z_stack)
        return np.sum(((DS_f - pred)/eDS_f)**2)

    try:
        res = differential_evolution(chi2_nfw, [(12,16),(1,20)], seed=42, maxiter=1000)
        logM, c = res.x; chi2 = res.fun; dof = len(R_f)-2
        pred = nfw_DeltaSigma(R_f, 10**logM, c, z_stack)
        results['NFW'] = {
            'logM': logM, 'c': c,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2+2*2, 'pred': pred,
        }
        print(f'  NFW: logM={logM:.2f}, c={c:.1f}, chi2/dof={chi2/max(dof,1):.2f}')
    except Exception as e:
        print(f'  NFW failed: {e}')

    def chi2_2h(p):
        logM, c, b = p
        pred = nfw_DeltaSigma(R_f, 10**logM, c, z_stack) + two_halo_DS(R_f, b, z_stack)
        return np.sum(((DS_f - pred)/eDS_f)**2)

    try:
        res = differential_evolution(chi2_2h, [(12,16),(1,20),(0.1,10)], seed=42, maxiter=1000)
        logM, c, b = res.x; chi2 = res.fun; dof = len(R_f)-3
        pred = nfw_DeltaSigma(R_f, 10**logM, c, z_stack) + two_halo_DS(R_f, b, z_stack)
        results['NFW+2halo'] = {
            'logM': logM, 'c': c, 'b': b,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2+2*3, 'pred': pred,
        }
        print(f'  NFW+2h: logM={logM:.2f}, c={c:.1f}, b={b:.2f}, chi2/dof={chi2/max(dof,1):.2f}')
    except Exception as e:
        print(f'  NFW+2h failed: {e}')

    if results:
        aic_min = min(r['AIC'] for r in results.values())
        for r in results.values():
            r['DAIC'] = r['AIC'] - aic_min

    return results


def main():
    print('=' * 60)
    print('cl1 + cl3 Stack (cl4/cl27 excluded)')
    print('=' * 60)

    R_bins = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])

    profiles = {}
    for cl_name, cl in CLUSTERS.items():
        print(f'\n--- {cl_name} (z={cl["z"]}) ---')

        src_file = find_source_file(cl_name)
        if src_file is None:
            print(f'  Source file not found for {cl_name}')
            continue

        sources = load_sources(src_file, cl['z'])
        if sources is None or len(sources) < 100:
            print(f'  Insufficient sources')
            continue

        prof = compute_DeltaSigma_profile(sources, cl['ra'], cl['dec'], cl['z'], R_bins)
        profiles[cl_name] = prof

        total_sn2 = 0
        print(f'\n  {"R[Mpc]":>8} {"N":>7} {"DS":>14} {"DS_x":>14}')
        for i in range(len(prof['R'])):
            print(f'  {prof["R"][i]:8.3f} {prof["n"][i]:7d} '
                  f'{prof["DS"][i]:14.3e} {prof["DS_x"][i]:14.3e}')
            if prof['e_DS'][i] < 1e18 and prof['e_DS'][i] > 0:
                total_sn2 += (prof['DS'][i]/prof['e_DS'][i])**2
        print(f'  S/N = {np.sqrt(total_sn2):.1f}')

    if len(profiles) < 1:
        print('\nERROR: No profiles computed.')
        return

    if len(profiles) >= 2:
        print(f'\n{"="*60}')
        print(f'Stacking {len(profiles)} clusters')
        print(f'{"="*60}')
        stacked = stack_profiles(list(profiles.values()))
        z_stack = np.mean([cl['z'] for cl in CLUSTERS.values()])
    else:
        cl_name = list(profiles.keys())[0]
        print(f'\n  Only {cl_name} available')
        stacked = profiles[cl_name]
        z_stack = CLUSTERS[cl_name]['z']

    print(f'\n  Stacked profile:')
    print(f'  {"R[Mpc]":>8} {"N_total":>8} {"DS":>14} {"DS_x":>14} {"e_DS":>14}')
    for i in range(len(stacked['R'])):
        print(f'  {stacked["R"][i]:8.3f} {stacked["n"][i]:8d} '
              f'{stacked["DS"][i]:14.3e} {stacked["DS_x"][i]:14.3e} '
              f'{stacked["e_DS"][i]:14.3e}')

    sn_stack = np.sqrt(np.sum(np.where(stacked['e_DS'] < 1e18,
                                        (stacked['DS']/stacked['e_DS'])**2, 0)))
    print(f'  Stack S/N = {sn_stack:.1f}')

    print(f'\n{"="*60}')
    print('Model Fitting')
    print(f'{"="*60}')
    fits = fit_all(stacked['R'], stacked['DS'], stacked['e_DS'], z_stack)

    if fits:
        print(f'\n--- AIC Comparison ---')
        print(f'{"Model":<15} {"chi2":>8} {"dof":>4} {"chi2/dof":>9} {"AIC":>8} {"DAIC":>7}')
        for name, f in sorted(fits.items(), key=lambda x: x[1]['AIC']):
            print(f'{name:<15} {f["chi2"]:8.2f} {f["dof"]:4d} {f["chi2_dof"]:9.2f} '
                  f'{f["AIC"]:8.2f} {f["DAIC"]:7.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    mask = stacked['n'] > 0
    ax.errorbar(stacked['R'][mask], stacked['DS'][mask],
                yerr=stacked['e_DS'][mask], fmt='ko', capsize=3, label='cl1+cl3 stack')

    colors_m = {'NFW': '#1a1a2e', 'NFW+2halo': '#e94560'}
    for name, f in fits.items():
        ax.plot(stacked['R'][mask], f['pred'], '-', color=colors_m.get(name, 'grey'),
                lw=2, label=f'{name} (DAIC={f["DAIC"]:.1f})')

    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('Delta-Sigma [Msun/Mpc^2]')
    ax.set_xscale('log')
    ax.set_title('(a) cl1+cl3 Stack: Delta-Sigma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(stacked['R'][mask], stacked['DS_x'][mask],
                yerr=stacked['e_DS'][mask], fmt='ko', capsize=3)
    ax.axhline(0, color='#e94560', ls='--', lw=2)
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('Delta-Sigma_x [Msun/Mpc^2]')
    ax.set_xscale('log')
    ax.set_title('(b) Cross component (null test)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'cl1+cl3 Stack Weak Lensing (S/N={sn_stack:.1f})', fontsize=13)
    plt.tight_layout()
    fig_path = OUTDIR / 'cl1_cl3_stack.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\nFigure: {fig_path}')

    summary = {
        'clusters_used': list(CLUSTERS.keys()),
        'clusters_excluded': ['cl4', 'cl27'],
        'exclusion_reason': 'No SDSS spectroscopic members found',
        'stack_SN': float(sn_stack),
        'fits': {name: {k: v for k, v in f.items() if k != 'pred'}
                 for name, f in fits.items()},
    }
    with open(OUTDIR / 'cl1_cl3_stack.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "cl1_cl3_stack.json"}')


if __name__ == '__main__':
    main()
