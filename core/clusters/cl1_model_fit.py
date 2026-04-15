#!/usr/bin/env python3
"""
cl1_model_fit.py
================
cl1の弱レンズシェアプロファイルに3モデルをフィットしてAIC比較。
"""

import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
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

Z_CL = 0.313

def E(z):
    return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def D_A(z):
    f = lambda zp: 1.0 / E(zp)
    chi, _ = quad(f, 0, z)
    return c_light / H0 * chi / (1+z)

RHO_CRIT_0 = 1.36e11  # Msun/Mpc^3 (H0=70)


def nfw_Sigma(R, M200, c, z):
    rho_c = RHO_CRIT_0 * E(z)**2
    r200 = (3*M200 / (4*np.pi*200*rho_c))**(1./3)
    rs = r200 / c
    rho_s = M200 / (4*np.pi*rs**3 * (np.log(1+c) - c/(1+c)))

    x = np.atleast_1d(R / rs).astype(float)
    result = np.zeros_like(x)

    lt1 = (x < 1) & (x > 0)
    gt1 = x > 1
    eq1 = np.abs(x - 1) < 1e-4

    if np.any(lt1):
        t = np.sqrt(1 - x[lt1]**2)
        result[lt1] = 1/(x[lt1]**2 - 1) * (1 - np.arctanh(t)/t)
    if np.any(gt1):
        t = np.sqrt(x[gt1]**2 - 1)
        result[gt1] = 1/(x[gt1]**2 - 1) * (1 - np.arctan(t)/t)
    if np.any(eq1):
        result[eq1] = 1./3

    return 2 * rs * rho_s * result


def nfw_DeltaSigma(R, M200, c, z):
    R = np.atleast_1d(R)
    Sigma_R = nfw_Sigma(R, M200, c, z)

    Sigma_mean = np.zeros_like(R)
    for i, Ri in enumerate(R):
        r_int = np.linspace(0.001*Ri, Ri, 500)
        S_int = nfw_Sigma(r_int, M200, c, z)
        Sigma_mean[i] = 2 * np.trapezoid(r_int * S_int, r_int) / Ri**2

    return Sigma_mean - Sigma_R


def two_halo_DeltaSigma(R, M200, b, z):
    rho_m = RHO_CRIT_0 * Omega_m * (1+z)**3
    r0 = 5.0 / 0.7
    gamma = 1.8

    from scipy.special import gamma as Gamma
    prefactor = rho_m * r0 * np.sqrt(np.pi) * Gamma((gamma-1)/2) / Gamma(gamma/2)
    Sigma_2h = b * prefactor * (R/r0)**(1-gamma)

    DS_2h = (gamma - 1) / (3 - gamma) * Sigma_2h

    return DS_2h


def membrane_DeltaSigma(R, M200, f_delta, r_delta, z):
    c = 5.0
    DS_nfw = nfw_DeltaSigma(R, M200 * (1 - f_delta), c, z)

    sigma_ring = 0.05
    M_ring = f_delta * M200

    Sigma_ring = M_ring / (2*np.pi*r_delta*sigma_ring*np.sqrt(2*np.pi)) * \
                 np.exp(-0.5*((R - r_delta)/sigma_ring)**2)

    Sigma_ring_mean = np.zeros_like(R)
    for i, Ri in enumerate(R):
        r_int = np.linspace(0.001, Ri, 300)
        S_int = M_ring / (2*np.pi*r_delta*sigma_ring*np.sqrt(2*np.pi)) * \
                np.exp(-0.5*((r_int - r_delta)/sigma_ring)**2)
        Sigma_ring_mean[i] = 2*np.trapezoid(r_int*S_int, r_int) / Ri**2

    DS_ring = Sigma_ring_mean - Sigma_ring

    return DS_nfw + DS_ring


def load_profile():
    for d in [Path('.'), Path('..'), OUTDIR, Path('cluster_stack_output')]:
        if not d.exists():
            continue
        for f in d.glob('*individual*shear*.csv'):
            print(f'Found binned profile: {f}')
            return load_binned_csv(f)

    print('Using hardcoded cl1 profile from v2 analysis')
    return hardcoded_cl1_profile()


def load_binned_csv(filepath):
    import csv
    print(f'Loading: {filepath}')

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        line = f.readline().strip()
        if line.startswith('#'):
            line = line[1:].strip()
        headers = [h.strip() for h in line.split(',')]
        print(f'  Columns: {headers}')

        r_col = None
        r_unit = 'unknown'
        for i, h in enumerate(headers):
            hl = h.lower()
            if 'kpc' in hl:
                r_col = i; r_unit = 'kpc'; break
            elif 'mpc' in hl:
                r_col = i; r_unit = 'Mpc'; break
            elif 'arcmin' in hl:
                r_col = i; r_unit = 'arcmin'; break
        if r_col is None:
            r_col = 0

        gt_col = next((i for i, h in enumerate(headers)
                       if 'gamma_t' in h.lower() or h.lower() == 'gt'), 1)
        eg_col = next((i for i, h in enumerate(headers)
                       if 'err' in h.lower() or 'e_g' in h.lower()), None)

        R, gt, eg = [], [], []
        for row_line in f:
            cols = row_line.strip().split(',')
            try:
                r_val = float(cols[r_col])
                if r_unit == 'kpc':
                    r_Mpc = r_val / 1000
                elif r_unit == 'arcmin':
                    r_Mpc = r_val / 60 * np.pi / 180 * D_A(Z_CL)
                else:
                    r_Mpc = r_val

                gt_val = float(cols[gt_col])
                eg_val = float(cols[eg_col]) if eg_col is not None else 0.01

                R.append(r_Mpc)
                gt.append(gt_val)
                eg.append(eg_val)
            except (ValueError, IndexError):
                pass

    print(f'  {len(R)} bins loaded')
    return np.array(R), np.array(gt), np.array(eg)


def hardcoded_cl1_profile():
    R_Mpc = np.array([0.07, 0.14, 0.24, 0.39, 0.59, 0.84, 1.22, 1.73, 2.45])
    gamma_t = np.array([0.035, 0.025, 0.018, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001])
    e_gamma = np.array([0.012, 0.006, 0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001])
    return R_Mpc, gamma_t, e_gamma


def fit_nfw(R, gt, eg):
    def model(params):
        logM, c, logA = params
        M200 = 10**logM
        A = 10**logA
        DS = nfw_DeltaSigma(R, M200, c, Z_CL)
        pred = A * DS
        return np.sum(((gt - pred) / eg)**2)

    try:
        result = differential_evolution(model,
            bounds=[(13, 16), (1, 15), (-18, -12)],
            seed=42, maxiter=500, tol=1e-6)
        logM, c, logA = result.x
        chi2 = result.fun
        dof = len(R) - 3

        DS_best = nfw_DeltaSigma(R, 10**logM, c, Z_CL)
        pred_best = 10**logA * DS_best

        return {
            'name': 'NFW',
            'logM200': logM, 'c200': c, 'logA': logA,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2 + 2*3,
            'pred': pred_best,
            'n_param': 3,
        }
    except Exception as e:
        print(f'  NFW fit failed: {e}')
        return None


def fit_nfw_2halo(R, gt, eg):
    def model(params):
        logM, c, logA, b = params
        M200 = 10**logM
        A = 10**logA
        DS_1h = nfw_DeltaSigma(R, M200, c, Z_CL)
        DS_2h = two_halo_DeltaSigma(R, M200, b, Z_CL)
        pred = A * (DS_1h + DS_2h)
        return np.sum(((gt - pred) / eg)**2)

    try:
        result = differential_evolution(model,
            bounds=[(13, 16), (1, 15), (-18, -12), (0.1, 10)],
            seed=42, maxiter=500, tol=1e-6)
        logM, c, logA, b = result.x
        chi2 = result.fun
        dof = len(R) - 4

        DS_1h = nfw_DeltaSigma(R, 10**logM, c, Z_CL)
        DS_2h = two_halo_DeltaSigma(R, 10**logM, b, Z_CL)
        pred = 10**logA * (DS_1h + DS_2h)

        return {
            'name': 'NFW+2halo',
            'logM200': logM, 'c200': c, 'logA': logA, 'b_lin': b,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2 + 2*4,
            'pred': pred,
            'n_param': 4,
        }
    except Exception as e:
        print(f'  NFW+2halo fit failed: {e}')
        return None


def fit_membrane(R, gt, eg):
    def model(params):
        logM, f_d, r_d, logA = params
        M200 = 10**logM
        A = 10**logA
        DS = membrane_DeltaSigma(R, M200, f_d, r_d, Z_CL)
        pred = A * DS
        return np.sum(((gt - pred) / eg)**2)

    try:
        result = differential_evolution(model,
            bounds=[(13, 16), (0.01, 0.5), (0.05, 1.0), (-18, -12)],
            seed=42, maxiter=500, tol=1e-6)
        logM, f_d, r_d, logA = result.x
        chi2 = result.fun
        dof = len(R) - 4

        DS = membrane_DeltaSigma(R, 10**logM, f_d, r_d, Z_CL)
        pred = 10**logA * DS

        return {
            'name': 'Membrane',
            'logM200': logM, 'f_delta': f_d, 'r_delta': r_d, 'logA': logA,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2 + 2*4,
            'pred': pred,
            'n_param': 4,
        }
    except Exception as e:
        print(f'  Membrane fit failed: {e}')
        return None


def main():
    print('=' * 60)
    print('cl1 Model Fitting: NFW vs NFW+2halo vs Membrane')
    print('=' * 60)

    R, gt, eg = load_profile()

    mask = (gt != 0) & (eg > 0) & (eg < 1) & (R > 0)
    R, gt, eg = R[mask], gt[mask], eg[mask]
    print(f'\nValid bins: {len(R)}')
    print(f'R range: [{R.min():.3f}, {R.max():.3f}] Mpc')
    print(f'gamma_t: {gt}')
    print(f'errors:  {eg}')

    print(f'\n--- Fitting ---')
    fits = {}

    print('\n  Model A: NFW')
    f_nfw = fit_nfw(R, gt, eg)
    if f_nfw:
        fits['NFW'] = f_nfw
        print(f'    logM={f_nfw["logM200"]:.2f}, c={f_nfw["c200"]:.1f}, '
              f'chi2/dof={f_nfw["chi2_dof"]:.2f}')

    print('\n  Model B: NFW + 2-halo')
    f_2h = fit_nfw_2halo(R, gt, eg)
    if f_2h:
        fits['NFW+2halo'] = f_2h
        print(f'    logM={f_2h["logM200"]:.2f}, c={f_2h["c200"]:.1f}, '
              f'b={f_2h["b_lin"]:.2f}, chi2/dof={f_2h["chi2_dof"]:.2f}')

    print('\n  Model C: Membrane delta-layer')
    f_mem = fit_membrane(R, gt, eg)
    if f_mem:
        fits['Membrane'] = f_mem
        print(f'    logM={f_mem["logM200"]:.2f}, f_d={f_mem["f_delta"]:.3f}, '
              f'r_d={f_mem["r_delta"]:.3f}, chi2/dof={f_mem["chi2_dof"]:.2f}')

    if fits:
        aic_min = min(f['AIC'] for f in fits.values())
        print(f'\n--- AIC Comparison ---')
        print(f'{"Model":<15} {"chi2":>7} {"dof":>4} {"chi2/dof":>9} {"AIC":>7} {"DAIC":>7} {"Best?":>6}')
        for name, f in sorted(fits.items(), key=lambda x: x[1]['AIC']):
            daic = f['AIC'] - aic_min
            best = '<--' if daic == 0 else ''
            print(f'{name:<15} {f["chi2"]:7.2f} {f["dof"]:4d} {f["chi2_dof"]:9.2f} '
                  f'{f["AIC"]:7.2f} {daic:7.2f} {best:>6}')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(R, gt, yerr=eg, fmt='ko', capsize=3, markersize=6, label='cl1 data')

    colors_m = {'NFW': '#1a1a2e', 'NFW+2halo': '#e94560', 'Membrane': '#2ecc71'}

    for name, f in fits.items():
        ax.plot(R, f['pred'], '-', color=colors_m.get(name, 'grey'), lw=2,
                label=f'{name} (chi2/dof={f["chi2_dof"]:.2f}, DAIC={f["AIC"]-aic_min:.1f})')

    ax.set_xlabel('R [Mpc]', fontsize=12)
    ax.set_ylabel('gamma_t', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('cl1 Weak Lensing: Model Comparison', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='grey', ls='--', alpha=0.3)

    plt.tight_layout()
    fig_path = OUTDIR / 'cl1_model_fit.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\nFigure: {fig_path}')

    summary = {name: {k: v for k, v in f.items() if k != 'pred'}
               for name, f in fits.items()}
    with open(OUTDIR / 'cl1_model_fit.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "cl1_model_fit.json"}')


if __name__ == '__main__':
    main()
