#!/usr/bin/env python3
"""
cl1_modelB_fit.py
=================
光伝播モデルB（式10a-10d, 提案G）をcl1弱レンズに組み込む。
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

Z_CL = 0.313

def E(z): return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def D_A(z):
    f = lambda zp: 1.0/E(zp)
    chi, _ = quad(f, 0, z)
    return c_light/H0*chi/(1+z)

RS_CL1 = 0.36  # Mpc


def nfw_profiles(R, M200, c, z):
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

    DeltaSigma = Sigma_mean - Sigma
    return Sigma, DeltaSigma, rs, rho_s


def T_tanh(r, rs, w=3.0):
    return 0.5*(1 + np.tanh(w*(r - rs)/rs))


def fold_density(r, rs, k_cross):
    T = T_tanh(r, rs)
    if r < 0.001:
        return 0.0
    return k_cross * (T/r)**2


def P_rewire(r, rs, delta, k_cross):
    n = fold_density(r, rs, k_cross)
    return 1 - np.exp(-np.pi * (2*delta)**2 * n)


def q_factor(R, rs, delta, k_cross, rho_s, rs_nfw):
    def rho_nfw(r):
        x = r / rs_nfw
        if x < 1e-6: return rho_s
        return rho_s / (x * (1+x)**2)

    l_max = 10.0
    n_pts = 200
    l_arr = np.linspace(-l_max, l_max, n_pts)

    rho_arr = np.array([rho_nfw(np.sqrt(R**2 + l**2)) for l in l_arr])
    Prw_arr = np.array([P_rewire(np.sqrt(R**2 + l**2), rs, delta, k_cross) for l in l_arr])

    num = np.trapezoid(rho_arr * Prw_arr, l_arr)
    den = np.trapezoid(rho_arr, l_arr)

    if den < 1e-30:
        return 0.0
    return num / den


def kappa_ratio_modelB(R, rs_membrane, delta, k_cross, alpha_rw,
                        xi_eff, beta_AB, rho_s, rs_nfw):
    R = np.atleast_1d(R)
    ratio = np.ones_like(R)

    for i, Ri in enumerate(R):
        q = q_factor(Ri, rs_membrane, delta, k_cross, rho_s, rs_nfw)
        kf = np.exp(-delta / xi_eff)
        factor = (1 + alpha_rw * beta_AB * q) * (q + (1 - q) * kf)
        ratio[i] = factor

    return ratio


def membrane_modelB_DeltaSigma(R, M200, c_nfw, delta, k_cross, alpha_rw, z):
    Sigma, DS_nfw, rs_nfw, rho_s = nfw_profiles(R, M200, c_nfw, z)

    rs_membrane = RS_CL1
    xi_eff = rs_membrane * 0.5
    beta_AB = 4.0 / (3*np.pi)

    ratio = kappa_ratio_modelB(R, rs_membrane, delta, k_cross, alpha_rw,
                                xi_eff, beta_AB, rho_s, rs_nfw)

    DS_membrane = ratio * DS_nfw
    return DS_membrane


def load_profile():
    for d in [Path('.'), Path('..'), OUTDIR, Path('cluster_stack_output')]:
        if not d.exists(): continue
        for f in d.glob('*individual*shear*.csv'):
            return load_binned(f)

    print('Using hardcoded cl1 profile')
    R = np.array([0.07, 0.14, 0.24, 0.39, 0.59, 0.84, 1.22, 1.73, 2.45])
    gt = np.array([0.035, 0.025, 0.018, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001])
    eg = np.array([0.012, 0.006, 0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001])
    return R, gt, eg


def load_binned(filepath):
    print(f'Loading: {filepath}')
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first = f.readline().strip()
        if first.startswith('#'): first = first[1:].strip()
        headers = [h.strip() for h in first.split(',')]
        print(f'  Headers: {headers}')

        r_col = next((i for i, h in enumerate(headers) if 'kpc' in h.lower()), 0)
        r_unit = 'kpc' if 'kpc' in headers[r_col].lower() else 'Mpc'
        gt_col = next((i for i, h in enumerate(headers) if 'gamma_t' in h.lower() or h.lower()=='gt'), 1)
        eg_col = next((i for i, h in enumerate(headers) if 'err' in h.lower()), None)

        R, gt, eg = [], [], []
        for line in f:
            cols = line.strip().split(',')
            try:
                r_val = float(cols[r_col])
                if r_unit == 'kpc':
                    r_val /= 1000

                gt_val = float(cols[gt_col])
                eg_val = float(cols[eg_col]) if eg_col is not None else 0.01

                R.append(r_val); gt.append(gt_val); eg.append(eg_val)
            except (ValueError, IndexError):
                pass
    print(f'  {len(R)} bins')
    return np.array(R), np.array(gt), np.array(eg)


def fit_models(R, gt, eg):
    mask = (eg > 0) & (eg < 1) & np.isfinite(gt)
    R_f, gt_f, eg_f = R[mask], gt[mask], eg[mask]
    if len(R_f) < 4:
        print('Too few bins')
        return {}

    results = {}

    def chi2_nfw(p):
        logM, c, logA = p
        DS = nfw_profiles(R_f, 10**logM, c, Z_CL)[1]
        pred = 10**logA * DS
        return np.sum(((gt_f - pred)/eg_f)**2)

    try:
        res = differential_evolution(chi2_nfw, [(13,16),(1,15),(-18,-12)],
                                     seed=42, maxiter=1000)
        logM, c, logA = res.x
        DS = nfw_profiles(R_f, 10**logM, c, Z_CL)[1]
        pred = 10**logA * DS
        chi2 = res.fun; dof = len(R_f)-3
        results['NFW'] = {
            'logM': logM, 'c': c, 'logA': logA,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2+2*3, 'pred': pred, 'n_param': 3,
        }
        print(f'  NFW: logM={logM:.2f}, c={c:.1f}, chi2/dof={chi2/max(dof,1):.2f}')
    except Exception as e:
        print(f'  NFW failed: {e}')

    def chi2_memB_simple(p):
        logM, c, logA, delta = p
        DS_mem = membrane_modelB_DeltaSigma(R_f, 10**logM, c, delta,
                                             k_cross=0.1, alpha_rw=2.0, z=Z_CL)
        pred = 10**logA * DS_mem
        return np.sum(((gt_f - pred)/eg_f)**2)

    try:
        res = differential_evolution(chi2_memB_simple,
            [(13,16),(1,15),(-18,-12),(0.001,0.5)],
            seed=42, maxiter=1000, tol=1e-5)
        logM, c, logA, delta = res.x
        DS_mem = membrane_modelB_DeltaSigma(R_f, 10**logM, c, delta,
                                             k_cross=0.1, alpha_rw=2.0, z=Z_CL)
        pred = 10**logA * DS_mem
        chi2 = res.fun; dof = len(R_f)-4
        results['MembraneB_simple'] = {
            'logM': logM, 'c': c, 'logA': logA, 'delta': delta,
            'k_cross': 0.1, 'alpha_rw': 2.0,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2+2*4, 'pred': pred, 'n_param': 4,
        }
        print(f'  MemB_simple: logM={logM:.2f}, c={c:.1f}, delta={delta:.4f}, '
              f'chi2/dof={chi2/max(dof,1):.2f}')
    except Exception as e:
        print(f'  MemB_simple failed: {e}')

    def chi2_memB_full(p):
        logM, c, logA, delta, k_cross, alpha_rw = p
        DS_mem = membrane_modelB_DeltaSigma(R_f, 10**logM, c, delta,
                                             k_cross=k_cross, alpha_rw=alpha_rw, z=Z_CL)
        pred = 10**logA * DS_mem
        return np.sum(((gt_f - pred)/eg_f)**2)

    try:
        res = differential_evolution(chi2_memB_full,
            [(13,16),(1,15),(-18,-12),(0.001,0.5),(0.01,10),(0.1,5)],
            seed=42, maxiter=1500, tol=1e-5)
        logM, c, logA, delta, k_cross, alpha_rw = res.x
        DS_mem = membrane_modelB_DeltaSigma(R_f, 10**logM, c, delta,
                                             k_cross=k_cross, alpha_rw=alpha_rw, z=Z_CL)
        pred = 10**logA * DS_mem
        chi2 = res.fun; dof = len(R_f)-6
        results['MembraneB_full'] = {
            'logM': logM, 'c': c, 'logA': logA,
            'delta': delta, 'k_cross': k_cross, 'alpha_rw': alpha_rw,
            'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/max(dof,1),
            'AIC': chi2+2*6, 'pred': pred, 'n_param': 6,
        }
        print(f'  MemB_full: delta={delta:.4f}, k={k_cross:.3f}, alpha={alpha_rw:.2f}, '
              f'chi2/dof={chi2/max(dof,1):.2f}')
    except Exception as e:
        print(f'  MemB_full failed: {e}')

    if results:
        aic_min = min(r['AIC'] for r in results.values())
        for r in results.values():
            r['DAIC'] = r['AIC'] - aic_min

    return results


def main():
    print('=' * 60)
    print('cl1 Weak Lensing: Model B (Eqs. 10a-10d) Fit')
    print('=' * 60)

    R, gt, eg = load_profile()
    mask = (eg > 0) & (eg < 1) & np.isfinite(gt) & (R > 0)
    R, gt, eg = R[mask], gt[mask], eg[mask]
    print(f'Valid bins: {len(R)}, R range: [{R.min():.3f}, {R.max():.3f}] Mpc')

    fits = fit_models(R, gt, eg)

    if fits:
        print(f'\n{"="*60}')
        print('AIC Comparison')
        print(f'{"="*60}')
        print(f'{"Model":<20} {"chi2":>7} {"dof":>4} {"chi2/dof":>9} '
              f'{"AIC":>7} {"DAIC":>7} {"n_p":>4}')
        for name, f in sorted(fits.items(), key=lambda x: x[1]['AIC']):
            print(f'{name:<20} {f["chi2"]:7.2f} {f["dof"]:4d} {f["chi2_dof"]:9.2f} '
                  f'{f["AIC"]:7.2f} {f["DAIC"]:7.2f} {f["n_param"]:4d}')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(R, gt, yerr=eg, fmt='ko', capsize=3, markersize=6, label='cl1 data')

    colors_m = {'NFW': '#1a1a2e', 'MembraneB_simple': '#e94560', 'MembraneB_full': '#2ecc71'}
    for name, f in fits.items():
        ax.plot(R, f['pred'], '-', color=colors_m.get(name, 'grey'), lw=2,
                label=f'{name} (DAIC={f["DAIC"]:.1f})')

    ax.set_xlabel('R [Mpc]', fontsize=12)
    ax.set_ylabel('gamma_t', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('cl1: NFW vs Membrane Model B (Eqs. 10a-10d)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='grey', ls='--', alpha=0.3)

    plt.tight_layout()
    fig_path = OUTDIR / 'cl1_modelB_fit.png'
    plt.savefig(fig_path, dpi=150)
    print(f'\nFigure: {fig_path}')

    summary = {name: {k: v for k, v in f.items() if k != 'pred'}
               for name, f in fits.items()}
    with open(OUTDIR / 'cl1_modelB_fit.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'Results: {OUTDIR / "cl1_modelB_fit.json"}')


if __name__ == '__main__':
    main()
