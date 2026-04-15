# -*- coding: utf-8 -*-
"""
brouwer_kids_sensitivity_chi2.py

Sensitivity test for KiDS hot-gas correction per M* bin.
"""

import os, math, sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

a0 = 1.2e-10
G_SI = 6.674e-11
Msun = 1.989e30
kpc_m = 3.0857e19
ESD_TO_GOBS = 5.580e-13

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

MSTAR_BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
MSTAR_BIN_LOGMIN = [8.5, 10.3, 10.6, 10.8]
MSTAR_BIN_LOG_CENTERS = [
    0.5 * (MSTAR_BIN_EDGES[i] + MSTAR_BIN_EDGES[i + 1]) for i in range(4)
]
N_BINS = 4
N_GBAR = 15
N_TOT = N_BINS * N_GBAR

F_GAS_GRID = [0.3, 0.5, 1.0, 2.0, 3.0]

LOG_MSTAR_REF = 11.0
GAMMA_RACC = 0.4


def load_esd(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'g_bar': data[:, 0],
        'ESD_t': data[:, 1],
        'error': data[:, 3],
        'bias':  data[:, 4],
    }


def load_covmatrix(filepath):
    data = np.loadtxt(filepath, comments='#')
    pairs_m = list(zip(data[:, 0], data[:, 2]))
    pairs_n = list(zip(data[:, 1], data[:, 3]))
    all_pairs = sorted(set(pairs_m))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    n = len(all_pairs)
    assert n == N_TOT, f"Expected {N_TOT} pairs, got {n}"
    cov_raw = np.zeros((n, n))
    bias_cov = np.zeros((n, n))
    for row in data:
        lm_m, lm_n, gb_i, gb_j = row[0], row[1], row[2], row[3]
        i = pair_to_idx[(lm_m, gb_i)]
        j = pair_to_idx[(lm_n, gb_j)]
        cov_raw[i, j] = row[4]
        bias_cov[i, j] = row[6]
    cov_corrected = cov_raw / bias_cov
    return cov_corrected, all_pairs


def membrane_rar(g_bar, gc):
    return (g_bar + np.sqrt(g_bar**2 + 4 * gc * g_bar)) / 2


def mond_rar(g_bar):
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    return g_bar / (1.0 - np.exp(-x))


def gobs_to_esd(g_obs, bias):
    return g_obs * bias / ESD_TO_GOBS


def esd_to_gobs(ESD_t, bias):
    return ESD_TO_GOBS * ESD_t / bias


def fit_gc(g_bar, g_obs, g_err):
    def chi2(log_gc):
        gc = 10**log_gc
        gp = membrane_rar(g_bar, gc)
        res = np.log10(np.maximum(g_obs, 1e-30)) - np.log10(np.maximum(gp, 1e-30))
        w = 1.0 / np.maximum((g_err / (g_obs * np.log(10)))**2, 1e-10)
        return np.sum(w * res**2)
    result = minimize_scalar(chi2, bounds=(-14, -8), method='bounded')
    return 10**result.x


def estimate_vflat(log_mstar):
    Mbar = 10**log_mstar * 1.4
    return (G_SI * a0 * Mbar * Msun)**0.25 / 1e3


def estimate_hR(log_mstar):
    return 10**(0.35 * (log_mstar - 10.0) + 0.5)


def c15_predict_gc(log_mstar, Yd=0.5):
    vflat = estimate_vflat(log_mstar)
    hR = estimate_hR(log_mstar)
    return 0.584 * Yd**(-0.361) * math.sqrt(a0 * (vflat * 1e3)**2 / (hR * kpc_m))


def compute_correction_factor_scaled(C_nominal, f_gas):
    C_f = 1.0 - f_gas * (1.0 - C_nominal)
    return np.maximum(C_f, 0.01)


def mstar_dependent_feff(f_gas, log_mstar):
    ratio = 10**(GAMMA_RACC * (log_mstar - LOG_MSTAR_REF))
    return f_gas * min(1.0, ratio)


def compute_chi2_full_cov(data_esd_vec, model_esd_vec, cov_inv):
    residual = data_esd_vec - model_esd_vec
    return float(residual @ cov_inv @ residual)


def main():
    print("=" * 78)
    print("  SENSITIVITY TEST: f_gas + M*-dependent R_acc + formal chi2")
    print("=" * 78)

    f_nohg = BASE / "Fig-4-5-C1_KiDS-isolated_Nobins.txt"
    f_hg = BASE / "Fig-4_KiDS-isolated_hotgas_Nobins.txt"

    if not f_nohg.exists() or not f_hg.exists():
        print(f"ERROR: Fig-4 files not found in {BASE}")
        sys.exit(1)

    d_nohg_all = load_esd(f_nohg)
    d_hg_all = load_esd(f_hg)

    assert np.allclose(d_nohg_all['g_bar'], d_hg_all['g_bar']), "g_bar mismatch"
    g_bar_grid = d_nohg_all['g_bar']
    C_nominal = d_hg_all['ESD_t'] / d_nohg_all['ESD_t']

    print(f"\nNominal C(g_bar): min={C_nominal.min():.3f}, max={C_nominal.max():.3f}")
    print(f"g_bar grid: {len(g_bar_grid)} points, "
          f"[{g_bar_grid[0]/a0:.2e}, {g_bar_grid[-1]/a0:.2e}] a0")

    bin_data = []
    for i in range(N_BINS):
        fname = BASE / f"Fig-9_KiDS-isolated_Massbin-{i+1}.txt"
        if not fname.exists():
            print(f"ERROR: {fname} not found")
            sys.exit(1)
        d = load_esd(fname)
        if not np.allclose(d['g_bar'], g_bar_grid):
            from scipy.interpolate import interp1d
            log_C_interp = interp1d(
                np.log10(g_bar_grid), np.log10(C_nominal),
                bounds_error=False, fill_value='extrapolate')
            C_at_bin = 10**log_C_interp(np.log10(d['g_bar']))
            print(f"  Bin {i+1}: g_bar grid differs, interpolating C(g_bar)")
        else:
            C_at_bin = C_nominal.copy()
        d['C_nominal'] = C_at_bin
        bin_data.append(d)

    cov_file = BASE / "Fig-9_RAR-KiDS-isolated_Massbins_covmatrix.txt"
    if not cov_file.exists():
        print(f"ERROR: {cov_file} not found")
        sys.exit(1)

    cov_corrected, pair_map = load_covmatrix(cov_file)
    print(f"\nCovariance matrix: {cov_corrected.shape}")
    print(f"  Condition number: {np.linalg.cond(cov_corrected):.2e}")
    cond = np.linalg.cond(cov_corrected)
    if cond > 1e12:
        print(f"  WARNING: ill-conditioned ({cond:.1e}), regularizing 1%")
        diag = np.diag(cov_corrected)
        cov_corrected += 0.01 * np.diag(diag)

    pair_to_idx = {p: i for i, p in enumerate(pair_map)}

    def lookup_idx(logM_min, gb):
        target = (logM_min, gb)
        if target in pair_to_idx:
            return pair_to_idx[target]
        dists = [(abs(p[0]-logM_min) + abs(np.log10(p[1])-np.log10(gb)), i)
                 for i, p in enumerate(pair_map)]
        return min(dists, key=lambda x: x[0])[1]

    esd_raw_vec = np.zeros(N_TOT)
    bias_vec = np.ones(N_TOT)
    gbar_vec = np.zeros(N_TOT)
    bin_idx_vec = np.zeros(N_TOT, dtype=int)
    C_nom_vec = np.zeros(N_TOT)
    for k in range(N_BINS):
        for j in range(N_GBAR):
            gb = bin_data[k]['g_bar'][j]
            idx = lookup_idx(MSTAR_BIN_LOGMIN[k], gb)
            esd_raw_vec[idx] = bin_data[k]['ESD_t'][j]
            bias_vec[idx] = bin_data[k]['bias'][j]
            gbar_vec[idx] = gb
            bin_idx_vec[idx] = k
            C_nom_vec[idx] = bin_data[k]['C_nominal'][j]

    gc_c15 = [c15_predict_gc(MSTAR_BIN_LOG_CENTERS[k]) for k in range(N_BINS)]
    print("\nC15 predictions:")
    for k in range(N_BINS):
        print(f"  Bin {k+1} (logM*={MSTAR_BIN_LOG_CENTERS[k]:.2f}): "
              f"gc_C15 = {gc_c15[k]/a0:.3f} a0")

    def run_scenario(C_f_vec, label):
        esd_corr_vec = esd_raw_vec * C_f_vec
        C_outer = np.outer(C_f_vec, C_f_vec)
        cov_corr = cov_corrected * C_outer
        if np.linalg.cond(cov_corr) > 1e12:
            cov_corr += 0.01 * np.diag(np.diag(cov_corr))
        cov_inv_corr = np.linalg.inv(cov_corr)

        gc_fit = []
        for k in range(N_BINS):
            mask = (bin_idx_vec == k)
            gb_k = gbar_vec[mask]
            esd_k = esd_corr_vec[mask]
            bias_k = bias_vec[mask]
            go_k = esd_to_gobs(esd_k, bias_k)
            ge_k = esd_to_gobs(
                np.sqrt(np.maximum(np.diag(cov_corr)[mask], 0)), bias_k)
            valid = (gb_k > 0) & (go_k > 0) & np.isfinite(go_k) & (ge_k > 0)
            if valid.sum() < 3:
                gc_fit.append(np.nan)
                continue
            gc_fit.append(fit_gc(gb_k[valid], go_k[valid], ge_k[valid]))

        esd_mond_vec = np.zeros(N_TOT)
        esd_c15_vec = np.zeros(N_TOT)
        esd_memb_fit_vec = np.zeros(N_TOT)
        for k in range(N_BINS):
            mask = (bin_idx_vec == k)
            gb_k = gbar_vec[mask]
            bias_k = bias_vec[mask]
            esd_mond_vec[mask] = gobs_to_esd(mond_rar(gb_k), bias_k)
            esd_c15_vec[mask] = gobs_to_esd(membrane_rar(gb_k, gc_c15[k]), bias_k)
            if not np.isnan(gc_fit[k]):
                esd_memb_fit_vec[mask] = gobs_to_esd(
                    membrane_rar(gb_k, gc_fit[k]), bias_k)

        chi2_mond = compute_chi2_full_cov(esd_corr_vec, esd_mond_vec, cov_inv_corr)
        chi2_c15 = compute_chi2_full_cov(esd_corr_vec, esd_c15_vec, cov_inv_corr)
        chi2_fit = compute_chi2_full_cov(esd_corr_vec, esd_memb_fit_vec, cov_inv_corr)

        return {
            'label': label,
            'gc_fit': [g/a0 if not np.isnan(g) else np.nan for g in gc_fit],
            'chi2_mond': chi2_mond,
            'chi2_c15': chi2_c15,
            'chi2_fit': chi2_fit,
            'dchi2_mond_c15': chi2_mond - chi2_c15,
            'dchi2_mond_fit': chi2_mond - chi2_fit,
        }

    print("\n" + "=" * 78)
    print("  PART 1: UNIFORM f_gas SCALING (M*-independent correction)")
    print("=" * 78)

    results_uniform = []
    for f_gas in F_GAS_GRID:
        C_f_vec = compute_correction_factor_scaled(C_nom_vec, f_gas)
        r = run_scenario(C_f_vec, f"uniform f={f_gas}")
        r['f_gas'] = f_gas
        results_uniform.append(r)

        print(f"\n--- f_gas = {f_gas:.1f} ---")
        print(f"  {'Bin':>4s} {'logM*':>6s} {'gc_fit/a0':>10s} "
              f"{'gc_C15/a0':>10s} {'ratio':>8s}")
        for k in range(N_BINS):
            gc_f = r['gc_fit'][k]
            gc_c = gc_c15[k]/a0
            ratio = gc_f / gc_c if not np.isnan(gc_f) else float('nan')
            print(f"    {k+1:>2d}  {MSTAR_BIN_LOG_CENTERS[k]:6.2f}  "
                  f"{gc_f:10.3f}  {gc_c:10.3f}  {ratio:8.2f}")
        print(f"  chi2(MOND)     = {r['chi2_mond']:8.1f}  (dof=60)")
        print(f"  chi2(C15 pred) = {r['chi2_c15']:8.1f}")
        print(f"  chi2(memb fit) = {r['chi2_fit']:8.1f}")
        print(f"  Delta-chi2 (MOND - C15 pred) = {r['dchi2_mond_c15']:+.1f}")
        print(f"  Delta-chi2 (MOND - memb fit) = {r['dchi2_mond_fit']:+.1f}")

    print("\n--- gc-M* slopes (uniform f_gas) ---")
    print(f"  {'f_gas':>6s} {'slope_fit':>10s} {'slope_C15':>10s} {'MOND':>6s}")
    for r in results_uniform:
        logM = np.array(MSTAR_BIN_LOG_CENTERS)
        log_gc = np.log10([max(g, 1e-3) for g in r['gc_fit']])
        log_gc_c15 = np.log10([gc_c15[k]/a0 for k in range(N_BINS)])
        valid = np.isfinite(log_gc)
        if valid.sum() >= 3:
            s_fit = np.polyfit(logM[valid], log_gc[valid], 1)[0]
        else:
            s_fit = float('nan')
        s_c15 = np.polyfit(logM, log_gc_c15, 1)[0]
        print(f"  {r['f_gas']:6.1f}  {s_fit:+10.3f}  {s_c15:+10.3f}  {0.0:+6.3f}")

    print("\n" + "=" * 78)
    print(f"  PART 2: M*-DEPENDENT f_gas (R_acc ~ M*^{GAMMA_RACC})")
    print("=" * 78)

    results_mstar_dep = []
    for f_gas_base in F_GAS_GRID:
        f_eff = [mstar_dependent_feff(f_gas_base, MSTAR_BIN_LOG_CENTERS[k])
                 for k in range(N_BINS)]
        C_f_vec = np.zeros(N_TOT)
        for k in range(N_BINS):
            mask = (bin_idx_vec == k)
            C_f_vec[mask] = compute_correction_factor_scaled(
                C_nom_vec[mask], f_eff[k])
        r = run_scenario(C_f_vec, f"M*-dep base={f_gas_base}")
        r['f_gas_base'] = f_gas_base
        r['f_eff'] = f_eff
        results_mstar_dep.append(r)

        print(f"\n--- f_gas_base = {f_gas_base:.1f} ---")
        print(f"  {'Bin':>4s} {'logM*':>6s} {'f_eff':>6s} {'gc_fit/a0':>10s} "
              f"{'gc_C15/a0':>10s} {'ratio':>8s}")
        for k in range(N_BINS):
            gc_f = r['gc_fit'][k]
            gc_c = gc_c15[k]/a0
            ratio = gc_f / gc_c if not np.isnan(gc_f) else float('nan')
            print(f"    {k+1:>2d}  {MSTAR_BIN_LOG_CENTERS[k]:6.2f}  "
                  f"{f_eff[k]:6.2f}  {gc_f:10.3f}  {gc_c:10.3f}  {ratio:8.2f}")
        print(f"  chi2(MOND) = {r['chi2_mond']:.1f}, chi2(C15) = {r['chi2_c15']:.1f}, "
              f"chi2(fit) = {r['chi2_fit']:.1f}")
        print(f"  Delta-chi2 (MOND - C15) = {r['dchi2_mond_c15']:+.1f}")

    print("\n" + "=" * 78)
    print("  GRAND SUMMARY: Delta-chi2 (MOND - C15) across all scenarios")
    print("=" * 78)
    print(f"\n  {'f_gas':>6s} | {'Uniform':>12s} {'M*-dep':>12s} | "
          f"{'Interpretation':>30s}")
    print("  " + "-" * 70)
    for ru, rm in zip(results_uniform, results_mstar_dep):
        d_u = ru['dchi2_mond_c15']
        d_m = rm['dchi2_mond_c15']
        if d_u > 9:
            interp = "Strong C15 (>3sigma)"
        elif d_u > 4:
            interp = "Moderate C15 (~2sigma)"
        elif d_u > 1:
            interp = "Mild C15"
        elif d_u > -1:
            interp = "Indistinguishable"
        elif d_u > -4:
            interp = "Mild MOND"
        else:
            interp = "MOND preferred"
        print(f"  {ru['f_gas']:6.1f} | {d_u:+12.1f} {d_m:+12.1f} | {interp:>30s}")

    print("\n--- Bin 1 (logM*=9.4) sensitivity ---")
    print(f"  {'f_gas':>6s} {'Uniform gc/a0':>14s} {'M*-dep gc/a0':>14s} "
          f"{'C15 pred':>10s}")
    for ru, rm in zip(results_uniform, results_mstar_dep):
        gc_u = ru['gc_fit'][0]
        gc_m = rm['gc_fit'][0]
        print(f"  {ru['f_gas']:6.1f}  {gc_u:14.3f}  {gc_m:14.3f}  "
              f"{gc_c15[0]/a0:10.3f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        colors_fgas = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

        ax = axs[0, 0]
        for ri, r in enumerate(results_uniform):
            ax.plot(MSTAR_BIN_LOG_CENTERS, r['gc_fit'], 'o-',
                    color=colors_fgas[ri], ms=7, lw=1.5,
                    label=f"f_gas={r['f_gas']:.1f}")
        ax.plot(MSTAR_BIN_LOG_CENTERS, [gc_c15[k]/a0 for k in range(N_BINS)],
                'k^--', ms=9, lw=2, label='C15 prediction')
        ax.axhline(1.0, color='gray', ls=':', lw=1, label='MOND (gc=a0)')
        ax.set_xlabel('log10(M*/Msun)')
        ax.set_ylabel('gc / a0')
        ax.set_title('(a) Uniform f_gas: gc per M* bin')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax = axs[0, 1]
        for ri, r in enumerate(results_mstar_dep):
            ax.plot(MSTAR_BIN_LOG_CENTERS, r['gc_fit'], 's-',
                    color=colors_fgas[ri], ms=7, lw=1.5,
                    label=f"f_gas={r['f_gas_base']:.1f}")
        ax.plot(MSTAR_BIN_LOG_CENTERS, [gc_c15[k]/a0 for k in range(N_BINS)],
                'k^--', ms=9, lw=2, label='C15 prediction')
        ax.axhline(1.0, color='gray', ls=':', lw=1, label='MOND (gc=a0)')
        ax.set_xlabel('log10(M*/Msun)')
        ax.set_ylabel('gc / a0')
        ax.set_title(f'(b) M*-dependent f_gas (R_acc~M*^{GAMMA_RACC}): gc per M* bin')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax = axs[1, 0]
        fgs = [r['f_gas'] for r in results_uniform]
        dchi_u = [r['dchi2_mond_c15'] for r in results_uniform]
        dchi_m = [r['dchi2_mond_c15'] for r in results_mstar_dep]
        ax.plot(fgs, dchi_u, 'bo-', ms=8, lw=2, label='Uniform correction')
        ax.plot(fgs, dchi_m, 'rs-', ms=8, lw=2, label='M*-dependent correction')
        ax.axhline(0, color='k', ls='-', lw=0.5)
        ax.axhline(4, color='gray', ls='--', lw=1, alpha=0.5)
        ax.axhline(9, color='gray', ls=':', lw=1, alpha=0.5)
        ax.axhline(-4, color='gray', ls='--', lw=1, alpha=0.5)
        ax.text(3.1, 4.5, '~2sigma', fontsize=8, color='gray')
        ax.text(3.1, 9.5, '~3sigma', fontsize=8, color='gray')
        ax.set_xlabel('f_gas (M_hot/M*)')
        ax.set_ylabel('Delta-chi2 (MOND - C15)')
        ax.set_title('(c) Model comparison: +ve = C15 preferred')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axs[1, 1]
        gc_u_b1 = [r['gc_fit'][0] for r in results_uniform]
        gc_m_b1 = [r['gc_fit'][0] for r in results_mstar_dep]
        ax.plot(fgs, gc_u_b1, 'bo-', ms=8, lw=2, label='Uniform')
        ax.plot(fgs, gc_m_b1, 'rs-', ms=8, lw=2, label='M*-dependent')
        ax.axhline(gc_c15[0]/a0, color='k', ls='--', lw=2,
                   label=f'C15 pred = {gc_c15[0]/a0:.3f} a0')
        ax.axhline(1.0, color='gray', ls=':', lw=1, label='MOND')
        ax.set_xlabel('f_gas (M_hot/M*)')
        ax.set_ylabel('gc_Bin1 / a0')
        ax.set_title('(d) Bin 1 (logM*=9.4) gc sensitivity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = BASE / "brouwer_kids_sensitivity_chi2.png"
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved: {out_png}")
    except Exception as e:
        print(f"\n[Plot error: {e}]")

    print("\n" + "=" * 78)
    print("  ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == '__main__':
    main()
