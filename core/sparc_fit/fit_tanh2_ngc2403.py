# -*- coding: utf-8 -*-
"""
NGC 2403: 2段階tanhモデル
T(r) = f_e × tanh(w(r/r_s1 - 1)) + (1-f_e) × tanh(w(r/r_s2 - 1))
フリーパラメータ: v_flat, r_s1, r_s2, f_e, Υ_d (5個)
固定: w = 1.0
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------
# データ読み込み
# -----------------------------------------------
def load_galaxy(filepath):
    data = np.loadtxt(filepath, comments='#')
    r      = data[:, 0]
    v_obs  = data[:, 1]
    v_err  = data[:, 2]
    v_gas  = data[:, 3]
    v_disk = data[:, 4]
    v_bul  = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)
    return r, v_obs, v_err, v_gas, v_disk, v_bul

# -----------------------------------------------
# モデル
# -----------------------------------------------
def tanh_single(r, r_s, w=1.0):
    return 0.5 * (1 + np.tanh(w * (r/r_s - 1)))

def model_2stage(r, v_flat, r_s1, r_s2, f_e, upsilon_d,
                 v_gas, v_disk, v_bul, w=1.0):
    T1 = tanh_single(r, r_s1, w)
    T2 = tanh_single(r, r_s2, w)
    T_combined = f_e * T1 + (1 - f_e) * T2

    v2 = (upsilon_d * v_disk**2
          + v_gas**2
          + v_bul**2
          + v_flat**2 * T_combined)
    return np.sqrt(np.maximum(v2, 0))

def model_1stage(r, v_flat, r_s, upsilon_d,
                 v_gas, v_disk, v_bul, w=1.0):
    T = tanh_single(r, r_s, w)
    v2 = (upsilon_d * v_disk**2
          + v_gas**2
          + v_bul**2
          + v_flat**2 * T)
    return np.sqrt(np.maximum(v2, 0))

# -----------------------------------------------
# フィット: 2段階
# -----------------------------------------------
def fit_2stage(r, v_obs, v_err, v_gas, v_disk, v_bul):
    def wrapper(r, v_flat, r_s1, r_s2, f_e, upsilon_d):
        return model_2stage(r, v_flat, r_s1, r_s2, f_e, upsilon_d,
                            v_gas, v_disk, v_bul)

    # Multi-start
    best_chi2 = 1e12
    best_popt = None
    best_pcov = None

    for vf0 in [80, 110, 140]:
        for rs1_0 in [1.0, 2.0, 3.0]:
            for rs2_0 in [5.0, 8.0, 12.0]:
                for fe0 in [0.3, 0.5, 0.7]:
                    for ud0 in [0.3, 0.5, 0.8]:
                        try:
                            popt, pcov = curve_fit(
                                wrapper, r, v_obs,
                                p0=[vf0, rs1_0, rs2_0, fe0, ud0],
                                bounds=([30, 0.1, 1.0, 0.01, 0.05],
                                        [300, 15.0, 40.0, 0.99, 3.0]),
                                sigma=np.clip(v_err, 1, None),
                                absolute_sigma=True,
                                maxfev=10000)
                            v_fit = wrapper(r, *popt)
                            c2 = np.sum(((v_obs - v_fit)/np.clip(v_err,1,None))**2)
                            if c2 < best_chi2:
                                best_chi2 = c2
                                best_popt = popt
                                best_pcov = pcov
                        except:
                            continue

    if best_popt is None:
        return None, None, None, None

    perr = np.sqrt(np.diag(best_pcov))
    v_fit = wrapper(r, *best_popt)
    dof = max(len(r) - 5, 1)
    chi2_dof = best_chi2 / dof
    return best_popt, perr, chi2_dof, v_fit

# -----------------------------------------------
# フィット: 1段階（比較用）
# -----------------------------------------------
def fit_1stage(r, v_obs, v_err, v_gas, v_disk, v_bul):
    def wrapper(r, v_flat, r_s, upsilon_d):
        return model_1stage(r, v_flat, r_s, upsilon_d,
                            v_gas, v_disk, v_bul)
    best_chi2 = 1e12
    best_popt = None
    best_pcov = None

    for vf0 in [80, 110, 140, 180]:
        for rs0 in [1, 3, 5, 10, 15]:
            for ud0 in [0.3, 0.5, 0.8, 1.2]:
                try:
                    popt, pcov = curve_fit(
                        wrapper, r, v_obs,
                        p0=[vf0, rs0, ud0],
                        bounds=([20, 0.1, 0.05],
                                [350, 50, 3.0]),
                        sigma=np.clip(v_err, 1, None),
                        absolute_sigma=True,
                        maxfev=10000)
                    v_fit = wrapper(r, *popt)
                    c2 = np.sum(((v_obs - v_fit)/np.clip(v_err,1,None))**2)
                    if c2 < best_chi2:
                        best_chi2 = c2
                        best_popt = popt
                        best_pcov = pcov
                except:
                    continue

    if best_popt is None:
        return None, None, None, None

    perr = np.sqrt(np.diag(best_pcov))
    v_fit = wrapper(r, *best_popt)
    dof = max(len(r) - 3, 1)
    chi2_dof = best_chi2 / dof
    return best_popt, perr, chi2_dof, v_fit

# -----------------------------------------------
# プロット
# -----------------------------------------------
def plot_result(r, v_obs, v_err, v_gas, v_disk, v_bul,
                popt_2, chi2_2, popt_1, chi2_1):
    v_flat, r_s1, r_s2, f_e, upsilon_d = popt_2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    # --- 左: 2段階モデル ---
    ax = axes[0, 0]
    rp = np.linspace(0.01, r.max()*1.15, 400)
    vdi = np.interp(rp, r, v_disk)
    vgi = np.interp(rp, r, v_gas)
    vbi = np.interp(rp, r, v_bul)

    T1 = tanh_single(rp, r_s1)
    T2 = tanh_single(rp, r_s2)
    T_c = f_e * T1 + (1 - f_e) * T2

    v_mem_e = np.sqrt(np.maximum(v_flat**2 * f_e * T1, 0))
    v_mem_p = np.sqrt(np.maximum(v_flat**2 * (1-f_e) * T2, 0))
    v_bar   = np.sqrt(np.maximum(upsilon_d*vdi**2 + vgi**2 + vbi**2, 0))
    v_tot   = model_2stage(rp, *popt_2, vgi, vdi, vbi)

    ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', ms=4, capsize=2, label='data', zorder=5)
    ax.plot(rp, v_tot, 'r-', lw=2.5, label=f'2-stage ($\\chi^2$={chi2_2:.3f})')
    ax.plot(rp, v_mem_e, 'g--', lw=1.3, alpha=0.8,
            label=f'elastic ($f_e$={f_e:.2f}, $r_{{s1}}$={r_s1:.1f})')
    ax.plot(rp, v_mem_p, 'm--', lw=1.3, alpha=0.8,
            label=f'plastic ($r_{{s2}}$={r_s2:.1f})')
    ax.plot(rp, v_bar, 'b:', lw=1.2, alpha=0.7, label=f'baryon ($\\Upsilon_d$={upsilon_d:.2f})')
    ax.axvline(r_s1, color='g', alpha=0.3, ls=':', lw=1)
    ax.axvline(r_s2, color='m', alpha=0.3, ls=':', lw=1)
    ax.set_ylabel('v [km/s]')
    ax.set_title('NGC 2403: 2-stage tanh model', fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.15)

    # 残差
    ax = axes[1, 0]
    v_fit2 = model_2stage(r, *popt_2, v_gas, v_disk, v_bul)
    resid2 = (v_obs - v_fit2) / np.clip(v_err, 1, None)
    ax.plot(r, resid2, 'ro-', ms=4, lw=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(+2, color='gray', ls='--', lw=0.6)
    ax.axhline(-2, color='gray', ls='--', lw=0.6)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('residual / $\\sigma$')
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.15)

    # --- 右: 1段階モデル（比較） ---
    ax = axes[0, 1]
    vf1, rs1, ud1 = popt_1
    v_tot1 = model_1stage(rp, vf1, rs1, ud1, vgi, vdi, vbi)
    v_bar1 = np.sqrt(np.maximum(ud1*vdi**2 + vgi**2 + vbi**2, 0))
    T_1s = tanh_single(rp, rs1)
    v_mem1 = np.sqrt(np.maximum(vf1**2 * T_1s, 0))

    ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', ms=4, capsize=2, label='data', zorder=5)
    ax.plot(rp, v_tot1, 'r-', lw=2.5, label=f'1-stage ($\\chi^2$={chi2_1:.3f})')
    ax.plot(rp, v_mem1, 'g-.', lw=1.3, alpha=0.8,
            label=f'membrane ($r_s$={rs1:.1f})')
    ax.plot(rp, v_bar1, 'b:', lw=1.2, alpha=0.7, label=f'baryon ($\\Upsilon_d$={ud1:.2f})')
    ax.axvline(rs1, color='g', alpha=0.3, ls=':', lw=1)
    ax.set_ylabel('v [km/s]')
    ax.set_title('NGC 2403: 1-stage tanh (comparison)', fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.15)

    # 残差
    ax = axes[1, 1]
    v_fit1 = model_1stage(r, *popt_1, v_gas, v_disk, v_bul)
    resid1 = (v_obs - v_fit1) / np.clip(v_err, 1, None)
    ax.plot(r, resid1, 'ro-', ms=4, lw=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(+2, color='gray', ls='--', lw=0.6)
    ax.axhline(-2, color='gray', ls='--', lw=0.6)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('residual / $\\sigma$')
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.15)

    plt.suptitle('NGC 2403: 1-stage vs 2-stage tanh  (w=1.0 fixed)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('ngc2403_2stage_tanh.png', dpi=150, bbox_inches='tight')
    print("-> ngc2403_2stage_tanh.png saved")

# -----------------------------------------------
# メイン
# -----------------------------------------------
if __name__ == '__main__':
    data_dir = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
    filepath = data_dir / 'NGC2403_rotmod.dat'

    r, v_obs, v_err, v_gas, v_disk, v_bul = load_galaxy(filepath)
    v_err = np.clip(v_err, 1.0, None)

    print("=" * 55)
    print("NGC 2403: 2-stage tanh fitting")
    print("=" * 55)

    # 2段階フィット
    popt_2, perr_2, chi2_2, vfit_2 = fit_2stage(
        r, v_obs, v_err, v_gas, v_disk, v_bul)

    if popt_2 is not None:
        labels = ['v_flat', 'r_s1', 'r_s2', 'f_e', 'Upsilon_d']
        units  = ['km/s', 'kpc', 'kpc', '', '']
        print(f"\n2-stage: chi2/dof = {chi2_2:.4f}")
        print(f"{'param':<12} {'value':>10} {'error':>10} {'unit':>6}")
        print("-" * 42)
        for l, v, e, u in zip(labels, popt_2, perr_2, units):
            print(f"{l:<12} {v:10.3f} {e:10.3f} {u:>6}")
    else:
        print("2-stage fit failed!")

    # 1段階フィット（比較）
    popt_1, perr_1, chi2_1, vfit_1 = fit_1stage(
        r, v_obs, v_err, v_gas, v_disk, v_bul)

    if popt_1 is not None:
        labels1 = ['v_flat', 'r_s', 'Upsilon_d']
        print(f"\n1-stage: chi2/dof = {chi2_1:.4f}")
        print(f"{'param':<12} {'value':>10} {'error':>10}")
        print("-" * 35)
        for l, v, e in zip(labels1, popt_1, perr_1):
            print(f"{l:<12} {v:10.3f} {e:10.3f}")

    # AIC 比較
    if popt_2 is not None and popt_1 is not None:
        n = len(r)
        chi2_raw_2 = chi2_2 * max(n-5, 1)
        chi2_raw_1 = chi2_1 * max(n-3, 1)
        aic_2 = chi2_raw_2 + 2*5
        aic_1 = chi2_raw_1 + 2*3
        daic = aic_2 - aic_1
        print(f"\n=== AIC comparison ===")
        print(f"  AIC(2-stage) = {aic_2:.1f}  (5 params)")
        print(f"  AIC(1-stage) = {aic_1:.1f}  (3 params)")
        print(f"  dAIC = {daic:.1f}")
        if daic < -2:
            print(f"  -> 2-stage is significantly better")
        elif daic > 2:
            print(f"  -> 1-stage is sufficient (2-stage overfits)")
        else:
            print(f"  -> Models are comparable")

    # プロット
    if popt_2 is not None and popt_1 is not None:
        plot_result(r, v_obs, v_err, v_gas, v_disk, v_bul,
                    popt_2, chi2_2, popt_1, chi2_1)
