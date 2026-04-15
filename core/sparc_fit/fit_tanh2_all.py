# -*- coding: utf-8 -*-
"""
全5銀河: 1段階 vs 2段階 tanh モデル比較
Υ_d 拘束あり/なし 2パターン実行
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# -----------------------------------------------
# 設定
# -----------------------------------------------
GALAXIES = ['NGC3198', 'NGC6503', 'UGC02885', 'DDO154', 'NGC2403']
DATA_DIR = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
OUT_DIR  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1')

# -----------------------------------------------
# データ読み込み
# -----------------------------------------------
def load_sparc(filepath):
    data = np.loadtxt(filepath, comments='#')
    r      = data[:, 0]
    v_obs  = data[:, 1]
    v_err  = np.maximum(data[:, 2], 2.0)
    v_gas  = data[:, 3]
    v_disk = data[:, 4]
    v_bul  = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)
    return r, v_obs, v_err, v_gas, v_disk, v_bul

# -----------------------------------------------
# モデル
# -----------------------------------------------
def tanh_T(r, r_s, w=1.0):
    return 0.5 * (1.0 + np.tanh(w * (r - r_s) / r_s))

def v_baryons(v_gas, v_disk, v_bul, upsilon_d, upsilon_b=0.7):
    v2  = np.sign(v_gas)  * v_gas**2
    v2 += upsilon_d * np.sign(v_disk) * v_disk**2
    v2 += upsilon_b * np.sign(v_bul)  * v_bul**2
    return v2

def model_1stage(r, v_flat, r_s, upsilon_d, v_gas, v_disk, v_bul):
    T  = tanh_T(r, r_s)
    v2 = v_flat**2 * T + v_baryons(v_gas, v_disk, v_bul, upsilon_d)
    return np.sqrt(np.maximum(v2, 0))

def model_2stage(r, v_flat, r_s1, r_s2, f_e, upsilon_d,
                 v_gas, v_disk, v_bul):
    T1 = tanh_T(r, r_s1)
    T2 = tanh_T(r, r_s2)
    T  = f_e * T1 + (1 - f_e) * T2
    v2 = v_flat**2 * T + v_baryons(v_gas, v_disk, v_bul, upsilon_d)
    return np.sqrt(np.maximum(v2, 0))

# -----------------------------------------------
# フィッティング（マルチスタート）
# -----------------------------------------------
def fit_galaxy(r, v_obs, v_err, v_gas, v_disk, v_bul,
               upsilon_min=0.0):
    ud_lo = max(upsilon_min, 0.01)
    results = {}

    # --- 1段階 ---
    def wrap1(r, v_flat, r_s, ud):
        return model_1stage(r, v_flat, r_s, ud, v_gas, v_disk, v_bul)

    best1_chi2 = 1e12
    best1_popt = best1_pcov = None
    for vf0 in [60, 100, 140, 200]:
        for rs0 in [1, 3, 7, 15]:
            for ud0 in [0.3, 0.5, 0.8, 1.2]:
                try:
                    popt, pcov = curve_fit(
                        wrap1, r, v_obs, p0=[vf0, rs0, ud0],
                        bounds=([20, 0.1, ud_lo], [400, 50, 2.5]),
                        sigma=v_err, absolute_sigma=True, maxfev=20000)
                    vf = wrap1(r, *popt)
                    c2 = np.sum(((v_obs - vf) / v_err)**2)
                    if c2 < best1_chi2:
                        best1_chi2 = c2
                        best1_popt = popt
                        best1_pcov = pcov
                except:
                    continue

    if best1_popt is not None:
        v_fit1 = wrap1(r, *best1_popt)
        dof = max(len(r) - 3, 1)
        results['1stage'] = dict(
            popt=best1_popt, pcov=best1_pcov,
            chi2=best1_chi2/dof, aic=best1_chi2+2*3, v_fit=v_fit1)
    else:
        results['1stage'] = dict(error='fit failed')

    # --- 2段階 ---
    def wrap2(r, v_flat, r_s1, r_s2, f_e, ud):
        return model_2stage(r, v_flat, r_s1, r_s2, f_e, ud,
                            v_gas, v_disk, v_bul)

    best2_chi2 = 1e12
    best2_popt = best2_pcov = None
    for vf0 in [60, 100, 140, 200]:
        for rs1_0 in [0.5, 1.0, 2.0, 3.0]:
            for rs2_0 in [4, 8, 15]:
                for fe0 in [0.3, 0.5, 0.7]:
                    for ud0 in [0.3, 0.5, 0.8]:
                        try:
                            popt, pcov = curve_fit(
                                wrap2, r, v_obs,
                                p0=[vf0, rs1_0, rs2_0, fe0, ud0],
                                bounds=([20, 0.05, 0.5, 0.0, ud_lo],
                                        [400, 15, 60, 1.0, 2.5]),
                                sigma=v_err, absolute_sigma=True,
                                maxfev=20000)
                            if popt[1] > popt[2]:
                                popt[1], popt[2] = popt[2], popt[1]
                                popt[3] = 1.0 - popt[3]
                            vf = wrap2(r, *popt)
                            c2 = np.sum(((v_obs - vf) / v_err)**2)
                            if c2 < best2_chi2:
                                best2_chi2 = c2
                                best2_popt = popt
                                best2_pcov = pcov
                        except:
                            continue

    if best2_popt is not None:
        v_fit2 = wrap2(r, *best2_popt)
        dof = max(len(r) - 5, 1)
        results['2stage'] = dict(
            popt=best2_popt, pcov=best2_pcov,
            chi2=best2_chi2/dof, aic=best2_chi2+2*5, v_fit=v_fit2)
    else:
        results['2stage'] = dict(error='fit failed')

    return results

# -----------------------------------------------
# プロット
# -----------------------------------------------
def plot_galaxy(ax_main, ax_res, r, v_obs, v_err,
                v_gas, v_disk, v_bul, res, galaxy_name):
    ax_main.errorbar(r, v_obs, yerr=v_err, fmt='ko', ms=3,
                     capsize=1.5, zorder=5, label='data')

    if '2stage' in res and 'popt' in res['2stage']:
        p = res['2stage']['popt']
        v_flat, r_s1, r_s2, f_e, ud = p
        rp = np.linspace(0.01, r.max()*1.1, 300)
        vdi = np.interp(rp, r, v_disk)
        vgi = np.interp(rp, r, v_gas)
        vbi = np.interp(rp, r, v_bul)
        v_tot = model_2stage(rp, *p, vgi, vdi, vbi)
        ax_main.plot(rp, v_tot, 'r-', lw=2,
                     label=f"2-stage $\\chi^2$={res['2stage']['chi2']:.2f}")
        T1 = tanh_T(rp, r_s1)
        T2 = tanh_T(rp, r_s2)
        ax_main.plot(rp, v_flat*np.sqrt(np.maximum(f_e*T1, 0)), 'g--',
                     lw=1.2, alpha=0.8, label=f'elastic $r_{{s1}}$={r_s1:.1f}')
        ax_main.plot(rp, v_flat*np.sqrt(np.maximum((1-f_e)*T2, 0)), 'm--',
                     lw=1.2, alpha=0.8, label=f'plastic $r_{{s2}}$={r_s2:.1f}')
        v_bar = np.sqrt(np.maximum(v_baryons(vgi, vdi, vbi, ud), 0))
        ax_main.plot(rp, v_bar, 'b:', lw=1, alpha=0.6, label=f'baryon $\\Upsilon_d$={ud:.2f}')
        ax_main.axvline(r_s1, color='g', alpha=0.2, ls=':', lw=0.8)
        ax_main.axvline(r_s2, color='m', alpha=0.2, ls=':', lw=0.8)
        resid = (v_obs - res['2stage']['v_fit']) / v_err
        ax_res.plot(r, resid, 'ro-', ms=3, lw=0.6, label='2-stage')

    if '1stage' in res and 'popt' in res['1stage']:
        rp = np.linspace(0.01, r.max()*1.1, 300)
        vdi = np.interp(rp, r, v_disk)
        vgi = np.interp(rp, r, v_gas)
        vbi = np.interp(rp, r, v_bul)
        p1 = res['1stage']['popt']
        v_tot1 = model_1stage(rp, *p1, vgi, vdi, vbi)
        ax_main.plot(rp, v_tot1, 'c--', lw=1.5, alpha=0.7,
                     label=f"1-stage $\\chi^2$={res['1stage']['chi2']:.2f}")
        resid1 = (v_obs - res['1stage']['v_fit']) / v_err
        ax_res.plot(r, resid1, 'b^-', ms=2.5, lw=0.4, alpha=0.5, label='1-stage')

    ax_main.set_title(galaxy_name, fontsize=10)
    ax_main.set_ylabel('v [km/s]')
    ax_main.legend(fontsize=6, loc='lower right')
    ax_main.set_xlim(0, None)
    ax_main.set_ylim(0, None)
    ax_main.grid(True, alpha=0.15)

    ax_res.axhline(0, color='k', lw=0.8)
    ax_res.axhline(+2, color='gray', lw=0.5, ls='--')
    ax_res.axhline(-2, color='gray', lw=0.5, ls='--')
    ax_res.set_ylabel('resid/$\\sigma$', fontsize=7)
    ax_res.set_ylim(-5, 5)
    ax_res.set_xlim(0, None)
    ax_res.legend(fontsize=5)
    ax_res.grid(True, alpha=0.1)

# -----------------------------------------------
# サマリー
# -----------------------------------------------
def print_summary(all_results, label):
    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"{'='*75}")
    print(f"{'galaxy':<12} {'chi2_1':>8} {'chi2_2':>8} {'dAIC':>8} "
          f"{'r_s1':>6} {'r_s2':>6} {'f_e':>5} {'Ud':>5} {'vflat':>7}")
    print("-"*75)

    for gal, res in all_results.items():
        c1  = res['1stage'].get('chi2', float('nan'))
        c2  = res['2stage'].get('chi2', float('nan'))
        a1  = res['1stage'].get('aic',  float('nan'))
        a2  = res['2stage'].get('aic',  float('nan'))
        da  = a2 - a1 if not (np.isnan(a1) or np.isnan(a2)) else float('nan')

        if 'popt' in res['2stage']:
            p = res['2stage']['popt']
            vf, rs1, rs2, fe, ud = p
        else:
            vf = rs1 = rs2 = fe = ud = float('nan')

        verdict = 'v' if da < -2 else ('=' if abs(da) <= 2 else '^')
        print(f"{gal:<12} {c1:8.3f} {c2:8.3f} {da:8.1f} "
              f"{rs1:6.2f} {rs2:6.2f} {fe:5.2f} {ud:5.2f} {vf:7.1f}  {verdict}")

    print("-"*75)
    print("  v=2-stage wins, ^=1-stage wins, ==comparable")

# -----------------------------------------------
# Run one configuration
# -----------------------------------------------
def run_config(constrain_upsilon, upsilon_min):
    ud_min = upsilon_min if constrain_upsilon else 0.0
    tag = "constrained" if constrain_upsilon else "free"
    label = f"Upsilon_d >= {upsilon_min}" if constrain_upsilon else "Upsilon_d free"

    fig, axes = plt.subplots(
        len(GALAXIES)*2, 1,
        figsize=(10, 4.5*len(GALAXIES)),
        gridspec_kw={'height_ratios': [3,1]*len(GALAXIES)})

    all_results = {}
    for i, gal in enumerate(GALAXIES):
        fpath = DATA_DIR / f'{gal}_rotmod.dat'
        if not fpath.exists():
            print(f"[SKIP] {gal}: file not found")
            continue

        r, v_obs, v_err, v_gas, v_disk, v_bul = load_sparc(fpath)
        res = fit_galaxy(r, v_obs, v_err, v_gas, v_disk, v_bul,
                         upsilon_min=ud_min)
        all_results[gal] = res
        plot_galaxy(axes[i*2], axes[i*2+1],
                    r, v_obs, v_err, v_gas, v_disk, v_bul, res, gal)

    axes[-1].set_xlabel('r [kpc]')
    plt.suptitle(f'1-stage vs 2-stage tanh  ({label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outname = OUT_DIR / f'all_galaxies_2stage_{tag}.png'
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> {outname}")

    print_summary(all_results, f"Summary ({label})")
    return all_results

# -----------------------------------------------
# メイン
# -----------------------------------------------
if __name__ == '__main__':
    print("=" * 75)
    print("  Run A: Upsilon_d free")
    print("=" * 75)
    res_free = run_config(constrain_upsilon=False, upsilon_min=0.30)

    print("\n\n")
    print("=" * 75)
    print("  Run B: Upsilon_d >= 0.30")
    print("=" * 75)
    res_const = run_config(constrain_upsilon=True, upsilon_min=0.30)

    # --- 比較表 ---
    print(f"\n\n{'='*75}")
    print("  Upsilon constraint effect")
    print(f"{'='*75}")
    print(f"{'galaxy':<12} {'chi2_free':>10} {'chi2_const':>11} {'Ud_free':>8} {'Ud_const':>9}")
    print("-"*55)
    for gal in GALAXIES:
        if gal in res_free and gal in res_const:
            c_f = res_free[gal]['2stage'].get('chi2', float('nan'))
            c_c = res_const[gal]['2stage'].get('chi2', float('nan'))
            ud_f = res_free[gal]['2stage']['popt'][4] if 'popt' in res_free[gal]['2stage'] else float('nan')
            ud_c = res_const[gal]['2stage']['popt'][4] if 'popt' in res_const[gal]['2stage'] else float('nan')
            print(f"{gal:<12} {c_f:10.3f} {c_c:11.3f} {ud_f:8.2f} {ud_c:9.2f}")
