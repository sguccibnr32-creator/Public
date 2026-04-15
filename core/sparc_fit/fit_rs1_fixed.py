# -*- coding: utf-8 -*-
"""
r_s1 = 0.6×h_R 固定での2段階フィット
Υ_d 縮退の解消を検証
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

UPSILON_MIN = 0.05
W_FIX       = 1.0

DATA_DIR = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
OUT_DIR  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1')

GALAXY_PARAMS = {
    'NGC1003' : {'h_R': 2.31, 'rs1_fixed': 2.31*0.6},
    'NGC2403' : {'h_R': 1.75, 'rs1_fixed': 1.75*0.6},
    'NGC2903' : {'h_R': 2.60, 'rs1_fixed': 2.60*0.6},
    'NGC6015' : {'h_R': 3.31, 'rs1_fixed': 3.31*0.6},
    'UGC00128': {'h_R': 5.20, 'rs1_fixed': 5.20*0.6},
}

PREV_FREE = {
    'NGC1003' : {'r_s1':3.08,'r_s2':18.25,'f_e':0.39,'ud':0.30,'chi2':1.62},
    'NGC2403' : {'r_s1':0.79,'r_s2': 3.79,'f_e':0.20,'ud':0.30,'chi2':3.29},
    'NGC2903' : {'r_s1':1.56,'r_s2': 3.93,'f_e':0.59,'ud':0.30,'chi2':7.19},
    'NGC6015' : {'r_s1':1.62,'r_s2': 9.77,'f_e':0.78,'ud':0.30,'chi2':6.90},
    'UGC00128': {'r_s1':3.79,'r_s2':11.12,'f_e':0.59,'ud':0.30,'chi2':1.24},
}

def load_sparc(filepath):
    data = np.loadtxt(filepath, comments='#')
    mask = data[:, 1] > 0
    data = data[mask]
    r, v_obs = data[:,0], data[:,1]
    v_err = np.maximum(data[:,2], 2.0)
    v_gas, v_disk = data[:,3], data[:,4]
    v_bul = data[:,5] if data.shape[1] > 5 else np.zeros_like(r)
    return r, v_obs, v_err, v_gas, v_disk, v_bul

def tanh_T(r, r_s):
    return 0.5 * (1.0 + np.tanh(W_FIX * (r - r_s) / r_s))

def v2_bar(v_gas, v_disk, v_bul, ud, ub=0.7):
    return (np.sign(v_gas)*v_gas**2 + ud*np.sign(v_disk)*v_disk**2
            + ub*np.sign(v_bul)*v_bul**2)

def model_2s_rs1fixed(r, vf, rs2, fe, ud, rs1_fixed, vg, vd, vb):
    T = fe*tanh_T(r, rs1_fixed) + (1-fe)*tanh_T(r, rs2)
    return np.sqrt(np.maximum(vf**2*T + v2_bar(vg, vd, vb, ud), 0))

def model_2s_free(r, vf, rs1, rs2, fe, ud, vg, vd, vb):
    T = fe*tanh_T(r, rs1) + (1-fe)*tanh_T(r, rs2)
    return np.sqrt(np.maximum(vf**2*T + v2_bar(vg, vd, vb, ud), 0))

def fit_galaxy(gal, r, v_obs, v_err, v_gas, v_disk, v_bul):
    rs1_fixed = GALAXY_PARAMS[gal]['rs1_fixed']
    rm, vm, n = r.max(), v_obs.max(), len(r)
    results = {}

    # r_s1 fixed (4 params: vf, rs2, fe, ud)
    def wf(r, vf, rs2, fe, ud):
        return model_2s_rs1fixed(r, vf, rs2, fe, ud, rs1_fixed, v_gas, v_disk, v_bul)

    best_f = None
    for p0 in [[vm*0.9,rm*0.4,0.5,0.5], [vm*0.9,rm*0.6,0.3,0.4],
                [vm*0.8,rm*0.3,0.7,0.6], [vm*1.0,rm*0.5,0.4,0.3]]:
        try:
            po, pc = curve_fit(wf, r, v_obs, p0=p0,
                               bounds=([5,rs1_fixed*1.2,0.0,UPSILON_MIN],[500,rm*3,1.0,2.0]),
                               sigma=v_err, absolute_sigma=True, maxfev=30000)
            vf = wf(r, *po)
            c2 = np.sum(((v_obs-vf)/v_err)**2)
            aic = c2 + 2*4
            if best_f is None or aic < best_f['aic']:
                best_f = dict(popt=po, pcov=pc, chi2=c2/max(n-4,1), aic=aic, v_fit=vf)
        except: continue
    results['fixed'] = best_f

    # r_s1 free (5 params: vf, rs1, rs2, fe, ud)
    def wv(r, vf, rs1, rs2, fe, ud):
        return model_2s_free(r, vf, rs1, rs2, fe, ud, v_gas, v_disk, v_bul)

    best_v = None
    for p0 in [[vm*0.9,rs1_fixed,rm*0.4,0.5,0.5], [vm*0.9,rs1_fixed,rm*0.6,0.3,0.4],
                [vm*0.8,rs1_fixed*0.5,rm*0.3,0.7,0.3]]:
        try:
            po, pc = curve_fit(wv, r, v_obs, p0=p0,
                               bounds=([5,0.01,0.5,0.0,UPSILON_MIN],[500,rm,rm*3,1.0,2.0]),
                               sigma=v_err, absolute_sigma=True, maxfev=30000)
            if po[1] > po[2]:
                po[1], po[2] = po[2], po[1]
                po[3] = 1.0 - po[3]
            vf = wv(r, *po)
            c2 = np.sum(((v_obs-vf)/v_err)**2)
            aic = c2 + 2*5
            if best_v is None or aic < best_v['aic']:
                best_v = dict(popt=po, pcov=pc, chi2=c2/max(n-5,1), aic=aic, v_fit=vf)
        except: continue
    results['free'] = best_v
    return results

def plot_one(ax_m, ax_r, r, vo, ve, vg, vd, vb, gal, rf, rv):
    rs1f = GALAXY_PARAMS[gal]['rs1_fixed']
    h_R  = GALAXY_PARAMS[gal]['h_R']
    rp = np.linspace(0.01, r.max()*1.1, 300)
    vgi, vdi, vbi = np.interp(rp,r,vg), np.interp(rp,r,vd), np.interp(rp,r,vb)

    ax_m.errorbar(r, vo, yerr=ve, fmt='ko', ms=3, capsize=1.5, zorder=5, label='data')

    if rf:
        vf, rs2, fe, ud = rf['popt']
        T1 = tanh_T(rp, rs1f); T2 = tanh_T(rp, rs2)
        v_tot = model_2s_rs1fixed(rp, vf, rs2, fe, ud, rs1f, vgi, vdi, vbi)
        ax_m.plot(rp, v_tot, 'r-', lw=2, label=f"fixed $\\chi^2$={rf['chi2']:.2f}, $\\Upsilon_d$={ud:.2f}")
        ax_m.plot(rp, vf*np.sqrt(np.maximum(fe*T1,0)), 'g--', lw=1.2, alpha=0.8,
                  label=f'elastic ($r_{{s1}}$={rs1f:.2f})')
        ax_m.plot(rp, vf*np.sqrt(np.maximum((1-fe)*T2,0)), 'm--', lw=1.2, alpha=0.8,
                  label=f'plastic ($r_{{s2}}$={rs2:.1f})')
        v_bar = np.sqrt(np.maximum(v2_bar(vgi,vdi,vbi,ud),0))
        ax_m.plot(rp, v_bar, 'b:', lw=1, alpha=0.5, label='baryon')
        ax_m.axvline(rs1f, color='g', alpha=0.3, ls=':', lw=0.8)
        ax_m.axvline(rs2, color='m', alpha=0.3, ls=':', lw=0.8)
        resid = (vo - rf['v_fit'])/ve
        ax_r.plot(r, resid, 'ro-', ms=3, lw=0.6, label='fixed')

    if rv:
        v_tot_v = model_2s_free(rp, *rv['popt'], vgi, vdi, vbi)
        ax_m.plot(rp, v_tot_v, 'c--', lw=1.5, alpha=0.7,
                  label=f"free $\\chi^2$={rv['chi2']:.2f}, $\\Upsilon_d$={rv['popt'][4]:.2f}")
        resid_v = (vo - rv['v_fit'])/ve
        ax_r.plot(r, resid_v, 'b^-', ms=2.5, lw=0.4, alpha=0.5, label='free')

    ax_m.set_title(f'{gal}  ($h_R$={h_R:.2f}, $r_{{s1}}$ fixed={rs1f:.2f} kpc)', fontsize=9)
    ax_m.set_ylabel('v [km/s]')
    ax_m.legend(fontsize=6, loc='lower right')
    ax_m.set_xlim(0,None); ax_m.set_ylim(0,None)
    ax_m.grid(True, alpha=0.15)
    ax_r.axhline(0, color='k', lw=0.8)
    ax_r.axhline(+2, color='gray', lw=0.5, ls='--')
    ax_r.axhline(-2, color='gray', lw=0.5, ls='--')
    ax_r.set_ylabel('resid/$\\sigma$', fontsize=7)
    ax_r.set_ylim(-5,5); ax_r.set_xlim(0,None)
    ax_r.legend(fontsize=6); ax_r.grid(True, alpha=0.1)

# ============================================================
# Main
# ============================================================
gals = list(GALAXY_PARAMS.keys())
fig, axes = plt.subplots(len(gals)*2, 1, figsize=(10, 5*len(gals)),
                          gridspec_kw={'height_ratios': [3,1]*len(gals)})

all_res = {}
for i, gal in enumerate(gals):
    fpath = DATA_DIR / f'{gal}_rotmod.dat'
    r, vo, ve, vg, vd, vb = load_sparc(fpath)
    print(f"[{i+1}/5] {gal}...")
    res = fit_galaxy(gal, r, vo, ve, vg, vd, vb)
    all_res[gal] = res
    plot_one(axes[i*2], axes[i*2+1], r, vo, ve, vg, vd, vb, gal, res['fixed'], res['free'])

axes[-1].set_xlabel('r [kpc]')
plt.suptitle('r_s1 = 0.6 h_R fixed vs r_s1 free  (2-stage tanh)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(OUT_DIR / 'rs1_fixed_comparison.png', dpi=150, bbox_inches='tight')
print("-> rs1_fixed_comparison.png saved")

# Summary
print(f"\n{'='*80}")
print("  r_s1 fixed vs free: comparison summary")
print('='*80)
print(f"{'galaxy':<12} {'rs1_fix':>8} {'rs2_fix':>8} {'fe_fix':>7} {'Ud_fix':>7} {'chi2_fix':>9} | "
      f"{'rs1_free':>9} {'rs2_free':>9} {'Ud_free':>8} {'chi2_free':>10}")
print('-'*95)

ud_fixed, ud_free, rs2_fixed, rs2_free = [], [], [], []
for gal in gals:
    res = all_res[gal]
    rf, rv = res['fixed'], res['free']
    rs1f = GALAXY_PARAMS[gal]['rs1_fixed']
    if rf:
        vf, rs2, fe, ud = rf['popt']
        ud_fixed.append(ud); rs2_fixed.append(rs2)
        fs = f"{rs1f:8.2f} {rs2:8.2f} {fe:7.2f} {ud:7.3f} {rf['chi2']:9.3f}"
    else:
        fs = "  [FAIL]"
    if rv:
        ud_v = rv['popt'][4]; rs1_v = rv['popt'][1]; rs2_v = rv['popt'][2]
        ud_free.append(ud_v); rs2_free.append(rs2_v)
        vs = f"{rs1_v:9.2f} {rs2_v:9.2f} {ud_v:8.3f} {rv['chi2']:10.3f}"
    else:
        vs = "  [FAIL]"
    print(f"{gal:<12} {fs} | {vs}")

print('='*95)
print(f"\nUpsilon_d comparison:")
print(f"  Fixed model: median={np.median(ud_fixed):.3f}  mean={np.mean(ud_fixed):.3f}  "
      f"range=[{np.min(ud_fixed):.3f}, {np.max(ud_fixed):.3f}]")
print(f"  Free  model: median={np.median(ud_free):.3f}  mean={np.mean(ud_free):.3f}  "
      f"range=[{np.min(ud_free):.3f}, {np.max(ud_free):.3f}]")
print(f"\nr_s2 comparison:")
print(f"  Fixed model: median={np.median(rs2_fixed):.2f}  mean={np.mean(rs2_fixed):.2f}")
print(f"  Free  model: median={np.median(rs2_free):.2f}  mean={np.mean(rs2_free):.2f}")

# Verdict
print(f"\n{'='*80}")
ud_med = np.median(ud_fixed)
if ud_med > 0.25:
    print(f"  -> Upsilon_d degeneracy RESOLVED (median={ud_med:.3f} > 0.25)")
    print(f"     Fixing r_s1 = 0.6 h_R breaks the r_s1/Upsilon_d degeneracy")
else:
    print(f"  -> Upsilon_d degeneracy NOT resolved (median={ud_med:.3f} <= 0.25)")
    print(f"     r_s1 fixing alone is insufficient")
# chi2 penalty
chi2_fixed = [all_res[g]['fixed']['chi2'] for g in gals if all_res[g]['fixed']]
chi2_free  = [all_res[g]['free']['chi2'] for g in gals if all_res[g]['free']]
print(f"\n  chi2/dof (fixed): median={np.median(chi2_fixed):.3f}")
print(f"  chi2/dof (free):  median={np.median(chi2_free):.3f}")
dc = np.median(chi2_fixed) - np.median(chi2_free)
print(f"  delta chi2 = {dc:+.3f}  ({'acceptable' if abs(dc) < 1 else 'significant penalty'})")
print('='*80)
