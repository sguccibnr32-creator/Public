# -*- coding: utf-8 -*-
"""
全SPARC: 1段階 vs 2段階 tanh モデル比較
Υ_d >= 0.30 拘束、w=1.0 固定
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

UPSILON_MIN   = 0.30
UPSILON_B_FIX = 0.70
W_FIX         = 1.0
DAIC_THRESHOLD = -10.0
V_ERR_MIN     = 2.0

DATA_DIR = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
OUT_DIR  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1')

def load_sparc(filepath):
    try:
        data = np.loadtxt(filepath, comments='#')
    except:
        return None
    if data.ndim == 1 or len(data) < 4:
        return None
    r, v_obs, v_err = data[:,0], data[:,1], np.maximum(data[:,2], V_ERR_MIN)
    v_gas, v_disk = data[:,3], data[:,4]
    v_bul = data[:,5] if data.shape[1] > 5 else np.zeros_like(r)
    mask = (v_obs > 0) & (r > 0)
    if mask.sum() < 4:
        return None
    return dict(r=r[mask], v_obs=v_obs[mask], v_err=v_err[mask],
                v_gas=v_gas[mask], v_disk=v_disk[mask], v_bul=v_bul[mask])

def tanh_T(r, r_s):
    return 0.5 * (1.0 + np.tanh(W_FIX * (r - r_s) / r_s))

def v2_baryons(v_gas, v_disk, v_bul, ud):
    v2  = np.sign(v_gas) * v_gas**2
    v2 += ud * np.sign(v_disk) * v_disk**2
    v2 += UPSILON_B_FIX * np.sign(v_bul) * v_bul**2
    return v2

def model_1s(r, vf, rs, ud, vg, vd, vb):
    return np.sqrt(np.maximum(vf**2 * tanh_T(r, rs) + v2_baryons(vg, vd, vb, ud), 0))

def model_2s(r, vf, rs1, rs2, fe, ud, vg, vd, vb):
    T = fe * tanh_T(r, rs1) + (1-fe) * tanh_T(r, rs2)
    return np.sqrt(np.maximum(vf**2 * T + v2_baryons(vg, vd, vb, ud), 0))

def fit_one(d):
    r, vo, ve = d['r'], d['v_obs'], d['v_err']
    vg, vd, vb = d['v_gas'], d['v_disk'], d['v_bul']
    n = len(r)
    r_max, v_max = r.max(), vo.max()
    out = {}

    # 1-stage (multi-start)
    def w1(r, vf, rs, ud):
        return model_1s(r, vf, rs, ud, vg, vd, vb)
    best1_c2 = 1e12
    best1_po = best1_pc = None
    for vf0 in [v_max*0.7, v_max*1.0, v_max*1.3]:
        for rs0 in [r_max*0.1, r_max*0.3, r_max*0.6]:
            for ud0 in [0.3, 0.5, 0.8]:
                try:
                    po, pc = curve_fit(w1, r, vo, p0=[vf0, rs0, ud0],
                                       bounds=([5, 0.05, UPSILON_MIN], [500, r_max*3, 2.5]),
                                       sigma=ve, absolute_sigma=True, maxfev=20000)
                    vf = w1(r, *po)
                    c2 = np.sum(((vo-vf)/ve)**2)
                    if c2 < best1_c2:
                        best1_c2 = c2; best1_po = po; best1_pc = pc
                except: continue
    if best1_po is not None:
        vf1 = w1(r, *best1_po)
        out['1s'] = dict(popt=best1_po, chi2=best1_c2/max(n-3,1),
                         aic=best1_c2+2*3, v_fit=vf1, ok=True)
    else:
        out['1s'] = dict(ok=False, chi2=np.nan, aic=np.nan)

    # 2-stage (multi-start)
    def w2(r, vf, rs1, rs2, fe, ud):
        return model_2s(r, vf, rs1, rs2, fe, ud, vg, vd, vb)
    best2_c2 = 1e12
    best2_po = best2_pc = None
    for vf0 in [v_max*0.8, v_max*1.1]:
        for rs1_0 in [r_max*0.03, r_max*0.1, r_max*0.2]:
            for rs2_0 in [r_max*0.3, r_max*0.5, r_max*0.8]:
                for fe0 in [0.3, 0.5, 0.7]:
                    try:
                        po, pc = curve_fit(w2, r, vo,
                                           p0=[vf0, rs1_0, rs2_0, fe0, 0.5],
                                           bounds=([5,0.01,0.1,0.0,UPSILON_MIN],
                                                   [500,r_max*1.5,r_max*3,1.0,2.5]),
                                           sigma=ve, absolute_sigma=True, maxfev=20000)
                        if po[1] > po[2]:
                            po[1], po[2] = po[2], po[1]
                            po[3] = 1.0 - po[3]
                        vf = w2(r, *po)
                        c2 = np.sum(((vo-vf)/ve)**2)
                        if c2 < best2_c2:
                            best2_c2 = c2; best2_po = po; best2_pc = pc
                    except: continue
    if best2_po is not None:
        vf2 = w2(r, *best2_po)
        out['2s'] = dict(popt=best2_po, chi2=best2_c2/max(n-5,1),
                         aic=best2_c2+2*5, v_fit=vf2, ok=True)
    else:
        out['2s'] = dict(ok=False, chi2=np.nan, aic=np.nan)

    if out['1s'].get('ok') and out['2s'].get('ok'):
        out['daic'] = out['2s']['aic'] - out['1s']['aic']
        out['grade'] = '2stage' if out['daic'] < DAIC_THRESHOLD else '1stage'
    else:
        out['daic'] = np.nan
        out['grade'] = 'failed' if not out['1s'].get('ok') else '1stage'
    return out

# ============================================================
# Main
# ============================================================
files = sorted(DATA_DIR.glob('*_rotmod.dat'))
print(f"SPARC galaxies: {len(files)}")
print(f"Upsilon_d >= {UPSILON_MIN}, w = {W_FIX}, dAIC threshold = {DAIC_THRESHOLD}")
print()

results = {}
for i, fp in enumerate(files):
    gal = fp.stem.replace('_rotmod', '')
    d = load_sparc(fp)
    if d is None:
        continue
    res = fit_one(d)
    results[gal] = res
    c1 = res['1s'].get('chi2', np.nan)
    c2 = res['2s'].get('chi2', np.nan)
    da = res.get('daic', np.nan)
    g  = res.get('grade', '?')
    if i < 10 or i % 20 == 0:
        print(f"[{i+1:3d}/{len(files)}] {gal:<18s} c1={c1:6.2f} c2={c2:6.2f} dAIC={da:7.1f} -> {g}")

# ============================================================
# Summary
# ============================================================
gc = {'2stage': 0, '1stage': 0, 'failed': 0}
for g, r in results.items():
    gc[r.get('grade', 'failed')] += 1
total = sum(gc.values())

print(f"\n{'='*80}")
print(f"SUMMARY (N={total})")
print(f"{'='*80}")
print(f"  2-stage wins (dAIC<{DAIC_THRESHOLD}): {gc['2stage']:3d} ({gc['2stage']/total*100:.0f}%)")
print(f"  1-stage sufficient:                 {gc['1stage']:3d} ({gc['1stage']/total*100:.0f}%)")
print(f"  Failed:                             {gc['failed']:3d}")

# chi2 statistics
c1_all = [r['1s']['chi2'] for r in results.values() if r['1s'].get('ok')]
c2_all = [r['2s']['chi2'] for r in results.values() if r['2s'].get('ok')]
print(f"\n  chi2/dof (1-stage): median={np.median(c1_all):.3f}, mean={np.mean(c1_all):.3f}")
print(f"  chi2/dof (2-stage): median={np.median(c2_all):.3f}, mean={np.mean(c2_all):.3f}")

# 2-stage galaxy statistics
rs1_2s, rs2_2s, fe_2s, ratio_2s = [], [], [], []
for g, r in results.items():
    if r.get('grade') != '2stage' or not r['2s'].get('ok'):
        continue
    p = r['2s']['popt']
    rs1_2s.append(p[1]); rs2_2s.append(p[2]); fe_2s.append(p[3])
    if p[1] > 0: ratio_2s.append(p[2]/p[1])

if rs1_2s:
    print(f"\n--- 2-stage galaxies (N={len(rs1_2s)}) ---")
    for nm, arr in [('r_s1 [kpc]', rs1_2s), ('r_s2 [kpc]', rs2_2s),
                    ('f_e', fe_2s), ('r_s2/r_s1', ratio_2s)]:
        a = np.array(arr)
        print(f"  {nm:<14s} median={np.median(a):.3f}  "
              f"[{np.percentile(a,16):.3f}, {np.percentile(a,84):.3f}]")

# Key galaxies
print(f"\n--- Key galaxies ---")
for gal in ['NGC3198','NGC2403','NGC6503','UGC02885','DDO154','NGC2841','NGC4214','NGC1705']:
    if gal not in results: continue
    r = results[gal]
    c1 = r['1s'].get('chi2', np.nan)
    c2 = r['2s'].get('chi2', np.nan)
    da = r.get('daic', np.nan)
    g  = r.get('grade', '?')
    if r['2s'].get('ok'):
        p = r['2s']['popt']
        print(f"  {gal:<12s} c1={c1:.2f} c2={c2:.2f} dAIC={da:.1f} "
              f"rs1={p[1]:.2f} rs2={p[2]:.2f} fe={p[3]:.2f} Ud={p[4]:.2f} -> {g}")
    else:
        print(f"  {gal:<12s} c1={c1:.2f} c2=FAIL -> {g}")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# All f_e distribution
fe_all = [r['2s']['popt'][3] for r in results.values() if r['2s'].get('ok')]
axes[0,0].hist(fe_all, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0,0].set_xlabel('f_e (elastic fraction)')
axes[0,0].set_ylabel('N galaxies')
axes[0,0].set_title('f_e distribution (all)')

# r_s1 distribution (2-stage only)
if rs1_2s:
    axes[0,1].hist(rs1_2s, bins=20, color='green', alpha=0.7, label='r_s1')
    axes[0,1].hist(rs2_2s, bins=20, color='magenta', alpha=0.5, label='r_s2')
    axes[0,1].set_xlabel('[kpc]')
    axes[0,1].set_ylabel('N')
    axes[0,1].set_title('r_s1 & r_s2 (2-stage galaxies)')
    axes[0,1].legend()

# r_s1 vs r_s2
rs1_1s = [r['2s']['popt'][1] for r in results.values()
          if r['2s'].get('ok') and r.get('grade')=='1stage']
rs2_1s = [r['2s']['popt'][2] for r in results.values()
          if r['2s'].get('ok') and r.get('grade')=='1stage']
if rs1_2s:
    axes[0,2].scatter(rs1_1s, rs2_1s, c='gray', s=15, alpha=0.4, label='1-stage OK')
    axes[0,2].scatter(rs1_2s, rs2_2s, c='red', s=25, alpha=0.7, label='2-stage wins')
    lim = max(max(rs2_2s), max(rs2_1s) if rs2_1s else 1) * 1.05
    axes[0,2].plot([0,lim],[0,lim],'k--',lw=0.8)
    axes[0,2].set_xlabel('r_s1 [kpc]')
    axes[0,2].set_ylabel('r_s2 [kpc]')
    axes[0,2].set_title('r_s1 vs r_s2')
    axes[0,2].legend(fontsize=8)

# chi2 comparison
c1_ok = [r['1s']['chi2'] for r in results.values() if r['1s'].get('ok') and r['2s'].get('ok')]
c2_ok = [r['2s']['chi2'] for r in results.values() if r['1s'].get('ok') and r['2s'].get('ok')]
gr_ok = [r.get('grade','?') for r in results.values() if r['1s'].get('ok') and r['2s'].get('ok')]
c1a, c2a = np.array(c1_ok), np.array(c2_ok)
is2 = np.array([g=='2stage' for g in gr_ok])
mask_plot = (c1a < 30) & (c2a < 30)
axes[1,0].scatter(c1a[mask_plot & ~is2], c2a[mask_plot & ~is2], c='gray', s=15, alpha=0.4)
axes[1,0].scatter(c1a[mask_plot & is2], c2a[mask_plot & is2], c='red', s=25, alpha=0.7)
axes[1,0].plot([0,25],[0,25],'k--',lw=0.8)
axes[1,0].set_xlabel('chi2/dof (1-stage)')
axes[1,0].set_ylabel('chi2/dof (2-stage)')
axes[1,0].set_title('Fit quality comparison')
axes[1,0].set_xlim(0,25); axes[1,0].set_ylim(0,25)

# dAIC distribution
daic_all = [r.get('daic',np.nan) for r in results.values()]
daic_arr = np.array([x for x in daic_all if not np.isnan(x)])
axes[1,1].hist(np.clip(daic_arr, -100, 50), bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1,1].axvline(DAIC_THRESHOLD, color='red', ls='--', lw=1.5, label=f'threshold={DAIC_THRESHOLD}')
axes[1,1].axvline(0, color='k', ls='-', lw=0.8)
axes[1,1].set_xlabel('dAIC (2-stage - 1-stage)')
axes[1,1].set_ylabel('N')
axes[1,1].set_title('dAIC distribution')
axes[1,1].legend()

# r_s2/r_s1 ratio for 2-stage
if ratio_2s:
    axes[1,2].hist(ratio_2s, bins=20, color='purple', alpha=0.7)
    axes[1,2].axvline(np.median(ratio_2s), color='red', ls='--',
                      label=f'median={np.median(ratio_2s):.1f}')
    axes[1,2].set_xlabel('r_s2 / r_s1')
    axes[1,2].set_ylabel('N')
    axes[1,2].set_title('Scale ratio (2-stage galaxies)')
    axes[1,2].legend()

plt.suptitle(f'SPARC Full Sample: 1-stage vs 2-stage tanh (N={total}, Ud>={UPSILON_MIN})',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_DIR / 'sparc_distributions.png', dpi=150, bbox_inches='tight')
print(f"\n-> sparc_distributions.png saved")

# CSV
with open(OUT_DIR / 'sparc_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['galaxy','chi2_1s','chi2_2s','daic','rs1','rs2','fe','ud','vflat','grade'])
    for gal, r in sorted(results.items()):
        c1 = r['1s'].get('chi2', '')
        c2 = r['2s'].get('chi2', '')
        da = r.get('daic', '')
        g  = r.get('grade', 'failed')
        if r['2s'].get('ok'):
            p = r['2s']['popt']
            w.writerow([gal, f'{c1:.4f}', f'{c2:.4f}', f'{da:.2f}',
                        f'{p[1]:.4f}', f'{p[2]:.4f}', f'{p[3]:.4f}', f'{p[4]:.4f}',
                        f'{p[0]:.1f}', g])
        else:
            w.writerow([gal, f'{c1:.4f}' if isinstance(c1,float) else '', '', '', '', '', '', '', '', g])
print(f"-> sparc_results.csv saved")
