# -*- coding: utf-8 -*-
"""
N-1 Layer 2b: Circularity check + Yd-hR hidden correlation.
"""
import os, sys, glob, warnings
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _fp in ['/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
            '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
            r'C:\Windows\Fonts\msgothic.ttc']:
    try: _fm.fontManager.addfont(_fp)
    except: pass
for fontname in ['IPAGothic', 'MS Gothic', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = fontname
        break
    except: continue
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

a0     = 1.2e-10
G_SI   = 6.674e-11
kpc_m  = 3.0857e19

BASE   = os.path.dirname(os.path.abspath(__file__))
ROTMOD = os.path.join(BASE, 'Rotmod_LTG')
PHASE1 = os.path.join(BASE, 'phase1', 'sparc_results.csv')
TA3    = os.path.join(BASE, 'TA3_gc_independent.csv')

for p, label in [(ROTMOD,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),
                 (TA3,'TA3_gc_independent.csv')]:
    if not os.path.exists(p):
        print(f'[ERROR] {label} not found: {p}'); sys.exit(1)


def load_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip()
    sep = ',' if ',' in header else None
    data = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        cols = [c.strip() for c in f.readline().strip().split(sep)]
        rows = []
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append([p.strip() for p in line.split(sep)])
    for i, col in enumerate(cols):
        vals = []
        for row in rows:
            if i < len(row):
                try:    vals.append(float(row[i]))
                except: vals.append(row[i])
            else: vals.append(np.nan)
        data[col] = vals
    return data

def find_name_col(data):
    for c in ['galaxy','Galaxy','name','Name','GALAXY']:
        if c in data: return c
    for k,v in data.items():
        if isinstance(v[0], str): return k
    return list(data.keys())[0]

def get_key(info, candidates, default=None):
    for c in candidates:
        if c in info:
            try: return float(info[c])
            except: return info[c]
    return default

def load_rotmod(filepath):
    cols = [[] for _ in range(8)]
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 6: continue
            try:
                for j in range(min(len(parts),8)):
                    cols[j].append(float(parts[j]))
                for j in range(len(parts),8):
                    cols[j].append(0.0)
            except ValueError: continue
    return tuple(np.array(c) for c in cols)


print('[1] Loading data...')
phase1 = load_csv(PHASE1)
ta3    = load_csv(TA3)
p1_nc  = find_name_col(phase1)
ta3_nc = find_name_col(ta3)

galaxy_info = {}
for i, name in enumerate(phase1[p1_nc]):
    name = str(name).strip()
    info = {}
    for k in phase1:
        if k == p1_nc: continue
        try:    info[k] = float(phase1[k][i])
        except: info[k] = phase1[k][i]
    galaxy_info[name] = info

for i, name in enumerate(ta3[ta3_nc]):
    name = str(name).strip()
    if name in galaxy_info:
        for k in ta3:
            if k == ta3_nc: continue
            try:    galaxy_info[name][k] = float(ta3[k][i])
            except: galaxy_info[name][k] = ta3[k][i]


print('[2] Processing galaxies...')
results = []
rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, '*.dat')))

for fpath in rotmod_files:
    gname = os.path.splitext(os.path.basename(fpath))[0].replace('_rotmod','').strip()
    info = None
    for key in [gname, gname.upper(), gname.lower()]:
        if key in galaxy_info:
            info = galaxy_info[key]; break
    if info is None: continue

    ud    = get_key(info, ['upsilon_d','Upsilon_d','Ud','ud','Yd'])
    gc_a0 = get_key(info, ['gc_over_a0','gc/a0','gc_ratio'])
    vflat = get_key(info, ['vflat','Vflat','v_flat'])

    if ud is None or gc_a0 is None or vflat is None: continue
    if np.isnan(ud) or np.isnan(gc_a0) or gc_a0<=0 or vflat<=0 or ud<=0: continue

    try:
        rad, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = load_rotmod(fpath)
    except: continue
    if len(rad) < 3: continue

    vds = np.sqrt(max(ud, 0.01)) * np.abs(vdisk)
    rpk_idx = np.argmax(vds)
    rpk = rad[rpk_idx]
    if rpk < 0.01 or rpk >= rad.max()*0.9: continue
    hR_kpc = rpk / 2.15
    hR_m   = hR_kpc * kpc_m

    r_m = rad * kpc_m
    vbar2 = np.abs(vgas**2 + ud * np.sign(vdisk)*vdisk**2
                   + 0.7 * np.sign(vbul)*vbul**2)
    g_N = np.zeros_like(r_m)
    mask_r = r_m > 0
    g_N[mask_r] = vbar2[mask_r] * 1e6 / r_m[mask_r]

    gc_si = gc_a0 * a0
    vflat_ms = vflat * 1e3
    GS0_proxy = vflat_ms**2 / hR_m

    n_out = min(3, len(rad))
    g_N_outer = np.mean(g_N[-n_out:])

    if g_N_outer > 0 and gc_si > 0:
        gf = (g_N_outer + np.sqrt(g_N_outer**2 + 4*gc_si*g_N_outer)) / 2
        gm = np.sqrt(gc_si * g_N_outer)
        corr_gc = gf / gm
    else:
        corr_gc = np.nan

    if g_N_outer > 0:
        gf_a0 = (g_N_outer + np.sqrt(g_N_outer**2 + 4*a0*g_N_outer)) / 2
        gm_a0 = np.sqrt(a0 * g_N_outer)
        corr_a0 = gf_a0 / gm_a0
    else:
        corr_a0 = np.nan

    results.append({
        'galaxy':     gname,
        'gc_a0':      gc_a0,
        'log_gc':     np.log10(gc_a0),
        'vflat':      vflat,
        'Yd':         ud,
        'log_Yd':     np.log10(ud),
        'hR_kpc':     hR_kpc,
        'log_hR':     np.log10(hR_kpc),
        'log_vflat':  np.log10(vflat),
        'GS0_proxy':  GS0_proxy,
        'log_GS0':    np.log10(GS0_proxy/a0),
        'g_N_outer':  g_N_outer,
        'corr_gc':    corr_gc,
        'log_corr_gc': np.log10(corr_gc) if np.isfinite(corr_gc) and corr_gc>0 else np.nan,
        'corr_a0':    corr_a0,
        'log_corr_a0': np.log10(corr_a0) if np.isfinite(corr_a0) and corr_a0>0 else np.nan,
    })

N = len(results)
print(f'  Processed: {N} galaxies')
if N < 10:
    print('[ERROR] Too few.'); sys.exit(1)

log_gc      = np.array([r['log_gc'] for r in results])
log_hR      = np.array([r['log_hR'] for r in results])
log_vflat   = np.array([r['log_vflat'] for r in results])
log_GS0     = np.array([r['log_GS0'] for r in results])
log_Yd      = np.array([r['log_Yd'] for r in results])
Yd_arr      = np.array([r['Yd'] for r in results])
log_corr_gc = np.array([r['log_corr_gc'] for r in results])
log_corr_a0 = np.array([r['log_corr_a0'] for r in results])

print('\n' + '='*70)
print('PART 1: Yd vs hR hidden correlation')
print('='*70)

mask1 = np.isfinite(log_Yd) & np.isfinite(log_hR)
rho_Yd_hR, p_Yd_hR = stats.spearmanr(log_Yd[mask1], log_hR[mask1])
sl_yh, int_yh, r_yh, _, se_yh = stats.linregress(log_hR[mask1], log_Yd[mask1])
print(f'\n  [1a] Yd vs hR:')
print(f'    Spearman rho = {rho_Yd_hR:.4f}, p = {p_Yd_hR:.2e}')
print(f'    Linear: log(Yd) = {sl_yh:.3f}*log(hR) + {int_yh:.3f}, R^2={r_yh**2:.3f}')

mask1v = np.isfinite(log_Yd) & np.isfinite(log_vflat)
rho_Yd_vf, p_Yd_vf = stats.spearmanr(log_Yd[mask1v], log_vflat[mask1v])
print(f'\n  [1b] Yd vs vflat: rho = {rho_Yd_vf:.4f}, p = {p_Yd_vf:.2e}')

mask1g = np.isfinite(log_Yd) & np.isfinite(log_gc)
rho_Yd_gc, p_Yd_gc = stats.spearmanr(log_Yd[mask1g], log_gc[mask1g])
print(f'\n  [1c] Yd vs gc:    rho = {rho_Yd_gc:.4f}, p = {p_Yd_gc:.2e}')

mask_p1 = np.isfinite(log_gc) & np.isfinite(log_hR) & np.isfinite(log_vflat) & np.isfinite(log_Yd)
X_ctrl = np.column_stack([log_vflat[mask_p1], log_Yd[mask_p1], np.ones(mask_p1.sum())])
y_gc = log_gc[mask_p1]
y_hR = log_hR[mask_p1]

b_gc = np.linalg.lstsq(X_ctrl, y_gc, rcond=None)[0]
res_gc = y_gc - X_ctrl @ b_gc
b_hR = np.linalg.lstsq(X_ctrl, y_hR, rcond=None)[0]
res_hR = y_hR - X_ctrl @ b_hR
rho_hR_Yd, p_hR_Yd = stats.spearmanr(res_gc, res_hR)

X_ctrl0 = np.column_stack([log_vflat[mask_p1], np.ones(mask_p1.sum())])
b_gc0 = np.linalg.lstsq(X_ctrl0, y_gc, rcond=None)[0]
res_gc0 = y_gc - X_ctrl0 @ b_gc0
b_hR0 = np.linalg.lstsq(X_ctrl0, y_hR, rcond=None)[0]
res_hR0 = y_hR - X_ctrl0 @ b_hR0
rho_hR_orig, p_hR_orig = stats.spearmanr(res_gc0, res_hR0)

print(f'\n  [1d] hR partial correlations:')
print(f'    gc vs hR | vflat:          rho = {rho_hR_orig:.4f}, p = {p_hR_orig:.4e}')
print(f'    gc vs hR | (vflat, Yd):    rho = {rho_hR_Yd:.4f}, p = {p_hR_Yd:.4e}')
red_yd = (1 - abs(rho_hR_Yd)/abs(rho_hR_orig))*100 if abs(rho_hR_orig)>0 else 0
print(f'    Reduction by adding Yd:    {red_yd:.1f}%')

X_m3 = np.column_stack([log_vflat[mask_p1], log_hR[mask_p1], log_Yd[mask_p1],
                         np.ones(mask_p1.sum())])
b3 = np.linalg.lstsq(X_m3, y_gc, rcond=None)[0]
ss3 = np.sum((y_gc - X_m3 @ b3)**2)
ss_tot = np.sum((y_gc - y_gc.mean())**2)
r2_3 = 1 - ss3/ss_tot
mse3 = ss3 / (len(y_gc)-4)
se3 = np.sqrt(np.diag(mse3 * np.linalg.inv(X_m3.T @ X_m3)))

print(f'\n  [1e] Multivariate: vflat + hR + Yd')
print(f'    log(gc) = {b3[0]:.4f}*log(vflat) + {b3[1]:.4f}*log(hR) '
      f'+ {b3[2]:.4f}*log(Yd) + {b3[3]:.4f}')
print(f'    SE:        {se3[0]:.4f}                 {se3[1]:.4f}                '
      f'{se3[2]:.4f}')
print(f'    R^2 = {r2_3:.4f}')
t_yd = abs(b3[2]) / se3[2]
p_yd_coeff = 2*(1-stats.t.cdf(t_yd, df=len(y_gc)-4))
print(f'    p(Yd coeff = 0) = {p_yd_coeff:.4f}')

X_m2 = np.column_stack([log_vflat[mask_p1], log_hR[mask_p1], np.ones(mask_p1.sum())])
b2 = np.linalg.lstsq(X_m2, y_gc, rcond=None)[0]
ss2 = np.sum((y_gc - X_m2 @ b2)**2)
r2_2 = 1 - ss2/ss_tot
print(f'    Compare: vflat+hR only R^2 = {r2_2:.4f}')
print(f'    Delta R^2 from Yd: +{r2_3 - r2_2:.4f}')

print('\n' + '='*70)
print('PART 2: gc-free correction (gc -> a0)')
print('='*70)

valid_c = np.isfinite(log_corr_gc) & np.isfinite(log_corr_a0)
arr_corr_gc = np.array([r['corr_gc'] for r in results])
arr_corr_a0 = np.array([r['corr_a0'] for r in results])
print(f'\n  [2a] Correction distributions (N={np.sum(valid_c)}):')
print(f'    With gc:  median = {np.nanmedian(arr_corr_gc[valid_c]):.4f}')
print(f'    With a0:  median = {np.nanmedian(arr_corr_a0[valid_c]):.4f}')

rho_cc, p_cc = stats.spearmanr(log_corr_gc[valid_c], log_corr_a0[valid_c])
print(f'\n  [2b] corr_gc vs corr_a0: rho = {rho_cc:.4f}, p = {p_cc:.2e}')

mask_cg = np.isfinite(log_corr_gc) & np.isfinite(log_gc)
rho_cg_gc, p_cg_gc = stats.spearmanr(log_corr_gc[mask_cg], log_gc[mask_cg])
mask_ca = np.isfinite(log_corr_a0) & np.isfinite(log_gc)
rho_ca_gc, p_ca_gc = stats.spearmanr(log_corr_a0[mask_ca], log_gc[mask_ca])
print(f'\n  [2c] Circularity check:')
print(f'    corr_gc vs gc: rho = {rho_cg_gc:.4f}, p = {p_cg_gc:.2e}  (circular)')
print(f'    corr_a0 vs gc: rho = {rho_ca_gc:.4f}, p = {p_ca_gc:.2e}  (gc-free)')

mask_2m = (np.isfinite(log_gc) & np.isfinite(log_vflat) &
           np.isfinite(log_hR) & np.isfinite(log_corr_a0))

print(f'\n  [2d] Multivariate with gc-free correction (N={np.sum(mask_2m)})')

y2 = log_gc[mask_2m]
ss_tot2 = np.sum((y2 - y2.mean())**2)

X_nc = np.column_stack([log_vflat[mask_2m], log_hR[mask_2m], np.ones(mask_2m.sum())])
b_nc = np.linalg.lstsq(X_nc, y2, rcond=None)[0]
r2_nc = 1 - np.sum((y2 - X_nc @ b_nc)**2)/ss_tot2
mse_nc = np.sum((y2 - X_nc @ b_nc)**2) / (len(y2)-3)
se_nc = np.sqrt(np.diag(mse_nc * np.linalg.inv(X_nc.T @ X_nc)))

print(f'    G1: vflat+hR: R^2 = {r2_nc:.4f}')
print(f'        vflat^{b_nc[0]:.3f}+/-{se_nc[0]:.3f}, hR^{b_nc[1]:.3f}+/-{se_nc[1]:.3f}')

X_cg = np.column_stack([log_vflat[mask_2m], log_hR[mask_2m],
                         log_corr_gc[mask_2m], np.ones(mask_2m.sum())])
b_cg = np.linalg.lstsq(X_cg, y2, rcond=None)[0]
r2_cg = 1 - np.sum((y2 - X_cg @ b_cg)**2)/ss_tot2
mse_cg = np.sum((y2 - X_cg @ b_cg)**2) / (len(y2)-4)
se_cg = np.sqrt(np.diag(mse_cg * np.linalg.inv(X_cg.T @ X_cg)))
t_cg = abs(b_cg[2]) / se_cg[2]
p_cg = 2*(1-stats.t.cdf(t_cg, df=len(y2)-4))

print(f'\n    G2: vflat+hR+corr_gc (circular): R^2 = {r2_cg:.4f}')
print(f'        vflat^{b_cg[0]:.3f}, hR^{b_cg[1]:.3f}, corr_gc^{b_cg[2]:.3f}+/-{se_cg[2]:.3f}')
print(f'        p(corr_gc=0) = {p_cg:.4f}')

X_ca = np.column_stack([log_vflat[mask_2m], log_hR[mask_2m],
                         log_corr_a0[mask_2m], np.ones(mask_2m.sum())])
b_ca = np.linalg.lstsq(X_ca, y2, rcond=None)[0]
r2_ca = 1 - np.sum((y2 - X_ca @ b_ca)**2)/ss_tot2
mse_ca = np.sum((y2 - X_ca @ b_ca)**2) / (len(y2)-4)
se_ca = np.sqrt(np.diag(mse_ca * np.linalg.inv(X_ca.T @ X_ca)))
t_ca = abs(b_ca[2]) / se_ca[2]
p_ca = 2*(1-stats.t.cdf(t_ca, df=len(y2)-4))

print(f'\n    G3: vflat+hR+corr_a0 (gc-free): R^2 = {r2_ca:.4f}')
print(f'        vflat^{b_ca[0]:.3f}+/-{se_ca[0]:.3f}, hR^{b_ca[1]:.3f}+/-{se_ca[1]:.3f}, '
      f'corr_a0^{b_ca[2]:.3f}+/-{se_ca[2]:.3f}')
print(f'        p(corr_a0=0) = {p_ca:.4f}')

X_ctrl2 = np.column_stack([log_vflat[mask_2m], log_corr_a0[mask_2m],
                            np.ones(mask_2m.sum())])
b_gc2 = np.linalg.lstsq(X_ctrl2, y2, rcond=None)[0]
res_gc2 = y2 - X_ctrl2 @ b_gc2
y_hR2 = log_hR[mask_2m]
b_hR2 = np.linalg.lstsq(X_ctrl2, y_hR2, rcond=None)[0]
res_hR2 = y_hR2 - X_ctrl2 @ b_hR2
rho_hR_a0, p_hR_a0 = stats.spearmanr(res_gc2, res_hR2)

X_ctrl0b = np.column_stack([log_vflat[mask_2m], np.ones(mask_2m.sum())])
b_gc0b = np.linalg.lstsq(X_ctrl0b, y2, rcond=None)[0]
res_gc0b = y2 - X_ctrl0b @ b_gc0b
b_hR0b = np.linalg.lstsq(X_ctrl0b, y_hR2, rcond=None)[0]
res_hR0b = y_hR2 - X_ctrl0b @ b_hR0b
rho_hR_0b, p_hR_0b = stats.spearmanr(res_gc0b, res_hR0b)

print(f'\n  [2e] hR partial correlations:')
print(f'    gc vs hR | vflat:                rho = {rho_hR_0b:.4f}')
print(f'    gc vs hR | (vflat, corr_a0):     rho = {rho_hR_a0:.4f}, p = {p_hR_a0:.4e}')
red_a0 = (1 - abs(rho_hR_a0)/abs(rho_hR_0b))*100 if abs(rho_hR_0b)>0 else 0
print(f'    Reduction (gc-free correction):  {red_a0:.1f}%')

print('\n' + '='*70)
print('PART 3: Full model comparison')
print('='*70)

mask_all = (np.isfinite(log_gc) & np.isfinite(log_vflat) & np.isfinite(log_hR)
            & np.isfinite(log_Yd) & np.isfinite(log_corr_a0))

y_all = log_gc[mask_all]
ss_all = np.sum((y_all - y_all.mean())**2)

models = {}

X_1 = np.column_stack([log_vflat[mask_all], log_hR[mask_all], np.ones(mask_all.sum())])
b_1 = np.linalg.lstsq(X_1, y_all, rcond=None)[0]
r2_m1 = 1 - np.sum((y_all - X_1 @ b_1)**2)/ss_all
models['vflat+hR'] = {'r2': r2_m1, 'k': 2, 'beta': b_1}

X_2 = np.column_stack([log_vflat[mask_all], log_hR[mask_all], log_Yd[mask_all],
                        np.ones(mask_all.sum())])
b_2_m = np.linalg.lstsq(X_2, y_all, rcond=None)[0]
r2_m2 = 1 - np.sum((y_all - X_2 @ b_2_m)**2)/ss_all
mse_m2 = np.sum((y_all - X_2 @ b_2_m)**2) / (len(y_all)-4)
se_m2 = np.sqrt(np.diag(mse_m2 * np.linalg.inv(X_2.T @ X_2)))
models['vflat+hR+Yd'] = {'r2': r2_m2, 'k': 3, 'beta': b_2_m, 'se': se_m2}

X_3 = np.column_stack([log_vflat[mask_all], log_hR[mask_all], log_corr_a0[mask_all],
                        np.ones(mask_all.sum())])
b_3_m = np.linalg.lstsq(X_3, y_all, rcond=None)[0]
r2_m3 = 1 - np.sum((y_all - X_3 @ b_3_m)**2)/ss_all
mse_m3 = np.sum((y_all - X_3 @ b_3_m)**2) / (len(y_all)-4)
se_m3 = np.sqrt(np.diag(mse_m3 * np.linalg.inv(X_3.T @ X_3)))
models['vflat+hR+corr_a0'] = {'r2': r2_m3, 'k': 3, 'beta': b_3_m, 'se': se_m3}

X_4 = np.column_stack([log_vflat[mask_all], log_hR[mask_all], log_Yd[mask_all],
                        log_corr_a0[mask_all], np.ones(mask_all.sum())])
b_4 = np.linalg.lstsq(X_4, y_all, rcond=None)[0]
r2_m4 = 1 - np.sum((y_all - X_4 @ b_4)**2)/ss_all
mse_m4 = np.sum((y_all - X_4 @ b_4)**2) / (len(y_all)-5)
se_m4 = np.sqrt(np.diag(mse_m4 * np.linalg.inv(X_4.T @ X_4)))
models['vflat+hR+Yd+corr_a0'] = {'r2': r2_m4, 'k': 4, 'beta': b_4, 'se': se_m4}

n_all = mask_all.sum()
print(f'\n  N = {n_all}')
print(f'\n  {"Model":<28s} | {"R^2":>6s} | {"AIC":>8s} | k')
print('  ' + '-'*55)
for name, m in models.items():
    k = m['k']
    r2 = m['r2']
    ss_r = (1-r2) * ss_all
    aic = n_all * np.log(ss_r/n_all) + 2*(k+1)
    m['aic'] = aic
    print(f'  {name:<28s} | {r2:>6.4f} | {aic:>8.1f} | {k}')

aic_base = models['vflat+hR']['aic']
print(f'\n  dAIC vs baseline (vflat+hR):')
for name, m in models.items():
    print(f'    {name:<28s}: dAIC = {m["aic"]-aic_base:>+7.1f}')

print(f'\n  Full model (M4) coefficients:')
print(f'    vflat:   {b_4[0]:.4f} +/- {se_m4[0]:.4f}')
print(f'    hR:      {b_4[1]:.4f} +/- {se_m4[1]:.4f}')
print(f'    Yd:      {b_4[2]:.4f} +/- {se_m4[2]:.4f}')
print(f'    corr_a0: {b_4[3]:.4f} +/- {se_m4[3]:.4f}')

X_fin = np.column_stack([log_vflat[mask_all], log_Yd[mask_all],
                          log_corr_a0[mask_all], np.ones(mask_all.sum())])
b_fin_gc = np.linalg.lstsq(X_fin, y_all, rcond=None)[0]
res_fin_gc = y_all - X_fin @ b_fin_gc
y_hR_all = log_hR[mask_all]
b_fin_hR = np.linalg.lstsq(X_fin, y_hR_all, rcond=None)[0]
res_fin_hR = y_hR_all - X_fin @ b_fin_hR
rho_fin, p_fin = stats.spearmanr(res_fin_gc, res_fin_hR)

print(f'\n  [Final] gc vs hR | (vflat, Yd, corr_a0):')
print(f'    rho = {rho_fin:.4f}, p = {p_fin:.4e}')
print(f'    Original (| vflat only): rho = {rho_hR_orig:.4f}')
total_red = (1 - abs(rho_fin)/abs(rho_hR_orig))*100 if abs(rho_hR_orig)>0 else 0
print(f'    Total reduction: {total_red:.1f}%')

print('\n[3] Generating figures...')
fig, axes = plt.subplots(2, 3, figsize=(17, 11))

ax = axes[0, 0]
m = np.isfinite(log_Yd) & np.isfinite(log_hR)
ax.scatter(log_hR[m], log_Yd[m], s=12, alpha=0.5, c='steelblue', edgecolors='none')
xf = np.linspace(log_hR[m].min(), log_hR[m].max(), 100)
ax.plot(xf, sl_yh*xf + int_yh, 'r-', lw=2, label=f'rho={rho_Yd_hR:.3f}')
ax.set_xlabel('log(hR / kpc)')
ax.set_ylabel('log(Yd)')
ax.set_title('1: Yd vs hR')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
vc = np.isfinite(log_corr_gc) & np.isfinite(log_corr_a0)
ax.scatter(log_corr_a0[vc], log_corr_gc[vc], s=12, alpha=0.5, c='darkorange', edgecolors='none')
lim = [min(log_corr_a0[vc].min(), log_corr_gc[vc].min()),
       max(log_corr_a0[vc].max(), log_corr_gc[vc].max())]
ax.plot(lim, lim, 'k--', lw=1, label='1:1')
ax.set_xlabel('log(corr_a0)')
ax.set_ylabel('log(corr_gc)')
ax.set_title(f'2: Two corrections (rho={rho_cc:.3f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(log_corr_a0[mask_ca], log_gc[mask_ca], s=12, alpha=0.5, c='green', edgecolors='none')
ax.set_xlabel('log(corr_a0)')
ax.set_ylabel('log(gc/a0)')
ax.set_title(f'3: corr_a0 vs gc (rho={rho_ca_gc:.3f})')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
names = list(models.keys())
r2s = [models[n]['r2'] for n in names]
cols = ['steelblue', 'darkorange', 'green', 'red']
ax.bar(range(len(names)), r2s, color=cols[:len(names)], alpha=0.7,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('+','+\n') for n in names], fontsize=7)
ax.set_ylabel('R^2')
ax.set_title('4: Model R^2')
for i,v in enumerate(r2s):
    ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
stages = ['| vflat', '| vflat,Yd', '| vflat,corr_a0', '| vflat,Yd,corr_a0']
rhos = [rho_hR_orig, rho_hR_Yd, rho_hR_a0, rho_fin]
barcolors = ['steelblue', 'darkorange', 'green', 'red']
ax.bar(range(len(stages)), [abs(r) for r in rhos], color=barcolors, alpha=0.7,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=7, rotation=15)
ax.set_ylabel('|rho| (hR partial)')
ax.set_title('5: hR partial reduction')
for i,v in enumerate(rhos):
    ax.text(i, abs(v)+0.01, f'{v:.3f}', ha='center', fontsize=8)
ax.axhline(0.1, color='grey', ls=':', label='negligible')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 2]
resid_full = y_all - X_4 @ b_4
ax.hist(resid_full, bins=25, color='purple', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', ls='--')
ax.set_xlabel('Residual [dex]')
ax.set_ylabel('Count')
ax.set_title(f'6: Full model residual (sigma={np.std(resid_full):.3f})')

fig.suptitle(f'N-1 Layer 2b: Circularity + Yd check (N={N})', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_N1_layer2b.png'), dpi=150)
print('  -> fig_N1_layer2b.png')

outcsv = os.path.join(BASE, 'N1_layer2b_results.csv')
cols_out = ['galaxy','gc_a0','log_gc','vflat','Yd','log_Yd','hR_kpc','log_hR',
            'log_vflat','log_GS0','corr_gc','log_corr_gc','corr_a0','log_corr_a0']
with open(outcsv, 'w', encoding='utf-8') as f:
    f.write(','.join(cols_out)+'\n')
    for r in results:
        f.write(','.join(str(r.get(c,'')) for c in cols_out)+'\n')
print(f'  -> {outcsv}')

print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)

print(f'\n  PART 1: Yd-hR correlation')
if abs(rho_Yd_hR) > 0.3 and p_Yd_hR < 0.01:
    print(f'    >>> Yd and hR ARE correlated (rho={rho_Yd_hR:.3f})')
else:
    print(f'    >>> Yd and hR are NOT strongly correlated (rho={rho_Yd_hR:.3f})')

print(f'\n  PART 2: Circularity')
print(f'    corr_gc vs gc: rho = {rho_cg_gc:.3f} (circular)')
print(f'    corr_a0 vs gc: rho = {rho_ca_gc:.3f} (gc-free)')
if abs(rho_ca_gc) < abs(rho_cg_gc) * 0.5:
    print(f'    >>> Layer 2 R^2 boost was LARGELY circular')
elif abs(rho_ca_gc) < abs(rho_cg_gc) * 0.8:
    print(f'    >>> Layer 2 R^2 boost was PARTIALLY circular')
else:
    print(f'    >>> Layer 2 R^2 boost was GENUINE')

print(f'\n  PART 3: hR partial after all controls')
print(f'    Original:  rho = {rho_hR_orig:.4f}')
print(f'    Final:     rho = {rho_fin:.4f}')
print(f'    Reduction: {total_red:.1f}%')
if abs(rho_fin) < 0.1:
    print(f'    >>> hR effect FULLY EXPLAINED')
elif abs(rho_fin) < 0.2:
    print(f'    >>> hR effect MOSTLY explained ({total_red:.0f}%)')
else:
    print(f'    >>> hR effect PARTIALLY explained ({total_red:.0f}%)')

print('\n[DONE]')
