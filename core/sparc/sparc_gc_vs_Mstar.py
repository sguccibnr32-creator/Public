# -*- coding: utf-8 -*-
"""
V-1b: gc vs M* direct regression.
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
Msun   = 1.989e30
pc_m   = 3.0857e16

BASE   = os.path.dirname(os.path.abspath(__file__))
ROTMOD = os.path.join(BASE, 'Rotmod_LTG')
PHASE1 = os.path.join(BASE, 'phase1', 'sparc_results.csv')
TA3    = os.path.join(BASE, 'TA3_gc_independent.csv')

for p, label in [(ROTMOD,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
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
    if np.isnan(ud) or np.isnan(gc_a0) or gc_a0<=0 or vflat<=0: continue

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

    vflat_ms = vflat * 1e3
    GS0_proxy = vflat_ms**2 / hR_m

    gc_si = gc_a0 * a0
    M_bar_btfr = vflat_ms**4 / (gc_si * G_SI)
    M_bar_btfr_sun = M_bar_btfr / Msun

    vdisk_peak_ms = np.max(np.sqrt(max(ud,0.01)) * np.abs(vdisk)) * 1e3
    M_disk_direct = 2.0 * vdisk_peak_ms**2 * hR_m / (0.56 * G_SI)
    M_disk_direct_sun = M_disk_direct / Msun

    vgas_peak_ms = np.max(np.abs(vgas)) * 1e3
    M_gas_direct = 2.0 * vgas_peak_ms**2 * hR_m / (0.56 * G_SI) if vgas_peak_ms > 0 else 0
    M_gas_direct_sun = M_gas_direct / Msun

    M_bar_direct_sun = M_disk_direct_sun + M_gas_direct_sun

    Sigma_bar = M_disk_direct / (2 * np.pi * hR_m**2)
    G_Sigma_bar = G_SI * Sigma_bar

    results.append({
        'galaxy':           gname,
        'gc_over_a0':       gc_a0,
        'log_gc':           np.log10(gc_a0),
        'vflat':            vflat,
        'Yd':               ud,
        'hR_kpc':           hR_kpc,
        'GS0_proxy':        GS0_proxy,
        'log_GS0':          np.log10(GS0_proxy / a0),
        'log_Mbar_btfr':    np.log10(M_bar_btfr_sun) if M_bar_btfr_sun>0 else np.nan,
        'M_disk_sun':       M_disk_direct_sun,
        'M_gas_sun':        M_gas_direct_sun,
        'M_bar_direct_sun': M_bar_direct_sun,
        'log_Mbar_direct':  np.log10(M_bar_direct_sun) if M_bar_direct_sun>0 else np.nan,
        'log_Mdisk':        np.log10(M_disk_direct_sun) if M_disk_direct_sun>0 else np.nan,
        'G_Sigma_bar':      G_Sigma_bar,
        'log_GSbar':        np.log10(G_Sigma_bar/a0) if G_Sigma_bar>0 else np.nan,
        'log_vflat':        np.log10(vflat),
        'log_hR':           np.log10(hR_kpc),
        'vdisk_peak':       vdisk_peak_ms/1e3,
    })

N = len(results)
print(f'  Processed: {N} galaxies')
if N < 10:
    print('[ERROR] Too few galaxies.'); sys.exit(1)

log_gc    = np.array([r['log_gc'] for r in results])
log_GS0   = np.array([r['log_GS0'] for r in results])
log_Mbar  = np.array([r['log_Mbar_direct'] for r in results])
log_Mdisk = np.array([r['log_Mdisk'] for r in results])
log_GSbar = np.array([r['log_GSbar'] for r in results])
log_vflat = np.array([r['log_vflat'] for r in results])
log_hR    = np.array([r['log_hR'] for r in results])
log_Mbtfr = np.array([r['log_Mbar_btfr'] for r in results])


def regtest(x, y, label, expected_slopes=None):
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 10:
        print(f'  [{label}] N={n} too few'); return None

    slope, intercept, r_val, p_val, se = stats.linregress(xv, yv)
    rho_s, p_s = stats.spearmanr(xv, yv)

    print(f'\n  [{label}]  N={n}')
    print(f'    slope    = {slope:.4f} +/- {se:.4f}')
    print(f'    R^2      = {r_val**2:.4f}')
    print(f'    Spearman = {rho_s:.4f}  (p={p_s:.2e})')

    if expected_slopes:
        for es in expected_slopes:
            t = abs(slope - es) / se
            p = 2*(1-stats.t.cdf(t, df=n-2))
            v = 'PASS' if p>0.05 else 'MARGINAL' if p>0.01 else 'FAIL'
            print(f'    p(slope={es:.4f}) = {p:.4f}  [{v}]')

    return {'slope':slope, 'se':se, 'intercept':intercept,
            'r2':r_val**2, 'rho':rho_s, 'p_s':p_s, 'n':n,
            'xv':xv, 'yv':yv}


print('\n' + '='*70)
print('V-1b: gc vs M* direct regression')
print('='*70)

print('\n--- A: Baseline proxy (vflat^2/hR) ---')
rA = regtest(log_GS0, log_gc, 'proxy vflat^2/hR', [0.5])

print('\n--- B: gc vs M_bar_direct (gc-independent, BT) ---')
rB = regtest(log_Mbar, log_gc, 'M_bar direct (BT)', [1/3, 1/4, 0.2, 0.125])

print('\n--- C: gc vs M_disk only (gc-independent) ---')
rC = regtest(log_Mdisk, log_gc, 'M_disk only', [1/3, 1/4, 0.2, 0.125])

print('\n--- D: gc vs Sigma_bar (gc-independent, V-1kai recheck) ---')
rD = regtest(log_GSbar, log_gc, 'G*Sigma_bar (BT)', [1/3, 0.5])

print('\n--- E: gc vs vflat only ---')
rE = regtest(log_vflat, log_gc, 'vflat only', [1.0, 2.0])

print('\n--- F: gc vs hR only ---')
rF = regtest(log_hR, log_gc, 'hR only', [-0.5, -1.0])

print('\n--- G: gc vs M_bar BTFR (circular, diagnostic) ---')
rG = regtest(log_Mbtfr, log_gc, 'M_bar BTFR (circular)', [1/3, 1/4])

print('\n--- H: Multivariate log(gc) = a*log(vflat) + b*log(hR) + c ---')
mask = np.isfinite(log_gc) & np.isfinite(log_vflat) & np.isfinite(log_hR)
X = np.column_stack([log_vflat[mask], log_hR[mask], np.ones(mask.sum())])
y = log_gc[mask]
beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ beta
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_multi = 1 - ss_res/ss_tot
n_multi = len(y)
mse = ss_res / (n_multi - 3)
se_beta = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))

print(f'  N = {n_multi}')
print(f'  log(gc) = {beta[0]:.4f}*log(vflat) + {beta[1]:.4f}*log(hR) + {beta[2]:.4f}')
print(f'  SE:       {se_beta[0]:.4f}              {se_beta[1]:.4f}              {se_beta[2]:.4f}')
print(f'  R^2 = {r2_multi:.4f}')
print(f'  vflat exponent: {beta[0]:.3f} +/- {se_beta[0]:.3f}')
print(f'  hR exponent:    {beta[1]:.3f} +/- {se_beta[1]:.3f}')

print(f'\n  Proxy decomposition test:')
print(f'    (vflat^2/hR)^0.5 predicts: vflat^1.0, hR^-0.5')
print(f'    Observed:                   vflat^{beta[0]:.3f}, hR^{beta[1]:.3f}')
t_vf = abs(beta[0] - 1.0) / se_beta[0]
p_vf = 2*(1-stats.t.cdf(t_vf, df=n_multi-3))
t_hr = abs(beta[1] - (-0.5)) / se_beta[1]
p_hr = 2*(1-stats.t.cdf(t_hr, df=n_multi-3))
print(f'    p(vflat=1.0) = {p_vf:.4f}')
print(f'    p(hR=-0.5)   = {p_hr:.4f}')

print('\n--- I: Partial correlations ---')
mask_p = np.isfinite(log_gc) & np.isfinite(log_vflat) & np.isfinite(log_hR)
gc_p = log_gc[mask_p]
vf_p = log_vflat[mask_p]
hr_p = log_hR[mask_p]

s_vf_gc, i_vf_gc, _, _, _ = stats.linregress(vf_p, gc_p)
res_gc = gc_p - (s_vf_gc * vf_p + i_vf_gc)
s_vf_hr, i_vf_hr, _, _, _ = stats.linregress(vf_p, hr_p)
res_hr = hr_p - (s_vf_hr * vf_p + i_vf_hr)
rho_partial, p_partial = stats.spearmanr(res_gc, res_hr)
print(f'  gc vs hR | vflat: rho_partial = {rho_partial:.4f}, p = {p_partial:.4e}')

s_hr_gc, i_hr_gc, _, _, _ = stats.linregress(hr_p, gc_p)
res_gc2 = gc_p - (s_hr_gc * hr_p + i_hr_gc)
s_hr_vf, i_hr_vf, _, _, _ = stats.linregress(hr_p, vf_p)
res_vf2 = vf_p - (s_hr_vf * hr_p + i_hr_vf)
rho_partial2, p_partial2 = stats.spearmanr(res_gc2, res_vf2)
print(f'  gc vs vflat | hR: rho_partial = {rho_partial2:.4f}, p = {p_partial2:.4e}')

print('\n[3] Generating figures...')

fig, axes = plt.subplots(2, 3, figsize=(17, 11))

ax = axes[0,0]
if rA:
    ax.scatter(rA['xv'], rA['yv'], s=10, alpha=0.5, c='steelblue', edgecolors='none')
    xf = np.linspace(rA['xv'].min(), rA['xv'].max(), 100)
    ax.plot(xf, rA['slope']*xf+rA['intercept'], 'r-', lw=2,
            label=f'slope={rA["slope"]:.3f}, R^2={rA["r2"]:.3f}')
    ax.plot(xf, 0.5*xf+rA['intercept'], 'b--', lw=1, label='theory: 0.5')
    ax.set_xlabel('log(vflat^2/hR / a0)')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('A: Proxy (baseline)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax = axes[0,1]
if rB:
    ax.scatter(rB['xv'], rB['yv'], s=10, alpha=0.5, c='darkorange', edgecolors='none')
    xf = np.linspace(rB['xv'].min(), rB['xv'].max(), 100)
    ax.plot(xf, rB['slope']*xf+rB['intercept'], 'r-', lw=2,
            label=f'slope={rB["slope"]:.3f}, R^2={rB["r2"]:.3f}')
    for es, col, ls in [(1/3,'blue','--'),(1/4,'green','--'),(0.125,'purple',':')]:
        ax.plot(xf, es*xf+rB['intercept'], color=col, ls=ls, lw=1, label=f'{es:.3f}')
    ax.set_xlabel('log(M_bar / M_sun)')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('B: M_bar direct (BT)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

ax = axes[0,2]
if rD:
    ax.scatter(rD['xv'], rD['yv'], s=10, alpha=0.5, c='green', edgecolors='none')
    xf = np.linspace(rD['xv'].min(), rD['xv'].max(), 100)
    ax.plot(xf, rD['slope']*xf+rD['intercept'], 'r-', lw=2,
            label=f'slope={rD["slope"]:.3f}, R^2={rD["r2"]:.3f}')
    ax.set_xlabel('log(G*Sigma_bar / a0)')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('C: Sigma_bar (recheck)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax = axes[1,0]
if rE:
    ax.scatter(rE['xv'], rE['yv'], s=10, alpha=0.5, c='crimson', edgecolors='none')
    xf = np.linspace(rE['xv'].min(), rE['xv'].max(), 100)
    ax.plot(xf, rE['slope']*xf+rE['intercept'], 'r-', lw=2,
            label=f'slope={rE["slope"]:.3f}, R^2={rE["r2"]:.3f}')
    ax.set_xlabel('log(vflat / km/s)')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('D: vflat only')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax = axes[1,1]
if rF:
    ax.scatter(rF['xv'], rF['yv'], s=10, alpha=0.5, c='teal', edgecolors='none')
    xf = np.linspace(rF['xv'].min(), rF['xv'].max(), 100)
    ax.plot(xf, rF['slope']*xf+rF['intercept'], 'r-', lw=2,
            label=f'slope={rF["slope"]:.3f}, R^2={rF["r2"]:.3f}')
    ax.set_xlabel('log(hR / kpc)')
    ax.set_ylabel('log(gc/a0)')
    ax.set_title('E: hR only')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax = axes[1,2]
labels_bar = []
r2_bar = []
cols_bar = []
for label, reg, col in [
    ('Proxy\nvflat^2/hR', rA, 'steelblue'),
    ('M_bar\n(direct)', rB, 'darkorange'),
    ('Sigma_bar\n(direct)', rD, 'green'),
    ('vflat\nonly', rE, 'crimson'),
    ('hR\nonly', rF, 'teal'),
    ('M_bar\n(BTFR circ)', rG, 'grey'),
]:
    if reg:
        labels_bar.append(label)
        r2_bar.append(reg['r2'])
        cols_bar.append(col)

labels_bar.append('vflat+hR\nmultivar')
r2_bar.append(r2_multi)
cols_bar.append('purple')

xpos = np.arange(len(labels_bar))
ax.bar(xpos, r2_bar, color=cols_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(xpos)
ax.set_xticklabels(labels_bar, fontsize=8)
ax.set_ylabel('R^2', fontsize=11)
ax.set_title('F: Predictive power comparison', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(r2_bar):
    ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=8)

fig.suptitle(f'V-1b: What drives gc? (N={N})', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_gc_vs_Mstar.png'), dpi=150)
print('  -> fig_gc_vs_Mstar.png')

outcsv = os.path.join(BASE, 'gc_vs_Mstar_results.csv')
cols_out = ['galaxy','gc_over_a0','log_gc','vflat','Yd','hR_kpc',
            'log_GS0','log_Mbar_direct','log_Mdisk','log_GSbar',
            'log_vflat','log_hR','log_Mbar_btfr','vdisk_peak']
with open(outcsv, 'w', encoding='utf-8') as f:
    f.write(','.join(cols_out)+'\n')
    for r in results:
        f.write(','.join(str(r.get(c,'')) for c in cols_out)+'\n')
print(f'\n[4] Saved: {outcsv}')

print('\n'+'='*70)
print('FINAL SUMMARY')
print('='*70)

print(f'\n  {"Predictor":<25s} | {"slope":>8s} +/- {"SE":>6s} | {"R^2":>6s} | {"Spearman":>8s}')
print('  '+'-'*70)
for label, reg in [
    ('Proxy vflat^2/hR', rA),
    ('M_bar direct (BT)', rB),
    ('M_disk only', rC),
    ('G*Sigma_bar (BT)', rD),
    ('vflat only', rE),
    ('hR only', rF),
    ('M_bar BTFR (circ)', rG),
]:
    if reg:
        print(f'  {label:<25s} | {reg["slope"]:>8.4f} +/- {reg["se"]:>6.4f} | '
              f'{reg["r2"]:>6.4f} | {reg["rho"]:>8.4f}')
print(f'  {"vflat+hR multivar":<25s} | {"--":>8s}     {"--":>6s} | {r2_multi:>6.4f} | {"--":>8s}')

print(f'\n  Multivariate decomposition:')
print(f'    gc ~ vflat^{beta[0]:.3f} * hR^{beta[1]:.3f}')
print(f'    Proxy predicts: vflat^1.0 * hR^-0.5')
print(f'    p(vflat=1.0) = {p_vf:.4f}')
print(f'    p(hR=-0.5)   = {p_hr:.4f}')

print(f'\n  Partial correlations:')
print(f'    gc vs hR | vflat:  rho={rho_partial:.4f}, p={p_partial:.4e}')
print(f'    gc vs vflat | hR:  rho={rho_partial2:.4f}, p={p_partial2:.4e}')

print('\n  KEY QUESTION: Is gc driven by M* (mass) or proxy (vflat^2/hR)?')
if rB and rA:
    if rB['r2'] > rA['r2'] * 0.8:
        print('  >>> M* has comparable predictive power -> mass is the driver')
    elif rB['r2'] < 0.1:
        print('  >>> M* (direct) has NO predictive power -> proxy encodes something else')
    else:
        print('  >>> Intermediate case')

print('\n[DONE]')
