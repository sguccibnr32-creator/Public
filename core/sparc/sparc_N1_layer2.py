# -*- coding: utf-8 -*-
"""
N-1 Layer 2: MOND transition correction analysis.
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

    r_m = rad * kpc_m
    vbar2 = (vgas**2 + ud * np.sign(vdisk)*vdisk**2
             + 0.7 * np.sign(vbul)*vbul**2)
    vbar2 = np.abs(vbar2)
    g_N = np.zeros_like(r_m)
    mask_r = r_m > 0
    g_N[mask_r] = vbar2[mask_r] * 1e6 / r_m[mask_r]

    gc_si = gc_a0 * a0

    n_outer = min(3, len(rad))
    g_N_outer = np.mean(g_N[-n_outer:])
    x_outer = g_N_outer / gc_si if gc_si > 0 else np.nan

    r_peak_target = 2.2 * hR_kpc
    idx_peak = np.argmin(np.abs(rad - r_peak_target))
    g_N_peak = g_N[idx_peak]
    x_peak = g_N_peak / gc_si if gc_si > 0 else np.nan

    if g_N_outer > 0 and gc_si > 0:
        g_obs_full = (g_N_outer + np.sqrt(g_N_outer**2 + 4*gc_si*g_N_outer)) / 2
        g_obs_mond = np.sqrt(gc_si * g_N_outer)
        correction = g_obs_full / g_obs_mond
    else:
        correction = np.nan

    vflat_ms = vflat * 1e3
    GS0_proxy = vflat_ms**2 / hR_m

    if np.isfinite(correction) and correction > 0:
        GS0_corrected = GS0_proxy / correction
    else:
        GS0_corrected = np.nan

    corrections_all = np.ones_like(g_N)
    for i_r in range(len(g_N)):
        if g_N[i_r] > 0 and gc_si > 0:
            gf = (g_N[i_r] + np.sqrt(g_N[i_r]**2 + 4*gc_si*g_N[i_r])) / 2
            gm = np.sqrt(gc_si * g_N[i_r])
            corrections_all[i_r] = gf / gm
    pos = g_N > 0
    mean_correction = np.mean(corrections_all[pos]) if pos.any() else np.nan

    results.append({
        'galaxy':         gname,
        'gc_over_a0':     gc_a0,
        'log_gc':         np.log10(gc_a0),
        'vflat':          vflat,
        'Yd':             ud,
        'hR_kpc':         hR_kpc,
        'log_hR':         np.log10(hR_kpc),
        'log_vflat':      np.log10(vflat),
        'GS0_proxy':      GS0_proxy,
        'log_GS0':        np.log10(GS0_proxy/a0),
        'x_outer':        x_outer,
        'log_x_outer':    np.log10(x_outer) if x_outer > 0 else np.nan,
        'x_peak':         x_peak,
        'g_N_outer':      g_N_outer,
        'g_N_peak':       g_N_peak,
        'correction':     correction,
        'log_correction': np.log10(correction) if np.isfinite(correction) and correction>0 else np.nan,
        'mean_correction': mean_correction,
        'GS0_corrected':  GS0_corrected,
        'log_GS0_corr':   np.log10(GS0_corrected/a0) if np.isfinite(GS0_corrected) and GS0_corrected>0 else np.nan,
        'r_outer_kpc':    rad[-1],
        'r_outer_over_hR': rad[-1] / hR_kpc if hR_kpc > 0 else np.nan,
    })

N = len(results)
print(f'  Processed: {N} galaxies')
if N < 10:
    print('[ERROR] Too few galaxies.'); sys.exit(1)

log_gc      = np.array([r['log_gc'] for r in results])
log_hR      = np.array([r['log_hR'] for r in results])
log_vflat   = np.array([r['log_vflat'] for r in results])
log_GS0     = np.array([r['log_GS0'] for r in results])
x_outer     = np.array([r['x_outer'] for r in results])
log_x_outer = np.array([r['log_x_outer'] for r in results])
correction  = np.array([r['correction'] for r in results])
log_corr    = np.array([r['log_correction'] for r in results])
log_GS0_c   = np.array([r['log_GS0_corr'] for r in results])
r_over_hR   = np.array([r['r_outer_over_hR'] for r in results])

print('\n' + '='*70)
print('N-1 Layer 2: MOND transition correction analysis')
print('='*70)

valid = np.isfinite(x_outer) & (x_outer > 0)
print(f'\n[A] x_outer = g_N(r_outer)/gc distribution (N={np.sum(valid)})')
print(f'    median = {np.nanmedian(x_outer[valid]):.4f}')
print(f'    IQR = [{np.nanpercentile(x_outer[valid],25):.4f}, '
      f'{np.nanpercentile(x_outer[valid],75):.4f}]')
print(f'    range = [{np.nanmin(x_outer[valid]):.4f}, {np.nanmax(x_outer[valid]):.4f}]')
print(f'    fraction x_outer > 0.1: {np.sum(x_outer[valid]>0.1)/np.sum(valid):.1%}')
print(f'    fraction x_outer > 0.5: {np.sum(x_outer[valid]>0.5)/np.sum(valid):.1%}')

mask_b = np.isfinite(log_x_outer) & np.isfinite(log_hR)
rho_xh, p_xh = stats.spearmanr(log_x_outer[mask_b], log_hR[mask_b])
print(f'\n[B] x_outer vs hR correlation')
print(f'    Spearman rho(log x_outer, log hR) = {rho_xh:.4f}, p = {p_xh:.2e}')

slope_xh, int_xh, r_xh, _, se_xh = stats.linregress(log_hR[mask_b], log_x_outer[mask_b])
print(f'    Linear: log(x_outer) = {slope_xh:.3f}*log(hR) + {int_xh:.3f}, R^2={r_xh**2:.3f}')

valid_c = np.isfinite(correction)
print(f'\n[C] Correction factor (N={np.sum(valid_c)})')
print(f'    median = {np.nanmedian(correction[valid_c]):.4f}')
print(f'    IQR = [{np.nanpercentile(correction[valid_c],25):.4f}, '
      f'{np.nanpercentile(correction[valid_c],75):.4f}]')
print(f'    range = [{np.nanmin(correction[valid_c]):.4f}, '
      f'{np.nanmax(correction[valid_c]):.4f}]')

mask_d = np.isfinite(log_corr) & np.isfinite(log_hR)
rho_ch, p_ch = stats.spearmanr(log_corr[mask_d], log_hR[mask_d])
print(f'\n[D] Correction vs hR')
print(f'    Spearman rho = {rho_ch:.4f}, p = {p_ch:.2e}')

mask_e = np.isfinite(log_GS0) & np.isfinite(log_gc)
slope_0, int_0, r_0, _, se_0 = stats.linregress(log_GS0[mask_e], log_gc[mask_e])
print(f'\n[E] Baseline: log(gc) vs log(proxy/a0)')
print(f'    slope = {slope_0:.4f} +/- {se_0:.4f}, R^2 = {r_0**2:.4f}')

mask_f = np.isfinite(log_GS0_c) & np.isfinite(log_gc)
if np.sum(mask_f) > 10:
    slope_c, int_c, r_c, _, se_c = stats.linregress(log_GS0_c[mask_f], log_gc[mask_f])
    print(f'\n[F] Corrected proxy regression')
    print(f'    slope = {slope_c:.4f} +/- {se_c:.4f}, R^2 = {r_c**2:.4f}')
else:
    slope_c, r_c, se_c = np.nan, np.nan, np.nan

mask_g = (np.isfinite(log_gc) & np.isfinite(log_vflat) &
          np.isfinite(log_hR) & np.isfinite(log_corr))

print(f'\n[G] Multivariate analysis (N={np.sum(mask_g)})')

X1 = np.column_stack([log_vflat[mask_g], log_hR[mask_g], np.ones(mask_g.sum())])
y = log_gc[mask_g]
beta1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
ss_res1 = np.sum((y - X1 @ beta1)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_1 = 1 - ss_res1/ss_tot
mse1 = ss_res1 / (len(y)-3)
se1 = np.sqrt(np.diag(mse1 * np.linalg.inv(X1.T @ X1)))

print(f'  G1: log(gc) = {beta1[0]:.4f}*log(vflat) + {beta1[1]:.4f}*log(hR) + {beta1[2]:.4f}')
print(f'      SE:        {se1[0]:.4f}                 {se1[1]:.4f}')
print(f'      R^2 = {r2_1:.4f}')

X2 = np.column_stack([log_vflat[mask_g], log_hR[mask_g], log_corr[mask_g],
                       np.ones(mask_g.sum())])
beta2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
ss_res2 = np.sum((y - X2 @ beta2)**2)
r2_2 = 1 - ss_res2/ss_tot
mse2 = ss_res2 / (len(y)-4)
se2 = np.sqrt(np.diag(mse2 * np.linalg.inv(X2.T @ X2)))

print(f'\n  G2: log(gc) = {beta2[0]:.4f}*log(vflat) + {beta2[1]:.4f}*log(hR) '
      f'+ {beta2[2]:.4f}*log(corr) + {beta2[3]:.4f}')
print(f'      SE:        {se2[0]:.4f}                 {se2[1]:.4f}                '
      f'{se2[2]:.4f}')
print(f'      R^2 = {r2_2:.4f}  (vs {r2_1:.4f} without correction)')

t_corr = abs(beta2[2]) / se2[2]
p_corr = 2*(1-stats.t.cdf(t_corr, df=len(y)-4))
print(f'      p(correction coeff = 0) = {p_corr:.4f}')

print(f'\n  hR coefficient comparison:')
print(f'    Without correction: {beta1[1]:.4f} +/- {se1[1]:.4f}')
print(f'    With correction:    {beta2[1]:.4f} +/- {se2[1]:.4f}')
delta_hR = abs(beta2[1]) - abs(beta1[1])
print(f'    |hR coeff| change:  {delta_hR:+.4f}')
if abs(beta2[1]) < abs(beta1[1]) * 0.5:
    print(f'    >>> hR effect ABSORBED by correction (>50% reduction)')
elif abs(beta2[1]) < abs(beta1[1]) * 0.8:
    print(f'    >>> hR effect PARTIALLY absorbed')
else:
    print(f'    >>> hR effect NOT absorbed by correction')

print(f'\n[H] Partial correlations controlling for correction')

X_ctrl = np.column_stack([log_vflat[mask_g], log_corr[mask_g]])
y_gc = log_gc[mask_g]
y_hR = log_hR[mask_g]

beta_gc = np.linalg.lstsq(np.column_stack([X_ctrl, np.ones(len(X_ctrl))]),
                            y_gc, rcond=None)[0]
res_gc = y_gc - np.column_stack([X_ctrl, np.ones(len(X_ctrl))]) @ beta_gc

beta_hr = np.linalg.lstsq(np.column_stack([X_ctrl, np.ones(len(X_ctrl))]),
                            y_hR, rcond=None)[0]
res_hR = y_hR - np.column_stack([X_ctrl, np.ones(len(X_ctrl))]) @ beta_hr

rho_partial_new, p_partial_new = stats.spearmanr(res_gc, res_hR)
print(f'  gc vs hR | (vflat, correction): rho = {rho_partial_new:.4f}, p = {p_partial_new:.4e}')

X_ctrl0 = log_vflat[mask_g].reshape(-1,1)
beta_gc0 = np.linalg.lstsq(np.column_stack([X_ctrl0, np.ones(len(X_ctrl0))]),
                             y_gc, rcond=None)[0]
res_gc0 = y_gc - np.column_stack([X_ctrl0, np.ones(len(X_ctrl0))]) @ beta_gc0
beta_hr0 = np.linalg.lstsq(np.column_stack([X_ctrl0, np.ones(len(X_ctrl0))]),
                             y_hR, rcond=None)[0]
res_hR0 = y_hR - np.column_stack([X_ctrl0, np.ones(len(X_ctrl0))]) @ beta_hr0
rho_partial_orig, p_partial_orig = stats.spearmanr(res_gc0, res_hR0)

print(f'  gc vs hR | vflat only:           rho = {rho_partial_orig:.4f}, p = {p_partial_orig:.4e}')
print(f'  Reduction: {rho_partial_orig:.4f} -> {rho_partial_new:.4f} '
      f'({(1-abs(rho_partial_new)/abs(rho_partial_orig))*100:.1f}%)')

print('\n[3] Generating figures...')

fig, axes = plt.subplots(2, 3, figsize=(17, 11))

ax = axes[0, 0]
xo_valid = x_outer[np.isfinite(x_outer) & (x_outer > 0)]
ax.hist(np.log10(xo_valid), bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(np.log10(0.1), color='orange', ls='--', lw=2, label='x=0.1')
ax.axvline(np.log10(0.5), color='red', ls='--', lw=2, label='x=0.5')
ax.set_xlabel('log(x_outer) = log(g_N(r_outer)/gc)')
ax.set_ylabel('Count')
ax.set_title('A: MOND departure parameter')
ax.legend(fontsize=8)

ax = axes[0, 1]
mb = np.isfinite(log_x_outer) & np.isfinite(log_hR)
ax.scatter(log_hR[mb], log_x_outer[mb], s=10, alpha=0.5, c='steelblue', edgecolors='none')
xf = np.linspace(log_hR[mb].min(), log_hR[mb].max(), 100)
ax.plot(xf, slope_xh*xf + int_xh, 'r-', lw=2,
        label=f'rho={rho_xh:.3f}')
ax.set_xlabel('log(hR / kpc)')
ax.set_ylabel('log(x_outer)')
ax.set_title('B: x_outer vs hR')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
md = np.isfinite(log_corr) & np.isfinite(log_hR)
ax.scatter(log_hR[md], log_corr[md], s=10, alpha=0.5, c='darkorange', edgecolors='none')
ax.set_xlabel('log(hR / kpc)')
ax.set_ylabel('log(correction)')
ax.set_title(f'C: Correction vs hR (rho={rho_ch:.3f})')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
mf = np.isfinite(log_GS0) & np.isfinite(log_gc)
ax.scatter(log_GS0[mf], log_gc[mf], s=10, alpha=0.3, c='steelblue',
           edgecolors='none', label=f'Original R^2={r_0**2:.3f}')
mfc = np.isfinite(log_GS0_c) & np.isfinite(log_gc)
if np.sum(mfc) > 5 and np.isfinite(r_c):
    ax.scatter(log_GS0_c[mfc], log_gc[mfc], s=10, alpha=0.3, c='red',
               edgecolors='none', label=f'Corrected R^2={r_c**2:.3f}')
ax.set_xlabel('log(proxy / a0)')
ax.set_ylabel('log(gc / a0)')
ax.set_title('D: Original vs corrected proxy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(res_hR0, res_gc0, s=10, alpha=0.3, c='steelblue', edgecolors='none',
           label=f'Original: rho={rho_partial_orig:.3f}')
ax.scatter(res_hR, res_gc, s=10, alpha=0.3, c='red', edgecolors='none',
           label=f'+ correction: rho={rho_partial_new:.3f}')
ax.set_xlabel('hR residual')
ax.set_ylabel('gc residual')
ax.set_title('E: hR partial correlation')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
labels = ['vflat+hR\n(orig)', 'vflat+hR\n+corr', 'proxy\norig', 'proxy\ncorr']
r2s = [r2_1, r2_2, r_0**2, r_c**2 if np.isfinite(r_c) else 0]
cols = ['steelblue', 'darkorange', 'green', 'red']
ax.bar(range(len(labels)), r2s, color=cols, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('R^2')
ax.set_title('F: Model comparison')
for i, v in enumerate(r2s):
    ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'N-1 Layer 2: MOND transition correction (N={N})', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_N1_layer2.png'), dpi=150)
print('  -> fig_N1_layer2.png')

outcsv = os.path.join(BASE, 'N1_layer2_results.csv')
cols_out = ['galaxy','gc_over_a0','log_gc','vflat','hR_kpc','log_hR','log_vflat',
            'x_outer','log_x_outer','correction','log_correction',
            'GS0_proxy','log_GS0','GS0_corrected','log_GS0_corr',
            'r_outer_kpc','r_outer_over_hR']
with open(outcsv, 'w', encoding='utf-8') as f:
    f.write(','.join(cols_out)+'\n')
    for r in results:
        f.write(','.join(str(r.get(c,'')) for c in cols_out)+'\n')
print(f'  -> {outcsv}')

print('\n' + '='*70)
print('VERDICT')
print('='*70)

print(f'\n  x_outer median: {np.nanmedian(x_outer[valid]):.4f}')
if np.nanmedian(x_outer[valid]) < 0.05:
    print('  >>> Most galaxies are deep in MOND regime (x_outer << 1)')
elif np.nanmedian(x_outer[valid]) < 0.3:
    print('  >>> Moderate MOND departure for typical galaxies')
else:
    print('  >>> Significant MOND departure')

print(f'\n  hR partial correlation:')
print(f'    Original (gc vs hR | vflat):              rho = {rho_partial_orig:.4f}')
print(f'    After correction (gc vs hR | vflat,corr): rho = {rho_partial_new:.4f}')
reduction = (1-abs(rho_partial_new)/abs(rho_partial_orig))*100
print(f'    Reduction: {reduction:.1f}%')

if reduction > 70:
    print('  >>> hR effect EXPLAINED by MOND transition correction')
    print('  >>> N-1 Layer 2: RESOLVED')
elif reduction > 30:
    print('  >>> hR effect PARTIALLY explained')
    print('  >>> N-1 Layer 2: PARTIAL')
else:
    print('  >>> hR effect NOT explained by transition')
    print('  >>> N-1 Layer 2: hR contribution is INDEPENDENT')

print(f'\n  Correction coefficient in multivariate:')
print(f'    coeff = {beta2[2]:.4f} +/- {se2[2]:.4f}, p = {p_corr:.4f}')
if p_corr < 0.05:
    print('  >>> Correction is statistically significant')
else:
    print('  >>> Correction is NOT significant')

print('\n[DONE]')
