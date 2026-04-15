# -*- coding: utf-8 -*-
"""
V-1 revised: gc vs Sigma_bar^(1/3) verification.
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
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

a0      = 1.2e-10
G_SI    = 6.674e-11
kpc_m   = 3.0857e19
Msun    = 1.989e30
pc_m    = 3.0857e16

Sigma_crit = a0 / G_SI
Sigma_crit_Msun_pc2 = Sigma_crit / Msun * pc_m**2
print(f'Milgrom critical surface density: {Sigma_crit_Msun_pc2:.0f} M_sun/pc^2')

BASE   = os.path.dirname(os.path.abspath(__file__))
ROTMOD = os.path.join(BASE, 'Rotmod_LTG')
PHASE1 = os.path.join(BASE, 'phase1', 'sparc_results.csv')
TA3    = os.path.join(BASE, 'TA3_gc_independent.csv')

for p, label in [(ROTMOD, 'Rotmod_LTG'), (PHASE1, 'sparc_results.csv'),
                 (TA3, 'TA3_gc_independent.csv')]:
    if not os.path.exists(p):
        print(f'[ERROR] {label} not found: {p}')
        sys.exit(1)


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
            if not line:
                continue
            rows.append([p.strip() for p in line.split(sep)])
    for i, col in enumerate(cols):
        vals = []
        for row in rows:
            if i < len(row):
                try:    vals.append(float(row[i]))
                except: vals.append(row[i])
            else:
                vals.append(np.nan)
        data[col] = vals
    return data

def find_name_col(data):
    for c in ['galaxy', 'Galaxy', 'name', 'Name', 'GALAXY']:
        if c in data:
            return c
    for k, v in data.items():
        if isinstance(v[0], str):
            return k
    return list(data.keys())[0]

def get_key(info, candidates, default=None):
    for c in candidates:
        if c in info:
            try:    return float(info[c])
            except: return info[c]
    return default


print('[1] Loading pipeline data...')
phase1 = load_csv(PHASE1)
ta3    = load_csv(TA3)

p1_nc = find_name_col(phase1)
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

print(f'  phase1: {len(phase1[p1_nc])} galaxies')
print(f'  TA3:    {len(ta3[ta3_nc])} galaxies')


def load_rotmod(filepath):
    cols = [[] for _ in range(8)]
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                for j in range(min(len(parts), 8)):
                    cols[j].append(float(parts[j]))
                for j in range(len(parts), 8):
                    cols[j].append(0.0)
            except ValueError:
                continue
    return tuple(np.array(c) for c in cols)


print('[2] Processing galaxies...')
results = []
rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, '*.dat')))
print(f'  Found {len(rotmod_files)} rotmod files')

for fpath in rotmod_files:
    gname = os.path.splitext(os.path.basename(fpath))[0]
    gname_clean = gname.replace('_rotmod', '').strip()

    info = None
    for key in [gname_clean, gname, gname_clean.upper(), gname_clean.lower()]:
        if key in galaxy_info:
            info = galaxy_info[key]
            break
    if info is None:
        continue

    ud      = get_key(info, ['upsilon_d', 'Upsilon_d', 'Ud', 'ud', 'Yd'])
    gc_a0   = get_key(info, ['gc_over_a0', 'gc/a0', 'gc_ratio'])
    vflat   = get_key(info, ['vflat', 'Vflat', 'v_flat'])

    if ud is None or gc_a0 is None or vflat is None:
        continue
    if np.isnan(ud) or np.isnan(gc_a0) or gc_a0 <= 0 or vflat <= 0:
        continue

    try:
        rad, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = load_rotmod(fpath)
    except:
        continue
    if len(rad) < 3:
        continue

    vds = np.sqrt(max(ud, 0.01)) * np.abs(vdisk)
    rpk_idx = np.argmax(vds)
    rpk = rad[rpk_idx]
    if rpk < 0.01 or rpk >= rad.max() * 0.9:
        continue
    hR_kpc = rpk / 2.15

    hR_m = hR_kpc * kpc_m
    vflat_ms = vflat * 1e3
    GS0_proxy = vflat_ms**2 / hR_m

    gc = gc_a0 * a0

    G_Sigma_bar_C = GS0_proxy**2 / (2 * np.pi * gc)

    vdisk_peak = np.max(np.sqrt(max(ud, 0.01)) * np.abs(vdisk))
    vdisk_peak_ms = vdisk_peak * 1e3
    G_Sigma0_bar_D = vdisk_peak_ms**2 / (0.56 * np.pi * hR_m)

    vgas_peak_ms = np.max(np.abs(vgas)) * 1e3
    G_Sigma_gas = vgas_peak_ms**2 / (0.56 * np.pi * hR_m) if vgas_peak_ms > 0 else 0
    G_Sigma_bar_E = G_Sigma0_bar_D + G_Sigma_gas

    results.append({
        'galaxy':         gname_clean,
        'gc_over_a0':     gc_a0,
        'log_gc_a0':      np.log10(gc_a0),
        'vflat':          vflat,
        'Yd':             ud,
        'hR_kpc':         hR_kpc,
        'GS0_proxy':      GS0_proxy,
        'log_GS0_proxy':  np.log10(GS0_proxy / a0),
        'G_Sbar_C':       G_Sigma_bar_C,
        'log_GSbar_C':    np.log10(G_Sigma_bar_C / a0) if G_Sigma_bar_C > 0 else np.nan,
        'G_Sbar_D':       G_Sigma0_bar_D,
        'log_GSbar_D':    np.log10(G_Sigma0_bar_D / a0) if G_Sigma0_bar_D > 0 else np.nan,
        'G_Sbar_E':       G_Sigma_bar_E,
        'log_GSbar_E':    np.log10(G_Sigma_bar_E / a0) if G_Sigma_bar_E > 0 else np.nan,
        'vdisk_peak':     vdisk_peak,
        'vgas_peak':      np.max(np.abs(vgas)),
    })

N = len(results)
print(f'  Processed: {N} galaxies')

if N < 10:
    print('[ERROR] Too few galaxies.')
    sys.exit(1)

log_gc = np.array([r['log_gc_a0'] for r in results])

log_GSbar_D = np.array([r['log_GSbar_D'] for r in results])
valid_D = np.isfinite(log_GSbar_D) & np.isfinite(log_gc)

log_GSbar_E = np.array([r['log_GSbar_E'] for r in results])
valid_E = np.isfinite(log_GSbar_E) & np.isfinite(log_gc)

log_GSbar_C = np.array([r['log_GSbar_C'] for r in results])
valid_C = np.isfinite(log_GSbar_C) & np.isfinite(log_gc)

log_GS0 = np.array([r['log_GS0_proxy'] for r in results])
valid_0 = np.isfinite(log_GS0) & np.isfinite(log_gc)

print('\n' + '='*70)
print('V-1 revised: gc vs Sigma_bar scaling test')
print('='*70)

def do_regression(x, y, mask, label, expected_slope):
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 5:
        print(f'\n[{label}] N={n} -- too few points')
        return None

    slope, intercept, r_val, p_val, se = stats.linregress(xv, yv)
    rho_s, p_s = stats.spearmanr(xv, yv)

    t_stat = abs(slope - expected_slope) / se
    p_slope = 2 * (1 - stats.t.cdf(t_stat, df=n-2))

    print(f'\n[{label}]  N = {n}')
    print(f'  slope     = {slope:.4f} +/- {se:.4f}')
    print(f'  expected  = {expected_slope}')
    print(f'  p(slope={expected_slope}) = {p_slope:.4f}')
    print(f'  R^2       = {r_val**2:.4f}')
    print(f'  Spearman  = {rho_s:.4f}, p = {p_s:.2e}')
    print(f'  intercept = {intercept:.4f}  (eta = 10^{intercept:.3f} = {10**intercept:.4f})')

    verdict = 'PASS' if p_slope > 0.05 else 'MARGINAL' if p_slope > 0.01 else 'FAIL'
    print(f'  VERDICT:  {verdict}')

    return {
        'slope': slope, 'se': se, 'intercept': intercept,
        'r2': r_val**2, 'rho': rho_s, 'p_spearman': p_s,
        'p_slope': p_slope, 'n': n, 'verdict': verdict,
        'eta_prime': 10**intercept
    }

print('\n--- Test 0: Original proxy (alpha=0.5 expected, baseline) ---')
r0 = do_regression(log_GS0, log_gc, valid_0,
                    'GS0_proxy = vflat^2/hR', 0.5)

print('\n--- Test D: V_disk peak Sigma_bar (1/3 expected, INDEPENDENT) ---')
rD = do_regression(log_GSbar_D, log_gc, valid_D,
                    'G*Sigma_bar from Vdisk_peak (BT)', 1/3)

print('\n--- Test E: disk+gas Sigma_bar (1/3 expected, INDEPENDENT) ---')
rE = do_regression(log_GSbar_E, log_gc, valid_E,
                    'G*Sigma_bar disk+gas (BT)', 1/3)

print('\n--- Test C: BTFR-decontaminated (1/3 expected, CIRCULAR CHECK) ---')
rC = do_regression(log_GSbar_C, log_gc, valid_C,
                    'G*Sigma_bar BTFR-decontam (circular)', 1/3)

print('\n--- Cross-check: does 0.5 work for true Sigma_bar? ---')
rD5 = do_regression(log_GSbar_D, log_gc, valid_D,
                     'G*Sigma_bar (BT) vs slope=0.5', 0.5)
rE5 = do_regression(log_GSbar_E, log_gc, valid_E,
                     'G*Sigma_bar disk+gas vs slope=0.5', 0.5)

G_Sbar_D_arr = np.array([r['G_Sbar_D'] for r in results])
Sbar_ratio = G_Sbar_D_arr / (G_SI * Sigma_crit)
print(f'\n--- Diagnostic ---')
print(f'  Sigma_bar / Sigma_crit (Milgrom):')
print(f'    median = {np.nanmedian(Sbar_ratio):.4f}')
print(f'    IQR = [{np.nanpercentile(Sbar_ratio, 25):.4f}, '
      f'{np.nanpercentile(Sbar_ratio, 75):.4f}]')
print(f'    range = [{np.nanmin(Sbar_ratio):.4f}, {np.nanmax(Sbar_ratio):.4f}]')

print('\n[3] Generating figures...')

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, logx, mask, reg, title, expected in [
    (axes[0], log_GS0, valid_0, r0,
     'Proxy: vflat^2/hR (baseline)', 0.5),
    (axes[1], log_GSbar_D, valid_D, rD,
     'V_disk peak (BT, independent)', 1/3),
    (axes[2], log_GSbar_E, valid_E, rE,
     'Disk+Gas (BT, independent)', 1/3),
]:
    if reg is None:
        continue
    xv, yv = logx[mask], log_gc[mask]
    ax.scatter(xv, yv, s=10, alpha=0.5, c='steelblue', edgecolors='none')
    xfit = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xfit, reg['slope']*xfit + reg['intercept'], 'r-', lw=2,
            label=f'fit: slope={reg["slope"]:.3f}+/-{reg["se"]:.3f}')
    ax.plot(xfit, expected*xfit + reg['intercept'], 'b--', lw=1.5,
            label=f'theory: slope={expected:.3f}')
    ax.set_xlabel('log(G*Sigma / a0)', fontsize=10)
    ax.set_ylabel('log(gc / a0)', fontsize=10)
    ax.set_title(f'{title}\np(slope={expected:.2f})={reg["p_slope"]:.3f} [{reg["verdict"]}]',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

fig.suptitle(f'V-1 revised: gc vs Sigma_bar scaling (N={N})', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_v1kai_slope_test.png'), dpi=150)
print('  -> fig_v1kai_slope_test.png')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
vals = log_GSbar_D[valid_D]
ax.hist(vals, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', ls='--', lw=2, label='G*Sigma_bar = a0')
ax.set_xlabel('log(G*Sigma_bar / a0)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('A: True baryon surface density', fontsize=11)
ax.legend(fontsize=9)

ax = axes[0, 1]
methods = []
slopes  = []
errors  = []
colors_bar = []
for label, reg, col in [
    ('Proxy\n(vflat^2/hR)', r0, 'steelblue'),
    ('Vdisk peak\n(BT)', rD, 'darkorange'),
    ('Disk+Gas\n(BT)', rE, 'green'),
    ('BTFR decontam\n(circular)', rC, 'grey'),
]:
    if reg is not None:
        methods.append(label)
        slopes.append(reg['slope'])
        errors.append(reg['se'])
        colors_bar.append(col)

x_pos = np.arange(len(methods))
ax.bar(x_pos, slopes, yerr=errors, color=colors_bar, alpha=0.7,
       capsize=5, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='blue', ls='--', lw=1.5, label='0.5 (proxy prediction)')
ax.axhline(1/3, color='red', ls='--', lw=1.5, label='1/3 (Sigma_bar prediction)')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel('Slope', fontsize=10)
ax.set_title('B: Slope comparison', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
if rD is not None:
    xv = log_GSbar_D[valid_D]
    yv = log_gc[valid_D]
    resid = yv - (1/3 * xv + rD['intercept'])
    ax.hist(resid, bins=25, color='darkorange', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', ls='--')
    ax.set_xlabel('Residual from 1/3 law [dex]', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'C: Residual (sigma={np.std(resid):.3f} dex)', fontsize=11)

ax = axes[1, 1]
if r0 is not None:
    xv = log_GS0[valid_0]
    yv = log_gc[valid_0]
    resid0 = yv - (0.5 * xv + r0['intercept'])
    ax.hist(resid0, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', ls='--')
    ax.set_xlabel('Residual from 0.5 law (proxy) [dex]', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'D: Proxy residual (sigma={np.std(resid0):.3f} dex)', fontsize=11)

fig.suptitle(f'V-1 revised summary (N={N})', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_v1kai_4panel.png'), dpi=150)
print('  -> fig_v1kai_4panel.png')

outcsv = os.path.join(BASE, 'v1kai_results.csv')
cols_out = ['galaxy', 'gc_over_a0', 'log_gc_a0', 'vflat', 'Yd', 'hR_kpc',
            'GS0_proxy', 'log_GS0_proxy',
            'G_Sbar_D', 'log_GSbar_D', 'G_Sbar_E', 'log_GSbar_E',
            'G_Sbar_C', 'log_GSbar_C',
            'vdisk_peak', 'vgas_peak']
with open(outcsv, 'w', encoding='utf-8') as f:
    f.write(','.join(cols_out) + '\n')
    for r in results:
        f.write(','.join(str(r.get(c, '')) for c in cols_out) + '\n')
print(f'\n[4] Saved: {outcsv}')

print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)

print('\n  Method                          | slope   +/- SE    | expected | p(match) | R^2    | verdict')
print('  ' + '-'*100)
for label, reg, expected in [
    ('Proxy vflat^2/hR (baseline)',  r0, 0.5),
    ('Vdisk peak BT (independent)',  rD, 1/3),
    ('Disk+Gas BT (independent)',    rE, 1/3),
    ('BTFR decontam (circular)',     rC, 1/3),
    ('Vdisk peak BT vs 0.5',         rD5, 0.5),
    ('Disk+Gas BT vs 0.5',           rE5, 0.5),
]:
    if reg is None:
        continue
    print(f'  {label:33s} | {reg["slope"]:.4f} +/- {reg["se"]:.4f} | '
          f'{expected:.4f}   | {reg["p_slope"]:.4f}   | '
          f'{reg["r2"]:.4f} | {reg["verdict"]}')

print('\n  Key question: Is the slope with TRUE Sigma_bar closer to 1/3 or 0.5?')
if rD is not None and rD5 is not None:
    if rD['p_slope'] > rD5['p_slope']:
        print('  >>> 1/3 is BETTER fit for true Sigma_bar')
    elif rD5['p_slope'] > rD['p_slope']:
        print('  >>> 0.5 is BETTER fit for true Sigma_bar')
    else:
        print('  >>> Inconclusive')

print('\n[DONE]')
