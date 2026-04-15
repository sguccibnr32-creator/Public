# -*- coding: utf-8 -*-
"""
V-1: Condition-14 quantitative mechanism verification.
SPARC 175銀河の g_N(r) プロファイルから f_p（塑性領域質量分率）を
数値計算し、観測された gc との相関を検証する。
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

a0 = 1.2e-10
G_SI = 6.674e-11
kpc_m = 3.0857e19
Msun = 1.989e30

BASE = os.path.dirname(os.path.abspath(__file__))
ROTMOD = os.path.join(BASE, 'Rotmod_LTG')
PHASE1 = os.path.join(BASE, 'phase1', 'sparc_results.csv')
TA3    = os.path.join(BASE, 'TA3_gc_independent.csv')

for p, label in [(ROTMOD, 'Rotmod_LTG'), (PHASE1, 'sparc_results.csv'), (TA3, 'TA3_gc_independent.csv')]:
    if not os.path.exists(p):
        print(f'[ERROR] {label} not found: {p}')
        sys.exit(1)


def load_csv_flexible(path):
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
            parts = line.split(sep)
            rows.append([p.strip() for p in parts])
    for i, col in enumerate(cols):
        vals = []
        for row in rows:
            if i < len(row):
                try: vals.append(float(row[i]))
                except: vals.append(row[i])
            else:
                vals.append(np.nan)
        data[col] = vals
    return data


print('[1] Loading pipeline results...')
phase1 = load_csv_flexible(PHASE1)
ta3    = load_csv_flexible(TA3)


def find_name_col(data):
    for candidate in ['galaxy', 'Galaxy', 'name', 'Name', 'GALAXY']:
        if candidate in data:
            return candidate
    for k, v in data.items():
        if isinstance(v[0], str):
            return k
    return list(data.keys())[0]


p1_name_col = find_name_col(phase1)
ta3_name_col = find_name_col(ta3)

galaxy_info = {}
for i, name in enumerate(phase1[p1_name_col]):
    name = str(name).strip()
    info = {}
    for k in phase1:
        if k == p1_name_col: continue
        try: info[k] = float(phase1[k][i])
        except: info[k] = phase1[k][i]
    galaxy_info[name] = info

for i, name in enumerate(ta3[ta3_name_col]):
    name = str(name).strip()
    if name in galaxy_info:
        for k in ta3:
            if k == ta3_name_col: continue
            try: galaxy_info[name][k] = float(ta3[k][i])
            except: galaxy_info[name][k] = ta3[k][i]

print(f'  phase1: {len(phase1[p1_name_col])} galaxies')
print(f'  TA3:    {len(ta3[ta3_name_col])} galaxies')


def compute_fp(radii_kpc, vgas, vdisk, vbul, upsilon_d, upsilon_b=0.7):
    r_m = radii_kpc * kpc_m

    vbar2 = (vgas**2
             + upsilon_d * np.sign(vdisk) * vdisk**2
             + upsilon_b * np.sign(vbul) * vbul**2)
    vbar2 = np.abs(vbar2)

    g_N = np.zeros_like(r_m)
    mask = r_m > 0
    g_N[mask] = (vbar2[mask] * 1e6) / r_m[mask]

    weight = np.abs(vdisk**2) * upsilon_d + np.abs(vgas**2)
    weight = np.maximum(weight, 1e-6)

    plastic_mask = g_N < a0
    total_weight = np.sum(weight)
    if total_weight <= 0:
        return np.nan, g_N

    f_p = np.sum(weight[plastic_mask]) / total_weight
    return f_p, g_N


def load_rotmod(filepath):
    rad, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = [], [], [], [], [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 6: continue
            try:
                rad.append(float(parts[0]))
                vobs.append(float(parts[1]))
                errv.append(float(parts[2]))
                vgas.append(float(parts[3]))
                vdisk.append(float(parts[4]))
                vbul.append(float(parts[5]))
                if len(parts) > 6: sbdisk.append(float(parts[6]))
                else: sbdisk.append(0.0)
                if len(parts) > 7: sbbul.append(float(parts[7]))
                else: sbbul.append(0.0)
            except ValueError:
                continue
    return (np.array(rad), np.array(vobs), np.array(errv),
            np.array(vgas), np.array(vdisk), np.array(vbul),
            np.array(sbdisk), np.array(sbbul))


print('[2] Processing SPARC galaxies...')

results = []
rotmod_files = sorted(glob.glob(os.path.join(ROTMOD, '*.dat')))
print(f'  Found {len(rotmod_files)} rotmod files')


def get_key(info, candidates, default=None):
    for c in candidates:
        if c in info:
            return info[c]
    return default


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

    ud = get_key(info, ['upsilon_d', 'Upsilon_d', 'Ud', 'ud', 'Yd'], None)
    gc_a0 = get_key(info, ['gc_over_a0', 'gc/a0', 'gc_ratio'], None)
    vflat_val = get_key(info, ['vflat', 'Vflat', 'v_flat'], None)

    if ud is None or gc_a0 is None or vflat_val is None:
        continue

    try:
        ud = float(ud)
        gc_a0 = float(gc_a0)
        vflat_val = float(vflat_val)
    except (ValueError, TypeError):
        continue

    if np.isnan(ud) or np.isnan(gc_a0) or gc_a0 <= 0:
        continue

    try:
        rad, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = load_rotmod(fpath)
    except:
        continue

    if len(rad) < 3:
        continue

    f_p, g_N = compute_fp(rad, vgas, vdisk, vbul, ud)
    if np.isnan(f_p):
        continue

    c_membrane = 1.0 - f_p

    vds = np.sqrt(np.maximum(ud, 0.01)) * np.abs(vdisk)
    rpk_idx = np.argmax(vds)
    rpk = rad[rpk_idx]
    if rpk < 0.01 or rpk >= rad.max() * 0.9:
        continue
    hR = rpk / 2.15
    hR_m = hR * kpc_m
    vflat_ms = vflat_val * 1e3
    GS0 = vflat_ms**2 / hR_m

    sqrt_ratio = np.sqrt(GS0 / a0)

    results.append({
        'galaxy': gname_clean,
        'f_p': f_p,
        'c_membrane': c_membrane,
        'gc_over_a0': gc_a0,
        'log_gc_a0': np.log10(gc_a0),
        'vflat': vflat_val,
        'Yd': ud,
        'hR_kpc': hR,
        'GS0': GS0,
        'log_GS0_a0': np.log10(GS0 / a0),
        'sqrt_GS0_a0': sqrt_ratio,
        'n_points': len(rad),
        'g_N_max': np.max(g_N),
        'g_N_min': np.min(g_N[g_N > 0]) if np.any(g_N > 0) else 0,
    })

N = len(results)
print(f'  Successfully processed: {N} galaxies')

if N < 10:
    print('[ERROR] Too few galaxies.')
    sys.exit(1)

print('\n' + '='*60)
print('V-1: f_p vs gc correlation analysis')
print('='*60)

fp_arr    = np.array([r['f_p'] for r in results])
c_arr     = np.array([r['c_membrane'] for r in results])
gc_arr    = np.array([r['gc_over_a0'] for r in results])
log_gc    = np.array([r['log_gc_a0'] for r in results])
sqrt_arr  = np.array([r['sqrt_GS0_a0'] for r in results])
log_GS0   = np.array([r['log_GS0_a0'] for r in results])

rho1, p1 = stats.spearmanr(fp_arr, log_gc)
print(f'\n[Test 1] f_p vs log(gc/a0):')
print(f'  Spearman rho = {rho1:.3f}, p = {p1:.2e}')

rho2, p2 = stats.spearmanr(c_arr, gc_arr)
print(f'\n[Test 2] c = 1-f_p vs gc/a0:')
print(f'  Spearman rho = {rho2:.3f}, p = {p2:.2e}')

rho3, p3 = stats.spearmanr(c_arr, sqrt_arr)
print(f'\n[Test 3] c = 1-f_p vs sqrt(G*Sigma0/a0):')
print(f'  Spearman rho = {rho3:.3f}, p = {p3:.2e}')

valid = (c_arr > 0) & (sqrt_arr > 0)
log_c = np.log10(c_arr[valid])
log_sqrt = np.log10(sqrt_arr[valid])
log_GS0_v = log_GS0[valid]

slope, intercept, r_val, p_slope, se_slope = stats.linregress(log_GS0_v, log_c)
print(f'\n[Test 4] log(c) vs log(G*Sigma0/a0) regression:')
print(f'  slope = {slope:.3f} +/- {se_slope:.3f}')
print(f'  Expected slope = 0.5 (sqrt scaling)')
print(f'  p(slope=0.5) = {2*(1-stats.t.cdf(abs(slope-0.5)/se_slope, df=sum(valid)-2)):.3f}')
print(f'  R^2 = {r_val**2:.3f}')
print(f'  intercept = {intercept:.3f}')

eta_eff = 10**intercept
print(f'\n[Derived] eta_eff = 10^{intercept:.3f} = {eta_eff:.3f}')

print(f'\n--- Summary ---')
print(f'N = {N}')
print(f'f_p: median = {np.median(fp_arr):.3f}, '
      f'IQR = [{np.percentile(fp_arr,25):.3f}, {np.percentile(fp_arr,75):.3f}]')
print(f'c = 1-f_p: median = {np.median(c_arr):.3f}, '
      f'IQR = [{np.percentile(c_arr,25):.3f}, {np.percentile(c_arr,75):.3f}]')
print(f'gc/a0: median = {np.median(gc_arr):.3f}')

g_N_max_arr = np.array([r['g_N_max'] for r in results])
GS0_arr = np.array([r['GS0'] for r in results])
ratio_peak = g_N_max_arr / GS0_arr
print(f'\n[BT check] g_N_max / (G*Sigma0): median = {np.median(ratio_peak):.2f} (expect ~1.8)')

print('\n[3] Generating figures...')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(fp_arr, log_gc, s=12, alpha=0.6, c=log_GS0, cmap='viridis', edgecolors='none')
ax.set_xlabel('f_p (g_N < a0 mass fraction)', fontsize=11)
ax.set_ylabel('log(gc / a0)', fontsize=11)
ax.set_title(f'V-1: f_p vs gc  (N={N}, rho={rho1:.3f}, p={p1:.1e})', fontsize=12)
plt.colorbar(ax.collections[0], ax=ax, label='log(G*Sigma0/a0)')
z = np.polyfit(fp_arr, log_gc, 1)
xline = np.linspace(0, 1, 100)
ax.plot(xline, np.polyval(z, xline), 'r--', lw=1.5, label=f'linear fit: slope={z[0]:.2f}')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_fp_vs_gc.png'), dpi=150)
print('  -> fig_fp_vs_gc.png')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(sqrt_arr, c_arr, s=12, alpha=0.6, c='steelblue', edgecolors='none')
ax.set_xlabel('sqrt(G*Sigma0 / a0)', fontsize=11)
ax.set_ylabel('c = 1 - f_p', fontsize=11)
ax.set_title(f'c vs sqrt scaling  (rho={rho3:.3f})', fontsize=12)
xref = np.linspace(0, np.max(sqrt_arr)*1.1, 100)
ax.plot(xref, eta_eff * xref, 'r-', lw=1.5, label=f'c = {eta_eff:.2f} * sqrt(GS0/a0)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(log_GS0_v, log_c, s=12, alpha=0.6, c='darkorange', edgecolors='none')
ax.set_xlabel('log(G*Sigma0 / a0)', fontsize=11)
ax.set_ylabel('log(c)', fontsize=11)
ax.set_title(f'Slope: {slope:.3f} +/- {se_slope:.3f} (expect 0.5)', fontsize=12)
xfit = np.linspace(log_GS0_v.min(), log_GS0_v.max(), 100)
ax.plot(xfit, slope*xfit + intercept, 'r-', lw=1.5, label=f'fit: slope={slope:.3f}')
ax.plot(xfit, 0.5*xfit + intercept, 'b--', lw=1.0, label='theory: slope=0.5')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_c_vs_sqrt_scaling.png'), dpi=150)
print('  -> fig_c_vs_sqrt_scaling.png')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.hist(fp_arr, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(np.median(fp_arr), color='red', ls='--', label=f'median={np.median(fp_arr):.2f}')
ax.set_xlabel('f_p', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A: f_p distribution', fontsize=12)
ax.legend()

ax = axes[0, 1]
ax.scatter(c_arr, gc_arr, s=12, alpha=0.6, c='steelblue', edgecolors='none')
ax.set_xlabel('c = 1 - f_p', fontsize=11)
ax.set_ylabel('gc / a0', fontsize=11)
ax.set_title(f'B: c vs gc/a0  (rho={rho2:.3f})', fontsize=12)
xline = np.linspace(0, 1, 100)
ax.plot(xline, xline, 'r--', lw=1, label='gc/a0 = c')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.hist(ratio_peak[np.isfinite(ratio_peak)], bins=30, color='darkorange', alpha=0.7, edgecolor='white')
ax.axvline(1.8, color='red', ls='--', lw=2, label='BT: 1.8')
ax.axvline(np.median(ratio_peak), color='blue', ls='--', label=f'median={np.median(ratio_peak):.2f}')
ax.set_xlabel('g_N_max / (G*Sigma0)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('C: Binney-Tremaine check', fontsize=12)
ax.legend(fontsize=9)

ax = axes[1, 1]
if np.sum(valid) > 5:
    residual = log_c - (0.5 * log_GS0_v + intercept)
    ax.hist(residual, bins=25, color='green', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', ls='--')
    ax.set_xlabel('log(c) - [0.5*log(GS0/a0) + const]', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'D: Residual (sigma={np.std(residual):.3f} dex)', fontsize=12)

fig.suptitle(f'V-1: Condition-14 Mechanism Verification (N={N})', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_fp_mechanism_summary.png'), dpi=150)
print('  -> fig_fp_mechanism_summary.png')

outcsv = os.path.join(BASE, 'fp_verification_results.csv')
with open(outcsv, 'w', encoding='utf-8') as f:
    cols = ['galaxy', 'f_p', 'c_membrane', 'gc_over_a0', 'log_gc_a0',
            'vflat', 'Yd', 'hR_kpc', 'GS0', 'log_GS0_a0', 'sqrt_GS0_a0',
            'g_N_max', 'g_N_min', 'n_points']
    f.write(','.join(cols) + '\n')
    for r in results:
        f.write(','.join(str(r[c]) for c in cols) + '\n')
print(f'\n[4] Results saved: {outcsv}')

print('\n' + '='*60)
print('VERDICT')
print('='*60)

verdict_fp_gc = 'PASS' if (rho1 < -0.5 and p1 < 0.001) else 'FAIL'
verdict_slope = 'PASS' if abs(slope - 0.5) < 2*se_slope else 'MARGINAL' if abs(slope-0.5) < 3*se_slope else 'FAIL'
verdict_bt = 'PASS' if 1.0 < np.median(ratio_peak) < 3.0 else 'CHECK'

print(f'  [V-1a] f_p vs gc correlation:      {verdict_fp_gc}  (rho={rho1:.3f}, p={p1:.1e})')
print(f'  [V-1b] sqrt scaling slope:          {verdict_slope}  (slope={slope:.3f}+/-{se_slope:.3f}, expect 0.5)')
print(f'  [V-1c] BT coefficient check:        {verdict_bt}  (median ratio={np.median(ratio_peak):.2f}, expect ~1.8)')
print(f'  [V-2]  eta effective:               {eta_eff:.3f}')
print()

if verdict_fp_gc == 'PASS' and verdict_slope in ('PASS', 'MARGINAL'):
    print('  >>> Condition-14 mechanism SUPPORTED: f_p determines gc via sqrt scaling')
    print(f'  >>> c = 1-f_p ~ {eta_eff:.2f} * sqrt(G*Sigma0/a0)')
else:
    print('  >>> Results require further investigation')

print('\n[DONE]')
