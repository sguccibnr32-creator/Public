# -*- coding: utf-8 -*-
"""
全SPARC銀河で (vflat, w) 2パラメータフィット（rs自己無撞着固定）
w_eff の分布が二峰性か連続かを判定
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

G_C      = 1.2e-10
KPC_TO_M = 3.086e19

def rs_selfconsistent(vflat_kms, r_kpc, v_disk_kms):
    vf2_si = (vflat_kms * 1e3)**2
    def equation(rs_kpc):
        vd = np.interp(rs_kpc, r_kpc, v_disk_kms)
        vd2_si = (vd * 1e3)**2
        return rs_kpc - (vd2_si + 0.5*vf2_si) / G_C / KPC_TO_M
    try:
        f_lo, f_hi = equation(0.01), equation(500.0)
        if f_lo * f_hi > 0:
            return 0.5 * vf2_si / G_C / KPC_TO_M
        return brentq(equation, 0.01, 500.0, xtol=0.001, maxiter=200)
    except:
        return 0.5 * vf2_si / G_C / KPC_TO_M

def fit_tanh_w_free(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def chi2(params):
        vf, w = params
        if not (10 < vf < 500): return 1e9
        if not (0.05 < w < 30): return 1e9
        rs = rs_selfconsistent(vf, r, vd)
        if rs < 0.001: return 1e9
        x  = r / rs
        Fx = 0.5*(1+np.tanh(w*(x-1)))
        v2 = vd**2 + vg**2 + vb**2 + vf**2*Fx
        vm = np.sqrt(np.clip(v2, 0, None))
        return np.sum(((vobs-vm)/verr)**2)

    # Grid search for initial values
    best_c2, best_p0 = 1e12, [100, 1.5]
    for vf0 in np.linspace(20, 350, 12):
        for w0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            c = chi2([vf0, w0])
            if c < best_c2:
                best_c2 = c
                best_p0 = [vf0, w0]

    res = minimize(chi2, best_p0, method='Nelder-Mead',
                   options={'xatol':0.1, 'fatol':0.05, 'maxiter':5000})
    vf, w = res.x
    w = np.clip(w, 0.05, 30)
    rs = rs_selfconsistent(vf, r, vd)
    dof = max(len(r)-2, 1)
    return vf, rs, w, res.fun/dof, res.fun

# ============================================================
data_dir = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
out_dir  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン')
cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

files = sorted(data_dir.glob("*_rotmod.dat"))
print(f"Processing {len(files)} galaxies...")

records = []
for i, fp in enumerate(files):
    name = fp.stem.replace('_rotmod','')
    try:
        df = pd.read_csv(fp, sep=r'\s+', comment='#', names=cols)
    except:
        continue
    if len(df) < 5:
        continue
    try:
        vf, rs, w, c2dof, c2raw = fit_tanh_w_free(df)
    except:
        continue

    records.append({
        'galaxy': name, 'vflat': round(vf,1),
        'rs': round(rs,2), 'w_eff': round(w,3),
        'chi2_dof': round(c2dof,3), 'n_pts': len(df),
    })
    if i < 5 or i % 25 == 0:
        print(f"  [{i+1:3d}] {name:15s} vf={vf:6.1f} rs={rs:6.2f} w={w:5.3f} chi2={c2dof:.3f}")

df_res = pd.DataFrame(records)
df_res.to_csv(out_dir/'w_eff_sparc_results.csv', index=False, encoding='utf-8-sig')

# ============================================================
# 統計
# ============================================================
w = df_res['w_eff'].values
# Exclude extreme boundary hits
w_good = w[(w > 0.06) & (w < 29)]

print(f"\n{'='*60}")
print(f"=== w_eff 統計（N={len(df_res)}、境界除外後 N={len(w_good)}）===")
print(f"  中央値 : {np.median(w_good):.3f}")
print(f"  平均   : {np.mean(w_good):.3f}")
print(f"  std    : {np.std(w_good):.3f}")
print(f"  min    : {np.min(w_good):.3f}")
print(f"  max    : {np.max(w_good):.3f}")
print(f"  25%ile : {np.percentile(w_good,25):.3f}")
print(f"  75%ile : {np.percentile(w_good,75):.3f}")

print(f"\n  w < 1.0            : {(w_good < 1.0).sum()} galaxies")
print(f"  1.0 <= w <= 3.0    : {((w_good>=1.0)&(w_good<=3.0)).sum()} galaxies")
print(f"  w > 3.0            : {(w_good > 3.0).sum()} galaxies")
print(f"  w > 5.0            : {(w_good > 5.0).sum()} galaxies")

# Boundary hits
print(f"\n  w boundary hit (<=0.06 or >=29): {len(w)-len(w_good)} galaxies")

# Key galaxies
print(f"\n=== 個別銀河 ===")
for gal in ['NGC3198','NGC2841','NGC2403','NGC6503','NGC4214','NGC1705','DDO154']:
    row = df_res[df_res['galaxy']==gal]
    if len(row):
        r = row.iloc[0]
        print(f"  {gal:12s}: vf={r['vflat']:6.1f} rs={r['rs']:6.2f} w={r['w_eff']:6.3f} chi2={r['chi2_dof']:.3f}")

# ============================================================
# Bimodality test (Hartigan's dip test approximation)
# Use kernel density + peak detection
# ============================================================
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

print(f"\n=== 二峰性検定 ===")
kde = gaussian_kde(w_good, bw_method=0.2)
x_kde = np.linspace(0, min(15, w_good.max()+1), 1000)
y_kde = kde(x_kde)
peaks, props = find_peaks(y_kde, height=0.01, distance=50)
print(f"  KDE peaks at w = {', '.join([f'{x_kde[p]:.2f}' for p in peaks])}")
print(f"  Number of peaks: {len(peaks)}")

# Valley between peaks
if len(peaks) >= 2:
    i1, i2 = peaks[0], peaks[1]
    valley_idx = i1 + np.argmin(y_kde[i1:i2])
    print(f"  Valley at w = {x_kde[valley_idx]:.2f}")
    print(f"  Peak1 height = {y_kde[peaks[0]]:.4f}")
    print(f"  Valley height = {y_kde[valley_idx]:.4f}")
    print(f"  Peak2 height = {y_kde[peaks[1]]:.4f}")
    ratio = y_kde[valley_idx] / min(y_kde[peaks[0]], y_kde[peaks[1]])
    print(f"  Valley/peak ratio = {ratio:.3f}")
    if ratio < 0.5:
        print(f"  → 二峰性あり（valley/peak < 0.5）")
    else:
        print(f"  → 二峰性なし（valley/peak >= 0.5、連続分布）")

# ============================================================
# プロット
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: w_eff histogram
ax = axes[0]
ax.hist(w_good, bins=30, color='steelblue', edgecolor='white', alpha=0.8,
        density=True, label='histogram')
ax.plot(x_kde, y_kde, 'r-', lw=2, label='KDE')
for p in peaks:
    ax.axvline(x_kde[p], color='red', ls='--', lw=1, alpha=0.7)
ax.axvline(np.median(w_good), color='orange', ls='-', lw=1.5,
           label=f'median={np.median(w_good):.2f}')
ax.set_xlabel('w_eff (effective steepness)', fontsize=11)
ax.set_ylabel('density', fontsize=11)
ax.set_title(f'w_eff distribution (N={len(w_good)})', fontsize=11)
ax.set_xlim(0, min(15, np.percentile(w_good, 98)+2))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 2: w_eff vs v_flat
ax = axes[1]
vf_vals = df_res['vflat'].values
mask = (w > 0.06) & (w < 29)
ax.scatter(vf_vals[mask], w[mask], alpha=0.4, s=20, c='steelblue')
# Highlight key galaxies
for gal, color in [('NGC3198','red'),('NGC2841','green'),('NGC6503','orange'),
                   ('NGC4214','purple'),('NGC1705','magenta')]:
    row = df_res[df_res['galaxy']==gal]
    if len(row):
        r = row.iloc[0]
        ax.plot(r['vflat'], r['w_eff'], '*', ms=14, color=color)
        ax.annotate(gal, (r['vflat'], r['w_eff']),
                    xytext=(4,4), textcoords='offset points', fontsize=7, color=color)

# w_elastic reference curve
vf_plot = np.linspace(20, 350, 200)
w_el = 0.772 * (vf_plot/100)**(-0.286)
ax.plot(vf_plot, w_el, 'k--', lw=1.5, label='w_elastic (4-gal fit)')
ax.set_xlabel('v_flat [km/s]', fontsize=11)
ax.set_ylabel('w_eff', fontsize=11)
ax.set_title('w_eff vs v_flat', fontsize=11)
ax.set_ylim(0, min(15, np.percentile(w[mask], 98)+2))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: chi2 vs w_eff
ax = axes[2]
c2 = df_res['chi2_dof'].values
ax.scatter(w[mask], c2[mask], alpha=0.4, s=20, c='steelblue')
ax.set_xlabel('w_eff', fontsize=11)
ax.set_ylabel('chi2/dof', fontsize=11)
ax.set_title('Fit quality vs w_eff', fontsize=11)
ax.set_ylim(0, 30)
ax.set_xlim(0, min(15, np.percentile(w[mask], 98)+2))
ax.axhline(2, color='green', ls='--', lw=0.8, label='chi2=2')
ax.axhline(5, color='orange', ls='--', lw=0.8, label='chi2=5')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.suptitle('w_eff (model-free): SPARC Full Sample', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir/'w_eff_sparc_summary.png', dpi=150, bbox_inches='tight')
print(f"\nw_eff_sparc_summary.png saved")
