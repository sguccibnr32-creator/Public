# -*- coding: utf-8 -*-
"""
v2.7 SPARC 全銀河解析
w_elastic = 0.772*(v_flat/100)^(-0.286)
w_total   = w_elastic / (1 - f_p)
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, curve_fit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 定数
# ============================================================
G_C      = 1.2e-10
KPC_TO_M = 3.086e19

# ============================================================
# コア関数
# ============================================================
def rs_selfconsistent(vflat_kms, r_kpc, v_disk_kms):
    vf2_si = (vflat_kms * 1e3)**2
    def equation(rs_kpc):
        vd = np.interp(rs_kpc, r_kpc, v_disk_kms)
        vd2_si = (vd * 1e3)**2
        rhs_kpc = (vd2_si + 0.5*vf2_si) / G_C / KPC_TO_M
        return rs_kpc - rhs_kpc
    try:
        f_lo = equation(0.01)
        f_hi = equation(500.0)
        if f_lo * f_hi > 0:
            return 0.5 * vf2_si / G_C / KPC_TO_M
        return brentq(equation, 0.01, 500.0, xtol=0.001, maxiter=200)
    except Exception:
        return 0.5 * vf2_si / G_C / KPC_TO_M

def w_elastic(vflat_kms):
    return 0.772 * (vflat_kms / 100)**(-0.286)

def w_total(vflat_kms, fp):
    fp = np.clip(fp, 0, 0.999)
    return w_elastic(vflat_kms) / (1 - fp)

def v_model_v27(r_kpc, vflat, fp, v_disk, v_gas, v_bul, rs_override=None):
    if rs_override is None:
        rs = rs_selfconsistent(vflat, r_kpc, v_disk)
    else:
        rs = rs_override
    w  = w_total(vflat, fp)
    x  = r_kpc / rs
    Fx = 0.5 * (1 + np.tanh(w * (x - 1)))
    v2 = v_disk**2 + v_gas**2 + v_bul**2 + vflat**2 * Fx
    return np.sqrt(np.clip(v2, 0, None)), rs

# ============================================================
# フィット
# ============================================================
def fit_v27(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1.0, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def chi2(params):
        vf, fp = params
        if not (10 < vf < 400): return 1e9
        if not (0 <= fp < 0.999): return 1e9
        vm, _ = v_model_v27(r, vf, fp, vd, vg, vb)
        return np.sum(((vobs - vm)/verr)**2)

    # grid search for initial values
    best_c2, best_p0 = 1e12, [100, 0.0]
    for vf0 in np.linspace(30, 350, 12):
        for fp0 in np.linspace(0, 0.95, 8):
            c = chi2([vf0, fp0])
            if c < best_c2:
                best_c2 = c
                best_p0 = [vf0, fp0]

    res = minimize(chi2, best_p0, method='Nelder-Mead',
                   options={'xatol':0.5, 'fatol':0.1, 'maxiter':5000})
    vf_best, fp_best = res.x
    fp_best = np.clip(fp_best, 0, 0.999)
    vm, rs = v_model_v27(r, vf_best, fp_best, vd, vg, vb)
    dof = max(len(r) - 2, 1)
    return {
        'vflat': vf_best, 'fp': fp_best, 'rs': rs,
        'w': w_total(vf_best, fp_best),
        'chi2': res.fun / dof, 'chi2_raw': res.fun, 'n_pts': len(r),
    }

def fit_tanh_ref(df):
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1.0, None)
    vd, vg, vb = df['Vdisk'].values, df['Vgas'].values, df['Vbul'].values

    def model(r, vflat, rs):
        x  = r / rs
        Fx = 0.5*(1+np.tanh(1.5*(x-1)))
        v2 = vd**2+vg**2+vb**2+vflat**2*Fx
        return np.sqrt(np.clip(v2, 0, None))
    try:
        popt, _ = curve_fit(model, r, vobs, p0=[100, 5],
                            bounds=([10,0.1],[400,200]),
                            sigma=verr, maxfev=5000)
        vm  = model(r, *popt)
        dof = max(len(r)-2, 1)
        c2  = np.sum(((vobs-vm)/verr)**2)
        return popt[0], popt[1], c2/dof, c2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# ============================================================
# メイン
# ============================================================
data_dir = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
out_dir  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン')
cols = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

files = sorted(data_dir.glob("*_rotmod.dat"))
print(f"処理対象：{len(files)} 銀河")
print(f"{'銀河':15s} {'vflat':>7} {'fp':>6} {'rs':>6} {'w':>5} {'chi2_v27':>9} {'chi2_tanh':>10}")
print("-"*70)

records = []
for i, fp_path in enumerate(files):
    name = fp_path.stem.replace('_rotmod','')
    try:
        df = pd.read_csv(fp_path, sep=r'\s+', comment='#', names=cols)
    except:
        continue
    if len(df) < 5:
        continue

    try:
        res = fit_v27(df)
    except:
        continue

    vft, rst, c2t, c2t_raw = fit_tanh_ref(df)

    # AIC
    n = res['n_pts']
    aic_v27  = res['chi2_raw'] + 2*2
    aic_tanh = c2t_raw + 2*2 if not np.isnan(c2t_raw) else np.nan
    daic = aic_v27 - aic_tanh if not np.isnan(aic_tanh) else np.nan

    records.append({
        'galaxy': name,
        'vflat': round(res['vflat'], 1),
        'fp': round(res['fp'], 4),
        'rs_v27': round(res['rs'], 2),
        'w': round(res['w'], 3),
        'chi2_v27': round(res['chi2'], 3),
        'vflat_tanh': round(vft, 1) if not np.isnan(vft) else np.nan,
        'rs_tanh': round(rst, 2) if not np.isnan(rst) else np.nan,
        'chi2_tanh': round(c2t, 3) if not np.isnan(c2t) else np.nan,
        'dAIC': round(daic, 1) if not np.isnan(daic) else np.nan,
        'n_pts': res['n_pts'],
    })

    if i < 10 or i % 20 == 0:
        c2ts = f"{c2t:10.3f}" if not np.isnan(c2t) else "       N/A"
        print(f"{name:15s} {res['vflat']:7.1f} {res['fp']:6.3f} "
              f"{res['rs']:6.2f} {res['w']:5.2f} {res['chi2']:9.3f} {c2ts}")

df_res = pd.DataFrame(records)
df_res.to_csv(out_dir/'v27_sparc_results.csv', index=False, encoding='utf-8-sig')

# ============================================================
# 特定銀河の確認
# ============================================================
print("\n" + "="*70)
print("=== 個別銀河確認 ===")
for gal in ['NGC3198','NGC2841','NGC2403','NGC6503','NGC4214','NGC1705','DDO154']:
    row = df_res[df_res['galaxy']==gal]
    if len(row) == 0:
        print(f"  {gal}: not found")
        continue
    r = row.iloc[0]
    print(f"  {gal:12s}: vflat={r['vflat']:6.1f}  fp={r['fp']:.3f}  "
          f"rs={r['rs_v27']:6.2f}  w={r['w']:.3f}  "
          f"chi2_v27={r['chi2_v27']:.3f}  chi2_tanh={r['chi2_tanh']}")

# ============================================================
# 統計サマリー
# ============================================================
print("\n" + "="*70)
print("=== 統計サマリー ===")
print(f"総銀河数：{len(df_res)}")

fp = df_res['fp'].dropna()
print(f"\nf_p 分布：")
print(f"  中央値：{fp.median():.3f}")
print(f"  平均  ：{fp.mean():.3f}")
print(f"  std   ：{fp.std():.3f}")
print(f"  f_p < 0.1（弾性支配）：{(fp < 0.1).sum()} 銀河")
print(f"  0.1 <= f_p < 0.5（混合）：{((fp>=0.1)&(fp<0.5)).sum()} 銀河")
print(f"  f_p >= 0.5（塑性支配）：{(fp >= 0.5).sum()} 銀河")

# chi2 比較
good = df_res[df_res['chi2_tanh'].notna() & (df_res['chi2_v27']<10) & (df_res['chi2_tanh']<10)]
print(f"\nchi2 < 10 の銀河（両モデル）：{len(good)} 件")
if len(good) > 0:
    print(f"  v2.7  chi2 中央値：{good['chi2_v27'].median():.3f}")
    print(f"  tanh  chi2 中央値：{good['chi2_tanh'].median():.3f}")

daic = df_res['dAIC'].dropna()
print(f"\ndAIC 分布（v2.7 - tanh）：")
print(f"  中央値：{daic.median():.1f}")
print(f"  dAIC < 0（v2.7 有利）：{(daic < 0).sum()} 銀河")
print(f"  |dAIC| < 2（同等）：{(daic.abs() < 2).sum()} 銀河")
print(f"  dAIC > 0（tanh 有利）：{(daic > 0).sum()} 銀河")

# v2.7 wins with chi2 < 5
v27_good = df_res[(df_res['chi2_v27'] < 5)]
print(f"\nchi2_v27 < 5：{len(v27_good)} 銀河 / {len(df_res)}")
v27_good2 = df_res[(df_res['chi2_v27'] < 2)]
print(f"chi2_v27 < 2：{len(v27_good2)} 銀河 / {len(df_res)}")

# ============================================================
# プロット
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

# --- 1: f_p histogram ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(fp, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(fp.median(), color='red', ls='--', lw=1.5,
            label=f'median={fp.median():.2f}')
ax1.set_xlabel('plastic fraction f_p')
ax1.set_ylabel('N galaxies')
ax1.set_title('f_p distribution (all SPARC)')
ax1.legend()

# --- 2: f_p vs v_flat ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(df_res['vflat'], df_res['fp'], alpha=0.5, s=25, c='steelblue')
# highlight key galaxies
for gal in ['NGC3198','NGC2841','NGC6503','NGC4214','NGC1705']:
    row = df_res[df_res['galaxy']==gal]
    if len(row) > 0:
        r = row.iloc[0]
        ax2.plot(r['vflat'], r['fp'], 'r*', ms=12)
        ax2.annotate(gal, (r['vflat'], r['fp']),
                     xytext=(4,4), textcoords='offset points', fontsize=7, color='red')

# trend line
from scipy.stats import linregress
mask_ok = (df_res['fp'] < 0.999) & (df_res['fp'] >= 0) & (df_res['vflat'] > 0)
if mask_ok.sum() > 5:
    sl, ic, rv, pv, _ = linregress(df_res.loc[mask_ok,'vflat'], df_res.loc[mask_ok,'fp'])
    vf_plot = np.linspace(20, 350, 200)
    ax2.plot(vf_plot, sl*vf_plot + ic, 'r--', lw=1.5,
             label=f'r={rv:.2f}, p={pv:.3f}')
    ax2.legend(fontsize=8)
ax2.set_xlabel('v_flat [km/s]')
ax2.set_ylabel('f_p')
ax2.set_title('f_p vs v_flat')

# --- 3: w distribution ---
ax3 = fig.add_subplot(gs[0, 2])
w_vals = df_res['w'].dropna()
w_clip = w_vals[w_vals < 10]
ax3.hist(w_clip, bins=25, color='coral', edgecolor='white', alpha=0.8)
ax3.axvline(w_clip.median(), color='red', ls='--', lw=1.5,
            label=f'median={w_clip.median():.2f}')
ax3.set_xlabel('w (effective)')
ax3.set_ylabel('N galaxies')
ax3.set_title('w_total distribution')
ax3.legend()

# --- 4: chi2 comparison ---
ax4 = fig.add_subplot(gs[1, 0])
mask_c = (df_res['chi2_v27'] < 30) & (df_res['chi2_tanh'] < 30) & df_res['chi2_tanh'].notna()
ax4.scatter(df_res.loc[mask_c,'chi2_tanh'], df_res.loc[mask_c,'chi2_v27'],
            alpha=0.5, s=20, c='steelblue')
lim = 25
ax4.plot([0,lim],[0,lim],'k--',lw=0.8)
ax4.set_xlabel('chi2/dof (tanh free)')
ax4.set_ylabel('chi2/dof (v2.7)')
ax4.set_title('Fit quality comparison')
ax4.set_xlim(0, lim)
ax4.set_ylim(0, lim)

# count above/below diagonal
n_below = ((df_res.loc[mask_c,'chi2_v27'] < df_res.loc[mask_c,'chi2_tanh'])).sum()
n_above = ((df_res.loc[mask_c,'chi2_v27'] > df_res.loc[mask_c,'chi2_tanh'])).sum()
ax4.text(2, 22, f'v2.7 better: {n_below}\ntanh better: {n_above}', fontsize=9)

# --- 5: dAIC distribution ---
ax5 = fig.add_subplot(gs[1, 1])
daic_clip = daic.clip(-100, 100)
ax5.hist(daic_clip, bins=30, color='coral', edgecolor='white', alpha=0.8)
ax5.axvline(0, color='k', ls='--', lw=1.0)
ax5.axvline(daic.median(), color='red', ls='-', lw=1.5,
            label=f'median={daic.median():.1f}')
ax5.set_xlabel('dAIC (v2.7 - tanh)')
ax5.set_ylabel('N galaxies')
ax5.set_title('dAIC distribution\n(negative = v2.7 wins)')
ax5.legend(fontsize=8)

# --- 6: rs comparison ---
ax6 = fig.add_subplot(gs[1, 2])
mask_r = (df_res['rs_v27'] < 50) & (df_res['rs_tanh'] < 50) & df_res['rs_tanh'].notna()
sc = ax6.scatter(df_res.loc[mask_r,'rs_tanh'], df_res.loc[mask_r,'rs_v27'],
                 c=df_res.loc[mask_r,'fp'], cmap='RdYlBu_r',
                 alpha=0.6, s=25, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax6, label='f_p')
ax6.plot([0,45],[0,45],'k--',lw=0.8, label='1:1')
ax6.set_xlabel('r_s (tanh) [kpc]')
ax6.set_ylabel('r_s (v2.7) [kpc]')
ax6.set_title('r_s comparison (color=f_p)')
ax6.legend(fontsize=8)
ax6.set_xlim(0, 45)
ax6.set_ylim(0, 45)

plt.suptitle('v2.7 Model: SPARC Full Sample Analysis', fontsize=14, fontweight='bold')
plt.savefig(out_dir/'v27_sparc_summary.png', dpi=150, bbox_inches='tight')
print("\nv27_sparc_summary.png saved")
