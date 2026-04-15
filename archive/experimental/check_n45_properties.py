"""
g_c = a0 で T-3 r_s が見つかった45銀河の特性を調べる
-> どんな銀河がMOND遷移半径をデータ範囲内に持つか
"""
import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt
import csv, os, sys, warnings
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

a0    = 1.2e-10
kpc2m = 3.086e19
kms2ms= 1e3

base_dir    = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
rotmod_dir  = os.path.join(base_dir, 'Rotmod_LTG')

def load_sparc_results(path):
    rows = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gal  = row['galaxy']
                vf   = float(row.get('vflat','')     or 'nan')
                rs   = float(row.get('rs2','') or row.get('rs1','') or 'nan')
                ups  = float(row.get('ud','')         or 'nan')
                chi2 = float(row.get('chi2_1s','')    or 'nan')
                if np.isfinite(vf) and np.isfinite(rs) and vf>0 and rs>0:
                    rows[gal] = {'vflat':vf,'r_s':rs,
                                 'upsilon':ups,'chi2':chi2}
            except:
                pass
    return rows

def load_baryon_profile(filepath, upsilon_d, upsilon_b=0.7):
    try:
        data = np.loadtxt(filepath, comments='#')
    except:
        return None
    if data.ndim==1 or len(data)<4:
        return None
    mask = data[:,1]>0
    data = data[mask]
    if len(data)<4:
        return None
    r     = data[:,0]
    v_gas = data[:,3]
    v_disk= data[:,4]
    v_bul = data[:,5] if data.shape[1]>5 else np.zeros_like(r)
    v2_bar= (np.sign(v_gas)*v_gas**2
           + upsilon_d*np.sign(v_disk)*v_disk**2
           + upsilon_b*np.sign(v_bul)*v_bul**2)
    v2_bar= np.maximum(v2_bar, 0)
    v_bar = np.sqrt(v2_bar)
    g_N   = v2_bar * kms2ms**2 / (r * kpc2m)
    v_obs = data[:,1]
    return dict(r=r, g_N=g_N, v_bar=v_bar, v_obs=v_obs,
                g_N_max=g_N.max(), g_N_min=g_N.min())

params_all = load_sparc_results(results_csv)

found_t3  = []   # g_N = a0 の交差あり
no_cross_high = []  # 全域 g_N > a0（ニュートン）
no_cross_low  = []  # 全域 g_N < a0（MOND）

for gal, params in params_all.items():
    dat_path = os.path.join(rotmod_dir, f'{gal}_rotmod.dat')
    if not os.path.exists(dat_path):
        continue
    ups  = max(params['upsilon'] if np.isfinite(params['upsilon']) else 0.5, 0.05)
    bary = load_baryon_profile(dat_path, upsilon_d=ups)
    if bary is None:
        continue

    r   = bary['r']
    g_N = bary['g_N']

    # g_N = a0 の交差を探す
    crossings = []
    for i in range(len(r)-1):
        d1 = g_N[i]   - a0
        d2 = g_N[i+1] - a0
        if d1*d2 <= 0 and d1!=d2:
            r_cross = r[i]+(r[i+1]-r[i])*(-d1)/(d2-d1)
            crossings.append(r_cross)

    rec = dict(
        galaxy  = gal,
        vflat   = params['vflat'],
        r_s_tanh= params['r_s'],
        upsilon = ups,
        chi2    = params['chi2'],
        g_N_max = bary['g_N_max']/a0,
        g_N_min = bary['g_N_min']/a0,
        r_max   = r.max(),
    )

    if crossings:
        rs_t3 = crossings[0]
        try:
            fi = interpolate.interp1d(r, bary['v_bar'],
                                       kind='linear',
                                       fill_value='extrapolate')
            vbar_rs = float(fi(rs_t3))
        except:
            continue
        if vbar_rs > 0:
            rec['rs_t3']   = rs_t3
            rec['f_obs']   = (vbar_rs/params['vflat'])**2
            found_t3.append(rec)
    elif g_N.min() > a0:
        no_cross_high.append(rec)
    else:
        no_cross_low.append(rec)

# -----------------------------------------------
# 統計比較
# -----------------------------------------------
print(f"\n{'='*65}")
print(f"  45銀河の特性分析")
print(f"{'='*65}")

groups = [
    ('T-3交差あり', found_t3),
    ('全域g_N>a0(Newton)', no_cross_high),
    ('全域g_N<a0(MOND)',   no_cross_low),
]

print(f"\n{'グループ':<25} {'N':>4} {'v_flat中央値':>13} "
      f"{'g_N_max/a0中央値':>16} {'g_N_min/a0中央値':>16}")
print('-'*80)

for label, recs in groups:
    if not recs:
        continue
    vf   = np.array([r['vflat']   for r in recs])
    gnmx = np.array([r['g_N_max'] for r in recs])
    gnmn = np.array([r['g_N_min'] for r in recs])
    print(f"{label:<25} {len(recs):>4} {np.median(vf):>13.1f} "
          f"{np.median(gnmx):>16.3f} {np.median(gnmn):>16.4f}")

# -----------------------------------------------
# T-3交差あり銀河の詳細
# -----------------------------------------------
phi      = (1+5**0.5)/2
f_theory = phi/2

f_arr  = np.array([r['f_obs']    for r in found_t3])
vf_arr = np.array([r['vflat']    for r in found_t3])
rs_t3  = np.array([r['rs_t3']   for r in found_t3])
rs_th  = np.array([r['r_s_tanh']for r in found_t3])
rs_rat = rs_th / rs_t3

print(f"\n{'='*65}")
print(f"  T-3交差あり銀河：詳細 (N={len(found_t3)})")
print(f"{'='*65}")
print(f"f_obs：中央値={np.median(f_arr):.4f}  "
      f"std={np.std(f_arr):.4f}  "
      f"（理論値phi/2={f_theory:.4f}）")
print(f"rs_tanh/rs_t3：中央値={np.median(rs_rat):.3f}  "
      f"std={np.std(rs_rat):.3f}")

# f_obs vs v_flat
r_fvf, p_fvf = stats.spearmanr(vf_arr, f_arr)
print(f"\nf vs v_flat：Spearman r={r_fvf:+.4f}, p={p_fvf:.4f}")

# r_s^T3 スケーリング
s_t3, i_t3, r_t3, p_t3, se_t3 = stats.linregress(
    np.log10(vf_arr), np.log10(rs_t3))
print(f"r_s^T3 ∝ v_flat^{s_t3:.3f} +/- {se_t3:.3f}  r={r_t3:.3f}")

# -----------------------------------------------
# プロット
# -----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1: 3グループのv_flat分布
ax = axes[0, 0]
vf_t3  = [r['vflat'] for r in found_t3]
vf_high= [r['vflat'] for r in no_cross_high]
vf_low = [r['vflat'] for r in no_cross_low]
bins   = np.logspace(1, 3, 25)
ax.hist(vf_t3,   bins=bins, alpha=0.7, color='steelblue',
        label=f'T3 crossing N={len(vf_t3)}')
ax.hist(vf_high, bins=bins, alpha=0.5, color='red',
        label=f'g_N>a0 all N={len(vf_high)}')
ax.hist(vf_low,  bins=bins, alpha=0.5, color='green',
        label=f'g_N<a0 all N={len(vf_low)}')
ax.set_xscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('count')
ax.set_title('v_flat distribution by group')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2: f_obs vs v_flat（45銀河）
ax = axes[0, 1]
ax.scatter(vf_arr, f_arr, s=30, alpha=0.6, c='steelblue')
ax.axhline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.axhline(np.median(f_arr), color='orange', lw=1.5, ls=':',
           label=f'median={np.median(f_arr):.4f}')
ax.set_xscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('f = v_bar(r_s^T3)^2/v_flat^2')
ax.set_title(f'f vs v_flat (N={len(found_t3)})\n'
             f'Spearman r={r_fvf:+.3f} p={p_fvf:.4f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 2)

# 3: r_s^T3 vs v_flat
ax = axes[1, 0]
ax.scatter(vf_arr, rs_t3, s=20, alpha=0.6, c='steelblue',
           label=f'r_s^T3: slope={s_t3:.3f}')
ax.scatter(vf_arr, rs_th, s=10, alpha=0.3, c='orange',
           marker='^', label='r_s_tanh')
vfl = np.logspace(np.log10(vf_arr.min()),
                   np.log10(vf_arr.max()), 100)
ax.plot(vfl, 10**i_t3*vfl**s_t3, 'b-', lw=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('r_s [kpc]')
ax.set_title(f'r_s^T3 ~ v_flat^{s_t3:.3f}\n'
             f'tanh: 0.795  expect: 0.68')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4: rs_tanh/rs_t3 分布
ax = axes[1, 1]
ax.hist(rs_rat[rs_rat<20], bins=30, color='steelblue', alpha=0.7)
ax.axvline(np.median(rs_rat), color='red', lw=2, ls='--',
           label=f'median={np.median(rs_rat):.2f}')
ax.axvline(1.0, color='k', lw=1, ls=':',
           label='ratio=1 (match)')
ax.set_xlabel('r_s_tanh / r_s^(T3)')
ax.set_ylabel('count')
ax.set_title(f'tanh r_s is ~{np.median(rs_rat):.1f}x T-3 r_s\n'
             f'(median={np.median(rs_rat):.2f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('N=45: Properties of galaxies with T-3 r_s in data range\n'
             '(g_c = a0, crossing g_N = a0 exists within observed r)',
             fontsize=11)
plt.tight_layout()
plt.savefig('n45_properties.png', dpi=150)
plt.close()
print("\n-> n45_properties.png saved")

print(f"\n{'='*65}")
print("  結論")
print('='*65)
print(f"T-3交差あり（N={len(found_t3)}）：中間質量銀河")
print(f"全域g_N>a0  （N={len(no_cross_high)}）：高面輝度・大質量銀河")
print(f"全域g_N<a0  （N={len(no_cross_low)}）：低面輝度・矮小銀河")
print(f"\nMOND遷移半径がデータ範囲内に存在するのは")
print(f"「中間的な表面輝度を持つ銀河」に限られる")
