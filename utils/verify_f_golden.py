import numpy as np
from scipy import stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import glob, os, csv, warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 定数
# -----------------------------------------------
phi   = (1 + 5**0.5) / 2   # 黄金比
f_theory = phi / 2           # = 0.8090...
a0    = 1.2e-10              # m/s^2
kpc2m = 3.086e19             # m/kpc
kms2ms= 1e3

print(f"黄金比 phi = {phi:.6f}")
print(f"理論値 f = phi/2 = {f_theory:.6f}")
print(f"理論値 v_c(r_s)/v_flat = sqrt(phi+0.5) = {(phi+0.5)**0.5:.6f}")
print(f"理論値 v_bar(r_s)/v_flat = sqrt(phi/2) = {(phi/2)**0.5:.6f}")

# -----------------------------------------------
# v3.0 5銀河での検証
# -----------------------------------------------
v30_galaxies = {
    'IC2574'  : {'vflat': 56,  'rs': 7.7,  'Ups': 1.90, 'gc_a0': 0.060},
    'NGC0300' : {'vflat': 77,  'rs': 5.5,  'Ups': 1.67, 'gc_a0': 0.206},
    'NGC3198' : {'vflat': 119, 'rs': 11.0, 'Ups': 0.96, 'gc_a0': 0.397},
    'UGC02885': {'vflat': 225, 'rs': 37.2, 'Ups': 1.01, 'gc_a0': 0.391},
    'NGC2841' : {'vflat': 256, 'rs': 3.3,  'Ups': 0.45, 'gc_a0': 2.970},
}

print(f"\n{'='*70}")
print("  v3.0 5銀河：f = v^2_bar(r_s)/v^2_flat の実測値")
print(f"{'='*70}")
print(f"{'銀河':<12} {'gc/a0':>8} {'gc_calc':>9} "
      f"{'f_obs':>8} {'f_theory':>9} {'残差%':>8}")
print("-"*70)

f_obs_v30 = []
for name, d in v30_galaxies.items():
    vf  = d['vflat']
    rs  = d['rs']
    gc  = d['gc_a0']

    # gc = f*v^2_flat/r_s から f を逆算
    gc_SI   = gc * a0
    vf_SI   = vf * kms2ms
    rs_SI   = rs * kpc2m
    f_obs   = gc_SI * rs_SI / vf_SI**2

    # gc_calc = f_theory*v^2_flat/r_s
    gc_calc = f_theory * vf_SI**2 / rs_SI / a0

    resid_pct = (f_obs - f_theory) / f_theory * 100
    f_obs_v30.append(f_obs)

    print(f"{name:<12} {gc:>8.3f} {gc_calc:>9.3f} "
          f"{f_obs:>8.4f} {f_theory:>9.4f} {resid_pct:>+8.1f}%")

f_obs_arr = np.array(f_obs_v30)
print(f"\n実測f：中央値={np.median(f_obs_arr):.4f}  "
      f"平均={np.mean(f_obs_arr):.4f}  "
      f"std={np.std(f_obs_arr):.4f}")
print(f"理論f：{f_theory:.4f}  (phi/2)")
print(f"乖離：{(np.median(f_obs_arr)-f_theory)/f_theory*100:+.1f}%")

# -----------------------------------------------
# SPARC全銀河での r_s vs v_flat の検証
# -----------------------------------------------
print(f"\n{'='*70}")
print("  SPARC全銀河：r_s vs v_flat のスケーリング")
print(f"{'='*70}")

# sparc_results.csv を読む
csv_path = 'sparc_results.csv'
vflat_all, rs_all, gc_all = [], [], []
if not os.path.exists(csv_path):
    print(f"[SKIP] {csv_path} が見つかりません")
else:
    with open(csv_path, newline='', encoding='utf-8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            try:
                vf  = float(row['vflat'])  if row.get('vflat')  else None
                rs  = float(row['r_s'])    if row.get('r_s')    else None
                gc  = float(row.get('gc_ratio','') or 'nan')
                if None not in [vf, rs] and vf > 0 and rs > 0:
                    vflat_all.append(vf)
                    rs_all.append(rs)
                    gc_all.append(gc)
            except:
                pass

    vflat_arr = np.array(vflat_all)
    rs_arr    = np.array(rs_all)

    if len(vflat_arr) >= 10:
        log_vf = np.log10(vflat_arr)
        log_rs = np.log10(rs_arr)
        s, i, r, p, se = stats.linregress(log_vf, log_rs)
        print(f"N = {len(vflat_arr)}")
        print(f"r_s ∝ v_flat^{s:.3f} +/- {se:.3f}")
        print(f"r = {r:.3f},  p = {p:.4e}")
        print(f"\n期待値（g_c ∝ v_flat^1.32から）：r_s ∝ v_flat^0.68")
        print(f"実測値：r_s ∝ v_flat^{s:.3f}")
        print(f"乖離：{s - 0.68:+.3f}")

# -----------------------------------------------
# sparc_gc.csv からf_obsを全銀河で計算
# -----------------------------------------------
gc_csv = 'sparc_gc.csv'
f_obs_all_arr = None
if os.path.exists(gc_csv):
    print(f"\n{'='*70}")
    print("  SPARC全銀河：f_obs = gc*r_s/v_flat^2 の分布")
    print(f"{'='*70}")

    f_obs_all = []
    with open(gc_csv, newline='', encoding='utf-8') as gcf:
        reader = csv.DictReader(gcf)
        for row in reader:
            try:
                vf = float(row.get('vflat','nan'))
                rs = float(row.get('r_s','nan'))
                gc = float(row.get('gc_ratio','nan'))
                if all(np.isfinite([vf,rs,gc])) and vf>0 and rs>0 and gc>0:
                    gc_SI = gc * a0
                    vf_SI = vf * kms2ms
                    rs_SI = rs * kpc2m
                    f_obs_all.append(gc_SI * rs_SI / vf_SI**2)
            except:
                pass

    if f_obs_all:
        f_arr = np.array(f_obs_all)
        f_arr = f_arr[np.isfinite(f_arr) & (f_arr > 0) & (f_arr < 10)]
        f_obs_all_arr = f_arr
        print(f"N = {len(f_arr)}")
        print(f"f_obs：中央値={np.median(f_arr):.4f}  "
              f"平均={np.mean(f_arr):.4f}  "
              f"std={np.std(f_arr):.4f}")
        print(f"f理論値（phi/2）= {f_theory:.4f}")
        print(f"中央値の乖離：{(np.median(f_arr)-f_theory)/f_theory*100:+.1f}%")

        _, p_sw = stats.shapiro(np.log10(f_arr[:50]))
        print(f"log(f)の正規性（Shapiro-Wilk）p = {p_sw:.4f}")

# -----------------------------------------------
# プロット
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1: v_bar(r_s)^2/v_flat^2 の分布（v3.0 5銀河）
ax = axes[0]
names_v30 = list(v30_galaxies.keys())
colors_v30 = ['steelblue']*5
bars = ax.bar(names_v30, f_obs_v30, color=colors_v30, alpha=0.7)
ax.axhline(f_theory, color='red', lw=2, ls='--',
           label=f'f = phi/2 = {f_theory:.4f}')
ax.axhline(np.median(f_obs_v30), color='orange', lw=1.5, ls=':',
           label=f'median = {np.median(f_obs_v30):.4f}')
ax.set_ylabel('f = gc * r_s / v_flat^2')
ax.set_title('v3.0 5 galaxies\nf_obs vs f_theory (phi/2)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2: r_s vs v_flat（全銀河）
ax = axes[1]
if len(vflat_all) >= 10:
    ax.scatter(vflat_arr, rs_arr, c='steelblue', s=15, alpha=0.4)
    vf_line = np.logspace(np.log10(vflat_arr.min()),
                           np.log10(vflat_arr.max()), 100)
    # 実測スケーリング
    rs_fit = 10**i * vf_line**s
    ax.plot(vf_line, rs_fit, 'r-', lw=2,
            label=f'obs: r_s = v_flat^{s:.2f}')
    # 理論予測（指数0.68）
    norm = np.median(rs_arr) / np.median(vflat_arr)**0.68
    rs_theory = norm * vf_line**0.68
    ax.plot(vf_line, rs_theory, 'g--', lw=2,
            label='theory: r_s = v_flat^0.68')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('v_flat [km/s]')
    ax.set_ylabel('r_s [kpc]')
    ax.set_title(f'r_s vs v_flat (N={len(vflat_arr)})\n'
                 f'slope={s:.3f} (expect 0.68)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No SPARC data available',
            ha='center', va='center', transform=ax.transAxes)

# 3: g_c ∝ v_flat^1.32 再確認
ax = axes[2]
if os.path.exists(gc_csv):
    vf_gc, gc_gc = [], []
    with open(gc_csv, newline='', encoding='utf-8') as gcf:
        reader = csv.DictReader(gcf)
        for row in reader:
            try:
                vf = float(row.get('vflat','nan'))
                gc = float(row.get('gc_ratio','nan'))
                if np.isfinite(vf) and np.isfinite(gc) and vf>0 and gc>0:
                    vf_gc.append(vf)
                    gc_gc.append(gc)
            except:
                pass
    if vf_gc:
        vf_arr2 = np.array(vf_gc)
        gc_arr2 = np.array(gc_gc)
        ax.scatter(vf_arr2, gc_arr2, c='steelblue', s=10, alpha=0.3)

        # 観測フィット
        s2, i2, r2, p2, se2 = stats.linregress(
            np.log10(vf_arr2), np.log10(gc_arr2))
        vf_l = np.logspace(np.log10(vf_arr2.min()),
                            np.log10(vf_arr2.max()), 100)
        ax.plot(vf_l, 10**i2 * vf_l**s2, 'r-', lw=2,
                label=f'obs: slope={s2:.2f}')

        # 理論予測（新条件から）
        norm2 = np.median(gc_arr2) / np.median(vf_arr2)**1.32
        ax.plot(vf_l, norm2 * vf_l**1.32, 'g--', lw=2,
                label='theory: slope=1.32')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('v_flat [km/s]')
        ax.set_ylabel('g_c / a0')
        ax.set_title(f'g_c vs v_flat (N={len(vf_arr2)})\n'
                     f'r={r2:.3f} p={p2:.2e}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No gc data',
                ha='center', va='center', transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, 'sparc_gc.csv not found',
            ha='center', va='center', transform=ax.transAxes)

plt.suptitle(f'f = phi/2 = {f_theory:.4f} verification\n'
             'v_bar^2(r_s) = (phi/2) v_flat^2 as additional condition',
             fontsize=11)
plt.tight_layout()
plt.savefig('verify_f_golden.png', dpi=150)
plt.close()
print("\n-> verify_f_golden.png saved")

# -----------------------------------------------
# 理論的帰結のまとめ
# -----------------------------------------------
print(f"\n{'='*60}")
print("  理論的帰結のまとめ")
print('='*60)
print(f"\nv3.0 + 新条件 v^2_bar(r_s) = f*v^2_flat から：")
print(f"  f = phi/2 = {f_theory:.6f}  （内部整合から決定）")
print(f"")
print(f"  g_c = (phi/2) * v^2_flat / r_s")
print(f"")
print(f"  g_c ∝ v_flat^1.32 が成立する条件：")
print(f"    r_s ∝ v_flat^0.68")
print(f"")
print(f"  次の問い：r_s ∝ v_flat^0.68 は")
print(f"  v3.0の方程式系から導出できるか？")
print(f"  （または追加条件として組み込むか？）")
