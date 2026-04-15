import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv, os, sys, warnings
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 定数
# -----------------------------------------------
phi      = (1 + 5**0.5) / 2
f_theory = phi / 2          # = 0.8090
a0       = 1.2e-10          # m/s^2
kpc2m    = 3.086e19
kms2ms   = 1e3

# -----------------------------------------------
# CSVパス
# -----------------------------------------------
base_dir    = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
gc_csv      = os.path.join(base_dir, 'phase1', 'sparc_gc.csv')

if len(sys.argv) >= 3:
    results_csv = sys.argv[1]
    gc_csv      = sys.argv[2]

# -----------------------------------------------
# sparc_results.csv から vflat・r_s を読む
# -----------------------------------------------
def load_results(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"列名（{Path(path).name}）：{headers}")
        for row in reader:
            try:
                vf = float(row.get('vflat','') or 'nan')
                # rs1 が2-stage fitのr_s
                rs = float(row.get('rs2','') or row.get('rs1','') or 'nan')
                if np.isfinite(vf) and np.isfinite(rs) and vf>0 and rs>0:
                    rows.append({'vflat':vf, 'r_s':rs,
                                 'galaxy':row.get('galaxy','')})
            except:
                pass
    return rows

# sparc_gc.csv から gc_ratio・vflat を読む
def load_gc(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"列名（{Path(path).name}）：{headers}")
        for row in reader:
            try:
                vf  = float(row.get('vflat','') or 'nan')
                gc  = float(row.get('gc_ratio','') or 'nan')
                rs  = float(row.get('rs','') or 'nan')
                ups = float(row.get('upsilon','') or 'nan')
                T   = float(row.get('T_type','') or 'nan')
                if all(np.isfinite([vf,gc,rs])) and vf>0 and gc>0 and rs>0:
                    rows.append({'vflat':vf,'gc':gc,'r_s':rs,
                                 'upsilon':ups,'T_type':T,
                                 'galaxy':row.get('galaxy','')})
            except:
                pass
    return rows

# -----------------------------------------------
# メイン解析
# -----------------------------------------------
results_ok = os.path.exists(results_csv)
gc_ok      = os.path.exists(gc_csv)

if not results_ok:
    print(f"[ERROR] {results_csv} が見つかりません")
if not gc_ok:
    print(f"[ERROR] {gc_csv} が見つかりません")
if not (results_ok and gc_ok):
    sys.exit(1)

res_data = load_results(results_csv)
gc_data  = load_gc(gc_csv)

print(f"\n読み込み：sparc_results={len(res_data)}行, sparc_gc={len(gc_data)}行")

# -----------------------------------------------
# 解析1：r_s vs v_flat のスケーリング
# -----------------------------------------------
vf_res = np.array([d['vflat'] for d in res_data])
rs_res = np.array([d['r_s']   for d in res_data])

log_vf = np.log10(vf_res)
log_rs = np.log10(rs_res)
s_rs, i_rs, r_rs, p_rs, se_rs = stats.linregress(log_vf, log_rs)

print(f"\n{'='*60}")
print(f"  解析1：r_s ∝ v_flat^alpha")
print(f"{'='*60}")
print(f"N = {len(vf_res)}")
print(f"alpha（実測）= {s_rs:.4f} +/- {se_rs:.4f}")
print(f"alpha（期待）= 0.68  （g_c ∝ v_flat^1.32から）")
print(f"r = {r_rs:.4f},  p = {p_rs:.4e}")
print(f"乖離：{s_rs - 0.68:+.4f}")

# -----------------------------------------------
# 解析2：f_obs = gc * r_s / v_flat^2 の分布
# -----------------------------------------------
f_obs_all = []
vf_f, gc_f, rs_f, ups_f, T_f = [], [], [], [], []

for d in gc_data:
    gc_SI = d['gc'] * a0
    vf_SI = d['vflat'] * kms2ms
    rs_SI = d['r_s'] * kpc2m
    f_obs = gc_SI * rs_SI / vf_SI**2
    if np.isfinite(f_obs) and 0 < f_obs < 20:
        f_obs_all.append(f_obs)
        vf_f.append(d['vflat'])
        gc_f.append(d['gc'])
        rs_f.append(d['r_s'])
        ups_f.append(d['upsilon'])
        T_f.append(d['T_type'])

f_arr  = np.array(f_obs_all)
vf_arr = np.array(vf_f)
gc_arr = np.array(gc_f)
rs_arr = np.array(rs_f)
ups_arr= np.array(ups_f)
T_arr  = np.array(T_f)

print(f"\n{'='*60}")
print(f"  解析2：f_obs = gc * r_s / v_flat^2 の分布")
print(f"{'='*60}")
print(f"N = {len(f_arr)}")
print(f"中央値 = {np.median(f_arr):.4f}  （理論値：{f_theory:.4f}）")
print(f"平均   = {np.mean(f_arr):.4f}")
print(f"std    = {np.std(f_arr):.4f}")
print(f"CV     = {np.std(f_arr)/np.mean(f_arr):.4f}")
print(f"中央値の乖離：{(np.median(f_arr)-f_theory)/f_theory*100:+.2f}%")

# t検定：f_obs の平均が f_theory と有意に異なるか
t_stat, p_t = stats.ttest_1samp(f_arr, f_theory)
print(f"\nt検定（H0: mean=phi/2）：t={t_stat:.3f}, p={p_t:.4f}")
try:
    _, p_mw = stats.wilcoxon(f_arr - f_theory)
    print(f"Wilcoxon検定（H0: median=phi/2）：p={p_mw:.4f}")
except:
    print("Wilcoxon検定：実行できず")

# -----------------------------------------------
# 解析3：f_obs vs 銀河物性の相関
# -----------------------------------------------
print(f"\n{'='*60}")
print(f"  解析3：f_obs の銀河依存性")
print(f"{'='*60}")

valid_ups = np.isfinite(ups_arr) & (ups_arr > 0)
valid_T   = np.isfinite(T_arr)

r_ups, p_ups = np.nan, np.nan
r_T, p_T = np.nan, np.nan

if valid_ups.sum() >= 10:
    r_ups, p_ups = stats.spearmanr(ups_arr[valid_ups], f_arr[valid_ups])
    print(f"f vs Upsilon_d：Spearman r={r_ups:.4f}, p={p_ups:.4f}")

if valid_T.sum() >= 10:
    r_T, p_T = stats.spearmanr(T_arr[valid_T], f_arr[valid_T])
    print(f"f vs T_type  ：Spearman r={r_T:.4f}, p={p_T:.4f}")

r_vf, p_vf = stats.spearmanr(vf_arr, f_arr)
print(f"f vs v_flat  ：Spearman r={r_vf:.4f}, p={p_vf:.4f}")

r_rs2, p_rs2 = stats.spearmanr(rs_arr, f_arr)
print(f"f vs r_s     ：Spearman r={r_rs2:.4f}, p={p_rs2:.4f}")

# -----------------------------------------------
# 解析4：g_c ∝ v_flat^1.32 の再確認
# -----------------------------------------------
log_vf2 = np.log10(vf_arr)
log_gc2 = np.log10(gc_arr)
s_gc, i_gc, r_gc, p_gc, se_gc = stats.linregress(log_vf2, log_gc2)

print(f"\n{'='*60}")
print(f"  解析4：g_c ∝ v_flat^slope の再確認（gc_csvから）")
print(f"{'='*60}")
print(f"N = {len(vf_arr)}")
print(f"slope = {s_gc:.4f} +/- {se_gc:.4f}  （期待：1.32）")
print(f"r = {r_gc:.4f},  p = {p_gc:.4e}")

# -----------------------------------------------
# プロット
# -----------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1: r_s vs v_flat
ax = axes[0, 0]
ax.scatter(vf_res, rs_res, s=10, alpha=0.4, c='steelblue')
vfl = np.logspace(np.log10(vf_res.min()), np.log10(vf_res.max()), 100)
ax.plot(vfl, 10**i_rs * vfl**s_rs, 'r-', lw=2,
        label=f'obs: alpha={s_rs:.3f}')
norm_th = np.median(rs_res) / np.median(vf_res)**0.68
ax.plot(vfl, norm_th * vfl**0.68, 'g--', lw=2,
        label='expect: alpha=0.68')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('r_s [kpc]')
ax.set_title(f'r_s vs v_flat (N={len(vf_res)})\n'
             f'slope={s_rs:.3f} expect=0.68')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2: f_obs 分布
ax = axes[0, 1]
bins = np.linspace(0, 5, 50)
ax.hist(f_arr, bins=bins, color='steelblue', alpha=0.7,
        edgecolor='white')
ax.axvline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2 = {f_theory:.4f}')
ax.axvline(np.median(f_arr), color='orange', lw=2, ls=':',
           label=f'median = {np.median(f_arr):.4f}')
ax.axvline(np.mean(f_arr), color='green', lw=1.5, ls='-.',
           label=f'mean = {np.mean(f_arr):.4f}')
ax.set_xlabel('f = gc * r_s / v_flat^2')
ax.set_ylabel('count')
ax.set_title(f'f_obs distribution (N={len(f_arr)})\n'
             f'CV={np.std(f_arr)/np.mean(f_arr):.3f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# 3: f_obs vs v_flat
ax = axes[0, 2]
sc = ax.scatter(vf_arr, f_arr, s=10, alpha=0.4,
                c=np.log10(vf_arr), cmap='viridis')
ax.axhline(f_theory, color='red', lw=1.5, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.axhline(np.median(f_arr), color='orange', lw=1, ls=':')
plt.colorbar(sc, ax=ax, label='log10(v_flat)')
ax.set_xscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('f_obs')
ax.set_title(f'f vs v_flat\nSpearman r={r_vf:.3f} p={p_vf:.4f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4: f_obs vs Upsilon
ax = axes[1, 0]
if valid_ups.sum() >= 10:
    ax.scatter(ups_arr[valid_ups], f_arr[valid_ups],
               s=10, alpha=0.4, c='steelblue')
    ax.axhline(f_theory, color='red', lw=1.5, ls='--',
               label=f'phi/2={f_theory:.4f}')
    ax.set_xlabel('Upsilon_d')
    ax.set_ylabel('f_obs')
    ax.set_title(f'f vs Upsilon\nSpearman r={r_ups:.3f} p={p_ups:.4f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient Upsilon data',
            ha='center', va='center', transform=ax.transAxes)

# 5: f_obs vs T_type
ax = axes[1, 1]
if valid_T.sum() >= 10:
    T_unique = np.unique(T_arr[valid_T])
    f_by_T   = [f_arr[valid_T & (T_arr==t)] for t in T_unique]
    ax.boxplot(f_by_T, positions=T_unique, widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.5))
    ax.axhline(f_theory, color='red', lw=1.5, ls='--',
               label=f'phi/2={f_theory:.4f}')
    ax.set_xlabel('Hubble T type')
    ax.set_ylabel('f_obs')
    ax.set_title(f'f vs T_type\nSpearman r={r_T:.3f} p={p_T:.4f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient T_type data',
            ha='center', va='center', transform=ax.transAxes)

# 6: g_c vs v_flat 再確認
ax = axes[1, 2]
ax.scatter(vf_arr, gc_arr, s=10, alpha=0.3, c='steelblue')
vfl2 = np.logspace(np.log10(vf_arr.min()),
                    np.log10(vf_arr.max()), 100)
ax.plot(vfl2, 10**i_gc * vfl2**s_gc, 'r-', lw=2,
        label=f'obs: {s_gc:.3f}')
norm3 = np.median(gc_arr) / np.median(vf_arr)**1.32
ax.plot(vfl2, norm3 * vfl2**1.32, 'g--', lw=2,
        label='theory: 1.32')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('g_c / a0')
ax.set_title(f'g_c vs v_flat (N={len(vf_arr)})\n'
             f'slope={s_gc:.3f} r={r_gc:.3f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Full SPARC verification: f = phi/2 = {f_theory:.4f}\n'
             f'v_bar^2(r_s) = (phi/2) v_flat^2',
             fontsize=12)
plt.tight_layout()
plt.savefig('verify_f_full_sparc.png', dpi=150)
plt.close()
print("\n-> verify_f_full_sparc.png saved")

# -----------------------------------------------
# 最終サマリー
# -----------------------------------------------
print(f"\n{'='*60}")
print("  最終サマリー")
print('='*60)
print(f"\n【新条件：v^2_bar(r_s) = f*v^2_flat, f = phi/2】")
print(f"  f理論値  = {f_theory:.4f}")
print(f"  f中央値  = {np.median(f_arr):.4f}  "
      f"（乖離{(np.median(f_arr)-f_theory)/f_theory*100:+.1f}%）")
print(f"  f CV     = {np.std(f_arr)/np.mean(f_arr):.4f}")
print(f"\n【r_s ∝ v_flat^alpha】")
print(f"  alpha実測 = {s_rs:.3f}  （期待：0.68）")
print(f"\n【g_c ∝ v_flat^slope】")
print(f"  slope実測 = {s_gc:.3f}  （期待：1.32）")
print(f"\n【整合条件】")
print(f"  alpha + slope = {s_rs + s_gc:.3f}  （理論：0.68+1.32=2.00）")
print(f"  -> 2に近いほど g_c = f*v_flat^2/r_s と整合")
