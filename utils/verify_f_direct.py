"""
f = v^2_bar(r_s) / v^2_flat を循環論法なしで検証する

手順：
1. sparc_results.csvからr_s（フィット値）を取得
2. 各銀河の_rotmod.datファイルからv_barを直接読む
3. r = r_s でのv_barを線形補間
4. f_obs = v_bar(r_s)^2 / v_flat^2 を計算
5. phi/2 = 0.8090 との比較
"""
import numpy as np
from scipy import stats, interpolate
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
f_theory = phi / 2    # = 0.8090
print(f"検証目標：f = phi/2 = {f_theory:.6f}")

# -----------------------------------------------
# パス設定
# -----------------------------------------------
base_dir    = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
rotmod_dir  = os.path.join(base_dir, 'Rotmod_LTG')

# -----------------------------------------------
# sparc_results.csv 読み込み
# -----------------------------------------------
def load_sparc_results(path):
    """r_s, vflat, upsilon を取得"""
    rows = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gal   = row['galaxy']
                vflat = float(row.get('vflat','') or 'nan')
                # rs2 = 2-stage fit の r_s
                rs    = float(row.get('rs2','') or row.get('rs1','') or 'nan')
                ups   = float(row.get('ud','') or 'nan')
                grade = row.get('grade','')
                if np.isfinite(vflat) and np.isfinite(rs) and vflat>0 and rs>0:
                    rows[gal] = {'vflat':vflat, 'r_s':rs,
                                 'upsilon':ups, 'grade':grade}
            except:
                pass
    return rows

# -----------------------------------------------
# SPARCデータから v_bar(r_s) を直接計算
# -----------------------------------------------
def load_sparc_dat(filepath, upsilon_d, upsilon_b=0.7):
    """
    SPARC _rotmod.datファイルを読み込み、バリオン合計速度を計算
    列：r, v_obs, v_err, v_gas, v_disk, v_bul

    v_bar^2 = v_gas^2 + Ups_d * v_disk^2 + Ups_b * v_bul^2
    """
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception:
        return None

    if data.ndim == 1 or len(data) < 3:
        return None

    # v_obs > 0 のデータのみ
    mask = data[:, 1] > 0
    data = data[mask]
    if len(data) < 3:
        return None

    r      = data[:, 0]
    v_obs  = data[:, 1]
    v_gas  = data[:, 3]
    v_disk = data[:, 4]
    v_bul  = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)

    # バリオン速度^2（符号保存）
    v2_bar = (np.sign(v_gas)  * v_gas**2
            + upsilon_d * np.sign(v_disk) * v_disk**2
            + upsilon_b * np.sign(v_bul)  * v_bul**2)

    # 負にならないよう制限
    v2_bar = np.maximum(v2_bar, 0)
    v_bar  = np.sqrt(v2_bar)

    return dict(r=r, v_obs=v_obs, v_bar=v_bar, v2_bar=v2_bar)

def interpolate_at(r, v, r_target):
    """r_target での v を線形補間で取得"""
    if len(r) < 2:
        return np.nan
    try:
        f_interp = interpolate.interp1d(
            r, v, kind='linear', fill_value='extrapolate')
        return float(f_interp(r_target))
    except:
        return np.nan

# -----------------------------------------------
# メイン処理
# -----------------------------------------------
if not os.path.exists(results_csv):
    print(f"[ERROR] {results_csv} が見つかりません")
    sys.exit(1)
if not os.path.exists(rotmod_dir):
    print(f"[ERROR] {rotmod_dir} が見つかりません")
    sys.exit(1)

sparc_params = load_sparc_results(results_csv)
print(f"sparc_results.csv：{len(sparc_params)}銀河")

# -----------------------------------------------
# 各銀河でf_obsを計算
# -----------------------------------------------
records = []

for gal, params in sparc_params.items():
    dat_path = os.path.join(rotmod_dir, f'{gal}_rotmod.dat')
    if not os.path.exists(dat_path):
        continue

    ups  = params['upsilon']
    if not np.isfinite(ups):
        ups = 0.5
    ups = max(ups, 0.05)

    d = load_sparc_dat(dat_path, upsilon_d=ups)
    if d is None:
        continue

    r    = d['r']
    vbar = d['v_bar']
    vflat= params['vflat']
    rs   = params['r_s']

    # r_sでのv_barを補間
    vbar_rs = interpolate_at(r, vbar, rs)
    if not np.isfinite(vbar_rs) or vbar_rs <= 0:
        continue

    f_obs = (vbar_rs / vflat)**2

    # r_sでのv_obsも取得（比較用）
    vobs_rs = interpolate_at(r, d['v_obs'], rs)

    records.append(dict(
        galaxy   = gal,
        vflat    = vflat,
        r_s      = rs,
        vbar_rs  = vbar_rs,
        vobs_rs  = vobs_rs,
        f_obs    = f_obs,
        upsilon  = ups,
        grade    = params['grade'],
        r_max    = r.max(),
        extrapolated = (rs < r.min() or rs > r.max())
    ))

# -----------------------------------------------
# 結果の分析
# -----------------------------------------------
n_total      = len(records)
n_interp     = sum(1 for rec in records if not rec['extrapolated'])
n_extrap     = sum(1 for rec in records if rec['extrapolated'])

f_arr  = np.array([rec['f_obs']  for rec in records])
vf_arr = np.array([rec['vflat']  for rec in records])
rs_arr = np.array([rec['r_s']    for rec in records])
ups_arr= np.array([rec['upsilon']for rec in records])

# 外挿を除いたデータで主解析
mask_interp = np.array([not rec['extrapolated'] for rec in records])
f_i    = f_arr[mask_interp]
vf_i   = vf_arr[mask_interp]

print(f"\n{'='*65}")
print(f"  f_obs = v^2_bar(r_s) / v^2_flat：独立検証結果")
print(f"{'='*65}")
print(f"総数：{n_total}銀河（補間：{n_interp}、外挿：{n_extrap}）")
print(f"\n【全銀河（N={len(f_arr)}）】")
print(f"  中央値 = {np.median(f_arr):.4f}  （理論値 phi/2={f_theory:.4f}）")
print(f"  平均   = {np.mean(f_arr):.4f}")
print(f"  std    = {np.std(f_arr):.4f}")
print(f"  CV     = {np.std(f_arr)/np.mean(f_arr):.4f}")
print(f"  乖離   = {(np.median(f_arr)-f_theory)/f_theory*100:+.2f}%")

if len(f_i) >= 5:
    print(f"\n【補間のみ（N={len(f_i)}）】")
    print(f"  中央値 = {np.median(f_i):.4f}")
    print(f"  平均   = {np.mean(f_i):.4f}")
    print(f"  std    = {np.std(f_i):.4f}")

# 統計検定
t_stat, p_t = stats.ttest_1samp(f_arr, f_theory)
print(f"\n統計検定（H0: f = phi/2 = {f_theory:.4f}）：")
print(f"  t検定   ：t={t_stat:.3f}, p={p_t:.4f}")
try:
    _, p_wilcox = stats.wilcoxon(f_arr - f_theory)
    print(f"  Wilcoxon：p={p_wilcox:.4f}")
except:
    p_wilcox = np.nan
    print("  Wilcoxon：実行できず")

if p_t > 0.05:
    print(f"  -> f = phi/2 を棄却できない（p={p_t:.4f} > 0.05）")
else:
    print(f"  -> f = phi/2 を棄却（p={p_t:.4f} < 0.05）")
    print(f"  -> 実測中央値 {np.median(f_arr):.4f} が正しい補正値")

# -----------------------------------------------
# f_obs の銀河依存性
# -----------------------------------------------
print(f"\n{'='*65}")
print(f"  f_obs の銀河依存性（系統的偏りの確認）")
print(f"{'='*65}")

r_vf, p_vf = stats.spearmanr(vf_arr, f_arr)
r_rs, p_rs = stats.spearmanr(rs_arr, f_arr)
r_up, p_up = stats.spearmanr(ups_arr, f_arr)
print(f"  f vs v_flat  ：Spearman r={r_vf:+.4f}, p={p_vf:.4f}")
print(f"  f vs r_s     ：Spearman r={r_rs:+.4f}, p={p_rs:.4f}")
print(f"  f vs Upsilon ：Spearman r={r_up:+.4f}, p={p_up:.4f}")

# -----------------------------------------------
# f_obs -> 修正された g_c 指数の計算
# -----------------------------------------------
f_median = np.median(f_arr)
print(f"\n{'='*65}")
print(f"  f = {f_median:.4f}（実測中央値）を使った場合の予言")
print(f"{'='*65}")
print(f"  g_c = {f_median:.4f} * v_flat^2 / r_s")
print(f"  r_s ∝ v_flat^0.795（実測）を代入：")
print(f"  g_c ∝ v_flat^{2 - 0.795:.3f} = v_flat^1.205")
print(f"  観測値：g_c ∝ v_flat^1.318")
print(f"  残差：{1.318 - (2-0.795):+.4f}")

# -----------------------------------------------
# プロット
# -----------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1: f_obs 分布（ヒストグラム）
ax = axes[0, 0]
f_plot = f_arr[f_arr < np.percentile(f_arr, 95)]
ax.hist(f_plot, bins=30, color='steelblue',
        alpha=0.7, edgecolor='white')
ax.axvline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.axvline(np.median(f_arr), color='orange', lw=2, ls=':',
           label=f'median={np.median(f_arr):.4f}')
ax.axvline(np.mean(f_arr), color='green', lw=1.5, ls='-.',
           label=f'mean={np.mean(f_arr):.4f}')
ax.set_xlabel('f = v_bar(r_s)^2 / v_flat^2')
ax.set_ylabel('count')
ax.set_title(f'f_obs distribution (N={len(f_arr)})\n'
             f'DIRECT measurement (no circular argument)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2: f_obs vs v_flat
ax = axes[0, 1]
colors = ['red' if rec['extrapolated'] else 'steelblue'
          for rec in records]
ax.scatter(vf_arr, f_arr, s=15, alpha=0.5, c=colors)
ax.axhline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.axhline(np.median(f_arr), color='orange', lw=1.5, ls=':',
           label=f'median={np.median(f_arr):.4f}')
ax.set_xscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('f_obs')
ax.set_title(f'f vs v_flat (blue=interp, red=extrap)\n'
             f'Spearman r={r_vf:+.3f} p={p_vf:.4f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(np.percentile(f_arr, 97), 5))

# 3: v_bar(r_s) vs v_flat
ax = axes[0, 2]
vbar_rs_arr = np.array([rec['vbar_rs'] for rec in records])
ax.scatter(vf_arr, vbar_rs_arr, s=15, alpha=0.5,
           c='steelblue', label='v_bar(r_s)')
vf_line = np.linspace(vf_arr.min(), vf_arr.max(), 100)
ax.plot(vf_line, (f_theory)**0.5 * vf_line, 'r--', lw=2,
        label=f'sqrt(phi/2)*v_flat ({f_theory**0.5:.3f}*v)')
ax.plot(vf_line, vf_line, 'k:', lw=1, alpha=0.5,
        label='v_bar=v_flat')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('v_bar(r_s) [km/s]')
ax.set_title('v_bar at r_s vs v_flat\n'
             f'theory: v_bar(r_s) = {f_theory**0.5:.3f} * v_flat')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4: f_obs vs r_s
ax = axes[1, 0]
ax.scatter(rs_arr, f_arr, s=15, alpha=0.5, c='steelblue')
ax.axhline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.set_xscale('log')
ax.set_xlabel('r_s [kpc]')
ax.set_ylabel('f_obs')
ax.set_title(f'f vs r_s\nSpearman r={r_rs:+.3f} p={p_rs:.4f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(np.percentile(f_arr, 97), 5))

# 5: f_obs vs Upsilon
ax = axes[1, 1]
ax.scatter(ups_arr, f_arr, s=15, alpha=0.5, c='steelblue')
ax.axhline(f_theory, color='red', lw=2, ls='--',
           label=f'phi/2={f_theory:.4f}')
ax.set_xlabel('Upsilon_d')
ax.set_ylabel('f_obs')
ax.set_title(f'f vs Upsilon\nSpearman r={r_up:+.3f} p={p_up:.4f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(np.percentile(f_arr, 97), 5))

# 6: サマリー
ax = axes[1, 2]
ax.axis('off')

p_w_str = f"{p_wilcox:.4f}" if np.isfinite(p_wilcox) else "N/A"
summary = [
    f"f = v_bar(r_s)^2 / v_flat^2",
    f"  Direct measurement",
    f"",
    f"N = {n_total} ({n_interp} interp, {n_extrap} extrap)",
    f"",
    f"f_obs median = {np.median(f_arr):.4f}",
    f"f_obs mean   = {np.mean(f_arr):.4f}",
    f"f_obs std    = {np.std(f_arr):.4f}",
    f"phi/2        = {f_theory:.4f}",
    f"deviation    = {(np.median(f_arr)-f_theory)/f_theory*100:+.1f}%",
    f"",
    f"t-test  p = {p_t:.4f}",
    f"Wilcoxon p = {p_w_str}",
    f"",
    f"f vs v_flat: r={r_vf:+.3f}",
    f"f vs r_s   : r={r_rs:+.3f}",
    f"f vs Ups   : r={r_up:+.3f}",
]

for i, line in enumerate(summary):
    ax.text(0.05, 0.97 - i*0.055, line,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

ax.set_title('Summary')

plt.suptitle(f'f = v_bar(r_s)^2 / v_flat^2: Direct verification\n'
             f'Theory: f = phi/2 = {f_theory:.4f}',
             fontsize=12)
plt.tight_layout()
plt.savefig('verify_f_direct.png', dpi=150)
plt.close()
print("\n-> verify_f_direct.png saved")

# -----------------------------------------------
# 個別銀河の詳細（上位・下位各5銀河）
# -----------------------------------------------
records_sorted = sorted(records, key=lambda x: x['f_obs'])
print(f"\n{'='*65}")
print("  f_obs の小さい銀河 Top5")
print(f"{'='*65}")
print(f"{'銀河':<14} {'f_obs':>7} {'v_flat':>8} "
      f"{'r_s':>6} {'v_bar(rs)':>10} {'外挿':>5}")
print('-'*55)
for rec in records_sorted[:5]:
    print(f"{rec['galaxy']:<14} {rec['f_obs']:>7.4f} "
          f"{rec['vflat']:>8.1f} {rec['r_s']:>6.2f} "
          f"{rec['vbar_rs']:>10.2f} "
          f"{'Yes' if rec['extrapolated'] else 'No':>5}")

print(f"\n  f_obs の大きい銀河 Top5")
print(f"{'='*65}")
print(f"{'銀河':<14} {'f_obs':>7} {'v_flat':>8} "
      f"{'r_s':>6} {'v_bar(rs)':>10} {'外挿':>5}")
print('-'*55)
for rec in records_sorted[-5:]:
    print(f"{rec['galaxy']:<14} {rec['f_obs']:>7.4f} "
          f"{rec['vflat']:>8.1f} {rec['r_s']:>6.2f} "
          f"{rec['vbar_rs']:>10.2f} "
          f"{'Yes' if rec['extrapolated'] else 'No':>5}")
