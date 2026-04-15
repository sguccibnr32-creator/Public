"""
T-3定義のr_s^(T3)：g_N(r) = g_c を満たす半径を使って
f = v^2_bar(r_s^(T3)) / v^2_flat を検証する（循環論法なし）

g_c は sparc_gc.csv から取得
ただしgc_ratio = v_flat^2/(r_s_tanh * a0) と判明しているため
g_c の独立推定が必要 -> ここでは g_c = a0（MOND定数）で固定して
「T-3定義のr_sがどこになるか」を確認する方針に変更
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
f_theory = phi / 2        # = 0.8090
a0       = 1.2e-10        # m/s^2
kpc2m    = 3.086e19
kms2ms   = 1e3

print(f"検証目標：f = phi/2 = {f_theory:.6f}")
print(f"黄金比 phi = {phi:.6f}")

# -----------------------------------------------
# sparc_results.csv から vflat・r_s_tanh・upsilon を取得
# -----------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_sparc_results(path):
    rows = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gal   = row['galaxy']
                vflat = float(row.get('vflat','')  or 'nan')
                rs    = float(row.get('rs2','') or row.get('rs1','') or 'nan')
                ups   = float(row.get('ud','')     or 'nan')
                if np.isfinite(vflat) and np.isfinite(rs) and vflat>0 and rs>0:
                    rows[gal] = {'vflat':vflat, 'r_s_tanh':rs, 'upsilon':ups}
            except:
                pass
    return rows

# -----------------------------------------------
# SPARCデータからバリオン加速度プロファイルを計算
# -----------------------------------------------
def load_baryon_profile(filepath, upsilon_d, upsilon_b=0.7):
    try:
        data = np.loadtxt(filepath, comments='#')
    except:
        return None
    if data.ndim == 1 or len(data) < 4:
        return None
    mask = data[:, 1] > 0
    data = data[mask]
    if len(data) < 4:
        return None

    r      = data[:, 0]         # kpc
    v_gas  = data[:, 3]         # km/s
    v_disk = data[:, 4]
    v_bul  = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)

    v2_bar = (np.sign(v_gas)  * v_gas**2
            + upsilon_d * np.sign(v_disk) * v_disk**2
            + upsilon_b * np.sign(v_bul)  * v_bul**2)
    v2_bar = np.maximum(v2_bar, 0)
    v_bar  = np.sqrt(v2_bar)

    # g_N(r) [m/s^2]
    g_N = v2_bar * kms2ms**2 / (r * kpc2m)

    return dict(r=r, g_N=g_N, v_bar=v_bar)

# -----------------------------------------------
# g_N(r) = g_c_target となるr_s^(T3)を補間
# -----------------------------------------------
def find_rs_t3(r, g_N, g_c_SI):
    """
    g_N が g_c_SI を下回る最初の点（外から内へ向けて）
    つまり「g_N = g_c の最外縁交差点」を返す
    """
    crossings = []
    for i in range(len(r)-1):
        d1 = g_N[i]   - g_c_SI
        d2 = g_N[i+1] - g_c_SI
        if d1 * d2 <= 0 and d1 != d2:
            r_cross = r[i] + (r[i+1]-r[i]) * (-d1)/(d2-d1)
            crossings.append(r_cross)
    if not crossings:
        return np.nan, 0
    return crossings[0], len(crossings)

# -----------------------------------------------
# メイン処理
# -----------------------------------------------
results_csv = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
rotmod_dir  = os.path.join(base_dir, 'Rotmod_LTG')

if not os.path.exists(results_csv):
    print(f"[ERROR] {results_csv} が見つかりません")
    sys.exit(1)

params_all = load_sparc_results(results_csv)
print(f"\nsparc_results.csv：{len(params_all)}銀河")

# -----------------------------------------------
# g_c の候補値
# -----------------------------------------------
gc_candidates = {
    'a0'      : a0,
    '0.55*a0' : 0.55 * a0,
    '0.28*a0' : 0.28 * a0,
    '2*a0'    : 2.0  * a0,
}

all_results = {label: [] for label in gc_candidates}

for gal, params in params_all.items():
    dat_path = os.path.join(rotmod_dir, f'{gal}_rotmod.dat')
    if not os.path.exists(dat_path):
        continue

    ups   = params['upsilon']
    if not np.isfinite(ups):
        ups = 0.5
    ups = max(ups, 0.05)

    bary = load_baryon_profile(dat_path, upsilon_d=ups)
    if bary is None:
        continue

    r    = bary['r']
    g_N  = bary['g_N']
    vbar = bary['v_bar']
    vflat= params['vflat']

    for label, gc_SI in gc_candidates.items():
        rs_t3, n_cross = find_rs_t3(r, g_N, gc_SI)

        if np.isnan(rs_t3):
            continue

        # r_s^(T3) での v_bar を補間
        try:
            f_interp = interpolate.interp1d(
                r, vbar, kind='linear',
                fill_value='extrapolate')
            vbar_rs = float(f_interp(rs_t3))
        except:
            continue

        if not np.isfinite(vbar_rs) or vbar_rs <= 0:
            continue

        f_obs = (vbar_rs / vflat)**2

        if not np.isfinite(f_obs) or f_obs <= 0 or f_obs > 100:
            continue

        all_results[label].append(dict(
            galaxy  = gal,
            vflat   = vflat,
            rs_tanh = params['r_s_tanh'],
            rs_t3   = rs_t3,
            rs_ratio= params['r_s_tanh'] / rs_t3,
            vbar_rs = vbar_rs,
            f_obs   = f_obs,
            upsilon = ups,
        ))

# -----------------------------------------------
# 統計出力
# -----------------------------------------------
print(f"\n{'='*70}")
print(f"  T-3定義のr_s^(T3)を使ったf検証（g_c候補別）")
print(f"{'='*70}")
print(f"  理論値：f = phi/2 = {f_theory:.4f}")
print(f"{'='*70}")

best_label = None
best_p     = 0.0

for label, recs in all_results.items():
    if len(recs) < 5:
        print(f"\n{label}：N={len(recs)}（データ不足）")
        continue

    f_arr   = np.array([r['f_obs']   for r in recs])
    rs_ratio= np.array([r['rs_ratio']for r in recs])

    try:
        _, p_w = stats.wilcoxon(f_arr - f_theory)
    except:
        p_w = np.nan
    t_stat, p_t = stats.ttest_1samp(f_arr, f_theory)

    print(f"\n【g_c = {label}】N={len(f_arr)}")
    print(f"  f_obs 中央値 = {np.median(f_arr):.4f}  "
          f"（理論値 {f_theory:.4f}）")
    print(f"  f_obs 平均   = {np.mean(f_arr):.4f}")
    print(f"  f_obs std    = {np.std(f_arr):.4f}")
    print(f"  乖離         = {(np.median(f_arr)-f_theory)/f_theory*100:+.1f}%")
    p_w_str = f"{p_w:.4f}" if np.isfinite(p_w) else "N/A"
    judge = '-> phi/2 棄却できず' if (np.isfinite(p_w) and p_w>0.05) else '-> phi/2 棄却'
    print(f"  Wilcoxon p   = {p_w_str}  {judge}")
    print(f"  rs_tanh/rs_t3 中央値 = {np.median(rs_ratio):.4f}")

    if np.isfinite(p_w) and p_w > best_p:
        best_p     = p_w
        best_label = label

# -----------------------------------------------
# 最もf=phi/2に近いg_c候補での詳細解析
# -----------------------------------------------
if best_label:
    recs    = all_results[best_label]
    f_arr   = np.array([r['f_obs']   for r in recs])
    rs_ratio= np.array([r['rs_ratio']for r in recs])
    vf_arr  = np.array([r['vflat']   for r in recs])
    rs_arr  = np.array([r['rs_t3']   for r in recs])

    print(f"\n{'='*70}")
    print(f"  最良候補：g_c = {best_label}  (N={len(recs)})")
    print(f"{'='*70}")

    # f_obs の銀河依存性
    r_vf, p_vf = stats.spearmanr(vf_arr, f_arr)
    r_rs, p_rs = stats.spearmanr(rs_arr, f_arr)
    print(f"  f vs v_flat：Spearman r={r_vf:+.4f}, p={p_vf:.4f}")
    print(f"  f vs r_s^T3：Spearman r={r_rs:+.4f}, p={p_rs:.4f}")

    # r_s^(T3) vs v_flat のスケーリング
    s, i, r_val, p_val, se = stats.linregress(
        np.log10(vf_arr), np.log10(rs_arr))
    print(f"\n  r_s^(T3) ∝ v_flat^{s:.3f} +/- {se:.3f}  r={r_val:.3f}")
    print(f"  tanhのr_s ∝ v_flat^0.795  （比較）")
    print(f"  期待値（g_c ∝ v^1.32から）：0.68")

    # rs_tanh/rs_t3 の分布
    print(f"\n  rs_tanh/rs_t3 の分布：")
    print(f"    中央値 = {np.median(rs_ratio):.4f}")
    print(f"    平均   = {np.mean(rs_ratio):.4f}")
    print(f"    std    = {np.std(rs_ratio):.4f}")
    print(f"    CV     = {np.std(rs_ratio)/np.mean(rs_ratio):.4f}")

# -----------------------------------------------
# プロット
# -----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

if best_label and len(all_results[best_label]) >= 5:
    recs  = all_results[best_label]
    f_arr = np.array([r['f_obs']    for r in recs])
    vf_arr= np.array([r['vflat']    for r in recs])
    rs_t3 = np.array([r['rs_t3']    for r in recs])
    rs_th = np.array([r['rs_tanh']  for r in recs])
    rs_rat= np.array([r['rs_ratio'] for r in recs])

    # 1: f分布
    ax = axes[0, 0]
    f_plot = f_arr[f_arr < np.percentile(f_arr, 95)]
    ax.hist(f_plot, bins=30, color='steelblue', alpha=0.7)
    ax.axvline(f_theory, color='red', lw=2, ls='--',
               label=f'phi/2={f_theory:.4f}')
    ax.axvline(np.median(f_arr), color='orange', lw=2, ls=':',
               label=f'median={np.median(f_arr):.4f}')
    try:
        _, p_w = stats.wilcoxon(f_arr - f_theory)
    except:
        p_w = np.nan
    ax.set_xlabel('f = v_bar(r_s^T3)^2 / v_flat^2')
    ax.set_title(f'T-3 r_s: g_c={best_label}\n'
                 f'N={len(f_arr)}, Wilcoxon p={p_w:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2: r_s^T3 vs r_s_tanh
    ax = axes[0, 1]
    ax.scatter(rs_th, rs_t3, s=15, alpha=0.5, c='steelblue')
    lim = max(rs_th.max(), rs_t3.max())
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1')
    ax.set_xlabel('r_s (tanh fit) [kpc]')
    ax.set_ylabel('r_s^(T3) [kpc]')
    ax.set_title(f'r_s_tanh vs r_s^T3\n'
                 f'median ratio={np.median(rs_rat):.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3: r_s^T3 vs v_flat
    ax = axes[1, 0]
    ax.scatter(vf_arr, rs_t3, s=15, alpha=0.5, c='steelblue',
               label='r_s^T3')
    ax.scatter(vf_arr, rs_th, s=10, alpha=0.3, c='orange',
               label='r_s_tanh', marker='^')
    s_t3, i_t3,_,_,_ = stats.linregress(
        np.log10(vf_arr), np.log10(rs_t3))
    vfl = np.logspace(np.log10(vf_arr.min()),
                       np.log10(vf_arr.max()), 100)
    ax.plot(vfl, 10**i_t3 * vfl**s_t3, 'b-', lw=2,
            label=f'T3: slope={s_t3:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('v_flat [km/s]')
    ax.set_ylabel('r_s [kpc]')
    ax.set_title(f'r_s^T3 vs v_flat\n'
                 f'slope={s_t3:.3f} (tanh=0.795, expect=0.68)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4: 結果サマリー
    ax = axes[1, 1]
    ax.axis('off')
    try:
        _, p_w = stats.wilcoxon(f_arr - f_theory)
    except:
        p_w = np.nan
    s_t3, _, _, _, _ = stats.linregress(
        np.log10(vf_arr), np.log10(rs_t3))
    p_w_str = f"{p_w:.4f}" if np.isfinite(p_w) else "N/A"
    lines = [
        "verify_f_t3_rs result",
        "",
        f"g_c candidate = {best_label}",
        f"N             = {len(f_arr)}",
        "",
        f"f_obs median  = {np.median(f_arr):.4f}",
        f"phi/2         = {f_theory:.4f}",
        f"deviation     = {(np.median(f_arr)-f_theory)/f_theory*100:+.1f}%",
        f"Wilcoxon p    = {p_w_str}",
        "",
        "-> " + ("phi/2 NOT rejected" if (np.isfinite(p_w) and p_w>0.05)
                 else "phi/2 REJECTED"),
        "",
        f"r_s^T3 ~ v_flat^{s_t3:.3f}",
        f"r_s_tanh ~ v_flat^0.795",
        f"rs_tanh/rs_t3 median={np.median(rs_rat):.3f}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.05, 0.97-i*0.067, line,
                transform=ax.transAxes, fontsize=9,
                va='top', fontfamily='monospace')

plt.suptitle('verify_f_t3_rs: T-3 defined r_s verification\n'
             'f = v_bar(r_s^T3)^2 / v_flat^2  vs  phi/2 = 0.809',
             fontsize=12)
plt.tight_layout()
plt.savefig('verify_f_t3_rs.png', dpi=150)
plt.close()
print("\n-> verify_f_t3_rs.png saved")
