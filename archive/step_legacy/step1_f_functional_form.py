#!/usr/bin/env python3
"""
Step 1: f^(tanh)(v_flat) の関数形特定
"""
import csv
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. データ読み込み
# ============================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)

detail_path = os.path.join(base_dir, "f_tanh_robustness_detail.csv")
if not os.path.exists(detail_path):
    print("[ERROR] f_tanh_robustness_detail.csv not found")
    sys.exit(1)

_, detail_rows = load_csv(detail_path)
print(f"[INFO] f_tanh_robustness_detail.csv: {len(detail_rows)} rows")

# sparc_results.csv for extra columns
extra = {}
src_path = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
if os.path.exists(src_path):
    _, src_rows = load_csv(src_path)
    for row in src_rows:
        name = row.get('galaxy', '').strip()
        if name:
            extra[name] = row

# ============================================================
# 2. 解析用配列
# ============================================================
names = []
v_flat = []
f_tanh = []
r_s = []
upsilon_d = []
is_extrap = []
grade_arr = []

for d in detail_rows:
    gal = d['galaxy'].strip()
    vf = float(d['v_flat'])
    ft = float(d['f_tanh'])
    rs = float(d['r_s_tanh'])
    ud = float(d['upsilon_d'])
    ext = d['is_extrap'].strip()

    names.append(gal)
    v_flat.append(vf)
    f_tanh.append(ft)
    r_s.append(rs)
    upsilon_d.append(ud)
    is_extrap.append(ext == 'True')

    gr = extra.get(gal, {}).get('grade', '')
    grade_arr.append(gr)

v_flat = np.array(v_flat)
f_tanh = np.array(f_tanh)
r_s = np.array(r_s)
upsilon_d = np.array(upsilon_d)
is_extrap = np.array(is_extrap)
N_total = len(v_flat)

mask_in = ~is_extrap
mask_pos = f_tanh > 0
mask_main = mask_in & mask_pos

print(f"\n[INFO] N_total={N_total}, non-extrap={mask_in.sum()}, f>0={mask_pos.sum()}, main={mask_main.sum()}")

# ============================================================
# 3. 関数形フィッティング
# ============================================================
lv = np.log10(v_flat[mask_main])
lf = np.log10(f_tanh[mask_main])
vf_m = v_flat[mask_main]
ft_m = f_tanh[mask_main]
N_fit = mask_main.sum()

print(f"\n{'='*70}")
print(f"Function form fitting (N={N_fit}, non-extrap, f>0)")
print(f"{'='*70}")

results = {}

# Model 1: power law
p1 = np.polyfit(lv, lf, 1)
pred1 = np.polyval(p1, lv)
rss1 = np.sum((lf - pred1)**2)
bic1 = N_fit * np.log(rss1/N_fit) + 2*np.log(N_fit)
r1 = np.corrcoef(lf, pred1)[0,1]
results['power_law'] = {'params': p1, 'rss': rss1, 'bic': bic1, 'r': r1, 'k': 2}
print(f"\nModel 1: power law  log(f) = {p1[1]:.3f} + {p1[0]:.3f}*log(v)")
print(f"  -> f ~ v^{p1[0]:.3f}")
print(f"  r={r1:.4f}, RSS={rss1:.3f}, BIC={bic1:.1f}")

# Model 2: log-quadratic
p2 = np.polyfit(lv, lf, 2)
pred2 = np.polyval(p2, lv)
rss2 = np.sum((lf - pred2)**2)
bic2 = N_fit * np.log(rss2/N_fit) + 3*np.log(N_fit)
r2 = np.corrcoef(lf, pred2)[0,1]
results['log_quadratic'] = {'params': p2, 'rss': rss2, 'bic': bic2, 'r': r2, 'k': 3}
lv_min = -p2[1]/(2*p2[0]) if abs(p2[0]) > 1e-10 else np.nan
print(f"\nModel 2: log-quad  log(f) = {p2[2]:.3f} + {p2[1]:.3f}*log(v) + {p2[0]:.3f}*log(v)^2")
if np.isfinite(lv_min):
    print(f"  minimum at log(v)={lv_min:.3f} -> v_flat={10**lv_min:.1f} km/s")
print(f"  r={r2:.4f}, RSS={rss2:.3f}, BIC={bic2:.1f}")

# Model 0: constant
pred0 = np.mean(lf) * np.ones_like(lf)
rss0 = np.sum((lf - pred0)**2)
bic0 = N_fit * np.log(rss0/N_fit) + 1*np.log(N_fit)
results['constant'] = {'rss': rss0, 'bic': bic0, 'k': 1}
print(f"\nModel 0: constant  f = 10^{np.mean(lf):.3f} = {10**np.mean(lf):.3f}")
print(f"  RSS={rss0:.3f}, BIC={bic0:.1f}")

# Model 4: broken line
def broken_line(lv, a, b1, b2, lv_break):
    return np.where(lv < lv_break,
                    a + b1*(lv - lv_break),
                    a + b2*(lv - lv_break))
try:
    p4, _ = curve_fit(broken_line, lv, lf, p0=[np.median(lf), -0.5, 0.5, 1.9], maxfev=10000)
    pred4 = broken_line(lv, *p4)
    rss4 = np.sum((lf - pred4)**2)
    bic4 = N_fit * np.log(rss4/N_fit) + 4*np.log(N_fit)
    r4 = np.corrcoef(lf, pred4)[0,1]
    results['broken_line'] = {'params': p4, 'rss': rss4, 'bic': bic4, 'r': r4, 'k': 4}
    print(f"\nModel 4: broken line  break at v={10**p4[3]:.1f} km/s")
    print(f"  slope_low={p4[1]:.3f}, slope_high={p4[2]:.3f}")
    print(f"  r={r4:.4f}, RSS={rss4:.3f}, BIC={bic4:.1f}")
except Exception as e:
    print(f"\nModel 4: broken line fit failed ({e})")

# BIC comparison
print(f"\n--- BIC comparison (lower=better) ---")
for nm, res in sorted(results.items(), key=lambda x: x[1]['bic']):
    print(f"  {nm:20s}: BIC={res['bic']:8.1f}  (k={res['k']})")

# ============================================================
# 4. Correlations
# ============================================================
print(f"\n{'='*70}")
print("Correlations")
print(f"{'='*70}")

ud_m = upsilon_d[mask_main]
rs_m = r_s[mask_main]

rho_ud, p_ud = spearmanr(ft_m, ud_m)
print(f"\nf vs Ups_d:  Spearman rho={rho_ud:.3f}, p={p_ud:.2e}")

rho_rs, p_rs = spearmanr(ft_m, rs_m)
print(f"f vs r_s:    Spearman rho={rho_rs:.3f}, p={p_rs:.2e}")

rho_uv, p_uv = spearmanr(ud_m, vf_m)
print(f"Ups_d vs v_flat: Spearman rho={rho_uv:.3f}, p={p_uv:.2e}")

# Multivariate
lud = np.log10(np.maximum(ud_m, 0.01))
X = np.column_stack([np.ones(N_fit), lv, lud])
beta, _, _, _ = np.linalg.lstsq(X, lf, rcond=None)
pred_mv = X @ beta
rss_mv = np.sum((lf - pred_mv)**2)
bic_mv = N_fit * np.log(rss_mv/N_fit) + 3*np.log(N_fit)
r_mv = np.corrcoef(lf, pred_mv)[0,1]
print(f"\nMultivariate: log(f) = {beta[0]:.3f} + {beta[1]:.3f}*log(v) + {beta[2]:.3f}*log(Ups)")
print(f"  r={r_mv:.4f}, RSS={rss_mv:.3f}, BIC={bic_mv:.1f}")

# Multivariate + quadratic
X2 = np.column_stack([np.ones(N_fit), lv, lv**2, lud])
beta2, _, _, _ = np.linalg.lstsq(X2, lf, rcond=None)
pred_mv2 = X2 @ beta2
rss_mv2 = np.sum((lf - pred_mv2)**2)
bic_mv2 = N_fit * np.log(rss_mv2/N_fit) + 4*np.log(N_fit)
r_mv2 = np.corrcoef(lf, pred_mv2)[0,1]
print(f"\nMV+quad: log(f) = {beta2[0]:.3f} + {beta2[1]:.3f}*log(v) + {beta2[2]:.3f}*log(v)^2 + {beta2[3]:.3f}*log(Ups)")
print(f"  r={r_mv2:.4f}, RSS={rss_mv2:.3f}, BIC={bic_mv2:.1f}")

# ============================================================
# 5. ビン別統計
# ============================================================
print(f"\n{'='*70}")
print("v_flat bin statistics (non-extrap, f>0)")
print(f"{'='*70}")

sort_idx = np.argsort(vf_m)
n_per_bin = N_fit // 8
print(f"\nEqual-N bins (N~{n_per_bin} each):")
print(f"{'v_flat range':>20s}  {'N':>4s}  {'f_med':>8s}  {'f_mean':>8s}  {'f_std':>8s}  {'v_bar/v_flat':>12s}")

bin_vf_centers = []
bin_f_medians = []

for i in range(8):
    i0 = i * n_per_bin
    i1 = (i+1)*n_per_bin if i < 7 else N_fit
    idx = sort_idx[i0:i1]
    vf_bin = vf_m[idx]
    ft_bin = ft_m[idx]
    vr_bin = np.sqrt(np.abs(ft_bin)) * np.sign(ft_bin)
    label = f"{vf_bin.min():.0f}-{vf_bin.max():.0f}"
    print(f"{label:>20s}  {len(idx):4d}  {np.median(ft_bin):8.3f}  {np.mean(ft_bin):8.3f}  {np.std(ft_bin):8.3f}  {np.median(vr_bin):12.3f}")
    bin_vf_centers.append(np.median(vf_bin))
    bin_f_medians.append(np.median(ft_bin))

# ============================================================
# 6. プロット
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0, 0]
ax.scatter(vf_m, ft_m, s=8, alpha=0.4, c='steelblue')
ax.scatter(bin_vf_centers, bin_f_medians, s=60, c='red', zorder=5,
           edgecolors='black', label='Bin medians')
vv = np.linspace(vf_m.min(), vf_m.max(), 200)
lvv = np.log10(vv)
ax.plot(vv, 10**np.polyval(p1, lvv), 'g--', label=f'Power law (a={p1[0]:.2f})')
ax.plot(vv, 10**np.polyval(p2, lvv), 'r-', label=f'Log-quad')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('f^(tanh)')
ax.set_title('(a) f vs v_flat')
ax.legend(fontsize=7)
ax.set_ylim(-0.2, 2.0)

ax = axes[0, 1]
ax.scatter(lv, lf, s=8, alpha=0.4, c='steelblue')
ax.plot(lvv, np.polyval(p1, lvv), 'g--', label='Power law')
ax.plot(lvv, np.polyval(p2, lvv), 'r-', label='Log-quadratic')
ax.set_xlabel('log(v_flat)')
ax.set_ylabel('log(f)')
ax.set_title('(b) log-log')
ax.legend(fontsize=8)

ax = axes[0, 2]
resid2 = lf - np.polyval(p2, lv)
ax.scatter(lv, resid2, s=8, alpha=0.4, c='steelblue')
ax.axhline(0, color='gray', ls='--')
ax.set_xlabel('log(v_flat)')
ax.set_ylabel('Residual')
ax.set_title(f'(c) Log-quad residuals (std={np.std(resid2):.3f})')

ax = axes[1, 0]
ax.scatter(ud_m, ft_m, s=8, alpha=0.4, c='orange')
ax.set_xlabel('Upsilon_d')
ax.set_ylabel('f^(tanh)')
ax.set_title(f'(d) f vs Ups_d (rho={rho_ud:.3f})')

ax = axes[1, 1]
ax.scatter(vf_m, ud_m, s=8, alpha=0.4, c='green')
ax.set_xlabel('v_flat [km/s]')
ax.set_ylabel('Upsilon_d')
ax.set_title(f'(e) Ups_d vs v_flat (rho={rho_uv:.3f})')

ax = axes[1, 2]
ax.scatter(pred_mv2, lf, s=8, alpha=0.4, c='purple')
xx = [min(lf.min(), pred_mv2.min()), max(lf.max(), pred_mv2.max())]
ax.plot(xx, xx, 'k--')
ax.set_xlabel('Predicted log(f)')
ax.set_ylabel('Observed log(f)')
ax.set_title(f'(f) MV+quad fit (r={r_mv2:.3f})')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step1_f_functional_form.png'), dpi=150)
plt.close()
print(f"\n[SAVED] step1_f_functional_form.png")

# ============================================================
# 7. 結論
# ============================================================
print(f"\n{'='*70}")
print("Step 1 Conclusion")
print(f"{'='*70}")

best = min(results.items(), key=lambda x: x[1]['bic'])
print(f"\nBest model (BIC): {best[0]} (BIC={best[1]['bic']:.1f})")
for nm, res in results.items():
    d = res['bic'] - best[1]['bic']
    if d > 0:
        print(f"  vs {nm}: dBIC = +{d:.1f}")

print(f"\nf(v_flat) is NOT constant (constant BIC={results['constant']['bic']:.1f})")
if np.isfinite(lv_min):
    print(f"Log-quad minimum at v_flat ~ {10**lv_min:.0f} km/s")
print(f"Ups_d contribution: coeff={beta[2]:.3f}")

if __name__ == "__main__":
    pass
