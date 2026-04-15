#!/usr/bin/env python3
"""
Step 2: r_s_tanh / h_R vs v_flat の解析
"""
import csv
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir   = os.path.dirname(os.path.abspath(__file__))
rotmod_dir = os.path.join(base_dir, 'Rotmod_LTG')

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
print(f"[INFO] detail.csv: {len(detail_rows)} rows")

# sparc_results.csv
extra = {}
src_path = os.path.join(base_dir, 'phase1', 'sparc_results.csv')
if os.path.exists(src_path):
    _, src_rows = load_csv(src_path)
    for row in src_rows:
        name = row.get('galaxy', '').strip()
        if name:
            extra[name] = row

# ============================================================
# 2. V_disk ピークから h_R を推定
# ============================================================
def read_sparc_dat(galaxy_name):
    path = os.path.join(rotmod_dir, f"{galaxy_name}_rotmod.dat")
    if os.path.exists(path):
        try:
            data = np.loadtxt(path, comments='#')
            rad    = data[:, 0]
            v_obs  = data[:, 1]
            v_gas  = data[:, 3]
            v_disk = data[:, 4]
            v_bul  = data[:, 5] if data.shape[1] > 5 else np.zeros_like(rad)
            return rad, v_obs, v_gas, v_disk, v_bul
        except:
            pass
    return None, None, None, None, None

def estimate_hR_from_vdisk(rad, v_disk):
    if rad is None or len(rad) < 3:
        return None, None
    abs_vd = np.abs(v_disk)
    if len(rad) >= 4:
        try:
            r_fine = np.linspace(rad.min(), rad.max(), 500)
            spl = UnivariateSpline(rad, abs_vd, s=len(rad)*0.5,
                                   k=min(3, len(rad)-1))
            vd_fine = spl(r_fine)
            i_peak = np.argmax(vd_fine)
            r_peak = r_fine[i_peak]
        except:
            i_peak = np.argmax(abs_vd)
            r_peak = rad[i_peak]
    else:
        i_peak = np.argmax(abs_vd)
        r_peak = rad[i_peak]

    if r_peak >= rad.max() * 0.95:
        return None, r_peak
    h_R = r_peak / 2.15
    return h_R, r_peak

# ============================================================
# 3. メインループ
# ============================================================
results = []

for d in detail_rows:
    gal = d['galaxy'].strip()
    vf = float(d['v_flat'])
    ft = float(d['f_tanh'])
    rs = float(d['r_s_tanh'])
    ud = float(d['upsilon_d'])
    ext = d['is_extrap'].strip() == 'True'

    rad, v_obs, v_gas, v_disk, v_bul = read_sparc_dat(gal)
    h_R = None
    r_peak = None
    h_R_source = None

    if rad is not None:
        h_R, r_peak = estimate_hR_from_vdisk(rad, v_disk)
        if h_R is not None:
            h_R_source = 'vpeak'

    if h_R is None or h_R <= 0:
        continue

    rs_over_hR = rs / h_R
    rs_over_rpeak = rs / (2.15 * h_R) if h_R > 0 else None

    results.append({
        'name': gal, 'v_flat': vf, 'f_tanh': ft,
        'r_s': rs, 'upsilon_d': ud, 'h_R': h_R,
        'h_R_source': h_R_source, 'r_peak': r_peak,
        'rs_over_hR': rs_over_hR, 'rs_over_rpeak': rs_over_rpeak,
        'is_extrap': ext,
    })

print(f"\n[INFO] h_R obtained: {len(results)} / {len(detail_rows)} galaxies")
res_in = [r for r in results if not r['is_extrap']]
print(f"  non-extrap: {len(res_in)}")

# ============================================================
# 4. r_s_tanh / h_R 基本統計
# ============================================================
print(f"\n{'='*70}")
print("r_s_tanh / h_R statistics")
print(f"{'='*70}")

def stats_block(label, data):
    arr = np.array([d['rs_over_hR'] for d in data])
    print(f"\n--- {label} (N={len(arr)}) ---")
    print(f"  median: {np.median(arr):.3f}")
    print(f"  mean:   {np.mean(arr):.3f}")
    print(f"  std:    {np.std(arr):.3f}")
    print(f"  25%ile: {np.percentile(arr, 25):.3f}")
    print(f"  75%ile: {np.percentile(arr, 75):.3f}")
    return arr

rh_all = stats_block("All", results)
rh_in = stats_block("Non-extrap", res_in)

# ============================================================
# 5. r_s/h_R vs v_flat
# ============================================================
print(f"\n{'='*70}")
print("r_s_tanh / h_R vs v_flat")
print(f"{'='*70}")

vf_arr = np.array([d['v_flat'] for d in res_in])
rh_arr = np.array([d['rs_over_hR'] for d in res_in])
ft_arr = np.array([d['f_tanh'] for d in res_in])
rs_arr = np.array([d['r_s'] for d in res_in])
hR_arr = np.array([d['h_R'] for d in res_in])
ud_arr = np.array([d['upsilon_d'] for d in res_in])
N = len(vf_arr)

mask_pos = (rh_arr > 0) & (vf_arr > 0)
lv = np.log10(vf_arr[mask_pos])
lrh = np.log10(rh_arr[mask_pos])
N_pos = mask_pos.sum()

rp, pp = pearsonr(lv, lrh)
rs_corr, ps = spearmanr(vf_arr[mask_pos], rh_arr[mask_pos])
print(f"\nlog(r_s/h_R) vs log(v_flat): Pearson r={rp:.4f}, p={pp:.2e}")
print(f"Spearman rho={rs_corr:.4f}, p={ps:.2e}")

# power law
p1 = np.polyfit(lv, lrh, 1)
print(f"\nPower law: r_s/h_R ~ v_flat^{p1[0]:.3f}")

# log-quadratic
p2 = np.polyfit(lv, lrh, 2)
lv_min2 = -p2[1]/(2*p2[0]) if abs(p2[0]) > 1e-6 else None
print(f"Log-quad: {p2[2]:.3f} + {p2[1]:.3f}*log(v) + {p2[0]:.3f}*log(v)^2")
if lv_min2 and np.isfinite(lv_min2):
    print(f"  extremum at v_flat ~ {10**lv_min2:.1f} km/s")

# BIC
pred1 = np.polyval(p1, lv); pred2 = np.polyval(p2, lv)
pred0 = np.mean(lrh)*np.ones(N_pos)
rss0 = np.sum((lrh-pred0)**2); rss1 = np.sum((lrh-pred1)**2); rss2 = np.sum((lrh-pred2)**2)
bic0 = N_pos*np.log(rss0/N_pos)+1*np.log(N_pos)
bic1 = N_pos*np.log(rss1/N_pos)+2*np.log(N_pos)
bic2 = N_pos*np.log(rss2/N_pos)+3*np.log(N_pos)
print(f"\nBIC: const={bic0:.1f}  power={bic1:.1f}  log-quad={bic2:.1f}")

# ============================================================
# 6. h_R vs v_flat
# ============================================================
print(f"\n{'='*70}")
print("h_R vs v_flat")
print(f"{'='*70}")

lhR = np.log10(hR_arr[mask_pos])
p_hR = np.polyfit(lv, lhR, 1)
r_hR, p_hR_p = pearsonr(lv, lhR)
print(f"\nh_R ~ v_flat^{p_hR[0]:.3f}  (r={r_hR:.4f}, p={p_hR_p:.2e})")

# ============================================================
# 7. alpha decomposition
# ============================================================
print(f"\n{'='*70}")
print("alpha = 0.795 decomposition: r_s = (r_s/h_R) * h_R")
print(f"{'='*70}")

lrs = np.log10(rs_arr[mask_pos])
p_rs = np.polyfit(lv, lrs, 1)
r_rs, _ = pearsonr(lv, lrs)

print(f"\nDirect: r_s ~ v_flat^{p_rs[0]:.3f}  (r={r_rs:.4f})")
print(f"\nDecomposition:")
print(f"  beta  = {p_hR[0]:.3f}  (h_R ~ v^beta)")
print(f"  gamma = {p1[0]:.3f}  (r_s/h_R ~ v^gamma)")
print(f"  beta+gamma = {p_hR[0]+p1[0]:.3f}  (cf. direct alpha={p_rs[0]:.3f})")

# ============================================================
# 8. ビン別統計
# ============================================================
print(f"\n{'='*70}")
print("v_flat bin statistics")
print(f"{'='*70}")

sort_idx = np.argsort(vf_arr[mask_pos])
n_per = N_pos // 6
bin_vf = []; bin_rh = []

print(f"\n{'v_flat range':>18s}  {'N':>4s}  {'r_s/h_R med':>12s}  {'h_R med':>10s}  {'r_s med':>10s}  {'f med':>8s}")
for i in range(6):
    i0 = i*n_per; i1 = (i+1)*n_per if i<5 else N_pos
    idx = sort_idx[i0:i1]
    vf_b=vf_arr[mask_pos][idx]; rh_b=rh_arr[mask_pos][idx]
    hR_b=hR_arr[mask_pos][idx]; rs_b=rs_arr[mask_pos][idx]
    ft_b=ft_arr[mask_pos][idx]
    label=f"{vf_b.min():.0f}-{vf_b.max():.0f}"
    print(f"{label:>18s}  {len(idx):4d}  {np.median(rh_b):12.3f}  {np.median(hR_b):10.3f}  {np.median(rs_b):10.3f}  {np.median(ft_b):8.3f}")
    bin_vf.append(np.median(vf_b)); bin_rh.append(np.median(rh_b))

# ============================================================
# 9. プロット
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

vv = np.linspace(lv.min(), lv.max(), 200)

ax = axes[0, 0]
ax.scatter(lv, lrh, s=8, alpha=0.4, c='steelblue')
ax.scatter(np.log10(bin_vf), np.log10(bin_rh), s=60, c='red', zorder=5,
           edgecolors='black', label='Bin medians')
ax.plot(vv, np.polyval(p1, vv), 'g--', label=f'Power: g={p1[0]:.3f}')
ax.plot(vv, np.polyval(p2, vv), 'r-', label='Log-quad')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s / h_R)')
ax.set_title(f'(a) r_s/h_R vs v_flat (r={rp:.3f})'); ax.legend(fontsize=7)

ax = axes[0, 1]
ax.scatter(lv, lhR, s=8, alpha=0.4, c='orange')
ax.plot(vv, np.polyval(p_hR, vv), 'k--', label=f'h_R ~ v^{p_hR[0]:.2f}')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(h_R) [kpc]')
ax.set_title(f'(b) h_R vs v_flat (beta={p_hR[0]:.3f})'); ax.legend(fontsize=8)

ax = axes[0, 2]
ax.scatter(lv, lrs, s=8, alpha=0.4, c='green')
ax.plot(vv, np.polyval(p_rs, vv), 'k-', label=f'r_s ~ v^{p_rs[0]:.2f} (direct)')
ax.plot(vv, np.polyval(p_hR, vv)+np.polyval(p1, vv), 'r--',
        label=f'h_R*(r_s/h_R) ~ v^{p_hR[0]+p1[0]:.2f}')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s) [kpc]')
ax.set_title('(c) alpha decomposition'); ax.legend(fontsize=7)

ax = axes[1, 0]
rh_plot = rh_arr[mask_pos]
ax.hist(rh_plot[rh_plot < np.percentile(rh_plot, 98)], bins=40,
        color='steelblue', edgecolor='white')
ax.axvline(np.median(rh_plot), color='red', ls='--',
           label=f'Median={np.median(rh_plot):.2f}')
ax.axvline(2.15, color='gray', ls=':', label='r_peak/h_R=2.15')
ax.set_xlabel('r_s_tanh / h_R'); ax.set_ylabel('Count')
ax.set_title('(d) r_s/h_R distribution'); ax.legend(fontsize=8)

ax = axes[1, 1]
ax.scatter(rh_arr[mask_pos], ft_arr[mask_pos], s=8, alpha=0.4, c='purple')
rho_frh, _ = spearmanr(rh_arr[mask_pos], ft_arr[mask_pos])
ax.set_xlabel('r_s_tanh / h_R'); ax.set_ylabel('f^(tanh)')
ax.set_title(f'(e) f vs r_s/h_R (rho={rho_frh:.3f})')

ax = axes[1, 2]
rs_rpeak = rs_arr[mask_pos] / (2.15 * hR_arr[mask_pos])
ax.scatter(vf_arr[mask_pos], rs_rpeak, s=8, alpha=0.4, c='teal')
ax.axhline(1.0, color='gray', ls='--', label='r_s = r_peak')
ax.set_xlabel('v_flat [km/s]'); ax.set_ylabel('r_s_tanh / r_peak')
ax.set_title(f'(f) r_s vs disk peak (med={np.median(rs_rpeak):.2f})')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step2_rs_hR_relation.png'), dpi=150)
plt.close()
print(f"\n[SAVED] step2_rs_hR_relation.png")

# ============================================================
# 10. CSV出力
# ============================================================
out_csv = os.path.join(base_dir, "step2_rs_hR_detail.csv")
with open(out_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['galaxy','v_flat','r_s_tanh','h_R','h_R_source',
                     'rs_over_hR','f_tanh','upsilon_d','is_extrap'])
    for d in sorted(results, key=lambda x: x['v_flat']):
        writer.writerow([
            d['name'], f"{d['v_flat']:.1f}", f"{d['r_s']:.3f}",
            f"{d['h_R']:.3f}", d['h_R_source'],
            f"{d['rs_over_hR']:.4f}", f"{d['f_tanh']:.4f}",
            f"{d['upsilon_d']:.3f}", d['is_extrap']
        ])
print(f"[SAVED] {out_csv}")

# ============================================================
# 11. 結論
# ============================================================
rs_rpeak_med = np.median(rs_rpeak)
print(f"\n{'='*70}")
print("Step 2 Conclusion")
print(f"{'='*70}")
print(f"\nalpha decomposition: r_s = (r_s/h_R) * h_R")
print(f"  beta  (h_R ~ v^beta)     = {p_hR[0]:.3f}")
print(f"  gamma (r_s/h_R ~ v^gamma)= {p1[0]:.3f}")
print(f"  beta + gamma             = {p_hR[0]+p1[0]:.3f}")
print(f"  direct alpha             = {p_rs[0]:.3f}")
print(f"\nr_s/h_R median = {np.median(rh_plot):.2f}")
print(f"r_s/r_peak median = {rs_rpeak_med:.2f}")
print(f"-> r_s_tanh is {rs_rpeak_med:.1f}x the V_disk peak radius")
print(f"\nf vs r_s/h_R: Spearman rho = {rho_frh:.3f}")
