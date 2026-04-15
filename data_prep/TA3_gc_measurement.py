#!/usr/bin/env python3
"""
T-A3: Independent g_c measurement via RAR fitting
No r_s used -> completely avoids circular argument
"""
import csv
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import spearmanr, pearsonr, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir   = os.path.dirname(os.path.abspath(__file__))
rotmod_dir = os.path.join(base_dir, 'Rotmod_LTG')
csv_path   = os.path.join(base_dir, 'phase1', 'sparc_results.csv')

a0_unit = 3703.0  # a0 in (km/s)^2/kpc

# ============================================================
# Data loading
# ============================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)

_, src_rows = load_csv(csv_path)
print(f"[INFO] sparc_results.csv: {len(src_rows)} rows")

galaxies = {}
for row in src_rows:
    name = row.get('galaxy', '').strip()
    if not name: continue
    ud = float(row.get('ud', 'nan'))
    vf = float(row.get('vflat', 'nan'))
    rs = float(row.get('rs1', row.get('rs2', 'nan')))
    if np.isfinite(ud) and np.isfinite(vf) and np.isfinite(rs) and ud > 0 and vf > 0 and rs > 0:
        galaxies[name] = {'upsilon_d': ud, 'vflat': vf, 'rs_tanh': rs}

print(f"[INFO] {len(galaxies)} galaxies loaded")

def read_dat(name):
    path = os.path.join(rotmod_dir, f"{name}_rotmod.dat")
    if os.path.exists(path):
        try: return np.loadtxt(path, comments='#')
        except: pass
    return None

# ============================================================
# MOND fitting
# ============================================================
def mond_gobs(gN, gc):
    return (gN + np.sqrt(gN**2 + 4*gc*gN)) / 2

def fit_gc_single(gN_arr, gobs_arr):
    mask = (gN_arr > 0) & (gobs_arr > 0)
    gN = gN_arr[mask]; gobs = gobs_arr[mask]
    N = len(gN)
    if N < 3:
        return None, None, None, N

    log_gobs = np.log10(gobs)

    def chi2(log_gc):
        gc = 10**log_gc
        model = mond_gobs(gN, gc)
        model = np.maximum(model, 1e-30)
        return np.sum((log_gobs - np.log10(model))**2)

    log_gc_grid = np.linspace(-1.0, 2.0, 100) + np.log10(a0_unit)
    chi2_grid = [chi2(lg) for lg in log_gc_grid]
    i_best = np.argmin(chi2_grid)

    result = minimize_scalar(chi2,
        bounds=(log_gc_grid[max(0,i_best-5)],
                log_gc_grid[min(len(log_gc_grid)-1,i_best+5)]),
        method='bounded')

    gc_fit = 10**result.x
    gc_over_a0 = gc_fit / a0_unit
    chi2_min = result.fun
    chi2_dof = chi2_min / max(N - 1, 1)

    gc_err = None
    try:
        target = chi2_min + 1.0
        def f_lo(lg): return chi2(lg) - target
        lg_lo = brentq(f_lo, log_gc_grid[0], result.x)
        lg_hi = brentq(f_lo, result.x, log_gc_grid[-1])
        gc_err = (10**lg_hi / a0_unit - 10**lg_lo / a0_unit) / 2
    except:
        pass

    return gc_over_a0, chi2_dof, gc_err, N

# ============================================================
# Main loop
# ============================================================
results = []
n_fail = 0; n_no_dat = 0

for name, gal in galaxies.items():
    data = read_dat(name)
    if data is None:
        n_no_dat += 1; continue

    rad = data[:, 0]; v_obs = data[:, 1]
    v_gas = data[:, 3]; v_disk = data[:, 4]
    v_bul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(rad)
    ud = gal['upsilon_d']

    vbar2 = ud*np.sign(v_disk)*v_disk**2 + np.sign(v_gas)*v_gas**2 + np.sign(v_bul)*v_bul**2

    mask = rad > 0.01
    r = rad[mask]
    gN = np.maximum(vbar2[mask], 0) / r
    gobs = v_obs[mask]**2 / r

    gc_a0, chi2_dof, gc_err, N_pts = fit_gc_single(gN, gobs)

    if gc_a0 is None:
        n_fail += 1; continue

    results.append({
        'name': name, 'vflat': gal['vflat'], 'rs_tanh': gal['rs_tanh'],
        'upsilon_d': ud, 'gc_over_a0': gc_a0, 'gc_err': gc_err,
        'chi2_dof': chi2_dof, 'N_pts': N_pts,
        'gc_ratio_circular': gal['vflat']**2 / (gal['rs_tanh'] * a0_unit),
    })

N_total = len(results)
print(f"\n[INFO] Fit success: {N_total}, fail: {n_fail}, no .dat: {n_no_dat}")

# ============================================================
# Analysis
# ============================================================
gc_arr = np.array([d['gc_over_a0'] for d in results])
vf_arr = np.array([d['vflat'] for d in results])
ud_arr = np.array([d['upsilon_d'] for d in results])
rs_arr = np.array([d['rs_tanh'] for d in results])
chi2_arr = np.array([d['chi2_dof'] for d in results])
gc_circ = np.array([d['gc_ratio_circular'] for d in results])

mask_good = chi2_arr < 10
gc_good = gc_arr[mask_good]
N_good = mask_good.sum()

print(f"\n{'='*70}")
print(f"g_c / a0 distribution (N={N_total})")
print(f"{'='*70}")
print(f"Good fits (chi2/dof < 10): {N_good}/{N_total}")

for label, arr in [("All", gc_arr), ("Good fits", gc_good)]:
    print(f"\n--- {label} (N={len(arr)}) ---")
    print(f"  median: {np.median(arr):.4f}")
    print(f"  mean:   {np.mean(arr):.4f}")
    print(f"  std:    {np.std(arr):.4f}")
    print(f"  CV:     {np.std(arr)/np.mean(arr):.3f}")
    print(f"  25%ile: {np.percentile(arr, 25):.4f}")
    print(f"  75%ile: {np.percentile(arr, 75):.4f}")

# Universality test
print(f"\n{'='*70}")
print("g_c universality test")
print(f"{'='*70}")

try:
    stat, p_w = wilcoxon(gc_good - 1.0)
    print(f"\nWilcoxon (H0: g_c = a0): p = {p_w:.2e}")
except Exception as e:
    print(f"\nWilcoxon failed: {e}"); p_w = None

print(f"CV = {np.std(gc_good)/np.mean(gc_good):.3f}")

# Correlations
print(f"\n{'='*70}")
print("g_c vs galaxy parameters (good fits)")
print(f"{'='*70}")

vf_good = vf_arr[mask_good]; ud_good = ud_arr[mask_good]
rs_good = rs_arr[mask_good]; gc_circ_good = gc_circ[mask_good]

rho_vf, p_vf = spearmanr(gc_good, vf_good)
rho_ud, p_ud = spearmanr(gc_good, ud_good)
rho_rs, p_rs = spearmanr(gc_good, rs_good)
rho_circ, p_circ = spearmanr(gc_good, gc_circ_good)

print(f"\ng_c vs v_flat:    rho={rho_vf:.3f}, p={p_vf:.2e}")
print(f"g_c vs Ups_d:     rho={rho_ud:.3f}, p={p_ud:.2e}")
print(f"g_c vs r_s_tanh:  rho={rho_rs:.3f}, p={p_rs:.2e}")
print(f"g_c(RAR) vs g_c(circular): rho={rho_circ:.3f}, p={p_circ:.2e}")

# Power law fit
mask_pos = (gc_good > 0) & (vf_good > 0)
p_fit = None
if mask_pos.sum() > 10:
    lg = np.log10(gc_good[mask_pos]); lv = np.log10(vf_good[mask_pos])
    p_fit = np.polyfit(lv, lg, 1)
    r_fit, _ = pearsonr(lv, lg)
    print(f"\nlog(g_c/a0) = {p_fit[1]:.3f} + {p_fit[0]:.3f}*log(v_flat)")
    print(f"  -> g_c ~ v_flat^{p_fit[0]:.3f}  (r={r_fit:.3f})")
    print(f"  cf. circular: g_c ~ v_flat^1.32")

# Binned statistics
print(f"\n{'='*70}")
print("v_flat binned g_c/a0")
print(f"{'='*70}")
sort_idx = np.argsort(vf_good)
n_per = max(N_good // 6, 1)
print(f"\n{'v_flat range':>18s}  {'N':>4s}  {'g_c/a0 med':>12s}  {'std':>10s}  {'CV':>6s}")
for i in range(6):
    i0 = i*n_per; i1 = (i+1)*n_per if i<5 else N_good
    idx = sort_idx[i0:i1]
    vf_b = vf_good[idx]; gc_b = gc_good[idx]
    cv_b = np.std(gc_b)/np.mean(gc_b) if np.mean(gc_b) > 0 else 999
    print(f"{vf_b.min():.0f}-{vf_b.max():.0f}:".rjust(18) +
          f"  {len(idx):4d}  {np.median(gc_b):12.4f}  {np.std(gc_b):10.4f}  {cv_b:6.3f}")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0, 0]
ax.hist(gc_good[gc_good<10], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(1.0, color='red', ls='--', lw=2, label='g_c = a0')
ax.axvline(np.median(gc_good), color='black', ls='-', label=f'Med={np.median(gc_good):.3f}')
ax.set_xlabel('g_c / a0'); ax.set_ylabel('Count')
ax.set_title(f'(a) g_c distribution (N={N_good})'); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.scatter(vf_good, gc_good, s=10, alpha=0.4, c='steelblue')
ax.axhline(1.0, color='red', ls='--', alpha=0.5)
ax.set_xlabel('v_flat [km/s]'); ax.set_ylabel('g_c / a0')
ax.set_xscale('log'); ax.set_yscale('log')
if p_fit is not None:
    vv = np.logspace(np.log10(vf_good.min()), np.log10(vf_good.max()), 100)
    ax.plot(vv, 10**np.polyval(p_fit, np.log10(vv)), 'k-', alpha=0.5,
            label=f'~v^{p_fit[0]:.2f}')
    ax.legend(fontsize=8)
ax.set_title(f'(b) g_c vs v_flat (rho={rho_vf:.3f})')

ax = axes[0, 2]
ax.scatter(ud_good, gc_good, s=10, alpha=0.4, c='orange')
ax.axhline(1.0, color='red', ls='--', alpha=0.5)
ax.set_xlabel('Ups_d'); ax.set_ylabel('g_c / a0'); ax.set_yscale('log')
ax.set_title(f'(c) g_c vs Ups_d (rho={rho_ud:.3f})')

# (d) RAR
ax = axes[1, 0]
count = 0
for d in results:
    if count >= 50: break
    data = read_dat(d['name'])
    if data is None: continue
    rad=data[:,0]; v_obs=data[:,1]; v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=d['upsilon_d']
    vbar2=ud*np.sign(v_disk)*v_disk**2+np.sign(v_gas)*v_gas**2+np.sign(v_bul)*v_bul**2
    m=(rad>0.01)&(vbar2>0)
    if m.sum()<2: continue
    gN=vbar2[m]/rad[m]; gobs=v_obs[m]**2/rad[m]
    ax.scatter(np.log10(gN/a0_unit), np.log10(gobs/a0_unit), s=1, alpha=0.1, c='gray')
    count += 1

gN_line = np.logspace(-3, 2, 500) * a0_unit
for gc_val, col, lab in [(1.0,'red','g_c=a0'),(0.5,'blue','0.5a0'),(2.0,'green','2a0')]:
    gobs_line = mond_gobs(gN_line, gc_val*a0_unit)
    ax.plot(np.log10(gN_line/a0_unit), np.log10(gobs_line/a0_unit), color=col, label=lab, lw=1.5)
ax.plot([-3,2],[-3,2],'k:',alpha=0.3)
ax.set_xlabel('log(g_N/a0)'); ax.set_ylabel('log(g_obs/a0)')
ax.set_title('(d) RAR with MOND curves'); ax.legend(fontsize=7)
ax.set_xlim(-3,1.5); ax.set_ylim(-2.5,1.5)

ax = axes[1, 1]
ax.scatter(gc_circ_good, gc_good, s=10, alpha=0.4, c='purple')
mx = max(gc_circ_good.max(), gc_good.max())
ax.plot([0.001,mx],[0.001,mx],'k--',alpha=0.3)
ax.set_xlabel('g_c (circular) / a0'); ax.set_ylabel('g_c (RAR) / a0')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title(f'(e) RAR vs circular (rho={rho_circ:.3f})')

ax = axes[1, 2]
ax.hist(chi2_arr[chi2_arr<20], bins=50, color='green', edgecolor='white', alpha=0.7)
ax.axvline(1.0, color='red', ls='--', label='chi2/dof=1')
ax.set_xlabel('chi2/dof'); ax.set_ylabel('Count')
ax.set_title('(f) Fit quality'); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'TA3_gc_measurement.png'), dpi=150)
plt.close()
print(f"\n[SAVED] TA3_gc_measurement.png")

# CSV
out_csv = os.path.join(base_dir, "TA3_gc_independent.csv")
with open(out_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['galaxy','v_flat','rs_tanh','upsilon_d',
                     'gc_over_a0','gc_err','chi2_dof','N_pts','gc_circular'])
    for d in sorted(results, key=lambda x: x['vflat']):
        writer.writerow([
            d['name'], f"{d['vflat']:.1f}", f"{d['rs_tanh']:.3f}",
            f"{d['upsilon_d']:.3f}", f"{d['gc_over_a0']:.4f}",
            f"{d['gc_err']:.4f}" if d['gc_err'] else "NA",
            f"{d['chi2_dof']:.4f}", d['N_pts'],
            f"{d['gc_ratio_circular']:.4f}",
        ])
print(f"[SAVED] {out_csv}")

# Final
print(f"\n{'='*70}")
print("T-A3 Conclusion")
print(f"{'='*70}")
print(f"""
  Independent g_c via RAR fitting (no r_s used)
  N = {N_good} (good fits)
  g_c/a0 median = {np.median(gc_good):.4f}
  CV = {np.std(gc_good)/np.mean(gc_good):.3f}
  g_c vs v_flat: rho={rho_vf:.3f}
  g_c(RAR) vs g_c(circular): rho={rho_circ:.3f}
""")
