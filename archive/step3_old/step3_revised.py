#!/usr/bin/env python3
"""
Step 3 revised: C-1 circular argument fix
V_peak from SPARC .dat directly (no r_s dependency)
"""
import csv
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import i0, i1, k0, k1
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir   = os.path.dirname(os.path.abspath(__file__))
rotmod_dir = os.path.join(base_dir, 'Rotmod_LTG')
csv_path   = os.path.join(base_dir, 'phase1', 'sparc_results.csv')

a0_unit = 3703.0
kpc_to_m = 3.086e19

# ============================================================
# Data loading
# ============================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)

_, src_rows = load_csv(csv_path)
galaxies = {}
for row in src_rows:
    name = row.get('galaxy','').strip()
    if not name: continue
    ud = float(row.get('ud','nan'))
    vf = float(row.get('vflat','nan'))
    rs = float(row.get('rs1', row.get('rs2','nan')))
    if np.isfinite(ud) and np.isfinite(vf) and np.isfinite(rs) and ud>0 and vf>0 and rs>0:
        galaxies[name] = {'upsilon_d': ud, 'vflat': vf, 'rs_tanh': rs}

def read_dat(name):
    path = os.path.join(rotmod_dir, f"{name}_rotmod.dat")
    if os.path.exists(path):
        try: return np.loadtxt(path, comments='#')
        except: pass
    return None

# ============================================================
# Part A: Measure V_peak, h_R from SPARC
# ============================================================
print("="*70)
print("Part A: SPARC measured V_peak, h_R")
print("="*70)

meas = []
for name, gal in galaxies.items():
    data = read_dat(name)
    if data is None: continue
    rad=data[:,0]; v_obs=data[:,1]; v_gas=data[:,3]; v_disk=data[:,4]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud = gal['upsilon_d']

    vbar2 = ud*np.sign(v_disk)*v_disk**2 + np.sign(v_gas)*v_gas**2 + np.sign(v_bul)*v_bul**2
    vbar = np.sqrt(np.maximum(vbar2, 0.0))
    vdisk_scaled = np.sqrt(ud) * np.abs(v_disk)

    if len(rad) < 3: continue

    V_peak_bar = np.max(vbar)
    i_pb = np.argmax(vbar); r_peak_bar = rad[i_pb]

    V_peak_disk = np.max(vdisk_scaled)
    i_pd = np.argmax(vdisk_scaled); r_peak_disk = rad[i_pd]

    peak_at_edge = (r_peak_disk >= rad.max() * 0.9)
    h_R_est = r_peak_disk / 2.15 if not peak_at_edge else None

    meas.append({
        'name': name, 'vflat': gal['vflat'], 'rs_tanh': gal['rs_tanh'],
        'upsilon_d': ud, 'V_peak_bar': V_peak_bar, 'V_peak_disk': V_peak_disk,
        'r_peak_bar': r_peak_bar, 'r_peak_disk': r_peak_disk,
        'h_R_est': h_R_est, 'peak_at_edge': peak_at_edge,
    })

print(f"Measured: {len(meas)} galaxies")

vf_m = np.array([d['vflat'] for d in meas])
Vp_disk = np.array([d['V_peak_disk'] for d in meas])
Vp_bar = np.array([d['V_peak_bar'] for d in meas])

mask = (vf_m > 0) & (Vp_disk > 0)
lv = np.log10(vf_m[mask])
lVp_disk = np.log10(Vp_disk[mask])
lVp_bar = np.log10(Vp_bar[mask])

p_Vp_disk = np.polyfit(lv, lVp_disk, 1)
p_Vp_bar = np.polyfit(lv, lVp_bar, 1)
r_disk, _ = pearsonr(lv, lVp_disk)
r_bar, _ = pearsonr(lv, lVp_bar)

print(f"\nV_peak(disk) ~ v_flat^{p_Vp_disk[0]:.3f}  (r={r_disk:.3f})")
print(f"V_peak(bar)  ~ v_flat^{p_Vp_bar[0]:.3f}  (r={r_bar:.3f})")
print(f"\nV_peak(disk)/v_flat: median={np.median(Vp_disk/vf_m):.3f}")
print(f"V_peak(bar)/v_flat:  median={np.median(Vp_bar/vf_m):.3f}")

# h_R vs v_flat
hR_data = [(d['vflat'], d['h_R_est']) for d in meas if d['h_R_est'] is not None]
p_hR = None
if hR_data:
    vf_hR = np.array([x[0] for x in hR_data])
    hR_arr = np.array([x[1] for x in hR_data])
    mask_hR = (vf_hR > 0) & (hR_arr > 0)
    if mask_hR.sum() >= 5:
        p_hR = np.polyfit(np.log10(vf_hR[mask_hR]), np.log10(hR_arr[mask_hR]), 1)
        r_hR, _ = pearsonr(np.log10(vf_hR[mask_hR]), np.log10(hR_arr[mask_hR]))
        print(f"\nh_R ~ v_flat^{p_hR[0]:.3f}  (r={r_hR:.3f}, N={mask_hR.sum()})")

# ============================================================
# Part B: Synthetic galaxies -> alpha derivation
# ============================================================
print(f"\n{'='*70}")
print("Part B: Synthetic galaxies (non-circular V_peak)")
print(f"{'='*70}")

# Freeman disk
def freeman_bessel(y):
    y = np.clip(y, 1e-8, 50.0)
    return y**2 * (i0(y)*k0(y) - i1(y)*k1(y))

y_fine = np.linspace(0.01, 5.0, 5000)
vd2_shape = freeman_bessel(y_fine)
bessel_peak = vd2_shape[np.argmax(vd2_shape)]

def v_disk_freeman(r, h_R, V_peak):
    y = np.clip(r/(2.0*h_R), 1e-6, 50.0)
    v2 = V_peak**2 * freeman_bessel(y) / bessel_peak
    return np.sqrt(np.maximum(v2, 0.0))

def v_obs_from_mond(r_kpc, v_bar, g_c):
    r_m = r_kpc * kpc_to_m; v_ms = v_bar * 1e3
    gN = np.where(r_m > 0, v_ms**2/r_m, 0.0)
    go = (gN + np.sqrt(gN**2 + 4*g_c*gN)) / 2
    return np.sqrt(r_m * go) / 1e3

def fit_tanh(r, v_obs, v_bar):
    def model(r, vf, rs):
        T = 0.5*(1.0+np.tanh((r-rs)/rs))
        return np.sqrt(v_bar**2+vf**2*T)
    try:
        popt, _ = curve_fit(model, r, v_obs, p0=[v_obs[-1], r[len(r)//3]],
                           bounds=([0.1,0.01],[500,r[-1]*3]), maxfev=5000)
        return popt[0], popt[1]
    except:
        return None, None

def Vpeak_nocirc(vf):
    return 10**np.polyval(p_Vp_disk, np.log10(vf))

def hR_from_vflat(vf):
    if p_hR is not None:
        return 10**np.polyval(p_hR, np.log10(vf))
    return 2.5 * (vf/100.0)**0.6

vflat_grid = np.logspace(np.log10(20), np.log10(300), 40)
scenarios = {'g_c=a0': 1.2e-10, 'g_c=0.5a0': 0.6e-10, 'g_c=2a0': 2.4e-10}
all_results = {}

for sc_name, g_c in scenarios.items():
    rs_list, vf_list, hR_list = [], [], []
    for vf in vflat_grid:
        h_R = hR_from_vflat(vf)
        V_peak = Vpeak_nocirc(vf)
        r = np.linspace(0.1*h_R, 15.0*h_R, 300)
        v_bar = v_disk_freeman(r, h_R, V_peak)
        v_obs = v_obs_from_mond(r, v_bar, g_c)
        vf_fit, rs_fit = fit_tanh(r, v_obs, v_bar)
        if vf_fit is None or rs_fit <= 0: continue
        rs_list.append(rs_fit); vf_list.append(vf); hR_list.append(h_R)

    rs_a=np.array(rs_list); vf_a=np.array(vf_list); hR_a=np.array(hR_list)
    N=len(rs_a)
    if N < 5:
        print(f"  {sc_name}: insufficient (N={N})"); continue

    lv=np.log10(vf_a); lrs=np.log10(rs_a); lhR=np.log10(hR_a); lrh=np.log10(rs_a/hR_a)
    p_a=np.polyfit(lv,lrs,1); p_b=np.polyfit(lv,lhR,1); p_g=np.polyfit(lv,lrh,1)
    rpeak_a=2.15*hR_a; rs_rpeak=rs_a/rpeak_a

    print(f"\n--- {sc_name} (N={N}) ---")
    print(f"  alpha = {p_a[0]:.4f}")
    print(f"  beta  = {p_b[0]:.4f}")
    print(f"  gamma = {p_g[0]:.4f}")
    print(f"  b+g   = {p_b[0]+p_g[0]:.4f}")
    print(f"  r_s/r_peak median = {np.median(rs_rpeak):.4f}")

    all_results[sc_name] = {
        'vf':vf_a,'rs':rs_a,'hR':hR_a,
        'alpha':p_a[0],'beta':p_b[0],'gamma':p_g[0],
        'rs_rpeak':rs_rpeak,'p_alpha':p_a,
    }

# ============================================================
# Part C: Comparison
# ============================================================
print(f"\n{'='*70}")
print("Old Step 3 vs Revised comparison")
print(f"{'='*70}")
print(f"\n{'':>20s}  {'alpha':>8s}  {'beta':>8s}  {'gamma':>8s}  {'r_s/r_pk':>10s}")
print(f"{'Observed':>20s}  {'0.883':>8s}  {'0.600':>8s}  {'0.284':>8s}  {'0.98':>10s}")
print(f"{'Old (circular)':>20s}  {'0.801':>8s}  {'0.600':>8s}  {'0.201':>8s}  {'0.571':>10s}")
for name, res in all_results.items():
    print(f"{('Revised '+name):>20s}  {res['alpha']:8.3f}  {res['beta']:8.3f}  {res['gamma']:8.3f}  {np.median(res['rs_rpeak']):10.3f}")

# ============================================================
# Part D: V_peak comparison
# ============================================================
print(f"\n{'='*70}")
print("V_peak input comparison: circular vs non-circular")
print(f"{'='*70}")
vv = np.array([30, 50, 75, 100, 150, 200, 300])
print(f"\n{'v_flat':>8s}  {'V_pk(old/circ)':>16s}  {'V_pk(new/meas)':>18s}  {'diff%':>6s}")
for vf in vv:
    lv_t = np.log10(vf)
    log_f_old = 5.80 - 6.89*lv_t + 1.84*lv_t**2
    f_old = np.clip(10**log_f_old, 0.05, 2.0)
    Vp_old = np.sqrt(f_old) * vf
    Vp_new = Vpeak_nocirc(vf)
    diff = 100*(Vp_new-Vp_old)/Vp_old if Vp_old>0 else 0
    print(f"{vf:8.0f}  {Vp_old:16.1f}  {Vp_new:18.1f}  {diff:+6.1f}%")

# ============================================================
# Part E: Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {'g_c=a0':'blue','g_c=0.5a0':'green','g_c=2a0':'orange'}

ax = axes[0, 0]
ax.scatter(vf_m, Vp_disk, s=8, alpha=0.4, c='blue', label='V_pk(disk)')
ax.scatter(vf_m, Vp_bar, s=8, alpha=0.4, c='red', label='V_pk(bar)')
vv_l = np.linspace(vf_m.min(), vf_m.max(), 100)
ax.plot(vv_l, 10**np.polyval(p_Vp_disk, np.log10(vv_l)), 'b--',
        label=f'disk: ~v^{p_Vp_disk[0]:.2f}')
ax.plot(vv_l, vv_l, 'k:', alpha=0.3)
ax.set_xlabel('v_flat'); ax.set_ylabel('V_peak')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('(a) V_peak vs v_flat (SPARC direct)'); ax.legend(fontsize=7)

ax = axes[0, 1]
ax.scatter(vf_m, Vp_disk/vf_m, s=8, alpha=0.3, c='blue', label='Measured')
vv2 = np.logspace(np.log10(20), np.log10(300), 100)
lv2 = np.log10(vv2)
f_old = np.clip(10**(5.80-6.89*lv2+1.84*lv2**2), 0.05, 2.0)
ax.plot(vv2, np.sqrt(f_old), 'r-', label='Old (circular f)')
ax.plot(vv2, 10**np.polyval(p_Vp_disk, lv2)/vv2, 'g--', label='New (measured)')
ax.set_xlabel('v_flat'); ax.set_ylabel('V_peak/v_flat')
ax.set_xscale('log'); ax.set_title('(b) V_pk/v_flat: old vs new'); ax.legend(fontsize=8)

ax = axes[0, 2]
for name, res in all_results.items():
    ax.plot(np.log10(res['vf']), np.log10(res['rs']), '-',
            color=colors.get(name,'gray'), label=f"{name}: a={res['alpha']:.3f}")
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s)')
ax.set_title('(c) r_s vs v_flat (revised)'); ax.legend(fontsize=7)

ax = axes[1, 0]
if 'g_c=a0' in all_results:
    res = all_results['g_c=a0']
    labels = ['Observed','Old\n(circular)','Revised']
    betas = [0.600, 0.600, res['beta']]
    gammas = [0.284, 0.201, res['gamma']]
    x = np.arange(3)
    ax.bar(x, betas, 0.35, label='beta', color='steelblue')
    ax.bar(x, gammas, 0.35, bottom=betas, label='gamma', color='coral')
    for i in range(3):
        ax.text(i, betas[i]+gammas[i]+0.02, f'a={betas[i]+gammas[i]:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Exponent'); ax.set_title('(d) alpha decomposition'); ax.legend(fontsize=8)

ax = axes[1, 1]
for name, res in all_results.items():
    ax.plot(res['vf'], res['rs_rpeak'], '-', color=colors.get(name,'gray'),
            label=f"{name}: {np.median(res['rs_rpeak']):.2f}")
ax.axhline(0.98, color='black', ls='--', label='Obs=0.98')
ax.axhline(0.571, color='red', ls=':', label='Old=0.571')
ax.set_xlabel('v_flat'); ax.set_ylabel('r_s/r_peak')
ax.set_title('(e) r_s/r_peak revised'); ax.legend(fontsize=7)

ax = axes[1, 2]
if p_hR is not None and hR_data:
    ax.scatter(vf_hR, hR_arr, s=8, alpha=0.3, c='orange')
    vv3 = np.logspace(np.log10(vf_hR.min()), np.log10(vf_hR.max()), 100)
    ax.plot(vv3, 10**np.polyval(p_hR, np.log10(vv3)), 'k--',
            label=f'h_R~v^{p_hR[0]:.2f}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('v_flat'); ax.set_ylabel('h_R [kpc]')
    ax.set_title(f'(f) h_R vs v_flat (b={p_hR[0]:.3f})'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step3_revised.png'), dpi=150)
plt.close()
print(f"\n[SAVED] step3_revised.png")

# Final
print(f"\n{'='*70}")
print("C-1 Fix Conclusion")
print(f"{'='*70}")
if 'g_c=a0' in all_results:
    res = all_results['g_c=a0']
    print(f"""
  V_peak input:
    Old: V_peak = sqrt(f(v_flat))*v_flat  <- f depends on r_s (CIRCULAR)
    New: V_peak(disk) ~ v_flat^{p_Vp_disk[0]:.3f}  <- from .dat directly (NO r_s)

  alpha results:
    Old:     alpha=0.801  (beta=0.600, gamma=0.201)
    Revised: alpha={res['alpha']:.3f}  (beta={res['beta']:.3f}, gamma={res['gamma']:.3f})
    Observed: alpha=0.883  (beta=0.600, gamma=0.284)

  Impact of circular fix: |old-new| = {abs(0.801-res['alpha']):.3f}
""")
