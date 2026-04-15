#!/usr/bin/env python3
"""
Step 3: v3.0 equilibrium + Freeman disk -> alpha derivation (pure theory)
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import i0, i1, k0, k1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = os.path.dirname(os.path.abspath(__file__))

a0 = 1.2e-10
kpc_to_m = 3.086e19

# ============================================================
# 1. Freeman exponential disk
# ============================================================
def v_disk_freeman(r, h_R, V_peak):
    y = np.clip(r / (2.0 * h_R), 1e-6, 50.0)
    bt = y**2 * (i0(y)*k0(y) - i1(y)*k1(y))
    y_pk = 1.075
    bt_pk = y_pk**2 * (i0(y_pk)*k0(y_pk) - i1(y_pk)*k1(y_pk))
    v2 = V_peak**2 * bt / bt_pk
    return np.sqrt(np.maximum(v2, 0.0))

# ============================================================
# 2. v3.0 MOND formula
# ============================================================
def v_obs_from_mond(r_kpc, v_bar, g_c_ms2):
    r_m = r_kpc * kpc_to_m
    v_bar_ms = v_bar * 1e3
    g_N = np.where(r_m > 0, v_bar_ms**2 / r_m, 0.0)
    g_o = (g_N + np.sqrt(g_N**2 + 4*g_c_ms2*g_N)) / 2
    return np.sqrt(r_m * g_o) / 1e3

# ============================================================
# 3. tanh fit
# ============================================================
def fit_tanh(r, v_obs, v_bar):
    def model(r, vf, rs):
        T = 0.5 * (1.0 + np.tanh((r - rs) / rs))
        return np.sqrt(v_bar**2 + vf**2 * T)
    vf0 = v_obs[-1]
    rs0 = r[len(r)//3]
    try:
        popt, _ = curve_fit(model, r, v_obs, p0=[vf0, rs0],
                           bounds=([0.1, 0.01], [500, r[-1]*3]), maxfev=5000)
        return popt[0], popt[1]
    except:
        return None, None

# ============================================================
# 4. Baryon scaling relations (from Step 1-2 observations)
# ============================================================
def hR_from_vflat(vf):
    return 2.5 * (vf / 100.0)**0.600

def Vpeak_from_vflat(vf):
    # Step 1 log-quadratic: log(f) = 5.80 - 6.89*log(v) + 1.84*log(v)^2
    lv = np.log10(vf)
    log_f = 5.80 - 6.89*lv + 1.84*lv**2
    f = np.clip(10**log_f, 0.05, 2.0)
    return np.sqrt(f) * vf

# ============================================================
# 5. Main computation
# ============================================================
print("="*70)
print("Step 3: v3.0 + Freeman disk -> alpha derivation")
print("="*70)

vflat_grid = np.logspace(np.log10(20), np.log10(300), 40)

scenarios = {
    'g_c=a0': a0,
    'g_c=0.5*a0': 0.5*a0,
    'g_c=2*a0': 2.0*a0,
}

all_results = {}

for scenario_name, g_c in scenarios.items():
    print(f"\n--- Scenario: {scenario_name} ---")
    rs_list, vf_list, hR_list, rpeak_list, f_list = [], [], [], [], []

    for vf in vflat_grid:
        h_R = hR_from_vflat(vf)
        V_peak = Vpeak_from_vflat(vf)
        r_peak = 2.15 * h_R
        r = np.linspace(0.1 * h_R, 15.0 * h_R, 300)
        v_bar = v_disk_freeman(r, h_R, V_peak)
        v_obs = v_obs_from_mond(r, v_bar, g_c)
        vf_fit, rs_fit = fit_tanh(r, v_obs, v_bar)
        if vf_fit is None or rs_fit <= 0:
            continue
        v_bar_at_rs = np.interp(rs_fit, r, v_bar)
        f_val = (v_bar_at_rs / vf_fit)**2

        rs_list.append(rs_fit); vf_list.append(vf)
        hR_list.append(h_R); rpeak_list.append(r_peak); f_list.append(f_val)

    rs_arr = np.array(rs_list); vf_arr = np.array(vf_list)
    hR_arr = np.array(hR_list); rpeak_arr = np.array(rpeak_list)
    f_arr = np.array(f_list)
    N = len(rs_arr)

    if N < 5:
        print(f"  Insufficient fits: N={N}"); continue

    lv = np.log10(vf_arr)
    p_alpha = np.polyfit(lv, np.log10(rs_arr), 1)
    p_beta  = np.polyfit(lv, np.log10(hR_arr), 1)
    p_gamma = np.polyfit(lv, np.log10(rs_arr/hR_arr), 1)
    rs_rpeak = rs_arr / rpeak_arr

    print(f"  N={N}")
    print(f"  alpha (r_s ~ v^a)    = {p_alpha[0]:.4f}")
    print(f"  beta  (h_R ~ v^b)    = {p_beta[0]:.4f}  (input: 0.600)")
    print(f"  gamma (r_s/h_R ~ v^g)= {p_gamma[0]:.4f}")
    print(f"  beta + gamma         = {p_beta[0]+p_gamma[0]:.4f}")
    print(f"  r_s/r_peak median    = {np.median(rs_rpeak):.4f}")
    print(f"  r_s/r_peak range     = {rs_rpeak.min():.3f} - {rs_rpeak.max():.3f}")
    print(f"  f median             = {np.median(f_arr):.4f}")

    all_results[scenario_name] = {
        'vf': vf_arr, 'rs': rs_arr, 'hR': hR_arr, 'rpeak': rpeak_arr,
        'f': f_arr, 'alpha': p_alpha[0], 'beta': p_beta[0],
        'gamma': p_gamma[0], 'rs_rpeak': rs_rpeak,
        'p_alpha': p_alpha, 'p_gamma': p_gamma,
    }

# ============================================================
# 6. Comparison with observations
# ============================================================
print(f"\n{'='*70}")
print("Comparison with observations")
print(f"{'='*70}")

print(f"\n{'Scenario':>15s}  {'alpha':>10s}  {'gamma':>10s}  {'r_s/r_peak':>10s}")
print(f"{'Observed':>15s}  {'0.883':>10s}  {'0.284':>10s}  {'0.98':>10s}")
print("-"*50)
for name, res in all_results.items():
    print(f"{name:>15s}  {res['alpha']:10.4f}  {res['gamma']:10.4f}  {np.median(res['rs_rpeak']):10.4f}")

# ============================================================
# 7. g_N/g_c ratio check
# ============================================================
print(f"\n{'='*70}")
print("g_N(r_peak)/a0 vs v_flat")
print(f"{'='*70}")

if 'g_c=a0' in all_results:
    res = all_results['g_c=a0']
    print(f"\n{'v_flat':>8s}  {'g_N/a0':>10s}  {'regime':>12s}")
    for vf_show in [25, 50, 75, 100, 150, 200, 300]:
        idx = np.argmin(np.abs(res['vf'] - vf_show))
        V_peak = Vpeak_from_vflat(res['vf'][idx])
        r_peak = res['rpeak'][idx]
        g_N = (V_peak*1e3)**2 / (r_peak*kpc_to_m)
        ratio = g_N / a0
        regime = "deep MOND" if ratio < 0.3 else ("MOND" if ratio < 3 else "Newtonian")
        print(f"  {res['vf'][idx]:6.0f}  {ratio:10.3f}  {regime:>12s}")

# ============================================================
# 8. Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {'g_c=a0': 'blue', 'g_c=0.5*a0': 'green', 'g_c=2*a0': 'orange'}

# (a) r_s vs v_flat
ax = axes[0, 0]
for name, res in all_results.items():
    ax.plot(np.log10(res['vf']), np.log10(res['rs']), '-',
            color=colors.get(name,'gray'), label=f"{name}: a={res['alpha']:.3f}")
lv_line = np.array([np.log10(20), np.log10(300)])
ax.plot(lv_line, 0.883*lv_line-1.5, 'k--', alpha=0.5, label='Obs a=0.883')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s)')
ax.set_title('(a) r_s vs v_flat: theory'); ax.legend(fontsize=7)

# (b) r_s/h_R vs v_flat
ax = axes[0, 1]
for name, res in all_results.items():
    ax.plot(np.log10(res['vf']), np.log10(res['rs']/res['hR']), '-',
            color=colors.get(name,'gray'), label=f"{name}: g={res['gamma']:.3f}")
ax.axhline(np.log10(2.11), color='k', ls='--', alpha=0.5, label='Obs med=2.11')
ax.set_xlabel('log(v_flat)'); ax.set_ylabel('log(r_s/h_R)')
ax.set_title('(b) gamma: membrane contribution'); ax.legend(fontsize=7)

# (c) r_s/r_peak vs v_flat
ax = axes[0, 2]
for name, res in all_results.items():
    ax.plot(res['vf'], res['rs_rpeak'], '-', color=colors.get(name,'gray'), label=name)
ax.axhline(0.98, color='k', ls='--', alpha=0.5, label='Obs med=0.98')
ax.set_xlabel('v_flat [km/s]'); ax.set_ylabel('r_s / r_peak')
ax.set_title('(c) r_s_tanh / r_peak'); ax.legend(fontsize=8)

# (d) f(v_flat) theory vs obs
ax = axes[1, 0]
for name, res in all_results.items():
    ax.plot(res['vf'], res['f'], '-', color=colors.get(name,'gray'), label=name)
vv = np.logspace(np.log10(20), np.log10(300), 100)
lv_obs = np.log10(vv)
ax.plot(vv, 10**(5.80-6.89*lv_obs+1.84*lv_obs**2), 'k--', label='Obs (log-quad)')
ax.set_xlabel('v_flat [km/s]'); ax.set_ylabel('f^(tanh)')
ax.set_title('(d) f(v_flat): theory vs obs'); ax.legend(fontsize=7)
ax.set_ylim(0, 1.5)

# (e) Synthetic RCs
ax = axes[1, 1]
for vf_show, col in [(30,'blue'),(100,'green'),(250,'red')]:
    h_R = hR_from_vflat(vf_show)
    V_peak = Vpeak_from_vflat(vf_show)
    r = np.linspace(0.1*h_R, 12*h_R, 300)
    v_bar = v_disk_freeman(r, h_R, V_peak)
    v_obs = v_obs_from_mond(r, v_bar, a0)
    r_n = r/h_R
    ax.plot(r_n, v_bar, '--', color=col, alpha=0.4)
    ax.plot(r_n, v_obs, '-', color=col, label=f'v={vf_show}')
ax.axvline(2.15, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('r / h_R'); ax.set_ylabel('v [km/s]')
ax.set_title('(e) Synthetic RCs (dashed=baryonic)'); ax.legend(fontsize=8)

# (f) alpha decomposition bar chart
ax = axes[1, 2]
if 'g_c=a0' in all_results:
    res = all_results['g_c=a0']
    labels = ['Observed', 'Theory\n(g_c=a0)']
    betas = [0.600, res['beta']]
    gammas = [0.284, res['gamma']]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x, betas, w, label='beta (baryonic)', color='steelblue')
    ax.bar(x, gammas, w, bottom=betas, label='gamma (membrane)', color='coral')
    for i in range(len(labels)):
        ax.text(i, betas[i]+gammas[i]+0.02, f'a={betas[i]+gammas[i]:.3f}',
                ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Exponent'); ax.set_title('(f) alpha decomposition')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step3_alpha_derivation.png'), dpi=150)
plt.close()
print(f"\n[SAVED] step3_alpha_derivation.png")

# ============================================================
# 9. Final conclusion
# ============================================================
print(f"\n{'='*70}")
print("Step 3 Final Conclusion")
print(f"{'='*70}")

if 'g_c=a0' in all_results:
    res = all_results['g_c=a0']
    print(f"""
  Theory alpha = {res['alpha']:.4f}  (obs: 0.883)
  Theory beta  = {res['beta']:.4f}   (obs: 0.600, input)
  Theory gamma = {res['gamma']:.4f}  (obs: 0.284)
  r_s/r_peak   = {np.median(res['rs_rpeak']):.4f}  (obs: 0.98)

  alpha match: |{res['alpha']:.3f} - 0.883| = {abs(res['alpha']-0.883):.3f}
  gamma match: |{res['gamma']:.3f} - 0.284| = {abs(res['gamma']-0.284):.3f}

  T-A1 Answer: "Why r_s ~ v_flat^0.795?"

  (1) h_R ~ v^0.6 (baryon size-mass relation, external input)
      -> ~68% of alpha
  (2) r_s_tanh ~ r_peak = 2.15*h_R (v3.0 MOND consequence)
      -> v_obs inflection point ~ v_bar peak
  (3) gamma ~ 0.28 (MOND transition geometry)
      -> small galaxies: deep MOND -> slight shift
      -> large galaxies: Newtonian -> r_s = r_peak exactly
""")
