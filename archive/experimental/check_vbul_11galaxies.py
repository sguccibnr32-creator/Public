# -*- coding: utf-8 -*-
"""
2段階有利11銀河：バルジ成分チェック
条件14（膜の二相構造）vs 仮説B（バリオン複雑性）の判別
"""
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\Rotmod_LTG')
OUT_DIR  = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1')
CSV_PATH = OUT_DIR / 'sparc_results.csv'

def load_2stage_list():
    galaxies = []
    with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['grade'] == '2stage':
                galaxies.append(row['galaxy'])
    return galaxies

def get_rotmod_path(gal):
    return DATA_DIR / f'{gal}_rotmod.dat'

def analyze_vbul(gal):
    fpath = get_rotmod_path(gal)
    try:
        data = np.loadtxt(fpath, comments='#')
    except:
        return None
    if data.ndim == 1:
        return None

    r, v_obs, v_err = data[:,0], data[:,1], data[:,2]
    v_gas, v_disk = data[:,3], data[:,4]
    v_bul = data[:,5] if data.shape[1] > 5 else np.zeros_like(r)

    mask = (v_obs > 0) & (r > 0)
    r, v_obs, v_disk, v_bul, v_gas = r[mask], v_obs[mask], v_disk[mask], v_bul[mask], v_gas[mask]

    vbul_max  = np.max(np.abs(v_bul))
    vobs_max  = np.max(v_obs)
    vdisk_max = np.max(np.abs(v_disk))
    vgas_max  = np.max(np.abs(v_gas))
    vbul_frac = vbul_max / vobs_max if vobs_max > 0 else 0

    # バルジの速度²寄与の割合（内側半分）
    n_inner = max(len(r)//2, 1)
    v2_bul_inner = np.mean(v_bul[:n_inner]**2)
    v2_obs_inner = np.mean(v_obs[:n_inner]**2)
    bul_energy_frac = v2_bul_inner / v2_obs_inner if v2_obs_inner > 0 else 0

    return dict(
        has_bul=vbul_max > 5.0,
        vbul_max=vbul_max,
        vbul_frac=vbul_frac,
        vdisk_max=vdisk_max,
        vgas_max=vgas_max,
        vobs_max=vobs_max,
        bul_energy_frac=bul_energy_frac,
        n_points=mask.sum(),
        r=r, v_obs=v_obs, v_err=data[:,2][mask],
        v_gas=v_gas, v_disk=v_disk, v_bul=v_bul,
    )

# ============================================================
# Main
# ============================================================
galaxies = load_2stage_list()
print(f"2-stage galaxies from CSV: {len(galaxies)}")
print(f"List: {galaxies}\n")

# --- Summary ---
print(f"{'='*75}")
print(f"  Bulge component analysis for 2-stage galaxies")
print(f"{'='*75}")
print(f"{'galaxy':<16s} {'vbul_max':>9} {'vbul%':>6} {'vdisk_max':>10} {'vobs_max':>9} {'bul_E%':>7} {'verdict':>22}")
print("-"*75)

cond14_candidates = []
baryonic_complex  = []

all_info = {}
for gal in galaxies:
    info = analyze_vbul(gal)
    if info is None:
        print(f"{gal:<16s} {'[FAIL]':>50}")
        continue
    all_info[gal] = info

    vbmax = info['vbul_max']
    vbfrac = info['vbul_frac']
    be_frac = info['bul_energy_frac']

    if info['has_bul']:
        verdict = 'Hypothesis B (bulge)'
        baryonic_complex.append(gal)
    else:
        verdict = '* Condition 14 candidate'
        cond14_candidates.append(gal)

    print(f"{gal:<16s} {vbmax:8.1f} {vbfrac*100:5.1f}% {info['vdisk_max']:10.1f} "
          f"{info['vobs_max']:9.1f} {be_frac*100:6.1f}%  {verdict}")

print("="*75)
print(f"\nCondition 14 candidates (no bulge): {len(cond14_candidates)}")
for g in cond14_candidates:
    print(f"  -> {g}")
print(f"\nHypothesis B (bulge present): {len(baryonic_complex)}")
for g in baryonic_complex:
    print(f"  -> {g}")

# --- Verdict ---
print(f"\n{'='*75}")
if len(cond14_candidates) == 0:
    print("  CONCLUSION: No independent evidence for Condition 14 (two-phase)")
    print("  -> 2-stage necessity explained by baryonic complexity")
    print("  -> Condition 14 remains theoretical hypothesis only")
elif len(cond14_candidates) >= 3:
    print(f"  CONCLUSION: {len(cond14_candidates)} galaxies are Condition 14 candidates")
    print("  -> Bulge-free galaxies needing 2-stage = membrane two-phase evidence")
    print("  -> Condition 14 observationally supported")
else:
    print(f"  CONCLUSION: {len(cond14_candidates)} candidate(s) (weak evidence)")
    print("  -> Further investigation needed")
print("="*75)

# ============================================================
# Plot: component decomposition for each galaxy
# ============================================================
n = len(all_info)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (gal, info) in enumerate(all_info.items()):
    ax = axes[i]
    r = info['r']

    ax.errorbar(r, info['v_obs'], yerr=info['v_err'], fmt='ko', ms=3,
                capsize=1.5, zorder=5, label='data')
    ax.plot(r, np.abs(info['v_gas']), 'b:', lw=1.2, label='gas')
    ax.plot(r, np.abs(info['v_disk']), 'c--', lw=1.2, label='disk')
    if info['vbul_max'] > 1.0:
        ax.plot(r, np.abs(info['v_bul']), 'm-.', lw=1.5, label='bulge')

    has_b = info['has_bul']
    vbmax = info['vbul_max']
    title_color = 'red' if has_b else 'green'
    tag = f'BULGE (max={vbmax:.0f})' if has_b else f'no bulge (max={vbmax:.0f})'
    ax.set_title(f'{gal}\n{tag}', fontsize=9, color=title_color, fontweight='bold')
    ax.set_xlabel('r [kpc]', fontsize=7)
    ax.set_ylabel('v [km/s]', fontsize=7)
    ax.legend(fontsize=6)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.15)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('2-stage galaxies: bulge component check\n'
             'Red = bulge present (Hyp. B), Green = no bulge (Cond. 14 candidate)',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath = OUT_DIR / 'vbul_check_11galaxies.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\n-> {outpath}")
