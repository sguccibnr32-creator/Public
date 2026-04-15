#!/usr/bin/env python3
"""
sparc_eta_derivation.py (TA3+phase1 adapted)
First-principles derivation of eta from deep-MOND gas fraction.
"""
import os, csv, warnings
import numpy as np
from scipy.stats import spearmanr, pearsonr
from numpy.linalg import lstsq
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
a0 = 1.2e-10
kpc_m = 3.086e19

def load_pipeline():
    data = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                data[name] = {'vflat': float(row.get('vflat', '0')),
                              'Yd': float(row.get('ud', '0.5'))}
            except: pass
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                if name in data and gc_a0 > 0:
                    data[name]['gc'] = gc_a0 * a0
            except: pass
    return {k: v for k, v in data.items() if 'gc' in v and v['vflat'] > 0}

def load_rotcurve(gname):
    fname = os.path.join(ROTMOD, f"{gname}_rotmod.dat")
    if not os.path.exists(fname): return None
    rad, vobs, vgas, vdisk, vbul = [], [], [], [], []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split()
            if len(p) < 6: continue
            try:
                rad.append(float(p[0])); vobs.append(float(p[1]))
                vgas.append(float(p[3])); vdisk.append(float(p[4])); vbul.append(float(p[5]))
            except: continue
    if len(rad) < 5: return None
    return {'r': np.array(rad), 'vobs': np.array(vobs),
            'vgas': np.array(vgas), 'vdisk': np.array(vdisk), 'vbul': np.array(vbul)}

def compute_hR(rc, Yd):
    vds = np.sqrt(max(Yd, 0.01)) * np.abs(rc['vdisk'])
    rpk = rc['r'][np.argmax(vds)]
    if rpk < 0.01 or rpk >= rc['r'].max() * 0.9: return None
    return rpk / 2.15

def compute_gc_deep(rc, Yd=0.5):
    r_m = rc['r'] * kpc_m
    v_bar2 = (Yd * rc['vdisk']**2 + rc['vgas']**2 + rc['vbul']**2) * 1e6
    v_obs2 = (rc['vobs'] * 1e3)**2
    gN = v_bar2 / r_m; gobs = v_obs2 / r_m
    mask = gN > 0
    if np.sum(mask) < 3: return None
    gc_pts = gobs[mask]**2 / gN[mask]
    gc_med = np.median(gc_pts)
    if gc_med <= 0: return None
    x = gN[mask] / gc_med
    deep = x < 1.0
    return np.median(gc_pts[deep]) if np.sum(deep) >= 3 else gc_med

def main():
    print("=" * 70)
    print("eta first-principles: deep-MOND gas fraction hypothesis")
    print("=" * 70)
    print("""
Theory: eta ~ Yd^beta with beta = -(1 - f_gas_deep)
Observed: beta = -0.41 --> predicted f_gas_deep = 0.59
""")
    pipe = load_pipeline()
    results = []
    for gname in sorted(pipe.keys()):
        gd = pipe[gname]
        gc_obs, vflat = gd['gc'], gd['vflat']
        Yd = gd.get('Yd', 0.5)
        if gc_obs <= 0 or vflat <= 0: continue
        rc = load_rotcurve(gname)
        if rc is None: continue
        hR = compute_hR(rc, Yd)
        if hR is None: continue
        gc_deep = compute_gc_deep(rc, Yd)
        if gc_deep is None or gc_deep <= 0: continue

        r_m = rc['r'] * kpc_m
        v_disk2 = rc['vdisk']**2 * 1e6
        v_gas2 = rc['vgas']**2 * 1e6
        v_bul2 = rc['vbul']**2 * 1e6
        gN_disk = Yd * v_disk2 / r_m
        gN_gas = v_gas2 / r_m
        gN_bul = v_bul2 / r_m
        gN_tot = gN_disk + gN_gas + gN_bul

        x = gN_tot / gc_deep
        deep = (x < 1.0) & (gN_tot > 0)
        if np.sum(deep) < 3: continue

        gN_nonYd = gN_gas[deep] + gN_bul[deep]
        gN_tot_d = gN_tot[deep]
        f_gas = np.sum(gN_nonYd) / np.sum(gN_tot_d)
        w = gN_tot_d**2
        f_gas_w = np.sum(gN_nonYd * w / gN_tot_d) / np.sum(w)
        f_gas_pts = gN_nonYd / gN_tot_d

        GSigma0 = (vflat * 1e3)**2 / (hR * kpc_m)
        eta_obs = gc_obs / np.sqrt(a0 * GSigma0)
        beta_pred = -(1 - f_gas)

        results.append({
            'name': gname, 'rc': rc, 'gc_obs': gc_obs, 'gc_deep': gc_deep,
            'vflat': vflat, 'hR': hR, 'Yd': Yd,
            'f_gas': f_gas, 'f_gas_w': f_gas_w,
            'f_gas_med': np.median(f_gas_pts),
            'beta_pred': beta_pred, 'eta_obs': eta_obs,
            'n_deep': int(np.sum(deep)),
        })

    N = len(results)
    print(f"Galaxies: {N}\n")
    if N < 30:
        print("Too few."); return

    f_gas = np.array([r['f_gas'] for r in results])
    f_gas_w = np.array([r['f_gas_w'] for r in results])
    beta_pred = np.array([r['beta_pred'] for r in results])
    eta_obs = np.array([r['eta_obs'] for r in results])
    Yd = np.array([r['Yd'] for r in results])
    vflat = np.array([r['vflat'] for r in results])
    hR = np.array([r['hR'] for r in results])

    # (1) f_gas distribution
    print("=" * 70)
    print("(1) f_gas_deep distribution")
    print("=" * 70)
    print(f"  mean   = {np.mean(f_gas):.4f}")
    print(f"  median = {np.median(f_gas):.4f}")
    print(f"  std    = {np.std(f_gas):.4f}")
    print(f"  IQR    = [{np.percentile(f_gas,25):.4f}, {np.percentile(f_gas,75):.4f}]")
    print(f"  weighted median = {np.median(f_gas_w):.4f}")
    print(f"  predicted = 0.59 (from beta_obs=-0.41)")
    print(f"  delta = {abs(np.median(f_gas)-0.59):.4f} "
          f"({abs(np.median(f_gas)-0.59)/0.59*100:.1f}%)")

    # (2) beta test
    print("\n" + "=" * 70)
    print("(2) beta prediction")
    print("=" * 70)
    log_eta = np.log10(eta_obs)
    log_Yd = np.log10(Yd)
    X = np.column_stack([log_Yd, np.ones(N)])
    sol, _, _, _ = lstsq(X, log_eta, rcond=None)
    beta_global = sol[0]
    print(f"  beta_obs (global OLS)    = {beta_global:.4f}")
    print(f"  beta_obs (expected)      = -0.41")
    print(f"  beta_pred median         = {np.median(beta_pred):.4f}")
    print(f"  -(1-median(f_gas))       = {-(1-np.median(f_gas)):.4f}")

    # (3) model comparison
    print("\n" + "=" * 70)
    print("(3) Model comparison")
    print("=" * 70)
    null_var = np.sum((log_eta - np.mean(log_eta))**2)
    # A: Yd^-0.41 fixed
    pred_A_core = -0.41 * log_Yd
    C_A = np.median(log_eta - pred_A_core)
    pred_A = C_A + pred_A_core
    R2_A = 1 - np.sum((log_eta - pred_A)**2)/null_var
    # B: per-galaxy
    pred_B_core = beta_pred * log_Yd
    C_B = np.median(log_eta - pred_B_core)
    pred_B = C_B + pred_B_core
    R2_B = 1 - np.sum((log_eta - pred_B)**2)/null_var
    # C: global fit
    pred_C = sol[1] + sol[0]*log_Yd
    R2_C = 1 - np.sum((log_eta - pred_C)**2)/null_var
    print(f"  A: Yd^-0.41 fixed:     R2={R2_A:.4f}")
    print(f"  B: Yd^-(1-f_gas_i):    R2={R2_B:.4f}")
    print(f"  C: Yd^beta_fit:        R2={R2_C:.4f}, beta={sol[0]:.4f}")

    # (4) f_gas determinants
    print("\n" + "=" * 70)
    print("(4) f_gas determinants")
    print("=" * 70)
    for pname, pvals in [('log(Yd)', log_Yd),
                          ('log(vflat)', np.log10(vflat)),
                          ('log(hR)', np.log10(hR)),
                          ('log(vflat^2/hR)', np.log10(vflat**2/hR))]:
        rho, p = spearmanr(f_gas, pvals)
        print(f"  rho(f_gas, {pname:18s}) = {rho:+.3f} (p={p:.2e})")

    # (5) numerical derivative
    print("\n" + "=" * 70)
    print("(5) Numerical d ln(gc)/d ln(Yd)")
    print("=" * 70)
    beta_num = []
    beta_pred_matched = []
    for r in results:
        rc = r['rc']; Yd0 = r['Yd']
        gc_lo = compute_gc_deep(rc, Yd0*0.9)
        gc_hi = compute_gc_deep(rc, Yd0*1.1)
        if gc_lo is None or gc_hi is None or gc_lo <= 0 or gc_hi <= 0: continue
        b = (np.log(gc_hi) - np.log(gc_lo)) / (np.log(1.1) - np.log(0.9))
        beta_num.append(b)
        beta_pred_matched.append(r['beta_pred'])
    beta_num = np.array(beta_num)
    beta_pred_matched = np.array(beta_pred_matched)
    print(f"  numerical: mean={np.mean(beta_num):.4f}, median={np.median(beta_num):.4f}")
    print(f"  analytical median = {np.median(beta_pred_matched):.4f}")
    print(f"  observed eta exponent = -0.41")
    if len(beta_num) > 5:
        r_bn, p_bn = pearsonr(beta_num, beta_pred_matched)
        print(f"  r(num, ana) = {r_bn:.4f} (p={p_bn:.2e})")

    # (6) residual std
    print("\n" + "=" * 70)
    print("(6) Residual scatter")
    print("=" * 70)
    print(f"  total log_eta std  = {np.std(log_eta):.4f}")
    print(f"  A residual std     = {np.std(log_eta - pred_A):.4f}")
    print(f"  B residual std     = {np.std(log_eta - pred_B):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    med_f = np.median(f_gas)
    print(f"""
  f_gas_deep median  = {med_f:.4f} (predicted: 0.59)
  beta_predicted     = {np.median(beta_pred):.4f} (obs: -0.41, global fit: {beta_global:.4f})
  beta_numerical     = {np.median(beta_num):.4f}
  Model A R2         = {R2_A:.4f}
  Model B R2         = {R2_B:.4f}

Criteria:
  (a) f_gas_deep in [0.50, 0.68]?  {'YES' if 0.50 <= med_f <= 0.68 else 'NO'}
  (b) beta_num approx beta_ana?    r={r_bn:.3f} {'YES' if abs(r_bn) > 0.5 else 'NO'}
  (c) R2_B >= R2_A?                {'YES' if R2_B >= R2_A else 'NO'}
""")
    print("Done.")

if __name__ == '__main__':
    main()
