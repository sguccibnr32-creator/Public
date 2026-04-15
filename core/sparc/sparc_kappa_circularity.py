#!/usr/bin/env python3
"""
sparc_kappa_circularity.py (TA3+phase1 adapted)
Circularity test for M3/M6/M8 R^2=0.55
"""
import os, csv, warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
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
    if len(rad) < 8: return None
    return {'r': np.array(rad), 'vobs': np.array(vobs),
            'vgas': np.array(vgas), 'vdisk': np.array(vdisk), 'vbul': np.array(vbul)}

def compute_hR(rc, Yd):
    vds = np.sqrt(max(Yd, 0.01)) * np.abs(rc['vdisk'])
    rpk = rc['r'][np.argmax(vds)]
    if rpk < 0.01 or rpk >= rc['r'].max() * 0.9: return None
    return rpk / 2.15

def compute_gc_deep_from_indices(rc, Yd, indices):
    r_m = rc['r'][indices] * kpc_m
    v_bar2 = (Yd * rc['vdisk'][indices]**2 + rc['vgas'][indices]**2 + rc['vbul'][indices]**2) * 1e6
    v_obs2 = (rc['vobs'][indices] * 1e3)**2
    gN = v_bar2 / r_m
    gobs = v_obs2 / r_m
    mask = gN > 0
    if np.sum(mask) < 3: return None
    gc_pts = gobs[mask]**2 / gN[mask]
    gc_med = np.median(gc_pts)
    if gc_med <= 0: return None
    x = gN[mask] / gc_med
    deep = x < 1.0
    return np.median(gc_pts[deep]) if np.sum(deep) >= 3 else gc_med

def compute_gc_deep(rc, Yd=0.5):
    return compute_gc_deep_from_indices(rc, Yd, np.arange(len(rc['r'])))

def compute_quantities(rc, gc_deep, Yd=0.5):
    r_m = rc['r'] * kpc_m
    v_bar2 = (Yd * rc['vdisk']**2 + rc['vgas']**2 + rc['vbul']**2) * 1e6
    v_obs2 = (rc['vobs'] * 1e3)**2
    gN = np.maximum(v_bar2 / r_m, 0)
    epsilon = np.sqrt(gN / gc_deep)
    u_elastic = np.mean(epsilon**2)
    E_kin = np.mean(v_obs2)
    strain_kin = u_elastic / (E_kin / kpc_m**2) if E_kin > 0 else 0
    eps_max = np.max(epsilon)
    if len(epsilon) >= 5:
        d2 = np.gradient(np.gradient(epsilon, r_m), r_m)
        curv = np.mean(d2**2)
    else:
        curv = 0
    return {'strain_kin': strain_kin, 'eps_max': eps_max, 'curvature': curv}

def fit_M6(log_excess, log_sk):
    finite = np.isfinite(log_sk) & np.isfinite(log_excess)
    if np.sum(finite) < 10: return None, 999
    X = log_sk[finite].reshape(-1, 1)
    sol, _, _, _ = lstsq(X, log_excess[finite], rcond=None)
    pred = sol[0] * log_sk
    pred[~finite] = 0
    chi2 = np.sum((log_excess - pred)**2)
    return sol[0], chi2

def fit_M3(log_excess, log_em):
    finite = np.isfinite(log_em) & np.isfinite(log_excess)
    if np.sum(finite) < 10: return None, None, 999
    lem = log_em[finite]; e = log_excess[finite]
    best = None
    for lec in np.linspace(np.percentile(lem, 10), np.percentile(lem, 90), 20):
        for b in np.linspace(-1, 1, 20):
            excess = np.maximum(lem - lec, 0)
            chi2 = np.sum((e - b * excess)**2)
            if best is None or chi2 < best[0]:
                best = (chi2, lec, b)
    def chi2_f(p):
        exc = np.maximum(lem - p[0], 0)
        return np.sum((e - p[1] * exc)**2)
    res = minimize(chi2_f, [best[1], best[2]], method='Nelder-Mead')
    return res.x[0], res.x[1], res.fun

def main():
    print("=" * 70)
    print("Circularity test: M3/M6/M8 R^2=0.55 real or artifact?")
    print("=" * 70)

    pipe = load_pipeline()
    galaxies = []
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
        sq = compute_quantities(rc, gc_deep, Yd)
        if sq['strain_kin'] <= 0: continue
        galaxies.append({'name': gname, 'gc_obs': gc_obs, 'gc_deep': gc_deep,
                         'vflat': vflat, 'hR': hR, 'Yd': Yd, 'rc': rc, **sq})

    N = len(galaxies)
    print(f"Galaxies: {N}\n")
    if N < 50:
        print("Too few galaxies."); return

    gc_obs = np.array([g['gc_obs'] for g in galaxies])
    gc_deep = np.array([g['gc_deep'] for g in galaxies])
    strain_kin = np.array([g['strain_kin'] for g in galaxies])
    eps_max = np.array([g['eps_max'] for g in galaxies])
    vflat = np.array([g['vflat'] for g in galaxies])
    hR = np.array([g['hR'] for g in galaxies])

    log_excess = np.log10(gc_obs / gc_deep)
    log_sk = np.log10(np.maximum(strain_kin, 1e-50))
    log_em = np.log10(np.maximum(eps_max, 1e-10))
    chi2_null = np.sum(log_excess**2)

    a_m6, chi2_m6 = fit_M6(log_excess, log_sk)
    R2_m6 = 1 - chi2_m6 / chi2_null
    lec_m3, beta_m3, chi2_m3 = fit_M3(log_excess, log_em)
    R2_m3 = 1 - chi2_m3 / chi2_null
    print(f"Baseline M6: a={a_m6:.4f}, R^2={R2_m6:.4f}")
    print(f"Baseline M3: eps_c={10**lec_m3:.4f}, beta={beta_m3:.3f}, R^2={R2_m3:.4f}")

    # T1: Split-half
    print("\n" + "=" * 70)
    print("T1: Split-half gc_deep (50 trials)")
    print("=" * 70)
    np.random.seed(42)
    R2_split_m6, R2_split_m3 = [], []
    for trial in range(50):
        gc_A = np.zeros(N); gc_B = np.zeros(N)
        valid = np.ones(N, dtype=bool)
        for i, g in enumerate(galaxies):
            rc = g['rc']
            n_pts = len(rc['r'])
            perm = np.random.permutation(n_pts)
            half = n_pts // 2
            gA = compute_gc_deep_from_indices(rc, g['Yd'], perm[:half])
            gB = compute_gc_deep_from_indices(rc, g['Yd'], perm[half:])
            if gA is None or gB is None or gA <= 0 or gB <= 0:
                valid[i] = False; continue
            gc_A[i] = gA; gc_B[i] = gB
        if np.sum(valid) < 50: continue
        le_B = np.log10(gc_obs[valid] / gc_B[valid])
        sk_A, em_A = [], []
        for i in np.where(valid)[0]:
            sq = compute_quantities(galaxies[i]['rc'], gc_A[i], galaxies[i]['Yd'])
            sk_A.append(sq['strain_kin']); em_A.append(sq['eps_max'])
        log_sk_A = np.log10(np.maximum(np.array(sk_A), 1e-50))
        log_em_A = np.log10(np.maximum(np.array(em_A), 1e-10))
        a_s, chi2_s = fit_M6(le_B, log_sk_A)
        if a_s is not None:
            R2_split_m6.append(1 - chi2_s / np.sum(le_B**2))
        _, _, chi2_s3 = fit_M3(le_B, log_em_A)
        R2_split_m3.append(1 - chi2_s3 / np.sum(le_B**2))
    R2_split_m6 = np.array(R2_split_m6)
    R2_split_m3 = np.array(R2_split_m3)
    print(f"M6 split-half R^2: mean={np.mean(R2_split_m6):.4f} "
          f"(base={R2_m6:.4f}, ratio={np.mean(R2_split_m6)/R2_m6:.2f})")
    print(f"M3 split-half R^2: mean={np.mean(R2_split_m3):.4f} "
          f"(base={R2_m3:.4f}, ratio={np.mean(R2_split_m3)/R2_m3:.2f})")

    # T2: shuffle
    print("\n" + "=" * 70)
    print("T2: gc_deep shuffle (500 trials)")
    print("=" * 70)
    np.random.seed(42)
    R2_shuf = []
    for _ in range(500):
        perm = np.random.permutation(N)
        gc_s = gc_deep[perm]
        le_s = np.log10(gc_obs / gc_s)
        sk_s = []
        for i in range(N):
            sq = compute_quantities(galaxies[i]['rc'], gc_s[i], galaxies[i]['Yd'])
            sk_s.append(sq['strain_kin'])
        log_sk_s = np.log10(np.maximum(np.array(sk_s), 1e-50))
        a_s, chi2_s = fit_M6(le_s, log_sk_s)
        if a_s is not None:
            R2_shuf.append(1 - chi2_s / np.sum(le_s**2))
    R2_shuf = np.array(R2_shuf)
    p_val = np.mean(R2_shuf >= R2_m6)
    print(f"M6 shuffle R^2: mean={np.mean(R2_shuf):.4f}, max={np.max(R2_shuf):.4f}")
    print(f"Real R^2={R2_m6:.4f}, p={p_val:.4f}")

    # T3: gc_obs direct (no gc_deep)
    print("\n" + "=" * 70)
    print("T3: Direct gc_obs prediction without gc_deep")
    print("=" * 70)
    log_gc_obs = np.log10(gc_obs)
    gN_mean, E_kin = [], []
    for g in galaxies:
        rc = g['rc']
        r_m = rc['r'] * kpc_m
        v_bar2 = (g['Yd'] * rc['vdisk']**2 + rc['vgas']**2 + rc['vbul']**2) * 1e6
        v_obs2 = (rc['vobs'] * 1e3)**2
        gN_mean.append(np.mean(np.maximum(v_bar2 / r_m, 0)))
        E_kin.append(np.mean(v_obs2))
    gN_mean = np.array(gN_mean); E_kin = np.array(E_kin)
    proxy = gN_mean / (E_kin / kpc_m**2)
    log_proxy = np.log10(np.maximum(proxy, 1e-50))
    finite = np.isfinite(log_proxy)
    X = np.column_stack([log_proxy[finite], np.ones(np.sum(finite))])
    sol, _, _, _ = lstsq(X, log_gc_obs[finite], rcond=None)
    pred = sol[0] * log_proxy + sol[1]
    chi2_null_d = np.sum((log_gc_obs - np.mean(log_gc_obs))**2)
    R2_direct = 1 - np.sum((log_gc_obs - pred)**2) / chi2_null_d
    print(f"gc_obs ~ proxy (gc_deep-free): R^2={R2_direct:.4f}")
    log_GS = np.log10(vflat**2 / hR)
    X2 = np.column_stack([log_GS, np.ones(N)])
    sol2, _, _, _ = lstsq(X2, log_gc_obs, rcond=None)
    R2_gs = 1 - np.sum((log_gc_obs - (sol2[0]*log_GS + sol2[1]))**2) / chi2_null_d
    print(f"gc_obs ~ vflat^2/hR (geometric): R^2={R2_gs:.4f}")
    X3 = np.column_stack([log_proxy[finite], log_GS[finite], np.ones(np.sum(finite))])
    sol3, _, _, _ = lstsq(X3, log_gc_obs[finite], rcond=None)
    pred3 = sol3[0]*log_proxy + sol3[1]*log_GS + sol3[2]
    R2_comb = 1 - np.sum((log_gc_obs - pred3)**2) / chi2_null_d
    print(f"gc_obs ~ proxy + vflat^2/hR: R^2={R2_comb:.4f}")
    print(f"  dR^2 from proxy over geometric: {R2_comb - R2_gs:.4f}")

    # T4: perturbation
    print("\n" + "=" * 70)
    print("T4: gc_deep perturbation")
    print("=" * 70)
    np.random.seed(42)
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        R2_p = []
        for _ in range(50):
            noise = 10**(np.random.normal(0, sigma, N))
            gc_p = gc_deep * noise
            le_p = np.log10(gc_obs / gc_p)
            sk_p = []
            for i in range(N):
                sq = compute_quantities(galaxies[i]['rc'], gc_p[i], galaxies[i]['Yd'])
                sk_p.append(sq['strain_kin'])
            log_sk_p = np.log10(np.maximum(np.array(sk_p), 1e-50))
            a_p, chi2_p = fit_M6(le_p, log_sk_p)
            if a_p is not None:
                R2_p.append(1 - chi2_p / np.sum(le_p**2))
        print(f"  sigma={sigma:.2f} dex: R^2={np.mean(R2_p):.4f} "
              f"(ratio={np.mean(R2_p)/R2_m6:.2f})")

    # T5: direct correlation
    print("\n" + "=" * 70)
    print("T5: Direct correlation")
    print("=" * 70)
    rho, p = spearmanr(log_excess, log_sk)
    print(f"rho(log_excess, log_strain_kin) = {rho:+.3f} (p={p:.2e})")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  T1 Split-half M6: {np.mean(R2_split_m6):.4f} / {R2_m6:.4f} = {np.mean(R2_split_m6)/R2_m6:.2f}
  T1 Split-half M3: {np.mean(R2_split_m3):.4f} / {R2_m3:.4f} = {np.mean(R2_split_m3)/R2_m3:.2f}
  T2 Shuffle: mean={np.mean(R2_shuf):.4f}, p={p_val:.4f}
  T3 Direct proxy: R^2={R2_direct:.4f}, dR^2 over vflat^2/hR = {R2_comb-R2_gs:.4f}
  T5 rho = {rho:+.3f}

Criterion:
  split>0.7 AND p<0.01 AND dR^2>0.02  -> real signal
  split<0.3 OR shuffle R^2>0.3        -> circular artifact
  otherwise mixed
""")
    print("Done.")

if __name__ == '__main__':
    main()
