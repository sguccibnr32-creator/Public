#!/usr/bin/env python3
"""
sparc_M3_M6_independence.py (TA3+phase1 adapted)
M3 (eps threshold) vs M6 (u_strain/E_kin) independence test.
"""
import os, csv, warnings
import numpy as np
from scipy.optimize import minimize
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

def compute_quantities(rc, gc_deep, Yd=0.5):
    r_m = rc['r'] * kpc_m
    v_bar2 = (Yd * rc['vdisk']**2 + rc['vgas']**2 + rc['vbul']**2) * 1e6
    v_obs2 = (rc['vobs'] * 1e3)**2
    gN = np.maximum(v_bar2 / r_m, 0)
    epsilon = np.sqrt(gN / gc_deep)
    u_el = np.mean(epsilon**2)
    E_kin = np.mean(v_obs2)
    strain_kin = u_el / (E_kin / kpc_m**2) if E_kin > 0 else 0
    eps_max = np.max(epsilon)
    if len(epsilon) >= 5:
        d2 = np.gradient(np.gradient(epsilon, r_m), r_m)
        curv = np.mean(d2**2)
    else:
        curv = 0
    deps = np.gradient(epsilon, r_m)
    E_grad = np.mean(deps**2)
    ratio = E_grad / u_el if u_el > 0 else 0
    return {'strain_kin': strain_kin, 'eps_max': eps_max,
            'curvature': curv, 'ratio': ratio, 'eps_mean': np.mean(epsilon)}

def build_dataset():
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
        sq = compute_quantities(rc, gc_deep, Yd)
        if sq['strain_kin'] <= 0: continue
        results.append({'name': gname, 'gc_obs': gc_obs, 'gc_deep': gc_deep,
                        'vflat': vflat, 'hR': hR, 'Yd': Yd,
                        'rmax_hR': rc['r'][-1]/hR if hR > 0 else 0, **sq})
    print(f"Dataset: {len(results)} galaxies\n")
    return results

def fit_M3(log_excess, log_em, mask=None):
    if mask is None: mask = np.ones(len(log_excess), dtype=bool)
    lem = log_em[mask]; e = log_excess[mask]
    best = None
    for lec in np.linspace(np.percentile(lem, 5), np.percentile(lem, 95), 20):
        for b in np.linspace(-1, 1, 20):
            exc = np.maximum(lem - lec, 0)
            chi2 = np.sum((e - b * exc)**2)
            if best is None or chi2 < best[0]:
                best = (chi2, lec, b)
    res = minimize(lambda p: np.sum((e - p[1]*np.maximum(lem-p[0], 0))**2),
                   [best[1], best[2]], method='Nelder-Mead')
    return res.x

def pred_M3(params, log_em):
    log_ec, beta = params
    return beta * np.maximum(log_em - log_ec, 0)

def fit_M6(log_excess, log_sk, mask=None):
    if mask is None: mask = np.ones(len(log_excess), dtype=bool)
    finite = np.isfinite(log_sk[mask])
    X = log_sk[mask][finite].reshape(-1, 1)
    sol, _, _, _ = lstsq(X, log_excess[mask][finite], rcond=None)
    return sol[0]

def pred_M6(a, log_sk):
    return a * log_sk

def aicc(chi2, n, k):
    aic = chi2 + 2*k
    if n - k - 1 > 0:
        return aic + 2*k*(k+1) / (n - k - 1)
    return aic

def partial_corr(x, y, z):
    cx = np.polyfit(z, x, 1); cy = np.polyfit(z, y, 1)
    return spearmanr(x - np.polyval(cx, z), y - np.polyval(cy, z))

def partial_corr_multi(target, y, controls):
    X = np.column_stack([*controls, np.ones(len(target))])
    ct, _, _, _ = lstsq(X, target, rcond=None)
    cy, _, _, _ = lstsq(X, y, rcond=None)
    return spearmanr(target - X @ ct, y - X @ cy)

def main():
    print("=" * 70)
    print("M3 vs M6 independence test")
    print("=" * 70)

    data = build_dataset()
    N = len(data)
    if N < 50:
        print("Too few galaxies."); return

    gc_obs = np.array([d['gc_obs'] for d in data])
    gc_deep = np.array([d['gc_deep'] for d in data])
    strain_kin = np.array([d['strain_kin'] for d in data])
    eps_max = np.array([d['eps_max'] for d in data])
    curvature = np.array([d['curvature'] for d in data])
    vflat = np.array([d['vflat'] for d in data])
    hR = np.array([d['hR'] for d in data])
    rmax_hR = np.array([d['rmax_hR'] for d in data])

    log_excess = np.log10(gc_obs / gc_deep)
    log_sk = np.log10(np.maximum(strain_kin, 1e-50))
    log_em = np.log10(np.maximum(eps_max, 1e-10))
    log_curv = np.log10(np.maximum(curvature, 1e-80))
    log_vf = np.log10(vflat)
    log_hR = np.log10(hR)
    chi2_null = np.sum(log_excess**2)

    params_M3 = fit_M3(log_excess, log_em)
    pred_m3 = pred_M3(params_M3, log_em)
    resid_m3 = log_excess - pred_m3
    chi2_m3 = np.sum(resid_m3**2)

    a_M6 = fit_M6(log_excess, log_sk)
    pred_m6 = pred_M6(a_M6, log_sk)
    resid_m6 = log_excess - pred_m6
    chi2_m6 = np.sum(resid_m6**2)

    print(f"M3: log_ec={params_M3[0]:.3f} (eps_c={10**params_M3[0]:.4f}), "
          f"beta={params_M3[1]:.3f}, R2={1-chi2_m3/chi2_null:.4f}")
    print(f"M6: a={a_M6:.4f}, R2={1-chi2_m6/chi2_null:.4f}")

    # T1
    print("\n" + "=" * 70)
    print("T1: residual correlation")
    print("=" * 70)
    rho_resid, p_resid = spearmanr(resid_m3, resid_m6)
    rho_resid_p, p_resid_p = pearsonr(resid_m3, resid_m6)
    print(f"Spearman rho(resid_M3, resid_M6) = {rho_resid:+.4f} (p={p_resid:.2e})")
    print(f"Pearson  r(resid_M3, resid_M6)   = {rho_resid_p:+.4f} (p={p_resid_p:.2e})")

    # T2
    print("\n" + "=" * 70)
    print("T2: Combined M3+M6")
    print("=" * 70)
    def fit_combined(le, lem, lsk, mask=None):
        if mask is None: mask = np.ones(len(le), dtype=bool)
        e = le[mask]; em = lem[mask]; sk = lsk[mask]
        def chi2f(p):
            lec, b, a = p
            return np.sum((e - b*np.maximum(em-lec, 0) - a*sk)**2)
        best = None
        for lec0 in np.linspace(np.percentile(em, 10), np.percentile(em, 90), 8):
            for b0 in [-0.3, -0.1, 0.1]:
                for a0i in [-0.01, 0.0, 0.01]:
                    res = minimize(chi2f, [lec0, b0, a0i], method='Nelder-Mead',
                                   options={'maxiter': 3000})
                    if best is None or res.fun < best.fun:
                        best = res
        return best.x, best.fun

    params_comb, chi2_comb = fit_combined(log_excess, log_em, log_sk)
    R2_comb = 1 - chi2_comb / chi2_null
    aicc_null = aicc(chi2_null, N, 0)
    aicc_m3 = aicc(chi2_m3, N, 2)
    aicc_m6 = aicc(chi2_m6, N, 1)
    aicc_comb = aicc(chi2_comb, N, 3)
    print(f"Combined: log_ec={params_comb[0]:.3f}, beta={params_comb[1]:.3f}, "
          f"a_M6={params_comb[2]:.4f}")
    print(f"  {'Model':<22s} {'k':>3s} {'R2':>7s} {'dAICc':>8s}")
    print(f"  {'M0 null':<22s} {0:>3d} {0.0:>7.4f} {0.0:>+8.1f}")
    print(f"  {'M6':<22s} {1:>3d} {1-chi2_m6/chi2_null:>7.4f} {aicc_m6-aicc_null:>+8.1f}")
    print(f"  {'M3':<22s} {2:>3d} {1-chi2_m3/chi2_null:>7.4f} {aicc_m3-aicc_null:>+8.1f}")
    print(f"  {'M3+M6':<22s} {3:>3d} {R2_comb:>7.4f} {aicc_comb-aicc_null:>+8.1f}")
    dA_vs_m3 = aicc_comb - aicc_m3
    dA_vs_m6 = aicc_comb - aicc_m6
    print(f"\n  M3+M6 vs M3: dAICc={dA_vs_m3:+.1f}")
    print(f"  M3+M6 vs M6: dAICc={dA_vs_m6:+.1f}")

    # T2b: LOO-CV
    print("\n--- LOO-CV ---")
    loo_comb, loo_m3, loo_m6 = [], [], []
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        try:
            p_c, _ = fit_combined(log_excess, log_em, log_sk, mask)
            pred_i = p_c[1]*max(log_em[i]-p_c[0], 0) + p_c[2]*log_sk[i]
            loo_comb.append((log_excess[i] - pred_i)**2)
        except: pass
        p3 = fit_M3(log_excess, log_em, mask)
        pred_i3 = p3[1]*max(log_em[i]-p3[0], 0)
        loo_m3.append((log_excess[i] - pred_i3)**2)
        a6 = fit_M6(log_excess, log_sk, mask)
        loo_m6.append((log_excess[i] - a6*log_sk[i])**2)
    mse_null = np.mean(log_excess**2)
    mse_m3 = np.mean(loo_m3); mse_m6 = np.mean(loo_m6); mse_comb = np.mean(loo_comb)
    print(f"  MSE: null={mse_null:.5f} M6={mse_m6:.5f} M3={mse_m3:.5f} Comb={mse_comb:.5f}")
    print(f"  ratio: M6={mse_m6/mse_null:.3f} M3={mse_m3/mse_null:.3f} Comb={mse_comb/mse_null:.3f}")

    # T3
    print("\n" + "=" * 70)
    print("T3: Incremental R2")
    print("=" * 70)
    sol_a, _, _, _ = lstsq(log_sk.reshape(-1,1), resid_m3, rcond=None)
    pred_a = sol_a[0]*log_sk
    R2_m6on3 = 1 - np.sum((resid_m3 - pred_a)**2)/np.sum(resid_m3**2)
    exc_m3 = np.maximum(log_em - params_M3[0], 0)
    sol_b, _, _, _ = lstsq(exc_m3.reshape(-1,1), resid_m6, rcond=None)
    pred_b = sol_b[0]*exc_m3
    R2_m3on6 = 1 - np.sum((resid_m6 - pred_b)**2)/np.sum(resid_m6**2)
    print(f"M6 on M3 residuals: R2={R2_m6on3:.4f}")
    print(f"M3 on M6 residuals: R2={R2_m3on6:.4f}")

    # T4
    print("\n" + "=" * 70)
    print("T4: hR partial correlation reduction")
    print("=" * 70)
    gc_m3 = gc_deep * 10**pred_m3
    gc_m6 = gc_deep * 10**pred_m6
    gc_comb = gc_deep * 10**(params_comb[1]*np.maximum(log_em-params_comb[0], 0)
                              + params_comb[2]*log_sk)
    log_gc_obs = np.log10(gc_obs)
    log_gc_m3 = np.log10(np.maximum(gc_m3, 1e-30))
    log_gc_m6 = np.log10(np.maximum(gc_m6, 1e-30))
    log_gc_comb = np.log10(np.maximum(gc_comb, 1e-30))
    rho_obs, _ = partial_corr(log_gc_obs, log_hR, log_vf)
    rho_m3p, _ = partial_corr(log_gc_m3, log_hR, log_vf)
    rho_m6p, _ = partial_corr(log_gc_m6, log_hR, log_vf)
    rho_combp, _ = partial_corr(log_gc_comb, log_hR, log_vf)
    print(f"  rho(gc, hR | vflat):")
    print(f"    gc_obs:   {rho_obs:+.4f}")
    print(f"    gc_M3:    {rho_m3p:+.4f} ({(1-abs(rho_m3p)/abs(rho_obs))*100:+.1f}%)")
    print(f"    gc_M6:    {rho_m6p:+.4f} ({(1-abs(rho_m6p)/abs(rho_obs))*100:+.1f}%)")
    print(f"    gc_M3+M6: {rho_combp:+.4f} ({(1-abs(rho_combp)/abs(rho_obs))*100:+.1f}%)")
    rho_obs2, _ = partial_corr_multi(log_gc_obs, log_hR, [log_vf, rmax_hR])
    rho_comb2, _ = partial_corr_multi(log_gc_comb, log_hR, [log_vf, rmax_hR])
    print(f"\n  rho(gc, hR | vflat, rmax/hR):")
    print(f"    gc_obs:   {rho_obs2:+.4f}")
    print(f"    gc_M3+M6: {rho_comb2:+.4f}")

    # T5
    print("\n" + "=" * 70)
    print("T5: indicator correlation")
    print("=" * 70)
    rho_ind, p_ind = spearmanr(log_em, log_sk)
    print(f"  rho(log_eps_max, log_strain_kin) = {rho_ind:+.4f} (p={p_ind:.2e})")
    m3_ind = np.maximum(log_em - params_M3[0], 0)
    rho_mi, p_mi = spearmanr(m3_ind, log_sk)
    print(f"  rho(M3_ind, log_sk) = {rho_mi:+.4f} (p={p_mi:.2e})")
    rho_ind_vf, p_ind_vf = partial_corr(log_em, log_sk, log_vf)
    print(f"  rho(log_em, log_sk | vflat) = {rho_ind_vf:+.4f} (p={p_ind_vf:.2e})")

    # T6
    print("\n" + "=" * 70)
    print("T6: M3/M6/M8 triangle")
    print("=" * 70)
    rho_36, _ = spearmanr(log_em, log_sk)
    rho_38, _ = spearmanr(log_em, log_curv)
    rho_68, _ = spearmanr(log_sk, log_curv)
    print(f"  rho(M3,M6)={rho_36:+.3f} rho(M3,M8)={rho_38:+.3f} rho(M6,M8)={rho_68:+.3f}")

    # T7
    print("\n" + "=" * 70)
    print("T7: shuffle independence")
    print("=" * 70)
    np.random.seed(42)
    R2_sh = []
    for _ in range(1000):
        sk_s = np.random.permutation(log_sk)
        sol_s, _, _, _ = lstsq(sk_s.reshape(-1,1), resid_m3, rcond=None)
        pred_s = sol_s[0]*sk_s
        R2_sh.append(1 - np.sum((resid_m3 - pred_s)**2)/np.sum(resid_m3**2))
    R2_sh = np.array(R2_sh)
    p_sh = np.mean(R2_sh >= R2_m6on3)
    print(f"  Real R2={R2_m6on3:.4f}, shuffle mean={np.mean(R2_sh):.4f}, p={p_sh:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
  T1 resid rho:       {rho_resid:+.4f}
  T2 dAICc vs M3/M6:  {dA_vs_m3:+.1f} / {dA_vs_m6:+.1f}
  T2b LOO ratio:      M3={mse_m3/mse_null:.3f} M6={mse_m6/mse_null:.3f} Comb={mse_comb/mse_null:.3f}
  T3 dR2:             M6->M3res={R2_m6on3:.4f}, M3->M6res={R2_m3on6:.4f}
  T4 hR partial:      obs={rho_obs:+.4f} Comb={rho_combp:+.4f}
  T5 indicator rho:   {rho_ind:+.4f}
  T6 M6-M8:           {rho_68:+.3f}
  T7 shuffle p:       {p_sh:.4f}
""")
    print("Done.")

if __name__ == '__main__':
    main()
