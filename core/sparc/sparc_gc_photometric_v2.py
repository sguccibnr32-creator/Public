#!/usr/bin/env python3
"""
sparc_gc_photometric_v2.py
Approach C revised: use SPARC_Lelli2016c.mrt photometric data directly.

Key insight: SPARC provides SBdisk [solLum/pc^2] = disk central surface brightness
  -> Sigma_star = Yd * SBdisk [solMass/pc^2]
  -> G * Sigma_star = G * Yd * SBdisk
  gc = eta * (a0 * G * Sigma_bar)^alpha

This avoids using Vdisk outer values (which failed in v1).
"""
import os, csv, warnings
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
MRT = os.path.join(BASE, "SPARC_Lelli2016c.mrt")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
a0 = 1.2e-10
G_SI = 6.674e-11
kpc_m = 3.0857e19
pc_m = 3.0857e16
Msun = 1.989e30

def parse_mrt():
    """Parse SPARC_Lelli2016c.mrt using whitespace splitting."""
    data = {}
    in_data = False
    sep_count = 0
    with open(MRT, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep_count += 1
                if sep_count >= 4:
                    in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                # 0=Galaxy 1=T 2=D 3=eD 4=fD 5=Inc 6=eInc
                # 7=L36 8=eL36 9=Reff 10=SBeff 11=Rdisk 12=SBdisk
                # 13=MHI 14=RHI 15=Vflat 16=eVflat 17=Q 18+=Ref
                name = parts[0]
                T = int(parts[1])
                D = float(parts[2])
                Inc = float(parts[5])
                L36 = float(parts[7])       # 10^9 solLum
                Reff = float(parts[9])       # kpc
                SBeff = float(parts[10])     # solLum/pc^2
                Rdisk = float(parts[11])     # kpc = hR
                SBdisk = float(parts[12])    # solLum/pc^2
                MHI = float(parts[13])       # 10^9 solMass
                RHI = float(parts[14])       # kpc
                Vflat = float(parts[15])     # km/s
                e_Vflat = float(parts[16])   # km/s
                Q = int(parts[17])
                data[name] = {
                    'T': T, 'D': D, 'Inc': Inc, 'L36': L36,
                    'Reff': Reff, 'SBeff': SBeff,
                    'Rdisk': Rdisk, 'SBdisk': SBdisk,
                    'MHI': MHI, 'RHI': RHI,
                    'Vflat': Vflat, 'e_Vflat': e_Vflat, 'Q': Q,
                }
            except (ValueError, IndexError):
                continue
    return data

def load_gc():
    """Load gc from TA3."""
    gc = {}
    with open(TA3, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                if gc_a0 > 0:
                    gc[name] = gc_a0 * a0
            except: pass
    return gc

def load_Yd():
    """Load Yd from phase1."""
    yd = {}
    with open(PHASE1, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                yd[name] = float(row.get('ud', '0.5'))
            except: pass
    return yd

def fit_alpha(log_gc, log_gsigma, n_boot=1000):
    """gc = eta * (a0 * G*Sigma)^alpha"""
    def resid(p, x, y):
        return p[0] + p[1]*x - y
    log_x = np.log10(a0) + log_gsigma
    res = optimize.least_squares(resid, [0, 0.5], args=(log_x, log_gc),
                                  bounds=([-np.inf, 0.01], [np.inf, 2.0]))
    alpha = res.x[1]; eta = 10**res.x[0]
    pred = res.x[0] + res.x[1]*log_x
    R2 = 1 - np.sum((log_gc - pred)**2)/np.sum((log_gc - log_gc.mean())**2)
    np.random.seed(42)
    boots = []
    ng = len(log_gc)
    for _ in range(n_boot):
        idx = np.random.choice(ng, ng, replace=True)
        try:
            rb = optimize.least_squares(resid, res.x, args=(log_x[idx], log_gc[idx]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            boots.append(rb.x[1])
        except: pass
    err = np.std(boots) if boots else np.nan
    z = (alpha - 0.5)/err if err > 0 else np.inf
    p05 = 2 * stats.norm.sf(abs(z))
    return alpha, err, eta, R2, p05, pred

def main():
    print("=" * 70)
    print("Approach C v2: Photometric SBdisk from SPARC_Lelli2016c.mrt")
    print("=" * 70)

    mrt = parse_mrt()
    gc_data = load_gc()
    yd_data = load_Yd()
    print(f"MRT galaxies: {len(mrt)}")
    print(f"TA3 gc values: {len(gc_data)}")
    print(f"Phase1 Yd values: {len(yd_data)}")

    # Build dataset
    records = []
    for name, m in mrt.items():
        if name not in gc_data: continue
        gc = gc_data[name]
        Yd = yd_data.get(name, 0.5)
        vflat = m['Vflat']
        hR = m['Rdisk']       # kpc - TRUE photometric hR
        SBdisk = m['SBdisk']  # solLum/pc^2
        L36 = m['L36']        # 10^9 solLum
        MHI = m['MHI']        # 10^9 solMass
        RHI = m['RHI']        # kpc
        Q = m['Q']
        if gc <= 0 or vflat <= 0 or hR <= 0 or SBdisk <= 0:
            continue

        # G * Sigma_star_0 [m/s^2]
        # SBdisk [solLum/pc^2] -> Sigma_star = Yd * SBdisk [solMass/pc^2]
        # -> G * Sigma_star [m/s^2] = G * Yd * SBdisk * Msun / pc^2
        G_Sigma_star = G_SI * Yd * SBdisk * Msun / (pc_m**2)

        # Gas surface density: Sigma_gas = MHI * 1.33 / (pi * RHI^2)
        # (1.33 for helium correction)
        if MHI > 0 and RHI > 0:
            Sigma_gas_Msun_pc2 = (MHI * 1.33e9) / (np.pi * (RHI * 1e3)**2)  # Msun/pc^2
            G_Sigma_gas = G_SI * Sigma_gas_Msun_pc2 * Msun / (pc_m**2)
        else:
            Sigma_gas_Msun_pc2 = 0
            G_Sigma_gas = 0

        G_Sigma_bar = G_Sigma_star + G_Sigma_gas
        G_Sigma_obs = (vflat * 1e3)**2 / (hR * kpc_m)  # Vobs-based proxy

        records.append({
            'name': name, 'gc': gc, 'vflat': vflat, 'hR': hR,
            'Yd': Yd, 'Q': Q,
            'SBdisk': SBdisk, 'L36': L36, 'MHI': MHI, 'RHI': RHI,
            'Sigma_gas': Sigma_gas_Msun_pc2,
            'G_Sigma_star': G_Sigma_star,
            'G_Sigma_gas': G_Sigma_gas,
            'G_Sigma_bar': G_Sigma_bar,
            'G_Sigma_obs': G_Sigma_obs,
        })

    N = len(records)
    print(f"\nValid galaxies: {N}")

    gc_arr = np.array([r['gc'] for r in records])
    log_gc = np.log10(gc_arr)
    vflat = np.array([r['vflat'] for r in records])
    hR = np.array([r['hR'] for r in records])
    Yd = np.array([r['Yd'] for r in records])
    SBdisk = np.array([r['SBdisk'] for r in records])
    G_Sigma_star = np.array([r['G_Sigma_star'] for r in records])
    G_Sigma_gas = np.array([r['G_Sigma_gas'] for r in records])
    G_Sigma_bar = np.array([r['G_Sigma_bar'] for r in records])
    G_Sigma_obs = np.array([r['G_Sigma_obs'] for r in records])

    # ================================================================
    # M1: G*Sigma_bar (star+gas, fixed Yd from SPS)
    # ================================================================
    print("\n" + "=" * 70)
    print("M1: G*Sigma_bar (Yd*SBdisk + gas, Yd=SPS fixed)")
    print("=" * 70)
    v1 = G_Sigma_bar > 0
    log_gs1 = np.log10(G_Sigma_bar[v1])
    a1, e1, eta1, r2_1, p1, pred1 = fit_alpha(log_gc[v1], log_gs1)
    print(f"  alpha={a1:.3f}+/-{e1:.3f}, eta={eta1:.3f}, R2={r2_1:.3f}, p(0.5)={p1:.4f}")

    # ================================================================
    # M2: G*Sigma_star only (no gas)
    # ================================================================
    print("\n" + "=" * 70)
    print("M2: G*Sigma_star only (Yd*SBdisk, no gas)")
    print("=" * 70)
    v2 = G_Sigma_star > 0
    log_gs2 = np.log10(G_Sigma_star[v2])
    a2, e2, eta2, r2_2, p2, pred2 = fit_alpha(log_gc[v2], log_gs2)
    print(f"  alpha={a2:.3f}+/-{e2:.3f}, eta={eta2:.3f}, R2={r2_2:.3f}, p(0.5)={p2:.4f}")

    # ================================================================
    # M3: G*Sigma_obs (vflat^2/hR, reference)
    # ================================================================
    print("\n" + "=" * 70)
    print("M3: G*Sigma_obs = vflat^2/hR (reference)")
    print("=" * 70)
    v3 = G_Sigma_obs > 0
    log_gs3 = np.log10(G_Sigma_obs[v3])
    a3, e3, eta3, r2_3, p3, pred3 = fit_alpha(log_gc[v3], log_gs3)
    print(f"  alpha={a3:.3f}+/-{e3:.3f}, eta={eta3:.3f}, R2={r2_3:.3f}, p(0.5)={p3:.4f}")

    # ================================================================
    # M4: Yd free + alpha simultaneous (photometric)
    # ================================================================
    print("\n" + "=" * 70)
    print("M4: free Yd + alpha simultaneous (photometric)")
    print("=" * 70)
    # gc = eta * (a0 * G * (Yd * SBdisk + gas) * Msun/pc^2)^alpha
    # log_gc = log_eta + alpha * log(a0 * G * (Yd_free * SB + gas_density) * Msun/pc^2)
    SB_raw = SBdisk  # solLum/pc^2
    Sigma_gas = np.array([r['Sigma_gas'] for r in records])  # Msun/pc^2

    def model_4(p, SB, Sg):
        log_eta, alpha, log_Yd = p
        Yd_f = 10**log_Yd
        Sigma_bar = Yd_f * SB + Sg  # Msun/pc^2
        Sigma_bar = np.maximum(Sigma_bar, 1e-10)
        G_Sigma = G_SI * Sigma_bar * Msun / (pc_m**2)
        return log_eta + alpha * np.log10(a0 * G_Sigma)

    def resid_4(p, SB, Sg, y):
        return model_4(p, SB, Sg) - y

    res4 = optimize.least_squares(resid_4, [0, 0.5, np.log10(0.5)],
                                    args=(SB_raw, Sigma_gas, log_gc),
                                    bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
    a4 = res4.x[1]; Yd4 = 10**res4.x[2]; eta4 = 10**res4.x[0]
    pred4 = model_4(res4.x, SB_raw, Sigma_gas)
    R2_4 = 1 - np.sum((log_gc - pred4)**2)/np.sum((log_gc - log_gc.mean())**2)

    np.random.seed(42)
    boots4 = {"alpha": [], "Yd": []}
    for _ in range(1000):
        idx = np.random.choice(N, N, replace=True)
        try:
            rb = optimize.least_squares(resid_4, res4.x,
                args=(SB_raw[idx], Sigma_gas[idx], log_gc[idx]),
                bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
            boots4["alpha"].append(rb.x[1]); boots4["Yd"].append(10**rb.x[2])
        except: pass
    e4 = np.std(boots4["alpha"]); Yd4_err = np.std(boots4["Yd"])
    z4 = (a4 - 0.5)/e4 if e4 > 0 else np.inf
    p4 = 2 * stats.norm.sf(abs(z4))
    print(f"  alpha={a4:.3f}+/-{e4:.3f}, Yd={Yd4:.3f}+/-{Yd4_err:.3f}")
    print(f"  eta={eta4:.3f}, R2={R2_4:.3f}, p(0.5)={p4:.4f}")

    # ================================================================
    # M5: SBdisk only (no Yd, absorbed into eta)
    # ================================================================
    print("\n" + "=" * 70)
    print("M5: SBdisk only (Yd absorbed, Sigma_L as proxy)")
    print("=" * 70)
    G_Sigma_L = G_SI * SBdisk * Msun / (pc_m**2)  # at Yd=1
    v5 = G_Sigma_L > 0
    log_gs5 = np.log10(G_Sigma_L[v5])
    a5, e5, eta5, r2_5, p5, pred5 = fit_alpha(log_gc[v5], log_gs5)
    Yd_implied = (eta5 / eta3)**(1/a5) if a5 > 0 else np.nan
    print(f"  alpha={a5:.3f}+/-{e5:.3f}, eta'={eta5:.3f}, R2={r2_5:.3f}, p(0.5)={p5:.4f}")
    print(f"  implied Yd ~ {Yd_implied:.3f} (from eta'/eta_obs)")

    # ================================================================
    # Independence check
    # ================================================================
    print("\n" + "=" * 70)
    print("Independence: G*Sigma_star vs G*Sigma_obs")
    print("=" * 70)
    rho_io, p_io = stats.pearsonr(np.log10(G_Sigma_star[v1]),
                                   np.log10(G_Sigma_obs[v1]))
    print(f"  r(G*Sigma_star, G*Sigma_obs) = {rho_io:.3f} (p={p_io:.2e})")
    rho_Lo, p_Lo = stats.pearsonr(np.log10(SBdisk), np.log10(G_Sigma_obs))
    print(f"  r(SBdisk, vflat^2/hR) = {rho_Lo:.3f}")

    # ================================================================
    # 50/50 generalization
    # ================================================================
    print("\n" + "=" * 70)
    print("50/50 split generalization (20x)")
    print("=" * 70)
    gen = {"M1_bar": [], "M2_star": [], "M3_obs": [], "M4_free": [], "M5_SB": []}
    for seed in range(20):
        np.random.seed(seed*11 + 7)
        idx = np.arange(N); np.random.shuffle(idx)
        half = N // 2
        tr, te = idx[:half], idx[half:]
        def resid_s(p, x, y): return p[0] + p[1]*x - y
        # M1
        try:
            lx = np.log10(a0) + np.log10(G_Sigma_bar)
            rb = optimize.least_squares(resid_s, [0, 0.5], args=(lx[tr], log_gc[tr]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            rb_te = optimize.least_squares(resid_s, rb.x, args=(lx[te], log_gc[te]),
                                            bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            gen["M1_bar"].append(rb_te.x[1])
        except: pass
        # M2
        try:
            lx = np.log10(a0) + np.log10(G_Sigma_star)
            rb = optimize.least_squares(resid_s, [0, 0.5], args=(lx[tr], log_gc[tr]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            rb_te = optimize.least_squares(resid_s, rb.x, args=(lx[te], log_gc[te]),
                                            bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            gen["M2_star"].append(rb_te.x[1])
        except: pass
        # M3
        try:
            lx = np.log10(a0) + np.log10(G_Sigma_obs)
            rb = optimize.least_squares(resid_s, [0, 0.5], args=(lx[tr], log_gc[tr]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            rb_te = optimize.least_squares(resid_s, rb.x, args=(lx[te], log_gc[te]),
                                            bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            gen["M3_obs"].append(rb_te.x[1])
        except: pass
        # M4
        try:
            rb = optimize.least_squares(resid_4, res4.x,
                args=(SB_raw[tr], Sigma_gas[tr], log_gc[tr]),
                bounds=([-5, 0.01, -1.5], [5, 2.0, 1.0]))
            def rfy(p, SB, Sg, y, lYd):
                Yd_f = 10**lYd; Sb = np.maximum(Yd_f*SB + Sg, 1e-10)
                return p[0] + p[1]*np.log10(a0*G_SI*Sb*Msun/(pc_m**2)) - y
            rb_te = optimize.least_squares(rfy, [rb.x[0], rb.x[1]],
                args=(SB_raw[te], Sigma_gas[te], log_gc[te], rb.x[2]),
                bounds=([-5, 0.01], [5, 2.0]))
            gen["M4_free"].append(rb_te.x[1])
        except: pass
        # M5
        try:
            lx = np.log10(a0) + np.log10(G_Sigma_L)
            rb = optimize.least_squares(resid_s, [0, 0.5], args=(lx[tr], log_gc[tr]),
                                         bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            rb_te = optimize.least_squares(resid_s, rb.x, args=(lx[te], log_gc[te]),
                                            bounds=([-np.inf, 0.01], [np.inf, 2.0]))
            gen["M5_SB"].append(rb_te.x[1])
        except: pass

    print(f"\n  {'model':<12s} {'a_test':>8s} {'std':>8s} {'covers 0.5':>12s}")
    for m in gen:
        if gen[m]:
            at = np.array(gen[m])
            cov = abs(at.mean()-0.5) < 2*at.std()
            print(f"  {m:<12s} {at.mean():>8.3f} {at.std():>8.3f} "
                  f"{'YES' if cov else 'NO':>12s}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  {'model':<30s} {'alpha':>7s} {'err':>7s} {'R2':>6s} {'p(0.5)':>8s}")
    print(f"  {'M1 Yd*SBdisk+gas (SPS Yd)':<30s} {a1:>7.3f} {e1:>7.3f} {r2_1:>6.3f} {p1:>8.4f}")
    print(f"  {'M2 Yd*SBdisk only':<30s} {a2:>7.3f} {e2:>7.3f} {r2_2:>6.3f} {p2:>8.4f}")
    print(f"  {'M3 vflat^2/hR (ref)':<30s} {a3:>7.3f} {e3:>7.3f} {r2_3:>6.3f} {p3:>8.4f}")
    print(f"  {'M4 free Yd+alpha':<30s} {a4:>7.3f} {e4:>7.3f} {R2_4:>6.3f} {p4:>8.4f}")
    print(f"  {'M5 SBdisk only (no Yd)':<30s} {a5:>7.3f} {e5:>7.3f} {r2_5:>6.3f} {p5:>8.4f}")
    print(f"\n  Key: M4 Yd_free = {Yd4:.3f}+/-{Yd4_err:.3f} (SPS typical ~0.5)")
    print(f"  Independence: r(Sigma_star, Sigma_obs) = {rho_io:.3f}")
    print(f"  SBdisk-only: implied Yd ~ {Yd_implied:.3f}")

    # Figure
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        lx = np.log10(a0 * G_Sigma_bar[v1])
        ax.scatter(lx, log_gc[v1], s=10, alpha=0.5, c='steelblue')
        xf = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xf, np.log10(eta1) + a1*xf, 'r-', lw=2,
                label=f'M1: a={a1:.3f}+/-{e1:.3f}')
        ax.set_xlabel('log(a0*G*Sigma_bar)'); ax.set_ylabel('log gc')
        ax.set_title(f'(a) M1: Photometric (R2={r2_1:.3f})')
        ax.legend()

        ax = axes[0, 1]
        ax.hist(boots4["alpha"], bins=40, alpha=0.5, color='steelblue', label='M4')
        ax.axvline(0.5, color='red', ls='--', label='0.5')
        ax.axvline(a3, color='blue', ls=':', label=f'M3 ref: {a3:.3f}')
        ax.set_xlabel('alpha'); ax.set_title('(b) M4 bootstrap')
        ax.legend()

        ax = axes[1, 0]
        ax.scatter(np.log10(G_Sigma_star[v1]), np.log10(G_Sigma_obs[v1]),
                   s=10, alpha=0.5, c='steelblue')
        lim = [min(np.log10(G_Sigma_star[v1]).min(), np.log10(G_Sigma_obs[v1]).min())-0.2,
               max(np.log10(G_Sigma_star[v1]).max(), np.log10(G_Sigma_obs[v1]).max())+0.2]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel('log G*Sigma_star'); ax.set_ylabel('log G*Sigma_obs')
        ax.set_title(f'(c) Independence r={rho_io:.3f}')

        ax = axes[1, 1]
        cols = ['steelblue','orange','gray','green','red']
        for i, (m, c) in enumerate(zip(gen, cols)):
            if gen[m]:
                at = np.array(gen[m])
                ax.scatter([i]*len(at), at, c=c, s=30, alpha=0.5)
                ax.errorbar(i, at.mean(), yerr=at.std(), color=c, capsize=5)
        ax.axhline(0.5, color='red', ls='--')
        ax.set_xticks(range(len(gen)))
        ax.set_xticklabels(list(gen.keys()), fontsize=7, rotation=20)
        ax.set_ylabel('alpha test')
        ax.set_title('(d) 50/50 generalization')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'gc_photometric_v2.png'), dpi=150)
        print("\nFigure saved.")
    except Exception as e:
        print(f"fig error: {e}")
    print("\nDone.")

if __name__ == '__main__':
    main()
