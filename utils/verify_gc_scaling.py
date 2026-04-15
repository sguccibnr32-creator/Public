import numpy as np
from scipy import stats
from scipy.optimize import brentq
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# v3.0フィット結果（5銀河）
galaxies = {
    'IC2574'  : {'vflat': 56,  'rs': 7.7,  'Upsilon': 1.90, 'gc_a0': 0.060},
    'NGC0300' : {'vflat': 77,  'rs': 5.5,  'Upsilon': 1.67, 'gc_a0': 0.206},
    'NGC3198' : {'vflat': 119, 'rs': 11.0, 'Upsilon': 0.96, 'gc_a0': 0.397},
    'UGC02885': {'vflat': 225, 'rs': 37.2, 'Upsilon': 1.01, 'gc_a0': 0.391},
    'NGC2841' : {'vflat': 256, 'rs': 3.3,  'Upsilon': 0.45, 'gc_a0': 2.970},
}

a0 = 1.2e-10      # m/s^2
kpc2m = 3.086e19  # m/kpc
kms2ms = 1e3      # m/s per km/s

names  = list(galaxies.keys())
vflat  = np.array([galaxies[g]['vflat'] for g in names])
rs     = np.array([galaxies[g]['rs']    for g in names])
gc_obs = np.array([galaxies[g]['gc_a0'] for g in names])

# -----------------------------------------------
# 検証1：g_c = v_flat²/r_s が成立するか
# -----------------------------------------------
gc_calc = (vflat*kms2ms)**2 / (rs*kpc2m) / a0  # g_c/a_0

print("=== 検証1：g_c = v_flat^2/r_s ===")
print(f"{'銀河':<12} {'gc_obs':>8} {'gc_calc':>8} {'比率':>8}")
print("-"*40)
for i, name in enumerate(names):
    ratio = gc_obs[i] / gc_calc[i]
    print(f"{name:<12} {gc_obs[i]:>8.3f} {gc_calc[i]:>8.3f} {ratio:>8.3f}")

# -----------------------------------------------
# 検証2：g_c ∝ v_flat^1.32
# -----------------------------------------------
log_vflat = np.log10(vflat)
log_gc    = np.log10(gc_obs)
slope, intercept, r, p, se = stats.linregress(log_vflat, log_gc)
print(f"\n=== 検証2：g_c ∝ v_flat^slope ===")
print(f"slope = {slope:.3f} +/- {se:.3f}  (期待値：1.32)")
print(f"r = {r:.3f},  p = {p:.4f}")

# -----------------------------------------------
# 検証3：黄金比条件 v^2_bar(r_s) ≈ 1.118 v^2_flat
# -----------------------------------------------
phi = (1 + 5**0.5) / 2
expected_ratio = phi - 0.5  # ≈ 1.118

print(f"\n=== 検証3：黄金比条件 ===")
print(f"期待値 v^2_bar(r_s)/v^2_flat = phi - 0.5 = {expected_ratio:.4f}")
print(f"phi = {phi:.4f}（黄金比）")
print(f"v_c(r_s)/v_flat = sqrt(phi) = {phi**0.5:.4f}")

# -----------------------------------------------
# 検証4：(2-4β)/(1-β) = 1.32 の解
# -----------------------------------------------
beta = brentq(lambda b: (2-4*b)/(1-b) - slope, 0.01, 0.49)
print(f"\n=== 検証4：必要なbeta ===")
print(f"g_c ∝ v_flat^{slope:.3f} を再現するbeta = {beta:.4f}")
print(f"r_d ∝ M_bar^{beta:.4f}")
print(f"M_bar ∝ v_flat^{2/(1-beta):.4f}  （有効BTFR指数）")

# 理論予言：各指数
print(f"\n=== スケーリング関係まとめ ===")
print(f"beta = {beta:.3f}")
print(f"M_bar ∝ v_flat^{2/(1-beta):.3f}")
print(f"r_d   ∝ v_flat^{2*beta/(1-beta):.3f}")
print(f"r_s   ∝ v_flat^{2*beta/(1-beta):.3f}  (r_s = kappa*r_d)")
print(f"g_c   ∝ v_flat^{(2-4*beta)/(1-beta):.3f}  (期待：1.32)")
