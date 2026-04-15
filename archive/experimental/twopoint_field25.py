"""
field=2〜5 の 2 点相関関数（field=1 の縦縞系統を除外）
各 field 別 + 合算の C(r) を計算し、相関長 ξ を推定する
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

filename = 'sss2_zeff_proxy.csv'

print("データ読み込み中...")
df = pd.read_csv(filename).dropna(subset=['z_dlt_r10','ra','dec'])
print(f"全行数: {len(df)}")

# field=1 を除外
df25 = df[df['field'].isin([2,3,4,5])].copy().reset_index(drop=True)
df1  = df[df['field'] == 1].copy().reset_index(drop=True)
print(f"field=2〜5: {len(df25)} 行")
print(f"field=1   : {len(df1)} 行（除外）")

deg2mpc = 60 * 0.173   # 1 deg ≈ 10.38 Mpc @z=0.5

def two_point_corr(sub, n_pairs=500000, r_min=0.5, r_max=60, n_bins=25,
                   label=''):
    """2 点相関関数 C(r) = <v(0)v(r)> / <v²> を計算"""
    np.random.seed(42)
    n_sample = min(20000, len(sub))
    if len(sub) > n_sample:
        idx = np.random.choice(len(sub), n_sample, replace=False)
        sub = sub.iloc[idx].reset_index(drop=True)

    x = sub['ra'].values  * deg2mpc
    y = sub['dec'].values * deg2mpc
    v = sub['z_dlt_r10'].values
    var_v = np.var(v)
    if var_v < 1e-12:
        return None, None, None

    bins  = np.linspace(r_min, r_max, n_bins + 1)
    r_mid = 0.5 * (bins[:-1] + bins[1:])

    i1 = np.random.randint(0, len(sub), n_pairs)
    i2 = np.random.randint(0, len(sub), n_pairs)
    same = (i1 == i2)
    i1, i2 = i1[~same], i2[~same]

    dx = x[i1] - x[i2]
    dy = y[i1] - y[i2]
    r  = np.sqrt(dx**2 + dy**2)
    vv = v[i1] * v[i2]

    C_r = np.zeros(n_bins)
    N_r = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (r >= bins[b]) & (r < bins[b+1])
        N_r[b] = mask.sum()
        if N_r[b] > 10:
            C_r[b] = np.mean(vv[mask]) / var_v

    return r_mid, C_r, N_r

def fit_twocomp(r_mid, C_r, N_r):
    """2 成分指数フィット"""
    valid = (N_r > 50) & np.isfinite(C_r)
    results = {}

    # 単純指数（r<25 Mpc のみ）
    v1 = valid & (r_mid < 25)
    if v1.sum() >= 3:
        try:
            popt, _ = curve_fit(lambda r,xi,A: A*np.exp(-r/xi),
                                r_mid[v1], C_r[v1], p0=[5,0.9],
                                bounds=([0.5,0],[50,5]))
            results['xi1'] = popt[0]; results['A1'] = popt[1]
        except: pass

    # 指数 + floor
    if valid.sum() >= 4:
        try:
            popt2, _ = curve_fit(lambda r,xi,A,C0: A*np.exp(-r/xi)+C0,
                                 r_mid[valid], C_r[valid],
                                 p0=[5,0.8,0.04],
                                 bounds=([0.5,0,0],[50,2,0.2]))
            results['xi_floor'] = popt2[0]
            results['A_floor']  = popt2[1]
            results['C0']       = popt2[2]
        except: pass

    # 2 成分
    if valid.sum() >= 5:
        try:
            popt3, _ = curve_fit(lambda r,x1,A1,x2,A2:
                                 A1*np.exp(-r/x1)+A2*np.exp(-r/x2),
                                 r_mid[valid], C_r[valid],
                                 p0=[4,0.7,30,0.08],
                                 bounds=([0.5,0,5,0],[20,2,200,1]))
            results['xi_s'] = popt3[0]; results['A_s'] = popt3[1]
            results['xi_l'] = popt3[2]; results['A_l'] = popt3[3]
        except: pass

    return results

# ── 計算 ─────────────────────────────────────────────────
print("\n=== 統計サマリー（コピペしてください）===\n")

# field=2〜5 合算
print("【field=2〜5 合算の C(r)】")
r_mid, C25, N25 = two_point_corr(df25, label='field2-5')
print(f"{'r [Mpc]':10s}  {'C(r)':8s}  {'N_pairs':8s}")
print("-"*32)
for i in range(len(r_mid)):
    if N25[i] > 10:
        print(f"{r_mid[i]:10.2f}  {C25[i]:8.4f}  {int(N25[i]):8d}")

fit25 = fit_twocomp(r_mid, C25, N25)
print()
print("【field=2〜5 フィット結果】")
if 'xi1' in fit25:
    print(f"  単純指数（r<25 Mpc）: ξ={fit25['xi1']:.2f} Mpc, A={fit25['A1']:.4f}")
if 'xi_floor' in fit25:
    print(f"  指数+floor:           ξ={fit25['xi_floor']:.2f} Mpc, "
          f"A={fit25['A_floor']:.4f}, C0={fit25['C0']:.4f}")
if 'xi_s' in fit25:
    print(f"  2成分: ξ_s={fit25['xi_s']:.2f} Mpc (A={fit25['A_s']:.4f}), "
          f"ξ_l={fit25['xi_l']:.2f} Mpc (A={fit25['A_l']:.4f})")

# field=1（縦縞あり）
print()
print("【field=1 の C(r)（縦縞系統の確認用）】")
r_mid1, C1, N1 = two_point_corr(df1, label='field1')
if r_mid1 is not None:
    for i in range(len(r_mid1)):
        if N1[i] > 10:
            print(f"{r_mid1[i]:10.2f}  {C1[i]:8.4f}  {int(N1[i]):8d}")
    fit1 = fit_twocomp(r_mid1, C1, N1)
    if 'xi1' in fit1:
        print(f"  単純指数（r<25 Mpc）: ξ={fit1['xi1']:.2f} Mpc")

# field 別
print()
print("【field 別 C(r) と ξ】")
print(f"{'field':6s}  {'n_pix':7s}  {'ξ_1(Mpc)':10s}  {'ξ_floor(Mpc)':13s}  {'C0':6s}")
print("-"*52)

field_results = {}
for fid in [1,2,3,4,5]:
    sub = df[df['field'] == fid]
    if len(sub) < 100: continue
    rm, Cf, Nf = two_point_corr(sub, label=f'field{fid}')
    if rm is None: continue
    ff = fit_twocomp(rm, Cf, Nf)
    xi1     = f"{ff['xi1']:.2f}"     if 'xi1'     in ff else '—'
    xi_fl   = f"{ff['xi_floor']:.2f}" if 'xi_floor' in ff else '—'
    c0_str  = f"{ff['C0']:.4f}"      if 'C0'      in ff else '—'
    print(f"  {fid:4d}  {len(sub):7d}  {xi1:10s}  {xi_fl:13s}  {c0_str:6s}")
    field_results[fid] = ff

# field=1 vs 2〜5 の比較
print()
print("【field=1 vs 2〜5 の比較（縦縞系統の定量評価）】")
xi_1 = field_results.get(1,{}).get('xi1', None)
xi_25 = fit25.get('xi1', None)
if xi_1 and xi_25:
    print(f"  field=1  ξ = {xi_1:.2f} Mpc")
    print(f"  field=2〜5 ξ = {xi_25:.2f} Mpc")
    print(f"  差 Δξ = {xi_1-xi_25:.2f} Mpc  "
          f"{'（系統あり ✗）' if abs(xi_1-xi_25)>1 else '（整合 ✓）'}")

print()
print("【全体 vs field=2〜5 の比較（field=1 の影響評価）】")
r_mid_all, C_all, N_all = two_point_corr(df, label='all')
fit_all = fit_twocomp(r_mid_all, C_all, N_all)
xi_all  = fit_all.get('xi1', None)
xi_25v  = fit25.get('xi1', None)
if xi_all and xi_25v:
    print(f"  全体    ξ = {xi_all:.2f} Mpc")
    print(f"  field=2〜5 ξ = {xi_25v:.2f} Mpc")
    print(f"  差 Δξ = {xi_all-xi_25v:.2f} Mpc  "
          f"{'（field=1 の影響大）' if abs(xi_all-xi_25v)>0.5 else '（field=1 の影響小）'}")