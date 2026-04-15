"""
z_dlt_r10 の 2 点相関関数を計算して相関長 ξ を推定するスクリプト
手元の sss2_zeff_proxy.csv に対して実行する
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

filename = 'sss2_zeff_proxy.csv'   # ← extract_zeff_v3.py が出力したファイル

print("データ読み込み中...")
df = pd.read_csv(filename).dropna(subset=['z_dlt_r10','ra','dec'])
print(f"有効行数: {len(df)}")

# ── 単位変換：deg → Mpc（z=0.5, H0=70 として）─────────────
# 1 arcmin ≈ 0.173 Mpc @z=0.5（平坦 ΛCDM, Ω_m=0.3）
deg2mpc = 60 * 0.173   # 1deg ≈ 10.38 Mpc

# ── フィールドごとに分割（大きすぎるので field 単位で処理）──
# まず field=2（最大, 198032行）で計算
field_id = 2
sub = df[df['field'] == field_id].copy()
print(f"\nfield={field_id}: {len(sub)} 行で計算")

# ── ランダムサブサンプリング（速度のため 20,000 点）────────
np.random.seed(42)
n_sample = 20000
if len(sub) > n_sample:
    idx = np.random.choice(len(sub), n_sample, replace=False)
    sub = sub.iloc[idx].reset_index(drop=True)
    print(f"サブサンプル: {n_sample} 点")

x = sub['ra'].values  * deg2mpc
y = sub['dec'].values * deg2mpc
v = sub['z_dlt_r10'].values

# ── 2 点相関関数 C(r) = <v(0) v(r)> / <v²> ─────────────────
print("2 点相関関数を計算中...")

# ペア距離のビン
r_min, r_max = 0.5, 60   # Mpc
n_bins = 25
bins = np.linspace(r_min, r_max, n_bins + 1)
r_mid = 0.5 * (bins[:-1] + bins[1:])

# ランダムペアサンプリング（全ペアは計算不可）
n_pairs = 500000
i1 = np.random.randint(0, len(sub), n_pairs)
i2 = np.random.randint(0, len(sub), n_pairs)
same = i1 == i2
i1, i2 = i1[~same], i2[~same]

dx = x[i1] - x[i2]
dy = y[i1] - y[i2]
r  = np.sqrt(dx**2 + dy**2)
vv = v[i1] * v[i2]
var_v = np.var(v)

# ビン平均
C_r = np.zeros(n_bins)
N_r = np.zeros(n_bins)
for b in range(n_bins):
    mask = (r >= bins[b]) & (r < bins[b+1])
    N_r[b] = mask.sum()
    if N_r[b] > 10:
        C_r[b] = np.mean(vv[mask]) / var_v

print()
print("=== 2 点相関関数 C(r) ===")
print(f"{'r [Mpc]':10s}  {'C(r)':8s}  {'N_pairs':8s}")
print("-"*32)
for i in range(n_bins):
    if N_r[i] > 10:
        print(f"{r_mid[i]:10.2f}  {C_r[i]:8.4f}  {int(N_r[i]):8d}")

# ── 指数フィットで相関長 ξ を推定 ───────────────────────────
valid = (N_r > 50) & (C_r > 0)
if valid.sum() >= 3:
    def exp_fit(r, xi, A):
        return A * np.exp(-r / xi)
    try:
        popt, pcov = curve_fit(exp_fit, r_mid[valid], C_r[valid],
                               p0=[10.0, 1.0], bounds=([0.1,0],[200,10]))
        xi_fit, A_fit = popt
        xi_err = np.sqrt(pcov[0,0])
        print()
        print(f"=== 指数フィット C(r) = A × exp(-r/ξ) ===")
        print(f"ξ = {xi_fit:.2f} ± {xi_err:.2f} Mpc")
        print(f"A = {A_fit:.4f}")
        print()
        print(f"T-7 との比較：")
        print(f"  HSC proxy の相関長 ξ_HSC = {xi_fit:.1f} Mpc")
        print(f"  SPARC の ξ ≈ r_s/3（典型 r_s≈5 kpc → ξ≈1.7 kpc）")
        print(f"  スケール比 = {xi_fit*1e3:.0f}（HSC: Mpc, SPARC: kpc）")
        print(f"  → HSC は大スケール構造の n_fold 相関長を測定")
    except Exception as e:
        print(f"フィット失敗: {e}")
else:
    print("有効ビン不足（正の C(r) が少ない）")

# ── 全 field の結果もまとめて出力 ───────────────────────────
print()
print("=== field 別の z_dlt_r10 統計 ===")
for fid in sorted(df['field'].unique()):
    s = df.loc[df['field']==fid, 'z_dlt_r10']
    print(f"field={fid}: n={len(s)}, mean={s.mean():.5f}, "
          f"std={s.std():.5f}, max={s.max():.5f}")