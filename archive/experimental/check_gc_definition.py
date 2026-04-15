"""
sparc_gc.csvのgc_ratioがどのように計算されたかを確認する。

定義A：gc_ratio = v_flat^2 / (r_s * a0)  <- トートロジー
定義B：gc_ratio = 独立なフィットパラメータ  <- 意味がある
"""
import numpy as np
import csv, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

a0    = 1.2e-10   # m/s^2
kpc2m = 3.086e19
kms2ms= 1e3

base_dir = os.path.dirname(os.path.abspath(__file__))
gc_csv   = os.path.join(base_dir, 'phase1', 'sparc_gc.csv')

if len(sys.argv) >= 2:
    gc_csv = sys.argv[1]

if not os.path.exists(gc_csv):
    print(f"[ERROR] {gc_csv} が見つかりません")
    sys.exit(1)

# -----------------------------------------------
# Step 1：列名の確認
# -----------------------------------------------
with open(gc_csv, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

print(f"列名：{headers}")
print(f"行数：{len(rows)}")
print(f"\n先頭3行：")
for row in rows[:3]:
    print(dict(row))

# -----------------------------------------------
# Step 2：gc_ratio = v_flat^2 / (r_s * a0) か確認
# -----------------------------------------------
print(f"\n{'='*65}")
print("  定義A検定：gc_ratio * r_s * a0 / v_flat^2 = 1.000 か？")
print(f"{'='*65}")

ratios = []
mismatches = []

for row in rows:
    try:
        gc  = float(row.get('gc_ratio','') or 'nan')
        vf  = float(row.get('vflat','')    or 'nan')
        rs  = float(row.get('rs','')       or 'nan')
        gal = row.get('galaxy','?')

        if not all(np.isfinite([gc, vf, rs])):
            continue
        if gc <= 0 or vf <= 0 or rs <= 0:
            continue

        # 定義Aなら gc = vf^2/rs/a0 -> gc * rs * a0 / vf^2 = 1
        gc_SI   = gc * a0
        vf_SI   = vf * kms2ms
        rs_SI   = rs * kpc2m
        ratio   = gc_SI * rs_SI / vf_SI**2

        ratios.append(ratio)
        if abs(ratio - 1.0) > 0.001:
            mismatches.append((gal, ratio, gc, vf, rs))

    except Exception as e:
        pass

ratios = np.array(ratios)
print(f"\nN = {len(ratios)}")
print(f"ratio = gc*r_s*a0/v_flat^2：")
print(f"  中央値 = {np.median(ratios):.6f}")
print(f"  平均   = {np.mean(ratios):.6f}")
print(f"  std    = {np.std(ratios):.8f}")
print(f"  min    = {ratios.min():.6f}")
print(f"  max    = {ratios.max():.6f}")

# -----------------------------------------------
# Step 3：判定
# -----------------------------------------------
print(f"\n{'='*65}")
print("  判定")
print(f"{'='*65}")

if np.std(ratios) < 1e-6:
    print("★ 定義A：gc_ratio = v_flat^2 / (r_s * a0)")
    print("  -> gc_ratio は v_flat と r_s から計算されたトートロジー")
    print("  -> g_c ∝ v_flat^1.32 は r_s ∝ v_flat^0.68 の言い換え")
    print("  -> C-2は深刻：独立な発見ではない")

elif np.median(ratios) < 0.01 or np.median(ratios) > 100:
    print("★ 定義B：gc_ratio は独立なフィットパラメータ")
    print("  -> g_c ∝ v_flat^1.32 は独立した発見")

elif abs(np.median(ratios) - 1.0) < 0.05:
    print("★ 定義A（ほぼ確実）：")
    print(f"  中央値={np.median(ratios):.4f} ≈ 1.000")
    print(f"  std={np.std(ratios):.6f}")
    print("  -> わずかなばらつきは数値誤差")
    print("  -> C-2は深刻")

else:
    print(f"★ 定義Bの可能性：中央値={np.median(ratios):.4f} != 1.000")
    print(f"  std={np.std(ratios):.4f}")
    print("  -> gc_ratio は独立に決定されている")

# -----------------------------------------------
# Step 4：スクリプト生成元の特定
# -----------------------------------------------
print(f"\n{'='*65}")
print("  追加確認：gc_ratioの計算方法")
print(f"{'='*65}")

# gc_ratio が何と相関するか確認
gc_list, vf_list, rs_list = [], [], []
for r in rows:
    try:
        gc = float(r.get('gc_ratio','nan'))
        vf = float(r.get('vflat','nan'))
        rs = float(r.get('rs','nan'))
        if all(np.isfinite([gc,vf,rs])) and gc>0 and vf>0 and rs>0:
            gc_list.append(gc)
            vf_list.append(vf)
            rs_list.append(rs)
    except:
        pass

gc_v = np.array(gc_list)
vf_v = np.array(vf_list)
rs_v = np.array(rs_list)

from scipy import stats
r_gc_vf, p_gc_vf = stats.pearsonr(np.log10(vf_v), np.log10(gc_v))
r_gc_rs, p_gc_rs = stats.pearsonr(np.log10(rs_v), np.log10(gc_v))

s_vf, _, _, _, _ = stats.linregress(np.log10(vf_v), np.log10(gc_v))
s_rs, _, _, _, _ = stats.linregress(np.log10(rs_v), np.log10(gc_v))

print(f"log(gc) vs log(v_flat)：r={r_gc_vf:.4f}, slope={s_vf:.4f}")
print(f"log(gc) vs log(r_s)   ：r={r_gc_rs:.4f}, slope={s_rs:.4f}")
print(f"\nもし定義Aなら：")
print(f"  gc = v^2/r -> log(gc) = 2*log(v) - log(r)")
print(f"  -> slope_vf=+2 slope_rs=-1 が期待値")
print(f"実測：slope_vf={s_vf:.3f}  slope_rs={s_rs:.3f}")

if abs(s_vf - 2.0) < 0.1 and abs(s_rs + 1.0) < 0.1:
    print(f"\n-> 定義A確定（slope_vf≈2, slope_rs≈-1）")
elif abs(s_rs + 1.0) > 0.3:
    print(f"\n-> 定義B可能性あり（slope_rs!=-1）")
else:
    print(f"\n-> 判定保留")

# -----------------------------------------------
# Step 5：gc_siとの比較
# -----------------------------------------------
print(f"\n{'='*65}")
print("  gc_si列との関係確認")
print(f"{'='*65}")

gc_si_list = []
gc_ratio_list = []
for r in rows:
    try:
        gc_si = float(r.get('gc_si','nan'))
        gc_r  = float(r.get('gc_ratio','nan'))
        if np.isfinite(gc_si) and np.isfinite(gc_r) and gc_si>0 and gc_r>0:
            gc_si_list.append(gc_si)
            gc_ratio_list.append(gc_r)
    except:
        pass

gc_si_a = np.array(gc_si_list)
gc_r_a  = np.array(gc_ratio_list)

if len(gc_si_a) > 0:
    ratio_si = gc_si_a / (gc_r_a * a0)
    print(f"gc_si / (gc_ratio * a0)：")
    print(f"  中央値={np.median(ratio_si):.6f}")
    print(f"  std   ={np.std(ratio_si):.8f}")
    if np.std(ratio_si) < 1e-6:
        print("  -> gc_si = gc_ratio * a0 確認済み（gc_ratioはgc/a0の無次元量）")
