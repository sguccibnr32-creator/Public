# -*- coding: utf-8 -*-
"""
v2.0 tanh モデルフィッティング
================================
モデル:
  v²(r) = Υ_d × v²_disk + v²_gas + v²_bul + v²_flat × T(r, r_s)
  T(r, r_s) = 0.5 × [1 + tanh(w × (r/r_s - 1))]

フリーパラメータ: v_flat, r_s, Υ_d（ディスク M/L 比）
固定: w = 1.0

使用例:
  python -X utf8 fit_tanh.py
  python -X utf8 fit_tanh.py --galaxies NGC3198 NGC2403
  python -X utf8 fit_tanh.py --all
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import argparse
import json

# ============================================================
# 定数
# ============================================================
W_FIXED = 1.0

# ============================================================
# データ読み込み
# ============================================================
ROTMOD_COLS = ['Rad', 'Vobs', 'errV', 'Vgas',
               'Vdisk', 'Vbul', 'SBdisk', 'SBbul']

def load_galaxy(filepath):
    """rotmod.dat を読み込んで DataFrame を返す"""
    df = pd.read_csv(filepath, sep=r'\s+',
                     comment='#', names=ROTMOD_COLS)
    return df

# ============================================================
# モデル
# ============================================================
def v_model(r, vflat, rs, upsilon_d, vdisk, vgas, vbul,
            w=W_FIXED):
    """
    v2.0 tanh モデル

    Parameters
    ----------
    r       : array, 半径 [kpc]
    vflat   : float, 平坦速度 [km/s]
    rs      : float, 遷移半径 [kpc]
    upsilon_d : float, ディスク M/L 比スケーリング
                SPARC は Υ=0.5 (3.6μm) で計算済み
                → upsilon_d=1.0 でそのまま採用
    vdisk   : array, ディスク速度成分 [km/s]
    vgas    : array, ガス速度成分 [km/s]
    vbul    : array, バルジ速度成分 [km/s]
    w       : float, tanh 傾き（固定）

    Returns
    -------
    v_total : array, モデル速度 [km/s]
    """
    x = r / rs
    Fx = 0.5 * (1.0 + np.tanh(w * (x - 1.0)))
    v2 = (upsilon_d * vdisk**2
          + vgas**2
          + vbul**2
          + vflat**2 * Fx)
    return np.sqrt(np.clip(v2, 0, None))

# ============================================================
# フィッティング
# ============================================================
def fit_galaxy(df, w=W_FIXED,
               vflat_bounds=(10, 400),
               rs_bounds=(0.1, 200),
               upsilon_bounds=(0.1, 5.0)):
    """
    3パラメータフィット: v_flat, r_s, Υ_d

    Returns
    -------
    dict : フィット結果
    """
    r    = df['Rad'].values
    vobs = df['Vobs'].values
    verr = np.clip(df['errV'].values, 1.0, None)
    vd   = df['Vdisk'].values
    vg   = df['Vgas'].values
    vb   = df['Vbul'].values

    def chi2(params):
        vf, rs, ud = params
        if not (vflat_bounds[0] < vf < vflat_bounds[1]):
            return 1e9
        if not (rs_bounds[0] < rs < rs_bounds[1]):
            return 1e9
        if not (upsilon_bounds[0] < ud < upsilon_bounds[1]):
            return 1e9
        vm = v_model(r, vf, rs, ud, vd, vg, vb, w=w)
        return np.sum(((vobs - vm) / verr)**2)

    # グリッドサーチで初期値を探索
    best_c2 = 1e12
    best_p0 = [100, 5, 1.0]
    for vf0 in np.linspace(30, 350, 10):
        for rs0 in [0.5, 1, 2, 5, 10, 20, 40]:
            for ud0 in [0.3, 0.5, 1.0, 1.5, 2.0]:
                c = chi2([vf0, rs0, ud0])
                if c < best_c2:
                    best_c2 = c
                    best_p0 = [vf0, rs0, ud0]

    res = minimize(chi2, best_p0,
                   method='Nelder-Mead',
                   options={'xatol': 0.1,
                            'fatol': 0.05,
                            'maxiter': 10000})

    vf, rs, ud = res.x
    ud = np.clip(ud, upsilon_bounds[0], upsilon_bounds[1])
    rs = np.clip(rs, rs_bounds[0], rs_bounds[1])

    vm   = v_model(r, vf, rs, ud, vd, vg, vb, w=w)
    n    = len(r)
    dof  = max(n - 3, 1)
    chi2_total = res.fun
    chi2_dof   = chi2_total / dof

    # 残差
    residuals = (vobs - vm) / verr

    return {
        'vflat'      : round(float(vf), 2),
        'rs'         : round(float(rs), 3),
        'upsilon_d'  : round(float(ud), 3),
        'w'          : w,
        'chi2_total' : round(float(chi2_total), 3),
        'chi2_dof'   : round(float(chi2_dof), 4),
        'n_pts'      : n,
        'dof'        : dof,
        'rms_resid'  : round(float(np.std(residuals)), 3),
        'r'          : r.tolist(),
        'vobs'       : vobs.tolist(),
        'verr'       : df['errV'].values.tolist(),
        'vm'         : vm.tolist(),
        'vdisk'      : vd.tolist(),
        'vgas'       : vg.tolist(),
        'vbul'       : vb.tolist(),
    }

# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='v2.0 tanh model fitting')
    parser.add_argument('--galaxies', nargs='+',
                        default=None,
                        help='銀河名リスト')
    parser.add_argument('--all', action='store_true',
                        help='全銀河をフィット')
    parser.add_argument('--rotmod-dir',
                        default=None,
                        help='rotmod データディレクトリ')
    parser.add_argument('--w', type=float,
                        default=W_FIXED,
                        help=f'w パラメータ（default={W_FIXED}）')
    parser.add_argument('--output', default='fit_results.csv',
                        help='出力 CSV')
    args = parser.parse_args()

    # データディレクトリ解決
    if args.rotmod_dir:
        data_dir = Path(args.rotmod_dir)
    else:
        candidates = [
            Path('data/Rotmod_LTG'),
            Path('../Rotmod_LTG'),
            Path(r'D:\ドキュメント\エントロピー\新膜宇宙論'
                 r'\これまでの軌跡\パイソン\Rotmod_LTG'),
        ]
        data_dir = None
        for c in candidates:
            if c.exists():
                data_dir = c
                break
        if data_dir is None:
            print("ERROR: rotmod ディレクトリが見つかりません")
            print("--rotmod-dir で指定してください")
            return

    # 対象銀河の決定
    available = {
        f.stem.replace('_rotmod', ''): f
        for f in data_dir.glob('*_rotmod.dat')
    }

    if args.all:
        targets = sorted(available.keys())
    elif args.galaxies:
        targets = args.galaxies
    else:
        targets = ['NGC3198', 'NGC2403', 'DDO154',
                   'UGC02885', 'NGC6503']

    # フィット実行
    print(f"w = {args.w} (fixed)")
    print(f"対象: {len(targets)} 銀河")
    print()
    print(f"{'galaxy':15s} {'vflat':>7} {'rs':>7} "
          f"{'Υ_d':>5} {'chi2/dof':>9} "
          f"{'n':>4} {'rms':>6}")
    print("-" * 60)

    records = []
    results_detail = {}

    for name in targets:
        fp = available.get(name)
        if fp is None:
            print(f"  {name}: not found")
            continue

        df = load_galaxy(fp)
        if len(df) < 5:
            print(f"  {name}: too few points ({len(df)})")
            continue

        res = fit_galaxy(df, w=args.w)
        results_detail[name] = res

        # chi2 判定
        c = res['chi2_dof']
        if c < 2:
            grade = 'A'
        elif c < 5:
            grade = 'B'
        elif c < 15:
            grade = 'C'
        else:
            grade = 'D'

        print(f"{name:15s} {res['vflat']:7.1f} "
              f"{res['rs']:7.2f} "
              f"{res['upsilon_d']:5.2f} "
              f"{c:9.3f} "
              f"{res['n_pts']:4d} "
              f"{res['rms_resid']:6.2f}  {grade}")

        records.append({
            'galaxy'    : name,
            'vflat'     : res['vflat'],
            'rs'        : res['rs'],
            'upsilon_d' : res['upsilon_d'],
            'w'         : res['w'],
            'chi2_dof'  : res['chi2_dof'],
            'chi2_total': res['chi2_total'],
            'n_pts'     : res['n_pts'],
            'rms_resid' : res['rms_resid'],
        })

    # CSV 保存
    if records:
        df_out = pd.DataFrame(records)
        out_path = Path(args.output)
        df_out.to_csv(out_path, index=False,
                      encoding='utf-8-sig')
        print(f"\n→ {out_path} ({len(records)} 銀河)")

    # JSON 保存（プロット用）
    json_path = Path(args.output).with_suffix('.json')
    # r, vobs 等の配列を含む詳細データ
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_detail, f,
                  ensure_ascii=False, indent=2)
    print(f"→ {json_path} (detail)")

if __name__ == '__main__':
    main()
