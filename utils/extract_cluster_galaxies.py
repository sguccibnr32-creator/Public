# -*- coding: utf-8 -*-
"""
HSC Y3 弱レンズカタログからクラスター周辺天体を抽出
9.6GB CSV をチャンク読みで処理
"""
import pandas as pd
import numpy as np
import time, sys
from pathlib import Path

HSC_PATH = Path(r"D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1")
OUT_DIR  = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1")

# 実際の列名（ヘッダから確認済み）
COLS_USE = [
    '# object_id',
    'i_ra', 'i_dec',
    'i_hsmshaperegauss_e1',
    'i_hsmshaperegauss_e2',
    'i_hsmshaperegauss_derived_weight',
    'i_hsmshaperegauss_derived_shear_bias_m',
    'i_hsmshaperegauss_derived_shear_bias_c1',
    'i_hsmshaperegauss_derived_shear_bias_c2',
    'i_hsmshaperegauss_resolution',
    'i_apertureflux_10_mag',
    'hsc_y3_zbin',
    'b_mode_mask',
]

# HSC Y3 フットプリント内の既知銀河団
# RA, Dec は HSC footprint (主に春の空 + 秋の一部)
CLUSTERS = {
    # 名前: (RA, Dec, z_cluster, M200_approx [1e14 M_sun])
    'A2142':   (239.583, 27.233, 0.091, 8.0),   # Abell 2142 — HSC外かも
    'RXJ1347': (206.878, -11.753, 0.451, 15.0),  # massive, HSC-S field?
    # HSC Wide のメインフィールド内クラスター
    'GAMA09':  (129.0, 0.5, 0.30, 3.0),          # GAMA field test point
    'GAMA15':  (217.0, 0.5, 0.30, 3.0),
    'XMM-LSS': (35.0, -4.5, 0.30, 3.0),          # XMM-LSS field
    'VVDS':    (150.0, 2.2, 0.30, 3.0),           # COSMOS/VVDS
    'HECTOMAP':(243.0, 43.0, 0.30, 3.0),          # Hectomap
}

def extract_around_cluster(csv_path, ra_cen, dec_cen,
                           radius_deg=0.5, chunksize=500000):
    """
    クラスター周辺天体をチャンク読みで抽出
    """
    cos_dec = np.cos(np.radians(dec_cen))
    ra_min  = ra_cen - radius_deg / cos_dec
    ra_max  = ra_cen + radius_deg / cos_dec
    dec_min = dec_cen - radius_deg
    dec_max = dec_cen + radius_deg

    results = []
    n_total = 0

    for chunk in pd.read_csv(csv_path, usecols=COLS_USE,
                              chunksize=chunksize, comment=None):
        n_total += len(chunk)
        mask = (
            (chunk['i_ra']  >= ra_min)  &
            (chunk['i_ra']  <= ra_max)  &
            (chunk['i_dec'] >= dec_min) &
            (chunk['i_dec'] <= dec_max) &
            (chunk['b_mode_mask'] == 1)
        )
        if mask.sum() > 0:
            results.append(chunk[mask].copy())

    if results:
        return pd.concat(results, ignore_index=True), n_total
    return pd.DataFrame(), n_total

def angular_distance(ra1, dec1, ra2, dec2):
    """角距離 [deg]"""
    d2r = np.pi / 180
    cos_d = (np.sin(dec1*d2r)*np.sin(dec2*d2r) +
             np.cos(dec1*d2r)*np.cos(dec2*d2r)*np.cos((ra1-ra2)*d2r))
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

# ============================================================
# Step 1: まず各フィールドに天体があるか確認（小領域スキャン）
# ============================================================
if __name__ == '__main__':
    print(f"HSC catalog: {HSC_PATH}")
    print(f"File size: {HSC_PATH.stat().st_size / 1e9:.1f} GB")

    # まず RA/Dec の分布を確認（最初の100万行）
    print("\n=== RA/Dec range (first 1M rows) ===")
    sample = pd.read_csv(HSC_PATH, usecols=['i_ra','i_dec'],
                         nrows=1000000)
    print(f"  RA:  [{sample['i_ra'].min():.2f}, {sample['i_ra'].max():.2f}]")
    print(f"  Dec: [{sample['i_dec'].min():.2f}, {sample['i_dec'].max():.2f}]")

    # フルスキャンで RA/Dec ヒストグラムを取得
    print("\n=== Full scan: RA/Dec range + field density ===")
    ra_all, dec_all = [], []
    n_total = 0
    t0 = time.time()

    for chunk in pd.read_csv(HSC_PATH, usecols=['i_ra','i_dec'],
                              chunksize=2000000):
        n_total += len(chunk)
        # 各クラスター候補位置の天体数を数える
        for name, (ra_c, dec_c, z, m) in CLUSTERS.items():
            cos_dec = np.cos(np.radians(dec_c))
            mask = (
                (np.abs(chunk['i_ra'] - ra_c) < 0.5/cos_dec) &
                (np.abs(chunk['i_dec'] - dec_c) < 0.5)
            )
            if mask.sum() > 0:
                if name not in [n for n,_ in ra_all]:
                    ra_all.append((name, mask.sum()))
                else:
                    for i, (n_, c_) in enumerate(ra_all):
                        if n_ == name:
                            ra_all[i] = (n_, c_ + mask.sum())
                            break

        if n_total % 10000000 < 2000000:
            elapsed = time.time() - t0
            print(f"  {n_total/1e6:.0f}M rows scanned ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total: {n_total} rows in {elapsed:.0f}s")
    print(f"\n  Field density around cluster candidates:")
    for name, count in ra_all:
        print(f"    {name:<12s}: {count:,d} sources within 0.5 deg")

    # 最も密度の高いフィールドで本抽出
    if ra_all:
        best_name = max(ra_all, key=lambda x: x[1])
        print(f"\n  -> Best field: {best_name[0]} ({best_name[1]:,d} sources)")
        ra_c, dec_c, z_c, m_c = CLUSTERS[best_name[0]]

        print(f"\n=== Extracting {best_name[0]} (r<0.5 deg) ===")
        t1 = time.time()
        df_cluster, _ = extract_around_cluster(HSC_PATH, ra_c, dec_c,
                                                radius_deg=0.5)
        t2 = time.time()
        print(f"  Extracted {len(df_cluster)} sources in {t2-t1:.0f}s")

        if len(df_cluster) > 0:
            # 角距離を計算
            df_cluster['theta_deg'] = angular_distance(
                df_cluster['i_ra'].values, df_cluster['i_dec'].values,
                ra_c, dec_c)

            # z_bin 分布
            print(f"\n  z_bin distribution:")
            print(df_cluster['hsc_y3_zbin'].value_counts().sort_index().to_string())

            # 保存
            outpath = OUT_DIR / f'hsc_{best_name[0]}_sources.csv'
            df_cluster.to_csv(outpath, index=False, encoding='utf-8-sig')
            print(f"\n  -> {outpath} saved ({len(df_cluster)} sources)")

            # 基本統計
            print(f"\n  Statistics:")
            print(f"    RA:  [{df_cluster['i_ra'].min():.3f}, {df_cluster['i_ra'].max():.3f}]")
            print(f"    Dec: [{df_cluster['i_dec'].min():.3f}, {df_cluster['i_dec'].max():.3f}]")
            print(f"    e1 mean: {df_cluster['i_hsmshaperegauss_e1'].mean():.5f}")
            print(f"    e2 mean: {df_cluster['i_hsmshaperegauss_e2'].mean():.5f}")
            print(f"    weight mean: {df_cluster['i_hsmshaperegauss_derived_weight'].mean():.3f}")
            print(f"    mag_i range: [{df_cluster['i_apertureflux_10_mag'].min():.1f}, "
                  f"{df_cluster['i_apertureflux_10_mag'].max():.1f}]")
    else:
        print("\n  No sources found near any candidate cluster position.")
        print("  Checking actual RA/Dec coverage...")
        # RA/Dec のユニークな範囲を確認
        ra_ranges = []
        for chunk in pd.read_csv(HSC_PATH, usecols=['i_ra','i_dec'],
                                  chunksize=5000000, nrows=10000000):
            ra_ranges.append((chunk['i_ra'].min(), chunk['i_ra'].max(),
                             chunk['i_dec'].min(), chunk['i_dec'].max()))
        for i, (rmin, rmax, dmin, dmax) in enumerate(ra_ranges):
            print(f"    Chunk {i}: RA=[{rmin:.1f},{rmax:.1f}], Dec=[{dmin:.1f},{dmax:.1f}]")
