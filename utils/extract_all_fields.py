# -*- coding: utf-8 -*-
"""
HSC Y3 から XMM-LSS, VVDS, GAMA09 フィールドも抽出
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path

HSC_PATH = Path(r"D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1")
OUT_DIR  = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1")

# 各フィールドの中心と抽出半径
FIELDS = {
    'XMM-LSS': (35.0, -4.5, 8.0),   # RA, Dec, radius_deg (広めに取る)
    'VVDS':    (150.0, 2.2, 5.0),
    'GAMA09':  (129.0, 0.5, 5.0),
    'GAMA15':  (217.0, 0.5, 5.0),    # 既に抽出済みだが広範囲で再抽出
    'HECTOMAP':(243.0, 43.0, 5.0),
}

# クラスターターゲット位置（必要な範囲を確認）
CLUSTERS = [
    ('J1415.2-0030', 213.809, -0.501),
    ('J1115.8+0129', 168.975,  1.496),
    ('J1023.6+0411', 155.912,  4.186),
    ('J0157.4-0550',  29.351, -5.840),
    ('J0231.7-0451',  37.947, -4.856),
    ('J1311.5-0120', 197.875, -1.335),
    ('J1258.6-0145', 194.671, -1.757),
    ('J0106.8+0103',  16.710,  1.055),
    ('J1217.6+0339', 184.419,  3.663),
]

# 各クラスター周辺 0.6 deg を確実にカバーするフィールド範囲を計算
print("Cluster -> Field mapping:")
for cname, cra, cdec in CLUSTERS:
    best_field = None
    best_dist = 999
    for fname, (fra, fdec, frad) in FIELDS.items():
        d = np.sqrt((cra-fra)**2 + (cdec-fdec)**2)
        if d < frad and d < best_dist:
            best_dist = d
            best_field = fname
    print(f"  {cname:<18s} RA={cra:7.3f} Dec={cdec:+7.3f} -> {best_field or 'NO FIELD'}")

# 全クラスターをカバーする最小矩形を各フィールドで計算
# 実際にはクラスター位置ベースで直接抽出する方が効率的
# 全クラスター周辺 0.6 deg を一括抽出

print(f"\nExtracting sources around {len(CLUSTERS)} clusters...")
t0 = time.time()

COLS = ['# object_id','i_ra','i_dec',
        'i_hsmshaperegauss_e1','i_hsmshaperegauss_e2',
        'i_hsmshaperegauss_derived_weight',
        'i_hsmshaperegauss_derived_shear_bias_m',
        'i_hsmshaperegauss_derived_shear_bias_c1',
        'i_hsmshaperegauss_derived_shear_bias_c2',
        'i_hsmshaperegauss_resolution',
        'i_apertureflux_10_mag',
        'hsc_y3_zbin','b_mode_mask']

RADIUS = 0.6  # deg around each cluster

cluster_sources = {c[0]: [] for c in CLUSTERS}
n_total = 0

for chunk in pd.read_csv(HSC_PATH, usecols=COLS, chunksize=2000000):
    n_total += len(chunk)
    # b_mode_mask フィルタ
    chunk = chunk[chunk['b_mode_mask'] == 1]

    for cname, cra, cdec in CLUSTERS:
        cos_dec = np.cos(np.radians(cdec))
        mask = (
            (np.abs(chunk['i_ra'] - cra) < RADIUS / cos_dec) &
            (np.abs(chunk['i_dec'] - cdec) < RADIUS)
        )
        if mask.sum() > 0:
            cluster_sources[cname].append(chunk[mask].copy())

    if n_total % 10000000 < 2000000:
        print(f"  {n_total/1e6:.0f}M rows ({time.time()-t0:.0f}s)")

print(f"\nDone: {n_total} rows in {time.time()-t0:.0f}s")

# 保存
for cname, chunks in cluster_sources.items():
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        # 角距離計算
        cra = [c[1] for c in CLUSTERS if c[0]==cname][0]
        cdec = [c[2] for c in CLUSTERS if c[0]==cname][0]
        cos_d = np.cos(np.radians(cdec))
        dra = (df['i_ra'] - cra) * cos_d * 60  # arcmin
        ddec = (df['i_dec'] - cdec) * 60
        df['theta_arcmin'] = np.sqrt(dra**2 + ddec**2)

        safe_name = cname.replace('+','p').replace('-','m')
        outpath = OUT_DIR / f'hsc_{safe_name}.csv'
        df.to_csv(outpath, index=False, encoding='utf-8-sig')
        print(f"  {cname:<18s}: {len(df):6d} sources -> {outpath.name}")

        # z_bin summary
        zb = df['hsc_y3_zbin'].value_counts().sort_index()
        print(f"    z_bin: {dict(zb)}")
    else:
        print(f"  {cname:<18s}: 0 sources (outside HSC footprint)")
