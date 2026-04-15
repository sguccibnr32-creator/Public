import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# --- SSS2候補49件を読み込み ---
sss2 = pd.read_csv('cluster_catalog.csv')
print(f'SSS2候補: {len(sss2)}件')

# --- PSZ2カタログをVizierから取得 ---
print('PSZ2カタログをVizierから取得中...')
v = Vizier(columns=['*'], row_limit=-1)
result = v.get_catalogs('J/A+A/594/A27/psz2')
psz2 = result[0].to_pandas()
print(f'PSZ2: {len(psz2)}件取得')

# カラム名確認
print('PSZ2カラム:', list(psz2.columns[:10]))

# --- 座標列を特定 ---
# PSZ2のRA/Dec列名を探す
ra_col = [c for c in psz2.columns if c.upper() in ('RA', '_RAJ2000', 'RAJ2000')][0]
dec_col = [c for c in psz2.columns if c.upper() in ('DEC', '_DEJ2000', 'DEJ2000', 'DE', 'DEC_')][0]
print(f'PSZ2座標列: {ra_col}, {dec_col}')

# --- SkyCoordオブジェクト作成 ---
sss2_coords = SkyCoord(ra=sss2['ra_center'].values*u.deg,
                       dec=sss2['dec_center'].values*u.deg)
psz2_coords = SkyCoord(ra=psz2[ra_col].values*u.deg,
                       dec=psz2[dec_col].values*u.deg)

# --- クロスマッチ (5 arcmin以内) ---
RADIUS = 5.0  # arcmin
idx, sep, _ = sss2_coords.match_to_catalog_sky(psz2_coords)
matched = sep.to(u.arcmin).value < RADIUS

print(f'\n=== クロスマッチ結果 (マッチ半径: {RADIUS} arcmin) ===')
print(f'SSS2候補中でPSZ2と一致: {matched.sum()}件 / {len(sss2)}件')

# --- 結果テーブル ---
rows = []
for i, (m, j, s) in enumerate(zip(matched, idx, sep.to(u.arcmin).value)):
    sss2_row = sss2.iloc[i]
    entry = {
        'sss2_id': int(sss2_row['cluster_id']),
        'sss2_ra': round(sss2_row['ra_center'], 4),
        'sss2_dec': round(sss2_row['dec_center'], 4),
        'sss2_conc_max': round(sss2_row['conc_max'], 3),
        'sss2_z_dlt': round(sss2_row['z_dlt_mean'], 3),
        'field': int(sss2_row['field']),
        'matched': m,
        'sep_arcmin': round(s, 2),
    }
    if m:
        psz2_row = psz2.iloc[j]
        # PSZ2のname/z列を探す
        name_col = next((c for c in psz2.columns if 'NAME' in c.upper() or 'name' in c), None)
        z_col = next((c for c in psz2.columns if c in ('z', 'redshift', 'Redshift', 'REDSHIFT')), None)
        entry['psz2_name'] = psz2_row[name_col] if name_col else ''
        entry['psz2_ra'] = round(float(psz2_row[ra_col]), 4)
        entry['psz2_dec'] = round(float(psz2_row[dec_col]), 4)
        entry['psz2_z'] = round(float(psz2_row[z_col]), 3) if z_col else np.nan
    else:
        entry['psz2_name'] = ''
        entry['psz2_ra'] = np.nan
        entry['psz2_dec'] = np.nan
        entry['psz2_z'] = np.nan
    rows.append(entry)

df_result = pd.DataFrame(rows)

# --- マッチしたもの表示 ---
matched_df = df_result[df_result['matched']].reset_index(drop=True)
print('\nマッチしたSSS2候補:')
print(matched_df[['sss2_id','sss2_ra','sss2_dec','sss2_conc_max','sss2_z_dlt',
                   'field','sep_arcmin','psz2_name','psz2_z']].to_string(index=False))

# --- 未マッチ ---
unmatched_df = df_result[~df_result['matched']].reset_index(drop=True)
print(f'\n未マッチ: {len(unmatched_df)}件')
print(unmatched_df[['sss2_id','sss2_ra','sss2_dec','sss2_conc_max','field',
                     'sep_arcmin']].to_string(index=False))

# --- 保存 ---
df_result.to_csv('psz2_sss2_crossmatch.csv', index=False)
print('\n結果を psz2_sss2_crossmatch.csv に保存しました')
