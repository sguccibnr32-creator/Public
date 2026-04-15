import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'
COLS = ['object_id','ra','dec','e1','e2','sigma_e','rms_e','weight',
        'shear_m','shear_c1','shear_c2','resolution','mag_i','zbin','b_mode_mask',
        'ra2','dec2','e1_2','e2_2']

df_cl = pd.read_csv('psz2_hsc_ready.csv')

# J0231と重複するPSZ2クラスターを除外
EXCLUDE_RA  = [37.930]
EXCLUDE_DEC = [-4.861]
mask_dup = pd.Series([False]*len(df_cl))
for ra_ex, dec_ex in zip(EXCLUDE_RA, EXCLUDE_DEC):
    mask_dup |= ((df_cl['RA']-ra_ex).abs()<0.05) & ((df_cl['Dec']-dec_ex).abs()<0.05)
df_cl = df_cl[~mask_dup].reset_index(drop=True)
print(f'解析対象: {len(df_cl)}件（J0231重複除外後）')
print()

results = {}
for _, cl in df_cl.iterrows():
    cn   = cl['Name'].replace(' ','_')
    DA   = cosmo.angular_diameter_distance(cl['z']).value
    r_search = cl['r500'] * 8.0 / DA * (180/np.pi)
    cd   = np.cos(np.radians(cl['Dec']))

    chunks = []
    reader = pd.read_csv(PATH, comment='#', header=None, names=COLS, chunksize=500000)
    for chunk in reader:
        m = (
            (chunk['ra']  >= cl['RA'] - r_search/cd) &
            (chunk['ra']  <= cl['RA'] + r_search/cd) &
            (chunk['dec'] >= cl['Dec'] - r_search)   &
            (chunk['dec'] <= cl['Dec'] + r_search)   &
            (chunk['b_mode_mask'] == 1) &
            (chunk['resolution'] >= 0.3) &
            (chunk['zbin'] >= 1)
        )
        if m.sum() > 0:
            chunks.append(chunk[m])

    if not chunks:
        print(f'{cn}: データなし')
        continue

    src = pd.concat(chunks, ignore_index=True)
    dra  = (src['ra'].values  - cl['RA'])  * cd
    ddec =  src['dec'].values - cl['Dec']
    sep  = np.sqrt(dra**2 + ddec**2)
    src  = src[sep <= r_search].copy()

    # レンズ後赤方偏移より手前の銀河を除外
    z_min_src = cl['z'] + 0.1
    ZSRC = {1:0.4, 2:0.65, 3:0.85, 4:1.05}
    src = src[src['zbin'].map(ZSRC) >= z_min_src].copy()

    src.to_csv(f'psz2_{cn}_src.csv', index=False)
    print(f'{cl["Name"][:30]:32s} z={cl["z"]:.3f}  N={len(src):6d}  field={cl["hsc_field"]}')
    results[cn] = len(src)

print()
print(f'=== 完了: {len(results)}クラスター ===')
total = sum(results.values())
print(f'総背景銀河数: {total:,}')
