import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'

COLS = [
    'object_id','ra','dec',
    'e1','e2','sigma_e','rms_e','weight',
    'shear_m','shear_c1','shear_c2',
    'resolution','mag_i','zbin','b_mode_mask',
    'ra2','dec2','e1_2','e2_2'
]

CLUSTERS = {
    'J0201': {'ra':30.652,  'dec':-4.955, 'z_l':0.196, 'r500':0.676},
    'J0231': {'ra':37.941,  'dec':-4.762, 'z_l':0.184, 'r500':0.760},
    'J1311': {'ra':197.863, 'dec': 0.403, 'z_l':0.183, 'r500':0.985},
    'J2337': {'ra':354.378, 'dec':-0.022, 'z_l':0.282, 'r500':0.952},
    'J1415': {'ra':213.698, 'dec': 0.186, 'z_l':0.143, 'r500':0.560},
    'J0158': {'ra':29.697,  'dec':-4.888, 'z_l':0.230, 'r500':0.545},
}

N_OLD = {'J0201':12754,'J0231':16263,'J1311':10213,'J2337':17913,'J1415':16351,'J0158':7214}

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

print('=== 検索半径を r500x8 に拡大して再抽出 ===')

results = {}

for cl_name, cl in CLUSTERS.items():
    D_A = cosmo.angular_diameter_distance(cl['z_l']).value
    r_search = cl['r500'] * 8.0 / D_A * (180/np.pi)
    cos_dec = np.cos(np.radians(cl['dec']))
    ra_min = cl['ra'] - r_search / cos_dec
    ra_max = cl['ra'] + r_search / cos_dec
    dec_min = cl['dec'] - r_search
    dec_max = cl['dec'] + r_search

    chunks = []
    reader = pd.read_csv(PATH, comment='#', header=None, names=COLS, chunksize=500000)
    for chunk in reader:
        mask = (
            (chunk['ra'] >= ra_min) & (chunk['ra'] <= ra_max) &
            (chunk['dec'] >= dec_min) & (chunk['dec'] <= dec_max) &
            (chunk['b_mode_mask'] == 1) &
            (chunk['resolution'] >= 0.3) &
            (chunk['zbin'] >= 1)
        )
        sub = chunk[mask]
        if len(sub) > 0:
            chunks.append(sub)

    if not chunks:
        print(cl_name, 'データなし')
        continue

    df_cl = pd.concat(chunks, ignore_index=True)
    dra = (df_cl['ra'] - cl['ra']) * cos_dec
    ddec = df_cl['dec'] - cl['dec']
    sep = np.sqrt(dra**2 + ddec**2)
    df_cl = df_cl[sep <= r_search].copy()
    df_cl['sep_deg'] = sep[sep <= r_search].values

    m_mean = df_cl['shear_m'].mean()
    cf = 1.0 / (1.0 + m_mean)
    n_old = N_OLD.get(cl_name, 0)
    ratio = len(df_cl) / n_old if n_old > 0 else 0

    results[cl_name] = {
        'N': len(df_cl), 'm': m_mean, 'cf': cf,
        'r_search_deg': r_search, 'df': df_cl
    }
    print(cl_name, 'N=', len(df_cl), '旧N=', n_old, '倍率=', round(ratio,2),
          'm=', round(m_mean,4), '補正係数=', round(cf,4))

print()
print('=== zbin別内訳（参考）===')
for cl_name, res in results.items():
    zb = res['df']['zbin'].value_counts().sort_index()
    print(cl_name, dict(zb))

print()
print('=== m補正がκ_Gに与える影響（試算）===')
kappa_G_old = {'J0201':-0.017,'J0231':-0.043,'J1311':-0.063,'J2337':-0.086}
print('クラスター  κ_G旧値   m補正後    変化')
for cl_name, kg in kappa_G_old.items():
    if cl_name in results:
        cf = results[cl_name]['cf']
        kg_new = kg * cf
        print(cl_name, round(kg,4), round(kg_new,4), round((cf-1)*100,1), '%増')
