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

print('=== 全天カタログ抽出開始（35M行） ===')

results = {}

for cl_name, cl in CLUSTERS.items():
    D_A = cosmo.angular_diameter_distance(cl['z_l']).value
    r_search = cl['r500'] * 3.0 / D_A * (180/np.pi)
    cos_dec = np.cos(np.radians(cl['dec']))
    ra_min = cl['ra'] - r_search / cos_dec
    ra_max = cl['ra'] + r_search / cos_dec
    dec_min = cl['dec'] - r_search
    dec_max = cl['dec'] + r_search

    print(cl_name, 'r_search=', round(r_search,3), 'deg')

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
        print('  データなし')
        continue

    df_cl = pd.concat(chunks, ignore_index=True)
    dra = (df_cl['ra'] - cl['ra']) * np.cos(np.radians(cl['dec']))
    ddec = df_cl['dec'] - cl['dec']
    sep = np.sqrt(dra**2 + ddec**2)
    df_cl = df_cl[sep <= r_search].copy()

    df_cl['e1_corr'] = (df_cl['e1'] - df_cl['shear_c1']) / (1 + df_cl['shear_m'])
    df_cl['e2_corr'] = (df_cl['e2'] - df_cl['shear_c2']) / (1 + df_cl['shear_m'])

    m_mean = df_cl['shear_m'].mean()
    results[cl_name] = {'N': len(df_cl), 'm': m_mean, 'cf': 1/(1+m_mean)}

    n_old = N_OLD.get(cl_name, 0)
    ratio = len(df_cl) / n_old if n_old > 0 else 0
    print('  N=', len(df_cl), ' m=', round(m_mean,4), ' 補正係数=', round(1/(1+m_mean),4), ' 旧N比=', round(ratio,2), 'x')

print()
print('=== サマリー ===')
print('クラスター  N_Y3    N_旧    倍率   m補正係数')
for cl_name in CLUSTERS:
    if cl_name not in results:
        continue
    res = results[cl_name]
    n_old = N_OLD.get(cl_name, 0)
    ratio = res['N'] / n_old if n_old > 0 else 0
    print(cl_name, res['N'], n_old, round(ratio,2), round(res['cf'],4))
