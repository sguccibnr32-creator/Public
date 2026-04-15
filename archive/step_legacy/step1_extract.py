import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'
COLS = ['object_id','ra','dec','e1','e2','sigma_e','rms_e','weight',
        'shear_m','shear_c1','shear_c2','resolution','mag_i','zbin','b_mode_mask',
        'ra2','dec2','e1_2','e2_2']

CLUSTERS = {
    'J0201':{'ra':30.652,  'dec':-4.955,'z_l':0.196,'r_s':0.169,'r500':0.676,'rho_s':4.8e14,'SP':0.610},
    'J0231':{'ra':37.941,  'dec':-4.762,'z_l':0.184,'r_s':0.190,'r500':0.760,'rho_s':4.2e14,'SP':0.528},
    'J1311':{'ra':197.863, 'dec': 0.403,'z_l':0.183,'r_s':0.246,'r500':0.985,'rho_s':3.5e14,'SP':0.736},
    'J2337':{'ra':354.378, 'dec':-0.022,'z_l':0.282,'r_s':0.238,'r500':0.952,'rho_s':3.6e14,'SP':0.636},
    'J1415':{'ra':213.698, 'dec': 0.186,'z_l':0.143,'r_s':0.050,'r500':0.560,'rho_s':6.45e13,'SP':0.202},
    'J0158':{'ra':29.697,  'dec':-4.888,'z_l':0.230,'r_s':0.133,'r500':0.545,'rho_s':6.4e12,'SP':0.677},
}

for cn, cl in CLUSTERS.items():
    DA = cosmo.angular_diameter_distance(cl['z_l']).value
    rs = cl['r500'] * 8.0 / DA * (180/np.pi)
    cd = np.cos(np.radians(cl['dec']))
    chunks = []
    for chunk in pd.read_csv(PATH, comment='#', header=None, names=COLS, chunksize=500000):
        m = ((chunk['ra']  >= cl['ra'] -rs/cd) & (chunk['ra']  <= cl['ra'] +rs/cd) &
             (chunk['dec'] >= cl['dec']-rs)     & (chunk['dec'] <= cl['dec']+rs)     &
             (chunk['b_mode_mask']==1) & (chunk['resolution']>=0.3) & (chunk['zbin']>=1))
        if m.sum()>0: chunks.append(chunk[m])
    if not chunks: print(cn,'なし'); continue
    df = pd.concat(chunks, ignore_index=True)
    dra = (df['ra'].values - cl['ra'])*cd
    ddec = df['dec'].values - cl['dec']
    sep = np.sqrt(dra**2+ddec**2)
    df = df[sep<=rs].copy()
    df.to_csv(cn+'_src.csv', index=False)
    print(cn, 'N=', len(df), '保存完了')

print('完了')
