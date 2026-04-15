import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'

COLS = [
    'object_id','ra','dec',
    'e1','e2','sigma_e','rms_e','weight',
    'shear_m','shear_c1','shear_c2',
    'resolution','mag_i','zbin','b_mode_mask',
    'ra2','dec2','e1_2','e2_2'
]

CLUSTERS = {
    'J0201': {'ra':30.652,  'dec':-4.955, 'z_l':0.196, 'r_s':0.169, 'r500':0.676, 'rho_s':4.8e14, 'SP':0.610, 'dxi':0.302},
    'J0231': {'ra':37.941,  'dec':-4.762, 'z_l':0.184, 'r_s':0.190, 'r500':0.760, 'rho_s':4.2e14, 'SP':0.528, 'dxi':0.296},
    'J1311': {'ra':197.863, 'dec': 0.403, 'z_l':0.183, 'r_s':0.246, 'r500':0.985, 'rho_s':3.5e14, 'SP':0.736, 'dxi':0.260},
    'J2337': {'ra':354.378, 'dec':-0.022, 'z_l':0.282, 'r_s':0.238, 'r500':0.952, 'rho_s':3.6e14, 'SP':0.636, 'dxi':0.337},
    'J1415': {'ra':213.698, 'dec': 0.186, 'z_l':0.143, 'r_s':0.050, 'r500':0.560, 'rho_s':6.45e13,'SP':0.202, 'dxi':0.207},
    'J0158': {'ra':29.697,  'dec':-4.888, 'z_l':0.230, 'r_s':0.133, 'r500':0.545, 'rho_s':6.4e12, 'SP':0.677, 'dxi':0.352},
}

def sigma_cr(z_l, z_s, cosmo):
    c = 2.998e8
    G = 6.674e-11
    Msun = 1.989e30
    Mpc = 3.086e22
    Ds = cosmo.angular_diameter_distance(z_s).value
    Dl = cosmo.angular_diameter_distance(z_l).value
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    if Dls <= 0 or Ds <= 0:
        return np.inf
    Sigma_cr = (c**2/(4*np.pi*G)) * (Ds/(Dl*Dls)) / (Msun/Mpc**2)
    return Sigma_cr

def nfw_gamma_t(R_mpc, rho_s, r_s):
    x = R_mpc / r_s
    def g(x):
        out = np.zeros_like(x, dtype=float)
        m1 = x < 1
        m2 = x == 1
        m3 = x > 1
        out[m1] = np.log(x[m1]/2) + 1/np.sqrt(1-x[m1]**2) * np.arccosh(1/x[m1])
        out[m2] = 1 + np.log(0.5)
        out[m3] = np.log(x[m3]/2) + 1/np.sqrt(x[m3]**2-1) * np.arccos(1/x[m3])
        return out
    def f(x):
        out = np.zeros_like(x, dtype=float)
        m1 = x < 1
        m2 = x == 1
        m3 = x > 1
        out[m1] = 1/np.sqrt(1-x[m1]**2) * np.arctanh(np.sqrt(1-x[m1]**2))
        out[m2] = 1.0
        out[m3] = 1/np.sqrt(x[m3]**2-1) * np.arctan(np.sqrt(x[m3]**2-1))
        return out
    kappa_s = rho_s * r_s
    Sigma_bar = 4 * kappa_s * r_s * g(x) / x**2
    Sigma = 2 * kappa_s * r_s * f(x)
    return Sigma_bar - Sigma

z_bins_center = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0}
R_bins = np.array([0.05, 0.15, 0.30, 0.50, 0.75, 1.00, 1.40, 1.80, 2.20])
R_mid = 0.5 * (R_bins[:-1] + R_bins[1:])

print('=== 全天カタログ読み込み・γ_t計算 ===')

all_results = {}

for cl_name, cl in CLUSTERS.items():
    print(f'\n--- {cl_name} ---')
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
        continue

    df = pd.concat(chunks, ignore_index=True)
    dra = (df['ra'] - cl['ra']) * cos_dec
    ddec = df['dec'] - cl['dec']
    sep_deg = np.sqrt(dra**2 + ddec**2)
    df = df[sep_deg <= r_search].copy()
    sep_deg = sep_deg[sep_deg.index.isin(df.index)]

    R_mpc = sep_deg.values * np.pi/180 * D_A

    phi = np.arctan2(df['dec'].values - cl['dec'],
                     (df['ra'].values - cl['ra']) * cos_dec)

    e1_c = (df['e1'].values - df['shear_c1'].values) / (1 + df['shear_m'].values)
    e2_c = (df['e2'].values - df['shear_c2'].values) / (1 + df['shear_m'].values)

    et = -(e1_c * np.cos(2*phi) + e2_c * np.sin(2*phi))
    w = df['weight'].values
    zbin = df['zbin'].values

    Scr = np.array([sigma_cr(cl['z_l'], z_bins_center.get(int(z), 0.7), cosmo)
                    for z in zbin])

    gt_bins = np.zeros(len(R_mid))
    gt_nfw_bins = np.zeros(len(R_mid))
    n_bins = np.zeros(len(R_mid))

    for i, (r1, r2) in enumerate(zip(R_bins[:-1], R_bins[1:])):
        in_bin = (R_mpc >= r1) & (R_mpc < r2) & np.isfinite(Scr) & (Scr > 0)
        if in_bin.sum() < 5:
            gt_bins[i] = np.nan
            gt_nfw_bins[i] = np.nan
            continue
        wi = w[in_bin]
        eti = et[in_bin]
        Scri = Scr[in_bin]
        gt_bins[i] = np.average(eti, weights=wi)
        nfw_val = nfw_gamma_t(np.array([R_mid[i]]), cl['rho_s'], cl['r_s'])[0]
        gt_nfw_bins[i] = nfw_val / np.mean(Scri)
        n_bins[i] = in_bin.sum()

    kappa_G = gt_bins - gt_nfw_bins

    rs_mpc = cl['r_s']
    r500_mpc = cl['r500']
    inner = (R_mid < rs_mpc) & np.isfinite(kappa_G)
    outer = (R_mid >= rs_mpc) & (R_mid < r500_mpc) & np.isfinite(kappa_G)

    kg_inner = np.nanmean(kappa_G[inner]) if inner.sum() > 0 else np.nan
    kg_outer = np.nanmean(kappa_G[outer]) if outer.sum() > 0 else np.nan
    Q = abs(kg_inner) / abs(kg_outer) if (kg_outer != 0 and np.isfinite(kg_outer)) else np.nan
    kg_int = np.nanmean(kappa_G[R_mid < r500_mpc])

    all_results[cl_name] = {
        'Q': Q, 'kg_inner': kg_inner, 'kg_outer': kg_outer,
        'kg_int': kg_int, 'N': len(df),
        'R_mid': R_mid, 'kappa_G': kappa_G, 'gt_obs': gt_bins, 'gt_nfw': gt_nfw_bins
    }
    print(f'  N={len(df)}, Q={Q:.3f}, kg_inner={kg_inner:.4f}, kg_outer={kg_outer:.4f}, kg_int={kg_int:.4f}')

print()
print('=== T-14 Q値まとめ ===')
print('クラスター  Q旧   Q新   判定  SP     kg_int旧  kg_int新')
Q_old = {'J0201':3.71,'J0231':1.48,'J1311':3.04,'J2337':2.69}
kg_old = {'J0201':-0.017,'J0231':-0.043,'J1311':-0.063,'J2337':-0.086}
for cl_name, res in all_results.items():
    q_o = Q_old.get(cl_name, '-')
    kg_o = kg_old.get(cl_name, '-')
    sp = CLUSTERS[cl_name]['SP']
    print(cl_name, q_o, round(res['Q'],2), 'Q>1' if res['Q']>1 else 'Q<1',
          sp, kg_o, round(res['kg_int'],4))

from scipy import stats
Qs = [res['Q'] for res in all_results.values() if np.isfinite(res['Q'])]
t_stat, p_val = stats.ttest_1samp(Qs, 1.0)
print()
print('t検定 H0:Q=1 ->  N=', len(Qs), ' t=', round(t_stat,3), ' p=', round(p_val,4))
print('平均Q=', round(np.mean(Qs),3))
