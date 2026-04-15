import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

df = pd.read_csv('psz2_in_hsc.csv')
print(f'読み込み: {len(df)}件')

# r500 を MSZ から計算
# MSZ [10^14 M_sun] -> M500 -> r500
rho_crit0 = 2.775e11  # M_sun/Mpc^3 (z=0)
def calc_r500(row):
    z = row['z']
    M500 = row['MSZ'] * 1e14  # M_sun
    Ez2 = cosmo.efunc(z)**2
    rho_crit = rho_crit0 * Ez2
    r500 = (3*M500 / (4*np.pi*500*rho_crit))**(1/3)
    return round(r500, 4)

def calc_rho_s(row):
    # NFW concentration c500=3.5 (典型値)
    c500 = 3.5
    z = row['z']
    M500 = row['MSZ'] * 1e14
    Ez2 = cosmo.efunc(z)**2
    rho_crit = rho_crit0 * Ez2
    r500 = (3*M500 / (4*np.pi*500*rho_crit))**(1/3)
    r_s = r500 / c500
    rho_s = (500/3) * rho_crit * c500**3 / (np.log(1+c500) - c500/(1+c500))
    return round(rho_s, 3), round(r_s, 4)

df['r500'] = df.apply(calc_r500, axis=1)
df['rho_s'] = df.apply(lambda r: calc_rho_s(r)[0], axis=1)
df['r_s']   = df.apply(lambda r: calc_rho_s(r)[1], axis=1)

# HSC-SSP 実フットプリント（厳密範囲）
hsc_strict = [
    (29, 43,  -7,  -1, 'XMM-LSS'),
    (327, 347, -4,   4, 'VVDS'),
    (174, 193, -3,   6, 'WIDE12H'),
    (208, 225, -2,   6, 'GAMA15'),
    (127, 143, -3,   5, 'GAMA09'),
]
def in_hsc(row):
    for ra1,ra2,d1,d2,fn in hsc_strict:
        if ra1<=row['RA']<=ra2 and d1<=row['Dec']<=d2:
            return fn
    return ''

df['hsc_field'] = df.apply(in_hsc, axis=1)
df_hsc = df[df['hsc_field']!=''].copy()

# z>0.05, z<0.7 の範囲に絞る（弱レンズ解析可能範囲）
df_hsc = df_hsc[(df_hsc['z']>=0.05)&(df_hsc['z']<=0.70)].copy()

print(f'HSCフットプリント内・z範囲内: {len(df_hsc)}件')
print()
print(df_hsc[['hsc_field','Name','RA','Dec','z','MSZ','SNR','r500','r_s','rho_s','MCXC']].to_string(index=False))
df_hsc.to_csv('psz2_hsc_ready.csv', index=False)
print()
print('保存: psz2_hsc_ready.csv')
print()

# Miyaoka 4クラスターとの比較
print('=== 参考：Miyaoka 4クラスター ===')
miy = [
    ('J0201', 30.652, -4.955, 0.196, 0.676, 4.8e14, 0.169),
    ('J0231', 37.941, -4.762, 0.184, 0.760, 4.2e14, 0.190),
    ('J1311',197.863,  0.403, 0.183, 0.985, 3.5e14, 0.246),
    ('J2337',354.378, -0.022, 0.282, 0.952, 3.6e14, 0.238),
]
print(f'{'名前':8s} {'RA':8s} {'Dec':7s} {'z':6s} {'r500':6s} {'rho_s':10s} {'r_s':6s}')
for m in miy:
    print(f'{m[0]:8s} {m[1]:8.3f} {m[2]:7.3f} {m[3]:6.3f} {m[4]:6.3f} {m[5]:10.2e} {m[6]:6.3f}')
