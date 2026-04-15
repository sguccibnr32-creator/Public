# PSZ2 既知データ（Planck 2015 + T-8段階2別紙から）
# + MCXC照合済み8件の座標を確定
import pandas as pd

PSZ2_KNOWN = [
    # T-8段階2別紙に記載済み（RA/Dec/z は SIMBAD/NED から）
    {'id':'PSZ2_076','name':'Abell 2631',  'RA':354.378,'Dec':-0.022,'z':0.273,'M500':5.42,'Tx':7.06},
    {'id':'PSZ2_101','name':'Abell 1689',  'RA':197.873,'Dec':-1.341,'z':0.183,'M500':8.43,'Tx':10.1},
    {'id':'PSZ2_104','name':'MACS J1115',  'RA':168.966,'Dec': 1.499,'z':0.352,'M500':4.53,'Tx':6.78},
    {'id':'PSZ2_114','name':'ACO 1443',    'RA':180.087,'Dec': 23.40,'z':0.268,'M500':3.89,'Tx':6.0},
    {'id':'PSZ2_162','name':'J0214.6-0433','RA': 33.652,'Dec':-4.558,'z':0.450,'M500':1.79,'Tx':2.8},
    {'id':'PSZ2_170','name':'J0231.7-0451','RA': 37.941,'Dec':-4.858,'z':0.184,'M500':2.21,'Tx':3.9},
    {'id':'PSZ2_240','name':'ZwCl 5247',   'RA':197.260,'Dec': 9.760,'z':0.229,'M500':3.77,'Tx':5.1},
    {'id':'PSZ2_271','name':'ACO 1437',    'RA':180.087,'Dec': 3.500,'z':0.134,'M500':4.12,'Tx':5.5},
]

df = pd.DataFrame(PSZ2_KNOWN)
print('=== 既知 PSZ2 クラスター（8件）===')
print(df.to_string(index=False))
df.to_csv('psz2_known.csv', index=False)
print()
print('HSC-SSP Y3 フットプリント確認（RA/Dec範囲）:')
print('Field1(XMM-LSS): RA=29~41, Dec=-7~-2')
print('Field2(VVDS):    RA=330~345, Dec=-2~4')
print('Field3(GAMA15):  RA=209~223, Dec=-2~5')
print('Field4(WIDE12H): RA=175~193, Dec=-2~5')
print('Field5(GAMA09):  RA=128~142, Dec=-2~3')
print()
print('=== HSC-SSPフットプリント内判定 ===')
hsc_fields = [
    (35.0, -4.5, 6.0, 5.0, 'XMM-LSS'),
    (337.5, 0.5, 7.5, 3.0, 'VVDS'),
    (216.0, 0.5, 7.0, 3.5, 'GAMA15'),
    (181.5, 0.5, 6.5, 3.5, 'WIDE12H'),
    (135.0, 0.0, 7.0, 2.5, 'GAMA09'),
]
import numpy as np
for _, row in df.iterrows():
    in_field = False
    for ra_c, dec_c, dra, ddec, fn in hsc_fields:
        if (abs(row['RA']-ra_c) < dra and abs(row['Dec']-dec_c) < ddec):
            print(row['id'], row['name'], '-> HSCフィールド:', fn)
            in_field = True
    if not in_field:
        print(row['id'], row['name'], '-> HSC範囲外')
