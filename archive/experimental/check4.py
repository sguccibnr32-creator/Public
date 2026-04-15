import pandas as pd
import numpy as np

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'

COLS = [
    'object_id','ra','dec',
    'e1','e2',
    'sigma_e','rms_e','weight',
    'shear_m','shear_c1','shear_c2',
    'resolution','mag_i',
    'zbin','b_mode_mask',
    'ra2','dec2','e1_2','e2_2'
]

df = pd.read_csv(PATH, comment='#', header=None, names=COLS, nrows=50000)

print('=== 基本統計 ===')
print('行数(先頭50000):', len(df))

print()
print('=== resolution ===')
r = df['resolution']
print('mean=', round(r.mean(),3))
print('<0.3の割合=', round((r<0.3).mean()*100,1), '%')

print()
print('=== shear bias m ===')
m = df['shear_m']
print('mean=', round(m.mean(),4))
print('std=', round(m.std(),4))
print('min=', round(m.min(),4), ' max=', round(m.max(),4))

print()
print('=== additive bias ===')
print('c1 mean=', round(df['shear_c1'].mean(),5))
print('c2 mean=', round(df['shear_c2'].mean(),5))

print()
print('=== weight ===')
w = df['weight']
print('mean=', round(w.mean(),3), ' min=', round(w.min(),3), ' max=', round(w.max(),3))
print('weight=0の割合:', round((w==0).mean()*100,1), '%')

print()
print('=== zbin分布 ===')
print(df['zbin'].value_counts().sort_index())

print()
print('=== b_mode_mask分布 ===')
print(df['b_mode_mask'].value_counts())

print()
print('=== RA/Dec範囲 ===')
print('RA:', round(df['ra'].min(),3), '~', round(df['ra'].max(),3))
print('Dec:', round(df['dec'].min(),3), '~', round(df['dec'].max(),3))

print()
print('=== 総行数カウント ===')
df_all = pd.read_csv(PATH, comment='#', header=None, names=COLS)
print('総行数:', len(df_all))
print('RA範囲(全体):', round(df_all['ra'].min(),3), '~', round(df_all['ra'].max(),3))
