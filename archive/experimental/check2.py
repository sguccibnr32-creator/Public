import pandas as pd
import numpy as np

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'
df = pd.read_csv(PATH, comment='#', nrows=50000)
df.columns = df.columns.str.strip()

print('=== resolution分布 ===')
r = df['i_hsmshaperegauss_resolution']
print('mean=', round(r.mean(),3), ' <0.3の割合=', round((r<0.3).mean()*100,1), '%')

print('=== shear bias m ===')
m = df['i_hsmshaperegauss_derived_shear_bias_m']
print('mean=', round(m.mean(),4), ' std=', round(m.std(),4))
print('min=', round(m.min(),4), ' max=', round(m.max(),4))

print('=== additive bias c1 c2 ===')
c1 = df['i_hsmshaperegauss_derived_shear_bias_c1']
c2 = df['i_hsmshaperegauss_derived_shear_bias_c2']
print('c1 mean=', round(c1.mean(),5))
print('c2 mean=', round(c2.mean(),5))

print('=== b_mode_mask分布 ===')
print(df['b_mode_mask'].value_counts())

print('=== hsc_y3_zbin分布 ===')
print(df['hsc_y3_zbin'].value_counts().sort_index())

print('=== weight分布 ===')
w = df['i_hsmshaperegauss_derived_weight']
print('mean=', round(w.mean(),3), ' min=', round(w.min(),3), ' max=', round(w.max(),3))
print('weight=0の割合:', round((w==0).mean()*100,1), '%')

print('=== 総行数 ===')
df2 = pd.read_csv(PATH, comment='#')
print('総行数:', len(df2))
