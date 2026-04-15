from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd

Vizier.ROW_LIMIT = 500
v = Vizier(columns=['PSZ2','RAJ2000','DEJ2000','Redshift','MSZ'])

centers = [
    ('XMM-LSS', 35.0, -4.5),
    ('GAMA09',  135.0, 0.0),
    ('GAMA15',  215.0, 0.5),
    ('VVDS',    337.0, 0.5),
    ('WIDE12H', 180.0, 0.0),
]

all_rows = []
for name, ra, dec in centers:
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    try:
        result = v.query_region(c, radius=8*u.deg, catalog='B/planck/plancksz2')
        if result and len(result) > 0:
            t = result[0]
            print(name, len(t), '件')
            for row in t:
                all_rows.append({col: str(row[col]) for col in t.colnames})
        else:
            print(name, '0件')
    except Exception as e:
        print(name, 'エラー:', e)

if all_rows:
    df = pd.DataFrame(all_rows).drop_duplicates(subset=['PSZ2'])
    print()
    print(df[['PSZ2','RAJ2000','DEJ2000','Redshift','MSZ']].to_string())
    df.to_csv('psz2_result.csv', index=False)
    print('保存:', len(df), '件')
else:
    print('取得なし - カタログ名を確認します')
    try:
        cats = Vizier.find_catalogs('Planck SZ2')
        for k,v2 in list(cats.items())[:5]:
            print(k, v2.description[:60])
    except Exception as e:
        print('カタログ検索失敗:', e)
