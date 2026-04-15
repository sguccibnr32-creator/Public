import urllib.request, re, pandas as pd

BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# HSC-SSPフィールド中心座標で検索
fields = [
    ('XMM-LSS', 35.0, -4.5, 480),
    ('VVDS',   337.0,  1.5, 480),
    ('WIDE12H',182.0,  1.5, 600),
    ('GAMA15', 216.0,  1.5, 480),
    ('GAMA09', 135.0,  0.5, 480),
]

results = []
for fname, ra, dec, r_arcmin in fields:
    url = (BASE +
        f'?-source=B/planck/plancksz2'
        f'&-c={ra}+{dec}&-c.rm={r_arcmin}'
        f'&-out=PSZ2&-out=RAJ2000&-out=DEJ2000&-out=Redshift&-out=MSZ'
        f'&-out.max=200')
    print(f'検索: {fname}', end=' ')
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        for row in rows:
            tds = re.findall(r'<TD>(.*?)</TD>', row)
            if len(tds) >= 4:
                results.append({'field':fname,'PSZ2':tds[0].strip(),
                    'RA':tds[1].strip(),'Dec':tds[2].strip(),
                    'z':tds[3].strip(),'MSZ':tds[4].strip() if len(tds)>4 else ''})
        print(f'-> {len(rows)}件')
    except Exception as e:
        print(f'-> エラー: {e}')

if results:
    df = pd.DataFrame(results).drop_duplicates('PSZ2')
    print()
    print(df.to_string(index=False))
    df.to_csv('psz2_in_hsc.csv', index=False)
    print(f'\n保存: psz2_in_hsc.csv ({len(df)}件)')
else:
    print('取得なし')
