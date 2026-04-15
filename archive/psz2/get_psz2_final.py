import urllib.request, re, pandas as pd, numpy as np

BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# HSC-SSPフィールド中心・半径
fields = [
    ('XMM-LSS', 35.0, -4.5, 480),
    ('VVDS',   337.0,  1.5, 480),
    ('WIDE12H',182.0,  1.5, 600),
    ('GAMA15', 216.0,  1.5, 480),
    ('GAMA09', 135.0,  0.5, 480),
]

all_rows = []
for fname, ra, dec, r_arcmin in fields:
    url = (BASE +
        '?-source=J/A+A/594/A27/psz2'
        '&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=z&-out=MSZ&-out=SNR&-out=MCXC'
        f'&-c={ra}+{dec}&-c.rm={r_arcmin}'
        '&-out.max=200')
    print(f'検索: {fname} (r={r_arcmin}arcmin)', end=' ')
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        n = 0
        for row in rows:
            tds = re.findall(r'<TD>(.*?)</TD>', row)
            if len(tds) >= 4:
                all_rows.append({
                    'field': fname,
                    'Name':  tds[0].strip(),
                    'RA':    tds[1].strip(),
                    'Dec':   tds[2].strip(),
                    'z':     tds[3].strip(),
                    'MSZ':   tds[4].strip() if len(tds)>4 else '',
                    'SNR':   tds[5].strip() if len(tds)>5 else '',
                    'MCXC':  tds[6].strip() if len(tds)>6 else '',
                })
                n += 1
        print(f'-> {n}件')
    except Exception as e:
        print(f'-> エラー: {e}')

if not all_rows:
    print('0件 - 半径を広げて再試行')
    # 半径を2倍に
    for fname, ra, dec, r_arcmin in fields:
        url = (BASE +
            '?-source=J/A+A/594/A27/psz2'
            '&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=z&-out=MSZ&-out=SNR'
            f'&-c={ra}+{dec}&-c.rm={r_arcmin*2}'
            '&-out.max=200')
        print(f'再検索: {fname} (r={r_arcmin*2}arcmin)', end=' ')
        try:
            data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
            rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
            n = 0
            for row in rows:
                tds = re.findall(r'<TD>(.*?)</TD>', row)
                if len(tds) >= 4:
                    all_rows.append({
                        'field': fname, 'Name': tds[0].strip(),
                        'RA': tds[1].strip(), 'Dec': tds[2].strip(),
                        'z':  tds[3].strip(), 'MSZ': tds[4].strip() if len(tds)>4 else '',
                        'SNR': tds[5].strip() if len(tds)>5 else '',
                    })
                    n += 1
            print(f'-> {n}件')
        except Exception as e:
            print(f'-> エラー: {e}')

if all_rows:
    df = pd.DataFrame(all_rows).drop_duplicates('Name')
    df['RA'] = pd.to_numeric(df['RA'], errors='coerce')
    df['Dec'] = pd.to_numeric(df['Dec'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df['MSZ'] = pd.to_numeric(df['MSZ'], errors='coerce')
    df['SNR'] = pd.to_numeric(df['SNR'], errors='coerce')

    # z>0のみ（redshiftが測定済み）
    df_z = df[df['z'] > 0].copy()
    print()
    print(f'=== 取得結果（z測定済み: {len(df_z)}/{len(df)}件）===')
    print(df_z[['field','Name','RA','Dec','z','MSZ','SNR','MCXC']].to_string(index=False))
    df_z.to_csv('psz2_in_hsc.csv', index=False)
    print(f'\n保存: psz2_in_hsc.csv')
else:
    print('取得なし - raw responseを確認します')
    url = (BASE + '?-source=J/A+A/594/A27/psz2'
           '&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=z'
           '&-c=35.0+-4.5&-c.rm=480&-out.max=5')
    data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
    print(data[:2000])
