import urllib.request, re
import pandas as pd

# PSZ2カタログ（Planck 2015 SZ）からHSC-SSPフットプリント内のクラスターを取得
# CDS HTTP経由
BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# PSZ2 全カタログをRA/Dec範囲で抽出
# HSC-SSPフットプリント: 複数フィールド
# RA~30, Dec~-5 / RA~38, Dec~-5 / RA~198, Dec~0 / RA~214, Dec~0 / RA~354, Dec~0

fields = [
    {'name':'XMM-LSS', 'ra':35.0, 'dec':-4.5, 'r':8.0},
    {'name':'GAMA09',  'ra':135.0,'dec': 0.0,  'r':8.0},
    {'name':'GAMA15',  'ra':215.0,'dec': 0.5,  'r':8.0},
    {'name':'VVDS',    'ra':337.0,'dec': 0.5,  'r':8.0},
    {'name':'WIDE12H', 'ra':180.0,'dec': 0.0,  'r':8.0},
]

results = []

for f in fields:
    ra_min = f['ra'] - f['r']
    ra_max = f['ra'] + f['r']
    dec_min = f['dec'] - f['r']
    dec_max = f['dec'] + f['r']

    url = (BASE +
           '?-source=B/planck/plancksz2&-out.max=200' +
           '&-out=PSZ2&-out=RA&-out=Dec&-out=Redshift&-out=MSZ' +
           f'&RA={f["ra"]}&DEC={f["dec"]}&-c.rm={f["r"]*60}')

    print(f'検索中: {f["name"]} RA={f["ra"]} Dec={f["dec"]}')
    try:
        req = urllib.request.urlopen(url, timeout=30)
        data = req.read().decode('utf-8','ignore')

        # TR行を抽出
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        for row in rows:
            tds = re.findall(r'<TD>(.*?)</TD>', row)
            if len(tds) >= 4:
                results.append({
                    'field': f['name'],
                    'PSZ2': tds[0].strip(),
                    'RA':   tds[1].strip(),
                    'Dec':  tds[2].strip(),
                    'z':    tds[3].strip(),
                    'MSZ':  tds[4].strip() if len(tds)>4 else '',
                })
        print(f'  -> {len(rows)} 件')
    except Exception as e:
        print(f'  -> エラー: {e}')

if results:
    df = pd.DataFrame(results)
    df.to_csv('psz2_hsc_fields.csv', index=False)
    print()
    print('=== 取得結果 ===')
    print(df.to_string())
    print(f'\n保存: psz2_hsc_fields.csv ({len(df)}件)')
else:
    print('取得できませんでした')
    print()
    print('=== 代替: 既知の17クラスター確認用 ===')
    # 前セッションで使用済みの8件を確認
    known = [
        {'PSZ2':'G114.99-34.22','name':'Abell 2631','RA':354.378,'Dec':-0.022,'z':0.273,'M500':5.42},
        {'PSZ2':'G236.92-26.65','name':'Abell 1689', 'RA':197.873,'Dec':-1.341,'z':0.183,'M500':8.43},
        {'PSZ2':'G271.18-30.95','name':'MACS J1115','RA':168.966,'Dec': 1.499,'z':0.352,'M500':4.53},
    ]
    for k in known:
        print(k)
