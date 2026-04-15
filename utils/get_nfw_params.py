import urllib.request, re, pandas as pd

BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# 対象クラスター（MCXC名で検索）
TARGETS = {
    'G167': {'mcxc':'J0214.6-0433', 'ra':33.684, 'dec':-4.586},
    'G273': {'mcxc':'J1200.4+0320', 'ra':180.109,'dec': 3.348},
    'G286': {'mcxc':'',             'ra':185.703,'dec': 2.116},
    'G223': {'mcxc':'',             'ra':131.371,'dec': 3.475},
    'G228': {'mcxc':'',             'ra':140.561,'dec': 3.753},
    'G231': {'mcxc':'',             'ra':139.058,'dec':-0.412},
}

results = {}

# 1) MCXC カタログ（J/A+A/534/A109）から M500・r500 取得
print('=== MCXC から M500・r500 取得 ===')
for cn, info in TARGETS.items():
    url = (BASE +
        '?-source=J/A+A/534/A109/mcxc'
        '&-out=MCXC&-out=RAJ2000&-out=DEJ2000&-out=z&-out=M500&-out=R500'
        f'&-c={info["ra"]}+{info["dec"]}&-c.rm=15'
        '&-out.max=5')
    try:
        data = urllib.request.urlopen(url, timeout=20).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        for row in rows:
            tds = re.findall(r'<TD>(.*?)</TD>', row)
            if len(tds) >= 5:
                results[cn] = {
                    'MCXC': tds[0].strip(), 'RA': tds[1].strip(),
                    'Dec':  tds[2].strip(), 'z':  tds[3].strip(),
                    'M500': tds[4].strip(), 'R500': tds[5].strip() if len(tds)>5 else '',
                }
                print(f'{cn}: {tds[0].strip()} M500={tds[4].strip()} R500={tds[5].strip() if len(tds)>5 else "?"}')
                break
        else:
            print(f'{cn}: MCXC未収録')
    except Exception as e:
        print(f'{cn}: エラー {e}')

# 2) 座標検索でより広い半径（30arcmin）で再試行
print()
print('=== 座標検索（r=30arcmin）===')
for cn, info in TARGETS.items():
    if cn in results: continue
    url = (BASE +
        '?-source=J/A+A/534/A109/mcxc'
        '&-out=MCXC&-out=RAJ2000&-out=DEJ2000&-out=z&-out=M500&-out=R500'
        f'&-c={info["ra"]}+{info["dec"]}&-c.rm=30'
        '&-out.max=3')
    try:
        data = urllib.request.urlopen(url, timeout=20).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        if rows:
            for row in rows:
                tds = re.findall(r'<TD>(.*?)</TD>', row)
                if len(tds) >= 4:
                    print(f'{cn}: 近傍 {tds[0].strip()} M500={tds[4].strip() if len(tds)>4 else "?"}')
        else:
            print(f'{cn}: 未収録')
    except Exception as e:
        print(f'{cn}: エラー {e}')

# 3) LoCuSS 弱レンズカタログ（Okabe+2016 J/MNRAS/461/3794）
print()
print('=== LoCuSS NFW fit（Okabe+2016）===')
for cn, info in TARGETS.items():
    url = (BASE +
        '?-source=J/MNRAS/461/3794/table1'
        '&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=z&-out=M200&-out=rs'
        f'&-c={info["ra"]}+{info["dec"]}&-c.rm=20'
        '&-out.max=3')
    try:
        data = urllib.request.urlopen(url, timeout=20).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        if rows:
            for row in rows:
                tds = re.findall(r'<TD>(.*?)</TD>', row)
                if len(tds) >= 4:
                    print(f'{cn}: LoCuSS {tds[0].strip()} M200={tds[4].strip() if len(tds)>4 else "?"}')
        else:
            print(f'{cn}: LoCuSS未収録')
    except Exception as e:
        print(f'{cn}: エラー {e}')

# 4) HSC-SSP 公開弱レンズカタログ（Umetsu+2020）
print()
print('=== Umetsu+2020（HSC弱レンズ）===')
for cn, info in TARGETS.items():
    url = (BASE +
        '?-source=J/ApJS/247/43/table1'
        '&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=z&-out=M200c&-out=rs'
        f'&-c={info["ra"]}+{info["dec"]}&-c.rm=20'
        '&-out.max=3')
    try:
        data = urllib.request.urlopen(url, timeout=20).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        if rows:
            for row in rows:
                tds = re.findall(r'<TD>(.*?)</TD>', row)
                if len(tds) >= 3:
                    print(f'{cn}: Umetsu {tds[0].strip()}')
        else:
            print(f'{cn}: Umetsu未収録')
    except Exception as e:
        print(f'{cn}: エラー {e}')
