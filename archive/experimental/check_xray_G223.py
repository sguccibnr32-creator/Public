import urllib.request, re, pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

TARGETS = {
    'G223': {'ra':131.371, 'dec': 3.475, 'z':0.327, 'MSZ':5.272e14},
    'G228': {'ra':140.561, 'dec': 3.753, 'z':0.270, 'MSZ':5.782e14},
    'G231': {'ra':139.058, 'dec':-0.412, 'z':0.332, 'MSZ':4.868e14},
}

def vizier_query(source, ra, dec, r_am, cols):
    col_str = ''.join(f'&-out={c}' for c in cols)
    url = f'{BASE}?-source={source}{col_str}&-c={ra}+{dec}&-c.rm={r_am}&-out.max=5'
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
        rows = re.findall(r'<TR>(.*?)</TR>', data, re.DOTALL)
        results = []
        for row in rows:
            tds = re.findall(r'<TD>(.*?)</TD>', row)
            if len(tds) >= 2:
                results.append({c: tds[i].strip() if i<len(tds) else '' for i,c in enumerate(cols)})
        return results
    except Exception as e:
        return [{'error': str(e)[:60]}]

CATALOGS = [
    ('J/A+A/641/A135/table1', 'Lovisari+2020 PSZ2xChandra',
     ['PSZ2','RAJ2000','DEJ2000','z','kT','M500'], 15),
    ('J/MNRAS/423/1024/xcs-dr1', 'XCS DR1 (Mehrtens+2012)',
     ['XCSJ','RAJ2000','DEJ2000','zsp','TX','M500'], 15),
    ('J/A+A/534/A109/mcxc', 'MCXC (Piffaretti+2011)',
     ['MCXC','RAJ2000','DEJ2000','z','M500','R500'], 10),
    ('J/ApJS/182/12/table1', 'ACCEPT (Cavagnolo+2009)',
     ['ObsID','Name','RAJ2000','DEJ2000','z','__kT_'], 12),
    ('VII/84/table', 'ACO (Abell+1989)',
     ['ACO','RAJ2000','DEJ2000','z'], 15),
    ('J/A+A/432/381/table2', 'REFLEX II',
     ['RXCJ','RAJ2000','DEJ2000','z','LX','M500'], 15),
]

print('=== X線・光学カタログ照合 ===')
for cn, tgt in TARGETS.items():
    print(f'\n--- {cn} (RA={tgt["ra"]:.3f}, Dec={tgt["dec"]:.3f}, z={tgt["z"]:.3f}) ---')
    found = False
    for cat_id, cat_name, cols, r_am in CATALOGS:
        results = vizier_query(cat_id, tgt['ra'], tgt['dec'], r_am, cols)
        if results and 'error' not in results[0]:
            print(f'  [{cat_name}]: {len(results)}件')
            for r in results[:2]: print(f'    {r}')
            found = True
        elif results and 'error' in results[0]:
            print(f'  [{cat_name}]: エラー {results[0]["error"]}')
        else:
            print(f'  [{cat_name}]: 未収録')
    if not found:
        print('  -> 全カタログ未収録 → X線観測なしの可能性')

print()
print('=== HEASARC Chandra/XMM 観測確認 ===')
try:
    from astroquery.heasarc import Heasarc
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    h = Heasarc()
    for cn, tgt in TARGETS.items():
        coord = SkyCoord(ra=tgt['ra']*u.deg, dec=tgt['dec']*u.deg)
        print(f'\n--- {cn} ---')
        for catalog, label in [('chanmaster','Chandra'),('xmmmaster','XMM')]:
            try:
                t = h.query_region(coord, catalog=catalog, radius='15 arcmin')
                if len(t) > 0:
                    print(f'  {label}: {len(t)}件')
                    for row in t[:3]:
                        cols_try = ['obsid','obs_id','time','date_obs','exposure','pn_filter']
                        info = {c: str(row[c]) for c in cols_try if c in t.colnames}
                        print(f'    {info}')
                else:
                    print(f'  {label}: 観測なし')
            except Exception as e:
                print(f'  {label}: {e}')
except ImportError:
    print('astroquery未インストール: pip install astroquery')
    print()
    print('ブラウザで確認するURL:')
    for cn, tgt in TARGETS.items():
        print(f'{cn} Chandra: https://cda.harvard.edu/chaser/?target={tgt["ra"]},{tgt["dec"]}&resolver=none&radius=15&coordtype=equatorial')
        print(f'{cn} XMM: https://nxsa.esac.esa.int/nxsa-web/#search?RA={tgt["ra"]}&DEC={tgt["dec"]}&SR=0.25')
