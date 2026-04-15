import urllib.request, json, re
from astropy.time import Time

# HEASARC TAP サービスで CHANMASTER テーブルから直接取得
def get_chandra_meta(obsid):
    # TAP クエリ
    query = f"SELECT obsid,name,ra,dec,grating,instrument,exposure,time FROM chanmaster WHERE obsid={obsid}"
    url = (f"https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync?"
           f"REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY={urllib.request.quote(query)}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Python/3'})
        data = urllib.request.urlopen(req, timeout=30).read()
        js = json.loads(data)
        cols = [c['name'] for c in js.get('metadata',[])]
        rows = js.get('data',[])
        if rows:
            return dict(zip(cols, rows[0]))
    except Exception as e:
        return {'error': str(e)[:80]}
    return {}

print('=== Chandra CHANMASTER メタデータ（TAP経由）===')
for obsid in ['26729','29163','11768']:
    meta = get_chandra_meta(obsid)
    if 'error' in meta:
        print(f'ObsID {obsid}: {meta["error"]}')
    else:
        mjd = float(meta.get('time',0)) if meta.get('time') else 0
        date = Time(mjd,format='mjd').iso[:10] if mjd>0 else '?'
        print(f'ObsID {obsid}:')
        print(f'  Target Name: {meta.get("name","?")}')
        print(f'  RA/Dec: {meta.get("ra","?")} / {meta.get("dec","?")}')
        print(f'  Instrument: {meta.get("instrument","?")}')
        print(f'  Grating: {meta.get("grating","?")}')
        print(f'  Exposure: {meta.get("exposure","?")}s')
        print(f'  Date: {date}')
    print()

print()
print('=== XMM CHANMASTER 対応（TAP）===')
def get_xmm_meta(obsid):
    query = f"SELECT obsid,target_name,ra,dec,duration,date_obs FROM xmmmaster WHERE obsid='{obsid}'"
    url = (f"https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync?"
           f"REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY={urllib.request.quote(query)}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Python/3'})
        data = urllib.request.urlopen(req, timeout=30).read()
        js = json.loads(data)
        cols = [c['name'] for c in js.get('metadata',[])]
        rows = js.get('data',[])
        if rows: return dict(zip(cols, rows[0]))
    except Exception as e:
        return {'error': str(e)[:80]}
    return {}

for obsid in ['0650381801','0804410101']:
    meta = get_xmm_meta(obsid)
    if 'error' in meta:
        print(f'XMM {obsid}: {meta["error"]}')
    elif meta:
        print(f'XMM {obsid}: target={meta.get("target_name","?")} '
              f'RA={meta.get("ra","?")} date={str(meta.get("date_obs","?"))[:10]}')
    else:
        print(f'XMM {obsid}: データなし')
