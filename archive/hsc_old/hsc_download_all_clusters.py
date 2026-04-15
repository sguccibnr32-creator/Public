#!/usr/bin/env python3
"""全クラスターの photo-z 一括ダウンロード"""
import requests, re, sys, io, time, os
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

base_dir=os.path.dirname(os.path.abspath(__file__))
base="https://hsc-release.mtk.nao.ac.jp"
auth=("sgucci@local","tMqhusKXiPmqKW1nlh0fa0VFm3JtaHbLCLwVuU17")

CLUSTERS=[
    ("cl1",140.45,-0.25),("cl2",142.21,-0.12),("cl3",140.30,-0.35),
    ("cl4",182.68,-0.28),("cl5",216.75,-0.20),("cl6",335.40,-0.95),
    ("cl7",342.43,-0.57),("cl8",139.00,-0.41),
]

s=requests.Session()
r0=s.get(f"{base}/datasearch/",auth=auth)
csrf=re.search(r'csrf-token.*?content="([^"]+)"',r0.text).group(1)
print(f"ログイン OK, CSRF取得")

def submit(sql):
    headers={"X-CSRF-Token":csrf,"X-Requested-With":"XMLHttpRequest"}
    data={"catalog_job[sql]":sql,"catalog_job[out_format]":"csv",
          "catalog_job[include_metainfo]":"0","catalog_job[release_version]":"pdr3"}
    r=s.post(f"{base}/datasearch/catalog_jobs",data=data,headers=headers,auth=auth,allow_redirects=False)
    return r.status_code in [302,303]

def wait_latest(max_wait=900):
    """最新ジョブの完了待ち -> download_key を返す"""
    for i in range(max_wait//10):
        r=s.get(f"{base}/datasearch/catalog_jobs",auth=auth)
        tbody=re.search(r'<tbody>(.*?)</tbody>',r.text,re.DOTALL)
        if not tbody: time.sleep(10); continue
        first=re.search(r'<tr[^>]*>(.*?)</tr>',tbody.group(1),re.DOTALL)
        if not first: time.sleep(10); continue
        row=first.group(1)
        if 'done' in row.lower():
            dk=re.search(r'download/([a-f0-9]+)',row)
            return dk.group(1) if dk else None
        elif 'error' in row.lower():
            return "ERROR"
        if i%6==0: print(f"    待機中... ({i*10}秒)")
        time.sleep(10)
    return None

def download(key,outpath):
    r=s.get(f"{base}/datasearch/catalog_jobs/download/{key}",auth=auth)
    with open(outpath,"w",encoding="utf-8") as f: f.write(r.text)
    lines=r.text.strip().split('\n')
    return len(lines)-1  # ヘッダー除く

radius=0.083  # 5 arcmin in deg
for name,ra,dec in CLUSTERS:
    outpath=os.path.join(base_dir,f"hsc_photoz_{name}.csv")
    if os.path.exists(outpath) and os.path.getsize(outpath)>1000:
        print(f"\n{name}: 既存 ({os.path.getsize(outpath)//1024} KB) -> スキップ")
        continue

    sql=f"""
SELECT f.object_id, f.ra, f.dec,
       m.photoz_best, m.photoz_err68_min, m.photoz_err68_max,
       m.stellar_mass, f.i_cmodel_mag, f.i_extendedness_value
FROM pdr3_wide.forced f
LEFT JOIN pdr3_wide.photoz_mizuki m ON f.object_id = m.object_id
WHERE f.ra BETWEEN {ra-radius/max(0.1,abs(dec)*0.017+0.01)} AND {ra+radius/max(0.1,abs(dec)*0.017+0.01)}
  AND f.dec BETWEEN {dec-radius} AND {dec+radius}
  AND f.i_cmodel_mag < 24.5
  AND f.isprimary = true
  AND m.photoz_best BETWEEN 0.01 AND 2.0
"""
    print(f"\n{name}: RA={ra}, Dec={dec}")
    print(f"  投入中...")
    if not submit(sql):
        print(f"  投入失敗"); continue

    print(f"  実行待ち...")
    key=wait_latest()
    if key is None:
        print(f"  タイムアウト"); continue
    if key=="ERROR":
        print(f"  エラー"); continue

    nrows=download(key,outpath)
    print(f"  完了: {nrows} 行 -> {outpath}")
    time.sleep(3)

print("\n全クラスター処理完了。")
