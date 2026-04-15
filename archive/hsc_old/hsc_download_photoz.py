#!/usr/bin/env python3
"""
HSC-SSP PDR3 データダウンロード (Web form API方式)
"""
import re,time,sys,os,io
from pathlib import Path
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

try:
    import requests
except ImportError:
    print("uv run --with requests python hsc_download_photoz.py"); sys.exit(1)

base_dir=os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR=Path(base_dir)
BASE_URL="https://hsc-release.mtk.nao.ac.jp"
AUTH=("sgucci@local","tMqhusKXiPmqKW1nlh0fa0VFm3JtaHbLCLwVuU17")

CLUSTERS=[
    {"name":"cl1","ra":140.45,"dec":-0.25,"label":"GAMA09H #1 (sgm=6.4)"},
    {"name":"cl2","ra":142.21,"dec":-0.12,"label":"GAMA09H #2 (S/N=8.0)"},
    {"name":"cl3","ra":140.30,"dec":-0.35,"label":"GAMA09H #3 (S/N=7.2)"},
    {"name":"cl4","ra":182.68,"dec":-0.28,"label":"WIDE12H (S/N=6.8)"},
    {"name":"cl5","ra":216.75,"dec":-0.20,"label":"GAMA15H (S/N=6.4)"},
    {"name":"cl6","ra":335.40,"dec":-0.95,"label":"VVDS (sgm=5.9)"},
    {"name":"cl7","ra":342.43,"dec":-0.57,"label":"VVDS #2 (sgm=5.2)"},
    {"name":"cl8","ra":139.00,"dec":-0.41,"label":"GAMA09H #4 (S/N=6.2)"},
]

class HSC_API:
    def __init__(self):
        self.s=requests.Session()
        self._get_csrf()

    def _get_csrf(self):
        r=self.s.get(f"{BASE_URL}/datasearch/",auth=AUTH)
        m=re.search(r'csrf-token.*?content="([^"]+)"',r.text)
        self.csrf=m.group(1) if m else ""
        uid=re.search(r'login_user_id=(\d+)',r.text)
        if uid: print(f"  ログイン成功 (user_id={uid.group(1)})")
        else: print("  ログイン失敗？")

    def submit(self,sql,release="pdr3"):
        """ジョブ投入 -> ジョブ一覧からIDを返す"""
        headers={"X-CSRF-Token":self.csrf,"X-Requested-With":"XMLHttpRequest"}
        data={
            "catalog_job[sql]":sql,
            "catalog_job[out_format]":"csv",
            "catalog_job[include_metainfo]":"0",
            "catalog_job[release_version]":release,
        }
        r=self.s.post(f"{BASE_URL}/datasearch/catalog_jobs",
                     data=data,headers=headers,auth=AUTH,allow_redirects=False)
        if r.status_code in [302,303]:
            print(f"    ジョブ投入成功 (302)")
            # ジョブ一覧ページからIDを取得
            time.sleep(1)
            return self._get_latest_job_id()
        else:
            print(f"    投入失敗: {r.status_code}")
            return None

    def _get_latest_job_id(self):
        """ジョブ一覧ページから最新ジョブIDを取得"""
        r=self.s.get(f"{BASE_URL}/datasearch/catalog_jobs",auth=AUTH)
        # download_key または job ID を探す
        # パターン: /datasearch/catalog_jobs/download/XXXX
        keys=re.findall(r'/datasearch/catalog_jobs/download/([a-f0-9]+)',r.text)
        if keys:
            print(f"    最新ジョブキー: {keys[0][:16]}...")
            return keys[0]
        # hiddenフォームからジョブIDを探す
        ids=re.findall(r'catalog_jobs/(\d+)/(?:hide|cancel)',r.text)
        if ids:
            print(f"    最新ジョブID: {ids[0]}")
            return ids[0]
        # status APIで確認
        r2=self.s.get(f"{BASE_URL}/datasearch/api/catalog_jobs/status.json",
                     auth=AUTH,headers={"Accept":"application/json"})
        if r2.status_code==200:
            try:
                jobs=r2.json()
                print(f"    status API: {str(jobs)[:200]}")
                return jobs
            except: pass
        print("    ジョブID取得失敗")
        return None

    def wait_and_download(self,job_key,output_path,max_wait=300):
        """ジョブ完了待ち+ダウンロード"""
        if job_key is None: return False

        # download_keyの場合は直接ダウンロード試行
        url=f"{BASE_URL}/datasearch/catalog_jobs/download/{job_key}"
        elapsed=0
        while elapsed<max_wait:
            r=self.s.get(url,auth=AUTH,stream=True,allow_redirects=True)
            ct=r.headers.get('Content-Type','')
            if 'text/csv' in ct or 'application/octet-stream' in ct or 'text/plain' in ct:
                with open(output_path,'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                sz=os.path.getsize(output_path)/1024
                print(f"    ダウンロード完了: {output_path.name} ({sz:.0f} KB)")
                return True
            elif 'text/html' in ct:
                # まだ実行中かもしれない
                if 'running' in r.text.lower() or 'queued' in r.text.lower():
                    time.sleep(5); elapsed+=5
                    if elapsed%30==0: print(f"    待機中... ({elapsed}秒)")
                    continue
                # 完了ページかもしれない - download_keyを再取得
                keys=re.findall(r'/datasearch/catalog_jobs/download/([a-f0-9]+)',r.text)
                if keys and keys[0]!=job_key:
                    job_key=keys[0]; continue
                # エラーチェック
                if 'error' in r.text.lower()[:500]:
                    print(f"    エラー検出"); return False
                time.sleep(5); elapsed+=5
                if elapsed%30==0: print(f"    待機中... ({elapsed}秒)")
            else:
                print(f"    不明なContent-Type: {ct}")
                time.sleep(5); elapsed+=5

        print(f"    タイムアウト ({max_wait}秒)"); return False

    def query(self,sql,output_path,release="pdr3"):
        """投入->待機->ダウンロード"""
        print(f"    SQL投入中...")
        key=self.submit(sql,release)
        if key is None: return False
        return self.wait_and_download(key,output_path)


def photoz_query(ra,dec,radius_arcsec=300):
    return f"""
SELECT
    object_id, ra, dec,
    photoz_best, photoz_err68_min, photoz_err68_max,
    i_cmodel_mag, i_extendedness_value
FROM
    pdr3_wide.forced
    LEFT JOIN pdr3_wide.photoz_mizuki USING (object_id)
WHERE
    cone_search(coord, {ra}, {dec}, {radius_arcsec})
    AND i_cmodel_mag < 24.5
    AND photoz_best IS NOT NULL
    AND photoz_best BETWEEN 0.01 AND 2.0
    AND i_extendedness_value = 1
"""

# ====================================================================
print("="*70)
print("HSC-SSP PDR3 データダウンロード")
print("="*70)

api=HSC_API()

# Phase 1: クラスター photo-z
print(f"\n{'='*70}")
print("[Phase 1] クラスター候補周辺 photo-z")
print("="*70)

for cl in CLUSTERS:
    outpath=OUTPUT_DIR/f"hsc_photoz_{cl['name']}.csv"
    print(f"\n  --- {cl['label']} ---")
    if outpath.exists():
        sz=outpath.stat().st_size/1024
        print(f"    既存 ({sz:.0f} KB) -> スキップ"); continue
    sql=photoz_query(cl['ra'],cl['dec'])
    success=api.query(sql,outpath)
    if not success:
        print(f"    失敗。次へ。")
    time.sleep(3)  # API負荷軽減

# 結果確認
print(f"\n{'='*70}")
print("[結果確認]")
print("="*70)

for f in sorted(OUTPUT_DIR.glob("hsc_photoz_*.csv")):
    sz=f.stat().st_size/1024
    try:
        n=sum(1 for _ in open(f,encoding='utf-8'))-1
        print(f"  {f.name}: {n:,} 行, {sz:.0f} KB")
    except:
        print(f"  {f.name}: {sz:.0f} KB")

print("\n完了。")
