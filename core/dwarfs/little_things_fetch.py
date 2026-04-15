#!/usr/bin/env python3
"""
LITTLE THINGS データ取得・前処理
Oh+2015 (AJ 149, 180) 26銀河の物性+回転曲線
"""
import os,sys,io,json,numpy as np
from pathlib import Path
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

try: import requests
except: print("uv run --with requests ..."); sys.exit(1)

base_dir=os.path.dirname(os.path.abspath(__file__))
OUTDIR=Path(os.path.join(base_dir,"little_things_data"))
OUTDIR.mkdir(exist_ok=True)

a0=1.2e-10; G_SI=6.674e-11; kms=1e3; kpc_m=3.086e19

print("="*70)
print("LITTLE THINGS データ取得・前処理")
print("="*70)

# ====================================================================
# Oh+2015 Table 2 の物理量（論文公開値）
# ====================================================================
galaxies=[
    ("CVnIdwA",3.6,-12.4,0.39,15.0,48,"Im"),
    ("DDO43",7.8,-15.1,1.31,35.0,41,"Im"),
    ("DDO46",6.1,-14.7,0.96,30.0,40,"Im"),
    ("DDO47",5.2,-15.5,1.89,55.0,38,"Im"),
    ("DDO50",3.4,-16.6,1.06,38.0,49,"Im"),
    ("DDO52",10.3,-15.4,1.67,52.0,44,"Im"),
    ("DDO53",3.6,-13.8,0.58,20.0,31,"Im"),
    ("DDO70",1.3,-14.2,0.53,22.0,50,"Im"),
    ("DDO87",7.7,-15.1,1.55,42.0,43,"Im"),
    ("DDO101",6.4,-15.0,0.77,35.0,51,"Im"),
    ("DDO126",4.9,-14.9,0.96,35.0,64,"Im"),
    ("DDO133",3.5,-14.8,1.17,35.0,43,"Im"),
    ("DDO154",3.7,-14.2,0.74,47.0,66,"Im"),
    ("DDO168",4.3,-15.7,0.92,53.0,46,"Im"),
    ("DDO210",0.9,-10.9,0.18,10.0,61,"Im"),
    ("DDO216",1.1,-13.7,0.36,15.0,63,"dIrr"),
    ("F564-V3",8.7,-14.0,1.22,25.0,56,"Im"),
    ("Haro29",5.9,-14.6,0.47,25.0,60,"BCD"),
    ("Haro36",9.3,-16.0,1.12,45.0,72,"Im"),
    ("IC1613",0.7,-15.3,0.75,23.0,48,"IB"),
    ("LeoA",0.8,-12.1,0.29,12.0,58,"IBm"),
    ("NGC1569",3.4,-18.2,0.35,50.0,63,"IBm"),
    ("NGC2366",3.4,-16.8,1.22,50.0,64,"IB"),
    ("NGC3738",4.9,-17.1,0.40,60.0,16,"Im"),
    ("NGC4163",3.0,-14.4,0.34,20.0,32,"Im"),
    ("WLM",1.0,-14.4,0.69,35.0,74,"IB"),
]

print(f"  銀河数: {len(galaxies)}")

# SPARC重複チェック（SPARCに含まれる銀河）
sparc_names=set()
sparc_csv=os.path.join(base_dir,'phase1','sparc_results.csv')
if os.path.exists(sparc_csv):
    import csv
    with open(sparc_csv,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sparc_names.add(row.get('galaxy','').strip().upper())

# 物理量計算
results=[]
for name,dist,MB,Rd,Vf,inc,morph in galaxies:
    vf_si=Vf*kms; hR_si=Rd*kpc_m
    GS=vf_si**2/hR_si
    gc_pred=np.sqrt(a0*GS)
    in_sparc=name.upper() in sparc_names or name.replace('DDO','DDO').upper() in sparc_names
    results.append({
        'name':name,'dist':dist,'MB':MB,'Rd':Rd,'Vflat':Vf,'inc':inc,'morph':morph,
        'G_Sigma0':GS,'gc_pred':gc_pred,'in_sparc':in_sparc,
    })

# サマリー
print(f"\n  {'銀河':<12s} {'Vf':>4s} {'Rd':>5s} {'logGS':>8s} {'gc/a0':>7s} {'SPARC':>6s}")
print(f"  {'-'*46}")
n_indep=0
for r in results:
    sp='YES' if r['in_sparc'] else 'NO'
    if not r['in_sparc']: n_indep+=1
    print(f"  {r['name']:<12s} {r['Vflat']:>4.0f} {r['Rd']:>5.2f} "
          f"{np.log10(r['G_Sigma0']):>8.3f} {r['gc_pred']/a0:>7.3f} {sp:>6s}")

print(f"\n  SPARC外（独立検証用）: {n_indep} 銀河")

# JSON保存
jpath=OUTDIR/"little_things_properties.json"
with open(jpath,"w") as f:
    json.dump([{k:(v if not isinstance(v,bool) else int(v)) for k,v in r.items()} for r in results],f,indent=2)
print(f"  -> {jpath.name}")

# ====================================================================
print(f"\n{'='*70}")
print("[回転曲線データ取得]")
print("="*70)

def fetch_vizier(suffix,fname):
    url=f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/149/180/{suffix}&-out.max=10000"
    print(f"  取得: {url[:80]}...")
    try:
        r=requests.get(url,timeout=60)
        if r.status_code==200 and len(r.text)>500 and "Error" not in r.text:
            fpath=OUTDIR/fname
            fpath.write_text(r.text,encoding='utf-8')
            lines=[l for l in r.text.split('\n') if not l.startswith('#') and l.strip()]
            print(f"    成功: {len(lines)} 行 -> {fname}")
            return r.text
        else:
            print(f"    失敗 ({len(r.text)} bytes)")
    except Exception as e:
        print(f"    エラー: {e}")
    return None

def fetch_cds(fname_on_server,local_name):
    url=f"https://cdsarc.cds.unistra.fr/ftp/J/AJ/149/180/{fname_on_server}"
    print(f"  取得: {url}")
    try:
        r=requests.get(url,timeout=60)
        if r.status_code==200 and len(r.text)>100:
            fpath=OUTDIR/local_name
            fpath.write_text(r.text,encoding='utf-8')
            print(f"    成功: {len(r.text)} bytes -> {local_name}")
            return r.text
        else:
            print(f"    失敗: HTTP {r.status_code}")
    except Exception as e:
        print(f"    エラー: {e}")
    return None

# 複数のテーブル名を試行
rc_data=None
for suffix in ["table3","table4","table5","table6","table7","rotcur","rc"]:
    rc_data=fetch_vizier(suffix,f"vizier_{suffix}.tsv")
    if rc_data: break

# CDS直接取得
if not rc_data:
    print("\n  VizieR失敗。CDS FTPを試行...")
    for fname in ["table3.dat","table4.dat","table5.dat","ReadMe"]:
        txt=fetch_cds(fname,fname)
        if txt and fname!="ReadMe":
            rc_data=txt; break

# ReadMe（テーブル構造の説明）を取得
print("\n  ReadMe取得...")
readme=fetch_cds("ReadMe","ReadMe.txt")
if readme:
    # テーブル定義部分を抽出
    in_table=False
    for line in readme.split('\n'):
        if 'table' in line.lower() and ('byte' in line.lower() or 'format' in line.lower()):
            in_table=True
        if in_table:
            print(f"    {line[:100]}")
        if in_table and line.strip()=='' and len(line.strip())==0:
            in_table=False

# ====================================================================
print(f"\n{'='*70}")
print("[結果]")
print("="*70)

gs_vals=[r['G_Sigma0'] for r in results]
gc_vals=[r['gc_pred']/a0 for r in results]
gs_indep=[r['G_Sigma0'] for r in results if not r['in_sparc']]
gc_indep=[r['gc_pred']/a0 for r in results if not r['in_sparc']]

print(f"""
  LITTLE THINGS 物性データ:
    全銀河: {len(galaxies)}
    SPARC外: {n_indep}

  log(G*Sigma0) 範囲: [{np.log10(min(gs_vals)):.2f}, {np.log10(max(gs_vals)):.2f}]
  gc_pred/a0 範囲: [{min(gc_vals):.3f}, {max(gc_vals):.3f}]
  gc_pred/a0 中央値: {np.median(gc_vals):.3f}

  SPARC外のみ:
    gc_pred/a0 中央値: {np.median(gc_indep):.3f}
    範囲: [{min(gc_indep):.3f}, {max(gc_indep):.3f}]

  幾何平均法則の予測:
    矮小銀河 (低Sigma0) -> gc < a0
    中央値 {np.median(gc_indep):.3f} < 1.0 (MOND)
    -> 回転曲線フィットで確認が必要

  回転曲線データ: {'取得成功' if rc_data else '取得失敗（手動DL必要）'}
""")

if not rc_data:
    print("  手動ダウンロード先:")
    print("    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/AJ/149/180")
    print("    または: https://cdsarc.cds.unistra.fr/ftp/J/AJ/149/180/")

print("完了。")
