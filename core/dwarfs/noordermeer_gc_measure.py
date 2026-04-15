#!/usr/bin/env python3
"""
Noordermeer+2007 早期型銀河: g_c 独立測定
VizieR TAP で回転曲線データ取得 -> RAR フィット -> 幾何平均法則検証
"""
import numpy as np,pandas as pd,sys,os,io
from scipy import stats,optimize
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

try: import requests
except: print("uv run --with requests ..."); sys.exit(1)

base_dir=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=Path(os.path.join(base_dir,"noordermeer")); DATA_DIR.mkdir(exist_ok=True)

a0=1.2e-10; G_SI=6.674e-11; kms=1e3; kpc_m=3.086e19
TAP_URL="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

print("="*70)
print("Noordermeer+2007 早期型銀河: g_c 独立測定")
print("="*70)

# ====================================================================
def vizier_tap(query,fname):
    fpath=DATA_DIR/fname
    if fpath.exists() and fpath.stat().st_size>100:
        print(f"  キャッシュ: {fname}")
        return pd.read_csv(fpath)
    params={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":query}
    try:
        r=requests.get(TAP_URL,params=params,timeout=120)
        if r.status_code==200 and len(r.text)>50:
            from io import StringIO
            lines=[l for l in r.text.split('\n') if not l.startswith('#') and l.strip()]
            if len(lines)>1:
                df=pd.read_csv(StringIO('\n'.join(lines)))
                df.to_csv(fpath,index=False); print(f"  保存: {fname} ({len(df)} 行)")
                return df
        print(f"  レスポンス短い ({len(r.text)} bytes)")
    except Exception as e: print(f"  エラー: {e}")
    return None

# ====================================================================
print(f"\n{'='*70}")
print("[Step 1] VizieR テーブル検索")
print("="*70)

# テーブル検索
print("  テーブル一覧を検索中...")
df_t=vizier_tap("""
SELECT table_name, description FROM TAP_SCHEMA.tables
WHERE table_name LIKE '%385/1359%' OR table_name LIKE '%381/1463%'
ORDER BY table_name
""","tables.csv")
if df_t is not None and len(df_t)>0:
    for _,r in df_t.iterrows():
        print(f"    {r.iloc[0]}: {str(r.iloc[1])[:60]}")

# 各テーブルを試行
print(f"\n  回転曲線テーブル探索...")
candidates=[
    "J/MNRAS/385/1359/table2","J/MNRAS/385/1359/table3",
    "J/MNRAS/385/1359/tablea1","J/MNRAS/385/1359/tablea2",
    "J/MNRAS/381/1463/table1","J/MNRAS/381/1463/table2",
    "J/MNRAS/381/1463/table3","J/MNRAS/381/1463/tablea1",
]

df_rc=None; rc_table=None
for tbl in candidates:
    print(f"\n  試行: {tbl}")
    q=f'SELECT TOP 5 * FROM "{tbl}"'
    try:
        r=requests.get(TAP_URL,params={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q},timeout=30)
        if r.status_code==200:
            lines=[l for l in r.text.split('\n') if not l.startswith('#') and l.strip()]
            if len(lines)>1:
                from io import StringIO
                df_test=pd.read_csv(StringIO('\n'.join(lines)))
                print(f"    成功: カラム={list(df_test.columns)}")
                # 全データ取得
                fname=tbl.replace('/','_')+".csv"
                df_full=vizier_tap(f'SELECT * FROM "{tbl}"',fname)
                if df_full is not None and len(df_full)>0:
                    if df_rc is None or len(df_full)>len(df_rc):
                        df_rc=df_full; rc_table=tbl
                        print(f"    採用: {len(df_rc)} 行")
            else:
                print(f"    データなし")
        else:
            print(f"    HTTP {r.status_code}")
    except Exception as e:
        print(f"    失敗: {e}")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 2] データ構造確認")
print("="*70)

if df_rc is not None:
    print(f"  採用テーブル: {rc_table}")
    print(f"  行数: {len(df_rc)}, カラム数: {len(df_rc.columns)}")
    print(f"\n  カラム一覧:")
    for i,col in enumerate(df_rc.columns):
        dt=str(df_rc[col].dtype); nn=int(df_rc[col].notna().sum())
        if pd.api.types.is_numeric_dtype(df_rc[col]):
            mn=f"{df_rc[col].min():.3g}"; mx=f"{df_rc[col].max():.3g}"
        else:
            vals=df_rc[col].dropna()
            mn=str(vals.iloc[0])[:15] if len(vals)>0 else "NaN"; mx=f"({df_rc[col].nunique()} unique)"
        print(f"    {i:>2d} {col:<25s} {dt:<10s} N={nn:>5d} [{mn} ~ {mx}]")

    # 銀河名カラム
    name_col=None
    for c in df_rc.columns:
        if df_rc[c].dtype=='object':
            nu=df_rc[c].nunique()
            if 2<nu<100: name_col=c; break
    if name_col:
        gals=df_rc[name_col].unique()
        print(f"\n  銀河名カラム: {name_col}")
        print(f"  銀河数: {len(gals)}")
        for g in gals[:15]:
            print(f"    {g}: {(df_rc[name_col]==g).sum()} 点")

    # バリオン成分カラム探索
    print(f"\n  バリオン成分カラム探索:")
    bc={}
    for kw,key in [('Rad','R'),('Vrot','V_obs'),('Vobs','V_obs'),
                    ('Vgas','V_gas'),('Vdis','V_disk'),('Vbul','V_bulge'),
                    ('Vbar','V_bar'),('e_Vrot','V_err'),('e_V','V_err'),
                    ('HI','V_gas'),('star','V_disk'),('gas','V_gas')]:
        for c in df_rc.columns:
            if kw.lower() in c.lower() and key not in bc:
                bc[key]=c; print(f"    {key} <- {c}")
    print(f"\n  検出結果: {bc}")
else:
    print("  回転曲線データなし")
    bc={}

# ====================================================================
print(f"\n{'='*70}")
print("[Step 3] g_c 測定")
print("="*70)

GPARAMS={
    "NGC2841":{"v_flat":305,"h_R":3.5,"in_sparc":True},
    "NGC5533":{"v_flat":240,"h_R":5.2,"in_sparc":False},
    "NGC7331":{"v_flat":250,"h_R":3.1,"in_sparc":True},
    "NGC4138":{"v_flat":200,"h_R":1.8,"in_sparc":False},
    "NGC4389":{"v_flat":120,"h_R":1.2,"in_sparc":False},
    "NGC4013":{"v_flat":180,"h_R":2.4,"in_sparc":True},
    "NGC3992":{"v_flat":260,"h_R":4.1,"in_sparc":False},
    "NGC5055":{"v_flat":210,"h_R":3.0,"in_sparc":True},
    "NGC3953":{"v_flat":230,"h_R":3.5,"in_sparc":False},
    "NGC4051":{"v_flat":155,"h_R":1.7,"in_sparc":False},
    "NGC2903":{"v_flat":195,"h_R":2.1,"in_sparc":True},
    "NGC3198":{"v_flat":150,"h_R":3.0,"in_sparc":True},
    "NGC2998":{"v_flat":215,"h_R":3.8,"in_sparc":False},
}

def rar_fit(R_kpc,V_obs,V_bar,V_err=None):
    Rm=R_kpc*kpc_m
    gobs=(V_obs*kms)**2/Rm; gN=(V_bar*kms)**2/Rm
    ok=(gN>1e-15)&(gobs>1e-15)&np.isfinite(gN)&np.isfinite(gobs)
    if ok.sum()<3: return None,None,None
    go=gobs[ok]; gn=gN[ok]
    ge=0.15*go if V_err is None else np.maximum(2*V_obs[ok]*kms*V_err[ok]*kms/Rm[ok],0.1*go)
    def c2(lg):
        gc=10**lg; gm=0.5*(gn+np.sqrt(gn**2+4*gc*gn))
        return np.sum(((go-gm)/ge)**2)
    gg=np.logspace(-12,-8,100); c2v=[c2(np.log10(g)) for g in gg]
    bi=np.argmin(c2v); gcb=gg[bi]; c2b=c2v[bi]
    try:
        res=optimize.minimize_scalar(c2,bounds=(np.log10(gcb)-1,np.log10(gcb)+1),method='bounded')
        if res.fun<c2b: gcb=10**res.x; c2b=res.fun
    except: pass
    dc2=np.array(c2v)-c2b; m68=dc2<1
    lo=gg[m68].min() if m68.sum()>0 else gcb
    hi=gg[m68].max() if m68.sum()>0 else gcb
    return gcb,(lo,hi),c2b/max(ok.sum()-1,1)

results=[]
if df_rc is not None and name_col and ('V_obs' in bc or len(bc)>=2):
    has_bar='V_bar' in bc or ('V_gas' in bc and 'V_disk' in bc)
    if not has_bar:
        print("  バリオン分離カラムなし。利用可能カラムで代用を試みます。")

    for gname,gp in GPARAMS.items():
        # 銀河名マッチ
        mk=df_rc[name_col].astype(str).str.contains(gname[-4:],case=False,na=False)
        if not mk.any():
            mk=df_rc[name_col].astype(str).str.contains(gname.replace('NGC',''),case=False,na=False)
        if not mk.any(): continue

        dg=df_rc[mk].copy()
        # カラム取得
        R=dg[bc['R']].values if 'R' in bc else dg.iloc[:,0].values
        Vo=dg[bc['V_obs']].values if 'V_obs' in bc else dg.iloc[:,1].values
        Ve=dg[bc['V_err']].values if 'V_err' in bc else None

        if 'V_bar' in bc:
            Vb=dg[bc['V_bar']].values
        elif 'V_gas' in bc and 'V_disk' in bc:
            Vg=dg[bc['V_gas']].values; Vd=dg[bc['V_disk']].values
            Vbu=dg[bc['V_bulge']].values if 'V_bulge' in bc else np.zeros(len(dg))
            Vb=np.sqrt(np.maximum(np.sign(Vg)*Vg**2+np.sign(Vd)*Vd**2+np.sign(Vbu)*Vbu**2,0))
        else:
            continue

        ok=np.isfinite(R)&np.isfinite(Vo)&np.isfinite(Vb)&(R>0)&(np.abs(Vo)>0)
        if ok.sum()<3: print(f"  {gname}: データ不足 ({ok.sum()})"); continue

        gc,ci,c2d=rar_fit(R[ok],np.abs(Vo[ok]),np.abs(Vb[ok]),
                          np.abs(Ve[ok]) if Ve is not None else None)
        if gc is None or gc<1e-12:
            print(f"  {gname}: フィット失敗"); continue

        vf=gp['v_flat']; hR=gp['h_R']
        GS=(vf*kms)**2/(hR*kpc_m); gc_gm=np.sqrt(a0*GS)

        results.append({'galaxy':gname,'v_flat':vf,'h_R':hR,
                       'gc':gc,'gc_lo':ci[0],'gc_hi':ci[1],
                       'gc_a0':gc/a0,'gc_gm_a0':gc_gm/a0,
                       'GS':GS,'c2dof':c2d,'in_sparc':gp['in_sparc'],'n_pts':ok.sum()})
        print(f"  {gname}: gc={gc/a0:.3f} a0, geomean={gc_gm/a0:.3f} a0, "
              f"chi2/dof={c2d:.2f}, N={ok.sum()}")
else:
    print("  回転曲線データなしまたはカラム不足 -> 測定スキップ")

# ====================================================================
print(f"\n{'='*70}")
print("[Step 4] 結果サマリー")
print("="*70)

if results:
    df_r=pd.DataFrame(results)
    df_r.to_csv(os.path.join(base_dir,'noordermeer_gc_results.csv'),index=False)
    print(f"  測定成功: {len(df_r)} 銀河")

    print(f"\n  {'銀河':<12s} {'gc/a0':>7s} {'[68%CI]':>16s} {'geom/a0':>8s} {'ratio':>6s} {'SPARC':>6s}")
    print(f"  {'-'*60}")
    for _,r in df_r.iterrows():
        ratio=r['gc']/np.sqrt(a0*r['GS'])
        print(f"  {r['galaxy']:<12s} {r['gc_a0']:>7.3f} [{r['gc_lo']/a0:.3f},{r['gc_hi']/a0:.3f}] "
              f"{r['gc_gm_a0']:>8.3f} {ratio:>6.2f} {'YES' if r['in_sparc'] else 'NO':>6s}")

    # SPARC外のみ
    di=df_r[~df_r['in_sparc']]
    if len(di)>=3:
        print(f"\n  SPARC外 ({len(di)} 銀河) の alpha フィット:")
        x=np.log10(di['GS'].values/a0); y=np.log10(di['gc'].values/a0)
        sl,ic,r,p,se=stats.linregress(x,y)
        t05=(sl-0.5)/se; p05=2*stats.t.sf(abs(t05),len(x)-2)
        print(f"    alpha = {sl:.3f} +/- {se:.3f}")
        print(f"    alpha=0.5 検定: t={t05:.2f}, p={p05:.4f} -> {'整合' if p05>0.05 else '不整合'}")
        print(f"    SPARC alpha=0.545 との差: {abs(sl-0.545):.3f}")

        # gc/a0 統計
        print(f"    gc/a0 中央値: {np.median(di['gc_a0']):.3f}")
        print(f"    gc_geomean/a0 中央値: {np.median(di['gc_gm_a0']):.3f}")
        print(f"    SPARC gc/a0 中央値: 0.825")

    # プロット
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    ax=axes[0]
    sp=df_r[df_r['in_sparc']]; ns=df_r[~df_r['in_sparc']]
    if len(ns)>0:
        ax.errorbar(ns['gc_gm_a0'],ns['gc_a0'],
                   yerr=[ns['gc_a0']-ns['gc_lo']/a0,ns['gc_hi']/a0-ns['gc_a0']],
                   fmt='o',color='coral',ms=8,capsize=3,label=f'SPARC external ({len(ns)})')
    if len(sp)>0:
        ax.errorbar(sp['gc_gm_a0'],sp['gc_a0'],
                   yerr=[sp['gc_a0']-sp['gc_lo']/a0,sp['gc_hi']/a0-sp['gc_a0']],
                   fmt='s',color='steelblue',ms=8,capsize=3,label=f'SPARC overlap ({len(sp)})')
    rng=np.linspace(0.01,5,100)
    ax.plot(rng,rng,'g-',lw=2,label='geomean=measured')
    ax.axhline(1,color='blue',ls='--',alpha=0.5,label='MOND (gc=a0)')
    ax.set_xlabel('gc_geomean / a0'); ax.set_ylabel('gc_measured / a0')
    ax.set_title('(a) Geomean prediction vs measurement'); ax.legend(fontsize=7)

    ax=axes[1]
    ax.scatter(np.log10(df_r['GS']/a0),np.log10(df_r['gc']/a0),
              c=['coral' if not s else 'steelblue' for s in df_r['in_sparc']],s=50,edgecolors='black')
    xr=np.linspace(-1,2,100)
    ax.plot(xr,0.5*xr,'g-',lw=2,label='alpha=0.5')
    ax.plot(xr,0.545*xr,'r--',lw=1.5,label='SPARC alpha=0.545')
    ax.axhline(0,color='blue',ls=':',alpha=0.5,label='MOND')
    ax.set_xlabel('log(G*Sigma0/a0)'); ax.set_ylabel('log(gc/a0)')
    ax.set_title('(b) RAR scaling'); ax.legend(fontsize=7)

    ax=axes[2]
    ratio_all=df_r['gc'].values/np.sqrt(a0*df_r['GS'].values)
    ax.bar(range(len(df_r)),ratio_all,
          color=['coral' if not s else 'steelblue' for s in df_r['in_sparc']],
          alpha=0.7,edgecolor='black')
    ax.axhline(1,color='g',ls='-',lw=2,label='geomean exact')
    ax.set_xticks(range(len(df_r)))
    ax.set_xticklabels([g[:8] for g in df_r['galaxy']],rotation=45,fontsize=7)
    ax.set_ylabel('gc_measured / gc_geomean'); ax.set_title('(c) Prediction accuracy')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'noordermeer_gc_verification.png'),dpi=150)
    plt.close()
    print(f"\n  -> noordermeer_gc_verification.png, noordermeer_gc_results.csv")
else:
    print("  測定結果なし")
    print(f"  VizieR取得状況: {'成功' if df_rc is not None else '失敗'}")
    if df_rc is not None:
        print(f"  行数: {len(df_rc)}, カラム: {list(df_rc.columns)[:10]}")
        print(f"  検出バリオンカラム: {bc}")
    print(f"\n  手動ダウンロード先:")
    print(f"    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/385/1359")
    print(f"    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/381/1463")

print("\n完了。")
