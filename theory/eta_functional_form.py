#!/usr/bin/env python3
"""
eta(disk_dom, Ud) の経験的関数形の決定
基底: g_c = eta0 * sqrt(a0 * G * Sigma0)
残差: R = log(g_c) - log(eta0 * sqrt(a0 * G * Sigma0))
目標: R = f(disk_dom, Ud) の関数形を特定 + LOO-CV
"""
import csv,os,sys,numpy as np
from scipy.stats import linregress,spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
warnings.filterwarnings('ignore')

base_dir=os.path.dirname(os.path.abspath(__file__))
rotmod_dir=os.path.join(base_dir,'Rotmod_LTG')
csv_path=os.path.join(base_dir,'phase1','sparc_results.csv')

a0_si=1.2e-10; G_si=6.674e-11; kpc_m=3.086e19; kms_ms=1e3

def load_csv(path):
    with open(path,"r",encoding="utf-8-sig") as fh:
        reader=csv.DictReader(fh); return reader.fieldnames,list(reader)

def read_dat(name):
    path=os.path.join(rotmod_dir,f"{name}_rotmod.dat")
    if os.path.exists(path):
        try: return np.loadtxt(path,comments='#')
        except: pass
    return None

# ====================================================================
print("="*70)
print("eta(disk_dom, Ud) の経験的関数形の決定")
print("="*70)

# データ読み込み・物理量構築
_,src_rows=load_csv(csv_path)
gal_info={}
for row in src_rows:
    name=row.get('galaxy','').strip()
    if not name: continue
    ud=float(row.get('ud','nan')); vf=float(row.get('vflat','nan'))
    if np.isfinite(ud) and np.isfinite(vf) and ud>0 and vf>0:
        gal_info[name]={'ud':ud,'vflat':vf}

gc_dict={}
gcp=os.path.join(base_dir,"TA3_gc_independent.csv")
if os.path.exists(gcp):
    _,gcr=load_csv(gcp)
    for row in gcr:
        n_=row.get('galaxy','').strip()
        gc=float(row.get('gc_over_a0','nan'))
        if n_ and np.isfinite(gc) and gc>0: gc_dict[n_]=gc

results=[]
for name,info in gal_info.items():
    if name not in gc_dict: continue
    data=read_dat(name)
    if data is None or data.shape[0]<5: continue
    rad=data[:,0]; v_disk=data[:,4]; v_gas=data[:,3]
    v_bul=data[:,5] if data.shape[1]>5 else np.zeros_like(rad)
    ud=info['ud']

    mask=rad>0.01; r=rad[mask]
    vd=v_disk[mask]; vg=v_gas[mask]; vb=v_bul[mask]
    if len(r)<5: continue

    vds=np.sqrt(ud)*np.abs(vd)
    V_pk=np.max(vds); i_pk=np.argmax(vds); r_pk=r[i_pk]
    if r_pk>=r.max()*0.9 or r_pk<0.01: continue
    h_R=r_pk/2.15

    vbar2=ud*np.sign(vd)*vd**2+np.sign(vg)*vg**2+np.sign(vb)*vb**2
    v_bar=np.sqrt(np.maximum(vbar2,0.0))
    V_pk_b=np.max(v_bar)
    disk_dom=V_pk/info['vflat']
    compact=V_pk_b/h_R
    gas_frac=np.mean(np.abs(vg)**2/np.maximum(v_bar**2,0.01))
    Sigma0=V_pk**2/h_R  # proxy [(km/s)^2/kpc]

    results.append({
        'name':name,'gc_a0':gc_dict[name],
        'vflat':info['vflat'],'ud':ud,'h_R':h_R,
        'V_peak':V_pk,'V_peak_bar':V_pk_b,
        'disk_dom':disk_dom,'compact':compact,
        'gas_frac':gas_frac,'Sigma0':Sigma0,
    })

N=len(results)
print(f"  有効銀河数: {N}")

# 配列構築
gc_a0=np.array([d['gc_a0'] for d in results])
gc_si=gc_a0*a0_si
vflat_kms=np.array([d['vflat'] for d in results])
vflat_ms=vflat_kms*kms_ms
hR_kpc=np.array([d['h_R'] for d in results])
hR_m=hR_kpc*kpc_m
ud_arr=np.array([d['ud'] for d in results])
disk_dom=np.array([d['disk_dom'] for d in results])
compact=np.array([d['compact'] for d in results])
V_pk=np.array([d['V_peak'] for d in results])
gas_frac=np.array([d['gas_frac'] for d in results])
Sigma0=np.array([d['Sigma0'] for d in results])

G_S0=vflat_ms**2/hR_m
log_gc=np.log10(gc_si)
log_gc_a0=np.log10(gc_a0)
log_vf=np.log10(vflat_kms)
log_hR=np.log10(hR_kpc)

# 幾何平均モデルの残差
gc_gm=np.sqrt(a0_si*G_S0)
eta0=np.median(gc_si/gc_gm)
log_gc_gm=np.log10(eta0*gc_gm)
R=log_gc-log_gc_gm  # 残差

print(f"  eta0 = {eta0:.4f}")
print(f"  幾何平均残差: std={np.std(R):.4f} dex, |R|中央値={np.median(np.abs(R)):.4f} dex")

# ====================================================================
print(f"\n{'='*70}")
print("【検証1】残差と全候補変数の相関スクリーニング")
print("="*70)

candidates={
    'disk_dom':disk_dom,
    'log_Ud':np.log10(ud_arr),
    'log_compact':np.log10(np.maximum(compact,0.01)),
    'log_Vpeak':np.log10(np.maximum(V_pk,0.1)),
    'log_vflat':log_vf,
    'log_hR':log_hR,
    'log_Sigma0':np.log10(np.maximum(Sigma0,0.01)),
    'gas_frac':gas_frac,
    'Ud':ud_arr,
    'log_GS0':np.log10(G_S0),
}

print(f"\n  {'変数':<18s} {'Pearson r':>10s} {'Spearman rho':>13s} {'p値':>12s} {'判定':>4s}")
print(f"  {'-'*62}")

corr_results=[]
for vname in sorted(candidates.keys()):
    vals=candidates[vname]
    valid=np.isfinite(vals)&np.isfinite(R)
    if np.sum(valid)<10: continue
    r_p,p_p=pearsonr(vals[valid],R[valid])
    r_s,p_s=spearmanr(vals[valid],R[valid])
    sig="**" if p_s<0.001 else "*" if p_s<0.05 else ""
    print(f"  {vname:<18s} {r_p:>+10.3f} {r_s:>+13.3f} {p_s:>12.2e} {sig:>4s}")
    corr_results.append((vname,r_p,r_s,p_s))

corr_results.sort(key=lambda x:abs(x[2]),reverse=True)
top_vars=[v for v,rp,rs,ps in corr_results if ps<0.01]
print(f"\n  有意な変数 (p<0.01): {top_vars}")

# ====================================================================
print(f"\n{'='*70}")
print("【検証2】単変数モデル比較")
print("="*70)

for vname in top_vars[:5]:
    vals=candidates[vname]
    valid=np.isfinite(vals)&np.isfinite(R)
    xv,yv=vals[valid],R[valid]
    if len(xv)<15: continue

    print(f"\n  --- {vname} ---")
    results_m=[]

    # 線形
    sl,ic,r,p,se=linregress(xv,yv)
    pred=sl*xv+ic; ss=np.sum((yv-pred)**2)
    aic=len(xv)*np.log(ss/len(xv))+2*2
    results_m.append(('線形',2,ss,aic,r))

    # 2次
    coeffs=np.polyfit(xv,yv,2)
    pred2=np.polyval(coeffs,xv); ss2=np.sum((yv-pred2)**2)
    aic2=len(xv)*np.log(ss2/len(xv))+2*3
    r2=np.corrcoef(pred2,yv)[0,1] if np.std(pred2)>0 else 0
    results_m.append(('2次',3,ss2,aic2,r2))

    # 対数 (xv>0の場合)
    if np.all(xv>0):
        lx=np.log10(xv)
        sl3,ic3,r3,p3,_=linregress(lx,yv)
        pred3=sl3*lx+ic3; ss3=np.sum((yv-pred3)**2)
        aic3=len(xv)*np.log(ss3/len(xv))+2*2
        results_m.append(('対数',2,ss3,aic3,r3))

    print(f"    {'関数形':<10s} {'k':>3s} {'RSS':>9s} {'AIC':>9s} {'r':>8s}")
    print(f"    {'-'*42}")
    for nm,k,ss,aic,r in results_m:
        print(f"    {nm:<10s} {k:>3d} {ss:>9.4f} {aic:>9.1f} {r:>+8.3f}")
    best=min(results_m,key=lambda x:x[3])
    print(f"    -> 最良: {best[0]} (AIC={best[3]:.1f})")

# ====================================================================
print(f"\n{'='*70}")
print("【検証3】2変数モデルの構築")
print("="*70)

# disk_dom と log(Ud) が主要変数
log_dd=np.log10(np.maximum(disk_dom,0.01))
log_ud=np.log10(np.maximum(ud_arr,0.01))
valid2=np.isfinite(disk_dom)&np.isfinite(ud_arr)&np.isfinite(R)&(disk_dom>0)&(ud_arr>0)
dd_v=disk_dom[valid2]; ud_v=ud_arr[valid2]; R_v=R[valid2]
log_dd_v=np.log10(dd_v); log_ud_v=np.log10(ud_v)
n_v=len(R_v)

print(f"  disk_dom: 中央値={np.median(dd_v):.3f}, [{dd_v.min():.3f}, {dd_v.max():.3f}]")
print(f"  Ud:       中央値={np.median(ud_v):.3f}, [{ud_v.min():.3f}, {ud_v.max():.3f}]")
print(f"  有効: N={n_v}")

model_results=[]

# M0: 定数
ss0=np.sum((R_v-np.mean(R_v))**2)
aic0=n_v*np.log(ss0/n_v)+2*1
model_results.append(('M0: 定数',1,ss0,aic0,0,None,None))

# M1: log(dd) + log(Ud)
X1=np.column_stack([log_dd_v,log_ud_v,np.ones(n_v)])
b1,_,_,_=np.linalg.lstsq(X1,R_v,rcond=None)
p1=X1@b1; ss1=np.sum((R_v-p1)**2); r1=np.corrcoef(p1,R_v)[0,1]
aic1=n_v*np.log(ss1/n_v)+2*3
model_results.append(('M1: a*log(dd)+b*log(Ud)+c',3,ss1,aic1,r1,b1,
    f'R = {b1[0]:+.3f}*log(dd) {b1[1]:+.3f}*log(Ud) {b1[2]:+.3f}'))

# M2: 線形 dd + Ud
X2=np.column_stack([dd_v,ud_v,np.ones(n_v)])
b2,_,_,_=np.linalg.lstsq(X2,R_v,rcond=None)
p2=X2@b2; ss2=np.sum((R_v-p2)**2); r2=np.corrcoef(p2,R_v)[0,1]
aic2=n_v*np.log(ss2/n_v)+2*3
model_results.append(('M2: a*dd+b*Ud+c',3,ss2,aic2,r2,b2,
    f'R = {b2[0]:+.4f}*dd {b2[1]:+.4f}*Ud {b2[2]:+.4f}'))

# M3: log(dd) + Ud (混合)
X3=np.column_stack([log_dd_v,ud_v,np.ones(n_v)])
b3,_,_,_=np.linalg.lstsq(X3,R_v,rcond=None)
p3=X3@b3; ss3=np.sum((R_v-p3)**2); r3=np.corrcoef(p3,R_v)[0,1]
aic3=n_v*np.log(ss3/n_v)+2*3
model_results.append(('M3: a*log(dd)+b*Ud+c',3,ss3,aic3,r3,b3,
    f'R = {b3[0]:+.3f}*log(dd) {b3[1]:+.4f}*Ud {b3[2]:+.4f}'))

# M4: disk_dom のみ (線形)
X4=np.column_stack([dd_v,np.ones(n_v)])
b4,_,_,_=np.linalg.lstsq(X4,R_v,rcond=None)
p4=X4@b4; ss4=np.sum((R_v-p4)**2); r4=np.corrcoef(p4,R_v)[0,1]
aic4=n_v*np.log(ss4/n_v)+2*2
model_results.append(('M4: a*dd+b (単変数)',2,ss4,aic4,r4,b4,
    f'R = {b4[0]:+.4f}*dd {b4[1]:+.4f}'))

# M5: log(Ud) のみ
X5=np.column_stack([log_ud_v,np.ones(n_v)])
b5,_,_,_=np.linalg.lstsq(X5,R_v,rcond=None)
p5=X5@b5; ss5=np.sum((R_v-p5)**2); r5=np.corrcoef(p5,R_v)[0,1]
aic5=n_v*np.log(ss5/n_v)+2*2
model_results.append(('M5: a*log(Ud)+b (単変数)',2,ss5,aic5,r5,b5,
    f'R = {b5[0]:+.3f}*log(Ud) {b5[1]:+.4f}'))

# M6: log(dd) のみ
X6=np.column_stack([log_dd_v,np.ones(n_v)])
b6,_,_,_=np.linalg.lstsq(X6,R_v,rcond=None)
p6=X6@b6; ss6=np.sum((R_v-p6)**2); r6=np.corrcoef(p6,R_v)[0,1]
aic6=n_v*np.log(ss6/n_v)+2*2
model_results.append(('M6: a*log(dd)+b (単変数)',2,ss6,aic6,r6,b6,
    f'R = {b6[0]:+.3f}*log(dd) {b6[1]:+.4f}'))

# M7: dd + log(Ud) + log(compact)
log_cmp_v=np.log10(np.maximum(compact[valid2],0.01))
X7=np.column_stack([dd_v,log_ud_v,log_cmp_v,np.ones(n_v)])
b7,_,_,_=np.linalg.lstsq(X7,R_v,rcond=None)
p7=X7@b7; ss7=np.sum((R_v-p7)**2); r7=np.corrcoef(p7,R_v)[0,1]
aic7=n_v*np.log(ss7/n_v)+2*4
model_results.append(('M7: dd+log(Ud)+log(cmp)+c',4,ss7,aic7,r7,b7,
    f'R = {b7[0]:+.4f}*dd {b7[1]:+.3f}*log(Ud) {b7[2]:+.3f}*log(cmp) {b7[3]:+.4f}'))

print(f"\n  {'モデル':<36s} {'k':>3s} {'RSS':>9s} {'AIC':>9s} {'dAIC':>7s} {'r':>7s}")
print(f"  {'-'*75}")
for nm,k,ss,aic,r,_,_ in model_results:
    print(f"  {nm:<36s} {k:>3d} {ss:>9.4f} {aic:>9.1f} {aic-aic0:>+7.1f} {r:>+7.3f}")

best_model=min(model_results[1:],key=lambda x:x[3])
print(f"\n  最良モデル: {best_model[0]}")
if best_model[6]:
    print(f"    式: {best_model[6]}")

# ====================================================================
print(f"\n{'='*70}")
print("【検証4】最終モデルの精度評価")
print("="*70)

# 最良モデルの残差を全データに適用
R_pred_full=np.zeros(N)
bm=best_model
bm_name=bm[0]

# 最良モデルを識別して適用
if 'M1' in bm_name:
    for i in range(N):
        if disk_dom[i]>0 and ud_arr[i]>0:
            R_pred_full[i]=bm[5][0]*np.log10(disk_dom[i])+bm[5][1]*np.log10(ud_arr[i])+bm[5][2]
    X_best=X1; b_best=b1
elif 'M2' in bm_name:
    for i in range(N):
        R_pred_full[i]=bm[5][0]*disk_dom[i]+bm[5][1]*ud_arr[i]+bm[5][2]
    X_best=X2; b_best=b2
elif 'M3' in bm_name:
    for i in range(N):
        if disk_dom[i]>0:
            R_pred_full[i]=bm[5][0]*np.log10(disk_dom[i])+bm[5][1]*ud_arr[i]+bm[5][2]
    X_best=X3; b_best=b3
elif 'M4' in bm_name:
    for i in range(N):
        R_pred_full[i]=bm[5][0]*disk_dom[i]+bm[5][1]
    X_best=X4; b_best=b4
elif 'M5' in bm_name:
    for i in range(N):
        if ud_arr[i]>0:
            R_pred_full[i]=bm[5][0]*np.log10(ud_arr[i])+bm[5][1]
    X_best=X5; b_best=b5
elif 'M6' in bm_name:
    for i in range(N):
        if disk_dom[i]>0:
            R_pred_full[i]=bm[5][0]*np.log10(disk_dom[i])+bm[5][1]
    X_best=X6; b_best=b6
elif 'M7' in bm_name:
    for i in range(N):
        if disk_dom[i]>0 and ud_arr[i]>0 and compact[i]>0:
            R_pred_full[i]=bm[5][0]*disk_dom[i]+bm[5][1]*np.log10(ud_arr[i])+bm[5][2]*np.log10(compact[i])+bm[5][3]
    X_best=X7; b_best=b7
else:
    X_best=X1; b_best=b1
    for i in range(N):
        if disk_dom[i]>0 and ud_arr[i]>0:
            R_pred_full[i]=b1[0]*np.log10(disk_dom[i])+b1[1]*np.log10(ud_arr[i])+b1[2]

log_gc_final=log_gc_gm+R_pred_full
resid_final=log_gc-log_gc_final
r_final=np.corrcoef(log_gc_final,log_gc)[0,1]

# 比較モデル
resid_mond=log_gc-np.log10(a0_si)
resid_gm=R.copy()

# 6変数経験的 (簡約版: vflat + hR)
log_gc_6var=(-2.175+2.015*log_vf-1.294*log_hR)+np.log10(a0_si)
resid_6var=log_gc-log_gc_6var

print(f"\n  {'モデル':<45s} {'r':>7s} {'残差std':>8s} {'改善率':>7s}")
print(f"  {'-'*70}")
print(f"  {'MOND (g_c=a0)':<45s} {'---':>7s} {np.std(resid_mond):>8.4f} {'---':>7s}")
print(f"  {'sqrt(a0*G*S0) eta固定':<45s} {np.corrcoef(log_gc_gm,log_gc)[0,1]:>7.4f} {np.std(resid_gm):>8.4f} {(1-np.std(resid_gm)/np.std(resid_mond))*100:>6.1f}%")
print(f"  {'sqrt(a0*G*S0) * eta(dd,Ud)':<45s} {r_final:>7.4f} {np.std(resid_final):>8.4f} {(1-np.std(resid_final)/np.std(resid_mond))*100:>6.1f}%")
print(f"  {'6変数経験的 (vflat+hR簡約版)':<45s} {np.corrcoef(log_gc_6var,log_gc)[0,1]:>7.4f} {np.std(resid_6var):>8.4f} {(1-np.std(resid_6var)/np.std(resid_mond))*100:>6.1f}%")

# 残差構造チェック
print(f"\n  eta(dd,Ud) モデル残差構造:")
for lab,vals in [("log(v_flat)",log_vf),("log(h_R)",log_hR),
                  ("log(Ud)",np.log10(ud_arr)),("disk_dom",disk_dom),
                  ("log(compact)",np.log10(np.maximum(compact,0.01)))]:
    rho,p=spearmanr(vals,resid_final)
    sig=" *有意*" if p<0.05 else ""
    print(f"    残差 vs {lab:<15s}: rho={rho:+.3f}, p={p:.2e}{sig}")

# ====================================================================
print(f"\n{'='*70}")
print("【検証5】LOO-CV による過剰適合チェック")
print("="*70)

# 最良モデルの LOO-CV
n_best=X_best.shape[1]  # パラメータ数
loo_resid=np.zeros(n_v)
for i in range(n_v):
    m=np.ones(n_v,dtype=bool); m[i]=False
    b_loo,_,_,_=np.linalg.lstsq(X_best[m],R_v[m],rcond=None)
    loo_resid[i]=R_v[i]-X_best[i]@b_loo

insample_resid=R_v-X_best@b_best
insample_std=np.std(insample_resid)
loo_std=np.std(loo_resid)

print(f"  LOO-CV 結果 ({bm_name}):")
print(f"    インサンプル残差 std = {insample_std:.4f} dex")
print(f"    LOO-CV 残差 std     = {loo_std:.4f} dex")
print(f"    劣化率 = {(loo_std/insample_std-1)*100:.1f}%")

# 全体モデルの LOO 残差推定
# total_resid^2 ≈ gm_resid^2 - insample^2 + loo^2
gm_resid_std=np.std(R_v)
total_loo=np.sqrt(max(gm_resid_std**2-insample_std**2+loo_std**2,0))
print(f"    幾何平均残差 std    = {gm_resid_std:.4f} dex")
print(f"    全体LOO残差推定     = {total_loo:.4f} dex")
print(f"    MOND比LOO改善率     = {(1-total_loo/np.std(resid_mond))*100:.1f}%")

# M1 も常に LOO-CV（最良でなくても参考に）
if 'M1' not in bm_name:
    loo1=np.zeros(n_v)
    for i in range(n_v):
        m=np.ones(n_v,dtype=bool); m[i]=False
        bl,_,_,_=np.linalg.lstsq(X1[m],R_v[m],rcond=None)
        loo1[i]=R_v[i]-X1[i]@bl
    print(f"\n  参考: M1 (log(dd)+log(Ud)) LOO-CV:")
    print(f"    インサンプル std = {np.std(R_v-X1@b1):.4f}")
    print(f"    LOO-CV std     = {np.std(loo1):.4f}")

# ====================================================================
print(f"\n{'='*70}")
print("プロット生成中...")
print("="*70)

fig,axes=plt.subplots(2,3,figsize=(18,12))
fig.suptitle(r'$\eta(disk\_dom, \Upsilon_d)$ Functional Form',fontsize=14,fontweight='bold')

# (a) R vs disk_dom
ax=axes[0,0]
ax.scatter(disk_dom,R,s=15,alpha=0.5,c='steelblue')
sl_dd,ic_dd,_,_,_=linregress(disk_dom,R)
xr=np.linspace(disk_dom.min(),disk_dom.max(),100)
ax.plot(xr,sl_dd*xr+ic_dd,'r-',lw=2)
rho_dd,_=spearmanr(disk_dom,R)
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xlabel(r'$disk\_dom$ ($V_{peak}/v_{flat}$)')
ax.set_ylabel('Residual R [dex]')
ax.set_title(f'(a) R vs disk_dom ($\\rho$={rho_dd:.3f})')

# (b) R vs log(Ud)
ax=axes[0,1]
ax.scatter(np.log10(ud_arr),R,s=15,alpha=0.5,c='coral')
sl_ud,ic_ud,_,_,_=linregress(np.log10(ud_arr),R)
xr=np.linspace(np.log10(ud_arr).min(),np.log10(ud_arr).max(),100)
ax.plot(xr,sl_ud*xr+ic_ud,'r-',lw=2)
rho_ud,_=spearmanr(np.log10(ud_arr),R)
ax.axhline(0,color='k',ls='--',alpha=0.3)
ax.set_xlabel(r'$\log(\Upsilon_d)$')
ax.set_ylabel('Residual R [dex]')
ax.set_title(f'(b) R vs $\\log(\\Upsilon_d)$ ($\\rho$={rho_ud:.3f})')

# (c) 2変数: predicted vs observed R
ax=axes[0,2]
ax.scatter(X_best@b_best,R_v,s=15,alpha=0.5,c='steelblue')
dg=np.linspace(min(X_best@b_best),max(X_best@b_best),100)
ax.plot(dg,dg,'r--',lw=1)
ax.set_xlabel('Predicted R [dex]')
ax.set_ylabel('Observed R [dex]')
ax.set_title(f'(c) Best model: {bm_name[:20]} (r={bm[4]:.3f})')

# (d) 最終: observed vs predicted g_c
ax=axes[1,0]
ax.scatter(log_gc_gm,log_gc,s=10,alpha=0.3,c='green',
           label=f'$\\eta$ fixed ({np.std(resid_gm):.3f})')
ax.scatter(log_gc_final,log_gc,s=10,alpha=0.5,c='coral',
           label=f'$\\eta$(dd,Ud) ({np.std(resid_final):.3f})')
dg=np.linspace(log_gc.min(),log_gc.max(),100)
ax.plot(dg,dg,'k--',alpha=0.3)
ax.set_xlabel(r'$\log(g_c)$ predicted')
ax.set_ylabel(r'$\log(g_c)$ observed')
ax.set_title('(d) Final model comparison')
ax.legend(fontsize=8)

# (e) 残差分布
ax=axes[1,1]
bins=np.linspace(-1,1,40)
ax.hist(resid_mond,bins=bins,alpha=0.2,color='gray',
        label=f'MOND ({np.std(resid_mond):.3f})')
ax.hist(resid_gm,bins=bins,alpha=0.3,color='green',
        label=f'$\\eta$ fixed ({np.std(resid_gm):.3f})')
ax.hist(resid_final,bins=bins,alpha=0.5,color='coral',
        label=f'$\\eta$(dd,Ud) ({np.std(resid_final):.3f})')
ax.set_xlabel('Residual [dex]'); ax.set_ylabel('Count')
ax.set_title('(e) Residual distributions')
ax.legend(fontsize=8)

# (f) LOO-CV 残差比較
ax=axes[1,2]
ax.hist(insample_resid,bins=30,alpha=0.5,color='steelblue',
        label=f'In-sample ({insample_std:.3f})')
ax.hist(loo_resid,bins=30,alpha=0.5,color='coral',
        label=f'LOO-CV ({loo_std:.3f})')
ax.set_xlabel('Residual [dex]'); ax.set_ylabel('Count')
ax.set_title(f'(f) LOO-CV check (degradation: {(loo_std/insample_std-1)*100:.1f}%)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'eta_functional_form_verification.png'),dpi=150)
plt.close()
print("  -> eta_functional_form_verification.png 保存完了")

# ====================================================================
print(f"\n{'='*70}")
print("【総合評価】")
print("="*70)
print(f"""
{'='*60}
  eta(disk_dom, Ud) の関数形の決定結果
{'='*60}

  幾何平均法則の残差構造:
    残差 std = {np.std(R):.4f} dex（eta固定の幾何平均）

  主要な残差相関:""")
for vn,rp,rs,ps in corr_results[:5]:
    sig="**" if ps<0.001 else "*" if ps<0.05 else ""
    print(f"    {vn:<18s}: rho={rs:+.3f} {sig}")

print(f"""
  最良モデル: {best_model[0]}""")
if best_model[6]:
    print(f"    式: {best_model[6]}")
print(f"""    r = {best_model[4]:.4f}

  最終理論モデル:
    g_c = eta(dd,Ud) * sqrt(a0 * G * Sigma0)
    eta(dd,Ud) = {eta0:.4f} * 10^(R_pred)

  精度比較:
    MOND:              残差 {np.std(resid_mond):.4f} dex
    幾何平均(eta固定):  残差 {np.std(resid_gm):.4f} dex  ({(1-np.std(resid_gm)/np.std(resid_mond))*100:.1f}% 改善)
    幾何平均(eta可変):  残差 {np.std(resid_final):.4f} dex  ({(1-np.std(resid_final)/np.std(resid_mond))*100:.1f}% 改善)
    6変数経験的:        残差 {np.std(resid_6var):.4f} dex  ({(1-np.std(resid_6var)/np.std(resid_mond))*100:.1f}% 改善)

  LOO-CV:
    劣化率 = {(loo_std/insample_std-1)*100:.1f}%
    全体LOO残差推定 = {total_loo:.4f} dex
    MOND比LOO改善率 = {(1-total_loo/np.std(resid_mond))*100:.1f}%

  理論構造の階層:
    第1層: g_c ~ sqrt(a0 * G*Sigma0)   <- 確立 (alpha=0.5)
    第2層: eta = eta(disk_dom, Ud)       <- 本検証
    第3層: compact等の追加変数           <- 6変数モデル残余
""")
print("完了。")
