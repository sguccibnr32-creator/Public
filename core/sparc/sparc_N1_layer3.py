# -*- coding: utf-8 -*-
"""
N-1 Layer 3: U(eps;c) self-consistency algebra verification.
"""
import os, sys, glob, warnings
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _fp in ['/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
            '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
            r'C:\Windows\Fonts\msgothic.ttc']:
    try: _fm.fontManager.addfont(_fp)
    except: pass
for fontname in ['IPAGothic', 'MS Gothic', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = fontname
        break
    except: continue
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

a0=1.2e-10; G_SI=6.674e-11; kpc_m=3.0857e19; Msun=1.989e30; pc_m=3.0857e16

BASE=os.path.dirname(os.path.abspath(__file__))
ROTMOD=os.path.join(BASE,'Rotmod_LTG')
PHASE1=os.path.join(BASE,'phase1','sparc_results.csv')
TA3=os.path.join(BASE,'TA3_gc_independent.csv')

for p,label in [(ROTMOD,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not os.path.exists(p): print(f'[ERROR] {label} not found: {p}'); sys.exit(1)

def load_csv(path):
    with open(path,'r',encoding='utf-8-sig') as f: header=f.readline().strip()
    sep=',' if ',' in header else None
    data={}
    with open(path,'r',encoding='utf-8-sig') as f:
        cols=[c.strip() for c in f.readline().strip().split(sep)]; rows=[]
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append([p.strip() for p in line.split(sep)])
    for i,col in enumerate(cols):
        vals=[]
        for row in rows:
            if i<len(row):
                try: vals.append(float(row[i]))
                except: vals.append(row[i])
            else: vals.append(np.nan)
        data[col]=vals
    return data

def find_name_col(data):
    for c in ['galaxy','Galaxy','name','Name','GALAXY']:
        if c in data: return c
    for k,v in data.items():
        if isinstance(v[0],str): return k
    return list(data.keys())[0]

def get_key(info,candidates,default=None):
    for c in candidates:
        if c in info:
            try: return float(info[c])
            except: return info[c]
    return default

def load_rotmod(filepath):
    cols=[[] for _ in range(8)]
    with open(filepath,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                for j in range(min(len(parts),8)): cols[j].append(float(parts[j]))
                for j in range(len(parts),8): cols[j].append(0.0)
            except ValueError: continue
    return tuple(np.array(c) for c in cols)

def U(eps,c):
    if eps>=1.0 or eps<0: return np.nan
    return -eps-eps**2/2.0-c*np.log(1.0-eps)

def U_prime(eps,c):
    if eps>=1.0: return np.nan
    return -1.0-eps+c/(1.0-eps)

def U_double_prime(eps,c):
    if eps>=1.0: return np.nan
    return -1.0+c/(1.0-eps)**2

def epsilon_eq(x,c):
    disc=(x+2.0)**2-4.0*c
    if disc<0: return np.nan
    return (-x+np.sqrt(disc))/2.0

print('[1] Loading data...')
phase1=load_csv(PHASE1); ta3=load_csv(TA3)
p1_nc=find_name_col(phase1); ta3_nc=find_name_col(ta3)

galaxy_info={}
for i,name in enumerate(phase1[p1_nc]):
    name=str(name).strip(); info={}
    for k in phase1:
        if k==p1_nc: continue
        try: info[k]=float(phase1[k][i])
        except: info[k]=phase1[k][i]
    galaxy_info[name]=info

for i,name in enumerate(ta3[ta3_nc]):
    name=str(name).strip()
    if name in galaxy_info:
        for k in ta3:
            if k==ta3_nc: continue
            try: galaxy_info[name][k]=float(ta3[k][i])
            except: galaxy_info[name][k]=ta3[k][i]

print('[2] Processing galaxies...')
results=[]
rotmod_files=sorted(glob.glob(os.path.join(ROTMOD,'*.dat')))

for fpath in rotmod_files:
    gname=os.path.splitext(os.path.basename(fpath))[0].replace('_rotmod','').strip()
    info=None
    for key in [gname,gname.upper(),gname.lower()]:
        if key in galaxy_info: info=galaxy_info[key]; break
    if info is None: continue

    ud=get_key(info,['upsilon_d','Upsilon_d','Ud','ud','Yd'])
    gc_a0=get_key(info,['gc_over_a0','gc/a0','gc_ratio'])
    vflat=get_key(info,['vflat','Vflat','v_flat'])

    if ud is None or gc_a0 is None or vflat is None: continue
    if np.isnan(ud) or np.isnan(gc_a0) or gc_a0<=0 or vflat<=0 or ud<=0: continue

    try: rad,vobs,errv,vgas,vdisk,vbul,sbdisk,sbbul=load_rotmod(fpath)
    except: continue
    if len(rad)<3: continue

    vds=np.sqrt(max(ud,0.01))*np.abs(vdisk)
    rpk_idx=np.argmax(vds); rpk=rad[rpk_idx]
    if rpk<0.01 or rpk>=rad.max()*0.9: continue
    hR_kpc=rpk/2.15; hR_m=hR_kpc*kpc_m

    vflat_ms=vflat*1e3; gc_si=gc_a0*a0
    GS0_proxy=vflat_ms**2/hR_m

    vdisk_peak_ms=np.max(np.sqrt(max(ud,0.01))*np.abs(vdisk))*1e3
    M_disk=2.0*vdisk_peak_ms**2*hR_m/(0.56*G_SI)
    vgas_peak_ms=np.max(np.abs(vgas))*1e3
    M_gas=2.0*vgas_peak_ms**2*hR_m/(0.56*G_SI) if vgas_peak_ms>0 else 0
    M_bar_direct=M_disk+M_gas; M_bar_sun=M_bar_direct/Msun

    Sigma_bar=M_disk/(2*np.pi*hR_m**2)
    G_Sigma_bar=G_SI*Sigma_bar

    if M_bar_direct>0:
        R_BTFR=vflat_ms**4/(G_SI*M_bar_direct)
        log_R_BTFR=np.log10(R_BTFR/a0)
    else: R_BTFR=np.nan; log_R_BTFR=np.nan

    btfr_ratio=R_BTFR/gc_si if gc_si>0 and np.isfinite(R_BTFR) else np.nan

    eta_sq=gc_si/(a0*GS0_proxy) if GS0_proxy>0 else np.nan
    log_eta_sq=np.log10(eta_sq) if np.isfinite(eta_sq) and eta_sq>0 else np.nan

    results.append({
        'galaxy':gname,'gc_a0':gc_a0,'log_gc':np.log10(gc_a0),
        'vflat':vflat,'Yd':ud,'log_Yd':np.log10(ud),
        'hR_kpc':hR_kpc,'log_hR':np.log10(hR_kpc),
        'log_vflat':np.log10(vflat),
        'GS0_proxy':GS0_proxy,'log_GS0':np.log10(GS0_proxy/a0),
        'M_bar_sun':M_bar_sun,
        'log_Mbar':np.log10(M_bar_sun) if M_bar_sun>0 else np.nan,
        'G_Sigma_bar':G_Sigma_bar,
        'log_GSbar':np.log10(G_Sigma_bar/a0) if G_Sigma_bar>0 else np.nan,
        'R_BTFR':R_BTFR,'log_R_BTFR':log_R_BTFR,
        'btfr_ratio':btfr_ratio,
        'log_btfr_ratio':np.log10(btfr_ratio) if np.isfinite(btfr_ratio) and btfr_ratio>0 else np.nan,
        'eta_sq':eta_sq,'log_eta_sq':log_eta_sq,
    })

N=len(results)
print(f'  Processed: {N} galaxies')
if N<10: print('[ERROR] Too few.'); sys.exit(1)

log_gc=np.array([r['log_gc'] for r in results])
log_hR=np.array([r['log_hR'] for r in results])
log_vflat=np.array([r['log_vflat'] for r in results])
log_GS0=np.array([r['log_GS0'] for r in results])
log_Mbar=np.array([r['log_Mbar'] for r in results])
log_GSbar=np.array([r['log_GSbar'] for r in results])
log_R_BTFR=np.array([r['log_R_BTFR'] for r in results])
log_btfr_r=np.array([r['log_btfr_ratio'] for r in results])
log_eta_sq=np.array([r['log_eta_sq'] for r in results])
log_Yd=np.array([r['log_Yd'] for r in results])

# TEST 1
print('\n'+'='*70); print('TEST 1: BTFR residual vs hR'); print('='*70)
mask1=np.isfinite(log_btfr_r)&np.isfinite(log_hR)
print(f'\n  N = {np.sum(mask1)}')
br=np.array([r['btfr_ratio'] for r in results])
br_valid=br[np.isfinite(br)&(br>0)]
print(f'  BTFR ratio: median={np.nanmedian(br_valid):.2f}, IQR=[{np.nanpercentile(br_valid,25):.2f},{np.nanpercentile(br_valid,75):.2f}]')

rho_br_hR,p_br_hR=stats.spearmanr(log_btfr_r[mask1],log_hR[mask1])
sl_br,int_br,r_br,_,se_br=stats.linregress(log_hR[mask1],log_btfr_r[mask1])
print(f'\n  log(BTFR_ratio) vs log(hR): rho={rho_br_hR:.4f}, p={p_br_hR:.2e}')
print(f'    slope={sl_br:.3f}+/-{se_br:.3f}, R^2={r_br**2:.3f}')

mask1b=np.isfinite(log_R_BTFR)&np.isfinite(log_gc)
sl_rb,int_rb,r_rb,_,se_rb=stats.linregress(log_R_BTFR[mask1b],log_gc[mask1b])
rho_rb,p_rb=stats.spearmanr(log_R_BTFR[mask1b],log_gc[mask1b])
print(f'\n  log(R_BTFR/a0) vs log(gc/a0): slope={sl_rb:.3f}+/-{se_rb:.3f}, R^2={r_rb**2:.3f}, rho={rho_rb:.4f}')

# TEST 2
print('\n'+'='*70); print('TEST 2: gc vs Sigma_bar (re-confirm)'); print('='*70)
mask2=np.isfinite(log_GSbar)&np.isfinite(log_gc)
sl2,int2,r2,_,se2=stats.linregress(log_GSbar[mask2],log_gc[mask2])
rho2,p2=stats.spearmanr(log_GSbar[mask2],log_gc[mask2])
print(f'\n  N={np.sum(mask2)}, slope={sl2:.4f}+/-{se2:.4f}, R^2={r2**2:.4f}, rho={rho2:.4f}')

# TEST 3
print('\n'+'='*70); print('TEST 3: eta residual structure'); print('='*70)
mask3=np.isfinite(log_eta_sq)
print(f'\n  N={np.sum(mask3)}, median log(eta^2)={np.nanmedian(log_eta_sq[mask3]):.4f}, std={np.nanstd(log_eta_sq[mask3]):.4f}')
print(f'  eta=10^(median/2)={10**(np.nanmedian(log_eta_sq[mask3])/2):.4f}')

mask3s=np.isfinite(log_eta_sq)&np.isfinite(log_GSbar)
rho_eta_S,p_eta_S=stats.spearmanr(log_eta_sq[mask3s],log_GSbar[mask3s])
sl_eS,int_eS,r_eS,_,se_eS=stats.linregress(log_GSbar[mask3s],log_eta_sq[mask3s])
print(f'\n  log(eta^2) vs log(G*Sigma_bar/a0): rho={rho_eta_S:.4f}, slope={sl_eS:.3f}+/-{se_eS:.3f}, R^2={r_eS**2:.3f}')
print(f'  (Cancel prediction: slope=-1.0)')

mask3h=np.isfinite(log_eta_sq)&np.isfinite(log_hR)
rho_eta_hR,p_eta_hR=stats.spearmanr(log_eta_sq[mask3h],log_hR[mask3h])
print(f'  log(eta^2) vs log(hR): rho={rho_eta_hR:.4f}, p={p_eta_hR:.2e}')

mask3y=np.isfinite(log_eta_sq)&np.isfinite(log_Yd)
rho_eta_Yd,p_eta_Yd=stats.spearmanr(log_eta_sq[mask3y],log_Yd[mask3y])
sl_eY,_,_,_,se_eY=stats.linregress(log_Yd[mask3y],log_eta_sq[mask3y])
print(f'  log(eta^2) vs log(Yd): rho={rho_eta_Yd:.4f}, slope={sl_eY:.3f}')

# TEST 4
print('\n'+'='*70); print('TEST 4: BTFR direct check'); print('='*70)
log_vf4=4*log_vflat
mask4=np.isfinite(log_vf4)&np.isfinite(log_Mbar)
sl4,int4,r4,_,se4=stats.linregress(log_Mbar[mask4],log_vf4[mask4])
rho4,p4=stats.spearmanr(log_Mbar[mask4],log_vf4[mask4])
t4=abs(sl4-1.0)/se4; p4s=2*(1-stats.t.cdf(t4,df=np.sum(mask4)-2))
print(f'\n  N={np.sum(mask4)}, slope={sl4:.3f}+/-{se4:.3f}, R^2={r4**2:.3f}, p(slope=1)={p4s:.4f}')

mask4m=np.isfinite(log_vf4)&np.isfinite(log_Mbar)&np.isfinite(log_hR)
X4=np.column_stack([log_Mbar[mask4m],log_hR[mask4m],np.ones(mask4m.sum())])
y4=log_vf4[mask4m]
b4,_,_,_=np.linalg.lstsq(X4,y4,rcond=None)
ss4=np.sum((y4-X4@b4)**2); ss4t=np.sum((y4-y4.mean())**2)
r2_4m=1-ss4/ss4t
mse4m=ss4/(len(y4)-3); se4m=np.sqrt(np.diag(mse4m*np.linalg.inv(X4.T@X4)))
t_hR4=abs(b4[1])/se4m[1]; p_hR4=2*(1-stats.t.cdf(t_hR4,df=len(y4)-3))
print(f'\n  Multivar BTFR: log(vf^4)={b4[0]:.3f}*log(M_bar)+{b4[1]:.3f}*log(hR)')
print(f'    SE: {se4m[0]:.3f}, {se4m[1]:.3f}, R^2={r2_4m:.3f}')
print(f'    p(hR=0)={p_hR4:.4f} -> hR adds to BTFR? {"YES" if p_hR4<0.05 else "NO"}')

# TEST 5 skipped (no c_fit data in current pipeline)

# FIGURES
print('\n[3] Generating figures...')
fig,axes=plt.subplots(2,3,figsize=(17,11))

ax=axes[0,0]
br_log=log_btfr_r[np.isfinite(log_btfr_r)]
ax.hist(br_log,bins=30,color='steelblue',alpha=0.7,edgecolor='white')
ax.axvline(0,color='red',ls='--',lw=2,label='BTFR exact')
ax.set_xlabel('log(vflat^4/(G*M_bar*gc))'); ax.set_ylabel('Count')
ax.set_title('1: BTFR ratio'); ax.legend(fontsize=8)

ax=axes[0,1]
m=np.isfinite(log_btfr_r)&np.isfinite(log_hR)
ax.scatter(log_hR[m],log_btfr_r[m],s=10,alpha=0.5,c='steelblue',edgecolors='none')
xf=np.linspace(log_hR[m].min(),log_hR[m].max(),100)
ax.plot(xf,sl_br*xf+int_br,'r-',lw=2,label=f'rho={rho_br_hR:.3f}')
ax.set_xlabel('log(hR/kpc)'); ax.set_ylabel('log(BTFR ratio)')
ax.set_title('2: BTFR residual vs hR'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[0,2]
m3=np.isfinite(log_eta_sq)&np.isfinite(log_GSbar)
ax.scatter(log_GSbar[m3],log_eta_sq[m3],s=10,alpha=0.5,c='darkorange',edgecolors='none')
xf=np.linspace(log_GSbar[m3].min(),log_GSbar[m3].max(),100)
ax.plot(xf,sl_eS*xf+int_eS,'r-',lw=2,label=f'slope={sl_eS:.3f}')
ax.plot(xf,-1.0*xf+int_eS,'b--',lw=1,label='exact cancel=-1')
ax.set_xlabel('log(G*Sigma_bar/a0)'); ax.set_ylabel('log(eta^2)')
ax.set_title('3: Does eta cancel Sigma_bar?'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,0]
m4=np.isfinite(log_Mbar)&np.isfinite(log_vf4)
ax.scatter(log_Mbar[m4],log_vf4[m4],s=10,alpha=0.5,c='green',edgecolors='none')
xf=np.linspace(log_Mbar[m4].min(),log_Mbar[m4].max(),100)
ax.plot(xf,sl4*xf+int4,'r-',lw=2,label=f'slope={sl4:.3f}')
ax.plot(xf,1.0*xf+int4,'b--',lw=1,label='BTFR: 1.0')
ax.set_xlabel('log(M_bar)'); ax.set_ylabel('log(vflat^4)')
ax.set_title('4: BTFR quality'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,1]
m_rb=np.isfinite(log_R_BTFR)&np.isfinite(log_gc)
ax.scatter(log_R_BTFR[m_rb],log_gc[m_rb],s=10,alpha=0.5,c='crimson',edgecolors='none')
xf=np.linspace(log_R_BTFR[m_rb].min(),log_R_BTFR[m_rb].max(),100)
ax.plot(xf,sl_rb*xf+int_rb,'r-',lw=2,label=f'slope={sl_rb:.3f}')
ax.plot(xf,xf,'b--',lw=1,label='1:1')
ax.set_xlabel('log(R_BTFR/a0)'); ax.set_ylabel('log(gc/a0)')
ax.set_title('5: R_BTFR vs gc'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,2]
labels=['proxy\nalpha=0.5','M_bar\ndirect','Sigma_bar\ndirect','R_BTFR\n=vf^4/GM']
r2s=[stats.linregress(log_GS0[np.isfinite(log_GS0)&np.isfinite(log_gc)],
                       log_gc[np.isfinite(log_GS0)&np.isfinite(log_gc)])[2]**2,
     stats.linregress(log_Mbar[np.isfinite(log_Mbar)&np.isfinite(log_gc)],
                       log_gc[np.isfinite(log_Mbar)&np.isfinite(log_gc)])[2]**2,
     r2**2,r_rb**2]
cols=['steelblue','darkorange','green','crimson']
ax.bar(range(len(labels)),r2s,color=cols,alpha=0.7,edgecolor='black',linewidth=0.5)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel('R^2'); ax.set_title('6: gc prediction comparison')
for i,v in enumerate(r2s): ax.text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=8)
ax.grid(True,alpha=0.3,axis='y')

fig.suptitle(f'N-1 Layer 3: Self-consistency (N={N})',fontsize=14,y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE,'fig_N1_layer3.png'),dpi=150)
print('  -> fig_N1_layer3.png')

outcsv=os.path.join(BASE,'N1_layer3_results.csv')
cols_out=['galaxy','gc_a0','log_gc','vflat','Yd','hR_kpc','log_hR',
          'log_GS0','log_Mbar','log_GSbar','log_R_BTFR','log_btfr_ratio','log_eta_sq']
with open(outcsv,'w',encoding='utf-8') as f:
    f.write(','.join(cols_out)+'\n')
    for r in results:
        f.write(','.join(str(r.get(c,'')) for c in cols_out)+'\n')
print(f'  -> {outcsv}')

print('\n'+'='*70); print('VERDICT'); print('='*70)

print(f'\n  TEST 1: BTFR residual vs hR')
if abs(rho_br_hR)>0.3 and p_br_hR<0.001:
    print(f'    >>> BTFR has hR dependence (rho={rho_br_hR:.3f})')
elif abs(rho_br_hR)>0.15 and p_br_hR<0.05:
    print(f'    >>> Weak hR dependence (rho={rho_br_hR:.3f})')
else:
    print(f'    >>> No significant hR dependence (rho={rho_br_hR:.3f})')

print(f'\n  TEST 2: gc vs Sigma_bar: R^2={r2**2:.3f} (unchanged)')

print(f'\n  TEST 3: eta cancellation')
if abs(sl_eS+1.0)<2*se_eS:
    print(f'    >>> eta DOES cancel Sigma_bar (slope={sl_eS:.3f} ~ -1.0)')
elif abs(sl_eS)>0.3:
    print(f'    >>> eta PARTIALLY cancels (slope={sl_eS:.3f})')
else:
    print(f'    >>> eta does NOT cancel (slope={sl_eS:.3f})')

print(f'\n  TEST 4: BTFR quality: slope={sl4:.3f}, p(1)={p4s:.4f}')
if p_hR4<0.05: print(f'    >>> hR adds to BTFR (p={p_hR4:.4f})')
else: print(f'    >>> hR does NOT add to BTFR')

print(f'\n  SYNTHESIS:')
print(f'    gc-Sigma_bar contradiction resolved if eta has Sigma_bar dependence')
print(f'    that cancels: slope={sl_eS:.3f} (prediction -1.0)')

print('\n[DONE]')
