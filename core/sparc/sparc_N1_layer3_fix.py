# -*- coding: utf-8 -*-
"""
N-1 Layer 3 fix: eta^2 = gc^2 / (a0 * GS0) correction + Test 3 re-run.
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
    try: plt.rcParams['font.family'] = fontname; break
    except: continue
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

a0=1.2e-10;G_SI=6.674e-11;kpc_m=3.0857e19;Msun=1.989e30

BASE=os.path.dirname(os.path.abspath(__file__))
ROTMOD=os.path.join(BASE,'Rotmod_LTG')
PHASE1=os.path.join(BASE,'phase1','sparc_results.csv')
TA3=os.path.join(BASE,'TA3_gc_independent.csv')
for p,l in [(ROTMOD,'Rotmod_LTG'),(PHASE1,'sparc_results.csv'),(TA3,'TA3_gc_independent.csv')]:
    if not os.path.exists(p): print(f'[ERROR] {l}: {p}'); sys.exit(1)

def load_csv(path):
    with open(path,'r',encoding='utf-8-sig') as f: header=f.readline().strip()
    sep=',' if ',' in header else None; data={}
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
def get_key(info,cands,default=None):
    for c in cands:
        if c in info:
            try: return float(info[c])
            except: return info[c]
    return default
def load_rotmod(fp):
    cols=[[] for _ in range(8)]
    with open(fp,'r') as f:
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
results=[]; rotmod_files=sorted(glob.glob(os.path.join(ROTMOD,'*.dat')))
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
    rpk=rad[np.argmax(vds)]
    if rpk<0.01 or rpk>=rad.max()*0.9: continue
    hR_kpc=rpk/2.15; hR_m=hR_kpc*kpc_m
    vflat_ms=vflat*1e3; gc_si=gc_a0*a0
    GS0_proxy=vflat_ms**2/hR_m
    vdpk=np.max(np.sqrt(max(ud,0.01))*np.abs(vdisk))*1e3
    M_disk=2.0*vdpk**2*hR_m/(0.56*G_SI)
    vgpk=np.max(np.abs(vgas))*1e3
    M_gas=2.0*vgpk**2*hR_m/(0.56*G_SI) if vgpk>0 else 0
    M_bar=M_disk+M_gas; M_bar_sun=M_bar/Msun
    Sigma_bar=M_disk/(2*np.pi*hR_m**2); G_Sigma_bar=G_SI*Sigma_bar
    # FIXED: eta^2 = gc^2 / (a0 * GS0)
    eta_sq=gc_si**2/(a0*GS0_proxy) if GS0_proxy>0 else np.nan
    eta_val=np.sqrt(eta_sq) if np.isfinite(eta_sq) and eta_sq>0 else np.nan
    if M_bar>0:
        R_BTFR=vflat_ms**4/(G_SI*M_bar); btfr_ratio=R_BTFR/gc_si if gc_si>0 else np.nan
    else: R_BTFR=np.nan; btfr_ratio=np.nan
    results.append({
        'galaxy':gname,'gc_a0':gc_a0,'log_gc':np.log10(gc_a0),
        'vflat':vflat,'Yd':ud,'log_Yd':np.log10(ud),
        'hR_kpc':hR_kpc,'log_hR':np.log10(hR_kpc),'log_vflat':np.log10(vflat),
        'GS0_proxy':GS0_proxy,'log_GS0':np.log10(GS0_proxy/a0),
        'M_bar_sun':M_bar_sun,'log_Mbar':np.log10(M_bar_sun) if M_bar_sun>0 else np.nan,
        'G_Sigma_bar':G_Sigma_bar,
        'log_GSbar':np.log10(G_Sigma_bar/a0) if G_Sigma_bar>0 else np.nan,
        'eta_sq':eta_sq,
        'log_eta_sq':np.log10(eta_sq) if np.isfinite(eta_sq) and eta_sq>0 else np.nan,
        'eta':eta_val,'log_eta':np.log10(eta_val) if np.isfinite(eta_val) and eta_val>0 else np.nan,
        'btfr_ratio':btfr_ratio,
        'log_btfr_r':np.log10(btfr_ratio) if np.isfinite(btfr_ratio) and btfr_ratio>0 else np.nan,
    })

N=len(results); print(f'  Processed: {N} galaxies')
if N<10: print('[ERROR] Too few.'); sys.exit(1)

log_gc=np.array([r['log_gc'] for r in results])
log_hR=np.array([r['log_hR'] for r in results])
log_vflat=np.array([r['log_vflat'] for r in results])
log_GS0=np.array([r['log_GS0'] for r in results])
log_Mbar=np.array([r['log_Mbar'] for r in results])
log_GSbar=np.array([r['log_GSbar'] for r in results])
log_eta_sq=np.array([r['log_eta_sq'] for r in results])
log_eta=np.array([r['log_eta'] for r in results])
log_Yd=np.array([r['log_Yd'] for r in results])
eta_arr=np.array([r['eta'] for r in results])

# SANITY CHECK
print('\n'+'='*70); print('SANITY CHECK: eta (FIXED)'); print('='*70)
m_eta=np.isfinite(log_eta)
print(f'\n  N={np.sum(m_eta)}')
print(f'  eta: median={np.nanmedian(eta_arr[m_eta]):.4f}')
print(f'  IQR=[{np.nanpercentile(eta_arr[m_eta],25):.4f},{np.nanpercentile(eta_arr[m_eta],75):.4f}]')
print(f'  range=[{np.nanmin(eta_arr[m_eta]):.4f},{np.nanmax(eta_arr[m_eta]):.4f}]')
for r in results:
    if 0.2<r['gc_a0']<0.3 and 80<r['vflat']<120:
        print(f'\n  Verification: {r["galaxy"]}')
        print(f'    gc={r["gc_a0"]:.3f}a0={r["gc_a0"]*a0:.3e}')
        print(f'    GS0={r["GS0_proxy"]:.3e}, eta={r["eta"]:.4f}')
        print(f'    Check: eta*sqrt(a0*GS0)={r["eta"]*np.sqrt(a0*r["GS0_proxy"]):.3e} vs gc={r["gc_a0"]*a0:.3e}')
        break

# TEST 3 CORRECTED
print('\n'+'='*70); print('TEST 3 CORRECTED: eta structure'); print('='*70)

def regprint(label,lx,ly):
    m=np.isfinite(lx)&np.isfinite(ly)
    if m.sum()<10: print(f'  [{label}] N={m.sum()} too few'); return None,None,None,None,None,None
    sl,it,r,_,se=stats.linregress(lx[m],ly[m])
    rho,p=stats.spearmanr(lx[m],ly[m])
    print(f'\n  [{label}] N={m.sum()}')
    print(f'    slope={sl:.4f}+/-{se:.4f}, R^2={r**2:.4f}, rho={rho:.4f}, p={p:.2e}')
    return sl,se,r**2,rho,p,m.sum()

sl_eS,se_eS,r2_eS,rho_eS,p_eS,_=regprint('eta vs G*Sigma_bar',log_GSbar,log_eta)
if sl_eS is not None:
    t=abs(sl_eS-(-0.5))/se_eS; p_c=2*(1-stats.t.cdf(t,df=163-2))
    print(f'    p(slope=-0.5)={p_c:.4f} (full cancel prediction)')

sl_eH,se_eH,r2_eH,rho_eH,p_eH,_=regprint('eta vs hR',log_hR,log_eta)
sl_eV,se_eV,r2_eV,rho_eV,p_eV,_=regprint('eta vs vflat',log_vflat,log_eta)

sl_eY,se_eY,r2_eY,rho_eY,p_eY,_=regprint('eta vs Yd',log_Yd,log_eta)
if sl_eY is not None:
    t=abs(sl_eY-(-0.44))/se_eY; p_y44=2*(1-stats.t.cdf(t,df=163-2))
    print(f'    p(slope=-0.44)={p_y44:.4f} (5-5 prediction)')

sl_eM,se_eM,r2_eM,rho_eM,p_eM,_=regprint('eta vs M_bar',log_Mbar,log_eta)
sl_eG,se_eG,r2_eG,rho_eG,p_eG,_=regprint('eta vs GS0_proxy',log_GS0,log_eta)
if sl_eG is not None:
    print(f'    effective alpha = 0.5 + {sl_eG:.4f} = {0.5+sl_eG:.4f}')

# SYNTHESIS
print('\n'+'='*70); print('SYNTHESIS'); print('='*70)
print(f'\n  {"Variable":<20s} | {"slope":>8s}+/-{"SE":>6s} | {"R^2":>6s} | {"rho":>6s}')
print('  '+'-'*60)
for lb,sl,se,r2,rho in [('G*Sigma_bar',sl_eS,se_eS,r2_eS,rho_eS),
    ('hR',sl_eH,se_eH,r2_eH,rho_eH),('vflat',sl_eV,se_eV,r2_eV,rho_eV),
    ('Yd',sl_eY,se_eY,r2_eY,rho_eY),('M_bar',sl_eM,se_eM,r2_eM,rho_eM),
    ('GS0_proxy',sl_eG,se_eG,r2_eG,rho_eG)]:
    if sl is not None:
        print(f'  {lb:<20s} | {sl:>8.4f}+/-{se:>6.4f} | {r2:>6.4f} | {rho:>6.4f}')

# FIGURES
print('\n[3] Generating figures...')
fig,axes=plt.subplots(2,3,figsize=(17,11))

ax=axes[0,0]
ev=eta_arr[np.isfinite(eta_arr)&(eta_arr>0)]
ax.hist(np.log10(ev),bins=30,color='steelblue',alpha=0.7,edgecolor='white')
ax.axvline(np.log10(np.nanmedian(ev)),color='red',ls='--',label=f'median={np.nanmedian(ev):.4f}')
ax.set_xlabel('log10(eta)'); ax.set_ylabel('Count'); ax.set_title('1: eta distribution (FIXED)')
ax.legend(fontsize=8)

ax=axes[0,1]
m=np.isfinite(log_eta)&np.isfinite(log_GSbar)
ax.scatter(log_GSbar[m],log_eta[m],s=10,alpha=0.5,c='darkorange',edgecolors='none')
xf=np.linspace(log_GSbar[m].min(),log_GSbar[m].max(),100)
if sl_eS is not None:
    ax.plot(xf,sl_eS*xf+(np.nanmedian(log_eta[m])-sl_eS*np.nanmedian(log_GSbar[m])),'r-',lw=2,label=f'slope={sl_eS:.3f}')
    it_plot=np.nanmedian(log_eta[m])-(-0.5)*np.nanmedian(log_GSbar[m])
    ax.plot(xf,-0.5*xf+it_plot,'b--',lw=1.5,label='full cancel: -0.5')
ax.set_xlabel('log(G*Sigma_bar/a0)'); ax.set_ylabel('log(eta)')
ax.set_title('2: eta vs Sigma_bar'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[0,2]
m=np.isfinite(log_eta)&np.isfinite(log_Yd)
ax.scatter(log_Yd[m],log_eta[m],s=10,alpha=0.5,c='green',edgecolors='none')
xf=np.linspace(log_Yd[m].min(),log_Yd[m].max(),100)
if sl_eY is not None:
    ax.plot(xf,sl_eY*xf+(np.nanmedian(log_eta[m])-sl_eY*np.nanmedian(log_Yd[m])),'r-',lw=2,label=f'slope={sl_eY:.3f}')
ax.set_xlabel('log(Yd)'); ax.set_ylabel('log(eta)')
ax.set_title('3: eta vs Yd'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,0]
m=np.isfinite(log_eta)&np.isfinite(log_GS0)
ax.scatter(log_GS0[m],log_eta[m],s=10,alpha=0.5,c='crimson',edgecolors='none')
if sl_eG is not None:
    xf=np.linspace(log_GS0[m].min(),log_GS0[m].max(),100)
    ax.plot(xf,sl_eG*xf+(np.nanmedian(log_eta[m])-sl_eG*np.nanmedian(log_GS0[m])),'r-',lw=2,label=f'slope={sl_eG:.3f}')
ax.set_xlabel('log(GS0/a0)'); ax.set_ylabel('log(eta)')
ax.set_title('4: eta vs proxy'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,1]
m=np.isfinite(log_eta)&np.isfinite(log_vflat)
ax.scatter(log_vflat[m],log_eta[m],s=10,alpha=0.5,c='teal',edgecolors='none')
if sl_eV is not None:
    xf=np.linspace(log_vflat[m].min(),log_vflat[m].max(),100)
    ax.plot(xf,sl_eV*xf+(np.nanmedian(log_eta[m])-sl_eV*np.nanmedian(log_vflat[m])),'r-',lw=2,label=f'slope={sl_eV:.3f}')
ax.set_xlabel('log(vflat)'); ax.set_ylabel('log(eta)')
ax.set_title('5: eta vs vflat'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,2]
labels_bar=['Sigma_bar','hR','vflat','Yd','M_bar','GS0_proxy']
r2s_bar=[r2_eS or 0,r2_eH or 0,r2_eV or 0,r2_eY or 0,r2_eM or 0,r2_eG or 0]
cols_bar=['darkorange','teal','crimson','green','steelblue','grey']
ax.bar(range(len(labels_bar)),r2s_bar,color=cols_bar,alpha=0.7,edgecolor='black',linewidth=0.5)
ax.set_xticks(range(len(labels_bar))); ax.set_xticklabels(labels_bar,fontsize=8,rotation=15)
ax.set_ylabel('R^2 with log(eta)'); ax.set_title('6: What determines eta?')
for i,v in enumerate(r2s_bar): ax.text(i,v+0.005,f'{v:.3f}',ha='center',fontsize=8)
ax.grid(True,alpha=0.3,axis='y')

fig.suptitle(f'N-1 Layer 3 FIX: eta structure (N={N})',fontsize=14,y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE,'fig_N1_layer3_fix.png'),dpi=150)
print('  -> fig_N1_layer3_fix.png')

outcsv=os.path.join(BASE,'N1_layer3_fix_results.csv')
cols_out=['galaxy','gc_a0','log_gc','vflat','Yd','log_Yd','hR_kpc','log_hR',
          'log_GS0','log_Mbar','log_GSbar','eta','log_eta','log_eta_sq','log_btfr_r']
with open(outcsv,'w',encoding='utf-8') as f:
    f.write(','.join(cols_out)+'\n')
    for r in results: f.write(','.join(str(r.get(c,'')) for c in cols_out)+'\n')
print(f'  -> {outcsv}')

print('\n'+'='*70); print('VERDICT (CORRECTED)'); print('='*70)
print(f'\n  eta median={np.nanmedian(eta_arr[m_eta]):.4f}')
print(f'\n  KEY: Does eta cancel Sigma_bar?')
if sl_eS is not None:
    print(f'    slope={sl_eS:.4f}+/-{se_eS:.4f} (prediction -0.5)')
    if abs(sl_eS-(-0.5))<2*se_eS:
        print(f'    >>> YES: consistent with -0.5 -> gc INDEPENDENT of Sigma_bar')
        print(f'    >>> Layer 3: RESOLVED')
    elif abs(sl_eS)>0.2:
        print(f'    >>> PARTIAL cancellation')
    else:
        print(f'    >>> NO cancellation')
if sl_eG is not None:
    print(f'\n  effective alpha = 0.5 + {sl_eG:.4f} = {0.5+sl_eG:.4f}')
print('\n[DONE]')
