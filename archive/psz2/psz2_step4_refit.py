import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import stats

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
c_light=2.998e8; G=6.674e-11; Msun=1.989e30; Mpc_m=3.086e22
CONST=c_light**2/(4*np.pi*G)*Mpc_m/Msun

def scr(z_l,z_s):
    if z_s<=z_l+0.1: return np.inf
    Dl=cosmo.angular_diameter_distance(z_l).value
    Ds=cosmo.angular_diameter_distance(z_s).value
    Dls=cosmo.angular_diameter_distance_z1z2(z_l,z_s).value
    if Dls<=0: return np.inf
    return CONST*Ds/(Dl*Dls)

def nfw_ds(R,rho_s,r_s):
    x=float(R/r_s)
    ks=rho_s*r_s
    if x<0.01: x=0.01
    if abs(x-1)<1e-4: x=1.0
    if x<1:
        sq=np.sqrt(max(1-x**2,1e-12))
        h=np.arctanh(sq)/sq+np.log(x/2)
        f=1/(x**2-1)*(1-2/sq*np.arctanh(np.sqrt(max((1-x)/(1+x),1e-12))))
    elif x==1:
        h=1+np.log(0.5); f=1/3.0
    else:
        sq=np.sqrt(x**2-1)
        h=np.arctan(sq)/sq+np.log(x/2)
        f=1/(x**2-1)*(1-2/sq*np.arctan(np.sqrt(max((x-1)/(x+1),1e-12))))
    return 4*ks/x**2*h-2*ks*f

ZSRC={1:0.4,2:0.65,3:0.85,4:1.05}
Rb=np.array([0.05,0.10,0.15,0.22,0.30,0.42,0.58,0.80,1.10,1.50])
Rm=0.5*(Rb[:-1]+Rb[1:])

CLUSTERS={
    'J0201':{'ra':30.652, 'dec':-4.955,'z_l':0.196,'r_s':0.169,'r500':0.676,'rho_s':4.8e14, 'src':'J0201_src.csv','type':'Miyaoka'},
    'J0231':{'ra':37.941, 'dec':-4.762,'z_l':0.184,'r_s':0.190,'r500':0.760,'rho_s':4.2e14, 'src':'J0231_src.csv','type':'Miyaoka'},
    'J1311':{'ra':197.863,'dec': 0.403,'z_l':0.183,'r_s':0.246,'r500':0.985,'rho_s':3.5e14, 'src':'J1311_src.csv','type':'Miyaoka'},
    'J2337':{'ra':354.378,'dec':-0.022,'z_l':0.282,'r_s':0.238,'r500':0.952,'rho_s':3.6e14, 'src':'J2337_src.csv','type':'Miyaoka'},
    'G167' :{'ra':33.684, 'dec':-4.586,'z_l':0.139,'r_s':0.2934,'r500':0.6458,'rho_s':2.164e15,'src':'psz2_PSZ2_G167.98-59.95_src.csv','type':'PSZ2'},
    'G273' :{'ra':180.109,'dec': 3.348,'z_l':0.134,'r_s':0.4499,'r500':1.220, 'rho_s':1.806e15,'src':'psz2_PSZ2_G273.59+63.27_src.csv','type':'PSZ2'},
    'G286' :{'ra':185.703,'dec': 2.116,'z_l':0.229,'r_s':0.2886,'r500':0.8205,'rho_s':2.226e15,'src':'psz2_PSZ2_G286.39+64.06_src.csv','type':'PSZ2'},
    'G223' :{'ra':131.371,'dec': 3.475,'z_l':0.327,'r_s':0.3206,'r500':0.9041,'rho_s':2.202e15,'src':'psz2_PSZ2_G223.47+26.85_src.csv','type':'PSZ2'},
    'G228' :{'ra':140.561,'dec': 3.753,'z_l':0.270,'r_s':0.3279,'r500':0.899, 'rho_s':2.137e15,'src':'psz2_PSZ2_G228.50+34.95_src.csv','type':'PSZ2'},
    'G231' :{'ra':139.058,'dec':-0.412,'z_l':0.332,'r_s':0.3100,'r500':0.8893,'rho_s':2.237e15,'src':'psz2_PSZ2_G231.79+31.48_src.csv','type':'PSZ2'},
}

results={}
print(f'{"クラスター":8s} {"タイプ":8s} {"N":8s} {"Q":8s} {"判定":6s} {"ki":9s} {"ko":9s} {"kg":9s}')
for cn,cl in CLUSTERS.items():
    try: df=pd.read_csv(cl['src'])
    except: print(cn,'CSVなし'); continue
    if len(df)<100: continue

    DA=cosmo.angular_diameter_distance(cl['z_l']).value
    cd=np.cos(np.radians(cl['dec']))
    dra=(df['ra'].values-cl['ra'])*cd
    ddec=df['dec'].values-cl['dec']
    R=np.sqrt(dra**2+ddec**2)*np.pi/180*DA
    phi=np.arctan2(ddec,dra)
    e1c=(df['e1'].values-df['shear_c1'].values)/(1+df['shear_m'].values)
    e2c=(df['e2'].values-df['shear_c2'].values)/(1+df['shear_m'].values)
    et=-(e1c*np.cos(2*phi)+e2c*np.sin(2*phi))
    w=df['weight'].values
    Sc=np.array([scr(cl['z_l'],ZSRC.get(int(z),0.7)) for z in df['zbin'].values])
    DS=np.where(np.isfinite(Sc),et*Sc,np.nan)
    Scr_m=np.nanmean(Sc[np.isfinite(Sc)])

    rs=cl['r_s']; r5=cl['r500']
    gt_o=np.full(len(Rm),np.nan)
    gt_n=np.full(len(Rm),np.nan)
    nb=np.zeros(len(Rm))
    for i,(r1,r2) in enumerate(zip(Rb[:-1],Rb[1:])):
        inn=(R>=r1)&(R<r2)&np.isfinite(DS)
        if inn.sum()<5: continue
        et_b=et[inn]; w_b=w[inn]
        em=np.average(et_b,weights=w_b)
        es=np.sqrt(np.average((et_b-em)**2,weights=w_b))
        keep=np.abs(et_b-em)<3.0*es
        if keep.sum()<5: continue
        gt_o[i]=np.average(DS[inn][keep],weights=w_b[keep])/Scr_m
        gt_n[i]=nfw_ds(Rm[i],cl['rho_s'],rs)/Scr_m
        nb[i]=keep.sum()

    kG=gt_o-gt_n
    inner=(Rm<rs)&np.isfinite(kG)
    outer=(Rm>=rs)&(Rm<r5)&np.isfinite(kG)
    ki=np.nanmean(kG[inner]) if inner.sum()>0 else np.nan
    ko=np.nanmean(kG[outer]) if outer.sum()>0 else np.nan
    Q=abs(ki)/abs(ko) if (np.isfinite(ko) and ko!=0 and np.isfinite(ki)) else np.nan
    kg=np.nanmean(kG[(Rm<r5)&np.isfinite(kG)])
    mm=df['shear_m'].mean()

    judg='Q>1★' if (np.isfinite(Q) and Q>1) else ('Q<1' if np.isfinite(Q) else 'nan')
    print(f'{cn:8s} {cl["type"]:8s} {len(df):8d} '
          f'{round(Q,3) if np.isfinite(Q) else "nan":8} {judg:6s} '
          f'{round(ki,4) if np.isfinite(ki) else "nan":9} '
          f'{round(ko,4) if np.isfinite(ko) else "nan":9} '
          f'{round(kg,4) if np.isfinite(kg) else "nan"}')
    results[cn]={'Q':Q,'ki':ki,'ko':ko,'kg':kg,'N':len(df),'type':cl['type']}

print()
valid_Q=[r['Q'] for r in results.values() if np.isfinite(r['Q']) and r['Q']>0]
print(f'有効Q値: {len(valid_Q)}件  Q>1: {sum(1 for q in valid_Q if q>1)}件  平均Q={np.mean(valid_Q):.3f}')
t,p=stats.ttest_1samp(valid_Q,1.0)
print(f't検定  t={t:.3f}  p={p:.4f}')
try:
    st,pw=stats.wilcoxon([q-1 for q in valid_Q],alternative='greater')
    print(f'Wilcoxon  stat={st:.1f}  p={pw:.4f}')
except: pass

print()
print('=== Miyaoka 4件のみ（旧解析の再現確認）===')
Qm=[r['Q'] for cn,r in results.items() if r['type']=='Miyaoka' and np.isfinite(r['Q'])]
t2,p2=stats.ttest_1samp(Qm,1.0)
print(f'N={len(Qm)}  平均Q={np.mean(Qm):.3f}  t={t2:.3f}  p={p2:.4f}')
