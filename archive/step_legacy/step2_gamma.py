import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

c=2.998e8; G=6.674e-11; Msun=1.989e30; Mpc=3.086e22
CONST = c**2/(4*np.pi*G)*Mpc/Msun

def scr(z_l, z_s):
    if z_s <= z_l+0.05: return np.inf
    Dl=cosmo.angular_diameter_distance(z_l).value
    Ds=cosmo.angular_diameter_distance(z_s).value
    Dls=cosmo.angular_diameter_distance_z1z2(z_l,z_s).value
    if Dls<=0: return np.inf
    return CONST*Ds/(Dl*Dls)

def nfw_ds(R, rho_s, r_s):
    x=float(R/r_s)
    ks=rho_s*r_s
    if x<1:
        sq=np.sqrt(max(1-x**2,1e-12))
        h=np.arctanh(sq)/sq+np.log(x/2)
        f=1/(x**2-1)*(1-2/sq*np.arctanh(np.sqrt(max((1-x)/(1+x),0))))
    elif x==1:
        h=1+np.log(0.5); f=1/3.0
    else:
        sq=np.sqrt(x**2-1)
        h=np.arctan(sq)/sq+np.log(x/2)
        f=1/(x**2-1)*(1-2/sq*np.arctan(np.sqrt(max((x-1)/(x+1),0))))
    return 4*ks/x**2*h - 2*ks*f

CLUSTERS = {
    'J0201':{'z_l':0.196,'r_s':0.169,'r500':0.676,'rho_s':4.8e14,'SP':0.610},
    'J0231':{'z_l':0.184,'r_s':0.190,'r500':0.760,'rho_s':4.2e14,'SP':0.528},
    'J1311':{'z_l':0.183,'r_s':0.246,'r500':0.985,'rho_s':3.5e14,'SP':0.736},
    'J2337':{'z_l':0.282,'r_s':0.238,'r500':0.952,'rho_s':3.6e14,'SP':0.636},
    'J1415':{'z_l':0.143,'r_s':0.050,'r500':0.560,'rho_s':6.45e13,'SP':0.202},
    'J0158':{'z_l':0.230,'r_s':0.133,'r500':0.545,'rho_s':6.4e12,'SP':0.677},
}
ZSRC={1:0.4, 2:0.65, 3:0.85, 4:1.05}
RA={'J0201':30.652,'J0231':37.941,'J1311':197.863,'J2337':354.378,'J1415':213.698,'J0158':29.697}
DEC={'J0201':-4.955,'J0231':-4.762,'J1311':0.403,'J2337':-0.022,'J1415':0.186,'J0158':-4.888}

results={}
for cn, cl in CLUSTERS.items():
    df=pd.read_csv(cn+'_src.csv')
    DA=cosmo.angular_diameter_distance(cl['z_l']).value
    cd=np.cos(np.radians(DEC[cn]))
    dra=(df['ra'].values-RA[cn])*cd
    ddec=df['dec'].values-DEC[cn]
    R=np.sqrt(dra**2+ddec**2)*np.pi/180*DA
    phi=np.arctan2(df['dec'].values-DEC[cn],(df['ra'].values-RA[cn])*cd)
    e1c=(df['e1'].values-df['shear_c1'].values)/(1+df['shear_m'].values)
    e2c=(df['e2'].values-df['shear_c2'].values)/(1+df['shear_m'].values)
    et=-(e1c*np.cos(2*phi)+e2c*np.sin(2*phi))
    w=df['weight'].values
    Sc=np.array([scr(cl['z_l'],ZSRC.get(int(z),0.7)) for z in df['zbin'].values])
    DS=np.where(np.isfinite(Sc), et*Sc, np.nan)

    rs=cl['r_s']; r5=cl['r500']
    Rb=np.array([0.02,rs*0.4,rs*0.8,rs,rs*1.5,r5*0.6,r5,r5*1.5])
    Rm=0.5*(Rb[:-1]+Rb[1:])
    Scr_mean=np.nanmean(Sc[np.isfinite(Sc)])

    gt_o=np.full(len(Rm),np.nan)
    gt_n=np.full(len(Rm),np.nan)
    nb=np.zeros(len(Rm))
    for i,(r1,r2) in enumerate(zip(Rb[:-1],Rb[1:])):
        inn=(R>=r1)&(R<r2)&np.isfinite(DS)
        if inn.sum()<5: continue
        gt_o[i]=np.average(DS[inn],weights=w[inn])/Scr_mean
        gt_n[i]=nfw_ds(Rm[i],cl['rho_s'],rs)/Scr_mean
        nb[i]=inn.sum()

    kG=gt_o-gt_n
    inner=(Rm<rs)&np.isfinite(kG)
    outer=(Rm>=rs)&(Rm<r5)&np.isfinite(kG)
    ki=np.nanmean(kG[inner]) if inner.sum()>0 else np.nan
    ko=np.nanmean(kG[outer]) if outer.sum()>0 else np.nan
    Q=abs(ki)/abs(ko) if (np.isfinite(ko) and ko!=0) else np.nan
    kg=np.nanmean(kG[(Rm<r5)&np.isfinite(kG)])
    mm=df['shear_m'].mean()

    results[cn]={'Q':Q,'ki':ki,'ko':ko,'kg':kg,'N':len(df),'m':mm,
                 'Rm':Rm,'kG':kG,'gt_o':gt_o,'gt_n':gt_n,'nb':nb}

    print(cn,'N=',len(df),'m=',round(mm,4),'Q=',round(Q,3) if np.isfinite(Q) else 'nan',
          'ki=',round(ki,4) if np.isfinite(ki) else 'nan',
          'ko=',round(ko,4) if np.isfinite(ko) else 'nan',
          'kg=',round(kg,4) if np.isfinite(kg) else 'nan')

print()
print('=== T-14 Q値まとめ ===')
Qold={'J0201':3.71,'J0231':1.48,'J1311':3.04,'J2337':2.69}
kgold={'J0201':-0.017,'J0231':-0.043,'J1311':-0.063,'J2337':-0.086}
print('クラスター  N      m補正   Q旧   Q新    判定  SP    kg旧   kg新')
for cn,res in results.items():
    sp=CLUSTERS[cn]['SP']
    print(cn, res['N'], round(1/(1+res['m']),4),
          Qold.get(cn,'-'),
          round(res['Q'],2) if np.isfinite(res['Q']) else 'nan',
          'Q>1' if (np.isfinite(res['Q']) and res['Q']>1) else 'Q<1',
          sp, kgold.get(cn,'-'),
          round(res['kg'],4) if np.isfinite(res['kg']) else 'nan')

Qs=[r['Q'] for r in results.values() if np.isfinite(r['Q'])]
t,p=stats.ttest_1samp(Qs,1.0)
print()
print('t検定 N=',len(Qs),'平均Q=',round(np.mean(Qs),3),'t=',round(t,3),'p=',round(p,4))

print()
print('=== プロファイル詳細 ===')
for cn,res in results.items():
    print(cn)
    for j in range(len(res['Rm'])):
        if np.isfinite(res['kG'][j]):
            print('  R=',round(res['Rm'][j],3),
                  'gt_obs=',round(res['gt_o'][j],5),
                  'gt_nfw=',round(res['gt_n'][j],5),
                  'kG=',round(res['kG'][j],5),
                  'N=',int(res['nb'][j]))
