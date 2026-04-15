# 検証: J0231のQ値が旧値(1.48)と大きく異なる理由を調べる
import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
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
cl={'ra':37.941,'dec':-4.762,'z_l':0.184,'r_s':0.190,'r500':0.760,'rho_s':4.2e14}

df=pd.read_csv('J0231_src.csv')
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
Rb=np.array([0.05,0.10,0.15,0.22,0.30,0.42,0.58,0.80,1.10,1.50])
Rm=0.5*(Rb[:-1]+Rb[1:])
print('J0231 プロファイル詳細')
print(f'{'R[Mpc]':8s} {'gt_obs':9s} {'gt_nfw':9s} {'kG':9s} {'N':6s} {'<rs?':6s}')
for i,(r1,r2) in enumerate(zip(Rb[:-1],Rb[1:])):
    inn=(R>=r1)&(R<r2)&np.isfinite(DS)
    if inn.sum()<5: continue
    gto=np.average(DS[inn],weights=w[inn])/Scr_m
    gtn=nfw_ds(Rm[i],cl['rho_s'],rs)/Scr_m
    kg=gto-gtn
    flag='★内側' if Rm[i]<rs else '  外側'
    print(f'{Rm[i]:8.3f} {gto:9.5f} {gtn:9.5f} {kg:9.5f} {int(inn.sum()):6d} {flag}')
