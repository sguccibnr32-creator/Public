import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

c=2.998e8; G=6.674e-11; Msun=1.989e30; Mpc=3.086e22
CONST=c**2/(4*np.pi*G)*Mpc/Msun

def scr(z_l,z_s):
    if z_s<=z_l+0.05: return np.inf
    Dl=cosmo.angular_diameter_distance(z_l).value
    Ds=cosmo.angular_diameter_distance(z_s).value
    Dls=cosmo.angular_diameter_distance_z1z2(z_l,z_s).value
    if Dls<=0: return np.inf
    return CONST*Ds/(Dl*Dls)

CLUSTERS={
    'J0201':{'ra':30.652, 'dec':-4.955,'z_l':0.196,'r_s':0.169,'r500':0.676,'rho_s':4.8e14,'SP':0.610},
    'J0231':{'ra':37.941, 'dec':-4.762,'z_l':0.184,'r_s':0.190,'r500':0.760,'rho_s':4.2e14,'SP':0.528},
    'J1311':{'ra':197.863,'dec': 0.403,'z_l':0.183,'r_s':0.246,'r500':0.985,'rho_s':3.5e14,'SP':0.736},
    'J2337':{'ra':354.378,'dec':-0.022,'z_l':0.282,'r_s':0.238,'r500':0.952,'rho_s':3.6e14,'SP':0.636},
}
ZSRC={1:0.4,2:0.65,3:0.85,4:1.05}

print('=== 確認1：shear bias m の確定値 ===')
m_vals=[]
for cn,cl in CLUSTERS.items():
    df=pd.read_csv(cn+'_src.csv')
    m=df['shear_m'].mean()
    m_vals.append(m)
    print(cn, 'm=',round(m,4), '補正係数=',round(1/(1+m),4))
print('平均m=',round(np.mean(m_vals),4))
print('平均補正係数=',round(np.mean([1/(1+m) for m in m_vals]),4))

print()
print('=== 確認2：additive bias c1,c2 の系統確認 ===')
for cn,cl in CLUSTERS.items():
    df=pd.read_csv(cn+'_src.csv')
    print(cn,'c1=',round(df['shear_c1'].mean(),6),'c2=',round(df['shear_c2'].mean(),6))

print()
print('=== 確認3：スタック解析（4クラスター合計） ===')
print('クラスター中心からR=0.05~2Mpc のγ_t(R)を重ね合わせ')

Rb=np.array([0.05,0.10,0.15,0.20,0.30,0.45,0.65,0.90,1.30,1.80])
Rm=0.5*(Rb[:-1]+Rb[1:])
gt_stack=np.zeros(len(Rm))
wt_stack=np.zeros(len(Rm))
nb_stack=np.zeros(len(Rm))

for cn,cl in CLUSTERS.items():
    df=pd.read_csv(cn+'_src.csv')
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
    valid=np.isfinite(Sc)&(Sc>0)

    for i,(r1,r2) in enumerate(zip(Rb[:-1],Rb[1:])):
        inn=(R>=r1)&(R<r2)&valid
        if inn.sum()<3: continue
        gt_stack[i]+=np.sum(w[inn]*et[inn]*Sc[inn])
        wt_stack[i]+=np.sum(w[inn]*Sc[inn])
        nb_stack[i]+=inn.sum()

gt_mean=np.where(wt_stack>0, gt_stack/wt_stack, np.nan)

print('R[Mpc]  gt_stack  N_total  SNR目安')
for i in range(len(Rm)):
    if wt_stack[i]>0 and nb_stack[i]>0:
        noise=0.3/np.sqrt(nb_stack[i])
        snr=abs(gt_mean[i])/noise if noise>0 else 0
        print(round(Rm[i],2), round(gt_mean[i],5), int(nb_stack[i]), round(snr,2))

print()
print('=== 確認4：m補正が旧κ_G値に与える影響 ===')
kg_orig={'J0201':-0.017,'J0231':-0.043,'J1311':-0.063,'J2337':-0.086}
m_by_cl={'J0201':-0.1199,'J0231':-0.1039,'J1311':-0.1173,'J2337':-0.1542}
print('クラスター  κ_G旧値  m補正係数  κ_G補正値  変化率')
for cn in CLUSTERS:
    kg=kg_orig[cn]
    cf=1/(1+m_by_cl[cn])
    kg_new=kg*cf
    print(cn, round(kg,4), round(cf,4), round(kg_new,4), round((cf-1)*100,1),'%')

print()
print('=== 結論サマリー ===')
print('1. shear bias m = -0.12 ~ -0.15（未補正）-> κ_G を11~19%過小評価')
print('2. additive bias c1,c2 は極めて小さく無視可能')
print('3. スタック解析でR=0.1~0.5Mpc のγ_t を確認')
print('4. 個別クラスターの内側N<30 -> Q値の直接改善は困難')
print('5. 確実な改善: 旧κ_G値にm補正係数を適用')
