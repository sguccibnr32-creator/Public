import pandas as pd, numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize_scalar
from scipy import stats

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
c_light=2.998e8; G=6.674e-11; Msun=1.989e30; Mpc_m=3.086e22
CONST=c_light**2/(4*np.pi*G)*Mpc_m/Msun
rho_crit0=2.775e11  # M_sun/Mpc^3

def rho_crit(z):
    return rho_crit0 * cosmo.efunc(z)**2

def nfw_params_from_M500(M500_1e14, z, c500=3.5):
    M500 = M500_1e14 * 1e14
    rc = rho_crit(z)
    r500 = (3*M500/(4*np.pi*500*rc))**(1/3)
    r_s  = r500 / c500
    rho_s= (500/3)*rc*c500**3/(np.log(1+c500)-c500/(1+c500))
    return r500, r_s, rho_s

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

# === ステップ1: MCXC実測値でG167・G273のNFWパラメータ更新 ===
print('=== MCXC実測値からNFWパラメータ計算 ===')
MCXC = {
    'G167': {'M500':1.7898, 'z':0.139, 'ra':33.684,  'dec':-4.586, 'src':'psz2_PSZ2_G167.98-59.95_src.csv'},
    'G273': {'M500':4.1194, 'z':0.134, 'ra':180.109, 'dec': 3.348, 'src':'psz2_PSZ2_G273.59+63.27_src.csv'},
}
for cn, info in MCXC.items():
    r500,r_s,rho_s = nfw_params_from_M500(info['M500'], info['z'])
    print(f'{cn}: M500={info["M500"]}e14 Msun  r500={r500:.4f}Mpc  r_s={r_s:.4f}Mpc  rho_s={rho_s:.4e}')
    MCXC[cn]['r500']=r500; MCXC[cn]['r_s']=r_s; MCXC[cn]['rho_s']=rho_s

# === ステップ2: rho_s を gamma_t から直接フィット（r_s固定） ===
print()
print('=== rho_s 拘束フィット（r_s固定、gt_obs>0ビンのみ使用） ===')

ALL_CL = {
    'G167': {**MCXC['G167']},
    'G273': {**MCXC['G273']},
    'G286': {'ra':185.703,'dec': 2.116,'z_l':0.229,'r_s':0.236,'r500':0.825,
             'src':'psz2_PSZ2_G286.39+64.06_src.csv'},
    'G223': {'ra':131.371,'dec': 3.475,'z_l':0.327,'r_s':0.247,'r500':0.865,
             'src':'psz2_PSZ2_G223.47+26.85_src.csv'},
    'G228': {'ra':140.561,'dec': 3.753,'z_l':0.270,'r_s':0.260,'r500':0.911,
             'src':'psz2_PSZ2_G228.50+34.95_src.csv'},
    'G231': {'ra':139.058,'dec':-0.412,'z_l':0.332,'r_s':0.240,'r500':0.841,
             'src':'psz2_PSZ2_G231.79+31.48_src.csv'},
}
# z_l キーの統一
for cn in ['G167','G273']:
    ALL_CL[cn]['z_l'] = ALL_CL[cn]['z']

fit_results = {}
for cn, cl in ALL_CL.items():
    try: df = pd.read_csv(cl['src'])
    except: print(cn,'CSV なし'); continue

    DA  = cosmo.angular_diameter_distance(cl['z_l']).value
    cd  = np.cos(np.radians(cl['dec']))
    dra = (df['ra'].values - cl['ra'])*cd
    ddec= df['dec'].values - cl['dec']
    R   = np.sqrt(dra**2+ddec**2)*np.pi/180*DA
    phi = np.arctan2(ddec,dra)
    e1c = (df['e1'].values-df['shear_c1'].values)/(1+df['shear_m'].values)
    e2c = (df['e2'].values-df['shear_c2'].values)/(1+df['shear_m'].values)
    et  = -(e1c*np.cos(2*phi)+e2c*np.sin(2*phi))
    w   = df['weight'].values
    Sc  = np.array([scr(cl['z_l'],ZSRC.get(int(z),0.7)) for z in df['zbin'].values])
    DS  = np.where(np.isfinite(Sc),et*Sc,np.nan)
    Scr_m = np.nanmean(Sc[np.isfinite(Sc)])

    gt_o = np.full(len(Rm),np.nan)
    gt_err= np.full(len(Rm),np.nan)
    nb   = np.zeros(len(Rm))
    for i,(r1,r2) in enumerate(zip(Rb[:-1],Rb[1:])):
        inn=(R>=r1)&(R<r2)&np.isfinite(DS)
        if inn.sum()<5: continue
        et_b=et[inn]; w_b=w[inn]
        em=np.average(et_b,weights=w_b)
        es=np.sqrt(np.average((et_b-em)**2,weights=w_b))
        keep=np.abs(et_b-em)<3.0*es
        if keep.sum()<5: continue
        gt_o[i] =np.average(DS[inn][keep],weights=w_b[keep])/Scr_m
        gt_err[i]=np.std(DS[inn][keep])/(np.sqrt(keep.sum())*Scr_m)
        nb[i]=keep.sum()

    # フィット: R>r_s の外側ビンのみ使用（内側は混在が大きい）
    # かつ gt_obs>0 かつ N>=20
    r_s = cl['r_s']
    use = (Rm >= r_s*0.5) & np.isfinite(gt_o) & (nb>=20) & (gt_o>0)
    if use.sum() < 2:
        # 外側ビンが不足 → 全ビンで試す
        use = np.isfinite(gt_o) & (nb>=10) & np.isfinite(gt_err) & (gt_err>0)

    if use.sum() < 2:
        print(f'{cn}: フィット不能（有効ビン{use.sum()}）')
        fit_results[cn] = {'rho_s_fit': cl.get('rho_s',3.5e14), 'r_s': r_s, 'chi2': 999}
        continue

    def chi2_func(log_rho):
        rho = 10**log_rho
        pred = np.array([nfw_ds(rm,rho,r_s)/Scr_m for rm in Rm[use]])
        resid = (gt_o[use]-pred)
        if np.any(gt_err[use]>0):
            return np.sum((resid/gt_err[use])**2)
        return np.sum(resid**2)

    res = minimize_scalar(chi2_func, bounds=(12,17), method='bounded')
    rho_fit = 10**res.x
    chi2_min = res.fun / max(use.sum()-1,1)

    fit_results[cn] = {'rho_s_fit': rho_fit, 'r_s': r_s, 'chi2': chi2_min,
                       'rho_s_orig': cl.get('rho_s',3.12e14)}
    print(f'{cn}: rho_s_fit={rho_fit:.3e}  '
          f'rho_s_orig={cl.get("rho_s",3.12e14):.3e}  '
          f'chi2/dof={chi2_min:.2f}  use_bins={use.sum()}')

print()
print('=== NFWパラメータ確定値まとめ ===')
print(f'{"クラスター":8s} {"rho_s_fit":12s} {"r_s":8s} {"出典":20s}')
sources = {'G167':'MCXC実測', 'G273':'MCXC実測',
           'G286':'HSC直接フィット','G223':'HSC直接フィット',
           'G228':'HSC直接フィット','G231':'HSC直接フィット'}
for cn,res in fit_results.items():
    print(f'{cn:8s} {res["rho_s_fit"]:.4e}  {res["r_s"]:.4f}  {sources.get(cn,"推定")}')

# 確定パラメータをCSV保存
out = []
for cn,res in fit_results.items():
    out.append({'cluster':cn, 'rho_s':res['rho_s_fit'],
                'r_s':res['r_s'], 'chi2':res['chi2'],
                'source':sources.get(cn,'')})
pd.DataFrame(out).to_csv('psz2_nfw_params.csv', index=False)
print('\n保存: psz2_nfw_params.csv')
