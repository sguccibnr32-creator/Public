#!/usr/bin/env python3
"""
Condition-14 non-linear membrane stiffness systematic test.
Uses TA3+phase1.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr
from numpy.linalg import lstsq
import csv, warnings
warnings.filterwarnings('ignore')

BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
ROTMOD = os.path.join(BASE, "Rotmod_LTG")
PHASE1 = os.path.join(BASE, "phase1", "sparc_results.csv")
TA3 = os.path.join(BASE, "TA3_gc_independent.csv")
a0 = 1.2e-10
kpc_m = 3.086e19

def load_pipeline():
    data={}
    with open(PHASE1,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name=row.get('galaxy','').strip()
            try: data[name]={'vflat':float(row.get('vflat','0')),'Yd':float(row.get('ud','0.5'))}
            except: pass
    with open(TA3,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name=row.get('galaxy','').strip()
            try:
                gc_a0=float(row.get('gc_over_a0','0'))
                if name in data and gc_a0>0: data[name]['gc']=gc_a0*a0
            except: pass
    return {k:v for k,v in data.items() if 'gc' in v and v['vflat']>0}

def load_rotcurve(gname):
    fname=os.path.join(ROTMOD,f"{gname}_rotmod.dat")
    if not os.path.exists(fname): return None
    rad,vobs,vgas,vdisk,vbul=[],[],[],[],[]
    with open(fname,'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts)<6: continue
            try:
                rad.append(float(parts[0])); vobs.append(float(parts[1]))
                vgas.append(float(parts[3])); vdisk.append(float(parts[4])); vbul.append(float(parts[5]))
            except: continue
    if len(rad)<5: return None
    return {'r':np.array(rad),'vobs':np.array(vobs),'vgas':np.array(vgas),
            'vdisk':np.array(vdisk),'vbul':np.array(vbul)}

def compute_gc_deep(rc,Yd=0.5):
    r_m=rc['r']*kpc_m
    v_bar2=(Yd*rc['vdisk']**2+rc['vgas']**2+rc['vbul']**2)*1e6
    v_obs2=(rc['vobs']*1e3)**2
    gN=v_bar2/r_m; gobs=v_obs2/r_m
    mask=gN>0; gc_pts=gobs[mask]**2/gN[mask]
    if len(gc_pts)<3: return None
    gc_med=np.median(gc_pts)
    if gc_med<=0: return None
    x=gN[mask]/gc_med; deep=x<1.0
    return np.median(gc_pts[deep]) if np.sum(deep)>=3 else gc_med

def compute_hR(rc,Yd):
    vds=np.sqrt(max(Yd,0.01))*np.abs(rc['vdisk'])
    rpk=rc['r'][np.argmax(vds)]
    if rpk<0.01 or rpk>=rc['r'].max()*0.9: return None
    return rpk/2.15

def compute_extended_strain(rc,gc_deep,Yd=0.5):
    r_kpc=rc['r']; r_m=r_kpc*kpc_m
    v_bar2=(Yd*rc['vdisk']**2+rc['vgas']**2+rc['vbul']**2)*1e6
    v_obs2=(rc['vobs']*1e3)**2
    gN=np.maximum(v_bar2/r_m,0)
    epsilon=np.sqrt(gN/gc_deep)
    deps=np.zeros_like(epsilon)
    deps[0]=(epsilon[1]-epsilon[0])/(r_m[1]-r_m[0])
    deps[-1]=(epsilon[-1]-epsilon[-2])/(r_m[-1]-r_m[-2])
    for i in range(1,len(epsilon)-1):
        deps[i]=(epsilon[i+1]-epsilon[i-1])/(r_m[i+1]-r_m[i-1])
    E_grad=np.mean(deps**2); E_local=np.mean(epsilon**2)
    ratio=E_grad/E_local if E_local>0 else 0
    eps_max=np.max(epsilon); eps_mean=np.mean(epsilon)
    eps_outer=np.mean(epsilon[-max(1,len(epsilon)//3):])
    u_e=np.mean(epsilon**2); u_c=np.mean(epsilon**3)
    E_kin=np.mean(v_obs2)
    sk=u_e/(E_kin/kpc_m**2) if E_kin>0 else 0
    gn=np.std(deps)/(np.mean(np.abs(deps))+1e-50)
    if len(epsilon)>=5:
        d2=np.gradient(np.gradient(epsilon,r_m),r_m)
        curv=np.mean(d2**2)
    else: curv=0
    return {'ratio':ratio,'E_grad':E_grad,'E_local':E_local,
            'eps_max':eps_max,'eps_mean':eps_mean,'eps_outer':eps_outer,
            'u_elastic':u_e,'u_cubic':u_c,'strain_kin':sk,
            'grad_nonuniform':gn,'curvature':curv}

def build_dataset():
    pipe=load_pipeline()
    results=[]
    for gname,gd in sorted(pipe.items()):
        gc_obs=gd['gc']; vflat=gd['vflat']; Yd=gd.get('Yd',0.5)
        if gc_obs<=0 or vflat<=0: continue
        rc=load_rotcurve(gname)
        if rc is None: continue
        hR=compute_hR(rc,Yd)
        if hR is None: continue
        gc_deep=compute_gc_deep(rc,Yd=Yd)
        if gc_deep is None or gc_deep<=0: continue
        sq=compute_extended_strain(rc,gc_deep,Yd=Yd)
        if sq['ratio']==0: continue
        results.append({'name':gname,'gc_obs':gc_obs,'gc_deep':gc_deep,
                        'vflat':vflat,'hR':hR,'Yd':Yd,
                        'rmax_hR':rc['r'][-1]/hR if hR>0 else 0,**sq})
    print(f"Dataset: {len(results)} galaxies\n")
    return results

def aicc(chi2,n,k):
    aic=chi2+2*k
    return aic+2*k*(k+1)/(n-k-1) if n-k-1>0 else aic

class ModelTester:
    def __init__(self,data):
        self.N=len(data)
        self.gc_obs=np.array([d['gc_obs'] for d in data])
        self.gc_deep=np.array([d['gc_deep'] for d in data])
        self.log_excess=np.log10(self.gc_obs/self.gc_deep)
        self.ratio=np.array([d['ratio'] for d in data])
        self.eps_max=np.array([d['eps_max'] for d in data])
        self.eps_mean=np.array([d['eps_mean'] for d in data])
        self.u_elastic=np.array([d['u_elastic'] for d in data])
        self.u_cubic=np.array([d['u_cubic'] for d in data])
        self.strain_kin=np.array([d['strain_kin'] for d in data])
        self.grad_nonuniform=np.array([d['grad_nonuniform'] for d in data])
        self.curvature=np.array([d['curvature'] for d in data])
        self.vflat=np.array([d['vflat'] for d in data])
        self.hR=np.array([d['hR'] for d in data])
        self.chi2_null=np.sum(self.log_excess**2)
        self.mse_null=np.mean(self.log_excess**2)
        print(f"log_excess: mean={np.mean(self.log_excess):.4f}, std={np.std(self.log_excess):.4f}")

    def test_model(self,name,n_params,fit_func,predict_func,loo=True):
        print(f"\n{'-'*60}")
        print(f"Model: {name} ({n_params} params)")
        print(f"{'-'*60}")
        params=fit_func()
        if params is None: print("  FIT FAILED"); return None
        pred=predict_func(params)
        resid=self.log_excess-pred
        chi2=np.sum(resid**2)
        R2=1-chi2/self.chi2_null
        _aicc=aicc(chi2,self.N,n_params)
        _aicc_null=aicc(self.chi2_null,self.N,0)
        dAICc=_aicc-_aicc_null
        print(f"  params: {params}")
        print(f"  chi2={chi2:.3f}, R2={R2:.4f}, dAICc={dAICc:+.1f}")
        print(f"  resid std={np.std(resid):.4f}")

        log_vf=np.log10(self.vflat); log_hR=np.log10(self.hR)
        gc_model=self.gc_deep*10**pred
        log_gc_m=np.log10(np.maximum(gc_model,1e-30))
        def pcorr(x,y,z):
            cx=np.polyfit(z,x,1); cy=np.polyfit(z,y,1)
            return spearmanr(x-np.polyval(cx,z),y-np.polyval(cy,z))
        rho_obs,_=pcorr(np.log10(self.gc_obs),log_hR,log_vf)
        rho_model,_=pcorr(log_gc_m,log_hR,log_vf)
        print(f"  hR partial: gc_obs={rho_obs:+.3f}, model={rho_model:+.3f}")

        if loo and self.N<=200:
            loo_err=[]
            for i in range(self.N):
                mask=np.ones(self.N,dtype=bool); mask[i]=False
                p_loo=fit_func(mask)
                if p_loo is None: continue
                pi=predict_func(p_loo,idx=i)
                val=pi[0] if hasattr(pi,'__len__') else pi
                loo_err.append((self.log_excess[i]-val)**2)
            mse_loo=np.mean(loo_err) if loo_err else 999
            print(f"  LOO-CV MSE={mse_loo:.6f} (null={self.mse_null:.6f}, ratio={mse_loo/self.mse_null:.4f})")

        return {'name':name,'n_params':n_params,'chi2':chi2,'R2':R2,
                'dAICc':dAICc,'rho_hR':rho_model,'params':params,
                'resid_std':np.std(resid)}

    def test_M1(self):
        r=self.ratio; le=self.log_excess
        def fit(mask=None):
            rr=r if mask is None else r[mask]
            e=le if mask is None else le[mask]
            def chi2(k):
                return np.sum((e-np.log10(np.maximum(1+k*rr,1e-10)))**2)
            rsc=np.median(rr); km=5.0/rsc if rsc>0 else 1e40
            res=minimize_scalar(chi2,bounds=(-km,km),method='bounded')
            return [res.x]
        def pred(p,idx=None):
            k=p[0]; rr=r if idx is None else np.array([r[idx]])
            return np.log10(np.maximum(1+k*rr,1e-10))
        return self.test_model("M1: kappa=const",1,fit,pred)

    def test_M2(self):
        r=self.ratio; le=self.log_excess
        def fit(mask=None):
            rr=r if mask is None else r[mask]
            e=le if mask is None else le[mask]
            def chi2(p):
                k0,n=p
                val=k0*np.abs(rr)**n*np.sign(rr)
                return np.sum((e-np.log10(np.maximum(1+val,1e-10)))**2)
            best=None
            for n0 in [0.3,0.5,1.0,1.5,2.0]:
                k0_i=0.01/(np.median(np.abs(rr))**n0+1e-50)
                for ks in [1,-1]:
                    try:
                        res=minimize(chi2,[ks*k0_i,n0],method='Nelder-Mead',
                                     options={'maxiter':5000})
                        if best is None or res.fun<best.fun: best=res
                    except: pass
            return list(best.x) if best else None
        def pred(p,idx=None):
            k0,n=p
            rr=r if idx is None else np.array([r[idx]])
            val=k0*np.abs(rr)**n*np.sign(rr)
            return np.log10(np.maximum(1+val,1e-10))
        return self.test_model("M2: k0*ratio^n",2,fit,pred)

    def test_M3(self):
        em_all=self.eps_max; le=self.log_excess
        def fit(mask=None):
            em=em_all if mask is None else em_all[mask]
            e=le if mask is None else le[mask]
            lem=np.log10(np.maximum(em,1e-10))
            def chi2(p):
                log_ec,beta=p
                excess=np.maximum(lem-log_ec,0)
                return np.sum((e-beta*excess)**2)
            best=None
            for lec in np.linspace(np.percentile(lem,10),np.percentile(lem,90),10):
                for b in [-1,-0.5,0.5,1]:
                    try:
                        res=minimize(chi2,[lec,b],method='Nelder-Mead',options={'maxiter':3000})
                        if best is None or res.fun<best.fun: best=res
                    except: pass
            return list(best.x) if best else None
        def pred(p,idx=None):
            log_ec,beta=p
            em=em_all if idx is None else np.array([em_all[idx]])
            lem=np.log10(np.maximum(em,1e-10))
            return beta*np.maximum(lem-log_ec,0)
        return self.test_model("M3: eps threshold",2,fit,pred)

    def test_M4(self):
        r=self.ratio; em=self.eps_mean; le=self.log_excess
        def fit(mask=None):
            rr=r if mask is None else r[mask]
            emm=em if mask is None else em[mask]
            e=le if mask is None else le[mask]
            eps_ref=np.median(emm)
            def chi2(p):
                k0,m=p
                mod=k0*rr*(emm/eps_ref)**m
                return np.sum((e-np.log10(np.maximum(1+mod,1e-10)))**2)
            k_i=0.01/(np.median(rr)+1e-50)
            best=None
            for m0 in [-2,-1,0,1,2]:
                for ks in [1,-1]:
                    try:
                        res=minimize(chi2,[ks*k_i,m0],method='Nelder-Mead',options={'maxiter':5000})
                        if best is None or res.fun<best.fun: best=res
                    except: pass
            return list(best.x) if best else None
        def pred(p,idx=None):
            k0,m=p
            rr=r if idx is None else np.array([r[idx]])
            emm=em if idx is None else np.array([em[idx]])
            eps_ref=np.median(self.eps_mean)
            mod=k0*rr*(emm/eps_ref)**m
            return np.log10(np.maximum(1+mod,1e-10))
        return self.test_model("M4: k0*ratio*(eps/eps_ref)^m",2,fit,pred)

    def _linear_model(self,name,qty):
        le=self.log_excess
        def fit(mask=None):
            q=qty if mask is None else qty[mask]
            e=le if mask is None else le[mask]
            log_q=np.log10(np.maximum(q,1e-80))
            fin=np.isfinite(log_q)
            if fin.sum()<10: return None
            X=log_q[fin].reshape(-1,1)
            sol,_,_,_=lstsq(X,e[fin],rcond=None)
            return [sol[0]]
        def pred(p,idx=None):
            a=p[0]
            q=qty if idx is None else np.array([qty[idx]])
            return a*np.log10(np.maximum(q,1e-80))
        return self.test_model(name,1,fit,pred)

    def test_M5(self): return self._linear_model("M5: <eps^3>/<eps^2>",self.u_cubic/(self.u_elastic+1e-50))
    def test_M6(self): return self._linear_model("M6: u_strain/E_kin",self.strain_kin)
    def test_M7(self): return self._linear_model("M7: grad nonuniform",self.grad_nonuniform)
    def test_M8(self): return self._linear_model("M8: eps curvature",self.curvature)

def main():
    print("="*70)
    print("Condition-14 non-linear membrane stiffness systematic test")
    print("="*70)
    data=build_dataset()
    mt=ModelTester(data)

    results=[]
    for test_fn in [mt.test_M1,mt.test_M2,mt.test_M3,mt.test_M4,
                     mt.test_M5,mt.test_M6,mt.test_M7,mt.test_M8]:
        try:
            r=test_fn()
            if r: results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n"+"="*70); print("Model comparison summary"); print("="*70)
    print(f"{'Model':<35s} {'k':>2s} {'dAICc':>7s} {'R^2':>7s} {'rho(hR)':>8s} {'std':>7s}")
    print("-"*70)
    print(f"{'M0: null':<35s} {'0':>2s} {'0.0':>7s} {'0.0':>7s} {'--':>8s} {np.std(mt.log_excess):>7.4f}")
    for r in sorted(results,key=lambda x:x['dAICc']):
        print(f"{r['name']:<35s} {r['n_params']:>2d} {r['dAICc']:>+7.1f} "
              f"{r['R2']:>7.4f} {r['rho_hR']:>+8.3f} {r['resid_std']:>7.4f}")

    if results:
        best=min(results,key=lambda x:x['dAICc'])
        print(f"\nBest: {best['name']} (dAICc={best['dAICc']:+.1f})")
        if best['dAICc']>=0: print("-> null (gc=gc_deep) is best. All models rejected.")
        elif best['dAICc']>-2: print("-> Weak evidence only.")
        elif best['dAICc']>-6: print("-> Moderate evidence.")
        else: print("-> Strong evidence.")

    print("\n[DONE]")

if __name__=='__main__': main()
