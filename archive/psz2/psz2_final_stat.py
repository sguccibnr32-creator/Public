import numpy as np
from scipy import stats

# 全Q値（NFW更新後）
Q_all = {
    'J0201':3.695,'J0231':0.077,'J1311':10.915,'J2337':2.524,
    'G167':115.917,'G273':2.239,'G286':np.nan,
    'G223':1.455,'G228':6.920,'G231':4.573,
}

print('=== Q値統計（全サンプル）===')
valid = [q for q in Q_all.values() if np.isfinite(q)]
print(f'有効N={len(valid)}, Q>1: {sum(1 for q in valid if q>1)}件')
t,p = stats.ttest_1samp(valid,1.0)
st,pw = stats.wilcoxon([q-1 for q in valid],alternative='greater')
print(f't検定  p={p:.4f}')
print(f'Wilcoxon  p={pw:.4f}')

print()
print('=== G167除外（ko≈0 外れ値）===')
valid2 = [q for cn,q in Q_all.items() if np.isfinite(q) and cn!='G167']
print(f'有効N={len(valid2)}, Q>1: {sum(1 for q in valid2 if q>1)}件, 平均Q={np.mean(valid2):.3f}')
t2,p2 = stats.ttest_1samp(valid2,1.0)
st2,pw2 = stats.wilcoxon([q-1 for q in valid2],alternative='greater')
print(f't検定  t={t2:.3f}  p={p2:.4f}')
print(f'Wilcoxon  stat={st2:.1f}  p={pw2:.4f}')

print()
print('=== J0231除外（Q=0.077 内側N=20・不安定）===')
valid3 = [q for cn,q in Q_all.items() if np.isfinite(q) and cn not in ['G167','J0231']]
print(f'有効N={len(valid3)}, Q>1: {sum(1 for q in valid3 if q>1)}件, 平均Q={np.mean(valid3):.3f}')
t3,p3 = stats.ttest_1samp(valid3,1.0)
st3,pw3 = stats.wilcoxon([q-1 for q in valid3],alternative='greater')
print(f't検定  t={t3:.3f}  p={p3:.4f}')
print(f'Wilcoxon  stat={st3:.1f}  p={pw3:.4f}')

print()
print('=== PSZ2のみ（G167除外）===')
PSZ2_Q = {'G273':2.239,'G223':1.455,'G228':6.920,'G231':4.573}
vp = list(PSZ2_Q.values())
print(f'N={len(vp)}, Q>1: {sum(1 for q in vp if q>1)}件, 平均Q={np.mean(vp):.3f}')
tp,pp = stats.ttest_1samp(vp,1.0)
stp,pwp = stats.wilcoxon([q-1 for q in vp],alternative='greater')
print(f't検定  t={tp:.3f}  p={pp:.4f}')
print(f'Wilcoxon  stat={stp:.1f}  p={pwp:.4f}')

print()
print('=== 主解析：Miyaoka4 + PSZ2_4（G167・G286・J0231除外）===')
main_Q = {'J0201':3.695,'J1311':10.915,'J2337':2.524,
          'G273':2.239,'G223':1.455,'G228':6.920,'G231':4.573}
vm = list(main_Q.values())
print(f'N={len(vm)}, Q>1: {sum(1 for q in vm if q>1)}件, 平均Q={np.mean(vm):.3f}')
tm,pm = stats.ttest_1samp(vm,1.0)
stm,pwm = stats.wilcoxon([q-1 for q in vm],alternative='greater')
print(f't検定  t={tm:.3f}  p={pm:.4f}')
print(f'Wilcoxon  stat={stm:.1f}  p={pwm:.4f}')
