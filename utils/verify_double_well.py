import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# v3.0のU(ε)とその構造
# -----------------------------------------------
def U(eps, clip=1e-6):
    eps = np.clip(eps, 0, 1-clip)
    return -eps - eps**2/2 - np.log(1-eps)

def dU(eps, clip=1e-6):
    eps = np.clip(eps, 0, 1-clip)
    return -1 - eps + 1/(1-eps)

def d2U(eps, clip=1e-6):
    eps = np.clip(eps, 0, 1-clip)
    return -1 + 1/(1-eps)**2

# テイラー展開（ε=0付近）
def U_taylor(eps):
    return eps**3/3 + eps**4/4 + eps**5/5

# -----------------------------------------------
# 自由エネルギー F = U(ε) - x・ε （x = g_N/g_c）
# -----------------------------------------------
def F(eps, x, clip=1e-6):
    return U(eps, clip) - x * eps

def dF(eps, x, clip=1e-6):
    return dU(eps, clip) - x

# -----------------------------------------------
# 平衡点の探索（dF/dε = 0）
# -----------------------------------------------
def find_equilibria(x, n_init=100):
    """全ての平衡点を探索"""
    eps_grid = np.linspace(0.001, 0.999, n_init)
    equilibria = []
    for e0 in eps_grid:
        try:
            sol = fsolve(lambda e: dF(e, x), e0,
                        full_output=True)
            e_eq = sol[0][0]
            if 0 < e_eq < 1:
                # 重複除去
                is_new = all(abs(e_eq - eq) > 1e-6
                            for eq in equilibria)
                if is_new:
                    equilibria.append(e_eq)
        except:
            pass
    return sorted(equilibria)

# -----------------------------------------------
# 二安定ポテンシャルの候補：U_modified
# -----------------------------------------------
def U_modified(eps, alpha=0.5, clip=1e-6):
    """
    二安定構造を持つ修正ポテンシャル
    U_mod = U(eps) - alpha * eps^2 * (1-eps)

    alpha=0：v3.0のU(ε)（単安定）
    alpha>0：ε=0 と ε=ε* の二安定構造
    """
    eps = np.clip(eps, 0, 1-clip)
    return U(eps, clip) - alpha * eps**2 * (1-eps)

def dU_modified(eps, alpha=0.5, clip=1e-6):
    eps = np.clip(eps, 0, 1-clip)
    return dU(eps, clip) - alpha * (2*eps*(1-eps) - eps**2)

def d2U_modified(eps, alpha=0.5, clip=1e-6):
    eps = np.clip(eps, 0, 1-clip)
    return d2U(eps, clip) - alpha * (2*(1-eps) - 4*eps + 2*eps)

# -----------------------------------------------
# ヒステリシスループの計算
# -----------------------------------------------
def hysteresis_loop(x_array, alpha=0.5):
    """
    x = g_N/g_c を増加→減少させた時の
    ε_eq の経路（ヒステリシス）
    """
    eps_up   = []  # x増加経路
    eps_down = []  # x減少経路

    # 初期状態：ε=0（折り畳み）
    e_current = 0.01

    for x in x_array:
        # 現在の状態付近の安定点を探す
        candidates = []
        for e0 in [e_current, 0.01, 0.99]:
            try:
                sol = fsolve(lambda e: dU_modified(e, alpha) - x,
                             e0, full_output=True)
                e_eq = sol[0][0]
                if (0.001 < e_eq < 0.999 and
                    d2U_modified(e_eq, alpha) > 0):  # 安定点
                    candidates.append(e_eq)
            except:
                pass

        if candidates:
            # 現在の状態に最も近い安定点を選択
            e_current = min(candidates,
                           key=lambda e: abs(e-e_current))
        eps_up.append(e_current)

    # x減少経路（初期状態：ε大）
    e_current = eps_up[-1]
    for x in reversed(x_array):
        candidates = []
        for e0 in [e_current, 0.99, 0.01]:
            try:
                sol = fsolve(lambda e: dU_modified(e, alpha) - x,
                             e0, full_output=True)
                e_eq = sol[0][0]
                if (0.001 < e_eq < 0.999 and
                    d2U_modified(e_eq, alpha) > 0):
                    candidates.append(e_eq)
            except:
                pass

        if candidates:
            e_current = min(candidates,
                           key=lambda e: abs(e-e_current))
        eps_down.append(e_current)

    return np.array(eps_up), np.array(list(reversed(eps_down)))

# -----------------------------------------------
# メインプロット
# -----------------------------------------------
eps = np.linspace(0.001, 0.995, 1000)
x_arr = np.linspace(0, 3, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# -----------------------------------------------
# 1: U(ε)の構造（v3.0）
# -----------------------------------------------
ax = axes[0, 0]
ax.plot(eps, U(eps), 'b-', lw=2, label='U(eps) v3.0')
ax.plot(eps, U_taylor(eps), 'r--', lw=1.5,
        label='eps^3/3 (Taylor)')
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.axvline(0, color='k', lw=0.8, ls=':')
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 2.0)
ax.set_xlabel('epsilon')
ax.set_ylabel('U(epsilon)')
ax.set_title('U(eps) v3.0\neps=0 is minimum (U=0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# U'(ε)の符号確認
ax2 = ax.twinx()
ax2.plot(eps, dU(eps), 'g-', lw=1, alpha=0.5,
         label="dU/deps")
ax2.axhline(0, color='g', lw=0.5, ls='--')
ax2.set_ylabel("dU/deps", color='g')
ax2.legend(loc='upper right', fontsize=7)

# -----------------------------------------------
# 2: 修正ポテンシャル（alpha別）
# -----------------------------------------------
ax = axes[0, 1]
for alpha in [0.0, 0.3, 0.6, 1.0, 1.5]:
    ax.plot(eps, U_modified(eps, alpha),
            lw=1.5, label=f'alpha={alpha}')
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('epsilon')
ax.set_ylabel('U_modified(epsilon)')
ax.set_title('Modified U: alpha増加で\n二安定構造が出現するか')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -----------------------------------------------
# 3: d²U/dε² の符号（安定性）
# -----------------------------------------------
ax = axes[0, 2]
for alpha in [0.0, 0.6, 1.0, 1.5]:
    d2 = np.array([d2U_modified(e, alpha) for e in eps])
    ax.plot(eps, d2, lw=1.5, label=f'alpha={alpha}')
ax.axhline(0, color='k', lw=1.5, ls='--',
           label='stability boundary')
ax.set_xlim(0, 1)
ax.set_ylim(-2, 5)
ax.set_xlabel('epsilon')
ax.set_ylabel("d2U/deps2")
ax.set_title('安定性（d2U>0が安定）\nd2U<0の領域で二安定が可能')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -----------------------------------------------
# 4: 自由エネルギー F = U - x*eps の地形
# -----------------------------------------------
ax = axes[1, 0]
for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
    ax.plot(eps, F(eps, x), lw=1.5,
            label=f'x=g_N/g_c={x}')
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xlim(0, 1)
ax.set_ylim(-3, 1)
ax.set_xlabel('epsilon')
ax.set_ylabel('F = U - x*eps')
ax.set_title('自由エネルギー地形（v3.0）\nx増加で平衡点が移動')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -----------------------------------------------
# 5: ヒステリシスループ（修正ポテンシャル）
# -----------------------------------------------
ax = axes[1, 1]
for alpha in [0.5, 0.8, 1.2]:
    try:
        up, down = hysteresis_loop(x_arr, alpha)
        ax.plot(x_arr, up, '-',
                lw=1.5, label=f'alpha={alpha} (up)')
        ax.plot(x_arr, down, '--',
                lw=1.5, label=f'alpha={alpha} (down)')
    except:
        pass

ax.set_xlabel('x = g_N / g_c')
ax.set_ylabel('epsilon_eq')
ax.set_title('ヒステリシスループ\n（条件8との整合性）')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -----------------------------------------------
# 6: 二安定状態の模式図
# -----------------------------------------------
ax = axes[1, 2]

# 概念図：ポテンシャルエネルギーの谷
eps_schematic = np.linspace(0, 1, 500)
# 二安定ポテンシャルの例
U_schematic = (2*(eps_schematic - 0.3)**2 *
               (eps_schematic - 0.85)**2 * 20 - 0.3)

ax.plot(eps_schematic, U_schematic, 'k-', lw=2)
ax.axhline(-0.3, color='gray', lw=0.5, ls=':')

# 二つの谷
ax.annotate('', xy=(0.3, -0.3), xytext=(0.3, 0.3),
            arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('', xy=(0.85, -0.3), xytext=(0.85, 0.3),
            arrowprops=dict(arrowstyle='->', color='red'))

ax.text(0.30, -0.45, 'eps=0\n(折り畳み)\n局所安定',
        ha='center', fontsize=9, color='blue')
ax.text(0.85, -0.45, 'eps=1\n(展開)\n大域安定?',
        ha='center', fontsize=9, color='red')

# エネルギー障壁
ax.annotate('エネルギー障壁\n(条件12)',
            xy=(0.57, 0.5), fontsize=8,
            ha='center', color='green')

ax.set_xlabel('epsilon')
ax.set_ylabel('U_effective')
ax.set_title('可能性C：二安定構造の概念図\n条件8・12との接続')
ax.set_xlim(0, 1.1)
ax.set_ylim(-0.7, 1.0)
ax.grid(True, alpha=0.3)

plt.suptitle('可能性C検証：U(epsilon)の二安定構造\n'
             'eps=0（局所安定・折り畳み）vs eps→1（大域安定・展開）',
             fontsize=12)
plt.tight_layout()
plt.savefig('double_well_verification.png', dpi=150)
plt.close()
print("-> double_well_verification.png saved")

# -----------------------------------------------
# 数値的結論
# -----------------------------------------------
print(f"\n{'='*60}")
print("  U(eps)の構造的分析")
print('='*60)

print("\n【v3.0のU(eps)のテイラー展開（eps=0付近）】")
print("  U(eps) = eps^3/3 + eps^4/4 + ...")
print("  -> eps=0でU=0（最小値）")
print("  -> eps>0でU>0（単調増加）")
print("  -> 単安定：eps=0のみが安定点")

print("\n【d2U/deps2の符号】")
eps_test = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
for e in eps_test:
    d2 = d2U(e)
    print(f"  eps={e:.2f}: d2U={d2:+.4f} "
          f"({'安定' if d2>0 else '不安定'})")

print("\n【二安定に必要な条件】")
print("  d2U < 0 の領域が存在する必要がある")
print(f"  v3.0のU(eps): d2U = -1 + 1/(1-eps)^2")
print(f"  d2U=0 のとき: eps = 1 - 1 = 0")
print(f"  -> eps>0で常にd2U>0")
print(f"  -> v3.0のU(eps)は単安定")

print("\n【可能性Cに必要な修正】")
print("  U_mod(eps) = U(eps) - alpha*eps^2*(1-eps)")
print("  alphaが十分大きい時にd2U_mod<0の領域が出現")
for alpha in [0.3, 0.6, 1.0, 1.5]:
    d2_min = min(d2U_modified(e, alpha)
                for e in np.linspace(0.01, 0.99, 1000))
    has_unstable = d2_min < 0
    print(f"  alpha={alpha}: d2U_min={d2_min:.4f} "
          f"-> {'二安定可能' if has_unstable else '単安定'}")

print(f"\n{'='*60}")
