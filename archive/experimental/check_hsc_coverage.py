# check_hsc_coverage.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# HSC密度マップ読み込み
# -----------------------------------------------
DENSITY_MAP = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\hscssp_pdr2_wide_densitymap.csv"

print("HSC密度マップ読み込み中...")
dm = pd.read_csv(DENSITY_MAP)

print(f"行数：{len(dm)}")
print(f"列名：{list(dm.columns)}")
print(f"\n先頭5行：")
print(dm.head())
print(f"\n基本統計：")
print(dm.describe())

# -----------------------------------------------
# Miyaoka 22クラスター
# -----------------------------------------------
clusters = [
    ('J0157.4-0550', 29.35125, -5.84000, 0.1289, 1.29),
    ('J0231.7-0451', 37.94708, -4.85583, 0.1843, 2.01),
    ('J0201.7-0212', 30.43417, -2.20083, 0.1960, 4.27),
    ('J1330.8-0152',202.70792, -1.87278, 0.0852, 2.13),
    ('J0158.4-0146', 29.61833, -1.78083, 0.1632, 1.45),
    ('J1258.6-0145',194.67125, -1.75694, 0.0845, 3.47),
    ('J1311.5-0120',197.87500, -1.33528, 0.1832,12.50),
    ('J0153.5-0118', 28.38333, -1.31222, 0.2438, 3.62),
    ('J2337.6+0016',354.41917,  0.27667, 0.2779, 6.90),
    ('J1415.2-0030',213.80917, -0.50111, 0.1403, 1.91),
    ('J0152.7+0100', 28.18167,  1.01611, 0.2270, 5.53),
    ('J0106.8+0103', 16.70958,  1.05472, 0.2537, 5.36),
    ('J1115.8+0129',168.97500,  1.49556, 0.3499,12.30),
    ('J0105.0+0201', 16.25958,  2.03000, 0.1967, 2.54),
    ('J1113.3+0231',168.33625,  2.53222, 0.0780, 1.09),
    ('J1401.0+0252',210.25958,  2.88000, 0.2528, 1.97),
    ('J1200.4+0320',180.10583,  3.33361, 0.1339,36.70),
    ('J2311.5+0338',347.88792,  3.64361, 0.2998,10.40),
    ('J1217.6+0339',184.41917,  3.66250, 0.0766, 2.74),
    ('J1023.6+0411',155.91167,  4.18639, 0.2850,18.10),
    ('J2256.9+0532',344.23792,  5.54694, 0.1696, 1.77),
    ('J1256.4+0440',194.11042,  4.66666, 0.2300, 1.71),
]

# S_plastic既知値
S_plastic_known = {
    'J1023.6+0411': 0.869,
    'J0157.4-0550': 0.840,
    'J1217.6+0339': 0.826,
    'J1311.5-0120': 0.736,
    'J1115.8+0129': 0.729,
    'J0231.7-0451': 0.528,
    'J1258.6-0145': 0.520,
    'J0106.8+0103': 0.484,
    'J1415.2-0030': 0.202,
}

# -----------------------------------------------
# 列名を自動判定してRA/Dec列を特定
# -----------------------------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

ra_col  = find_col(dm, ['ra','RA','i_ra','ra_center','RA_center'])
dec_col = find_col(dm, ['dec','Dec','DEC','i_dec','dec_center','DEC_center'])
den_col = find_col(dm, ['density','nobj','count','ngal','weight','nd3_r10','nd5_r10'])

print(f"\n使用列：RA={ra_col}, Dec={dec_col}, density={den_col}")

if ra_col is None or dec_col is None:
    print("\n[警告] RA/Dec列が自動判定できません")
    print("列名を確認して手動指定してください")
    import sys; sys.exit(0)

ra_map  = dm[ra_col].values
dec_map = dm[dec_col].values

# -----------------------------------------------
# クラスターとHSCピクセルのマッチング
# 各クラスターから30arcmin以内のピクセルを探す
# -----------------------------------------------
MATCH_RADIUS_DEG = 0.5   # 30 arcmin

print(f"\n{'='*70}")
print(f"  Miyaoka 22クラスター × HSC-SSP Wide カバレッジ確認")
print(f"  マッチ半径：{MATCH_RADIUS_DEG*60:.0f} arcmin")
print('='*70)
print(f"{'クラスター':<20} {'RA':>8} {'Dec':>7} {'z':>6} "
      f"{'HSCピクセル数':>13} {'S_plastic':>10} {'判定':>6}")
print('-'*70)

covered       = []
not_covered   = []
results_list  = []

for name, ra, dec, z, lx in clusters:
    # 角距離計算（平面近似）
    cos_dec = np.cos(np.radians(dec))
    dra     = (ra_map - ra) * cos_dec
    ddec    = dec_map - dec
    dist    = np.sqrt(dra**2 + ddec**2)

    n_pix   = np.sum(dist <= MATCH_RADIUS_DEG)
    in_hsc  = n_pix > 0
    sp      = S_plastic_known.get(name, np.nan)
    sp_str  = f"{sp:.3f}" if not np.isnan(sp) else "  -  "
    flag    = 'O' if in_hsc else 'X'

    if in_hsc:
        covered.append(name)
        # 最近接ピクセルの密度
        min_dist = dist.min()
    else:
        not_covered.append(name)
        min_dist = dist.min()

    results_list.append(dict(
        name=name, ra=ra, dec=dec, z=z, lx=lx,
        n_pix=n_pix, in_hsc=in_hsc,
        min_dist_arcmin=min_dist*60,
        s_plastic=sp
    ))

    print(f"{name:<20} {ra:>8.3f} {dec:>7.3f} {z:>6.4f} "
          f"{n_pix:>13d} {sp_str:>10} {flag:>6}")

print('='*70)
print(f"\nHSCカバレッジ内：{len(covered)}/{len(clusters)} クラスター")
print(f"\n[O] Coverage found:")
for c in covered:
    print(f"   {c}"
          + (f"  [S_plastic={S_plastic_known[c]:.3f}]"
             if c in S_plastic_known else ""))

print(f"\n[X] No coverage:")
for c in not_covered:
    r = next(r for r in results_list if r['name']==c)
    print(f"   {c}  (最近接={r['min_dist_arcmin']:.1f}arcmin)")

# -----------------------------------------------
# プロット：HSCフットプリント + クラスター位置
# -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：全天マップ
ax = axes[0]
# HSCピクセル（密度でカラー）
if den_col:
    sc = ax.scatter(ra_map, dec_map,
                    c=dm[den_col].values,
                    s=0.1, cmap='Blues', alpha=0.5,
                    vmin=0, vmax=np.percentile(dm[den_col],95))
    plt.colorbar(sc, ax=ax, label=den_col)
else:
    ax.scatter(ra_map, dec_map, s=0.1,
               c='steelblue', alpha=0.3)

# クラスター位置
for r in results_list:
    color = 'red' if r['in_hsc'] else 'gray'
    ms    = 80    if r['in_hsc'] else 40
    ax.scatter(r['ra'], r['dec'],
               c=color, s=ms, zorder=5,
               marker='*' if r['in_hsc'] else 'x')
    if r['in_hsc']:
        ax.annotate(r['name'].split('.')[0],
                    (r['ra'], r['dec']),
                    fontsize=6, color='red',
                    xytext=(3,3),
                    textcoords='offset points')

ax.set_xlabel('RA [deg]')
ax.set_ylabel('Dec [deg]')
ax.set_title(f'HSC-SSP Wide フットプリント\n'
             f'赤★=Miyaoka HSC内({len(covered)}個) '
             f'灰×=範囲外({len(not_covered)}個)')
ax.invert_xaxis()

# 右：カバレッジ内クラスターの拡大
ax = axes[1]
covered_data = [r for r in results_list if r['in_hsc']]

if covered_data:
    ra_cov  = np.array([r['ra']  for r in covered_data])
    dec_cov = np.array([r['dec'] for r in covered_data])
    sp_cov  = np.array([r['s_plastic'] for r in covered_data])

    # HSCピクセル（カバレッジ内付近のみ）
    ra_range  = [ra_cov.min()-2,  ra_cov.max()+2]
    dec_range = [dec_cov.min()-2, dec_cov.max()+2]
    mask_hsc  = ((ra_map  >= ra_range[0])  &
                 (ra_map  <= ra_range[1])  &
                 (dec_map >= dec_range[0]) &
                 (dec_map <= dec_range[1]))

    if mask_hsc.sum() > 0:
        ax.scatter(ra_map[mask_hsc], dec_map[mask_hsc],
                   s=0.5, c='steelblue', alpha=0.3)

    # クラスターをS_plasticでカラーリング
    mask_sp = ~np.isnan(sp_cov)
    if mask_sp.sum() > 0:
        sc2 = ax.scatter(ra_cov[mask_sp], dec_cov[mask_sp],
                         c=sp_cov[mask_sp], cmap='RdYlGn_r',
                         s=150, zorder=5, marker='*',
                         vmin=0, vmax=1,
                         label='S_plastic既知')
        plt.colorbar(sc2, ax=ax, label='S_plastic')
    if (~mask_sp).sum() > 0:
        ax.scatter(ra_cov[~mask_sp], dec_cov[~mask_sp],
                   c='black', s=100, zorder=5,
                   marker='*', label='S_plastic未知')

    for r in covered_data:
        ax.annotate(r['name'],
                    (r['ra'], r['dec']),
                    fontsize=7, zorder=6,
                    xytext=(4,4),
                    textcoords='offset points')

    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_title('HSCカバレッジ内クラスター\n'
                 '色=S_plastic（赤=多・緑=少）')
    ax.legend(fontsize=8)
    ax.invert_xaxis()
else:
    ax.text(0.5, 0.5, 'カバレッジ内クラスターなし',
            ha='center', va='center',
            transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('hsc_miyaoka_coverage.png', dpi=150)
plt.close()
print("\n→ hsc_miyaoka_coverage.png 保存完了")
