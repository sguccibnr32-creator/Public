#!/usr/bin/env python3
"""
Phase B Step 3: Three-field analysis (G09 + G12 + G15) with Bin 5 exclusion.
"""

import numpy as np
import os
import sys
import time
from collections import defaultdict

G_SI    = 6.67430e-11
Msun    = 1.98892e30
Mpc_m   = 3.08568e22
c_ms    = 2.99792e8
H0      = 70.0
a0_si   = 1.2e-10
deg2rad = np.pi / 180.0

N_GBAR_BINS = 15
GBAR_MIN = 1e-15
GBAR_MAX = 5e-12

MSTAR_EDGES_FULL = [8.5, 10.3, 10.6, 10.8, 11.0, 12.0]
MSTAR_EDGES_CUT  = [8.5, 10.3, 10.6, 10.8, 11.0]

ZBIN_ZEFF = {1: 0.44, 2: 0.75, 3: 1.01, 4: 1.30}
MULT_BIAS = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

Om0 = 0.3; OL0 = 0.7

FIELDS = {
    'G09': {'ra_min': 129.0, 'ra_max': 141.0, 'dec_min': -2.0, 'dec_max': 3.0},
    'G12': {'ra_min': 174.0, 'ra_max': 186.0, 'dec_min': -3.0, 'dec_max': 2.0},
    'G15': {'ra_min': 211.5, 'ra_max': 223.5, 'dec_min': -2.0, 'dec_max': 3.0},
}

N_JK_RA  = 4
N_JK_DEC = 2
N_JK_PER_FIELD = N_JK_RA * N_JK_DEC


def _E(z): return np.sqrt(Om0*(1+z)**3 + OL0)


def comoving_dist(z, n=500):
    if z <= 0: return 0.0
    zz = np.linspace(0, z, n)
    return (c_ms/1e3/H0) * np.trapezoid(1.0/_E(zz), zz)


_zt = np.linspace(0, 2, 2000)
_ct = np.array([comoving_dist(z) for z in _zt])


def _chi(z): return np.interp(z, _zt, _ct)
def ang_diam_dist(z): return _chi(z)/(1+z)


def Sigma_crit_inv(zl, zs):
    if zs <= zl + 0.05: return 0.0
    cl, cs = _chi(zl), _chi(zs)
    Dl = cl/(1+zl)*Mpc_m; Ds = cs/(1+zs)*Mpc_m
    Dls = (cs-cl)/(1+zs)*Mpc_m
    if Dls <= 0 or Ds <= 0: return 0.0
    return 4*np.pi*G_SI/c_ms**2 * Dl*Dls/Ds


BASE = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン"
HSC_FILE = r"E:\スバル望遠鏡データ\931720.csv.gz.1"
GAMA_DIR = r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\GAMA_DR4"
OUT_DIR  = os.path.join(BASE, "phase_b_output")
os.makedirs(OUT_DIR, exist_ok=True)


def build_all_lens_catalogs(logm_max=11.0):
    from astropy.io import fits as afits

    print("=== Building lens catalogs for all fields ===")
    print(f"  logM* cut: < {logm_max}")

    sci_file   = os.path.join(GAMA_DIR, "gkvScienceCatv02.fits")
    mass_file  = os.path.join(GAMA_DIR, "StellarMassesGKVv24.fits")
    match_file = os.path.join(GAMA_DIR, "gkvGamaIIMatchesv01.fits")
    g3c_file   = os.path.join(GAMA_DIR, "G3CGalv10.fits")

    print("  Loading gkvScienceCat...")
    with afits.open(sci_file) as h:
        sci = h[1].data
        sci_uid = np.array(sci['uberID'], dtype=np.int64)
        sci_ra  = np.array(sci['RAcen'], dtype=np.float64)
        sci_dec = np.array(sci['Deccen'], dtype=np.float64)
        sci_z   = np.array(sci['Z'], dtype=np.float64)
        sci_nq  = np.array(sci['NQ'], dtype=np.int32)
        sci_uc  = np.array(sci['uberclass'], dtype=np.int32)

    print(f"    {len(sci_uid)} rows")

    print("  Loading StellarMasses...")
    with afits.open(mass_file) as h:
        mass = h[1].data
        mass_uid = np.array(mass['uberID'], dtype=np.int64)
        mass_logm = np.array(mass['logmstar'], dtype=np.float64)

    print("  Loading gkvGamaIIMatches...")
    with afits.open(match_file) as h:
        mat = h[1].data
        mat_uid = np.array(mat['uberID'], dtype=np.int64)
        mat_cataid = np.array(mat['CATAID'], dtype=np.int64)

    print("  Loading G3CGal...")
    with afits.open(g3c_file) as h:
        g3c = h[1].data
        g3c_cataid = np.array(g3c['CATAID'], dtype=np.int64)
        g3c_rank = np.array(g3c['RankIterCen'], dtype=np.int32)
        g3c_gid = np.array(g3c['GroupID'], dtype=np.int64)

    print("  Building lookup dicts...")
    # Valid mass entries only (logmstar > 0)
    m_mask = (mass_uid > 0)
    mass_dict = dict(zip(mass_uid[m_mask], mass_logm[m_mask]))

    # Valid CATAID matches only
    u_mask = mat_cataid > 0
    uid2cat = dict(zip(mat_uid[u_mask], mat_cataid[u_mask]))

    # G3C - need unique by CATAID
    _, uq_idx = np.unique(g3c_cataid, return_index=True)
    cat2group = {int(g3c_cataid[i]): (int(g3c_gid[i]), int(g3c_rank[i]))
                 for i in uq_idx}

    lens_catalogs = {}
    for field_name, bounds in FIELDS.items():
        ra_min, ra_max = bounds['ra_min'], bounds['ra_max']
        dec_min, dec_max = bounds['dec_min'], bounds['dec_max']

        # Vectorized field + quality cuts
        in_field = ((sci_ra >= ra_min) & (sci_ra <= ra_max) &
                    (sci_dec >= dec_min) & (sci_dec <= dec_max))
        in_uclass = np.isin(sci_uc, [1, 2])
        in_nq = sci_nq >= 3
        in_z = (sci_z >= 0.05) & (sci_z <= 0.5)

        base_mask = in_field & in_uclass & in_nq & in_z
        idx_candidate = np.where(base_mask)[0]

        ras, decs, zs, logms = [], [], [], []
        for i in idx_candidate:
            uid = int(sci_uid[i])
            lm = mass_dict.get(uid, np.nan)
            if np.isnan(lm): continue
            if lm >= logm_max: continue

            # Isolation via G3C
            cataid = uid2cat.get(uid, -1)
            if cataid >= 0:
                ginfo = cat2group.get(cataid, None)
                if ginfo is not None:
                    gid, rank = ginfo
                    # Not BCG and in group -> skip
                    if gid != 0 and rank != 1:
                        continue

            ras.append(sci_ra[i]); decs.append(sci_dec[i])
            zs.append(sci_z[i]); logms.append(lm)

        ras = np.array(ras); decs = np.array(decs)
        zs = np.array(zs); logms = np.array(logms)

        print(f"  {field_name}: field={in_field.sum()} -> "
              f"base={base_mask.sum()} -> isolated+M*<{logm_max}={len(ras)}")

        lens_catalogs[field_name] = {
            'ra': ras, 'dec': decs, 'z': zs, 'logm': logms,
            'Mgal': 10**logms * Msun
        }

    return lens_catalogs


def load_hsc_all_fields():
    print(f"\n=== Loading HSC sources (all 3 fields) ===")
    t0 = time.time()

    sources = {fn: {'ra':[], 'dec':[], 'e1':[], 'e2':[], 'w':[], 'zbin':[]}
               for fn in FIELDS}
    row_count = 0

    # Detect gzip
    with open(HSC_FILE, 'rb') as fb:
        is_gz = (fb.read(2) == b'\x1f\x8b')

    if is_gz:
        import gzip
        fh = gzip.open(HSC_FILE, 'rt', errors='replace')
    else:
        fh = open(HSC_FILE, 'r', encoding='utf-8', errors='replace')

    with fh as f:
        header = None
        first_data = None
        for line in f:
            if line.startswith('#'):
                header = line.lstrip('#').strip().split(',')
                continue
            else:
                if header is None:
                    header = line.strip().split(',')
                else:
                    first_data = line
                break

        col = {name.strip(): i for i, name in enumerate(header)}
        idx_ra   = col.get('i_ra')
        idx_dec  = col.get('i_dec')
        idx_e1   = col.get('i_hsmshaperegauss_e1')
        idx_e2   = col.get('i_hsmshaperegauss_e2')
        idx_w    = col.get('i_hsmshaperegauss_derived_weight')
        idx_zbin = col.get('hsc_y3_zbin')

        def process(line):
            nonlocal row_count
            parts = line.strip().split(',')
            try:
                ra  = float(parts[idx_ra])
                dec = float(parts[idx_dec])
            except (ValueError, IndexError):
                return
            matched = None
            for fn, b in FIELDS.items():
                if b['ra_min'] <= ra <= b['ra_max'] and b['dec_min'] <= dec <= b['dec_max']:
                    matched = fn
                    break
            if matched is None:
                return
            try:
                e1 = float(parts[idx_e1])
                e2 = float(parts[idx_e2])
                w  = float(parts[idx_w])
                zb = int(float(parts[idx_zbin]))
            except (ValueError, IndexError):
                return
            if w <= 0 or zb < 1 or zb > 4:
                return
            s = sources[matched]
            s['ra'].append(ra); s['dec'].append(dec)
            s['e1'].append(e1); s['e2'].append(e2)
            s['w'].append(w); s['zbin'].append(zb)

        if first_data is not None:
            row_count += 1
            process(first_data)

        for line in f:
            row_count += 1
            process(line)

            if row_count % 5_000_000 == 0:
                counts = {fn: len(sources[fn]['ra']) for fn in FIELDS}
                dt = time.time() - t0
                print(f"  {row_count/1e6:.0f}M rows, {counts}, {dt:.0f}s", flush=True)

    result = {}
    for fn in FIELDS:
        s = sources[fn]
        ra  = np.array(s['ra'],  dtype=np.float64)
        dec = np.array(s['dec'], dtype=np.float64)
        e1  = np.array(s['e1'],  dtype=np.float64)
        e2  = np.array(s['e2'],  dtype=np.float64)
        w   = np.array(s['w'],   dtype=np.float64)
        zb  = np.array(s['zbin'],dtype=np.int32)

        if len(w) > 0:
            e_sq = e1**2 + e2**2
            e_rms_sq = np.sum(w * e_sq) / np.sum(w)
            R_resp = 1.0 - e_rms_sq
        else:
            R_resp = 0.7

        result[fn] = {'ra': ra, 'dec': dec, 'e1': e1, 'e2': e2,
                      'w': w, 'zbin': zb, 'R_resp': R_resp}
        print(f"  {fn}: {len(ra):,} sources, R_resp={R_resp:.4f}")

    dt = time.time() - t0
    print(f"  Total: {row_count/1e6:.1f}M rows, {dt:.0f}s")
    return result


def compute_field_esd(lenses, sources, field_name, mstar_edges):
    from scipy.spatial import cKDTree

    l_ra, l_dec, l_z = lenses['ra'], lenses['dec'], lenses['z']
    l_logm, l_Mgal = lenses['logm'], lenses['Mgal']
    s_ra, s_dec = sources['ra'], sources['dec']
    s_e1, s_e2 = sources['e1'], sources['e2']
    s_w, s_zbin = sources['w'], sources['zbin']
    R_resp = sources['R_resp']

    n_lens = len(l_ra)
    n_mbins = len(mstar_edges) - 1
    bounds = FIELDS[field_name]

    print(f"\n--- {field_name}: {n_lens} lenses, {len(s_ra):,} sources ---")

    if n_lens == 0 or len(s_ra) == 0:
        print(f"  SKIP: insufficient data")
        return None

    gbar_edges = np.logspace(np.log10(GBAR_MIN), np.log10(GBAR_MAX), N_GBAR_BINS+1)
    gbar_centers = np.sqrt(gbar_edges[:-1] * gbar_edges[1:])

    esd_num   = np.zeros(N_GBAR_BINS)
    esd_x_num = np.zeros(N_GBAR_BINS)
    esd_den   = np.zeros(N_GBAR_BINS)
    esd_npair = np.zeros(N_GBAR_BINS, dtype=np.int64)
    esd_var   = np.zeros(N_GBAR_BINS)

    esd_num_m   = np.zeros((n_mbins, N_GBAR_BINS))
    esd_den_m   = np.zeros((n_mbins, N_GBAR_BINS))
    esd_npair_m = np.zeros((n_mbins, N_GBAR_BINS), dtype=np.int64)
    esd_var_m   = np.zeros((n_mbins, N_GBAR_BINS))

    jk_num = np.zeros((N_JK_PER_FIELD, N_GBAR_BINS))
    jk_den = np.zeros((N_JK_PER_FIELD, N_GBAR_BINS))
    jk_num_m = np.zeros((N_JK_PER_FIELD, n_mbins, N_GBAR_BINS))
    jk_den_m = np.zeros((N_JK_PER_FIELD, n_mbins, N_GBAR_BINS))

    def get_patch(ra, dec):
        ira  = int(np.clip((ra - bounds['ra_min']) / (bounds['ra_max']-bounds['ra_min']) * N_JK_RA, 0, N_JK_RA-1))
        idec = int(np.clip((dec- bounds['dec_min'])/ (bounds['dec_max']-bounds['dec_min'])* N_JK_DEC,0, N_JK_DEC-1))
        return idec * N_JK_RA + ira

    s_xyz = np.column_stack([
        np.cos(s_dec*deg2rad)*np.cos(s_ra*deg2rad),
        np.cos(s_dec*deg2rad)*np.sin(s_ra*deg2rad),
        np.sin(s_dec*deg2rad)])
    tree = cKDTree(s_xyz)

    # Precompute Sigma_crit per lens
    sc_inv_table = np.zeros((n_lens, 5))
    for i in range(n_lens):
        for zb in range(1, 5):
            sc_inv_table[i, zb] = Sigma_crit_inv(l_z[i], ZBIN_ZEFF[zb])

    total_pairs = 0
    t0 = time.time()

    for i in range(n_lens):
        zl = l_z[i]; Dl = ang_diam_dist(zl); Mgal = l_Mgal[i]; lm = l_logm[i]
        patch = get_patch(l_ra[i], l_dec[i])

        mbin = -1
        for mb in range(n_mbins):
            if mstar_edges[mb] <= lm < mstar_edges[mb+1]:
                mbin = mb; break

        lx = np.array([np.cos(l_dec[i]*deg2rad)*np.cos(l_ra[i]*deg2rad),
                        np.cos(l_dec[i]*deg2rad)*np.sin(l_ra[i]*deg2rad),
                        np.sin(l_dec[i]*deg2rad)])

        th_max = min(3.0/Dl, np.pi/2)
        ch_max = 2*np.sin(th_max/2)
        idx = tree.query_ball_point(lx, ch_max)
        if len(idx) == 0: continue
        idx = np.array(idx)

        cos_th = np.clip(s_xyz[idx,0]*lx[0]+s_xyz[idx,1]*lx[1]+s_xyz[idx,2]*lx[2], -1, 1)
        theta = np.arccos(cos_th)
        R_Mpc = theta * Dl
        R_m = R_Mpc * Mpc_m

        mask_R = (R_Mpc >= 0.03) & (R_Mpc <= 3.0)
        if not np.any(mask_R): continue

        idx_g = idx[mask_R]; R_m_g = R_m[mask_R]
        gbar = G_SI * Mgal / R_m_g**2
        gb_bin = np.searchsorted(gbar_edges, gbar) - 1
        valid = (gb_bin >= 0) & (gb_bin < N_GBAR_BINS)
        if not np.any(valid): continue

        idx_v = idx_g[valid]; gb_v = gb_bin[valid]

        dra = (s_ra[idx_v]-l_ra[i])*np.cos(l_dec[i]*deg2rad)*deg2rad
        ddec= (s_dec[idx_v]-l_dec[i])*deg2rad
        phi = np.arctan2(ddec, dra)
        c2p = np.cos(2*phi); s2p = np.sin(2*phi)

        et = -(s_e1[idx_v]*c2p + s_e2[idx_v]*s2p) / 2.0
        ex =  (s_e1[idx_v]*s2p - s_e2[idx_v]*c2p) / 2.0

        # Vectorized Sigma_crit lookup
        szb_v = s_zbin[idx_v]
        sc_inv_v = sc_inv_table[i, szb_v]
        good = sc_inv_v > 0
        if not np.any(good): continue

        idx_v = idx_v[good]; gb_v = gb_v[good]
        et = et[good]; ex = ex[good]
        szb_v = szb_v[good]; sc_inv_v = sc_inv_v[good]

        m_b = np.array([MULT_BIAS.get(int(z), 0.0) for z in szb_v])
        corr = (1.0+m_b)*R_resp
        Sc_v = 1.0/sc_inv_v
        W = s_w[idx_v] * sc_inv_v**2
        et_c = et/corr; ex_c = ex/corr
        val_t = W*et_c*Sc_v
        val_x = W*ex_c*Sc_v
        var_c = W**2 * (et_c*Sc_v)**2

        np.add.at(esd_num, gb_v, val_t)
        np.add.at(esd_x_num, gb_v, val_x)
        np.add.at(esd_den, gb_v, W)
        np.add.at(esd_npair, gb_v, 1)
        np.add.at(esd_var, gb_v, var_c)
        np.add.at(jk_num[patch], gb_v, val_t)
        np.add.at(jk_den[patch], gb_v, W)

        if mbin >= 0:
            np.add.at(esd_num_m[mbin], gb_v, val_t)
            np.add.at(esd_den_m[mbin], gb_v, W)
            np.add.at(esd_npair_m[mbin], gb_v, 1)
            np.add.at(esd_var_m[mbin], gb_v, var_c)
            np.add.at(jk_num_m[patch, mbin], gb_v, val_t)
            np.add.at(jk_den_m[patch, mbin], gb_v, W)

        total_pairs += len(et)
        if (i+1) % 2000 == 0:
            dt = time.time()-t0
            print(f"  {field_name}: {i+1}/{n_lens}, {total_pairs/1e6:.2f}M pairs, {dt:.0f}s", flush=True)

    dt = time.time()-t0
    print(f"  {field_name}: done, {total_pairs/1e6:.2f}M pairs, {dt:.0f}s")

    mv = esd_den > 0
    esd = np.full(N_GBAR_BINS, np.nan); esd_x = np.full(N_GBAR_BINS, np.nan)
    esd[mv] = esd_num[mv]/esd_den[mv]; esd_x[mv] = esd_x_num[mv]/esd_den[mv]

    jk_esd = np.zeros((N_JK_PER_FIELD, N_GBAR_BINS))
    for p in range(N_JK_PER_FIELD):
        n_loo = esd_num - jk_num[p]; d_loo = esd_den - jk_den[p]
        g = d_loo > 0
        jk_esd[p,g] = n_loo[g]/d_loo[g]; jk_esd[p,~g] = np.nan
    jk_mean = np.nanmean(jk_esd, axis=0)
    jk_cov = np.zeros((N_GBAR_BINS, N_GBAR_BINS))
    for p in range(N_JK_PER_FIELD):
        d = jk_esd[p]-jk_mean; d[np.isnan(d)] = 0
        jk_cov += np.outer(d, d)
    jk_cov *= (N_JK_PER_FIELD-1)/N_JK_PER_FIELD
    jk_err = np.sqrt(np.diag(jk_cov))

    gobs = 4*G_SI*esd; gobs_x = 4*G_SI*esd_x
    gobs_err = 4*G_SI*jk_err
    ana_err = np.full(N_GBAR_BINS, np.nan)
    ana_err[mv] = np.sqrt(esd_var[mv])/esd_den[mv]
    bad = (gobs_err <= 0)|np.isnan(gobs_err)
    gobs_err[bad] = 4*G_SI*ana_err[bad]

    mbins_res = []
    for mb in range(n_mbins):
        mv_m = esd_den_m[mb] > 0
        e_mb = np.full(N_GBAR_BINS, np.nan)
        e_mb[mv_m] = esd_num_m[mb,mv_m]/esd_den_m[mb,mv_m]

        jk_mb = np.zeros((N_JK_PER_FIELD, N_GBAR_BINS))
        for p in range(N_JK_PER_FIELD):
            n_l = esd_num_m[mb]-jk_num_m[p,mb]; d_l = esd_den_m[mb]-jk_den_m[p,mb]
            g = d_l > 0; jk_mb[p,g] = n_l[g]/d_l[g]; jk_mb[p,~g] = np.nan
        jk_m_mean = np.nanmean(jk_mb, axis=0)
        jk_v = np.zeros(N_GBAR_BINS)
        for p in range(N_JK_PER_FIELD):
            d = jk_mb[p]-jk_m_mean; d[np.isnan(d)] = 0; jk_v += d**2
        jk_v *= (N_JK_PER_FIELD-1)/N_JK_PER_FIELD
        ge_mb = 4*G_SI*np.sqrt(jk_v)
        ae = np.full(N_GBAR_BINS, np.nan)
        ae[mv_m] = np.sqrt(esd_var_m[mb,mv_m])/esd_den_m[mb,mv_m]
        bad_m = (ge_mb <= 0)|np.isnan(ge_mb)
        ge_mb[bad_m] = 4*G_SI*ae[bad_m]

        mbins_res.append({
            'mlo': mstar_edges[mb], 'mhi': mstar_edges[mb+1],
            'gobs': 4*G_SI*e_mb, 'gobs_err': ge_mb,
            'npair': esd_npair_m[mb]
        })

    return {
        'field': field_name, 'gbar': gbar_centers,
        'gobs': gobs, 'gobs_x': gobs_x, 'gobs_err': gobs_err,
        'npair': esd_npair, 'n_lens': n_lens, 'n_pairs': total_pairs,
        'jk_cov_gobs': jk_cov * (4*G_SI)**2,
        'mbins': mbins_res,
        '_esd_num': esd_num, '_esd_den': esd_den, '_esd_var': esd_var,
        '_esd_num_m': esd_num_m, '_esd_den_m': esd_den_m,
        '_esd_var_m': esd_var_m, '_esd_npair_m': esd_npair_m,
        '_jk_num': jk_num, '_jk_den': jk_den,
        '_jk_num_m': jk_num_m, '_jk_den_m': jk_den_m,
    }


def combine_fields(field_results, mstar_edges):
    print("\n=== Combining all fields ===")
    n_mbins = len(mstar_edges) - 1
    gbar = field_results[0]['gbar']

    esd_num = sum(r['_esd_num'] for r in field_results)
    esd_den = sum(r['_esd_den'] for r in field_results)
    esd_var = sum(r['_esd_var'] for r in field_results)
    esd_npair = sum(r['npair'] for r in field_results)

    mv = esd_den > 0
    esd = np.full(N_GBAR_BINS, np.nan)
    esd[mv] = esd_num[mv]/esd_den[mv]

    all_jk_num = np.concatenate([r['_jk_num'] for r in field_results], axis=0)
    all_jk_den = np.concatenate([r['_jk_den'] for r in field_results], axis=0)
    n_jk_all = all_jk_num.shape[0]

    jk_esd = np.zeros((n_jk_all, N_GBAR_BINS))
    for p in range(n_jk_all):
        n_l = esd_num - all_jk_num[p]; d_l = esd_den - all_jk_den[p]
        g = d_l > 0; jk_esd[p,g] = n_l[g]/d_l[g]; jk_esd[p,~g] = np.nan
    jk_mean = np.nanmean(jk_esd, axis=0)
    jk_cov = np.zeros((N_GBAR_BINS, N_GBAR_BINS))
    for p in range(n_jk_all):
        d = jk_esd[p]-jk_mean; d[np.isnan(d)] = 0
        jk_cov += np.outer(d, d)
    jk_cov *= (n_jk_all-1)/n_jk_all
    jk_err = np.sqrt(np.diag(jk_cov))

    gobs = 4*G_SI*esd
    gobs_err = 4*G_SI*jk_err
    ana_err = np.full(N_GBAR_BINS, np.nan)
    ana_err[mv] = np.sqrt(esd_var[mv])/esd_den[mv]
    bad = (gobs_err <= 0)|np.isnan(gobs_err)
    gobs_err[bad] = 4*G_SI*ana_err[bad]

    mbins_comb = []
    for mb in range(n_mbins):
        en_m = sum(r['_esd_num_m'][mb] for r in field_results)
        ed_m = sum(r['_esd_den_m'][mb] for r in field_results)
        ev_m = sum(r['_esd_var_m'][mb] for r in field_results)
        np_m = sum(r['_esd_npair_m'][mb] for r in field_results)

        mv_m = ed_m > 0
        e_mb = np.full(N_GBAR_BINS, np.nan)
        e_mb[mv_m] = en_m[mv_m]/ed_m[mv_m]

        all_jk_nm = np.concatenate([r['_jk_num_m'][:,mb,:] for r in field_results], axis=0)
        all_jk_dm = np.concatenate([r['_jk_den_m'][:,mb,:] for r in field_results], axis=0)
        jk_mb = np.zeros((n_jk_all, N_GBAR_BINS))
        for p in range(n_jk_all):
            nl = en_m - all_jk_nm[p]; dl = ed_m - all_jk_dm[p]
            g = dl > 0; jk_mb[p,g] = nl[g]/dl[g]; jk_mb[p,~g] = np.nan
        jkm = np.nanmean(jk_mb, axis=0)
        jkv = np.zeros(N_GBAR_BINS)
        for p in range(n_jk_all):
            d = jk_mb[p]-jkm; d[np.isnan(d)] = 0; jkv += d**2
        jkv *= (n_jk_all-1)/n_jk_all
        ge_mb = 4*G_SI*np.sqrt(jkv)
        ae_m = np.full(N_GBAR_BINS, np.nan)
        ae_m[mv_m] = np.sqrt(ev_m[mv_m])/ed_m[mv_m]
        bad_m = (ge_mb <= 0)|np.isnan(ge_mb)
        ge_mb[bad_m] = 4*G_SI*ae_m[bad_m]

        mbins_comb.append({
            'mlo': mstar_edges[mb], 'mhi': mstar_edges[mb+1],
            'gobs': 4*G_SI*e_mb, 'gobs_err': ge_mb, 'npair': np_m
        })

    n_total_lens = sum(r['n_lens'] for r in field_results)
    n_total_pairs = sum(r['n_pairs'] for r in field_results)
    print(f"  Combined: {n_total_lens} lenses, {n_total_pairs/1e6:.2f}M pairs")

    return {
        'field': 'ALL', 'gbar': gbar,
        'gobs': gobs, 'gobs_err': gobs_err, 'npair': esd_npair,
        'jk_cov_gobs': jk_cov*(4*G_SI)**2,
        'mbins': mbins_comb,
        'n_lens': n_total_lens, 'n_pairs': n_total_pairs,
        'n_jk': n_jk_all
    }


def fit_gc(gbar, gobs, gobs_err):
    from scipy.optimize import minimize_scalar, brentq
    mask = np.isfinite(gbar)&np.isfinite(gobs)&np.isfinite(gobs_err)
    mask &= (gobs>0)&(gbar>0)&(gobs_err>0)
    if np.sum(mask)<3: return np.nan, np.nan
    gb,go,ge = gbar[mask],gobs[mask],gobs_err[mask]
    def chi2(lgc):
        gc=10**lgc; m=gb/(1-np.exp(-np.sqrt(gb/gc))); return np.sum(((go-m)/ge)**2)
    res = minimize_scalar(chi2, bounds=(-14,-8), method='bounded')
    gc = 10**res.x; c2min = res.fun
    try: lhi = brentq(lambda x: chi2(x)-c2min-1, res.x, res.x+2)
    except: lhi = res.x+0.5
    try: llo = brentq(lambda x: chi2(x)-c2min-1, res.x-2, res.x)
    except: llo = res.x-0.5
    return gc, (10**lhi-10**llo)/2


def chi2_comparison(gbar, gobs, gobs_err, jk_cov=None):
    mask = np.isfinite(gbar)&np.isfinite(gobs)&np.isfinite(gobs_err)
    mask &= (gobs>0)&(gbar>0)&(gobs_err>0)
    if np.sum(mask)<3: return None
    gb,go,ge = gbar[mask],gobs[mask],gobs_err[mask]
    idx_m = np.where(mask)[0]

    use_cov = False; ci = None
    if jk_cov is not None:
        cov = jk_cov[np.ix_(idx_m,idx_m)].copy()
        diag = np.diag(cov); bad = diag<=0
        if np.any(bad):
            for k in np.where(bad)[0]: cov[k,k] = ge[k]**2
        try: ci = np.linalg.inv(cov); use_cov = True
        except: pass

    def c2(model):
        r = go-model
        return float(r@ci@r) if use_cov else float(np.sum((r/ge)**2))

    c2_mond = c2(gb/(1-np.exp(-np.sqrt(gb/a0_si))))
    gc,gce = fit_gc(gbar, gobs, gobs_err)
    c2_c15 = c2(gb/(1-np.exp(-np.sqrt(gb/gc)))) if np.isfinite(gc) else np.inf
    n = int(np.sum(mask))
    return {'chi2_mond':c2_mond,'chi2_c15':c2_c15,'gc':gc,'gc_err':gce,
            'dchi2':c2_mond-c2_c15,'dAIC':c2_mond-c2_c15-2,'n':n,'cov':use_cov}


def gc_slope(gbar, mbins_res):
    logm_c, gc_vals, gc_errs = [],[],[]
    for r in mbins_res:
        mask = np.isfinite(r['gobs'])&(r['gobs_err']>0)&(r['npair']>30)
        if np.sum(mask)<3: continue
        gc,gce = fit_gc(gbar[mask], r['gobs'][mask], r['gobs_err'][mask])
        if np.isfinite(gc) and gc > 0 and gce > 0:
            logm_c.append((r['mlo']+r['mhi'])/2)
            gc_vals.append(gc/a0_si); gc_errs.append(gce/a0_si)
    if len(logm_c)<2: return None
    x = np.array(logm_c); y = np.log10(gc_vals)
    ye = np.array(gc_errs)/(np.array(gc_vals)*np.log(10))
    w = 1/ye**2; S=np.sum(w); Sx=np.sum(w*x); Sxx=np.sum(w*x**2); Sxy=np.sum(w*x*y)
    det = S*Sxx-Sx**2
    if det<=0: return None
    sl = (S*Sxy-Sx*np.sum(w*y))/det; sle = np.sqrt(S/det)
    return sl, sle, x, np.array(gc_vals), np.array(gc_errs)


def plot_results(field_results, combined):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gbar = combined['gbar']
    gb_p = np.logspace(-5, 2, 200)*a0_si
    g_mond = gb_p/(1-np.exp(-np.sqrt(gb_p/a0_si)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0,0]
    colors_f = {'G09':'blue','G12':'green','G15':'red'}
    for r in field_results:
        fn = r['field']
        m = np.isfinite(r['gobs'])&(r['npair']>50)&(r['gobs_err']>0)
        if np.any(m):
            ax.errorbar(gbar[m]/a0_si, r['gobs'][m]/a0_si, yerr=r['gobs_err'][m]/a0_si,
                        fmt='o', ms=4, color=colors_f[fn], alpha=0.7, capsize=2,
                        label=f"{fn} ({r['n_lens']} L)")
    ax.plot(gb_p/a0_si, g_mond/a0_si, 'k-', lw=2, alpha=0.5, label='MOND')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar/a0'); ax.set_ylabel('g_obs/a0')
    ax.set_title('Per-field comparison'); ax.legend(fontsize=7)
    ax.set_xlim(1e-5,1e2); ax.set_ylim(1e-3,1e2)

    ax = axes[0,1]
    r = combined
    m = np.isfinite(r['gobs'])&(r['npair']>100)&(r['gobs_err']>0)
    if np.any(m):
        ax.errorbar(gbar[m]/a0_si, r['gobs'][m]/a0_si, yerr=r['gobs_err'][m]/a0_si,
                    fmt='s', ms=5, color='black', capsize=3, label=f"Combined ({r['n_lens']} L)")
    ax.plot(gb_p/a0_si, g_mond/a0_si, 'r-', lw=2, label='MOND')
    ax.plot(gb_p/a0_si, gb_p/a0_si, 'k--', lw=1, alpha=0.3)
    chi2r = chi2_comparison(gbar, r['gobs'], r['gobs_err'], r['jk_cov_gobs'])
    if chi2r and np.isfinite(chi2r['gc']):
        gc = chi2r['gc']
        g_c15 = gb_p/(1-np.exp(-np.sqrt(gb_p/gc)))
        ax.plot(gb_p/a0_si, g_c15/a0_si, 'b--', lw=2,
                label=f"C15 gc={gc/a0_si:.2f}a0")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar/a0'); ax.set_ylabel('g_obs/a0')
    ax.set_title(f"Combined 3-field (Bin5 excl)"); ax.legend(fontsize=7)
    ax.set_xlim(1e-5,1e2); ax.set_ylim(1e-3,1e2)

    colors_m = ['purple','blue','green','orange']
    for mb_i, rb in enumerate(combined['mbins'][:4]):
        ax = axes[(mb_i+2)//3, (mb_i+2)%3]
        m = np.isfinite(rb['gobs'])&(rb['npair']>30)&(rb['gobs_err']>0)
        gc,gce = fit_gc(gbar[m], rb['gobs'][m], rb['gobs_err'][m]) if np.any(m) else (np.nan,np.nan)
        lab = f"gc={gc/a0_si:.2f}+/-{gce/a0_si:.2f}" if np.isfinite(gc) else "?"
        if np.any(m):
            ax.errorbar(gbar[m]/a0_si, rb['gobs'][m]/a0_si, yerr=rb['gobs_err'][m]/a0_si,
                        fmt='o', ms=4, color=colors_m[mb_i], capsize=2, label=lab)
        ax.plot(gb_p/a0_si, g_mond/a0_si, 'r-', lw=1.5, alpha=0.5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('g_bar/a0'); ax.set_ylabel('g_obs/a0')
        ax.set_title(f"Bin {mb_i+1}: [{rb['mlo']:.1f},{rb['mhi']:.1f})")
        ax.legend(fontsize=7); ax.set_xlim(1e-5,1e2); ax.set_ylim(1e-3,1e2)

    plt.tight_layout()
    fp = os.path.join(OUT_DIR, "phase_b_three_field_rar.png")
    plt.savefig(fp, dpi=150); plt.close()
    print(f"  Saved: {fp}")


def main():
    print("="*64)
    print("Phase B Step 3: Three-field (G09+G12+G15), Bin5 excluded")
    print("="*64)

    lens_cats = build_all_lens_catalogs(logm_max=11.0)
    source_cats = load_hsc_all_fields()

    mstar_edges = MSTAR_EDGES_CUT
    field_results = []
    for fn in ['G09', 'G12', 'G15']:
        if len(lens_cats[fn]['ra']) == 0 or len(source_cats[fn]['ra']) == 0:
            print(f"  {fn}: SKIP (no data)")
            continue
        res = compute_field_esd(lens_cats[fn], source_cats[fn], fn, mstar_edges)
        if res is not None:
            field_results.append(res)

    if len(field_results) == 0:
        print("ERROR: No fields produced results")
        return

    print("\n=== Per-field results ===")
    gbar = field_results[0]['gbar']
    for r in field_results:
        chi2r = chi2_comparison(gbar, r['gobs'], r['gobs_err'], r['jk_cov_gobs'])
        gc_a0 = chi2r['gc']/a0_si if chi2r and np.isfinite(chi2r['gc']) else np.nan
        gce   = chi2r['gc_err']/a0_si if chi2r and np.isfinite(chi2r['gc_err']) else np.nan
        daic  = chi2r['dAIC'] if chi2r else np.nan
        print(f"  {r['field']}: {r['n_lens']} L, {r['n_pairs']/1e6:.2f}M pairs, "
              f"gc={gc_a0:.2f}+/-{gce:.2f} a0, dAIC={daic:+.1f}")

    print("\n=== Field consistency ===")
    gc_per = []
    for r in field_results:
        c = chi2_comparison(gbar, r['gobs'], r['gobs_err'], r['jk_cov_gobs'])
        if c and np.isfinite(c['gc']):
            gc_per.append((r['field'], c['gc']/a0_si, c['gc_err']/a0_si))
    if len(gc_per) >= 2:
        vals = [x[1] for x in gc_per]; errs = [x[2] for x in gc_per]
        w = [1/e**2 for e in errs]
        gc_wm = sum(v*wi for v,wi in zip(vals,w))/sum(w)
        gc_wm_err = 1/np.sqrt(sum(w))
        chi2_cons = sum(((v-gc_wm)/e)**2 for v,e in zip(vals,errs))
        ndof_cons = len(vals)-1
        from scipy.stats import chi2 as chi2_dist
        p_cons = chi2_dist.sf(chi2_cons, ndof_cons)
        for fn, gc, gce in gc_per:
            print(f"  {fn}: gc = {gc:.3f} +/- {gce:.3f} a0")
        print(f"  Weighted mean: gc = {gc_wm:.3f} +/- {gc_wm_err:.3f} a0")
        print(f"  Consistency chi2={chi2_cons:.2f} (dof={ndof_cons}, p={p_cons:.3f})")
        print(f"  -> Fields {'CONSISTENT' if p_cons > 0.05 else 'INCONSISTENT'}")

    combined = combine_fields(field_results, mstar_edges)

    print("\n=== Combined chi^2: C15 vs MOND ===")
    chi2r = chi2_comparison(gbar, combined['gobs'], combined['gobs_err'],
                             combined['jk_cov_gobs'])
    if chi2r:
        print(f"  MOND:  chi2={chi2r['chi2_mond']:.1f} (dof={chi2r['n']})")
        print(f"  C15:   chi2={chi2r['chi2_c15']:.1f} (dof={chi2r['n']-1}, "
              f"gc={chi2r['gc']/a0_si:.3f}+/-{chi2r['gc_err']/a0_si:.3f} a0)")
        print(f"  Dchi2={chi2r['dchi2']:+.1f}, DAIC={chi2r['dAIC']:+.1f}")
        print(f"  Cov: {'full JK' if chi2r['cov'] else 'diagonal'} "
              f"({combined['n_jk']} patches)")

    print("\n=== Combined gc-M* slope ===")
    sr = gc_slope(gbar, combined['mbins'])
    if sr:
        sl, sle, x, gcv, gce = sr
        print(f"  slope = {sl:+.3f} +/- {sle:.3f}")
        from scipy.stats import norm
        t_c15 = (sl-0.075)/sle; t_m = sl/sle
        print(f"  vs C15 (+0.075): t={t_c15:+.2f}, p={2*norm.sf(abs(t_c15)):.3f}")
        print(f"  vs MOND (0.000): t={t_m:+.2f},  p={2*norm.sf(abs(t_m)):.3f}")
        for i in range(len(x)):
            print(f"    Bin {i+1} (logM*={x[i]:.1f}): gc={gcv[i]:.3f}+/-{gce[i]:.3f} a0")

    print("\n=== Saving ===")
    for r in field_results:
        fp = os.path.join(OUT_DIR, f"phase_b_{r['field']}_rar.txt")
        with open(fp, 'w') as f:
            f.write(f"# {r['field']}: {r['n_lens']} lenses, {r['n_pairs']} pairs\n")
            f.write("# g_bar  g_obs  g_obs_err  N_pairs\n")
            for k in range(N_GBAR_BINS):
                f.write(f"{gbar[k]:.6e}  {r['gobs'][k]:.6e}  "
                        f"{r['gobs_err'][k]:.6e}  {r['npair'][k]}\n")

    fp = os.path.join(OUT_DIR, "phase_b_combined_rar.txt")
    with open(fp, 'w') as f:
        f.write(f"# Combined 3-field: {combined['n_lens']} lenses, "
                f"{combined['n_pairs']} pairs, Bin5 excluded\n")
        f.write("# g_bar  g_obs  g_obs_err  N_pairs\n")
        for k in range(N_GBAR_BINS):
            f.write(f"{gbar[k]:.6e}  {combined['gobs'][k]:.6e}  "
                    f"{combined['gobs_err'][k]:.6e}  {combined['npair'][k]}\n")

    for mb, rb in enumerate(combined['mbins']):
        fp = os.path.join(OUT_DIR, f"phase_b_combined_mbin_{mb+1}.txt")
        with open(fp, 'w') as f:
            f.write(f"# Combined Bin {mb+1} [{rb['mlo']:.1f},{rb['mhi']:.1f})\n")
            f.write("# g_bar  g_obs  g_obs_err  N_pairs\n")
            for k in range(N_GBAR_BINS):
                f.write(f"{gbar[k]:.6e}  {rb['gobs'][k]:.6e}  "
                        f"{rb['gobs_err'][k]:.6e}  {rb['npair'][k]}\n")

    fp = os.path.join(OUT_DIR, "phase_b_combined_jk_cov.txt")
    np.savetxt(fp, combined['jk_cov_gobs'],
               header=f"Combined JK covariance, {combined['n_jk']} patches")

    plot_results(field_results, combined)
    print("\n=== Phase B Step 3 complete ===")


if __name__ == '__main__':
    main()
