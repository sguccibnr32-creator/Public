# -*- coding: utf-8 -*-
"""
LITTLE THINGS x SPARC Cross-Match & C15 Verification (Stage 1)
"""

import os, csv, math
import numpy as np
from pathlib import Path

a0 = 1.2e-10

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
TA3_FILE    = BASE / "TA3_gc_independent.csv"
PHASE1_FILE = BASE / "phase1" / "sparc_results.csv"
MRT_FILE    = BASE / "SPARC_Lelli2016c.mrt"

OUT_CSV = BASE / "littlethings_sparc_c15.csv"
OUT_TXT = BASE / "littlethings_sparc_c15_report.txt"

LITTLE_THINGS = {
    "CVnIdwA":   ["CVnIdwA", "UGCA292", "UGC07559"],
    "DDO043":    ["DDO043", "DDO43", "UGC03860"],
    "DDO046":    ["DDO046", "DDO46", "UGC03966"],
    "DDO047":    ["DDO047", "DDO47", "UGC03974"],
    "DDO050":    ["DDO050", "DDO50", "UGC04305", "HoII", "HolmbergII"],
    "DDO052":    ["DDO052", "DDO52", "UGC04426"],
    "DDO053":    ["DDO053", "DDO53", "UGC04459"],
    "DDO063":    ["DDO063", "DDO63", "UGC05139", "HoI", "HolmbergI"],
    "DDO069":    ["DDO069", "DDO69", "LeoA", "UGC05364"],
    "DDO070":    ["DDO070", "DDO70", "SextansB", "UGC05373"],
    "DDO075":    ["DDO075", "DDO75", "SextansA", "UGCA205"],
    "DDO087":    ["DDO087", "DDO87", "UGC05918"],
    "DDO101":    ["DDO101", "UGC06900"],
    "DDO126":    ["DDO126", "UGC07559"],
    "DDO133":    ["DDO133", "UGC07698"],
    "DDO154":    ["DDO154", "UGC08024", "NGC4789A"],
    "DDO155":    ["DDO155", "GR8"],
    "DDO168":    ["DDO168", "UGC08320"],
    "DDO210":    ["DDO210", "Aquarius"],
    "DDO216":    ["DDO216", "Pegasus", "UGC12613", "PegDIG"],
    "F564-V3":   ["F564-V3"],
    "Haro29":    ["Haro29", "Mrk209", "UGCA281"],
    "Haro36":    ["Haro36", "UGC07950"],
    "IC10":      ["IC10", "UGC00192"],
    "IC1613":    ["IC1613", "DDO8", "UGC00668"],
    "Mrk178":    ["Mrk178", "UGC06541"],
    "NGC1569":   ["NGC1569", "UGC03056", "VIIZw16"],
    "NGC2366":   ["NGC2366", "UGC03851", "DDO42"],
    "NGC3738":   ["NGC3738", "UGC06565", "Arp234"],
    "UGC8508":   ["UGC8508", "UGC08508", "IZw60"],
    "VIIZw403":  ["VIIZw403", "UGC06456"],
    "WLM":       ["WLM", "DDO221", "UGCA444"],
}


def load_mrt(path):
    """Use established split()-based parser for SPARC_Lelli2016c.mrt.
    Fields: [0]Galaxy [1]T [2]D [4]Inc [7]L36 [9]Reff [10]SBeff
            [11]Rdisk [12]SBdisk0 [13]MHI [14]RHI [15]Vflat [17]Q"""
    data = {}
    if not path.exists():
        print(f"[WARN] MRT not found: {path}")
        return data
    in_data = False
    sep = 0
    with open(path, 'r') as f:
        for line in f:
            if line.strip().startswith('---'):
                sep += 1
                if sep >= 4:
                    in_data = True
                continue
            if not in_data:
                continue
            p = line.split()
            if len(p) < 18:
                continue
            try:
                data[p[0]] = {
                    'Ttype': int(p[1]),
                    'D':     float(p[2]),
                    'Inc':   float(p[4]),
                    'L36':   float(p[7]),
                    'Reff':  float(p[9]),
                    'SBeff': float(p[10]),
                    'Rdisk': float(p[11]),
                    'SBdisk0': float(p[12]),
                    'MHI':   float(p[13]),
                    'RHI':   float(p[14]),
                    'vflat': float(p[15]),
                    'Q':     float(p[17]),
                }
            except (ValueError, IndexError):
                continue
    return data


def load_ta3(path):
    data = {}
    if not path.exists(): return data
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                gc_a0 = float(row.get('gc_over_a0', '0'))
                rs = float(row.get('rs_tanh', '0'))
                if gc_a0 > 0 and name:
                    data[name] = {'gc_a0': gc_a0, 'rs_tanh': rs}
            except: pass
    return data


def load_phase1(path):
    data = {}
    if not path.exists(): return data
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('galaxy', '').strip()
            try:
                ud = float(row.get('ud', '0.5'))
                vf = float(row.get('vflat', '0'))
                if name:
                    data[name] = {'Yd': ud, 'vflat': vf}
            except: pass
    return data


def normalize_name(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


def find_sparc_match(lt_name, lt_aliases, sparc_names):
    sparc_norm = {normalize_name(n): n for n in sparc_names}
    for alias in lt_aliases:
        norm = normalize_name(alias)
        if norm in sparc_norm:
            return sparc_norm[norm]
    return None


def c15_predict(Yd, vflat_kms, hR_kpc):
    vflat_m = vflat_kms * 1e3
    hR_m = hR_kpc * 3.086e19
    Sigma_dyn = vflat_m**2 / hR_m
    gc = 0.584 * Yd**(-0.361) * math.sqrt(a0 * Sigma_dyn)
    return gc


def main():
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 70)
    log("LITTLE THINGS x SPARC Cross-Match & C15 Verification (Stage 1)")
    log("=" * 70)

    mrt = load_mrt(MRT_FILE)
    ta3 = load_ta3(TA3_FILE)
    ph1 = load_phase1(PHASE1_FILE)
    log(f"\nData loaded: MRT={len(mrt)}, TA3={len(ta3)}, phase1={len(ph1)}")

    sparc_all = set(mrt.keys()) | set(ta3.keys()) | set(ph1.keys())
    log(f"Unique SPARC galaxy names: {len(sparc_all)}")

    log("\n" + "-" * 70)
    log("CROSS-MATCH: LITTLE THINGS vs SPARC")
    log("-" * 70)

    matches = []
    no_match = []
    for lt_name, aliases in sorted(LITTLE_THINGS.items()):
        sparc_name = find_sparc_match(lt_name, aliases, sparc_all)
        if sparc_name:
            matches.append((lt_name, sparc_name))
            log(f"  MATCH: {lt_name:15s} -> {sparc_name}")
        else:
            no_match.append(lt_name)

    log(f"\nMatched: {len(matches)} / {len(LITTLE_THINGS)}")
    log(f"No match: {len(no_match)}")
    if no_match:
        log(f"  Unmatched: {', '.join(no_match)}")

    if len(matches) == 0:
        log("\n[ERROR] No matches.")
        with open(OUT_TXT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        return

    log("\n" + "-" * 70)
    log("C15 PARAMETERS FOR MATCHED GALAXIES")
    log("-" * 70)
    log(f"{'LT_name':15s} {'SPARC':15s} {'T':>3s} {'vflat':>6s} "
        f"{'hR':>6s} {'Yd':>5s} {'gc/a0':>7s} "
        f"{'gc_C15':>7s} {'ratio':>7s} {'lgRes':>7s}")

    results = []
    for lt_name, sp_name in matches:
        m = mrt.get(sp_name, {})
        t = ta3.get(sp_name, {})
        p = ph1.get(sp_name, {})

        vflat = p.get('vflat') or m.get('vflat')
        Yd = p.get('Yd', 0.5)
        hR = m.get('Rdisk')
        gc_a0 = t.get('gc_a0')
        ttype = m.get('Ttype')

        missing = []
        if not vflat or vflat <= 0: missing.append('vflat')
        if not hR or hR <= 0: missing.append('hR')
        if not gc_a0 or gc_a0 <= 0: missing.append('gc')

        if missing:
            log(f"  {lt_name:15s} {sp_name:15s} SKIP ({','.join(missing)})")
            continue

        gc_obs = gc_a0 * a0
        gc_pred = c15_predict(Yd, vflat, hR)
        gc_pred_a0 = gc_pred / a0
        ratio = gc_obs / gc_pred
        lg_res = math.log10(ratio)

        rec = {
            'lt_name': lt_name, 'sparc_name': sp_name,
            'Ttype': ttype, 'vflat': vflat, 'hR': hR, 'Yd': Yd,
            'gc_obs_a0': gc_a0, 'gc_c15_a0': gc_pred_a0,
            'ratio': ratio, 'lg_residual': lg_res,
        }
        results.append(rec)

        tt_str = f"{ttype:3d}" if ttype is not None else "N/A"
        log(f"  {lt_name:15s} {sp_name:15s} {tt_str:>3s} {vflat:6.1f} "
            f"{hR:6.2f} {Yd:5.2f} {gc_a0:7.3f} "
            f"{gc_pred_a0:7.3f} {ratio:7.3f} {lg_res:+7.3f}")

    log(f"\nComplete records: {len(results)}")

    if len(results) == 0:
        with open(OUT_TXT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        return

    log("\n" + "-" * 70)
    log("C15 VERIFICATION STATISTICS (LITTLE THINGS subset)")
    log("-" * 70)

    lg_res = np.array([r['lg_residual'] for r in results])
    log(f"N = {len(results)}")
    log(f"log10(gc_obs/gc_C15):")
    log(f"  median = {np.median(lg_res):+.4f} dex")
    log(f"  mean   = {np.mean(lg_res):+.4f} dex")
    log(f"  std    = {np.std(lg_res):.4f} dex")
    log(f"  range  = [{lg_res.min():+.3f}, {lg_res.max():+.3f}]")

    log(f"\nSPARC full-sample C15 scatter: 0.286 dex")
    log(f"LITTLE THINGS subset scatter:  {np.std(lg_res):.3f} dex")
    if np.std(lg_res) < 0.35:
        log(f"  -> Consistent")
    else:
        log(f"  -> ELEVATED")

    if len(results) >= 5:
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(lg_res, 0.0)
            log(f"\nBias test (1-sample t, H0: mean=0):")
            log(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
            if len(results) >= 6:
                w_stat, w_p = stats.wilcoxon(lg_res)
                log(f"  Wilcoxon: W = {w_stat:.1f}, p = {w_p:.4f}")
        except ImportError:
            log("  scipy not available, skipping t-test")

    log("\n" + "-" * 70)
    log("CONTEXT: LT SUBSET vs SPARC FULL SAMPLE")
    log("-" * 70)

    full_results = []
    full_names = []
    for sp_name in ta3:
        m = mrt.get(sp_name, {})
        p = ph1.get(sp_name, {})
        t = ta3[sp_name]
        vflat = p.get('vflat') or m.get('vflat')
        Yd = p.get('Yd', 0.5)
        hR = m.get('Rdisk')
        gc_a0 = t.get('gc_a0')
        if vflat and vflat > 0 and hR and hR > 0 and gc_a0 and gc_a0 > 0:
            gc_pred = c15_predict(Yd, vflat, hR)
            r = gc_a0 * a0 / gc_pred
            full_results.append(math.log10(r))
            full_names.append(sp_name)

    full_res = np.array(full_results)
    log(f"SPARC full sample: N={len(full_res)}, scatter={np.std(full_res):.4f} dex")

    lt_sparc_names = set(sp for _, sp in matches)
    lt_idx = [i for i, n in enumerate(full_names) if n in lt_sparc_names]
    non_lt_idx = [i for i, n in enumerate(full_names) if n not in lt_sparc_names]

    if lt_idx:
        lt_res_full = full_res[lt_idx]
        non_lt_res = full_res[non_lt_idx]
        log(f"\nLT subset in full pipeline: N={len(lt_idx)}")
        log(f"  scatter = {np.std(lt_res_full):.4f} dex")
        log(f"  median  = {np.median(lt_res_full):+.4f} dex")
        log(f"Non-LT remainder: N={len(non_lt_idx)}")
        log(f"  scatter = {np.std(non_lt_res):.4f} dex")
        log(f"  median  = {np.median(non_lt_res):+.4f} dex")

        try:
            from scipy import stats
            if len(lt_idx) >= 3 and len(non_lt_idx) >= 3:
                ks_stat, ks_p = stats.ks_2samp(lt_res_full, non_lt_res)
                log(f"\nKS test (LT vs non-LT residuals):")
                log(f"  D = {ks_stat:.4f}, p = {ks_p:.4f}")
                if ks_p > 0.05:
                    log(f"  -> Same distribution (C15 holds uniformly)")
                else:
                    log(f"  -> Distributions differ")
        except ImportError:
            pass

    log("\n" + "-" * 70)
    log("PHYSICAL PROPERTIES OF MATCHED GALAXIES")
    log("-" * 70)

    vflats = np.array([r['vflat'] for r in results])
    hRs = np.array([r['hR'] for r in results])
    gc_obs_arr = np.array([r['gc_obs_a0'] for r in results])
    log(f"vflat range: [{vflats.min():.1f}, {vflats.max():.1f}] km/s  (SPARC full: ~25-300)")
    log(f"hR range:    [{hRs.min():.2f}, {hRs.max():.2f}] kpc")
    log(f"gc/a0 range: [{gc_obs_arr.min():.3f}, {gc_obs_arr.max():.3f}]  (SPARC median: 0.24)")
    log(f"\nvflat < 75 km/s (dwarf regime): {np.sum(vflats < 75)}/{len(results)}")

    log("\n" + "-" * 70)
    log("ANOMALY CHECK: vflat 50-75 km/s bin")
    log("-" * 70)
    anomaly_bin = [r for r in results if 50 <= r['vflat'] <= 75]
    if anomaly_bin:
        log(f"Galaxies in 50-75 km/s bin: {len(anomaly_bin)}")
        for r in anomaly_bin:
            log(f"  {r['lt_name']:15s} vflat={r['vflat']:.1f} "
                f"gc/a0={r['gc_obs_a0']:.3f} ratio={r['ratio']:.3f} "
                f"lgRes={r['lg_residual']:+.3f}")
        anom_res = np.array([r['lg_residual'] for r in anomaly_bin])
        log(f"  median residual = {np.median(anom_res):+.3f} dex")
    else:
        log("  No galaxies in this bin")

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    log(f"\nCSV saved: {OUT_CSV}")

    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f"Report saved: {OUT_TXT}")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"LITTLE THINGS galaxies in SPARC: {len(matches)}")
    log(f"Complete C15 records:            {len(results)}")
    log(f"C15 scatter (LT subset):         {np.std(lg_res):.3f} dex")
    log(f"C15 median bias:                 {np.median(lg_res):+.3f} dex")
    log(f"SPARC full scatter:              0.286 dex")


if __name__ == '__main__':
    main()
