#!/usr/bin/env python3
"""
cl1/cl3/cl4/cl27 ソース銀河 + photo-z を HSC CAS から取得。
INNER JOIN で photo-z 必須、z_src > z_cl + 0.1 フィルタ済み。

uv run --with requests python fetch_cluster_photoz.py
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
from hscSspQuery import get_session, submit_job, wait_for_job, download_result

# (id, RA, Dec, z_spec)
CLUSTERS = [
    ("cl1",  140.45012, -0.25117, 0.313),
    ("cl3",  140.50893, -0.34883, 0.319),
    ("cl4",  140.50336, -0.44150, 0.324),
    ("cl27", 140.45036, -0.48161, 0.320),
]

R_DEG = 1.0  # ±1度 = 2度角正方形
RELEASE = "pdr3"
FMT = "csv"

def make_query(cid, ra, dec, z_cl):
    z_min = z_cl + 0.1
    return f"""-- {cid} (z={z_cl}) with photo-z INNER JOIN
SELECT
    w.object_id,
    w.i_ra,
    w.i_dec,
    w.i_hsmshaperegauss_e1,
    w.i_hsmshaperegauss_e2,
    w.i_hsmshaperegauss_derived_weight,
    w.i_hsmshaperegauss_derived_shear_bias_m,
    w.i_hsmshaperegauss_derived_shear_bias_c1,
    w.i_hsmshaperegauss_derived_shear_bias_c2,
    w.b_mode_mask,
    a.photoz_best,
    a.photoz_err68_min,
    a.photoz_err68_max
FROM
    s19a_wide.weaklensing_hsm_regauss AS w
    INNER JOIN pdr3_wide.photoz_demp AS a USING (object_id)
WHERE
    w.b_mode_mask = 1
    AND w.i_ra BETWEEN {ra - R_DEG} AND {ra + R_DEG}
    AND w.i_dec BETWEEN {dec - R_DEG} AND {dec + R_DEG}
    AND a.photoz_best > {z_min}
    AND a.photoz_best < 2.0
    AND w.i_hsmshaperegauss_resolution > 0.3
"""

def main():
    user = os.environ.get("HSC_USER", "")
    pw = os.environ.get("HSC_PASSWORD", "")
    if not user:
        user = input("HSC user: ").strip()
    if not pw:
        import getpass
        pw = getpass.getpass("HSC password: ")

    print("Authenticating...")
    session, token = get_session(user, pw)
    print("Authenticated.\n")

    jobs = []
    for cid, ra, dec, z in CLUSTERS:
        outf = f"{cid}_sources_photoz.csv"
        if Path(outf).exists():
            print(f"  {cid}: {outf} already exists, skipping.")
            continue
        sql = make_query(cid, ra, dec, z)
        z_min = z + 0.1
        print(f"Submitting {cid} (z={z}, z_src>{z_min:.3f})...")
        print(f"  RA=[{ra-R_DEG:.3f},{ra+R_DEG:.3f}] Dec=[{dec-R_DEG:.3f},{dec+R_DEG:.3f}]")
        try:
            jid = submit_job(session, token, sql, RELEASE, FMT)
            print(f"  job_id={jid}")
            jobs.append((cid, jid, outf))
        except Exception as e:
            print(f"  submit error: {e}")

    if not jobs:
        print("\nAll files exist or all submissions failed.")
        return

    print(f"\nWaiting for {len(jobs)} jobs...")
    for cid, jid, outf in jobs:
        try:
            wait_for_job(session, jid, poll_interval=10)
            print(f"  {cid} (job {jid}): done, downloading...")
            download_result(session, jid, outf)
            with open(outf) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {cid}: {n} sources -> {outf}")
        except Exception as e:
            print(f"  {cid} (job {jid}): error: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
