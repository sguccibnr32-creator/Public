#!/usr/bin/env python3
"""
cl3, cl4, cl5, cl27 のソース銀河を HSC CAS から一括取得。
hscSspQuery.py を内部で利用。

uv run --with requests python fetch_cluster_sources.py

認証情報は環境変数 HSC_USER / HSC_PASSWORD、または対話入力。
"""

import sys, os, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
from hscSspQuery import get_session, submit_job, wait_for_job, download_result

# クラスター定義: (id, RA, Dec, z_spec)
CLUSTERS = [
    ("cl3",  140.50893, -0.34883, 0.319),
    ("cl4",  140.50336, -0.44150, 0.324),
    ("cl5",  140.33710, -0.24947, 0.316),
    ("cl27", 140.45036, -0.48161, 0.320),
]

R_DEG = 1.0  # 検索半径 [deg]
RELEASE = "pdr3"
FMT = "csv"

def make_query(cid, ra, dec, z_cl):
    z_src_min = z_cl + 0.1
    return f"""-- {cid} (z={z_cl})
SELECT
    object_id,
    w.i_ra, w.i_dec,
    w.i_hsmshaperegauss_e1,
    w.i_hsmshaperegauss_e2,
    w.i_hsmshaperegauss_derived_weight,
    w.i_hsmshaperegauss_derived_shear_bias_m,
    w.i_hsmshaperegauss_derived_shear_bias_c1,
    w.i_hsmshaperegauss_derived_shear_bias_c2,
    w.b_mode_mask
FROM
    s19a_wide.weaklensing_hsm_regauss AS w
WHERE
    w.b_mode_mask = 1
    AND w.i_ra BETWEEN {ra - R_DEG} AND {ra + R_DEG}
    AND w.i_dec BETWEEN {dec - R_DEG} AND {dec + R_DEG}
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
        outf = Path(f"{cid}_sources.csv")
        if outf.exists():
            print(f"  {cid}: {outf} already exists, skipping.")
            continue
        sql = make_query(cid, ra, dec, z)
        print(f"Submitting {cid} (RA={ra:.3f} Dec={dec:.3f} z={z})...")
        try:
            jid = submit_job(session, token, sql, RELEASE, FMT)
            print(f"  job_id={jid}")
            jobs.append((cid, jid, str(outf)))
        except Exception as e:
            print(f"  submit error: {e}")

    if not jobs:
        print("\nAll files exist or all submissions failed.")
        return

    # ポーリング
    print(f"\nWaiting for {len(jobs)} jobs...")
    for cid, jid, outf in jobs:
        try:
            wait_for_job(session, jid, poll_interval=10)
            print(f"  {cid} (job {jid}): done, downloading...")
            download_result(session, jid, outf)
            # 行数確認
            with open(outf) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {cid}: {n} sources saved to {outf}")
        except Exception as e:
            print(f"  {cid} (job {jid}): error: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
