#!/usr/bin/env python3
"""HSC SSP Catalog Query Tool (compatible interface with original hscSspQuery.py)."""

import argparse
import re
import sys
import time
import requests
from html.parser import HTMLParser

BASE_URL = "https://hsc-release.mtk.nao.ac.jp/datasearch"


class TokenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.authenticity_token = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") == "authenticity_token":
            self.authenticity_token = attrs.get("value")


def get_session(user, password):
    session = requests.Session()
    session.auth = (user, password)
    r = session.get(BASE_URL + "/", timeout=30)
    r.raise_for_status()
    parser = TokenParser()
    parser.feed(r.text)
    if not parser.authenticity_token:
        raise RuntimeError("Authentication failed or could not retrieve form token.")
    return session, parser.authenticity_token


def submit_job(session, auth_token, sql, release, out_format):
    data = {
        "utf8": "✓",
        "authenticity_token": auth_token,
        "catalog_job[name]": f"query-{int(time.time())}",
        "catalog_job[release_version]": release,
        "catalog_job[sql]": sql,
        "catalog_job[include_metainfo_to_body]": "0",
        "catalog_job[out_format]": out_format,
    }
    r = session.post(BASE_URL + "/catalog_jobs", data=data,
                     allow_redirects=True, timeout=60)
    r.raise_for_status()
    m = re.search(r'/catalog_jobs/(\d+)', r.url + r.text)
    if not m:
        raise RuntimeError("Could not determine job ID after submission.")
    return m.group(1)


def wait_for_job(session, job_id, poll_interval=5):
    print(f"Waiting for job {job_id}...", file=sys.stderr)
    while True:
        r = session.get(BASE_URL + "/catalog_jobs", timeout=30)
        idx = r.text.find(f">{job_id}<")
        if idx >= 0:
            chunk = r.text[idx:idx+800]
            m = re.search(r"class='job-([^']+)'>([^<]+)<", chunk)
            if m:
                status = m.group(2).strip()
                print(f"  status: {status}", file=sys.stderr)
                if status == "done":
                    return
                if status in ("error", "failed", "cancelled"):
                    err = re.search(r"data-error='([^']*)'", chunk)
                    msg = err.group(1) if err else "unknown error"
                    msg = msg.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&")
                    raise RuntimeError(f"Job failed: {msg}")
        time.sleep(poll_interval)


def get_download_url(session, job_id):
    r = session.get(BASE_URL + "/catalog_jobs", timeout=30)
    idx = r.text.find(f">{job_id}<")
    if idx >= 0:
        chunk = r.text[idx:idx+1500]
        m = re.search(r'href="(/datasearch/catalog_jobs/download/[a-f0-9]+)"', chunk)
        if m:
            return "https://hsc-release.mtk.nao.ac.jp" + m.group(1)
    raise RuntimeError(f"No download link found for job {job_id}.")


def download_result(session, job_id, output_file):
    url = get_download_url(session, job_id)
    r = session.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved: {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Query the HSC SSP catalog database.")
    parser.add_argument("--user", required=True, help="HSC SSP username")
    parser.add_argument("--password", required=True, help="HSC SSP password")
    parser.add_argument("--release", default="pdr3",
                        help="Data release (e.g. pdr3, pdr2)")
    parser.add_argument("--output-format", default="csv",
                        choices=["csv", "csv.gz", "fits", "sqlite3"])
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--sql", required=True, help="SQL query string")
    parser.add_argument("--nomail", action="store_true",
                        help="Do not send notification email (ignored)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    print(f"Release : {args.release}", file=sys.stderr)
    print(f"Output  : {args.output}", file=sys.stderr)
    print(f"SQL     : {args.sql[:80]}{'...' if len(args.sql)>80 else ''}", file=sys.stderr)

    if not args.yes:
        ans = input("Submit this query? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.", file=sys.stderr)
            sys.exit(0)

    print("Authenticating...", file=sys.stderr)
    session, auth_token = get_session(args.user, args.password)
    print("Authenticated.", file=sys.stderr)

    print("Submitting query...", file=sys.stderr)
    job_id = submit_job(session, auth_token, args.sql,
                        args.release, args.output_format)
    print(f"Job ID: {job_id}", file=sys.stderr)

    wait_for_job(session, job_id)

    print("Downloading result...", file=sys.stderr)
    download_result(session, job_id, args.output)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
