#!/usr/bin/env python3
"""HSC SSP Catalog Query Tool - uses the datasearch web API."""

import argparse
import sys
import time
import requests
from html.parser import HTMLParser

BASE_URL = "https://hsc-release.mtk.nao.ac.jp/datasearch"


class TokenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.csrf_token = None
        self.authenticity_token = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name") == "csrf-token":
            self.csrf_token = attrs.get("content")
        if tag == "input" and attrs.get("name") == "authenticity_token":
            self.authenticity_token = attrs.get("value")


def get_session(user, password):
    session = requests.Session()
    session.auth = (user, password)
    r = session.get(BASE_URL + "/", timeout=30)
    r.raise_for_status()
    parser = TokenParser()
    parser.feed(r.text)
    return session, parser.authenticity_token, parser.csrf_token


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
    # After submit, we are redirected to the job page; extract job ID from URL
    job_url = r.url  # e.g. /datasearch/catalog_jobs/12345
    job_id = job_url.rstrip("/").split("/")[-1]
    if not job_id.isdigit():
        # Try to find job ID in page
        import re
        m = re.search(r'/catalog_jobs/(\d+)', r.text)
        if m:
            job_id = m.group(1)
        else:
            print("Response URL:", job_url)
            print(r.text[:500])
            raise RuntimeError("Could not determine job ID after submission")
    return job_id


def wait_for_job(session, csrf_token, job_id, poll_interval=5):
    import re
    jobs_url = BASE_URL + "/catalog_jobs"
    print(f"Job {job_id} submitted. Waiting for completion...", file=sys.stderr)
    while True:
        r = session.get(jobs_url, timeout=30)
        # Parse status from job list HTML: <td class='job-done'>done</td>
        # Find the row for our job_id
        idx = r.text.find(f">{job_id}<")
        if idx >= 0:
            chunk = r.text[idx:idx+800]
            m = re.search(r"class='job-([^']+)'>([^<]+)<", chunk)
            if m:
                status = m.group(2).strip()
                print(f"  status: {status}", file=sys.stderr)
                if status in ("done",):
                    return
                if status in ("error", "failed", "cancelled"):
                    err = re.search(r"data-error='([^']*)'", chunk)
                    msg = err.group(1) if err else "unknown error"
                    raise RuntimeError(f"Job {job_id} failed: {msg}")
        time.sleep(poll_interval)


def get_download_key(session, job_id):
    import re
    r = session.get(BASE_URL + "/catalog_jobs", timeout=30)
    idx = r.text.find(f">{job_id}<")
    if idx >= 0:
        chunk = r.text[idx:idx+1500]
        m = re.search(r'href="(/datasearch/catalog_jobs/download/[a-f0-9]+)"', chunk)
        if m:
            return m.group(1)
    raise RuntimeError(f"Could not find download link for job {job_id}")


def download_result(session, csrf_token, job_id, output_file):
    path = get_download_key(session, job_id)
    download_url = "https://hsc-release.mtk.nao.ac.jp" + path
    r = session.get(download_url, stream=True, timeout=120)
    r.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="HSC SSP catalog query tool")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--release", default="pdr3")
    parser.add_argument("--output-format", default="csv", choices=["csv", "csv.gz", "fits", "sqlite3"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--sql", required=True)
    args = parser.parse_args()

    print("Authenticating...", file=sys.stderr)
    session, auth_token, csrf_token = get_session(args.user, args.password)
    if not auth_token:
        raise RuntimeError("Authentication failed or could not get form token")
    print("Authenticated.", file=sys.stderr)

    print("Submitting query...", file=sys.stderr)
    job_id = submit_job(session, auth_token, args.sql, args.release, args.output_format)
    print(f"Job ID: {job_id}", file=sys.stderr)

    wait_for_job(session, csrf_token, job_id)

    print("Downloading result...", file=sys.stderr)
    download_result(session, csrf_token, job_id, args.output)


if __name__ == "__main__":
    main()
