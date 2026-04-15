#!/usr/bin/env python3
"""
Fetch MCXC and ACT-DR5 cluster catalogs from VizieR,
filter by HSC field footprint and z < 0.20.
"""

import pandas as pd
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

# ── HSC field definitions ─────────────────────────────────────────────────────
HSC_FIELDS = {
    "GAMA09":   dict(ra_min=129, ra_max=152, dec_min=-2, dec_max=5),
    "WIDE12H":  dict(ra_min=173, ra_max=196, dec_min=-2, dec_max=2),
    "GAMA15":   dict(ra_min=207, ra_max=223, dec_min=-2, dec_max=2),
    "VVDS":     dict(ra_min=330, ra_max=363, dec_min=-1, dec_max=4),
    "ELAIS-N1": dict(ra_min=240, ra_max=250, dec_min=42, dec_max=46),
}

def in_hsc_fields(ra, dec):
    """Return field name if (ra, dec) falls in any HSC field, else None."""
    for name, f in HSC_FIELDS.items():
        if f["ra_min"] <= ra <= f["ra_max"] and f["dec_min"] <= dec <= f["dec_max"]:
            return name
    return None

def to_decimal_deg(val, unit):
    """Convert a coordinate value (decimal or sexagesimal string) to float degrees."""
    try:
        return float(val)
    except (ValueError, TypeError):
        # sexagesimal string
        sc = SkyCoord(ra=val if unit == "ra" else "00:00:00",
                      dec=val if unit == "dec" else "00:00:00",
                      unit=(u.hourangle if unit == "ra" else u.deg,
                            u.deg if unit == "dec" else u.hourangle))
        return sc.ra.deg if unit == "ra" else sc.dec.deg


def add_field_column(df, ra_col="RAJ2000", dec_col="DEJ2000"):
    # Detect if coordinates are sexagesimal
    sample_ra  = str(df[ra_col].iloc[0])
    sample_dec = str(df[dec_col].iloc[0])
    is_sexa = " " in sample_ra or ":" in sample_ra

    if is_sexa:
        coords = SkyCoord(
            ra=df[ra_col].values,
            dec=df[dec_col].values,
            unit=(u.hourangle, u.deg),
        )
        ra_vals  = coords.ra.deg
        dec_vals = coords.dec.deg
    else:
        ra_vals  = df[ra_col].astype(float).values
        dec_vals = df[dec_col].astype(float).values

    df = df.copy()
    df["_ra_deg"]  = ra_vals
    df["_dec_deg"] = dec_vals
    df["hsc_field"] = [
        in_hsc_fields(r, d) for r, d in zip(ra_vals, dec_vals)
    ]
    return df[df["hsc_field"].notna()].reset_index(drop=True)

# ── Vizier setup ──────────────────────────────────────────────────────────────
viz = Vizier(columns=["*"], row_limit=-1)

# ═════════════════════════════════════════════════════════════════════════════
# 1. MCXC  (Piffaretti et al. 2011, A&A 534, A109)
# ═════════════════════════════════════════════════════════════════════════════
print("Fetching MCXC...")
r_mcxc = viz.get_catalogs("J/A+A/534/A109/mcxc")
mcxc = r_mcxc[0].to_pandas()
print(f"  Total: {len(mcxc)} clusters")

# z filter
mcxc = mcxc[mcxc["z"] < 0.20].copy()
print(f"  After z < 0.20: {len(mcxc)}")

# HSC field filter
mcxc = add_field_column(mcxc, "RAJ2000", "DEJ2000")
print(f"  In HSC fields:  {len(mcxc)}")

# Tidy output
mcxc_out = mcxc.rename(columns={
    "MCXC":     "Name",
    "_ra_deg":  "RA",
    "_dec_deg": "Dec",
    "z":        "z",
    "L500":     "L500_1e44ergs",
    "M500":     "M500_1e14Msun",
    "R500":     "R500_Mpc",
}).copy()
mcxc_out = mcxc_out[["Name", "RA", "Dec", "z", "L500_1e44ergs",
                      "M500_1e14Msun", "R500_Mpc", "hsc_field"]]
mcxc_out = mcxc_out.sort_values(["hsc_field", "z"]).reset_index(drop=True)

mcxc_out.to_csv("mcxc_hsc.csv", index=False)
print(f"  Saved -> mcxc_hsc.csv")

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)
print(mcxc_out.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# 2. ACT-DR5  (Hilton et al. 2021, ApJS 253, 3)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFetching ACT-DR5...")
r_act = viz.get_catalogs("J/ApJS/253/3")
act = r_act[0].to_pandas()   # main cluster table
print(f"  Total: {len(act)} clusters")

# z filter (ACT-DR5 z column may have NaN for photo-z failures)
act = act[act["z"].notna() & (act["z"] < 0.20)].copy()
print(f"  After z < 0.20: {len(act)}")

# HSC field filter
act = add_field_column(act, "RAJ2000", "DEJ2000")
print(f"  In HSC fields:  {len(act)}")

# M500cC is in units of 1e14 Msun (check: column description says 10^14 Msun)
act_out = act.rename(columns={
    "ACT-CL":   "Name",
    "_ra_deg":  "RA",
    "_dec_deg": "Dec",
    "z":        "z",
    "M500cC":   "M500cC_1e14Msun",
    "M200m":    "M200m_1e14Msun",
    "SNR":      "SNR",
}).copy()

keep = ["Name", "RA", "Dec", "z", "SNR",
        "M500cC_1e14Msun", "M200m_1e14Msun", "hsc_field"]
keep = [c for c in keep if c in act_out.columns]
act_out = act_out[keep].sort_values(["hsc_field", "z"]).reset_index(drop=True)

act_out.to_csv("act_dr5_hsc.csv", index=False)
print(f"  Saved -> act_dr5_hsc.csv")
print(act_out.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary (z < 0.20, in HSC fields) ===")
print(f"MCXC   : {len(mcxc_out)} clusters")
for field in HSC_FIELDS:
    n = (mcxc_out["hsc_field"] == field).sum()
    if n: print(f"  {field}: {n}")
print(f"ACT-DR5: {len(act_out)} clusters")
for field in HSC_FIELDS:
    n = (act_out["hsc_field"] == field).sum()
    if n: print(f"  {field}: {n}")
