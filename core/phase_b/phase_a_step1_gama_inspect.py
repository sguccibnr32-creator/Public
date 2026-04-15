# -*- coding: utf-8 -*-
"""
phase_a_step1_gama_inspect.py

Step 1: Inspect GAMA FITS file structures.
"""

import os, sys
import numpy as np
from pathlib import Path

GAMA_DIR = Path(r"D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\GAMA_DR4")

GAMA_FILES = {
    'SpecObj':           ['SpecObjv27.fits', 'SpecObj.fits'],
    'StellarMasses':     ['StellarMassesGKVv24.fits', 'StellarMassesGKV.fits'],
    'G3CGal':            ['G3CGalv10.fits', 'G3CGal.fits'],
    'InputCat':          ['gkvInputCatv02.fits', 'gkvScienceCatv02.fits',
                          'InputCatA.fits', 'InputCat.fits'],
    'ScienceCat':        ['gkvScienceCatv02.fits'],
}

KEY_COLUMNS = {
    'id':       ['CATAID', 'cataid', 'uberID', 'UBERID', 'OBJID'],
    'ra':       ['RA', 'RAcen', 'RAJ2000', 'ra', 'RA_J2000'],
    'dec':      ['DEC', 'Deccen', 'DECJ2000', 'dec'],
    'z':        ['Z', 'Z_HELIO', 'Z_TONRY', 'z', 'REDSHIFT'],
    'z_qual':   ['NQ', 'Z_QUAL', 'nQ', 'QUALITY'],
    'logmstar': ['logmstar', 'LOGMSTAR', 'logMstar', 'mstar'],
    'sersic_n': ['GALINDEX_r', 'SERSIC_N', 'GAL_INDEX_r', 'nser_r'],
    'mag_r':    ['mag_rt', 'MAG_AUTO_r', 'PETROMAG_R', 'Rpetro', 'mag_r'],
    'group':    ['GroupID', 'GROUPID', 'G3CGalID'],
    'rank':     ['RankIterCen', 'RANKITERCEN', 'Rank'],
    'nfof':     ['Nfof', 'NFOF', 'Multiplicity'],
    'uberclass': ['uberclass', 'UBERCLASS', 'class'],
}


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def find_file(category):
    candidates = GAMA_FILES.get(category, [])
    for name in candidates:
        path = GAMA_DIR / name
        if path.exists():
            return path
    return None


def inspect_fits(filepath, category):
    from astropy.io import fits
    print(f"\n  File: {filepath.name}")
    print(f"  Size: {filepath.stat().st_size/1e6:.1f} MB")

    found_keys = {}
    with fits.open(filepath, memmap=True) as hdul:
        print(f"  HDUs: {len(hdul)}")

        for hi, hdu in enumerate(hdul):
            if not hasattr(hdu, 'columns') or hdu.columns is None:
                continue

            nrows = len(hdu.data) if hdu.data is not None else 0
            ncols = len(hdu.columns)
            print(f"\n  HDU[{hi}] '{hdu.name}': {nrows:,} rows x {ncols} columns")

            # Match key columns
            for ci, col in enumerate(hdu.columns):
                for key_name, patterns in KEY_COLUMNS.items():
                    if key_name in found_keys:
                        continue
                    if col.name in patterns:
                        found_keys[key_name] = col.name
                        break
                    if col.name.lower() in [p.lower() for p in patterns]:
                        found_keys[key_name] = col.name
                        break

            # Show first 30 columns (for overview)
            print(f"\n  First 30 columns:")
            for ci, col in enumerate(hdu.columns[:30]):
                indicator = ''
                for k, v in found_keys.items():
                    if v == col.name:
                        indicator = f'<- {k}'
                        break
                print(f"    [{ci:3d}] {col.name:35s} {col.format:>8s}  "
                      f"{str(col.unit or '-'):>10s}  {indicator}")
            if ncols > 30:
                print(f"    ... ({ncols-30} more columns)")

            # Key column summary with samples
            print(f"\n  Key columns found:")
            for key_name, col_name in found_keys.items():
                try:
                    sample = hdu.data[col_name][:3]
                    sample_str = ', '.join([str(v)[:20] for v in sample])
                    print(f"    {key_name:>10s} -> {col_name:25s}  sample: [{sample_str}]")
                except Exception as e:
                    print(f"    {key_name:>10s} -> {col_name:25s}  (read err)")

            # Any column with 'mass', 'mag', 'sersic', 'group' etc.
            print(f"\n  Other potentially useful columns (search: mass/mag/sersic/group/ran/dec):")
            keywords = ['mass', 'mag', 'sersic', 'nser', 'group', 'bcg', 'rank',
                        'ra', 'dec', 'z_', 'logmstar', 'uber', 'redshift']
            found_other = []
            for col in hdu.columns:
                cl = col.name.lower()
                if any(k in cl for k in keywords):
                    if col.name not in found_keys.values():
                        found_other.append(col.name)
            for name in found_other[:40]:
                print(f"    {name}")
            if len(found_other) > 40:
                print(f"    ... ({len(found_other)-40} more)")

            break  # Only first table HDU

    return found_keys


def main():
    section("GAMA FITS STRUCTURE INSPECTOR")

    print(f"\n  GAMA directory: {GAMA_DIR}")
    if not GAMA_DIR.exists():
        print(f"  ERROR: Directory not found!")
        return

    import glob
    all_fits = sorted(glob.glob(str(GAMA_DIR / "*.fits")))
    print(f"\n  FITS files found ({len(all_fits)}):")
    for f in all_fits:
        p = Path(f)
        print(f"    {p.name:50s}  ({p.stat().st_size/1e6:.1f} MB)")

    results = {}
    for category in GAMA_FILES:
        section(f"{category}")
        filepath = find_file(category)
        if filepath is None:
            print(f"  NOT FOUND. Tried: {GAMA_FILES.get(category, [])}")
            continue
        found_keys = inspect_fits(filepath, category)
        results[category] = {'path': filepath, 'keys': found_keys}

    section("CROSS-TABLE JOIN FEASIBILITY")
    id_cols = {}
    for cat, info in results.items():
        id_col = info['keys'].get('id')
        if id_col:
            id_cols[cat] = id_col
            print(f"  {cat:20s}: ID column = {id_col}")
        else:
            print(f"  {cat:20s}: NO ID column found")

    section("CONFIG FOR STEP 2")
    print(f"\n  GAMA_CONFIG = {{")
    for cat, info in results.items():
        keys = info['keys']
        path = info['path']
        print(f"      '{cat}': {{")
        print(f"          'path': r'{path}',")
        for k, v in keys.items():
            print(f"          '{k}': '{v}',")
        print(f"      }},")
    print(f"  }}")

    print(f"\n{'='*72}\n  STEP 1 COMPLETE\n{'='*72}")


if __name__ == '__main__':
    main()
