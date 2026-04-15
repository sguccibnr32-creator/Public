# -*- coding: utf-8 -*-
"""
compare_wl_catalogs.py

Compare two weak lensing catalogs for consistency:
  1. 931720.csv.gz.1 (Subaru/HSC-SSP Y3)
  2. KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits (KiDS DR4)
"""

import os, sys, gzip
import numpy as np
from pathlib import Path

FILE_SUBARU = Path(r"E:\スバル望遠鏡データ\931720.csv.gz.1")
FILE_KIDS   = Path(r"E:\スバル望遠鏡データ\KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits")

MAX_PREVIEW = 100
MAX_SAMPLE = 500000


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def is_gzip(filepath):
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except Exception:
        return False


# ====================================================================
#  PART 1: FILE INSPECTION
# ====================================================================

def inspect_csv_gz(filepath):
    section(f"FILE 1: {filepath.name}")
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None

    fsize = filepath.stat().st_size
    print(f"  Size: {fsize:,} bytes ({fsize/1e9:.2f} GB)")

    actually_gzip = is_gzip(filepath)
    print(f"  Extension suggests gzip: {str(filepath).endswith(('.gz', '.gz.1'))}")
    print(f"  Magic number confirms gzip: {actually_gzip}")

    try:
        if actually_gzip:
            opener = lambda p: gzip.open(p, 'rt', errors='replace')
        else:
            opener = lambda p: open(p, 'r', encoding='utf-8', errors='replace')

        with opener(filepath) as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line.rstrip())
                if i >= MAX_PREVIEW:
                    break

        header_line = lines[0]
        if header_line.startswith('#'):
            print(f"  Header type: comment-style (#)")
            for i, line in enumerate(lines):
                if not line.startswith('#'):
                    header_line = lines[i-1].lstrip('#').strip() if i > 0 else line
                    break

        for delim_name, delim in [('comma', ','), ('tab', '\t'), ('pipe', '|')]:
            if delim in header_line:
                cols = [c.strip() for c in header_line.split(delim)]
                if len(cols) > 3:
                    print(f"  Delimiter: {delim_name}")
                    break
        else:
            cols = header_line.split()
            print(f"  Delimiter: whitespace")

        print(f"  Columns ({len(cols)}):")
        for i, c in enumerate(cols):
            print(f"    [{i:3d}] {c}")

        print(f"\n  First data lines:")
        for line in lines[1:6]:
            if not line.startswith('#'):
                print(f"    {line[:120]}{'...' if len(line) > 120 else ''}")

        print(f"\n  Counting rows...")
        row_count = 0
        with opener(filepath) as f:
            for line in f:
                if not line.startswith('#'):
                    row_count += 1
                if row_count > 10000000:
                    print(f"  Rows: > 10,000,000 (stopped counting)")
                    break
            else:
                print(f"  Rows: {row_count:,} (including header)")

        return {'type': 'csv', 'columns': cols, 'rows': row_count,
                'lines': lines, 'path': filepath}

    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None


def inspect_fits(filepath):
    section(f"FILE 2: {filepath.name}")
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None

    fsize = filepath.stat().st_size
    print(f"  Size: {fsize:,} bytes ({fsize/1e9:.2f} GB)")

    try:
        from astropy.io import fits
        with fits.open(filepath, memmap=True) as hdul:
            print(f"  HDU count: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                print(f"    HDU[{i}]: {hdu.name} ({type(hdu).__name__})")
                if hasattr(hdu, 'data') and hdu.data is not None:
                    try:
                        nrows = len(hdu.data)
                        print(f"      Rows: {nrows:,}")
                    except Exception:
                        pass
            for hdu in hdul:
                if hasattr(hdu, 'columns') and hdu.columns is not None:
                    cols = [c.name for c in hdu.columns]
                    nrows = len(hdu.data) if hdu.data is not None else 0
                    return {'type': 'fits', 'columns': cols, 'rows': nrows,
                            'path': filepath}
    except ImportError:
        print("  ERROR: astropy not installed.")
        return None
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None
    return None


# ====================================================================
#  PART 2: COLUMN MAPPING (with HSC support + dedup)
# ====================================================================

def identify_columns(info):
    """Identify key WL columns, handling KiDS (lensfit) and HSC-SSP (regauss).

    Uses first occurrence to avoid HSC duplicate columns 15-18.
    """
    cols = info['columns']
    cols_lower = [c.lower() for c in cols]
    # First occurrence index for dedup
    seen_cols = {}
    for i, c in enumerate(cols_lower):
        if c not in seen_cols:
            seen_cols[c] = i

    def find_col(exact_matches):
        for pattern in exact_matches:
            pl = pattern.lower()
            if pl in seen_cols:
                return cols[seen_cols[pl]]
        return None

    mapping = {}
    mapping['ra'] = find_col(
        ['RAJ2000', 'ALPHA_J2000', 'RA', 'ra', 'RA_J2000', 'i_ra'])
    mapping['dec'] = find_col(
        ['DECJ2000', 'DEJ2000', 'DELTA_J2000', 'DEC', 'dec', 'DEC_J2000', 'i_dec'])
    mapping['e1'] = find_col(
        ['e1', 'E1', 'i_hsmshaperegauss_e1'])
    mapping['e2'] = find_col(
        ['e2', 'E2', 'i_hsmshaperegauss_e2'])
    mapping['weight'] = find_col(
        ['weight', 'WEIGHT', 'lensfit_weight',
         'i_hsmshaperegauss_derived_weight'])
    mapping['z'] = find_col(
        ['Z_B', 'z_b', 'PHOTO_Z', 'photo_z', 'Z_BEST', 'z_best',
         'Z_PHOT', 'z_phot', 'photoz', 'PHOTOZ', 'Z_ML', 'z_ml',
         'hsc_y3_zbin'])

    mapping = {k: v for k, v in mapping.items() if v is not None}
    return mapping


# ====================================================================
#  PART 3: DATA LOADING
# ====================================================================

def load_csv_sample(info, mapping, max_rows=MAX_SAMPLE):
    import csv
    filepath = info['path']
    if is_gzip(filepath):
        open_func = lambda: gzip.open(filepath, 'rt', errors='replace')
    else:
        open_func = lambda: open(filepath, 'r', encoding='utf-8', errors='replace')

    data = {k: [] for k in mapping}
    col_idx = {}
    header_found = False
    header_candidate = None
    count = 0

    with open_func() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('#'):
                candidate = line.lstrip('#').strip()
                parts = [c.strip() for c in candidate.split(',')]
                if len(parts) < 3:
                    parts = candidate.split()
                if len(parts) >= 3:
                    header_candidate = parts
                continue

            if not header_found:
                test_parts = [c.strip() for c in line.split(',')]
                if len(test_parts) < 3:
                    test_parts = line.split()

                try:
                    float(test_parts[0])
                    header = header_candidate if header_candidate else test_parts
                    is_data = True
                except ValueError:
                    header = test_parts
                    is_data = False

                # Use first occurrence of each header field (HSC dedup)
                seen = {}
                for j, h in enumerate(header):
                    if h not in seen:
                        seen[h] = j
                for key, col_name in mapping.items():
                    if col_name in seen:
                        col_idx[key] = seen[col_name]
                    else:
                        for h, j in seen.items():
                            if h.lower() == col_name.lower():
                                col_idx[key] = j
                                break
                header_found = True

                if not col_idx:
                    print(f"  WARNING: No columns matched in header: {header[:10]}")
                    return {k: np.array([]) for k in mapping}

                if not is_data:
                    continue

            parts = [c.strip() for c in line.split(',')]
            if len(parts) < 3:
                parts = line.split()

            try:
                for key, idx in col_idx.items():
                    data[key].append(float(parts[idx]))
                count += 1
            except (ValueError, IndexError):
                continue

            if count >= max_rows:
                break

    print(f"  Loaded {count:,} data rows")
    return {k: np.array(v) for k, v in data.items()}


def load_fits_sample(info, mapping, max_rows=MAX_SAMPLE):
    from astropy.io import fits
    with fits.open(info['path'], memmap=True) as hdul:
        for hdu in hdul:
            if hasattr(hdu, 'columns') and hdu.columns is not None:
                nrows = min(len(hdu.data), max_rows)
                data = {}
                for key, col_name in mapping.items():
                    if col_name in hdu.columns.names:
                        data[key] = np.array(hdu.data[col_name][:nrows], dtype=float)
                return data
    return {}


# ====================================================================
#  PART 4: COMPARISON
# ====================================================================

def compare_catalogs(data1, data2, label1, label2):
    from scipy import stats
    section("STATISTICAL COMPARISON")

    for key in ['ra', 'dec', 'e1', 'e2', 'z', 'weight']:
        if key not in data1 or key not in data2:
            continue
        d1, d2 = data1[key], data2[key]
        d1 = d1[np.isfinite(d1)]
        d2 = d2[np.isfinite(d2)]
        if len(d1) == 0 or len(d2) == 0:
            continue

        is_zbin_discrete = (key == 'z' and
                            (np.all(d1 == d1.astype(int)) or np.all(d2 == d2.astype(int))))

        print(f"\n  --- {key} ---")
        if is_zbin_discrete:
            print(f"  NOTE: one catalog uses discrete z-bin index; KS not meaningful.")
        print(f"  {'':>20s}  {label1[:20]:>20s}  {label2[:20]:>20s}")
        print(f"  {'N':>20s}  {len(d1):>20,d}  {len(d2):>20,d}")
        print(f"  {'min':>20s}  {np.min(d1):>20.6f}  {np.min(d2):>20.6f}")
        print(f"  {'max':>20s}  {np.max(d1):>20.6f}  {np.max(d2):>20.6f}")
        print(f"  {'mean':>20s}  {np.mean(d1):>20.6f}  {np.mean(d2):>20.6f}")
        print(f"  {'median':>20s}  {np.median(d1):>20.6f}  {np.median(d2):>20.6f}")
        print(f"  {'std':>20s}  {np.std(d1):>20.6f}  {np.std(d2):>20.6f}")

        n_ks = min(50000, len(d1), len(d2))
        ks_stat, ks_p = stats.ks_2samp(
            np.random.choice(d1, n_ks, replace=False),
            np.random.choice(d2, n_ks, replace=False))
        print(f"  {'KS test':>20s}  D={ks_stat:.4f}, p={ks_p:.2e}")

    if 'ra' in data1 and 'ra' in data2 and 'dec' in data1 and 'dec' in data2:
        section("SKY COVERAGE OVERLAP")
        ra1_min, ra1_max = np.min(data1['ra']), np.max(data1['ra'])
        de1_min, de1_max = np.min(data1['dec']), np.max(data1['dec'])
        ra2_min, ra2_max = np.min(data2['ra']), np.max(data2['ra'])
        de2_min, de2_max = np.min(data2['dec']), np.max(data2['dec'])

        print(f"\n  {label1}:")
        print(f"    RA:  [{ra1_min:.3f}, {ra1_max:.3f}]")
        print(f"    Dec: [{de1_min:.3f}, {de1_max:.3f}]")
        print(f"\n  {label2}:")
        print(f"    RA:  [{ra2_min:.3f}, {ra2_max:.3f}]")
        print(f"    Dec: [{de2_min:.3f}, {de2_max:.3f}]")

        ra_overlap = max(0, min(ra1_max, ra2_max) - max(ra1_min, ra2_min))
        de_overlap = max(0, min(de1_max, de2_max) - max(de1_min, de2_min))

        if ra_overlap > 0 and de_overlap > 0:
            print(f"\n  OVERLAP DETECTED:")
            print(f"    RA overlap:  {ra_overlap:.3f} deg")
            print(f"    Dec overlap: {de_overlap:.3f} deg")
            print(f"    Overlap area: ~{ra_overlap * de_overlap:.1f} sq deg (rough)")
        else:
            print(f"\n  NO SKY OVERLAP between the two catalogs.")


def identify_survey(info):
    cols_lower = [c.lower() for c in info['columns']]
    cols_str = ' '.join(cols_lower)
    surveys = []
    if any(x in cols_str for x in ['hsc', 'i_cmodel', 'g_cmodel', 'tract',
                                     'isprimary', 'ideblend']):
        surveys.append('HSC-SSP (Subaru Hyper Suprime-Cam)')
    if any(x in cols_str for x in ['kids', 'lensfit', 'som_flag', 'theli',
                                     'gal_id', 'autocal']):
        surveys.append('KiDS (Kilo-Degree Survey)')
    if any(x in cols_str for x in ['des', 'metacal', 'im3shape']):
        surveys.append('DES (Dark Energy Survey)')
    if any(x in cols_str for x in ['cfht', 'cfhtlens', 'erben']):
        surveys.append('CFHTLenS')
    if any(x in cols_str for x in ['e1', 'e2', 'shear', 'ellipticity']):
        surveys.append('(Generic weak lensing catalog)')
    if any(x in cols_str for x in ['esd', 'delta_sigma', 'excess_surface']):
        surveys.append('(ESD / galaxy-galaxy lensing profile)')
    return surveys


def main():
    print("=" * 72)
    print("  WEAK LENSING CATALOG COMPARISON")
    print("  HSC-SSP Y3 vs KiDS DR4.1 SOM-gold WL")
    print("=" * 72)

    info1 = inspect_csv_gz(FILE_SUBARU)
    info2 = inspect_fits(FILE_KIDS)

    if info1 is None and info2 is None:
        print("\nERROR: Neither file could be read.")
        return

    section("SURVEY IDENTIFICATION")
    if info1:
        surveys1 = identify_survey(info1)
        print(f"\n  File 1: {', '.join(surveys1) if surveys1 else 'Unknown'}")
    if info2:
        surveys2 = identify_survey(info2)
        print(f"  File 2: {', '.join(surveys2) if surveys2 else 'Unknown'}")

    section("COLUMN MAPPING")
    map1 = map2 = {}
    if info1:
        map1 = identify_columns(info1)
        print(f"\n  File 1 key columns:")
        for k, v in map1.items():
            print(f"    {k:>10s} -> {v}")
    if info2:
        map2 = identify_columns(info2)
        print(f"\n  File 2 key columns:")
        for k, v in map2.items():
            print(f"    {k:>10s} -> {v}")

    if info1 and info2 and map1 and map2:
        section("LOADING DATA SAMPLES")
        print(f"  Loading up to {MAX_SAMPLE:,} rows from each...")

        data1 = {}
        data2 = {}
        try:
            if info1['type'] == 'csv':
                data1 = load_csv_sample(info1, map1)
            else:
                data1 = load_fits_sample(info1, map1)
            if data1:
                print(f"  File 1: loaded {len(next(iter(data1.values()))):,} rows")
        except Exception as e:
            print(f"  File 1 load error: {e}")

        try:
            if info2['type'] == 'fits':
                data2 = load_fits_sample(info2, map2)
            else:
                data2 = load_csv_sample(info2, map2)
            if data2:
                print(f"  File 2: loaded {len(next(iter(data2.values()))):,} rows")
        except Exception as e:
            print(f"  File 2 load error: {e}")

        if data1 and data2:
            compare_catalogs(data1, data2,
                             info1['path'].name, info2['path'].name)

    section("SUMMARY")
    print(f"""
  1. SURVEY IDENTITY: same or different?
  2. DATA INTEGRITY: RA [0,360], Dec [-90,90], |e1,e2|<1, z>0
  3. SAME region: cross-match ~1 arcsec, r > 0.8 expected
""")
    print("=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
