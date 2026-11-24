from pathlib import Path
import warnings

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

MICRONS_UNIT = u.def_unit("microns", u.um)

JWST_FILTERS = [
    "F070W",
    "F090W",
    "F115W",
    "F150W",
    "F162M",
    "F182M",
    "F200W",
    "F210M",
    "F250M",
    "F277W",
    "F300M",
    "F335M",
    "F356W",
    "F410M",
    "F430M",
    "F444W",
    "F460M",
    "F480M",
    "F560W",
    "F770W",
    "F1000W",
    "F1280W",
    "F1500W",
    "F1800W",
    "F2100W",
    "F2550W",
]

MIRI_FILTERS = [
    "F560W",
    "F770W",
    "F1000W",
    "F1280W",
    "F1500W",
    "F1800W",
    "F2100W",
    "F2550W",
]

SEDPY_FILTER_MAP = {
    "F070W": "jwst_f070w",
    "F090W": "jwst_f090w",
    "F1000W": "jwst_f1000w",
    "F115W": "jwst_f115w",
    "F1280W": "jwst_f1280w",
    "F1500W": "jwst_f1500w",
    "F150W": "jwst_f150w",
    "F162M": "jwst_f162m",
    "F1800W": "jwst_f1800w",
    "F182M": "jwst_f182m",
    "F200W": "jwst_f200w",
    "F2100W": "jwst_f2100w",
    "F210M": "jwst_f210m",
    "F250M": "jwst_f250m",
    "F2550W": "jwst_f2550w",
    "F277W": "jwst_f277w",
    "F300M": "jwst_f300m",
    "F335M": "jwst_f335m",
    "F356W": "jwst_f356w",
    "F410M": "jwst_f410m",
    "F430M": "jwst_f430m",
    "F435W": "acs_wfc_f435w",
    "F444W": "jwst_f444w",
    "F460M": "jwst_f460m",
    "F480M": "jwst_f480m",
    "F560W": "jwst_f560w",
    "F606W": "acs_wfc_f606w",
    "F770W": "jwst_f770w",
    "F775W": "acs_wfc_f775w",
    "F814W": "acs_wfc_f814w",
}


def match_coord(cat, ra, dec, radius=0.2):
    """Match a given RA and DEC to the nearest source in a catalog within a specified radius."""
    
    cat_coords = SkyCoord(cat["RA"], cat["DEC"], unit=u.deg)
    target = SkyCoord(ra * u.deg, dec * u.deg)

    sep = target.separation(cat_coords)
    idx = np.argmin(sep)

    if sep[idx] > radius * u.arcsec:
        raise ValueError(
            f"No match within {radius}\"; nearest is {sep[idx].arcsec:.3f}\""
        )

    return idx, sep[idx]


def create_jades_galaxy_photometry_table(
    nircam_row,
    miri_row,
    filters,
    waveffs,
    nircam_phot_type="KRON",
    miri_phot_type="KRON",
):
    """Create a table of photometry for a given galaxy from the JADES catalog."""

    tb = Table()

    use_filters = []
    use_sedpy_filters = []
    use_waveffs = []
    use_data = []
    use_err = []
    use_valid = []

    for filt, waveff in zip(filters, waveffs):
        if filt not in JWST_FILTERS:
            continue

        if filt in MIRI_FILTERS:
            row = miri_row
            phot_type = miri_phot_type
        else:
            row = nircam_row
            phot_type = nircam_phot_type

        if row is None:
            continue

        data_col = f"{filt.upper()}_{phot_type}"
        err_col = f"{filt.upper()}_{phot_type}_e"
        if data_col not in row.colnames or err_col not in row.colnames:
            continue

        data = row[data_col]
        err = row[err_col]

        # Skip masked or null photometry
        if np.ma.is_masked(data) or np.ma.is_masked(err):
            continue

        data = float(data)
        err = float(err)
        if not np.isfinite(data) or not np.isfinite(err):
            continue

        use_filters.append(filt)
        use_sedpy_filters.append(SEDPY_FILTER_MAP[filt])
        use_waveffs.append(waveff)
        use_data.append(data)
        use_err.append(err)
        use_valid.append(1)

    tb["FILTER"] = use_filters
    tb["SEDPY_FILTER"] = use_sedpy_filters
    tb["WAVE_PIVOT"] = use_waveffs
    tb["DATA"] = use_data
    tb["ERR"] = use_err
    tb["VALID"] = use_valid
    tb["WAVE_PIVOT"].unit = u.um
    tb["DATA"].unit = u.nJy
    tb["ERR"].unit = u.nJy

    return tb


def write_photometry_table(tb, out_file, nirspec_id, nircam_id, ra, dec):
    """Write a photometry table with source metadata in the primary header."""

    header = fits.Header()
    header["HIERARCH NIRSPECID"] = nirspec_id
    header["NIRCAMID"] = nircam_id
    header["RA"] = float(ra)
    header["DEC"] = float(dec)

    primary_hdu = fits.PrimaryHDU(header=header)
    table_hdu = fits.table_to_hdu(tb)
    fits.HDUList([primary_hdu, table_hdu]).writeto(out_file, overwrite=True)


def main():

    RESEARCH_DIR = Path("/Users/Jonah/PhD/Research")
    phot_dir = RESEARCH_DIR / "lapis" / "data_processed" / "photometry"
    cat_dir = RESEARCH_DIR / "lapis" / "data_external" / "JADES" / "dr5" / "catalogues"

    # Load the JADES photometry catalogues
    gs_cat_file = "hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits"
    gn_cat_file = "hlsp_jades_jwst_nircam_goods-n_photometry_v5.0_catalog.fits"

    nircam_hdu = "KRON"
    miri_hdu = "MIRI"
    gs_cat_tb = Table.read(cat_dir / gs_cat_file, hdu=nircam_hdu)
    gn_cat_tb = Table.read(cat_dir / gn_cat_file, hdu=nircam_hdu)
    gs_miri_tb = Table.read(cat_dir / gs_cat_file, hdu=miri_hdu)
    gn_miri_tb = Table.read(cat_dir / gn_cat_file, hdu=miri_hdu)

    # Load filter information
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'microns' did not parse as fits unit.*")
        with u.add_enabled_units([MICRONS_UNIT]):
            filt_tb = Table.read(cat_dir / gs_cat_file, hdu="FILTERS")
    filters = filt_tb["BAND"]
    wave_pivots = filt_tb["WAVE_PIVOT"]

    # Combine into single table
    cat_tb = vstack([gs_cat_tb, gn_cat_tb], metadata_conflicts="silent")

    # Define RA and DEC for target sample
    baker_coords = {
        72127: (189.26572,  62.16839),
        1397:  (53.06227,  -27.87504),
        8290:  (53.08188,  -27.82880),
        80660: (189.27545,  62.21414),
        8777:  (53.10821,  -27.82519),
    }

    # Select the nearest match in the JADES catalogue
    cat_ids = []

    for baker_id in baker_coords:
        idx, sep = match_coord(cat_tb, *baker_coords[baker_id])
        cat_ids.append(idx)

    # Extract JADES photometry for the matched sources
    for baker_id, cat_id in zip(baker_coords, cat_ids):
        baker_row = cat_tb[cat_id]
        if cat_id < len(gs_cat_tb):
            miri_tb = gs_miri_tb
        else:
            miri_tb = gn_miri_tb

        miri_match = np.where(miri_tb["ID"] == baker_row["ID"])[0]
        miri_row = miri_tb[miri_match[0]] if len(miri_match) > 0 else None

        # Create a photometry table for the matched source
        phot_tb = create_jades_galaxy_photometry_table(
            baker_row,
            miri_row,
            filters,
            wave_pivots,
            nircam_phot_type="KRON",
            miri_phot_type="KRON",
        )
        phot_tb_file = f"{baker_id}_jades_dr5_photometry.fits"
        write_photometry_table(
            phot_tb,
            phot_dir / phot_tb_file,
            nirspec_id=int(baker_id),
            nircam_id=int(baker_row["ID"]),
            ra=float(baker_row["RA"]),
            dec=float(baker_row["DEC"]),
        )

if __name__ == "__main__":
    main()
