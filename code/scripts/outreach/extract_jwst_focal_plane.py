"""Extract JWST SIAF aperture geometry and project it onto the sky."""

import os
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.table import Table
import pysiaf
from pysiaf.utils import rotations


def extract_jwst_focal_plane():
    """Extract the defined quadrilateral apertures from the installed SIAF.

    Returns
    -------
    focal_plane : astropy.table.Table
        Aperture reference positions, orientations, ideal vertices, and telescope
        V2/V3 vertices. Angular coordinates are in arcseconds, except for
        V3IdlYAngle, which is in degrees. Main marks the standard display apertures.
        Entries without a complete quadrilateral definition are excluded.
    """
    main_apertures = {ap.AperName for group in pysiaf.siaf.get_main_apertures() for ap in group}
    attributes = ["V2Ref", "V3Ref", "V3IdlYAngle", "VIdlParity"]
    vertex_attributes = [f"{axis}IdlVert{i}" for axis in ["X", "Y"] for i in range(1, 5)]
    rows = []

    for instrument in ["NIRCam", "NIRSpec", "NIRISS", "MIRI", "FGS"]:
        for aperture in pysiaf.Siaf(instrument).apertures.values():
            values = [getattr(aperture, name) for name in attributes + vertex_attributes]
            if any(value is None for value in values) or not np.all(np.isfinite(values)):
                continue

            # Convert the published vertices using the standard SIAF convention.
            x_idl, y_idl = np.array(values[4:]).reshape(2, 4)
            v2, v3 = aperture.idl_to_tel(x_idl, y_idl)
            rows.append([instrument, aperture.AperName, aperture.AperType,
                         aperture.AperName in main_apertures, *values, *v2, *v3])

    names = ["Instrument", "AperName", "AperType", "Main", *attributes, *vertex_attributes]
    names += [f"V{axis}Vert{i}" for axis in [2, 3] for i in range(1, 5)]
    focal_plane = Table(rows=rows, names=names)
    for name in ["V2Ref", "V3Ref", *vertex_attributes, *names[-8:]]:
        focal_plane[name].unit = u.arcsec
    focal_plane["V3IdlYAngle"].unit = u.deg
    focal_plane.meta = {
        "pysiaf_version": pysiaf.__version__,
        "JWST_PRD_VERSION": pysiaf.JWST_PRD_VERSION,
        "source": "https://github.com/spacetelescope/pysiaf",
        "vertex_transform": "idl_to_tel: default planar_approximation (STScI example)",
        "scope": "Nominal SIAF quadrilaterals; not calibrated exposure coverage.",
    }
    return focal_plane


def project_jwst_focal_plane(focal_plane, reference_aperture, ra, dec, v3_pa):
    """Project aperture vertices using a pointing at an aperture reference point.

    Parameters
    ----------
    focal_plane : astropy.table.Table
        Geometry returned by extract_jwst_focal_plane, optionally filtered by aperture.
        Must include the reference aperture.
    reference_aperture : str
        SIAF aperture whose reference point is placed at ra, dec.
    ra, dec : float
        ICRS coordinates of that reference point in degrees.
    v3_pa : float
        Position angle of +V3 at that reference point, east of north in degrees.
        This is the local V3 angle (ROLL_REF in an exposure), not the detector PA
        or the V3 angle at the telescope boresight (PA_V3).

    Returns
    -------
    footprints : dict
        Aperture names mapped to (RA, Dec) arrays in degrees. Each polygon closes
        by repeating its first vertex.
    """
    reference = focal_plane[focal_plane["AperName"] == reference_aperture][0]
    attitude = rotations.attitude(reference["V2Ref"], reference["V3Ref"], ra, dec, v3_pa)
    footprints = {}
    for aperture in focal_plane:
        v2 = np.array([aperture[f"V2Vert{i}"] for i in [1, 2, 3, 4, 1]])
        v3 = np.array([aperture[f"V3Vert{i}"] for i in [1, 2, 3, 4, 1]])
        footprints[aperture["AperName"]] = rotations.pointing(attitude, v2, v3)
    return footprints


if __name__ == "__main__":
    instrument_data = Path(os.environ.get("INSTRUMENT_DATA", "/Users/Jonah/PhD/Research/instrument_data"))
    output_path = instrument_data / "jwst/geometry/jwst_focal_plane.ecsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    focal_plane = extract_jwst_focal_plane()
    focal_plane.write(output_path, format="ascii.ecsv", overwrite=True)
    print(f"Wrote {len(focal_plane)} apertures from {pysiaf.JWST_PRD_VERSION} to {output_path}")
