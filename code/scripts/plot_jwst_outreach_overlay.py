"""Create transparent JWST field outlines for outreach slides."""

from pathlib import Path

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.table import Table


project_root = Path(__file__).resolve().parents[2]
instrument_data = Path(os.environ.get("INSTRUMENT_DATA", "/Users/Jonah/PhD/Research/instrument_data"))
geometry = Table.read(instrument_data / "jwst/geometry/jwst_focal_plane.ecsv")
output_directory = project_root / "figures/outreach"
output_directory.mkdir(parents=True, exist_ok=True)

# Show one representative field per instrument and retain the NIRCam chip gaps.
aperture_names = {
    "NIRCam": [f"NRC{module}{chip}_FULL" for module in ["A", "B"] for chip in range(1, 5)],
    "NIRSpec": [f"NRS_FULL_MSA{quadrant}" for quadrant in range(1, 5)],
    "MIRI": ["MIRIM_ILLUM"],
    "NIRISS": ["NIS_CEN"],
}
colors = {"NIRCam": "#55d6ff", "NIRSpec": "#ffca62", "MIRI": "#ff6eaa",
          "NIRISS": "#78e6a4"}

for labelled in [True, False]:
    fig, ax = plt.subplots(figsize=(14, 7))
    all_vertices = []
    for instrument, names in aperture_names.items():
        instrument_vertices = []
        for aperture_name in names:
            row = geometry[geometry["AperName"] == aperture_name][0]
            vertices = np.array([[row[f"V2Vert{i}"], row[f"V3Vert{i}"]] for i in range(1, 5)]) / 60
            instrument_vertices.extend(vertices)
            ax.add_patch(Polygon(vertices, facecolor=colors[instrument], edgecolor="none", alpha=0.12))
            ax.add_patch(Polygon(vertices, fill=False, edgecolor=colors[instrument], lw=1.8))

        instrument_vertices = np.array(instrument_vertices)
        all_vertices.extend(instrument_vertices)
        if labelled:
            label_x = (instrument_vertices[:, 0].min() + instrument_vertices[:, 0].max()) / 2
            label_y = instrument_vertices[:, 1].max() + 0.22
            ax.text(label_x, label_y, instrument, color=colors[instrument], fontsize=17,
                    weight="bold", ha="center", va="bottom")

    all_vertices = np.array(all_vertices)
    ax.set_xlim(all_vertices[:, 0].max() + 0.25, all_vertices[:, 0].min() - 0.25)
    ax.set_ylim(all_vertices[:, 1].min() - 0.2, all_vertices[:, 1].max() + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    suffix = "labelled" if labelled else "unlabelled"
    for extension in ["png", "svg"]:
        fig.savefig(output_directory / f"jwst_overlay_{suffix}.{extension}", dpi=300,
                    transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
