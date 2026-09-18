"""Build a QGIS project showing both v3 phenology GeoTIFFs.

Day of year is circular -- 31 December sits next to 1 January -- so a ramp with
two distinct ends would invent a seam in the middle of the boreal winter and
split the southern growing season in half. The ramp below is matplotlib's
`twilight`, which is cyclic and perceptually uniform, sampled at 13 stops and
wrapped onto day 1..366 so the first and last colour are the same.

The raw `middle` band (band 3) is what the layers show by default: NaN wherever
MODIS derived no cycle, so gaps stay visible instead of being papered over. The
`middle_interp` band (band 6), which is what the inference date is actually read
from, carries the same values plus a nearest-valid fill and is loaded below,
switched off. v6 is visible at both resolutions; v2, the store inference uses today, is there
for comparison.

The GeoTIFFs are copied next to the project inside `qgis/` (6 MB for both) and
referenced by bare filename, so a clone of this repo opens on any machine. They
are copied from DATAPATH when missing; --refresh re-copies them.

Run with the system Python, which has PyQGIS (headless needs
QT_QPA_PLATFORM=offscreen):
    /usr/bin/python3 make_qgis_project.py
"""

import argparse
import os
import shutil

from qgis.core import (
    QgsApplication,
    QgsColorRampShader,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtGui import QColor

from paths import DATAPATH

HERE = os.path.dirname(os.path.abspath(__file__))
QGIS_DIR = os.path.join(HERE, "qgis")
OUT = os.path.join(QGIS_DIR, "phenology.qgs")

# matplotlib twilight, 13 stops, first == last
TWILIGHT = [
    (225, 216, 226),
    (179, 198, 206),
    (123, 160, 194),
    (97, 117, 186),
    (93, 67, 164),
    (77, 23, 110),
    (47, 20, 54),
    (88, 21, 71),
    (141, 44, 80),
    (178, 86, 82),
    (198, 137, 109),
    (211, 187, 171),
    (225, 216, 226),
]

MONTH_STARTS = [(1, "1 Jan"), (32, "1 Feb"), (60, "1 Mar"), (91, "1 Apr"),
                (121, "1 May"), (152, "1 Jun"), (182, "1 Jul"), (213, "1 Aug"),
                (244, "1 Sep"), (274, "1 Oct"), (305, "1 Nov"), (335, "1 Dec"),
                (366, "31 Dec")]

DOY_MIN, DOY_MAX = 1, 366

# (file, band, layer name, visible)
LAYERS = [
    ("modis_pheno_processed_v6_10km.tif", 3, "v6 10 km - middle (measured)", True),
    ("modis_pheno_processed_v6.tif", 3, "v6 40 km - middle (measured)", True),
    ("modis_pheno_processed_v2.tif", 3, "v2 40 km - middle (measured, in use today)", False),
    ("modis_pheno_processed_v6_10km.tif", 6, "v6 10 km - middle_interp", False),
    ("modis_pheno_processed_v6.tif", 6, "v6 40 km - middle_interp", False),
    ("modis_pheno_processed_v2.tif", 6, "v2 40 km - middle_interp (in use today)", False),
]


def _ramp_items():
    """Colour stops in day-of-year, labelled by month."""
    items = []
    span = DOY_MAX - DOY_MIN
    for i, (r, g, b) in enumerate(TWILIGHT):
        doy = DOY_MIN + span * i / (len(TWILIGHT) - 1)
        label = min(MONTH_STARTS, key=lambda m: abs(m[0] - doy))[1]
        items.append(
            QgsColorRampShader.ColorRampItem(doy, QColor(r, g, b), label))
    return items


def _cyclic_renderer(layer, band):
    ramp = QgsColorRampShader(DOY_MIN, DOY_MAX, None,
                              QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(_ramp_items())
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band,
                                                shader)
    # Fixed 1..366, never a per-layer stretch: the two stores have to stay
    # comparable, and a stretch would also break the wrap-around.
    renderer.setClassificationMin(DOY_MIN)
    renderer.setClassificationMax(DOY_MAX)
    return renderer


def _local_copy(filename, refresh):
    """The tif next to the project, copied from DATAPATH on first use."""
    local = os.path.join(QGIS_DIR, filename)
    if refresh or not os.path.exists(local):
        shutil.copyfile(os.path.join(DATAPATH, filename), local)
        print(f"copied {filename} -> qgis/")
    return local


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-copy the GeoTIFFs from DATAPATH")
    args = ap.parse_args()

    os.makedirs(QGIS_DIR, exist_ok=True)
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()

    project = QgsProject.instance()
    project.setTitle("Land-surface phenology v6 - day of year")
    root = project.layerTreeRoot()
    crs_set = False

    for filename, band, name, visible in LAYERS:
        path = _local_copy(filename, args.refresh)
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            raise SystemExit(f"cannot load {path}")
        layer.setRenderer(_cyclic_renderer(layer, band))
        project.addMapLayer(layer, False)
        node = root.addLayer(layer)
        node.setItemVisibilityChecked(visible)
        if not crs_set:
            # Without this the project opens with no CRS at all and QGIS asks for
            # one; the canvas matches the data, so nothing is reprojected.
            project.setCrs(layer.crs())
            crs_set = True
        print(f"added {name:28s} <- {filename}")

    # Store sources relative to the project file, which now sits next to the
    # tifs, so the paths survive a clone onto another machine.
    project.writeEntryBool("Paths", "/Absolute", False)
    project.write(OUT)
    app.exitQgis()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
