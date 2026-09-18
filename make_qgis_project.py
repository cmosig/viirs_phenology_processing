"""Build a QGIS project showing both v3 phenology GeoTIFFs.

Day of year is circular -- 31 December sits next to 1 January -- so a ramp with
two distinct ends would invent a seam in the middle of the boreal winter and
split the southern growing season in half. The ramp below is matplotlib's
`twilight`, which is cyclic and perceptually uniform, sampled at 13 stops and
wrapped onto day 1..366 so the first and last colour are the same.

Both stores are loaded on `middle_interp` (band 6), the variable the inference
date is taken from, with the 10 km store on top. The raw `middle` band (band 3,
NaN where MODIS derived no cycle) is added below, switched off, to see what is
measured and what is nearest-filled.

Run with the system Python, which has PyQGIS:
    /usr/bin/python3 make_qgis_project.py
"""

import os

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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phenology_v3.qgs")

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
    ("modis_pheno_processed_v3_10km.tif", 6, "v3 10 km - middle_interp", True),
    ("modis_pheno_processed_v3.tif", 6, "v3 40 km - middle_interp", True),
    ("modis_pheno_processed_v3_10km.tif", 3, "v3 10 km - middle (raw)", False),
    ("modis_pheno_processed_v3.tif", 3, "v3 40 km - middle (raw)", False),
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


def main():
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()

    project = QgsProject.instance()
    project.setTitle("MODIS phenology v3 - day of year")
    root = project.layerTreeRoot()

    for filename, band, name, visible in LAYERS:
        path = os.path.join(DATAPATH, filename)
        layer = QgsRasterLayer(path, name)
        if not layer.isValid():
            raise SystemExit(f"cannot load {path}")
        layer.setRenderer(_cyclic_renderer(layer, band))
        project.addMapLayer(layer, False)
        node = root.addLayer(layer)
        node.setItemVisibilityChecked(visible)
        print(f"added {name:28s} <- {filename}")

    project.write(OUT)
    app.exitQgis()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
