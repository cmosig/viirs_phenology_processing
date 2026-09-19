"""Export a modis_pheno_processed_v3*.zarr store to a 6-band GeoTIFF.

Bands are the same six variables the zarr carries, in order:
    1 start   2 end   3 middle   4 start_interp   5 end_interp   6 middle_interp

Values are day-of-year (the raw bands are NaN where MODIS derived no cycle; the *_interp
bands are nearest-valid-filled everywhere). CRS is the MODIS sinusoidal projection.

The zarr y axis ascends northward, so the array is flipped to the north-up row order
GeoTIFF expects.

Usage:
    python export_v3_geotiff.py                       # 40 km
    python export_v3_geotiff.py --factor 1            # 10 km
    python export_v3_geotiff.py --store <name.zarr> --out <file.tif>
"""

import argparse
from os.path import join

import numpy as np
import rasterio
import xarray as xr
from rasterio import crs, transform

from paths import DATAPATH

# Named, so QGIS shows "MODIS Sinusoidal" rather than "unnamed". There is no
# authority code for this grid: the MODIS/VIIRS sphere is R = 6371007.181 m,
# while ESRI:53008 uses 6371000, so it stays a custom CRS. Parameters are
# unchanged -- this only affects how the CRS is labelled.
MODIS_CRS = crs.CRS.from_string("""PROJCS["MODIS Sinusoidal",
GEOGCS["Unknown datum based upon the custom spheroid",
DATUM["Not specified (based on custom spheroid)",
SPHEROID["Custom spheroid",6371007.181,0]],
PRIMEM["Greenwich",0],
UNIT["degree",0.0174532925199433]],
PROJECTION["Sinusoidal"],
PARAMETER["longitude_of_center",0],
PARAMETER["false_easting",0],
PARAMETER["false_northing",0],
UNIT["Meter",1]]""")

BANDS = ["start", "end", "middle", "start_interp", "end_interp", "middle_interp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor", type=int, default=4,
                    help="4 -> the 40 km store (default), 1 -> the 10 km store")
    ap.add_argument("--version", default="v7",
                    help="version tag of the dates store (default: %(default)s)")
    ap.add_argument("--store", default=None, help="explicit zarr name, overrides --factor")
    ap.add_argument("--out", default=None, help="explicit output .tif path")
    args = ap.parse_args()

    suffix = "" if args.factor == 4 else f"_{10 * args.factor}km"
    store = args.store or f"modis_pheno_processed_{args.version}{suffix}.zarr"
    out = args.out or join(DATAPATH, store.replace(".zarr", ".tif"))

    da = xr.open_zarr(join(DATAPATH, store)).phenology40km.load()
    print(f"in : {join(DATAPATH, store)}  {dict(da.sizes)}")

    resx = float(da.x[1] - da.x[0])
    resy = float(da.y[1] - da.y[0])
    west = float(da.x[0]) - resx / 2
    east = float(da.x[-1]) + resx / 2
    south = float(da.y[0]) - resy / 2
    north = float(da.y[-1]) + resy / 2
    h, w = da.sizes["y"], da.sizes["x"]
    tr = transform.from_bounds(west, south, east, north, w, h)
    print(f"     {resx/1000:.2f} km px | bounds {west:.0f},{south:.0f},{east:.0f},{north:.0f}")

    # zarr rows run south -> north; GeoTIFF rows run north -> south
    data = np.stack([np.flipud(da.sel(var=b).values) for b in BANDS]).astype(np.float32)

    with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=len(BANDS),
                       dtype="float32", crs=MODIS_CRS, transform=tr, nodata=np.nan,
                       compress="DEFLATE", predictor=3, tiled=True,
                       blockxsize=256, blockysize=256) as dst:
        dst.write(data)
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)
        dst.update_tags(source=store, units="day of year",
                        note="raw bands are NaN where MODIS derived no cycle; "
                             "*_interp are nearest-valid filled")

    for i, b in enumerate(BANDS):
        v = data[i]
        print(f"     band {i+1} {b:<14} valid {np.isfinite(v).sum():>9,} "
              f"({100*np.isfinite(v).mean():5.1f}%)  range "
              f"{np.nanmin(v) if np.isfinite(v).any() else np.nan:.0f}-"
              f"{np.nanmax(v) if np.isfinite(v).any() else np.nan:.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
