"""Normalise the day-of-year composite to 0-255 per cell, with a missing mask.

Each cell's count curve is min-max scaled to its own range, so the curve says
*when* the season is, not how much forest the cell holds. Cells carried by too
few pixel-years are marked missing here too, for the same reason the dates
product drops them (see the note in 4_aggregate_to_dates.py): a curve built from
a handful of pixel-years is noise. The downstream nearest-neighbour fill
(sentinel_mortality/scripts/misc/fill_modis_phenology_nn.py) replaces every
masked cell.
"""

import argparse
from os.path import join

import numpy as np
import xarray as xr

from paths import DATAPATH

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aggregate", default="modispheno_aggregated_v5.zarr",
                    help="input composite under DATAPATH (default: %(default)s)")
parser.add_argument("--out", default=None,
                    help="output name (default: the input with _normalized)")
parser.add_argument("--min-pixel-years", type=int, default=200,
                    help=("mark cells carried by fewer than this many pixel-years as "
                          "missing (default: %(default)s; 0 disables)"))
args = parser.parse_args()

path = join(DATAPATH, args.aggregate)
out_path = join(DATAPATH, args.out or args.aggregate.replace(".zarr", "_normalized.zarr"))

ds = xr.open_zarr(path).load()

sparse = ds.phenology.max(dim="day") < args.min_pixel_years
print(f"cells below {args.min_pixel_years} pixel-years: {int(sparse.sum()):,}")

# min-max scale each cell's own curve
ds = ds - ds.min(dim="day")
ds = ds / ds.max(dim="day")
ds = ds * 255

nan_mask = ds.phenology.isnull().any(dim="day") | sparse
phenology = ds.phenology.fillna(0).astype("uint8")

ds = xr.Dataset(
    {
        "phenology": (["y", "x", "day"], phenology.data),
        "nan_mask": (["y", "x"], nan_mask.data),
    },
    coords={"y": ds.y.data, "x": ds.x.data, "day": ds.day.data},
)

print(f"missing cells (no curve or too sparse): {int(nan_mask.sum()):,} "
      f"of {nan_mask.size:,}")
# zarr v2, like every other store here: the nearest-neighbour fill downstream
# works on v2 arrays.
ds.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
print(f"wrote {out_path}")
