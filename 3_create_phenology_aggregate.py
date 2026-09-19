"""Build the 10 km, 10-year day-of-year composite from the MODIS phenology cube.

For every 10 km cell this counts, per day of year, how many forested pixel-years
had that day inside a growing season, from Onset_Greenness_Maximum (greenup) and
Onset_Greenness_Decrease (senescence) of both MODIS cycles.

The source cube is read tile by tile. Its chunks are (1, 2400, 2400), so the old
per-10-km-cell reads decompressed a whole 2400x2400 chunk for each 20x20 window;
one tile now covers 120x120 cells and every chunk is touched exactly once. The
interval marking is vectorised through a difference array instead of filling a
(years, 20, 20, 366) block per cell, and the output chunking matches one tile, so
each tile is a single chunk write rather than a read-modify-write of a 322^3 chunk.

    python 3_create_phenology_aggregate.py --workers 48
"""

import argparse
import datetime
from multiprocessing import Pool
from os.path import join

import dask.array
import numpy as np
import rasterio
import xarray as xr
import zarr
from numcodecs import Blosc
from rasterio import crs, windows
from tqdm import tqdm

from paths import DATAPATH

# Load the MODIS phenology data
variables_of_interest = ["Onset_Greenness_Maximum", "Onset_Greenness_Decrease"]

FOREST_MASK_PATH = "/net/scratch/cmosig/datasets/worldcover_aggregated.tif"

modiscrs = crs.CRS.from_string("""PROJCS["unnamed",\
GEOGCS["Unknown datum based upon the custom spheroid", \
DATUM["Not specified (based on custom spheroid)", \
SPHEROID["Custom spheroid",6371007.181,0]], \
PRIMEM["Greenwich",0],\
UNIT["degree",0.0174532925199433]],\
PROJECTION["Sinusoidal"], \
PARAMETER["longitude_of_center",0], \
PARAMETER["false_easting",0], \
PARAMETER["false_northing",0], \
UNIT["Meter",1]]""")

modis_x_size = 86400
modis_y_size = 33600
aggregation_factor = 20  # --> to 10km
modis_x_size_agg = modis_x_size // aggregation_factor
modis_y_size_agg = modis_y_size // aggregation_factor

TILE = 2400  # source chunk size, in 500 m pixels
TILE_CELLS = TILE // aggregation_factor  # 120 cells of 10 km per tile edge
N_CELLS = TILE_CELLS * TILE_CELLS
DAYS = 366
N_YEARS = 10

total_xmin = -20015109.354
total_xmax = 20015109.354
total_ymin = -6671703.118
total_ymax = 8895604.157333

pixel_size_x_agg = (total_xmax - total_xmin) / modis_x_size_agg
pixel_size_y_agg = (total_ymax - total_ymin) / modis_y_size_agg


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="modispheno_aggregated_v8.zarr",
                        help="output zarr under DATAPATH (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=48,
                        help="worker processes (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many tiles; for smoke tests")
    return parser.parse_args()


def day_offsets(time_labels):
    """Days from 1 Jan of each layer's year back to 2000-01-01.

    The layers are days since 2000-01-01, so turning them into day of year means
    subtracting the real day count -- NOT 366 * (year - 2000), which over-subtracts
    by 9 d in 2013 growing to 16 d in 2022 (~0.75 d/yr). That drift biased every
    date early and smeared the 10-year composite: in central Germany the true peak
    greenness DOY is 152 in both 2013 and 2022, the old formula returned 143 and 136.
    """
    return np.array([
        -(datetime.date(int(t.split("_")[0]), 1, 1) -
          datetime.date(2000, 1, 1)).days for t in time_labels
    ])


def segments(start, end):
    """Split [start, end) into day-of-year segments; empty ones have e <= s.

    Derived dates legitimately fall outside [0, 366): a cycle peaking in November
    and senescing the following February becomes start=324, end=406 against 1 Jan
    of the layer's year, and a greenup in the previous December comes out negative.
    Clipping at the year boundary would drop the January half of such a season;
    the composite is a day-of-year histogram, i.e. cyclic, so the tail goes to the
    start of the same year. Two segments are enough: a season is at most a year.
    """
    length = np.minimum(end - start, DAYS)
    first = np.mod(start, DAYS)
    stop = first + length
    return (first, np.minimum(stop, DAYS), np.zeros_like(start),
            np.maximum(stop - DAYS, 0))


def add_intervals(diff, cell, start, end, sign):
    """diff[cell, day] += sign for every day in [start, end).

    A difference array plus one cumulative sum at the end replaces marking every
    day of every interval, which is what made the per-cell version slow.
    """
    keep = end > start
    if not keep.any():
        return
    idx = cell[keep].astype(np.int64) * (DAYS + 1)
    size = diff.shape[0] * (DAYS + 1)
    flat = (np.bincount(idx + start[keep], minlength=size) -
            np.bincount(idx + end[keep], minlength=size))
    diff += sign * flat.reshape(diff.shape).astype(diff.dtype)


def cycle_intervals(onset_max, onset_dec):
    """The season of one cycle per pixel.

    A pixel-year contributes a season only when it has both of its own dates:

    senescence after maturity  -> [maturity, senescence);
    senescence before maturity -> [maturity, senescence + 366), a season that
    crosses New Year.

    Anything else does not contribute. v6 and earlier had three fallbacks that
    invented a missing endpoint, and each of them anchored an interval on a year
    boundary rather than on anything measured:

    * senescence before maturity -> `[0, senescence)`, which threw away the half
      of the season before New Year;
    * maturity missing -> `[0, senescence)`, a start the pixel never had;
    * senescence missing -> `[maturity, 366)`, an end the pixel never had.

    Outside the temperate zone those are most of the data: 58% of the
    contributing pixel-years over the Amazon, 36% over Borneo, 0.4% over
    Germany. Their intervals stack on day 0 and day 365, the count curve steps
    at the year boundary, and the peak rule reads the step as the mid-season
    date. v6 put 42,479 cells on day 0, 19x what an average day holds. v7
    removed the first two fallbacks and the pile moved to the other end of the
    year: 53,977 cells on day 365, 25x the mean. Only dropping all three takes
    the year boundary out of the histogram, which is what v8 does.

    Both cycles are treated the same: a senescence before its own maturity means
    the same thing in either.
    """
    have_max = ~np.isnan(onset_max)
    have_dec = ~np.isnan(onset_dec)
    both = have_max & have_dec

    after = np.zeros(onset_max.shape, dtype=bool)
    np.greater(onset_dec, onset_max, out=after, where=both)
    before = np.zeros(onset_max.shape, dtype=bool)
    np.less(onset_dec, onset_max, out=before, where=both)

    wrapped = both & before
    # senescence exactly on maturity is a zero-length season, dropped as before
    valid = (both & after) | wrapped

    start = np.nan_to_num(onset_max)
    end = np.where(wrapped,
                   np.nan_to_num(onset_dec) + float(DAYS),
                   np.nan_to_num(onset_dec))

    # astype truncates toward zero, as int() did per pixel
    return valid, start.astype(np.int32), end.astype(np.int32)


def tile_counts(reader, y0, x0, forest, time_labels):
    """Per-cell day-of-year counts for one tile.

    Returns (counts, has_source); cells without source data or forest stay NaN.
    """
    offsets = day_offsets(time_labels)
    cell = np.repeat(np.repeat(
        np.arange(N_CELLS, dtype=np.int32).reshape(TILE_CELLS, TILE_CELLS),
        aggregation_factor, axis=0), aggregation_factor, axis=1).ravel()

    diff = np.zeros((N_CELLS, DAYS + 1), dtype=np.int32)
    has_source = np.zeros(N_CELLS, dtype=bool)
    # A cell without a single forested pixel is left empty rather than written as
    # zeros: a flat curve has no cycle to extract, and the per-cell version never
    # visited such cells, so the interpolation filled them from their neighbours.
    has_forest = forest.reshape(TILE_CELLS, aggregation_factor, TILE_CELLS,
                                aggregation_factor).any(axis=(1, 3)).ravel()
    max_var, dec_var = variables_of_interest

    for year in range(N_YEARS):
        t1, t2 = 2 * year, 2 * year + 1
        raw = {(var, t): reader(var, t, y0, x0)
               for var in variables_of_interest for t in (t1, t2)}

        present = np.zeros((TILE, TILE), dtype=bool)
        for array in raw.values():
            present |= ~np.isnan(array)
        # "no source data at all" is tested before the forest mask, as before
        has_source |= present.reshape(TILE_CELLS, aggregation_factor, TILE_CELLS,
                                      aggregation_factor).any(axis=(1, 3)).ravel()

        masked = {key: np.where(forest, array, np.nan)
                  for key, array in raw.items()}
        valid_1, start_1, end_1 = cycle_intervals(masked[(max_var, t1)] + offsets[t1],
                                                  masked[(dec_var, t1)] + offsets[t1])
        valid_2, start_2, end_2 = cycle_intervals(masked[(max_var, t2)] + offsets[t2],
                                                  masked[(dec_var, t2)] + offsets[t2])
        if not (valid_1.any() or valid_2.any()):
            continue

        valid_1, valid_2 = valid_1.ravel(), valid_2.ravel()
        start_1, end_1 = start_1.ravel(), end_1.ravel()
        start_2, end_2 = start_2.ravel(), end_2.ravel()
        both = valid_1 & valid_2

        seg_1 = segments(start_1, end_1)
        seg_2 = segments(start_2, end_2)
        for start, end in ((seg_1[0], seg_1[1]), (seg_1[2], seg_1[3])):
            add_intervals(diff, cell[valid_1], start[valid_1], end[valid_1], 1)
        for start, end in ((seg_2[0], seg_2[1]), (seg_2[2], seg_2[3])):
            add_intervals(diff, cell[valid_2], start[valid_2], end[valid_2], 1)
        # The two cycles of one pixel-year are a union, not a sum: a day both of
        # them cover must not be counted twice.
        if both.any():
            for a_start, a_end in ((seg_1[0], seg_1[1]), (seg_1[2], seg_1[3])):
                for b_start, b_end in ((seg_2[0], seg_2[1]), (seg_2[2], seg_2[3])):
                    add_intervals(diff, cell[both],
                                  np.maximum(a_start, b_start)[both],
                                  np.minimum(a_end, b_end)[both], -1)

    has_source &= has_forest
    counts = np.cumsum(diff, axis=1)[:, :DAYS].astype(np.float32)
    counts[~has_source] = np.nan
    return counts.reshape(TILE_CELLS, TILE_CELLS, DAYS), has_source


_state = {}


def init_worker(out):
    dataset = xr.open_zarr(join(DATAPATH, "modispheno.zarr"),
                           chunks=None)[variables_of_interest]
    _state["dataset"] = dataset
    _state["time_labels"] = list(dataset.time.values)
    _state["mask"] = rasterio.open(FOREST_MASK_PATH)
    _state["out"] = zarr.open(join(DATAPATH, out), mode="r+")["phenology"]


def read_window(var, t, y0, x0):
    return _state["dataset"][var].isel(time=t,
                                       y=slice(y0, y0 + TILE),
                                       x=slice(x0, x0 + TILE)).values


def process_tile(tile):
    """One source chunk: 120x120 cells of 10 km."""
    y0, x0 = tile
    # The zarr y axis ascends northward while the mask GeoTIFF is north-up, so the
    # window covering zarr rows [y0, y0+TILE) starts at tif row N - y0 - TILE.
    # Without the -TILE the mask was read a full tile too far south, and at y0 = 0
    # the window fell off the raster entirely.
    window = windows.Window(col_off=x0,
                            row_off=modis_y_size - y0 - TILE,
                            width=TILE,
                            height=TILE)
    # ...and within that window the tif still runs north to south, so row i holds
    # zarr row y0 + TILE - 1 - i: flip it to match the cube's y axis.
    forest = (_state["mask"].read(1, window=window) > 0.5)[::-1]
    if not forest.any():
        return y0, x0, 0

    counts, has_source = tile_counts(read_window, y0, x0, forest,
                                      _state["time_labels"])
    if not has_source.any():
        return y0, x0, 0

    ys = slice(y0 // aggregation_factor, y0 // aggregation_factor + TILE_CELLS)
    xs = slice(x0 // aggregation_factor, x0 // aggregation_factor + TILE_CELLS)
    _state["out"][ys, xs, :] = counts
    return y0, x0, int(has_source.sum())


def create_store(path):
    """Empty store, chunked so that one tile is exactly one chunk."""
    dataset = xr.Dataset(
        data_vars=dict(phenology=xr.DataArray(data=dask.array.empty(
            (modis_y_size_agg, modis_x_size_agg, DAYS),
            dtype=np.float32,
            chunks=(TILE_CELLS, TILE_CELLS, DAYS)),
                                              dims=("y", "x", "day"))),
        coords=dict(
            y=np.linspace(total_ymin, total_ymax, modis_y_size_agg,
                          endpoint=False) + pixel_size_y_agg / 2,
            x=np.linspace(total_xmin, total_xmax, modis_x_size_agg,
                          endpoint=False) + pixel_size_x_agg / 2,
            day=np.arange(DAYS)),
    )
    dataset.to_zarr(path,
                    compute=False,
                    zarr_format=2,
                    encoding=dict(
                        phenology={
                            "chunks": (TILE_CELLS, TILE_CELLS, DAYS),
                            "write_empty_chunks": False,
                            "compressor": Blosc(cname="lz4"),
                            "_FillValue": np.nan,
                        }))


def main():
    args = _parse_args()
    create_store(join(DATAPATH, args.out))
    print(f"created {args.out}")

    tiles = [(y, x) for y in range(0, modis_y_size, TILE)
             for x in range(0, modis_x_size, TILE)]
    if args.limit is not None:
        tiles = tiles[:args.limit]

    cells = 0
    with Pool(args.workers, initializer=init_worker,
              initargs=(args.out,)) as pool:
        for _, _, n in tqdm(pool.imap_unordered(process_tile, tiles),
                            total=len(tiles),
                            unit="tile"):
            cells += n
    print(f"wrote {cells:,} cells of 10 km")


if __name__ == "__main__":
    main()
