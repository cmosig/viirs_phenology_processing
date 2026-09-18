import argparse

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import seaborn as sns
from paths import DATAPATH
from os.path import join

# --factor 4 (default) reproduces the shipped 40 km product; --factor 1 runs the same
# recipe on the native 10 km grid, for comparing what the 4x4 aggregation costs.
parser = argparse.ArgumentParser()
parser.add_argument("--factor", type=int, default=4,
                    help="spatial downsample factor from the 10 km grid (default 4 -> 40 km)")
parser.add_argument("--aggregate", default="modispheno_aggregated.zarr",
                    help="input composite from step 3 (default: %(default)s)")
parser.add_argument("--middle", choices=("peak", "threshold"), default="peak",
                    help=("how the middle date is placed: 'peak' (default) is the "
                          "day most pixel-years are in leaf, 'threshold' is the "
                          "midpoint between start and end, which v5 and earlier "
                          "used"))
parser.add_argument("--min-pixel-years", type=int, default=200,
                    help=("cells whose season is carried by fewer than this many "
                          "pixel-years are left to the interpolation (default: "
                          "%(default)s; see the note in the code)"))
parser.add_argument("--version", default="v3",
                    help="version tag of the output store and figure (default: %(default)s)")
args = parser.parse_args()
factor = args.factor
res_km = 10 * factor
suffix = "" if factor == 4 else f"_{res_km}km"

da = xr.open_zarr(join(DATAPATH, args.aggregate)).phenology
print("loading data...")
x = da.load().to_numpy()
x.shape

print(f"resampling... factor {factor} -> {res_km} km")
x_down = x.reshape(x.shape[0] // factor, factor, x.shape[1] // factor, factor,
                   x.shape[2])
# v3: nansum, not sum. Plain sum propagates NaN, so a 40km block went NaN if ANY of its
# 16 sub-cells was NaN instead of only if all were -- dropping every coastal/fragmented
# block and forcing it to be nearest-neighbour filled from far away (39% of blocks with
# at least one valid sub-cell).
all_nan_block = np.isnan(x_down).all(axis=(1, 3, 4))
x_down = np.nansum(x_down, axis=(1, 3))

# extract start, end, and middle from count curves
thresh = ((np.max(x_down, axis=2, keepdims=True) - np.min(x_down, axis=2, keepdims=True)) * 0.5) + np.min(x_down, axis=2, keepdims=True)
binmap = (x_down > thresh)


def get_cycle(arr):
    # detect up to two cycles

    # extract first cycle
    start_cycle_1 = np.argmax(arr)
    end_cycle_1 = np.argmin(arr[start_cycle_1:]) + start_cycle_1

    # extract second cycle if there is one
    second_cycle_exists = False
    start_cycle_2 = np.argmax(arr[end_cycle_1:]) + end_cycle_1
    second_cycle_exists = start_cycle_2 > end_cycle_1
    if second_cycle_exists:
        end_cycle_2 = np.argmin(arr[start_cycle_2:]) + start_cycle_2

        # if it cant find min -> then it goes until end
        end_cycle_2 = end_cycle_2 if end_cycle_2 > start_cycle_2 else 365

        # if first starts at 0 and second ends at 365 --> merge them --> done
        if start_cycle_1 == 0 and end_cycle_2 == 365:
            return_start = start_cycle_2
            return_end = end_cycle_1

        # else --> select longer one --> done
        if (end_cycle_1 - start_cycle_1) > (end_cycle_2 - start_cycle_2):
            return_start = start_cycle_1
            return_end = end_cycle_1
        else:
            return_start = start_cycle_2
            return_end = end_cycle_2

    else:
        # if only one cycle --> done
        return_start = start_cycle_1
        return_end = end_cycle_1

    # compute middle date
    if return_start < return_end:
        return_middle = return_start + ((return_end - return_start) // 2)
    else:
        return_middle = (return_end + ((return_end +
                                        (365 - return_start)) // 2)) % 366

    return np.array([return_start, return_end, return_middle],
                    dtype=np.float32)


print("extracting dates...")
cycle_dates = np.apply_along_axis(func1d=get_cycle, axis=2, arr=binmap)

if args.middle == "peak":
    # The date exists to pick the Sentinel-2 composite on which a leafless crown
    # most likely means a dead tree, so it belongs on the day the largest share of
    # pixel-years are in leaf, not halfway between the season bounds. On a
    # low-contrast tropical curve the midpoint drifts off the canopy plateau: over
    # the Amazon it sits below 80% of peak leaf-on in 20% of cells, while the peak
    # is at 100% by construction. Temperate and boreal dates move 2-3 d, inside one
    # composite step.
    #
    # Flat-topped curves have many days tied at the maximum, so the tied days are
    # averaged on the circle rather than taking the first, which would bias early
    # and jump between neighbouring cells.
    #
    # start and end stay the 50% crossings, but of the run that *contains the
    # peak* rather than the longest run. Picking by length leaves the peak outside
    # the kept season in 8% of Amazon cells (2.2% in SE Asia, 0% in Europe), which
    # would put middle outside [start, end]; following the peak also replaces the
    # length tie-break between two similar cycles with a meaningful one.
    print("placing the date on the peak of the curve...")
    tied = x_down >= x_down.max(axis=2, keepdims=True)
    peak_day = x_down.argmax(axis=2).astype(np.int32)

    # Walk out from the peak to the edges of the run it sits in, and of the
    # plateau of days tied with it. Averaging *all* tied days would put the middle
    # between two equal peaks in different seasons -- outside the run, and
    # describing neither season -- which matters because the in_pheno gate on the
    # training orthophotos tests `start <= acquisition <= end` against the same
    # triple.
    start = np.zeros(binmap.shape[:2], dtype=np.float32)
    end = np.zeros(binmap.shape[:2], dtype=np.float32)
    middle = np.zeros(binmap.shape[:2], dtype=np.float32)
    found = np.zeros(binmap.shape[:2], dtype=bool)
    # three copies, queried in the middle one: a run is at most a year long, so
    # both its edges are inside the window even when it wraps. Two copies are not
    # enough -- a 309-day run starting in the second copy runs off the end.
    days = np.arange(3 * 366, dtype=np.int16)
    def run_around_peak(mask_block, at):
        """[first, last+1) of the run of True containing `at`, on a tiled year."""
        tiled = np.concatenate([mask_block] * 3, axis=2)
        opens = tiled & ~np.roll(tiled, 1, axis=2)
        closes = tiled & ~np.roll(tiled, -1, axis=2)
        run_start = np.maximum.accumulate(
            np.where(opens, days, np.int16(-1)), axis=2)
        run_end = np.minimum.accumulate(
            np.where(closes, days + 1, np.int16(3 * 366))[:, :, ::-1],
            axis=2)[:, :, ::-1]
        first = np.take_along_axis(run_start, at, axis=2)[:, :, 0]
        last = np.take_along_axis(run_end, at, axis=2)[:, :, 0]
        return first, last, (first >= 0) & (last < 3 * 366)

    for lo in range(0, binmap.shape[0], 50):
        hi = min(lo + 50, binmap.shape[0])
        at = (peak_day[lo:hi] + 366)[:, :, None]
        s, e, ok_run = run_around_peak(binmap[lo:hi], at)
        ps, pe, ok_plateau = run_around_peak(tied[lo:hi], at)
        # midpoint of the plateau; it lies inside the run by construction
        mid = (ps + (pe - 1 - ps) / 2.0)
        start[lo:hi] = np.where(ok_run, s % 366, 0)
        end[lo:hi] = np.where(ok_run, e % 366, 0)
        middle[lo:hi] = np.where(ok_plateau, mid % 366, peak_day[lo:hi])
        found[lo:hi] = ok_run
    # a flat curve has no run at all; those cells keep the old triple and are
    # dropped below anyway if they are too sparse
    cycle_dates[found] = np.stack([start, end, middle], axis=2)[found]
    print(f"  peak inside a run for {100 * found.mean():.1f}% of cells")
# must follow the nansum above: nansum returns 0 for an all-NaN block, never NaN, so the
# no-data test has to come from the pre-aggregation array instead of from x_down.
cycle_dates[all_nan_block] = np.nan

# A 10 km cell holds at most 400 forested pixels x 10 years; a coastal or
# fragmented one holds a handful, and a date read off a curve built from a
# handful of pixel-years is unstable -- these are the isolated specks along
# coastlines. Subsampling well-sampled cells (>=2000 pixel-years, 120 cells over
# 4 tiles, 15 draws each) and comparing against their own full-sample date gives
# the 95th percentile of the sampling error:
#     5 px-yrs: 82 d   20: 27 d   50: 13 d   100: 8 d   200: 5 d   400: 3 d
# The date is only ever used to pick a 7-day composite, so the threshold is set
# where that error falls below the composite step: 200 pixel-years (p95 = 5 d).
# Below it the cell is dropped and filled by the interpolation below, which is
# what happens to a cell with no measured cycle at all.
sparse = np.nanmax(x_down, axis=2) < args.min_pixel_years
cycle_dates[sparse & ~all_nan_block] = np.nan
print(f"dropped {int((sparse & ~all_nan_block).sum()):,} cells with fewer than "
      f"{args.min_pixel_years} pixel-years "
      f"({100 * (sparse & ~all_nan_block).sum() / max((~all_nan_block).sum(), 1):.1f}% "
      f"of the measured cells)")

print("interpolating...")
# One nearest donor for the whole triple. Filling start, end and middle
# independently lets a cell take its start from one neighbour and its middle from
# another, so the filled season is not a season anyone measured -- and the
# in_pheno gate reads exactly these three bands.
missing = np.isnan(cycle_dates[:, :, 0])
_, donor = distance_transform_edt(missing, return_indices=True)
cin = cycle_dates[donor[0], donor[1], :]

print("plotting...")
# plotting
titles = [
    "Start of Range",
    "End of Range",
    "Middle of Range",
]
fig, axes = plt.subplots(nrows=6, figsize=(15, 15))

for i in range(3):
    axes[i * 2].set_title(titles[i])
    axes[i * 2 + 1].set_title(titles[i] + " Interpolated")
    sns.heatmap(np.flip(cycle_dates[:, :, i] / 30.5, 0),
                vmin=0,
                vmax=12,
                cmap="hsv",
                ax=axes[i * 2],
                cbar_kws=dict(label="Month (6 -> end of June)",
                              ticks=range(13)))
    sns.heatmap(np.flip(cin[:, :, i] / 30.5, 0),
                vmin=0,
                vmax=12,
                cmap="hsv",
                ax=axes[i * 2 + 1],
                cbar_kws=dict(label="Month (6 -> end of June)",
                              ticks=range(13)))

    axes[i * 2].axis("off")
    axes[i * 2 + 1].axis("off")
fig.tight_layout()
fig.savefig(f"phenology_dates_{args.version}{suffix}.png")

# save it as zarr
# Output cell centres are the mean of the input centres each block covers. This is exact
# for any factor and reproduces the shipped 40 km coordinates bit for bit; the previous
# arange construction used `da.x.max() - inresx / 2` where the extent needs `+ inresx / 2`,
# which happened to cancel at factor 4 but yields 4319 instead of 4320 columns at factor 1.
outx = da.x.values[:(len(da.x) // factor) * factor].reshape(-1, factor).mean(axis=1)
outy = da.y.values[:(len(da.y) // factor) * factor].reshape(-1, factor).mean(axis=1)

print("saving...")
daout = xr.DataArray(data=np.concatenate([cycle_dates, cin], axis=2),
                     coords=dict(y=outy,
                                 x=outx,
                                 var=[
                                     "start", "end", "middle", "start_interp",
                                     "end_interp", "middle_interp"
                                 ]),
                     dims=["y", "x", "var"])

daout.rename("phenology40km").to_zarr(
    join(DATAPATH, f"modis_pheno_processed_{args.version}{suffix}.zarr"))
