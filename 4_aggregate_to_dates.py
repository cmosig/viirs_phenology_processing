import argparse

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
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
# must follow the nansum above: nansum returns 0 for an all-NaN block, never NaN, so the
# no-data test has to come from the pre-aggregation array instead of from x_down.
cycle_dates[all_nan_block] = np.nan

print("interpolating...")
# Interpolate missing values
cin = cycle_dates.copy()
for i in range(3):
    x, y = np.indices(cin[:, :, i].shape)

    cin[:, :, i][np.isnan(cin[:, :, i])] = griddata(
        (x[~np.isnan(cycle_dates[:, :, i])],
         y[~np.isnan(cycle_dates[:, :, i])]),
        cycle_dates[:, :, i][~np.isnan(cycle_dates[:, :, i])],
        (x[np.isnan(cycle_dates[:, :, i])], y[np.isnan(cycle_dates[:, :, i])]),
        method="nearest")

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
