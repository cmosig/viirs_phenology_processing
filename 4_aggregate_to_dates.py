import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import seaborn as sns
from paths import DATAPATH
from os.path import jon

da = xr.open_zarr(join(DATAPATH, "modispheno_aggregated.zarr")).phenology
print("loading data...")
x = da.load().to_numpy()
x.shape

print("resampling...")
# downsample by factor 4 --> from 10km to 40km
factor = 4
x_down = x.reshape(x.shape[0] // factor, factor, x.shape[1] // factor, factor,
                   x.shape[2])
x_down = x_down.sum(axis=(1, 3))

# extract start, end, and middle from count curves
thresh = np.max(x_down, axis=2, keepdims=True) * 0.5
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
cycle_dates[np.isnan(x_down).all(axis=2)] = np.nan

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
fig.savefig("phenology_dates.png")

# save it as zarr
inresx = (da.x[1] - da.x[0]).item()
inresy = (da.y[1] - da.y[0]).item()

inminx = (da.x.min() - inresx / 2).item()
inminy = (da.y.min() - inresy / 2).item()
inmaxx = (da.x.max() - inresx / 2).item()
inmaxy = (da.y.max() - inresy / 2).item()

outresx = inresx * factor
outresy = inresy * factor

print("saving...")
daout = xr.DataArray(data=np.concatenate([cycle_dates, cin], axis=2),
                     coords=dict(y=np.arange(inminy + outresy / 2,
                                             inmaxy + outresy / 2, outresy),
                                 x=np.arange(inminx + outresx / 2,
                                             inmaxx + outresx / 2, outresx),
                                 var=[
                                     "start", "end", "middle", "start_interp",
                                     "end_interp", "middle_interp"
                                 ]),
                     dims=["y", "x", "var"])

daout.rename("phenology40km").to_zarr(
    join(DATAPATH, "modis_pheno_processed.zarr"))
