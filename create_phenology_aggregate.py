import xarray as xr
from numcodecs import Blosc
import numpy as np
from rasterio import windows, crs
import rasterio
import dask.array
from tqdm import tqdm
from multiprocessing import Pool
from os.path import join
import random

# Load the MODIS phenology data
variables_of_interest = ["Onset_Greenness_Maximum", "Onset_Greenness_Decrease"]

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


def process_chunk(inp):
    y, x, forest_mask = inp

    # required variables are:
    # - Onset_Greenness_Maximum
    # - Onset_Greenness_Decrease

    ds = xr.open_zarr("/scratch/cmosig/modispheno.zarr/",
                      chunks=None)[variables_of_interest]

    chunk = ds.isel(y=slice(y, y + aggregation_factor),
                    x=slice(x, x + aggregation_factor))

    # print(y, x, "chunk", chunk.x.min().item(), chunk.y.min().item(), chunk.x.max().item(), chunk.y.max().item())

    if all([chunk[var].isnull().all().item()
            for var in variables_of_interest]):
        # print("opening dataset, but empty chunk")
        return (y, x, None)

    # compute the offset dates as per the documentation
    # and reshape for future broadcasting
    offset_dates = np.array(
        list(
            map(lambda x: -366 * (int(x.split("_")[0]) - 2000),
                chunk.time.values))).reshape((20, 1, 1))

    # get bounds and switch from center coord to actual pixel bounds
    pixel_size_x = chunk.x[1] - chunk.x[0]
    pixel_size_y = chunk.y[1] - chunk.y[0]
    xmin = chunk.x.min() - pixel_size_x / 2
    xmax = chunk.x.max() + pixel_size_x / 2
    ymin = chunk.y.min() - pixel_size_y / 2
    ymax = chunk.y.max() + pixel_size_y / 2

    # mask out pixels with nan where the forest cover is less than 0.5
    chunk = chunk.where(forest_mask, np.nan)

    # shift values as per documentation
    onset_max = chunk.Onset_Greenness_Maximum.values + offset_dates
    onset_dec = chunk.Onset_Greenness_Decrease.values + offset_dates

    # setup vector for intermediate results
    inter = np.zeros(
        (onset_max.shape[0] // 2, onset_max.shape[1], onset_max.shape[2], 366),
        dtype=np.uint16)

    # create mask for nan values
    nan_mask = ~chunk.isnull()
    nan_mask_onset_max = nan_mask.Onset_Greenness_Maximum.values
    nan_mask_onset_dec = nan_mask.Onset_Greenness_Decrease.values

    # Case 1 onsetmax is not nan -> start value = onsetmax
    # Case 1A onsetdec is greater than onsetmax and not nan -> end value = onsetdec
    mask_case_1A = nan_mask_onset_max[::
                                      2, :, :] & nan_mask_onset_dec[::2, :, :] & (
                                          onset_dec[::2, :, :]
                                          > onset_max[::2, :, :])
    for ti, yi, xi in zip(*np.where(mask_case_1A)):
        start = int(onset_max[ti * 2, yi, xi])
        end = int(onset_dec[ti * 2, yi, xi])

        # set for the range
        inter[ti, yi, xi, start:end] = 1

    # Case 1B else -> end value = 366
    mask_case_1B = nan_mask_onset_max[::2, :, :] & (
        ~nan_mask_onset_dec[::2, :, :])
    for ti, yi, xi in zip(*np.where(mask_case_1B)):
        start = int(onset_max[ti * 2, yi, xi])
        end = 366

        # set for the range
        inter[ti, yi, xi, start:end] = 1

    # Case 2 onsetdec is non nan and smaller than onsetmax -> end value = onsetdec, start value = 0
    mask_case_2 = nan_mask_onset_dec[::2] & (
        (onset_dec[::2] < onset_max[::2]) | (~nan_mask_onset_max[::2]))
    for ti, yi, xi in zip(*np.where(mask_case_2)):
        start = 0
        end = int(onset_dec[ti * 2, yi, xi])

        # set for the range
        inter[ti, yi, xi, start:end] = 1

    # for data cycle 2
    # Case 3 onsetmax is not nan -> start value = onsetmax
    # Case 3A onsetdec is greater than onsetmax and not nan -> end value = onsetdec
    mask_case_3A = nan_mask_onset_max[1::2] & (
        onset_dec[1::2] > onset_max[1::2]) & nan_mask_onset_dec[1::2]
    for ti, yi, xi in zip(*np.where(mask_case_3A)):
        start = int(onset_max[ti * 2 + 1, yi, xi])
        end = int(onset_dec[ti * 2 + 1, yi, xi])

        # set for the range
        inter[ti, yi, xi, start:end] = 1

    # Case 3B else -> end value = 365
    mask_case_1B = nan_mask_onset_max[1::2, :, :] & (
        ~nan_mask_onset_dec[1::2, :, :])
    for ti, yi, xi in zip(*np.where(mask_case_1B)):
        start = int(onset_max[ti * 2, yi, xi])
        end = 366

        # set for the range
        inter[ti, yi, xi, start:end] = 1

    # 2 sum vectors across time and space dimensions
    return (y, x, np.sum(inter, axis=(0, 1, 2)))


modis_x_size = 86400
modis_y_size = 33600
aggregation_factor = 20  # --> to 10km
modis_x_size_agg = modis_x_size // aggregation_factor
modis_y_size_agg = modis_y_size // aggregation_factor

total_xmin = -20015109.354
total_xmax = 20015109.354
total_ymin = -6671703.118
total_ymax = 8895604.157333

# derive pixel size
pixel_size_x_agg = (total_xmax - total_xmin) / modis_x_size_agg
pixel_size_y_agg = (total_ymax - total_ymin) / modis_y_size_agg

out_path = "/scratch/cmosig/modispheno_aggregated.zarr"

# create zarr store for the results
ds = xr.Dataset(
    data_vars=dict(phenology=xr.DataArray(data=dask.array.empty(
        (modis_y_size_agg, modis_x_size_agg, 366), dtype=np.float32),
                                          dims=("y", "x", "day"))),
    coords=dict(
        y=np.linspace(total_ymin, total_ymax, modis_y_size_agg,
                      endpoint=False) + pixel_size_y_agg / 2,
        x=np.linspace(total_xmin, total_xmax, modis_x_size_agg,
                      endpoint=False) + pixel_size_x_agg / 2,
        day=np.arange(366)),
)
ds.to_zarr(out_path,
           compute=False,
           encoding=dict(
               phenology={
                   "write_empty_chunks": False,
                   "compressor": Blosc(cname="lz4"),
                   "_FillValue": np.nan,
               }))

pbar = tqdm(total=(modis_x_size // aggregation_factor) *
            (modis_y_size // aggregation_factor))


def index_generator():
    yis = list(range(0, modis_y_size, aggregation_factor))
    # random.shuffle(yis)

    forest_mask = None
    with rasterio.open(
            "/net/scratch/cmosig/datasets/worldcover_aggregated.tif") as dr:

        for y in yis:
            for x in range(0, modis_x_size, aggregation_factor):
                # xmin = total_xmin + x * pixel_size_x_agg
                # xmax = total_xmin + (x + aggregation_factor) * pixel_size_x_agg
                # ymin = total_ymin + (modis_y_size - y) * pixel_size_y_agg
                # ymax = total_ymin + (modis_y_size - y + aggregation_factor) * pixel_size_y_agg
                # win = windows.from_bounds(xmin, ymin, xmax, ymax, dr.transform)

                coloff = x
                rowoff = modis_y_size - y
                win = windows.Window(col_off=coloff,
                                     row_off=rowoff,
                                     width=aggregation_factor,
                                     height=aggregation_factor)

                if (dr.read_masks(1, window=win) == 0).all():
                    pbar.update(1)
                    continue

                forest_mask = dr.read(1, window=win)
                forest_mask = forest_mask > 0.5

                # print(y, x, "rasterio", xmin, ymin, xmax, ymax)
                # print(y, x, "rasterio", coloff, rowoff, coloff + aggregation_factor, rowoff + aggregation_factor)

                # if there is not forest in the chunk, return None
                if not forest_mask.any():
                    # print("empty mask")
                    pbar.update(1)
                    continue

                yield y, x, forest_mask


def callback(ret):
    y, x, result = ret
    if result is not None:
        # write the results to the zarr store
        xr.DataArray(
            data=result.reshape(1, 1, 366),
            dims=("y", "x", "day"),
            coords=dict(day=np.arange(366),
                        y=[ds.y.values[y // aggregation_factor]],
                        x=[ds.x.values[x // aggregation_factor]]),
        ).to_dataset(name="phenology").to_zarr(
            out_path,
            mode="r+",
            region="auto",
        )


pool = Pool(60)
results = pool.imap_unordered(process_chunk, index_generator())
for result in results:
    callback(result)
    pbar.update(1)

# for index in index_generator():
#     callback(process_chunk(index))
