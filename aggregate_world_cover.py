from rasterio import crs, transform, windows, warp
from os.path import join
import numpy as np
import rasterio
from tqdm import tqdm
from scipy import ndimage
from multiprocessing.pool import Pool
import random
from glob import glob

path_to_worldcover = "/net/scratch/cmosig/datasets/worldcover_modis_crs.vrt"
out_path = "/net/scratch/cmosig/datasets/worldcover_aggregated.tif"

# this is relative to the modis grid
reprojection_factor = 100

# y, x
modis_shape = (33600, 86400)
modis_xmin = -20015109.354
modis_xmax = 20015109.354
modis_ymin = -6671703.118
modis_ymax = 8895604.157333
modis_yres = (modis_ymax - modis_ymin) / modis_shape[0]
modis_xres = (modis_xmax - modis_xmin) / modis_shape[1]

modis_crs = crs.CRS.from_string(
    """PROJCS["unnamed", GEOGCS["Unknown datum based upon the custom spheroid", DATUM["Not specified (based on custom spheroid)", SPHEROID["Custom spheroid",6371007.181,0]], PRIMEM["Greenwich",0], UNIT["degree",0.0174532925199433]], PROJECTION["Sinusoidal"], PARAMETER["longitude_of_center",0], PARAMETER["false_easting",0], PARAMETER["false_northing",0], UNIT["Meter",1]]"""
)

# figure out target transform for worldcover -->
# transform into downsampled modis grid
target_transform = transform.from_bounds(modis_xmin, modis_ymin, modis_xmax,
                                         modis_ymax, modis_shape[1],
                                         modis_shape[0])


def compute_forest_fraction(index):
    yi, xi = index

    ymin = modis_ymin + (yi * modis_yres * reprojection_factor)
    ymax = modis_ymin + ((yi + 1) * modis_yres * reprojection_factor)
    xmin = modis_xmin + (xi * modis_xres * reprojection_factor)
    xmax = modis_xmin + ((xi + 1) * modis_xres * reprojection_factor)

    window = windows.from_bounds(xmin, ymin, xmax, ymax, worldcover_transform)

    with rasterio.open(
            "/net/scratch/cmosig/datasets/worldcover_modis_crs.vrt") as dr:
        if not dr.dataset_mask(window=window).any():
            return (index, None)

        forest_mask = (dr.read(1, window=window, boundless=True) == 10)

        if not forest_mask.any():
            return (index, None)

        assert forest_mask.shape[0] == forest_mask.shape[1]
        widthheight = forest_mask.shape[0]

        target_transform = transform.from_bounds(0, 0, widthheight,
                                                 widthheight,
                                                 reprojection_factor,
                                                 reprojection_factor)

        source_transform = transform.from_bounds(0, 0, widthheight,
                                                 widthheight,
                                                 forest_mask.shape[1],
                                                 forest_mask.shape[0])

        return_array = np.empty((reprojection_factor, reprojection_factor),
                                dtype=np.float32)

        warp.reproject(
            source=forest_mask.astype(np.float32),
            destination=return_array,
            src_crs=modis_crs,  # does not matter
            dst_crs=modis_crs,
            src_transform=source_transform,
            dst_transform=target_transform,
            resampling=warp.Resampling.average,
        )

        return (index, return_array)


worldcover_transform = rasterio.open(path_to_worldcover).transform

temp_agg_dir = "/net/scratch/cmosig/datasets/temp_agg_save"


def get_indices():
    yis = list(range(0, modis_shape[0] // reprojection_factor))
    random.shuffle(yis)
    for yi in yis:
        for xi in range(0, modis_shape[1] // reprojection_factor):
            yield (yi, xi)


def reproject_and_to_numpy():
    pool = Pool(60)
    pbar = tqdm(total=(modis_shape[0] // reprojection_factor) *
                (modis_shape[1] // reprojection_factor),
                desc="index")

    results = pool.imap_unordered(compute_forest_fraction, get_indices())

    for result in results:
        save_tile_numpy(result)


def numpy_to_geotiff():

    # init numpy array for the entire area
    outarr = np.zeros((modis_shape[0], modis_shape[1]), dtype=np.float32)

    files_to_load = glob(join(temp_agg_dir, "*.npy"))

    for file in tqdm(files_to_load, desc="loading"):
        yi, xi = file.split("/")[-1].split(".")[0].split("_")
        yi, xi = int(yi), int(xi)
        outarr[modis_shape[0] - ((yi+1) * reprojection_factor):modis_shape[0] -
               (yi * reprojection_factor),
               xi * reprojection_factor:(xi + 1) *
               reprojection_factor] = np.load(file)

    open_params = dict(
        driver='GTiff',
        height=modis_shape[0],
        width=modis_shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=modis_crs,
        transform=target_transform,
        compress="DEFLATE",
        tiled="YES",
        nodata=0,
    )

    # initialize the file
    dr = rasterio.open(out_path, 'w', **open_params)
    dr.write(outarr, 1)
    dr.close()


def save_tile_numpy(result):

    index, fractions_result = result
    yi, xi = index

    if fractions_result is not None:
        np.save(join(temp_agg_dir, f"{yi}_{xi}.npy"), fractions_result)

    pbar.update(1)


def save_tile(result):
    index, fractions_result = result
    yi, xi = index

    with rasterio.open(out_path, 'r+') as dst:
        win = windows.Window(xi * reprojection_factor,
                             modis_shape[0] - ((yi + 1) * reprojection_factor),
                             reprojection_factor, reprojection_factor)

        if fractions_result is not None:
            dst.write(fractions_result, window=win, indexes=1)
        else:
            dst.write(np.zeros((reprojection_factor, reprojection_factor),
                               dtype=np.float32),
                      window=win,
                      indexes=1)

        pbar.update(1)


numpy_to_geotiff()
