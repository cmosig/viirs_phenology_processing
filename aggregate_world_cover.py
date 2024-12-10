from rasterio import crs, transform, windows, warp
import numpy as np
import rasterio
from tqdm import tqdm
from parallel import paral
from scipy import ndimage

path_to_worldcover = "/net/scratch/cmosig/datasets/worldcover_modis_crs.vrt"
out_path = "/net/scratch/cmosig/datasets/worldcover_aggregated.tif"

# this is relative to the modis grid
reprojection_factor = 200

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
            return None

        forest_mask = (dr.read(1, window=window, boundless=True) == 10)

        if not forest_mask.any():
            return None

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

        return return_array


worldcover_transform = rasterio.open(path_to_worldcover).transform

with rasterio.open(out_path,
                   'w',
                   driver='GTiff',
                   height=modis_shape[0],
                   width=modis_shape[1],
                   count=1,
                   dtype=rasterio.float32,
                   crs=modis_crs,
                   transform=target_transform,
                   compress="DEFLATE",
                   tiled="YES") as dst:

    pbar = tqdm(total=modis_shape[0] // reprojection_factor, desc="yi index")
    for yi in range(0, modis_shape[0] // reprojection_factor):

        indices = []
        for xi in range(0, modis_shape[1] // reprojection_factor):
            indices.append((yi, xi))

        print("computing results...")
        fractions_result = paral(compute_forest_fraction,
                                 iters=[indices],
                                 num_cores=16)

        print("computed results. now writing")
        for xi in range(0, modis_shape[1] // reprojection_factor):

            win = windows.Window(
                xi * reprojection_factor,
                modis_shape[0] - ((yi + 1) * reprojection_factor),
                reprojection_factor, reprojection_factor)

            # agg_grid[modis_shape[0] -
            #          ((yi + 1) * reprojection_factor):modis_shape[0] -
            #          (yi * reprojection_factor),
            #          xi * reprojection_factor:(xi + 1) *
            #          reprojection_factor] = fractions_result[i]

            if fractions_result[xi] is not None:
                dst.write(fractions_result[xi], window=win, indexes=1)
            else:
                dst.write(np.zeros((reprojection_factor, reprojection_factor),
                                   dtype=np.float32),
                          window=win,
                          indexes=1)

        print("completed writing")
        pbar.update(1)
