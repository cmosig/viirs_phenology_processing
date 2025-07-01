import xarray as xr
import numpy as np
from rasterio import crs, warp

path = "/net/projects/tree_mortality/sentinel_project_new/modis_phenology/modispheno_aggregated_normalized.zarr"

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


def get_phenology_curve(lat: float, lon: float) -> np.ndarray | None:
    """
    Get the phenology curve for a given latitude and longitude. Returns None if
    nothing is found. Otherwise returns the phenology curve as a numpy array,
    where the values are between 0 and 256.
    """
    # transform lat lon into modis crs x y
    x, y = warp.transform(crs.CRS.from_epsg(4326), modiscrs, [lon], [lat])

    # open the dataset
    ds = xr.open_zarr(path)

    # get the nearest pixel
    ds = ds.sel(x=x, y=y, method="nearest")

    pheno = ds.phenology.values[0][0]
    is_nan = ds.nan_mask.values[0][0]

    if is_nan:
        return None
    else:
        return pheno


# Example usage for black forest nationalpark
print(get_phenology_curve(48.000, 8.000))
