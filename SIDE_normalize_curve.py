from paths import *
from os.path import join
import xarray as xr

path = join(DATAPATH, "modispheno_aggregated.zarr")

ds = xr.open_zarr(path).load()

# substract min across day dim
ds = ds - ds.min(dim="day")

# divide by max across day dim
ds = ds / ds.max(dim="day")

# multiply by 256 
ds = (ds * 255)

# create nan mask 
nan_mask = ds.phenology.isnull().any(dim="day")
phenology = ds.phenology.fillna(0).astype("uint8")

# save as new variable with dims ("y", "x")
ds = xr.Dataset(
    {
        "phenology": (["y", "x", "day"], phenology.data),
        "nan_mask": (["y", "x"], nan_mask.data)
    },
    coords={
        "y": ds.y.data,
        "x": ds.x.data,
        "day": ds.day.data,
    }
)

# save to zarr
ds.to_zarr(path.replace(".zarr", "_normalized.zarr"), mode="w", consolidated=True)
