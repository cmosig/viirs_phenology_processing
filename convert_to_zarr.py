import h5netcdf
from numcodecs import Blosc
import numpy as np
import xarray as xr
from glob import glob
from os.path import join
from os import listdir
from rasterio import crs, transform
from tqdm import tqdm
import dask.array

# ADJUST THIS ACCORDINGLY
ROOT_PATH = "/net/scratch/cmosig/datasets/modispheno/"

# these are all available variables
# data types read from manual
DATA_VARIABLES_DTYPE = {
    "Date_Mid_Greenup_Phase_1": np.uint16,
    "Date_Mid_Senescence_Phase_1": np.uint16,
    "EVI2_Growing_Season_Area_1": np.uint16,
    "EVI2_Onset_Greenness_Increase_1": np.uint16,
    "EVI2_Onset_Greenness_Maximum_1": np.uint16,
    "GLSP_QC_1": np.uint8,
    "Greenness_Agreement_Growing_Season_1": np.uint8,
    "Growing_Season_Length_1": np.uint16,
    "Onset_Greenness_Decrease_1": np.uint16,
    "Onset_Greenness_Increase_1": np.uint16,
    "Onset_Greenness_Maximum_1": np.uint16,
    "Onset_Greenness_Minimum_1": np.uint16,
    "PGQ_Growing_Season_1": np.uint8,
    "PGQ_Onset_Greenness_Decrease_1": np.uint8,
    "PGQ_Onset_Greenness_Increase_1": np.uint8,
    "PGQ_Onset_Greenness_Maximum_1": np.uint8,
    "PGQ_Onset_Greenness_Minimum_1": np.uint8,
    "Rate_Greenness_Decrease_1": np.uint16,
    "Rate_Greenness_Increase_1": np.uint16,
}

DATA_VARIABLES_FILL_VALUE = {
    "Date_Mid_Greenup_Phase_1": 32767,
    "Date_Mid_Senescence_Phase_1": 32767,
    "EVI2_Growing_Season_Area_1": 32767,
    "EVI2_Onset_Greenness_Increase_1": 32767,
    "EVI2_Onset_Greenness_Maximum_1": 32767,
    "GLSP_QC_1": 255,
    "Greenness_Agreement_Growing_Season_1": 255,
    "Growing_Season_Length_1": 32767,
    "Onset_Greenness_Decrease_1": 32767,
    "Onset_Greenness_Increase_1": 32767,
    "Onset_Greenness_Maximum_1": 32767,
    "Onset_Greenness_Minimum_1": 32767,
    "PGQ_Growing_Season_1": 255,
    "PGQ_Onset_Greenness_Decrease_1": 255,
    "PGQ_Onset_Greenness_Increase_1": 255,
    "PGQ_Onset_Greenness_Maximum_1": 255,
    "PGQ_Onset_Greenness_Minimum_1": 255,
    "Rate_Greenness_Decrease_1": 32767,
    "Rate_Greenness_Increase_1": 32767,
}

# ------------------------------------------------------------

PRODUCT_CRS = crs.CRS.from_string("""PROJCS["unnamed",\
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

PATH_TO_FILES = join(ROOT_PATH, "e4ftl01.cr.usgs.gov/VIIRS/VNP22Q2.001")

# should be 2013 - 2022
# and in the format "YYYY.MM.DD"
year_folder_names = sorted(listdir(PATH_TO_FILES))

# NOTE FOR TESTING
year_folder_names = year_folder_names[:1]

# and for each year we have two data cycles, meaning 2 timesteps per year
num_timesteps = len(year_folder_names) * 2

# as the grid size is 0-35 (=#36) in x and 1-15 (=#14) in y and each tile is 2400x2400 pixels
# this means the total shape size is 2400*36 x 2400*18=86400 x 43200
# NOTE there are more tiles than this v00 and v016 - v17, but these have never any data
# shape (y, x)
chunk_size = 2400
x_size = 86400
y_size = 33600

# this is outside all valid values
# (can't use nan as it's a float and we have integers)

# as derived from the metadata
total_xmin = -20015109.354
total_xmax = 20015109.354
total_ymin = -6671703.118
total_ymax = 8895604.157333

# derive pixel size
pixel_size_x = (total_xmax - total_xmin) / x_size
pixel_size_y = (total_ymax - total_ymin) / y_size

ds = xr.Dataset(
    data_vars=dict([(
        name,
        xr.DataArray(data=dask.array.empty(
            (len(year_folder_names) * 2, y_size, x_size),
            dtype=DATA_VARIABLES_DTYPE[name]),
                     dims=("time", "y", "x")),
    ) for name in DATA_VARIABLES_DTYPE.keys()]),
    coords=dict(
        time=[
            f"{n.split('.')[0]}_cycle_{i}" for n in year_folder_names
            for i in (1, 2)
        ],
        # need to shift because xarray uses center of pixel
        y=np.linspace(total_ymin, total_ymax, y_size, endpoint=False) +
        pixel_size_y / 2,
        x=np.linspace(total_xmin, total_xmax, x_size, endpoint=False) +
        pixel_size_x / 2,
    ),
).chunk({
    "time": 1,
    "y": chunk_size,
    "x": chunk_size
})

# write to zarr as template
ds.to_zarr(
    join(ROOT_PATH, "/scratch/cmosig/modispheno.zarr"),
    mode="w",
    write_empty_chunks=False,
    encoding=dict([(name, {
        "write_empty_chunks": False,
        "compressor": Blosc(cname="lz4"),
        "_FillValue": DATA_VARIABLES_FILL_VALUE[name],
    }) for name in DATA_VARIABLES_DTYPE.keys()]),
    compute=False,
)

# ------------------------------------------------------------


# iterate through variables, data cycles and years
def convert_variable(variable_name):
    # setup xr data array for the respective variable

    pbar_years = tqdm(year_folder_names, leave=False, dynamic_ncols=True)
    time_index = 0

    for year_folder in year_folder_names:
        pbar_cycle = tqdm((1, 2),
                          desc=f"Processing {year_folder}",
                          leave=False,
                          dynamic_ncols=True)
        pbar_years.set_description(f"Processing {year_folder}")

        for cycle in (1, 2):

            data_array = xr.DataArray(
                data=np.full(
                    (1, y_size, x_size),
                    fill_value=DATA_VARIABLES_FILL_VALUE[variable_name],
                    dtype=DATA_VARIABLES_DTYPE[variable_name]),
                dims=("time", "y", "x"),
                coords=dict(
                    time=[f"{year_folder.split('.')[0]}_cycle_{cycle}"],
                    y=ds.y,
                    x=ds.x,
                ),
            ).chunk({
                "time": 1,
                "y": chunk_size,
                "x": chunk_size
            }, )

            files = list(glob(join(PATH_TO_FILES, year_folder, "*.h5")))
            pbar_files = tqdm(files,
                              desc=f"Processing cycle {cycle}",
                              leave=False,
                              dynamic_ncols=True)
            pbar_cycle.set_description(f"Processing cycle {cycle}")

            for file in files:
                pbar_files.set_description(f"Processing {file.split('/')[-1]}")
                # read variable and write to respective area in data array

                f = h5netcdf.File(file, phony_dims='access')

                if variable_name not in f[
                        f"/HDFEOS/GRIDS/Cycle {cycle}/Data Fields"]:
                    pbar_files.update()
                    continue

                data = f[
                    f"/HDFEOS/GRIDS/Cycle {cycle}/Data Fields/{variable_name}"][:]

                # read chunk transform from metadata
                # black magic
                metadata = f["/HDFEOS INFORMATION/StructMetadata.0"].__array__(
                ).item().decode().split()
                minx, maxy = eval(metadata[8].split("=")[1])
                maxx, miny = eval(metadata[9].split("=")[1])

                data_array.loc[dict(
                    time=f"{year_folder.split('.')[0]}_cycle_{cycle}",
                    y=slice(miny, maxy),
                    x=slice(minx, maxx))] = np.flip(data, axis=0)

                pbar_files.update()

            # write to zarr
            data_array.rename(variable_name).to_zarr(
                join(ROOT_PATH, "/scratch/cmosig/modispheno.zarr"),
                mode="a",
                region=dict(time=slice(time_index, time_index + 1),
                            x=slice(None),
                            y=slice(None)),
            )
            time_index += 1

            pbar_cycle.update()
        pbar_years.update()


for variable in tqdm(DATA_VARIABLES_DTYPE,
                     desc="Variables",
                     dynamic_ncols=True):
    convert_variable(variable)
