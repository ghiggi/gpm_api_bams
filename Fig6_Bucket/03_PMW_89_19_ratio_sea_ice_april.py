import os 
import gpm
import matplotlib
import polars as pl 
from gpm.bucket import LonLatPartitioning
from gpm.utils.geospatial import extend_extent

# Define directories
fig_dir = "/ltenas8/data/tmp/GPM_Figures"

bucket_dir = "/t5500/ltenas8/data/GPM_Buckets"
bucket_dir = "/ltenas8/data/GPM_Buckets"

os.makedirs(fig_dir, exist_ok=True)
   
# Matplotlib settings
matplotlib.get_backend()
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10

gmi_name = "GMI"

#-------------------------------------------------------------------------.
#### Define extents
extent = [-90, 0, -70, 70]

#### Read and filter data
query_extent = extend_extent(extent, padding=1)
columns = [   
    'Tc_19V',
    'Tc_89V',
    'Tc_89H',
    'Quality_LF',
    # Coordinates
    'lon',
    'lat',
    'time',
]

#### Read Parquet Dataset
df_pl = gpm.bucket.read(
    bucket_dir=os.path.join(bucket_dir, gmi_name),
    columns=columns,
    use_pyarrow=False,  # use rust parquet reader
    extent=query_extent,
    backend="polars", 
    # backend="polars_lazy",
    parallel="auto", #"auto", # "row_groups", "columns"
)

# Filter by month
df_pl = df_pl.filter(pl.col("time").dt.month() == 5)

# Filter by quality
# - 1: possible sunglint  
# - 2: possible RFI 
df_pl = df_pl.filter(pl.col("Quality_LF").is_in([0, 1, 2]))

# Filter by extent 
df_pl = df_pl.filter(
    (pl.col("lon") > extent[0]) & (pl.col("lon") < extent[1]) &
    (pl.col("lat") > extent[2]) & (pl.col("lat") < extent[3])
)

#-----------------------------------------------------------------------------.
#### Compute spatial statstics 
# Compute ratio
df_pl = df_pl.with_columns(
    (pl.col("Tc_89V") / pl.col("Tc_19V")).alias("RV89_19"),    # Li et a., 2022
    (pl.col("Tc_89V") - pl.col("Tc_19V")).alias("DV89_19"), 
    (pl.col("Tc_89V") - pl.col("Tc_89H")).alias("PD89"),
)

# Define custom spatial partitioning over which to compute statistics
partitioning = LonLatPartitioning(size=0.1, extent=extent)

# Add geographic partition centroid coordinate 
df_pl = partitioning.add_centroids(df_pl,
                                    x="lon", y="lat", 
                                    x_coord="lon_bin",
                                    y_coord="lat_bin")

# Compute statistics 
grouped_df = df_pl.group_by(*partitioning.levels)
list_expressions = [
    pl.col("RV89_19").median().alias("RV89_19_median"),
    pl.col("DV89_19").median().alias("DV89_19_median"),
    pl.col("PD89").median().alias("PD89_median"),
    ]
df = grouped_df.agg(*list_expressions)

# Conversion to xarray
ds_sea_ice_ratio = partitioning.to_xarray(df, spatial_coords=("lon_bin","lat_bin"))
ds_sea_ice_ratio = ds_sea_ice_ratio.rename({"lon_bin": "longitude", "lat_bin": "latitude"})
 
# Save statistics
ds_sea_ice_ratio.to_netcdf("/ltenas8/data/tmp/PMW_sea_ice_april.nc")

#-----------------------------------------------------------------------------.
