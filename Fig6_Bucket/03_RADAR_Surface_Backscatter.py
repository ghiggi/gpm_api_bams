import os 
import gpm
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.utils.geospatial import extend_extent

# Define directories
fig_dir = "/ltenas8/data/tmp/GPM_Figures"
bucket_dir = "/t5500/ltenas8/data/GPM_Buckets"
bucket_dir = "/ltenas8/data/GPM_Buckets"

os.makedirs(fig_dir, exist_ok=True)
   
bucket_name = "DPR_DrySurface"

#-----------------------------------------------------------------------------.
#### Define extents
extent = [0, 90, -70, 70]

#### Read and filter data
query_extent = extend_extent(extent, padding=1)
variable = "sigmaZeroMeasured_Ku"
variable = "sigmaZeroCorrected_Ku"

columns = [
    variable,
    # Coordinates
    "lon",
    "lat", 
    'gpm_cross_track_id',
]

#### Read Parquet Dataset
df_pl = gpm.bucket.read(
    bucket_dir=os.path.join(bucket_dir, bucket_name),
    columns=columns,
    use_pyarrow=False,  # use rust parquet reader
    extent=query_extent,
    use_statistics=False,
    low_memory=False,
    backend="polars", 
    parallel="auto",# "row_groups", "columns"
)

# Select data at nadir 
df_pl = df_pl.filter(pl.col("gpm_cross_track_id") == 23)

# Filter by extent 
df_pl = df_pl.filter(
    (pl.col("lon") > extent[0]) & (pl.col("lon") < extent[1]) &
    (pl.col("lat") > extent[2]) & (pl.col("lat") < extent[3])
)

#-----------------------------------------------------------------------------.
#### Compute spatial statstics 
# Define custom spatial partitioning over which to compute statistics
partitioning = LonLatPartitioning(size=0.1, extent=extent)

# Add geographic partition centroid coordinate
df_pl = partitioning.add_centroids(df_pl,
                                    x="lon", y="lat", 
                                    x_coord="lon_bin",
                                    y_coord="lat_bin")

# Compute statistics 
grouped_df = df_pl.group_by(*partitioning.levels)
 
list_expressions = (
    pl.col("sigmaZeroCorrected_Ku").median().alias("median_NCRS"),
)

df = grouped_df.agg(*list_expressions)

# Conversion to xarray
ds = partitioning.to_xarray(df, spatial_coords=("lon_bin","lat_bin"))
ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})
 
# Save statistics
ds.to_netcdf("/ltenas8/data/tmp/Radar_NRCS_surface_stats.nc")

#-----------------------------------------------------------------------------.
