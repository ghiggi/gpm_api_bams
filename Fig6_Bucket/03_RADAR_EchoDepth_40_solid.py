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

bucket_name = "DPR_RainySurface"

#-----------------------------------------------------------------------------.
#### Define extents
extent = [-180, -90, -70, 70]

#### Read and filter data
variable = "EchoDepth40dBZ_solid_phase"
variable = "EchoDepth45dBZ_solid_phase"
query_extent = extend_extent(extent, padding=2)
columns = [variable,
           # Coordinates
           "lon",
           "lat", 
]

#### Read Parquet Dataset
df_pl = gpm.bucket.read(
    bucket_dir=os.path.join(bucket_dir, bucket_name),
    columns=columns,
    use_pyarrow=False,  # use rust parquet reader
    extent=query_extent,
    backend="polars", 
    parallel="auto", #"auto", # "row_groups", "columns"
)

# Select only non null raws and dept > 2000
df_pl = df_pl.filter(~pl.col(variable).is_null() & pl.col(variable).gt(1000))

# Filter by extent 
df_pl = df_pl.filter(
    (pl.col("lon") > extent[0]) & (pl.col("lon") < extent[1]) &
    (pl.col("lat") > extent[2]) & (pl.col("lat") < extent[3])
)

#-----------------------------------------------------------------------------.
#### Compute spatial statstics 

# Define custom spatial partitioning over which to compute statistics
partitioning = LonLatPartitioning(size=2, extent=extent)

# Add geographic partition centroid coordinate
df_pl = partitioning.add_centroids(df_pl,
                                    x="lon", y="lat", 
                                    x_coord="lon_bin",
                                    y_coord="lat_bin")

# Compute statistics 
grouped_df = df_pl.group_by(*partitioning.levels)
 
list_expressions = (
    pl.col(f"{variable}").max().name.prefix("max_"),
    pl.col(f"{variable}").gt(3000).sum().name.suffix("_GT_3KM"),
    pl.col(f"{variable}").gt(4000).sum().name.suffix("_GT_4KM"),
    pl.col(f"{variable}").gt(8000).sum().name.suffix("_GT_8KM"),
)

df = grouped_df.agg(*list_expressions)

# Conversion to xarray
ds = partitioning.to_xarray(df, spatial_coords=("lon_bin","lat_bin"))
ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})
 
# Save statistics
ds.to_netcdf(f"/ltenas8/data/tmp/Radar_{variable}_stats.nc")

#-----------------------------------------------------------------------------.