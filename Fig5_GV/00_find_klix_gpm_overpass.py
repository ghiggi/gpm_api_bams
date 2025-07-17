import os 
import polars as pl 
import gpm.bucket 
import radar_api
from radar_api.io import get_radar_location, get_radar_time_coverage
from gpm.bucket.analysis import get_list_overpass_time
import numpy as np
import pandas as pd
        
# Define directory paths
gpm_gv_dir = "/t5500/ltenas8/data/GPM_GV"
bucket_dir = "/ltenas8/data/GPM_Buckets/DPR_RainySurface"
bucket_dir = "/t5500/ltenas8/data/GPM_Buckets/DPR_RainySurface"

### Define GPM-GV overpass search settings
radar_distance = 150_000
minimum_number_rainy_footprints = 100
        
####--------------------------------------------------------------------------.
# List available radar networks 
networks = radar_api.available_networks()

# Define radar network
network = "NEXRAD"
 
# Define radars to process 
# radars = radar_api.available_radars(network=network)
radars = ["KLIX"]

# Find GPM overpass for each specified radar
for radar in radars:
	print(f"Finding GPM overpass for {network} radar {radar}")
	####------------------------------------------------------------------.
	#### Define bucket options
	columns = ["time", "lon", "lat", 
		   "precipRateNearSurface",
	]
	# Try to retrieve location of the radar
	try:
	    radar_location = get_radar_location(network=network, radar=radar)
	except Exception: 
	    print(f"The radar location is unavailable for {network} radar {radar}.")
	    continue

	# Try to recover temporal availabilty of the radar
	try:
	    radar_start_time, radar_end_time = get_radar_time_coverage(network=network, radar=radar)
	except Exception: 
	    print(f"The radar time coverage information is unavailable for {network} radar {radar}.")
	    continue

	#---------------------------------------------------------------------. 
	#### Read Parquet Dataset with polars 
	try:
	    df_pl = gpm.bucket.read(bucket_dir=bucket_dir,
		                    point=radar_location,
		                    distance=radar_distance,
		                    columns=columns,
		                    use_pyarrow=False,  # use rust parquet reader
		                    parallel="auto", # "row_groups", "columns"
		                    )
	except Exception: 
	    print(f"No GPM overpass for {network} radar {radar}.")
	    continue

	# Filter by intensity 
	df_pl = df_pl.filter((pl.col("precipRateNearSurface") > 0.1))

	# Sort by time 
	df_pl = df_pl.sort("time")

	# Convert to pandas
	df = df_pl.to_pandas()

	# Filter by radar temporal availability 
	df = df[np.logical_and(df["time"] >= radar_start_time, df["time"] <= radar_end_time)]

	if len(df) == 0:
	    print(f"No GPM overpass for {network} radar {radar} between {radar_start_time} and {radar_end_time}.")
	    continue 

	####------------------------------------------------------------------.
	#### Identify time periods with GPM overpass 
	list_time_periods = get_list_overpass_time(timesteps=df['time'].to_numpy(), 
		                                   interval=np.array(10, dtype="m8[m]")
		                                   )  
	n_overpass = len(list_time_periods)
	start_time, end_time = list_time_periods[0]
	list_overpass = []
	for i, (start_time, end_time) in enumerate(list_time_periods):
	    df_overpass = df[np.logical_and(df["time"] >= start_time, df["time"] <=end_time)]
	    n_rainy_footprints = len(df_overpass) 
	    max_precip = max(df_overpass["precipRateNearSurface"])
	    if n_rainy_footprints >= minimum_number_rainy_footprints:
    		dict_overpass = {"start_time": start_time, 
    		                 "end_time": end_time, 
    		                 "n_footprints": n_rainy_footprints, 
    		                 "max_surface_precip":  max_precip}
    		list_overpass.append(dict_overpass)

	# Check presence of GPM overpass matching the filtering criteria
	if len(list_overpass) == 0: 
	    print(f"No GPM overpass for {network} radar {radar} matching the filtering criteria.")
	    continue

	# Define overpass summary table
	df_summary = pd.DataFrame(list_overpass)

	# Save the overpass table
	gpm_gv_overpass_dir = os.path.join(gpm_gv_dir, "Overpass", network)
	os.makedirs(gpm_gv_overpass_dir, exist_ok=True)
	filepath = os.path.join(gpm_gv_overpass_dir, f"{radar}.parquet")
	df_summary.to_parquet(filepath)

####------------------------------------------------------------------.

 
