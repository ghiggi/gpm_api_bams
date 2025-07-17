import os
import radar_api
import pandas as pd
import gpm
import gpm.gv # xradar_dev accessor
from gpm.gv import ( 
    volume_matching, 
    retrieve_ds_sr,
)
from gpm.utils.geospatial import merge_extents
from radar_api.utils.xradar import get_nexrad_datatree_from_pyart


#### Define directory paths
# Define GPM GV directory
gv_base_dir = "/home/ghiggi/data/GPM_GV"
gv_base_dir = "/t5500/ltenas8/data/GPM_GV"

# List available radar networks 
# networks = radar_api.available_networks()

# List available radar networks 
# radars = radar_api.available_radars(network=network)

# Define network and radar
network = "NEXRAD"
radar = "KLIX"
storage="GES_DISC"

#### Define SR/GR volume matching settings
radar_band = "S"
beamwidth_gr = 1 
z_min_threshold_gr = 0  
z_min_threshold_sr = 10

# Define GPM-GV Overpass directory
gpm_gv_overpass_dir = os.path.join(gv_base_dir, "Overpass", network)

# Define GPM-GV Overpass Table for the specific radar
filepath = os.path.join(gpm_gv_overpass_dir, f"{radar}.parquet")

# Check availability of GPM-GV Overpass Table for the specific radar
if not os.path.exists(filepath):
    raise ValueError(f"GPM-GV Overpass Table for {network} radar {radar} is unavailable.")

# Define GPM-GV directories
gv_volume_dir = os.path.join(gv_base_dir, "Volumes", network, radar)
os.makedirs(gv_volume_dir, exist_ok=True)

gv_quicklook_dir = os.path.join(gv_base_dir, "Quicklooks", network, radar)
os.makedirs(gv_quicklook_dir, exist_ok=True)

# Open  GPM-GV Overpass Table
df_overpass_table = pd.read_parquet(filepath)
df_overpass_table  = df_overpass_table.sort_values(by="n_footprints", ascending=False)

# Perform SR-GR matchup over each GPM overpass
i = 0
n_overpasses = len(df_overpass_table)
for i in range(n_overpasses):
    print(f"{i+1}/{n_overpasses}")
    ####-----------------------------------------------------------------.
    #### Retrieve GPM overpass time
    start_time = df_overpass_table.iloc[i]["start_time"]
    end_time = df_overpass_table.iloc[i]["end_time"]  
    
    start_time_sr = start_time - pd.Timedelta(30, unit="seconds")
    end_time_sr = end_time + pd.Timedelta(30, unit="seconds")

    ####-----------------------------------------------------------------.    
    #### Download GPM required data
    product_type = "RS"
    storage = "GES_DISC"
    version = 7
    for product in ["1B-Ku", "2A-DPR"]:
        gpm.download(
            product=product,
            product_type=product_type,
            version=version,
            start_time=start_time_sr,
            end_time=end_time_sr,
            storage=storage,
            force_download=False,
            verbose=False,
            progress_bar=False,
            check_integrity=True,
        )

    ####-----------------------------------------------------------------.    
    #### Download GR required data
    # Download files
    start_search_window = start_time - pd.Timedelta(5, unit="minutes")
    end_search_window = end_time + pd.Timedelta(5, unit="minutes")
    filepaths = radar_api.download_files(
        network=network,
        radar=radar,
        start_time=start_search_window,
        end_time=end_search_window,
        verbose=False,
        progress_bar=False,
        force_download=False,
    )
    filepaths = [filepath for filepath in filepaths if not filepath.endswith("_MDM")]
    
    # Check there are available files
    if len(filepaths) == 0:
        print(f"Unavailable data for {network} radar {radar} between {start_search_window} and {end_search_window}.")
        continue
    
    ####-----------------------------------------------------------------.    
    #### Open GR required data
    # - Select sweeps with acquisition time within 5 minutes from GPM overpass
    dict_dt = {}
    for filepath in filepaths: 

        # Open file
        # - TODO: in future use directly xradar open_datatree
        try:
            radar_obj = radar_api.open_pyart(filepath, network=network)
            dt_gr = get_nexrad_datatree_from_pyart(radar_obj)
        except Exception as e:
            print(f"Error while opening {filepath}: {str(e)}")
            continue
        
        # Select only sweeps with acquisition time difference less than 5 minutes 
        invalid_sweeps = []
        for sweep_group in dt_gr.xradar_dev.sweeps:
            sweep_start_time = dt_gr[sweep_group]["time"].data[0]
            sweep_end_time = dt_gr[sweep_group]["time"].data[-1] 
            # If the sweep has been acquired 5 minutes apart from SR overpass, do not process the sweep
            if (sweep_end_time < start_search_window) or (sweep_start_time > end_search_window):
                invalid_sweeps.append(sweep_group)  
            
        # Subset datatree 
        dt_gr = dt_gr.drop_nodes(invalid_sweeps)
        if len(dt_gr.xradar_dev.sweeps) > 0: 
            dict_dt[os.path.basename(filepath)] = dt_gr
    
    # Check if some sweeps are available 
    if len(dict_dt) == 0: 
        print("No {network} radar {radar} sweeps acquired between between {start_search_window} and {end_search_window}.")
        continue 
    
 

    ####-----------------------------------------------------------------.
    #### Load GPM data 
    # Retrieve outer GR extent (in WGS84)
    outer_extent_gr = merge_extents([dt.xradar_dev.extent() for dt in dict_dt.values()])
    
    # Retrieve overpassing SR scans
    ds_sr = retrieve_ds_sr(start_time=start_time_sr, 
                           end_time=end_time_sr, 
                           extent_gr=outer_extent_gr,
                           download_sr=False)
    
    # Define quicklook overpass directory
    time_str = pd.Timestamp(ds_sr["time"][0].data.item()).strftime("%Y-%m-%dT%H%M")
    gv_quicklook_overpass_dir = os.path.join(gv_quicklook_dir, time_str)
    os.makedirs(gv_quicklook_overpass_dir, exist_ok=True)
    
    ####-----------------------------------------------------------------.
    #### Run SR/GR volume matching over each sweep      
    # Perform volume matching for each GR sweep 
    # - This typically takes 5-20 seconds per sweep to complete    
    list_gdf = []
    for filename, dt_gr in dict_dt.items():
        for sweep_group in dt_gr.xradar_dev.sweeps:
            try:
                # Retriete sweep
                ds_gr = dt_gr[sweep_group].to_dataset()
                
                # Define quicklook filepath 
                time_str = pd.Timestamp(ds_gr["time"][0].data.item()).strftime("%Y%m%d%H%M%S")
                filename = f"{network}_{radar}_{time_str}_{sweep_group}.png"
                quicklook_fpath = os.path.join(gv_quicklook_overpass_dir, filename)
                
                # Create matched volume database
                gdf_match = volume_matching(
                    ds_gr = ds_gr, 
                    ds_sr = ds_sr,
                    z_variable_gr="DBZH",
                    radar_band=radar_band,
                    beamwidth_gr=beamwidth_gr,
                    z_min_threshold_gr = z_min_threshold_gr,     
                    z_min_threshold_sr = z_min_threshold_sr,   
                    min_gr_range = 0,         
                    max_gr_range = 150_000,   
                    # gr_sensitivity_thresholds=None, 
                    # sr_sensitivity_thresholds=None, 
                    display_quicklook=False,
                    quicklook_fpath=quicklook_fpath,
                    download_sr=False, # require internet connection ! 
                )
                
            except Exception as e: 
                print(f"Volume matching error at {start_time}: {str(e)}")
                continue 
            
            # Append matching database
            if gdf_match is not None:
                gdf_match["filename"] = filename
                gdf_match["sweep_group"] = sweep_group
                list_gdf.append(gdf_match)

    # Concate volume matching results across all sweeps
    if len(list_gdf) == 0: 
        print("No matched volumes.")
        continue
    
    gdf_match = pd.concat(list_gdf)

    ####-----------------------------------------------------------------.
    #### Save GeoParquet to disk
    time_str = start_time.strftime("%Y%m%d%H%M")
    filename = f"GPM_GV_{product}_{network}_{radar}_{time_str}.parquet"
    filepath = os.path.join(gv_volume_dir,filename)
    gdf_match.to_parquet(filepath, geometry_encoding="WKB") # "geoarrow"

####--------------------------------------------------------------------------.
