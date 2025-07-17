import time
import os 
import gpm
import numpy as np
import xarray as xr
import dask
import logging
import numcodecs
import zarr
# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
zarr.blosc.use_threads = False
numcodecs.blosc.use_threads = False
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":  # https://github.com/dask/distributed/issues/2520
    ##-----------------------------------------------------------------------.
    #### Create daily, monthly and global aggregated datasets 
    # - Set before starting in the terminal : ulimit -n 999999 
    
    ####--------------------------------------------------------.
    #### Define path and country
    base_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Annual_Store"
    
    variable = "precipitation"

    zarr_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Global"
    os.makedirs(zarr_dir, exist_ok=True)
    
    ##-----------------------------------------------------------------------.
    #### Define Dask Cluster        
    # Create dask.distributed local cluster
    cluster = LocalCluster(
        processes=True,
        n_workers=21,
        threads_per_worker=1,
    )
    
    client = Client(cluster)
    
    # Set MALLOC_TRIM_THRESHOLD_
    # - To avoid "WARNING - Unmanaged memory use is high"
    def set_env(k,v):
        import os
        os.environ[k]=v
    client.run(set_env,'MALLOC_TRIM_THRESHOLD_','0')

    ####--------------------------------------------------------.
 
    #### Open dataset 
    # List files 
    t_i = time.time()  
    fpaths = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]
    ds = xr.open_mfdataset(fpaths, 
                           engine="zarr", 
                           combine='nested', # 'by_coords', 
                           concat_dim="time",  
                           coords="minimal",
                           # compat="override",
                           combine_attrs='override',
                           consolidated = True, 
                           parallel=False,
                           chunks={})
    
    # Move time to start of accumulation time ! 
    ds["time"] = ds["time"] - np.timedelta64(30, "m")
    
    ####--------------------------------------------------------.
    #### Select variable
    # Global statistics 
    # 30-min --> 20 TB --> 1 year: 908 GB 
    # hourly --> 8.6 TB
    # daily --> 360 GB 
    # monthly --> 10 GB 
    # annual --> ...
    
    # 17 TB at 100 MB/S --> read in 49 hours 
    # 17 TB at 1G MB/S --> read in 4.9 hours 

    subset_variables = ["precipitation", "IRprecipitation", "precipitationUncal"]
    ds = ds[subset_variables]

    print("Input Store time chunk:", ds.isel(lon=slice(0,50), lat=slice(0,50)).nbytes/(1024**3), "GB") # Each chunk 7 GB 
    print("Input Store size:", ds.nbytes/(1024**4), "TB")  # ~20TB
    
    ####--------------------------------------------------------.
    # Conversion from 30-minute averaged mm/hr to file total precipitation (mm)
    with xr.set_options(keep_attrs=True): 
        ds_30_mm = ds / 2
    for var in ds_30_mm.data_vars:
        ds_30_mm[var].attrs["units"] == "mm"
        ds_30_mm[var].attrs.pop("LongName")
    
    # Resample to daily 
    ds_daily = ds_30_mm.resample(time='1D').sum(method="blockwise")
    
    # Initialize consolidated dataset 
    daily_store = os.path.join(zarr_dir, "DailyStore.zarr")
    os.makedirs(daily_store, exist_ok=True)
    ds_template = ds_daily.chunk({"time": -1})
    ds_template = xr.zeros_like(ds_template)
    ds_template.to_zarr(daily_store, 
                        compute=False, 
                        consolidated=True, 
                        mode='w')

    # Create list of spatial blocks 
    def create_slices_from_chunks(chunksizes):
        slices = []
        start = 0
        for size in chunksizes:
            stop = start + size
            slices.append(slice(start, stop))
            start = stop
        return slices
    
    list_lon_slices = create_slices_from_chunks(ds_daily.chunksizes["lon"])
    list_lat_slices = create_slices_from_chunks(ds_daily.chunksizes["lat"])
    
    # Resample over each block 
    lat_slices = list_lat_slices[0]
    lon_slices = list_lon_slices[0]
    print("DailyStore time chunk:", ds_daily.isel(lon=lon_slices, lat=lat_slices).nbytes/(1024**3), "GB") # Each chunk 7 GB 
    print("DailyStore size:", ds_daily.nbytes/(1024**3), "GB")  # 360 GB
    
    n_blocks = len(list_lon_slices)*len(list_lat_slices)
    i = 0
    for lat_slices in list_lat_slices: 
        for lon_slices in list_lon_slices:
            i = i + 1
            print(f"{i}/{n_blocks}")
            t_block = time.time() 
            region = {"lon": lon_slices, "lat": lat_slices}
            ds_daily_block = ds_daily.isel(**region)
            ds_daily_block = ds_daily_block.chunk({"time": -1}) # TODO: block by year (to enable load only 1 year at time)
            region.update({"time": slice(0, ds_daily["time"].size)})
            ds_daily_block.to_zarr(daily_store, 
                                   region = region)
            print(time.time() - t_block)

    t_f = time.time() 
    print(t_f - t_i)
    
# -------------------------------------------------------------------------.
     