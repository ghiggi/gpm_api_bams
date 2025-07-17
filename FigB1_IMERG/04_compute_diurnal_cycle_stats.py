import time
import os 
import gpm 
import dask
import flox
import zarr
import numcodecs
import logging
import numpy as np
import pandas as pd
import xarray as xr
from gpm.utils.parallel import get_block_slices
    
zarr.blosc.use_threads = False
numcodecs.blosc.use_threads = False
# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})


from dask.distributed import Client, LocalCluster


def compute_diurnal_cycle_stats(da_block):
    # Define threshold DataArray
    thresholds = [0, 0.1, 0.5, 1, 2, 5, 10] # mm over one hour !
    thresholds = xr.DataArray(thresholds, dims="threshold")
    
    # Mask at various thresholds (0, 0.1, ...) 
    da_block = da_block.where(da_block >= thresholds)
    da_block = da_block.assign_coords({"threshold": thresholds})
        
    # Compute hours with precipitation 
    # - count() exclude the NaN values 
    da_n_hours = xr.ones_like(da_block.sel(threshold=0)).groupby("time.hour").sum(dim="time") 
    da_n_hours_valid = (~np.isnan(da_block.sel(threshold=0))).groupby("time.hour").sum(dim="time") 
    
    # Create grouped DataArray 
    da_grouped = da_block.groupby("time.hour")
    
    # Compute mean and total accumulations 
    da_mean = da_grouped.mean(dim="time") 
    da_accumulation = da_grouped.sum(dim="time")
    
    # Compute occurence
    da_n_hours_precip = da_grouped.count(dim="time")
    
    # Compute min, q25, median, q75, max 
    quantiles = [0, 0.010, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1]
    da_quantiles = da_grouped.quantile(q=quantiles, dim="time")
    da_quantiles = da_quantiles.rename({"quantile": "stat"})
    da_quantiles["stat"] = ["min", "q10", "q25", "median", "q75", "q90", "q95","q99", "max"]
        
    # Define Dataset
    ds_stats = da_quantiles.to_dataset(dim="stat")
    ds_stats["mean"] = da_mean
    ds_stats["accumulation"] = da_accumulation
    ds_stats["n_hours_precip"] = da_n_hours_precip
    ds_stats["n_hours_valid"] = da_n_hours_valid
    ds_stats["n_hours"] = da_n_hours   
     
    # Compute frequency
    ds_stats["frequency"] = ds_stats["n_hours_precip"] / ds_stats["n_hours_valid"] 
   
    # Compute fraction of valid hours
    ds_stats["fraction_valid_hours"] = ds_stats["n_hours_valid"] / ds_stats["n_hours"]  
 
    # Add local solar time hour as coordinate
    ds_stats["lst_hour"] = np.round(ds_stats["hour"] + ds_stats["lon"]/15) % 24
    ds_stats = ds_stats.set_coords("lst_hour")
    return ds_stats 


if __name__ == "__main__":  # https://github.com/dask/distributed/issues/2520
    # - Set before starting in the terminal : ulimit -n 999999 
    
    ####--------------------------------------------------------.
    #### Define path and variable
    # LTESRV8
    base_dir = "/t5500/export-ltesrv8/GPM/GPM_ZARR/IMERG_Annual_Store"
    zarr_dir = "/t5500/export-ltesrv8/GPM/GPM_ZARR/IMERG_Statistics"
    
    # LTESRV1
    base_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Annual_Store"
    zarr_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Statistics"
    
    os.makedirs(zarr_dir, exist_ok=True)
    
    variable = "precipitation"

    ##-----------------------------------------------------------------------.
    #### Define Dask Cluster        
    # Create dask.distributed local cluster
    cluster = LocalCluster(
        # n_workers=12,
        processes=True,
        n_workers=20,
        threads_per_worker=1,
        memory_limit="600GB",        
        # processes=True,  
        # threads_per_worker=1,
        # memory_limit="700GB",
        silence_logs=logging.WARN,
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
                           # inline_array=True, 
                           chunks={},
    )
    
    # Move time to start of accumulation time ! 
    ds["time"] = ds["time"] - np.timedelta64(30, "m")
    
    # Check that no missing timesteps
    assert len(np.unique(np.diff(ds["time"].data))) == 1
    
    # Select variable 
    da = ds[variable]
    
    # Subset time period 
    da = da.sel(time=slice('2001-01-01 00:00:00', '2022-12-31 23:59:59'))
        
    ####--------------------------------------------------------.
    #### Perform computations
    # Conversion from 30-minute averaged mm/hr to file total precipitation (mm)
    with xr.set_options(keep_attrs=True): 
        da_30_mm = da / 2  # --> /60*30
        da_30_mm.attrs["units"] == "mm"
        da_30_mm.attrs.pop("LongName")
    
    # Resample to hourly precipitation 
    da_hourly = da_30_mm.resample(time='1h').sum(method="blockwise")
    
    #### ---------------------------------------------------------------------.
    #### Define diurnal cycle zarr store 
    ds_stats = compute_diurnal_cycle_stats(da_hourly.chunk({"time": -1}))
    
    # Initialize consolidated dataset 
    store = os.path.join(zarr_dir, "DiurnalCycleStore.zarr")
    os.makedirs(store, exist_ok=True)
    ds_template = xr.zeros_like(ds_stats)
  
    ds_template.to_zarr(store, 
                        compute=False, 
                        consolidated=True, 
                        mode='w')
       
    #### ---------------------------------------------------------------------.
    #### Compute diurnal cycle
    # Define the block of chunks to process simultanously
    # - Input block size should be less (maybe half to be sure) the available RAM
    list_slices = get_block_slices(da_hourly, lon=3, lat=3) # 3*3 chunks
    n_blocks = len(list_slices)  
    print("Input Chunk Size:", da.isel(lon=slice(0,50), lat=slice(0,50)).nbytes/(1024**3), "GB") # Each chunk 7 GB 
    print("Input Block Size:", da.isel(**list_slices[0]).nbytes/(1024**3), "GB") 
    print("Input Store Size:", da.nbytes/(1024**4), "TB")  # ~18TB
    print("Output Block Size:", ds_stats.isel(**list_slices[0]).nbytes/(1024**3), "GB")  
    print("Output Store size:", ds_stats.nbytes/(1024**3), "GB")  # 26 GB
    print("Number of blocks:", n_blocks)
   
    # Compute and write over each block of data
    # - Reading one chunk and resample the data to hourly resolution involve a single thread / process 
    # - I put the hourly resolution data in memory (takes around 2 min for 7GB data ...)
    # - I parallelize the diurnal cycle stats computations over each pixel !
    # --> 5 min per block on LTESRV8 with persist() --> 9 days
    # --> 8 min with compute()
    region = list_slices[0]
    for i, region in enumerate(list_slices): 
        print(f"{i}/{n_blocks}")
        t_block = time.time() 
      
        # Compute stats (simple approach overwhelm dask scheduler)
        # ds_block = ds_stats.isel(**region)
        
        # Compute stats (alternative approach to not overwhelm dask scheduler)
        da_hourly_block = da_hourly.isel(**region)
        da_hourly_block = da_hourly_block.persist()
        ds_block = compute_diurnal_cycle_stats(
            da_block=da_hourly_block.chunk({"time": -1, "lon": 1, "lat": 1}))
        ds_block = ds_block.compute() 
        ds_block = ds_block.chunk({"hour": -1, "lat": 50, "lon": 50, "threshold": -1})
        ds_block["lst_hour"] = ds_block["lst_hour"].compute()
                
        # Write to zarr
        region.update({"hour": slice(0, ds_block["hour"].size), 
                       "threshold": slice(0, ds_block["threshold"].size)})
        ds_block.to_zarr(store, region = region)
        print(time.time() - t_block)

    t_f = time.time() 
    print(t_f - t_i)
        
    #### ---------------------------------------------------------------------.



