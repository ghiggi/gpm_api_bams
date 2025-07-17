import time
import os 
import gpm
import numpy as np
import xarray as xr
import dask
import logging

# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})

from dask.distributed import Client, LocalCluster


if __name__ == "__main__":  # https://github.com/dask/distributed/issues/2520
    ##-----------------------------------------------------------------------.
    #### Create monthly datasets 
    # - Set before starting in the terminal : ulimit -n 999999 
    
    ####--------------------------------------------------------.
    #### Define path and country   
    zarr_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Global"
    daily_store = os.path.join(zarr_dir, "DailyStore.zarr")
    monthly_store = os.path.join(zarr_dir, "MonthlyStore.zarr")
    
    os.makedirs(zarr_dir, exist_ok=True)
    
    ##-----------------------------------------------------------------------.
    #### Define Dask Cluster        
    # Create dask.distributed local cluster
    cluster = LocalCluster(
        # n_workers=12,
        processes=True,
        n_workers=21,
        # threads_per_worker=21,
        # processes=True,  
        memory_limit="700GB",
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
    ds = xr.open_zarr(daily_store, chunks={}, consolidated=True) 

    ####--------------------------------------------------------.
    #### Select variable
    # Global statistics 
    # 153 MB per 50X50 block
    # 400 GB per variable 
    # 1.2 TB total dataset
    print("Input Store time chunk:", ds.isel(lon=slice(0,50), lat=slice(0,50)).nbytes/(1024**2), "MB") # Each chunk 7 GB 
    print("Input Store size:", ds.nbytes/(1024**3), "GB")  # ~20TB
    
    ####--------------------------------------------------------.
    # Group by day and compute climatology
    ds_monthly = ds.resample(time="1ME").sum(method="blockwise")
    
    # Initialize consolidated dataset 
    ds_template = ds_monthly.chunk({"time": -1})
    ds_template = xr.zeros_like(ds_template)
    ds_template.to_zarr(monthly_store, 
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
    
    list_lon_slices = create_slices_from_chunks(ds_monthly.chunksizes["lon"])
    list_lat_slices = create_slices_from_chunks(ds_monthly.chunksizes["lat"])
    
    # Resample over each block 
    lat_slices = list_lat_slices[0]
    lon_slices = list_lon_slices[0]
    print("Destination Store time chunk:", ds_monthly.isel(lon=lon_slices, lat=lat_slices).nbytes/(1024**2), "MB") # 20 MB
    print("Destination Store size:", ds_monthly.nbytes/(1024**3), "GB")   # 53 MB
    
    n_blocks = len(list_lon_slices)*len(list_lat_slices)
    i = 0
    for lat_slices in list_lat_slices: 
        for lon_slices in list_lon_slices:
            i = i + 1
            print(f"{i}/{n_blocks}")
            t_block = time.time() 
            region = {"lon": lon_slices, "lat": lat_slices}
            ds_block = ds_monthly.isel(**region)
            ds_block = ds_block.chunk({"time": -1})
            region.update({"time": slice(0, ds_monthly["time"].size)})
            ds_block.to_zarr(monthly_store, 
                             region = region)
            print(time.time() - t_block)

    t_f = time.time() 
    print(t_f - t_i)
 
    # --------------------------------------------------------------------------.
