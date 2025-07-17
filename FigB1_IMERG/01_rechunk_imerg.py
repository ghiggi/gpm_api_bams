import dask
import os
import logging
import gpm
from gpm.io.local import get_local_filepaths
from gpm.utils.dask import clean_memory
from dask.distributed import Client, LocalCluster
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})


def rechunk_dataset(target_store, filepaths, year):
    import time
    import datetime
    import numcodecs
    import zarr
    import xarray as xr
    from gpm.utils.time import regularize_dataset
    from rechunker.api import parse_target_chunks_from_dim_chunks
    zarr.blosc.use_threads = False
    numcodecs.blosc.use_threads = False
    
    #--------------------------------------------------------------------------. 
    # Annual store: 1.8 TB in memory  
    # Rechunk by year (Blocks of 2TB of data)
    # - Without decoding, every int16 DataArray size 227 GB 
    # - The entire Dataset (all variables together) size 2 TB         
    # - decode_cf=True is what convert to float ! 
    #   --> It increase file size to read in memory
    # - BUG with mask_and_scale=False or decode_cf=False 
    #   --> https://github.com/pydata/xarray/issues/9053
    
    #--------------------------------------------------------------------------. 
    # Define target_chunks
    size = 50
    target_chunks = {'time': -1, 'lat': size, 'lon': size}
    
    # Define variables to rechunk
    variables = [
        "precipitationUncal",
        "precipitation",
        "MWprecipitation",
        "IRprecipitation",
        "precipitationQualityIndex",
    ]
        
    # Define open arguments
    # - decode_*: False to speed up !
    open_kwargs = {"decode_cf": False,   
                   "mask_and_scale": False,
                   "decode_coords": False,
                   "concat_characters": False,
                   # Fixed
                   "chunks": -1, # read as source chunks
                   "consolidated": True,
                   "zarr_version": 2,
                   } 
    
    # Open files in parallel 
    print(f"Open files lazily for year {year}")
    t_i_1 = time.time() 
    ds = xr.open_mfdataset(filepaths, engine="zarr",
                           combine='nested', # 'by_coords', 
                           concat_dim="time",  
                           coords="minimal",
                           # compat="override",
                           combine_attrs='override',
                           parallel=True,
                           **open_kwargs
                           )  
    # Subset variables 
    ds = ds[variables]
    
    # Decode coordinates
    ds = xr.decode_cf(ds,
                     decode_times=True,
                     decode_coords=True,
                     use_cftime=False, 
                     # False the rest !
                     concat_characters=False, 
                     mask_and_scale=False, 
                     drop_variables=None,
                     decode_timedelta=False)
    
    # Regularize dataset 
    ds = regularize_dataset(ds=ds, freq="30min")   
    
    # Sanitize dataset         
    # ds = ds.drop_vars("crsWGS84")
    ds["lon"].attrs.pop("_FillValue", None) 
    ds["lat"].attrs.pop("_FillValue", None) 
    ds["time"].attrs.pop("_FillValue", None) 
    _ = ds.attrs.pop("FileName", None)
    
    # Report time required to open the dataset 
    t_f_1 = time.time() 
    timedelta_str = str(datetime.timedelta(seconds=t_f_1 - t_i_1))
    print(f"Elapsed time for opening the dataset: {timedelta_str}")
    
    # Now loop over each variable 
    # var = "MWprecipitation"
    # var = variables[1]
    for i, var in enumerate(variables): 
        #---------------------------------------------------------------.
        #### Load variable
        print(f"- Loading variable {var}")
        t_i_var = time.time()
        
        # Select variable to rechunk
        tmp_ds = ds[[var]]
        
        # Read data into memory
        tmp_ds = tmp_ds.compute()
        
        # Print elapsed time for loading
        t_f_var = time.time() 
        timedelta_str = str(datetime.timedelta(seconds=t_f_var - t_i_var))
        print(f" - Elapsed time for loading {var}: {timedelta_str} .", end="\n")
        
        #---------------------------------------------------------------.
        #### Write variable
        print(f" - Writing variable {var}")       
        t_i_writing = time.time()          
        # Remove chunks information in encodings
        for k in [var, "time"]:
            _ = tmp_ds[k].encoding.pop("chunks", None)
            _ = tmp_ds[k].encoding.pop("preferred_chunks", None)
            
        # Set new chunks encoding 
        tmp_ds[var].encoding["chunks"] = parse_target_chunks_from_dim_chunks(tmp_ds, target_chunks)[var]
             
        # Define write mode 
        if i == 0: 
            mode = "w"
        else:
            mode =  "a"
            
        # Rechunk the DataArray 
        tmp_ds[var] = tmp_ds[var].chunk(target_chunks)
        
        # Write to store 
        # - If using multiprocessing, we get 'ValueError: bytes object is too large'
        with dask.config.set(scheduler='threads'):
            tmp_ds.to_zarr(store=target_store, mode=mode)
               
        # Print elapsed time    
        t_f_var = time.time() 
     
        timedelta_str = str(datetime.timedelta(seconds=t_f_var - t_i_writing))
        print(f" - Elapsed time for writing {var}: {timedelta_str} .", end="\n")
        timedelta_str = str(datetime.timedelta(seconds=t_f_var - t_i_var))
        print(f" - Elapsed time for rechunking {var}: {timedelta_str} .", end="\n")
    
    # Print total elapsed time  
    timedelta_str = str(datetime.timedelta(seconds=t_f_var - t_i_1))
    print(f"--> Rechunking of year {year} terminated. Elapsed time: {timedelta_str} .", end="\n")         
    
    return None


if __name__ == "__main__":  # https://github.com/dask/distributed/issues/2520
    ##-----------------------------------------------------------------------.
    #### Rechunk the dataset by year 
    # - Set before starting : ulimit -n 999999 
    # NOTE:
    # - ZipStore does not support parallel write but does support parallel read !
    #   Multithreading reads from ZipStore slow downs quite a lot because of GIL
    #   Multiprocessing reads alleviate the problem 
    # - Multiprocessing is required to avoid excessive slow down caused by GIL in 
    #   opening the dataset, load data in memory and concatenate (ZIP + array decompression) !
    # - Most numcodecs don't release the GIL, except (maybe) blosc compressors
    # - Once data are in memory, use multithreading for writing to ZarrStore !
    # - To compress a Zarr Directory Store, use: zip -r -0 (no compression!)
    # - Blosc compressor allows chunks of max 2.1 GBs
    # - Philosophy: "chunk as early as possible, and avoid rechunking as much as possible"
 
    ####-----------------------------------------------------------------------.
    # Define configurations for rechunking the Zarr Store              
    # src_zarr_store = "/ltenas8/data/GPM_ZARR/GPM"
    # target_base_dir = "/ltenas8/data/GPM_ZARR/IMERG_Annual_Store"
        
    src_zarr_store = "/ltesrv8/GPM"
    target_base_dir = "/ltesrv8/GPM/GPM_ZARR/IMERG_Annual_Store"
    
    # Define start and end year
    start_year = 2002
    end_year = 2022
    
    ##-----------------------------------------------------------------------.
    #### Define Dask Cluster
    # Set number of workers
    max_mem = "900GB" # "250GB" # , "90GB"
    num_workers = 23 # 3 # 10
    
    # available_workers = # os.cpu_count() - 1
    # num_workers = dask.config.get("num_workers", available_workers)
        
    # Create dask.distributed local cluster
    # --> Use multiprocessing to avoid netCDF multithreading locks ! 
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,  
        processes=True,  
        memory_limit="960GB",
        silence_logs=logging.WARN,
    )
    
    client = Client(cluster)
    
    # Set MALLOC_TRIM_THRESHOLD_
    # - To avoid "WARNING - Unmanaged memory use is high"
    def set_env(k,v):
        import os
        os.environ[k]=v
    client.run(set_env,'MALLOC_TRIM_THRESHOLD_','0')
   
    ##----------------------------------------------------------------------. 
    # Retrieve available filepaths 
    print("Retrieve available filepaths")
    with gpm.config.set({"base_dir": src_zarr_store}):
        filepaths_dict = get_local_filepaths(product="IMERG-FR",
                                             version=7,
                                             product_type="RS",
                                             groups="year")
        
  
    ##----------------------------------------------------------------------. 
    # Loop over the years 
    for year in range(start_year, end_year+1):
        # Retrieve filepaths
        filepaths = filepaths_dict[str(year)]
        
        # Define path of annual zarr store
        target_store = os.path.join(target_base_dir, f"{year}.zarr")
        os.makedirs(target_store, exist_ok=True)
        
        # Rechunk dataset
        rechunk_dataset(target_store=target_store,
                        filepaths=filepaths,
                        year=year)
        
        # Restart client
        clean_memory(client)
        client.restart()
   

    ####----------------------------------------------------------------------.
  
     
     