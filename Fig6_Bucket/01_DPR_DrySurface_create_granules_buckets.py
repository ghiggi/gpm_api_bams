import os 
import gpm # noqa
import dask 
import logging
from gpm.io.local import get_local_filepaths
from gpm.bucket import write_granules_bucket, LonLatPartitioning

# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})

from dask.distributed import Client, LocalCluster


if __name__ == "__main__": #  https://github.com/dask/distributed/issues/2520
    # Notes 
    # - Zarr input required for better perfomance 
    # - This code is penalized by HDF/netCDF locking 
    #   - Even if HDF5 is compiled with thread safety, the netcdf4 C library is not thread safe.
    #   - We use multiprocessing to get partially around that 
    # - This code use dask.delayed. dask.delayed works only with dask.distributed ! 
    # - libnetcdf requirements: 
    #    --> Either downgrade libnetcdf to 4.8.1 to avoid HDF error/warnings 
    #    --> Either upgrade to libnetcdf 4.9.2 to avoid HDF error/warnings
    #    --> https://github.com/SciTools/iris/issues/5187
    
    ####----------------------------------------------------------------------.
    #### Define Dask Distributed Cluster   
    # Set environment variable to avoid HDF locking
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Set number of workers 
    # dask.delayed run n_workers*2 concurrent processes
    available_workers = int(os.cpu_count()/2)
    num_workers = dask.config.get("num_workers", available_workers)
        
    # Create dask.distributed local cluster
    # --> Use multiprocessing to avoid netCDF multithreading locks ! 
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1, # important to set to 1 to avoid netcdf locking ! 
        processes=True,
        memory_limit="120GB",
        silence_logs=logging.WARN,
    )
    
    client = Client(cluster)
    
    ####----------------------------------------------------------------------.
    #### Define GridBucket product, variables and settings 
    
    # Define geographic grid bucket directory 
    bucket_dir = "/t5500/ltenas8/data/GPM_Buckets_Granules/DPR_Dry/"
    os.makedirs(bucket_dir)
      
    # Define partitioning
    spatial_partitioning = LonLatPartitioning(size=[2, 2], labels_decimals=0)
     
    # Define processing options 
    parallel = True
    max_dask_total_tasks = 1000
        
    # Define GPM product
    product = "2A-DPR"
    product_type = "RS"
    version = 7   
        
    ####----------------------------------------------------------------------.
    #### List all available files 
    print("Listing available granules")
    filepaths = get_local_filepaths(product=product, 
                                    product_type=product_type, 
                                    version=version)

    ####----------------------------------------------------------------------.
    #### Define the granule to dataframe conversion function 
    def create_dataframe_from_granule(filepath):
        #----------------------------------------------------------------------.
        #### Define the variables of interest
        # Define GPM variables relevant for footprints with precipitation
        variables = [            
            "heightZeroDeg",
            
            "binClutterFreeBottom",
            "binRealSurface",          

            "flagPrecip",      # 0 No precipitation, >1 precipitation
            
            "sunLocalTime", 
            "localZenithAngle",
            "landSurfaceType",
            "elevation",
            "flagSurfaceSnowfall",
            "snowIceCover",
            "seaIceConcentration",
            
            "flagSigmaZeroSaturation",
            "sigmaZeroCorrected",
            "sigmaZeroMeasured",
            "sigmaZeroNPCorrected",
            "snRatioAtRealSurface",
            
            "airTemperature",
           
        ]
        
        #----------------------------------------------------------------------------------.
        #### Open the granules 
        scan_mode = "FS" 
        open_granule_kwargs = {
            "scan_mode": scan_mode,
            "groups": None,
            "variables": variables,
            "decode_cf": True,
            "chunks": -1,
        }
        
        ds = gpm.open_granule_dataset(filepath, **open_granule_kwargs)
        
        #----------------------------------------------------------------------------------.
        #### Precompute the granules 
        ds = ds.compute()
        
        #---------------------------------------------------------------------.
        #### Compute custom variables / features    
        # Compute variables for both frequecies: Ku and Ka
        for r_f in ["Ku", "Ka"]:
            ds_f = ds.sel({"radar_frequency": r_f})
            ds[f"dataQuality_{r_f}"] = ds_f["dataQuality"]
            for var in ds.gpm.frequency_variables:
                # Standard variables
                ds[f"{var}_{r_f}"] = ds_f[var]     
                
            # Custom variables
            ds[f"heightRealSurface_{r_f}"] = ds_f.gpm.get_height_at_bin(bins=ds_f["binRealSurface"])
        
        ds["heightClutterFreeBottom"] = ds.gpm.retrieve("heightClutterFreeBottom")
        
        # Take maximum temperature 
        ds["airTemperatureMax"] = ds["airTemperature"].max(dim="range") - 273.15
        
        #---------------------------------------------------------------------.
        #### Discard unrelevant variables  
        ds = ds.drop_vars(ds.gpm.frequency_variables)
        ds = ds.drop_vars(ds.gpm.spatial_3d_variables)
        ds = ds.drop_vars([
            # Variables not used
            "binClutterFreeBottom", 
            "binRealSurface_Ku",
            "binRealSurface_Ka",            
            # Remove multifrequency and 3D coordinates
            "dataQuality",
            "height"]
        )
        
        #---------------------------------------------------------------------.
        #### Convert to pandas dataframe 
        df = ds.gpm.to_pandas_dataframe()
        
        #---------------------------------------------------------------------.
        #### Filter the dataframe 
        # Filter dataset where is not raining in the column 
        df = df.loc[df["flagPrecip"] == 0, :]
        
        #---------------------------------------------------------------------.
        return df 

    ####----------------------------------------------------------------------.
    #### Compute Granule Buckets
    # ---> Multiprocessing for high performance
    # ---> It process by batches of 2*n_workers
    writer_kwargs = {}
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=create_dataframe_from_granule,
        # Processing options
        parallel=True,
        max_concurrent_tasks=None,
        max_dask_total_tasks=max_dask_total_tasks,
        # Writer kwargs 
        **writer_kwargs,
    )
    ####----------------------------------------------------------------------.


    
 
 
    
    
    
    
  