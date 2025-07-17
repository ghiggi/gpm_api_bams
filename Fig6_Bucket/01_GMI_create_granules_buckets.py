import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import gpm # noqa
import xarray as xr
import dask 
import logging
from gpm.bucket import write_granules_bucket, LonLatPartitioning
from gpm.io.local import get_local_filepaths
from gpm.io.find import find_associated_filepath
from gpm.utils.orbit import get_orbit_mode

# Do not warn for invalid coordinates
gpm.config.set({"warn_invalid_geolocation": False})
gpm.config.set({"warn_non_contiguous_scans": False})

# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
xr.set_options(file_cache_maxsize=1)

from dask.distributed import Client, LocalCluster


if __name__ == "__main__": #  https://github.com/dask/distributed/issues/2520
    ####----------------------------------------------------------------------.
    #### Notes 
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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["BLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
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
        silence_logs=logging.ERROR,
    )
    
    client = Client(cluster)
        
    ####----------------------------------------------------------------------.
    #### Define GridBucket product, variables and settings 
    # Define geographic grid bucket directory 
    bucket_dir = "/t5500/ltenas8/data/GPM_Buckets_Granules/GMI"

    # Define partitioning
    spatial_partitioning = LonLatPartitioning(size=[4,4], labels_decimals=0)
    
    # Define processing options
    parallel = True
    max_dask_total_tasks = 1000
    
    # Define input granules 
    product = "1C-GMI-R"
    product_type = "RS"
    version = 7
    
    ####----------------------------------------------------------------------.
    #### List all available files 
    filepaths = get_local_filepaths(product=product, 
                                    product_type=product_type, 
                                    version=version)
    
    ####----------------------------------------------------------------------.
    #### Define the granule to dataframe conversion function 
    def create_dataframe_from_granule(filepath): 
        #---------------------------------------------------------------------.
        #### Retrieve the 2A-GMI-CLIM granule associated to the 1C-GMI-R product
        filepath_2a = find_associated_filepath(filepath, 
                                               product="2A-GMI-CLIM",
                                               version= 7)
        
        #---------------------------------------------------------------------.
        #### Define the variables of interest
        variables_1c = [
            [
             'Quality',
             #'FractionalGranuleNumber',
             # 'SCaltitude',
             'SClatitude', # --> to determine orbit_mode
             # 'SClongitude',
             'Tc',
             'incidenceAngle', 
             'sunGlintAngle',
             'sunLocalTime']
        ]

        variables_2a = [
            'pixelStatus', # 0=valid, >1 invalid pixel
            'qualityFlag', # 0=valid, 1: caution, 2: snow-covered surface, 3: channels missing 
            
            'airmassLiftIndex', # index of atmospheric conditions conductive to orographic precipitation
            'surfaceTypeIndex',
            "temp2mIndex",
            
            'cloudWaterPath',
            'iceWaterPath',
            'rainWaterPath',
            "totalColumnWaterVaporIndex",  # from model (ECMWF/GANAL)
            
            'surfacePrecipitation',
            'mostLikelyPrecipitation', # surface precipitation value (best Bayesian retrieval)
            'convectivePrecipitation',
            'frozenPrecipitation',
            'precip1stTertial', # 33.33 percentile of the distribution
            'precip2ndTertial', # 66.66 percentile of the distribution
            
            'precipitationYesNoFlag', # 0 non-rainy, 1 rainy
            'probabilityOfPrecip',    # 0-100
                    
            # Variables taken from 1C 
            # 'FractionalGranuleNumber',
            # 'SCaltitude',
            # 'SClatitude',
            # 'SClongitude',
            # 'L1CqualityFlag',   
            #  'sunGlintAngle',  
            # 'sunLocalTime',
            
             # Variables with nspecies diimension 
             # - profile species are: rain water content, cloud water content, snow water content, graupel/hail content and latent heating. 
             # - However, the graupel/hail content and latent heating  profiles are currently set to missing, with the 
             #   latter to be implemented in the next version (V8).
             # 'profileNumber',
             # 'profileScale',
             # 'profileTemp2mIndex',  
             # 'temp2mIndex',
             # 'totalColumnWaterVaporIndex', # 0 to 78 mm
        ]
        
        #---------------------------------------------------------------------.
        #### Open the granules 
        # Open scan S1 (low frequency channels) and S2 (high frequency channels)  
        # - 1C-GMI-R is already collocated channels across S1 and S2
        dt = gpm.open_granule_datatree(
            filepath=filepath,
            variables=variables_1c, 
            chunks=-1,   
            cache=False,
        )
        ds_s1 = dt["S1"].to_dataset()
        ds_s2 = dt["S2"].to_dataset()
 
        # Open 2A-CLIM product
        ds_2a = gpm.open_granule_dataset(filepath_2a, variables=variables_2a, 
                                         chunks=-1, cache=False)
        
        #---------------------------------------------------------------------.
        #### Concatenate together the scan modes of 1C-GMI-R
        # Take care of the different incidence/sun angle between LF and HF channels
        # - We just keep one sun local time 
        # Compute orbit mode
        ds_s1 = ds_s1.assign_coords({"orbit_mode": get_orbit_mode(ds_s1)})
    
        # Take care of the different incidence/sun angle between LF and HF channels
        # - We just keep one sun local time 
        ds_s1 = ds_s1.rename_vars({
            "incidenceAngle": "incidenceAngle_LF",
            "sunGlintAngle": "sunGlintAngle_LF",
            # "sunLocalTime": "sunLocalTime_LF",
            "Quality": "Quality_LF",
        }).squeeze()
    
        ds_s2 = ds_s2.rename_vars({
            "incidenceAngle": "incidenceAngle_HF",
            "sunGlintAngle": "sunGlintAngle_HF",
            # "sunLocalTime": "sunLocalTime_HF",
            "Quality": "Quality_HF",
        }).squeeze()
    
        # Concat 1C-R together
        ds_s2 = ds_s2.drop_vars(["lon", "lat", "sunLocalTime", "SCorientation", "SClatitude"]) 
        ds_1c = xr.merge((ds_s1, ds_s2))
        
        #---------------------------------------------------------------------.
        #### Unstack the Tc pmw_frequency dimension
        # Add a Tc variable for each frequency 
        for i, freq in enumerate(ds_1c["pmw_frequency"].values):
            # Select the data for the current frequency
            da_freq = ds_1c["Tc"].isel({"pmw_frequency": i})    
            # Create a variable name based on the frequency
            var_name = f"Tc_{freq}"
            # Add this data array as a variable to the dataset
            ds_1c[var_name] = da_freq
        ds_1c = ds_1c.drop_vars("Tc") 
            
        #---------------------------------------------------------------------.
        #### Combine all data together 
        ds = xr.merge((ds_1c, ds_2a))
        
        #---------------------------------------------------------------------.
        #### Read data in memory 
        ds = ds.compute()
        
        #---------------------------------------------------------------------.
        #### Convert to pandas dataframe 
        df = ds.gpm.to_pandas_dataframe()

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

    
 
 
    
    
    
    
  