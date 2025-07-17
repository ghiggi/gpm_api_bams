#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:47:00 2024

@author: ghiggi
"""
import os
import zarr
import dask 
import time 
import datetime
import logging
import gpm # noqa
from dask.diagnostics import ProgressBar
from gpm.io.local import get_local_filepaths
from gpm.encoding.routines import set_encoding
from gpm.encoding.encode_imerg_v7 import get_encoding_dict
from gpm.io.local import get_local_dir_tree_from_filename
from xencoding.zarr.numcodecs import get_compressor
from xencoding.checks.chunks import check_chunks
from xencoding.checks.zarr_compressor import check_compressor
from xencoding.zarr.writer import set_compressor, set_chunks

# dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})
dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})

from dask.distributed import Client, LocalCluster

def _rewrite_to_zarr(filepath, zarr_base_dir, force=True):
    """Rewrite a single GPM file to Zarr."""
    with dask.config.set(scheduler="single-threaded"): # very important !
        # Define ZarrStore file path 
        dir_tree = get_local_dir_tree_from_filename(filepath, product_type="RS", base_dir=zarr_base_dir)
        os.makedirs(dir_tree, exist_ok=True)
        filename = os.path.splitext(os.path.basename(filepath))[0]
        store_fpath = os.path.join(dir_tree, f"{filename}.zarr.zip")
       
        # Check what to do if exists
        if os.path.exists(store_fpath): 
            if force: 
                os.remove(store_fpath)
            else:
                return None 
    
        # Read all data into memory in a single chunk (faster) with chunks=-1
        ds = gpm.open_granule_dataset(filepath, chunks=-1, decode_cf=True)
       
        # Set compressor 
        compressor =  get_compressor(compressor_name="zstd", clevel=6)
        compressor_dict = check_compressor(ds, compressor)
        ds = set_compressor(ds, compressor_dict)
        
        # Set encoding 
        encoding_dict = get_encoding_dict()
        ds = set_encoding(ds, encoding_dict=encoding_dict)
       
        # Set chunks
        chunks_dict = {"time": 1, "lat": -1, "lon": -1}
        chunks_dict = check_chunks(ds, chunks=chunks_dict)
        ds = set_chunks(ds, chunks_dict=chunks_dict)
     
        # Write to Zarr Store
        with zarr.ZipStore(store_fpath, mode='w') as store:
            ds.to_zarr(store=store)


 
@dask.delayed
def rewrite_to_zarr(filepath, zarr_base_dir, force=True):
    """Rewrite a single GPM file to Zarr."""
    try: 
        _ = _rewrite_to_zarr(filepath, zarr_base_dir=zarr_base_dir, force=force)
        result = None
    except Exception: 
        result = filepath 
    return result


if __name__ == "__main__":  # https://github.com/dask/distributed/issues/2520
     
####-----------------------------------------------------------------------------.
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
     
     ####-----------------------------------------------------------------------------.
     # Define GPM Zarr directory 
     zarr_base_dir = "/t5500/ltenas8/data/GPM_ZARR"
     force = False
     # Retrieve available filepaths 
     filepaths = get_local_filepaths(product="IMERG-FR", version=7, product_type="RS")
     
     # Define delayed computations 
     list_delayed = [rewrite_to_zarr(filepath, zarr_base_dir=zarr_base_dir, force=force) for filepath in filepaths]
     
     # Run file conversion 
     print("Starting conversion !")
     t_i = time.time()
     
     with ProgressBar():
         results = dask.compute(*list_delayed) 
         
     t_f = time.time()
     timedelta_str = str(datetime.timedelta(seconds=round(t_f-t_i, 0)))
     print(f"Conversion terminated in {timedelta_str} !")
     
     ####-----------------------------------------------------------------------------.
     # Print filepaths which were not converted !
     bad_filepaths = [result for result in results if result is not None]
     if len(bad_filepaths) > 0:
         print("Error occured for the following filepaths")
         print(bad_filepaths)
     
     ####-----------------------------------------------------------------------------.