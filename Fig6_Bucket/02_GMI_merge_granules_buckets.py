####----------------------------------------------------------------------.
#### REQUIREMENT
# Before running this script, change the ulimit (Open File Limit) close to hard limit 
# --> https://stackoverflow.com/questions/34588/how-do-i-change-the-number-of-open-files-limit-in-linux
# --> See hard limit with: ulimit -Hn
# --> See current soft limit with: limit -Sn
# --> Modify soft limit with: ulimit -n 999999
from gpm.bucket import merge_granule_buckets


if __name__ == "__main__":
    ####----------------------------------------------------------------------.
    #### Define bucket filepaths 
    src_bucket_dir = "/t5500/ltenas8/data/GPM_Buckets_Granules/GMI"
    dst_bucket_dir = "/t5500/ltenas8/data/GPM_Buckets/GMI"  
    
    ####----------------------------------------------------------------------.
    #### Merge Parquet Datasets     
    merge_granule_buckets(
        # Bucket directories
        src_bucket_dir=src_bucket_dir,
        dst_bucket_dir=dst_bucket_dir,
        # Consolidation options
        row_group_size="200MB",
        max_file_size="2GB",
        max_open_files=0,
        use_threads=True,
        compression="snappy",
        compression_level=None, 
        # Scanner options
        batch_size=131_072,
        batch_readahead=10,  # 16
        fragment_readahead=20,  # 4
    )
    ####----------------------------------------------------------------------.


