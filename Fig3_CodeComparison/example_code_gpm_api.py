import gpm
# Define products to analyze
product = "2A-DPR"
product_type = "RS"
version = 7
# Define analysis time period
start_time = "2021-08-29 15:10:00"
end_time = "2021-08-29 15:18:00"
# Define geographic area of interest 
extent = [-95.5, -83.5, 25.5, 33.5]
# Download data
gpm.download(start_time=start_time, end_time=end_time,
    product=product, product_type=product_type, version=version)
# Open dataset
ds = gpm.open_dataset(start_time=start_time, end_time=end_time, 
    product=product, product_type=product_type, version=version)
# Select data over time period of interest 
ds = ds.gpm.sel(time=slice(start_time, end_time))
# Crop data to the area of interest
ds = ds.gpm.crop(extent)
# Display the surface precipitation map 
p = ds["precipRateNearSurface"].gpm.plot_map()
p.axes.set_extent(extent)
# Display the radar reflectivity cross-section
da_reflectivity = ds["zFactorFinal"].sel(radar_frequency="Ku")
da_transect = da_reflectivity.isel({"along_track": slice(54, 168), 'cross_track': 18})
da_transect.gpm.plot_cross_section(x="lon", y="height", zoom=False)