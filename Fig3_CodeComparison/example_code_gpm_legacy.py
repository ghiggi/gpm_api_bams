import gpm
import cartopy
import pycolorbar
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
# Define analysis time period
start_time = "2021-08-29 15:10:00"
end_time = "2021-08-29 15:18:00"
# Define geographic area of interest 
extent = [-95.5, -83.5, 25.5, 33.5]
# Define path of the file to open 
filepath = "[...]"
# Open the HDF file in read mode
ds = Dataset(filepath, mode="r")
# Extract required coordinates and variables into numpy arrays
lat = ds["/FS/Latitude"][:].data
lon = ds["/FS/Longitude"][:].data
ds_scan_time = ds["FS"]["ScanTime"]
dict_time = {k: ds_scan_time[k][:] for k in ["Year", "Month", "DayOfMonth", "Hour", "Minute", "Second"]}
time = pd.to_datetime(dict_time).to_numpy()
surface_precip = ds["FS"]["SLV"]["precipRateNearSurface"][:].data
surface_precip_fillvalue = ds["FS"]["SLV"]["precipRateNearSurface"].getncattr("_FillValue") # -9999.9
surface_precip[surface_precip == surface_precip_fillvalue] = np.nan
# Select data over time period of interest 
idx_time = np.where((time >= pd.Timestamp(start_time)) & (time <= pd.Timestamp(end_time)))[0]
time = time[idx_time]
lat = lat[idx_time, :]
lon = lon[idx_time, :]
surface_precip = surface_precip[idx_time, :]
# Crop data to the area of interest
idx_at, idx_ct = np.where(((lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])))
idx_along_track = sorted(np.unique(idx_at))
idx_cross_track = sorted(np.unique(idx_ct))
slc_along_track = slice(idx_along_track[0], idx_along_track[-1]+1)
slc_cross_track = slice(idx_cross_track[0], idx_cross_track[-1]+1)
time = time[slc_along_track]
lat = lat[slc_along_track, slc_cross_track]
lon = lon[slc_along_track, slc_cross_track]
surface_precip = surface_precip[slc_along_track, slc_cross_track]
# Display the surface precipitation map 
# - Get colormaps and colorbar arguments from GPM-API configurations files
plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs("precipRateNearSurface")
# - Initialize figure with geographic projection
fig, ax = plt.subplots(1,1, subplot_kw={"projection": ccrs.PlateCarree()})
# - Add background
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax.add_feature(cartopy.feature.BORDERS)
# - Add gridlines and labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,  linewidth=1, color="gray", alpha=0.1, linestyle="-")
gl.top_labels = False   
gl.right_labels = False   
gl.xlines = True
gl.ylines = True
# - Add the precipitation map 
p = ax.pcolormesh(lon, lat, surface_precip, **plot_kwargs)
# - Add swath lines 
ax.plot(lon[:, 0], lat[:, 0], transform=ccrs.Geodetic(), linestyle="--", color="k",)
ax.plot(lon[:, -1], lat[:, -1], transform=ccrs.Geodetic(), linestyle="--", color="k",)
# - Add the colorbar with pycolorbar
pycolorbar.plot_colorbar(p, ax=ax, **cbar_kwargs)
# - Set correct extent
ax.set_extent(extent)