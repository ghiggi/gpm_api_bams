import gpm # noqa
import os
import matplotlib
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
import pycolorbar
from pycolorbar.univariate import (
    get_discrete_cyclic_cmap,
    plot_circular_colorbar,
)
from pycolorbar.norm import CategorizeNorm

# Define directories
fig_dir = "/ltenas8/data/tmp/GPM_Figures"
zarr_dir = "/t5500/export-ltesrv8/GPM/GPM_ZARR"
zarr_dir = "/ltesrv8/GPM/GPM_ZARR"

os.makedirs(fig_dir, exist_ok=True)
   
# Matplotlib settings
matplotlib.get_backend()
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10

####-----------------------------------------------------------------------.
#### Load datasets
monthly_store = os.path.join(zarr_dir, "IMERG_Global/MonthlyStore.zarr")
ds_monthly = xr.open_zarr(monthly_store, consolidated=True)

diurnal_cycle_store = os.path.join(zarr_dir, "IMERG_Statistics/DiurnalCycleStore.zarr")
ds_hourly_stats = xr.open_zarr(diurnal_cycle_store, consolidated=True)

# Define precipitation DataArray
da_monthly = ds_monthly["precipitation"].compute()
da_monthly.name = "dummy"

# Compute monthly climatology
da_monthly_mean = da_monthly.groupby("time.month").mean()

####-----------------------------------------------------------------------.
#### Month with maximum precipitation
da_month_with_max_precip = da_monthly_mean.argmax("month")
da_month_with_max_precip.name = "Month with maximum precipitation"

#### Month with minimum precipitation
da_month_with_min_precip = da_monthly_mean.argmin("month")
da_month_with_min_precip.name = "Month with minimum precipitation"

#### Compute annual precipitation [in mm]
da_annual = da_monthly.resample(time='1YS').sum()  

#### Long-Term Mean Precipitation  (LTM)
da_annual_ltm = da_annual.mean(dim="time") 

#### Compute diurnal cycle peak hour
def get_peak_hour(da):
    idx_max_hour = da.argmax(dim="hour").compute()
    da_utc_hour = da["hour"].isel(hour=idx_max_hour) 
    da_lst_hour = np.round(da_utc_hour + da_utc_hour["lon"]/15) % 24
    return da_lst_hour 

da_lst_hour = get_peak_hour(ds_hourly_stats["accumulation"].sel(threshold=0))


####-----------------------------------------------------------------------.
#### Create figure    
# Define figure settings
dpi = 300
figsize = (8, 10)

# Define Cartopy projections
crs_proj = ccrs.EqualEarth()

# Create the map with month with max precipitation
fig, axes = plt.subplots(3, 1, 
                       subplot_kw={"projection": crs_proj},
                       figsize=figsize, dpi=dpi)

#----------------------------------------------------------------------.
#### - LTM Precipitation Plot
axes[0].coastlines(alpha=0.6, color="#2e2a2b", linewidth=0.3)
cbar_kwargs = {} 
cbar_kwargs["ticks"] = [100, 200, 500, 1000, 2000, 5000]
cbar_kwargs["ticklabels"] = [100, 200, 500, 1000, 2000, 5000]
cbar_kwargs["extend"] = "both"
cbar_kwargs["extendfrac"] = 0.05
cbar_kwargs["label"] = "Precipitation [mm/year]"
norm = LogNorm(vmin=100, vmax=5000)
p = da_annual_ltm.gpm.plot_map(
    ax=axes[0],
    cmap="YlGnBu",
    norm=norm, 
    add_background=False,
    add_gridlines=False, 
    add_labels=False, 
    add_colorbar=False, 
    cbar_kwargs=cbar_kwargs)
cbar = pycolorbar.plot_colorbar(p=p, ax=axes[0], cax=None, 
                         size="25%", pad=0.3,
                         **cbar_kwargs)
cbar.ax.set_aspect(8)
axes[0].set_title("Annual Precipitation")

#----------------------------------------------------------------------.
#### - Month with Maximum Precipitation Plot
# Define norm, ticks and ticklabels
data = np.arange(1, 13)
norm = Normalize(1, 12)  # for plot data
ticklabels = ["Jan", "Feb", "March", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ticks = np.linspace(0, 2*np.pi - (2*np.pi / len(ticklabels)), len(ticklabels))
ticks = ticks + np.diff(ticks)[0] / 2  # centered labels
n = int(norm.vmax - norm.vmin + 1)

# Retrieve cmap
cmap = pycolorbar.get_cmap("infinity")
cmap_discrete = get_discrete_cyclic_cmap(cmap, n)

# Create figure
axes[1].coastlines(alpha=0.6, color="#2e2a2b", linewidth=0.3)
p = da_month_with_max_precip.gpm.plot_map(
    ax=axes[1],
    add_background=False,
    add_gridlines=False, 
    add_labels=False, 
    add_colorbar=False, 
    cmap=cmap_discrete,
    norm=norm,
)
axes[1].set_title("Most Precipitating Month")
# axes[1].set_title("Month with maximum precipitation")
plot_circular_colorbar(
    ax=axes[1],
    cmap=cmap_discrete, 
    ticks=ticks,
    ticklabels=ticklabels, 
    size="25%",
    pad=0.3,
    # Approach
    use_wedges=True,
    r_min=0.2,
    r_max=0.5,
    antialiased=False,
    wedges_edgecolor="none",
    wedges_linewidths=None,
    # Contour
    add_contour=True,
    contour_color="black",
    contour_linewidth=None,
    # Ticks (not available for method="polar")
    add_ticks=True,
    ticklength=0.02,
    tickcolor="black",
    tickwidth=1,
    # Ticklabels
    ticklabels_pad=None,
    ticklabels_size=8,
    )

#----------------------------------------------------------------------.
#### - Diurnal Cycle Plot

# Define norm, ticks and ticklabels
n_max = 24 # # Diurnal Cycle: 0 - 24 --> Rescaled to 0-1 ([23-24] is 1, 24 is 0)
data = np.arange(0, n_max)
norm = Normalize(0, n_max)  # for plot data
ticklabels = np.arange(0, n_max)

# Define category norm 
boundaries = np.arange(0, n_max+1)
norm = CategorizeNorm(boundaries=boundaries, labels = boundaries[0:-1])
ticklabels = norm.ticklabels
ticks = np.linspace(0, np.pi*2 - (np.pi*2 / len(ticklabels)), len(ticklabels))
n = int(norm.vmax - norm.vmin)

ticklabels = ticklabels.astype(str)
ticklabels[1::2] = ""

# Retrieve cmap
cmap = pycolorbar.get_cmap("twilight_shifted")
cmap_discrete = get_discrete_cyclic_cmap(cmap, n)

# Create figure
axes[2].coastlines(alpha=0.6, color="#2e2a2b", linewidth=0.3)
p = da_lst_hour.gpm.plot_map(
    ax=axes[2],
    add_background=False,
    add_gridlines=False, 
    add_labels=False, 
    add_colorbar=False, 
    cmap=cmap_discrete,
    norm=norm,
)
axes[2].set_title("Most Precipitating Hour")
plot_circular_colorbar(
    ax=axes[2],
    cmap=cmap_discrete, 
    ticks=ticks,
    ticklabels=ticklabels, 
    size="25%",
    pad=0.3,
    # Approach
    use_wedges=True,
    r_min=0.2,
    r_max=0.5,
    antialiased=False,
    wedges_edgecolor="none",
    wedges_linewidths=None,
    # Contour
    add_contour=True,
    contour_color="black",
    contour_linewidth=None,
    # Ticks (not available for method="polar")
    add_ticks=True,
    ticklength=0.02,
    tickcolor="black",
    tickwidth=1,
    # Ticklabels
    ticklabels_pad=None,
    ticklabels_size=8,
    )

 
# Save figure
image_filepath = os.path.join(fig_dir, "IMERG_BAMS.png")
fig.savefig(image_filepath)
plt.show()
plt.close()
