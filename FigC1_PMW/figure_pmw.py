import os
import datetime
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr 

from matplotlib.colors import Normalize
from pycolorbar import BivariateColormap, available_bivariate_colormaps
import gpm
from cartopy.mpl.geoaxes import GeoAxes
import pycolorbar
import numpy as np
from pycolorbar.utils.mpl_legend import resize_cax, pad_cax
import matplotlib.gridspec as gridspec

# Matplotlib settings
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10

# Transparent and white 
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

# Define figure directory
fig_dir = "/home/ghiggi/GPM_FIGURES/"
os.makedirs(fig_dir, exist_ok=True)

product = "1C-GMI-R"
product_type = "RS"  # if ~48 h from real-time data, otherwise "RS" (Research) ...
version = 7

start_time = datetime.datetime.strptime("2021/08/29 15:12:00", "%Y/%m/%d %H:%M:%S")
end_time = datetime.datetime.strptime("2021/08/29 15:17:00", "%Y/%m/%d %H:%M:%S")

# Download data over specific time periods
gpm.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
)

# Open scan S1 (low frequency channels) and S2 (high frequency channels)  
# 1C-GMI-R already collocated channels across S1 and S2
dt = gpm.open_datatree(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
 
)
ds_s1 = dt["S1"].to_dataset()
ds_s2 = dt["S2"].to_dataset()
ds_s2 = ds_s2.drop_vars(["lon", "lat"]) 
ds = xr.concat((ds_s1, ds_s2), dim="pmw_frequency")

####---------------------------------------------------------------------------------------.
#### Retrieve PCT, PD and RGB composites

# Retrieve dataset of brightness temperature
ds_tc = ds["Tc"].gpm.unstack_dimension(dim="pmw_frequency", suffix="_")

# Compute polarization difference
ds_pd = ds.gpm.retrieve("polarization_difference") # ds.gpm.retrieve("PD")

# Compute polarization ratio
ds_pr = ds.gpm.retrieve("polarization_ratio") # ds.gpm.retrieve("PR")

# Compute polarization correct temperature
# - Remove surface water signature on land brightness temperature
ds_pct = ds.gpm.retrieve("polarization_corrected_temperature")   # ds.gpm.retrieve("PCT")
 
# Retrieve RGB composites
ds_rgb = ds.gpm.retrieve("rgb_composites")

####---------------------------------------------------------------------------------------.
#### Retrieve UMAP RGB composite
# Create Dataset with the variables over which to apply dimensionality reduction with UMAP
list(ds_tc)

variables = [
    'Tc_10V',
    'Tc_19V',
    'Tc_23V',
    'Tc_37V',
    'Tc_89V',
    'Tc_165V',
    "PD_10",
    'PD_19',
    'PD_37',
    'PD_89',
    'PD_165',
    'Tc_183V7',
]

ds_channels = ds_tc.copy()
ds_channels.update(ds_pd)
ds_channels = ds_channels[variables].compute()

####-------------------------------------------
#### Define figure settings
vmin = 80
vmax = 300
cmap = "Spectral_r"
extent = [-95, -83, 24, 35]
add_labels = False


#####------------------------------------------------------------------------------------------------------.
#### Create figure
#### Create figure without DPR
figsize = (8, 7.4)
dpi = 300
crs_proj = ccrs.PlateCarree()

fig = plt.figure(figsize=figsize, dpi=dpi)
gs = gridspec.GridSpec(4, 6, figure=fig,
                       height_ratios=[1, 1, 1, 1], 
                       width_ratios=[1, 1, 1, 0.45, 0.05, 0.45])

# Define Cartopy map axes
axes = np.zeros((4, 3)).astype("object")
for row in range(4):
    for col in range(3):
        axes[row, col] = fig.add_subplot(gs[row, col], projection=crs_proj)

# Create colorbar axes
cax_bt = fig.add_subplot(gs[0, 3])
cax_pct = fig.add_subplot(gs[1, 3])
cax_pd = fig.add_subplot(gs[1, 5])
cax_biv = fig.add_subplot(gs[2, 3:])

#### - Add 10H, 89H, 165H Plots
ds["Tc"].sel(pmw_frequency="10H").gpm.plot_map(ax=axes[0, 0], vmin=vmin, vmax=vmax, cmap="Spectral_r", add_labels=add_labels, add_colorbar=False)
ds["Tc"].sel(pmw_frequency="89H").gpm.plot_map(ax=axes[0, 1], vmin=vmin, vmax=vmax, cmap="Spectral_r", add_labels=add_labels, add_colorbar=False)
mappable_tc = ds["Tc"].sel(pmw_frequency="165H").gpm.plot_map(ax=axes[0, 2], vmin=vmin, vmax=vmax, extend="both", cmap="Spectral_r", add_labels=add_labels, add_colorbar=False)

#### - Add PCT10 PD89 PD166 Plots
mappable_pct_10 = ds_pct["PCT_10"].gpm.plot_map(ax=axes[1, 0], cmap="rainbow_PuBr_r",  vmin=280, vmax=300, add_labels=add_labels, add_colorbar=False) 
mappable_pd_89 = ds_pd["PD_89"].gpm.plot_map(ax=axes[1, 1],  cmap="icefire", vmin=0, vmax=20, add_labels=add_labels, add_colorbar=False)
mappable_pd_165 = ds_pd["PD_165"].gpm.plot_map(ax=axes[1, 2], cmap="icefire", vmin=0, vmax=20, add_labels=add_labels, add_colorbar=False)

#### - Add Bivariate Colormaps Plots (19, 37, 89)
n_pd = 10 
n_t = 10 
for i, freq in enumerate([19, 37,89]):
    # Extract PD and brightness temperature 
    da_pd = ds_pd[f"PD_{freq}"]
    da_t = ds_tc[f"Tc_{freq}V"]  
    # Define norms
    norm_pd = Normalize(vmin=0, vmax=11, clip=True)
    norm_t = Normalize(vmin=200, vmax=300, clip=True)
    # Define colormap
    bivariate_cmap = BivariateColormap.from_name(name='ziegler', n=(n_t, n_pd))
    # Map values to colors
    da_rgba = bivariate_cmap(x=da_t, y=da_pd, norm_x=norm_t, norm_y=norm_pd)
    # Plot colormap  
    p = da_rgba.gpm.plot_map(ax=axes[2, i], rgb="rgba", add_labels=add_labels)

#### - RGB composites 
ds_rgb["NRL_37"].gpm.plot_map(ax=axes[3, 0], rgb="rgb", add_labels=add_labels)
ds_rgb["NRL_89"].gpm.plot_map(ax=axes[3, 1], rgb="rgb", add_labels=add_labels)
ds_rgb["165 + 183 GHz"].gpm.plot_map(ax=axes[3, 2], rgb="rgb", add_labels=add_labels)

#### - Add legend titles
legend_location = "upper left"
axes[0, 0].legend(labels=["10 H GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[0, 1].legend(labels=["89 H GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[0, 2].legend(labels=["166 H GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[1, 0].legend(labels=["PCT 10 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[1, 1].legend(labels=["PD 89 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[1, 2].legend(labels=["PD 166 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[2, 0].legend(labels=["19 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[2, 1].legend(labels=["37 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[2, 2].legend(labels=["89 GHz"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[3, 0].legend(labels=["NRL 37 GHz RGB"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[3, 1].legend(labels=["NRL 89 GHz RGB"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)
axes[3, 2].legend(labels=["166 + 183 GHz RGB"], loc=legend_location, fancybox=True, framealpha=0.8, handlelength=0, handleheight=0, handletextpad=0)

#### - Remove bottom axis labels from all subplots except the bottom ones
n_rows = axes.shape[0]
n_cols = axes.shape[1]
for i in range(0, n_rows):
    for j in range(0, n_cols):
        axes[i, j].set_extent(extent)

for i in range(0, n_rows - 1):
    for j in range(0, n_cols):
        if isinstance(axes[i, j], GeoAxes):
            gl = axes[i, j]._gridliners[0]
            gl.bottom_labels = False
          
for i in range(0, n_rows):
    for j in range(1, n_cols):
        if isinstance(axes[i, j], GeoAxes):
            gl = axes[i, j]._gridliners[0]
            gl.left_labels = False

# Adjust layout
fig.tight_layout()

# Fine-tune spacing
fig.subplots_adjust(hspace=0.005, wspace=0.02)

#### - Add colorbar for Z plot
cbar_kwargs = {
    'extend': 'both',
    'extendfrac': 0.05,
    'extendrect': False,
}

cax_bt = resize_cax(cax_bt, width_percent=20, height_percent=90, x_alignment='left', y_alignment='center')
cax_bt = pad_cax(cax_bt, pad_left=30)
cbar_tc = pycolorbar.plot_colorbar(mappable_tc, cax=cax_bt, label="BT [K]", **cbar_kwargs)
 

cax_pct = resize_cax(cax_pct, width_percent=20, height_percent=90,  x_alignment='left', y_alignment='center')
cax_pct = pad_cax(cax_pct, pad_left=30)
cbar_pct = pycolorbar.plot_colorbar(mappable_pct_10, cax=cax_pct, label="PCT [K]", **cbar_kwargs)   

cax_pd = resize_cax(cax_pd, width_percent=15, height_percent=90,  x_alignment='left', y_alignment='center')
cbar_pd = pycolorbar.plot_colorbar(mappable_pd_89, cax=cax_pd, label="PD [K]", **cbar_kwargs)


cax_biv = resize_cax(cax_biv, width_percent=60, height_percent=60,  x_alignment='center', y_alignment='center')
cax_biv = pad_cax(cax_biv, pad_left=25, pad_bottom=15)

#### - Add bivariate colorbar 
bivariate_cmap.plot_colorbar(
    cax=cax_biv,
    origin="lower",
    xlabel="V [K]", ylabel="PD [K]",
)

#### - Save figure
image_fpath = os.path.join(fig_dir, "BAMS_PMW1.png") 
fig.savefig(image_fpath)





 

