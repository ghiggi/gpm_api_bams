import os 
import matplotlib
import pycolorbar
import matplotlib.pyplot as plt
import xarray as xr
from cartopy import crs as ccrs
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define directories
fig_dir = "/ltenas8/data/tmp/GPM_Figures"

bucket_dir = "/t5500/ltenas8/data/GPM_Buckets"
bucket_dir = "/ltenas8/data/GPM_Buckets"

os.makedirs(fig_dir, exist_ok=True)
   
# Matplotlib settings
matplotlib.get_backend()
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10

#-------------------------------------------------------------------------.
#### Open climatologies
ds_ncrs_stats = xr.open_dataset("/ltenas8/data/tmp/Radar_NRCS_surface_stats.nc")
ds_sea_ice_ratio = xr.open_dataset("/ltenas8/data/tmp/PMW_sea_ice_april.nc")
ds_pd_166 = xr.open_dataset("/ltenas8/data/tmp/PMW_PD166_LT160K_stats.nc")
ds_hail_stats = xr.open_dataset("/ltenas8/data/tmp/Radar_EchoDepth45dBZ_solid_phase_stats.nc")

#-------------------------------------------------------------------------.
#### Create figure

# Define function to plot vertical line on cartopy axes
def plot_vertical_line(lon, **kwargs):
    ax.plot([lon, lon], [-70, 70], transform=ccrs.PlateCarree(), **kwargs)

fig, ax = plt.subplots(1, 1, 
                       figsize=(8,3), 
                       dpi=1800, 
                       subplot_kw={"projection": ccrs.PlateCarree()})

#### - Hail
cmap = pycolorbar.get_cmap("Spectral_r")
cmap.set_under(alpha=0)
cbar_kwargs = {"label": 'Echo depth [km]', "extend": "both", "extendfrac": 0.06}
p_hail = (ds_hail_stats["max_EchoDepth45dBZ_solid_phase"]/1000).gpm.plot_map(
    ax=ax,
    cmap=cmap, 
    # vmax=1.2,
    add_colorbar=False, 
    add_background=False, 
    add_labels=False, 
    add_gridlines=False,
    vmin=1,
    vmax=7,
    extend="both",
    cbar_kwargs=cbar_kwargs,
)

#### - Sea Ice
cmap = pycolorbar.get_cmap("spectral_PuWh", interval=(0.05, 0.95))
p_pmw = ds_sea_ice_ratio["RV89_19_median"].gpm.plot_map(
    ax=ax,
    cmap=cmap, 
    vmin=0.9,
    vmax=1.2,
    add_colorbar=False, 
    add_background=False, 
    add_labels=False, 
    add_gridlines=False,
    fig_kwargs={"dpi":300},
)

#### - NCRS
cbar_kwargs = {"label": 'NCRS', "extend": "both", "extendfrac": 0.06}
p_ncrs = ds_ncrs_stats["median_NCRS"].gpm.plot_map(
    ax=ax,
    cmap="terrain_r",
    vmax=21,
    add_colorbar=False, 
    add_background=False, 
    add_labels=False, 
    add_gridlines=False,
    extend="both",
    cbar_kwargs=cbar_kwargs,
)

#### - PD165 
cmap = pycolorbar.get_cmap("icefire_noaa", interval=(0.3, 1))
p_pd166 = ds_pd_166["PD165_GT4"].gpm.plot_map(
    ax=ax,
    cmap=cmap,
    vmax=150,
    extend="max",
    cbar_kwargs={"extendfrac": 0.06},
    add_colorbar=False, 
    add_background=False, 
    add_labels=False, 
    add_gridlines=False,

)
 
# Add plot decorations
ax.coastlines(alpha=0.6, color="#2e2a2b", linewidth=0.3)
ax.set_extent([-180, 180, -65, 65])

# Add titles and x-coordinates in normalized Axes space
titles = ["Max DPR Solid-Phase ED45 [km]", 
          "Median GMI 89V / 19 V [-]",
          "Median DPR NRCS at near 0°[dB]", 
          "Counts GMI PD 166 > 4 K [#]"]
x_positions = [0.125, 0.375, 0.625, 0.875]

#### Add colorbars background
rect = patches.Rectangle(
    (0, 0), 1, 0.18,       # x0, y0, width, height in Axes coordinates
    transform=ax.transAxes, 
    facecolor='white', 
    alpha=0.7, 
    zorder=2               # it ensure it's drawn on top of the image but below colorbars/text
)
ax.add_patch(rect)

#### Add colorbars 
images = [p_hail,p_pmw,p_ncrs, p_pd166]
for x, im, title in zip(x_positions, images, titles):
    # Create an inset Axes at a particular location in "Axes coordinates"
    cax = inset_axes(
        ax,
        width="20%",          # fraction of the Axes width
        height="5%",          # fraction of the Axes height
        loc='lower left',     # anchor at the lower-left corner of the bounding box
        bbox_to_anchor=(x-0.1, 0.08, 1, 0.5),   # (x, y) offset in Axes coords
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    
    # Make a horizontal colorbar inside this inset Axes
    if title.startswith("Counts"):
        extend = "max"
    else: 
        extend = "both"
    cbar = pycolorbar.plot_colorbar(im, cax=cax, 
                                    orientation='horizontal',
                                    label_position="top",
                                    extend=extend,
                                    extendfrac=0.05)
    cbar.set_label(title, fontsize=6)
    cbar.ax.tick_params(axis='both', which='major', labelsize=6)

# Add vertical lines
plot_vertical_line(lon=-90, color="black", linewidth=0.5)
plot_vertical_line(lon=0, color="black", linewidth=0.5)
plot_vertical_line(lon=90, color="black", linewidth=0.5)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.tight_layout() 

fig.savefig(os.path.join(fig_dir, "BAMS_Bucket_Composite.png"))

#-------------------------------------------------------------------------.