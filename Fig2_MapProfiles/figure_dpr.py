import os
import datetime
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
from gpm.utils.geospatial import extend_geographic_extent
import gpm
import matplotlib.gridspec as gridspec
import pycolorbar
import matplotlib.ticker as mticker
import cartopy

# Matplotlib settings
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["savefig.transparent"] = False

# Define figure directory
fig_dir = "/home/ghiggi/GPM_FIGURES/"
os.makedirs(fig_dir, exist_ok=True)

####--------------------------------------------------------------------------.
#### Define analysis settings
# Define analysis time period
start_time = datetime.datetime.strptime("2021/08/29 15:10:00", "%Y/%m/%d %H:%M:%S")
end_time = datetime.datetime.strptime("2021/08/29 15:18:00", "%Y/%m/%d %H:%M:%S")

# Define products to analyze
products = [
    "2A-DPR",
    "2A-GMI",
    "2A-GPM-SLH",
    "2B-GPM-CSH",
    "2B-GPM-CORRA",
]

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
#### Download products
for product in products:
    print(product)
    gpm.download(product=product,
                      product_type=product_type,
                      version = version,
                      start_time=start_time,
                      end_time=end_time,
                      storage="GES_DISC",
                      force_download=False,
                      transfer_tool="CURL",
                      progress_bar=True,
                      verbose = True,
                      n_threads=1)

####--------------------------------------------------------------------------.
#### Define product-variable dictionary
product_var_dict = {
    "2A-DPR": [
        "precipRate",
        "zFactorFinal",
        "zFactorMeasured",
        "precipRateNearSurface",
        "zFactorFinalNearSurface",
        "paramDSD",
        "typePrecip",
        "heightZeroDeg",
    ],
    "2A-GMI": [
        "surfacePrecipitation",
    ],
    "2B-GPM-CSH": ["latentHeating"],
    "2A-GPM-SLH": ["latentHeating"],
    "2B-GPM-CORRA": [
      'precipTotDm',
      'precipTotLogNw',
      'precipTotMu',
      "precipTotRate",
      "correctedReflectFactor",
      # TO RETRIEVE HEIGHT
      "ellipsoidBinOffset", 
      "localZenithAngle",
    ],
}

####--------------------------------------------------------------------------.
#### Open the datasets
dict_product = {}
# product, variables = list(product_var_dict.items())[0]
# variables = product_var_dict[product]
for product, variables in product_var_dict.items():
    ds = gpm.open_dataset(
        product=product,
        start_time=start_time,
        end_time=end_time,
        chunks={},
        # Optional
        variables=variables,
        version=version,
        product_type=product_type,
        prefix_group=False,
       
    )
    dict_product[product] = ds


####--------------------------------------------------------------------------.
#### Retrieve datasets over AOI
# Define bounding box of interest
extent = [-95, -84, 26, 33]
bbox_extent = extend_geographic_extent(extent, padding=0.5)  

# Crop dataset
ds_gmi = dict_product["2A-GMI"].gpm.crop(bbox_extent)
ds_dpr = dict_product["2A-DPR"].gpm.crop(bbox_extent)
ds_corra = dict_product["2B-GPM-CORRA"].gpm.crop(bbox_extent)
ds_csh = dict_product["2B-GPM-CSH"].gpm.crop(bbox_extent)
ds_slh = dict_product["2A-GPM-SLH"].gpm.crop(bbox_extent)

# Check same coordinates
assert np.all(ds_csh["lat"] == ds_dpr["lat"]).item()
assert np.all(ds_csh["lon"] == ds_dpr["lon"]).item()
assert np.all(ds_corra["lon"] == ds_dpr["lon"]).item()
assert np.all(ds_corra["lat"] == ds_dpr["lat"]).item()

####--------------------------------------------------------------------------.
#### Extract transect line passing across the maximum intensity region

# Define transect isel dict 
variable = "precipRate"
isel_dict = ds_dpr[variable].gpm.locate_max_value(return_isel_dict=True)
transect_isel_dict = {"cross_track": isel_dict["cross_track"]}

transect_isel_dict = {"along_track": slice(54, 168),  'cross_track': 18}
# transect_isel_dict_csh = {"along_track": slice(45+2, 155+2),  'cross_track': 18+1} # CSH not aligned ! PRODUCT BUG !

ds_dpr_transect = ds_dpr.isel(transect_isel_dict)
ds_dpr_transect["precipRate"].gpm.plot_cross_section(x="lon", zoom=False)

ds_slh_transect = ds_slh.isel(transect_isel_dict)
ds_slh_transect["latentHeating"].gpm.plot_cross_section(zoom=False)

ds_csh_transect = ds_csh.isel(transect_isel_dict)
ds_csh_transect["latentHeating"].gpm.plot_cross_section(zoom=False)

# Extract transects
ds_dpr_transect = ds_dpr.isel(transect_isel_dict)
ds_dpr_transect = ds_dpr_transect.compute()

ds_csh_transect = ds_csh.isel(transect_isel_dict)
ds_csh_transect = ds_csh_transect.compute()

ds_corra_transect = ds_corra.isel(transect_isel_dict)
ds_corra_transect = ds_corra_transect.compute()

# Check same coordinates
assert np.all(ds_csh_transect["lat"] == ds_dpr_transect["lat"]).item()
assert np.all(ds_csh_transect["lon"] == ds_dpr_transect["lon"]).item()
assert np.all(ds_corra_transect["lon"] == ds_dpr_transect["lon"]).item()
assert np.all(ds_corra_transect["lat"] == ds_dpr_transect["lat"]).item()

# Zoom on eye
eye_extent = [-91,-89, 28.5, 29.5]
p = ds_csh["latentHeating"].max(dim="range").gpm.plot_map()
p.axes.set_extent(eye_extent)

p = ds_dpr["precipRate"].max(dim="range").gpm.plot_map()
p.axes.set_extent(eye_extent)

p = ds_corra["precipTotRate"].max(dim="range").gpm.plot_map()
p.axes.set_extent(eye_extent)

####------------------------------------------------------------------------------------------------
#### Define fields to plot
da_dpr = ds_dpr["precipRateNearSurface"].compute()
da_gmi = ds_gmi["surfacePrecipitation"].compute()
da_corra_transect = ds_corra_transect["precipTotRate"].compute()
da_csh_transect = ds_csh_transect["latentHeating"].compute()
da_dpr_transect = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ku").compute()
da_dpr_transect_ka = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ka").compute()

# Mask CSH below height where CORRA has NaN
# - CSH range from bottom to top  
# - CORRA, DPR: from top to bottom
buffer = 2
corra_last_valid_bin = np.isnan(da_corra_transect).argmax(dim="range")
corra_last_valid_height = da_corra_transect["height"].gpm.slice_range_at_bin(corra_last_valid_bin - buffer)
da_csh_transect = da_csh_transect.where(da_csh_transect["height"] > corra_last_valid_height)

# Mask GMI low precipitation 
# - 0.01 everywhere ! 
val, counts = np.unique(np.round(da_gmi.data,2), return_counts=True)
val[0:10]
counts[0:10]
attrs = da_gmi.attrs.copy()
with xr.set_options(keep_attrs=True):
    da_gmi = xr.where(da_gmi < 0.05, 0, da_gmi, keep_attrs=True)  
    da_gmi.attrs = attrs
    
# Set CORRA NaN and negative values to 0 
da_corra_transect = da_corra_transect.where(da_corra_transect > 0, 0)

# Set CSH NaN values to 0 
da_csh_transect = da_csh_transect.where(da_csh_transect != 0, 0)

# Infill missing data for range gates below clutter
da_corra_transect = da_corra_transect.gpm.infill_below_bin(bins=corra_last_valid_bin)

csh_last_valid_bin = np.isnan(da_csh_transect).argmin(dim="range") + 1
da_csh_transect = da_csh_transect.gpm.infill_below_bin(bins=csh_last_valid_bin)

da_gmi.gpm.plot_map()

ylim = (0, 15000)
p = da_dpr_transect.gpm.plot_cross_section(zoom=False)
p.axes.set_ylim(*ylim)
plt.show() 

p = da_dpr_transect_ka.gpm.plot_cross_section(zoom=False)
p.axes.set_ylim(*ylim)
plt.show() 

p = da_csh_transect.gpm.plot_cross_section(zoom=False, check_contiguity=False)
p.axes.set_ylim(*ylim)
plt.show() 

p = da_corra_transect.gpm.plot_cross_section(zoom=False)
p.axes.set_ylim(*ylim)
plt.show() 

da_corra_transect.gpm.plot_cross_section()
plt.plot(np.arange(ds_dpr_transect["heightZeroDeg"].size), ds_dpr_transect["heightZeroDeg"])


####--------------------------------------------------------------------------.
# %%  Create figure 
# Define figure settings
dpi = 300
figsize = (8, 6)  # Adjusted for the new layout
crs_proj = ccrs.PlateCarree()
y_lim = (0, 16_000)
y_lim = (0, 16)
legend_location = "upper left"
cartopy_linewidth = 0.2

xlim = (-91, -88.5) 
xticks = [-91, -90.5, -90, -89.5, -89, -88.5]
xticklabels = ["-91", "-90.5", "-90", "-89.5", "-89", "-88.5"]

#---------------------------------------------------------.
#### Create figure without DPR
fig = plt.figure(figsize=figsize, dpi=dpi)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       height_ratios=[1.8, 1, 1], 
                       width_ratios=[1, 1, 0.06])

# Create axes with appropriate projections
ax_dpr = fig.add_subplot(gs[0, 0], projection=crs_proj)
ax_gmi = fig.add_subplot(gs[0, 1], projection=crs_proj)
ax_corra = fig.add_subplot(gs[1, :2])  # Span both columns
ax_csh = fig.add_subplot(gs[2, :2])  # Span both columns

# Create colorbar axes
cax_precip = fig.add_subplot(gs[0:2, -1])  # Shared colorbar for first two plots
cax_lh = fig.add_subplot(gs[2, -1])    # Colorbar for bottom plot

#---------------------------------------------------------.
#### - Plot GPM DPR Precip
# - Add background
ax_dpr.coastlines(linewidth=cartopy_linewidth)
ax_dpr.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax_dpr.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax_dpr.add_feature(cartopy.feature.BORDERS, linewidth=cartopy_linewidth)

# - Add field
p1 = da_dpr.gpm.plot_map(ax=ax_dpr,
                     add_colorbar=False, 
                     add_background=False, 
                     add_gridlines=False, add_labels=False)
# - Set extent
ax_dpr.set_extent(extent)

# - Plot transect line
ds_dpr_transect.gpm.plot_transect_line(
    ax=ax_dpr, 
    color="white",
    line_kwargs={"linestyle": "-", "alpha": 1, "linewidth" : 1},
    add_direction=False,
    add_background=False,
    add_gridlines=False,
    add_labels=False, 
)


# - Add gridlines and ticklabels
gl = ax_dpr.gridlines(crs=ccrs.PlateCarree(),
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.top_labels = False  # gl.xlabels_top = False
gl.right_labels = False 
gl.left_labels = True
gl.bottom_labels = True
gl.xlocator = mticker.FixedLocator([-94, -90, -86])
gl.ylocator = mticker.FixedLocator([28, 30, 32])

# - Set title
title = da_dpr.gpm.title(add_timestep=False)
ax_dpr.set_title(title)
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_dpr.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)

#---------------------------------------------------------.
#### - Plot GPM GMI Precip
# - Add background
ax_gmi.coastlines(linewidth=cartopy_linewidth)
ax_gmi.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax_gmi.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax_gmi.add_feature(cartopy.feature.BORDERS, linewidth=cartopy_linewidth)

# - Add field
p2 = da_gmi.gpm.plot_map(ax=ax_gmi, 
                     add_gridlines=False,
                     add_labels=False, 
                     add_background=False,
                     add_colorbar=False)
# - Set extent
ax_gmi.set_extent(extent)

# - Plot transect line
ds_dpr_transect.gpm.plot_transect_line(
    ax=ax_gmi, 
    color="white",
    line_kwargs={"linestyle": "-", "alpha": 1, "linewidth" : 1},
    add_direction=False,
    add_background=False,
    add_gridlines=False,
    add_labels=False, 
)
# - Add DPR swath
ds_dpr.gpm.plot_swath_lines(
      ax=ax_gmi, color="black",
      add_background=False,
      add_gridlines=False,
      add_labels=False, 
      alpha=0.6,
      linestyle="--",
)

# - Add gridlines and ticklabels
gl = ax_gmi.gridlines(crs=ccrs.PlateCarree(),
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.top_labels = False  # gl.xlabels_top = False
gl.right_labels = False 
gl.left_labels = False
gl.bottom_labels = True
gl.xlocator = mticker.FixedLocator([-94, -90, -86])

# - Set title
title = da_gmi.gpm.title(add_timestep=False)
ax_gmi.set_title(title)
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_gmi.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)

#---------------------------------------------------------.
#### - Add shared colorbar for first two plots
plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(name="precipRateNearSurface")
cbar = pycolorbar.plot_colorbar(p1, cax=cax_precip, **cbar_kwargs)
cbar.ax.set_aspect(16)

#### - Plot GPM CORRA Precip cross-section
# - Display cross section
p3 = da_corra_transect.gpm.plot_cross_section(x="lon", y="height_km", ax=ax_corra, zoom=False, add_colorbar=False)
ax_corra.set_ylim(*y_lim)
ax_corra.set_ylabel("")

# - Set title
title = da_corra_transect.gpm.title(add_timestep=False)
ax_corra.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_corra.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
            handlelength=0, handleheight=0, handletextpad=0)

# - Drop x ticks and ticklabels 
ax_corra.xaxis.set_ticks([])
ax_corra.xaxis.set_ticklabels("")
ax_corra.set_xlabel("")

ax_corra.set_xlim(*xlim)

#---------------------------------------------------------.
#### - Plot GPM CSH Latent Heating cross-section
# - Display cross section
p_lh = da_csh_transect.gpm.plot_cross_section(x="lon", y="height_km", 
                                        vmin=-100, vmax=100,
                                        ax=ax_csh, zoom=False, 
                                        add_colorbar=False)
ax_csh.set_ylim(*y_lim)
ax_csh.set_ylabel("")
# - Set title
title = da_csh_transect.gpm.title(add_timestep=False)
ax_csh.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_csh.legend(handles=[handle], labels=[title], loc=legend_location, 
           fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)
ax_csh.set_xlim(*xlim)
ax_csh.set_xticks(xticks)
ax_csh.set_xticklabels(xticklabels)

#---------------------------------------------------------.
#### - Add colorbar for bottom plot
# fig.colorbar(p_lh, cax=cax_lh, label='Latent Heating', extendfrac=0.05, extend="both")

_, cbar_kwargs = gpm.get_plot_kwargs(name="latentHeating")
cbar_kwargs["extendfrac"] = 0.1
pycolorbar.plot_colorbar(p_lh, cax=cax_lh, **cbar_kwargs)

#### - Add label
fig.text(0.018, 0.33, "Height [km]", rotation=90, va='center')

# Adjust layout
fig.tight_layout()

# Fine-tune spacing
fig.subplots_adjust(hspace=0.025, wspace=0.025)

# Save figure
fig.savefig(os.path.join(fig_dir, "BAMS_Precip_Products.png"))
 
####--------------------------------------------------
#### Create with DPR reflectivity
figsize = (8, 8) 
fig = plt.figure(figsize=figsize, dpi=dpi)
gs = gridspec.GridSpec(4, 3, figure=fig,
                       height_ratios=[1.8, 1, 1, 1], 
                       width_ratios=[1, 1, 0.06])

# Create axes with appropriate projections
ax_dpr = fig.add_subplot(gs[0, 0], projection=crs_proj)
ax_gmi = fig.add_subplot(gs[0, 1], projection=crs_proj)
ax_corra = fig.add_subplot(gs[1, :2])  # Span both columns
ax_z = fig.add_subplot(gs[2, :2])  # Span both columns
ax_csh = fig.add_subplot(gs[3, :2])  # Span both columns

# Create colorbar axes
cax_precip = fig.add_subplot(gs[0:2, -1])  # Shared colorbar for first two plots
cax_z = fig.add_subplot(gs[2, -1])    # Colorbar for bottom plot
cax_lh = fig.add_subplot(gs[3, -1])    # Colorbar for bottom plot

#---------------------------------------------------------.
#### - Plot GPM DPR Precip
# - Add background
ax_dpr.coastlines(linewidth=cartopy_linewidth)
ax_dpr.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax_dpr.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax_dpr.add_feature(cartopy.feature.BORDERS, linewidth=cartopy_linewidth)

# - Add field
p1 = da_dpr.gpm.plot_map(ax=ax_dpr,
                     add_colorbar=False, 
                     add_background=False, 
                     add_gridlines=False, add_labels=False)
# - Set extent
ax_dpr.set_extent(extent)

# - Plot transect line
ds_dpr_transect.gpm.plot_transect_line(
    ax=ax_dpr, 
    color="white",
    line_kwargs={"linestyle": "-", "alpha": 1, "linewidth" : 1},
    add_direction=False,
    add_background=False,
    add_gridlines=False,
    add_labels=False, 
)


# - Add gridlines and ticklabels
gl = ax_dpr.gridlines(crs=ccrs.PlateCarree(),
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.top_labels = False  # gl.xlabels_top = False
gl.right_labels = False 
gl.left_labels = True
gl.bottom_labels = True
gl.xlocator = mticker.FixedLocator([-94, -90, -86])
gl.ylocator = mticker.FixedLocator([28, 30, 32])

# - Set title
title = da_dpr.gpm.title(add_timestep=False)
ax_dpr.set_title(title)
ax_dpr.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_dpr.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)

#---------------------------------------------------------.
#### - Plot GPM GMI Precip
# - Add background
ax_gmi.coastlines(linewidth=cartopy_linewidth)
ax_gmi.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax_gmi.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax_gmi.add_feature(cartopy.feature.BORDERS, linewidth=cartopy_linewidth)

# - Add field
p2 = da_gmi.gpm.plot_map(ax=ax_gmi, 
                     add_gridlines=False,
                     add_labels=False, 
                     add_background=False,
                     add_colorbar=False)
# - Set extent
ax_gmi.set_extent(extent)

# - Plot transect line
ds_dpr_transect.gpm.plot_transect_line(
    ax=ax_gmi, 
    color="white",
    line_kwargs={"linestyle": "-", "alpha": 1, "linewidth" : 1},
    add_direction=False,
    add_background=False,
    add_gridlines=False,
    add_labels=False, 
)
# - Add DPR swath
ds_dpr.gpm.plot_swath_lines(
      ax=ax_gmi, color="black",
      add_background=False,
      add_gridlines=False,
      add_labels=False, 
      alpha=0.6,
      linestyle="--",
)

# - Add gridlines and ticklabels
gl = ax_gmi.gridlines(crs=ccrs.PlateCarree(),
                  linewidth=1, color='gray', alpha=0.2, linestyle='-')
gl.top_labels = False  # gl.xlabels_top = False
gl.right_labels = False 
gl.left_labels = False
gl.bottom_labels = True
gl.xlocator = mticker.FixedLocator([-94, -90, -86])

# - Set title
title = da_gmi.gpm.title(add_timestep=False)
ax_gmi.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_gmi.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)

#---------------------------------------------------------.
#### - Add shared colorbar for first two plots
plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(name="precipRateNearSurface")
cbar = pycolorbar.plot_colorbar(p1, cax=cax_precip, **cbar_kwargs)
cbar.ax.set_aspect(16)

#### - Plot GPM DPR reflectivity cross-section
# - Display cross section
p_z = da_dpr_transect.gpm.plot_cross_section(x="lon", y="height_km", ax=ax_z, zoom=False, add_colorbar=False)
ax_z.set_ylim(*y_lim)
ax_z.set_ylabel("")

# - Set title
title = da_dpr_transect.gpm.title(add_timestep=False)
ax_z.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_z.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
            handlelength=0, handleheight=0, handletextpad=0)

# - Drop x ticks and ticklabels 
ax_z.xaxis.set_ticks([])
ax_z.xaxis.set_ticklabels("")
ax_z.set_xlabel("")
ax_z.set_xlim(*xlim)

#### - Plot GPM CORRA Precip cross-section
# - Display cross section
p3 = da_corra_transect.gpm.plot_cross_section(x="lon", y="height_km", ax=ax_corra, zoom=False, add_colorbar=False)
ax_corra.set_ylim(*y_lim)
ax_corra.set_ylabel("")

# - Set title
title = da_corra_transect.gpm.title(add_timestep=False)
ax_corra.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_corra.legend(handles=[handle], labels=[title], loc=legend_location, fancybox=True, framealpha=0.8,
            handlelength=0, handleheight=0, handletextpad=0)

# - Drop x ticks and ticklabels 
ax_corra.xaxis.set_ticks([])
ax_corra.xaxis.set_ticklabels("")
ax_corra.set_xlabel("")

ax_corra.set_xlim(*xlim)

#---------------------------------------------------------.
#### - Plot GPM CSH Latent Heating cross-section
# - Display cross section
p_lh = da_csh_transect.gpm.plot_cross_section(x="lon", y="height_km", 
                                        vmin=-100, vmax=100,
                                        ax=ax_csh, zoom=False, check_contiguity=False,
                                        add_colorbar=False)
ax_csh.set_ylim(*y_lim)
ax_csh.set_ylabel("")
# - Set title
title = da_csh_transect.gpm.title(add_timestep=False)
ax_csh.set_title("")
handle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, visible=False)
ax_csh.legend(handles=[handle], labels=[title], loc=legend_location, 
           fancybox=True, framealpha=0.8,
           handlelength=0, handleheight=0, handletextpad=0)
ax_csh.set_xlim(*xlim)
ax_csh.set_xticks(xticks)
ax_csh.set_xticklabels(xticklabels)

#---------------------------------------------------------.
#### - Add colorbar for Z plot
# fig.colorbar(p_z, cax=cax_z, label='Latent Heating', extendfrac=0.05, extend="both")

_, cbar_kwargs = gpm.get_plot_kwargs(name="zFactorFinalKu")
cbar_kwargs["label"] = "Reflectivity [dBZ]"
cbar_kwargs["extendfrac"] = 0.1
cbar_z = pycolorbar.plot_colorbar(p_z, cax=cax_z, **cbar_kwargs)
cbar_z.ax.yaxis.labelpad = 14

#---------------------------------------------------------.
#### - Add colorbar for LH plot
# fig.colorbar(p_lh, cax=cax_lh, label='Latent Heating', extendfrac=0.05, extend="both")

_, cbar_kwargs = gpm.get_plot_kwargs(name="latentHeating")
cbar_kwargs["extendfrac"] = 0.1
cbar_lh = pycolorbar.plot_colorbar(p_lh, cax=cax_lh, **cbar_kwargs)
list_ticklabeks = cbar_lh.ax.yaxis.get_ticklabels()
dir(list_ticklabeks[0])
ticklabels = [str(int(text_tick.get_position()[1])) for text_tick in cbar_lh.ax.yaxis.get_ticklabels()]
cbar_lh.ax.yaxis.set_ticklabels(ticklabels)
cbar_lh.ax.yaxis.labelpad = 5

#### - Add label
fig.text(0.018, 0.38, "Height [km]", rotation=90, va='center')

# Adjust layout
fig.tight_layout()

# Fine-tune spacing
fig.subplots_adjust(hspace=0.025, wspace=0.025)

# Save figure
fig.savefig(os.path.join(fig_dir, "BAMS_Precip_Products1.png"))

####--------------------------------------------------------------------------.

