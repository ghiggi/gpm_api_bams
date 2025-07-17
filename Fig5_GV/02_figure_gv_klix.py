import os
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import gpm
import gpm.gv # xradar_dev accessor
import geopandas as gpd
import glob
import radar_api
import pycolorbar
import matplotlib.gridspec as gridspec
from functools import reduce
from gpm.visualization.eda import plot_boxplot
from gpm.gv.plots import plot_gdf_map
from pycolorbar.norm import CategorizeNorm
from radar_api.download import define_local_filepath

####-----------------------------------------------------------------.
#### Define directory paths
# Define GPM GV directory
gv_base_dir = "/home/ghiggi/data/GPM_GV"
gv_base_dir = "/t5500/ltenas8/data/GPM_GV"

# Define figure directory
fig_dir = "/t5500/ltenas8/data/tmp/GPM_Figures"
os.makedirs(fig_dir, exist_ok=True)

####-----------------------------------------------------------------.
#### Define Matplotlib settings
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["legend.fontsize"] = 8

# Transparent and white 
# matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

####-----------------------------------------------------------------.
#### Define GV matching database 
# Define network and radar
network = "NEXRAD"
radar = "KLIX"

# List GPM-GV match ups
gv_volume_dir = os.path.join(gv_base_dir, "Volumes", network, radar)
filepaths = glob.glob(os.path.join(gv_volume_dir, "*.parquet"))

####-----------------------------------------------------------.
#### Define filtering criteria
# Keep only beams where aggregated gate reflectivities above the respective instrument sensitivities
# - GR_fraction_above_sensitivity
# - SR_fraction_above_sensitivity

# Select only stratiform precipitation outside the melting layer 
# --> Flag for stratiform, flag for bright band, above ML, below ML 
# --> Keep only stratiforms, discard bright band and where maybe bright band
# --> Keep only above (and below eventually) the melting layer 
# --> Discard convective and other ... 

# Select volume-averaged SR and GR moderate reflectivities values that are largely
#   unaffected by low SR sensitivity or attenuation of the SR beam
# --> Warren et al., 2018:  between 24 and 36 dBZ
 
# Remove GR samples contaminated by ground clutter, anomalous propagation, or beam blockage
# Remove footprints with poor volume matching 
# Remove footprints with nonuniform beam filling (SR/GR)

def filter_matched_volumes(gdf_match, sr_z_range, gr_z_range):
    # Define mask
    masks = [
        # Select SR scan with "normal" dataQuality (for entire cross-track scan)
        gdf_match["SR_dataQuality"] == 0,
       
        # Select SR beams with detected precipitation
        gdf_match["SR_flagPrecip"] > 0, 
        
        # Select only 'high quality' SR data
        # - qualityFlag == 1 indicates low quality retrievals
        # - qualityFlag == 2 indicates bad/missing retrievals
        gdf_match["SR_qualityFlag"] == 0,
                   
        # Select only beams with confident precipitation type
        gdf_match["SR_qualityTypePrecip"] == 1,
           
        # Select only stratiform precipitation
        # - SR_flagPrecipitationType == 2 indicates convective
        gdf_match["SR_flagPrecipitationType"] == 1,
    
        # Select only SR beams with reliable attenuation correction 
        gdf_match["SR_reliabFlag"].isin((1,2)), # or == 1 
    
    
        # Select only beams with reduced path attenuation 
        # gdf_match["SR_zFactorCorrection_Ku_max"]
        # gdf_match["SR_piaFinal"]
        # gdf_match["SR_pathAtten"]
      
        # Select only SR gates with no clutter 
        # - Removes a lot of points !
        gdf_match["SR_fraction_clutter"] < 0.1,
    
        # Select only SR gates not in the melting layer 
        gdf_match["SR_fraction_melting_layer"] == 0,
    
        # Select only SR gates with precipitation
        gdf_match["SR_fraction_no_precip"] < 0.1,
    
        # Select only SR gates with no hail 
        gdf_match["SR_fraction_hail"] == 0,
     
        # Select only SR gates with rain 
        # gdf_match["SR_fraction_rain"] == 1,
        
        # Select only SR gates with snow 
        # gdf_match["SR_fraction_snow"] == 1,
                            
        # Select SR beams only within given GR radius interval 
        # - Crisologo et al., 2018, Warren et al., 2018
        gdf_match["GR_range_min"] > 15_000,
        gdf_match["GR_range_max"] < 115_000,
    
        # Discard SR beams where scanning time difference > 5 minutes
        # - time_difference is in seconds !
        gdf_match["time_difference"] < 60*5,  
     
        # Discard SR beams where GR gates does not cover 80% of the horizontal area 
        gdf_match["GR_fraction_covered_area"] > 0.8,
    
        # # Filter footprints where volume ratio exceeds 60 
        # gdf_match["VolumeRatio"] > 60,
                        
        # Select only SR beams with detected bright band
        # - This can remove lot of matched volumes !
        # - We can just discard gates in the BB
        # - Schwaller et al., 2011: only stratiform rain above brightband
        # gdf_match["SR_qualityBB"] == 1,
        
        # Select only SR gates with snow 
        # gdf_match["SR_fraction_above_isotherm"] == 1,
        
        # Select only interval of reflectivities 
        # - Crisologo et al., 2018, Warren et al., 2018:  between 24 and 36 dBZ
        # - Schwaller et al., 2011: SR above 18 dBZ, GR: 15 dBZ (-3 dBZ error allowance)
        # --> Iterative filtering based on bias-corrected reflectivity (Protat et al., 2011)
        
        # gdf_match["SR_zFactorFinal_Ku_mean"] > 18, 
        # gdf_match["GR_Z_mean"] > 15,
        
        gdf_match["SR_zFactorFinal_Ku_mean"] > sr_z_range[0], 
        gdf_match["SR_zFactorFinal_Ku_mean"] < sr_z_range[1],
        gdf_match["GR_Z_mean"] > gr_z_range[0],
        gdf_match["GR_Z_mean"] < gr_z_range[1],
        
        # Select SR gates above minimum reflectivity
        # - 0.7 in Crisologo et al., 2018 and Warren et al., 2018
        # - 0.95 in Schwaller et al., 2011
        # gdf_match["SR_zFactorFinal_Ku_fraction_above_12dBZ"] > 0.95,
        
        # Select SR gates with GR above minimum reflectivity
        # - 0.7 in Crisologo et al., 2018
        # gdf_match["GR_Z_fraction_above_12dBZ"] > 0.95,
        
        # Discard SR beams with high NUBF 
        # gdf_match["SR_zFactorFinal_Ku_std"]  
        # gdf_match["SR_zFactorFinal_Ku_range"] < 10, 
        # gdf_match["SR_zFactorFinal_Ku_range"] < 5,
        # gdf_match["SR_zFactorFinal_Ku_cov"] < 0.5,
        # gdf_match["GR_Z_std"]
        # gdf_match["GR_Z_cov"] < 0.5,
        # gdf_match["GR_Z_range"] < 15,
    ]

    # Define final mask
    mask_final = reduce(np.logical_and, masks)
    gdf_match["filtering_mask"] = mask_final
    gdf_filtered = gdf_match[mask_final]
    return gdf_filtered

####-----------------------------------------------------------------.
#### Compute calibration statistics 
radar_band = "S"
sr_z_column = f"SR_zFactorFinal_{radar_band}_mean"
gr_z_column = "GR_Z_mean"
max_iterations = 10
minimum_number_of_samples = 50 # 150
sr_z_range_start = (24, 36)
sr_z_range_stop = (24, 36)
gr_z_range_start = (18, 40)
gr_z_range_stop = (24, 36)

# filepath = filepaths[2]
list_calibration_stats = []
for filepath in filepaths:
    ####-----------------------------------------------------------------.
    #### Load GPM-GV matched volume 
    gdf_match = gpd.read_parquet(filepath)
    timestep = str(gdf_match["SR_time"].iloc[0])
    n_matches = len(gdf_match)
    print(timestep)
    
    # Apply iterative filtering 
    # - Iterative calculation of bias is required when thresholding the reflectivity to
    #   account for the fact that, given a nonzero calibration,
    #   samples will be incorrectly included/excluded from the calculation of bias
    # - We recompute the bias iteratively
    sr_z_ranges = list(zip(np.linspace(sr_z_range_start[0], sr_z_range_stop[0], num=max_iterations), 
                           np.linspace(sr_z_range_start[1], sr_z_range_stop[1], num=max_iterations)))
    gr_z_ranges = list(zip(np.linspace(gr_z_range_start[0], gr_z_range_stop[0], num=max_iterations), 
                           np.linspace(gr_z_range_start[1], gr_z_range_stop[1], num=max_iterations)))
    sr_z_ranges.append(sr_z_range_stop)
    gr_z_ranges.append(gr_z_range_stop)
    offset_mean = 0
    offset_median = 0
    unsufficient_number_of_samples = False
    list_offset = []
    for i in range(0, max_iterations):
        df = gdf_match.copy()
        df[gr_z_column] = df[gr_z_column] - offset_median
        df = filter_matched_volumes(df,
                                    sr_z_range=sr_z_ranges[i], 
                                    gr_z_range=gr_z_ranges[i])
        if len(df) < minimum_number_of_samples:
            unsufficient_number_of_samples = True
            break
        
        delta_z = df[gr_z_column] - df[sr_z_column]
        offset_mean += np.nanmean(delta_z).round(2)
        offset_median += np.nanmedian(delta_z).round(2)
        list_offset.append(offset_median.round(2))
        
    # Compute other calibration statistics
    if not unsufficient_number_of_samples:
        print(f"Offset evolution from {list_offset[0]} to {list_offset[-1]} dBZ")
        bias = offset_mean
        robbias = offset_median
        delta_z_unbiased = delta_z + offset_median
        std = np.nanstd(delta_z_unbiased).round(2)
        mad = np.median(np.absolute(delta_z_unbiased - robbias)).round(2)
        quantiles = np.nanquantile(delta_z_unbiased, q=[0.1, 0.25, 0.75, 0.9]).round(2)
        q10, q25, q75, q90 = quantiles
        time = df["SR_time"].iloc[0].to_numpy()
        dict_bias_stats = {
            "mean": bias,
            "median": robbias,
            "std": std,
            "mad": mad,
            "q10": q10,
            "q25": q25,
            "q75": q75,
            "q90": q90,
            "iqr": q75 - q25,
            "min": delta_z_unbiased.min(),
            "max": delta_z_unbiased.max(),
            "n": len(delta_z_unbiased),        
        }
        df_stats = pd.DataFrame(dict_bias_stats, index=[time])
        list_calibration_stats.append(df_stats)
  
        # Report final number of matches
        n_matches_selected = len(df)
        n_matches_discarded = n_matches - n_matches_selected
        n_matches_selected_percentage = round(n_matches_selected/n_matches*100, 1)
        n_matches_discarded_percentage = round(n_matches_discarded/n_matches*100, 1)
        print(f"Number of matches selected: {n_matches_selected} ({n_matches_selected_percentage} %)")
        print(f"Number of matches discarded: {n_matches_discarded} ({n_matches_discarded_percentage} %)")


print(f"Number of selected overpass: {len(list_calibration_stats)}/{len(filepaths)}")

####--------------------------------------------------------------------------.
#### Define the relative calibration statistics table
df_absolute_calibration = pd.concat(list_calibration_stats)
df_absolute_calibration = df_absolute_calibration.sort_index()
 
# Compute relative calibration 
df_relative_calibration  = df_absolute_calibration.copy()
df_relative_calibration["time"] = df_relative_calibration.index 
offset = np.nanmedian(df_relative_calibration["median"])
for var in ["mean", "median", "q10", "q25", "q75", "q90", "min", "max"]: 
    df_relative_calibration[var] = df_relative_calibration[var] - offset

plt.scatter(df_relative_calibration["median"],
            df_relative_calibration["n"],
            s=1)
 
####-----------------------------------------------------------.
#### Display boxplot timeseries (at irregular timestep)
medianprops = dict(marker="s", markersize=3, color='#1f77b4')
medianprops = None
median_points_kwargs = {"marker":"o", "s": 5}
median_line_kwargs = {"linewidth": 1.5}

fig, ax = plt.subplots( 1,1, figsize=(8, 2), dpi=300)
bplot = plot_boxplot(df_relative_calibration, 
                  ax=ax, 
                  widths=0, 
                  # widths=0.6*15,
                  # medianprops=medianprops,
                  showfliers=False, 
                  showwhisker=False, 
                  showcaps=False,
                  add_median_points=True,
                  median_points_kwargs=median_points_kwargs,
                  add_median_line=False,
                  median_line_kwargs=median_line_kwargs,
                  patch_artist=True,
                  )
ax.set_ylim(-4, 4)
ax.axhline(0, color="black", alpha=0.8, linestyle="-", linewidth=0.5)
ax.axhline(-2, color="black", alpha=0.5, linestyle="--", linewidth=0.5)
ax.axhline(2, color="black", alpha=0.5, linestyle="--", linewidth=0.5)
ax.set_ylabel("Calibration Offset [dBZ]")

# Optionally, format the date labels
_ = ax.set_xticklabels(ax.get_xticklabels() , rotation=90)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))   

# Color boxes by number of colors 
boundaries = [-1, 50, 100, 200, 500, 10_000]
labels = ["<50", "50-100", "100-200", "200-500", ">500"]
norm = CategorizeNorm(boundaries=boundaries, labels=labels)
cmap = pycolorbar.get_cmap("Blues", len(labels), interval=(0.3, 1))
colors = cmap(norm(df_relative_calibration["n"]))
 
for patch, color in zip(bplot['boxes'], colors):
    patch.set_color(color)
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
 

#### Display boxplot (disregard irregular timestep)
median_line_kwargs = {"linewidth": 1.5}
fig, ax = plt.subplots( 1,1, figsize=(10, 5), dpi=300)
bplot = plot_boxplot(df_relative_calibration.reset_index(), 
                  ax=ax, 
                  showfliers=False, 
                  showwhisker=False, 
                  showcaps=False,
                  add_median_line=True, 
                  median_line_kwargs=median_line_kwargs,
                  widths=0.6)
ax.set_ylim(-4, 4)
ax.axhline(0, color="black", alpha=0.8, linestyle="-", linewidth=0.5)
ax.axhline(-2, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)
ax.axhline(2, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)
ax.set_ylabel("Calibration Offset [dBZ]")
# Subsample x axis 
current_ticklabels = ax.get_xticklabels() 
subsampled_ticks = [current_ticklabels[int(i)].get_position()[0] for i in  np.linspace(0, len(current_ticklabels)-1, 10)]
subsampled_ticklabels = [current_ticklabels[int(i)].get_text() for i in  np.linspace(0, len(current_ticklabels)-1, 10)]
ax.set_xticks(subsampled_ticks, labels=subsampled_ticklabels, rotation=90)


####--------------------------------------------------------------------------.
#### Plot KLIX sweeps 
# Define sweep to plot 
sweep = "sweep_2"

# Load SR-GV matched database
filename = "GPM_GV_2A-DPR_NEXRAD_KLIX_202108291513.parquet"
gv_filepath = os.path.join(gv_volume_dir, filename)
gdf_match = gpd.read_parquet(gv_filepath)

# Select sweep SR-GR data
gdf_sweep = gdf_match[gdf_match["sweep_group"] == sweep]
nexrad_filenames = np.unique(gdf_sweep["filename"])
nexrad_filename = nexrad_filenames[0]
if len(nexrad_filenames) > 1: 
    gdf_sweep = gdf_sweep[gdf_sweep["filename"] == nexrad_filename]

# Load GR data 
gr_filepath = define_local_filepath(filename=nexrad_filename,
                                    network=network, 
                                    radar=radar)
ds_gr = radar_api.open_dataset(gr_filepath, network=network, sweep=sweep)
ds_gr = ds_gr.xradar.georeference() # add x and y coordinates

print("Elevation:", ds_gr["elevation"].min().item())
print("Range:", np.diff(ds_gr["range"].data).min())

# Time offset
ds_gr["time"].values.min()
ds_gr["time"].values.max()

gdf_sweep["SR_time"].min()
gdf_sweep["SR_time"].max()
gdf_sweep["GR_time"].min()
gdf_sweep["GR_time"].max()

gdf_sweep["time_difference"].min()  # 18 s
gdf_sweep["time_difference"].max()  # 72

# Define Cartopy projection
ccrs_gr_aeqd = ccrs.AzimuthalEquidistant(central_longitude=ds_gr["longitude"].item(), 
                                         central_latitude=ds_gr["latitude"].item())
subplot_kwargs = {}
subplot_kwargs["projection"] = ccrs_gr_aeqd

# Define geographic extent
extent_xy = gdf_sweep.total_bounds[[0, 2, 1, 3]]

# Retrieve plot kwargs 
plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs("zFactorFinal", 
                                               user_plot_kwargs={"vmin": 15, "vmax":45})

# Define plot decorations
def add_decorations(ax, ds_gr, radar_size):
    # - Add radar location
    ax.scatter(0, 0, c="black", marker="X", s=radar_size)
    ax.scatter(0, 0, c="black", marker="X", s=radar_size)
    
    ds_gr.xradar_dev.plot_range_distance(
        distance=15_000,
        ax=ax,
        add_background=False,
        add_gridlines=False,
        add_labels=False,
        linestyle="dashed",
        linewidth=0.8,
        edgecolor="black",
    )
    ds_gr.xradar_dev.plot_range_distance(
        distance=100_000,
        ax=ax,
        add_background=False,
        add_gridlines=False,
        add_labels=False,
        linestyle="dashed",
        linewidth=0.8,
        edgecolor="black",
    )
    ds_gr.xradar_dev.plot_range_distance(
        distance=150_000,
        ax=ax,
        add_background=False,
        add_gridlines=False,
        add_labels=False,
        linestyle="dashed",
        linewidth=0.8,
        edgecolor="black",
    )

# Define figure settings
figsize = (8, 4)
dpi = 300

# Define radar size 
radar_size = 40

# Create the figure
fig, axes = plt.subplots(1, 3, 
                         width_ratios=[1, 1, 1.1],
                         subplot_kw=subplot_kwargs, 
                         figsize=figsize, dpi=dpi)

#### Plot SR data
axes[0].coastlines()
_ = plot_gdf_map(
    ax=axes[0],
    gdf=gdf_sweep,
    column=sr_z_column,
    title="GPM DPR Matched",
    extent_xy=extent_xy,
    # Gridline settings
    # grid_linewidth=grid_linewidth,
    # grid_color=grid_color,
    # Colorbar settings
    add_colorbar=False,
    # Plot settings
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)
add_decorations(ax=axes[0], ds_gr=ds_gr, radar_size=radar_size)

#### - Plot GR matched data
axes[1].coastlines()

_ = plot_gdf_map(
    ax=axes[1],
    gdf=gdf_sweep,
    column=gr_z_column,
    title=f"{network} {radar} Matched",
    extent_xy=extent_xy,
    # Gridline settings
    # grid_linewidth=grid_linewidth,
    # grid_color=grid_color,
    # Colorbar settings
    add_colorbar=False,
    # Plot settings
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)
add_decorations(ax=axes[1], ds_gr=ds_gr, radar_size=radar_size)

#### - Plot GR sweep data 
axes[2].coastlines()

p = ds_gr["DBZH"].where(ds_gr["DBZH"] > 0).xradar_dev.plot_map(
    ax=axes[2], 
    x="x", y="y",
    add_background=False,
    add_gridlines=False, 
    add_labels=False, 
    add_colorbar=True, 
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
    )
p.axes.set_xlim(extent_xy[0:2])
p.axes.set_ylim(extent_xy[2:4])
p.axes.set_title(f"{network} {radar} PPI")
add_decorations(ax=axes[2], ds_gr=ds_gr, radar_size=radar_size)


####--------------------------------------------------------------------------.
#### Create figure 
# Define figure settings
figsize = (8, 4)
dpi = 300
crs_proj = ccrs_gr_aeqd
cartopy_linewidth = 0.4

#---------------------------------------------------------.
#### - Create figure with complex GridSpec layout
fig = plt.figure(figsize=figsize, dpi=dpi)
gs = gridspec.GridSpec(2, 4, figure=fig,
                       height_ratios=[3, 1],   # First row taller, second row shorter
                       width_ratios=[1, 1, 1, 0.2]) # # Last column narrower for colorbar


# Create axes with appropriate projections
ax1 = fig.add_subplot(gs[0, 0], projection=crs_proj)
ax2 = fig.add_subplot(gs[0, 1], projection=crs_proj)
ax3 = fig.add_subplot(gs[0, 2], projection=crs_proj)
boxplot_ax = fig.add_subplot(gs[1, 0:3])  # Span both columns
cbar_ax = fig.add_subplot(gs[0, 3])  # Shared colorbar for first rows maps
legend_ax = fig.add_subplot(gs[1, 3]) 

#### - Plot GR PPI
ax1.coastlines(linewidth=cartopy_linewidth)
p1 = ds_gr["DBZH"].where(ds_gr["DBZH"] > 0).xradar_dev.plot_map(
    ax=ax1, 
    x="x", y="y",
    add_background=False,
    add_gridlines=False, 
    add_labels=False, 
    add_colorbar=False, 
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)
p1.axes.set_xlim(extent_xy[0:2])
p1.axes.set_ylim(extent_xy[2:4])
p1.axes.set_title(f"{network} {radar} PPI")
add_decorations(ax=ax1, ds_gr=ds_gr, radar_size=radar_size)

#### - Plot GR Matched
ax2.coastlines(linewidth=cartopy_linewidth)
p2 = plot_gdf_map(
    ax=ax2,
    gdf=gdf_sweep,
    column=gr_z_column,
    title=f"{network} {radar} Matched",
    extent_xy=extent_xy,
    add_colorbar=False,
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)
add_decorations(ax=ax2, ds_gr=ds_gr, radar_size=radar_size)

#### - Plot SR Matched
ax3.coastlines(linewidth=cartopy_linewidth)
p3 = plot_gdf_map(
    ax=ax3,
    gdf=gdf_sweep,
    column=sr_z_column,
    title="GPM DPR Matched",
    extent_xy=extent_xy,
    add_colorbar=False,
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)
add_decorations(ax=ax3, ds_gr=ds_gr, radar_size=radar_size)

#### - Add map colorbar  
cbar = pycolorbar.plot_colorbar(p1, cax=cbar_ax, **cbar_kwargs)
cbar.ax.set_aspect(0.6)

#### - Plot boxplot
# Define colors for boxplots 
boundaries = [49, 100, 200, 500, 10_000]
labels = ["50-100", "100-200", "200-500", ">500"]
norm = CategorizeNorm(boundaries=boundaries, labels=labels)
cmap = pycolorbar.get_cmap("Blues", len(labels), interval=(0.3, 1))
colors = cmap(norm(df_relative_calibration["n"]))

median_points_kwargs = {"marker":"o", "s": 1, "c": colors}
 
bplot = plot_boxplot(df_relative_calibration, 
                  ax=boxplot_ax, 
                  widths=0, 
                  showfliers=False, 
                  showwhisker=False, 
                  showcaps=False,
                  add_median_points=True,
                  median_points_kwargs=median_points_kwargs,
                  patch_artist=True,
                  )

x_min, x_max = boxplot_ax.get_xlim() 
boxplot_ax.set_xlim(x_min-10,x_max+10)

boxplot_ax.set_ylim(-6, 6)
boxplot_ax.axhline(0, color="black", alpha=0.8, linestyle="-", linewidth=0.5)
boxplot_ax.axhline(-2, color="black", alpha=0.5, linestyle="--", linewidth=0.5)
boxplot_ax.axhline(2, color="black", alpha=0.5, linestyle="--", linewidth=0.5)
boxplot_ax.set_ylabel("Offset [dBZ]",  fontsize='x-small')

# Optionally, format the date labels
# Format date labels
boxplot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
boxplot_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 6, 10]))   
_ = boxplot_ax.set_xticklabels(boxplot_ax.get_xticklabels(), rotation=90)

# Color boxes by number of colors 
for patch, color in zip(bplot['boxes'], colors):
    patch.set_color(color)
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
 
#### - Add boxplot legend 
# Create legend handles and labels
legend_handles = [plt.Line2D([0], [0], color=color, lw=1) for color in cmap(np.arange(0, len(labels)))]
legend_labels = labels

# Add legend to legend_ax 
legend_ax.legend(legend_handles, legend_labels, 
                 loc='center',
                 bbox_to_anchor=(1, 0.4),
                 title='Footprints',
                 frameon=False,
                 fancybox=False,
                 handlelength=0.5, 
                 fontsize='x-small', ncol=1, title_fontsize='x-small')
legend_ax.axis('off')

# Adjust layout
fig.tight_layout()

# Fine-tune spacing
fig.subplots_adjust(hspace=0.025, wspace=0.025)

# Save figure
fig.savefig(os.path.join(fig_dir, "BAMS_GV_KLIX.png"))
 
####--------------------------------------------------------------------------.


 
 
 
 



 
