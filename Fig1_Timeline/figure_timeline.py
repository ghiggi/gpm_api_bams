import os
import gpm
import datetime
import numpy as np
from gpm.io.products import get_product_start_time, get_info_dict
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.text as mtext
from matplotlib import gridspec
from matplotlib.markers import CARETRIGHT  # TICKLEFT, TICKRIGHT

# %% Define function to retrieve mpl date as function of date strings
def get_mdate(date_string): 
    date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    mdate = matplotlib.dates.date2num(date)
    return mdate

# %% Figure settings
np.seterr(all='ignore') # raise/ignore divisions by 0 and nans
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True

# Transparent and white 
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

# Define figure directory
fig_dir = "/home/ghiggi/GPM_FIGURES/"
os.makedirs(fig_dir, exist_ok=True)

# %% Define datasets characteristics 

current_year = 2025

radar_sensor_names = gpm.available_sensors(product_categories="RADAR")

pmw_sensor_names = gpm.available_satellites(product_categories="PMW", prefix_with_sensor=True)

####-------------------------------------------------------------------------------------------------------------------.
#### Define satellite orbit type 
# - sun_synchronous:
#   --> - two overpasses: when ascending (traveling from south to north) and once descending.
#   --> - Sun-synchronous orbits are often described by their equatorial crossing times
#   --> - cross the equator at the same local solar time each day (but drift over time unlead controlled orbits !)
# - non_sun_synchrnous: 
#   - overpass at different times 
#   - near-equatorial orbits
#   --> for diurnal cycle studies ! 
# --> https://www.remss.com/support/crossing-times/ 

# GMI, TMI, SAPHIR are the only non-sun_synchronous 
# AMSRE, AMSR2, SSMI-F08, METOP-A, METOP-B, METOP-C controlled orbits (crossing local time does not vary)
# Others sensors, the equatorial_crossing_time varies over time ! 

# Sun Synchronous Orbit
# - drifting
# - controlled 
# - "-" # for GMI/TMI/SAPHIR ! 
pmw_drifting_list = ['AMSUB-NOAA15','AMSUB-NOAA16', 'AMSUB-NOAA17',
                     'MHS-NOAA18', 'MHS-NOAA19', 
                     'SSMI-F10', 'SSMI-F11', 'SSMI-F13', 'SSMI-F14', 'SSMI-F15',
                     'SSMIS-F16', 'SSMIS-F17', 'SSMIS-F18', 'SSMIS-F19']
pmw_controlled_list = ['AMSR2-GCOMW1','AMSRE-AQUA',
                       'ATMS-NPP', 'ATMS-NOAA20', 'ATMS-NOAA21', 
                       'SSMI-F08', 
                       'MHS-METOPA', 'MHS-METOPB', 'MHS-METOPC']
others_list = ["DPR-GPM", "PR-TRMM", "GMI-GPM", "TMI-TRMM", "SAPHIR-MT1"]
orbit_type_dict = {} 
for sensor in pmw_drifting_list:
    orbit_type_dict[sensor] = "drifting"
for sensor in pmw_controlled_list:
    orbit_type_dict[sensor] = "controlled"
for sensor in others_list:
    orbit_type_dict[sensor] = "-"

####-------------------------------------------------------------------------------------------------------------------.
#### Define satellite coverage dictionary 

def get_product_temporal_coverage(product): 
    start_time = get_product_start_time(product)
    end_time = get_info_dict()[product]["end_time"]
    start_year = round(start_time.year + start_time.month/12, 2)
    if end_time is not None:
        end_year = round(end_time.year + end_time.month/12, 2)
    else: 
        end_year = None
    return start_year, end_year

pmw_coverage_dict = {}
for sensor in pmw_sensor_names:
    if "GPM" in sensor:  
        product = "2A-GMI-CLIM"
    elif "TRMM" in sensor: 
        product = "2A-TMI-CLIM"
    else:
        product = f"2A-{sensor}-CLIM"
    pmw_coverage_dict[sensor] = get_product_temporal_coverage(product)
    
radar_coverage_dict = {}
for sensor, satellite in [("DPR","GPM"), ("PR", "TRMM")]:
    name = f"{sensor}-{satellite}"
    product = f"2A-{sensor}"
    radar_coverage_dict[name] = get_product_temporal_coverage(product)

coverage_dict = {} 
coverage_dict.update(pmw_coverage_dict)
coverage_dict.update(radar_coverage_dict)

####-------------------------------------------------------------------------------------------------------------------.
#### Define satellite current status 
ongoing_dict = {} 
for sensor, (start_time, end_time) in coverage_dict.items(): 
    if end_time is None:
        ongoing_dict[sensor] = True 
        coverage_dict[sensor] = (start_time, current_year)
    else: 
        ongoing_dict[sensor] = False 

####-------------------------------------------------------------------------------------------------------------------.
#### Define sensor type (conically-scanning PMW imagers, cross-track PMW sounders, radar) 
# - conically scanning window-channel radiometers  
# - cross-track scanning water vapor sounding radiometers # SAPHIR / ATNS / MHS
pmw_sounders_list = ['AMSUB', 'ATMS', 'MHS', 'SAPHIR']
pmw_imagers_list = ['AMSR2','AMSRE','GMI','TMI', 'SSMI','SSMIS']
radar_list = ["DPR", "PR"]
sensor_type_dict = {} 
for sensor in pmw_sounders_list:
    sensor_type_dict[sensor] = "pmw_sounder"
for sensor in pmw_imagers_list:
    sensor_type_dict[sensor] = "pmw_imager"
for sensor in radar_list:
    sensor_type_dict[sensor] = "radar"


# %% Define sensor timeline characteristics 
info_dict = {}
for name in coverage_dict.keys(): 
    sensor, satellite = name.split("-")
    info_dict[name] = {}
    info_dict[name]["sensor"] = sensor
    info_dict[name]["satellite"] = satellite
    info_dict[name]["start_time"] = coverage_dict[name][0]
    info_dict[name]["end_time"] = coverage_dict[name][1]
    info_dict[name]["ongoing"] = ongoing_dict[name]
    info_dict[name]["sensor_type"] = sensor_type_dict[sensor]
    info_dict[name]["orbit_type"] = orbit_type_dict[name]

# Sort dictionary 
names = list(info_dict.keys())
start_times = [sensor_dict["start_time"] for sensor, sensor_dict in info_dict.items()]
combined = list(zip(start_times, names))
combined.sort()
 
info_dict = {name: info_dict[name] for start_time, name in combined}
  
# %% Define sensor_type colors 
cmap = plt.get_cmap('Paired')
colorpalette = cmap(np.arange(0,12))
                    
sensor_type_color_dict = {}
sensor_type_color_dict['radar'] = colorpalette[7]  # orange
sensor_type_color_dict['pmw_imager'] = colorpalette[1]   # light blue 
sensor_type_color_dict['pmw_sounder'] = colorpalette[0]     # dark blue   

sensor_type_label_dict = {} 
sensor_type_label_dict['radar'] = "Radar"
sensor_type_label_dict['pmw_imager'] = "PMW conically scanning imager"
sensor_type_label_dict['pmw_sounder'] = "PMW cross-track scanning sounder"   

orbit_type_hatch_dict = {}
orbit_type_hatch_dict['controlled'] = '/'
orbit_type_hatch_dict['drifting'] = None
orbit_type_hatch_dict['-'] = '||' # 'xx' 

orbit_type_label_dict = {}
orbit_type_label_dict['-'] = "Near-equatorial"
orbit_type_label_dict['controlled'] = "Stable sun-synchronous "
orbit_type_label_dict['drifting'] = "Drifting sun-synchronous "

#%% Create the figure

n_sensors = len(info_dict)

# Setting for timeline
start_year = min([sensor_dict["start_time"] for sensor, sensor_dict in info_dict.items()])
end_year =  max([sensor_dict["end_time"] for sensor, sensor_dict in info_dict.items()])

x_left_margin = 0.5
x_right_margin = 1
y_margin = 0.375
timeline_bar_height = 0.5
dx_lag_arrow = 0.5
dx_lag_text = 0.5
xlim = (start_year-x_left_margin, end_year+x_right_margin)
ylim = [y_margin, n_sensors + timeline_bar_height + y_margin]

# Create figure 
fig = plt.figure(dpi=300)
plt.gcf().set_size_inches([18.21,  9])

# Set graphic layout 
gs = gridspec.GridSpec(1, 4, width_ratios=[0.67,0.06,0.07,0.16])
gs.update(hspace=0, wspace=0)

# Set background colors
ax = fig.add_subplot(gs[0,0])
idx_bar = 0     
       
#----------------------------------------------------------------------------------------. 
#### - Add background color
n_before = 5 
color1 = 'w'
color2 = '0.9'
transparency = 0.7
ax.axhspan(ylim[0], ylim[1], facecolor=color1, alpha=transparency)

#----------------------------------------------------------------------------------------. 
#### - Add time lines 
for i, name in enumerate(list(info_dict.keys())):
    # Define current timeline index 
    idx_bar = idx_bar + 1

    # Add temporal coverage bars
    start_time = info_dict[name]["start_time"]
    end_time = info_dict[name]["end_time"]
    hatch=orbit_type_hatch_dict[info_dict[name]["orbit_type"]]
    ax.barh(y=idx_bar , 
            width=end_time - start_time,  
            left=start_time,
            height=timeline_bar_height,
            color=sensor_type_color_dict[info_dict[name]["sensor_type"]], 
            align='center',
            edgecolor='k', 
            hatch=hatch,
            alpha=transparency, 
            linewidth=1)
        
    # Add sensor name 
    if i == 0: 
        x_text_position = end_time + dx_lag_text
        ha = "left"
    else: 
        x_text_position = start_time - dx_lag_text
        ha = "right"
  
    ax.text(x=x_text_position,
            y=idx_bar, 
            s=name, 
            color='k',
            ha=ha,        # horizontal alignment
            va='center',  # vertical alignement
    ) 
    
    ## Add right arrow (for currently operational datasets)
    if info_dict[name]["ongoing"]:
        # If beyond lifetime (8 years), set satellite text in italic 
        lifetime = current_year - start_time
        if lifetime >= 8:
            color = "lightgray"
        else: 
            color = "black"
        x_arrow_position = current_year + dx_lag_arrow
        plt.plot(x_arrow_position,
                 idx_bar,
                 marker = CARETRIGHT,
                 markersize = 10,
                 color = color, 
                 markeredgecolor = 'k')
   
#-------------------------------------------------------------------------.
#### - Add legend patches 

# - Define sensor_type legend (colored patches)
sensor_type_handles = []
sensor_type_handles.append('Sensor')
for sensor_type, label in sensor_type_label_dict.items():
    patch = mpatches.Patch(color = sensor_type_color_dict[sensor_type], 
                           alpha = transparency, 
                           label = label)
    sensor_type_handles.append(patch)

# - Define orbit legend (hatched patches)
orbit_handles = [] 
orbit_handles.append('Orbit')
for orbit_type, label in orbit_type_label_dict.items():
    hatch = orbit_type_hatch_dict[orbit_type]
    if hatch is not None: 
        hatch = hatch*2
    patch = mpatches.Patch(
        facecolor="white",
        edgecolor = "black", 
        # alpha = 0.2,
        hatch = hatch,
        label = label)
    orbit_handles.append(patch)

# - Add operation right arrow legend 
aux_handles = []
aux_handles.append('Auxiliary information')     
operational_arrow = mlines.Line2D(
    [], [], color='k',
    marker=CARETRIGHT, 
    linestyle='None',
    markersize=10, 
    label='Operational',
)
aux_handles.append(operational_arrow) 
operational_beyond_lifetime_arrow = mlines.Line2D(
    [], [], color='lightgray',
    marker=CARETRIGHT, 
    linestyle='None',
    markersize=10, 
    label='Operational (beyond expected end-of-life)')
aux_handles.append(operational_beyond_lifetime_arrow)

legend_handles = sensor_type_handles + orbit_handles + aux_handles

# - Define class for sub-headers
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, weight="bold", **self.text_props)
        handlebox.add_artist(title)
        return title
    
# - Plot legend 
handles = legend_handles 
labels = [handle.get_label() if not isinstance(handle, str) else '' for handle in legend_handles]
plt.legend(handles, labels,
           loc = "lower left", 
           handler_map={str: LegendTitle()})

#-------------------------------------------------------------------------.
#### - Specify timeline axis settings
# Y-axis 
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylim(ylim)
ax.invert_yaxis()

# X-axis 
ax.set_xlim(xlim)


# %% Figure final settings 
plt.gcf().set_tight_layout(True)
 

# %% Save figure 
plt.savefig(os.path.join(fig_dir, 'gpm_sensors_timeline.jpg'), dpi=500)
