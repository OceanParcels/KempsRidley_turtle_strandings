import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pandas as pd
from matplotlib.lines import Line2D
from copy import copy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# find nearest coastal cell to defined beaching location, to release in water,
# remove corner points by using dx,dy
def nearestcoastcell(fieldMesh_x, fieldMesh_y, coastMask, lon, lat):
    dist = np.sqrt((fieldMesh_x - lon) ** 2 * coastMask + (fieldMesh_y - lat) ** 2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))

    dx, dy = 0.001, 0.001  # 0.068 km and 0.111 km at 52°N latitude, respectively

    startlon_release = fieldMesh_x[coords]
    if startlon_release > lon:  # to the left
        startlon_release = startlon_release - dx
        endlon_release = fieldMesh_x[coords[0], coords[1] - 1] + dx
    else:
        startlon_release = startlon_release + dx
        endlon_release = fieldMesh_x[coords[0], coords[1] + 1] - dx

    startlat_release = fieldMesh_y[coords]
    if startlat_release > lat:  # below cell
        startlat_release = startlat_release - dy
        endlat_release = fieldMesh_y[coords[0] - 1, coords[1]] + dy
    else:
        startlat_release = startlat_release + dy
        endlat_release = fieldMesh_y[coords[0] + 1, coords[1]] - dy
    return startlon_release, endlon_release, startlat_release, endlat_release, coords


home_dir = '/Users/dmanral/Desktop/Analysis/Ridley/'
ds_cur = xr.load_dataset(home_dir + 'data/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')

lons = ds_cur.longitude
lats = ds_cur.latitude
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)
coastMask = np.loadtxt(home_dir + 'data/coastalMask_297x_375y')

file_land = home_dir + 'data/landMask_297x_375y'
landMask = np.genfromtxt(file_land, delimiter=None)

lmask = np.genfromtxt(home_dir + 'data/true_landMask_296x_374y', delimiter=None)
color_land = copy(plt.get_cmap('Reds'))(0)
color_ocean = copy(plt.get_cmap('Reds'))(128)

fig = plt.figure(figsize=(16, 10), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
gl.xlines = False
gl.ylines = False
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14, 'color': 'k'}
gl.ylabel_style = {'size': 14, 'color': 'k'}
colormap = clr.ListedColormap(['skyblue', 'white'])
plt.title('Released particles for each stranding location', fontsize=20, pad=20)
# ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS, edgecolor='gray')

# ax.pcolormesh(fieldMesh_x[150:251, 200:226], fieldMesh_y[150:251, 200:226], lmask[150:250, 200:225], cmap=colormap)
# ax.scatter(fieldMesh_x, fieldMesh_y, c=landMask, s=20,
#            cmap='Reds_r', vmin=-0.05, vmax=0.05, edgecolors='k')

colors = np.array(('r', 'b', 'g', 'gold', 'pink'))
stations = pd.read_csv(home_dir + 'Locations_NL.csv')
np_sqrt = 100

for index, st in stations.iterrows():
    strand_lon, strand_lat = st['Longitude'], st['Latitude']
    startlon_release, endlon_release, startlat_release, endlat_release, coords = nearestcoastcell(fieldMesh_x,
                                                                                                  fieldMesh_y,
                                                                                                  coastMask,
                                                                                                  strand_lon,
                                                                                                  strand_lat)

    # 10x10 particles -> 100 particles homogeneously spread over grid cell
    re_lons = np.linspace(startlon_release, endlon_release, np_sqrt)
    re_lats = np.linspace(startlat_release, endlat_release, np_sqrt)
    fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)

    plt.scatter(fieldMesh_x_re, fieldMesh_y_re, c='r', alpha=0.5, s=0.5)
    plt.scatter(strand_lon, strand_lat, c='black', marker='x', s=200)
    ax.text(strand_lon + 0.1, strand_lat - 0.05, st['Location'],
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor='none'),
            fontsize=15)
ax.tick_params(axis='both', labelsize=13)
# ax.set_xlim(3.2, 5)
# ax.set_ylim(50, 55)
ax.set_xlim(1, 6)
ax.set_ylim(50, 55)

custom_lines = [Line2D([], [], color='black', marker='x', linestyle='None', markersize=12, label='Stranding location'),
                Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Released particles')]
ax.legend(handles=custom_lines, bbox_to_anchor=(.01, .93), loc='center left', borderaxespad=0.,
          framealpha=1, prop={'size': 14})

# custom_lines = [Line2D([0], [0], c=color_ocean, marker='o', markersize=10, markeredgecolor='k', lw=0),
#                 Line2D([0], [0], c=color_land, marker='o', markersize=10, markeredgecolor='k', lw=0)]
# ax.legend(custom_lines, ['ocean point', 'land point'], bbox_to_anchor=(.01, .93), loc='center left', borderaxespad=0.,
#           framealpha=1)

plt.savefig(home_dir + 'Plots/MapReleaseCells.jpeg',
            bbox_inches='tight',
            pad_inches=0.2)
# plt.show()
