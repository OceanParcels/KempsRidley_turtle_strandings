import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pandas as pd


# find nearest coastal cell to defined beaching location, to release in water,
# remove corner points by using dx,dy
# unfortunately have to handle release locations for Monster beach specially- as the point lies on land
# better way exists- but shortage of time.
def nearestcoastcell(fieldMesh_x, fieldMesh_y, coastMask, lon, lat, monster_flag):
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
    if startlat_release > lat and monster_flag:  # below cell
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

fig = plt.figure(figsize=(16, 8), dpi=300)
ax = plt.axes()
colormap = clr.ListedColormap(['skyblue', 'white'])
plt.title('Released particles for each stranding location', fontsize=20, pad=20)
# ax.pcolormesh(fieldMesh_x, fieldMesh_y, landMask, cmap=colormap)
ax.pcolormesh(fieldMesh_x[150:251, 175:226], fieldMesh_y[150:251, 175:226], landMask[150:250, 175:225], cmap=colormap,
              shading='flat')
# ax.pcolormesh(fieldMesh_x[150:250, 200:225], fieldMesh_y[150:250, 200:225], landMask[150:250, 200:225], cmap=colormap)
# ax.pcolormesh(fieldMesh_x[150:251, 200:226], fieldMesh_y[150:251, 200:226], coastMask[150:250, 200:225], cmap=colormap,
#               shading='flat')
plt.scatter(fieldMesh_x, fieldMesh_y, s=0.2, c='black')

colors = np.array(('r', 'b', 'g', 'gold', 'pink'))
stations = pd.read_csv(home_dir + 'Locations_NL.csv')
np_sqrt = 100

for index, st in stations.iterrows():
    strand_lon, strand_lat = st['Longitude'], st['Latitude']
    if st['Location'] == 'Monster':
        monster_flag = False
    else:
        monster_flag = True
    startlon_release, endlon_release, startlat_release, endlat_release, coords = nearestcoastcell(fieldMesh_x,
                                                                                                  fieldMesh_y,
                                                                                                  coastMask,
                                                                                                  strand_lon,
                                                                                                  strand_lat,
                                                                                                  monster_flag)

    # 10x10 particles -> 100 particles homogeneously spread over grid cell
    re_lons = np.linspace(startlon_release, endlon_release, np_sqrt)
    re_lats = np.linspace(startlat_release, endlat_release, np_sqrt)
    fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)

    plt.scatter(fieldMesh_x_re, fieldMesh_y_re, c='r', alpha=0.5, s=0.5)
    plt.scatter(strand_lon, strand_lat, c='black', marker='x', s=200)
    ax.text(strand_lon, strand_lat - 0.3, st['Location'],
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor='none'),
            fontsize=15)

ax.set_xlim(3.2, 5)
ax.set_ylim(50, 55)
plt.savefig(home_dir + 'Plots/NewReleaseCells.jpeg')
# plt.show()
