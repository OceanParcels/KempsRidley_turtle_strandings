import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import colors, colorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import sys

args = sys.argv
assert len(args) == 2

wind = args[1]
print(wind)
min_lon, max_lon = -15, 8
min_lat, max_lat = 42, 57

base_folder = '/nethome/manra003/analysis/KempRidley/'

# we need to coordinates file to access the corner points - glamf/gphif
U_ds = xr.open_dataset('/storage/shared/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')
lons = U_ds.longitude
lats = U_ds.latitude
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)
lmask = np.genfromtxt('/nethome/manra003/KempsRidley_turtle_strandings/data/landMask_297x_375y', delimiter=None)

stations = pd.read_csv('/nethome/manra003/KempsRidley_turtle_strandings/data/Locations_NL.csv')

for index, station in stations.iterrows():
# index=1
    s = station['Location']#[index]
    if wind == '0pWind':
        ds = xr.open_zarr(base_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes_120days_{1}.zarr'.format(wind, s))
    else:
        ds = xr.open_zarr(base_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes+wind_120days_{1}.zarr'.format(wind, s))

    fig, [ax, cax] = plt.subplots(1, 2, gridspec_kw={"width_ratios": [50, 1]}, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    colormap = colors.ListedColormap(['white', 'gainsboro'])
    gl = ax.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.pcolormesh(fieldMesh_x, fieldMesh_y, lmask, cmap=colormap)
    # ax.pcolormesh(fieldMesh_x[100:221, 145:241], fieldMesh_y[100:221, 145:241], true_lmask[100:220, 145:240],
    #                cmap=colormap)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # region Animation
    output_dt = timedelta(days=1)
    time_range = np.arange(np.nanmax(ds['time'].values),
                           np.nanmin(ds['time'].values) - np.timedelta64(output_dt),
                           -output_dt)
    print('Time_range: ', len(time_range))

    # release locations
    time_id = np.where(ds['time'] == time_range[0])

    theta1 = ds['theta']

    # temp_cmp = cmocean.cm.balance
    temp_cmp = plt.cm.coolwarm
    norm = colors.Normalize(vmin=5, vmax=20)

    cb1 = colorbar.ColorbarBase(cax, cmap=temp_cmp,
                                norm=norm,
                                orientation='vertical', label='Temperature (°C)')
    scatter = ax.scatter(ds['lon'].values[time_id], ds['lat'].values[time_id], s=1)
    strand_lon, strand_lat = station['Longitude'], station['Latitude']
    plt.scatter(strand_lon, strand_lat, c='black', marker='x', s=200)
    ax.text(strand_lon + 0.3, strand_lat - 0.1, s,
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor='none'),
            fontsize=8)
    # d=270
    # ax.scatter(ds.lon[:, d], ds.lat[:, d], c=ds.theta[:, d], cmap=temp_cmp, s=0.5)
    # plt.show()

    t = np.datetime_as_string(time_range[0], unit='m')
    title = ax.set_title('Particles at time = ' + t)


    def animate(i):
        t = np.datetime_as_string(time_range[i], unit='m')

        time_id = np.where(ds['time'] == time_range[i])
        title.set_text(s + ': Backward simulation at time = ' + t)

        scatter.set_offsets(np.c_[ds['lon'].values[time_id], ds['lat'].values[time_id]])
        scatter.set_color(temp_cmp(norm(theta1.values[time_id])))


    size = len(time_range)
    anim = FuncAnimation(fig, animate, frames=size, interval=200)
    anim.save(base_folder + 'outputs/{0}/{0}_{1}.mp4'.format(wind, s))
    # endregion

    print('animation saved')
