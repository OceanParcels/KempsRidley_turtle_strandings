"""
Code to plot the temperature of particles over time and days before stranding.
Note: Here, we ignore the temperature measurements at the release time, i.e. day 0
- it adds bias for particles very close to the coast due to temperature interpolation from sea to land cell.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib import colors
import pandas as pd
import util
import csv
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
import sys

args = sys.argv
assert len(args) == 2

wind = args[1]
print(wind)

home_folder = '/nethome/manra003/analysis/KempRidley/'
data_folder = '/nethome/manra003/KempsRidley_turtle_strandings/data/'

threshold_t10 = 10
threshold_t12 = 12
threshold_t14 = 14  # degree Celcius

n_particles = 10000
days = 120

figure_dpi = 300
stations = pd.read_csv(data_folder + 'Locations_NL.csv')

 # we need to coordinates file to access the corner points - glamf/gphif
true_lmask = np.genfromtxt(data_folder + 'true_landMask_296x_374y', delimiter=None)
U_ds = xr.open_dataset('/storage/shared/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')
lons = U_ds.longitude
lats = U_ds.latitude
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)

# summary file fields
with open(home_folder + 'outputs/summary_file_{0}.csv'.format(wind), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Location", "Min_time_10", "Max_time_10", "Mean_dis_10",
                     "Min_time_12", "Max_time_12", "Mean_dis_12",
                     "Min_time_14", "Max_time_14", "Mean_dis_14"])

#create all data-file 
full_data = np.zeros((len(stations),n_particles, 4, 3))
full_data [:,:,:,:] = np.NAN

# 'Westerschouwen-Schouwen', 'Monster', 'DenHelder', 'Westkapelle', 'IJmuiden'
for ind in stations.index:
    s = stations['Location'][ind]
    if wind == '0pWind':
        data_ds = xr.open_zarr(home_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes_120days_{1}.zarr'.format(wind, s))
    else:
        data_ds = xr.open_zarr(home_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes+wind_120days_{1}.zarr'.format(wind, s))
    
    assert data_ds.theta.shape==(10000,121) 
    
    # remove all zero temperature and beached particles (only at sea, beached=0)
    ds = data_ds.where((data_ds['theta'] != 0.) & (data_ds['beached'] == 0))
    mean_theta = ds.theta[:, 1:].mean(dim='trajectory', skipna=True).values

    # second plot:
    days_10 = np.empty((n_particles))
    days_10[:] = np.nan
    days_12 = np.empty((n_particles))
    days_12[:] = np.nan
    days_14 = np.empty((n_particles))
    days_14[:] = np.nan

    for i in range(n_particles):
        filter_10 = np.where(ds.theta[i, 1:] > threshold_t10)[0]
        if filter_10.size > 0 and filter_10[0] != 0:  # add condition: filter_10[0] != 0-> threshold is never crossed.
            # plus 1: since we are ignoring day 0 and np.where returns results from day 1 onwards
            days_10[i] = filter_10[0]
        filter_12 = np.where(ds.theta[i, 1:] > threshold_t12)[0]
        if filter_12.size > 0 and filter_12[0] != 0:
            days_12[i] = filter_12[0]
        filter_14 = np.where(ds.theta[i, 1:] > threshold_t14)[0]
        if filter_14.size > 0 and filter_14[0] != 0:
            days_14[i] = filter_14[0]

    print('Location: ', s)
    T10_t, T10_count = np.unique(days_10[~np.isnan(days_10)], return_counts=True)
    T12_t, T12_count = np.unique(days_12[~np.isnan(days_12)], return_counts=True)
    T14_t, T14_count = np.unique(days_14[~np.isnan(days_14)], return_counts=True)

    T10_min,T10_max,T12_min,T12_max,T14_min,T14_max=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    if T10_t.size > 0: T10_min, T10_max = np.nanmin(T10_t), np.nanmax(T10_t)
    if T12_t.size > 0: T12_min, T12_max = np.nanmin(T12_t), np.nanmax(T12_t)
    if T14_t.size > 0: T14_min, T14_max = np.nanmin(T14_t), np.nanmax(T14_t)
    print('min/max T 10 C: ', T10_min, T10_max)
    print('min/max T 12 C: ', T12_min, T12_max)
    print('min/max T 14 C: ', T14_min, T14_max)

    # region: Figure to plot the location of stranding event
    d10_ind = np.argwhere(~np.isnan(days_10)).flatten()
    d12_ind = np.argwhere(~np.isnan(days_12)).flatten()
    d14_ind = np.argwhere(~np.isnan(days_14)).flatten()

    lats_d10, lons_d10 = [(ds.lat[i, int(days_10[i])].values) for i in d10_ind], [(ds.lon[i, int(days_10[i])].values) for i in d10_ind]
    lats_d12, lons_d12 = [(ds.lat[i, int(days_12[i])].values) for i in d12_ind], [(ds.lon[i, int(days_12[i])].values) for i in d12_ind]
    lats_d14, lons_d14 = [(ds.lat[i, int(days_14[i])].values) for i in d14_ind], [(ds.lon[i, int(days_14[i])].values) for i in d14_ind]

    # compute average distance from stranding location
    dis_10t = [util.dist_pairs_km(x, y, stations['Longitude'][ind], stations['Latitude'][ind]) for x, y in
               zip(lons_d10, lats_d10)]
    dis_12t = [util.dist_pairs_km(x, y, stations['Longitude'][ind], stations['Latitude'][ind]) for x, y in
               zip(lons_d12, lats_d12)]
    dis_14t = [util.dist_pairs_km(x, y, stations['Longitude'][ind], stations['Latitude'][ind]) for x, y in
               zip(lons_d14, lats_d14)]
    
    print('mean distance 10 C: ', np.nanmean(dis_10t))
    print('mean distance 12 C: ', np.nanmean(dis_12t))
    print('mean distance 14 C: ', np.nanmean(dis_14t))

    def fill_data(st, tc, lats, lons, days, dist, indices):
        if len(lats) > 0 : 
            full_data[st, indices, 0, tc] = lats
            full_data[st, indices, 1, tc] = lons
            full_data[st, indices, 2, tc] = days[indices]
            full_data[st, indices, 3, tc] = dist
    fill_data(ind, 0, lats_d10, lons_d10, days_10, dis_10t, d10_ind)
    fill_data(ind, 1, lats_d12, lons_d12, days_12, dis_12t, d12_ind)
    fill_data(ind, 2, lats_d14, lons_d14, days_14, dis_14t, d14_ind)
        
    with open(home_folder + 'outputs/summary_file_{0}.csv'.format(wind), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([s, T10_min, T10_max, np.around(np.nanmean(dis_10t), 1),
                         T12_min, T12_max, np.around(np.nanmean(dis_12t), 1),
                         T14_min, T14_max, np.around(np.nanmean(dis_14t), 1)])

    custom_size = 18
    fig, ax = plt.subplots(ncols=2, nrows=1,
                           dpi=figure_dpi, figsize=(16, 8))
    fig.suptitle('Time_{0}_{1}_log'.format(wind, s))
    ax[0].axhline(threshold_t10, c='b', linestyle='--', label='10°C')
    ax[0].axhline(threshold_t12, c='orange', linestyle='--', label='12°C')
    ax[0].axhline(threshold_t14, c='red', linestyle='--', label='14°C')
    # ax[0].set_xlim(ds.time[1, 0].values - np.timedelta64(days, 'D'), ds.time[1, 0])
    ax[0].set_xlim(np.nanmin(ds.time[:, 1:].values), np.nanmax(ds.time[:, 1:].values))
    ax[0].set_ylim(7, 21)

    ax[0].scatter(ds.time[:, 1:], ds.theta[:, 1:], c='black', alpha=0.1, s=0.3)
    ax[0].plot(data_ds.time[0, 1:], mean_theta, c='magenta', label='mean temperature')
    ax[0].set_xlabel('Time', fontsize=custom_size)
    ax[0].tick_params('x', labelrotation=45, labelsize=custom_size)
    ax[0].tick_params(axis='y', labelsize=custom_size)
    ax[0].set_ylabel('Temperature (°C)', fontsize=custom_size)
    ax[0].set_xlabel('Time', fontsize=custom_size)
    ax[0].legend(loc='upper right', prop={'size': custom_size})

    ax[1].bar(T10_t, T10_count, color='b', label='10°C')
    ax[1].bar(T14_t, T14_count, color='tomato', label='14°C')
    ax[1].bar(T12_t, T12_count, color='orange', label='12°C', alpha=0.7)

    # reordering the labels
    handles, labels = ax[1].get_legend_handles_labels()

    # specify order
    order = [0, 2, 1]

    # pass handle & labels lists along with order as below
    ax[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', prop={'size': custom_size})

    # ax[1].legend(prop={'size': 12})
    ax[1].set_xlabel('Days before stranding', fontsize=custom_size)
    ax[1].set_ylabel('Number of particles crossing threshold temperatures', fontsize=custom_size)
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.001, 10000)
    ax[1].set_xlim(-1, 75)
    ax[1].tick_params(axis='both', labelsize=custom_size)
    # for i in range(10000):
    #     plt.plot(ds.time[i, 1:], ds.theta[i, 1:], c='royalblue')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(home_folder + 'outputs/{0}/Time_{0}_{1}_log.jpeg'.format(wind, s),
                bbox_inches='tight',
                pad_inches=0.2)
    # plt.show()
    print("First plot completed")

    custom_size = 15

    figure2 = plt.figure(figsize=(12, 8), dpi=figure_dpi)
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': custom_size, 'color': 'k'}
    gl.ylabel_style = {'size': custom_size, 'color': 'k'}
    ax2.add_feature(cfeature.LAND)

    # ax.coastlines(resolution='50m')
    # ax2.set_xlim(min_lon, max_lon)
    # ax2.set_ylim(min_lat, max_lat)
    ax2.set_title('Locations where temperature crosses threshold temperatures: {0}'.format(s))
    colormap = colors.ListedColormap(['white', 'lightgrey'])
    # ax2.pcolormesh(fieldMesh_x[130:201, 135:231], fieldMesh_y[130:201, 135:231], true_lmask[130:200, 135:230],
    #                cmap=colormap)
    ax2.pcolormesh(fieldMesh_x[130:221, 145:241], fieldMesh_y[130:221, 145:241], true_lmask[130:220, 145:240],
                   cmap=colormap)

    ax2.add_feature(cfeature.COASTLINE, edgecolor='gray')
    # ax2.add_feature(cfeature.BORDERS, edgecolor='lightgray')
    # ax2.add_feature(cfeature.LAND, color='lightgray')
    ax2.scatter(lons_d10, lats_d10, c='b', s=4, label='10°C')
    ax2.scatter(lons_d14, lats_d14, c='tomato', s=4, label='14°C', alpha=0.7)
    ax2.scatter(lons_d12, lats_d12, c='orange', s=4, label='12°C', alpha=0.7)
    ax2.scatter(stations['Longitude'][ind], stations['Latitude'][ind], c='black', marker='x', s=200,
                label='Stranding Location')

    custom_lines = [
        Line2D([], [], color='b', lw=8, label='10°C'),
        Line2D([], [], color='orange', lw=8, label='12°C'),
        Line2D([], [], color='tomato', lw=8, label='14°C'),
        Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Stranding location')]
    ax2.legend(handles=custom_lines, loc='upper left', borderpad=0.8, prop={'size': custom_size})

    plt.savefig(home_folder + 'outputs/{0}/Locations_{0}_{1}.jpeg'.format(wind, s),
                bbox_inches='tight',
                pad_inches=0.2)
    # plt.show()

    # endregion

np.save(home_folder + 'outputs/full_data_{0}.npy'.format(wind), full_data)