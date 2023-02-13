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

home_folder = '/Users/dmanral/Desktop/Analysis/Ridley/'

threshold_t10 = 10
threshold_t12 = 12
threshold_t14 = 14  # degree Celcius

n_particles = 10000
days = 120

figure_dpi = 300
stations = pd.read_csv(home_folder + 'Locations_NL.csv', index_col=1)
# summary file fields
with open(home_folder + 'summary_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Location", "Min_time_10", "Max_time_10", "Mean_dis_10",
                     "Min_time_12", "Max_time_12", "Mean_dis_12",
                     "Min_time_14", "Max_time_14", "Mean_dis_14"])

# 'Westerschouwen-Schouwen', 'Monster', 'DenHelder', 'Westkapelle', 'IJmuiden'
for ind in stations.index:
    s = stations['Location'][ind]
    # s = 'Westerschouwen-Schouwen'

    file = 'Sum_BK_NoWind_curr+stokes_120days_{0}'.format(s)
    data_ds = xr.open_dataset(home_folder + 'Simulations/{0}.nc'.format(file))

    # remove all zero temperature and beached particles (only at sea, beached=0)
    ds = data_ds.where((data_ds['theta'] != 0.) & (data_ds['beached'] == 0))
    mean_theta = ds.theta[:, 1:].mean(dim='traj', skipna=True).values

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
    # print('min max days for 10C: ', min(days_10), max(days_10))
    # print('min max days for 12C: ', min(days_12), max(days_12))
    # print('min max days for 14C: ', min(days_14), max(days_14))

    T10_t, T10_count = np.unique(days_10, return_counts=True)
    T12_t, T12_count = np.unique(days_12, return_counts=True)
    T14_t, T14_count = np.unique(days_14, return_counts=True)

    print('min/max T 10 C: ', min(T10_t), max(T10_t))
    print('min/max T 12 C: ', min(T12_t), max(T12_t))
    print('min/max T 14 C: ', min(T14_t), max(T14_t))

    # region: Figure to plot the location of stranding event
    d_10 = days_10[~np.isnan(days_10)].astype(int)
    d_12 = days_12[~np.isnan(days_12)].astype(int)
    d_14 = days_14[~np.isnan(days_14)].astype(int)

    lats_d10 = [ds.lat[i, j].values for i, j in zip(range(n_particles), d_10)]
    lons_d10 = [ds.lon[i, j].values for i, j in zip(range(n_particles), d_10)]
    lats_d12 = [ds.lat[i, j].values for i, j in zip(range(n_particles), d_12)]
    lons_d12 = [ds.lon[i, j].values for i, j in zip(range(n_particles), d_12)]
    lats_d14 = [ds.lat[i, j].values for i, j in zip(range(n_particles), d_14)]
    lons_d14 = [ds.lon[i, j].values for i, j in zip(range(n_particles), d_14)]

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

    with open(home_folder + 'summary_file.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([s, min(T10_t), max(T10_t), np.around(np.nanmean(dis_10t), 1),
                         min(T12_t), max(T12_t), np.around(np.nanmean(dis_12t), 1),
                         min(T14_t), max(T14_t), np.around(np.nanmean(dis_14t), 1)])

    custom_size = 15
    fig, ax = plt.subplots(ncols=2, nrows=1,
                           dpi=figure_dpi, figsize=(16, 8))
    fig.suptitle(file)
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
    plt.savefig(home_folder + 'Simulations/Plots/{0}_log.jpeg'.format(file),
                bbox_inches='tight',
                pad_inches=0.2)
    # plt.show()
    print("First plot completed")

    # we need to coordinates file to access the corner points - glamf/gphif
    model_mask_file = home_folder + 'data/landMask_297x_375y'
    landMask = np.genfromtxt(model_mask_file, delimiter=None)
    true_lmask = np.genfromtxt(home_folder + 'data/true_landMask_296x_374y', delimiter=None)
    U_ds = xr.open_dataset(home_folder + 'data/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')
    lons = U_ds.longitude
    lats = U_ds.latitude
    fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)

    custom_size = 11

    figure2 = plt.figure(figsize=(12, 8), dpi=figure_dpi)
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': custom_size, 'color': 'k'}
    gl.ylabel_style = {'size': custom_size, 'color': 'k'}
    # ax2.coastlines(resolution='50m')
    ax2.set_title('Locations where temperature crosses threshold temperatures: {0}'.format(s))
    colormap = colors.ListedColormap(['white', 'gainsboro'])
    # ax2 = plt.axes()
    ax2.pcolormesh(fieldMesh_x[130:201, 135:231], fieldMesh_y[130:201, 135:231], true_lmask[130:200, 135:230],
                   cmap=colormap)

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

    plt.savefig(home_folder + 'Simulations/Plots/ColdStunningLocations_{0}.jpeg'.format(s),
                bbox_inches='tight',
                pad_inches=0.2)
    # plt.show()

    # endregion
