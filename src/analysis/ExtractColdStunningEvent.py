"""
Code to plot the temperature of particles over time and days before stranding.
Note: Here, we ignore the temperature measurements at the release time, i.e. day 0
- it adds bias for particles very close to the coast due to temperature interpolation from sea to land cell.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import pandas as pd
import util
import csv
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sys
import time

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

# create all data-file
full_data = np.zeros((len(stations), n_particles, 4, 3))
full_data[:, :, :, :] = np.NAN

# 'Westenschouwen-Schouwen', 'Monster', 'DenHelder', 'Westkapelle', 'IJmuiden'
for ind in stations.index:
    s = stations['Location'][ind]
    print('Location: ', s)
    if wind == '0pWind':
        data_ds = xr.open_zarr(home_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes_120days_{1}.zarr'.format(wind, s))
    else:
        data_ds = xr.open_zarr(home_folder + 'simulations/{0}/Sum_BK_{0}_curr+stokes+wind_120days_{1}.zarr'.format(wind, s))
    
    assert data_ds.theta.shape == (10000, 121)

    zeroT_ds = data_ds.where(data_ds['theta'] == 0.)
    zero_traj, zero_obs = np.where(~np.isnan(zeroT_ds.lon.values) == True)
    assert zero_traj.size == 0 & zero_obs.size == 0
    print("asserts completed")
    # remove all beached particles (only at sea, beached=0)
    ds = data_ds.where(data_ds['beached'] == 0)
    # to not ignore day 0 temperatures
    mean_theta = ds.theta[:, :].mean(dim='trajectory', skipna=True).values

    # second plot:
    start = time.process_time()
    def filter_function(threshold_t, empty_value):
        days = np.empty((n_particles))
        days[:] = np.nan
        filter_t = np.where(ds.theta[:, :] > threshold_t)
        df = pd.DataFrame({'traj':filter_t[0],
                'days':filter_t[1]})
        grouped = df.groupby('traj').first()
        days[grouped.index] = (grouped.values).flatten()

        # if no indices were returned for a particle: it was always below the threshold, set empty value
        days[np.isnan(days)] = empty_value
        days = days - 1
        #if always above threshold, returns 0 in search, now -1 ; so now replace it with nan
        days[days == -1] = np.nan
        return days
        # if filter_t.size == 0:  # always below Tc,
        #     return empty_value  # -3: 10C, -2: 12C and -1 for 14C
        # elif filter_t[0] == 0:  # add condition: filter_10[0] != 0-> threshold is never crossed, always above Tc.
        #     return np.nan
        # else:
        #     return filter_t[0] - 1  # -1 to access the days before stranding.

 
    days_10 = filter_function(threshold_t10, -4) #stores -5
    days_12 = filter_function(threshold_t12, -3) #stores -4
    days_14 = filter_function(threshold_t14, -2)  #stores -3

    T10_t, T10_count = np.unique(days_10[~np.isnan(days_10)], return_counts=True)
    T12_t, T12_count = np.unique(days_12[~np.isnan(days_12)], return_counts=True)
    T14_t, T14_count = np.unique(days_14[~np.isnan(days_14)], return_counts=True)

    T10_min, T10_max, T12_min, T12_max, T14_min, T14_max = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if T10_t.size > 0: T10_min, T10_max = np.nanmin(T10_t), np.nanmax(T10_t)
    if T12_t.size > 0: T12_min, T12_max = np.nanmin(T12_t), np.nanmax(T12_t)
    if T14_t.size > 0: T14_min, T14_max = np.nanmin(T14_t), np.nanmax(T14_t)
    print('min/max T 10 C: ', T10_min, T10_max)
    print('min/max T 12 C: ', T12_min, T12_max)
    print('min/max T 14 C: ', T14_min, T14_max)
    print("days for threshold computed in ",time.process_time() - start)

    # region: Figure to plot the location of stranding event
    start2 = time.process_time()
    d10_ind = np.argwhere((~np.isnan(days_10)) & (days_10 >= 0)).flatten()
    d12_ind = np.argwhere((~np.isnan(days_12)) & (days_12 >= 0)).flatten()
    d14_ind = np.argwhere((~np.isnan(days_14)) & (days_14 >= 0)).flatten()

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
    print("distance for threshold crossing points computed in ",time.process_time() - start2)

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
    ax[0].set_ylim(7, 21)
    ax[0].scatter(ds.time[:, :], ds.theta[:, :], c='black', alpha=0.1, s=0.3)
    max_time = np.nanmax(ds.time[:, :].values) #always available
    ax[0].set_xlim(max_time - np.timedelta64(days, 'D'), max_time)
    ax[0].plot(data_ds.time[0, :], mean_theta, c='magenta', label='mean temperature')
    ax[0].set_xlabel('Time', fontsize=custom_size)
    ax[0].tick_params('x', labelrotation=60, labelsize=custom_size)
    ax[0].tick_params(axis='y', labelsize=custom_size)
    ax[0].set_ylabel('Temperature (°C)', fontsize=custom_size)
    ax[0].set_xlabel('Time', fontsize=custom_size)
    ax[0].legend(loc='upper right', prop={'size': custom_size})

    ax[1].bar(T10_t, T10_count, color='blue', label='10°C')
    ax[1].bar(T14_t, T14_count, color='tomato', label='14°C')
    ax[1].bar(T12_t, T12_count, color='orange', label='12°C', alpha=0.7)
    handles = [Patch(facecolor='b', edgecolor='none'),
            Patch(facecolor='orange', edgecolor='none'),
            Patch(facecolor='tomato', edgecolor='none')]
    labels = ['10°C', '12°C', '14°C']
    if (T10_t.size > 0 and T10_t[0] < 0) or (T12_t.size > 0 and T12_t[0] < 0) or (T14_t.size > 0 and T14_t[0] < 0):
        ax[1].fill_between([-6,-2],[10000,10000], facecolor="none", hatch="//", edgecolor="grey", linewidth=0)
        handles.append(Patch(facecolor='white',hatch='//'))
        labels.append("Always below")
        
    ax[1].legend(handles, labels,loc='upper right', prop={'size': custom_size})
    ax[1].set_xlabel('Days before stranding', fontsize=custom_size)
    ax[1].set_ylabel('Number of particles crossing threshold temperatures', fontsize=custom_size)
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.001, 10000)
    ax[1].set_xlim(-5, 75)
    ax[1].tick_params(axis='both', labelsize=custom_size)
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
    ax2.set_title('Locations where temperature crosses threshold temperatures: {0}'.format(s))
    colormap = colors.ListedColormap(['white', 'lightgrey'])

    latmin=0
    latmax=375
    lonmin=0
    lonmax=297
    ax2.pcolormesh(fieldMesh_x[latmin:latmax+1,lonmin:lonmax+1], fieldMesh_y[latmin:latmax+1,lonmin:lonmax+1], true_lmask[latmin:latmax,lonmin:lonmax],cmap=colormap)

    ax2.add_feature(cfeature.COASTLINE, edgecolor='gray')
    # ax2.coastlines(resolution='50m')
    ax2.scatter(lons_d10, lats_d10, c='b', s=4, label='10°C')
    ax2.scatter(lons_d14, lats_d14, c='tomato', s=4, label='14°C', alpha=0.7)
    ax2.scatter(lons_d12, lats_d12, c='orange', s=4, label='12°C', alpha=0.7)
    ax2.scatter(stations['Longitude'][ind], stations['Latitude'][ind], c='black', marker='x', s=200,
                label='Stranding Location')

    ax2.set_xlim(-5,6)
    ax2.set_ylim(48,55)

    custom_lines = [Patch(facecolor='b', edgecolor='none', label='10°C'),
                    Patch(facecolor='orange', edgecolor='none', label='12°C'),
                    Patch(facecolor='tomato', edgecolor='none', label='14°C'),
                    Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Stranding location')]
    ax2.legend(handles=custom_lines, loc='upper left', borderpad=0.8, prop={'size': custom_size})
    plt.savefig(home_folder + 'outputs/{0}/Locations_{0}_{1}.jpeg'.format(wind, s),
                    bbox_inches='tight',
                    pad_inches=0.2)
    # plt.show()
    print("plotting completed. Total time in ", time.process_time() - start)

    # endregion

np.save(home_folder + 'outputs/full_data_{0}.npy'.format(wind), full_data)