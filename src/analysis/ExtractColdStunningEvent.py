import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

home_folder = '/Users/dmanral/Desktop/Analysis/Ridley/'
# 'WesterschouwenSchouwen', 'Monster', 'DenHelder', 'Westkapelle', 'IJmuiden'
# for s in np.array(('Westkapelle', 'WesterschouwenSchouwen', 'Monster', 'DenHelder', 'IJmuiden')):
s = 'IJmuiden'
file = 'Sum_BK_NoWind_Beaching_curr+stokes_120days_{0}'.format(s)
data_ds = xr.open_dataset(home_folder + 'Simulations/{0}.nc'.format(file))

# remove all zero temperature- beached particles
ds = data_ds.where(data_ds['theta'] != 0.)
mean_theta = ds.theta[:, 1:].mean(dim='traj', skipna=True).values

threshold_t10 = 10
threshold_t12 = 12
threshold_t14 = 14  # degree Celcius

n_particles = 10000
days = 100

fig, ax = plt.subplots(ncols=2, nrows=1,
                       dpi=300, figsize=(16, 7))
fig.suptitle(file)
ax[0].axhline(threshold_t10, c='b', linestyle='--', label='10°C')
ax[0].axhline(threshold_t12, c='orange', linestyle='--', label='12°C')
ax[0].axhline(threshold_t14, c='red', linestyle='--', label='14°C')
ax[0].set_xlim(ds.time[1, 0].values - np.timedelta64(days, 'D'), ds.time[1, 0])

ax[0].scatter(ds.time[:, 1:], ds.theta[:, 1:], c='black', alpha=0.1, s=0.3)
ax[0].plot(ds.time[0, 1:], mean_theta, c='magenta', label='mean temperature')
ax[0].set_xlabel('Time', fontsize=12)
ax[0].tick_params('x', labelrotation=45)
ax[0].set_ylabel('Temperature (°C)', fontsize=12)
ax[0].legend(prop={'size': 12})
# second plot
days_10 = np.empty((n_particles))
days_10[:] = np.nan
days_12 = np.empty((n_particles))
days_12[:] = np.nan
days_14 = np.empty((n_particles))
days_14[:] = np.nan

for i in range(n_particles):
    filter_10 = np.where(ds.theta[i, 1:] > threshold_t10)[0]
    if filter_10.size > 0:
        days_10[i] = filter_10[0]
    filter_12 = np.where(ds.theta[i, 1:] > threshold_t12)[0]
    if filter_12.size > 0:
        days_12[i] = filter_12[0]
    filter_14 = np.where(ds.theta[i, 1:] > threshold_t14)[0]
    if filter_14.size > 0:
        days_14[i] = filter_14[0]

bins = np.linspace(0, 100, 51)
ax[1].hist(days_10, bins=bins, color='b', label='10°C')
ax[1].hist(days_12, bins=bins, color='orange', label='12°C', alpha=0.7)
ax[1].hist(days_14, bins=bins, color='r', label='14°C', alpha=0.7)
ax[1].legend(prop={'size': 12})
ax[1].set_xlabel('Days before threshold temperature', fontsize=12)
ax[1].set_ylabel('Number of particles', fontsize=12)
# for i in range(10000):
#     plt.plot(ds.time[i, 1:], ds.theta[i, 1:], c='royalblue')
# plt.show()
plt.savefig(home_folder + 'Plots/{0}_.jpeg'.format(file))
