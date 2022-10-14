import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import colors, colorbar
import cmocean

home_folder = '/Users/dmanral/Desktop/Analysis/Ridley/'

# we need to coordinates file to access the corner points - glamf/gphif
model_mask_file = home_folder + 'data/landMask_297x_375y'

landMask = np.genfromtxt(model_mask_file, delimiter=None)

file = 'Sum_BK_NoWind_Beaching_curr+stokes_120days_DenHelder'
ds = xr.open_dataset(home_folder + 'Simulations/{0}.nc'.format(file))
print(ds)

U_ds = xr.open_dataset(home_folder + 'data/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')
lons = U_ds.longitude
lats = U_ds.latitude
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)

fig, [ax, cax] = plt.subplots(1, 2, gridspec_kw={"width_ratios": [50, 1]})
colormap = colors.ListedColormap(['white', 'gainsboro'])

# ax.pcolormesh(fieldMesh_x, fieldMesh_y, landMask, cmap=colormap,
#               shading='auto')
ax.pcolormesh(fieldMesh_x[100:251, 100:226], fieldMesh_y[100:251, 100:226], landMask[100:250, 100:225], cmap=colormap,
              shading='auto')
# ax.pcolormesh(fieldMesh_x[150:251,175:226], fieldMesh_y[150:251,175:226], landMask[150:250,175:225], cmap=colormap)


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
temp_cmp = plt.cm.seismic
norm = colors.Normalize(vmin=5, vmax=20)

cb1 = colorbar.ColorbarBase(cax, cmap=temp_cmp,
                            norm=norm,
                            orientation='vertical', label='Temperature (°C)')
scatter = ax.scatter(ds['lon'].values[time_id], ds['lat'].values[time_id], s=1)
# d=270
# ax.scatter(ds.lon[:, d], ds.lat[:, d], c=ds.theta[:, d], cmap=temp_cmp, s=0.5)
# plt.show()

t = np.datetime_as_string(time_range[0], unit='m')
title = ax.set_title('Particles at z = 0 m and time = ' + t)


def animate(i):
    t = np.datetime_as_string(time_range[i], unit='m')

    time_id = np.where(ds['time'] == time_range[i])
    title.set_text('Particles at z = 0 m and time = ' + t)

    scatter.set_offsets(np.c_[ds['lon'].values[time_id], ds['lat'].values[time_id]])
    scatter.set_color(temp_cmp(norm(theta1.values[time_id])))


size = len(time_range)
anim = FuncAnimation(fig, animate, frames=size, interval=200)
anim.save(home_folder + 'Simulations/{0}.mp4'.format(file))
# endregion

print('animation saved')
