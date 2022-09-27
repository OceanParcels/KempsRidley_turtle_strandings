import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.colors as color
import pandas as pd

home_folder = '/Users/dmanral/Desktop/Analysis/Ridley/'

# we need to coordinates file to access the corner points - glamf/gphif
model_mask_file = home_folder + 'data/landMask_297x_375y'

landMask = np.genfromtxt(model_mask_file, delimiter=None)

file = 'Sum_BK_3pWind_Beaching_curr+wind_120days_WesterschouwenSchouwen_100p'
ds = xr.open_dataset(home_folder + 'Simulations/{0}.nc'.format(file))
print(ds)

U_ds = xr.open_dataset(home_folder + 'data/metoffice_foam1_amm7_NWS_CUR_dm20141211.nc')
lons = U_ds.longitude
lats = U_ds.latitude
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)

fig = plt.figure()
ax = plt.axes()
colormap = color.ListedColormap(['whitesmoke', 'grey'])
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

scatter = ax.scatter(ds['lon'].values[time_id], ds['lat'].values[time_id], s=1, c='blue')

t = np.datetime_as_string(time_range[0], unit='m')
title = ax.set_title('Particles at z = 0 m and time = ' + t)


def animate(i):
    t = np.datetime_as_string(time_range[i], unit='m')

    time_id = np.where(ds['time'] == time_range[i])
    title.set_text('Particles at z = 0 m and time = ' + t)

    scatter.set_offsets(np.c_[ds['lon'].values[time_id], ds['lat'].values[time_id]])


size = len(time_range)
anim = FuncAnimation(fig, animate, frames=size, interval=200)
anim.save(home_folder + 'Simulations/{0}.mp4'.format(file))
# endregion

print('animation saved')
