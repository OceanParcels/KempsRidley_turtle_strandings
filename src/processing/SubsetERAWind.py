'''
Program to subset wind data for the European Western Shelf and combine the slices
'''
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as clr
import numpy as np

left_min_lon = 340
left_max_lon = 359.75
right_min_lon = 0
right_max_lon = 10

min_lat = 40
max_lat = 65

years = [2007, 2006, 2008, 2011, 2014, 2021]
months=[[1],[8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]

for index, y in enumerate(years):
    for m in months[index]:
        ds = xr.open_dataset('/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_{0}{1}.nc'.format(y,str(m).zfill(2)))

        left_slice = ds.loc[{'latitude': slice(max_lat,min_lat),
                          'longitude': slice(left_min_lon, left_max_lon)}]

        right_slice = ds.loc[{'latitude': slice(max_lat, min_lat), #notice how the order is different
                          'longitude': slice(right_min_lon, right_max_lon)}]
        #change the values of longitude coordinates to negative.
        new_lon = left_slice.longitude.copy()
        new_lon = new_lon - 360
        left_slice = left_slice.assign_coords(longitude=new_lon)        
        result = [left_slice, right_slice]
        R = xr.combine_nested(result, concat_dim=["longitude"])
        R.to_netcdf('/nethome/manra003/analysis/KempRidley/ERA5_EUWsubset/reanalysis-era5-single-level_wind10m_{0}{1}.nc'.format(y,str(m).zfill(2)))
                         
                         