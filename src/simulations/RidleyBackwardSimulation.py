from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, Field, VectorField, Variable, AdvectionRK4
import numpy as np
from datetime import timedelta, datetime
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar
import math
from glob import glob
import turtle_kernels as tk
import sys

data_path = '/storage/shared/oceanparcels/input_data/'
home_dir = '/nethome/manra003/KempsRidley_turtle_strandings/'

#$station $field $windp $days
args = sys.argv
assert len(args) == 5

location = args[1]
print(location)

fields = args[2]
print(fields)

windp = args[3]
print(windp)

d_count = int(args[4])
print(d_count)

strand_lon, strand_lat = 4.1591, 52.0277
release_date = datetime(2011, 12, 12)

if windp=='1pWind':
    windage = 0.01
elif windp == '2pWind':
    windage=0.02
elif windp == '3pWind':
    windage =0.03
else:
    raise ValueError('check windage value again')
    
# region: load currents (reanalysis data- incorporates tides)
re_files = sorted(glob(data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_CUR_dm{0}*.nc'.format(release_date.year)))

filenames_re = {'U': re_files,
                'V': re_files}
variables_re = {'U': 'uo',
                'V': 'vo'}
dimensions_re = {'lat': 'latitude',
                 'lon': 'longitude',
                 'time': 'time'}

index0 = {'depth': [0]}
                  
fieldset_current = FieldSet.from_netcdf(filenames_re, variables_re,
                                    dimensions_re, indices=index0)

# region: load stokes
st_files = sorted(glob(data_path + 'CMEMS/NWSHELF_REANALYSIS_WAV_004_015/metoffice_wave_amm15_NWS_WAV_3hi{0}*.nc'.format(release_date.year)))

filenames_stokes = {'U_stokes': st_files,
                    'V_stokes': st_files}
variables_stokes = {'U_stokes': 'VSDX',
                    'V_stokes': 'VSDY'}
dimensions_stokes = {'lat': 'latitude',
                     'lon': 'longitude',
                     'time': 'time'}

fieldset_stokes = FieldSet.from_netcdf(filenames_stokes, variables_stokes, dimensions_stokes)
fieldset_stokes.U_stokes.units = GeographicPolar()
fieldset_stokes.V_stokes.units = Geographic()
#endregion

#region: load wind 
wind_files=sorted(glob(data_path + 'ERA5/reanalysis-era5-single-level_wind10m_{0}*.nc'.format(release_date.year)))

filenames_wind = {'U_wind': wind_files,
                  'V_wind': wind_files}
variables_wind = {'U_wind': 'u10',
                  'V_wind': 'v10'}
dimensions_wind = {'lat': 'latitude',
                   'lon': 'longitude',
                   'time': 'time'}

fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables_wind, dimensions_wind)

fieldset_wind.U_wind.set_scaling_factor(windage)
fieldset_wind.V_wind.set_scaling_factor(windage)
fieldset_wind.U_wind.units = GeographicPolar()
fieldset_wind.V_wind.units = Geographic()  
#endregion

if fields == 'curr+stokes':
    fieldset_all = FieldSet(U=fieldset_current.U+fieldset_stokes.U_stokes, 
                            V=fieldset_current.V+fieldset_stokes.V_stokes)
elif fields == 'curr+wind':
    fieldset_current.add_field(fieldset_wind.U_wind)
    fieldset_current.add_field(fieldset_wind.V_wind)
    fieldset_all = fieldset_current
    vectorField_wind = VectorField('UV_wind',fieldset_all.U_wind,fieldset_all.V_wind)
    fieldset_all.add_vector_field(vectorField_wind)    

elif fields == 'curr+stokes+wind':
    fieldset_all = FieldSet(U=fieldset_current.U+fieldset_stokes.U_stokes, 
                            V=fieldset_current.V+fieldset_stokes.V_stokes)
    fieldset_all.add_field(fieldset_wind.U_wind)
    fieldset_all.add_field(fieldset_wind.V_wind)
    vectorField_wind = VectorField('UV_wind',fieldset_all.U_wind,fieldset_all.V_wind)
    fieldset_all.add_vector_field(vectorField_wind) 
else:
    fields = 'curr'
    fieldset_all = FieldSet(U=fieldset_current.U, 
                            V=fieldset_current.V)

#region: load temperature metoffice_foam1_amm7_NWS_TEM_dm20010815.nc
tem_files = sorted(glob(data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_TEM_dm{0}*.nc'.format(release_date.year)))

filenames_tem = {'T': tem_files}
variables_tem = {'T': 'thetao'}
dimensions_tem = {'lat': 'latitude',
                 'lon': 'longitude',
                 'time': 'time'}

fieldset_temp = FieldSet.from_netcdf(filenames_tem, variables_tem,
                                    dimensions_tem, indices=index0)  

fieldset_all.add_field(fieldset_temp.T)
#endregion

lons = fieldset_current.U.lon
lats = fieldset_current.U.lat
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)
  

class TurtleParticle(JITParticle):
    theta = Variable('theta', dtype=np.float64, initial=-999.0, to_write=True)
    
coastMask=np.loadtxt(home_dir + 'data/coastalMask_297x_375y')
    
# find nearest coastal cell to defined beaching location, to release in water
def nearestcoastcell(lon, lat):
    dist = np.sqrt((fieldMesh_x - lon) ** 2 * coastMask + (fieldMesh_y - lat) ** 2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))
    startlon_release = fieldMesh_x[coords]
    endlon_release = fieldMesh_x[coords[0], coords[1] + 1]
    startlat_release = fieldMesh_y[coords]
    endlat_release = fieldMesh_y[coords[0] + 1, coords[1]]
    return startlon_release, endlon_release, startlat_release, endlat_release, coords


startlon_release, endlon_release, startlat_release, endlat_release, coords = nearestcoastcell(strand_lon, strand_lat)
# 10x10 particles -> 100 particles homogeneously spread over grid cell
np_sqrt = 100
re_lons = np.linspace(startlon_release, endlon_release, np_sqrt, endpoint=False)
re_lats = np.linspace(startlat_release, endlat_release, np_sqrt, endpoint=False)
fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)

    
pset = ParticleSet.from_list(fieldset=fieldset_all, pclass=TurtleParticle, lon=fieldMesh_x_re, lat=fieldMesh_y_re, time=release_date)

filename = "/nethome/manra003/sim_out/kempT/{0}/Sum_BK_{1}_{2}days_{3}.nc".format(windp,fields,d_count,location)

output_file = pset.ParticleFile(name=filename,
                                outputdt=timedelta(days=1))

if fields == 'curr+wind' or fields == 'curr+stokes+wind':
    kernels =  pset.Kernel(tk.AdvectionRK4_Wind)+ pset.Kernel(tk.SampleTemperature)
else:
    kernels =  pset.Kernel(AdvectionRK4)+ pset.Kernel(tk.SampleTemperature)


pset.execute(kernels,
             runtime=timedelta(days=d_count),
             dt= -1 * timedelta(minutes=10),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: tk.DeleteParticle})
output_file.close()
