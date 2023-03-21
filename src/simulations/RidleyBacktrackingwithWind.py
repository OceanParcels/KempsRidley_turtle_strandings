from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, Field, VectorField, Variable, AdvectionRK4
import numpy as np
from datetime import timedelta, datetime
import pandas as pd
from parcels.tools.converters import Geographic, GeographicPolar
from glob import glob
import with_wind_kernels as tk
import sim_util as util
import sys

import parcels
print(parcels.__version__)

data_path = '/storage/shared/oceanparcels/input_data/'
home_dir = '/nethome/manra003/KempsRidley_turtle_strandings/'

# $station $field $windp $days
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

if 'wind' in fields:
    if '01pWind' in windp:
        windage = 0.001
    elif '1pWind' in windp:
        windage = 0.01
    elif '2pWind' in windp:
        windage = 0.02
    elif '3pWind' in windp:
        windage = 0.03
    else:
        raise ValueError('check windage value again')
else:
    windp = '0pWind'
    windage = 0.0

stations = pd.read_csv(home_dir + 'data/Locations_NL.csv')
st = stations.loc[lambda stations: stations['Location'] == location]

strand_lon, strand_lat = st['Longitude'].values[0], st['Latitude'].values[0]
release_date = datetime.strptime(st['Date'].values[0] + ' 12:00:00', '%d/%m/%Y %H:%M:%S')


np_sqrt = 100

# region: load currents (reanalysis data- incorporates tides)
if location == 'IJmuiden':
    re_files = sorted(
        glob(data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_CUR_dm200[6-7]*.nc'))
else:
    re_files = sorted(glob(
        data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_CUR_dm{0}*.nc'.format(
            release_date.year)))

filenames_re = {'U_curr': re_files,
                'V_curr': re_files}
variables_re = {'U_curr': 'uo',
                'V_curr': 'vo'}
dimensions_re = {'lat': 'latitude',
                 'lon': 'longitude',
                 'time': 'time'}


index0 = {'depth': [0]}

fieldset_current = FieldSet.from_netcdf(filenames_re, variables_re,
                                        dimensions_re, indices=index0)
fieldset_current.U_curr.units = GeographicPolar()
fieldset_current.V_curr.units = Geographic()


# region: load stokes
if location == 'IJmuiden':
    st_files = sorted(
        glob(data_path + 'CMEMS/NWSHELF_REANALYSIS_WAV_004_015/metoffice_wave_amm15_NWS_WAV_3hi200[6-7]*.nc'))
else:
    st_files = sorted(
        glob(data_path + 'CMEMS/NWSHELF_REANALYSIS_WAV_004_015/metoffice_wave_amm15_NWS_WAV_3hi{0}*.nc'.format(
            release_date.year)))

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
# endregion

# region: load wind
if windage > 0:
    wind_folder='/nethome/manra003/analysis/KempRidley/ERA5_EUWsubset/'
    if location == 'IJmuiden':
        wind_files = sorted(glob(wind_folder + 'ERA5/reanalysis-era5-single-level_wind10m_200[6-7]*.nc'))
    else:
        wind_files = sorted(
            glob(wind_folder + 'ERA5/reanalysis-era5-single-level_wind10m_{0}*.nc'.format(release_date.year)))

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

    #Create full fieldset with all the fields summed together
    fieldset_all = FieldSet(U=fieldset_current.U_curr + fieldset_stokes.U_stokes + fieldset_wind.U_wind,
                            V=fieldset_current.V_curr + fieldset_stokes.V_stokes + fieldset_wind.V_wind)
else:
    #Create full fieldset with all the fields summed together
    fieldset_all = FieldSet(U=fieldset_current.U_curr + fieldset_stokes.U_stokes,
                            V=fieldset_current.V_curr + fieldset_stokes.V_stokes)


# also add u,v_curr to the fieldset_all- needed for land check.
fieldset_all.add_field(fieldset_current.U_curr)
fieldset_all.add_field(fieldset_current.V_curr)
fieldset_all.U_curr.units = GeographicPolar()
fieldset_all.V_curr.units = Geographic()
vectorField_current = VectorField('UV_current', fieldset_current.U_curr, fieldset_current.V_curr)
fieldset_all.add_vector_field(vectorField_current)

# region: load temperature metoffice_foam1_amm7_NWS_TEM_dm20010815.nc
if location == 'IJmuiden':
    tem_files = sorted(glob(
        data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_TEM_dm200[6-7]*.nc'))
else:
    tem_files = sorted(glob(
        data_path + 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/metoffice_foam1_amm7_NWS_TEM_dm{0}*.nc'.format(
            release_date.year)))

filenames_tem = {'T': tem_files}
variables_tem = {'T': 'thetao'}
dimensions_tem = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}

fieldset_temp = FieldSet.from_netcdf(filenames_tem, variables_tem,
                                     dimensions_tem, indices=index0)

fieldset_all.add_field(fieldset_temp.T)
fieldset_all.T.interp_method = 'linear_invdist_land_tracer'  # updated from the interpolation tutorial
# endregion

lons = fieldset_current.U_curr.lon
lats = fieldset_current.U_curr.lat
fieldMesh_x, fieldMesh_y = np.meshgrid(lons, lats)
coastMask = np.loadtxt(home_dir + 'data/coastalMask_297x_375y')

# region: load unbeaching land currents
file_displacement_U = home_dir + 'data/displacement_U_%ix_%iy' % (len(lons), len(lats))
file_displacement_V = home_dir + 'data/displacement_V_%ix_%iy' % (len(lons), len(lats))

landDisp_U = np.loadtxt(file_displacement_U)
landDisp_V = np.loadtxt(file_displacement_V)

U_land = Field('U_land', landDisp_U, lon=lons, lat=lats, mesh='spherical')
V_land = Field('V_land', landDisp_V, lon=lons, lat=lats, mesh='spherical')

fieldset_all.add_field(U_land)
fieldset_all.add_field(V_land)

fieldset_all.U_land.units = GeographicPolar()
fieldset_all.V_land.units = Geographic()

vectorField_unbeach = VectorField('UV_unbeach', U_land, V_land)
fieldset_all.add_vector_field(vectorField_unbeach)
# endregion


class TurtleParticle(JITParticle):
    theta = Variable('theta', dtype=np.float64, initial=fieldset_all.T, to_write=True)
    # beached : 0 at sea, 1 beached, -1 deleted, -2 unbeaching failed
    beached = Variable('beached', dtype=np.int32, initial=0., to_write=True)


startlon_release, endlon_release, startlat_release, endlat_release, coords = util.nearestcoastcell(fieldMesh_x,
                                                                                                   fieldMesh_y,
                                                                                                   coastMask,
                                                                                                   strand_lon,
                                                                                                   strand_lat)
# 10x10 particles -> 100 particles homogeneously spread over grid cell
re_lons = np.linspace(startlon_release, endlon_release, np_sqrt)
re_lats = np.linspace(startlat_release, endlat_release, np_sqrt)
fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)

pset = ParticleSet.from_list(fieldset=fieldset_all, pclass=TurtleParticle, lon=fieldMesh_x_re, lat=fieldMesh_y_re,
                             time=release_date)

filename = "/nethome/manra003/analysis/KempRidley/simulations/{0}/Sum_BK_{0}_{1}_{2}days_{3}".format(windp, fields, d_count, location)

output_file = pset.ParticleFile(name=filename,
                                outputdt=timedelta(days=1))


kernels = pset.Kernel(tk.CurrWindAdvectionRK4) + pset.Kernel(tk.BeachTesting) + pset.Kernel(
        tk.AttemptUnBeaching) + pset.Kernel(
        tk.SampleTemperature)

pset.execute(kernels,
             runtime=timedelta(days=d_count),
             dt=-1 * timedelta(minutes=10),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: tk.DeleteParticle})
output_file.close()
