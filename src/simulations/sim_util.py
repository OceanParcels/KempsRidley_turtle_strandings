import numpy as np


# find nearest coastal cell to defined beaching location, to release in water
def nearestcoastcell(fieldMesh_x, fieldMesh_y, coastMask, lon, lat):
    dist = np.sqrt((fieldMesh_x - lon) ** 2 * coastMask + (fieldMesh_y - lat) ** 2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))

    dx, dy = 0.001, 0.001  # 0.068 km and 0.111 km at 52°N latitude, respectively

    startlon_release = fieldMesh_x[coords]
    if startlon_release > lon:  # to the left
        startlon_release = startlon_release - dx
        endlon_release = fieldMesh_x[coords[0], coords[1] - 1] + dx
    else:
        startlon_release = startlon_release + dx
        endlon_release = fieldMesh_x[coords[0], coords[1] + 1] - dx

    startlat_release = fieldMesh_y[coords]
    if startlat_release > lat:  # below cell
        startlat_release = startlat_release - dy
        endlat_release = fieldMesh_y[coords[0] - 1, coords[1]] + dy
    else:
        startlat_release = startlat_release + dy
        endlat_release = fieldMesh_y[coords[0] + 1, coords[1]] - dy
    return startlon_release, endlon_release, startlat_release, endlat_release, coords
