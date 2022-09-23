import numpy as np

# find nearest coastal cell to defined beaching location, to release in water
def nearestcoastcell(fieldMesh_x, fieldMesh_y, coastMask, lon, lat):
    dist = np.sqrt((fieldMesh_x - lon) ** 2 * coastMask + (fieldMesh_y - lat) ** 2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))
    startlon_release = fieldMesh_x[coords]
    endlon_release = fieldMesh_x[coords[0], coords[1] + 1]
    startlat_release = fieldMesh_y[coords]
    endlat_release = fieldMesh_y[coords[0] + 1, coords[1]]
    return startlon_release, endlon_release, startlat_release, endlat_release, coords
