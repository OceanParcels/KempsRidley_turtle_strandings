def AdvectionRK4(particle, fieldset, time):
    """
    advection due to current,and wind:x% windage
    """
    if particle.beached == 0:
        # Runga-Kutte Fourth order 
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]

        u_cur = (u1 + 2*u2 + 2*u3 + u4) / 6.
        v_cur = (v1 + 2*v2 + 2*v3 + v4) / 6.
    
        particle.lon += u_cur * particle.dt
        particle.lat += v_cur * particle.dt

def AdvectionRK4_Wind(particle, fieldset, time):
    """
    advection due to current,and wind:x% windage
    """
    if particle.beached == 0:
        # Runga-Kutte Fourth order 
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]

        u_cur = (u1 + 2*u2 + 2*u3 + u4) / 6.
        v_cur = (v1 + 2*v2 + 2*v3 + v4) / 6.

        if (u_cur==0) and (v_cur==0):
            particle.lon +=0.
            particle.lat +=0.
        else:
            #Windage-already scaled down wind
            (u_wind, v_wind) = fieldset.UV_wind[time, particle.depth, particle.lat, particle.lon]
            particle.lon += (u_cur + u_wind) * particle.dt
            particle.lat += (v_cur + v_wind) * particle.dt

        
def SampleTemperature(particle, fieldset, time):
    if particle.beached == 0:
        particle.theta=fieldset.T[time, particle.depth, particle.lat, particle.lon]
   
    
def DeleteParticle(particle, fieldset, time):
    particle.delete()
    

def BeachTesting(particle, fieldset, time):
    if particle.beached == 0: # at sea
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            particle.beached = 1


def AttemptUnBeaching(particle, fieldset, time):
    if particle.beached == 1:
        (u_land, v_land) = fieldset.UV_unbeach[time, particle.depth, particle.lat, particle.lon]
        
        particle.lon += u_land * (-particle.dt)
        particle.lat += v_land * (-particle.dt)
        
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            particle.beached = -2 # unbeaching failed
            print('Unbeaching failed %.2f %.2f %d' % (particle.lat, particle.lon, particle.beached))
        else:
            particle.beached = 0 # at sea again
   