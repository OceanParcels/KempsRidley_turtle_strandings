def AdvectionRK4_Wind(particle, fieldset, time):
    """
    advection due to current,and wind:x% windage
    """
    
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

        u_total = u_cur + u_wind
        v_total = v_cur + v_wind

        particle.lon += u_total * particle.dt
        particle.lat += v_total * particle.dt

        
def SampleTemperature(particle, fieldset, time):
    particle.theta=fieldset.T[time, particle.depth, particle.lat, particle.lon]
   
    
def DeleteParticle(particle, fieldset, time):
    particle.delete()
   
