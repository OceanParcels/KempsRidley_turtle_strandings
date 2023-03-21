def CurrWindAdvectionRK4(particle, fieldset, time):
    """
    advection due to current and stokes drift combined together in UV
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

        u_tot = (u1 + 2*u2 + 2*u3 + u4) / 6.
        v_tot = (v1 + 2*v2 + 2*v3 + v4) / 6.
    
        particle.lon += u_tot * particle.dt
        particle.lat += v_tot * particle.dt

        
def SampleTemperature(particle, fieldset, time):
    if particle.beached == 0:
        particle.theta=fieldset.T[time+particle.dt, particle.depth, particle.lat, particle.lon]
   
    
def DeleteParticle(particle, fieldset, time):
    print('Particle deleted %d %.2f %.2f %d' % (particle.id, time, particle.lat, particle.lon, particle.beached))
    particle.delete()
    

def BeachTesting(particle, fieldset, time):
    if particle.beached == 0: # at sea
        (u_curr, v_curr) = fieldset.UV_current[time, particle.depth, particle.lat, particle.lon]
        if u_curr == 0 and v_curr == 0:
            particle.beached = 1


def AttemptUnBeaching(particle, fieldset, time):
    if particle.beached == 1:
        (u_land, v_land) = fieldset.UV_unbeach[time, particle.depth, particle.lat, particle.lon]
        
        particle.lon += u_land * (-particle.dt)
        particle.lat += v_land * (-particle.dt)
        
        (u, v) = fieldset.UV_current[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            particle.beached = -2 # unbeaching failed
            print('Unbeaching failed %d %.2f %.2f %d' % (particle.id, time, particle.lat, particle.lon, particle.beached))
        else:
            particle.beached = 0 # at sea again
