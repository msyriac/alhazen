import numpy as np

def dndz(z,z0=0.33):
    ans = (z**2.)* np.exp(-1.0*z/z0)/ (2.*z0**3.) 
    return ans




