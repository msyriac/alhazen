from enlib import lensing,powspec,utils,enmap,bench
import numpy as np, sys

powspec_file = "data/Aug6_highAcc_CDM_lenspotentialCls.dat"
ps = powspec.read_camb_full_lens(powspec_file).astype(np.float64)
geom = lambda x: enmap.fullsky_geometry(res=x*np.pi/180./60., proj="car")

shape,wcs = geom(4.0)

#obs = enmap.posmap(shape,wcs)

#sys.exit()

with bench.show("2arc"):
    full, = lensing.rand_map(shape,wcs,ps,lmax=4000,maplmax=4000,verbose=False,delta_theta = 2.*np.pi/180.,seed=1)


with bench.show("2arc"):
    full2, = lensing.rand_map(shape,wcs,ps,lmax=4000,maplmax=4000,verbose=True,seed=1)
    

import orphics.tools.io as io

io.quickPlot2d(full,io.dout_dir+"fmap.png")
io.quickPlot2d(full-full2,io.dout_dir+"diffmap.png")
print np.isclose(full,full2)
print full-full2
print np.max(np.abs(full-full2)) #/np.max(np.max(np.abs(full),np.abs(full2)))*100.
