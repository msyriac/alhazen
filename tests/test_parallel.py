from enlib import lensing,powspec,utils,enmap,bench
import numpy as np, sys

from multiprocessing import Pool



def func(a):

    return a**2+a #sum([x**2.+x for x in a])
    #return sum([x**2.+x for x in a])
    

if __name__=='__main__':

    p = Pool(12)
    print p._processes
    a = np.random.random((50000000,1)).ravel().tolist()
    print (len(a))
    with bench.show("serial"):
        b = [func(x) for x in a]

    print (len(a))
    with bench.show("parallel"):
        b = p.map(func, a)    

    p.close()

    sys.exit()


    # powspec_file = "data/Aug6_highAcc_CDM_lenspotentialCls.dat"
    # ps = powspec.read_camb_full_lens(powspec_file).astype(np.float64)
    geom = lambda x: enmap.fullsky_geometry(res=x*np.pi/180./60., proj="car")

    shape,wcs = geom(4.0)

    obs = enmap.posmap(shape,wcs)
