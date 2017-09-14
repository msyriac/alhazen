import numpy as np
from enlib import enmap,resample,lensing
import orphics.analysis.flatMaps as fmaps
from szar.counts import ClusterCosmology
import orphics.tools.io as io
import alhazen.lensTools as lt
from mpi4py import MPI
import sys
from scipy.linalg import pinv2

def nfwkappa(massOverh):
    zL = 0.7
    overdensity = 180.
    critical = False
    atClusterZ = False
    concentration = 3.2
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,pa.modrmap* 180.*60./np.pi,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return kappa


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

lmax = 8000
cc = ClusterCosmology(lmax=lmax,pickling=True)
theory = cc.theory

arc = 10.0
px = 1.0
lens_order = 5


shape,wcs = enmap.get_enmap_patch(arc,px,proj="car")



    
pa = fmaps.PatchArray(shape,wcs,dimensionless=False,skip_real=False)
pa.add_theory(theory,lmax)
pa.add_white_noise_with_atm(0.01,0.,0,1,0,1)



N = 10000
Nmasses = numcores #10


from alhazen.halos import NFWkappa

mrange = np.linspace(2.,4.,Nmasses)*1.e14
cs = []
cinvs = []



#for M in mrange:
if True:
    M = mrange[rank]
    kappa = nfwkappa(M)
    phi, fphi = lt.kappa_to_phi(kappa,pa.modlmap,return_fphi=True)
    #grad_phi = enmap.grad(phi)
    alpha_pix = enmap.grad_pixf(fphi)


    for i in range(N):
        cmb_map = pa.get_unlensed_cmb(seed=i)
        #lensed = lensing.lens_map(cmb_map, grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
        lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order)
        lensed += pa.get_noise_sim(seed=i*100000)

        cmb = lensed.reshape((1,shape[0]*shape[1]))
        if i==0:
            vec = cmb
        else:
            vec = np.append(vec,cmb,axis=0)
        if i%1000==0: print M,i
    print vec.shape
    print "Calculating cov..."
    c = np.cov(vec.T)
    
    print c
    print c.shape
    cs.append(c)
    #io.quickPlot2d(c,"cov.png")
    print "Inverting cov..."
    from btip import inpaintStamp as ins
    cinv = pinv2(c)
    cinvs.append(cinv)



np.save("c_"+str(rank),c)
np.save("cinv_"+str(rank),cinv)
sys.exit()
from alhazen.maxlike import lnlike

M = 3.5
kappa = nfwkappa(M)
phi, fphi = lt.kappa_to_phi(kappa,pa.modlmap,return_fphi=True)
#grad_phi = enmap.grad(phi)
alpha_pix = enmap.grad_pixf(fphi)

N = 1000

totlikes = 0.
for i in range(N):
    lnlikes = []
    cmb_map = pa.get_unlensed_cmb(seed=i+100000)
    lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order)

    for k,M in enumerate(mrange):
    
        print i,M
        lnlikes.append(lnlike(cs[k],cinvs[k],lensed))

    totlikes += np.array(lnlikes)

pl = io.Plotter()
pl.add(mrange,np.exp(-0.5*totlikes))
pl.done("lnlikes.png")
