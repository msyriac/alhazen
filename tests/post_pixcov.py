import numpy as np
from enlib import enmap,resample,lensing
import orphics.analysis.flatMaps as fmaps
from alhazen.halos import NFWkappa
from szar.counts import ClusterCosmology
import orphics.tools.io as io
import alhazen.lensTools as lt
from mpi4py import MPI
import sys

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

from alhazen.maxlike import lnlike

lmax = 8000
cc = ClusterCosmology(lmax=lmax,pickling=True)
theory = cc.theory

arc = 10.0
px = 1.0
lens_order = 5


shape,wcs = enmap.get_enmap_patch(arc,px,proj="car")



    
pa = fmaps.PatchArray(shape,wcs,dimensionless=False,skip_real=False)
pa.add_theory(theory,lmax)

Nmasses = 4
mrange = np.linspace(2.,4.,Nmasses)*1.e14


M = 3.5
kappa = nfwkappa(M)
phi, fphi = lt.kappa_to_phi(kappa,pa.modlmap,return_fphi=True)
#grad_phi = enmap.grad(phi)
alpha_pix = enmap.grad_pixf(fphi)

N = 1000

totlikes = 0.
allike = 1.
for i in range(N):
    lnlikes = []
    cmb_map = pa.get_unlensed_cmb(seed=i+100000)
    lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order)

    if i%100==0: print i
    for k,M in enumerate(mrange):
    
        c = np.load("c_"+str(k)+".npy")
        cinv = np.load("cinv_"+str(k)+".npy")
        lnlikeval = lnlike(c,cinv,lensed)
        lnlikes.append(lnlikeval)
        #print lnlikeval
    totlikes += np.array(lnlikes)
    print totlikes

pl = io.Plotter()
likes = np.exp(-0.5*totlikes)
print likes
likes /= likes.sum()
pl.add(mrange,likes)
pl.done("lnlikes.png")
