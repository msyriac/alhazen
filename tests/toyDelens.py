from enlib import enmap, lensing, powspec, utils
from szar.counts import ClusterCosmology
from alhazen.halos import NFWkappa
from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.tools.stats as stats
import os, sys
import numpy as np
import matplotlib.pyplot as plt

out_dir = os.environ['WWW']+"plots/halotest/lensorder5_"
lmax = 8000
#lens_order = 5

# === COSMOLOGY ===
cc = ClusterCosmology(lmax=lmax,pickling=True)
TCMB = 2.7255e6
theory = cc.theory




patch_width_arcmin = 40.
sim_pixel_scale = 0.1
pol = False
shape_sim, wcs_sim = enmap.get_enmap_patch(patch_width_arcmin,sim_pixel_scale,proj="car",pol=pol)
modr_sim = enmap.modrmap(shape_sim,wcs_sim) * 180.*60./np.pi
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)




massOverh = 2.e14
zL = 0.7
overdensity = 500.
critical = True
atClusterZ = True
concentration = 3.2
comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
comL = cc.results.comoving_radial_distance(zL)*cc.h
winAtLens = (comS-comL)/comS
kappa_map,r500 = NFWkappa(cc,massOverh,concentration,zL,modr_sim,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


phi, fphi = lt.kappa_to_phi(kappa_map,modlmap_sim,return_fphi=True)
# io.quickPlot2d(kappa_map,out_dir+"kappa_map.png")
# io.quickPlot2d(phi,out_dir+"phi.png")
alpha_pix = enmap.grad_pixf(fphi)


ps = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=False)

np.random.seed(200)

# NO BEAM, NO NOISE
# Unlensed
unlensed = enmap.rand_map(shape_sim,wcs_sim,ps)

test = False
if test:
    pl = io.Plotter(scaleY='log')


    for k,lens_order in enumerate(range(5,0,-1)):

        alpha = (lens_order-2+1.+1)/(5-2+1.+1)
        print lens_order, alpha

        # Lens with kappa1
        lensed = lensing.lens_map_flat_pix(unlensed, alpha_pix,order=lens_order)

        # Delens with kappa1
        simple_delensed = lensing.lens_map_flat_pix(lensed.copy(), -alpha_pix,order=lens_order)

        grad_phi = enmap.grad(phi)

        lens_residual = lensed - unlensed
        #io.quickPlot2d(lens_residual,out_dir+"lensres.png")
        simple_residual = simple_delensed - unlensed
        #io.quickPlot2d(simple_residual,out_dir+"simpleres.png")


        iters = []
        for nstep in range(1,8):
            iter_delensed = lensing.delens_map(lensed.copy(), grad_phi, nstep=nstep, order=lens_order, mode="spline", border="cyclic")
            # Check residual

            iter_residual = iter_delensed - unlensed
            iters.append(iter_residual)


            #io.quickPlot2d(iter_residual,out_dir+"iterres_"+str(nstep).zfill(2)+".png")


        bin_edges = np.arange(0.,20.,0.5)
        binner = stats.bin2D(modr_sim,bin_edges)

        cents, prof = binner.bin(lens_residual)
        pl.add(cents,np.abs(prof),ls="--",label=("no delensing" if k==0 else None),alpha=alpha,color="C0")
        cents, prof = binner.bin(simple_residual)
        pl.add(cents,np.abs(prof),ls="--",label=("anti-lensing"if k==0 else None),alpha=alpha,color="C1")


        for i,iteration in enumerate(iters):
            cents, prof = binner.bin(iteration)
            pl.add(cents,np.abs(prof),label=("iterative nstep="+str(1+i) if k==0 else None),alpha=alpha,color="C"+str(i+2))


    pl.legendOn(labsize=10)
    pl.done(out_dir+"iter_profs.png")


lens_order = 5
nstep = 9

iters = 2

unlensed = unlensed * TCMB
lensed = lensing.lens_map_flat_pix(unlensed, alpha_pix,order=lens_order)

iter_delensed = lensed
phi, fphi = lt.kappa_to_phi(kappa_map,modlmap_sim,return_fphi=True)
grad_phi = enmap.grad(phi)/(iters)
for i in range(1,iters+1):
    print i
    iter_delensed = lensing.delens_map(iter_delensed, grad_phi, nstep=nstep, order=lens_order, mode="spline", border="cyclic")
    
iter_residual = iter_delensed - unlensed
lens_residual = lensed - unlensed
io.quickPlot2d(lens_residual,out_dir+"lensres.png")
io.quickPlot2d(iter_residual,out_dir+"iterres.png")
