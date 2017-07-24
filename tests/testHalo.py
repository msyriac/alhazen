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

# === PARAMS ===

Nsims = 100
sim_pixel_scale = 0.1
analysis_pixel_scale = 0.5
patch_width_arcmin = 100.
periodic = True
beam_arcmin = 1.0
noise_T_uK_arcmin = 0.1
noise_P_uK_arcmin = 0.1
lmax = 8000
tellmax = 8000
pellmax = 8000
tellmin = 200
pellmin = 200
kellmax = 8500
kellmin = 200
gradCut = 2000
pol = False
debug = True

out_dir = os.environ['WWW']+"plots/"


# === COSMOLOGY ===
cc = ClusterCosmology(lmax=lmax,pickling=True)
TCMB = 2.7255e6
theory = cc.theory


# === TEMPLATE MAPS ===

bin_edges = np.arange(0.,10.0,0.2)

shape_sim, wcs_sim = enmap.get_enmap_patch(patch_width_arcmin,sim_pixel_scale,proj="car",pol=pol)
modr_sim = enmap.modrmap(shape_sim,wcs_sim) * 180.*60./np.pi
binner_sim = stats.bin2D(modr_sim,bin_edges)

shape_dat, wcs_dat = enmap.get_enmap_patch(patch_width_arcmin,analysis_pixel_scale,proj="car",pol=pol)
modr_dat = enmap.modrmap(shape_dat,wcs_dat) * 180.*60./np.pi
binner_dat = stats.bin2D(modr_dat,bin_edges)




# === CLUSTER ===

massOverh = 2.e14
zL = 0.7
overdensity = 180.
critical = False
atClusterZ = False
concentration = 3.2
comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
comL = cc.results.comoving_radial_distance(zL)*cc.h
winAtLens = (comS-comL)/comS
nfw_map,r500 = NFWkappa(cc,massOverh,concentration,zL,modr_sim,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
#cents, nkprofile = binner.bin(nfwMap)




# === EXPERIMENT ===

lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
modl_map_alt = enmap.modlmap(shape_sim,wcs_sim)
assert np.all(np.isclose(modlmap_sim,modl_map_alt))

ntfunc = cmb.get_noise_func(beam_arcmin,noise_T_uK_arcmin,ellmin=tellmin,ellmax=tellmax,TCMB=2.7255e6)
npfunc = cmb.get_noise_func(beam_arcmin,noise_P_uK_arcmin,ellmin=pellmin,ellmax=pellmax,TCMB=2.7255e6)

nT_sim = ntfunc(modlmap_sim)
nP_sim = npfunc(modlmap_sim)






# === ESTIMATOR ===

template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
modl_map_alt = enmap.modlmap(shape_dat,wcs_dat)
assert np.all(np.isclose(modlmap_dat,modl_map_alt))

nT = ntfunc(modlmap_dat)
nP = npfunc(modlmap_dat)


fMaskCMB_T = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellmin,lmax=tellmax)
fMaskCMB_P = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellmin,lmax=pellmax)
fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)
qest = Estimator(template_dat,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                 fmaskY2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                 fmaskKappa=fMask,
                 doCurl=False,
                 TOnly=not(pol),
                 halo=True,
                 gradCut=gradCut,verbose=True,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None)





# === ENMAP POWER ===

fine_ells = np.arange(0,lmax,1)
cltt = theory.uCl('TT',fine_ells)
clee = theory.uCl('EE',fine_ells)
clte = theory.uCl('TE',fine_ells)
ps = np.zeros((3,3,fine_ells.size))
ps[0,0] = cltt
ps[1,1] = clee
ps[0,1] = clte
ps[1,0] = clte

# === SIM AND RECON LOOP ===

for i in range(Nsims):
    print i

    unlensed = enmap.rand_map(shape_sim,wcs_sim,ps)
    if i==0 and debug:
        if pol:
            io.highResPlot2d(unlensed[0],out_dir+"tmap.png")
            io.highResPlot2d(unlensed[1],out_dir+"qmap.png")
            io.highResPlot2d(unlensed[2],out_dir+"umap.png")
        else:        
            io.highResPlot2d(unlensed,out_dir+"tmap.png")

