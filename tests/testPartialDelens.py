"""

- Lens the CMB with NFW Kappa.
- Delens with the same Kappa.
- Stack the residuals.

"""
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
import enlib.fft as fftfast

np.random.seed(100)

noiseless_cmb = False
noiseless_kappa = True
gaussian_kappa_noise = True
Nstack = 30


lens_order = 5
nstep = 7


beam_arcmin = 1. #1. #1. #1.0
noise_T_uK_arcmin = 1. #0.001 #1.0 #0.01
noise_P_uK_arcmin = 1. #0.001 #1.0 #0.01
lmax = 6500
tellmax = 6000
pellmax = 6000
tellmin = 200
pellmin = 200
kellmax = min(tellmax,pellmax)
kellmin = 200
gradCut = 2000
#pol_list = ['TT','EB','EE','ET','TE']
pol_list = ['TT']#,'EB']

out_dir = os.environ['WWW']+"plots/halotest/lensorder5_"

# === COSMOLOGY ===
cc = ClusterCosmology(lmax=lmax,pickling=True)
TCMB = 2.7255e6
theory = cc.theory

ps = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=False)


patch_width_arcmin = 40.
sim_pixel_scale = 0.1
pol = False
shape_sim, wcs_sim = enmap.get_enmap_patch(patch_width_arcmin,sim_pixel_scale,proj="car",pol=pol)
modr_sim = enmap.modrmap(shape_sim,wcs_sim) * 180.*60./np.pi
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
pix_ells = np.arange(0,modlmap_sim.max(),1)





# === EXPERIMENT ===


ntfunc = cmb.get_noise_func(beam_arcmin,noise_T_uK_arcmin,ellmin=tellmin,ellmax=tellmax,TCMB=2.7255e6)
npfunc = cmb.get_noise_func(beam_arcmin,noise_P_uK_arcmin,ellmin=pellmin,ellmax=pellmax,TCMB=2.7255e6)

if beam_arcmin>1.e-5:
    kbeam_sim = cmb.gauss_beam(modlmap_sim,beam_arcmin)
else:
    kbeam_sim = modlmap_sim*0.+1.
ps_noise = np.zeros((3,3,pix_ells.size))
ps_noise[0,0] = pix_ells*0.+(noise_T_uK_arcmin*np.pi/180./60./TCMB)**2.
ps_noise[1,1] = pix_ells*0.+(noise_P_uK_arcmin*np.pi/180./60./TCMB)**2.
ps_noise[2,2] = pix_ells*0.+(noise_P_uK_arcmin*np.pi/180./60./TCMB)**2.




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
alpha_pix = enmap.grad_pixf(fphi)
grad_phi_true = enmap.grad(phi)

res_stack = 0.

fMaskCMB_T = fmaps.fourierMask(lx_sim,ly_sim,modlmap_sim,lmin=tellmin,lmax=tellmax)
# io.quickPlot2d(np.fft.fftshift(fMaskCMB_T),out_dir+"presmooth.png")
# fMaskCMB_T = fmaps.smooth(fMaskCMB_T,modlmap_sim,gauss_sigma_arcmin=0.1)
# io.quickPlot2d(np.fft.fftshift(fMaskCMB_T),out_dir+"postsmooth.png")
# fMaskCMB_T[modlmap_sim<2]=0.

def f(rmap):
    fk = fftfast.fft(rmap,axes=[-2,-1])
    fk = np.nan_to_num(fk) *fMaskCMB_T
    return enmap.samewcs(fftfast.ifft(fk,axes=[-2,-1],normalize=True).real,rmap)
    

for i in range(Nstack):
    
    unlensed = enmap.rand_map(shape_sim,wcs_sim,ps)
    lensed = lensing.lens_map_flat_pix(unlensed, alpha_pix,order=lens_order)
    if noiseless_cmb:
        measured = lensed
    else:
        klteb = enmap.map2harm(lensed)
        klteb_beam = klteb*kbeam_sim
        lteb_beam = enmap.ifft(klteb_beam).real
        noise = enmap.rand_map(shape_sim,wcs_sim,ps_noise,scalar=True)
        observed = lteb_beam + noise

        fkmaps = fftfast.fft(observed,axes=[-2,-1])
        fkmaps = np.nan_to_num(fkmaps/kbeam_sim) *fMaskCMB_T
        measured = enmap.samewcs(fftfast.ifft(fkmaps,axes=[-2,-1],normalize=True).real,observed)
        if i==0: io.quickPlot2d((measured-lensed),out_dir+"test2.png")
        
    grad_phi = grad_phi_true
    delensed = lensing.delens_map(measured.copy(), grad_phi, nstep=nstep, order=lens_order, mode="spline", border="cyclic")

    residual = delensed - unlensed
    res_stack += residual

res_stack /= Nstack
p = res_stack*100./unlensed
p[np.abs(p)>100] = np.nan
io.quickPlot2d(p,out_dir+"resstack.png")
io.quickPlot2d((lensed-unlensed)*100./unlensed,out_dir+"lensres.png")

# io.quickPlot2d(res_stack,out_dir+"resstack.png")
# io.quickPlot2d((lensed-unlensed),out_dir+"lensres.png")


""" NOTES
# fmaskCMB_T seems to mess things up


"""
