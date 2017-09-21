import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute, MPIStats, SpectrumVerification
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
import warnings
import logging
logger = logging.getLogger()
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from enlib import enmap, lensing, resample
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
import argparse
from mpi4py import MPI

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


out_dir = "./"
analysis_section = "analysis"
sim_section = "sims"
expf_name = "experiment_noiseless"
cosmology_section = "cc_default"
lens_order = 5
delens_steps = 3
gradCut = None

Config = io.config_from_file("../halofg/input/recon.ini")

pol = True
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=False)


Nsims = 15
# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Nsims,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
# What am I doing?
my_tasks = each_tasks[rank]

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section,dimensionless=False)
parray_dat.add_theory(theory,lmax,orphics_is_dimensionless=False)
parray_sim.add_theory(theory,lmax,orphics_is_dimensionless=False)

lb = aio.ellbounds_from_config(Config,"reconstruction_pol",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']


lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
kellmin = 200
kellmax = 6000
lbin_edges = np.arange(kellmin,kellmax,50)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)


sverif_cmb = SpectrumVerification(mpibox,theory,shape_sim,wcs_sim,lbinner=lbinner_sim,pol=True)
sverif_kappa = SpectrumVerification(mpibox,theory,shape_dat,wcs_dat,lbinner=lbinner_dat,pol=False)

template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
if rank==0: io.quickPlot2d(nT,out_dir+"nt.png")
kbeam_dat = parray_dat.lbeam
kbeampass = kbeam_dat
if rank==0: io.quickPlot2d(kbeampass,out_dir+"kbeam.png")
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
                 kBeamX = kbeampass,
                 kBeamY = kbeampass,
                 doCurl=False,
                 TOnly=not(pol),
                 halo=True,
                 uEqualsL=True,
                 gradCut=gradCut,verbose=False,
                 bigell=lmax)


clkk2d = theory.gCl('kk',modlmap_dat)


for k,index in enumerate(my_tasks):
    if rank==0: print "Rank ", rank, " doing job ",k+1, " / ",len(my_tasks),"..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index,scalar=False)
    luteb,dummy = sverif_cmb.add_power("unlensed",unlensed)

    kappa = parray_sim.get_kappa(ktype="grf",vary=False,seed=index+1000000)
    phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
    grad_phi = enmap.grad(phi)
    lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order)
    llteb,dummy = sverif_cmb.add_power("lensed",lensed)


    pdelensed = lensing.delens_map(lensed, grad_phi, nstep=delens_steps, order=lens_order)
    lpteb,dummy = sverif_cmb.add_power("pdelensed",pdelensed)


    qest.updateTEB_X(llteb[0],llteb[1],llteb[2],alreadyFTed=True)
    qest.updateTEB_Y(alreadyFTed=True)
    with io.nostdout():
        rawkappa_TT = qest.getKappa("TT").real
        rawkappa_EB = qest.getKappa("EB").real

    kappa_recon_TT = enmap.ndmap(rawkappa_TT,wcs_dat)
    kappa_recon_EB = enmap.ndmap(rawkappa_EB,wcs_dat)

    if rank==0 and k==0:
        io.quickPlot2d(kappa_recon_TT,out_dir+"rtt.png")
        io.quickPlot2d(kappa_recon_EB,out_dir+"reb.png")

    # rphitt, rfphitt = lt.kappa_to_phi(kappa_recon_TT,parray_dat.modlmap,return_fphi=True)
    # rgrad_phitt = enmap.grad(rphitt)

    # rphieb, rfphieb = lt.kappa_to_phi(kappa_recon_EB,parray_dat.modlmap,return_fphi=True)
    # rgrad_phieb = enmap.grad(rphieb)

    lrtt,lreb = sverif_kappa.add_power("rttXreb",kappa_recon_TT,imap2=kappa_recon_EB)
    lrtt,likk = sverif_kappa.add_power("rttXikk",kappa_recon_TT,imap2=kappa)
    lreb,likk = sverif_kappa.add_power("rebXikk",kappa_recon_EB,imap2=kappa)

    # delens TT with EB and vice versa

    rtt2d = sverif_kappa.fcalc.f2power(lrtt,lrtt,pixel_units=False)
    reb2d = sverif_kappa.fcalc.f2power(lreb,lreb,pixel_units=False)

    thntt2d = qest.N.Nlkk['TT']
    thneb2d = qest.N.Nlkk['EB']

    wienerTT = clkk2d/(clkk2d+thntt2d)
    wienerEB = clkk2d/(clkk2d+thneb2d)


    if rank==0 and k==0:
        io.quickPlot2d(np.fft.fftshift(rtt2d),out_dir+"rtt2d.png")
        io.quickPlot2d(np.fft.fftshift(reb2d),out_dir+"reb2d.png")

        finells = np.arange(2,kellmax,1)
        clkk1d = theory.gCl('kk',finells)
        ntt2d = rtt2d - clkk2d
        neb2d = reb2d - clkk2d

        cents, rtt1d = lbinner_dat.bin(rtt2d)
        cents, reb1d = lbinner_dat.bin(reb2d)
        cents, ntt1d = lbinner_dat.bin(ntt2d)
        cents, neb1d = lbinner_dat.bin(neb2d)

        cents, thntt1d = lbinner_dat.bin(thntt2d)
        cents, thneb1d = lbinner_dat.bin(thneb2d)
        
        pl = io.Plotter(scaleX='linear',scaleY='log')
        pl.add(finells,clkk1d,color="k")

        pl.add(cents,rtt1d,ls="none",marker="o",alpha=0.1,label="clkk+ntt",color="C0")
        pl.add(cents,reb1d,ls="none",marker="o",alpha=0.1,label="clkk+neb",color="C1")

        pl.add(cents,ntt1d,ls="-.",label="data ntt",color="C0")
        pl.add(cents,neb1d,ls="-.",label="data neb",color="C1")
        pl.add(cents,thntt1d,ls="--",label="theory ntt",color="C0")
        pl.add(cents,thneb1d,ls="--",label="theory neb",color="C1")

        pl.legendOn(loc='lower left',labsize=10)
        pl._ax.set_xlim(2,kellmax)
        pl._ax.set_ylim(1.e-9,1.e-6)
        pl.done(out_dir+"clkkcomp.png")
    

    
    

    
mpibox.get_stats()
if rank==0:
    plot_list = ["unlensed","lensed","pdelensed"]
    spec_list = ["TT","EE","BB","TE"]
    for spec in spec_list:
        sverif_cmb.plot(spec,plot_list,out_dir+"cl"+spec+".png")
        sverif_cmb.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")

    spec = "kk"
    plot_list = ['rttXreb','rttXikk','rebXikk']
    sverif_kappa.plot(spec,plot_list,out_dir+"cl"+spec+".png",scale_spectrum=False)
    sverif_kappa.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")
