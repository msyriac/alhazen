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

def get_nfw(massOverh):
    from alhazen.halos import NFWkappa

    #massOverh = 2.e14
    zL = 0.7
    overdensity = 180.
    critical = False
    atClusterZ = False
    concentration = 3.2
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,parray_sim.modrmap* 180.*60./np.pi,winAtLens,
                              overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return kappa

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

import argparse
parser = argparse.ArgumentParser(description='Iteratively delens T with a cluster model that is close to true')
parser.add_argument("Nsims", type=int,help='Total number of sims.')
args = parser.parse_args()
Nsims = args.Nsims

out_dir = "./cluster_"
analysis_section = "analysis_arc"
sim_section = "sims_arc"
expf_name = "experiment_noiseless"
cosmology_section = "cc_cluster"
lens_order = 5
delens_steps = 3
gradCut = None

Config = io.config_from_file("../halofg/input/recon.ini")

pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=False)


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

lb = aio.ellbounds_from_config(Config,"reconstruction_cluster_lowell",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']


lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
lbin_edges = np.arange(kellmin,kellmax,50)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)


sverif_cmb = SpectrumVerification(mpibox,theory,shape_sim,wcs_sim,lbinner=lbinner_sim,pol=pol)
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
                 uEqualsL=False,
                 gradCut=gradCut,verbose=False,
                 bigell=lmax)


clkk2d = theory.gCl('kk',modlmap_dat)


kappa = get_nfw(2.e14)
phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)

kappa_model = 0.9*kappa.copy()


for k,index in enumerate(my_tasks):
    if rank==0: print "Rank ", rank, " doing job ",k+1, " / ",len(my_tasks),"..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index,scalar=False)
    luteb,dummy = sverif_cmb.add_power("unlensed",unlensed)

    lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order)
    llteb,dummy = sverif_cmb.add_power("lensed",lensed)


    pdelensed = lensing.delens_map(lensed, grad_phi, nstep=delens_steps, order=lens_order)
    lpteb,dummy = sverif_cmb.add_power("pdelensed",pdelensed)


    qest.updateTEB_X(llteb,alreadyFTed=True)
    qest.updateTEB_Y(alreadyFTed=True)
    with io.nostdout():
        rawkappa = qest.getKappa("TT").real

    kappa_recon = enmap.ndmap(rawkappa,wcs_dat)
    mpibox.add_to_stack("kapparecon",kappa_recon)

    if rank==0 and k==0:
        io.quickPlot2d(kappa,out_dir+"kappa.png")
        io.quickPlot2d(kappa_model,out_dir+"kappamodel.png")
        io.quickPlot2d(unlensed,out_dir+"unlensed.png")
        io.quickPlot2d(lensed-unlensed,out_dir+"difflensed.png")
        io.quickPlot2d(pdelensed-unlensed,out_dir+"diffpdelensed.png")
        io.quickPlot2d(kappa_recon,out_dir+"rtt.png")

    # rphitt, rfphitt = lt.kappa_to_phi(kappa_recon_TT,parray_dat.modlmap,return_fphi=True)
    # rgrad_phitt = enmap.grad(rphitt)

    # rphieb, rfphieb = lt.kappa_to_phi(kappa_recon_EB,parray_dat.modlmap,return_fphi=True)
    # rgrad_phieb = enmap.grad(rphieb)

    lrtt,likk = sverif_kappa.add_power("rttXikk",kappa_recon,imap2=kappa)


    

    
    

    
mpibox.get_stats()
mpibox.get_stacks()
if rank==0:

    io.quickPlot2d(mpibox.stacks['kapparecon'],out_dir+"reconstack.png")
    
    plot_list = ["unlensed","lensed","pdelensed"]
    spec_list = ["TT"]
    for spec in spec_list:
        sverif_cmb.plot(spec,plot_list,out_dir+"cl"+spec+".png")
        sverif_cmb.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")

    spec = "kk"
    plot_list = ['rttXikk']
    sverif_kappa.plot(spec,plot_list,out_dir+"cl"+spec+".png",scale_spectrum=False)
    sverif_kappa.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")
