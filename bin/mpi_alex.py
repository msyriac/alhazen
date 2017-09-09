import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute, MPIStats
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

# Runtime params that should be moved to command line
expf_name = "experiment_simple"
cosmology_section = "cc_alex"




# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
#parser.add_argument("-s", "--save",     type=str,  default=None)
args = parser.parse_args()
Nsims = args.nsim

analysis_section = "analysis_alex"

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
out_dir = os.environ['WWW']+"plots/alexsims_"  # for plots

# How many sims? Should I use saved files?

if Nsims is None: Nsims = 20
alex_cmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/lensedCMBMaps_set00_"+str(x).zfill(5)+"/order5_periodic_lensedCMB_T_1arcpix_7.fits"
alex_ucmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/lensedCMBMaps_set00_"+str(x).zfill(5)+"/unlensedCMB_T_1arcpix_7.fits"
alex_kappa_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/phiMaps_"+str(x).zfill(5)+"/kappaMap_1arcpix_7.fits"

    
Ntot = Nsims


# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print "At most ", max(num_each) , " tasks..."

# What am I doing?
my_tasks = each_tasks[rank]

if rank==0: print "Reading config..."

# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

if rank==0: print "Params..."

pol = False
shape_dat, wcs_dat = aio.enmap_from_config_section(Config,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
if rank==0:
    print "px: ", analysis_resolution,"shape : ", shape_dat
    area = enmap.area(shape_dat,wcs_dat)*(180./np.pi)**2.
    print "area: ", area, " sq.deg."
    fsky = area/41250.
    print "fsky: ",fsky
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
if rank==0: print "Ell bounds..."

lb = aio.ellbounds_from_config(Config,"reconstruction_alex",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']

if rank==0: print "Patches data..."

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=True,skip_real=True)

if rank==0: print "Attributes..."

lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
#modlmap_dat = enmap.modlmap(shape_dat,wcs_dat)
del lxmap_dat
del lymap_dat
del angmap_dat

if rank==0: print "Binners..."

lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)

if rank==0: print "Cosmology..."

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section)
parray_dat.add_theory(theory,lmax)
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

del lx_dat
del ly_dat

if rank==0: print "Initializing norm..."
with io.nostdout():
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
                     gradCut=10000,verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None,
                     bigell=9000)


px_dat = analysis_resolution
del fMask
del fMaskCMB_T
del fMaskCMB_P
del nT
del nP

k = -1
for index in my_tasks:
    
    k += 1
    if rank==0: print "Rank ", rank , " doing cutout ", index
    kappa = enmap.read_map(alex_kappa_file(index))

    ucmb = enmap.ndmap(enmap.read_map(alex_ucmb_file(index))/2.7255e6,wcs_dat)
    utt2d = fmaps.get_simple_power_enmap(ucmb)
    ccents,utt = lbinner_dat.bin(utt2d)
    mpibox.add_to_stats("ucl",utt)
    del ucmb
    del utt2d


    
    cmb = enmap.ndmap(enmap.read_map(alex_cmb_file(index))/2.7255e6,wcs_dat)
    ltt2d = fmaps.get_simple_power_enmap(cmb)
    ccents,ltt = lbinner_dat.bin(ltt2d)
    mpibox.add_to_stats("lcl",ltt)
    del ltt2d

    

    if rank==0: print "Reconstructing..."
    measured = cmb
    fkmaps = fftfast.fft(measured,axes=[-2,-1])
    qest.updateTEB_X(fkmaps,alreadyFTed=True)
    del fkmaps
    qest.updateTEB_Y()
    with io.nostdout():
        rawkappa = qest.getKappa("TT").real

    kappa_recon = enmap.ndmap(rawkappa,wcs_dat)
    #if save is not None: enmap.write_fits(save_func(index),kappa_recon)

    if rank==0: print "Calculating kappa powers and binning..."

    apower = fmaps.get_simple_power_enmap(enmap1=kappa_recon)

    
    data_power_2d_TT = fmaps.get_simple_power_enmap(measured)
    if rank==0 and index==0:
        io.quickPlot2d(measured,out_dir+"mcmb.png")
    del measured
    del cmb
    
    sd = qest.N.super_dumb_N0_TTTT(data_power_2d_TT)
    del data_power_2d_TT
    
    lcents,sdp = lbinner_dat.bin(sd)
    #np.savetxt(save_func(index,"superdumbn0"),np.vstack((lcents,sdp)).T)
    
    mpibox.add_to_stats("superdumbs",sdp)
    n0subbed = apower - sd
    lcents,rclkk = lbinner_dat.bin(n0subbed)
    #np.savetxt(save_func(index,"auto_n0_subbed"),np.vstack((lcents,rclkk)).T)
    mpibox.add_to_stats("auto_n0subbed",rclkk)


    
    downk = enmap.ndmap(kappa ,wcs_dat)
    cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=downk)
    ipower = fmaps.get_simple_power_enmap(enmap1=downk)
    lcents, cclkk = lbinner_dat.bin(cpower)
    del cpower
    #np.savetxt(save_func(index,"crosspower"),np.vstack((lcents,cclkk)).T)

    lcents, aclkk = lbinner_dat.bin(apower)
    del apower
    #np.savetxt(save_func(index,"autopower"),np.vstack((lcents,aclkk)).T)
    lcents, iclkk = lbinner_dat.bin(ipower)
    del ipower
    #np.savetxt(save_func(index,"inputpower"),np.vstack((lcents,iclkk)).T)

    mpibox.add_to_stats("cross",cclkk)
    mpibox.add_to_stats("ipower",iclkk)
    mpibox.add_to_stats("auto",aclkk)

    if rank==0 and index==0:
        io.quickPlot2d(kappa,out_dir+"inpkappa.png")
        io.quickPlot2d(kappa_recon,out_dir+"reconkappa.png")


mpibox.get_stacks()
mpibox.get_stats()




if rank==0:

    cstats = mpibox.stats['cross']
    istats = mpibox.stats['ipower']
    astats = mpibox.stats['auto']
    rstats = mpibox.stats['auto_n0subbed']
    nstats = mpibox.stats['superdumbs']

    diag = np.sqrt(np.diagonal(astats['cov'])*lcents*np.diff(lbin_edges)*fsky)
    diagr = np.sqrt(np.diagonal(rstats['cov'])*lcents*np.diff(lbin_edges)*fsky)

    pl = io.Plotter(scaleY='log')
    pl.addErr(lcents,cstats['mean'],yerr=cstats['errmean'],marker="o",label="recon x cross")
    pl.add(lcents,istats['mean'],marker="x",ls="none",label="input")
    pl.add(lcents,diag,ls="-.",lw=2,label="diag no n0sub")
    pl.add(lcents,diagr,ls="-.",lw=2,label="diag n0sub")
    lcents,nlkk = lbinner_dat.bin(qest.N.Nlkk['TT'])
    ellrange = np.arange(2,kellmax,1)
    clkk = theory.gCl("kk",ellrange)
    pl.addErr(lcents,astats['mean'],yerr=astats['errmean'],marker="o",alpha=0.5,label="raw")
    pl.addErr(lcents,rstats['mean'],yerr=rstats['errmean'],marker="o",alpha=0.5,label="auto n0subbed")
    pl.add(lcents,nlkk,ls="--",label="theory n0")
    pl.add(lcents,nstats['mean'],ls="--",label="superdumb n0")
    pl.add(ellrange,clkk,color="k")
    pl.legendOn(loc="lower left",labsize=9)
    pl.done(out_dir+"cpower.png")

    io.quickPlot2d(stats.cov2corr(astats['covmean']),out_dir+"corr.png")
    io.quickPlot2d(stats.cov2corr(rstats['covmean']),out_dir+"rcorr.png")

    pl = io.Plotter()
    ldiff = (cstats['mean']-istats['mean'])*100./istats['mean']
    lerr = cstats['errmean']*100./istats['mean']
    pl.addErr(lcents,ldiff,yerr=lerr,marker="o",ls="-")
    pl._ax.axhline(y=0.,ls="--",color="k")
    pl._ax.set_ylim(-5.,5.)
    pl.done(out_dir+"powerdiff.png")
    
    iltt2d = theory.lCl("TT",parray_dat.modlmap)
    ccents,iltt = lbinner_dat.bin(iltt2d)
    lclstats = mpibox.stats["lcl"]

    iutt2d = theory.uCl("TT",parray_dat.modlmap)
    ccents,iutt = lbinner_dat.bin(iutt2d)
    uclstats = mpibox.stats["ucl"]

    ltt = lclstats['mean']
    ltterr = lclstats['errmean']
    utt = uclstats['mean']
    utterr = uclstats['errmean']


    pl = io.Plotter()




    pdiff = (ltt-iltt)*100./iltt
    perr = 100.*ltterr/iltt

    
    pl.addErr(ccents+50,pdiff,yerr=perr,marker="o",ls="none",label="lensed")


    pdiff = (utt-iutt)*100./iutt
    perr = 100.*utterr/iutt

    
    pl.addErr(ccents+50,pdiff,yerr=perr,marker="x",ls="none",label="unlensed")
    
    
    pl.legendOn(labsize=10,loc="lower left")
    pl._ax.axhline(y=0.,ls="--",color="k")
    pl._ax.set_ylim(-5.,5.)
    pl.done(out_dir+"clttpdiff.png")



    pl = io.Plotter(scaleY='log',scaleX='log')

    pl.add(ccents,iltt*ccents**2.)
    pl.addErr(ccents,ltt*ccents**2.,yerr=ltterr*ccents**2.,marker="o",ls="none",label="lensed")

    pl.add(ccents,iutt*ccents**2.)
    pl.addErr(ccents,utt*ccents**2.,yerr=utterr*ccents**2.,marker="x",ls="none",label="unlensed")

    pl.legendOn(labsize=10)
    pl.done(out_dir+"clttp.png")
