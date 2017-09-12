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
analysis_section = "analysis_liu"
sim_section = "sims_liu"


# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("Exp", type=str,help='Experiment name.')
parser.add_argument("mass", type=str,help='massive/massless')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
args = parser.parse_args()
Ntot = args.nsim
mass = args.mass
expf_name = args.Exp

cosmology_section = "cc_jia_"+mass

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
out_dir = os.environ['WWW']+"plots/jiav2_"+mass+"_"+expf_name+"_"  # for plots


from peakaboo.liuSims import LiuConvergence
if Ntot is None: Ntot = 1000
liucon = LiuConvergence("/gpfs01/astro/workarea/msyriac/data/sims/jia/cmb/"+mass+"/")
save_func = lambda x,ftype: "/gpfs01/astro/workarea/msyriac/data/sims/jia/output/"+mass+"_"+ftype+"_"+expf_name+"_"+str(x).zfill(9)+".fits"

# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print "At most ", max(num_each) , " tasks..."

# What am I doing?
my_tasks = each_tasks[rank]


# Read config
iniFile = "../halofg/input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
lb = aio.ellbounds_from_config(Config,"reconstruction_jia",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']
parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=True)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section)
parray_dat.add_theory(theory,lmax)
gradCut = None
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
                     gradCut=gradCut,verbose=False,
                     bigell=9000)

    
pixratio = analysis_resolution/Config.getfloat(sim_section,"pixel_arcmin")
px_dat = analysis_resolution
lens_order = Config.getint(sim_section,"lens_order")
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
parray_sim.add_theory(theory,lmax)


k = -1
for index in my_tasks:
    
    kappa = liucon.get_kappa(index+1) #parray_sim.get_kappa(ktype="grf",vary=False)

    phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
    #alpha_pix = enmap.grad_pixf(fphi)
    grad_phi = enmap.grad(phi)
            
    if rank==0: print "Generating unlensed CMB..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index)
    if rank==0: print "Lensing..."
    #lensed = lensing.lens_map_flat_pix(unlensed.copy(), alpha_pix.copy(),order=lens_order)
    lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)

    if rank==0: print "Beam convolving..."

    flensed = fftfast.fft(lensed,axes=[-2,-1])
    flensed *= parray_sim.lbeam
    lensed = fftfast.ifft(flensed,axes=[-2,-1],normalize=True).real
    if rank==0: print "Adding noise..."
    
    noise = parray_sim.get_noise_sim(seed=index+20000)
    lensed += noise
    
    if rank==0: print "Downsampling..."
    cmb = lensed if abs(pixratio-1.)<1.e-3 else resample.resample_fft(lensed,shape_dat)
    cmb = enmap.ndmap(cmb,wcs_dat)
    if rank==0: print "Calculating powers for diagnostics..."
    utt2d = fmaps.get_simple_power_enmap(enmap.ndmap(unlensed if abs(pixratio-1.)<1.e-3 else resample.resample_fft(unlensed,shape_dat),wcs_dat))
    ltt2d = fmaps.get_simple_power_enmap(cmb)
    ccents,utt = lbinner_dat.bin(utt2d)
    ccents,ltt = lbinner_dat.bin(ltt2d)
    mpibox.add_to_stats("ucl",utt)
    mpibox.add_to_stats("lcl",ltt)
            

    if rank==0: print "Reconstructing..."
    measured = cmb
    fkmaps = fftfast.fft(measured,axes=[-2,-1])
    qest.updateTEB_X(fkmaps,alreadyFTed=True)
    qest.updateTEB_Y()
    with io.nostdout():
        rawkappa = qest.getKappa("TT").real



        
    kappa_recon = enmap.ndmap(rawkappa,wcs_dat)
    print "Saving and calculating powers..."
    enmap.write_fits(save_func(index,"kappa_recon"),kappa_recon)
        
    apower = fmaps.get_simple_power_enmap(enmap1=kappa_recon)

    
    data_power_2d_TT = fmaps.get_simple_power_enmap(measured)
    sd = qest.N.super_dumb_N0_TTTT(data_power_2d_TT)
    lcents,sdp = lbinner_dat.bin(sd)
    np.savetxt(save_func(index,"superdumbn0"),np.vstack((lcents,sdp)).T)
    
    mpibox.add_to_stats("superdumbs",sdp)
    n0subbed = apower - sd
    lcents,rclkk = lbinner_dat.bin(n0subbed)
    np.savetxt(save_func(index,"auto_n0_subbed"),np.vstack((lcents,rclkk)).T)
    mpibox.add_to_stats("auto_n0subbed",rclkk)


    
    if rank==0: print "Downsampling input kappa..."
    downk = enmap.ndmap(kappa  if abs(pixratio-1.)<1.e-3 else resample.resample_fft(kappa,shape_dat),wcs_dat)
    if rank==0: print "Calculating kappa powers and binning..."
    cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=downk)
    ipower = fmaps.get_simple_power_enmap(enmap1=downk)
    lcents, cclkk = lbinner_dat.bin(cpower)
    np.savetxt(save_func(index,"crosspower"),np.vstack((lcents,cclkk)).T)

    lcents, aclkk = lbinner_dat.bin(apower)
    np.savetxt(save_func(index,"autopower"),np.vstack((lcents,aclkk)).T)
    lcents, iclkk = lbinner_dat.bin(ipower)
    np.savetxt(save_func(index,"inputpower"),np.vstack((lcents,iclkk)).T)

    mpibox.add_to_stats("cross",cclkk)
    mpibox.add_to_stats("ipower",iclkk)
    mpibox.add_to_stats("auto",aclkk)

    if rank==0 and index==0:
        io.quickPlot2d(cmb,out_dir+"cmb.png")
        io.quickPlot2d(measured,out_dir+"mcmb.png")
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

    area = unlensed.area()*(180./np.pi)**2.
    print "area: ", area, " sq.deg."
    fsky = area/41250.
    print "fsky: ",fsky
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
    pl._ax.set_ylim(-20.,10.)
    pl.done(out_dir+"powerdiff.png")
    
    iutt2d = theory.uCl("TT",parray_dat.modlmap)
    iltt2d = theory.lCl("TT",parray_dat.modlmap)
    ccents,iutt = lbinner_dat.bin(iutt2d)
    ccents,iltt = lbinner_dat.bin(iltt2d)
    uclstats = mpibox.stats["ucl"]
    lclstats = mpibox.stats["lcl"]

    utt = uclstats['mean']
    ltt = lclstats['mean']
    utterr = uclstats['errmean']
    ltterr = lclstats['errmean']


    pl = io.Plotter()




    pdiff = (utt-iutt)*100./iutt
    perr = 100.*utterr/iutt

    pl.addErr(ccents+25,pdiff,yerr=perr,marker="x",ls="none",label="unlensed")

    pdiff = (ltt-iltt)*100./iltt
    perr = 100.*ltterr/iltt

    pl.addErr(ccents+50,pdiff,yerr=perr,marker="o",ls="none",label="lensed")
    pl.legendOn(labsize=10,loc="lower left")
    pl._ax.axhline(y=0.,ls="--",color="k")
    pl._ax.set_ylim(-20.,20.)
    pl.done(out_dir+"clttpdiff.png")



    pl = io.Plotter(scaleY='log',scaleX='log')

    pl.add(ccents,iutt*ccents**2.)
    pl.addErr(ccents,utt*ccents**2.,yerr=utterr*ccents**2.,marker="x",ls="none",label="unlensed")

    pl.add(ccents,iltt*ccents**2.)
    pl.addErr(ccents,ltt*ccents**2.,yerr=ltterr*ccents**2.,marker="o",ls="none",label="lensed")

    pl.legendOn(labsize=10)
    pl.done(out_dir+"clttp.png")
