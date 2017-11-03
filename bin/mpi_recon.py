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
from configparser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
import argparse
from mpi4py import MPI

# Runtime params that should be moved to command line
#analysis_section = "analysis"
analysis_section = "analysis_arc"
sim_section = "sims"
expf_name = "experiment_simple"
cosmology_section = "cc_cluster"



# analysis_section = "analysis_sigurd"
# sim_section = "sims_sigurd"
# expf_name = "experiment_simple"
# cosmology_section = "cc_sigurd"
#cosmology_section = "cc_default"

# cluster = False
# liu = False
# sigurd = True
# meanfield_sub = False
# simulated_cmb = False
# simulated_kappa = False
# periodic = False


# cluster = False
# liu = True
# simulated_cmb = True
# simulated_kappa = False
# periodic = True


cluster = True
liu = False
sigurd = False
simulated_cmb = True
simulated_kappa = True
periodic = True

# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("dirname", type=str,help='Directory name.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
parser.add_argument("-s", "--save",     type=str,  default=None)
args = parser.parse_args()
dirname = args.dirname
Nsims = args.nsim
save = args.save


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
out_dir = os.environ['WWW']+"plots/"  # for plots
map_root = os.environ['WORK2']+'/data/sehgal/' # for map inputs
save_dir = map_root + dirname # for saves
if save is not None: save_func = lambda x: save_dir + "/"+save+"_"+str(x).zfill(9)+".fits"

# check for saved kappas and cmbs
kappa_glob = sorted(glob.glob(save_dir+"/kappa*"))
cmb_glob = sorted(glob.glob(save_dir+"/cmb*"))

# How many did I find?
Ntotk = len(kappa_glob)
Ntotc = len(cmb_glob)



# How many sims? Should I use saved files?

if sigurd:
    assert not(liu)
    assert not(simulated_cmb)
    assert not(simulated_kappa)
    assert not(periodic)
    if Nsims is None: Nsims = 320
    #sigurd_cmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/cori/equator_curved_lensed_car_"+str(x).zfill(2)+".fits"
    #sigurd_phi_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/cori/equator_curved_phi_car_"+str(x).zfill(2)+".fits"
    sigurd_cmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/cori/v2/equator_curved_lensed_car_"+str(x).zfill(2)+".fits"
    sigurd_kappa_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/cori/v2/equator_curved_kappa_car_"+str(x).zfill(2)+".fits"

if liu:
    from peakaboo.liuSims import LiuConvergence
    assert simulated_cmb
    assert not(simulated_kappa)
    if Nsims is None: Nsims = 1000
    liucon = LiuConvergence("/gpfs01/astro/workarea/msyriac/data/jiav2/massless/")
    
if (simulated_cmb and simulated_kappa and (Nsims is not None)) or liu or sigurd:
    # If I'm simulating everything and Nsims is specified in the command line
    Ntot = Nsims
    # No need to load saved file names
    cmb_glob = [""]*Ntot
    kappa_glob = [""]*Ntot
else:
    # If I need saved files or if Nsims is not specified, do
    # as many as there are saved files.
    # Directory might be being modified, so only use
    # the lowest number found by all MPI cores
    if rank!=0:
        comm.send(Ntotk,dest=0,tag=99)
        comm.send(Ntotc,dest=0,tag=88)
        Nmin = None
    else:
        Ntotks = [Ntotk]
        Ntotcs = [Ntotc]
        for i in range(1,numcores):
            Ntotks.append(comm.recv(source=i,tag=99))
            Ntotcs.append(comm.recv(source=i,tag=88))
        Nmin = min(min(Ntotks,Ntotcs))
    Nmin = comm.bcast(Nmin,root=0)
    if rank==0: print(("Nmin : ", Nmin))
    cmb_glob = cmb_glob[:Nmin]
    kappa_glob = kappa_glob[:Nmin]
    Ntot = Nmin


# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print(("At most ", max(num_each) , " tasks..."))

# What am I doing?
my_tasks = each_tasks[rank]
my_kappa_files = [kappa_glob[i] for i in my_tasks]
my_cmb_files = [cmb_glob[i] for i in my_tasks]


# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
lb = aio.ellbounds_from_config(Config,"reconstruction",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']
parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=True)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
bin_edges = np.arange(0.,20.,analysis_resolution*2.)
binner_dat = stats.bin2D(parray_dat.modrmap*60.*180./np.pi,bin_edges)
binner_sim = stats.bin2D(parray_sim.modrmap*60.*180./np.pi,bin_edges)
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section)
if cluster: assert cc is not None
parray_dat.add_theory(theory,lmax)
gradCut = 2000 if cluster else 10000
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
                     uEqualsL=not(cluster),
                     gradCut=gradCut,verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None)

    
taper_percent = 30.0 if not(periodic) else 0.
pad_percent = 10.0 if not(periodic) else 0.
# taper_percent = 12.0 if not(periodic) else 0.
# pad_percent = 4.0 if not(periodic) else 0.
Ny,Nx = shape_dat
taper = fmaps.cosineWindow(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))
w2 = np.mean(taper**2.)
w3 = np.mean(taper**3.)
w4 = np.mean(taper**4.)
if rank==0:
    io.quickPlot2d(taper,out_dir+"taper.png")
    print(("w2 : " , w2))

pixratio = analysis_resolution/Config.getfloat(sim_section,"pixel_arcmin")
px_dat = analysis_resolution
if simulated_cmb or simulated_kappa:
    lens_order = Config.getint(sim_section,"lens_order")
    parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
    parray_sim.add_theory(theory,lmax)





random = True if "random" in dirname else False

if random or periodic or not(meanfield_sub):
    meanfield = 0.
else:
    meanfield = np.load("/gpfs01/astro/workarea/msyriac/data/sehgal/randoms_15041256892/reconstack.npy")

k = -1
for index,kappa_file,cmb_file in zip(my_tasks,my_kappa_files,my_cmb_files):
    assert kappa_file[-9:]==cmb_file[-9:]
    
    k += 1
    if rank==0: print(("Rank ", rank , " doing cutout ", index))
    if not(simulated_kappa):
        if liu:
            kappa = liucon.get_kappa(index+1)
        elif sigurd:
            kappa = enmap.read_map(sigurd_kappa_file(index))*taper
            #phi = enmap.read_map(sigurd_phi_file(index))
            #fkphi = fftfast.fft(phi*taper,axes=[-2,-1])
            #lmap = parray_sim.modlmap
            #fkkappa = fkphi * lmap * (lmap+1.)/2.
            #kappa = fftfast.ifft(fkkappa,axes=[-2,-1],normalize=True)
        else:
            hikappa = enmap.ndmap(np.load(kappa_file),wcs_dat)
            kappa = enmap.upgrade(hikappa,pixratio) if abs(pixratio-1.)>1.e-3 else hikappa
        # phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
        # alpha_pix = enmap.grad_pixf(fphi)
    else:
        if k==0:

            if cluster:
                from alhazen.halos import NFWkappa

                massOverh = 2.e14
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

            else:
                kappa = parray_sim.get_kappa(ktype="grf",vary=False)

    if simulated_cmb:
        phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
        #alpha_pix = enmap.grad_pixf(fphi)
        grad_phi = enmap.grad(phi)
            

    if not(simulated_cmb):
        if sigurd:
            cmb = enmap.read_map(sigurd_cmb_file(index))[0]/2.7255e6
            ltt2d = fmaps.get_simple_power_enmap(cmb*taper)
            ccents,ltt = lbinner_dat.bin(ltt2d)/w2
            mpibox.add_to_stats("lcl",ltt)
        else:
            cmb = np.load(cmb_file) / 1.072480e+09
    else:
        if rank==0: print("Generating unlensed CMB...")
        unlensed = parray_sim.get_unlensed_cmb(seed=index)
        if random and simulated_kappa:
            lensed = unlensed
        else:
            if rank==0: print("Lensing...")
            #lensed = lensing.lens_map_flat_pix(unlensed.copy(), alpha_pix.copy(),order=lens_order)
            lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
        if rank==0: print("Downsampling...")
        cmb = lensed if abs(pixratio-1.)<1.e-3 else resample.resample_fft(lensed,shape_dat)
        cmb = enmap.ndmap(cmb,wcs_dat)
        if not(cluster):
            if rank==0: print("Calculating powers for diagnostics...")
            #pxwindow =  fmaps.pixel_window_function(modlmap_dat,angmap_dat,px_dat,px_dat)
            hutt2d = fmaps.get_simple_power_enmap(unlensed*taper)/w2
            hltt2d = fmaps.get_simple_power_enmap(lensed*taper)/w2
            utt2d = fmaps.get_simple_power_enmap(enmap.ndmap(unlensed*taper if abs(pixratio-1.)<1.e-3 else resample.resample_fft(unlensed*taper,shape_dat),wcs_dat))/w2
            ltt2d = fmaps.get_simple_power_enmap(cmb*taper)/w2
            ccents,utt = lbinner_dat.bin(utt2d)
            ccents,ltt = lbinner_dat.bin(ltt2d)
            ccents,hutt = lbinner_sim.bin(hutt2d)
            ccents,hltt = lbinner_sim.bin(hltt2d)
            mpibox.add_to_stats("ucl",utt)
            mpibox.add_to_stats("lcl",ltt)
            mpibox.add_to_stats("hucl",hutt)
            mpibox.add_to_stats("hlcl",hltt)
                

    if rank==0: print("Filtering and binning input kappa...")
    #if not(sigurd):
    kappa = enmap.ndmap(fmaps.filter_map(kappa,kappa*0.+1.,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)
    if cluster:
        cents,kappa1d = binner_sim.bin(kappa)
        mpibox.add_to_stats("input_kappa1d",kappa1d)
    #if not(sigurd):
    mpibox.add_to_stack("input_kappa2d",kappa)
    

    if rank==0: print("Reconstructing...")
    measured = cmb * taper
    fkmaps = fftfast.fft(measured,axes=[-2,-1])
    qest.updateTEB_X(fkmaps,alreadyFTed=True)
    qest.updateTEB_Y()
    with io.nostdout():
        rawkappa = qest.getKappa("TT").real

    tapnorm = taper**2. if cluster else 1.
    kappa_recon = enmap.ndmap(np.nan_to_num(rawkappa/tapnorm)-meanfield,wcs_dat)
    if cluster: kappa_recon[parray_dat.modrmap*180.*60./np.pi>40.] = 0.
    kappa_recon = enmap.ndmap(fmaps.filter_map(kappa_recon,kappa_recon*0.+1.,parray_dat.modlmap,lowPass=kellmax,highPass=kellmin),wcs_dat)
    if save is not None: enmap.write_fits(save_func(index),kappa_recon)
    mpibox.add_to_stack("recon_kappa2d",kappa_recon)
    if not(cluster):
        apower = fmaps.get_simple_power_enmap(enmap1=kappa_recon)/w4


        if True: #not(sigurd):
            if rank==0: print("Downsampling input kappa...")
            downk = enmap.ndmap(kappa  if abs(pixratio-1.)<1.e-3 else resample.resample_fft(kappa,shape_dat),wcs_dat)
            if rank==0: print("Calculating kappa powers and binning...")

        

            cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=downk)/w3
            ipower = fmaps.get_simple_power_enmap(enmap1=downk)/w2
        # else:
        #     lmap = parray_dat.modlmap
        #     cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=phi*taper)*(lmap*(lmap+1.)/2.)/w3
        #     ipower = fmaps.get_simple_power_enmap(enmap1=phi*taper)*(lmap*(lmap+1.)/2.)**2./w2
                
        
        lcents, cclkk = lbinner_dat.bin(cpower)
        lcents, aclkk = lbinner_dat.bin(apower)
        lcents, iclkk = lbinner_dat.bin(ipower)



        mpibox.add_to_stats("cross",cclkk)
        mpibox.add_to_stats("ipower",iclkk)
        mpibox.add_to_stats("auto",aclkk)
    else:
        cents,kapparecon1d = binner_dat.bin(kappa_recon)
        mpibox.add_to_stats("recon_kappa1d",kapparecon1d)

    if rank==0 and index==0:
        io.quickPlot2d(cmb,out_dir+"cmb.png")
        io.quickPlot2d(measured,out_dir+"mcmb.png")
        #if not(sigurd):
        io.quickPlot2d(kappa,out_dir+"inpkappa.png")
        io.quickPlot2d(kappa_recon,out_dir+"reconkappa.png")


mpibox.get_stacks()
mpibox.get_stats()


# if rank==0:
#     pl = io.Plotter(scaleY='log')
#     ellrange = np.arange(2,3000,1)
#     clkk = theory.gCl("kk",ellrange)
#     pl.add(ellrange,clkk)
#     for i in range(Nsims):
#         iclkk = mpibox.vectors['ipower'][i,:]
#         pl.add(lcents,iclkk,alpha=0.1)
#     pl.done(out_dir+"clkk.png")


if rank==0:

    if cluster:
        kappa_stats = mpibox.stats["input_kappa1d"]
        kapparecon_stats = mpibox.stats["recon_kappa1d"]

        pl = io.Plotter()
        sgn = 1 if simulated_cmb else -1
        pl.addErr(cents,kappa_stats['mean'],yerr=kappa_stats['errmean'],ls="-")
        pl.addErr(cents,sgn*kapparecon_stats['mean'],yerr=kapparecon_stats['errmean'],ls="--")
        pl.done(out_dir+"kappa1d.png")

    
        if not(random):
            filename = "/profiles_simcmb_"+str(simulated_cmb)+"_simkap_"+str(simulated_kappa)+"_periodic_"+str(periodic)+".txt"
            np.savetxt(save_dir+filename,np.vstack((cents,kappa_stats['mean'],kappa_stats['errmean'],sgn*kapparecon_stats['mean'], \
                                                    kapparecon_stats['errmean'])).transpose(), \
                       header="# bin centers (arc) , input_kappa, input_kappa_err, recon_kappa, recon_kappa_err")


        io.quickPlot2d(stats.cov2corr(kappa_stats['cov']),out_dir+"kappa_corr.png")
        
    reconstack = mpibox.stacks["recon_kappa2d"]
    io.quickPlot2d(reconstack,out_dir+"reconstack.png")
    if not(sigurd):
        inpstack = mpibox.stacks["input_kappa2d"]
        io.quickPlot2d(inpstack,out_dir+"inpstack.png")
        inp = enmap.ndmap(inpstack if abs(pixratio-1.)<1.e-3 else resample.resample_fft(inpstack,shape_dat),wcs_dat)
        pdiff = np.nan_to_num((inp-reconstack)*100./inp)
        io.quickPlot2d(pdiff,out_dir+"pdiffstack.png",lim=20.)
    np.save(save_dir+"/reconstack",reconstack)


    if not(cluster):
        cstats = mpibox.stats['cross']
        istats = mpibox.stats['ipower']
        astats = mpibox.stats['auto']
        pl = io.Plotter(scaleY='log')
        pl.addErr(lcents,cstats['mean'],yerr=cstats['errmean'],marker="o")
        pl.add(lcents,istats['mean'],marker="x",ls="none")
        lcents,nlkk = lbinner_dat.bin(qest.N.Nlkk['TT'])
        ellrange = np.arange(2,kellmax,1)
        clkk = theory.gCl("kk",ellrange)
        pl.addErr(lcents,astats['mean'],yerr=astats['errmean'],marker="o",alpha=0.5)
        pl.add(lcents,nlkk,ls="--")
        pl.add(ellrange,clkk,color="k")
        pl.done(out_dir+"cpower.png")

        pl = io.Plotter()
        ldiff = (cstats['mean']-istats['mean'])*100./istats['mean']
        lerr = cstats['errmean']*100./istats['mean']
        pl.addErr(lcents,ldiff,yerr=lerr,marker="o",ls="-")
        pl._ax.axhline(y=0.,ls="--",color="k")
        pl.done(out_dir+"powerdiff.png")


    
        iutt2d = theory.uCl("TT",parray_dat.modlmap)
        iltt2d = theory.lCl("TT",parray_dat.modlmap)
        ccents,iutt = lbinner_dat.bin(iutt2d)
        ccents,iltt = lbinner_dat.bin(iltt2d)
        if not(sigurd): uclstats = mpibox.stats["ucl"]
        lclstats = mpibox.stats["lcl"]
        if not(sigurd): huclstats = mpibox.stats["hucl"]
        if not(sigurd): hlclstats = mpibox.stats["hlcl"]

        if not(sigurd): utt = uclstats['mean']
        ltt = lclstats['mean']
        if not(sigurd): utterr = uclstats['errmean']
        ltterr = lclstats['errmean']

        if not(sigurd): hutt = huclstats['mean']
        if not(sigurd): hltt = hlclstats['mean']
        if not(sigurd): hutterr = huclstats['errmean']
        if not(sigurd): hltterr = hlclstats['errmean']


        pl = io.Plotter()

        if not(sigurd):
            pdiff = (hutt-iutt)*100./iutt
            perr = 100.*hutterr/iutt

            pl.addErr(ccents-50,pdiff,yerr=perr,marker="x",ls="none",label="unlensed highres",alpha=0.5)

            pdiff = (hltt-iltt)*100./iltt
            perr = 100.*hltterr/iltt

            pl.addErr(ccents-25,pdiff,yerr=perr,marker="o",ls="none",label="lensed highres",alpha=0.5)



            pdiff = (utt-iutt)*100./iutt
            perr = 100.*utterr/iutt

            pl.addErr(ccents+25,pdiff,yerr=perr,marker="x",ls="none",label="unlensed")

        pdiff = (ltt-iltt)*100./iltt
        perr = 100.*ltterr/iltt

        pl.addErr(ccents+50,pdiff,yerr=perr,marker="o",ls="none",label="lensed")
        pl.legendOn(labsize=10,loc="lower left")
        pl._ax.axhline(y=0.,ls="--",color="k")
        pl.done(out_dir+"clttpdiff.png")



        pl = io.Plotter(scaleY='log',scaleX='log')

        if not(sigurd):
            pl.add(ccents,iutt*ccents**2.)
            pl.addErr(ccents,utt*ccents**2.,yerr=utterr*ccents**2.,marker="x",ls="none",label="unlensed")

        pl.add(ccents,iltt*ccents**2.)
        pl.addErr(ccents,ltt*ccents**2.,yerr=ltterr*ccents**2.,marker="o",ls="none",label="lensed")

        pl.legendOn(labsize=10)
        pl.done(out_dir+"clttp.png")
