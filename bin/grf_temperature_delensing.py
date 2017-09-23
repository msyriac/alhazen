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

import argparse
parser = argparse.ArgumentParser(description='Iteratively delens T with a cluster model that is close to true')
parser.add_argument("Nsims", type=int,help='Total number of sims.')
args = parser.parse_args()
Nsims = args.Nsims

out_dir = os.environ['WWW']+"plots/cluster_"
analysis_section = "analysis_arc"
sim_section = "sims_arc"
expf_name = "experiment_noiseless"
#cosmology_section = "cc_cluster"
cosmology_section = "cc_cluster_high"
#recon_section = "reconstruction_cluster_lowell"
recon_section = "reconstruction_cluster"
lens_order = 5
delens_steps = 5
gradCut = None

Config = io.config_from_file("../halofg/input/recon.ini")

pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=False)
arcmap = parray_sim.modrmap* 180.*60./np.pi

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

lb = aio.ellbounds_from_config(Config,recon_section,min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']


lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)
bin_edges = np.arange(0.,20.,analysis_resolution*2.)
binner_dat = stats.bin2D(parray_dat.modrmap*60.*180./np.pi,bin_edges)


sverif_cmb = SpectrumVerification(mpibox,theory,shape_sim,wcs_sim,lbinner=lbinner_sim,pol=pol)
sverif_kappa = SpectrumVerification(mpibox,theory,shape_dat,wcs_dat,lbinner=lbinner_dat,pol=False)

template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
# if rank==0: io.quickPlot2d(nT,out_dir+"nt.png")
kbeam_dat = parray_dat.lbeam
kbeampass = kbeam_dat
# if rank==0: io.quickPlot2d(kbeampass,out_dir+"kbeam.png")
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
                 bigell=lmax,
                 lEqualsU=False)




kappa = parray_sim.get_kappa("grf")
phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)



for k,index in enumerate(my_tasks):
    if rank==0: print "Rank ", rank, " doing job ",k+1, " / ",len(my_tasks),"..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index,scalar=False)
    luteb,dummy = sverif_cmb.add_power("unlensed",unlensed)

    lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order)
    lensed += parray_dat.get_noise_sim(seed=index+100000)
    llteb,dummy = sverif_cmb.add_power("lensed",lensed)

    

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
        # io.quickPlot2d(pdelensed-unlensed,out_dir+"diffpdelensed.png")
        io.quickPlot2d(kappa_recon,out_dir+"rtt.png")


    lrtt,likk = sverif_kappa.add_power("rttXikk",kappa_recon,imap2=kappa)


    # # BEGIN ITERATIVE DELENSING
    # kappa_iter_recon = kappa_model.copy()

    # niter = 10

    # if rank==0 and k==0:
    #     pl = io.Plotter(scaleY='log')
        
    # fc = enmap.FourierCalc(shape_dat,wcs_dat)
        
    # for j in range(niter):

    #     iqest = Estimator(template_dat,
    #                       dtheory,
    #                       noiseX2dTEB=[nT,nP,nP],
    #                       noiseY2dTEB=[nT,nP,nP],
    #                       fmaskX2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
    #                       fmaskY2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
    #                       fmaskKappa=fMask,
    #                       kBeamX = kbeampass,
    #                       kBeamY = kbeampass,
    #                       TOnly=not(pol),
    #                       halo=True,
    #                       uEqualsL=True,
    #                       gradCut=gradCut,
    #                       bigell=lmax)

        

    #     kappa_iter_recon = enmap.ndmap(fmaps.filter_map(kappa_iter_recon,kappa*0.+1.,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)
        
    #     # convert kappa to alpha
    #     rphitt, rfphitt = lt.kappa_to_phi(kappa_iter_recon,parray_dat.modlmap,return_fphi=True)
    #     rgrad_phitt = enmap.grad(rphitt)

    #     # delens original lensed with current model
    #     delensed = lensing.delens_map(lensed.copy(), rgrad_phitt, nstep=delens_steps, order=lens_order)
    #     delensed = enmap.ndmap(fmaps.filter_map(delensed,delensed*0.+1.,parray_sim.modlmap,lowPass=tellmax,highPass=tellmin),wcs_sim)

    #     # get fft of delensed map and reconstruct
    #     llteb = enmap.fft(delensed,normalize=False)
    #     qest.updateTEB_X(llteb,alreadyFTed=True)
    #     qest.updateTEB_Y(llteb,alreadyFTed=True)
    #     with io.nostdout():
    #         rawkappa = qest.getKappa("TT").real
    #     kappa_recon = enmap.ndmap(rawkappa,wcs_dat)

            
    #     if j==0:
    #         # ps_noise = np.zeros((1,1,modlmap_dat.shape[0],modlmap_dat.shape[1]))
    #         # ps_noise[0,0] = qest.N.Nlkk['TT']
    #         # ngen = enmap.MapGen(shape_dat,wcs_dat,ps_noise)

    #         # cents,kp1d = lbinner_dat.bin(qest.N.Nlkk['TT'])
    #         # pl.add(cents,kp1d,ls="-")

    #         cluster_power = theory.gCl('kk',modlmap_dat)
    #         wiener = cluster_power*np.nan_to_num(1./(cluster_power+qest.N.Nlkk['TT']))
    #         wiener[fMask<1.] = 0.
    #         #wiener = cluster_power*np.nan_to_num(1./(qest.N.Nlkk['TT']))
    #         if rank==0 and k==0:
    #             cents,cluster1d = lbinner_dat.bin(cluster_power)
    #             cents,n1d = lbinner_dat.bin(qest.N.Nlkk['TT'])
    #             pl = io.Plotter(scaleY='log')
    #             pl.add(cents,cluster1d)
    #             pl.add(cents,n1d)
    #             pl.done(out_dir+"cluster1d.png")
    #             io.quickPlot2d(np.fft.fftshift(wiener),out_dir+"wiener2d.png")
    #             cents, wiener1d = lbinner_dat.bin(wiener)
    #             pl = io.Plotter()
    #             pl.add(cents,wiener1d)
    #             pl.done(out_dir+"wiener1d.png")

    #         pass
            
    #     #kappa_recon = ngen.get_map(index+j)
    #     # cents,kp1d = lbinner_dat.bin(fc.power2d(kappa_recon)[0])
    #     # pl.add(cents,kp1d,alpha=0.5,ls="--",label=str(j))
    #     #fwhm = 1.5
    #     #kappa_recon = fmaps.smooth(kappa_recon,modlmap_sim,fwhm)
    #     kappa_recon = enmap.ndmap(fmaps.filter_map(kappa_recon,wiener,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)
    #     arcmax = 5.
    #     conv = np.mean(np.abs(kappa_recon[arcmap<arcmax]/kappa[arcmap<arcmax]))
    #     if rank==0 and k==0: print j,conv*100.

    #     # update model with residual
    #     kappa_iter_recon = kappa_iter_recon + kappa_recon
    #     #kappa_iter_recon = kappa_model + kappa_recon
        
    #     if rank==0 and k==0:
    #         io.quickPlot2d(kappa_iter_recon,out_dir+"rtt_itertot_"+str(j)+".png")
    #         io.highResPlot2d(delensed,out_dir+"iterdelensed_"+str(j)+".png")
    #         io.quickPlot2d(kappa_recon,out_dir+"rtt_iterinst_"+str(j)+".png")

    # # if rank==0 and k==0:
    # #     pl.legendOn()
    # #     pl.done(out_dir+"n1ds.png")
    
    # mpibox.add_to_stack("kappaiterrecon",kappa_iter_recon)
    # rcents, recon1d = binner_dat.bin(kappa_iter_recon)
    # mpibox.add_to_stats("kappaiterrecon1d",recon1d)
    

    
mpibox.get_stats()
mpibox.get_stacks()
if rank==0:

    io.quickPlot2d(mpibox.stacks['kapparecon'],out_dir+"reconstack.png")
    # io.quickPlot2d(mpibox.stacks['kappaiterrecon'],out_dir+"iterreconstack.png")
    
    plot_list = ["unlensed","lensed"]#,"pdelensed"]
    spec_list = ["TT"]
    for spec in spec_list:
        sverif_cmb.plot(spec,plot_list,out_dir+"cl"+spec+".png")
        sverif_cmb.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")

    spec = "kk"
    plot_list = ['rttXikk']
    sverif_kappa.plot(spec,plot_list,out_dir+"cl"+spec+".png",scale_spectrum=False)
    sverif_kappa.plot_diff(spec,plot_list,out_dir+"cl"+spec+"diff.png")


    inpkappa = enmap.ndmap(fmaps.filter_map(kappa,kappa*0.+1.,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)

    rcents, inpkappa1d = binner_dat.bin(inpkappa)
    kapparecon_stats = mpibox.stats["kapparecon1d"]
    # kappaiterrecon_stats = mpibox.stats["kappaiterrecon1d"]

    pl = io.Plotter(scaleX='log',scaleY='log')
    pl.add(rcents,inpkappa1d)
    pl.addErr(rcents,kapparecon_stats['mean'],yerr=kapparecon_stats['errmean'],ls="-")
    # pl.addErr(rcents,kappaiterrecon_stats['mean'],yerr=kappaiterrecon_stats['errmean'],ls="--")
    pl._ax.set_xlim(0.1,10.)
    pl._ax.set_ylim(0.001,0.63)
    pl.done(out_dir+"kappa1d.png")


    
    pl = io.Plotter()
    
    rec = kapparecon_stats['mean']
    inp = inpkappa1d
    recerr = kapparecon_stats['errmean']
    diff = (rec-inp)/inp
    differr = recerr/inp
    
    pl.addErr(rcents,diff,yerr=differr,marker="o",ls="-")

    # rec = kappaiterrecon_stats['mean']
    # inp = inpkappa1d
    # recerr = kappaiterrecon_stats['errmean']
    # diff = (rec-inp)/inp
    # differr = recerr/inp
    
    # pl.addErr(rcents,diff,yerr=differr,marker="d",ls="--")

    
    pl.legendOn(labsize=8,loc="lower right")
    pl.hline()
    pl._ax.set_ylim(-0.30,0.10)
    pl._ax.set_xlim(0.,10.)
    pl.done(out_dir+"diffper.png")

