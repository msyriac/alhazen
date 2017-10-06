import numpy as np
import sys, os
from orphics.analysis.pipeline import mpi_distribute, MPIStats, SpectrumVerification
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from enlib import enmap, lensing, resample
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
import enlib.fft as fftfast
import argparse
from alhazen.quadraticEstimator import Estimator
from mpi4py import MPI


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

import argparse
parser = argparse.ArgumentParser(description='Verify lensing reconstruction with sims. Calculate N1 TTTT from sims')
parser.add_argument("Out", type=str,help='Output Directory Name (not path, that\'s specified in ini')
parser.add_argument("Recon", type=str,help='Recon section')
parser.add_argument("Exp", type=str,help='Experiment section')
parser.add_argument("-s", "--covseed",     type=int,  default=0)
parser.add_argument("-g", "--grad_cut",     type=int,  default=None)
parser.add_argument("-N", "--num_sims",     type=int,  default=None, help="If you don't want to use all the sims, specify the number here.")
parser.add_argument("-n", "--N1", action='store_true',help='Do N1.')
parser.add_argument("-p", "--paper", action='store_true',help='Paper specific plots.')
parser.add_argument("-u", "--unlensed", action='store_true',help='Make unlensed.')

args = parser.parse_args()
doN1 = args.N1
paper = args.paper
unlensed = args.unlensed

pout_dir = os.environ['WWW']+"plots/sims_"+args.Out+"_g_"+str(args.grad_cut)+"_cseed_"+str(args.covseed)+"_"+args.Exp+"_"+args.Recon+"_unlensed_"+str(args.unlensed)+"_"

rConfig = io.config_from_file("../halofg/input/recon.ini")
recon_section = args.Recon
expf_name = args.Exp
fout_dir = rConfig.get('general','output_dir')+args.Out+"/"


Config = io.config_from_file(fout_dir+"config.ini")
analysis_section = Config.get("general","analysis")
sim_section = Config.get("general","sims")
cosmology_section = Config.get("general","cosmology")
Nsims = Config.getint("general","Nsims")
Nsets = Config.getint("general","Nsets")
polsims = Config.getboolean("general","pol")
pol = False

shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
pixratio = analysis_resolution/Config.getfloat(sim_section,"pixel_arcmin")
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)

parray_dat = aio.patch_array_from_config(rConfig,expf_name,shape_dat,wcs_dat,dimensionless=False)

# Efficiently distribute sims over MPI cores
Nuse = Nsims if args.num_sims is None else args.num_sims
assert args.num_sims < Nsims
num_each,each_tasks = mpi_distribute(Nuse,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
# What am I doing?
my_tasks = each_tasks[rank]

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section,dimensionless=False)
parray_dat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)



lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lb = aio.ellbounds_from_config(rConfig,recon_section,min_ell*8)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']

lbin_edges = np.arange(kellmin,kellmax,600)
#lbin_edges = np.logspace(np.log10(kellmin),np.log10(kellmax),50)
lbinner = stats.bin2D(modlmap_dat,lbin_edges)


template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
kbeam_dat = parray_dat.lbeam
kbeampass = kbeam_dat
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
                     gradCut=args.grad_cut,bigell=lmax)

covseed = 0


def l(cseed,kseed,returnk=False,index=None):
    cname = fout_dir+"lensed_covseed_"+str(args.covseed).zfill(3)+"_cmbseed_"+str(cseed).zfill(5)+"_kseed_"+str(kseed).zfill(5)+".hdf"
    if unlensed:
        seedroot = (covseed)*Nsets*Nsims
        lensedt = parray_dat.get_unlensed_cmb(seed=seedroot+cseed,scalar=False)
    else:
        lensedt = enmap.read_map(cname)[0] if polsims else enmap.read_map(cname)
    # -- add beam and noise if you want --
    if "noiseless" not in expf_name:
        assert index is not None
        if rank==0: print "Adding beam..."
        flensed = fftfast.fft(lensedt,axes=[-2,-1])
        flensed *= parray_dat.lbeam
        lensedt = fftfast.ifft(flensed,axes=[-2,-1],normalize=True).real
        if rank==0: print "Adding noise..."
        seedroot = (covseed+1)*Nsets*Nsims # WARNING: noise sims will be correlated with CMB from the next covseed
        nseed = seedroot+index
        noise = parray_dat.get_noise_sim(seed=nseed)
        if paper:
            cents, noise1d = lbinner.bin(power(noise)[0])
            mpibox.add_to_stats('noisett',noise1d)        
        lensedt += noise
        
    lensedt = enmap.ndmap(lensedt,wcs_dat)
    
    if returnk:
        kname = fout_dir+"kappa_covseed_"+str(args.covseed).zfill(3)+"_kseed_"+str(kseed).zfill(5)+".hdf"
        return lensedt,enmap.read_map(kname)
    else:
        return lensedt

def qe(xmap,ymap=None,fted=False):
    qest.updateTEB_X(xmap,alreadyFTed=fted)
    if ymap is not None:
        qest.updateTEB_Y(ymap,alreadyFTed=fted)
    else:
        qest.updateTEB_Y(alreadyFTed=fted)
    return qest.getKappa('TT')

fc = enmap.FourierCalc(shape_dat,wcs_dat)
def power(map1,map2=None):
    return fc.power2d(map1,map2)

def powerf1f2(kmap1,kmap2):
    return fc.f2power(kmap1,kmap2)

def powerf1(map1,kmap2):
    return fc.f1power(map1,kmap2)



for k,index in enumerate(my_tasks):

    i = index

    if rank==0: print "Rank ", rank, " doing task ",k, " / ",len(my_tasks), " ... "

    S0,inpk = l(4*Nsims+i,5*Nsims+i,returnk=True,index=i) 
    if doN1:
        # === N1 calculation ===
        if rank==0: print "N0MC ... "
        n1 = 0.
        S = l(i,Nsims+i,index=i) 
        Sp = l(2*Nsims+i,3*Nsims+i,index=i) 
        # S and Sp have different unlensed CMB and different kappa
        kappa = qe(S,Sp)
        kappap = qe(Sp,S)
        pwr,k1,k1 = power(kappa,kappa)
        pwrp,kp = powerf1(kappap,k1)
        n0mc2d = pwr+pwrp
        cents, p1d = lbinner.bin(n0mc2d)
        mpibox.add_to_stats('n0mc',p1d)
        n1 += -n0mc2d
        if rank==0: print "N1MC ... "
        # ALREADY DONE ABOVE: S0,inpk = l(4*Nsims+i,5*Nsims+i,returnk=True,index=i) 
        Sp,inpk2 = l(6*Nsims+i,5*Nsims+i,returnk=True,index=i)
        assert np.all(np.isclose(inpk,inpk2))
        # S and Sp have different unlensed CMB but same kappa
        kappa = qe(S0,Sp)
        kappap = qe(Sp,S0)
        pwr,k1,k1 = power(kappa,kappa)
        pwrp,kp = powerf1(kappap,k1)
        n1 += (pwr+pwrp)
        mpibox.add_to_stack('n1',n1)
        # === N1 calculation done ===

    # kappa verification
    if rank==0: print "Kappa verification ... "
    kappa = qe(S0)
    if rank==0: print "Powers ... "
    ixi,ki,ki = power(inpk)
    ixr,kr = powerf1(kappa,ki)
    rxr_raw = powerf1f2(kr,kr)
    if rank==0: print "Binning ... "
    cents, ixi1d = lbinner.bin(ixi)
    cents, ixr1d = lbinner.bin(ixr)
    cents, rxr_raw1d = lbinner.bin(rxr_raw)
    mpibox.add_to_stats('ixi',ixi1d)
    mpibox.add_to_stats('ixr',ixr1d)
    mpibox.add_to_stats('rxr_raw',rxr_raw1d)

    # sdn0
    if rank==0: print "SDN0 ... "
    lpower,ks,ks = power(S0,S0)
    sd = qest.N.super_dumb_N0_TTTT(lpower)
    lcents,sdp = lbinner.bin(sd)
    mpibox.add_to_stats("superdumbs",sdp)
    n0subbed = rxr_raw - sd
    lcents,rxr = lbinner.bin(n0subbed)
    mpibox.add_to_stats("rxr",rxr)
    mpibox.add_to_stack("rxr2d",n0subbed)
    

    if rank==0: print rank,index,len(my_tasks)

mpibox.get_stacks()
mpibox.get_stats()
if rank==0:
    if doN1: n12d = mpibox.stacks['n1']
    rxr2d = mpibox.stacks['rxr2d']

    if doN1: 
        unb2d = rxr2d-n12d
        cents,unb1d = lbinner.bin(unb2d)
    
        cents,n11d = lbinner.bin(n12d)

    clkk2d = theory.gCl('kk',modlmap_dat)
    tot2d = qest.N.Nlkk['TT']+clkk2d
    cents,tot1d = lbinner.bin(tot2d)

    norm2d = qest.N.Nlkk['TT']
    cents,norm1d = lbinner.bin(norm2d)

    
    
    if doN1: n0mc = mpibox.stats['n0mc']
    ixi = mpibox.stats['ixi']
    ixr = mpibox.stats['ixr']
    rxr_raw = mpibox.stats['rxr_raw']
    rxr = mpibox.stats['rxr']
    sdn0 = mpibox.stats['superdumbs']


    area = S0.area()*(180./np.pi)**2.
    print "area: ", area, " sq.deg."
    fsky = area/41250.
    print "fsky: ",fsky
    diagraw = np.sqrt(np.diagonal(rxr_raw['cov'])*cents*np.diff(lbin_edges)*fsky)
    diagn0sub = np.sqrt(np.diagonal(rxr['cov'])*cents*np.diff(lbin_edges)*fsky)
    idiag = np.sqrt(np.diagonal(ixr['cov'])*cents*np.diff(lbin_edges)*fsky)
    cents,clkk1d = lbinner.bin(clkk2d)
    idiag_est = clkk1d+(3.*(idiag**2.-clkk1d**2.)/clkk1d)

    
    io.quickPlot2d(stats.cov2corr(rxr['covmean']),pout_dir+"rcorr.png")
    io.quickPlot2d(stats.cov2corr(rxr_raw['covmean']),pout_dir+"corr.png")

    # if paper and not(unlensed):
    #     np.save(pout_dir+"covmat_"+str(area)+"sqdeg.npy",rxr['cov'])
    #     np.save(pout_dir+"lbin_edges_"+str(area)+"sqdeg.npy",lbin_edges)
    #     import cPickle as pickle
    #     pickle.dump((cents,mpibox.stats['noisett']['mean']),open(pout_dir+"noise.pkl",'wb'))
    
    ellrange = np.arange(2,kellmax,1)
    clkk = theory.gCl('kk',ellrange)
    pl = io.Plotter(scaleY='log',labelX="$L$",labelY="$C^{\\kappa\\kappa}_L$")#,scaleX='log')
    pl.add(ellrange,clkk,color="k",lw=3)
    io.save_cols(pout_dir+"_plot_clkk_theory.txt",(ellrange,clkk))
    pl.addErr(cents-140,ixr['mean'],yerr=ixr['errmean'],ls="none",marker="o",label='Input x Reconstruction')
    io.save_cols(pout_dir+"_plot_inputXrecon.txt",(cents,ixr['mean'],ixr['errmean']))
    pl.addErr(cents,rxr_raw['mean'],yerr=rxr_raw['errmean'],ls="none",marker="o",label='Total Autospectrum',alpha=0.5)
    io.save_cols(pout_dir+"_plot_reconXrecon_raw.txt",(cents,rxr_raw['mean'],rxr_raw['errmean']))
    if not(paper): pl.addErr(cents,rxr['mean'],yerr=rxr['errmean'],ls="none",marker="o",label='rxr sdn0')
    # pl.addErr(cents,ixi['mean'],yerr=ixi['errmean'],ls="none",marker="x",color="k",label='Input x Reconstruction')
    #pl.add(cents,ixi['mean'],ls="none",marker="x",color="k",label='Input x Input')
    if not(paper) and doN1: pl.add(cents+140,unb1d,marker="o",label='Debiased Reconstruction Autospectrum',ls="none")
    if not(paper): pl.add(cents,diagraw,ls="--",marker="^",label="diag raw",alpha=0.5)
    pl.add(cents,diagn0sub,ls="--",marker="^",label="Variance of Power")
    io.save_cols(pout_dir+"_plot_varPower.txt",(cents,diagn0sub))
    #pl.add(cents,idiag_est,ls="--",marker="^",label="Variance of Power from Cross")
    #pl.add(cents,diagcross,ls="--",marker="^",label="diag cross")

    if not(paper): pl.add(cents,norm1d,ls="-.",label='theory n0')
    if not(paper): pl.add(cents,sdn0['mean'],ls="--",label='sdn0')
    if not(paper) and doN1: pl.add(cents,n0mc['mean'],marker="^",ls="none",alpha=0.5,label='n0mc')
    pl.add(cents,tot1d,ls="--",label='Theory Total Power')
    io.save_cols(pout_dir+"_plot_totPower.txt",(cents,tot1d))
    if not(paper) and doN1: pl.add(cents,n11d,ls="-.",label='Montecarlo N1 Bias')
    if "small" in args.Out:
        pl._ax.set_ylim(1.e-11,1.e-7)
    else:
        pl._ax.set_ylim(1.e-9 ,1.e-6)
    if paper:
        pl._ax.set_xlim(2000,kellmax)
    else:
        pl._ax.set_xlim(kellmin,kellmax)
    pl.legendOn('lower left')
    filetype = "pdf" if paper else "png"
    pl.done(pout_dir+"clkkverify."+filetype)


    pl = io.Plotter()
    diff = (ixr['mean']-ixi['mean'])/ixi['mean']
    differr = (ixr['errmean'])/ixi['mean']
    pl.addErr(cents,diff,yerr=differr)
    if doN1:
        diff = (unb1d-ixi['mean'])/ixi['mean']
        pl.add(cents,diff,marker="o")
    pl.hline()
    pl._ax.set_xlim(kellmin,kellmax)
    pl._ax.set_ylim(-0.1,0.05)
    pl.done(pout_dir+"clkkverifydiff.png")
