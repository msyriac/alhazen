from enlib import enmap
from orphics.theory.cosmology import Cosmology
from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
import alhazen.io as aio
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.tools.stats as stats
import enlib.fft as fftfast
import os, sys
import numpy as np
from mpi4py import MPI
import argparse
from ConfigParser import SafeConfigParser 
import cPickle as pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')

parser.add_argument("Nsims", type=int,help='Total number of sims.')
parser.add_argument("Exp", type=str,help='Experiment name.')
parser.add_argument("ExpFilter", type=str,help='Filter Experiment name.')


args = parser.parse_args()
# === PARAMS ===

np.random.seed(rank)

Nsims = args.Nsims
exp_name = args.Exp
expf_name = args.ExpFilter

# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

pol = Config.getboolean("reconstruction","pol")

output_dir = Config.get("general","output_dir")


lmax,tellmin,tellmax,pellmin,pellmax,kellmin,kellmax = aio.ellbounds_from_config(Config,"reconstruction")
gradCut = None
pol_list = ['TT']
debug = False
out_dir = os.environ['WWW']+"plots/halotest/smallpatch_"



# === SIM AND RECON LOOP ===

kappa_stack = {}
apowers = {}
cpowers = {}
    
for polcomb in pol_list:
    kappa_stack[polcomb] = 0.
    apowers[polcomb] = []
    cpowers[polcomb] = []
    

super_dumbs = []    
n0subs = []
# BE CAREFUL WITH THE GORRAM SEEDS!!!!

#mapfile = lambda x: "/home/msyriac/data/sigurdsims/"+exp_name+"_curved_lensed_car_"+str(x).zfill(2)+".fits"
mapfile = lambda x: "/gpfs01/astro/workarea/msyriac/data/sigurd_sims/"+exp_name+"_curved_lensed_car_"+str(x).zfill(2)+".fits"


# === COSMOLOGY ===
cc = Cosmology(lmax=lmax,pickling=True)
theory = cc.theory
TCMB = 2.7255e6

fine_ells = np.arange(2,5000,1)

#for i in range(Nsims):

Nsims = int(50/numcores)
irange = range(rank*Nsims,(rank+1)*Nsims)
for i,k in enumerate(irange):
    print i
    measured = enmap.read_map(mapfile(k))
    measured = measured[0]/TCMB


    
    if i==0:
        shape_dat = measured.shape
        wcs_dat = measured.wcs

        # === ESTIMATOR ===

        template_dat = fmaps.simple_flipper_template_from_enmap(measured.shape,measured.wcs)
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
        

        taper_percent = 15.0
        Ny,Nx = shape_dat
        taper = fmaps.cosineWindow(Ny,Nx,lenApodY=taper_percent*min(Ny,Nx)/100.,lenApodX=taper_percent*min(Ny,Nx)/100.,padY=0,padX=0)
        tapered = measured*taper
        # io.quickPlot2d(tapered,"map.png")
        w2 = np.mean(taper**2.)
        w4 = np.mean(taper**4.)

        
        debug_edges = np.arange(400,8000,150)
        dbinner_dat = stats.bin2D(modlmap_dat,debug_edges)


        # p2d = fmaps.get_simple_power_enmap(tapered)/w2
        # cents,dcltt = dbinner_dat.bin(p2d)
        # cltt = theory.lCl("TT",fine_ells)
        # ucltt = theory.uCl("TT",fine_ells)
        # pl = io.Plotter(scaleY='log')
        # pl.add(cents,dcltt*cents**2.)
        # pl.add(fine_ells,cltt*fine_ells**2.)
        # pl.add(fine_ells,ucltt*fine_ells**2.)
        # pl.done("cls.png")


        
        nT = modlmap_dat*0.
        nP = modlmap_dat*0.
        kbeam_dat = modlmap_dat*0.+1.


        fMaskCMB_T = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellmin,lmax=tellmax)
        fMaskCMB_P = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellmin,lmax=pellmax)
        fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)
        kbeampass = kbeam_dat
            
        qest = Estimator(template_dat,
                         cc.theory,
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
                         loadPickledNormAndFilters=None,
                         savePickledNormAndFilters=None)



    measured = measured * taper
    fkmaps = fftfast.fft(measured,axes=[-2,-1])


    if pol:
        qest.updateTEB_X(fkmaps[0],fkmaps[1],fkmaps[2],alreadyFTed=True)
    else:
        qest.updateTEB_X(fkmaps,alreadyFTed=True)

    qest.updateTEB_Y()


    
    
    for polcomb in pol_list:
        print "Reconstructing",polcomb ," for ", i , " ..."
        kappa_recon = enmap.samewcs(qest.getKappa(polcomb).real,measured)
        # if i==0: io.quickPlot2d(kappa_recon,out_dir+"kappa_recon_single.png")
        kappa_recon -= kappa_recon.mean()

        kpower = fmaps.get_simple_power_enmap(kappa_recon)/w4
        cents_pwr, aclkk = dbinner_dat.bin(kpower)
        apowers[polcomb].append(aclkk)

        m = measured[0] if pol else measured
        data_power_2d_TT = fmaps.get_simple_power_enmap(m)
        sd = qest.N.super_dumb_N0_TTTT(data_power_2d_TT)/w2**2.
        cents_pwr, sdp = dbinner_dat.bin(sd)
        super_dumbs.append(sdp)

        
        
        n0sub = kpower - sd
        cents_pwr, n0subbed = dbinner_dat.bin(n0sub)
        n0subs.append(n0subbed)

        p2d = fmaps.get_simple_power_enmap(tapered)/w2
        cents,dcltt = dbinner_dat.bin(p2d)

        
        pickle.dump((cents_pwr,n0subbed),open(output_dir+exp_name+"_clkk_n0subbed_"+str(k).zfill(2)+".pkl",'wb'))
        pickle.dump((cents_pwr,aclkk),open(output_dir+exp_name+"_rawclkk_"+str(k).zfill(2)+".pkl",'wb'))
        pickle.dump((cents_pwr,sdp),open(output_dir+exp_name+"_superdumbn0_"+str(k).zfill(2)+".pkl",'wb'))
        pickle.dump((cents,dcltt),open(output_dir+exp_name+"_cltt_"+str(k).zfill(2)+".pkl",'wb'))
        # sys.exit()
        
sys.exit()
clkk = theory.gCl("kk",fine_ells)



astats = {}

for polcomb in pol_list:
    astats[polcomb] = stats.getStats(apowers[polcomb])

n0stats = stats.getStats(n0subs)
    
pl = io.Plotter(scaleY='log')
pl.add(fine_ells,clkk,alpha=0.2)
vals = []
for j,polcomb in enumerate(pol_list):
    pl.addErr(cents_pwr,astats[polcomb]['mean'],yerr=astats[polcomb]['errmean'],label=polcomb,ls="none",marker="o",alpha=0.5)
    nlkk2d = qest.N.Nlkk[polcomb]
    cents, nlkk = dbinner_dat.bin(nlkk2d)
    pl.add(cents,nlkk,ls="--",label=polcomb)

pl.legendOn(labsize=8)

vals = np.asarray(vals).ravel().tolist()


pl.addErr(cents_pwr,n0stats['mean'],yerr=n0stats['errmean'],label="N0subbed",ls="none",marker="o")



pl._ax.set_xlim(2,5000)
#pl._ax.set_ylim(min(vals),max(vals))
pl._ax.set_ylim(1.e-10,5.e-7)
pl.done(out_dir+"clkkrecon.png")


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(fine_ells,clkk)
polcomb = "TT"
nlkk2d = qest.N.Nlkk[polcomb]
cents, nlkk = dbinner_dat.bin(nlkk2d)
pl.add(cents,nlkk,ls="--",label=polcomb)
for k,sd in enumerate(super_dumbs):
    pl.add(cents_pwr,sd,ls="-",alpha=0.2,label=str(k))

    
pl.legendOn(labsize=6)

pl._ax.set_xlim(2,5000)
pl._ax.set_ylim(1.e-10,5.e-7)
pl.done(out_dir+"superdumb.png")




