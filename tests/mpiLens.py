from enlib import enmap, lensing, powspec, utils
from szar.counts import ClusterCosmology
from alhazen.halos import NFWkappa
from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')

parser.add_argument("Nsims", type=int,help='Total number of sims.')
parser.add_argument("Exp", type=str,help='Experiment name.')
parser.add_argument("-c", "--cluster", action='store_true',help='Simulate a cluster kappa instead of GRF kappa.')
parser.add_argument("-p", "--pol", action='store_true',help='Do polarization.')


args = parser.parse_args()
# === PARAMS ===

np.random.seed(rank)

Nsims = args.Nsims
cluster = args.cluster
exp_name = args.Exp
pol = args.pol

# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)



lens_order = Config.getint("sims","lens_order")

pixratio = Config.getfloat("analysis","pixel_arcmin")/Config.getfloat("sims","pixel_arcmin")


pol = Config.getboolean("reconstruction","pol")
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,"sims","analysis",pol=pol)    
parray_sim = aio.patch_array_from_config(Config,exp_name,shape_sim,wcs_sim,dimensionless=True)
parray_dat = aio.patch_array_from_config(Config,exp_name,shape_dat,wcs_dat,dimensionless=True)
lmax,tellmin,tellmax,pellmin,pellmax,kellmin,kellmax = aio.ellbounds_from_config(Config,"reconstruction")
            

if cluster:
    gradCut = 2000
else:
    gradCut = None

if pol:
    pol_list = ['TT','EB','EE','ET','TE','TB']
else:
    pol_list = ['TT']
    
debug = False

out_dir = os.environ['WWW']+"plots/halotest/smallpatch_"



# === COSMOLOGY ===
cc = ClusterCosmology(lmax=lmax,pickling=True)
parray_sim.add_cosmology(cc)
parray_dat.add_cosmology(cc)
theory = cc.theory

kappa_map = parray_sim.get_kappa(ktype="grf",vary=False)
phi, fphi = lt.kappa_to_phi(kappa_map,parray_sim.modlmap,return_fphi=True)
io.quickPlot2d(kappa_map,out_dir+"kappa_map.png")
io.quickPlot2d(phi,out_dir+"phi.png")
alpha_pix = enmap.grad_pixf(fphi)




# === EXPERIMENT ===

fine_ells = parray_dat.fine_ells

kbeam_sim = parray_sim.lbeam





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


for i in range(Nsims):
    print i
    unlensed = parray_sim.get_unlensed_cmb(seed=(200+i))
    lensed = lensing.lens_map_flat_pix(unlensed.copy(), alpha_pix.copy(),order=lens_order)

    #m, = lensing.rand_map(shape, wcs, ps, lmax=lmax, maplmax=maplmax, seed=(seed,i))

    klteb = enmap.map2harm(lensed.copy())
    klteb_beam = klteb*kbeam_sim
    lteb_beam = enmap.ifft(klteb_beam).real
    noise = parray_sim.get_noise_sim(seed=(300+i))
    observed = lteb_beam + noise
    measured = enmap.downgrade(observed,pixratio)
    if i==0:


        shape_dat, wcs_dat = measured.shape, measured.wcs
        modr_dat = parray_dat.modrmap * 180.*60./np.pi


        # === ESTIMATOR ===

        template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)

        debug_edges = np.arange(200,8000,150)
        dbinner_dat = stats.bin2D(modlmap_dat,debug_edges)

        
        nT = parray_dat.nT
        nP = parray_dat.nP
        # nT[modlmap_dat>tellmax]=np.inf
        # nP[modlmap_dat>pellmax]=np.inf
        # nT[modlmap_dat<tellmin]=np.inf
        # nP[modlmap_dat<pellmin]=np.inf
        kbeam_dat = parray_dat.lbeam


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
                         uEqualsL=not(cluster),
                         gradCut=gradCut,verbose=False,
                         loadPickledNormAndFilters=None,
                         savePickledNormAndFilters=None)



    if i==0 and debug:
        
        teb = enmap.ifft(enmap.map2harm(unlensed)).real
        lteb = enmap.ifft(klteb).real
        if pol:
            io.quickPlot2d(unlensed[0],out_dir+"tmap.png")
            io.quickPlot2d(unlensed[1],out_dir+"qmap.png")
            io.quickPlot2d(unlensed[2],out_dir+"umap.png")
            io.quickPlot2d(teb[1],out_dir+"emap.png")
            io.quickPlot2d(teb[2],out_dir+"bmap.png")
            io.quickPlot2d(lensed[0],out_dir+"ltmap.png")
            io.quickPlot2d(lensed[1],out_dir+"lqmap.png")
            io.quickPlot2d(lensed[2],out_dir+"lumap.png")
            io.quickPlot2d(lteb[1],out_dir+"lemap.png")
            io.quickPlot2d(lteb[2],out_dir+"lbmap.png")
        else:        
            io.quickPlot2d(unlensed,out_dir+"tmap.png")
            io.quickPlot2d(lensed,out_dir+"ltmap.png")

        
        t = teb[0,:,:]
        e = teb[1,:,:]
        b = teb[2,:,:]
        nt = noise[0,:,:]
        ne = noise[1,:,:]
        nb = noise[2,:,:]
        ntt2d = np.nan_to_num(fmaps.get_simple_power_enmap(nt)/kbeam_sim**2.)
        nee2d = np.nan_to_num(fmaps.get_simple_power_enmap(ne)/kbeam_sim**2.)
        nbb2d = np.nan_to_num(fmaps.get_simple_power_enmap(nb)/kbeam_sim**2.)
        
        utt2d = fmaps.get_simple_power_enmap(t)
        uee2d = fmaps.get_simple_power_enmap(e)
        ute2d = fmaps.get_simple_power_enmap(enmap1=t,enmap2=e)
        ubb2d = fmaps.get_simple_power_enmap(b)
        debug_edges = np.arange(tellmin,tellmax,80)
        dbinner = stats.bin2D(parray_sim.modlmap,debug_edges)
        cents, utt = dbinner.bin(utt2d)
        cents, uee = dbinner.bin(uee2d)
        cents, ute = dbinner.bin(ute2d)
        cents, ntt = dbinner.bin(ntt2d)
        cents, nee = dbinner.bin(nee2d)
        cents, nbb = dbinner.bin(nbb2d)
        #cents, ubb = dbinner.bin(ubb2d)


        tl = lteb[0,:,:]
        el = lteb[1,:,:]
        bl = lteb[2,:,:]
        ltt2d = fmaps.get_simple_power_enmap(tl)
        lee2d = fmaps.get_simple_power_enmap(el)
        lte2d = fmaps.get_simple_power_enmap(enmap1=tl,enmap2=el)
        lbb2d = fmaps.get_simple_power_enmap(bl)
        cents, ltt = dbinner.bin(ltt2d)
        cents, lee = dbinner.bin(lee2d)
        cents, lte = dbinner.bin(lte2d)
        cents, lbb = dbinner.bin(lbb2d)


        lcltt, lclee, lclte, lclbb = cmb.unpack_cmb_theory(theory,fine_ells,lensed=True)
        cltt, clee, clte, clbb = cmb.unpack_cmb_theory(theory,fine_ells,lensed=False)

        
        pl = io.Plotter(scaleY='log',scaleX='log')
        pl.add(cents,utt*cents**2.,color="C0",marker="o",ls="none")
        pl.add(cents,uee*cents**2.,color="C1",marker="o",ls="none")
        #pl.add(cents,ubb*cents**2.,color="C2",ls="-")
        pl.add(fine_ells,cltt*fine_ells**2.,color="C0",ls="--")
        pl.add(fine_ells,clee*fine_ells**2.,color="C1",ls="--")
        #pl.add(fine_ells,clbb*fine_ells**2.,color="C2",ls="--")
        pl.done(out_dir+"ccomp.png")

        pl = io.Plotter(scaleX='log')
        pl.add(cents,ute*cents**2.,color="C0",marker="o",ls="none")
        pl.add(fine_ells,clte*fine_ells**2.,color="C0",ls="--")
        pl.done(out_dir+"ccompte.png")

        dbinner2 = stats.bin2D(parray_dat.modlmap,debug_edges)
        cents, ntt2 = dbinner2.bin(nT/parray_dat.lbeam**2.)
        cents, nee2 = dbinner2.bin(nP/parray_dat.lbeam**2.)

        
        pl = io.Plotter(scaleY='log')#,scaleX='log')
        pl.add(fine_ells,lcltt*fine_ells**2.,color="C0",ls="--",lw=3)
        pl.add(fine_ells,lclee*fine_ells**2.,color="C1",ls="--",lw=3)
        pl.add(fine_ells,lclbb*fine_ells**2.,color="C2",ls="--",lw=3)
        pl.add(cents,ltt*cents**2.,color="C0",marker="o",ls="none",markersize=3)
        pl.add(cents,lee*cents**2.,color="C1",marker="o",ls="none",markersize=3)
        pl.add(cents,lbb*cents**2.,color="C2",marker="o",ls="none",markersize=3)
        pl.add(cents,ntt2*cents**2.,color="C3",ls="-.",alpha=0.4)
        pl.add(cents,nee2*cents**2.,color="C4",ls="-.",alpha=0.4)
        pl.add(cents,ntt*cents**2.,color="C0",ls="-.",alpha=0.4)
        pl.add(cents,nee*cents**2.,color="C1",ls="-.",alpha=0.4)
        pl.add(cents,nbb*cents**2.,color="C2",ls="-.",alpha=0.4)
        pl.done(out_dir+"lccomp.png")

        pl = io.Plotter(scaleX='log')
        pl.add(cents,lte*cents**2.,color="C0",ls="-")
        pl.add(fine_ells,lclte*fine_ells**2.,color="C0",ls="--")
        pl.done(out_dir+"lccompte.png")


    fkmaps = fftfast.fft(measured,axes=[-2,-1])


    if pol:
        qest.updateTEB_X(fkmaps[0],fkmaps[1],fkmaps[2],alreadyFTed=True)
    else:
        qest.updateTEB_X(fkmaps,alreadyFTed=True)

    qest.updateTEB_Y()


    
    
    for polcomb in pol_list:
        print "Reconstructing",polcomb ," for ", i , " ..."
        kappa_recon = enmap.samewcs(qest.getKappa(polcomb).real,measured)
        if i==0: io.quickPlot2d(kappa_recon,out_dir+"kappa_recon_single.png")
        kappa_recon -= kappa_recon.mean()

        downk = enmap.downgrade(kappa_map,pixratio)
        kpower = fmaps.get_simple_power_enmap(kappa_recon)
        cents_pwr, aclkk = dbinner_dat.bin(kpower)
        cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=downk)
        cents_pwr, cclkk = dbinner_dat.bin(cpower)
        apowers[polcomb].append(aclkk)
        cpowers[polcomb].append(cclkk)

        kappa_stack[polcomb] += kappa_recon


        if polcomb=="TT":
            m = measured[0] if pol else measured
            data_power_2d_TT = fmaps.get_simple_power_enmap(m)
            sd = qest.N.super_dumb_N0_TTTT(data_power_2d_TT)
            cents_pwr, sdp = dbinner_dat.bin(sd)
            super_dumbs.append(sdp)

            n0sub = kpower - sd
            cents_pwr, n0subbed = dbinner_dat.bin(n0sub)
            n0subs.append(n0subbed)

        

clkk = parray_dat.clkk



astats = {}
cstats = {}

for polcomb in pol_list:
    kappa_stack[polcomb] /= Nsims
    k = kappa_stack[polcomb]
    io.quickPlot2d(k,out_dir+"kappa_recon_"+polcomb+".png")
    astats[polcomb] = stats.getStats(apowers[polcomb])
    cstats[polcomb] = stats.getStats(cpowers[polcomb])

n0stats = stats.getStats(n0subs)
    
pl = io.Plotter(scaleY='log')
pl.add(fine_ells,clkk,alpha=0.2)
vals = []
for j,polcomb in enumerate(pol_list):
    vals.append((cstats[polcomb]['mean']+cstats[polcomb]['errmean']).ravel())
    vals.append((cstats[polcomb]['mean']-cstats[polcomb]['errmean']).ravel())
    pl.addErr(cents_pwr,cstats[polcomb]['mean'],yerr=cstats[polcomb]['errmean'],label=polcomb,ls="none",marker="o",alpha=0.5)
    nlkk2d = qest.N.Nlkk[polcomb]
    cents, nlkk = dbinner_dat.bin(nlkk2d)
    pl.add(cents,nlkk,ls="--",label=polcomb)

pl.legendOn(labsize=8)

vals = np.asarray(vals).ravel().tolist()


pl.addErr(cents_pwr,n0stats['mean'],yerr=n0stats['errmean'],label="N0subbed",ls="none",marker="o")

dbinner_sim = stats.bin2D(parray_sim.modlmap,debug_edges)
kpower2d = fmaps.get_simple_power_enmap(kappa_map)
cents_pwr, input_kappa = dbinner_sim.bin(kpower2d)
pl.add(cents_pwr,input_kappa,marker="x",ls="none",color="black")


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




