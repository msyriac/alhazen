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

# === PARAMS ===

Nsims = 200

sim_pixel_scale = 1.0
analysis_pixel_scale = 1.0
patch_width_arcmin = 25.*60.
cluster = False

# sim_pixel_scale = 0.1
# analysis_pixel_scale = 0.2
# patch_width_arcmin = 70.
# cluster = True

lens_order = 3
maxlike = True

periodic = True

beam_arcmin = 0.1 #1.0
noise_T_uK_arcmin = 0.01
noise_P_uK_arcmin = 0.01
lmax = 8000
tellmax = 8000
pellmax = 8000
tellmin = 200
pellmin = 200
kellmax = 8500
kellmin = 200
gradCut = 10000
#pol_list = ['TT','EB','EE','ET','TE']
pol_list = ['TT','EB']
debug = True

out_dir = os.environ['WWW']+"plots/halotest/smallpatch_"



# === COSMOLOGY ===
cc = ClusterCosmology(lmax=lmax,pickling=True)
TCMB = 2.7255e6
theory = cc.theory


# === TEMPLATE MAPS ===

pol = False if pol_list==['TT'] else True

fine_ells = np.arange(0,lmax,1)
bin_edges = np.arange(0.,15.0,0.4)

shape_sim, wcs_sim = enmap.get_enmap_patch(patch_width_arcmin,sim_pixel_scale,proj="car",pol=pol)
modr_sim = enmap.modrmap(shape_sim,wcs_sim) * 180.*60./np.pi



# === LENS ===

lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
pix_ells = np.arange(0,modlmap_sim.max(),1)
modl_map_alt = enmap.modlmap(shape_sim,wcs_sim)
assert np.all(np.isclose(modlmap_sim,modl_map_alt))


if cluster:
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
    #cents, nkprofile = binner.bin(kappa_map)

    model_massOverh = 2.e14
else:
    clkk = theory.gCl("kk",fine_ells)
    clkk.resize((1,1,clkk.size))
    kappa_map = enmap.rand_map(shape_sim[-2:],wcs_sim,cov=clkk,scalar=True)
    if debug:
        pkk = fmaps.get_simple_power_enmap(kappa_map)
        debug_edges = np.arange(kellmin,kellmax,80)
        dbinner = stats.bin2D(modlmap_sim,debug_edges)
        cents, bclkk = dbinner.bin(pkk)
        clkk.resize((clkk.shape[-1]))
        pl = io.Plotter(scaleY='log',scaleX='log')
        pl.add(fine_ells,clkk)
        pl.add(cents,bclkk)
        pl.done(out_dir+"clkk.png")
phi, fphi = lt.kappa_to_phi(kappa_map,modlmap_sim,return_fphi=True)
io.quickPlot2d(kappa_map,out_dir+"kappa_map.png")
io.quickPlot2d(phi,out_dir+"phi.png")
alpha_pix = enmap.grad_pixf(fphi)




# === EXPERIMENT ===


ntfunc = cmb.get_noise_func(beam_arcmin,noise_T_uK_arcmin,ellmin=tellmin,ellmax=tellmax,TCMB=2.7255e6)
npfunc = cmb.get_noise_func(beam_arcmin,noise_P_uK_arcmin,ellmin=pellmin,ellmax=pellmax,TCMB=2.7255e6)

kbeam_sim = cmb.gauss_beam(modlmap_sim,beam_arcmin)
ps_noise = np.zeros((3,3,pix_ells.size))
ps_noise[0,0] = pix_ells*0.+(noise_T_uK_arcmin*np.pi/180./60./TCMB)**2.
ps_noise[1,1] = pix_ells*0.+(noise_P_uK_arcmin*np.pi/180./60./TCMB)**2.
ps_noise[2,2] = pix_ells*0.+(noise_P_uK_arcmin*np.pi/180./60./TCMB)**2.






# === ENMAP POWER ===


ps = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=False)

# === SIM AND RECON LOOP ===

kappa_stack = {}
for polcomb in pol_list:
    kappa_stack[polcomb] = 0.


for i in range(Nsims):
    print i

    unlensed = enmap.rand_map(shape_sim,wcs_sim,ps)
    lensed = lensing.lens_map_flat_pix(unlensed, alpha_pix,order=lens_order)
    klteb = enmap.map2harm(lensed)
    klteb_beam = klteb*kbeam_sim
    lteb_beam = enmap.ifft(klteb_beam).real
    noise = enmap.rand_map(shape_sim,wcs_sim,ps_noise,scalar=True)
    observed = lteb_beam + noise
    measured = enmap.downgrade(observed,analysis_pixel_scale/sim_pixel_scale)
    if i==0:


        shape_dat, wcs_dat = measured.shape, measured.wcs
        modr_dat = enmap.modrmap(shape_dat,wcs_dat) * 180.*60./np.pi
        binner_dat = stats.bin2D(modr_dat,bin_edges)

        if maxlike and cluster:
            init_kappa_model,r500_init = NFWkappa(cc,model_massOverh,concentration,zL,modr_dat,winAtLens,
                                                  overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


        # === ESTIMATOR ===

        template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
        modl_map_alt = enmap.modlmap(shape_dat,wcs_dat)
        assert np.all(np.isclose(modlmap_dat,modl_map_alt))

        nT = ntfunc(modlmap_dat)
        nP = npfunc(modlmap_dat)
        kbeam_dat = cmb.gauss_beam(modlmap_dat,beam_arcmin)


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
                         doCurl=False,
                         TOnly=not(pol),
                         halo=True,
                         gradCut=gradCut,verbose=True,
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
        dbinner = stats.bin2D(modlmap_sim,debug_edges)
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

        pl = io.Plotter(scaleY='log',scaleX='log')
        pl.add(cents,ltt*cents**2.,color="C0",marker="o",ls="none")
        pl.add(cents,lee*cents**2.,color="C1",marker="o",ls="none")
        pl.add(cents,lbb*cents**2.,color="C2",marker="o",ls="none")
        pl.add(cents,ntt*cents**2.,color="C0",ls="-.",alpha=0.4)
        pl.add(cents,nee*cents**2.,color="C1",ls="-.",alpha=0.4)
        pl.add(cents,nbb*cents**2.,color="C2",ls="-.",alpha=0.4)
        pl.add(fine_ells,lcltt*fine_ells**2.,color="C0",ls="--")
        pl.add(fine_ells,lclee*fine_ells**2.,color="C1",ls="--")
        pl.add(fine_ells,lclbb*fine_ells**2.,color="C2",ls="--")
        pl.done(out_dir+"lccomp.png")

        pl = io.Plotter(scaleX='log')
        pl.add(cents,lte*cents**2.,color="C0",ls="-")
        pl.add(fine_ells,lclte*fine_ells**2.,color="C0",ls="--")
        pl.done(out_dir+"lccompte.png")

        
    fkmaps = enmap.fft(measured,normalize=False)
    fkmaps = np.nan_to_num(fkmaps/kbeam_dat)


    if maxlike and cluster:
        polcomb = "TT"
        maps = enmap.ifft(fkmaps*fMaskCMB_T,normalize=True).real
        kappa_model = init_kappa_model
        k = 0
        io.quickPlot2d(kappa_model,out_dir+"kappa_iter_"+str(k).zfill(3)+".png")
        #io.quickPlot2d(maps,out_dir+"map_iter_"+str(k).zfill(3)+".png")


        import flipper.fft as fftfast
        from scipy.integrate import simps
        Ny,Nx = shape_dat[-2:]
        pixScaleY, pixScaleX = enmap.pixshape(shape_dat,wcs_dat)
        Ukappa = init_kappa_model
        Uft = fftfast.fft(Ukappa,axes=[-2,-1])
        Upower = np.real(Uft*Uft.conjugate())
        Nl2d = qest.N.Nlkk[polcomb]
        area = Nx*Ny*pixScaleX*pixScaleY
        Upower = Upower *area / (Nx*Ny)**2
        wfilter = np.nan_to_num(Upower/Nl2d)
        

        # wfilter = np.nan_to_num(1./(qest.N.Nlkk[polcomb]))
        wfilter[wfilter>1.e90] = 0.
        # print wfilter.max()
        wfilter = wfilter/wfilter.max()

        debug_edges = np.arange(kellmin,kellmax,80)
        dbinner = stats.bin2D(modlmap_dat,debug_edges)
        cents, bwf = dbinner.bin(wfilter)
        pl = io.Plotter()
        pl.add(cents,bwf)
        pl.done(out_dir+"bwf.png")
        #wfilter = np.nan_to_num(qest.N.clkk2d/(qest.N.clkk2d+qest.N.Nlkk[polcomb]))
        
        while k<50:



            
            kappa_model = enmap.samewcs(fmaps.filter_map(kappa_model,wfilter*0.+1.,
                                                         modlmap_dat,lowPass=kellmax,highPass=kellmin),init_kappa_model)
            
            phi_model = lt.kappa_to_phi(kappa_model,modlmap_dat)
            grad_phi = enmap.grad(phi_model)
            maps = lensing.delens_map(maps, grad_phi, nstep=7, order=lens_order, mode="spline", border="cyclic")
            
            fkmaps = enmap.fft(maps,normalize=True)
            qest.updateTEB_X(fkmaps,alreadyFTed=True)
            qest.updateTEB_Y()
            kappa_recon = enmap.samewcs(qest.getKappa(polcomb).real,measured)
            io.quickPlot2d(kappa_recon,out_dir+"kappa_recon_"+str(k).zfill(3)+".png")
            kappa_model = kappa_model + kappa_recon
            k += 1
            maps_filt = fmaps.filter_map(maps,wfilter*0.+1.,
                                         modlmap_dat,lowPass=tellmax,highPass=tellmin)
            io.quickPlot2d(maps_filt,out_dir+"map_iter_"+str(k).zfill(3)+".png")
            io.quickPlot2d(kappa_model,out_dir+"kappa_iter_"+str(k).zfill(3)+".png")
        sys.exit()
    else:
        if pol:
            qest.updateTEB_X(fkmaps[0],fkmaps[1],fkmaps[2],alreadyFTed=True)
        else:
            qest.updateTEB_X(fkmaps,alreadyFTed=True)

        qest.updateTEB_Y()


        for polcomb in pol_list:
            print "Reconstructing",polcomb ," for ", i , " ..."
            kappa_recon = enmap.samewcs(qest.getKappa(polcomb).real,measured)
            if i==0: io.quickPlot2d(kappa_recon,out_dir+"kappa_recon_single.png")

            kappa_stack[polcomb] += kappa_recon
    



if cluster:        
    recon_profiles = {}
    binner_dat = stats.bin2D(modr_dat,bin_edges)

    for polcomb in pol_list:
        kappa_stack[polcomb] /= Nsims
        io.quickPlot2d(kappa_stack[polcomb],out_dir+"kappa_recon_"+polcomb+".png")
        cents, recon_profiles[polcomb] = binner_dat.bin(kappa_stack[polcomb])


    down_input = enmap.downgrade(kappa_map,analysis_pixel_scale/sim_pixel_scale)
    io.quickPlot2d(down_input,out_dir+"down_inp.png")
    inp_kappa = fmaps.filter_map(down_input,down_input.copy()*0.+1.,modlmap_dat,lowPass=kellmax,highPass=kellmin)
    io.quickPlot2d(inp_kappa,out_dir+"filt_inp.png")
    cents, inp_profile = binner_dat.bin(inp_kappa)

    pl = io.Plotter()
    pl.add(cents,inp_profile,ls="--")
    for polcomb in pol_list:
        pl.add(cents,recon_profiles[polcomb],label=polcomb)
    pl.legendOn()
    pl.done(out_dir+"recon_profiles.png")


    pl = io.Plotter()
    for polcomb in pol_list:
        pl.add(cents,(recon_profiles[polcomb]-inp_profile)*100./inp_profile,label=polcomb)
    pl.legendOn()
    pl._ax.set_ylim(-20.,20.)
    pl.done(out_dir+"recon_profiles_per.png")




