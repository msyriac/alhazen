from orphics.tools.stats import bin2D
from enlib import enmap, bench, curvedsky
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np
import sys
import orphics.analysis.flatMaps as fmaps
from orphics.tools.mpi import MPI,mpi_distribute, MPIStats
import healpy as hp
from enlib import enmap

Nsims = 16

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    
Ntot = Nsims
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print(("At most ", max(num_each) , " tasks..."))
my_tasks = each_tasks[rank]



theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
ells = np.arange(0,7000,1)
#pfunc = lambda x: theory.lCl('TT',x)
#pfunc = lambda x: x*0.+1.
pfunc = lambda x: x*0.+(10.*np.pi/180./60.)**2. #theory.gCl('kk',x)
ps = pfunc(ells).reshape((1,1,7000))

wdeg = 40.
hdeg = 15.
yoffset = 60.
pix = 0.5
lmax = 7000

fshape, fwcs = enmap.fullsky_geometry(res=pix*np.pi/180./60., proj="car")

shape,wcs = enmap.rect_geometry(width_arcmin=wdeg*60.,px_res_arcmin=pix,height_arcmin=hdeg*60.,yoffset_degree=yoffset)



with bench.show("taper"):
    #mg = enmap.MapGen(shape,wcs,ps)
    taper,sw2 = fmaps.get_taper(shape,taper_percent = 18.0,pad_percent = 4.0,weight=None)

pxover = 0.5
    
for k,index in enumerate(my_tasks):

    with bench.show("full_sky"):
        fullsky = curvedsky.rand_map(fshape, fwcs, ps, lmax=lmax,dtype=np.float32)


    map_south = fullsky.submap(enmap.box(shape,wcs))*taper


    if k==0:
        if rank==0:
            with bench.show("Rot init..."):
                r = fmaps.MapRotatorEquator(shape,wcs,wdeg,hdeg,width_multiplier=0.6,
                                            height_multiplier=1.2,downsample=True,verbose=True,pix_target_override_arcmin=pxover)
        else:
            r = fmaps.MapRotatorEquator(shape,wcs,wdeg,hdeg,width_multiplier=0.6,
                                        height_multiplier=1.2,downsample=True,pix_target_override_arcmin=pxover)

    
    rotmap = r.rotate(map_south)

    if k==0:
        rottap = r.rotate(taper)
        #tmg = enmap.MapGen(r.shape_final,r.wcs_final,ps)
        w2 = np.mean(rottap**2.)
        fc = enmap.FourierCalc(r.shape_final,r.wcs_final)
        modlmap = enmap.modlmap(r.shape_final,r.wcs_final)
        bin_edges = np.arange(100,lmax,40)
        binner = bin2D(modlmap,bin_edges)


    map_test = fullsky.submap(enmap.box(r.shape_final,r.wcs_final))
    del fullsky
    if k==0:
        rect_taper,rw2 = fmaps.get_taper(map_test.shape,taper_percent = 18.0,pad_percent = 4.0,weight=None)
        rfc = enmap.FourierCalc(map_test.shape,map_test.wcs)
        rmodlmap = enmap.modlmap(map_test.shape,map_test.wcs)
        rbinner = bin2D(rmodlmap,bin_edges)

    map_test *= rect_taper
    #map_test = tmg.get_map(seed=index+int(1e6))*rottap

    alm = curvedsky.map2alm(map_south,lmax=lmax)
    cls = hp.alm2cl(alm)/sw2
    del alm

    if rank==0 and k==0:
        prefix = io.dout_dir+"Oct28_"
        # io.highResPlot2d(map_south,prefix+"smap.png")
        del map_south
        # io.highResPlot2d(rotmap,prefix+"rotmap.png")
        # io.highResPlot2d(rottap,prefix+"taper.png")
        # io.highResPlot2d(map_test,prefix+"testmap.png")


    p2d_rot,_,_ = fc.power2d(rotmap)/w2
    p2d,_,_ = rfc.power2d(map_test)/rw2


    cents,crot = binner.bin(p2d_rot)
    cents,ctest = rbinner.bin(p2d)

    mpibox.add_to_stack("sht",cls)
    mpibox.add_to_stack("crot",crot)
    mpibox.add_to_stack("ctest",ctest)

    if rank==0: print(("Done with ", k+1," / ",len(my_tasks)))

mpibox.get_stacks()

if rank==0:
    p2d_theory = pfunc(modlmap)
    cents,ctheory = binner.bin(p2d_theory)

    crot = mpibox.stacks["crot"]
    ctest = mpibox.stacks["ctest"]
    shts = mpibox.stacks["sht"]
    ells = np.arange(0,shts.size,1)
    sht_theory = pfunc(ells)

    

    pl = io.Plotter()
    pl.add(ells,(shts-sht_theory)/sht_theory,label="south sht vs theory",alpha=0.3)
    pl.add(cents,(crot-ctest)/ctest,label="rot vs test")
    pl.add(cents,(crot-ctheory)/ctheory,label="rot vs theory")
    pl.add(cents,(ctest-ctheory)/ctheory,label="test vs theory")
    pl.hline()
    pl.legendOn()
    pl._ax.set_xlim(0,5000)
    pl.done(prefix+"cldiff.png")


    pl = io.Plotter(scaleY='log')
    pl.add(ells,(shts*ells**2.))
    pl.add(ells,(sht_theory*ells**2.))
    pl.legendOn()
    pl._ax.set_xlim(0,5000)
    pl.done(prefix+"clsht.png")

