from orphics.tools.stats import bin2D
from enlib import enmap, bench
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np
import sys
import orphics.analysis.flatMaps as fmaps
from orphics.analysis.pipeline import mpi_distribute, MPIStats
from mpi4py import MPI

Nsims = 1024

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    
Ntot = Nsims
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
my_tasks = each_tasks[rank]



theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

wdeg = 40.
hdeg = 15.
yoffset = 60.
pix = 1.0
lmax = 5000
shape,wcs = enmap.rect_geometry(width_arcmin=wdeg*60.,px_res_arcmin=pix,height_arcmin=hdeg*60.,yoffset_degree=yoffset)


ells = np.arange(0,7000,1)
#pfunc = lambda x: theory.lCl('TT',x)
#pfunc = lambda x: x*0.+1.
pfunc = lambda x: theory.gCl('kk',x)
ps = pfunc(ells).reshape((1,1,7000))

mg = enmap.MapGen(shape,wcs,ps)
taper,w2 = fmaps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)

pxover = 1.0
if rank==0:
    with bench.show("Rot init..."):
        r = fmaps.MapRotatorEquator(shape,wcs,wdeg,hdeg,width_multiplier=0.6,height_multiplier=1.2,downsample=True,verbose=True,pix_target_override_arcmin=pxover)
else:
    r = fmaps.MapRotatorEquator(shape,wcs,wdeg,hdeg,width_multiplier=0.6,height_multiplier=1.2,downsample=True,pix_target_override_arcmin=pxover)
    
for k,index in enumerate(my_tasks):
    map_south = mg.get_map(seed=index)*taper

    
    rotmap = r.rotate(map_south)

    if k==0:
        rottap = r.rotate(taper)
        tmg = enmap.MapGen(r.shape_final,r.wcs_final,ps)
        w2 = np.mean(rottap**2.)
        fc = enmap.FourierCalc(r.shape_final,r.wcs_final)
        modlmap = enmap.modlmap(r.shape_final,r.wcs_final)
        bin_edges = np.arange(100,lmax,40)
        binner = bin2D(modlmap,bin_edges)
        
    map_test = tmg.get_map(seed=index+int(1e6))*rottap


    if rank==0 and k==0:
        prefix = io.dout_dir+"Oct28_"
        # io.highResPlot2d(map_south,prefix+"smap.png")
        del map_south
        # io.highResPlot2d(rotmap,prefix+"rotmap.png")
        # io.highResPlot2d(rottap,prefix+"taper.png")
        # io.highResPlot2d(map_test,prefix+"testmap.png")



    p2d_rot,_,_ = fc.power2d(rotmap)/w2
    p2d,_,_ = fc.power2d(map_test)/w2


    cents,crot = binner.bin(p2d_rot)
    cents,ctest = binner.bin(p2d)

    mpibox.add_to_stack("crot",crot)
    mpibox.add_to_stack("ctest",ctest)

    if rank==0: print "Done with ", k+1," / ",len(my_tasks)

mpibox.get_stacks()

if rank==0:
    p2d_theory = pfunc(modlmap)
    cents,ctheory = binner.bin(p2d_theory)

    crot = mpibox.stacks["crot"]
    ctest = mpibox.stacks["ctest"]

    pl = io.Plotter()
    pl.add(cents,(crot-ctest)/ctest,label="rot vs test")
    pl.add(cents,(crot-ctheory)/ctheory,label="rot vs theory")
    pl.add(cents,(ctest-ctheory)/ctheory,label="test vs theory")
    pl.hline()
    pl.legendOn()
    pl.done(prefix+"cldiff.png")

