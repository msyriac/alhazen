from orphics.analysis.pipeline import mpi_distribute, MPIStats
from mpi4py import MPI
import healpy as hp
import orphics.analysis.flatMaps as fmaps
import orphics.tools.stats as stats
import orphics.tools.cmb as cmb
import flipper.liteMap as lm
import flipper.fftTools as ft
import orphics.tools.io as io
import numpy as np

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

Ntot = 256

num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
my_tasks = each_tasks[rank]

nside = 4096
deg = 10.
px = 0.5

bin_edges = np.arange(100,4000,40)


theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

ellrange = np.arange(0,10000,1)
cls = theory.lCl('TT',ellrange)

lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin = px, pixScaleYarcmin=px)


for k,i in enumerate(my_tasks):

    with io.nostdout():
        hpmap,alm = hp.synfast(cls,nside,alm=True,pol=False)

    lmap.loadDataFromHealpixMap(hpmap, interpolate = False, hpCoords = "J2000")
    del hpmap
    if k==0:
        taper,w2 = fmaps.get_taper(lmap.data.shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)

    lmap.data *= taper
    if rank==0 and k==0:
        io.quickPlot2d(lmap.data,io.dout_dir+"flipper_map.png")
    p = ft.powerFromLiteMap(lmap)
    if k==0:
        binner = stats.bin2D(p.modLMap,bin_edges)


    #pwin = hp.pixwin(nside)
    #pells = np.arange(0,pwin.size)
    #from scipy.interpolate import interp1d
    pwin2d = 1. #interp1d(pells,pwin,bounds_error=False,fill_value=0.)(p.modLMap)
    p2d = np.nan_to_num(p.powerMap/w2/pwin2d**2.)

    cents,p1d = binner.bin(p2d)

    cls = hp.alm2cl(alm)

    mpibox.add_to_stack("full",cls)
    mpibox.add_to_stack("cut",p1d)

    if rank==0: print k+1, " / ", len(my_tasks), " done."


mpibox.get_stacks()

if rank==0:

    fineells = np.arange(0,cls.size,1)
    fulltt = theory.lCl('TT',fineells)
    fulldiff = (mpibox.stacks['full']-fulltt)/fulltt

    cents,cuttt = binner.bin(theory.lCl('TT',p.modLMap))
    cutdiff = (mpibox.stacks['cut']-cuttt)/cuttt


    pl = io.Plotter()
    pl.add(fineells,fulldiff)
    pl.add(cents,cutdiff)
    pl.hline()
    pl._ax.set_ylim(-0.05,0.1)
    pl.done(io.dout_dir+"pdiff_flipper.png")
    

