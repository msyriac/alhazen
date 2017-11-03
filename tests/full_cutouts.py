from enlib import enmap, curvedsky
from orphics.analysis.pipeline import mpi_distribute, MPIStats
from mpi4py import MPI
import healpy as hp
import numpy as np
import orphics.analysis.flatMaps as fmaps
import orphics.tools.stats as stats
import orphics.tools.cmb as cmb
import orphics.tools.io as io

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

Ntot = 32

num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print(("At most ", max(num_each) , " tasks..."))
my_tasks = each_tasks[rank]

deg = 40.
px = 2.0
shape,wcs = enmap.rect_geometry(deg*60.,px,proj="car")
bbox = enmap.box(shape,wcs)

bin_edges = np.arange(100,4000,40)


theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)


for k,i in enumerate(my_tasks):
    
    map_file = "/gpfs01/astro/workarea/msyriac/data/sims/sigurd/cori/v6full/fullsky_curved_lensed_car_"+str(i).zfill(2)+".fits"
    lmap_file = "/gpfs01/astro/workarea/msyriac/data/sims/sigurd/cori/v61600/equator_curved_lensed_car_"+str(i).zfill(2)+".fits"
    imap = enmap.read_map(map_file)[0]
    lmap = enmap.read_map(lmap_file)[0]
    smap = imap.submap(bbox)

    if k==0:
        shape = smap.shape
        wcs = smap.wcs
        fc = enmap.FourierCalc(shape,wcs)
        taper,w2 = fmaps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
        binner = stats.bin2D(enmap.modlmap(shape,wcs),bin_edges)

        shape2 = lmap.shape
        wcs2 = lmap.wcs
        fc2 = enmap.FourierCalc(shape2,wcs2)
        taper2,w22 = fmaps.get_taper(shape2,taper_percent = 12.0,pad_percent = 3.0,weight=None)
        binner2 = stats.bin2D(enmap.modlmap(shape2,wcs2),bin_edges)
        
        ells = np.arange(0,5000,1)
        ps = theory.lCl('TT',ells).reshape((1,1,5000))
        shapeN,wcsN = enmap.rect_geometry(1.2*deg*60.,px,proj="car")
        mg = enmap.MapGen(shapeN,wcsN,ps)

        mgP = enmap.MapGen(shape,wcs,ps)


        
    p2d,_,_ = fc.power2d(smap*taper)/w2
    cents,p1d = binner.bin(p2d)

    p2d2,_,_ = fc2.power2d(lmap*taper2)/w22
    cents,p1d2 = binner2.bin(p2d2)

    nmap = curvedsky.rand_map(shape,wcs,ps) #mg.get_map()
    p2d,_,_ = fc.power2d(nmap*taper)/w2
    cents,p1d3 = binner.bin(p2d)

    nmap = mg.get_map().submap(enmap.box(shape,wcs))
    fc4 = enmap.FourierCalc(nmap.shape,nmap.wcs)
    taper4,w24 = fmaps.get_taper(nmap.shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
    binner4 = stats.bin2D(nmap.modlmap(),bin_edges)
    p2d,_,_ = fc4.power2d(nmap*taper4)/w24
    shape4 = nmap.shape
    wcs4 = nmap.wcs
    cents,p1d4 = binner4.bin(p2d)
    if rank==0 and k==0: io.quickPlot2d(nmap*taper4,io.dout_dir+"nptapered.png")


    nmap = mgP.get_map()
    p2d,_,_ = fc.power2d(nmap*taper)/w2
    cents,p1d5 = binner.bin(p2d)

    
    alm = curvedsky.map2alm(imap.astype(np.float64),lmax=4000)
    del imap
    cls = hp.alm2cl(alm)

    mpibox.add_to_stack("full",cls)
    mpibox.add_to_stack("cut",p1d)
    mpibox.add_to_stack("cut2",p1d2)
    mpibox.add_to_stack("cut3",p1d3)
    mpibox.add_to_stack("cut4",p1d4)
    mpibox.add_to_stack("cut5",p1d5)

    print((k,i))


mpibox.get_stacks()

if rank==0:

    fineells = np.arange(0,cls.size,1)
    fulltt = theory.lCl('TT',fineells)
    fulldiff = (mpibox.stacks['full']-fulltt)/fulltt

    cents,cuttt = binner.bin(theory.lCl('TT',enmap.modlmap(shape,wcs)))
    cutdiff = (mpibox.stacks['cut']-cuttt)/cuttt

    cents,cuttt2 = binner2.bin(theory.lCl('TT',enmap.modlmap(shape2,wcs2)))
    cutdiff2 = (mpibox.stacks['cut2']-cuttt2)/cuttt2

    cents,cuttt3 = binner.bin(theory.lCl('TT',enmap.modlmap(shape,wcs)))
    cutdiff3 = (mpibox.stacks['cut3']-cuttt3)/cuttt3

    cents,cuttt4 = binner4.bin(theory.lCl('TT',enmap.modlmap(shape4,wcs4)))
    cutdiff4 = (mpibox.stacks['cut4']-cuttt4)/cuttt4

    cents,cuttt5 = binner.bin(theory.lCl('TT',enmap.modlmap(shape,wcs)))
    cutdiff5 = (mpibox.stacks['cut5']-cuttt5)/cuttt5

    import orphics.tools.io as io
    pl = io.Plotter()
    pl.add(fineells,fulldiff,label="SHT gen, SHT power",alpha=0.4)
    pl.add(cents,cutdiff,label="cutout from saved full-sky SHT-gen, FFT power")
    pl.add(cents,cutdiff2,label="saved cut-sky SHT-gen, FFT power")
    pl.add(cents,cutdiff3,label="on-the-fly cut-sky SHT-gen, FFT power")
    pl.add(cents,cutdiff4,label="FFT-gen cutout non-periodic, FFT power")
    pl.add(cents,cutdiff5,label="FFT-gen periodic tapered, FFT power")
    pl.hline()
    pl.legendOn()
    pl._ax.set_ylim(-0.05,0.1)
    pl.done(io.dout_dir+"pdiff.png")
    

