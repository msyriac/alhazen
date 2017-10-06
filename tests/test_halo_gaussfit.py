import numpy as np
from enlib import enmap
import orphics.tools.io as io
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import orphics.analysis.flatMaps as fmaps
import orphics.tools.stats as stats
from szar.counts import ClusterCosmology
import os,sys
from mpi4py import MPI

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

def get_nfw(massOverh,zL=0.7,concentration=3.2):
    from alhazen.halos import NFWkappa

    #massOverh = 2.e14
    overdensity = 180.
    critical = False
    atClusterZ = False
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,modrmap* 180.*60./np.pi,winAtLens,
                              overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return kappa

cc = ClusterCosmology(lmax=8500,pickling=True)
widtharc = 50.
px = 0.5
shape,wcs = enmap.get_enmap_patch(widtharc,px)
modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
rbins = np.arange(0.,10.,1.0)
rbinner = stats.bin2D(modrmap*60.*180./np.pi,rbins)
true_mass = 2.e14
true2d = get_nfw(true_mass)
kellmin = 200
kellmax = 8500
true2d = enmap.ndmap(fmaps.filter_map(true2d,true2d*0.+1.,modlmap,lowPass=kellmax,highPass=kellmin),wcs)


Nsims = 1000
cov = np.ones((1,1,8500))*0.00000001
ngen = enmap.MapGen(shape,wcs,cov)

out_dir = os.environ['WWW']+"plots/cgauss_"

# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Nsims,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
# What am I doing?
my_tasks = each_tasks[rank]


for k,index in enumerate(my_tasks):

    noise = ngen.get_map()
    mkappa = true2d+noise
    if k==0 and rank==0: io.quickPlot2d(noise,out_dir+"nstamp.png")
    print k
    cents, prof = rbinner.bin(mkappa)
    mpibox.add_to_stats('prof',prof)
    mpibox.add_to_stack('mkappa',mkappa)

mpibox.get_stacks()
mpibox.get_stats()

if rank==0:
    io.quickPlot2d(mpibox.stacks['mkappa'],out_dir+"stack.png")

    mean = mpibox.stats['prof']['mean']
    cov = mpibox.stats['prof']['covmean']
    siginv = np.linalg.inv(cov)
    
    chisq = np.dot(np.dot(mean,siginv),mean)
    print np.sqrt(chisq)

    mass_range = np.linspace(1.e14,5.e14,300)
    Likes = []
    for k,m in enumerate(mass_range):
        trial = get_nfw(m)
        kellmin = 200
        kellmax = 8500
        trial = enmap.ndmap(fmaps.filter_map(trial,true2d*0.+1.,modlmap,lowPass=kellmax,highPass=kellmin),wcs)
        cents,theory = rbinner.bin(trial)
        Likes.append(np.exp(-0.5*stats.fchisq(mean,siginv,theory,amp=1.)))
        if k%10==0: print m

    Likes = np.array(Likes)
    Likes /= Likes.sum()
    
    pl = io.Plotter()
    pl.add(mass_range,Likes)
    pl._ax.axvline(x=true_mass,ls="--")
    pl.done(out_dir+"likes.png")

    

