import numpy as np
from enlib import enmap,resample,lensing
import orphics.analysis.flatMaps as fmaps
from szar.counts import ClusterCosmology
import orphics.tools.io as io
import alhazen.lensTools as lt
from mpi4py import MPI
import sys, os
from scipy.linalg import pinv2
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import argparse
import cPickle as pickle

# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("-M", "--mass",     type=float)
args = parser.parse_args()
M = args.mass


out_dir = os.environ['WWW']+"plots/"
def nfwkappa(massOverh):
    zL = 0.7
    overdensity = 180.
    critical = False
    atClusterZ = False
    concentration = 3.2
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,pa.modrmap* 180.*60./np.pi,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return kappa


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

lmax = 8000
cc = ClusterCosmology(lmax=lmax,pickling=True)
theory = cc.theory

arc = 10.0
px = 0.5
lens_order = 5


shape,wcs = enmap.get_enmap_patch(arc,px,proj="car")



    
pa = fmaps.PatchArray(shape,wcs,dimensionless=False,skip_real=False)
pa.add_theory(theory,lmax)
pa.add_white_noise_with_atm(0.1,0.,0,1,0,1)



N = 130000
Ntot = N

# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."

from alhazen.halos import NFWkappa

kappa = nfwkappa(M)
phi, fphi = lt.kappa_to_phi(kappa,pa.modlmap,return_fphi=True)
#grad_phi = enmap.grad(phi)
alpha_pix = enmap.grad_pixf(fphi)

# What am I doing?
my_tasks = each_tasks[rank]


for k,index in enumerate(my_tasks):
    
    cmb_map = pa.get_unlensed_cmb(seed=2*index)
    lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order)
    noise = pa.get_noise_sim(seed=2*index+1)
    measured = lensed + noise

    if index==0 or index==1:
        io.quickPlot2d(cmb_map,out_dir+"cmbmap_"+str(index)+".png")
        io.quickPlot2d(lensed,out_dir+"lcmbmap_"+str(index)+".png")
        io.quickPlot2d(noise,out_dir+"ncmbmap_"+str(index)+".png")
        io.quickPlot2d(measured,out_dir+"mcmbmap_"+str(index)+".png")

    #cmb = measured.reshape((1,shape[0]*shape[1]))
    cmb = measured.reshape((shape[0]*shape[1]))
    mpibox.add_to_stats("vec",cmb)

    if k%1000==0: print M,k

mpibox.get_stats()

if rank==0:
    cov = mpibox.stats["vec"]["cov"]

    print "Inverting cov..."
    cinv = pinv2(cov)

    s,logdet = np.linalg.slogdet(cov)
    #print s,logdet
    try:
        assert s>0
    except:
        print "ERROR: Negative log"
        print s,logdet
        
        sys.exit()

    pickle.dump((M,logdet,cinv),open("c_"+str(M)+".pkl",'wb'))
    io.quickPlot2d(cov,out_dir+"cov_"+str(M)+".png")
    io.quickPlot2d(cinv,out_dir+"cinv_"+str(M)+".png")
        
