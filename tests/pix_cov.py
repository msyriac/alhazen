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
import enlib.fft as fftfast

# Parse command line
parser = argparse.ArgumentParser(description='Generate pixel-pixel covmats of lensed CMB.')
parser.add_argument("Out", type=str,help='Output Directory Name (not path, that\'s specified in ini')
parser.add_argument("Exp", type=str,help='Experiment section for beam')
parser.add_argument("-s", "--covseed",     type=int,  default=0)
parser.add_argument("-N", "--num_sims",     type=int,  default=130000, help="Number of sims for covmat.")
parser.add_argument("-n", "--N1", action='store_true',help='Do N1.')
parser.add_argument("-p", "--paper", action='store_true',help='Paper specific plots.')
parser.add_argument("-u", "--unlensed", action='store_true',help='Make unlensed.')
parser.add_argument("-M", "--massindex",     type=int)
args = parser.parse_args()
Npoints = 60
mrange = np.linspace(1.0,3.0,Npoints)*1.e14
M = mrange[args.massindex]



out_dir = os.environ['WWW']+"plots/maxlike_hdv_nodim_"


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
pa.add_theory(cc,theory,lmax)
pa.add_gaussian_beam(1.0)
pa.add_white_noise_with_atm(3.0,0.,0,1,0,1)



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
    
    cmb_map = pa.get_unlensed_cmb(seed=(args.massindex*Ntot)+index)
    lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=lens_order) if np.abs(M)>1.e-3 else cmb_map
    flensed = fftfast.fft(lensed,axes=[-2,-1])
    flensed *= pa.lbeam
    lensed = fftfast.ifft(flensed,axes=[-2,-1],normalize=True).real
    # noise = pa.get_noise_sim(seed=(mrange.size*Ntot)+(args.massindex*Ntot)+index)
    # measured = lensed + noise
    measured = lensed

    # if index==0 or index==1:
    #     io.quickPlot2d(cmb_map,out_dir+"cmbmap_"+str(index)+".png")
    #     io.quickPlot2d(lensed,out_dir+"lcmbmap_"+str(index)+".png")
    #     io.quickPlot2d(noise,out_dir+"ncmbmap_"+str(index)+".png")
    #     io.quickPlot2d(measured,out_dir+"mcmbmap_"+str(index)+".png")

    cmb = measured.reshape((shape[0]*shape[1]))
    mpibox.add_to_stats("vec",cmb)

    if k%1000==0 and rank==0: print M,k, " / ", len(my_tasks)

mpibox.get_stats()

if rank==0:
    cov = mpibox.stats["vec"]["cov"]

    # print "Inverting cov..."
    # cinv = pinv2(cov)

    # s,logdet = np.linalg.slogdet(cov)
    # print s,logdet
    # try:
    #     assert s>0
    # except:
    #     print "ERROR: Negative log"
    #     print M,s,logdet
        
    #     sys.exit()

    pickle.dump((M,cov),open(out_dir+"c_"+str(args.massindex)+".pkl",'wb'))
    #pickle.dump((M,logdet,cov,cinv),open(out_dir+"c_"+str(args.massindex)+".pkl",'wb'))
    # io.quickPlot2d(cov,out_dir+"cov_"+str(M)+".png")
    # io.quickPlot2d(cinv,out_dir+"cinv_"+str(M)+".png")
        
