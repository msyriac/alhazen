from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from enlib import enmap, curvedsky, lensing
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.analysis.flatMaps as fmaps
import numpy as np
import healpy as hp
from mpi4py import MPI
from orphics.analysis.pipeline import mpi_distribute, MPIStats, SpectrumVerification

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

sim_root = "/gpfs01/astro/workarea/msyriac/data/sims/sigurd/cori/v6full64/"
save_dir = "/gpfs01/astro/workarea/msyriac/data/depot/distortions/fullsky_spectra64_"

def load_fullsky(sim_root,prefix,k):

    uiqu = enmap.read_map(sim_root+prefix+"_"+str(k).zfill(2)+".fits")
    
    return uiqu

def map2power(iqu,mpibox):
    from orphics.tools.stats import bin2D, bin1D
    bin_edges = np.arange(200,4000,40)
        
    
    print ("Map 2 alm...")
    alm = curvedsky.map2alm(iqu.astype("float64"),lmax=5000)
    del iqu
    cls = hp.alm2cl(alm)
    del alm
    fineells = np.arange(0,cls.shape[1],1)

    print ("Binning...")
    lbinner = bin1D(bin_edges)
    def b(cls):
        ells,cl1d = lbinner.binned(fineells,fineells*cls)
        ells,norm = lbinner.binned(fineells,fineells)
        cl1d /= norm
        return ells,cl1d


    ells,cltt = b(cls[0,:])
    ells,clee = b(cls[1,:])
    ells,clbb = b(cls[2,:])
    ells,clte = b(cls[3,:])
    ells,cleb = b(cls[4,:])
    ells,cltb = b(cls[5,:])

    mpibox.add_to_stats("TT",cltt)
    mpibox.add_to_stats("EE",clee)
    mpibox.add_to_stats("BB",clbb)
    mpibox.add_to_stats("TE",clte)
    mpibox.add_to_stats("EB",cleb)
    mpibox.add_to_stats("TB",cltb)

    return ells

Ntot = 8
# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print ("At most ", max(num_each) , " tasks...")

# What am I doing?
my_tasks = each_tasks[rank]

k = -1
for index in my_tasks:
    
    k += 1

    cmb_map = load_fullsky(sim_root,"fullsky_curved_lensed_car",index)
    print (rank,k,cmb_map.shape)
    ells = map2power(cmb_map,mpibox)

mpibox.get_stats()
if rank==0:
    TT = mpibox.stats['TT']['mean']
    EE = mpibox.stats['EE']['mean']
    BB = mpibox.stats['BB']['mean']
    EB = mpibox.stats['EB']['mean']
    TE = mpibox.stats['TE']['mean']
    TB = mpibox.stats['TB']['mean']

    io.save_cols(save_dir+"fullsky_binned_average_spec_v664.txt",(ells,TT,EE,BB,TE,EB,TB))
    

#     if lensed:
#         ells,tcltt = b(theory.lCl('TT',fineells))
#         ells,tclee = b(theory.lCl('EE',fineells))
#         ells,tclte = b(theory.lCl('TE',fineells))
#         ells,tclbb = b(theory.lCl('BB',fineells))
#     else:        
#         ells,tcltt = b(theory.uCl('TT',fineells))
#         ells,tclee = b(theory.uCl('EE',fineells))
#         ells,tclte = b(theory.uCl('TE',fineells))
#         ells,tclbb = b(fineells*0.)

#     pl = io.Plotter(scaleY='log')
#     pl.add(ells,cltt*ells**2.)
#     pl.add(ells,clee*ells**2.)
#     pl.add(ells,tcltt*ells**2.,color="k")
#     pl.add(ells,tclee*ells**2.,color="k")
#     if lensed:
#         pl.add(ells,clbb*ells**2.)        
#         pl.add(ells,tclbb*ells**2.,color="k")
#     pl._ax.set_ylim(1.e-3,5e5)
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_lensed_"+str(lensed)+".png")


#     pl = io.Plotter(labelX="$\\ell$",labelY="$\Delta C_{\\ell}/C_{\\ell}$")
#     if True:#(sshape is not None) and (swcs is not None):
#         pl.add(mells,mcltt,label="TT cut-sky",ls="--")
#         pl.add(mells,mclee,label="EE cut-sky",ls="--")
#     pl.add(ells,np.nan_to_num((cltt-tcltt)/tcltt),label="TT Alex")
#     #pl.add(ells,np.nan_to_num((clte-tclte)/tclee),label="TE")
#     pl.add(ells,np.nan_to_num((clee-tclee)/tclee),label="EE Alex")
#     if lensed: pl.add(ells,np.nan_to_num((clbb-tclbb)/tclbb),label="BB Alex")
#     pl._ax.set_xlim(2,4000)
#     pl._ax.set_ylim(-0.1,0.1)
#     pl.hline()
#     pl.legendOn(loc="lower left")
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_lensed_"+str(lensed)+"_diff.png")


#     pl = io.Plotter()
#     pl.add(ells,tclte*ells**2.,color="k")
#     pl.add(ells,clte*ells**2.)
#     pl.add(ells,cleb*ells**2.)
#     pl.add(ells,cltb*ells**2.)
    
#     if not(lensed): pl.add(ells,clbb*ells**2.)
#     pl.done(io.dout_dir+"sigurd_alex_clcomp_cross_lensed_"+str(lensed)+".png")

# cambRoot = sim_root + "cosmo2017"
# theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
    
# # uiqu = load_alex(sim_root,"uTquMap")
# # print (uiqu.shape)
# # pix_arc = np.sqrt(uiqu.pixsize())*180.*60./np.pi
# # print (pix_arc, " arcmin pixel")

# # map2power(uiqu)

# liqu = load_alex(sim_root,"lTquMap")
# map2power(liqu,lensed=True)

