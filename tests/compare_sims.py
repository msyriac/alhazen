
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

save_dir = "/gpfs01/astro/workarea/msyriac/data/depot/distortions/fullsky_spectra64_"

from orphics.tools.stats import bin2D, bin1D
bin_edges = np.arange(200,4000,40)

lbinner = bin1D(bin_edges)

ells,cltt,clee,clbb,clte,cleb,cltb = np.loadtxt(save_dir+"fullsky_binned_average_spec_v664.txt",unpack=True)

fineells = np.arange(0,ells[-1],1)

def b(cls):
    cls = np.nan_to_num(cls)
    ells,cl1d = lbinner.binned(fineells,fineells*cls)
    ells,norm = lbinner.binned(fineells,fineells)
    cl1d /= norm
    return ells,cl1d


theory_file_root = "data/Aug6_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)



ells,tcltt = b(theory.lCl('TT',fineells))
ells,tclee = b(theory.lCl('EE',fineells))
ells,tclte = b(theory.lCl('TE',fineells))
ells,tclbb = b(theory.lCl('BB',fineells))


pl = io.Plotter(scaleY='log')
pl.add(ells,cltt*ells**2.)
pl.add(ells,clee*ells**2.)
pl.add(ells,tcltt*ells**2.,color="k")
pl.add(ells,tclee*ells**2.,color="k")
pl.add(ells,clbb*ells**2.)        
pl.add(ells,tclbb*ells**2.,color="k")
pl._ax.set_ylim(1.e-3,5e5)
pl.done(io.dout_dir+"sigurd_alex_clcomp_.png")

cut_root = "/gpfs01/astro/workarea/msyriac/data/depot/distortions/"
mells,mcltt,_ = np.loadtxt(cut_root+"distspectrav6unlensed_equator_car_TT_unlensed.txt",unpack=True)
mells,mclee,_ = np.loadtxt(cut_root+"distspectrav6unlensed_equator_car_EE_unlensed.txt",unpack=True)


pl = io.Plotter(labelX="$\\ell$",labelY="$\Delta C_{\\ell}/C_{\\ell}$")
if True:
    pl.add(mells,mcltt,label="TT cut-sky",ls="--")
    pl.add(mells,mclee,label="EE cut-sky",ls="--")
pl.add(ells,np.nan_to_num((cltt-tcltt)/tcltt),label="TT Full-sky")
pl.add(ells,np.nan_to_num((clee-tclee)/tclee),label="EE Full-sky")
pl.add(ells,np.nan_to_num((clbb-tclbb)/tclbb),label="BB Full-sky")
pl._ax.set_xlim(2,4000)
pl._ax.set_ylim(-0.1,0.1)
pl.hline()
pl.legendOn(loc="lower left")
pl.done(io.dout_dir+"sigurd_alex_clcomp__diff.png")


pl = io.Plotter()
pl.add(ells,tclte*ells**2.,color="k")
pl.add(ells,clte*ells**2.)
pl.add(ells,cleb*ells**2.)
pl.add(ells,cltb*ells**2.)


pl.done(io.dout_dir+"sigurd_alex_clcomp_cross.png")


