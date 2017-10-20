import numpy as np
import sys, os
from orphics.analysis.pipeline import mpi_distribute, MPIStats, SpectrumVerification
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from enlib import enmap, lensing, resample
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
import enlib.fft as fftfast
import argparse
from alhazen.quadraticEstimator import Estimator
from mpi4py import MPI


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

# Runtime params that should be moved to command line
expf_name = "experiment_noiseless"
cosmology_section = "cc_nam_low"
iau_convention = False

# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("Region", type=str,help='equator/south')
parser.add_argument("Projection", type=str,help='CAR/CEA')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
#parser.add_argument("-s", "--save",     type=str,  default=None)
args = parser.parse_args()
Nsims = args.nsim
region = args.Region
projection = args.Projection
#save = args.save

save_dir = "/gpfs01/astro/workarea/msyriac/data/depot/distortions/distspectrav61600_"+region+"_"+projection+"_"
analysis_section = "analysis_sigurd_"+region+"_"+projection+"_1600"

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
pout_dir = os.environ['WWW']+"plots/distsimsv61600_"+region+"_"+projection+"_"  # for plots
#save_dir = map_root + dirname # for saves
#if save is not None: save_func = lambda x: save_dir + "/"+save+"_"+str(x).zfill(9)+".fits"

# How many sims? Should I use saved files?

if Nsims is None: Nsims = 320
sigurd_cmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sims/sigurd/cori/v61600/"+region+"_curved_lensed_"+projection+"_"+str(x).zfill(2)+".fits"
sigurd_kappa_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/sims/sigurd/cori/v61600/"+region+"_curved_kappa_"+projection+"_"+str(x).zfill(2)+".fits"

    
Ntot = Nsims


# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print "At most ", max(num_each) , " tasks..."

# What am I doing?
my_tasks = each_tasks[rank]

if rank==0: print "Reading config..."

# Read config
iniFile = "../halofg/input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

if rank==0: print "Params..."

pol = True
shape_dat, wcs_dat = aio.enmap_from_config_section(Config,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
if rank==0: print "Ell bounds..."

if rank==0: print "Patches data..."

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)

if rank==0: print "Attributes..."

lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)

if rank==0: print "Binners..."

kellmin = 200
kellmax = 4000
lbin_edges = np.arange(kellmin,kellmax,40)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)

if rank==0: print "Cosmology..."

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section,dimensionless=False)
parray_dat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)


    
taper_percent = 14.0
pad_percent = 6.0
Ny,Nx = shape_dat[-2:]
taper = fmaps.cosineWindow(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))
w2 = np.mean(taper**2.)
if rank==0:
    io.quickPlot2d(taper,pout_dir+"taper.png")
    print "w2 : " , w2

px_dat = analysis_resolution

sverif_cmb = SpectrumVerification(mpibox,theory,shape_dat,wcs_dat,lbinner=lbinner_dat,pol=pol,iau_convention=iau_convention)



k = -1
for index in my_tasks:
    
    k += 1
    if rank==0: print "Rank ", rank , " doing cutout ", index
    cmb = enmap.read_map(sigurd_cmb_file(index))
    llteb,dummy = sverif_cmb.add_power("lensed",cmb*taper,norm=w2,twod_stack=True)



mpibox.get_stacks()
mpibox.get_stats()




if rank==0:

    p2d_tt = sverif_cmb.mpibox.stacks['lensed_p2d'][0,0]
    np.save(save_dir+"p2d_tt.npy",p2d_tt)
    io.quickPlot2d(np.fft.fftshift(np.log10(p2d_tt)),pout_dir+"p2dtt.png",aspect="auto",lim=[-22,2])

    plot_list = ["lensed"]
    spec_list = ["TT","EE","BB"] if pol else ['TT']
    pl = io.Plotter(scaleY='log')
    for spec in spec_list:
        sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
    pl.done(pout_dir+"cl.png")

    if pol:
        plot_list = ["lensed"]
        spec_list = ["TE"]
        pl = io.Plotter()
        for spec in spec_list:
            sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
        pl.done(pout_dir+"clte.png")

        plot_list = ["lensed"]
        spec_list = ["EB","TB"]
        for spec in spec_list:
            sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,skip_uzero=False)
        pl.done(pout_dir+"clzero.png")

    plot_list = ["lensed"]
    spec_list = ["TT","EE","BB"] if pol else ['TT']
    pl = io.Plotter()
    for spec in spec_list:
        sverif_cmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],ylim=[-0.1,0.05],pl=pl,save_root=save_dir)
    pl.done(pout_dir+"cldiffrat.png")

    if pol:
        plot_list = ["lensed"]
        spec_list = ["TE","EB","TB"]
        pl = io.Plotter()
        for spec in spec_list:
            sverif_cmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,ratio=False)
        pl.done(pout_dir+"cldiff.png")

    
