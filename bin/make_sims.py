import numpy as np
import sys, os
from orphics.analysis.pipeline import mpi_distribute, MPIStats, SpectrumVerification
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from enlib import enmap, lensing, resample
import alhazen.lensTools as lt
from configparser import SafeConfigParser 
import enlib.fft as fftfast
import argparse
from mpi4py import MPI


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

import argparse
parser = argparse.ArgumentParser(description='Make flat-sky sims meant for Clkk analysis')
parser.add_argument("Analysis", type=str,help='Analysis section name')
parser.add_argument("Sims", type=str,help='Sim section name')
parser.add_argument("Cosmology", type=str,help='Cosmology section name')
parser.add_argument("Kappa", type=str,help='Kappa section name')
parser.add_argument("Out", type=str,help='Output Directory Name (not path, that\'s specified in ini')
parser.add_argument("Nsims", type=int,help='Total number of sims')
parser.add_argument("-s", "--covseed",     type=int,  default=0)
parser.add_argument("-x", "--kellmin",     type=int,  default=100)
parser.add_argument("-y", "--kellmax",     type=int,  default=3000)
parser.add_argument("-t", "--skip_pol", action='store_true',help='Skip polarization')
parser.add_argument("-v", "--verify_only", action='store_true',help='Does not generate sims. Reads them from disk and verifies them.')

args = parser.parse_args()
Nsims = args.Nsims
Nsets = 7
vonly = args.verify_only

Config = io.config_from_file("../halofg/input/recon.ini")
pout_dir = os.environ['WWW']+"plots/sims_"
fout_dir = Config.get('general','output_dir')+args.Out+"/"



if rank==0: # POSSIBLE RACE CONDITION!!!
    if not os.path.exists(fout_dir):
        os.makedirs(fout_dir)
analysis_section = args.Analysis
sim_section = args.Sims
kappa_section = args.Kappa
expf_name = "experiment_noiseless"
cosmology_section = args.Cosmology


if not(vonly):
    keep_sections = [analysis_section,sim_section,kappa_section,cosmology_section,expf_name]
    for section in Config.sections():
        if section not in keep_sections: Config.remove_section(section)

    Config.add_section("general")
    Config.set("general","analysis",analysis_section)
    Config.set("general","sims",sim_section)
    Config.set("general","kappa",kappa_section)
    Config.set("general","cosmology",cosmology_section)
    Config.set("general","cosmology",cosmology_section)
    Config.set("general","Nsims",str(Nsims))
    Config.set("general","Nsets",str(Nsets))
    Config.set("general","pol",str(not(args.skip_pol)))
    Config.write(open(fout_dir+"config.ini",'w'))


lens_order = Config.get(sim_section,'lens_order')


pol = not(args.skip_pol)
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
if rank==0:
    print(("Dat shape : ", shape_dat))
    print(("Sim shape : ",shape_sim))
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
pixratio = analysis_resolution/Config.getfloat(sim_section,"pixel_arcmin")
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=False)



# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Nsims,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print(("At most ", max(num_each) , " tasks..."))
# What am I doing?
my_tasks = each_tasks[rank]

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section,dimensionless=False)
parray_dat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
parray_sim.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)

lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
kellmin = args.kellmin
kellmax = args.kellmax
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)

sverif_cmb = SpectrumVerification(mpibox,theory,shape_sim,wcs_sim,lbinner=lbinner_sim,pol=pol)
sverif_dcmb = SpectrumVerification(mpibox,theory,shape_dat,wcs_dat,lbinner=lbinner_dat,pol=pol)
sverif_kappa = SpectrumVerification(mpibox,theory,shape_sim[-2:],wcs_sim,lbinner=lbinner_sim,pol=False)
sverif_dkappa = SpectrumVerification(mpibox,theory,shape_dat[-2:],wcs_dat,lbinner=lbinner_dat,pol=False)



seedroot = args.covseed*Nsets*Nsims


def sim(cseed,kseed,skip_kappa_verif=False):

    if not(vonly):
        unlensed = parray_sim.get_unlensed_cmb(seed=seedroot+cseed,scalar=False)
        kappa = aio.kappa_from_config(Config,kappa_section,parray_sim,seed=seedroot+kseed)
        lensed = parray_sim.get_lensed(unlensed, order=lens_order, mode="spline", border="cyclic")

        luteb,dummy = sverif_cmb.add_power("unlensed",unlensed)
        llteb,dummy = sverif_cmb.add_power("lensed",lensed)
        if not(skip_kappa_verif): lk,dummy = sverif_kappa.add_power("kappa",kappa)

    cname = fout_dir+"lensed_covseed_"+str(args.covseed).zfill(3)+"_cmbseed_"+str(cseed).zfill(5)+"_kseed_"+str(kseed).zfill(5)+".hdf"
    kname = fout_dir+"kappa_covseed_"+str(args.covseed).zfill(3)+"_kseed_"+str(kseed).zfill(5)+".hdf"

    if vonly:
        dlensed = enmap.read_map(cname)
        dkappa = enmap.read_map(kname)
    else:
        dlensed = lensed if abs(pixratio-1.)<1.e-3 else enmap.ndmap(resample.resample_fft(lensed,shape_dat),wcs_dat)
        dkappa = kappa if abs(pixratio-1.)<1.e-3 else enmap.ndmap(resample.resample_fft(kappa,shape_dat[-2:]),wcs_dat)
        dlensed.write(cname)
        dkappa.write(kname)


    dllteb,dummy = sverif_dcmb.add_power("dlensed",dlensed)
    if not(skip_kappa_verif): dlk,dummy = sverif_dkappa.add_power("dkappa",dkappa)




for k,index in enumerate(my_tasks):
    if rank==0: print(("Rank ", rank, " doing job ",k+1, " / ",len(my_tasks),"..."))
    sim(index,Nsims+index)
    sim(2*Nsims+index,3*Nsims+index)
    sim(4*Nsims+index,5*Nsims+index)
    sim(6*Nsims+index,5*Nsims+index,skip_kappa_verif=True) # skip kappa verification because kappa is repeated
    



    

    
mpibox.get_stats()
if rank==0:

    plot_list = ["unlensed","lensed","dlensed"] if not(vonly) else ['dlensed']
    spec_list = ["TT","EE","BB"] if pol else ['TT']
    pl = io.Plotter(scaleY='log')
    for spec in spec_list:
        if not(vonly): sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
        sverif_dcmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
    pl.done(pout_dir+"cl.png")

    if pol:
        plot_list = ["unlensed","lensed","dlensed"] if not(vonly) else ['dlensed']
        spec_list = ["TE"]
        pl = io.Plotter()
        for spec in spec_list:
            if not(vonly): sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
            sverif_dcmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
        pl.done(pout_dir+"clte.png")

        if not(vonly):
            plot_list = ["unlensed"]
            spec_list = ["BB","EB","TB"]
            pl = io.Plotter()
            for spec in spec_list:
                if not(vonly): sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,skip_uzero=False)
                sverif_dcmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,skip_uzero=False)
        plot_list = ["lensed","dlensed"] if not(vonly) else ['dlensed']
        spec_list = ["EB","TB"]
        for spec in spec_list:
            if not(vonly): sverif_cmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,skip_uzero=False)
            sverif_dcmb.plot(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,skip_uzero=False)
        pl.done(pout_dir+"clzero.png")

    plot_list = ["unlensed","lensed","dlensed"] if not(vonly) else ['dlensed']
    spec_list = ["TT","EE","BB"] if pol else ['TT']
    pl = io.Plotter()
    for spec in spec_list:
        if not(vonly): sverif_cmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
        sverif_dcmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],pl=pl)
    pl.done(pout_dir+"cldiffrat.png")

    if pol:
        plot_list = ["unlensed","lensed","dlensed"] if not(vonly) else ['dlensed']
        spec_list = ["TE","EB","TB"]
        pl = io.Plotter()
        for spec in spec_list:
            if not(vonly): sverif_cmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,ratio=False)
            sverif_dcmb.plot_diff(spec,plot_list,xlim=[kellmin,kellmax],pl=pl,ratio=False)
        pl.done(pout_dir+"cldiff.png")

    
    spec = "kk"
    plot_list = ['kappa','dkappa'] if not(vonly) else ['dkappa']
    pl = io.Plotter(scaleY='log')
    if not(vonly): sverif_kappa.plot(spec,plot_list,scale_spectrum=False,pl=pl)
    sverif_dkappa.plot(spec,plot_list,scale_spectrum=False,pl=pl)
    pl.done(pout_dir+"clkk.png")
    
    spec = "kk"
    plot_list = ['kappa','dkappa']if not(vonly) else ['dkappa']
    pl = io.Plotter()
    if not(vonly): sverif_kappa.plot_diff(spec,plot_list,pl=pl)
    sverif_dkappa.plot_diff(spec,plot_list,pl=pl)
    pl.done(pout_dir+"clkkdiff.png")
    

