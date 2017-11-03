import sys, os
import numpy as np
from enlib import enmap # iau_convention must be False
import orphics.analysis.flatMaps as fmaps
import orphics.tools.stats as stats
import orphics.tools.io as io
from orphics.theory.cosmology import Cosmology


def plot_stats(cents,cont,ells,ells_pp,theory):
    if len(cont['tt'])<3: return
    
    print("Calculating stats...")
    st = {}
    for spec in ['tt','ee','te','bb','pp']:
        st[spec] = stats.getStats(cont[spec])

    print("Plotting...")
    pl = io.Plotter(scaleY='log',scaleX='log')
    for spec in ['tt','ee','bb']:
        pl.add(ells,theory[spec]*ells**2.,lw=2)
        pl.addErr(cents,st[spec]['mean']*cents**2.,yerr=st[spec]['errmean']*cents**2.,ls="none",marker="o")

    pl.done(os.environ['WORK']+"/web/plots/clsauto.png")
    pl = io.Plotter(scaleX='log')
    for spec in ['te']:
        pl.add(ells,theory[spec]*ells**2.,lw=2)
        pl.addErr(cents,st[spec]['mean']*cents**2.,yerr=st[spec]['errmean']*cents**2.,ls="none",marker="o")

    pl.done(os.environ['WORK']+"/web/plots/clste.png")
    pl = io.Plotter(scaleY='log')
    for spec in ['pp']:
        pl.add(ells_pp,theory[spec],lw=2)
        pl.addErr(cents,st[spec]['mean'],yerr=st[spec]['errmean'],ls="none",marker="o")
    pl._ax.set_xlim(2,3000)
    pl.done(os.environ['WORK']+"/web/plots/clspp.png")


root_dir = os.environ['WORK'] + "/data/sigurdsims/south"


Nsims = 32
taper_percent = 20.0  # smaller values cause larger biases in phi and B power




map_name = lambda x: root_dir+"_curved_lensed_car_"+str(x).zfill(2)+".fits"
phi_name = lambda x: root_dir+"_curved_phi_car_"+str(x).zfill(2)+".fits"


ps_name = "../tenki/cl_lensinput.dat"
ps_lensed_name = "../tenki/cl_lensed.dat"
cl = {}
ells_pp,d3,d4,d5,d6,cl['pp'],d1,d2 = np.loadtxt(ps_name,unpack=True)
ells,cl['tt'],cl['ee'],cl['bb'],cl['te'] = np.loadtxt(ps_lensed_name,unpack=True)

container = {}
theory = {}
for spec in ['tt','ee','te','bb','pp']:
    container[spec] = []
    if spec=='pp':
        theory[spec] = cl[spec]*2.*np.pi/4. #/ells_pp**2./(ells_pp+1.)**2.
    else:
        theory[spec] = cl[spec]*2.*np.pi/ells/(ells+1.)

for i in range(Nsims):

    imap = enmap.read_map(map_name(i)) 
    phi = enmap.read_map(phi_name(i)) 
    print(("=========== Loaded ", i,"==========="))

    if i==0:
        Ny,Nx = imap.shape[-2:]
        #taper = fmaps.cosineWindow(Ny,Nx,lenApodY=taper_percent*min(Ny,Nx)/100.,lenApodX=taper_percent*min(Ny,Nx)/100.,padY=0,padX=0)
        taper = fmaps.cosineWindow(Ny,Nx,lenApodY=taper_percent*Ny/100.,lenApodX=taper_percent*Nx/100.,padY=0,padX=0)
        w2 = np.mean(taper**2.)
        w4 = np.mean(taper**4.)

        modlmap = imap.modlmap()
        ellmax = 6000
        ellmin = 200
        ellwidth = 40
        bin_edges = np.arange(ellmin,ellmax,ellwidth)
        binner = stats.bin2D(modlmap,bin_edges)

        
    imap = imap*taper
    phi = phi*taper
    if i==0: io.quickPlot2d(phi,os.environ['WORK']+"/web/plots/phimap.png")


    print("IQU to TEB...")
    teb = enmap.ifft(enmap.map2harm(imap)).real

    print("Powers...")

    t = teb[0,:,:]
    e = teb[1,:,:]
    b = teb[2,:,:]
    spec2d = {}
    spec2d['tt'] = np.nan_to_num(fmaps.get_simple_power_enmap(t))/w2
    spec2d['ee'] = np.nan_to_num(fmaps.get_simple_power_enmap(e))/w2
    spec2d['bb'] = np.nan_to_num(fmaps.get_simple_power_enmap(b))/w2
    spec2d['te'] = np.nan_to_num(fmaps.get_simple_power_enmap(t,enmap2=e))/w2
    spec2d['pp'] = np.nan_to_num(fmaps.get_simple_power_enmap(phi))/w2*(modlmap*(modlmap+1.))**2./4.

    print("Binning...")
    
    for spec in ['tt','ee','te','bb','pp']:
        cents, clww = binner.bin(spec2d[spec])
        container[spec].append(clww)


    plot_stats(cents,container,ells,ells_pp,theory)
