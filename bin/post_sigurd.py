from enlib import enmap
import numpy as np
import orphics.tools.io as io
import os,sys

lmax = 2500
autokk = {}
inputkk = {}
#meanfield = {}
#meanfieled_pwr = {}
cross = {}
n0 = {}
for region in ['equator','south']:
    print region
    save_dir = "/gpfs01/astro/workarea/msyriac/data/depot/distortions/distspectrav4mfsub_"+region+"_"
    autokk[region] = np.loadtxt(save_dir+"autokk.txt",unpack=True)
    inputkk[region] = np.loadtxt(save_dir+"ikk.txt",unpack=True)
    cross[region] = np.loadtxt(save_dir+"rxikk.txt",unpack=True)
    n0[region] = np.loadtxt(save_dir+"sdn0.txt",unpack=True)
    

ells = inputkk["equator"][0]


# First let's look at the difference of the inputs
pl = io.Plotter(labelX="$L$",labelY="$\Delta C_L^{II}/C_L^{II}$")
eq = inputkk["equator"][1]
so = inputkk["south"][1]
pl.add(ells,(so-eq)/eq)
pl.hline()
pl._ax.set_xlim(0,lmax)
pl.done(io.dout_dir+"dist_inpdiff.png")



# Next diff w.r.t input
pl = io.Plotter(labelX="$L$",labelY="$(C_L^{IR}-C_L^{II})/C_L^{II}$")
for ls,region in zip(("-","--"),['equator','south']):
    inp = inputkk[region][1]
    crossp = cross[region][1]
    pl.add(ells,(crossp-inp)/inp,ls=ls,label=region)
pl.hline()
pl.legendOn()
pl._ax.set_xlim(0,lmax)
pl.done(io.dout_dir+"dist_crossdiff.png")



# Next absolute diff of the two
pl = io.Plotter(labelX="$L$",labelY="$(C_s^{RR}-C_e^{RR})$")
autoeq = autokk["equator"][1]
autoso = autokk["south"][1]
pl.add(ells,(autoso-autoeq))
pl.hline()
pl._ax.set_xlim(0,lmax)
pl.done(io.dout_dir+"dist_autodiff.png")



pl = io.Plotter(labelX="$L$",labelY="$C_L$")
for k,(col,ls,region) in enumerate(zip(("C0","C1"),("-","--"),['equator','south'])):
    pl.add(ells,inputkk[region][1],ls=ls,color="k",alpha=0.7)
    pl.add(ells+k*20,cross[region][1],color="C"+str(k*2+1),marker="^",ls="none",label=region+" cross")
    pl.addErr(ells,autokk[region][1],yerr=autokk[region][2],ls="none",marker="o",label=region+" n0 subbed auto",color="C"+str(k*2))
    #pl.add(ells,ells*inputkk[region][1],ls=ls)
    #pl.addErr(ells,ells*autokk[region][1],yerr=ells*autokk[region][2],ls="none",marker="o")
pl.hline()
pl.legendOn(loc="upper right")
pl._ax.set_xlim(0,lmax)
pl.done(io.dout_dir+"dist_auto.png")


# Next absolute diff of the two
pl = io.Plotter(labelX="$L$",labelY="$(C_s^{RR}-C_e^{RR})/C_e^{RR}$")
autoeq = autokk["equator"][1]
autoso = autokk["south"][1]
pl.add(ells,(autoso-autoeq)/autoeq)
pl.hline()
pl.hline(y=0.2,ls="-")
pl._ax.set_ylim(-0.01,1.)
pl._ax.set_xlim(0,lmax)
pl.done(io.dout_dir+"dist_autodiffrat.png")
