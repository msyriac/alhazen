"""


A script to explore how enlib treats power
spectra.


"""


import os,sys
from enlib import enmap,powspec,utils
from orphics.theory.cosmology import Cosmology
import orphics.tools.io as io
import numpy as np
from scipy.interpolate import interp1d
import orphics.tools.stats as stats


out_dir = os.environ['WWW']

ps = powspec.read_spectrum("data/cl_lensed.dat")
print((ps.shape))

TCMB = 2.7255e6
cc = Cosmology(lmax=3000,pickling=True)
ells = np.arange(2,3000,1)

cctt = cc.theory.lCl('TT',ells)*TCMB**2.
ccte = cc.theory.lCl('TE',ells)*TCMB**2.
ccee = cc.theory.lCl('EE',ells)*TCMB**2.
ccbb = cc.theory.lCl('BB',ells)*TCMB**2.


enells = np.asarray(list(range(ps.shape[2])))[2:]
entt = ps[0,0,2:]
ente = ps[0,1,2:]
enet = ps[1,0,2:]
enee = ps[1,1,2:]
enbb = ps[2,2,2:]

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,cctt*ells**2.,color="C0",ls="-")
pl.add(ells,ccee*ells**2.,color="C1",ls="-")
pl.add(ells,ccbb*ells**2.,color="C2",ls="-")
pl.add(enells,entt*enells**2.,color="C0",ls="--")
pl.add(enells,enee*enells**2.,color="C1",ls="--")
pl.add(enells,enbb*enells**2.,color="C2",ls="--")
pl.done(out_dir+"ccomp.png")


pl = io.Plotter(scaleX='log')
pl.add(ells,ccte*ells**2.,color="C0",ls="-")
pl.add(enells,ente*enells**2.,color="C0",ls="--")
pl.add(enells,enet*enells**2.,color="C0",ls="-.")
pl.done(out_dir+"ccompte.png")



# === TEMPLATE MAPS ===
px = 0.5
arc = 10.*60.
hwidth = arc/2.
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")

shape = (3,)+shape

np.random.seed(100)

ps2d = enmap.spec2flat(shape,wcs,ps)



ps2dtest = ps2d[0,0]
psfunc = interp1d(ells,cctt,fill_value=0.,bounds_error=False)
modlmap = enmap.modlmap(shape,wcs)
ps2dalt = psfunc(modlmap)

io.quickPlot2d(np.log(ps2dtest),out_dir+"ps1.png")
io.quickPlot2d(np.log(ps2dalt),out_dir+"ps2.png")
#io.quickPlot2d(np.nan_to_num(ps2dtest/ps2dalt),out_dir+"psrat.png")


ellbin_edges = np.arange(2,3000,60)
binner = stats.bin2D(modlmap,ellbin_edges)
cents,ps1dtest = binner.bin(ps2dtest)
cents,ps1dalt = binner.bin(ps2dalt)


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,cctt*ells**2.,color="C1",ls="-")
pl.add(cents,ps1dtest*cents**2.,color="C0",ls="-")
pl.add(cents,ps1dalt*cents**2.,color="C0",ls="--")
pl.done(out_dir+"ccomp1d.png")
