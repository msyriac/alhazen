import os,sys
from orphics.theory.cosmology import Cosmology
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np

camb_lmax = 8500
cc = Cosmology(lmax=camb_lmax,pickling=True,dimensionless=False)
theory = cc.theory

lmax_of_interest = 6000
assert (lmax_of_interest+2000)<camb_lmax

ellrange = np.arange(0,lmax_of_interest+2000,1)
clkk = theory.gCl('kk',ellrange)
dtheory = cmb.get_lensed_cls(theory,ellrange,clkk,lmax_of_interest)

#sys.exit()
    
ellrange = np.arange(2,lmax_of_interest,1)
ucltt = theory.uCl('TT',ellrange)
lcltt = theory.lCl('TT',ellrange)
clcltt = dtheory.lCl('TT',ellrange)
clkk = theory.gCl('kk',ellrange)
pl = io.Plotter(scaleY='log')
pl.add(ellrange,ucltt*ellrange**2.)
pl.add(ellrange,lcltt*ellrange**2.)
pl.add(ellrange,clcltt*ellrange**2.)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dcltt.png")



pl = io.Plotter()
pl.add(ellrange,(lcltt-ucltt)*ellrange**2.)
pl.add(ellrange,(clcltt-ucltt)*ellrange**2.)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttdiff.png")

pl = io.Plotter()
pl.add(ellrange,(lcltt-ucltt)/ucltt)
pl.add(ellrange,(clcltt-ucltt)/ucltt)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttfrac.png")


pl = io.Plotter()
pl.add(ellrange,(clcltt-lcltt)/lcltt,ls="--")
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttfrac2.png")




