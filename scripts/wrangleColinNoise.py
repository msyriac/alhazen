"""
Feb 21, 2017

This script reads in Colin Hill's noise curves and outputs them
in an alhazen-friendly format
"""

import numpy as np
import glob
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from orphics.tools.io import Plotter
import os

TCMB = 2.7255 # kelvin!

fileList = glob.glob("data/CMBdustsynch_*")

cambRoot = "data/ell28k_highacc"
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
pl = Plotter(scaleY='log')

for i,filen in enumerate(fileList):

    ells,d1,Nls,d2 = np.loadtxt(filen,unpack=True)
    Nls = Nls/TCMB**2.

    if i==0:
        pl.add(ells,theory.lCl('EE',ells),lw=2)

    lab = ""
    if "5m" in filen:
        lab = "5m"
        ls = "-"
    else:
        lab = "6m"
        ls = "--"

    if "synchGal" in filen:
        lab += "_dust_synch"
    elif "dustGal" in filen:
        lab += "_dust"
    pl.add(ells,Nls,label=lab,ls=ls)


    np.savetxt("data/colin_"+lab+".txt",np.vstack((ells,Nls)).transpose())

pl.legendOn(labsize=10,loc='lower left')
pl.done("output/colin.png")

    
    


