import os,sys
import camb.correlations as corr
from orphics.theory.cosmology import Cosmology
import orphics.tools.io as io
import numpy as np

camb_lmax = 8500
cc = Cosmology(lmax=camb_lmax,pickling=True,dimensionless=False)
theory = cc.theory


def get_corr_cls(theory,lmax):
    
    ellrange = np.arange(0,lmax+2000,1)
    mulfact = ellrange*(ellrange+1.)/2./np.pi
    ucltt = theory.uCl('TT',ellrange)*mulfact
    uclee = theory.uCl('EE',ellrange)*mulfact
    uclbb = theory.uCl('BB',ellrange)*mulfact
    uclte = theory.uCl('TE',ellrange)*mulfact
    clpp = theory.gCl('kk',ellrange)*4./2./np.pi

    cmbarr = np.vstack((ucltt,uclee,uclbb,uclte)).T
    # print cmbarr.shape
    # print cmbarr[0,:]
    # print cmbarr[1,:]
    # print cmbarr[2,:]
    print "Calculating lensed cls..."
    lcls = corr.lensed_cls(cmbarr,clpp)
    cellrange = cellrange.ravel()[2:lmax]
    print lcls.shape
    clcltt = lcls[2:lmax,0]
    clcltt = np.nan_to_num(clcltt/cellrange/(cellrange+1.)*2.*np.pi)
    print clcltt

    import orphics.tools.cmb as cmb
    dtheory = cmb.TheorySpectra()

    dtheory.loadCls(ellrange,ucltt,'TT',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ell,uclte,'TE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ell,uclee,'EE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ell,uclbb,'BB',lensed=False,interporder="linear",lpad=lpad)
    theory.loadGenericCls(ell,clkk,"kk",lpad=lpad)

    ell, lcltt, lclee, lclbb, lclte = np.loadtxt(lFile,unpack=True,usecols=[0,1,2,3,4])
    mult = 2.*np.pi/ell/(ell+1.)/TCMB**2.
    lcltt *= mult
    lclee *= mult
    lclte *= mult
    lclbb *= mult
    theory.loadCls(ell,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)


    return ellrange, cmbarr,clpp


lmax_of_interest = 6000
assert (lmax_of_interest+2000)<camb_lmax

cellrange, cmbarr, clpp = get_corr_cls(theory,lmax_of_interest)

#sys.exit()
    
ellrange = np.arange(2,lmax_of_interest,1)
ucltt = theory.uCl('TT',ellrange)
lcltt = theory.lCl('TT',ellrange)
clkk = theory.gCl('kk',ellrange)
pl = io.Plotter(scaleY='log')
pl.add(ellrange,ucltt*ellrange**2.)
pl.add(ellrange,lcltt*ellrange**2.)
pl.add(cellrange,clcltt*cellrange**2.)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dcltt.png")



pl = io.Plotter()
pl.add(ellrange,(lcltt-ucltt)*ellrange**2.)
pl.add(cellrange,(clcltt-ucltt)*cellrange**2.)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttdiff.png")

pl = io.Plotter()
pl.add(ellrange,(lcltt-ucltt)/ucltt)
pl.add(cellrange,(clcltt-ucltt)/ucltt)
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttfrac.png")


pl = io.Plotter()
pl.add(cellrange,(clcltt-lcltt)/lcltt,ls="--")
pl._ax.set_xlim(2,lmax_of_interest)
pl.done("dclttfrac2.png")




