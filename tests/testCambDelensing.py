import os,sys
import camb.correlations as corr
from orphics.theory.cosmology import Cosmology
import orphics.tools.io as io
import numpy as np

camb_lmax = 8500
cc = Cosmology(lmax=camb_lmax,pickling=True,dimensionless=False)
theory = cc.theory


def get_lensed_cls(theory,ells,clkk,lmax):
    
    ellrange = np.arange(0,lmax+2000,1)
    mulfact = ellrange*(ellrange+1.)/2./np.pi
    ucltt = theory.uCl('TT',ellrange)*mulfact
    uclee = theory.uCl('EE',ellrange)*mulfact
    uclbb = theory.uCl('BB',ellrange)*mulfact
    uclte = theory.uCl('TE',ellrange)*mulfact
    from scipy.interpolate import interp1d
    clkkfunc = interp1d(ells,clkk)
    clpp = clkkfunc(ellrange)*4./2./np.pi

    cmbarr = np.vstack((ucltt,uclee,uclbb,uclte)).T
    print "Calculating lensed cls..."
    lcls = corr.lensed_cls(cmbarr,clpp)
    ellrange = cellrange.ravel()[:lmax]
    lclall = lcls[:lmax,0]
    lclall = np.nan_to_num(lclall/cellrange/(cellrange+1.)*2.*np.pi)
    #clcltt = lcls[:lmax,0]
    #clcltt = np.nan_to_num(clcltt/cellrange/(cellrange+1.)*2.*np.pi)
    #print clcltt
    lpad = lmax
    import orphics.tools.cmb as cmb
    dtheory = cmb.TheorySpectra()
    mult = 1./multfact
    ucltt *= mult
    uclee *= mult
    uclte *= mult
    uclbb *= mult
    dtheory.loadCls(ellrange,ucltt,'TT',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,uclte,'TE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,uclee,'EE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,uclbb,'BB',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadGenericCls(ells,clkk,"kk",lpad=lpad)

    lcltt = lcall[:,0]
    lclee = lcall[:,1]
    lclbb = lcall[:,2]
    lclte = lcall[:,3]
    #lcltt *= mult
    #lclee *= mult
    #lclte *= mult
    #lclbb *= mult
    dtheory.loadCls(ellrange,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(ellrange,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)


    return dtheory


lmax_of_interest = 6000
assert (lmax_of_interest+2000)<camb_lmax

ellrange = np.arange(0,lmax_of_interest+2000,1)
clkk = theory.gCl('kk',ellrange)
dtheory = get_lensed_cls(theory,ellrange,clkk,lmax_of_interest)

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




