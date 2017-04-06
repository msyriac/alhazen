import matplotlib
matplotlib.use('Agg')
import os
from alhazen.shear import dndz
import numpy as np
from orphics.tools import io
import alhazen.halos as halos

from scipy.interpolate import interp1d

ngalArcminSquare = 20.


# fzrange,nz = np.loadtxt("../cmb-lensing-projections/data/hscd6.csv",unpack=True)
# zfunc = interp1d(fzrange,nz)


zmax = 3.0
w = 0.1
zrange = np.arange(0.,zmax-w/2.,w)+w/2.
z_edges = np.arange(0.,zmax+w,w)

nz = dndz(zrange)
#nz = zfunc(zrange)

# sys.exit()


norm = np.trapz(nz,zrange)
nz /= norm

fracgals = []
for i,z in enumerate(zrange):
    fracgal = np.trapz(dndz(zrange[i:]),zrange[i:,])
    fracgals.append(fracgal)
    print z

pl = io.Plotter()
pl.add(zrange,fracgals)
outDir = os.environ['WWW']+"plots/"
pl.done(outDir + "fracgals.png")


print "imports"
from orphics.theory.gaussianCov import LensForecast
from orphics.theory.cosmology import LimberCosmology
from szlib.szcounts import ClusterCosmology
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
print "imports done"

iniFile = "../SZ_filter/input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmologyName = 'params' # from ini file
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')

maximum_ell = 8000
cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "../SZ_filter/data/cltt_lensed_Feb18.txt")
lc = LimberCosmology(cosmoDict,constDict,lmax=maximum_ell,pickling=True,numz=100,kmax=42.47,nonlinear=True)
#lc = LimberCosmology(cosmoDict,constDict,lmax=maximum_ell,pickling=True,numz=100,kmax=10.0,nonlinear=True)

ellrange = np.arange(100,maximum_ell,40)




M500 = 10**14.7 #5.e14
c500 = 1.18
# zlens = 0.5
delta = 500

lc.addNz("gal",z_edges,nz,bias=None,magbias=None,numzIntegral=300) # hack
lc.generateCls(ellrange,autoOnly=True,zmin=0.)
Clss = lc.getCl("gal","gal")

Clkk = lc.getCl("cmb","cmb")
ls,Nlkk = np.loadtxt("../SZ_filter/data/LA_all_Nl.txt",unpack=True,delimiter=",")
LF = LensForecast()
LF.loadKK(ellrange,Clkk,ls,Nlkk)

sns = []
snsCMB = []
winsGal = []
winsCMB = []

stdsGal = []
stdsCMB = []

import cPickle as pickle
mexpgrid,zgrid,errgrid = pickle.load(open("../SZ_filter/data/owl2.pkl",'rb'))
sngrid = 1./errgrid
print sngrid[np.where(np.isclose(mexpgrid,np.log10(M500))),:].ravel()

#sys.exit()

sigGals = []
sigCMBs = []

pl = Plotter(scaleY='log')


for i,z in enumerate(zrange):
    #if z<2.5: continue

    zlens = z
    lc.addStepNz("gal"+str(i),zlens,zmax,bias=None,magbias=None,numzIntegral=300)
    R500 = cc.rdel_c(M500,zlens,delta)

    win = lc.kernels["gal"+str(i)]["window_z"](zlens)
    comL  = (cc.results.comoving_radial_distance(zlens) )*cc.h
    winsGal.append(win*comL*(1.+zlens)/R500**2.)
    winsCMB.append(lc.kernels["cmb"]["window_z"](zlens)*comL*(1.+zlens)/R500**2.)

    
    LF.loadSS(ellrange,Clss,ngal=ngalArcminSquare*fracgals[i],shapeNoise=0.3)
    totCl = Clss+LF.Nls['ss'](ellrange)

    pl.add(ellrange,totCl,color="darkslateblue",alpha=(i+1.)/len(zrange))

    
    snGal,k500gal,stdgal = halos.NFWMatchedFilterSN(cc,np.log10(M500),c500,zlens,ellrange,totCl,maximum_ell,overdensity=500.,critical=True,atClusterZ=True,winAtLens=win)
    sigGals.append(k500gal)
    stdsGal.append(stdgal)
    sns.append(snGal)

    snCMB,k500CMB,stdcmb = halos.NFWMatchedFilterSN(cc,np.log10(M500),c500,zlens,ellrange,Clkk+LF.Nls['kk'](ellrange),maximum_ell,overdensity=500.,critical=True,atClusterZ=True,winAtLens=None)
    snsCMB.append(snCMB)
    sigCMBs.append(k500CMB)
    stdsCMB.append(stdcmb)

pl.done(outDir+"clss.png")
    
sns = np.array(sns)
snsCMB = np.array(snsCMB)
sigGals = np.array(sigGals)
sigCMBs = np.array(sigCMBs)
stdsGal = np.array(stdsGal)
stdsCMB = np.array(stdsCMB)

pl = io.Plotter()
pl.add(zrange,snsCMB)
pl.add(zrange,sns,ls="--")
pl.add(zgrid,sngrid[np.where(np.isclose(mexpgrid,np.log10(M500))),:].ravel(),ls="--",label="from nick")
#outDir = "output/"#os.environ['WWW']+"plots/"
pl.done(outDir + "sns.png")

pl = io.Plotter()
pl.add(zrange,sigGals,ls="--")
pl.add(zrange,sigCMBs)
#outDir = "output/"#os.environ['WWW']+"plots/"
pl.done(outDir + "sigs.png")

pl = io.Plotter()
pl.add(zrange,stdsGal,ls="--")
pl.add(zrange,stdsCMB)
#outDir = "output/"#os.environ['WWW']+"plots/"
pl.done(outDir + "stds.png")


pl = io.Plotter()
pl.add(zrange,winsGal,ls="--")
pl.add(zrange,winsCMB)
#outDir = "output/"#os.environ['WWW']+"plots/"
pl.done(outDir + "wins.png")
