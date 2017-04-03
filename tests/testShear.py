import matplotlib
matplotlib.use('Agg')
import os
from alhazen.shear import dndz
import numpy as np
from orphics.tools import io
import alhazen.halos as halos

ngalArcminSquare = 30.

zmax = 3.0
w = 0.1
zrange = np.arange(0.,zmax-w/2.,w)+w/2.
z_edges = np.arange(0.,zmax+w,w)

nz = dndz(zrange)
print np.trapz(nz,zrange)

fracgals = []
for i,z in enumerate(zrange):
    fracgal = np.trapz(dndz(zrange[i:]),zrange[i:,])
    fracgals.append(fracgal)
    print z

# pl = io.Plotter()
# pl.add(zrange,fracgals)
# outDir = os.environ['WWW']+"plots/"
# pl.done(outDir + "fracgals.png")


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

cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "../SZ_filter/data/cltt_lensed_Feb18.txt")
#lc = LimberCosmology(cosmoDict,constDict,lmax=6000,pickling=True,numz=100,kmax=42.47,nonlinear=True)
lc = LimberCosmology(cosmoDict,constDict,lmax=6000,pickling=True,numz=100,kmax=10.0,nonlinear=True)

ellrange = np.arange(100,6000,40)




M500 = 5.e14
c500 = 1.18
zlens = 0.5
delta = 500
R500 = cc.rdel_c(M500,zlens,delta)

lc.addNz("gal",z_edges,nz,bias=None,magbias=None,numzIntegral=300) # hack
lc.generateCls(ellrange,autoOnly=True,zmin=0.)
Clss = lc.getCl("gal","gal")

Clkk = lc.getCl("cmb","cmb")
ls,Nlkk = np.loadtxt("../SZ_filter/data/LA_pol_Nl.txt",unpack=True,delimiter=",")
LF = LensForecast()
LF.loadKK(ellrange,Clkk,ls,Nlkk)

sns = []
snsCMB = []

for i,z in enumerate(zrange):

    new_nz = nz.copy()
    new_nz[zrange<z] = 0.
    norm = np.trapz(new_nz,zrange)
    new_nz = new_nz/norm
    lc.addNz("gal"+str(i),z_edges,new_nz,bias=None,magbias=None,numzIntegral=300) # hack

    win = lc.kernels["gal"+str(i)]["window_z"](zlens)

    LF.loadSS(ellrange,Clss,ngal=ngalArcminSquare*fracgals[i],shapeNoise=0.3)
    totCl = Clss+LF.Nls['ss'](ellrange)

    
    sn = halos.NFWMatchedFilterSN(cc,np.log10(M500),c500,zlens,ellrange,totCl,6000,overdensity=500.,critical=True,atClusterZ=True,winAtLens=win)

    sns.append(sn)

    sn = halos.NFWMatchedFilterSN(cc,np.log10(M500),c500,zlens,ellrange,Clkk+LF.Nls['kk'](ellrange),6000,overdensity=500.,critical=True,atClusterZ=True,winAtLens=None)
    snsCMB.append(sn)

    
sns = np.array(sns)
snsCMB = np.array(snsCMB)
np.savetxt("data/owlsns.txt",np.vstack((zrange,sns)).transpose())
np.savetxt("data/cmbsns.txt",np.vstack((zrange,snsCMB)).transpose())

pl = io.Plotter()
pl.add(zrange,snsCMB)
pl.add(zrange,sns,ls="--")
outDir = "output/"#os.environ['WWW']+"plots/"
pl.done(outDir + "sns.png")
