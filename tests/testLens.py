print "Importing modules..."
import matplotlib
matplotlib.use('Agg')
from enlib import enmap,utils
import numpy as np
from alhazen.halos import NFWkappa
from alhazen.lensTools import alphaMaker
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from szlib.szcounts import ClusterCosmology
print "Done importing modules..."

cosmologyName = 'LACosmology' # from ini file

iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 8000

cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax)


massOverh = 2.e14
concentration = 3.2
zL = 0.7
sourceZ = 1100.
overdensity = 180.
critical = False
atClusterZ = False


px = 0.5
arc = 100
hwidth = arc/2.

deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
thetaMap = enmap.posmap(shape, wcs)
thetaMap = np.sum(thetaMap**2,0)**0.5

# pl = Plotter()
# pl.plot2d(thetaMap)
# pl.done("output/theta.png")



kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,sourceZ,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


a = alphaMaker(thetaMap)
alpha = a.kappaToAlpha(kappaMap,test=True)

print alpha.shape

alphamod = np.sum(alpha**2,0)**0.5

# pl = Plotter()
# pl.plot2d(kappaMap)
# pl.done("output/kappa.png")

# pl = Plotter()
# pl.plot2d(alpha[0,:,:])
# pl.done("output/alphax.png")

# pl = Plotter()
# pl.plot2d(alpha[1,:,:])
# pl.done("output/alphay.png")

pl = Plotter()
pl.plot2d(alphamod)
pl.done("output/alphamod.png")
