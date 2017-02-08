print "Importing modules..."
import matplotlib
matplotlib.use('Agg')
from enlib import enmap,utils,lensing,powspec
import numpy as np
from alhazen.halos import NFWkappa
from alhazen.lensTools import alphaMaker
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from szlib.szcounts import ClusterCosmology
print "Done importing modules..."

# === COSMOLOGY ===
cosmologyName = 'LACosmology' # from ini file
iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
lmax = 5000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax)

# === NFW CLUSTER ===
massOverh = 2.e14
concentration = 3.2
zL = 0.7
sourceZ = 1100.
overdensity = 180.
critical = False
atClusterZ = False

# === TEMPLATE MAP ===
px = 0.5
arc = 300
hwidth = arc/2.
hwidthTen = 5.
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
shapeTen, wcsTen = enmap.geometry(pos=[[-hwidthTen*arcmin,-hwidthTen*arcmin],[hwidthTen*arcmin,hwidthTen*arcmin]], res=px*arcmin, proj="car")
thetaMap = enmap.posmap(shape, wcs)
thetaMap = np.sum(thetaMap**2,0)**0.5


# === KAPPA MAP ===
kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,sourceZ,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
print "kappaint ", kappaMap[thetaMap*60.*180./np.pi<10.].mean()


# === DEFLECTION MAP ===
a = alphaMaker(thetaMap)
alpha = a.kappaToAlpha(kappaMap)
alphamod = 180.*60.*np.sum(alpha**2,0)**0.5/np.pi
print "alphaint ", alphamod[thetaMap*60.*180./np.pi<10.].mean()
pos = kappaMap.posmap() + alpha
pix = kappaMap.sky2pix(pos, safe=False)

# === CMB POWER SPECTRUM ===      
ps = powspec.read_spectrum("data/cl_lensinput.dat")


# === RUN N SIMS ===      
N = 100
for i in range(N):
    print i
    map = enmap.rand_map(shape, wcs, ps)
    lensedMap = lensing.displace_map(map, pix,order=5)
    
    if i==0:
        pl = Plotter()
        #pl.plot2d(lensedMap[250:-250,250:-250]-map[250:-250,250:-250])
        pl.plot2d(enmap.project(lensedMap,shapeTen,wcsTen))
        pl.done("output/lensed.png")


    
