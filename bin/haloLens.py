print("Importing modules...")
import matplotlib
matplotlib.use('Agg')
from enlib import enmap,utils,lensing,powspec
import numpy as np
from alhazen.halos import NFWkappa
from alhazen.lensTools import alphaMaker
from configparser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
import orphics.tools.io as io
from szar.counts import ClusterCosmology
from orphics.tools.stats import bin2D
from szar.sims import BattagliaSims, getKappaSZ
from enlib.fft import fft,ifft
from alhazen.quadraticEstimator import Estimator
import os
print("Done importing modules...")


out_dir = os.environ['WWW']+"lenstests/"

# === COSMOLOGY ===
cosmologyName = 'params' # from ini file
iniFile = "../szar/input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
lmax = 8000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)
TCMB = 2.7255e6

# === NFW CLUSTER ===
massOverh = 2.e14
concentration = 3.2
zL = 0.7
sourceZ = 1100.

# overdensity = 180.
# critical = False
# atClusterZ = False

overdensity = 500.
critical = True
atClusterZ = True


# === TEMPLATE MAP ===
px = 0.1
arc = 50
hwidth = arc/2.
pxDown = 0.2
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
shapeDown, wcsDown = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=pxDown*arcmin, proj="car")
thetaMap = enmap.posmap(shape, wcs)
thetaMap = np.sum(thetaMap**2,0)**0.5
thetaMapDown = enmap.posmap(shapeDown, wcsDown)
thetaMapDown = np.sum(thetaMapDown**2,0)**0.5


comL = cc.results.comoving_radial_distance(zL)*cc.h
comS = cc.results.comoving_radial_distance(sourceZ)*cc.h
winAtLens = (comS-comL)/comS

kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


# === CMB POWER SPECTRUM ===      
ps = powspec.read_spectrum("data/cl_lensinput.dat")


# === DEFLECTION MAP ===
a = alphaMaker(thetaMap)
alpha = a.kappaToAlpha(kappaMap,test=False)
pos = thetaMap.posmap() + alpha
pix = thetaMap.sky2pix(pos, safe=False)

gradfit = 2.*np.pi/180./60.

from scipy.ndimage.interpolation import rotate

N = 100
avgrot = 0.
for i in range(N):
    map = enmap.rand_map(shape, wcs, ps)/TCMB

    gmap = enmap.grad(map)
    angle = np.arctan2(gmap[0][np.where(thetaMap<gradfit)].mean(),gmap[1][np.where(thetaMap<gradfit)].mean())*180./np.pi
    
    lensedTQU = lensing.displace_map(map, pix,order=5)+0.j


    diffmap =  (lensedTQU-map).real
    rotmap = rotate(diffmap, angle,reshape=False)
    avgrot += rotmap
    
    print((i,angle))
    #io.highResPlot2d(map,out_dir+"unl"+str(i)+".png")
    #io.highResPlot2d(rotmap,out_dir+"rot"+str(i)+".png")


avgrot /= N

io.highResPlot2d(avgrot,out_dir+"avgrot.png")

    
