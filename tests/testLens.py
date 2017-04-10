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
from orphics.tools.stats import bin2D
from szlib.sims import BattagliaSims
from enlib.fft import fft,ifft
import os
print "Done importing modules..."


outDir = os.environ['WWW']+"plots/kappatest/"

def getKappaSZ(bSims,snap,massIndex,px,thetaMapshape):
    b = bSims
    #snap = 44
    #massIndex = 40
    PIX = 2048
    maps, z, kappaSimDat, szMapuKDat, projectedM500, trueM500, trueR500, pxInRad, pxInRad = b.getMaps(snap,massIndex,freqGHz=150.)
    pxIn = pxInRad * 180.*60./np.pi
    hwidth = PIX*pxIn/2.
    
    # input pixelization
    shapeSim, wcsSim = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=pxIn*arcmin, proj="car")
    kappaSim = enmap.enmap(kappaSimDat,wcsSim)
    szMapuK = enmap.enmap(szMapuKDat,wcsSim)
    
    # downgrade to native
    shapeOut, wcsOut = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
    kappaMap = enmap.project(kappaSim,shapeOut,wcsOut)
    szMap = enmap.project(szMapuK,shapeOut,wcsOut)

    print thetaMapshape
    print szMap.shape
    diffPad = ((np.array(thetaMapshape) - np.array(szMap.shape))/2.+0.5).astype(int)
    
    apodWidth = 25
    kappaMap = enmap.pad(enmap.apod(kappaMap,apodWidth),diffPad)[:-1,:-1]
    szMap = enmap.pad(enmap.apod(szMap,apodWidth),diffPad)[:-1,:-1]
    print szMap.shape
    assert szMap.shape==thetaMap.shape

    print z, projectedM500
    # pl = Plotter()
    # pl.plot2d(kappaMap)
    # pl.done(outDir+"kappasim.png")
    # pl = Plotter()
    # pl.plot2d(szMap)
    # pl.done(outDir+"szsim.png")
    # sys.exit()

    print "kappaint ", kappaMap[thetaMap*60.*180./np.pi<10.].mean()
    return kappaMap,szMap


# === COSMOLOGY ===
cosmologyName = 'LACosmology' # from ini file
iniFile = "../SZ_filter/input/params.ini"
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
# massOverh = 2.e15
# concentration = 3.2
# zL = 1.0
sourceZ = 1100.
overdensity = 180.
critical = False
atClusterZ = False

# === TEMPLATE MAP ===
px = 0.2
arc = 100
hwidth = arc/2.
hwidthTen = 5.
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
shapeTen, wcsTen = enmap.geometry(pos=[[-hwidthTen*arcmin,-hwidthTen*arcmin],[hwidthTen*arcmin,hwidthTen*arcmin]], res=px*arcmin, proj="car")
thetaMap = enmap.posmap(shape, wcs)
thetaMap = np.sum(thetaMap**2,0)**0.5


# === KAPPA MAP ===
comL = cc.results.comoving_radial_distance(zL)*cc.h
zstar = 1100.
comS = cc.results.comoving_radial_distance(zstar)*cc.h
winAtLens = (comS-comL)/comS

kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
arcmax = 20.
dt = 0.05
thetaRange = np.arange(dt,arcmax,dt)
kappa1d,r5001d = NFWkappa(cc,massOverh,concentration,zL,thetaRange,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
snap = 43
b = BattagliaSims(constDict)

# === CMB POWER SPECTRUM ===      
ps = powspec.read_spectrum("data/cl_lensinput.dat")



# === QUADRATIC ESTIMATOR INITIALIZATION ===      

class template:
    pass

templateLM = template()
templateLM.Ny, templateLM.Nx = thetaMap.shape
templateLM.pixScaleY, templateLM.pixScaleX = thetaMap.pixshape()

from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
lxMap,lyMap,modLMap,angMap,lx,ly = fmaps.getFTAttributesFromLiteMap(templateLM)

pol = False

if pol:
    #polCombList = ["TT","ET","EB"]
    polCombList = ["EB"]
    shape = (3,)+shape
else:
    polCombList = ["TT"]


theory = cc.theory
nT,nP,nP = fmaps.whiteNoise2D([3.0,3.0,3.0],0.01,modLMap,TCMB = TCMB)
gradCut = 2000
cmbellmin = 200
cmbellmax = 8000
kellmin = 100
kellmax = 8000

fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)
qest = Estimator(templateLM,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[fMaskCMB]*3,
                 fmaskY2dTEB=[fMaskCMB]*3,
                 fmaskKappa=fMask,
                 doCurl=False,
                 TOnly=not(pol),
                 halo=True,
                 gradCut=gradCut,verbose=True,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None)


# bin_edges = np.arange(kellmin,kellmax,20)
# b = bin2D(modLMap,bin_edges)
# ls,Nls = b.bin(qest.N.Nlkk['EB'])

# pl = Plotter(scaleY='log')
# pl.add(bin_edges,theory.gCl("kk",bin_edges))
# pl.add(ls,Nls)
# pl.done(outDir+"nl.png")

szX = False
szY = False

# szX = True
# szY = True

# szX = True
# szY = False

# szX = False
# szY = True


# === RUN N SIMS ===      
kappaStack = thetaMap*0.
trueKappaStack = thetaMap*0.
szStack = thetaMap*0.
lX = thetaMap*0.
lY = thetaMap*0.
N = 100
massIndices = range(300)
for i in range(N):
    print i
    map = enmap.rand_map(shape, wcs, ps)/TCMB


    massIndex = massIndices[i]
    inputKappaMap, szMap = getKappaSZ(b,snap,massIndex,px,thetaMap.shape)
    #inputKappaMap = kappaMap
    #szMap = 0.

    trueKappaStack += inputKappaMap
    szStack += szMap

    # === DEFLECTION MAP ===
    a = alphaMaker(thetaMap)
    alpha = a.kappaToAlpha(inputKappaMap,test=False)
    alphamod = 180.*60.*np.sum(alpha**2,0)**0.5/np.pi
    print "alphaint ", alphamod[thetaMap*60.*180./np.pi<10.].mean()
    pos = thetaMap.posmap() + alpha
    pix = thetaMap.sky2pix(pos, safe=False)





    lensedTQU = lensing.displace_map(map, pix,order=5)
    #lensedMapX = ifft(enmap.map2harm(lensedTQU),axes=[-2,-1],normalize=True).real 
    lensedMapX = enmap.ifft(enmap.map2harm(lensedTQU),normalize=True).real 
    lensedMapY = lensedMapX.copy()

    if szX:
        lensedMapX += (szMap/TCMB)
    if szY:
        lensedMapY += (szMap/TCMB)
    
    # if i==0:
    #     pl = Plotter()
    #     pl.plot2d(enmap.project(lensedMapX,shapeTen,wcsTen))
    #     pl.done(outDir+"lensedX.png")
    #     pl = Plotter()
    #     pl.plot2d(enmap.project(lensedMapY,shapeTen,wcsTen))
    #     pl.done(outDir+"lensedY.png")
    #     pl = Plotter()
    #     pl.plot2d(enmap.project(lensedMapY,shapeTen,wcsTen)-enmap.project(map,shapeTen,wcsTen))
    #     pl.done(outDir+"diff.png")

    
    fotX = fft(lensedMapX,axes=[-2,-1])
    fotY = fft(lensedMapY,axes=[-2,-1])

    print "Reconstructing" , i , " ..."
    qest.updateTEB_X(fotX,alreadyFTed=True)
    qest.updateTEB_Y(fotY,alreadyFTed=True)
    # qest.updateTEB_X(fotX[0],fotX[1],fotX[2],alreadyFTed=True)
    # qest.updateTEB_Y(fotY[0],fotY[1],fotY[2],alreadyFTed=True)
    kappa = qest.getKappa(polCombList[0]).real
        
    kappaStack += kappa
    # lX += lensedMapX
    # lY += lensedMapY



pl = Plotter()
pl.plot2d(kappaStack/N)
pl.done(outDir+"recon.png")

pl = Plotter()
pl.plot2d(trueKappaStack/N)
pl.done(outDir+"truestack.png")

pl = Plotter()
pl.plot2d(szStack/N)
pl.done(outDir+"szstack.png")

filtInput = fmaps.stepFunctionFilterLiteMap(trueKappaStack/N,modLMap,kellmax)

pl = Plotter()
pl.plot2d(filtInput)
pl.done(outDir+"filtinput.png")


dt = 0.2
thetaRange = np.arange(dt,arcmax,dt)
breal = bin2D(thetaMap*180.*60./np.pi,thetaRange)
cents,inps = breal.bin(trueKappaStack/N)
cents,inpsFilt = breal.bin(filtInput)
cents,recons = breal.bin(kappaStack/N)

pl = Plotter()
#pl.add(thetaRange,kappa1d)
pl.add(cents,inps,ls="--")
pl.add(cents,inpsFilt)
pl.add(cents,recons)
pl.done(outDir+"profiles.png")



# fotX = enmap.fft(lX,normalize=False)
# fotY = enmap.fft(lY,normalize=False)

# print "Reconstructing" , i , " ..."
# qest.updateTEB_X(fotX,alreadyFTed=True)
# qest.updateTEB_Y(fotY,alreadyFTed=True)
# kappa = qest.getKappa(polCombList[0]).real

# cleanKappa = kappaStack - kappa
# pl = Plotter()
# pl.plot2d(cleanKappa/N)
# pl.done(outDir+"cleanrecon.png")
