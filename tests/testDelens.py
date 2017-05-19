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
from szlib.sims import BattagliaSims, getKappaSZ
from enlib.fft import fft,ifft
import os
print "Done importing modules..."


outDir = os.environ['WWW']+"plots/kappatest/"



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
sourceZ = 1100.

overdensity = 500.
critical = True
atClusterZ = True

# === TEMPLATE MAP ===
px = 0.2
arc = 100
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


# === KAPPA MAP ===
comL = cc.results.comoving_radial_distance(zL)*cc.h
comS = cc.results.comoving_radial_distance(sourceZ)*cc.h
winAtLens = (comS-comL)/comS

kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
arcmax = 10.
snap = 43

# === CMB POWER SPECTRUM ===      
ps = powspec.read_spectrum("data/cl_lensinput.dat")



# === QUADRATIC ESTIMATOR INITIALIZATION ===      

class template:
    pass
    
templateLM = template()
templateLM.Ny, templateLM.Nx = thetaMapDown.shape
Ny, Nx = thetaMapDown.shape
templateLM.pixScaleY, templateLM.pixScaleX = thetaMapDown.pixshape()

from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
lxMap,lyMap,modLMap,angMap,lx,ly = fmaps.getFTAttributesFromLiteMap(templateLM)

polCombList = ["TT"]


# simRoot1 = "/astro/astronfs01/workarea/msyriac/cmbSims/"
# beamPath = simRoot1 + "beam_0.txt"
# l,beamells = np.loadtxt(beamPath,unpack=True,usecols=[0,1])
fwhm = 1.0
tht_fwhm= np.deg2rad(fwhm / 60.)
l = np.arange(0.,10000.)
beamells = np.exp(-(tht_fwhm**2.)*(l**2.) / (8.*np.log(2.)))
beamTemplate = fmaps.makeTemplate(l,beamells,modLMap)

noiseT = 0.1
noiseP = np.sqrt(2.)*noiseT
whiteNoiseT = (np.pi / (180. * 60))**2.  * noiseT**2. / TCMB**2.  
whiteNoiseP = (np.pi / (180. * 60))**2.  * noiseP**2. / TCMB**2.  


ellNoise = np.arange(0,modLMap.max())
Ntt = ellNoise*0.+np.nan_to_num(whiteNoiseT)
Npp = ellNoise*0.+np.nan_to_num(whiteNoiseP)
Ntt[0] = 0.
Npp[0] = 0.
gGenT = fmaps.GRFGen(templateLM,ellNoise,Ntt,bufferFactor=1)
gGenP1 = fmaps.GRFGen(templateLM,ellNoise,Npp,bufferFactor=1)
gGenP2 = fmaps.GRFGen(templateLM,ellNoise,Npp,bufferFactor=1)

theory = cc.theory
fot = np.zeros((templateLM.Ny,templateLM.Nx))+0.j
filt_noiseT = fot.copy()*0.+np.nan_to_num(gGenT.power/ beamTemplate[:,:]**2.)
filt_noiseE = fot.copy()*0.+np.nan_to_num(gGenP1.power/ beamTemplate[:,:]**2.)
filt_noiseB = fot.copy()*0.+np.nan_to_num(gGenP2.power/ beamTemplate[:,:]**2.)


gradCut = 2000
cmbellmin = 20
cmbellmax = 8000
kellmin = 20
kellmax = 8000

fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)
qest = Estimator(templateLM,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[filt_noiseT,filt_noiseE,filt_noiseB],
                 noiseY2dTEB=[filt_noiseT,filt_noiseE,filt_noiseB],
                 fmaskX2dTEB=[fMaskCMB]*3,
                 fmaskY2dTEB=[fMaskCMB]*3,
                 fmaskKappa=fMask,
                 doCurl=False,
                 TOnly=not(pol),
                 halo=True,
                 gradCut=gradCut,verbose=False,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None)



#win = fmaps.initializeCosineWindowData(Ny,Nx,lenApod=100,pad=10)
win = 1.
w2 = np.mean(win**2.)


# === RUN N SIMS ===      
kappaStack = 0.
kappaFix = 0.
trueKappaStack = 0.
szStack = 0.
N = 1000
massIndices = range(300)
avgM500 = 0.
avgz = 0.

# remove to test norm dependence
ells = np.array(range(ps[0,0,:].size))
ps[0,0,:] = theory.uCl('TT',ells)*TCMB**2.

# pl = Plotter(scaleY='log',scaleX='log')
# pl.add(ells,ps[0,0,:]*ells**2.)
# pl.add(ells,theory.uCl('TT',ells)*ells**2.*TCMB**2.)
# pl.done(outDir+"cls.png")
# sys.exit()


for i in range(N):
    map = enmap.rand_map(shape, wcs, ps)/TCMB


    inputKappaMap = kappaMap

    if int(pxDown/px)>1:
        inpDown = enmap.downgrade(inputKappaMap,pxDown/px)
    else:
        inpDown = inputKappaMap
    trueKappaStack += inpDown

    # === DEFLECTION MAP ===
    a = alphaMaker(thetaMap)
    alpha = a.kappaToAlpha(inputKappaMap,test=False)
    alphamod = 180.*60.*np.sum(alpha**2,0)**0.5/np.pi
    pos = thetaMap.posmap() + alpha
    pix = thetaMap.sky2pix(pos, safe=False)

    lensedTQU = lensing.displace_map(map, pix,order=5)+0.j
    lensedMapX = lensedTQU
    lensedMapY = lensedMapX.copy()

    if int(pxDown/px)>1:
        lensedMapX = enmap.downgrade(lensedMapX,pxDown/px)
        lensedMapY = enmap.downgrade(lensedMapY,pxDown/px)
        
    lensedMapX = fmaps.convolveBeam(lensedMapX,modLMap,beamTemplate)
    lensedMapY = fmaps.convolveBeam(lensedMapY,modLMap,beamTemplate)

    if noiseT>1.e-3: lensedMapX = lensedMapX + gGenT.getMap(stepFilterEll=None)
    if noiseT>1.e-3: lensedMapY = lensedMapY + gGenT.getMap(stepFilterEll=None)

    lensedMapX = lensedMapX*win
    lensedMapY = lensedMapY*win

    if i==0:
        pl = Plotter()
        pl.plot2d(lensedMapX)
        pl.done(outDir+"lensed.png")

    
    
    fotX = np.nan_to_num(fft(lensedMapX,axes=[-2,-1])/ beamTemplate[:,:])
    fotY = np.nan_to_num(fft(lensedMapY,axes=[-2,-1])/ beamTemplate[:,:])


    if i%1==0: print "Reconstructing" , i , " ..."
    qest.updateTEB_X(fotX,alreadyFTed=True)
    qest.updateTEB_Y(fotY,alreadyFTed=True)
    kappa = qest.getKappa(polCombList[0]).real/w2

    kappaStack += kappa



kappaStack /= N
trueKappaStack /= N


stepmin = kellmin

kappaStack = fmaps.stepFunctionFilterLiteMap(kappaStack,modLMap,kellmax,ellMin=stepmin)


pl = Plotter()
pl.plot2d(kappaStack)
pl.done(outDir+"recon.png")



pl = Plotter()
pl.plot2d(trueKappaStack)
pl.done(outDir+"truestack.png")

pl = Plotter()
pl.plot2d(szStack)
pl.done(outDir+"szstack.png")

filtInput = fmaps.stepFunctionFilterLiteMap(trueKappaStack,modLMap,kellmax,ellMin=stepmin)
# filtSim = fmaps.stepFunctionFilterLiteMap(simKappa,modLMap,kellmax)

pl = Plotter()
pl.plot2d(filtInput)
pl.done(outDir+"filtinput.png")

# pl = Plotter()
# pl.plot2d(filtInput)
# pl.done(outDir+"filtsim.png")


dt = 0.2
thetaRange = np.arange(0.,arcmax,dt)
breal = bin2D(thetaMapDown*180.*60./np.pi,thetaRange)
#cents,inps = breal.bin(trueKappaStack)
cents,inpsFilt = breal.bin(filtInput)
cents,recons = breal.bin(kappaStack)
# cents,reconsfix = breal.bin(kappaFix)
# cents,simFilt = breal.bin(filtSim)

pl = Plotter()
#pl.add(cents,inps,ls="--")
pl.add(cents,inpsFilt)
# pl.add(cents,simFilt,ls="-.")
pl.add(cents,recons)
# pl.add(cents,reconsfix,ls="-.")
pl._ax.axhline(y=0.,ls="--",alpha=0.5)
pl.done(outDir+"profiles.png")

pl = Plotter()
#pl.add(cents,inps,ls="--")
#pl.add(cents,inpsFilt)
# pl.add(cents,simFilt,ls="-.")
pl.add(cents,(recons-inpsFilt)*100./inpsFilt.max())
#pl.add(cents,reconsfix,ls="-.")
pl._ax.axhline(y=0.,ls="--",alpha=0.5)
pl.done(outDir+"percent.png")



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
