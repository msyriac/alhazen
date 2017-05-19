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
import os,sys
import alhazen.planckInterface as pint
import healpy as hp
import flipper.liteMap as lm
from scipy.ndimage.interpolation import zoom
print "Done importing modules..."


ras,decs = pint.getCatalogRADecsPlanck(sncut=5.)
#ras,decs = pint.getCatalogRADecsRedmapper()
print ras.min(),ras.max()
print decs.min(),decs.max()

print len(ras)
#sys.exit()
p143Loc = '/astro/astronfs01/workarea/msyriac/PlanckClusters/WPR2_CMB_muK.fits'
hpPlanck = hp.read_map(p143Loc)
 

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





# === QUADRATIC ESTIMATOR INITIALIZATION ===      


from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator




# === RUN N SIMS ===      
kappaStack = 0.
stampStack = 0.
maxN = None #10000 #None
pixScale = 1.0
widthArc = 160.
width = widthArc/60.
Np = np.int(width/pixScale*60.+0.5)
tfact = 2.7255e6

np.random.seed(400)

#bin_edge , hist = np.histogram(decs,bins=60)
# from orphics.tools.io import Plotter
# import os, sys
# pl = Plotter()
# pl._ax.hist(decs)
# pl.done("output/dechist.png")
# sys.exit()
# mindec = np.cos(min(decs)*np.pi/180.)
# maxdec = np.cos(max(decs)*np.pi/180.)

doMf = False

if doMf:
    maxN = 100000


i=0
for ra,dec in zip(ras[:maxN],decs[:maxN]):
    #for k in range(maxN):
    i+=1
    if i%10==0: print i,ra,dec

    if doMf:
        ra = np.random.uniform(min(ras),max(ras))
        dec = decs[k%len(decs)]

    raLeft = ra - width/2.
    raRight = ra + width/2.
    decLeft = dec - width/2.
    decRight = dec + width/2.

    fieldCoords = (raLeft,decLeft,raRight,decRight)
    smap = lm.makeEmptyCEATemplateAdvanced(*fieldCoords,pixScaleXarcmin=pixScale,pixScaleYarcmin=pixScale)
    smap.loadDataFromHealpixMap(hpPlanck,hpCoords="GALACTIC")


    stamp = smap.data.copy() / tfact
    stamp = zoom(stamp,zoom=(float(Np)/stamp.shape[0],float(Np)/stamp.shape[1]))
    #print ra, dec, stamp.shape
    
    if i==1:
        
        lxMap,lyMap,modLMap,angMap,lx,ly = fmaps.getFTAttributesFromLiteMap(smap)
        xMap,yMap,modRMap,xx,yy = fmaps.getRealAttributes(smap)
        polCombList = ["TT"]

        Ny, Nx = stamp.shape
        win = fmaps.initializeCosineWindowData(Ny,Nx,lenApod=40,pad=5)
        #win = 1.
        w2 = np.mean(win**2.)



        fwhm = 5.0
        tht_fwhm= np.deg2rad(fwhm / 60.)
        l = np.arange(0.,10000.)
        beamells = np.exp(-(tht_fwhm**2.)*(l**2.) / (8.*np.log(2.)))
        beamTemplate = fmaps.makeTemplate(l,beamells,modLMap)
        #print beamTemplate.shape

        noiseT = 42.0
        noiseP = np.sqrt(2.)*noiseT
        whiteNoiseT = (np.pi / (180. * 60))**2.  * noiseT**2. / TCMB**2.  
        whiteNoiseP = (np.pi / (180. * 60))**2.  * noiseP**2. / TCMB**2.  


        ellNoise = np.arange(0,modLMap.max())
        Ntt = ellNoise*0.+np.nan_to_num(whiteNoiseT)
        Npp = ellNoise*0.+np.nan_to_num(whiteNoiseP)
        Ntt[0] = 0.
        Npp[0] = 0.
        gGenT = fmaps.GRFGen(smap,ellNoise,Ntt,bufferFactor=1)
        gGenP1 = fmaps.GRFGen(smap,ellNoise,Npp,bufferFactor=1)
        gGenP2 = fmaps.GRFGen(smap,ellNoise,Npp,bufferFactor=1)

        theory = cc.theory
        fot = np.zeros((smap.Ny,smap.Nx))+0.j
        filt_noiseT = fot.copy()*0.+np.nan_to_num(gGenT.power/ beamTemplate[:,:]**2.)
        filt_noiseE = fot.copy()*0.+np.nan_to_num(gGenP1.power/ beamTemplate[:,:]**2.)
        filt_noiseB = fot.copy()*0.+np.nan_to_num(gGenP2.power/ beamTemplate[:,:]**2.)


        gradCut = 2000
        cmbellmin = 20
        cmbellmax = 4000
        kellmin = 20
        kellmax = 4000

        fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
        fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)
        qest = Estimator(smap,
                         theory,
                         theorySpectraForNorm=None,
                         noiseX2dTEB=[filt_noiseT,filt_noiseE,filt_noiseB],
                         noiseY2dTEB=[filt_noiseT,filt_noiseE,filt_noiseB],
                         fmaskX2dTEB=[fMaskCMB]*3,
                         fmaskY2dTEB=[fMaskCMB]*3,
                         fmaskKappa=fMask,
                         doCurl=False,
                         TOnly=True,
                         halo=True,
                         gradCut=gradCut,verbose=False,
                         loadPickledNormAndFilters=None,
                         savePickledNormAndFilters=None)

    
    lensedMapX = stamp.copy()*win
    lensedMapY = stamp.copy()*win

        
    try:
        fotX = np.nan_to_num(fft(lensedMapX,axes=[-2,-1])/ beamTemplate[:,:])
    except:
        print "skipping ", i, ra, dec
        i-=1
        continue
    fotY = np.nan_to_num(fft(lensedMapY,axes=[-2,-1])/ beamTemplate[:,:])


    if i%10==0: print "Reconstructing" , i , " ..."
    qest.updateTEB_X(fotX,alreadyFTed=True)
    qest.updateTEB_Y(fotY,alreadyFTed=True)
    kappa = qest.getKappa(polCombList[0]).real/w2
        
    kappaStack += kappa

N = i
kappaStack /= N

saveMf = False
try:
    if doMf:
        np.savetxt("data/meanfield.dat",kappaStack)
        print "saved meanfield"
        saveMf = True
except:
    pass

if not(saveMf):
    try:    
        mf = np.loadtxt("data/meanfield.dat")
        kappaStack -= mf
        print "subtracted meanfield"
    except:
        pass

    

stepmin = kellmin

kappaStack = fmaps.stepFunctionFilterLiteMap(kappaStack,modLMap,kellmax,ellMin=stepmin)


pl = Plotter()
pl.plot2d(kappaStack)
pl.done(outDir+"recon.png")



dt = pixScale
arcmax = 20.
thetaRange = np.arange(0.,arcmax,dt)
breal = bin2D(modRMap*180.*60./np.pi,thetaRange)
cents,recons = breal.bin(kappaStack)

pl = Plotter()
pl.add(cents,recons)
pl._ax.axhline(y=0.,ls="--",alpha=0.5)
pl.done(outDir+"profiles.png")

