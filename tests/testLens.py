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
print "Done importing modules..."

# === COSMOLOGY ===
cosmologyName = 'LACosmology' # from ini file
iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
lmax = 8000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax)
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
kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,sourceZ,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
print "kappaint ", kappaMap[thetaMap*60.*180./np.pi<10.].mean()


# === DEFLECTION MAP ===
a = alphaMaker(thetaMap)
alpha = a.kappaToAlpha(kappaMap,test=True)
alphamod = 180.*60.*np.sum(alpha**2,0)**0.5/np.pi
print "alphaint ", alphamod[thetaMap*60.*180./np.pi<10.].mean()
pos = kappaMap.posmap() + alpha
pix = kappaMap.sky2pix(pos, safe=False)

# === CMB POWER SPECTRUM ===      
ps = powspec.read_spectrum("data/cl_lensinput.dat")



# === QUADRATIC ESTIMATOR INITIALIZATION ===      

class template:
    pass

templateLM = template()
templateLM.Ny, templateLM.Nx = kappaMap.shape
templateLM.pixScaleY, templateLM.pixScaleX = kappaMap.pixshape()

from orphics.analysis import flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(templateLM)
lmap = kappaMap.lmap 

pol = True

if pol:
    #polCombList = ["TT","ET","EB"]
    polCombList = ["EB"]
    shape = (3,)+shape
else:
    polCombList = ["TT"]

theory = cc.theory
nT,nP = fmaps.whiteNoise2D([0.01,0.01,0.01],0.01,modLMap,TCMB = TCMB)
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
                 TOnly=False,
                 halo=True,
                 gradCut=gradCut,verbose=True,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None)


bin_edges = np.arange(kellmin,kellmax,20)
b = bin2D(modLMap,bin_edges)
ls,Nls = b.bin(qest.N.Nlkk['EB'])

pl = Plotter(scaleY='log')
pl.add(bin_edges,theory.gCl("kk",bin_edges))
pl.add(ls,Nls)
pl.done("output/nl.png")

# === RUN N SIMS ===      
kappaStack = kappaMap*0.
N = 10
for i in range(N):
    print i
    map = enmap.rand_map(shape, wcs, ps)/TCMB
    lensedTQU = lensing.displace_map(map, pix,order=5)
    lensedMap = enmap.ifft(enmap.map2harm(lensedTQU)).real 
    
    if i==0:
        pl = Plotter()
        pl.plot2d(enmap.project(lensedMap[0],shapeTen,wcsTen))
        pl.done("output/lensed.png")
        pl = Plotter()
        pl.plot2d(enmap.project(lensedMap[0],shapeTen,wcsTen)-enmap.project(map[0],shapeTen,wcsTen))
        pl.done("output/diff.png")


    
    fot,foe,fob = enmap.fft(lensedMap,normalize=False)

    print "Reconstructing" , i , " ..."
    qest.updateTEB_X(fot,foe,fob,alreadyFTed=True)
    qest.updateTEB_Y(alreadyFTed=True)
    # kappaMvft = 0.
    # Ntotinv = 0.
    # for polComb in polCombList:
    #     kappaft,nlinv = qest.getKappa(polComb,weightedFt=True)
    #     kappaMvft += kappaft
    #     Ntotinv += nlinv
    # kappa = enmap.ifft(np.nan_to_num(kappaMvft/Ntotinv),normalize=False).real
    kappa = qest.getKappa(polCombList[0]).real
        
    kappaStack += kappa



pl = Plotter()
#pl.plot2d(enmap.project(kappaStack,shapeTen,wcsTen))
pl.plot2d(kappaStack/N)
pl.done("output/recon.png")
