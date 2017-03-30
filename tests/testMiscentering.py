from orphics.tools.output import Plotter
import flipper.liteMap as lm
from szlib.szcounts import ClusterCosmology
from orphics.tools.io import dictFromSection,listFromConfig
import orphics.tools.io as io
from ConfigParser import SafeConfigParser 
from alhazen.halos import NFWMatchedFilterSN
import numpy as np
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax

#Mexp = 14.4 #np.log10(2.e14)
Mexp = np.log10(5.e14)
z = 1.5
c = 1.18

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file

iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 8000

cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)
theory = cc.theory




beamrange = np.arange(0.5,5.0,0.25)

# Make a CMB Noise Curve
cambRoot = "data/ell28k_highacc"
gradCut = 2000
halo = True
#beamX = 1.0
#beamY = 1.0
noiseTX = 10.0
noisePX = 14.14
noiseTY = 10.0
noisePY = 14.14
tellmin = 2
tellmax = 8000
gradCut = 2000
pellmin = 2
pellmax = 8000
polComb = 'EB'
kmin = 100
kmax = getMax(polComb,tellmax,pellmax)

deg = 10.
px = 0.5
dell = 10
bin_edges = np.arange(kmin,kmax,dell)+dell
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    
pl = Plotter(scaleY='log',scaleX='log')

sns = []
snsM = []

overdensity=500.
critical=True
atClusterZ=True
kellmax = 8000

outDir = "/gpfs01/astro/www/msyriac/plots/"

for beam in beamrange:
    beamX = beam
    beamY = beam
    print beam
    myNls.updateNoise(beamX,noiseTX,noisePX,tellmin,tellmax,pellmin,pellmax,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY)
    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

    #ls,Nls = np.loadtxt("../SZ_filter/data/LA_pol_Nl.txt",unpack=True,delimiter=',')



    sn,kappa = NFWMatchedFilterSN(cc,Mexp,c,z,ells=ls,Nls=Nls,kellmax=kellmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,returnKappa=True)
    # io.quickPlot2d(kappa,outDir+"kappa.png")
    
    snM,kappa = NFWMatchedFilterSN(cc,Mexp,c,z,ells=ls,Nls=Nls,kellmax=kellmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,rayleighSigmaArcmin=beam/2.,returnKappa=True)
    # io.quickPlot2d(kappa,outDir+"kappaM.png")
    # sys.exit()
    
    sns.append(sn)
    snsM.append(snM)
    pl.add(ellkk,4.*Clkk/2./np.pi)
    pl.add(ls,4.*Nls/2./np.pi,label=str(beam))
    
    
pl.legendOn(loc='lower left',labsize=10)
pl.done(outDir+"nlEB.png")



pl = Plotter(labelX='beam (arcmin)',labelY='S/N per cluster (TT)',ftsize=14)
pl.add(beamrange,sns,ls="--")
pl.add(beamrange,snsM)
pl._ax.tick_params(axis='both', which='major', labelsize=12)
pl._ax.tick_params(axis='both', which='minor', labelsize=12)
pl.done(outDir+"snEB.png")





