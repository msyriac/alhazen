import matplotlib
matplotlib.use('Agg')
from orphics.tools.io import Plotter,dictFromSection,listFromConfig,getLensParams
from orphics.theory.gaussianCov import LensForecast
from orphics.theory.cosmology import Cosmology
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from scipy.interpolate import interp1d
import numpy as np
from alhazen.quadraticEstimator import NlGenerator,getMax
import orphics.analysis.flatMaps as fmaps 
import flipper.liteMap as lm
from orphics.tools.stats import timeit, bin2D
from ConfigParser import SafeConfigParser 


cosmologyName = 'Planck15' 
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = Config.getint('general','camb_ellmax')
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = Cosmology(cosmoDict,constDict,lmax)
theory = cc.theory

expX = Config.get('general','X')
expY = Config.get('general','Y')

beamX,beamFileX,fgFileX,noiseTX,noisePX,tellminX,tellmaxX,pellminX, \
    pellmaxX,lxcutTX,lycutTX,lxcutPX,lycutPX,lkneeTX,alphaTX,lkneePX,alphaPX = getLensParams(Config,expX)
beamY,beamFileY,fgFileY,noiseTY,noisePY,tellminY,tellmaxY,pellminY, \
    pellmaxY,lxcutTY,lycutTY,lxcutPY,lycutPY,lkneeTY,alphaTY,lkneePY,alphaPY = getLensParams(Config,expY)

cmb_bin_edges = np.arange(10,9000,10)


TCMB = Config.getfloat('general','TCMB')
gradCut = Config.getint('general','gradCut')
halo = Config.getboolean('general','halo')
fsky = Config.getfloat('general','sqDeg')/41250.
kmin = 40


deg = 10.
px = 0.5
dell = 10

kellrange = np.arange(80.,2100.,10.)


Nlmvinv = 0.
pl = Plotter(scaleY='log')
for polComb in ['TT','TE','EE','EB']:
    kmax = getMax(polComb,tellmaxY,pellmaxY)
    bin_edges = np.arange(kmin,kmax,dell)+dell
    lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
    myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
    myNls.updateBins(bin_edges)

    nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                      pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                      noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                      pellminY=pellminY,pellmaxY=pellmaxY,lkneesX=(lkneeTX,lkneePX),alphasX=(alphaTX,alphaPX), \
                                        lkneesY=(lkneeTY,lkneePY),alphasY=(alphaTY,alphaPY),lxcutTX=lxcutTX, \
                                        lxcutTY=lxcutTY,lycutTX=lycutTX,lycutTY=lycutTY, \
                                        lxcutPX=lxcutPX,lxcutPY=lxcutPY,lycutPX=lycutPX,lycutPY=lycutPY, \
                                        fgFileX=fgFileX,beamFileX=beamFileX,fgFileY=fgFileY,beamFileY=beamFileY )


    cbinner = bin2D(myNls.N.modLMap,cmb_bin_edges)
    ells, Nells = cbinner.bin(nTX)

    pl = Plotter(scaleY='log')
    pl.add(ells,Nells*ells**2.*TCMB**2.)
    pl.add(ells,Nells*ells**2.*TCMB**2.)
    tells,tnlstt = np.loadtxt('data/louisCls.dat',delimiter=',',unpack=True)
    pl.add(tells,tnlstt)
    pl.done("output/compnl.png")
    sys.exit()
           
