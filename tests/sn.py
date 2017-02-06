import matplotlib
matplotlib.use('Agg')
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
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

#cambRoot = "data/ell28k_highacc"
#theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)


cosmologyName = 'Planck15' # from ini file
lmax = 5000
iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = Cosmology(cosmoDict,constDict,lmax)
theory = cc.theory

fgfileX = None
beamFileX = None
fgfileY = None
beamFileY = None
TCMB = 2.725e6
gradCut = 10000
halo = True
#beamX = 7.0
beamX = 1.4
beamY = 1.4

# D56 S2+S3
noiseTX = np.sqrt(1./(1./17.**2.+1./11.7**2.))#16.0
noisePX = np.sqrt(2.)*noiseTX
noiseTY = np.sqrt(1./(1./17.**2.+1./11.7**2.))#16.0
noisePY = np.sqrt(2.)*noiseTY
# fsky = 430./41250.
# fgfileX = "data/foreground_powers.txt"
# beamFileX = "data/beam_3.txt"
# fgfileY = "data/foreground_powers.txt"
# beamFileY = "data/beam_3.txt"

# D56 S2
# noiseTX = 17.0
# noisePX = np.sqrt(2.)*noiseTX
# noiseTY = 17.0
# noisePY = np.sqrt(2.)*noiseTY
fsky = 430./41250.
# fgfile = "data/foreground_powers.txt"
beamFile = "data/beam_3.txt"

# D5+6 S2
# noiseTX = 11.3
# noisePX = np.sqrt(2.)*noiseTX
# noiseTY = 11.3
# noisePY = np.sqrt(2.)*noiseTY
# fsky = 128./41250.



# BOSS-N
# noiseTX = 32.83
# noisePX = np.sqrt(2.)*noiseTX
# noiseTY = 32.83 #16.0
# noisePY = np.sqrt(2.)*noiseTY
# fsky = 2100./41250.


tellminX = 500 #1000
tellmaxX = 3000
pellminX = 500 #1000
pellmaxX = 3000
tellminY = 500 #1000
tellmaxY = 3000
pellminY = 500 #1000
pellmaxY = 3000

# tellminX = 1000
# tellmaxX = 3000
# pellminX = 1000
# pellmaxX = 3000
# tellminY = 1000
# tellmaxY = 3000
# pellminY = 1000
# pellmaxY = 3000

lxcutTX = 90
lycutTX = 50
lxcutTY = lxcutTX
lycutTY = lycutTX
lxcutPX = lxcutTX
lycutPX = lycutTX
lxcutPY = lxcutTX
lycutPY = lycutTX

# lxcutTX = 0
# lycutTX = 0
# lxcutTY = 50
# lycutTY = 90
# lxcutPX = 0
# lycutPX = 0
# lxcutPY = 50
# lycutPY = 90


# tellminX = 2 #1000
# tellmaxX = 3000
# pellminX = 2 #1000
# pellmaxX = 3000
# tellminY = 500 #1000
# tellmaxY = 3000
# pellminY = 500 #1000
# pellmaxY = 3000


# lkneeX = [4129,312]
# alphaX = [-4.65,-3.05]
# lkneeY = [4129,312]
# alphaY = [-4.65,-3.05]


# ACT fits
lkneeX = [3294.8,1868.8]
alphaX = [-3.22,-0.59]
lkneeY = [3294.8,1868.8]
alphaY = [-3.22,-0.59]

# lkneeX = [0,0]
# alphaX = [1,1]
# lkneeY = [3294.8,1868.8]
# alphaY = [-3.22,-0.59]



# lkneeX = [4100,2000]
# alphaX = [-4.7,-2]
# lkneeY = [4100,2000]
# alphaY = [-4.7,-2]

# lkneeX = [0,0]
# alphaX = [1,1]
# lkneeY = [3400,330]
# alphaY = [-4.7,-3.8]


# lkneeX = [0,0]
# alphaX = [1,1]
# lkneeY = [0,0]
# alphaY = [1,1]

kmin = 40


deg = 10.
px = 0.5
dell = 10

kellrange = np.arange(80.,2100.,10.)

Clkk = theory.gCl("kk",kellrange)

cmb_bin_edges = np.arange(10,9000,10)

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
                      pellminY=pellminY,pellmaxY=pellmaxY,lkneesX=lkneeX,alphasX=alphaX, \
                                        lkneesY=lkneeY,alphasY=alphaY,lxcutTX=lxcutTX, \
                                        lxcutTY=lxcutTY,lycutTX=lycutTX,lycutTY=lycutTY, \
                                        lxcutPX=lxcutPX,lxcutPY=lxcutPY,lycutPX=lycutPX,lycutPY=lycutPY, \
                                        fgFileX=fgfileX,beamFileX=beamFileX,fgFileY=fgfileY,beamFileY=beamFileY )

    cmbbins = np.arange(50.,5000.,10.)
    binner = bin2D(myNls.N.modLMap,cmbbins)
    ells,nlpp = binner.bin(nPY)
    pl = Plotter(scaleY='log')
    pl.add(ells,theory.lCl("BB",ells)*ells**2.)
    pl.add(ells,nlpp*ells**2.)

    nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
                      pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
                      noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
                      pellminY=pellminY,pellmaxY=pellmaxY,lxcutTX=lxcutTX, \
                                        lxcutTY=lxcutTY,lycutTX=lycutTX,lycutTY=lycutTY, \
                                        lxcutPX=lxcutPX,lxcutPY=lxcutPY,lycutPX=lycutPX,lycutPY=lycutPY, \
                                        fgFileX=fgfileX,beamFileX=beamFileX,fgFileY=fgfileY,beamFileY=beamFileY )

    ells,nlppw = binner.bin(nPY)
    pl.add(ells,nlppw*ells**2.,ls="--")

    
    pl.done("output/clbb.png")
    sys.exit()


    # myNls.updateNoise(beamY,noiseTY,noisePY,tellminY,tellmaxY, \
    #                   pellminY,pellmaxY,beamY=beamX,noiseTY=noiseTX, \
    #                   noisePY=noisePX,tellminY=tellminX,tellmaxY=tellmaxX, \
    #                   pellminY=pellminX,pellmaxY=pellmaxX,lkneesX=lkneeY,alphasX=alphaY, \
    #                   lkneesY=lkneeX,alphasY=alphaX)
                    
    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
    nlfunc = interp1d(ls,Nls,bounds_error=False,fill_value = np.inf)

    nlnow = nlfunc(kellrange)
    Nlmvinv +=  (1./nlnow)

    pl.add(ls,4.*Nls/2./np.pi,label=polComb,ls="--")

    LF = LensForecast()
    LF.loadKK(kellrange,Clkk,ls,Nls)#kellrange,nlnow)
    sn,errs = LF.sn(kellrange,fsky,"kk")
    print polComb, sn


pl.add(kellrange,4.*Clkk/2./np.pi)

Nlmv = 1./Nlmvinv
pl.add(kellrange,4.*Nlmv/2./np.pi,label="mv",color='black')
pl.legendOn(loc='lower right',labsize=12)
pl._ax.set_xlim(kellrange.min(),kellrange.max())
pl.done("output/projnl.png")

Nlmvfunc = interp1d(kellrange,Nlmv,bounds_error=False,fill_value=np.inf)

snrange = np.arange(80.,2100.,240.)
Clkk = theory.gCl("kk",snrange)


LF = LensForecast()
LF.loadKK(snrange,Clkk,snrange,Nlmvfunc(snrange))
sn,errs = LF.sn(snrange,fsky,"kk")
print errs
print "mv", sn
