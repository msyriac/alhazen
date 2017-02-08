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
from orphics.tools.stats import timeit, bin2D,coreBinner
from ConfigParser import SafeConfigParser 
from szlib.szcounts import ClusterCosmology

cosmologyName = 'LACosmology' #Planck15' 
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = Config.getint('general','camb_ellmax')
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax)
theory = cc.theory

expX = Config.get('general','X')
expY = Config.get('general','Y')

beamX,beamFileX,fgFileX,noiseTX,noisePX,tellminX,tellmaxX,pellminX, \
    pellmaxX,lxcutTX,lycutTX,lxcutPX,lycutPX,lkneeTX,alphaTX,lkneePX,alphaPX = getLensParams(Config,expX)
beamY,beamFileY,fgFileY,noiseTY,noisePY,tellminY,tellmaxY,pellminY, \
    pellmaxY,lxcutTY,lycutTY,lxcutPY,lycutPY,lkneeTY,alphaTY,lkneePY,alphaPY = getLensParams(Config,expY)


    
TCMB = Config.getfloat('general','TCMB')
gradCut = Config.getint('general','gradCut')
halo = Config.getboolean('general','halo')
fsky = Config.getfloat('general','sqDeg')/41250.

kmin = 40


deg = 10.
px = 0.5
dell = 10

kellrange = np.arange(80.,2100.,20.)
kfrange = np.arange(80.,2100.,1.)

Clkk = theory.gCl("kk",kfrange)

cmb_bin_edges = np.arange(10,9000,10)

Nlmvinv = 0.
pl = Plotter(scaleY='log')
for polComb in ['TT','TE','EE','EB','TB']:
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

    # cmbbins = np.arange(50.,5000.,10.)
    # binner = bin2D(myNls.N.modLMap,cmbbins)
    # ells,nlpp = binner.bin(nPY)
    # pl = Plotter(scaleY='log')
    # pl.add(ells,theory.lCl("BB",ells)*ells**2.)
    # pl.add(ells,nlpp*ells**2.)

    # nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX, \
    #                   pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY, \
    #                   noisePY=noisePY,tellminY=tellminY,tellmaxY=tellmaxY, \
    #                   pellminY=pellminY,pellmaxY=pellmaxY,lxcutTX=lxcutTX, \
    #                                     lxcutTY=lxcutTY,lycutTX=lycutTX,lycutTY=lycutTY, \
    #                                     lxcutPX=lxcutPX,lxcutPY=lxcutPY,lycutPX=lycutPX,lycutPY=lycutPY, \
    #                                     fgFileX=fgfileX,beamFileX=beamFileX,fgFileY=fgfileY,beamFileY=beamFileY )

    # ells,nlppw = binner.bin(nPY)
    # pl.add(ells,nlppw*ells**2.,ls="--")

    
    # pl.done("output/clbb.png")
    # sys.exit()


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
    LF.loadKK(kfrange,Clkk,ls,Nls)#kellrange,nlnow)
    sn,errs = LF.sn(kellrange,fsky,"kk")
    print polComb, sn


pl.add(kfrange,4.*Clkk/2./np.pi)

Nlmv = 1./Nlmvinv
pl.add(kellrange,4.*Nlmv/2./np.pi,label="mv",color='black')
pl.legendOn(loc='lower right',labsize=12)
pl._ax.set_xlim(kellrange.min(),kellrange.max())
pl.done("output/projnl.png")

print ls.shape
print Nlmv.shape
np.savetxt(expX+expY+"_nlmv.txt",np.vstack((np.array(kellrange),np.array(Nlmv))).transpose())
sys.exit()

Nlmvfunc = interp1d(kellrange,Nlmv,bounds_error=False,fill_value=np.inf)

snrange = np.arange(80.,2100.,240.)
Clkk = theory.gCl("kk",snrange)




LF = LensForecast()
LF.loadKK(snrange,Clkk,snrange,Nlmvfunc(snrange))
sn,errs = LF.sn(snrange,fsky,"kk")
print errs
print "mv", sn

Clkk = theory.gCl("kk",kfrange)
b = coreBinner(snrange)
cents,clkkbinned = b.binned(kfrange,Clkk) 

#Nlold = np.loadtxt("Nltemp.txt")


# pl = Plotter(scaleY='log',labelX="$L$",labelY="$[L(L+1)]^2C_L^{\\phi\\phi}/2\\pi$")
# pl.add(kfrange,4.*Clkk/2./np.pi)
# #pl.addErr(cents,4.*np.array(clkkbinned)/2./np.pi,yerr=np.array(errs)*4./2./np.pi,marker="x")
# pl.add(kellrange,4.*Nlmv/2./np.pi,color='black',label="ACTS15")
# pl.add(kellrange,4.*Nlold/2./np.pi,color='black',label="PlACTS15")
# pl.legendOn(loc='lower right',labsize=12)
# pl._ax.set_xlim(0,2200)
# pl._ax.set_ylim(-0.2e-7,1.6e-7)
# pl._ax.axhline(y=0.,ls="--",alpha=0.5)
# pl.done("output/errs.png")

#np.savetxt("Nltemp.txt",Nlmv)
fskyCMASS =  203./41250.
N = 12000. *fsky/fskyCMASS#148.#*12376./2950.
Mexp = 13.3
c = 5.0
#c = 1.18
z = 0.55
ls = kellrange
Nls = Nlmv
kellmax = 8000
# overdensity = 500
# critical = True
# atClusterZ = True
overdensity = 180
critical = False
atClusterZ = False
from alhazen.halos import NFWMatchedFilterSN
sn = NFWMatchedFilterSN(cc,Mexp,c,z,ells=ls,Nls=Nls,kellmax=kellmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

print 100./(sn*np.sqrt(N)), " %"
print "sn " , sn*np.sqrt(N)

from orphics.theory.limber import XCorrIntegrator

snrange = np.arange(80,2000,200)

xint = XCorrIntegrator(cosmoDict)
zcents,dndz = np.loadtxt("data/cmass_dndz.csv",unpack=True)
xint.addNz("cmass",zcents,dndz[1:],bias=2.0)
xint.generateCls(snrange)

Clgg = xint.getCl("cmass","cmass")
Clkg = xint.getCl("cmb","cmass")

ngal = 0.026

LF = LensForecast()
Clkk = theory.gCl("kk",snrange)
LF.loadKK(snrange,Clkk,ls,Nls)#kellrange,nlnow)
LF.loadKG(snrange,Clkg)#kellrange,nlnow)
LF.loadGG(snrange,Clgg,ngal=ngal)#kellrange,nlnow)
sn,errs = LF.sn(snrange,fsky,"kg")
print errs


frange = np.arange(40,2000,1)
xint.generateCls(frange)

Clgg = xint.getCl("cmass","cmass")
Clkg = xint.getCl("cmb","cmass")

b = coreBinner(snrange)
cents,clkgbinned = b.binned(frange,Clkg)

from scipy.interpolate import interp1d
clkgint = interp1d(frange,Clkg)

print "kg S/N " , sn
pl = Plotter(labelX="$L$",labelY="$LC_L$")
pl.add(frange,frange*Clkg)
pl.addErr(cents,clkgint(cents)*cents,yerr=np.array(errs)*cents,marker="o")
#pl.legendOn(loc='lower right',labsize=12)
pl._ax.set_xlim(0,2000)
#pl._ax.set_ylim(-0.2e-7,1.6e-7)
pl._ax.axhline(y=0.,ls="--",alpha=0.5)
pl.done("output/errs.png")


zcents,dndz = np.loadtxt("data/hscd6.csv",unpack=True)
xint.addNz("hsc",zcents,dndz[1:])
xint.generateCls(snrange)

Clss = xint.getCl("hsc","hsc")
Clks = xint.getCl("cmb","hsc")

ngal = 12.0


nlfile = "/astro/u/msyriac/repos/HSCxACT/data/actpolS2coaddN0_TT_6.txt"
ls,Nls = np.loadtxt(nlfile,unpack=True)

LF = LensForecast()
Clkk = theory.gCl("kk",snrange)
LF.loadKK(snrange,Clkk,ls,Nls)#kellrange,nlnow)
LF.loadKS(snrange,Clks)#kellrange,nlnow)
LF.loadSS(snrange,Clss,ngal=ngal)#kellrange,nlnow)
sn,errs = LF.sn(snrange,fsky,"ks")
print errs
print "ks ", sn

Clsg = xint.getCl("hsc","cmass")

ngalF = 0.026
LF = LensForecast()
LF.loadSS(snrange,Clss,ngal=ngal)#kellrange,nlnow)
LF.loadSG(snrange,Clsg)#kellrange,nlnow)
LF.loadGG(snrange,Clgg,ngal=ngalF)#kellrange,nlnow)
sn,errs = LF.sn(snrange,fsky,"sg")
print errs
print "sg ", sn

